#!/usr/bin/env python3
"""RAG-powered essay writer using Claude with editing and advanced retrieval."""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class RAGWriter:
    """RAG system for generating essays in the style of source material."""

    def __init__(self, index_dir: str = "index"):
        """Initialize the RAG writer."""
        self.index_dir = index_dir
        self.index = None
        self.metadata = None
        self.config = None
        self.model = None
        self.client = None
        self.embeddings = None  # Store embeddings for MMR

        # Session state
        self.top_k = 8  # Default number of chunks to retrieve
        self.use_mmr = True  # Use MMR by default for diversity
        self.mmr_lambda = 0.7  # Balance between relevance and diversity
        self.use_research = True  # Whether to retrieve indexed research
        self.last_prompt = None
        self.last_essay = None
        self.last_sources = None
        self.last_research_sources = None
        self.essay_history = []  # Track all essays in session
        self.excluded_sources = set()  # Sources to exclude from retrieval

        # Research sources (PDFs for information, not style)
        self.research_sources = []  # List of {'name': str, 'content': str}

        # Structure sources (for essay organization, not voice)
        self.structure_sources = []  # List of {'name': str, 'content': str}
        self.structure_budget = 20000  # Max chars for structure sources

        self._load_index()
        self._load_claude()

    def _load_index(self):
        """Load the FAISS index and metadata."""
        index_path = Path(self.index_dir)

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index directory '{self.index_dir}' not found. "
                "Run build_index.py first to create the index."
            )

        print("Loading index...")
        self.index = faiss.read_index(f"{self.index_dir}/vector.index")

        with open(f"{self.index_dir}/metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)

        with open(f"{self.index_dir}/config.pkl", 'rb') as f:
            self.config = pickle.load(f)

        # Load embeddings for MMR if available
        embeddings_path = f"{self.index_dir}/embeddings.npy"
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            print(f"Loaded embeddings for MMR retrieval")
        else:
            print("Note: Run 'rebuild' to enable MMR diversity retrieval")

        # Display index info
        style_count = self.config.get('style_chunks', self.config['num_chunks'])
        research_count = self.config.get('research_chunks', 0)
        print(f"Loaded index: {self.config['num_chunks']} chunks ({style_count} style, {research_count} research)")

        # Load the same model used for indexing
        print(f"Loading embedding model: {self.config['model_name']}...")
        self.model = SentenceTransformer(self.config['model_name'])

    def _load_claude(self):
        """Load Claude API client."""
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. "
                "Please create a .env file with your API key. "
                "Get one at: https://console.anthropic.com/"
            )

        self.client = Anthropic(api_key=api_key)
        print("Claude API initialized\n")

    def load_pdf(self, pdf_path: str) -> Optional[Dict]:
        """Load a PDF file as a research source.

        Returns dict with name and content, or None if failed.
        """
        if not PDF_SUPPORT:
            print("PDF support not available. Install PyMuPDF: pip install PyMuPDF")
            return None

        path = Path(pdf_path)
        if not path.exists():
            print(f"File not found: {pdf_path}")
            return None

        if path.suffix.lower() != '.pdf':
            print(f"Not a PDF file: {pdf_path}")
            return None

        try:
            doc = fitz.open(pdf_path)
            text_parts = []

            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num}]\n{text}")

            doc.close()

            content = "\n\n".join(text_parts)

            if not content.strip():
                print(f"Warning: No text extracted from {path.name}")
                return None

            return {
                'name': path.name,
                'path': str(path.absolute()),
                'content': content,
                'pages': len(text_parts)
            }

        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None

    def load_pdf_folder(self, folder_path: str) -> int:
        """Load all PDFs from a folder as research sources.

        Returns number of PDFs loaded.
        """
        path = Path(folder_path)
        if not path.exists() or not path.is_dir():
            print(f"Folder not found: {folder_path}")
            return 0

        pdf_files = list(path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return 0

        loaded = 0
        for pdf_file in pdf_files:
            # Skip if already loaded
            if any(r['name'] == pdf_file.name for r in self.research_sources):
                print(f"  Skipping {pdf_file.name} (already loaded)")
                continue

            result = self.load_pdf(str(pdf_file))
            if result:
                self.research_sources.append(result)
                print(f"  Loaded: {result['name']} ({result['pages']} pages)")
                loaded += 1

        return loaded

    def _smart_sample_structure(self, content: str, max_chars: int = 20000) -> Tuple[str, dict]:
        """Intelligently sample a document to capture structural patterns.

        Takes intro, evenly-spaced middle samples, and conclusion.
        Returns (sampled_content, stats_dict).
        """
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        total_paragraphs = len(paragraphs)
        total_chars = len(content)

        # If already under limit, return as-is
        if total_chars <= max_chars:
            return content, {
                'sampled': False,
                'total_paragraphs': total_paragraphs,
                'sampled_paragraphs': total_paragraphs,
                'total_chars': total_chars,
                'sampled_chars': total_chars
            }

        # Calculate how many paragraphs we can afford
        avg_para_len = total_chars / total_paragraphs if total_paragraphs > 0 else 1000
        target_paragraphs = int(max_chars / avg_para_len)
        target_paragraphs = max(target_paragraphs, 10)  # Minimum 10 paragraphs

        # Allocate: 20% intro, 60% middle samples, 20% conclusion
        intro_count = max(2, target_paragraphs // 5)
        conclusion_count = max(2, target_paragraphs // 5)
        middle_count = target_paragraphs - intro_count - conclusion_count

        # Get intro paragraphs
        intro_paras = paragraphs[:intro_count]

        # Get conclusion paragraphs
        conclusion_paras = paragraphs[-conclusion_count:]

        # Sample middle paragraphs evenly
        middle_start = intro_count
        middle_end = total_paragraphs - conclusion_count
        middle_pool = paragraphs[middle_start:middle_end]

        if len(middle_pool) <= middle_count:
            middle_paras = middle_pool
        else:
            # Evenly sample from middle
            step = len(middle_pool) / middle_count
            indices = [int(i * step) for i in range(middle_count)]
            middle_paras = [middle_pool[i] for i in indices]

        # Build sampled content with markers
        sampled_parts = []

        sampled_parts.append("=== OPENING SECTION ===")
        sampled_parts.extend(intro_paras)

        sampled_parts.append("\n=== MIDDLE SECTIONS (sampled) ===")
        for i, para in enumerate(middle_paras):
            if i > 0 and i % 3 == 0:
                sampled_parts.append("[...]")  # Indicate skipped content
            sampled_parts.append(para)

        sampled_parts.append("\n=== CLOSING SECTION ===")
        sampled_parts.extend(conclusion_paras)

        sampled_content = "\n\n".join(sampled_parts)
        sampled_paragraphs = len(intro_paras) + len(middle_paras) + len(conclusion_paras)

        return sampled_content, {
            'sampled': True,
            'total_paragraphs': total_paragraphs,
            'sampled_paragraphs': sampled_paragraphs,
            'total_chars': total_chars,
            'sampled_chars': len(sampled_content)
        }

    def load_structure_source(self, file_path: str, max_chars: int = 20000) -> Optional[Dict]:
        """Load a text or PDF file as a structure reference source.

        Long documents are intelligently sampled to capture structural patterns.
        Returns dict with name and content, or None if failed.
        """
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {file_path}")
            return None

        try:
            # Handle PDFs
            if path.suffix.lower() == '.pdf':
                if not PDF_SUPPORT:
                    print("PDF support not available. Install PyMuPDF: pip install PyMuPDF")
                    return None

                doc = fitz.open(file_path)
                text_parts = []

                for page_num, page in enumerate(doc, 1):
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(f"[Page {page_num}]\n{text}")

                doc.close()
                full_content = "\n\n".join(text_parts)
                pages = len(text_parts)
            else:
                # Handle text files
                with open(path, 'r', encoding='utf-8') as f:
                    full_content = f.read()
                pages = None

            if not full_content.strip():
                print(f"Warning: No text content in: {path.name}")
                return None

            # Smart sample if too long
            sampled_content, stats = self._smart_sample_structure(full_content, max_chars)

            result = {
                'name': path.name,
                'path': str(path.absolute()),
                'content': sampled_content,
                'full_content': full_content if stats['sampled'] else None,
                **stats
            }

            if pages is not None:
                result['pages'] = pages

            return result

        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retrieve_mmr(self, query: str, top_k: int = 8, lambda_param: float = 0.7) -> List[Dict]:
        """Retrieve chunks using Maximal Marginal Relevance for diversity.

        MMR balances relevance to the query with diversity among selected chunks.
        lambda_param: 1.0 = pure relevance, 0.0 = pure diversity
        """
        if self.embeddings is None:
            # Fall back to standard retrieval
            return self.retrieve(query, top_k=top_k)

        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

        # Get candidate pool (retrieve more than we need)
        candidate_k = min(top_k * 4, len(self.metadata))
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            candidate_k
        )

        # Filter out excluded sources
        valid_candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            filename = self.metadata[idx]['filename']
            if filename not in self.excluded_sources:
                valid_candidates.append((idx, dist))

        if not valid_candidates:
            print("Warning: All sources are excluded. Showing results anyway.")
            valid_candidates = [(idx, dist) for idx, dist in zip(indices[0], distances[0])]

        # MMR selection
        selected = []
        selected_embeddings = []
        candidate_pool = list(valid_candidates)

        while len(selected) < top_k and candidate_pool:
            best_score = float('-inf')
            best_idx = None
            best_pool_idx = None

            for pool_idx, (doc_idx, _) in enumerate(candidate_pool):
                doc_embedding = self.embeddings[doc_idx]

                # Relevance score (convert L2 distance to similarity)
                relevance = self._cosine_similarity(query_embedding, doc_embedding)

                # Diversity score (max similarity to already selected)
                if selected_embeddings:
                    max_sim = max(
                        self._cosine_similarity(doc_embedding, sel_emb)
                        for sel_emb in selected_embeddings
                    )
                else:
                    max_sim = 0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = doc_idx
                    best_pool_idx = pool_idx

            if best_idx is not None:
                selected.append(best_idx)
                selected_embeddings.append(self.embeddings[best_idx])
                candidate_pool.pop(best_pool_idx)

        # Build results
        results = []
        for idx in selected:
            results.append({
                **self.metadata[idx],
                'distance': float(np.linalg.norm(query_embedding - self.embeddings[idx]))
            })

        return results

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant chunks for a query."""
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Search index - get extra results to filter
        search_k = min(top_k * 3, len(self.metadata))
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)

        # Get metadata for results, filtering excluded sources
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            filename = self.metadata[idx]['filename']
            if filename not in self.excluded_sources:
                results.append({
                    **self.metadata[idx],
                    'distance': float(distance)
                })
                if len(results) >= top_k:
                    break

        # If we couldn't get enough after filtering, add some back with warning
        if len(results) < top_k and self.excluded_sources:
            print(f"Warning: Only found {len(results)} chunks after excluding sources")

        return results

    def retrieve_by_type(self, query: str, source_type: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks filtered by source type ('style' or 'research')."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Get more candidates to filter from
        search_k = min(top_k * 5, len(self.metadata))
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            meta = self.metadata[idx]
            # Filter by source type AND exclusions
            if meta.get('source_type') == source_type and meta['filename'] not in self.excluded_sources:
                results.append({
                    **meta,
                    'distance': float(distance)
                })
                if len(results) >= top_k:
                    break

        return results

    def retrieve_mmr_by_type(self, query: str, source_type: str, top_k: int = 8, 
                             lambda_param: float = 0.7) -> List[Dict]:
        """MMR retrieval filtered by source type ('style' or 'research')."""
        if self.embeddings is None:
            return self.retrieve_by_type(query, source_type, top_k=top_k)

        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]

        # Get larger candidate pool to filter from
        candidate_k = min(top_k * 6, len(self.metadata))
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            candidate_k
        )

        # Filter candidates by source type and exclusions
        valid_candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            meta = self.metadata[idx]
            if meta.get('source_type') == source_type and meta['filename'] not in self.excluded_sources:
                valid_candidates.append((idx, dist))

        if not valid_candidates:
            return []

        # MMR selection
        selected = []
        selected_embeddings = []
        candidate_pool = list(valid_candidates)

        while len(selected) < top_k and candidate_pool:
            best_score = float('-inf')
            best_idx = None
            best_pool_idx = None

            for pool_idx, (doc_idx, _) in enumerate(candidate_pool):
                doc_embedding = self.embeddings[doc_idx]

                # Relevance score
                relevance = self._cosine_similarity(query_embedding, doc_embedding)

                # Diversity score
                if selected_embeddings:
                    max_sim = max(
                        self._cosine_similarity(doc_embedding, sel_emb)
                        for sel_emb in selected_embeddings
                    )
                else:
                    max_sim = 0

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = doc_idx
                    best_pool_idx = pool_idx

            if best_idx is not None:
                selected.append(best_idx)
                selected_embeddings.append(self.embeddings[best_idx])
                candidate_pool.pop(best_pool_idx)

        # Build results
        results = []
        for idx in selected:
            results.append({
                **self.metadata[idx],
                'distance': float(np.linalg.norm(query_embedding - self.embeddings[idx]))
            })

        return results

    def generate_essay(self, prompt: str, top_k: Optional[int] = None,
                       max_tokens: int = 4000, revision_feedback: Optional[str] = None,
                       previous_essay: Optional[str] = None) -> Tuple[str, List[Dict], List[Dict]]:
        """Generate an essay using retrieved context.

        Returns tuple of (essay_text, style_sources_used, research_sources_used)
        """
        k = top_k or self.top_k

        # Retrieve STYLE chunks (for voice/tone)
        print(f"Retrieving style context (top {k} chunks, MMR={'on' if self.use_mmr else 'off'})...")
        if self.use_mmr:
            style_results = self.retrieve_mmr_by_type(prompt, 'style', top_k=k, lambda_param=self.mmr_lambda)
        else:
            style_results = self.retrieve_by_type(prompt, 'style', top_k=k)

        # Retrieve RESEARCH chunks (for facts/information) from index
        research_results = []
        if self.use_research and self.config.get('research_chunks', 0) > 0:
            print(f"Retrieving research context (top {k} chunks)...")
            if self.use_mmr:
                research_results = self.retrieve_mmr_by_type(prompt, 'research', top_k=k, lambda_param=self.mmr_lambda)
            else:
                research_results = self.retrieve_by_type(prompt, 'research', top_k=k)

        # Build style context
        style_parts = []
        for i, result in enumerate(style_results, 1):
            style_parts.append(f"[Style Source {i} - {result['filename']}]")
            style_parts.append(result['text'])
            style_parts.append("")

        style_context = "\n".join(style_parts)

        # Build research context from indexed chunks
        research_context = ""
        if research_results:
            research_parts = []
            for i, result in enumerate(research_results, 1):
                research_parts.append(f"[Research {i} - {result['filename']}]")
                research_parts.append(result['text'])
                research_parts.append("")
            research_context = "\n".join(research_parts)

        # Also include any dynamically loaded research sources (legacy support)
        if self.research_sources:
            legacy_parts = []
            for i, source in enumerate(self.research_sources, 1):
                content = source['content']
                if len(content) > 15000:
                    content = content[:15000] + "\n\n[... content truncated for length ...]"
                legacy_parts.append(f"[Additional Research {i} - {source['name']}]\n{content}")
            if research_context:
                research_context += "\n\n" + "\n\n".join(legacy_parts)
            else:
                research_context = "\n\n".join(legacy_parts)

        # Build structure context if available
        structure_context = ""
        if self.structure_sources:
            structure_parts = []
            for i, source in enumerate(self.structure_sources, 1):
                # Truncate very long files to avoid token limits
                content = source['content']
                if len(content) > 20000:
                    content = content[:20000] + "\n\n[... content truncated for length ...]"
                structure_parts.append(f"[Structure Reference {i} - {source['name']}]\n{content}")
            structure_context = "\n\n".join(structure_parts)

        # Build system prompt based on what sources are available
        has_structure = bool(self.structure_sources)
        has_research = bool(research_results) or bool(self.research_sources)

        if has_structure or has_research:
            distinctions = ["- STYLE SOURCES: Use these to capture the author's voice, tone, rhetorical techniques, vocabulary, sentence structure, and way of thinking. Your writing should feel like it was written by this author."]

            if has_structure:
                distinctions.append("- STRUCTURE SOURCES: Use these as a template for essay organization, section flow, how arguments are built and developed, paragraph structure, and overall narrative arc. Mimic the STRUCTURE but NOT the voice or vocabulary.")

            if has_research:
                distinctions.append("- RESEARCH SOURCES: Use these for factual information, data, and subject matter expertise. Extract relevant facts and insights, but do NOT let these influence your writing style or structure.")

            system_prompt = f"""You are an essay writer that combines elements from multiple source types.

IMPORTANT DISTINCTIONS:
{chr(10).join(distinctions)}

Your essay should:
1. Sound like the author from the STYLE SOURCES (voice, tone, word choice)
{('2. Be organized like the STRUCTURE SOURCES (sections, flow, argument development)' + chr(10)) if has_structure else ''}{'3. ' if has_structure else '2. '}Incorporate facts from RESEARCH SOURCES if provided"""
        else:
            system_prompt = """You are an essay writer that mimics the writing style, voice, and analytical approach of the source material provided.

Carefully analyze the voice, tone, rhetorical techniques, vocabulary, sentence structure, and thematic interests present in the source material. Then write an essay on the given topic that authentically captures this distinctive style.

Your essay should feel like it could have been written by the original author."""

        # Build user message
        structure_section = f"\n\nSTRUCTURE SOURCES (mimic this organization and flow):\n{structure_context}" if structure_context else ""
        research_section = f"\n\nRESEARCH SOURCES (for information only):\n{research_context}" if research_context else ""

        if revision_feedback and previous_essay:
            # Revision mode
            user_message = f"""Based on the following materials, revise this essay:

TOPIC: {prompt}

PREVIOUS ESSAY:
{previous_essay}

REVISION FEEDBACK:
{revision_feedback}

STYLE SOURCES (mimic this writing voice and tone):
{style_context}{structure_section}{research_section}

Please revise the essay according to the feedback while maintaining the style from STYLE SOURCES{' and structure from STRUCTURE SOURCES' if structure_context else ''}."""
            print("Revising essay with Claude Sonnet 4.5...\n")
        else:
            structure_instruction = " Follow the organizational patterns from STRUCTURE SOURCES." if structure_context else ""
            research_instruction = " Incorporate facts from RESEARCH SOURCES." if research_context else ""

            user_message = f"""Based on the following materials, write an essay on this topic:

TOPIC: {prompt}

STYLE SOURCES (mimic this writing voice and tone):
{style_context}{structure_section}{research_section}

Write a complete essay that captures the voice and tone from STYLE SOURCES.{structure_instruction}{research_instruction}"""
            print("Generating essay with Claude Sonnet 4.5...\n")

        # Call Claude API
        message = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        essay = message.content[0].text
        return essay, style_results, research_results

    def preview_sources(self, prompt: str, top_k: Optional[int] = None) -> List[Dict]:
        """Preview which sources would be retrieved for a prompt."""
        k = top_k or self.top_k

        if self.use_mmr:
            results = self.retrieve_mmr(prompt, top_k=k, lambda_param=self.mmr_lambda)
        else:
            results = self.retrieve(prompt, top_k=k)

        return results

    def get_all_sources(self) -> List[str]:
        """Get list of all unique source files."""
        sources = set()
        for meta in self.metadata:
            sources.add(meta['filename'])
        return sorted(sources)

    def get_sources_by_type(self) -> Tuple[List[str], List[str]]:
        """Get lists of source files by type (style, research)."""
        style_sources = set()
        research_sources = set()
        for meta in self.metadata:
            if meta.get('source_type') == 'research':
                research_sources.add(meta['filename'])
            else:
                style_sources.add(meta['filename'])
        return sorted(style_sources), sorted(research_sources)

    def print_help(self):
        """Print help information."""
        print("\n" + "=" * 60)
        print("COMMANDS")
        print("=" * 60)
        print("""
ESSAY GENERATION:
  <topic>           Write an essay on <topic>
  regenerate, r     Regenerate the last essay with same prompt

EDITING:
  edit <feedback>   Revise the last essay with your feedback
  e <feedback>      Short form of edit

INDEXED SOURCES (rebuilt with build_index.py):
  transcripts/      .txt files → style sources (for voice/tone)
  research/         .pdf files → research sources (for facts)
  
  Run 'python build_index.py' after adding files to either folder.
  Research PDFs are chunked and semantically searched, so even
  400-page documents work efficiently.

STRUCTURE SOURCES (for essay organization, not voice):
  structure <path>  Load a .txt or .pdf as a structure reference
  structures        List loaded structure sources
  struct-budget <N> Set max chars for structure (current: {})
  unload-struct <n> Remove a structure source
  clear-structure   Clear all structure sources

ADDITIONAL RESEARCH (load PDFs at runtime, not indexed):
  pdf <path>        Load a PDF as additional research
  pdfs <folder>     Load all PDFs from a folder
  research          List loaded additional research
  unload <name>     Remove an additional research source
  clear-research    Clear all additional research

RETRIEVAL SETTINGS:
  chunks <N>        Set chunks to retrieve per type (current: {})
  mmr on/off        Toggle MMR diversity (current: {})
  lambda <0-1>      Set MMR lambda (current: {:.1f})
                    Higher = more relevance, lower = more diversity
  use-research      Toggle indexed research retrieval (current: {})

SOURCE CONTROL:
  preview <topic>   Preview sources for a topic without generating
  sources           List all indexed source files
  exclude <file>    Exclude a source file from retrieval
  include <file>    Re-include an excluded source file
  excluded          Show currently excluded sources
  clear-excluded    Clear all exclusions

HISTORY:
  history           Show essay history for this session
  last              Show the last generated essay again

UTILITIES:
  save              Save the last essay to a file
  settings          Show current settings
  help              Show this help message
  quit, exit, q     Exit the program
""".format(self.structure_budget, self.top_k, 'on' if self.use_mmr else 'off', self.mmr_lambda, 'on' if self.use_research else 'off'))

    def interactive_mode(self):
        """Run interactive CLI mode."""
        style_count = self.config.get('style_chunks', self.config['num_chunks'])
        research_count = self.config.get('research_chunks', 0)
        
        print("=" * 60)
        print("RAG Essay Writer - Interactive Mode")
        print("=" * 60)
        print("\nType 'help' for all commands, or just enter your essay topic.")
        print(f"Index: {style_count} style chunks, {research_count} research chunks")
        print(f"Settings: {self.top_k} chunks per type, MMR {'on' if self.use_mmr else 'off'}, research {'on' if self.use_research else 'off'}")
        if self.excluded_sources:
            print(f"Excluded sources: {len(self.excluded_sources)}")
        print()

        while True:
            try:
                user_input = input(">> ").strip()

                if not user_input:
                    continue

                # Parse command
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                # Exit commands
                if cmd in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                # Help
                if cmd == 'help':
                    self.print_help()
                    continue

                # Regenerate
                if cmd in ['regenerate', 'r']:
                    if not self.last_prompt:
                        print("No previous essay to regenerate. Generate one first.\n")
                        continue
                    print(f"Regenerating essay on: {self.last_prompt}\n")
                    self._generate_and_display(self.last_prompt)
                    continue

                # Edit/revise
                if cmd in ['edit', 'e']:
                    if not self.last_essay:
                        print("No previous essay to edit. Generate one first.\n")
                        continue
                    if not args:
                        print("Please provide feedback for the revision.")
                        print("Example: edit Make it more concise and add more specific examples\n")
                        continue
                    print(f"Revising essay with feedback: {args}\n")
                    self._generate_and_display(
                        self.last_prompt,
                        revision_feedback=args,
                        previous_essay=self.last_essay
                    )
                    continue

                # Set chunks
                if cmd == 'chunks':
                    try:
                        n = int(args)
                        if n < 1:
                            raise ValueError()
                        self.top_k = n
                        print(f"Now retrieving top {n} chunks.\n")
                    except ValueError:
                        print("Please provide a positive integer. Example: chunks 12\n")
                    continue

                # MMR toggle
                if cmd == 'mmr':
                    if args.lower() == 'on':
                        self.use_mmr = True
                        print("MMR diversity retrieval enabled.\n")
                    elif args.lower() == 'off':
                        self.use_mmr = False
                        print("Standard retrieval enabled (no diversity weighting).\n")
                    else:
                        print("Usage: mmr on/off\n")
                    continue

                # Lambda setting
                if cmd == 'lambda':
                    try:
                        l = float(args)
                        if not 0 <= l <= 1:
                            raise ValueError()
                        self.mmr_lambda = l
                        print(f"MMR lambda set to {l:.2f}")
                        print(f"  (1.0 = pure relevance, 0.0 = pure diversity)\n")
                    except ValueError:
                        print("Please provide a number between 0 and 1. Example: lambda 0.6\n")
                    continue

                # Toggle indexed research retrieval
                if cmd == 'use-research':
                    if args.lower() == 'on':
                        self.use_research = True
                        print("Indexed research retrieval enabled.\n")
                    elif args.lower() == 'off':
                        self.use_research = False
                        print("Indexed research retrieval disabled (style sources only).\n")
                    else:
                        status = 'on' if self.use_research else 'off'
                        print(f"Research retrieval is currently: {status}")
                        print("Usage: use-research on/off\n")
                    continue

                # Load single PDF
                if cmd == 'pdf':
                    if not args:
                        print("Please provide a PDF path. Example: pdf research/paper.pdf\n")
                        continue
                    if not PDF_SUPPORT:
                        print("PDF support not available. Install PyMuPDF: pip install PyMuPDF\n")
                        continue
                    # Check if already loaded
                    pdf_name = Path(args).name
                    if any(r['name'] == pdf_name for r in self.research_sources):
                        print(f"'{pdf_name}' is already loaded.\n")
                        continue
                    result = self.load_pdf(args)
                    if result:
                        self.research_sources.append(result)
                        print(f"Loaded: {result['name']} ({result['pages']} pages)")
                        print(f"Total research sources: {len(self.research_sources)}\n")
                    continue

                # Load PDFs from folder
                if cmd == 'pdfs':
                    if not args:
                        print("Please provide a folder path. Example: pdfs research/\n")
                        continue
                    if not PDF_SUPPORT:
                        print("PDF support not available. Install PyMuPDF: pip install PyMuPDF\n")
                        continue
                    print(f"\nLoading PDFs from: {args}")
                    loaded = self.load_pdf_folder(args)
                    print(f"Loaded {loaded} PDF(s). Total research sources: {len(self.research_sources)}\n")
                    continue

                # List research sources
                if cmd == 'research':
                    if not self.research_sources:
                        print("No research sources loaded.")
                        print("Use 'pdf <path>' or 'pdfs <folder>' to load PDFs.\n")
                    else:
                        print(f"\nLoaded research sources ({len(self.research_sources)}):\n")
                        for i, source in enumerate(self.research_sources, 1):
                            chars = len(source['content'])
                            print(f"  {i}. {source['name']} ({source['pages']} pages, {chars:,} chars)")
                        print()
                    continue

                # Unload a research source
                if cmd == 'unload':
                    if not args:
                        print("Please provide a source name. Example: unload paper.pdf\n")
                        continue
                    matches = [r for r in self.research_sources if args.lower() in r['name'].lower()]
                    if len(matches) == 1:
                        self.research_sources.remove(matches[0])
                        print(f"Unloaded: {matches[0]['name']}\n")
                    elif len(matches) > 1:
                        print("Multiple matches found:")
                        for m in matches:
                            print(f"  - {m['name']}")
                        print("Please be more specific.\n")
                    else:
                        print(f"No research source matching '{args}' found.\n")
                    continue

                # Clear all research sources
                if cmd == 'clear-research':
                    count = len(self.research_sources)
                    self.research_sources.clear()
                    print(f"Cleared {count} research source(s).\n")
                    continue

                # Load structure source
                if cmd == 'structure':
                    if not args:
                        print("Please provide a file path. Example: structure book_chapter.txt\n")
                        continue
                    # Check if already loaded
                    struct_name = Path(args).name
                    if any(s['name'] == struct_name for s in self.structure_sources):
                        print(f"'{struct_name}' is already loaded.\n")
                        continue
                    result = self.load_structure_source(args, max_chars=self.structure_budget)
                    if result:
                        self.structure_sources.append(result)
                        pages_info = f" ({result['pages']} pages)" if result.get('pages') else ""
                        print(f"Loaded structure source: {result['name']}{pages_info}")
                        if result.get('sampled'):
                            print(f"  Full: {result['total_paragraphs']} paragraphs, {result['total_chars']:,} chars")
                            print(f"  Sampled: {result['sampled_paragraphs']} paragraphs, {result['sampled_chars']:,} chars")
                            print(f"  (intro + middle samples + conclusion)")
                        else:
                            print(f"  {result['total_paragraphs']} paragraphs, {result['total_chars']:,} chars (full)")
                        print(f"Total structure sources: {len(self.structure_sources)}\n")
                    continue

                # List structure sources
                if cmd == 'structures':
                    if not self.structure_sources:
                        print("No structure sources loaded.")
                        print("Use 'structure <path>' to load a .txt or .pdf as a structure reference.\n")
                    else:
                        print(f"\nLoaded structure sources ({len(self.structure_sources)}):\n")
                        for i, source in enumerate(self.structure_sources, 1):
                            sampled_tag = " [sampled]" if source.get('sampled') else ""
                            pages_info = f", {source['pages']} pages" if source.get('pages') else ""
                            print(f"  {i}. {source['name']} ({source['sampled_paragraphs']} paragraphs, {source['sampled_chars']:,} chars{pages_info}){sampled_tag}")
                        print()
                    continue

                # Set structure budget
                if cmd == 'struct-budget':
                    try:
                        n = int(args)
                        if n < 5000:
                            print("Minimum budget is 5000 characters.\n")
                            continue
                        self.structure_budget = n
                        print(f"Structure budget set to {n:,} characters.")
                        if self.structure_sources:
                            print("Note: Reload structure sources to apply new budget.\n")
                        else:
                            print()
                    except ValueError:
                        print("Please provide a number. Example: struct-budget 30000\n")
                    continue

                # Unload a structure source
                if cmd == 'unload-struct':
                    if not args:
                        print("Please provide a source name. Example: unload-struct book.txt\n")
                        continue
                    matches = [s for s in self.structure_sources if args.lower() in s['name'].lower()]
                    if len(matches) == 1:
                        self.structure_sources.remove(matches[0])
                        print(f"Unloaded: {matches[0]['name']}\n")
                    elif len(matches) > 1:
                        print("Multiple matches found:")
                        for m in matches:
                            print(f"  - {m['name']}")
                        print("Please be more specific.\n")
                    else:
                        print(f"No structure source matching '{args}' found.\n")
                    continue

                # Clear all structure sources
                if cmd == 'clear-structure':
                    count = len(self.structure_sources)
                    self.structure_sources.clear()
                    print(f"Cleared {count} structure source(s).\n")
                    continue

                # Preview sources
                if cmd == 'preview':
                    if not args:
                        print("Please provide a topic. Example: preview urban transit\n")
                        continue
                    print(f"\nPreviewing sources for: {args}\n")
                    results = self.preview_sources(args)
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['filename']} (distance: {result['distance']:.3f})")
                        preview = result['text'][:150].replace('\n', ' ')
                        print(f"   {preview}...")
                        print()
                    continue

                # List sources
                if cmd == 'sources':
                    style_sources, research_sources = self.get_sources_by_type()
                    print(f"\nIndexed source files:")
                    if style_sources:
                        print(f"\n  Style sources ({len(style_sources)}):")
                        for s in style_sources:
                            excluded = " [EXCLUDED]" if s in self.excluded_sources else ""
                            print(f"    - {s}{excluded}")
                    if research_sources:
                        print(f"\n  Research sources ({len(research_sources)}):")
                        for s in research_sources:
                            excluded = " [EXCLUDED]" if s in self.excluded_sources else ""
                            print(f"    - {s}{excluded}")
                    if not style_sources and not research_sources:
                        print("  No sources indexed.")
                    print()
                    continue

                # Exclude source
                if cmd == 'exclude':
                    if not args:
                        print("Please provide a filename. Example: exclude file.txt\n")
                        continue
                    # Find matching source
                    sources = self.get_all_sources()
                    matches = [s for s in sources if args.lower() in s.lower()]
                    if len(matches) == 1:
                        self.excluded_sources.add(matches[0])
                        print(f"Excluded: {matches[0]}\n")
                    elif len(matches) > 1:
                        print("Multiple matches found:")
                        for m in matches:
                            print(f"  - {m}")
                        print("Please be more specific.\n")
                    else:
                        print(f"No source file matching '{args}' found.\n")
                    continue

                # Include source
                if cmd == 'include':
                    if not args:
                        print("Please provide a filename. Example: include file.txt\n")
                        continue
                    matches = [s for s in self.excluded_sources if args.lower() in s.lower()]
                    if len(matches) == 1:
                        self.excluded_sources.remove(matches[0])
                        print(f"Re-included: {matches[0]}\n")
                    elif len(matches) > 1:
                        print("Multiple matches found:")
                        for m in matches:
                            print(f"  - {m}")
                        print("Please be more specific.\n")
                    else:
                        print(f"'{args}' is not in the excluded list.\n")
                    continue

                # Show excluded
                if cmd == 'excluded':
                    if not self.excluded_sources:
                        print("No sources are currently excluded.\n")
                    else:
                        print("\nCurrently excluded sources:")
                        for s in sorted(self.excluded_sources):
                            print(f"  - {s}")
                        print()
                    continue

                # Clear excluded
                if cmd == 'clear-excluded':
                    self.excluded_sources.clear()
                    print("All sources re-included.\n")
                    continue

                # History
                if cmd == 'history':
                    if not self.essay_history:
                        print("No essays generated in this session yet.\n")
                    else:
                        print(f"\nSession history ({len(self.essay_history)} essays):\n")
                        for i, entry in enumerate(self.essay_history, 1):
                            print(f"{i}. [{entry['type']}] {entry['prompt'][:60]}...")
                            print(f"   Generated at: {entry['timestamp']}")
                            print()
                    continue

                # Last essay
                if cmd == 'last':
                    if not self.last_essay:
                        print("No essay generated yet.\n")
                    else:
                        print("=" * 60)
                        print(f"LAST ESSAY - Topic: {self.last_prompt}")
                        print("=" * 60)
                        print()
                        print(self.last_essay)
                        print()
                        print("=" * 60)
                        print()
                    continue

                # Save
                if cmd == 'save':
                    if not self.last_essay:
                        print("No essay to save. Generate one first.\n")
                        continue
                    filename = input("Filename (e.g., essay.txt): ").strip()
                    if filename:
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(self.last_essay)
                        print(f"Saved to {filename}\n")
                    else:
                        print("Save cancelled.\n")
                    continue

                # Settings
                if cmd == 'settings':
                    style_count = self.config.get('style_chunks', self.config['num_chunks'])
                    research_count = self.config.get('research_chunks', 0)
                    print("\nCurrent Settings:")
                    print(f"  Chunks to retrieve (per type): {self.top_k}")
                    print(f"  MMR diversity: {'on' if self.use_mmr else 'off'}")
                    print(f"  MMR lambda: {self.mmr_lambda:.2f}")
                    print(f"  Use indexed research: {'on' if self.use_research else 'off'}")
                    print(f"  Structure budget: {self.structure_budget:,} chars")
                    print(f"\nIndexed Sources:")
                    print(f"  Style chunks (transcripts): {style_count}")
                    print(f"  Research chunks (PDFs): {research_count}")
                    print(f"  Excluded sources: {len(self.excluded_sources)}")
                    print(f"\nSession Sources:")
                    print(f"  Structure sources: {len(self.structure_sources)}")
                    print(f"  Additional research: {len(self.research_sources)}")
                    print(f"  Essays generated: {len(self.essay_history)}")
                    print()
                    continue

                # Otherwise, treat as essay prompt
                print()
                self._generate_and_display(user_input)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

    def _generate_and_display(self, prompt: str, revision_feedback: Optional[str] = None,
                              previous_essay: Optional[str] = None):
        """Generate essay and display with options."""
        import datetime

        essay, style_sources, research_sources = self.generate_essay(
            prompt,
            revision_feedback=revision_feedback,
            previous_essay=previous_essay
        )

        # Update state
        self.last_prompt = prompt
        self.last_essay = essay
        self.last_sources = style_sources
        self.last_research_sources = research_sources

        # Add to history
        entry_type = "revision" if revision_feedback else "new"
        self.essay_history.append({
            'prompt': prompt,
            'essay': essay,
            'style_sources': style_sources,
            'research_sources': research_sources,
            'type': entry_type,
            'feedback': revision_feedback,
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S")
        })

        # Display
        print("=" * 60)
        print("GENERATED ESSAY")
        print("=" * 60)
        print()
        print(essay)
        print()
        print("=" * 60)

        # Show style sources
        if style_sources:
            print(f"\nStyle sources used ({len(style_sources)}):")
            source_files = {}
            for s in style_sources:
                source_files[s['filename']] = source_files.get(s['filename'], 0) + 1
            for fname, count in source_files.items():
                print(f"  - {fname} ({count} chunk{'s' if count > 1 else ''})")

        # Show indexed research sources
        if research_sources:
            print(f"\nResearch sources used ({len(research_sources)}):")
            research_files = {}
            for r in research_sources:
                research_files[r['filename']] = research_files.get(r['filename'], 0) + 1
            for fname, count in research_files.items():
                print(f"  - {fname} ({count} chunk{'s' if count > 1 else ''})")

        # Show structure sources
        if self.structure_sources:
            print(f"\nStructure sources used ({len(self.structure_sources)}):")
            for s in self.structure_sources:
                sampled_info = " [sampled]" if s.get('sampled') else ""
                pages_info = f", {s['pages']} pages" if s.get('pages') else ""
                print(f"  - {s['name']} ({s['sampled_paragraphs']} paragraphs{pages_info}){sampled_info}")

        # Show dynamically loaded research (legacy)
        if self.research_sources:
            print(f"\nAdditional research sources ({len(self.research_sources)}):")
            for r in self.research_sources:
                print(f"  - {r['name']} ({r['pages']} pages)")

        print()
        print("Commands: 'edit <feedback>' to revise, 'regenerate' for new version, 'save' to save")
        print()


def main():
    """Main entry point."""
    import sys

    try:
        writer = RAGWriter()

        if len(sys.argv) > 1:
            # Single prompt mode
            prompt = " ".join(sys.argv[1:])
            essay, _, _ = writer.generate_essay(prompt)
            print(essay)
        else:
            # Interactive mode
            writer.interactive_mode()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
