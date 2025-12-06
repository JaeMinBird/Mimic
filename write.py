#!/usr/bin/env python3
"""RAG-powered essay writer using Claude with editing and advanced retrieval."""

import os
import json
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

try:
    import requests
    WEB_SUPPORT = True
except ImportError:
    WEB_SUPPORT = False


def load_settings() -> Dict:
    """Load settings from settings.json, creating default if not exists."""
    settings_path = Path("settings.json")
    
    default_settings = {
        "retrieval": {
            "chunks": 8,
            "mmr_enabled": True,
            "mmr_lambda": 0.7,
            "deduplicate": True,
            "dedup_threshold": 0.85,
            "use_research": True
        },
        "web_search": {
            "num_results": 5
        },
        "enhancement": {
            "auto_refine": False
        },
        "structure": {
            "budget": 20000
        }
    }
    
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                loaded = json.load(f)
                # Merge with defaults (in case new settings were added)
                for category, values in default_settings.items():
                    if category not in loaded:
                        loaded[category] = values
                    elif isinstance(values, dict):
                        for key, val in values.items():
                            if key not in loaded[category]:
                                loaded[category][key] = val
                return loaded
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse settings.json ({e}), using defaults")
            return default_settings
    else:
        # Create default settings file
        with open(settings_path, 'w') as f:
            json.dump(default_settings, f, indent=4)
        print("Created settings.json with default values")
        return default_settings


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

        # Load settings from settings.json
        settings = load_settings()
        
        # Session state (initialized from settings file)
        self.top_k = settings['retrieval']['chunks']
        self.use_mmr = settings['retrieval']['mmr_enabled']
        self.mmr_lambda = settings['retrieval']['mmr_lambda']
        self.use_research = settings['retrieval']['use_research']
        self.auto_refine = settings['enhancement']['auto_refine']
        self.deduplicate = settings['retrieval']['deduplicate']
        self.dedup_threshold = settings['retrieval']['dedup_threshold']
        self.web_results = settings['web_search']['num_results']
        self.structure_budget = settings['structure']['budget']
        self.last_prompt = None
        self.last_context = None  # User-provided context for last essay
        self.last_essay = None
        self.last_sources = None
        self.last_research_sources = None
        self.essay_history = []  # Track all essays in session
        self.excluded_sources = set()  # Sources to exclude from retrieval

        # Style analysis
        self.style_profile = None  # Analyzed style characteristics

        # Web research
        self.web_context = None  # Last web search results

        # Research sources (PDFs for information, not style)
        self.research_sources = []  # List of {'name': str, 'content': str}

        # Structure sources (for essay organization, not voice)
        self.structure_sources = []  # List of {'name': str, 'content': str}

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

    def deduplicate_chunks(self, chunks: List[Dict], threshold: float = 0.85) -> List[Dict]:
        """Remove near-duplicate chunks based on embedding similarity.
        
        Keeps the first (most relevant) chunk when duplicates are found.
        """
        if not chunks or self.embeddings is None:
            return chunks

        # Get indices for the chunks
        chunk_indices = []
        for chunk in chunks:
            # Find the index in metadata
            for i, meta in enumerate(self.metadata):
                if (meta['filename'] == chunk['filename'] and 
                    meta['chunk_index'] == chunk['chunk_index']):
                    chunk_indices.append(i)
                    break

        if len(chunk_indices) != len(chunks):
            return chunks  # Fallback if we can't find indices

        # Check similarity between chunks
        deduplicated = [chunks[0]]
        kept_indices = [chunk_indices[0]]

        for i in range(1, len(chunks)):
            chunk_embedding = self.embeddings[chunk_indices[i]]
            is_duplicate = False

            for kept_idx in kept_indices:
                kept_embedding = self.embeddings[kept_idx]
                similarity = self._cosine_similarity(chunk_embedding, kept_embedding)

                if similarity > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(chunks[i])
                kept_indices.append(chunk_indices[i])

        return deduplicated

    def analyze_style(self, num_samples: int = 15) -> str:
        """Analyze style sources and create a detailed style profile.
        
        Uses Claude to extract specific patterns from style sources.
        """
        print("Analyzing style sources...")

        # Get style chunks directly from metadata (no semantic search)
        # This ensures we actually get style sources regardless of query matching
        style_chunks = [
            {**meta, 'idx': i} 
            for i, meta in enumerate(self.metadata) 
            if meta.get('source_type') == 'style' and meta['filename'] not in self.excluded_sources
        ]

        if not style_chunks:
            print("No style sources found to analyze.")
            return None

        # Group by filename to ensure diversity across sources
        from collections import defaultdict
        chunks_by_file = defaultdict(list)
        for chunk in style_chunks:
            chunks_by_file[chunk['filename']].append(chunk)

        # Sample evenly from each source file
        samples = []
        files = list(chunks_by_file.keys())
        samples_per_file = max(1, num_samples // len(files))
        
        import random
        for filename in files:
            file_chunks = chunks_by_file[filename]
            # Take evenly spaced samples from each file
            if len(file_chunks) <= samples_per_file:
                samples.extend(file_chunks)
            else:
                step = len(file_chunks) // samples_per_file
                indices = [i * step for i in range(samples_per_file)]
                samples.extend([file_chunks[i] for i in indices])

        # Limit total samples
        if len(samples) > num_samples * 2:
            samples = random.sample(samples, num_samples * 2)

        print(f"Sampled {len(samples)} chunks from {len(files)} style sources.")

        # Build sample text
        sample_text = "\n\n---\n\n".join([
            f"[From {s['filename']}]\n{s['text']}" for s in samples
        ])

        # Ask Claude to analyze the style
        analysis_prompt = """Analyze these writing samples and extract a detailed style profile. Be specific and concrete.

WRITING SAMPLES:
{samples}

Provide a comprehensive style analysis covering:

1. SENTENCE STRUCTURE
   - Average sentence length (short/medium/long)
   - Sentence variety patterns
   - Use of fragments or run-ons
   - Punctuation habits (semicolons, dashes, parentheticals)

2. VOCABULARY & DICTION
   - Formality level (casual/conversational/formal/academic)
   - Characteristic words or phrases
   - Technical vs accessible language
   - Any repeated expressions or verbal tics

3. RHETORICAL TECHNIQUES
   - Use of questions (rhetorical, direct)
   - Analogies and metaphors style
   - How arguments are structured
   - Use of examples and evidence

4. PARAGRAPH PATTERNS
   - Typical paragraph length
   - How paragraphs begin and end
   - Transition patterns between ideas

5. VOICE & TONE
   - Overall persona (authoritative, humble, provocative, etc.)
   - Relationship with reader (distant, conversational, challenging)
   - Emotional register
   - Any distinctive quirks

6. SPECIFIC PATTERNS TO REPLICATE
   - List 5-7 concrete, imitable patterns
   - Include example phrases or structures

Be extremely specific. Instead of "uses vivid language," say "frequently uses unexpected adjective-noun pairings like 'ambitious silence' or 'reluctant clarity'."
""".format(samples=sample_text)

        print("Extracting style patterns with Claude...")
        message = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            messages=[{"role": "user", "content": analysis_prompt}]
        )

        self.style_profile = message.content[0].text
        print("Style profile created.\n")
        return self.style_profile

    def web_search(self, query: str, num_results: Optional[int] = None) -> Optional[str]:
        """Search the web for current information on a topic.
        
        Uses DuckDuckGo HTML search (no API key required).
        """
        num_results = num_results or self.web_results
        if not WEB_SUPPORT:
            print("Web search requires 'requests' package: pip install requests")
            return None

        print(f"Searching web for: {query}")

        try:
            # Use DuckDuckGo HTML search
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # DuckDuckGo lite version for simpler parsing
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"Search failed with status {response.status_code}")
                return None

            # Simple extraction of result snippets
            from html.parser import HTMLParser
            
            class DDGParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results = []
                    self.in_result = False
                    self.in_snippet = False
                    self.current_title = ""
                    self.current_snippet = ""

                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    if tag == 'a' and attrs_dict.get('class') == 'result__a':
                        self.in_result = True
                    if tag == 'a' and attrs_dict.get('class') == 'result__snippet':
                        self.in_snippet = True

                def handle_endtag(self, tag):
                    if tag == 'a' and self.in_result:
                        self.in_result = False
                    if tag == 'a' and self.in_snippet:
                        self.in_snippet = False
                        if self.current_title and self.current_snippet:
                            self.results.append({
                                'title': self.current_title.strip(),
                                'snippet': self.current_snippet.strip()
                            })
                        self.current_title = ""
                        self.current_snippet = ""

                def handle_data(self, data):
                    if self.in_result:
                        self.current_title += data
                    if self.in_snippet:
                        self.current_snippet += data

            parser = DDGParser()
            parser.feed(response.text)

            if not parser.results:
                print("No results found.")
                return None

            # Format results
            results_text = []
            for i, r in enumerate(parser.results[:num_results], 1):
                results_text.append(f"{i}. {r['title']}\n   {r['snippet']}")

            web_context = f"WEB SEARCH RESULTS FOR: {query}\n\n" + "\n\n".join(results_text)
            self.web_context = web_context
            print(f"Found {min(len(parser.results), num_results)} results.\n")
            return web_context

        except Exception as e:
            print(f"Web search error: {e}")
            return None

    def critique_essay(self, essay: str, style_context: str) -> str:
        """Use Claude Opus to critique an essay for style authenticity.
        
        Returns specific, actionable feedback for improvement.
        """
        print("Critiquing essay with Claude Opus...")

        critique_prompt = f"""You are a writing style expert. Compare this essay against the style sources and identify specific ways the essay fails to match the authentic voice.

STYLE SOURCES (the target voice to match):
{style_context[:15000]}

ESSAY TO CRITIQUE:
{essay}

Provide a detailed critique focusing on:

1. VOICE MISMATCHES
   - Where does the essay sound generic or AI-like?
   - What phrases feel out of character?
   - Where is the vocabulary too formal/informal compared to sources?

2. STRUCTURAL DIFFERENCES  
   - How do paragraph patterns differ?
   - Are sentences too uniform in length?
   - Missing rhetorical techniques from the source?

3. SPECIFIC FIXES (most important)
   - List 5-7 concrete changes to make
   - Be extremely specific: "Change X to Y" or "Add Z technique in paragraph N"
   - Focus on changes that will make the biggest authenticity improvement

Be harsh and specific. Vague feedback like "make it more conversational" is useless. 
Instead say: "Paragraph 3 uses 'furthermore' and 'additionally' - the source author never uses these transition words. Replace with shorter sentences that imply connection, or use 'And' to start sentences."
"""

        message = self.client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": critique_prompt}]
        )

        critique = message.content[0].text
        print("Critique complete.\n")
        return critique

    def refine_essay(self, essay: str, prompt: str, style_context: str) -> str:
        """Refine an essay based on Opus critique.
        
        Uses the full critique → revise pipeline.
        """
        # Get critique from Opus
        critique = self.critique_essay(essay, style_context)

        print("Refining essay based on critique...")

        # Revise with Sonnet
        revision_prompt = f"""Revise this essay based on the style critique below. Make every suggested change.

ORIGINAL ESSAY:
{essay}

STYLE CRITIQUE (follow these instructions precisely):
{critique}

STYLE REFERENCE:
{style_context[:10000]}

Rewrite the essay incorporating ALL the critique's suggestions. The goal is perfect style authenticity."""

        message = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=5000,
            messages=[{"role": "user", "content": revision_prompt}]
        )

        refined = message.content[0].text
        print("Refinement complete.\n")
        return refined

    def generate_essay(self, prompt: str, top_k: Optional[int] = None,
                       max_tokens: int = 4000, revision_feedback: Optional[str] = None,
                       previous_essay: Optional[str] = None,
                       user_context: Optional[str] = None) -> Tuple[str, List[Dict], List[Dict]]:
        """Generate an essay using retrieved context.

        Returns tuple of (essay_text, style_sources_used, research_sources_used)
        """
        k = top_k or self.top_k

        # Retrieve more chunks if deduplication is enabled
        retrieve_k = k * 2 if self.deduplicate else k

        # Retrieve STYLE chunks (for voice/tone)
        print(f"Retrieving style context (top {k} chunks, MMR={'on' if self.use_mmr else 'off'}, dedup={'on' if self.deduplicate else 'off'})...")
        if self.use_mmr:
            style_results = self.retrieve_mmr_by_type(prompt, 'style', top_k=retrieve_k, lambda_param=self.mmr_lambda)
        else:
            style_results = self.retrieve_by_type(prompt, 'style', top_k=retrieve_k)

        # Deduplicate if enabled
        if self.deduplicate and len(style_results) > k:
            style_results = self.deduplicate_chunks(style_results, self.dedup_threshold)[:k]

        # Retrieve RESEARCH chunks (for facts/information) from index
        research_results = []
        if self.use_research and self.config.get('research_chunks', 0) > 0:
            print(f"Retrieving research context (top {k} chunks)...")
            if self.use_mmr:
                research_results = self.retrieve_mmr_by_type(prompt, 'research', top_k=retrieve_k, lambda_param=self.mmr_lambda)
            else:
                research_results = self.retrieve_by_type(prompt, 'research', top_k=retrieve_k)

            # Deduplicate research too
            if self.deduplicate and len(research_results) > k:
                research_results = self.deduplicate_chunks(research_results, self.dedup_threshold)[:k]

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
        has_style_profile = bool(self.style_profile)
        has_web = bool(self.web_context)

        if has_structure or has_research:
            distinctions = ["- STYLE SOURCES: Use these to capture the author's voice, tone, rhetorical techniques, vocabulary, sentence structure, and way of thinking. Your writing should feel like it was written by this author."]

            if has_style_profile:
                distinctions.append("- STYLE PROFILE: This is an analyzed breakdown of the author's specific patterns. Follow these patterns precisely.")

            if has_structure:
                distinctions.append("- STRUCTURE SOURCES: Use these as a template for essay organization, section flow, how arguments are built and developed, paragraph structure, and overall narrative arc. Mimic the STRUCTURE but NOT the voice or vocabulary.")

            if has_research:
                distinctions.append("- RESEARCH SOURCES: Use these for factual information, data, and subject matter expertise. Extract relevant facts and insights, but do NOT let these influence your writing style or structure.")

            if has_web:
                distinctions.append("- WEB RESEARCH: Current information from the web. Use for timely facts and recent developments.")

            system_prompt = f"""You are an essay writer that combines elements from multiple source types.

IMPORTANT DISTINCTIONS:
{chr(10).join(distinctions)}

Your essay should:
1. Sound like the author from the STYLE SOURCES (voice, tone, word choice)
{('2. Follow the STYLE PROFILE patterns precisely' + chr(10)) if has_style_profile else ''}{('3. ' if has_style_profile else '2. ') + 'Be organized like the STRUCTURE SOURCES (sections, flow, argument development)' + chr(10) if has_structure else ''}{'4. ' if has_style_profile and has_structure else '3. ' if has_style_profile or has_structure else '2. '}Incorporate facts from RESEARCH SOURCES and WEB RESEARCH if provided"""
        else:
            style_profile_instruction = """

Additionally, a STYLE PROFILE with specific patterns has been provided. Follow these patterns precisely to achieve authentic style matching.""" if has_style_profile else ""

            system_prompt = f"""You are an essay writer that mimics the writing style, voice, and analytical approach of the source material provided.

Carefully analyze the voice, tone, rhetorical techniques, vocabulary, sentence structure, and thematic interests present in the source material. Then write an essay on the given topic that authentically captures this distinctive style.{style_profile_instruction}

Your essay should feel like it could have been written by the original author."""

        # Build user message sections
        structure_section = f"\n\nSTRUCTURE SOURCES (mimic this organization and flow):\n{structure_context}" if structure_context else ""
        research_section = f"\n\nRESEARCH SOURCES (for information only):\n{research_context}" if research_context else ""
        style_profile_section = f"\n\nSTYLE PROFILE (follow these patterns precisely):\n{self.style_profile}" if self.style_profile else ""
        web_section = f"\n\nWEB RESEARCH (current information):\n{self.web_context}" if self.web_context else ""

        if revision_feedback and previous_essay:
            # Revision mode
            user_message = f"""Based on the following materials, revise this essay:

TOPIC: {prompt}

PREVIOUS ESSAY:
{previous_essay}

REVISION FEEDBACK:
{revision_feedback}

STYLE SOURCES (mimic this writing voice and tone):
{style_context}{style_profile_section}{structure_section}{research_section}{web_section}

Please revise the essay according to the feedback while maintaining the style from STYLE SOURCES{' and structure from STRUCTURE SOURCES' if structure_context else ''}."""
            print("Revising essay with Claude Sonnet 4.5...\n")
        else:
            structure_instruction = " Follow the organizational patterns from STRUCTURE SOURCES." if structure_context else ""
            research_instruction = " Incorporate facts from RESEARCH SOURCES." if research_context else ""
            web_instruction = " Include relevant current information from WEB RESEARCH." if self.web_context else ""
            context_section = f"\n\nCONTEXT (important background for this essay):\n{user_context}" if user_context else ""
            context_instruction = " Consider the provided CONTEXT when framing your essay." if user_context else ""

            user_message = f"""Based on the following materials, write an essay on this topic:

TOPIC: {prompt}{context_section}

STYLE SOURCES (mimic this writing voice and tone):
{style_context}{style_profile_section}{structure_section}{research_section}{web_section}

Write a complete essay that captures the voice and tone from STYLE SOURCES.{structure_instruction}{research_instruction}{web_instruction}{context_instruction}"""
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

        # Auto-refine if enabled (skip for revisions)
        if self.auto_refine and not revision_feedback:
            print("Auto-refining with Opus critique...")
            essay = self.refine_essay(essay, prompt, style_context)

        return essay, style_results, research_results

    def preview_sources(self, prompt: str, top_k: Optional[int] = None) -> List[Dict]:
        """Preview which sources would be retrieved for a prompt."""
        k = top_k or self.top_k

        if self.use_mmr:
            results = self.retrieve_mmr(prompt, top_k=k, lambda_param=self.mmr_lambda)
        else:
            results = self.retrieve(prompt, top_k=k)

        return results

    def parse_write_command(self) -> Optional[Dict[str, str]]:
        """Parse the structured write command input.
        
        Prompts user for Prompt, Context, and Search fields.
        Returns dict with 'prompt', 'context', 'search' keys, or None if cancelled.
        """
        print("\n--- Write Essay ---")
        print("(Enter each field, or leave blank to skip. Type 'cancel' to abort.)\n")
        
        # Get prompt (required)
        prompt = input("Prompt: ").strip()
        if prompt.lower() == 'cancel':
            print("Cancelled.\n")
            return None
        if not prompt:
            print("Prompt is required. Cancelled.\n")
            return None
        
        # Get context (optional)
        context = input("Context: ").strip()
        if context.lower() == 'cancel':
            print("Cancelled.\n")
            return None
        
        # Get search query (optional)
        search = input("Search: ").strip()
        if search.lower() == 'cancel':
            print("Cancelled.\n")
            return None
        
        print()  # Blank line before generation starts
        
        return {
            'prompt': prompt,
            'context': context if context else None,
            'search': search if search else None
        }

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
  write             Start structured essay wizard (Prompt/Context/Search)
  regenerate, r     Regenerate the last essay with same settings

EDITING:
  edit <feedback>   Revise the last essay with your feedback
  e <feedback>      Short form of edit
  refine            Critique & refine last essay with Opus

STYLE ENHANCEMENT:
  analyze           Analyze style sources and create style profile
  profile           Show current style profile
  save-profile <n>  Save current profile to profiles/<n>.txt
  load-profile <n>  Load a saved profile
  profiles          List all saved profiles
  delete-profile    Delete a saved profile
  clear-profile     Clear the active style profile
  auto-refine       Toggle automatic Opus refinement (current: {})

WEB RESEARCH:
  web-results <N>   Set number of web results (current: {}, 1-20)
  web               Show last web search results
  clear-web         Clear web search context

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
  dedup on/off      Toggle chunk deduplication (current: {})

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
""".format(
            'on' if self.auto_refine else 'off',
            self.web_results,
            self.structure_budget, 
            self.top_k, 
            'on' if self.use_mmr else 'off', 
            self.mmr_lambda, 
            'on' if self.use_research else 'off',
            'on' if self.deduplicate else 'off'
        ))

    def interactive_mode(self):
        """Run interactive CLI mode."""
        style_count = self.config.get('style_chunks', self.config['num_chunks'])
        research_count = self.config.get('research_chunks', 0)
        
        print("=" * 60)
        print("RAG Essay Writer - Interactive Mode")
        print("=" * 60)
        print("\nType 'write' to start an essay, or 'help' for all commands.")
        print(f"Index: {style_count} style chunks, {research_count} research chunks")
        print(f"Retrieval: {self.top_k} chunks, MMR {'on' if self.use_mmr else 'off'}, dedup {'on' if self.deduplicate else 'off'}")
        print(f"Enhancement: auto-refine {'on' if self.auto_refine else 'off'}, profile {'active' if self.style_profile else 'none'}")
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

                # Write command - structured essay generation
                if cmd == 'write':
                    write_params = self.parse_write_command()
                    if write_params:
                        # Perform web search if requested
                        if write_params['search']:
                            self.web_search(write_params['search'])
                        
                        # Generate the essay
                        self._generate_and_display(
                            write_params['prompt'],
                            user_context=write_params['context']
                        )
                    continue

                # Regenerate
                if cmd in ['regenerate', 'r']:
                    if not self.last_prompt:
                        print("No previous essay to regenerate. Use 'write' first.\n")
                        continue
                    print(f"Regenerating essay on: {self.last_prompt}\n")
                    self._generate_and_display(self.last_prompt, user_context=self.last_context)
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

                # Toggle deduplication
                if cmd == 'dedup':
                    if args.lower() == 'on':
                        self.deduplicate = True
                        print("Chunk deduplication enabled.\n")
                    elif args.lower() == 'off':
                        self.deduplicate = False
                        print("Chunk deduplication disabled.\n")
                    else:
                        status = 'on' if self.deduplicate else 'off'
                        print(f"Deduplication is currently: {status}")
                        print("Usage: dedup on/off\n")
                    continue

                # Set web search results count
                if cmd == 'web-results':
                    try:
                        n = int(args)
                        if n < 1 or n > 20:
                            print("Please provide a number between 1 and 20.\n")
                            continue
                        self.web_results = n
                        print(f"Web search will now return {n} results.\n")
                    except ValueError:
                        print(f"Web results is currently: {self.web_results}")
                        print("Usage: web-results <N> (1-20)\n")
                    continue

                # Toggle auto-refine
                if cmd == 'auto-refine':
                    if args.lower() == 'on':
                        self.auto_refine = True
                        print("Auto-refine enabled (Opus will critique and refine each essay).\n")
                    elif args.lower() == 'off':
                        self.auto_refine = False
                        print("Auto-refine disabled.\n")
                    else:
                        status = 'on' if self.auto_refine else 'off'
                        print(f"Auto-refine is currently: {status}")
                        print("Usage: auto-refine on/off\n")
                    continue

                # Manual refine
                if cmd == 'refine':
                    if not self.last_essay:
                        print("No essay to refine. Generate one first.\n")
                        continue
                    # Get style context for critique
                    if self.use_mmr:
                        style_results = self.retrieve_mmr_by_type(self.last_prompt, 'style', top_k=self.top_k)
                    else:
                        style_results = self.retrieve_by_type(self.last_prompt, 'style', top_k=self.top_k)
                    style_context = "\n\n".join([r['text'] for r in style_results])
                    
                    refined = self.refine_essay(self.last_essay, self.last_prompt, style_context)
                    self.last_essay = refined
                    
                    print("=" * 60)
                    print("REFINED ESSAY")
                    print("=" * 60)
                    print()
                    print(refined)
                    print()
                    print("=" * 60)
                    print()
                    continue

                # Analyze style
                if cmd == 'analyze':
                    self.analyze_style()
                    if self.style_profile:
                        print("Style profile created. It will be used in future generations.")
                        print("Use 'profile' to view it, or 'clear-profile' to remove it.\n")
                    continue

                # Show style profile
                if cmd == 'profile':
                    if self.style_profile:
                        print("\n" + "=" * 60)
                        print("STYLE PROFILE")
                        print("=" * 60)
                        print(self.style_profile)
                        print("=" * 60 + "\n")
                    else:
                        print("No style profile active. Use 'analyze' to create one or 'load-profile <name>' to load a saved one.\n")
                    continue

                # Clear style profile
                if cmd == 'clear-profile':
                    self.style_profile = None
                    print("Style profile cleared.\n")
                    continue

                # Save style profile
                if cmd == 'save-profile':
                    if not self.style_profile:
                        print("No style profile to save. Use 'analyze' first.\n")
                        continue
                    if not args:
                        print("Please provide a name. Example: save-profile my_style\n")
                        continue
                    # Create profiles directory if needed
                    profiles_dir = Path("profiles")
                    profiles_dir.mkdir(exist_ok=True)
                    # Save profile
                    profile_name = args.strip().replace(" ", "_")
                    profile_path = profiles_dir / f"{profile_name}.txt"
                    with open(profile_path, 'w', encoding='utf-8') as f:
                        f.write(self.style_profile)
                    print(f"Style profile saved to: {profile_path}\n")
                    continue

                # Load style profile
                if cmd == 'load-profile':
                    if not args:
                        print("Please provide a profile name. Example: load-profile my_style")
                        print("Use 'profiles' to list saved profiles.\n")
                        continue
                    profile_name = args.strip().replace(" ", "_")
                    profile_path = Path("profiles") / f"{profile_name}.txt"
                    if not profile_path.exists():
                        # Try without .txt extension in case they included it
                        if args.strip().endswith('.txt'):
                            profile_path = Path("profiles") / args.strip()
                        if not profile_path.exists():
                            print(f"Profile not found: {profile_name}")
                            print("Use 'profiles' to list saved profiles.\n")
                            continue
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        self.style_profile = f.read()
                    print(f"Loaded style profile: {profile_name}\n")
                    continue

                # List saved profiles
                if cmd == 'profiles':
                    profiles_dir = Path("profiles")
                    if not profiles_dir.exists():
                        print("No saved profiles yet. Use 'save-profile <name>' after running 'analyze'.\n")
                        continue
                    profiles = list(profiles_dir.glob("*.txt"))
                    if not profiles:
                        print("No saved profiles yet. Use 'save-profile <name>' after running 'analyze'.\n")
                        continue
                    print(f"\nSaved style profiles ({len(profiles)}):\n")
                    for p in sorted(profiles):
                        size = p.stat().st_size
                        print(f"  - {p.stem} ({size:,} bytes)")
                    print()
                    continue

                # Delete a saved profile
                if cmd == 'delete-profile':
                    if not args:
                        print("Please provide a profile name. Example: delete-profile my_style\n")
                        continue
                    profile_name = args.strip().replace(" ", "_")
                    profile_path = Path("profiles") / f"{profile_name}.txt"
                    if not profile_path.exists():
                        print(f"Profile not found: {profile_name}\n")
                        continue
                    profile_path.unlink()
                    print(f"Deleted profile: {profile_name}\n")
                    continue

                # Show web context
                if cmd == 'web':
                    if self.web_context:
                        print("\n" + self.web_context + "\n")
                    else:
                        print("No web search results. Use 'write' and enter a Search query.\n")
                    continue

                # Clear web context
                if cmd == 'clear-web':
                    self.web_context = None
                    print("Web context cleared.\n")
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
                    print(f"  Deduplication: {'on' if self.deduplicate else 'off'}")
                    print(f"  Use indexed research: {'on' if self.use_research else 'off'}")
                    print(f"  Structure budget: {self.structure_budget:,} chars")
                    print(f"\nStyle Enhancement:")
                    print(f"  Style profile: {'active' if self.style_profile else 'none'}")
                    print(f"  Auto-refine (Opus): {'on' if self.auto_refine else 'off'}")
                    print(f"\nWeb Search:")
                    print(f"  Results per search: {self.web_results}")
                    print(f"  Web context: {'active' if self.web_context else 'none'}")
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

                # Unknown command
                print(f"Unknown command: {cmd}")
                print("Use 'write' to start an essay, or 'help' for all commands.\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")

    def _generate_and_display(self, prompt: str, revision_feedback: Optional[str] = None,
                              previous_essay: Optional[str] = None,
                              user_context: Optional[str] = None):
        """Generate essay and display with options."""
        import datetime

        essay, style_sources, research_sources = self.generate_essay(
            prompt,
            revision_feedback=revision_feedback,
            previous_essay=previous_essay,
            user_context=user_context
        )

        # Update state
        self.last_prompt = prompt
        self.last_context = user_context
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
