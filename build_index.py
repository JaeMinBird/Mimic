#!/usr/bin/env python3
"""Build vector index from transcript and research files."""

import os
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # Try to end at a sentence boundary
        if end < text_len:
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            boundary = max(last_period, last_newline)

            if boundary > chunk_size // 2:  # Only adjust if boundary is reasonable
                end = start + boundary + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

        if end >= text_len:
            break

    return chunks


def load_transcripts(transcripts_dir: str = "transcripts") -> List[Tuple[str, str]]:
    """Load all .txt files from transcripts directory."""
    transcripts = []
    transcripts_path = Path(transcripts_dir)

    if not transcripts_path.exists():
        print(f"Creating {transcripts_dir}/ directory...")
        transcripts_path.mkdir(exist_ok=True)
        print(f"Please add your .txt transcript files to the {transcripts_dir}/ folder")
        return []

    txt_files = list(transcripts_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {transcripts_dir}/")
        return []

    for txt_file in txt_files:
        print(f"Loading {txt_file.name}...")
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            transcripts.append((txt_file.name, content))

    return transcripts


def load_pdfs(research_dir: str = "research") -> List[Tuple[str, str]]:
    """Load all PDF files from research directory."""
    if not PDF_SUPPORT:
        print("PDF support not available. Install PyMuPDF: pip install PyMuPDF")
        return []

    documents = []
    research_path = Path(research_dir)

    if not research_path.exists():
        print(f"No {research_dir}/ directory found. Skipping research PDFs.")
        return []

    pdf_files = list(research_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No .pdf files found in {research_dir}/")
        return []

    for pdf_file in pdf_files:
        print(f"Loading {pdf_file.name}...")
        try:
            doc = fitz.open(pdf_file)
            text_parts = []

            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num}]\n{text}")

            doc.close()

            content = "\n\n".join(text_parts)
            if content.strip():
                documents.append((pdf_file.name, content))
            else:
                print(f"  Warning: No text extracted from {pdf_file.name}")

        except Exception as e:
            print(f"  Error reading {pdf_file.name}: {e}")

    return documents


def build_index(transcripts_dir: str = "transcripts",
                research_dir: str = "research",
                index_dir: str = "index",
                chunk_size: int = 1000,
                overlap: int = 200):
    """Build FAISS index from transcripts (style) and PDFs (research)."""

    # Load style sources (transcripts)
    print("Loading style sources (transcripts)...")
    transcripts = load_transcripts(transcripts_dir)
    print(f"Loaded {len(transcripts)} transcript(s)")

    # Load research sources (PDFs)
    print("\nLoading research sources (PDFs)...")
    pdfs = load_pdfs(research_dir)
    print(f"Loaded {len(pdfs)} PDF(s)")

    if not transcripts and not pdfs:
        print("\nNo sources found!")
        print(f"  - Add .txt files to {transcripts_dir}/ for style sources")
        print(f"  - Add .pdf files to {research_dir}/ for research sources")
        return

    # Chunk all sources
    print("\nChunking sources...")
    all_chunks = []
    chunk_metadata = []

    # Process style sources (transcripts)
    for filename, content in transcripts:
        chunks = chunk_text(content, chunk_size, overlap)
        print(f"  [style] {filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                'filename': filename,
                'chunk_index': i,
                'text': chunk,
                'source_type': 'style'
            })

    # Process research sources (PDFs)
    for filename, content in pdfs:
        chunks = chunk_text(content, chunk_size, overlap)
        print(f"  [research] {filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                'filename': filename,
                'chunk_index': i,
                'text': chunk,
                'source_type': 'research'
            })

    print(f"\nTotal chunks: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks created. Check your source files.")
        return

    # Load embedding model
    print("\nLoading embedding model (this may take a moment)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good quality

    # Create embeddings
    print("Creating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index
    print("\nBuilding FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(embeddings.astype('float32'))

    # Save everything
    print(f"\nSaving index to {index_dir}/...")
    Path(index_dir).mkdir(exist_ok=True)

    faiss.write_index(index, f"{index_dir}/vector.index")

    # Save embeddings for MMR retrieval
    np.save(f"{index_dir}/embeddings.npy", embeddings)

    with open(f"{index_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(chunk_metadata, f)

    # Count by source type
    style_count = sum(1 for m in chunk_metadata if m['source_type'] == 'style')
    research_count = sum(1 for m in chunk_metadata if m['source_type'] == 'research')

    with open(f"{index_dir}/config.pkl", 'wb') as f:
        pickle.dump({
            'model_name': 'all-MiniLM-L6-v2',
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_chunks': len(all_chunks),
            'style_chunks': style_count,
            'research_chunks': research_count
        }, f)

    print(f"\n✓ Index built successfully!")
    print(f"  - {len(all_chunks)} total chunks indexed")
    print(f"    • {style_count} style chunks (from transcripts)")
    print(f"    • {research_count} research chunks (from PDFs)")
    print(f"  - Dimension: {dimension}")
    print(f"  - MMR-enabled: embeddings.npy saved")


if __name__ == "__main__":
    import sys

    transcripts_dir = sys.argv[1] if len(sys.argv) > 1 else "transcripts"
    research_dir = sys.argv[2] if len(sys.argv) > 2 else "research"
    build_index(transcripts_dir=transcripts_dir, research_dir=research_dir)
