# RAG Essay Writer

A fast Retrieval-Augmented Generation (RAG) system that generates essays in the style of your source material using Claude Sonnet 4.5.

## Features

- Fast vector search using FAISS
- Quality embeddings with sentence-transformers
- Claude Sonnet 4.5 for natural essay generation
- Simple CLI interface
- **Style sources**: Learns writing style from transcript files
- **Research sources**: Indexes large PDFs (even 400+ pages) for semantic search

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Anthropic API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key

### 3. Configure Environment

Create a `.env` file in this directory:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
ANTHROPIC_API_KEY=sk-ant-...your-key-here
```

### 4. Add Your Source Files

**Style Sources (transcripts)** - Place `.txt` files in the `transcripts/` folder. These will be used to capture writing voice and tone:

```
transcripts/
  ├── video1_transcript.txt
  ├── video2_transcript.txt
  └── video3_transcript.txt
```

**Research Sources (PDFs)** - Place `.pdf` files in the `research/` folder. These will be chunked and indexed for semantic search, so even 400+ page documents work efficiently:

```
research/
  ├── paper1.pdf
  ├── book_chapter.pdf
  └── long_report.pdf
```

The system will automatically create these folders if they don't exist.

### 5. Build the Index

Run the indexing script to process your sources:

```bash
python build_index.py
```

This will:
- Load all `.txt` files from `transcripts/` as **style sources**
- Load all `.pdf` files from `research/` as **research sources**
- Chunk them intelligently (with page markers for PDFs)
- Create embeddings for semantic search
- Build a FAISS vector index
- Save to `index/` directory

You can also specify custom directories:
```bash
python build_index.py my_transcripts my_research
```

## Usage

### Interactive Mode

Run the essay writer in interactive mode:

```bash
python rag_writer.py
```

Then type your essay prompts and press Enter. Type `quit` to exit.

### Single Prompt Mode

Generate an essay from a single prompt:

```bash
python rag_writer.py "The impact of digital media on modern storytelling"
```

## How It Works

1. **Indexing**: Your sources are split into chunks, embedded, and indexed with FAISS:
   - **Style sources** (transcripts): Used to capture writing voice and tone
   - **Research sources** (PDFs): Used for factual information and data

2. **Retrieval**: When you provide a prompt, the system retrieves relevant chunks from both source types separately, keeping their purposes distinct

3. **Generation**: Claude Sonnet 4.5 receives:
   - Style chunks → to mimic writing voice, tone, vocabulary
   - Research chunks → to incorporate relevant facts and information
   - Structure sources (if loaded) → to follow organizational patterns

4. **Output**: Claude generates an essay that captures the voice of your style sources while incorporating facts from your research sources

## Project Structure

```
moose-rag/
├── build_index.py      # Index builder script
├── rag_writer.py       # Main RAG essay generator
├── requirements.txt    # Python dependencies
├── .env               # API key (create this)
├── .env.example       # Environment template
├── transcripts/       # Style sources (.txt) go here
├── research/          # Research sources (.pdf) go here
└── index/            # Generated index files
    ├── vector.index  # FAISS vector index
    ├── metadata.pkl  # Chunk metadata (with source_type tags)
    ├── embeddings.npy # Embeddings for MMR retrieval
    └── config.pkl    # Index configuration
```

## Customization

### Chunk Size

Edit `build_index.py` to adjust chunking:

```python
build_index(chunk_size=1000, overlap=200)
```

### Number of Retrieved Chunks

Edit `rag_writer.py` to retrieve more/fewer chunks:

```python
writer.generate_essay(prompt, top_k=8)  # Adjust top_k
```

### Max Essay Length

```python
writer.generate_essay(prompt, max_tokens=4000)  # Adjust max_tokens
```

## Tips

**Style Sources (transcripts)**
- More transcripts = better style capture
- Longer transcripts help the system understand the voice
- The system works best with 5+ transcript files

**Research Sources (PDFs)**
- Large PDFs (400+ pages) are fully supported
- Content is chunked and semantically searched
- Only relevant sections are retrieved, not the whole document
- Add research to the `research/` folder and rebuild the index

## Troubleshooting

**"Index directory not found"**
- Run `python build_index.py` first

**"ANTHROPIC_API_KEY not found"**
- Create `.env` file with your API key

**"No .txt files found"**
- Add transcript files to `transcripts/` folder

**Poor style matching**
- Add more transcript files
- Try adjusting `top_k` to retrieve more context
- Ensure transcripts represent the author's typical style
