# RAG Essay Writer

A powerful Retrieval-Augmented Generation (RAG) system that generates essays in the style of your source material. Supports multiple AI providers including Anthropic, OpenAI, Google, Mistral, and Groq.

## Features

- **Style Learning**: Learns writing voice from transcript files
- **Research Integration**: Indexes large PDFs (even 400+ pages) for semantic search
- **Style Analysis**: AI-powered extraction of writing patterns for enhanced authenticity
- **Multi-Provider Support**: Switch between Anthropic, OpenAI, Google, Mistral, and Groq
- **Multi-Model Refinement**: Optional critique and revision pipeline
- **Web Research**: Integrated web search for current information
- **Structure Templates**: Use existing essays/documents as organizational templates
- **Smart Retrieval**: MMR diversity + deduplication for higher quality context
- **GUI Interface**: Modern, minimal interface for easy use

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add keys for the providers you want to use:

```
ANTHROPIC_API_KEY=sk-ant-...your-key-here
OPENAI_API_KEY=sk-...your-key-here
GOOGLE_API_KEY=...your-key-here
MISTRAL_API_KEY=...your-key-here
GROQ_API_KEY=gsk_...your-key-here
```

You only need to add keys for the providers you plan to use.

### 3. Add Your Source Files

**Style Sources** - Place `.txt` files in `transcripts/`:
```
transcripts/
  author_transcript1.txt
  author_transcript2.txt
  ...
```

**Research Sources** - Place `.pdf` files in `research/`:
```
research/
  paper.pdf
  book_chapter.pdf
  ...
```

### 4. Build the Index

```bash
python build_index.py
```

This creates embeddings for all your sources, enabling semantic search.

## Usage

### GUI Mode

```bash
python gui.py
```

The GUI provides:
- Provider and model selection (hot-swappable)
- Source management with dropdown selection
- Prompt, context, and web search inputs
- Generate, regenerate, refine, and save buttons
- Real-time console output
- Retrieval settings with visual sliders

### CLI Mode

```bash
python write.py
```

Use the `write` command to start the structured essay wizard:

```
>> write

--- Write Essay ---
(Enter each field, or leave blank to skip. Type 'cancel' to abort.)

Prompt: The evolution of democracy in ancient Athens
Context: This is for an undergraduate history course, focusing on political philosophy
Search: Athenian democracy Solon Cleisthenes reforms

Searching web for: Athenian democracy Solon Cleisthenes reforms
Found 5 results.

Retrieving style context...
Retrieving research context...
Generating essay...
```

**Fields:**
- **Prompt** (required): Your essay topic
- **Context** (optional): Background information, audience, purpose
- **Search** (optional): Web search query for current information

## Commands Reference

### Essay Generation

| Command | Description |
|---------|-------------|
| `write` | Start structured essay wizard (Prompt/Context/Search) |
| `regenerate` | Regenerate the last essay with same settings |
| `edit <feedback>` | Revise the last essay based on your feedback |
| `refine` | Critique and refine last essay |

### Style Enhancement

| Command | Description |
|---------|-------------|
| `analyze` | Analyze style sources and create detailed style profile |
| `profile` | Show the current style profile |
| `save-profile <name>` | Save profile to `profiles/<name>.txt` |
| `load-profile <name>` | Load a saved profile |
| `profiles` | List all saved profiles |
| `delete-profile <name>` | Delete a saved profile |
| `clear-profile` | Clear the active profile |
| `auto-refine on/off` | Toggle automatic refinement |
| `rules on/off/show` | Toggle writing rules |

### Retrieval Settings

| Command | Description |
|---------|-------------|
| `chunks <N>` | Set chunks to retrieve per type (default: 8) |
| `mmr on/off` | Toggle MMR diversity retrieval |
| `lambda <0-1>` | Set MMR balance (1.0=relevance, 0.0=diversity) |
| `dedup on/off` | Toggle chunk deduplication |
| `use-research on/off` | Toggle indexed research retrieval |

### Structure Sources

| Command | Description |
|---------|-------------|
| `structure <path>` | Load a .txt or .pdf as structure reference |
| `structures` | List loaded structure sources |
| `struct-budget <N>` | Set max chars for structure sampling |
| `clear-structure` | Clear all structure sources |

### Source Control

| Command | Description |
|---------|-------------|
| `sources` | List all indexed source files |
| `preview <topic>` | Preview sources for a topic |
| `exclude <file>` | Exclude a source from retrieval |
| `include <file>` | Re-include an excluded source |

### Web Search

| Command | Description |
|---------|-------------|
| `web-results <N>` | Set number of search results (1-20, default: 5) |
| `web` | Show last web search results |
| `clear-web` | Clear web search context |

### Other

| Command | Description |
|---------|-------------|
| `settings` | Show current settings |
| `history` | Show essay history |
| `last` | Display the last essay |
| `save` | Save last essay to file |
| `help` | Show all commands |
| `quit` | Exit |

## Supported Providers and Models

### Anthropic
- claude-sonnet-4-5-20250929
- claude-opus-4-20250514
- claude-3-5-sonnet-20241022
- claude-3-opus-20240229
- claude-3-haiku-20240307

### OpenAI
- gpt-4o
- gpt-4o-mini
- gpt-4-turbo
- gpt-4
- gpt-3.5-turbo
- o1-preview
- o1-mini

### Google
- gemini-2.0-flash-exp
- gemini-1.5-pro
- gemini-1.5-flash

### Mistral
- mistral-large-latest
- mistral-medium-latest
- mistral-small-latest
- codestral-latest

### Groq
- llama-3.3-70b-versatile
- llama-3.1-8b-instant
- mixtral-8x7b-32768
- gemma2-9b-it

## Advanced Workflows

### Maximum Style Authenticity

```bash
# 1. Analyze your style sources
>> analyze
Sampled 14 chunks from 14 style sources.
Style profile created.

# 2. Save for future sessions
>> save-profile narrative_voice

# 3. Enable refinement
>> auto-refine on

# 4. Increase context diversity
>> chunks 20
>> lambda 0.5

# 5. Write
>> write
Prompt: ...
```

### Using Saved Profiles

```bash
# List available profiles
>> profiles
Saved style profiles (2):
  - narrative_voice (4,521 bytes)
  - academic_tone (3,892 bytes)

# Load one
>> load-profile narrative_voice
Loaded style profile: narrative_voice
```

### Structure Templates

Use an existing essay/document as an organizational template:

```bash
>> structure essays/example_essay.pdf
Loaded structure source: example_essay.pdf (12 pages)
  Sampled: 28 paragraphs, 18,756 chars

# Now essays will follow this organizational pattern
>> write
```

## How It Works

1. **Indexing**: Sources are chunked, embedded, and stored in a FAISS index
   - Style sources (transcripts) - tagged for voice/tone retrieval
   - Research sources (PDFs) - tagged for factual retrieval

2. **Retrieval**: Semantic search retrieves relevant chunks
   - MMR ensures diversity across sources
   - Deduplication removes near-duplicate chunks
   - Filtered by source type (style vs research)

3. **Generation**: The selected model generates the essay using:
   - Style chunks - to capture voice and tone
   - Research chunks - for facts and information
   - Style profile (if active) - specific patterns to follow
   - Structure source (if loaded) - organizational template
   - Web context (if searched) - current information
   - User context (if provided) - background/purpose

4. **Refinement** (if auto-refine is on):
   - Model critiques for style authenticity
   - Model revises based on critique

## Project Structure

```
rag-essay-writer/
  write.py            # Main essay writer (CLI mode)
  gui.py              # GUI application
  build_index.py      # Index builder script
  settings.json       # Default settings (edit to change defaults)
  requirements.txt    # Python dependencies
  .env                # API keys (create from .env.example)
  .env.example        # Example environment file
  transcripts/        # Style sources (.txt)
  research/           # Research sources (.pdf)
  profiles/           # Saved style profiles
  index/              # Generated index files
    vector.index
    metadata.pkl
    embeddings.npy
    config.pkl
```

## Settings File

The `settings.json` file controls default values for all settings:

```json
{
    "retrieval": {
        "chunks": 8,
        "mmr_enabled": true,
        "mmr_lambda": 0.7,
        "deduplicate": true,
        "dedup_threshold": 0.85,
        "use_research": true
    },
    "web_search": {
        "num_results": 5
    },
    "enhancement": {
        "auto_refine": false
    },
    "structure": {
        "budget": 20000
    },
    "writing_rules": {
        "enabled": true,
        "rules": [
            "NEVER use em-dashes. Use commas, periods, or parentheses instead.",
            "AVOID overused words: crucial, pivotal, landscape, navigate, delve...",
            "..."
        ]
    }
}
```

**Settings explained:**

| Setting | Description |
|---------|-------------|
| `chunks` | Number of chunks to retrieve per source type |
| `mmr_enabled` | Whether to use MMR diversity retrieval |
| `mmr_lambda` | Balance: 1.0 = pure relevance, 0.0 = pure diversity |
| `deduplicate` | Remove near-duplicate chunks |
| `dedup_threshold` | Similarity threshold for deduplication (0.0-1.0) |
| `use_research` | Whether to include indexed research PDFs |
| `num_results` | Number of web search results to retrieve |
| `auto_refine` | Automatically refine essays |
| `budget` | Max characters for structure source sampling |
| `writing_rules.enabled` | Whether to enforce writing rules |
| `writing_rules.rules` | List of rules the AI must follow |

Note: Changes made during a session do not modify `settings.json`. On the next startup, the program reloads from the file.

### Writing Rules

The `writing_rules` section helps avoid common AI writing quirks that make essays feel artificial. Default rules include:

- No em-dashes
- No cliched openings ("In today's world...")
- No overused AI words (crucial, delve, landscape, navigate, foster...)
- No hedging phrases ("It's important to note...")
- No generic conclusions
- Natural sentence variety

You can customize these rules in `settings.json`.

## Tips

**For Better Style Matching:**
- Add more transcript files (5+ recommended)
- Run `analyze` to create a style profile
- Enable `auto-refine` for critique-based refinement
- Use higher `chunks` (16-20) with lower `lambda` (0.4-0.5)

**For Large Research PDFs:**
- PDFs are automatically chunked and indexed
- Only semantically relevant sections are retrieved
- Rebuild index after adding new PDFs

**For Specific Essays:**
- Use the Context field to specify audience/purpose
- Load a structure source for organizational templates
- Use Search field for current/timely information

## Troubleshooting

**"No style sources found to analyze"**
- Rebuild index: `python build_index.py`
- Make sure `.txt` files are in `transcripts/`

**"Index directory not found"**
- Run `python build_index.py` first

**"API_KEY not found"**
- Create `.env` file from `.env.example`
- Add your API key for the provider you want to use

**Essays feel generic**
- Run `analyze` to create style profile
- Enable `auto-refine on`
- Add more style source transcripts
