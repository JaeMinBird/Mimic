# Hybrid RAG Pipeline Architecture

## Core Principle

**Raw corpus text is NEVER passed to the writer.**

RAG is used only for idea extraction. Style comes from a static Voice Specification.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     OFFLINE (Run Once)                          │
├─────────────────────────────────────────────────────────────────┤
│  Corpus Samples  ───►  Voice Analysis  ───►  VOICE SPEC        │
│                        (Gemini Pro)          (400-800 tokens)   │
│                                              Static artifact    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RUNTIME PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 1: RAG → Notes (Cheap Model)                       │  │
│  │ • Retrieve relevant passages                              │  │
│  │ • Extract ideas, arguments, metaphors, examples           │  │
│  │ • Compress to bullet notes                                │  │
│  │ • DISCARD raw text                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 2: Outline (Cheap Model)                           │  │
│  │ • Thesis statement                                        │  │
│  │ • Section plan                                            │  │
│  │ • Intent per section                                      │  │
│  │ • NO PROSE YET                                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 3: Draft A (Claude Opus, temp=0.9)                 │  │
│  │ Inputs: Voice Spec + Outline + Notes                      │  │
│  │ Style: EXPRESSIVE, take risks                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 4: Draft B (Claude Opus, temp=0.3)                 │  │
│  │ Inputs: Voice Spec + Outline + Notes                      │  │
│  │ Style: CONTROLLED, strict discipline                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 5: Synthesis (Claude Opus, temp=0.5)               │  │
│  │ • Compare A and B side-by-side                            │  │
│  │ • Keep strongest sentences from either                    │  │
│  │ • Delete redundancy                                       │  │
│  │ • Enforce paragraph flow                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 6: Editorial Critique (GPT-5/4o)                   │  │
│  │ • Flag weak arguments                                     │  │
│  │ • Flag padding                                            │  │
│  │ • Flag flat sections                                      │  │
│  │ • NO REWRITES - flags only                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ STAGE 7: Targeted Revision (Claude Opus)                 │  │
│  │ • Fix ONLY flagged areas                                  │  │
│  │ • Preserve everything else                                │  │
│  │ • Surgical precision                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                        FINAL ESSAY                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Stack

| Role | Model | Purpose |
|------|-------|---------|
| Notes extraction | Claude Haiku | Cheap, fast idea extraction |
| Outline | Claude Haiku | Cheap structural planning |
| Writer | Claude Opus 4 | Quality drafting and synthesis |
| Critic | GPT-4o / GPT-5 | External perspective, finds weaknesses |
| Voice Analysis | Gemini Pro (1M) | Long-context corpus analysis |

---

## Voice Specification

The Voice Spec is a **static artifact** created offline. It contains abstract rules, not examples:

```
### Sentence Rhythm
- Alternates between 8-15 word sentences and 20-30 word sentences
- Rarely uses sentences under 5 words except for emphasis
- Semicolons appear roughly once per paragraph

### Paragraph Shape
- Opens with assertion, not question
- 4-7 sentences typical
- Closing sentence often shorter, punchy

### Transitions
- Implicit connections preferred over explicit ("However", "Moreover")
- New paragraphs often start with concrete detail, not abstract claim
- Uses "And" to start sentences frequently

### Rhetorical Moves
- Builds argument through accumulation of examples
- Deploys rhetorical questions mid-paragraph, never as openers
- Qualifies claims with specific exceptions

### What to Avoid
- Em-dashes
- "It is important to note"
- Starting with "In today's world"
- Generic conclusions

### Emotional Modulation
- Restraint as default, intensity reserved for key moments
- Irony preferred over direct criticism
- Enthusiasm expressed through specificity, not exclamation
```

**Key**: The writer receives these RULES, never the original samples.

---

## What RAG Does and Doesn't Do

### RAG IS USED FOR:
- Retrieving relevant passages
- Extracting ideas, arguments, metaphors, examples
- Compressing to notes

### RAG IS BANNED FROM:
- Drafting
- Line editing
- Final synthesis
- Style transfer

The retrieved passages are processed into notes and then **discarded**. The writer never sees them.

---

## Usage

### 1. Create Voice Specification (once)

```bash
python pipeline.py analyze
```

This analyzes your corpus and creates `voice_spec.txt`.

### 2. Run Full Pipeline

```bash
python pipeline.py generate
```

### 3. Run Quick Pipeline (faster, single draft)

```bash
python pipeline.py quick
```

---

## Files

| File | Purpose |
|------|---------|
| `system_prompts.py` | All stage prompts and templates |
| `pipeline.py` | Pipeline implementation |
| `voice_spec.txt` | Your static voice specification |
| `write.py` | Original RAG system (used for retrieval) |
| `gui.py` | GUI interface |

---

## Key Rules

1. **Raw corpus text is never passed to the writer**
2. Voice comes from abstraction, not retrieval
3. Dual drafts (expressive + controlled) ensure quality
4. External critic provides unbiased feedback
5. Revisions are surgical, not wholesale
