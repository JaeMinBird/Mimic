#!/usr/bin/env python3
"""Hybrid RAG Pipeline - Quality-first multi-stage essay generation.

Architecture:
- RAG for idea extraction ONLY (notes, not style)
- Static Voice Specification (never retrieved)
- Multi-draft synthesis with different temperatures
- Cross-model editorial critique

Key Rule: Raw corpus text is NEVER passed to the writer.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

from system_prompts import (
    VOICE_SPEC_TEMPLATE,
    STAGE_1_NOTES_SYSTEM, STAGE_1_NOTES_USER,
    STAGE_2_OUTLINE_SYSTEM, STAGE_2_OUTLINE_USER,
    STAGE_3_DRAFT_A_SYSTEM, STAGE_3_DRAFT_A_USER,
    STAGE_4_DRAFT_B_SYSTEM, STAGE_4_DRAFT_B_USER,
    STAGE_5_SYNTHESIS_SYSTEM, STAGE_5_SYNTHESIS_USER,
    STAGE_6_CRITIQUE_SYSTEM, STAGE_6_CRITIQUE_USER,
    STAGE_7_REVISION_SYSTEM, STAGE_7_REVISION_USER,
    VOICE_ANALYSIS_SYSTEM, VOICE_ANALYSIS_USER,
    format_prompt,
)


@dataclass
class PipelineConfig:
    """Configuration for the hybrid pipeline."""
    
    # Model assignments
    notes_model: str = "claude-3-haiku-20240307"  # Cheap, fast
    outline_model: str = "claude-3-haiku-20240307"  # Cheap, fast
    writer_model: str = "claude-opus-4-20250514"  # Quality writer
    critic_model: str = "gpt-4o"  # External critic (or gpt-5 when available)
    analysis_model: str = "gemini-1.5-pro"  # Long context for corpus analysis
    
    # Temperature settings
    draft_a_temp: float = 0.9  # Expressive
    draft_b_temp: float = 0.3  # Controlled
    synthesis_temp: float = 0.5  # Balanced
    revision_temp: float = 0.4  # Precise
    
    # RAG settings (for notes extraction only)
    notes_chunks: int = 12
    
    # Pipeline control
    skip_critique: bool = False  # Skip external critique stage
    max_revision_rounds: int = 1


@dataclass
class PipelineState:
    """State container for pipeline execution."""
    
    topic: str = ""
    context: str = ""
    voice_spec: str = ""
    
    # Stage outputs
    retrieved_passages: str = ""
    notes: str = ""
    outline: str = ""
    draft_a: str = ""
    draft_b: str = ""
    synthesis: str = ""
    critique: str = ""
    final: str = ""
    
    # Metadata
    stages_completed: List[str] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=dict)


class HybridPipeline:
    """Multi-stage hybrid RAG pipeline for quality essay generation."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline."""
        load_dotenv()
        
        self.config = config or PipelineConfig()
        self.state = PipelineState()
        
        # Initialize clients lazily
        self._anthropic = None
        self._openai = None
        self._google = None
        
        # Load voice spec if exists
        self.voice_spec_path = Path("voice_spec.txt")
        if self.voice_spec_path.exists():
            self.state.voice_spec = self.voice_spec_path.read_text(encoding='utf-8')
            
    @property
    def anthropic(self):
        """Lazy load Anthropic client."""
        if self._anthropic is None:
            from anthropic import Anthropic
            self._anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._anthropic
    
    @property
    def openai(self):
        """Lazy load OpenAI client."""
        if self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai
    
    @property
    def google(self):
        """Lazy load Google client."""
        if self._google is None:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self._google = genai
        return self._google
    
    def _call_anthropic(self, system: str, user: str, model: str, 
                        temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """Call Anthropic API."""
        response = self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return response.content[0].text
    
    def _call_openai(self, system: str, user: str, model: str,
                     temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """Call OpenAI API."""
        response = self.openai.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        return response.choices[0].message.content
    
    def _call_google(self, prompt: str, model: str = "gemini-1.5-pro") -> str:
        """Call Google API."""
        model_instance = self.google.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        return response.text
    
    # =========================================================================
    # VOICE SPECIFICATION (Offline, Static)
    # =========================================================================
    
    def analyze_voice(self, samples: str, save: bool = True) -> str:
        """Analyze writing samples and create a Voice Specification.
        
        This should be run ONCE, offline. The result is a static artifact
        that never changes during generation.
        """
        print("Analyzing voice (this creates a permanent specification)...")
        
        # Use long-context model for corpus analysis
        try:
            combined_prompt = f"{VOICE_ANALYSIS_SYSTEM}\n\n{VOICE_ANALYSIS_USER.format(samples=samples)}"
            voice_spec = self._call_google(combined_prompt, model=self.config.analysis_model)
        except Exception:
            # Fallback to Anthropic
            voice_spec = self._call_anthropic(
                VOICE_ANALYSIS_SYSTEM,
                VOICE_ANALYSIS_USER.format(samples=samples),
                model="claude-sonnet-4-5-20250929",
                temperature=0.3,
                max_tokens=1500
            )
        
        if save:
            self.voice_spec_path.write_text(voice_spec, encoding='utf-8')
            print(f"Voice specification saved to {self.voice_spec_path}")
            
        self.state.voice_spec = voice_spec
        return voice_spec
    
    def load_voice_spec(self, path: Optional[str] = None) -> str:
        """Load a voice specification from file."""
        p = Path(path) if path else self.voice_spec_path
        if not p.exists():
            raise FileNotFoundError(f"Voice spec not found: {p}")
        self.state.voice_spec = p.read_text(encoding='utf-8')
        return self.state.voice_spec
    
    # =========================================================================
    # STAGE 1: RAG → Notes (Cheap Model)
    # =========================================================================
    
    def stage_1_extract_notes(self, topic: str, passages: str) -> str:
        """Extract compressed notes from retrieved passages.
        
        RAG is used here ONLY. Output is notes, not raw text.
        The writer never sees the original passages.
        """
        print("Stage 1: Extracting notes from retrieved passages...")
        
        self.state.topic = topic
        self.state.retrieved_passages = passages  # Stored but never passed to writer
        
        notes = self._call_anthropic(
            STAGE_1_NOTES_SYSTEM,
            STAGE_1_NOTES_USER.format(topic=topic, passages=passages),
            model=self.config.notes_model,
            temperature=0.3,
            max_tokens=2000
        )
        
        self.state.notes = notes
        self.state.stages_completed.append("notes")
        print(f"  Extracted {len(notes.split(chr(10)))} note lines")
        return notes
    
    # =========================================================================
    # STAGE 2: Outline (Cheap Model)
    # =========================================================================
    
    def stage_2_create_outline(self, context: str = "") -> str:
        """Create structural outline from notes.
        
        No writing yet - just thesis, sections, and intent.
        """
        print("Stage 2: Creating structural outline...")
        
        self.state.context = context
        
        outline = self._call_anthropic(
            STAGE_2_OUTLINE_SYSTEM,
            STAGE_2_OUTLINE_USER.format(
                topic=self.state.topic,
                context=context,
                notes=self.state.notes
            ),
            model=self.config.outline_model,
            temperature=0.4,
            max_tokens=1500
        )
        
        self.state.outline = outline
        self.state.stages_completed.append("outline")
        print("  Outline created")
        return outline
    
    # =========================================================================
    # STAGE 3: Draft A - Expressive (Claude Opus, High Temp)
    # =========================================================================
    
    def stage_3_draft_a(self) -> str:
        """Generate expressive first draft.
        
        Uses Voice Spec + Outline + Notes. NO raw corpus text.
        """
        print("Stage 3: Writing Draft A (expressive)...")
        
        if not self.state.voice_spec:
            raise ValueError("Voice specification required. Run analyze_voice() first.")
        
        system = STAGE_3_DRAFT_A_SYSTEM.format(voice_spec=self.state.voice_spec)
        
        draft_a = self._call_anthropic(
            system,
            STAGE_3_DRAFT_A_USER.format(
                outline=self.state.outline,
                notes=self.state.notes,
                context=self.state.context
            ),
            model=self.config.writer_model,
            temperature=self.config.draft_a_temp,
            max_tokens=5000
        )
        
        self.state.draft_a = draft_a
        self.state.stages_completed.append("draft_a")
        print(f"  Draft A: {len(draft_a.split())} words")
        return draft_a
    
    # =========================================================================
    # STAGE 4: Draft B - Controlled (Claude Opus, Low Temp)
    # =========================================================================
    
    def stage_4_draft_b(self) -> str:
        """Generate controlled second draft.
        
        Same inputs, stricter discipline.
        """
        print("Stage 4: Writing Draft B (controlled)...")
        
        system = STAGE_4_DRAFT_B_SYSTEM.format(voice_spec=self.state.voice_spec)
        
        draft_b = self._call_anthropic(
            system,
            STAGE_4_DRAFT_B_USER.format(
                outline=self.state.outline,
                notes=self.state.notes,
                context=self.state.context
            ),
            model=self.config.writer_model,
            temperature=self.config.draft_b_temp,
            max_tokens=5000
        )
        
        self.state.draft_b = draft_b
        self.state.stages_completed.append("draft_b")
        print(f"  Draft B: {len(draft_b.split())} words")
        return draft_b
    
    # =========================================================================
    # STAGE 5: Synthesis (Claude Opus)
    # =========================================================================
    
    def stage_5_synthesize(self) -> str:
        """Merge Draft A and Draft B into superior final draft."""
        print("Stage 5: Synthesizing drafts...")
        
        system = STAGE_5_SYNTHESIS_SYSTEM.format(voice_spec=self.state.voice_spec)
        
        synthesis = self._call_anthropic(
            system,
            STAGE_5_SYNTHESIS_USER.format(
                draft_a=self.state.draft_a,
                draft_b=self.state.draft_b
            ),
            model=self.config.writer_model,
            temperature=self.config.synthesis_temp,
            max_tokens=5000
        )
        
        self.state.synthesis = synthesis
        self.state.stages_completed.append("synthesis")
        print(f"  Synthesis: {len(synthesis.split())} words")
        return synthesis
    
    # =========================================================================
    # STAGE 6: Editorial Critique (External Model)
    # =========================================================================
    
    def stage_6_critique(self) -> str:
        """Get editorial critique from external model.
        
        Flags problems only - does not rewrite.
        """
        if self.config.skip_critique:
            print("Stage 6: Skipped (critique disabled)")
            self.state.critique = ""
            return ""
            
        print("Stage 6: Getting editorial critique...")
        
        try:
            critique = self._call_openai(
                STAGE_6_CRITIQUE_SYSTEM,
                STAGE_6_CRITIQUE_USER.format(draft=self.state.synthesis),
                model=self.config.critic_model,
                temperature=0.3,
                max_tokens=2000
            )
        except Exception as e:
            print(f"  OpenAI critique failed ({e}), using Anthropic fallback")
            critique = self._call_anthropic(
                STAGE_6_CRITIQUE_SYSTEM,
                STAGE_6_CRITIQUE_USER.format(draft=self.state.synthesis),
                model="claude-sonnet-4-5-20250929",
                temperature=0.3,
                max_tokens=2000
            )
        
        self.state.critique = critique
        self.state.stages_completed.append("critique")
        print("  Critique received")
        return critique
    
    # =========================================================================
    # STAGE 7: Targeted Revision (Claude Opus)
    # =========================================================================
    
    def stage_7_revise(self) -> str:
        """Revise ONLY the flagged problem areas."""
        if not self.state.critique:
            print("Stage 7: Skipped (no critique)")
            self.state.final = self.state.synthesis
            return self.state.final
            
        print("Stage 7: Targeted revision...")
        
        system = STAGE_7_REVISION_SYSTEM.format(voice_spec=self.state.voice_spec)
        
        final = self._call_anthropic(
            system,
            STAGE_7_REVISION_USER.format(
                draft=self.state.synthesis,
                critique=self.state.critique
            ),
            model=self.config.writer_model,
            temperature=self.config.revision_temp,
            max_tokens=5000
        )
        
        self.state.final = final
        self.state.stages_completed.append("revision")
        print(f"  Final: {len(final.split())} words")
        return final
    
    # =========================================================================
    # FULL PIPELINE
    # =========================================================================
    
    def run(self, topic: str, passages: str, context: str = "") -> str:
        """Run the complete pipeline.
        
        Args:
            topic: The essay topic
            passages: Retrieved passages (for notes extraction only)
            context: Optional context about audience, purpose, etc.
            
        Returns:
            The final essay
        """
        print("\n" + "=" * 60)
        print("HYBRID PIPELINE - Starting")
        print("=" * 60)
        print(f"Topic: {topic[:80]}...")
        print()
        
        # Stage 1: Extract notes (RAG → notes only)
        self.stage_1_extract_notes(topic, passages)
        
        # Stage 2: Create outline
        self.stage_2_create_outline(context)
        
        # Stage 3: Draft A (expressive)
        self.stage_3_draft_a()
        
        # Stage 4: Draft B (controlled)
        self.stage_4_draft_b()
        
        # Stage 5: Synthesize
        self.stage_5_synthesize()
        
        # Stage 6: Critique
        self.stage_6_critique()
        
        # Stage 7: Revise
        self.stage_7_revise()
        
        print()
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print(f"Stages: {' → '.join(self.state.stages_completed)}")
        print(f"Final word count: {len(self.state.final.split())}")
        print("=" * 60)
        
        return self.state.final
    
    def run_quick(self, topic: str, passages: str, context: str = "") -> str:
        """Run a faster pipeline (skip dual drafts and critique).
        
        Stages: Notes → Outline → Single Draft → Done
        """
        print("\n" + "=" * 60)
        print("HYBRID PIPELINE - Quick Mode")
        print("=" * 60)
        
        # Stage 1: Extract notes
        self.stage_1_extract_notes(topic, passages)
        
        # Stage 2: Create outline
        self.stage_2_create_outline(context)
        
        # Single draft (balanced temperature)
        print("Writing single draft...")
        system = STAGE_3_DRAFT_A_SYSTEM.format(voice_spec=self.state.voice_spec)
        
        draft = self._call_anthropic(
            system,
            STAGE_3_DRAFT_A_USER.format(
                outline=self.state.outline,
                notes=self.state.notes,
                context=self.state.context
            ),
            model=self.config.writer_model,
            temperature=0.6,
            max_tokens=5000
        )
        
        self.state.final = draft
        print(f"Final: {len(draft.split())} words")
        print("=" * 60)
        
        return draft


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for the hybrid pipeline."""
    import sys
    
    if len(sys.argv) < 2:
        print("Hybrid RAG Pipeline")
        print()
        print("Usage:")
        print("  python pipeline.py analyze    - Create voice spec from corpus")
        print("  python pipeline.py generate   - Run full pipeline (interactive)")
        print("  python pipeline.py quick      - Run quick pipeline (interactive)")
        return
    
    command = sys.argv[1].lower()
    pipeline = HybridPipeline()
    
    if command == "analyze":
        # Load samples from transcripts folder
        from write import RAGWriter
        writer = RAGWriter()
        
        # Get style chunks
        samples = []
        for meta in writer.metadata[:30]:  # Sample first 30 chunks
            if meta.get('source_type') == 'style':
                samples.append(meta['text'])
        
        if not samples:
            print("No style sources found. Add .txt files to transcripts/")
            return
            
        combined = "\n\n---\n\n".join(samples)
        voice_spec = pipeline.analyze_voice(combined)
        
        print("\n" + "=" * 60)
        print("VOICE SPECIFICATION")
        print("=" * 60)
        print(voice_spec)
        
    elif command in ["generate", "quick"]:
        # Check voice spec
        if not pipeline.state.voice_spec:
            print("No voice spec found. Run 'python pipeline.py analyze' first.")
            return
            
        # Get input
        print("Enter topic:")
        topic = input("> ").strip()
        if not topic:
            return
            
        print("Enter context (optional, press Enter to skip):")
        context = input("> ").strip()
        
        # Retrieve passages using existing RAG
        from write import RAGWriter
        writer = RAGWriter()
        
        if writer.use_mmr:
            results = writer.retrieve_mmr(topic, top_k=12)
        else:
            results = writer.retrieve(topic, top_k=12)
            
        passages = "\n\n---\n\n".join([r['text'] for r in results])
        
        # Run pipeline
        if command == "quick":
            essay = pipeline.run_quick(topic, passages, context)
        else:
            essay = pipeline.run(topic, passages, context)
        
        print("\n" + "=" * 60)
        print("FINAL ESSAY")
        print("=" * 60)
        print(essay)
        

if __name__ == "__main__":
    main()
