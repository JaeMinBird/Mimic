#!/usr/bin/env python3
"""Modern GUI for RAG Essay Writer - Swedish Minimalist Design.

Supports two modes:
- Classic: Traditional RAG with style chunks passed to writer
- Hybrid: Multi-stage pipeline with voice spec (no raw text to writer)
"""

import os
import sys
import threading
import datetime
from pathlib import Path
from typing import Optional, Dict, List
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk

# Import the core systems
from write import RAGWriter, load_settings
from pipeline import HybridPipeline, PipelineConfig


# Swedish minimalist color palette
COLORS = {
    "bg_primary": "#0a0a0a",
    "bg_secondary": "#111111",
    "bg_tertiary": "#1a1a1a",
    "bg_input": "#0f0f0f",
    "bg_tooltip": "#000000",
    "text_primary": "#ffffff",
    "text_secondary": "#888888",
    "text_muted": "#555555",
    "accent": "#d62828",
    "accent_hover": "#e63939",
    "border": "#2a2a2a",
}


class Tooltip:
    """Minimal tooltip with black background and white text."""
    
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)
        
    def _show(self, event=None):
        if self.tooltip_window:
            return
            
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 5
        y = self.widget.winfo_rooty()
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(bg=COLORS["bg_tooltip"])
        
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background=COLORS["bg_tooltip"],
            foreground=COLORS["text_primary"],
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 9),
            padx=8,
            pady=6,
            wraplength=200
        )
        label.pack()
        
        tw.configure(highlightbackground=COLORS["border"], highlightthickness=1)
        
    def _hide(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class OutputRedirector:
    """Redirect stdout to a text widget."""
    
    def __init__(self, text_widget: ctk.CTkTextbox, tag: str = "stdout"):
        self.text_widget = text_widget
        self.tag = tag
        
    def write(self, text: str):
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", text)
        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")
        
    def flush(self):
        pass


class ModelConfig:
    """Configuration for different AI model providers."""
    
    PROVIDERS = {
        "Anthropic": {
            "models": [
                "claude-opus-4-20250514",
                "claude-sonnet-4-5-20250929",
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307",
            ],
            "env_key": "ANTHROPIC_API_KEY",
            "default": "claude-opus-4-20250514",
        },
        "OpenAI": {
            "models": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "o1-preview",
                "o1-mini",
            ],
            "env_key": "OPENAI_API_KEY",
            "default": "gpt-4o",
        },
        "Google": {
            "models": [
                "gemini-2.0-flash-exp",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ],
            "env_key": "GOOGLE_API_KEY",
            "default": "gemini-2.0-flash-exp",
        },
        "Mistral": {
            "models": [
                "mistral-large-latest",
                "mistral-medium-latest",
                "mistral-small-latest",
            ],
            "env_key": "MISTRAL_API_KEY",
            "default": "mistral-large-latest",
        },
        "Groq": {
            "models": [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
            ],
            "env_key": "GROQ_API_KEY",
            "default": "llama-3.3-70b-versatile",
        },
    }
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        from dotenv import load_dotenv
        load_dotenv()
        available = []
        for provider, config in cls.PROVIDERS.items():
            if os.getenv(config["env_key"]):
                available.append(provider)
        return available if available else ["Anthropic"]
    
    @classmethod
    def get_models(cls, provider: str) -> List[str]:
        return cls.PROVIDERS.get(provider, {}).get("models", [])
    
    @classmethod
    def get_default_model(cls, provider: str) -> str:
        return cls.PROVIDERS.get(provider, {}).get("default", "")


class SourcesPanel(ctk.CTkFrame):
    """Panel for managing sources with checkbox exclusion."""
    
    def __init__(self, parent, writer: RAGWriter, **kwargs):
        super().__init__(parent, fg_color=COLORS["bg_secondary"], corner_radius=0, **kwargs)
        self.writer = writer
        self.checkboxes = {}
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        header = ctk.CTkLabel(
            self, text="Sources", 
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        header.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")
        
        self.source_type_var = ctk.StringVar(value="Style")
        self.source_dropdown = ctk.CTkOptionMenu(
            self,
            values=["Style", "Research", "Structure", "Extra PDFs"],
            variable=self.source_type_var,
            command=self._on_source_type_change,
            fg_color=COLORS["bg_tertiary"],
            button_color=COLORS["bg_tertiary"],
            button_hover_color=COLORS["border"],
            dropdown_fg_color=COLORS["bg_secondary"],
            dropdown_hover_color=COLORS["accent"],
            font=ctk.CTkFont(size=11),
            height=26,
            width=140
        )
        self.source_dropdown.grid(row=1, column=0, padx=12, pady=(0, 6), sticky="w")
        Tooltip(self.source_dropdown, "Style: indexed transcripts\nResearch: indexed PDFs\nStructure: org templates\nExtra PDFs: runtime loaded")
        
        self.source_scroll = ctk.CTkScrollableFrame(self, fg_color=COLORS["bg_input"], corner_radius=4)
        self.source_scroll.grid(row=2, column=0, padx=12, pady=(0, 6), sticky="nsew")
        self.source_scroll.grid_columnconfigure(0, weight=1)
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=3, column=0, padx=12, pady=(0, 12), sticky="ew")
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.add_btn = ctk.CTkButton(
            btn_frame, text="Add", font=ctk.CTkFont(size=11),
            fg_color=COLORS["bg_tertiary"], hover_color=COLORS["border"],
            height=26, command=self._add_source
        )
        self.add_btn.grid(row=0, column=0, padx=(0, 3), sticky="ew")
        
        self.clear_btn = ctk.CTkButton(
            btn_frame, text="Clear", font=ctk.CTkFont(size=11),
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            height=26, command=self._clear_sources
        )
        self.clear_btn.grid(row=0, column=1, padx=(3, 0), sticky="ew")
        
        self._refresh_list()
        
    def _on_source_type_change(self, value: str):
        self._refresh_list()
        if value in ["Style", "Research"]:
            self.add_btn.configure(state="disabled")
            self.clear_btn.configure(state="disabled")
        else:
            self.add_btn.configure(state="normal")
            self.clear_btn.configure(state="normal")
        
    def _toggle_source(self, source_name: str, var: ctk.BooleanVar):
        if var.get():
            self.writer.excluded_sources.discard(source_name)
        else:
            self.writer.excluded_sources.add(source_name)
        
    def _refresh_list(self):
        source_type = self.source_type_var.get()
        
        for widget in self.source_scroll.winfo_children():
            widget.destroy()
        self.checkboxes.clear()
        
        if source_type == "Style":
            style_sources, _ = self.writer.get_sources_by_type()
            if style_sources:
                for i, s in enumerate(style_sources):
                    var = ctk.BooleanVar(value=s not in self.writer.excluded_sources)
                    cb = ctk.CTkCheckBox(
                        self.source_scroll,
                        text=s[:28] + "..." if len(s) > 28 else s,
                        variable=var,
                        command=lambda name=s, v=var: self._toggle_source(name, v),
                        font=ctk.CTkFont(size=10), text_color=COLORS["text_secondary"],
                        fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                        border_color=COLORS["border"], checkmark_color=COLORS["text_primary"],
                        height=22, checkbox_width=14, checkbox_height=14
                    )
                    cb.grid(row=i, column=0, sticky="w", pady=1)
                    self.checkboxes[s] = var
                    Tooltip(cb, s)
            else:
                ctk.CTkLabel(self.source_scroll, text="No style sources.\nAdd .txt to transcripts/",
                    font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"]).grid(row=0, column=0, sticky="w")
                
        elif source_type == "Research":
            _, research_sources = self.writer.get_sources_by_type()
            if research_sources:
                for i, s in enumerate(research_sources):
                    var = ctk.BooleanVar(value=s not in self.writer.excluded_sources)
                    cb = ctk.CTkCheckBox(
                        self.source_scroll,
                        text=s[:28] + "..." if len(s) > 28 else s,
                        variable=var,
                        command=lambda name=s, v=var: self._toggle_source(name, v),
                        font=ctk.CTkFont(size=10), text_color=COLORS["text_secondary"],
                        fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                        border_color=COLORS["border"], checkmark_color=COLORS["text_primary"],
                        height=22, checkbox_width=14, checkbox_height=14
                    )
                    cb.grid(row=i, column=0, sticky="w", pady=1)
                    self.checkboxes[s] = var
                    Tooltip(cb, s)
            else:
                ctk.CTkLabel(self.source_scroll, text="No research sources.\nAdd .pdf to research/",
                    font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"]).grid(row=0, column=0, sticky="w")
                
        elif source_type == "Structure":
            if self.writer.structure_sources:
                for i, s in enumerate(self.writer.structure_sources):
                    name = s['name']
                    sampled = " [s]" if s.get('sampled') else ""
                    ctk.CTkLabel(self.source_scroll, text=f"{i+1}. {name[:24]}{sampled}",
                        font=ctk.CTkFont(size=10), text_color=COLORS["text_secondary"]).grid(row=i, column=0, sticky="w", pady=1)
            else:
                ctk.CTkLabel(self.source_scroll, text="No structure sources.",
                    font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"]).grid(row=0, column=0, sticky="w")
                
        elif source_type == "Extra PDFs":
            if self.writer.research_sources:
                for i, s in enumerate(self.writer.research_sources):
                    ctk.CTkLabel(self.source_scroll, text=f"{i+1}. {s['name'][:24]}",
                        font=ctk.CTkFont(size=10), text_color=COLORS["text_secondary"]).grid(row=i, column=0, sticky="w", pady=1)
            else:
                ctk.CTkLabel(self.source_scroll, text="No extra PDFs loaded.",
                    font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"]).grid(row=0, column=0, sticky="w")
        
    def _add_source(self):
        source_type = self.source_type_var.get()
        if source_type == "Structure":
            filepath = filedialog.askopenfilename(title="Select Structure Source", filetypes=[("Text and PDF", "*.txt *.pdf")])
            if filepath:
                result = self.writer.load_structure_source(filepath, max_chars=self.writer.structure_budget)
                if result:
                    self.writer.structure_sources.append(result)
                    self._refresh_list()
        elif source_type == "Extra PDFs":
            filepath = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF", "*.pdf")])
            if filepath:
                result = self.writer.load_pdf(filepath)
                if result:
                    self.writer.research_sources.append(result)
                    self._refresh_list()
                    
    def _clear_sources(self):
        source_type = self.source_type_var.get()
        if source_type == "Structure":
            self.writer.structure_sources.clear()
        elif source_type == "Extra PDFs":
            self.writer.research_sources.clear()
        self._refresh_list()


class SettingsPanel(ctk.CTkScrollableFrame):
    """Compact settings panel with mode selection and tooltips."""
    
    def __init__(self, parent, writer: RAGWriter, on_model_change: callable = None, on_mode_change: callable = None, **kwargs):
        super().__init__(parent, fg_color=COLORS["bg_secondary"], corner_radius=0, **kwargs)
        self.writer = writer
        self.on_model_change = on_model_change
        self.on_mode_change = on_mode_change
        
        self.grid_columnconfigure(0, weight=1)
        
        row = 0
        
        # MODE SECTION
        self._add_section_header("Mode", row)
        row += 1
        
        mode_frame = ctk.CTkFrame(self, fg_color="transparent")
        mode_frame.grid(row=row, column=0, padx=8, pady=2, sticky="ew")
        mode_frame.grid_columnconfigure(1, weight=1)
        
        self.mode_var = ctk.StringVar(value="Hybrid")
        mode_dropdown = ctk.CTkOptionMenu(
            mode_frame,
            values=["Hybrid", "Hybrid Quick", "Classic"],
            variable=self.mode_var,
            command=self._on_mode_change,
            fg_color=COLORS["accent"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["bg_secondary"],
            dropdown_hover_color=COLORS["accent"],
            font=ctk.CTkFont(size=11, weight="bold"),
            height=28
        )
        mode_dropdown.grid(row=0, column=0, columnspan=2, sticky="ew")
        Tooltip(mode_dropdown, "Hybrid: 7-stage pipeline, voice spec\nHybrid Quick: Single draft pipeline\nClassic: Traditional RAG")
        row += 1
        
        # Voice spec status
        self.voice_status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.voice_status_frame.grid(row=row, column=0, padx=8, pady=(4, 2), sticky="ew")
        
        self.voice_status = ctk.CTkLabel(
            self.voice_status_frame,
            text="Voice Spec: " + ("Loaded" if Path("voice_spec.txt").exists() else "Missing"),
            font=ctk.CTkFont(size=10),
            text_color=COLORS["accent"] if Path("voice_spec.txt").exists() else COLORS["text_muted"]
        )
        self.voice_status.pack(side="left")
        Tooltip(self.voice_status, "Static voice specification for hybrid mode.\nRun 'python pipeline.py analyze' to create.")
        row += 1
        
        # Model section
        self._add_section_header("Model", row, top_pad=6)
        row += 1
        
        # Provider
        provider_frame = ctk.CTkFrame(self, fg_color="transparent", height=28)
        provider_frame.grid(row=row, column=0, padx=8, pady=2, sticky="ew")
        provider_frame.grid_columnconfigure(1, weight=1)
        
        lbl = ctk.CTkLabel(provider_frame, text="Provider", font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"], width=50, anchor="w")
        lbl.grid(row=0, column=0, sticky="w")
        Tooltip(lbl, "AI provider for generation")
        
        available_providers = ModelConfig.get_available_providers()
        self.provider_var = ctk.StringVar(value=available_providers[0] if available_providers else "Anthropic")
        self.provider_dropdown = ctk.CTkOptionMenu(
            provider_frame, values=available_providers, variable=self.provider_var,
            command=self._on_provider_change,
            fg_color=COLORS["bg_tertiary"], button_color=COLORS["bg_tertiary"],
            button_hover_color=COLORS["border"], dropdown_fg_color=COLORS["bg_secondary"],
            dropdown_hover_color=COLORS["accent"], font=ctk.CTkFont(size=10), height=24
        )
        self.provider_dropdown.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        row += 1
        
        # Model
        model_frame = ctk.CTkFrame(self, fg_color="transparent", height=28)
        model_frame.grid(row=row, column=0, padx=8, pady=2, sticky="ew")
        model_frame.grid_columnconfigure(1, weight=1)
        
        lbl = ctk.CTkLabel(model_frame, text="Model", font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"], width=50, anchor="w")
        lbl.grid(row=0, column=0, sticky="w")
        Tooltip(lbl, "Specific model (writer in hybrid mode)")
        
        initial_models = ModelConfig.get_models(self.provider_var.get())
        self.model_var = ctk.StringVar(value=ModelConfig.get_default_model(self.provider_var.get()))
        self.model_dropdown = ctk.CTkOptionMenu(
            model_frame, values=initial_models if initial_models else ["claude-opus-4-20250514"],
            variable=self.model_var, command=self._on_model_change_internal,
            fg_color=COLORS["bg_tertiary"], button_color=COLORS["bg_tertiary"],
            button_hover_color=COLORS["border"], dropdown_fg_color=COLORS["bg_secondary"],
            dropdown_hover_color=COLORS["accent"], font=ctk.CTkFont(size=10), height=24
        )
        self.model_dropdown.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        row += 1
        
        # Retrieval section
        self._add_section_header("Retrieval", row, top_pad=6)
        row += 1
        
        # Chunks
        chunks_frame = ctk.CTkFrame(self, fg_color="transparent")
        chunks_frame.grid(row=row, column=0, padx=8, pady=2, sticky="ew")
        chunks_frame.grid_columnconfigure(1, weight=1)
        
        lbl = ctk.CTkLabel(chunks_frame, text="Chunks", font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"], width=50, anchor="w")
        lbl.grid(row=0, column=0, sticky="w")
        Tooltip(lbl, "Number of chunks to retrieve. In hybrid mode, used for notes extraction.")
        
        slider_frame = ctk.CTkFrame(chunks_frame, fg_color="transparent")
        slider_frame.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        slider_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(slider_frame, text="1", font=ctk.CTkFont(size=8), text_color=COLORS["text_muted"], width=12).grid(row=0, column=0)
        
        self.chunks_var = ctk.IntVar(value=writer.top_k)
        self.chunks_slider = ctk.CTkSlider(
            slider_frame, from_=1, to=30, number_of_steps=29,
            variable=self.chunks_var, command=self._on_chunks_change,
            fg_color=COLORS["bg_tertiary"], progress_color=COLORS["accent"],
            button_color=COLORS["text_primary"], button_hover_color=COLORS["text_secondary"], height=14
        )
        self.chunks_slider.grid(row=0, column=1, sticky="ew", padx=2)
        
        ctk.CTkLabel(slider_frame, text="30", font=ctk.CTkFont(size=8), text_color=COLORS["text_muted"], width=16).grid(row=0, column=2)
        
        self.chunks_value = ctk.CTkLabel(slider_frame, text=str(writer.top_k), font=ctk.CTkFont(size=10, weight="bold"), text_color=COLORS["accent"], width=20)
        self.chunks_value.grid(row=0, column=3, padx=(4, 0))
        row += 1
        
        # Lambda (MMR)
        lambda_frame = ctk.CTkFrame(self, fg_color="transparent")
        lambda_frame.grid(row=row, column=0, padx=8, pady=2, sticky="ew")
        lambda_frame.grid_columnconfigure(1, weight=1)
        
        lbl = ctk.CTkLabel(lambda_frame, text="MMR", font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"], width=50, anchor="w")
        lbl.grid(row=0, column=0, sticky="w")
        Tooltip(lbl, "MMR lambda: 1.0=relevance, 0.0=diversity")
        
        slider_frame2 = ctk.CTkFrame(lambda_frame, fg_color="transparent")
        slider_frame2.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        slider_frame2.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(slider_frame2, text="0", font=ctk.CTkFont(size=8), text_color=COLORS["text_muted"], width=12).grid(row=0, column=0)
        
        self.lambda_var = ctk.DoubleVar(value=writer.mmr_lambda)
        self.lambda_slider = ctk.CTkSlider(
            slider_frame2, from_=0, to=1, number_of_steps=20,
            variable=self.lambda_var, command=self._on_lambda_change,
            fg_color=COLORS["bg_tertiary"], progress_color=COLORS["accent"],
            button_color=COLORS["text_primary"], button_hover_color=COLORS["text_secondary"], height=14
        )
        self.lambda_slider.grid(row=0, column=1, sticky="ew", padx=2)
        
        ctk.CTkLabel(slider_frame2, text="1", font=ctk.CTkFont(size=8), text_color=COLORS["text_muted"], width=12).grid(row=0, column=2)
        
        self.lambda_value = ctk.CTkLabel(slider_frame2, text=f"{writer.mmr_lambda:.1f}", font=ctk.CTkFont(size=10, weight="bold"), text_color=COLORS["accent"], width=24)
        self.lambda_value.grid(row=0, column=3, padx=(4, 0))
        row += 1
        
        # Toggles
        toggle_row1 = ctk.CTkFrame(self, fg_color="transparent")
        toggle_row1.grid(row=row, column=0, padx=8, pady=4, sticky="ew")
        toggle_row1.grid_columnconfigure((0, 1), weight=1)
        
        self.mmr_var = ctk.BooleanVar(value=writer.use_mmr)
        mmr_cb = ctk.CTkCheckBox(
            toggle_row1, text="MMR", variable=self.mmr_var, command=self._on_mmr_toggle,
            font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"],
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            border_color=COLORS["border"], checkmark_color=COLORS["text_primary"],
            height=20, checkbox_width=14, checkbox_height=14
        )
        mmr_cb.grid(row=0, column=0, sticky="w")
        Tooltip(mmr_cb, "Maximal Marginal Relevance for diverse retrieval")
        
        self.dedup_var = ctk.BooleanVar(value=writer.deduplicate)
        dedup_cb = ctk.CTkCheckBox(
            toggle_row1, text="Dedup", variable=self.dedup_var, command=self._on_dedup_toggle,
            font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"],
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            border_color=COLORS["border"], checkmark_color=COLORS["text_primary"],
            height=20, checkbox_width=14, checkbox_height=14
        )
        dedup_cb.grid(row=0, column=1, sticky="w")
        Tooltip(dedup_cb, "Remove near-duplicate chunks")
        row += 1
        
        # Classic mode settings (hidden in hybrid)
        self.classic_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.classic_frame.grid(row=row, column=0, padx=8, pady=2, sticky="ew")
        self.classic_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.research_var = ctk.BooleanVar(value=writer.use_research)
        research_cb = ctk.CTkCheckBox(
            self.classic_frame, text="Research", variable=self.research_var, command=self._on_research_toggle,
            font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"],
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            border_color=COLORS["border"], checkmark_color=COLORS["text_primary"],
            height=20, checkbox_width=14, checkbox_height=14
        )
        research_cb.grid(row=0, column=0, sticky="w")
        Tooltip(research_cb, "Include indexed research in retrieval")
        
        self.rules_var = ctk.BooleanVar(value=writer.writing_rules_enabled)
        rules_cb = ctk.CTkCheckBox(
            self.classic_frame, text="Rules", variable=self.rules_var, command=self._on_rules_toggle,
            font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"],
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            border_color=COLORS["border"], checkmark_color=COLORS["text_primary"],
            height=20, checkbox_width=14, checkbox_height=14
        )
        rules_cb.grid(row=0, column=1, sticky="w")
        Tooltip(rules_cb, "Anti-AI writing rules (classic mode)")
        row += 1
        
        # Hybrid settings
        self._add_section_header("Pipeline", row, top_pad=6)
        row += 1
        
        self.hybrid_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.hybrid_frame.grid(row=row, column=0, padx=8, pady=2, sticky="ew")
        self.hybrid_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.critique_var = ctk.BooleanVar(value=True)
        critique_cb = ctk.CTkCheckBox(
            self.hybrid_frame, text="Critique", variable=self.critique_var,
            font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"],
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            border_color=COLORS["border"], checkmark_color=COLORS["text_primary"],
            height=20, checkbox_width=14, checkbox_height=14
        )
        critique_cb.grid(row=0, column=0, sticky="w")
        Tooltip(critique_cb, "External model critique stage (GPT)")
        
        self.dual_draft_var = ctk.BooleanVar(value=True)
        dual_cb = ctk.CTkCheckBox(
            self.hybrid_frame, text="Dual Draft", variable=self.dual_draft_var,
            font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"],
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            border_color=COLORS["border"], checkmark_color=COLORS["text_primary"],
            height=20, checkbox_width=14, checkbox_height=14
        )
        dual_cb.grid(row=0, column=1, sticky="w")
        Tooltip(dual_cb, "Generate both expressive and controlled drafts")
        row += 1
        
        # Profile section (classic mode)
        self._add_section_header("Style Profile", row, top_pad=6)
        row += 1
        
        profile_frame = ctk.CTkFrame(self, fg_color="transparent")
        profile_frame.grid(row=row, column=0, padx=8, pady=2, sticky="ew")
        profile_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        self.profile_status = ctk.CTkLabel(
            profile_frame, 
            text="None" if not writer.style_profile else "Active",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_muted"] if not writer.style_profile else COLORS["accent"]
        )
        self.profile_status.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))
        Tooltip(self.profile_status, "Classic mode style profile (analyzed from sources)")
        
        btn_style = {"font": ctk.CTkFont(size=10), "fg_color": COLORS["bg_tertiary"], "hover_color": COLORS["border"], "height": 24}
        
        analyze_btn = ctk.CTkButton(profile_frame, text="Analyze", command=self._analyze_style, **btn_style)
        analyze_btn.grid(row=1, column=0, padx=1, sticky="ew")
        Tooltip(analyze_btn, "Analyze sources for style profile")
        
        load_btn = ctk.CTkButton(profile_frame, text="Load", command=self._load_profile, **btn_style)
        load_btn.grid(row=1, column=1, padx=1, sticky="ew")
        
        save_btn = ctk.CTkButton(profile_frame, text="Save", command=self._save_profile, **btn_style)
        save_btn.grid(row=1, column=2, padx=1, sticky="ew")
        
        clear_btn = ctk.CTkButton(profile_frame, text="Clear", command=self._clear_profile,
            font=ctk.CTkFont(size=10), fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], height=24)
        clear_btn.grid(row=1, column=3, padx=1, sticky="ew")
        
    def _add_section_header(self, text: str, row: int, top_pad: int = 0):
        ctk.CTkLabel(self, text=text, font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_secondary"]).grid(row=row, column=0, padx=8, pady=(top_pad + 6, 4), sticky="w")
        
    def _on_mode_change(self, value: str):
        if self.on_mode_change:
            self.on_mode_change(value)
        
    def _on_provider_change(self, value: str):
        models = ModelConfig.get_models(value)
        default_model = ModelConfig.get_default_model(value)
        self.model_dropdown.configure(values=models if models else ["No models"])
        self.model_var.set(default_model if default_model else "")
        if self.on_model_change:
            self.on_model_change(value, default_model)
            
    def _on_model_change_internal(self, value: str):
        if self.on_model_change:
            self.on_model_change(self.provider_var.get(), value)
        
    def _on_chunks_change(self, value):
        self.writer.top_k = int(value)
        self.chunks_value.configure(text=str(int(value)))
        
    def _on_mmr_toggle(self):
        self.writer.use_mmr = self.mmr_var.get()
        
    def _on_lambda_change(self, value):
        self.writer.mmr_lambda = value
        self.lambda_value.configure(text=f"{value:.1f}")
        
    def _on_dedup_toggle(self):
        self.writer.deduplicate = self.dedup_var.get()
        
    def _on_research_toggle(self):
        self.writer.use_research = self.research_var.get()
        
    def _on_rules_toggle(self):
        self.writer.writing_rules_enabled = self.rules_var.get()
        
    def _analyze_style(self):
        def analyze():
            self.writer.analyze_style()
            self.after(0, self._update_profile_status)
        threading.Thread(target=analyze, daemon=True).start()
        
    def _update_profile_status(self):
        if self.writer.style_profile:
            self.profile_status.configure(text="Active", text_color=COLORS["accent"])
        else:
            self.profile_status.configure(text="None", text_color=COLORS["text_muted"])
            
    def _load_profile(self):
        profiles_dir = Path("profiles")
        if not profiles_dir.exists() or not list(profiles_dir.glob("*.txt")):
            messagebox.showinfo("No Profiles", "No saved profiles found.")
            return
        profiles = list(profiles_dir.glob("*.txt"))
            
        dialog = ctk.CTkToplevel(self)
        dialog.title("Load Profile")
        dialog.geometry("240x320")
        dialog.transient(self)
        dialog.grab_set()
        dialog.configure(fg_color=COLORS["bg_primary"])
        
        ctk.CTkLabel(dialog, text="Select profile", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["text_primary"]).pack(pady=(12, 6))
        
        listbox = tk.Listbox(dialog, bg=COLORS["bg_input"], fg=COLORS["text_primary"], 
            selectbackground=COLORS["accent"], selectforeground=COLORS["text_primary"],
            font=("Consolas", 10), borderwidth=0, highlightthickness=0)
        listbox.pack(fill="both", expand=True, padx=12, pady=6)
        
        for p in sorted(profiles):
            listbox.insert("end", p.stem)
            
        def load_selected():
            selection = listbox.curselection()
            if selection:
                name = listbox.get(selection[0])
                path = profiles_dir / f"{name}.txt"
                with open(path, 'r', encoding='utf-8') as f:
                    self.writer.style_profile = f.read()
                self._update_profile_status()
                dialog.destroy()
                
        ctk.CTkButton(dialog, text="Load", fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], command=load_selected).pack(pady=12)
        
    def _save_profile(self):
        if not self.writer.style_profile:
            messagebox.showwarning("No Profile", "No style profile to save.")
            return
        dialog = ctk.CTkInputDialog(text="Profile name:", title="Save Profile")
        name = dialog.get_input()
        if name:
            profiles_dir = Path("profiles")
            profiles_dir.mkdir(exist_ok=True)
            safe_name = name.strip().replace(" ", "_")
            path = profiles_dir / f"{safe_name}.txt"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.writer.style_profile)
            messagebox.showinfo("Saved", f"Profile saved: {safe_name}")
            
    def _clear_profile(self):
        self.writer.style_profile = None
        self._update_profile_status()
        
    def get_current_model(self) -> tuple:
        return self.provider_var.get(), self.model_var.get()
    
    def get_mode(self) -> str:
        return self.mode_var.get()
    
    def get_pipeline_config(self) -> dict:
        return {
            "skip_critique": not self.critique_var.get(),
            "dual_draft": self.dual_draft_var.get(),
        }


class RAGWriterGUI(ctk.CTk):
    """Main GUI application - Swedish Minimalist Design."""
    
    def __init__(self):
        super().__init__()
        
        ctk.set_appearance_mode("dark")
        
        self.title("RAG Essay Writer")
        self.geometry("1400x900")
        self.minsize(1000, 650)
        self.configure(fg_color=COLORS["bg_primary"])
        
        self.current_provider = "Anthropic"
        self.current_model = "claude-opus-4-20250514"
        self.current_mode = "Hybrid"
        
        self.grid_columnconfigure(0, weight=0, minsize=200)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=270)
        self.grid_rowconfigure(0, weight=1)
        
        self._init_systems()
        self._create_left_panel()
        self._create_main_panel()
        self._create_right_panel()
        
        self.bind("<Control-Return>", lambda e: self._generate_essay())
        self.bind("<Control-s>", lambda e: self._save_essay())
        
    def _init_systems(self):
        """Initialize both classic RAG and hybrid pipeline."""
        try:
            self.writer = RAGWriter()
            self.pipeline = HybridPipeline()
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            self.destroy()
            sys.exit(1)
            
    def _on_model_change(self, provider: str, model: str):
        self.current_provider = provider
        self.current_model = model
        print(f"Model: {provider}/{model}")
        
    def _on_mode_change(self, mode: str):
        self.current_mode = mode
        print(f"Mode: {mode}")
            
    def _create_left_panel(self):
        left_panel = ctk.CTkFrame(self, fg_color=COLORS["bg_secondary"], corner_radius=0)
        left_panel.grid(row=0, column=0, sticky="nsew")
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_rowconfigure(0, weight=1)
        
        self.sources_panel = SourcesPanel(left_panel, self.writer)
        self.sources_panel.grid(row=0, column=0, sticky="nsew")
        
    def _create_main_panel(self):
        main_panel = ctk.CTkFrame(self, fg_color=COLORS["bg_primary"], corner_radius=0)
        main_panel.grid(row=0, column=1, sticky="nsew")
        main_panel.grid_columnconfigure(0, weight=1)
        main_panel.grid_rowconfigure(3, weight=1)
        
        title = ctk.CTkLabel(main_panel, text="RAG Essay Writer",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"), text_color=COLORS["text_primary"])
        title.grid(row=0, column=0, padx=20, pady=(20, 12), sticky="w")
        
        # Input section
        input_frame = ctk.CTkFrame(main_panel, fg_color=COLORS["bg_secondary"], corner_radius=6)
        input_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 12))
        input_frame.grid_columnconfigure(0, weight=1)
        
        lbl = ctk.CTkLabel(input_frame, text="Prompt", font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS["text_secondary"])
        lbl.grid(row=0, column=0, padx=12, pady=(12, 2), sticky="w")
        Tooltip(lbl, "The main topic or question for your essay")
        
        self.prompt_entry = ctk.CTkTextbox(input_frame, height=90, font=ctk.CTkFont(size=12),
            fg_color=COLORS["bg_input"], text_color=COLORS["text_primary"],
            border_width=1, border_color=COLORS["border"])
        self.prompt_entry.grid(row=1, column=0, padx=12, pady=(0, 6), sticky="ew")
        
        lbl = ctk.CTkLabel(input_frame, text="Context", font=ctk.CTkFont(size=11), text_color=COLORS["text_muted"])
        lbl.grid(row=2, column=0, padx=12, pady=(6, 2), sticky="w")
        Tooltip(lbl, "Optional: audience, purpose, constraints")
        
        self.context_entry = ctk.CTkTextbox(input_frame, height=70, font=ctk.CTkFont(size=11),
            fg_color=COLORS["bg_input"], text_color=COLORS["text_primary"],
            border_width=1, border_color=COLORS["border"])
        self.context_entry.grid(row=3, column=0, padx=12, pady=(0, 6), sticky="ew")
        
        lbl = ctk.CTkLabel(input_frame, text="Web Search", font=ctk.CTkFont(size=11), text_color=COLORS["text_muted"])
        lbl.grid(row=4, column=0, padx=12, pady=(6, 2), sticky="w")
        Tooltip(lbl, "Optional search for current information")
        
        self.search_entry = ctk.CTkEntry(input_frame, font=ctk.CTkFont(size=11),
            fg_color=COLORS["bg_input"], text_color=COLORS["text_primary"],
            border_width=1, border_color=COLORS["border"],
            placeholder_text="Optional search query", placeholder_text_color=COLORS["text_muted"])
        self.search_entry.grid(row=5, column=0, padx=12, pady=(0, 12), sticky="ew")
        
        # Action buttons
        btn_frame = ctk.CTkFrame(main_panel, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 12))
        
        self.generate_btn = ctk.CTkButton(btn_frame, text="Generate", font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            height=36, width=120, command=self._generate_essay)
        self.generate_btn.pack(side="left", padx=(0, 6))
        Tooltip(self.generate_btn, "Generate essay (Ctrl+Enter)")
        
        self.regenerate_btn = ctk.CTkButton(btn_frame, text="Regenerate", font=ctk.CTkFont(size=11),
            fg_color=COLORS["bg_tertiary"], hover_color=COLORS["border"],
            height=36, width=100, command=self._regenerate_essay)
        self.regenerate_btn.pack(side="left", padx=(0, 6))
        
        self.save_btn = ctk.CTkButton(btn_frame, text="Save", font=ctk.CTkFont(size=11),
            fg_color=COLORS["bg_tertiary"], hover_color=COLORS["border"],
            height=36, width=70, command=self._save_essay)
        self.save_btn.pack(side="left")
        
        self.word_count_label = ctk.CTkLabel(btn_frame, text="", font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"])
        self.word_count_label.pack(side="right")
        
        # Output area
        output_frame = ctk.CTkFrame(main_panel, fg_color=COLORS["bg_secondary"], corner_radius=6)
        output_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 12))
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(output_frame, text="Output", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_secondary"]).grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")
        
        self.essay_output = ctk.CTkTextbox(output_frame, font=ctk.CTkFont(family="Georgia", size=12),
            fg_color=COLORS["bg_input"], text_color=COLORS["text_primary"],
            border_width=1, border_color=COLORS["border"], wrap="word")
        self.essay_output.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 6))
        
        # Edit bar
        edit_frame = ctk.CTkFrame(output_frame, fg_color="transparent")
        edit_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        edit_frame.grid_columnconfigure(0, weight=1)
        
        self.edit_entry = ctk.CTkEntry(edit_frame, placeholder_text="Feedback to edit essay",
            placeholder_text_color=COLORS["text_muted"], font=ctk.CTkFont(size=11),
            fg_color=COLORS["bg_input"], text_color=COLORS["text_primary"],
            border_width=1, border_color=COLORS["border"])
        self.edit_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        
        edit_btn = ctk.CTkButton(edit_frame, text="Edit", font=ctk.CTkFont(size=11),
            fg_color=COLORS["bg_tertiary"], hover_color=COLORS["border"], width=60, command=self._edit_essay)
        edit_btn.grid(row=0, column=1)
        
    def _create_right_panel(self):
        right_panel = ctk.CTkFrame(self, fg_color=COLORS["bg_secondary"], corner_radius=0)
        right_panel.grid(row=0, column=2, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight=3)
        right_panel.grid_rowconfigure(1, weight=1)
        
        self.settings_panel = SettingsPanel(right_panel, self.writer, 
            on_model_change=self._on_model_change, on_mode_change=self._on_mode_change)
        self.settings_panel.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        console_frame = ctk.CTkFrame(right_panel, fg_color=COLORS["bg_tertiary"], corner_radius=0)
        console_frame.grid(row=1, column=0, sticky="nsew")
        console_frame.grid_columnconfigure(0, weight=1)
        console_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(console_frame, text="Console", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_secondary"]).grid(row=0, column=0, padx=8, pady=(8, 4), sticky="w")
        
        self.console = ctk.CTkTextbox(console_frame, font=ctk.CTkFont(family="Consolas", size=9),
            fg_color=COLORS["bg_input"], text_color=COLORS["text_muted"], border_width=0)
        self.console.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        self.console.configure(state="disabled")
        
        self._original_stdout = sys.stdout
        sys.stdout = OutputRedirector(self.console)
        
    def _generate_essay(self):
        prompt = self.prompt_entry.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning("Missing Prompt", "Please enter a prompt.")
            return
            
        context = self.context_entry.get("1.0", "end").strip() or ""
        search = self.search_entry.get().strip() or None
        mode = self.settings_panel.get_mode()
        
        self.generate_btn.configure(state="disabled", text="...")
        self.essay_output.configure(state="normal")
        self.essay_output.delete("1.0", "end")
        self.essay_output.insert("1.0", f"Generating with {mode} mode...")
        self.essay_output.configure(state="disabled")
        
        def generate():
            try:
                # Web search if requested
                if search:
                    self.writer.web_search(search)
                
                if mode == "Classic":
                    # Classic RAG mode
                    essay, style_sources, research_sources = self.writer.generate_essay(prompt, user_context=context)
                else:
                    # Hybrid pipeline mode
                    # First retrieve passages using existing RAG
                    if self.writer.use_mmr:
                        results = self.writer.retrieve_mmr(prompt, top_k=self.writer.top_k)
                    else:
                        results = self.writer.retrieve(prompt, top_k=self.writer.top_k)
                    
                    passages = "\n\n---\n\n".join([r['text'] for r in results])
                    
                    # Configure pipeline
                    config = self.settings_panel.get_pipeline_config()
                    self.pipeline.config.skip_critique = config["skip_critique"]
                    self.pipeline.config.writer_model = self.current_model
                    
                    # Run pipeline
                    if mode == "Hybrid Quick":
                        essay = self.pipeline.run_quick(prompt, passages, context)
                    else:
                        if config["dual_draft"]:
                            essay = self.pipeline.run(prompt, passages, context)
                        else:
                            # Single draft variant
                            essay = self.pipeline.run_quick(prompt, passages, context)
                    
                    style_sources = results
                    research_sources = []
                
                self.after(0, lambda: self._display_essay(essay, style_sources, research_sources))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.generate_btn.configure(state="normal", text="Generate"))
                
        threading.Thread(target=generate, daemon=True).start()
        
    def _regenerate_essay(self):
        if not self.writer.last_prompt:
            messagebox.showinfo("No Essay", "No previous essay.")
            return
            
        # Re-run with same prompt
        self.prompt_entry.delete("1.0", "end")
        self.prompt_entry.insert("1.0", self.writer.last_prompt)
        if self.writer.last_context:
            self.context_entry.delete("1.0", "end")
            self.context_entry.insert("1.0", self.writer.last_context)
        self._generate_essay()
        
    def _edit_essay(self):
        if not self.writer.last_essay:
            messagebox.showinfo("No Essay", "No essay to edit.")
            return
        feedback = self.edit_entry.get().strip()
        if not feedback:
            messagebox.showwarning("Missing Feedback", "Enter feedback.")
            return
            
        self.generate_btn.configure(state="disabled", text="...")
        
        def edit():
            try:
                # Always use classic mode for edits
                essay, style_sources, research_sources = self.writer.generate_essay(
                    self.writer.last_prompt, revision_feedback=feedback,
                    previous_essay=self.writer.last_essay, user_context=self.writer.last_context
                )
                self.after(0, lambda: self._display_essay(essay, style_sources, research_sources))
                self.after(0, lambda: self.edit_entry.delete(0, "end"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.generate_btn.configure(state="normal", text="Generate"))
        threading.Thread(target=edit, daemon=True).start()
        
    def _display_essay(self, essay: str, style_sources: list, research_sources: list):
        self.essay_output.configure(state="normal")
        self.essay_output.delete("1.0", "end")
        self.essay_output.insert("1.0", essay)
        self.essay_output.configure(state="disabled")
        
        words = len(essay.split())
        self.word_count_label.configure(text=f"{words:,} words")
        
        self.writer.last_essay = essay
        self.writer.last_sources = style_sources
        self.writer.last_research_sources = research_sources
        
        prompt = self.prompt_entry.get("1.0", "end").strip()
        self.writer.last_prompt = prompt
        self.writer.last_context = self.context_entry.get("1.0", "end").strip()
        
        self.writer.essay_history.append({
            'prompt': prompt, 'essay': essay, 'style_sources': style_sources,
            'research_sources': research_sources, 'type': 'new',
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S")
        })
        
    def _save_essay(self):
        if not self.writer.last_essay:
            messagebox.showinfo("No Essay", "No essay to save.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("Markdown", "*.md"), ("All", "*.*")],
            title="Save Essay"
        )
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.writer.last_essay)
            messagebox.showinfo("Saved", f"Saved to: {filepath}")
            
    def destroy(self):
        sys.stdout = self._original_stdout
        super().destroy()


def main():
    app = RAGWriterGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
