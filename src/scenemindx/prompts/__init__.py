"""Versioned prompt templates namespace."""

from .loader import (
    PromptBundleSpec,
    PromptSpec,
    load_prompt,
    load_prompt_bundle_registry,
    load_prompt_manifest,
)
from .phase2b import Phase2bPromptModule, Phase2bPromptStage, load_phase2b_prompt_registry

__all__ = [
    "PromptBundleSpec",
    "PromptSpec",
    "load_prompt",
    "load_prompt_bundle_registry",
    "load_prompt_manifest",
    "Phase2bPromptModule",
    "Phase2bPromptStage",
    "load_phase2b_prompt_registry",
]
