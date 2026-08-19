"""Complexity-aware tool routing skeleton; no external model calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ComplexitySignals:
    """Provide complexity signals behavior."""
    scene_complexity: str = "unknown"
    text_density: str = "unknown"
    object_clutter: str = "unknown"
    small_object_level: str = "unknown"
    occlusion_level: str = "unknown"
    motion_blur: bool = False
    defocus_blur: bool = False
    reflection: bool = False
    ghosting: bool = False


@dataclass(frozen=True)
class RouteDecision:
    """Provide route decision behavior."""
    image_id: str
    complexity: str
    confidence: float
    branches: tuple[str, ...]
    reasons: tuple[str, ...]
    policy_version: str = "complexity_route_v1"

    def as_dict(self) -> dict[str, Any]:
        """Execute the as dict operation."""
        return {
            "image_id": self.image_id,
            "complexity": self.complexity,
            "confidence": self.confidence,
            "branches": list(self.branches),
            "reasons": list(self.reasons),
            "policy_version": self.policy_version,
        }


def decide_route(image_id: str, signals: ComplexitySignals | Mapping[str, Any]) -> RouteDecision:
    """Choose optional branches from precomputed signals only.

    This policy deliberately does not call OCR, Grounding, SAM, or a VLM. It
    makes no semantic claim and is intended for dry-run/configuration tests.
    """

    if not isinstance(signals, ComplexitySignals):
        signals = ComplexitySignals(**{key: value for key, value in dict(signals).items() if key in ComplexitySignals.__dataclass_fields__})
    branches = ["global_vlm"]
    reasons: list[str] = []
    hard = signals.scene_complexity == "complex" or signals.object_clutter == "high" or signals.occlusion_level == "high"
    text = signals.text_density == "high"
    presence = signals.reflection or signals.ghosting or signals.motion_blur or signals.defocus_blur
    if text:
        branches.append("ocr")
        reasons.append("text_density_high")
    if hard or signals.small_object_level in {"medium", "high"}:
        branches.extend(["grounding", "local_vlm"])
        reasons.append("clutter_or_small_object_or_occlusion")
    if presence:
        branches.append("presence_uncertainty")
        reasons.append("reflection_or_blur_or_ghosting")
    if hard and presence:
        branches.append("conflict_review")
        reasons.append("complex_presence_conflict_risk")
    complexity = signals.scene_complexity
    if complexity == "unknown":
        complexity = "complex" if len(branches) > 1 else "unknown"
    confidence = 0.9 if signals.scene_complexity in {"simple", "moderate", "complex"} else 0.4
    return RouteDecision(image_id, complexity, confidence, tuple(dict.fromkeys(branches)), tuple(reasons))
