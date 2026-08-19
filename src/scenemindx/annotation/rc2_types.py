"""Typed compact shared-fact structures for p3_shared_fact_v1_4_rc2."""

from __future__ import annotations

from typing import Literal, TypedDict


MediumType = Literal[
    "real_world_photo", "screenshot", "presentation_slide",
    "meme_or_reaction_image", "poster_or_advertisement", "document_or_paper",
    "webpage_or_user_interface", "chat_screenshot", "chart_or_infographic",
    "illustration_or_comic", "packaging", "photographed_screen_content",
    "mixed", "uncertain",
]
ClaimType = Literal["direct_observation", "reasonable_inference", "uncertain", "abstention"]


class VisualMedium(TypedDict):
    """Provide visual medium behavior."""
    primary_type: MediumType
    secondary_types: list[MediumType]
    evidence_refs: list[str]
    confidence: float
    uncertainty_reason: str | None


class DepictedContent(TypedDict):
    """Provide depicted content behavior."""
    summary: str | None
    real_world_event_status: Literal["directly_observed", "depicted_only", "mixed", "uncertain", "unknown"]
    evidence_refs: list[str]
    uncertainty_reason: str | None


class Attribute(TypedDict):
    """Provide attribute behavior."""
    key: str
    value: str


class Subject(TypedDict):
    """Provide subject behavior."""
    subject_id: str
    name: str
    salience: Literal["primary", "secondary", "background", "uncertain"]
    medium_layer: Literal["scene_direct", "depicted_inside_medium", "both", "uncertain"]
    attributes: list[Attribute]
    evidence_refs: list[str]
    confidence: float
    uncertainty_reason: str | None


class Activity(TypedDict):
    """Provide activity behavior."""
    subject_ref: str
    action: str
    evidence_refs: list[str]
    confidence: float
    uncertainty_reason: str | None


class Relationship(TypedDict):
    """Provide relationship behavior."""
    subject_ref: str
    relation: str
    object_ref: str | None
    object_text: str | None
    evidence_refs: list[str]
    confidence: float
    uncertainty_reason: str | None


class SelectedText(TypedDict):
    """Provide selected text behavior."""
    text: str
    language: str
    script_type: Literal["printed", "handwritten", "artistic", "digital", "mixed", "unknown"]
    legibility: Literal["none", "low", "partial", "high", "mixed"]
    direction: Literal["left_to_right", "right_to_left", "top_to_bottom", "mixed", "unknown"]
    region_ref: str | None
    confidence: float
    uncertainty_reason: str | None


class TextAssessment(TypedDict):
    """Provide text assessment behavior."""
    text_presence: Literal["none", "possible", "present"]
    legibility: Literal["none", "low", "partial", "high", "mixed"]
    suspected_false_positive: bool
    false_positive_contexts: list[Literal["footprint", "snow", "texture", "glare", "reflection", "compression_artifact", "other"]]
    selected_text: list[SelectedText]
    scene_inference_allowed: bool
    uncertainty_reason: str | None


class EvidenceItem(TypedDict):
    """Provide evidence item behavior."""
    evidence_id: str
    source_type: Literal["visual_global", "visual_region", "model_text_assessment"]
    content: str
    region_ref: str | None
    tool_ref: str | None
    confidence: float


class Claim(TypedDict):
    """Provide claim behavior."""
    claim_id: str
    claim_type: ClaimType
    text: str
    evidence_refs: list[str]
    confidence: float
    uncertainty_reason: str | None


class Rc2SharedFacts(TypedDict):
    """Provide rc2 shared facts behavior."""
    visual_medium: VisualMedium
    depicted_content: DepictedContent
    scene_summary: str | None
    subjects: list[Subject]
    activities: list[Activity]
    relationships: list[Relationship]
    text_assessment: TextAssessment
    evidence_items: list[EvidenceItem]
    claims: list[Claim]
