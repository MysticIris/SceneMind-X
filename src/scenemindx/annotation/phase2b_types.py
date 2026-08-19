"""Single typed source of truth for the Phase 2B annotation pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class _StrictModel(BaseModel):
    """Provide strict model behavior."""
    model_config = ConfigDict(extra="forbid", strict=True)


class VisualMedium(str, Enum):
    """Provide visual medium behavior."""
    REAL_WORLD_PHOTO = "real_world_photo"
    SCREENSHOT = "screenshot"
    PRESENTATION_SLIDE = "presentation_slide"
    MEME_OR_REACTION_IMAGE = "meme_or_reaction_image"
    POSTER_OR_ADVERTISEMENT = "poster_or_advertisement"
    DOCUMENT_OR_PAPER = "document_or_paper"
    WEBPAGE_OR_USER_INTERFACE = "webpage_or_user_interface"
    CHAT_SCREENSHOT = "chat_screenshot"
    CHART_OR_INFOGRAPHIC = "chart_or_infographic"
    ILLUSTRATION_OR_COMIC = "illustration_or_comic"
    PACKAGING = "packaging"
    PHOTOGRAPHED_SCREEN_CONTENT = "photographed_screen_content"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class MediaConfidence(str, Enum):
    """Provide media confidence behavior."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


ShortPhrase = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=96)]
ShortSummary = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
VisibleText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]


class MediaRouterOutput(_StrictModel):
    """Stage A output. It deliberately cannot carry semantic content."""

    visual_medium: VisualMedium
    confidence: MediaConfidence


class SemanticPayload(_StrictModel):
    """Stage B model-owned payload: nine flat semantic fields only."""

    depicted_content: ShortSummary | None = None
    scene_summary: ShortSummary
    subjects: Annotated[list[ShortPhrase], Field(default_factory=list, max_length=12)]
    activities: Annotated[list[ShortPhrase], Field(default_factory=list, max_length=8)]
    relationships: Annotated[list[ShortPhrase], Field(default_factory=list, max_length=12)]
    visible_text_candidates: Annotated[list[VisibleText], Field(default_factory=list, max_length=8)]
    observations: Annotated[list[ShortSummary], Field(min_length=1, max_length=12)]
    inference_candidates: Annotated[list[ShortSummary], Field(default_factory=list, max_length=8)]
    uncertainties: Annotated[list[ShortSummary], Field(default_factory=list, max_length=8)]


class Phase2bAnnotationMeta(_StrictModel):
    """Provide phase2b annotation meta behavior."""
    schema_version: Literal["visual_asset_annotation_phase2b_v1"]
    pipeline_version: Literal["phase2b_v1"]
    media_prompt_version: Literal["phase2b_media_router_v1"]
    semantic_prompt_version: Literal["phase2b_semantic_extractor_v1"]
    run_id: Annotated[str, StringConstraints(pattern=r"^phase2b_[A-Za-z0-9_\-]+$")]
    image_id: Annotated[str, StringConstraints(pattern=r"^[^/\\]+\.[A-Za-z0-9]+$")]
    source_split: Literal["train"]
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    generated_at: datetime
    model_name: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    model_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    media_max_new_tokens: Annotated[int, Field(ge=1, le=256)]
    semantic_max_new_tokens: Annotated[int, Field(ge=1, le=768)]
    constrained_decoding: Literal["none", "outlines_canary_only", "native"]


class CanonicalSubject(_StrictModel):
    """Provide canonical subject behavior."""
    subject_id: Annotated[str, StringConstraints(pattern=r"^sub_[0-9]{3}$")]
    text: ShortPhrase


class CanonicalActivity(_StrictModel):
    """Provide canonical activity behavior."""
    activity_id: Annotated[str, StringConstraints(pattern=r"^act_[0-9]{3}$")]
    text: ShortPhrase


class CanonicalRelationship(_StrictModel):
    """Provide canonical relationship behavior."""
    relationship_id: Annotated[str, StringConstraints(pattern=r"^rel_[0-9]{3}$")]
    text: ShortPhrase


class UnverifiedTextCandidate(_StrictModel):
    """Provide unverified text candidate behavior."""
    candidate_id: Annotated[str, StringConstraints(pattern=r"^txt_[0-9]{3}$")]
    text: VisibleText
    verification_source: Literal["vlm_only"]
    verification_status: Literal["unverified"]


class PolicyOverride(_StrictModel):
    """Provide policy override behavior."""
    path: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    detected_claim: ShortSummary
    action: Literal["ignored_verification_claim"]


class TextGovernance(_StrictModel):
    """Provide text governance behavior."""
    verification_source: Literal["vlm_only"]
    verification_status: Literal["unverified"]
    selected_text: None
    unverified_text_candidates: Annotated[list[UnverifiedTextCandidate], Field(max_length=8)]
    policy_overrides: Annotated[list[PolicyOverride], Field(max_length=16)]


class EvidenceItem(_StrictModel):
    """Provide evidence item behavior."""
    evidence_id: Annotated[str, StringConstraints(pattern=r"^ev_[0-9]{3}$")]
    source_type: Literal["visual_observation"]
    verification_status: Literal["vlm_unverified"]
    content: ShortSummary


class DirectObservationClaim(_StrictModel):
    """Provide direct observation claim behavior."""
    claim_id: Annotated[str, StringConstraints(pattern=r"^cl_[0-9]{3}$")]
    claim_type: Literal["direct_observation"]
    text: ShortSummary
    evidence_refs: Annotated[
        list[Annotated[str, StringConstraints(pattern=r"^ev_[0-9]{3}$")]],
        Field(min_length=1, max_length=1),
    ]
    verification_status: Literal["vlm_unverified"]


class CanonicalInferenceCandidate(_StrictModel):
    """Provide canonical inference candidate behavior."""
    inference_id: Annotated[str, StringConstraints(pattern=r"^inf_[0-9]{3}$")]
    text: ShortSummary
    verification_status: Literal["unverified"]
    promoted_to_claim: Literal[False]


class FinalCanonicalAnnotation(_StrictModel):
    """Provide final canonical annotation behavior."""
    annotation_meta: Phase2bAnnotationMeta
    visual_medium: VisualMedium
    medium_confidence: MediaConfidence
    depicted_content: ShortSummary | None
    scene_summary: ShortSummary
    subjects: Annotated[list[CanonicalSubject], Field(max_length=12)]
    activities: Annotated[list[CanonicalActivity], Field(max_length=8)]
    relationships: Annotated[list[CanonicalRelationship], Field(max_length=12)]
    text_governance: TextGovernance
    evidence_items: Annotated[list[EvidenceItem], Field(min_length=1, max_length=12)]
    claims: Annotated[list[DirectObservationClaim], Field(min_length=1, max_length=12)]
    inference_candidates: Annotated[list[CanonicalInferenceCandidate], Field(max_length=8)]
    uncertainties: Annotated[list[ShortSummary], Field(max_length=8)]
