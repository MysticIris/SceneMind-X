"""Single typed source of truth for the Phase 2A RC3 payload and annotation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


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


class TextPresence(str, Enum):
    """Provide text presence behavior."""
    NONE = "none"
    POSSIBLE = "possible"
    PRESENT = "present"


class Legibility(str, Enum):
    """Provide legibility behavior."""
    NONE = "none"
    LOW = "low"
    PARTIAL = "partial"
    HIGH = "high"
    MIXED = "mixed"


class Confidence(str, Enum):
    """Provide confidence behavior."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


ShortPhrase = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=48)]
ShortSummary = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=96)]
ShortUncertainty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
VisibleText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class Rc3TextAssessment(_StrictModel):
    """Provide rc3 text assessment behavior."""
    presence: TextPresence
    legibility: Legibility
    selected_text: Annotated[list[VisibleText], Field(min_length=1, max_length=4)] | None
    summary: ShortSummary | None
    uncertainty: ShortUncertainty | None

    @model_validator(mode="after")
    def validate_text_state(self) -> "Rc3TextAssessment":
        """Validate text state."""
        if self.presence is TextPresence.NONE:
            if self.legibility is not Legibility.NONE or self.selected_text is not None:
                raise ValueError("presence=none requires legibility=none and selected_text=null")
        elif self.presence is TextPresence.POSSIBLE:
            if self.selected_text is not None or self.uncertainty is None:
                raise ValueError("presence=possible requires selected_text=null and uncertainty")
        elif self.legibility is Legibility.NONE:
            raise ValueError("presence=present cannot use legibility=none")
        if self.selected_text is not None and self.legibility in {Legibility.NONE, Legibility.LOW}:
            raise ValueError("selected_text requires partial, high, or mixed legibility")
        return self


class Rc3Inference(_StrictModel):
    """Provide rc3 inference behavior."""
    text: ShortSummary
    support: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=1, max_length=3)]
    confidence: Confidence


class Rc3ModelPayload(_StrictModel):
    """Provide rc3 model payload behavior."""
    visual_medium: VisualMedium
    depicted_content: ShortSummary | None
    scene_summary: ShortSummary | None
    subjects: Annotated[list[ShortPhrase], Field(max_length=6)]
    activities: Annotated[list[ShortPhrase], Field(max_length=4)]
    relationships: Annotated[list[ShortPhrase], Field(max_length=4)]
    text_assessment: Rc3TextAssessment
    observations: Annotated[list[ShortSummary], Field(min_length=1, max_length=6)]
    inferences: Annotated[list[Rc3Inference], Field(max_length=4)]


class Rc3AnnotationMeta(_StrictModel):
    """Provide rc3 annotation meta behavior."""
    schema_version: Literal["visual_asset_annotation_v1_1_rc3"]
    prompt_version: Literal["p3_shared_fact_v1_4_rc3"]
    run_id: Annotated[str, StringConstraints(pattern=r"^run_[0-9]{8}_[0-9]{6}$")]
    image_id: Annotated[str, StringConstraints(pattern=r"^[^/\\]+\.[A-Za-z0-9]+$")]
    source_split: Literal["train"]
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    generated_at: datetime
    model_name: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    model_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    max_new_tokens: Annotated[int, Field(ge=1, le=768)]
    constrained_decoding: Literal["none", "xgrammar", "outlines", "guidance", "native"]


class Rc3CanonicalSubject(_StrictModel):
    """Provide rc3 canonical subject behavior."""
    subject_id: Annotated[str, StringConstraints(pattern=r"^sub_[0-9]{3}$")]
    text: ShortPhrase


class Rc3CanonicalActivity(_StrictModel):
    """Provide rc3 canonical activity behavior."""
    activity_id: Annotated[str, StringConstraints(pattern=r"^act_[0-9]{3}$")]
    text: ShortPhrase


class Rc3CanonicalRelationship(_StrictModel):
    """Provide rc3 canonical relationship behavior."""
    relationship_id: Annotated[str, StringConstraints(pattern=r"^rel_[0-9]{3}$")]
    text: ShortPhrase


class Rc3EvidenceItem(_StrictModel):
    """Provide rc3 evidence item behavior."""
    evidence_id: Annotated[str, StringConstraints(pattern=r"^ev_[0-9]{3}$")]
    source_type: Literal["visual_observation"]
    content: ShortSummary


class Rc3Claim(_StrictModel):
    """Provide rc3 claim behavior."""
    claim_id: Annotated[str, StringConstraints(pattern=r"^cl_[0-9]{3}$")]
    claim_type: Literal["reasonable_inference"]
    text: ShortSummary
    evidence_refs: Annotated[
        list[Annotated[str, StringConstraints(pattern=r"^ev_[0-9]{3}$")]],
        Field(min_length=1, max_length=3),
    ]
    confidence: Confidence


class Rc3CanonicalAnnotation(_StrictModel):
    """Provide rc3 canonical annotation behavior."""
    annotation_meta: Rc3AnnotationMeta
    visual_medium: VisualMedium
    depicted_content: ShortSummary | None
    scene_summary: ShortSummary | None
    subjects: Annotated[list[Rc3CanonicalSubject], Field(max_length=6)]
    activities: Annotated[list[Rc3CanonicalActivity], Field(max_length=4)]
    relationships: Annotated[list[Rc3CanonicalRelationship], Field(max_length=4)]
    text_assessment: Rc3TextAssessment
    evidence_items: Annotated[list[Rc3EvidenceItem], Field(min_length=1, max_length=6)]
    claims: Annotated[list[Rc3Claim], Field(max_length=4)]
