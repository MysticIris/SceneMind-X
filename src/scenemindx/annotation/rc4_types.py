"""Single typed source of truth for the Phase 2A RC4 payload and annotation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class _StrictModel(BaseModel):
    """Provide strict model behavior."""
    model_config = ConfigDict(extra="forbid", strict=True)


class Rc4VisualMedium(str, Enum):
    """Provide rc4 visual medium behavior."""
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


class Rc4TextPresence(str, Enum):
    """Provide rc4 text presence behavior."""
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class Rc4TextRole(str, Enum):
    """Provide rc4 text role behavior."""
    DECISION_CRITICAL = "decision_critical"
    CONTEXTUAL = "contextual"
    DECORATIVE = "decorative"
    INCIDENTAL = "incidental"
    MIXED = "mixed"
    NONE = "none"
    UNCERTAIN = "uncertain"


class Rc4TextReliability(str, Enum):
    """Provide rc4 text reliability behavior."""
    VERIFIED = "verified"
    LIKELY = "likely"
    UNCERTAIN = "uncertain"
    LOW = "low"
    UNREADABLE = "unreadable"
    NONE = "none"


ShortPhrase = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=96)]
ShortSummary = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
VisibleText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]


class Rc4ModelPayload(_StrictModel):
    """The model-owned payload: flat primitives and string arrays only."""

    visual_medium: Rc4VisualMedium
    depicted_content: ShortSummary | None = None
    scene_summary: ShortSummary
    subjects: Annotated[list[ShortPhrase], Field(default_factory=list, max_length=12)]
    activities: Annotated[list[ShortPhrase], Field(default_factory=list, max_length=8)]
    relationships: Annotated[list[ShortPhrase], Field(default_factory=list, max_length=12)]
    text_presence: Rc4TextPresence
    text_role: Rc4TextRole
    text_reliability: Rc4TextReliability
    visible_text_candidates: Annotated[list[VisibleText], Field(default_factory=list, max_length=8)]
    selected_text: VisibleText | None = None
    text_summary: ShortSummary | None = None
    text_uncertainty: ShortSummary | None = None
    observations: Annotated[list[ShortSummary], Field(min_length=1, max_length=12)]
    inference_candidates: Annotated[list[ShortSummary], Field(default_factory=list, max_length=8)]
    uncertainties: Annotated[list[ShortSummary], Field(default_factory=list, max_length=8)]

    @model_validator(mode="after")
    def validate_text_state(self) -> "Rc4ModelPayload":
        """Validate text state."""
        if self.selected_text is not None:
            raise ValueError("selected_text must remain null without OCR or human verification")
        if self.text_reliability is Rc4TextReliability.VERIFIED:
            raise ValueError("text_reliability=verified requires OCR or human verification")
        if self.text_presence is Rc4TextPresence.ABSENT:
            if self.text_role is not Rc4TextRole.NONE or self.text_reliability is not Rc4TextReliability.NONE:
                raise ValueError("text_presence=absent requires text_role=none and text_reliability=none")
            if self.visible_text_candidates or self.text_uncertainty is not None:
                raise ValueError("text_presence=absent cannot contain text candidates or uncertainty")
        else:
            if self.text_role is Rc4TextRole.NONE or self.text_reliability is Rc4TextReliability.NONE:
                raise ValueError("present or uncertain text requires a non-none role and reliability")
        return self


class Rc4AnnotationMeta(_StrictModel):
    """Provide rc4 annotation meta behavior."""
    schema_version: Literal["visual_asset_annotation_v1_1_rc4"]
    prompt_version: Literal["p3_shared_fact_v1_4_rc4"]
    run_id: Annotated[str, StringConstraints(pattern=r"^run_[0-9]{8}_[0-9]{6}$")]
    image_id: Annotated[str, StringConstraints(pattern=r"^[^/\\]+\.[A-Za-z0-9]+$")]
    source_split: Literal["train"]
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    generated_at: datetime
    model_name: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    model_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    max_new_tokens: Annotated[int, Field(ge=1, le=640)]
    constrained_decoding: Literal["none", "xgrammar", "outlines", "guidance", "native"]


class Rc4CanonicalSubject(_StrictModel):
    """Provide rc4 canonical subject behavior."""
    subject_id: Annotated[str, StringConstraints(pattern=r"^sub_[0-9]{3}$")]
    text: ShortPhrase


class Rc4CanonicalActivity(_StrictModel):
    """Provide rc4 canonical activity behavior."""
    activity_id: Annotated[str, StringConstraints(pattern=r"^act_[0-9]{3}$")]
    text: ShortPhrase


class Rc4CanonicalRelationship(_StrictModel):
    """Provide rc4 canonical relationship behavior."""
    relationship_id: Annotated[str, StringConstraints(pattern=r"^rel_[0-9]{3}$")]
    text: ShortPhrase


class Rc4TextCandidate(_StrictModel):
    """Provide rc4 text candidate behavior."""
    candidate_id: Annotated[str, StringConstraints(pattern=r"^txt_[0-9]{3}$")]
    text: VisibleText
    source_type: Literal["vlm_unverified"]
    verification_status: Literal["unverified"]
    role: Rc4TextRole
    reliability: Rc4TextReliability


class Rc4TextGovernance(_StrictModel):
    """Provide rc4 text governance behavior."""
    presence: Rc4TextPresence
    role: Rc4TextRole
    reliability: Rc4TextReliability
    selected_text: None
    selected_text_status: Literal["none", "withheld_unverified"]
    unverified_candidates: Annotated[list[Rc4TextCandidate], Field(max_length=8)]
    summary: ShortSummary | None
    uncertainty: ShortSummary | None


class Rc4EvidenceItem(_StrictModel):
    """Provide rc4 evidence item behavior."""
    evidence_id: Annotated[str, StringConstraints(pattern=r"^ev_[0-9]{3}$")]
    source_type: Literal["visual_observation"]
    verification_status: Literal["vlm_unverified"]
    content: ShortSummary


class Rc4DirectObservationClaim(_StrictModel):
    """Provide rc4 direct observation claim behavior."""
    claim_id: Annotated[str, StringConstraints(pattern=r"^cl_[0-9]{3}$")]
    claim_type: Literal["direct_observation"]
    text: ShortSummary
    evidence_refs: Annotated[
        list[Annotated[str, StringConstraints(pattern=r"^ev_[0-9]{3}$")]],
        Field(min_length=1, max_length=1),
    ]
    verification_status: Literal["vlm_unverified"]


class Rc4InferenceCandidate(_StrictModel):
    """Provide rc4 inference candidate behavior."""
    inference_id: Annotated[str, StringConstraints(pattern=r"^inf_[0-9]{3}$")]
    text: ShortSummary
    verification_status: Literal["unverified"]
    promoted_to_claim: Literal[False]


class Rc4CanonicalAnnotation(_StrictModel):
    """Provide rc4 canonical annotation behavior."""
    annotation_meta: Rc4AnnotationMeta
    visual_medium: Rc4VisualMedium
    depicted_content: ShortSummary | None
    scene_summary: ShortSummary
    subjects: Annotated[list[Rc4CanonicalSubject], Field(max_length=12)]
    activities: Annotated[list[Rc4CanonicalActivity], Field(max_length=8)]
    relationships: Annotated[list[Rc4CanonicalRelationship], Field(max_length=12)]
    text_governance: Rc4TextGovernance
    evidence_items: Annotated[list[Rc4EvidenceItem], Field(min_length=1, max_length=12)]
    claims: Annotated[list[Rc4DirectObservationClaim], Field(min_length=1, max_length=12)]
    inference_candidates: Annotated[list[Rc4InferenceCandidate], Field(max_length=8)]
    uncertainties: Annotated[list[ShortSummary], Field(max_length=8)]
