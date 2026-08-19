"""Phase 2B two-stage validation and deterministic canonicalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, TypeVar

from pydantic import BaseModel, ValidationError

from .phase2b_types import (
    FinalCanonicalAnnotation,
    MediaRouterOutput,
    SemanticPayload,
)
from .schema import AnnotationValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE2B_PIPELINE_VERSION = "phase2b_v1"
MEDIA_PROMPT_VERSION = "phase2b_media_router_v1"
SEMANTIC_PROMPT_VERSION = "phase2b_semantic_extractor_v1"
MEDIA_SCHEMA_ID = "phase2b_media_router_output_v1"
SEMANTIC_SCHEMA_ID = "phase2b_semantic_payload_v1"
CANONICAL_SCHEMA_ID = "visual_asset_annotation_phase2b_v1"
MEDIA_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "phase2b_media_router_output.schema.json"
SEMANTIC_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "phase2b_semantic_payload.schema.json"
CANONICAL_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "visual_asset_annotation_phase2b_v1.schema.json"
MEDIA_FIELDS = tuple(MediaRouterOutput.model_fields)
SEMANTIC_FIELDS = tuple(SemanticPayload.model_fields)

_VERIFICATION_CLAIM_RE = re.compile(
    r"(?:已经?|已|人工|OCR)核验|(?:human[_ -]?verified|ocr[_ -]?verified|verified)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StageParseResult:
    """Represent stage parse result data."""
    value: dict[str, Any] | None
    error: str | None
    repair_type: str | None
    json_unclosed: bool


@dataclass(frozen=True)
class CanonicalContext:
    """Provide canonical context behavior."""
    run_id: str
    image_id: str
    source_sha256: str
    generated_at: datetime
    model_name: str
    model_revision: str
    media_max_new_tokens: int
    semantic_max_new_tokens: int
    constrained_decoding: str = "none"


ModelT = TypeVar("ModelT", bound=BaseModel)


def _schema(model: type[BaseModel], schema_id: str) -> dict[str, Any]:
    value = model.model_json_schema(mode="validation")
    value.update({"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": schema_id})
    return value


def media_router_schema() -> dict[str, Any]:
    """Execute the media router schema operation."""
    return _schema(MediaRouterOutput, MEDIA_SCHEMA_ID)


def semantic_payload_schema() -> dict[str, Any]:
    """Execute the semantic payload schema operation."""
    return _schema(SemanticPayload, SEMANTIC_SCHEMA_ID)


def final_canonical_schema() -> dict[str, Any]:
    """Execute the final canonical schema operation."""
    return _schema(FinalCanonicalAnnotation, CANONICAL_SCHEMA_ID)


def _resolve_ref(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return root["$defs"][ref.rsplit("/", 1)[-1]]
    return schema


def _skeleton(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Any:
    schema = _resolve_ref(schema, root)
    if "anyOf" in schema:
        branches = [item for item in schema["anyOf"] if item.get("type") != "null"]
        return _skeleton(branches[0], root) if branches else None
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        return {key: _skeleton(value, root) for key, value in schema.get("properties", {}).items()}
    if kind == "array":
        return [_skeleton(schema.get("items", {}), root)]
    if kind == "null":
        return None
    return "<简体中文短语>"


def prompt_skeleton(model: type[BaseModel]) -> dict[str, Any]:
    """Execute the prompt skeleton operation."""
    schema = model.model_json_schema(mode="validation")
    value = _skeleton(schema, schema)
    if not isinstance(value, dict):
        raise RuntimeError("derived prompt skeleton is not an object")
    return value


def _validate(model: type[ModelT], value: Mapping[str, Any], label: str) -> ModelT:
    if any(isinstance(item, dict) for item in value.values()):
        raise AnnotationValidationError(f"{label} forbids nested objects")
    if any(isinstance(item, list) and any(not isinstance(child, str) for child in item) for item in value.values()):
        raise AnnotationValidationError(f"{label} permits string arrays only")
    try:
        return model.model_validate_json(json.dumps(value, ensure_ascii=False), strict=True)
    except ValidationError as exc:
        raise AnnotationValidationError(str(exc)) from exc


def validate_media_router(value: Mapping[str, Any]) -> MediaRouterOutput:
    """Validate media router."""
    return _validate(MediaRouterOutput, value, "media router output")


def validate_semantic_payload(value: Mapping[str, Any]) -> SemanticPayload:
    """Validate semantic payload."""
    return _validate(SemanticPayload, value, "semantic payload")


def validate_final_canonical(value: Mapping[str, Any]) -> FinalCanonicalAnnotation:
    """Validate final canonical."""
    try:
        annotation = FinalCanonicalAnnotation.model_validate_json(
            json.dumps(value, ensure_ascii=False), strict=True
        )
    except ValidationError as exc:
        raise AnnotationValidationError(str(exc)) from exc
    evidence_ids = [item.evidence_id for item in annotation.evidence_items]
    claim_ids = [item.claim_id for item in annotation.claims]
    if len(evidence_ids) != len(set(evidence_ids)) or len(claim_ids) != len(set(claim_ids)):
        raise AnnotationValidationError("canonical IDs must be unique")
    if len(annotation.claims) != len(annotation.evidence_items):
        raise AnnotationValidationError("claims and evidence must map one-to-one")
    for claim, evidence in zip(annotation.claims, annotation.evidence_items):
        if claim.evidence_refs != [evidence.evidence_id] or claim.text != evidence.content:
            raise AnnotationValidationError("claim does not match direct evidence")
    return annotation


def _walk_strings(value: Mapping[str, Any]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for key, item in value.items():
        if isinstance(item, str):
            found.append((key, item))
        elif isinstance(item, list):
            found.extend((f"{key}.{index}", child) for index, child in enumerate(item) if isinstance(child, str))
    return found


def detect_policy_overrides(value: Mapping[str, Any]) -> list[dict[str, str]]:
    """Execute the detect policy overrides operation."""
    return [
        {"path": path, "detected_claim": text, "action": "ignored_verification_claim"}
        for path, text in _walk_strings(value)
        if _VERIFICATION_CLAIM_RE.search(text)
    ]


def canonicalize_phase2b(
    media_value: Mapping[str, Any], semantic_value: Mapping[str, Any], context: CanonicalContext
) -> FinalCanonicalAnnotation:
    """Execute the canonicalize phase2b operation."""
    media = validate_media_router(media_value)
    semantic = validate_semantic_payload(semantic_value)
    annotation = {
        "annotation_meta": {
            "schema_version": CANONICAL_SCHEMA_ID,
            "pipeline_version": PHASE2B_PIPELINE_VERSION,
            "media_prompt_version": MEDIA_PROMPT_VERSION,
            "semantic_prompt_version": SEMANTIC_PROMPT_VERSION,
            "run_id": context.run_id,
            "image_id": context.image_id,
            "source_split": "train",
            "source_sha256": context.source_sha256,
            "generated_at": context.generated_at.isoformat(),
            "model_name": context.model_name,
            "model_revision": context.model_revision,
            "media_max_new_tokens": context.media_max_new_tokens,
            "semantic_max_new_tokens": context.semantic_max_new_tokens,
            "constrained_decoding": context.constrained_decoding,
        },
        "visual_medium": media.visual_medium,
        "medium_confidence": media.confidence,
        "depicted_content": semantic.depicted_content,
        "scene_summary": semantic.scene_summary,
        "subjects": [
            {"subject_id": f"sub_{index:03d}", "text": text}
            for index, text in enumerate(semantic.subjects, 1)
        ],
        "activities": [
            {"activity_id": f"act_{index:03d}", "text": text}
            for index, text in enumerate(semantic.activities, 1)
        ],
        "relationships": [
            {"relationship_id": f"rel_{index:03d}", "text": text}
            for index, text in enumerate(semantic.relationships, 1)
        ],
        "text_governance": {
            "verification_source": "vlm_only",
            "verification_status": "unverified",
            "selected_text": None,
            "unverified_text_candidates": [
                {
                    "candidate_id": f"txt_{index:03d}",
                    "text": text,
                    "verification_source": "vlm_only",
                    "verification_status": "unverified",
                }
                for index, text in enumerate(semantic.visible_text_candidates, 1)
            ],
            "policy_overrides": detect_policy_overrides(semantic_value),
        },
        "evidence_items": [
            {
                "evidence_id": f"ev_{index:03d}",
                "source_type": "visual_observation",
                "verification_status": "vlm_unverified",
                "content": text,
            }
            for index, text in enumerate(semantic.observations, 1)
        ],
        "claims": [
            {
                "claim_id": f"cl_{index:03d}",
                "claim_type": "direct_observation",
                "text": text,
                "evidence_refs": [f"ev_{index:03d}"],
                "verification_status": "vlm_unverified",
            }
            for index, text in enumerate(semantic.observations, 1)
        ],
        "inference_candidates": [
            {
                "inference_id": f"inf_{index:03d}",
                "text": text,
                "verification_status": "unverified",
                "promoted_to_claim": False,
            }
            for index, text in enumerate(semantic.inference_candidates, 1)
        ],
        "uncertainties": semantic.uncertainties,
    }
    return validate_final_canonical(annotation)


def _json_unclosed(text: str) -> bool:
    stripped = text.lstrip("\ufeff \t\r\n")
    if not stripped.startswith("{"):
        return False
    depth = 0
    in_string = False
    escaped = False
    for char in stripped:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return in_string or depth > 0


def parse_stage_json(raw_text: str) -> StageParseResult:
    """Parse stage json."""
    cleaned = raw_text.lstrip("\ufeff").strip()
    repair_type: str | None = None
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[len("```json") : -3].strip()
        repair_type = "surrounding_json_fence_removed"
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return StageParseResult(
            None,
            f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            repair_type,
            _json_unclosed(cleaned),
        )
    if not isinstance(value, dict):
        return StageParseResult(None, "root must be a JSON object", repair_type, False)
    return StageParseResult(value, None, repair_type, False)


def should_retry_stage(*, finish_reason: str | None, parse_result: StageParseResult) -> bool:
    """Execute the should retry stage operation."""
    return finish_reason in {"length", "max_tokens"} or parse_result.json_unclosed
