"""RC3 shallow-payload validation and deterministic canonical assembly."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import ValidationError

from .rc3_types import Rc3CanonicalAnnotation, Rc3ModelPayload
from .schema import AnnotationValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RC3_PROMPT_ID = "p3_shared_fact_v1_4_rc3"
RC3_PAYLOAD_SCHEMA_ID = "phase2a_rc3_model_payload_v1"
RC3_CANONICAL_SCHEMA_ID = "visual_asset_annotation_v1_1_rc3"
RC3_PAYLOAD_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "phase2a_rc3_model_payload.schema.json"
RC3_CANONICAL_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "visual_asset_annotation_v1_1_rc3.schema.json"
RC3_TOP_LEVEL_FIELDS = tuple(Rc3ModelPayload.model_fields)

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_FUSED_RE = re.compile(r"[\u3400-\u9fff][A-Za-z]{2,}|[A-Za-z]{2,}[\u3400-\u9fff]")
_FORMULA_RE = re.compile(r"\b[A-Za-z]\s*=\s*[A-Za-z0-9^+\-*/().]+")
_MODEL_RE = re.compile(r"\b(?:[A-Z]{1,6}[0-9][A-Za-z0-9-]*|[A-Za-z]+-[A-Za-z0-9-]+)\b")


@dataclass(frozen=True)
class Rc3ParseResult:
    """Represent rc3 parse result data."""
    value: dict[str, Any] | None
    error: str | None
    repair_type: str | None
    json_unclosed: bool


@dataclass(frozen=True)
class Rc3LanguageResult:
    """Represent rc3 language result data."""
    latin_share: float
    han_characters: int
    latin_letters: int
    candidate_threshold: float
    reasonable_exceptions: tuple[str, ...]
    violations: tuple[str, ...]
    qualified: bool

    def as_dict(self) -> dict[str, Any]:
        """Execute the as dict operation."""
        return asdict(self)


@dataclass(frozen=True)
class Rc3CanonicalContext:
    """Provide rc3 canonical context behavior."""
    run_id: str
    image_id: str
    source_sha256: str
    generated_at: datetime
    model_name: str
    model_revision: str
    max_new_tokens: int
    constrained_decoding: str = "none"


def rc3_payload_schema() -> dict[str, Any]:
    """Execute the rc3 payload schema operation."""
    schema = Rc3ModelPayload.model_json_schema(mode="validation")
    schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": RC3_PAYLOAD_SCHEMA_ID})
    return schema


def rc3_canonical_schema() -> dict[str, Any]:
    """Execute the rc3 canonical schema operation."""
    schema = Rc3CanonicalAnnotation.model_json_schema(mode="validation")
    schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": RC3_CANONICAL_SCHEMA_ID})
    return schema


def _resolve_schema_ref(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return schema
    return root["$defs"][ref.rsplit("/", 1)[-1]]


def _schema_skeleton(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Any:
    schema = _resolve_schema_ref(schema, root)
    if "anyOf" in schema:
        branches = [branch for branch in schema["anyOf"] if branch.get("type") != "null"]
        return _schema_skeleton(branches[0], root) if branches else None
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        return {key: _schema_skeleton(value, root) for key, value in schema.get("properties", {}).items()}
    if kind == "array":
        return [_schema_skeleton(schema.get("items", {}), root)]
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.5
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    return "<简体中文短语>"


def rc3_prompt_skeleton() -> dict[str, Any]:
    """Execute the rc3 prompt skeleton operation."""
    schema = rc3_payload_schema()
    value = _schema_skeleton(schema, schema)
    if not isinstance(value, dict) or tuple(value) != RC3_TOP_LEVEL_FIELDS:
        raise RuntimeError("derived RC3 skeleton does not match the payload type")
    return value


def validate_rc3_payload(value: Mapping[str, Any]) -> Rc3ModelPayload:
    """Validate rc3 payload."""
    try:
        payload = Rc3ModelPayload.model_validate_json(
            json.dumps(value, ensure_ascii=False), strict=True
        )
    except ValidationError as exc:
        raise AnnotationValidationError(str(exc)) from exc
    observation_count = len(payload.observations)
    for inference_index, inference in enumerate(payload.inferences):
        invalid = [index for index in inference.support if index >= observation_count]
        if invalid:
            raise AnnotationValidationError(
                f"inferences.{inference_index}.support contains out-of-range indices {invalid}"
            )
    return payload


def validate_rc3_canonical(value: Mapping[str, Any]) -> Rc3CanonicalAnnotation:
    """Validate rc3 canonical."""
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            default=lambda item: item.isoformat() if isinstance(item, datetime) else item.value,
        )
        annotation = Rc3CanonicalAnnotation.model_validate_json(serialized, strict=True)
    except ValidationError as exc:
        raise AnnotationValidationError(str(exc)) from exc
    evidence_ids = {item.evidence_id for item in annotation.evidence_items}
    if len(evidence_ids) != len(annotation.evidence_items):
        raise AnnotationValidationError("duplicate evidence_id")
    claim_ids = {item.claim_id for item in annotation.claims}
    if len(claim_ids) != len(annotation.claims):
        raise AnnotationValidationError("duplicate claim_id")
    dangling = sorted({ref for claim in annotation.claims for ref in claim.evidence_refs} - evidence_ids)
    if dangling:
        raise AnnotationValidationError(f"dangling evidence refs: {dangling}")
    return annotation


def canonicalize_rc3(value: Mapping[str, Any], context: Rc3CanonicalContext) -> Rc3CanonicalAnnotation:
    """Execute the canonicalize rc3 operation."""
    payload = validate_rc3_payload(value)
    annotation = {
        "annotation_meta": {
            "schema_version": RC3_CANONICAL_SCHEMA_ID,
            "prompt_version": RC3_PROMPT_ID,
            "run_id": context.run_id,
            "image_id": context.image_id,
            "source_split": "train",
            "source_sha256": context.source_sha256,
            "generated_at": context.generated_at,
            "model_name": context.model_name,
            "model_revision": context.model_revision,
            "max_new_tokens": context.max_new_tokens,
            "constrained_decoding": context.constrained_decoding,
        },
        "visual_medium": payload.visual_medium,
        "depicted_content": payload.depicted_content,
        "scene_summary": payload.scene_summary,
        "subjects": [
            {"subject_id": f"sub_{index:03d}", "text": text}
            for index, text in enumerate(payload.subjects, start=1)
        ],
        "activities": [
            {"activity_id": f"act_{index:03d}", "text": text}
            for index, text in enumerate(payload.activities, start=1)
        ],
        "relationships": [
            {"relationship_id": f"rel_{index:03d}", "text": text}
            for index, text in enumerate(payload.relationships, start=1)
        ],
        "text_assessment": payload.text_assessment.model_dump(mode="json"),
        "evidence_items": [
            {"evidence_id": f"ev_{index:03d}", "source_type": "visual_observation", "content": text}
            for index, text in enumerate(payload.observations, start=1)
        ],
        "claims": [
            {
                "claim_id": f"cl_{index:03d}",
                "claim_type": "reasonable_inference",
                "text": inference.text,
                "evidence_refs": [f"ev_{support + 1:03d}" for support in inference.support],
                "confidence": inference.confidence,
            }
            for index, inference in enumerate(payload.inferences, start=1)
        ],
    }
    return validate_rc3_canonical(annotation)


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


def parse_rc3_json(raw_text: str) -> Rc3ParseResult:
    """Parse rc3 json."""
    cleaned = raw_text.lstrip("\ufeff").strip()
    repair_type: str | None = None
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[len("```json") : -3].strip()
        repair_type = "surrounding_json_fence_removed"
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return Rc3ParseResult(None, f"{exc.msg} at line {exc.lineno} column {exc.colno}", repair_type, _json_unclosed(cleaned))
    if not isinstance(value, dict):
        return Rc3ParseResult(None, "root must be a JSON object", repair_type, False)
    return Rc3ParseResult(value, None, repair_type, False)


def should_retry_rc3(*, finish_reason: str | None, parse_result: Rc3ParseResult) -> bool:
    """Execute the should retry rc3 operation."""
    return finish_reason in {"length", "max_tokens"} or parse_result.json_unclosed


def _language_items(value: Mapping[str, Any]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for key in ("depicted_content", "scene_summary"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            items.append((key, raw.strip()))
    for key in ("subjects", "activities", "relationships", "observations"):
        raw = value.get(key)
        if isinstance(raw, list):
            items.extend((f"{key}.{index}", text.strip()) for index, text in enumerate(raw) if isinstance(text, str) and text.strip())
    assessment = value.get("text_assessment")
    if isinstance(assessment, Mapping):
        for key in ("summary", "uncertainty"):
            raw = assessment.get(key)
            if isinstance(raw, str) and raw.strip():
                items.append((f"text_assessment.{key}", raw.strip()))
    raw_inferences = value.get("inferences")
    if isinstance(raw_inferences, list):
        for index, inference in enumerate(raw_inferences):
            if isinstance(inference, Mapping) and isinstance(inference.get("text"), str):
                items.append((f"inferences.{index}.text", inference["text"].strip()))
    return items


def check_rc3_language(
    value: Mapping[str, Any], *, candidate_latin_share_max: float = 0.20,
    allowed_latin_terms: Iterable[str] = (),
) -> Rc3LanguageResult:
    """Check rc3 language."""
    violations: list[str] = []
    exceptions: list[str] = []
    cleaned_items: list[str] = []
    terms = sorted({term for term in allowed_latin_terms if term}, key=len, reverse=True)
    for path, text in _language_items(value):
        cleaned = text
        for term in terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(cleaned):
                exceptions.append(f"allowlisted_term:{path}:{term}")
                cleaned = pattern.sub(" ", cleaned)
        for label, pattern in (("formula", _FORMULA_RE), ("model_or_product", _MODEL_RE)):
            for matched in pattern.findall(cleaned):
                exceptions.append(f"{label}:{path}:{matched}")
            cleaned = pattern.sub(" ", cleaned)
        cleaned_items.append(cleaned)
        han = len(_HAN_RE.findall(cleaned))
        latin = len(_LATIN_RE.findall(cleaned))
        if latin >= 4 and han == 0:
            violations.append(f"ascii_only_narrative:{path}")
        if _FUSED_RE.search(cleaned):
            violations.append(f"abnormal_han_latin_fusion:{path}")
    assessment = value.get("text_assessment")
    if isinstance(assessment, Mapping) and isinstance(assessment.get("selected_text"), list):
        for index, text in enumerate(assessment["selected_text"]):
            if isinstance(text, str) and _LATIN_RE.search(text):
                exceptions.append(f"verbatim_visible_text:text_assessment.selected_text.{index}")
    combined = " ".join(cleaned_items)
    han_count = len(_HAN_RE.findall(combined))
    latin_count = len(_LATIN_RE.findall(combined))
    share = latin_count / (han_count + latin_count) if han_count + latin_count else 0.0
    if han_count == 0:
        violations.append("no_simplified_chinese_narrative")
    if share > candidate_latin_share_max:
        violations.append(f"latin_share_exceeds_candidate:{share:.6f}")
    unique = tuple(dict.fromkeys(violations))
    return Rc3LanguageResult(
        latin_share=share, han_characters=han_count, latin_letters=latin_count,
        candidate_threshold=candidate_latin_share_max,
        reasonable_exceptions=tuple(dict.fromkeys(exceptions)),
        violations=unique, qualified=not unique,
    )
