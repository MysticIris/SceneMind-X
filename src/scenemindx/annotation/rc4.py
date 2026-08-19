"""RC4 flat-payload validation and deterministic canonical assembly."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import ValidationError

from .rc4_types import (
    Rc4CanonicalAnnotation,
    Rc4ModelPayload,
    Rc4TextReliability,
    Rc4TextRole,
)
from .schema import AnnotationValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RC4_PROMPT_ID = "p3_shared_fact_v1_4_rc4"
RC4_PAYLOAD_SCHEMA_ID = "phase2a_rc4_flat_model_payload_v1"
RC4_CANONICAL_SCHEMA_ID = "visual_asset_annotation_v1_1_rc4"
RC4_PAYLOAD_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "phase2a_rc4_flat_model_payload.schema.json"
RC4_CANONICAL_SCHEMA_PATH = PROJECT_ROOT / "data" / "schemas" / "visual_asset_annotation_v1_1_rc4.schema.json"
RC4_TOP_LEVEL_FIELDS = tuple(Rc4ModelPayload.model_fields)

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_FUSED_RE = re.compile(r"[\u3400-\u9fff][A-Za-z]{2,}|[A-Za-z]{2,}[\u3400-\u9fff]")
_FORMULA_RE = re.compile(r"\b[A-Za-z]\s*=\s*[A-Za-z0-9^+\-*/().]+")
_MODEL_RE = re.compile(r"\b(?:[A-Z]{1,6}[0-9][A-Za-z0-9-]*|[A-Za-z]+-[A-Za-z0-9-]+)\b")


@dataclass(frozen=True)
class Rc4ParseResult:
    """Represent rc4 parse result data."""
    value: dict[str, Any] | None
    error: str | None
    repair_type: str | None
    json_unclosed: bool


@dataclass(frozen=True)
class Rc4LanguageResult:
    """Represent rc4 language result data."""
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
class Rc4CanonicalContext:
    """Provide rc4 canonical context behavior."""
    run_id: str
    image_id: str
    source_sha256: str
    generated_at: datetime
    model_name: str
    model_revision: str
    max_new_tokens: int
    constrained_decoding: str = "none"


def rc4_payload_schema() -> dict[str, Any]:
    """Execute the rc4 payload schema operation."""
    schema = Rc4ModelPayload.model_json_schema(mode="validation")
    schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": RC4_PAYLOAD_SCHEMA_ID})
    return schema


def rc4_canonical_schema() -> dict[str, Any]:
    """Execute the rc4 canonical schema operation."""
    schema = Rc4CanonicalAnnotation.model_json_schema(mode="validation")
    schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": RC4_CANONICAL_SCHEMA_ID})
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
    if kind == "null":
        return None
    return "<简体中文短语>"


def rc4_prompt_skeleton() -> dict[str, Any]:
    """Execute the rc4 prompt skeleton operation."""
    schema = rc4_payload_schema()
    value = _schema_skeleton(schema, schema)
    if not isinstance(value, dict) or tuple(value) != RC4_TOP_LEVEL_FIELDS:
        raise RuntimeError("derived RC4 skeleton does not match the payload type")
    value["selected_text"] = None
    return value


def _all_nested_values_are_strings(value: Mapping[str, Any]) -> bool:
    for item in value.values():
        if isinstance(item, dict):
            return False
        if isinstance(item, list) and any(not isinstance(child, str) for child in item):
            return False
    return True


def _candidate_propagation_paths(payload: Rc4ModelPayload) -> list[str]:
    if payload.text_role not in {
        Rc4TextRole.DECORATIVE,
        Rc4TextRole.INCIDENTAL,
        Rc4TextRole.UNCERTAIN,
    } and payload.text_reliability not in {
        Rc4TextReliability.UNCERTAIN,
        Rc4TextReliability.LOW,
        Rc4TextReliability.UNREADABLE,
    }:
        return []
    fields: list[tuple[str, str]] = [("scene_summary", payload.scene_summary)]
    if payload.depicted_content:
        fields.append(("depicted_content", payload.depicted_content))
    for key in ("subjects", "activities", "relationships", "observations", "inference_candidates"):
        fields.extend((f"{key}.{index}", text) for index, text in enumerate(getattr(payload, key)))
    return [
        path
        for candidate in payload.visible_text_candidates
        for path, text in fields
        if candidate in text
    ]


def validate_rc4_payload(value: Mapping[str, Any]) -> Rc4ModelPayload:
    """Validate rc4 payload."""
    if not _all_nested_values_are_strings(value):
        raise AnnotationValidationError("RC4 payload forbids nested objects and object arrays")
    try:
        payload = Rc4ModelPayload.model_validate_json(json.dumps(value, ensure_ascii=False), strict=True)
    except ValidationError as exc:
        raise AnnotationValidationError(str(exc)) from exc
    unsafe_paths = _candidate_propagation_paths(payload)
    if unsafe_paths:
        raise AnnotationValidationError(
            "unverified decorative/incidental text propagated into semantic fields: "
            + ", ".join(sorted(set(unsafe_paths)))
        )
    return payload


def validate_rc4_canonical(value: Mapping[str, Any]) -> Rc4CanonicalAnnotation:
    """Validate rc4 canonical."""
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            default=lambda item: item.isoformat() if isinstance(item, datetime) else item.value,
        )
        annotation = Rc4CanonicalAnnotation.model_validate_json(serialized, strict=True)
    except ValidationError as exc:
        raise AnnotationValidationError(str(exc)) from exc
    evidence_ids = [item.evidence_id for item in annotation.evidence_items]
    claim_ids = [item.claim_id for item in annotation.claims]
    candidate_ids = [item.candidate_id for item in annotation.text_governance.unverified_candidates]
    inference_ids = [item.inference_id for item in annotation.inference_candidates]
    for label, ids in (
        ("evidence_id", evidence_ids),
        ("claim_id", claim_ids),
        ("candidate_id", candidate_ids),
        ("inference_id", inference_ids),
    ):
        if len(ids) != len(set(ids)):
            raise AnnotationValidationError(f"duplicate {label}")
    evidence_set = set(evidence_ids)
    dangling = sorted({ref for claim in annotation.claims for ref in claim.evidence_refs} - evidence_set)
    if dangling:
        raise AnnotationValidationError(f"dangling evidence refs: {dangling}")
    if len(annotation.claims) != len(annotation.evidence_items):
        raise AnnotationValidationError("direct observation claims must map one-to-one to evidence")
    for index, claim in enumerate(annotation.claims):
        expected = annotation.evidence_items[index]
        if claim.evidence_refs != [expected.evidence_id] or claim.text != expected.content:
            raise AnnotationValidationError("direct observation claim does not match its evidence")
    return annotation


def canonicalize_rc4(value: Mapping[str, Any], context: Rc4CanonicalContext) -> Rc4CanonicalAnnotation:
    """Execute the canonicalize rc4 operation."""
    payload = validate_rc4_payload(value)
    annotation = {
        "annotation_meta": {
            "schema_version": RC4_CANONICAL_SCHEMA_ID,
            "prompt_version": RC4_PROMPT_ID,
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
        "text_governance": {
            "presence": payload.text_presence,
            "role": payload.text_role,
            "reliability": payload.text_reliability,
            "selected_text": None,
            "selected_text_status": "withheld_unverified" if payload.visible_text_candidates else "none",
            "unverified_candidates": [
                {
                    "candidate_id": f"txt_{index:03d}",
                    "text": text,
                    "source_type": "vlm_unverified",
                    "verification_status": "unverified",
                    "role": payload.text_role,
                    "reliability": payload.text_reliability,
                }
                for index, text in enumerate(payload.visible_text_candidates, start=1)
            ],
            "summary": payload.text_summary,
            "uncertainty": payload.text_uncertainty,
        },
        "evidence_items": [
            {
                "evidence_id": f"ev_{index:03d}",
                "source_type": "visual_observation",
                "verification_status": "vlm_unverified",
                "content": text,
            }
            for index, text in enumerate(payload.observations, start=1)
        ],
        "claims": [
            {
                "claim_id": f"cl_{index:03d}",
                "claim_type": "direct_observation",
                "text": text,
                "evidence_refs": [f"ev_{index:03d}"],
                "verification_status": "vlm_unverified",
            }
            for index, text in enumerate(payload.observations, start=1)
        ],
        "inference_candidates": [
            {
                "inference_id": f"inf_{index:03d}",
                "text": text,
                "verification_status": "unverified",
                "promoted_to_claim": False,
            }
            for index, text in enumerate(payload.inference_candidates, start=1)
        ],
        "uncertainties": payload.uncertainties,
    }
    return validate_rc4_canonical(annotation)


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


def parse_rc4_json(raw_text: str) -> Rc4ParseResult:
    """Parse rc4 json."""
    cleaned = raw_text.lstrip("\ufeff").strip()
    repair_type: str | None = None
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[len("```json") : -3].strip()
        repair_type = "surrounding_json_fence_removed"
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return Rc4ParseResult(
            None,
            f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            repair_type,
            _json_unclosed(cleaned),
        )
    if not isinstance(value, dict):
        return Rc4ParseResult(None, "root must be a JSON object", repair_type, False)
    return Rc4ParseResult(value, None, repair_type, False)


def should_retry_rc4(*, finish_reason: str | None, parse_result: Rc4ParseResult) -> bool:
    """Execute the should retry rc4 operation."""
    return finish_reason in {"length", "max_tokens"} or parse_result.json_unclosed


def _language_items(value: Mapping[str, Any]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for key in ("depicted_content", "scene_summary", "text_summary", "text_uncertainty"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            items.append((key, raw.strip()))
    for key in (
        "subjects",
        "activities",
        "relationships",
        "observations",
        "inference_candidates",
        "uncertainties",
    ):
        raw = value.get(key)
        if isinstance(raw, list):
            items.extend(
                (f"{key}.{index}", text.strip())
                for index, text in enumerate(raw)
                if isinstance(text, str) and text.strip()
            )
    return items


def check_rc4_language(
    value: Mapping[str, Any], *, candidate_latin_share_max: float = 0.20,
    allowed_latin_terms: Iterable[str] = (),
) -> Rc4LanguageResult:
    """Check rc4 language."""
    violations: list[str] = []
    exceptions: list[str] = []
    cleaned_items: list[str] = []
    terms = sorted({term for term in allowed_latin_terms if term}, key=len, reverse=True)
    candidates = [
        text for text in value.get("visible_text_candidates", [])
        if isinstance(text, str) and text.strip()
    ]
    for index, candidate in enumerate(candidates):
        if _LATIN_RE.search(candidate):
            exceptions.append(f"verbatim_visible_text:visible_text_candidates.{index}")
    for path, text in _language_items(value):
        cleaned = text
        for candidate in sorted(candidates, key=len, reverse=True):
            if candidate in cleaned:
                exceptions.append(f"quoted_visible_text:{path}")
                cleaned = cleaned.replace(candidate, " ")
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
    combined = " ".join(cleaned_items)
    han_count = len(_HAN_RE.findall(combined))
    latin_count = len(_LATIN_RE.findall(combined))
    share = latin_count / (han_count + latin_count) if han_count + latin_count else 0.0
    if han_count == 0:
        violations.append("no_simplified_chinese_narrative")
    if share > candidate_latin_share_max:
        violations.append(f"latin_share_exceeds_candidate:{share:.6f}")
    unique = tuple(dict.fromkeys(violations))
    return Rc4LanguageResult(
        latin_share=share,
        han_characters=han_count,
        latin_letters=latin_count,
        candidate_threshold=candidate_latin_share_max,
        reasonable_exceptions=tuple(dict.fromkeys(exceptions)),
        violations=unique,
        qualified=not unique,
    )
