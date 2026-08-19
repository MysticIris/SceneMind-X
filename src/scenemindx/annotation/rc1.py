"""Strict validation and parsing policy for p3_shared_fact_v1_4_rc1."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .rc1_types import Rc1SharedFacts
from .schema import AnnotationValidationError


RC1_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "data" / "schemas" / "visual_asset_annotation_v1_1_rc1.schema.json"
RC1_PROMPT_ID = "p3_shared_fact_v1_4_rc1"
LEGACY_SHORT_PAYLOAD_ID = "legacy_p3_v1_4_short_payload"
RC1_TOP_LEVEL_FIELDS = (
    "visual_medium", "depicted_content", "scene_summary", "subjects",
    "activities", "relationships", "text_assessment", "evidence_items",
    "claims", "short_caption", "dense_caption", "self_check",
)


@dataclass(frozen=True)
class Rc1GenerationPolicy:
    """Provide rc1 generation policy behavior."""
    max_new_tokens: int = 1024
    retry_max_new_tokens: int = 1280
    max_retries: int = 1
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class Rc1ParseResult:
    """Represent rc1 parse result data."""
    value: Rc1SharedFacts | None
    error: str | None
    repair_type: str | None
    json_unclosed: bool


def load_rc1_schema(path: str | Path = RC1_SCHEMA_PATH) -> dict[str, Any]:
    """Load rc1 schema."""
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise AnnotationValidationError("RC1 schema root must be an object")
    return schema


def _unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise AnnotationValidationError(f"{label}: IDs must be unique")


def _check_refs(refs: list[str], valid: set[str], label: str) -> None:
    dangling = sorted(set(refs) - valid)
    if dangling:
        raise AnnotationValidationError(f"{label}: dangling evidence refs {dangling}")


def validate_rc1_shared_facts(
    value: Mapping[str, Any],
    path: str | Path = RC1_SCHEMA_PATH,
    *,
    raw_model_output: bool = True,
) -> None:
    """Validate JSON Schema plus canonical IDs and references."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise AnnotationValidationError("RC1 validation requires jsonschema") from exc
    errors = sorted(
        Draft202012Validator(load_rc1_schema(path)).iter_errors(value),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise AnnotationValidationError(details)

    evidence_ids = [item["evidence_id"] for item in value["evidence_items"]]
    claim_ids = [item["claim_id"] for item in value["claims"]]
    subject_ids = [item["subject_id"] for item in value["subjects"]]
    _unique(evidence_ids, "evidence_items")
    _unique(claim_ids, "claims")
    _unique(subject_ids, "subjects")
    evidence_set = set(evidence_ids)
    subject_set = set(subject_ids)

    _check_refs(value["visual_medium"]["evidence_refs"], evidence_set, "visual_medium")
    _check_refs(value["depicted_content"]["evidence_refs"], evidence_set, "depicted_content")
    for group in ("subjects", "activities", "relationships", "claims"):
        for index, item in enumerate(value[group]):
            _check_refs(item["evidence_refs"], evidence_set, f"{group}.{index}")
    for index, item in enumerate(value["activities"]):
        if item["subject_ref"] not in subject_set:
            raise AnnotationValidationError(f"activities.{index}.subject_ref: dangling subject ref")
    for index, item in enumerate(value["relationships"]):
        if item["subject_ref"] not in subject_set:
            raise AnnotationValidationError(f"relationships.{index}.subject_ref: dangling subject ref")
        if item["object_ref"] is not None and item["object_ref"] not in subject_set:
            raise AnnotationValidationError(f"relationships.{index}.object_ref: dangling subject ref")

    if raw_model_output:
        disallowed_evidence = [
            item["evidence_id"] for item in value["evidence_items"]
            if item["source_type"] not in {"visual_global", "visual_region", "model_text_assessment"}
        ]
        if disallowed_evidence:
            raise AnnotationValidationError(
                f"evidence_items: raw model output cannot claim tool/human evidence {disallowed_evidence}"
            )
        for index, item in enumerate(value["text_assessment"]["ocr_candidates"]):
            if item["source_type"] != "model_visual":
                raise AnnotationValidationError(
                    f"text_assessment.ocr_candidates.{index}: raw model output source must be model_visual"
                )


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


def parse_rc1_json(raw_text: str) -> Rc1ParseResult:
    """Parse standard JSON with only the two proposal-approved wrappers."""

    cleaned = raw_text.lstrip("\ufeff").strip()
    repair_type: str | None = None
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[len("```json") : -3].strip()
        repair_type = "surrounding_json_fence_removed"
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return Rc1ParseResult(
            value=None,
            error=f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            repair_type=repair_type,
            json_unclosed=_json_unclosed(cleaned),
        )
    if not isinstance(value, dict):
        return Rc1ParseResult(None, "root must be a JSON object", repair_type, False)
    return Rc1ParseResult(value=value, error=None, repair_type=repair_type, json_unclosed=False)  # type: ignore[arg-type]


def should_retry_rc1(*, finish_reason: str | None, parse_result: Rc1ParseResult) -> bool:
    """Execute the should retry rc1 operation."""
    return finish_reason in {"length", "max_tokens"} or parse_result.json_unclosed


def build_legacy_p3_v1_3_compat_view(
    payload: Mapping[str, Any], *, image_id: str, source_sha256: str
) -> dict[str, Any]:
    """Expose a frozen legacy payload without pretending it is RC1 data."""

    return {
        "compatibility_view_version": "legacy_p3_v1_3_to_rc1_view_v1",
        "image_id": image_id,
        "source_sha256": source_sha256,
        "source_prompt_id": "gate1_d3_multistage_p3_v1",
        "target_prompt_id": RC1_PROMPT_ID,
        "migration_status": "requires_regeneration_no_fact_invention",
        "rc1_payload": None,
        "missing_rc1_fields": list(RC1_TOP_LEVEL_FIELDS),
        "source_payload_read_only_copy": deepcopy(dict(payload)),
    }
