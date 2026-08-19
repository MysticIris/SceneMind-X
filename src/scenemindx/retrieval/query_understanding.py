"""Strict, auditable Query Understanding helpers for Phase 5.3 Stage 2."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


QUERY_INTENT_FIELDS = (
    "literal_objects",
    "scene",
    "activities",
    "attributes",
    "relations",
    "mood",
    "color_tone",
    "composition",
    "usage_intent",
    "positive_constraints",
    "negative_constraints",
    "literal_query",
    "semantic_query",
    "visual_style_query",
    "uncertainties",
)
LIST_FIELDS = {
    "literal_objects",
    "scene",
    "activities",
    "attributes",
    "relations",
    "mood",
    "color_tone",
    "composition",
    "positive_constraints",
    "negative_constraints",
    "uncertainties",
}
STRING_FIELDS = set(QUERY_INTENT_FIELDS) - LIST_FIELDS
NEGATION_MARKERS = (
    "不要",
    "不含",
    "排除",
    "避免",
    "不能",
    "不可",
    "不是",
    "没有",
    "无",
    "not ",
    "without",
    "exclude",
    "avoid",
)
BACKEND_IDENTITY_PATTERN = re.compile(
    r"(?:\bIMG_[1-9]\d*\b|\b[\w.-]+\.(?:jpg|jpeg|png|webp|bmp)\b|"
    r"\b[a-f0-9]{64}\b)",
    re.IGNORECASE,
)


class QueryUnderstandingError(ValueError):
    """Raised when a model result violates the Stage 2 contract."""


def sha256_file(path: Path) -> str:
    """Execute the sha256 file operation."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_query_prompt(template: str, original_query: str) -> str:
    """Render query prompt."""
    query = str(original_query).strip()
    if not query:
        raise QueryUnderstandingError("original_query_empty")
    if len(query) > 2000:
        raise QueryUnderstandingError("original_query_too_long")
    marker = "{{ORIGINAL_QUERY_JSON}}"
    if template.count(marker) != 1:
        raise QueryUnderstandingError("prompt_template_marker_mismatch")
    return template.replace(marker, json.dumps(query, ensure_ascii=False))


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    text = str(raw_output).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise QueryUnderstandingError("json_object_not_found") from None
        try:
            value, end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise QueryUnderstandingError(f"json_parse_failed:{exc.msg}") from None
        trailing = text[start + end :].strip()
        if trailing:
            raise QueryUnderstandingError("contract_text_outside_json")
    if not isinstance(value, dict):
        raise QueryUnderstandingError("json_root_must_be_object")
    return value


def _clean_list(field: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        raise QueryUnderstandingError(f"{field}_must_be_array")
    if len(value) > 12:
        raise QueryUnderstandingError(f"{field}_too_many_items")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise QueryUnderstandingError(f"{field}_item_must_be_string")
        cleaned = " ".join(item.split())
        if not cleaned:
            raise QueryUnderstandingError(f"{field}_contains_empty_item")
        if len(cleaned) > 120:
            raise QueryUnderstandingError(f"{field}_item_too_long")
        if BACKEND_IDENTITY_PATTERN.search(cleaned):
            raise QueryUnderstandingError(f"{field}_contains_backend_identity")
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def validate_query_intent(
    value: Mapping[str, Any],
    *,
    original_query: str,
) -> dict[str, Any]:
    """Validate query intent."""
    query = str(original_query).strip()
    keys = set(value)
    expected = set(QUERY_INTENT_FIELDS)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise QueryUnderstandingError(
            f"query_intent_fields_mismatch:missing={missing}:unknown={unknown}"
        )
    normalized: dict[str, Any] = {}
    for field in QUERY_INTENT_FIELDS:
        raw = value[field]
        if field in LIST_FIELDS:
            normalized[field] = _clean_list(field, raw)
            continue
        if not isinstance(raw, str):
            raise QueryUnderstandingError(f"{field}_must_be_string")
        cleaned = " ".join(raw.split())
        if len(cleaned) > 1000:
            raise QueryUnderstandingError(f"{field}_too_long")
        if BACKEND_IDENTITY_PATTERN.search(cleaned):
            raise QueryUnderstandingError(f"{field}_contains_backend_identity")
        normalized[field] = cleaned
    if normalized["literal_query"] != query:
        raise QueryUnderstandingError("literal_query_must_preserve_original_exactly")
    if not normalized["semantic_query"]:
        raise QueryUnderstandingError("semantic_query_empty")
    lowered = query.casefold()
    if normalized["negative_constraints"] and not any(
        marker in lowered for marker in NEGATION_MARKERS
    ):
        raise QueryUnderstandingError("negative_constraints_not_grounded_in_query")
    return normalized


def parse_query_understanding(raw_output: str, *, original_query: str) -> dict[str, Any]:
    """Parse query understanding."""
    return validate_query_intent(
        _extract_json_object(raw_output),
        original_query=original_query,
    )


def parse_query_understanding_audited(
    raw_output: str,
    *,
    original_query: str,
) -> tuple[dict[str, Any], list[str]]:
    """Parse with one non-semantic repair: bind literal_query to backend input."""

    value = _extract_json_object(raw_output)
    repairs: list[str] = []
    if isinstance(value.get("literal_query"), str) and value["literal_query"] != original_query.strip():
        value = {**value, "literal_query": original_query.strip()}
        repairs.append("literal_query_backend_bound_to_original")
    return validate_query_intent(value, original_query=original_query), repairs


def build_query_text_variants(
    intent: Mapping[str, Any],
    *,
    original_query: str,
) -> dict[str, Any]:
    """Build query text variants."""
    validated = validate_query_intent(intent, original_query=original_query)
    literal = str(original_query).strip()
    semantic = validated["semantic_query"]
    visual = validated["visual_style_query"]
    expanded_parts = [semantic]
    if visual:
        expanded_parts.append(visual)
    expanded = "；".join(expanded_parts)
    return {
        "literal_query": literal,
        "semantic_query": semantic,
        "visual_style_query": visual,
        "three_way_components": [
            value for value in (literal, semantic, visual) if value
        ],
        "original_plus_expanded_components": [literal, expanded],
    }
