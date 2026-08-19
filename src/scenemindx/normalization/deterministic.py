"""Lossless repairs for frozen Phase 4B stage outputs."""
from __future__ import annotations
import json
import hashlib
from typing import Any, Mapping
from jsonschema import Draft202012Validator

NORMALIZER_VERSION = "deterministic_normalizer_v1"
UNREPAIRABLE = "semantic_field_missing_unrepairable"
SCHEMA_VERSION = "gate1_d3_semantic_review_payload_p3_v1_1"
PARSER_VERSION = "phase4b_service_parser_v1"
RULE_REGISTRY = {
    "single_json_string_to_declared_field": {"semantic_change_allowed": False, "description": "Map one literal JSON string to its declared single field."},
    "plain_scalar_to_declared_field": {"semantic_change_allowed": False, "description": "Retain a literal non-JSON scalar for its declared single field."},
    "empty_object_to_contractual_empty_string": {"semantic_change_allowed": False, "description": "Map contractual empty object to empty string."},
    "ordered_adjacent_json_strings": {"semantic_change_allowed": False, "description": "Map two adjacent JSON string literals by declared field order."},
    "single_string_in_unquoted_wrapper": {"semantic_change_allowed": False, "description": "Unwrap one literal JSON string from a one-value malformed object wrapper."},
    "single_string_set_wrapper": {"semantic_change_allowed": False, "description": "Unwrap one literal JSON string from a one-value set-like wrapper."},
}

def _json_string_sequence(text: str) -> list[str] | None:
    decoder = json.JSONDecoder(); values: list[str] = []; position = 0
    while position < len(text):
        while position < len(text) and (text[position].isspace() or text[position] == ','): position += 1
        if position >= len(text): break
        try: value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError: return None
        if not isinstance(value, str): return None
        values.append(value); position = end
    return values or None

def normalize_stage_outputs(stage_outputs: list[Mapping[str, Any]], schema: Mapping[str, Any]) -> dict[str, Any]:
    """Recover only literal values already present in raw stage output."""
    normalized: dict[str, str] = {}; applied: list[dict[str, str]] = []; failures: list[dict[str, Any]] = []
    registry = {
        "global_subjects": ["global_observation", "subjects"], "activities": ["activities"],
        "attributes": ["attributes"], "relations": ["relations"], "visible_text": ["visible_text_candidates"],
        "scene_hypotheses": ["scene_hypotheses"], "uncertainties": ["uncertainties"], "evidence": ["evidence_descriptions"],
    }
    for stage in stage_outputs:
        stage_id = str(stage.get("stage_id", "")); fields = [str(x) for x in stage.get("fields", [])] or registry.get(stage_id, [])
        parsed = stage.get("parsed_output"); raw = str(stage.get("raw_output") or "").strip()
        if isinstance(parsed, dict) and set(parsed) == set(fields) and all(isinstance(parsed.get(f), str) for f in fields):
            normalized.update({f: parsed[f] for f in fields}); continue
        if len(fields) == 1:
            value: str | None = None; rule = ""
            if isinstance(parsed, dict) and parsed == {} and fields[0] in {"uncertainties", "activities"}:
                value, rule = "", "empty_object_to_contractual_empty_string"
            else:
                values = _json_string_sequence(raw)
                if values and len(values) == 1: value, rule = values[0], "single_json_string_to_declared_field"
                elif raw.startswith("{") and raw.endswith("}"):
                    # Some one-field generations used an unquoted key or a
                    # set-like brace wrapper. Accept only exactly one quoted
                    # string and no second value; no semantic splitting occurs.
                    inner = raw[1:-1].strip()
                    quoted = _json_string_sequence(inner.split(":", 1)[-1].strip()) if ":" in inner else _json_string_sequence(inner)
                    if quoted and len(quoted) == 1:
                        value = quoted[0]
                        rule = "single_string_in_unquoted_wrapper" if ":" in inner else "single_string_set_wrapper"
                elif raw and not raw.startswith(("{", "[")): value, rule = raw.replace("```json", "").replace("```", "").strip(), "plain_scalar_to_declared_field"
            if value is not None:
                normalized[fields[0]] = value; applied.append({"stage_id": stage_id, "rule_id": rule}); continue
        if stage_id == "global_subjects" and fields == ["global_observation", "subjects"]:
            values = _json_string_sequence(raw)
            if values and len(values) == 2:
                normalized.update(dict(zip(fields, values))); applied.append({"stage_id": stage_id, "rule_id": "ordered_adjacent_json_strings"}); continue
        failures.append({"stage_id": stage_id, "reason": UNREPAIRABLE})
    errors = [e.message for e in Draft202012Validator(schema).iter_errors(normalized)]
    canonical_sha = hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    status = "repaired" if applied else ("unrepairable" if failures else "unchanged")
    return {"normalizer_version": NORMALIZER_VERSION, "parser_version": PARSER_VERSION, "schema_version": SCHEMA_VERSION,
            "normalized_output": normalized, "applied_rules": applied,
            "applied_rule_ids": [x["rule_id"] for x in applied], "rule_registry": RULE_REGISTRY,
            "failures": failures, "normalization_status": status, "canonical_output_sha256": canonical_sha,
            "schema_valid": not errors, "schema_errors": errors, "semantic_change": False,
            "semantic_change_allowed": False}
