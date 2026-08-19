"""Strict RC2 parsing, structure, reference, and Chinese-language gates."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .rc2_types import Rc2SharedFacts
from .schema import AnnotationValidationError


RC2_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "data" / "schemas" / "visual_asset_annotation_v1_1_rc2.schema.json"
RC2_PROMPT_ID = "p3_shared_fact_v1_4_rc2"
RC2_TOP_LEVEL_FIELDS = (
    "visual_medium", "depicted_content", "scene_summary", "subjects",
    "activities", "relationships", "text_assessment", "evidence_items", "claims",
)
_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_MODEL_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z._-]*\d[A-Za-z0-9._-]*|\d+[A-Za-z][A-Za-z0-9._-]*)(?![A-Za-z0-9])")
_FORMULA_RE = re.compile(r"(?<!\w)[A-Za-z0-9]+\s*[=+*/^<>]\s*[A-Za-z0-9+*/^().-]+")
_FUSED_RE = re.compile(r"[\u3400-\u9fff][A-Za-z]{2,}|[A-Za-z]{2,}[\u3400-\u9fff]")
_TRADITIONAL_ONLY = set("體臺裏後於這為與顯關係實現應當無資訊場圖證據載內識別輸觀確")


@dataclass(frozen=True)
class Rc2GenerationPolicy:
    """Provide rc2 generation policy behavior."""
    max_new_tokens: int = 640
    retry_max_new_tokens: int = 768
    max_retries: int = 1
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class Rc2ParseResult:
    """Represent rc2 parse result data."""
    value: Rc2SharedFacts | None
    error: str | None
    repair_type: str | None
    json_unclosed: bool


@dataclass(frozen=True)
class Rc2LanguageResult:
    """Represent rc2 language result data."""
    latin_share: float
    han_characters: int
    latin_letters: int
    candidate_threshold: float
    reasonable_exceptions: tuple[str, ...]
    violations: tuple[str, ...]
    rules: Mapping[str, bool]
    qualified: bool

    def as_dict(self) -> dict[str, Any]:
        """Execute the as dict operation."""
        return asdict(self)


def load_rc2_schema(path: str | Path = RC2_SCHEMA_PATH) -> dict[str, Any]:
    """Load rc2 schema."""
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise AnnotationValidationError("RC2 schema root must be an object")
    return schema


def _unique(values: Iterable[str], label: str) -> None:
    items = list(values)
    if len(items) != len(set(items)):
        raise AnnotationValidationError(f"{label}: IDs must be unique")


def _check_refs(refs: Iterable[str], valid: set[str], label: str) -> None:
    dangling = sorted(set(refs) - valid)
    if dangling:
        raise AnnotationValidationError(f"{label}: dangling evidence refs {dangling}")


def _normalized_text(value: str | None) -> str:
    return " ".join((value or "").split()).casefold()


def validate_rc2_structure(
    value: Mapping[str, Any],
    path: str | Path = RC2_SCHEMA_PATH,
    *,
    raw_model_output: bool = True,
) -> None:
    """Validate exact RC2 shape, canonical IDs, references, and fact ownership."""

    validate_rc2_schema(value, path)
    if raw_model_output and tuple(value) != RC2_TOP_LEVEL_FIELDS:
        raise AnnotationValidationError("<root>: top-level keys must use the exact RC2 order")

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

    if value["visual_medium"]["primary_type"] != "uncertain" and value["visual_medium"]["confidence"] >= 0.8:
        if not value["visual_medium"]["evidence_refs"]:
            raise AnnotationValidationError("visual_medium: confident value requires evidence")
    if value["depicted_content"]["summary"] is not None and not value["depicted_content"]["evidence_refs"]:
        raise AnnotationValidationError("depicted_content: non-null summary requires evidence")
    for group in ("subjects", "activities", "relationships"):
        for index, item in enumerate(value[group]):
            if item["confidence"] >= 0.8 and not item["evidence_refs"]:
                raise AnnotationValidationError(f"{group}.{index}: confident value requires evidence")

    meaningful = bool(
        value["scene_summary"]
        or value["subjects"]
        or value["activities"]
        or value["relationships"]
        or value["text_assessment"]["selected_text"]
    )
    if meaningful and (not value["evidence_items"] or not value["claims"]):
        raise AnnotationValidationError("non-empty shared facts require evidence_items and claims")

    text = value["text_assessment"]
    if text["text_presence"] == "none":
        if text["selected_text"] or text["legibility"] != "none" or text["scene_inference_allowed"]:
            raise AnnotationValidationError("text_assessment: text_presence none is inconsistent")
    if text["suspected_false_positive"] and not text["false_positive_contexts"]:
        raise AnnotationValidationError("text_assessment: suspected false positive requires context")

    scene = _normalized_text(value["scene_summary"])
    evidence_text = {_normalized_text(item["content"]) for item in value["evidence_items"]}
    claim_text = {_normalized_text(item["text"]) for item in value["claims"]}
    if scene and scene in evidence_text:
        raise AnnotationValidationError("evidence_items: content must not duplicate scene_summary")
    if scene and scene in claim_text:
        raise AnnotationValidationError("claims: text must not duplicate scene_summary")
    duplicate_claim_evidence = sorted((claim_text & evidence_text) - {""})
    if duplicate_claim_evidence:
        raise AnnotationValidationError("claims: conclusions must not duplicate evidence text")

    if raw_model_output:
        for index, item in enumerate(value["evidence_items"]):
            if item["tool_ref"] is not None:
                raise AnnotationValidationError(f"evidence_items.{index}: raw model output cannot bind a tool")


def validate_rc2_schema(value: Mapping[str, Any], path: str | Path = RC2_SCHEMA_PATH) -> None:
    """Validate only the formal machine JSON Schema."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise AnnotationValidationError("RC2 validation requires jsonschema") from exc
    errors = sorted(
        Draft202012Validator(load_rc2_schema(path)).iter_errors(value),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise AnnotationValidationError(details)


def _narrative_items(value: Mapping[str, Any]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []

    def add(path: str, text: Any) -> None:
        if isinstance(text, str) and text.strip():
            items.append((path, text.strip()))

    def mapping(candidate: Any) -> Mapping[str, Any]:
        return candidate if isinstance(candidate, Mapping) else {}

    def sequence(candidate: Any) -> list[Any]:
        return candidate if isinstance(candidate, list) else []

    visual_medium = mapping(value.get("visual_medium"))
    depicted_content = mapping(value.get("depicted_content"))
    text_assessment = mapping(value.get("text_assessment"))
    add("visual_medium.uncertainty_reason", visual_medium.get("uncertainty_reason"))
    add("depicted_content.summary", depicted_content.get("summary"))
    add("depicted_content.uncertainty_reason", depicted_content.get("uncertainty_reason"))
    add("scene_summary", value.get("scene_summary"))
    for index, raw_subject in enumerate(sequence(value.get("subjects"))):
        subject = mapping(raw_subject)
        add(f"subjects.{index}.name", subject.get("name"))
        add(f"subjects.{index}.uncertainty_reason", subject.get("uncertainty_reason"))
        for attr_index, raw_attribute in enumerate(sequence(subject.get("attributes"))):
            attribute = mapping(raw_attribute)
            add(f"subjects.{index}.attributes.{attr_index}.key", attribute.get("key"))
            add(f"subjects.{index}.attributes.{attr_index}.value", attribute.get("value"))
    for index, raw_activity in enumerate(sequence(value.get("activities"))):
        activity = mapping(raw_activity)
        add(f"activities.{index}.action", activity.get("action"))
        add(f"activities.{index}.uncertainty_reason", activity.get("uncertainty_reason"))
    for index, raw_relation in enumerate(sequence(value.get("relationships"))):
        relation = mapping(raw_relation)
        add(f"relationships.{index}.relation", relation.get("relation"))
        add(f"relationships.{index}.object_text", relation.get("object_text"))
        add(f"relationships.{index}.uncertainty_reason", relation.get("uncertainty_reason"))
    add("text_assessment.uncertainty_reason", text_assessment.get("uncertainty_reason"))
    for index, raw_selected in enumerate(sequence(text_assessment.get("selected_text"))):
        selected = mapping(raw_selected)
        add(f"text_assessment.selected_text.{index}.uncertainty_reason", selected.get("uncertainty_reason"))
    for index, raw_evidence in enumerate(sequence(value.get("evidence_items"))):
        evidence = mapping(raw_evidence)
        add(f"evidence_items.{index}.content", evidence.get("content"))
    for index, raw_claim in enumerate(sequence(value.get("claims"))):
        claim = mapping(raw_claim)
        add(f"claims.{index}.text", claim.get("text"))
        add(f"claims.{index}.uncertainty_reason", claim.get("uncertainty_reason"))
    return items


def _strip_exceptions(text: str, allowed_terms: Iterable[str], path: str) -> tuple[str, list[str]]:
    cleaned = text
    exceptions: list[str] = []
    for term in sorted({term for term in allowed_terms if term}, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(cleaned):
            exceptions.append(f"allowlisted_term:{path}:{term}")
            cleaned = pattern.sub(" ", cleaned)
    for label, pattern in (("formula", _FORMULA_RE), ("model_or_product", _MODEL_TOKEN_RE)):
        for matched in pattern.findall(cleaned):
            exceptions.append(f"{label}:{path}:{matched}")
        cleaned = pattern.sub(" ", cleaned)
    return cleaned, exceptions


def check_rc2_language(
    value: Mapping[str, Any],
    *,
    candidate_latin_share_max: float = 0.20,
    allowed_latin_terms: Iterable[str] = (),
) -> Rc2LanguageResult:
    """Evaluate Simplified-Chinese narrative while exempting visible foreign text."""

    violations: list[str] = []
    exceptions: list[str] = []
    cleaned_values: list[tuple[str, str]] = []
    for path, text in _narrative_items(value):
        cleaned, found = _strip_exceptions(text, allowed_latin_terms, path)
        exceptions.extend(found)
        cleaned_values.append((path, cleaned))
        han = len(_HAN_RE.findall(cleaned))
        latin = len(_LATIN_RE.findall(cleaned))
        if latin >= 4 and han == 0:
            violations.append(f"ascii_only_narrative:{path}")
        if _FUSED_RE.search(cleaned):
            violations.append(f"abnormal_han_latin_fusion:{path}")
        if any(char in _TRADITIONAL_ONLY for char in cleaned):
            violations.append(f"traditional_character_in_narrative:{path}")

    text_assessment = value.get("text_assessment")
    selected_text = text_assessment.get("selected_text", []) if isinstance(text_assessment, Mapping) else []
    for index, selected in enumerate(selected_text if isinstance(selected_text, list) else []):
        text = selected.get("text") if isinstance(selected, Mapping) else None
        if isinstance(text, str) and _LATIN_RE.search(text):
            exceptions.append(f"verbatim_visible_text:text_assessment.selected_text.{index}.text")

    combined = " ".join(text for _, text in cleaned_values)
    han_count = len(_HAN_RE.findall(combined))
    latin_count = len(_LATIN_RE.findall(combined))
    latin_share = latin_count / (han_count + latin_count) if han_count + latin_count else 0.0
    if han_count == 0:
        violations.append("no_simplified_chinese_narrative")
    if latin_share > candidate_latin_share_max:
        violations.append(f"latin_share_exceeds_candidate:{latin_share:.6f}")

    rules = {
        "has_simplified_chinese_narrative": han_count > 0,
        "no_ascii_only_narrative": not any(item.startswith("ascii_only_narrative:") for item in violations),
        "no_abnormal_han_latin_fusion": not any(item.startswith("abnormal_han_latin_fusion:") for item in violations),
        "no_traditional_character_signal": not any(item.startswith("traditional_character_in_narrative:") for item in violations),
        "latin_share_within_candidate": latin_share <= candidate_latin_share_max,
        "verbatim_visible_text_excluded": True,
    }
    unique_violations = tuple(dict.fromkeys(violations))
    return Rc2LanguageResult(
        latin_share=latin_share,
        han_characters=han_count,
        latin_letters=latin_count,
        candidate_threshold=candidate_latin_share_max,
        reasonable_exceptions=tuple(dict.fromkeys(exceptions)),
        violations=unique_violations,
        rules=rules,
        qualified=not unique_violations and all(rules.values()),
    )


def validate_rc2_shared_facts(
    value: Mapping[str, Any],
    path: str | Path = RC2_SCHEMA_PATH,
    *,
    raw_model_output: bool = True,
    candidate_latin_share_max: float = 0.20,
    allowed_latin_terms: Iterable[str] = (),
) -> Rc2LanguageResult:
    """Validate rc2 shared facts."""
    validate_rc2_structure(value, path, raw_model_output=raw_model_output)
    language = check_rc2_language(
        value,
        candidate_latin_share_max=candidate_latin_share_max,
        allowed_latin_terms=allowed_latin_terms,
    )
    if not language.qualified:
        raise AnnotationValidationError("language: " + "; ".join(language.violations))
    return language


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


def parse_rc2_json(raw_text: str) -> Rc2ParseResult:
    """Parse standard JSON; only a complete surrounding JSON fence is removable."""

    cleaned = raw_text.lstrip("\ufeff").strip()
    repair_type: str | None = None
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[len("```json") : -3].strip()
        repair_type = "surrounding_json_fence_removed"
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return Rc2ParseResult(
            value=None,
            error=f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            repair_type=repair_type,
            json_unclosed=_json_unclosed(cleaned),
        )
    if not isinstance(value, dict):
        return Rc2ParseResult(None, "root must be a JSON object", repair_type, False)
    return Rc2ParseResult(value=value, error=None, repair_type=repair_type, json_unclosed=False)  # type: ignore[arg-type]


def should_retry_rc2(*, finish_reason: str | None, parse_result: Rc2ParseResult) -> bool:
    """Allow the sole retry only for explicit truncation or unclosed JSON."""

    return finish_reason in {"length", "max_tokens"} or parse_result.json_unclosed
