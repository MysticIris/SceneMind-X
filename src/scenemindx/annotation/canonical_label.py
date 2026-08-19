"""Phase 6.0-A canonical pseudo-label normalization and validation.

The model owns only the semantic payload.  Asset identity, provenance,
verification state, quality decisions, and active-version state are attached
deterministically so malformed model output cannot silently become an active
label.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from scenemindx.task_text_policy import (
    candidate_partition,
    is_qualified_statement,
    prioritize_candidate_values,
    repair_filtered_text,
)


SCHEMA_VERSION = "scenemindx_canonical_pseudo_label_v2_candidate"
MODEL_PAYLOAD_VERSION = "phase6_0a_model_payload_v1"
PROMPT_ID = "phase6_0a_canonical_pseudo_label_v1"
ALLOWED_MEDIA = {
    "photograph",
    "screenshot",
    "illustration",
    "poster_document",
    "product_packaging",
    "mixed",
    "uncertain",
}
ALLOWED_TEXT_PRESENCE = {
    "none",
    "present_readable",
    "present_unreadable",
    "uncertain",
}
_INTERNAL_LEAK_RE = re.compile(
    r"(?:prompt|validator|trace|schema|json|asset_id|sha256|canonical|"
    r"模型指令|系统提示|内部字段)",
    re.IGNORECASE,
)
_UNVERIFIED_TEXT_ASSERTION_RE = re.compile(
    r"(?:写着|写有|印有|显示为|配有文字|文字(?:是|为)?|标题为|标牌为|"
    r"字幕为|号码为|型号为|[“「『‘'][^”」』’']{2,40}[”」』’'])"
)


def canonical_json(value: Any) -> str:
    """Execute the canonical json operation."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    """Execute the sha256 text operation."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text(value: Any, *, maximum: int = 200) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = " ".join(normalized.split()).strip()
    return normalized[:maximum]


def _texts(
    value: Any,
    *,
    maximum_items: int,
    maximum_length: int,
) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _text(item, maximum=maximum_length)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= maximum_items:
            break
    return result


def extract_json_object(raw_output: str) -> dict[str, Any]:
    """Extract the first balanced JSON object without semantic recovery."""

    text = str(raw_output or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    candidates = [text]
    smart_quote_repair = text.translate(str.maketrans({"“": '"', "”": '"'}))
    if smart_quote_repair != text:
        candidates.append(smart_quote_repair)
    for candidate in candidates:
        index = candidate.find("{")
        if index < 0:
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model_output_did_not_contain_json_object")


def normalize_model_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Perform lossless formatting normalization only."""

    display = payload.get("display") if isinstance(payload.get("display"), Mapping) else {}
    facts = payload.get("facts") if isinstance(payload.get("facts"), Mapping) else {}
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), Mapping) else {}
    text_evidence = (
        payload.get("text_evidence")
        if isinstance(payload.get("text_evidence"), Mapping)
        else {}
    )
    fallback = payload.get("fallback") if isinstance(payload.get("fallback"), Mapping) else {}
    payload_version = _text(payload.get("payload_version"), maximum=80)
    if payload_version == "phase6_0a model payload v1":
        payload_version = MODEL_PAYLOAD_VERSION
    return {
        "payload_version": payload_version,
        "display": {
            "theme": _text(display.get("theme"), maximum=24),
            "short_description": _text(display.get("short_description"), maximum=140),
            "micro_tags": _texts(display.get("micro_tags"), maximum_items=8, maximum_length=18),
        },
        "facts": {
            "visual_medium": _text(facts.get("visual_medium"), maximum=32),
            "scene": _text(facts.get("scene"), maximum=100),
            "subjects": _texts(facts.get("subjects"), maximum_items=6, maximum_length=40),
            "actions": _texts(facts.get("actions"), maximum_items=5, maximum_length=40),
            "attributes": _texts(facts.get("attributes"), maximum_items=6, maximum_length=40),
            "relations": _texts(facts.get("relations"), maximum_items=5, maximum_length=60),
        },
        "evidence": {
            "direct_observations": _texts(
                evidence.get("direct_observations"),
                maximum_items=8,
                maximum_length=80,
            ),
            "cautious_inferences": _texts(
                evidence.get("cautious_inferences"),
                maximum_items=4,
                maximum_length=80,
            ),
            "uncertainties": _texts(
                evidence.get("uncertainties"),
                maximum_items=4,
                maximum_length=80,
            ),
        },
        "text_evidence": {
            "presence": _text(text_evidence.get("presence"), maximum=32),
            "visual_candidates": prioritize_candidate_values(
                text_evidence.get("visual_candidates")
                if isinstance(text_evidence.get("visual_candidates"), list)
                else [],
                maximum_items=6,
                maximum_length=40,
            ),
        },
        "fallback": {
            "safe_caption": _text(fallback.get("safe_caption"), maximum=140),
            "safe_facts": _texts(
                fallback.get("safe_facts"),
                maximum_items=5,
                maximum_length=80,
            ),
        },
    }


def validate_model_payload(
    payload: Mapping[str, Any],
    *,
    after_sanitization: bool = False,
) -> list[str]:
    """Validate model payload."""
    errors: list[str] = []
    if payload.get("payload_version") != MODEL_PAYLOAD_VERSION:
        errors.append("payload_version_mismatch")
    display = payload.get("display", {})
    facts = payload.get("facts", {})
    evidence = payload.get("evidence", {})
    text_evidence = payload.get("text_evidence", {})
    fallback = payload.get("fallback", {})

    theme = str(display.get("theme", ""))
    description = str(display.get("short_description", ""))
    tags = display.get("micro_tags", [])
    if not 2 <= len(theme) <= 24:
        errors.append("display_theme_length")
    minimum_description = 8 if after_sanitization else 12
    if not minimum_description <= len(description) <= 140:
        errors.append("short_description_length")
    minimum_tags = 1 if after_sanitization else 3
    if not isinstance(tags, list) or not minimum_tags <= len(tags) <= 8:
        errors.append("micro_tags_count")
    if facts.get("visual_medium") not in ALLOWED_MEDIA:
        errors.append("visual_medium_invalid")
    if not isinstance(facts.get("subjects"), list) or not facts.get("subjects"):
        errors.append("subjects_empty")
    observations = evidence.get("direct_observations", [])
    minimum_observations = 1 if after_sanitization else 2
    if not isinstance(observations, list) or len(observations) < minimum_observations:
        errors.append("direct_observations_insufficient")
    presence = text_evidence.get("presence")
    if presence not in ALLOWED_TEXT_PRESENCE:
        errors.append("text_presence_invalid")
    candidates = text_evidence.get("visual_candidates", [])
    if presence in {"none", "present_unreadable", "uncertain"} and candidates:
        errors.append("unsafe_text_candidates_for_presence")
    safe_caption = str(fallback.get("safe_caption", ""))
    safe_facts = fallback.get("safe_facts", [])
    if not 8 <= len(safe_caption) <= 140:
        errors.append("safe_caption_length")
    if not isinstance(safe_facts, list) or not safe_facts:
        errors.append("safe_facts_empty")

    public_strings = [theme, description, *tags, safe_caption, *safe_facts]
    if any(_INTERNAL_LEAK_RE.search(value) for value in public_strings):
        errors.append("internal_term_leak")
    safe_strings = [safe_caption, *safe_facts]
    retained_values, blocked_values, _ = candidate_partition(text_evidence)
    retained_keys = {_text(value).casefold() for value in retained_values if _text(value)}
    blocked_keys = {_text(value).casefold() for value in blocked_values if _text(value)}
    if any(
        _UNVERIFIED_TEXT_ASSERTION_RE.search(value)
        and not any(candidate in value.casefold() for candidate in retained_keys)
        for value in safe_strings
    ):
        errors.append("unverified_specific_text_in_fallback")
    if blocked_keys and any(
        candidate in value.casefold()
        for candidate in blocked_keys
        for value in safe_strings
    ):
        errors.append("visual_text_candidate_promoted_to_fallback_fact")
    return sorted(set(errors))


def sanitize_unverified_text(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Remove unverified specific-text clauses without adding visual semantics."""

    value = json.loads(json.dumps(payload, ensure_ascii=False))
    retained_values, blocked_values, _ = candidate_partition(
        value.get("text_evidence", {})
    )
    retained = {_text(item).casefold() for item in retained_values if _text(item)}
    blocked = {_text(item).casefold() for item in blocked_values if _text(item)}
    actions: list[str] = []
    original_candidates = list(
        value.get("text_evidence", {}).get("visual_candidates", [])
    )
    value["text_evidence"]["visual_candidates"] = [
        item
        for item in original_candidates
        if _text(item).casefold() in retained
    ]
    if len(value["text_evidence"]["visual_candidates"]) != len(
        original_candidates
    ):
        actions.append("removed_low_confidence_text_candidate")

    def unsafe(
        text: str,
        *,
        exact_candidate: bool = False,
        candidate_substring: bool = False,
    ) -> bool:
        key = text.casefold()
        contains_retained = any(candidate in key for candidate in retained)
        if _UNVERIFIED_TEXT_ASSERTION_RE.search(text) and not contains_retained:
            return True
        if is_qualified_statement(text):
            return True
        if exact_candidate and key in blocked:
            return True
        return candidate_substring and any(candidate in key for candidate in blocked)

    def clean_sentence(text: str, field: str) -> str:
        parts = re.split(r"([，。；;])", text)
        kept: list[str] = []
        removed = False
        index = 0
        while index < len(parts):
            clause = parts[index]
            separator = parts[index + 1] if index + 1 < len(parts) else ""
            if clause.strip() and unsafe(clause):
                removed = True
            else:
                kept.extend([clause, separator])
            index += 2
        cleaned = repair_filtered_text("".join(kept), sentence=False)
        if removed:
            actions.append(f"removed_unverified_text_clause:{field}")
        return cleaned

    def redact_candidate_substrings(text: str, field: str) -> str:
        cleaned = text
        for candidate in sorted(blocked, key=len, reverse=True):
            if candidate and candidate in cleaned.casefold():
                cleaned = re.sub(re.escape(candidate), "", cleaned, flags=re.IGNORECASE)
                actions.append(f"redacted_visual_text_candidate:{field}")
        return repair_filtered_text(cleaned, sentence=False)

    display = value["display"]
    display["theme"] = redact_candidate_substrings(
        clean_sentence(display["theme"], "display.theme"),
        "display.theme",
    )
    display["short_description"] = redact_candidate_substrings(
        clean_sentence(
            display["short_description"],
            "display.short_description",
        ),
        "display.short_description",
    )
    display["short_description"] = repair_filtered_text(
        display["short_description"],
        sentence=True,
    )
    display["micro_tags"] = [
        item
        for item in display["micro_tags"]
        if not unsafe(item, exact_candidate=True)
    ]
    if len(display["theme"]) < 2 and display["micro_tags"]:
        display["theme"] = display["micro_tags"][0]
        actions.append("reused_model_micro_tag_for_theme")
    if len(display["short_description"]) < 12:
        safe_caption = value["fallback"]["safe_caption"]
        if not unsafe(safe_caption) and len(safe_caption) >= 8:
            display["short_description"] = safe_caption
            actions.append("reused_model_safe_caption_for_short_description")

    facts = value["facts"]
    facts["scene"] = repair_filtered_text(
        clean_sentence(facts["scene"], "facts.scene"),
        sentence=False,
    )
    for field in ("subjects", "actions", "attributes", "relations"):
        before = facts[field]
        facts[field] = [
            repair_filtered_text(item, sentence=field == "relations")
            for item in before
            if not unsafe(item, exact_candidate=True)
        ]
        if len(facts[field]) != len(before):
            actions.append(f"removed_unverified_text_item:facts.{field}")
    evidence = value["evidence"]
    for field in ("direct_observations", "cautious_inferences", "uncertainties"):
        before = evidence[field]
        evidence[field] = [
            repair_filtered_text(item, sentence=True)
            for item in before
            if not unsafe(item, exact_candidate=True)
        ]
        if len(evidence[field]) != len(before):
            actions.append(f"removed_unverified_text_item:evidence.{field}")
    fallback = value["fallback"]
    fallback["safe_caption"] = redact_candidate_substrings(
        clean_sentence(
            fallback["safe_caption"],
            "fallback.safe_caption",
        ),
        "fallback.safe_caption",
    )
    fallback["safe_caption"] = repair_filtered_text(
        fallback["safe_caption"],
        sentence=True,
    )
    before_facts = fallback["safe_facts"]
    fallback["safe_facts"] = [
        repair_filtered_text(item, sentence=True)
        for item in before_facts
        if not unsafe(item, exact_candidate=True, candidate_substring=True)
    ]
    if len(fallback["safe_facts"]) != len(before_facts):
        actions.append("removed_unverified_text_item:fallback.safe_facts")
    if not evidence["direct_observations"] and fallback["safe_facts"]:
        evidence["direct_observations"] = list(fallback["safe_facts"])
        actions.append("reused_model_safe_facts_for_direct_observations")
    return value, sorted(set(actions))


def build_canonical_label(
    payload: Mapping[str, Any],
    *,
    asset_id: str,
    asset_sha256: str,
    relative_path: str,
    asset_split: str = "train",
    prompt_sha256: str,
    model: str,
    model_revision: str,
    raw_output_sha256: str,
    source_run_id: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build canonical label."""
    if asset_split not in {
        "train",
        "val",
        "user_custom",
        "session_temporary",
    }:
        raise ValueError(f"unsupported_asset_split:{asset_split}")
    normalized = normalize_model_payload(payload)
    pre_sanitization_errors = validate_model_payload(normalized)
    sanitized, sanitization_actions = sanitize_unverified_text(normalized)
    sanitizable_pre_errors = {
        "unverified_specific_text_in_fallback",
        "visual_text_candidate_promoted_to_fallback_fact",
    }
    errors = sorted(
        set(
            [
                error
                for error in pre_sanitization_errors
                if error not in sanitizable_pre_errors
            ]
            + validate_model_payload(sanitized, after_sanitization=True)
        )
    )
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    identity_key = f"{asset_sha256}:{SCHEMA_VERSION}"
    label_id = f"cpl_{sha256_text(identity_key)[:24]}"
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "label_id": label_id,
        "asset": {
            "asset_id": str(asset_id),
            "sha256": str(asset_sha256).lower(),
            "split": asset_split,
            "relative_path": str(relative_path),
        },
        "display": sanitized["display"],
        "facts": sanitized["facts"],
        "evidence": sanitized["evidence"],
        "text_evidence": {
            **sanitized["text_evidence"],
            "verification_status": "unverified_visual_candidate_only",
            "verified_text": [],
            "verified_text_source": "not_available",
        },
        "fallback": {
            **sanitized["fallback"],
            "policy": "use_only_when_public_model_body_invalid",
            "must_not_override_valid_model_output": True,
        },
        "quality": {
            "status": "accepted" if not errors else "rejected",
            "schema_valid": not errors,
            "validation_errors": errors,
            "human_review_status": "pending",
            "sanitization_applied": bool(sanitization_actions),
            "sanitization_actions": sanitization_actions,
        },
        "provenance": {
            "source_run_id": source_run_id,
            "generated_at": timestamp,
            "prompt_id": PROMPT_ID,
            "prompt_sha256": prompt_sha256,
            "model": model,
            "model_revision": model_revision,
            "raw_output_sha256": raw_output_sha256,
            "normalizer": "phase6_0a_lossless_normalizer_v1",
            "validator": "phase6_0a_canonical_validator_v1",
        },
    }
    canonical["canonical_sha256"] = sha256_text(canonical_json(canonical))
    return canonical


@dataclass(frozen=True)
class PublicFallbackDecision:
    """Provide public fallback decision behavior."""
    text: str
    source: str
    applied: bool
    reason: str | None


def select_public_body(
    model_body: Any,
    canonical_label: Mapping[str, Any],
) -> PublicFallbackDecision:
    """Use fallback only when the public model body is absent or structurally unsafe."""

    if isinstance(model_body, str):
        body = _text(model_body, maximum=4000)
        if body and not _INTERNAL_LEAK_RE.search(body) and not body.startswith(("{", "[")):
            return PublicFallbackDecision(body, "model", False, None)
    quality = canonical_label.get("quality", {})
    if quality.get("status") != "accepted":
        return PublicFallbackDecision(
            "当前图片暂时没有可安全展示的描述。",
            "safe_refusal",
            True,
            "canonical_label_not_accepted",
        )
    fallback = canonical_label.get("fallback", {})
    caption = _text(fallback.get("safe_caption"), maximum=140)
    if not caption:
        return PublicFallbackDecision(
            "当前图片暂时没有可安全展示的描述。",
            "safe_refusal",
            True,
            "safe_caption_empty",
        )
    return PublicFallbackDecision(
        caption,
        "canonical_pseudo_label_safe_caption",
        True,
        "public_model_body_invalid",
    )
