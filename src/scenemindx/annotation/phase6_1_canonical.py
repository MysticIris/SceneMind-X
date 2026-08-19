"""Migrate and validate Phase 6.1 Canonical visual assets."""

from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from typing import Any, Mapping

from .canonical_closeout import (
    migrate_phase6_0a_label,
    validate_closeout_label,
)
from .canonical_label import canonical_json, sha256_text


SCHEMA_VERSION = "scenemindx_canonical_pseudo_label_v2_1_1_candidate"
MIGRATION_VERSION = "phase6_1_split_compatibility_migration_v1"
VALIDATOR_VERSION = "phase6_1_split_compatibility_validator_v1"
SUPPORTED_SOURCE_SCHEMAS = frozenset(
    {
        "scenemindx_canonical_pseudo_label_v2_candidate",
        "scenemindx_canonical_pseudo_label_v2_1_candidate",
        SCHEMA_VERSION,
    }
)
_RECOVERY_KEY_ALIASES = {
    "contract version": "contract_version",
    "contract_version": "contract_version",
    "display": "display",
    "fallback": "fallback",
    "evidence boundary": "evidence_boundary",
    "evidence_boundary": "evidence_boundary",
    "text evidence": "text_evidence",
    "text_evidence": "text_evidence",
    "theme": "theme",
    "short description": "short_description",
    "short_description": "short_description",
    "micro tags": "micro_tags",
    "micro_tags": "micro_tags",
    "safe facts": "safe_facts",
    "safe_facts": "safe_facts",
    "uncertainty": "uncertainty",
    "verification status": "verification_status",
    "verification_status": "verification_status",
}
_PRECISE_COUNT_RE = re.compile(
    r"(?:\d+|[一二三四五六七八九十百两]+)"
    r"(?:名|位|个|张|只|条|辆|棵|座|处|组|台|本|件|人)"
)
_KNOWN_SINGLE_INPUT_CARRIER_PHRASES = frozenset(
    {
        "这是一张图片",
        "这是一张图像",
        "这是一张照片",
        "这是一张截图",
    }
)


def _neutralize_recovery_public_precise_count(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized_value = value.strip().rstrip("。！？!?")
    if normalized_value in _KNOWN_SINGLE_INPUT_CARRIER_PHRASES:
        return value
    return _PRECISE_COUNT_RE.sub("", value)


def normalize_recovery_payload(
    payload: Mapping[str, Any],
    *,
    neutralize_public_counts: bool = True,
) -> dict[str, Any]:
    """Normalize formatting, public count safety, and frozen array bounds.

    Array bounding keeps the model's original order and values, never invents
    visual facts, and only removes values beyond the already-frozen Schema
    limits. Public precise-count fragments are deleted rather than replaced
    with another count, so the recovery path stays conservative.
    """

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                normalized_key = (
                    str(key)
                    .strip()
                    .replace("\u3000", " ")
                    .replace("-", " ")
                    .casefold()
                )
                normalized_key = " ".join(normalized_key.split())
                target = _RECOVERY_KEY_ALIASES.get(normalized_key, str(key))
                result[target] = normalize(item)
            return result
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    result = normalize(payload)
    contract = str(result.get("contract_version") or "")
    if contract.casefold() == "phase6_0b_recovery_payload_v1":
        result["contract_version"] = "phase6_0b_recovery_payload_v1"
    display = result.get("display")
    if isinstance(display, dict):
        if isinstance(display.get("micro_tags"), list):
            display["micro_tags"] = display["micro_tags"][:5]
    fallback = result.get("fallback")
    if isinstance(fallback, dict) and isinstance(fallback.get("safe_facts"), list):
        fallback["safe_facts"] = fallback["safe_facts"][:3]
    evidence = result.get("evidence_boundary")
    if isinstance(evidence, dict) and isinstance(evidence.get("uncertainty"), list):
        evidence["uncertainty"] = evidence["uncertainty"][:3]
    if neutralize_public_counts:
        if isinstance(display, dict):
            for key in ("theme", "short_description"):
                display[key] = _neutralize_recovery_public_precise_count(
                    display.get(key)
                )
            if isinstance(display.get("micro_tags"), list):
                display["micro_tags"] = [
                    _neutralize_recovery_public_precise_count(item)
                    for item in display["micro_tags"]
                ]
        if isinstance(fallback, dict) and isinstance(
            fallback.get("safe_facts"),
            list,
        ):
            fallback["safe_facts"] = [
                _neutralize_recovery_public_precise_count(item)
                for item in fallback["safe_facts"]
            ]
    return result


def validate_recovery_payload_safety(
    payload: Mapping[str, Any],
) -> list[str]:
    """Validate recovery payload safety."""
    public_values = [
        str((payload.get("display") or {}).get("theme") or ""),
        str((payload.get("display") or {}).get("short_description") or ""),
        *[
            str(item)
            for item in (payload.get("display") or {}).get("micro_tags") or []
        ],
        *[
            str(item)
            for item in (payload.get("fallback") or {}).get("safe_facts") or []
        ],
    ]
    errors: list[str] = []
    has_unsupported_precise_count = False
    for value in public_values:
        normalized_value = value.strip().rstrip("。！？!?")
        if normalized_value in _KNOWN_SINGLE_INPUT_CARRIER_PHRASES:
            continue
        if _PRECISE_COUNT_RE.search(value):
            has_unsupported_precise_count = True
            break
    if has_unsupported_precise_count:
        errors.append("recovery_public_precise_count")
    return errors


def _finalize(
    label: Mapping[str, Any],
    *,
    asset_split: str,
    source_run_id: str,
    source_label: Mapping[str, Any],
    migrated_at: str | None = None,
) -> dict[str, Any]:
    if asset_split not in {
        "train",
        "val",
        "user_custom",
        "session_temporary",
    }:
        raise ValueError(f"unsupported_asset_split:{asset_split}")
    source_schema = str(source_label.get("schema_version") or "")
    if source_schema not in SUPPORTED_SOURCE_SCHEMAS:
        raise ValueError(f"unsupported_source_schema:{source_schema}")

    value = copy.deepcopy(dict(label))
    source_label_id = str(source_label.get("label_id") or "")
    value["schema_version"] = SCHEMA_VERSION
    value["asset"]["split"] = asset_split
    asset_sha = str(value["asset"]["sha256"]).lower()
    value["label_id"] = f"cpl_{sha256_text(f'{asset_sha}:{SCHEMA_VERSION}')[:24]}"
    if source_label_id and source_label_id != value["label_id"]:
        value["supersedes"] = list(
            dict.fromkeys([source_label_id, *(value.get("supersedes") or [])])
        )
    value["provenance"] = {
        **dict(value.get("provenance") or {}),
        "source_run_id": source_run_id,
        "source_schema_version": source_schema,
        "source_label_id": source_label_id,
        "split_compatibility_migrated_at": (
            migrated_at or datetime.now(timezone.utc).isoformat()
        ),
        "split_compatibility_migration_version": MIGRATION_VERSION,
        "split_compatibility_validator": VALIDATOR_VERSION,
    }
    value.pop("canonical_sha256", None)
    value["canonical_sha256"] = sha256_text(canonical_json(value))
    return value


def migrate_phase6_0a_to_phase6_1(
    old_label: Mapping[str, Any],
    *,
    asset_split: str,
    legacy_review: Mapping[str, Any] | None,
    source_run_id: str,
    recovered_at_tokens: int | None = None,
    migrated_at: str | None = None,
) -> dict[str, Any]:
    """Execute the migrate phase6 0a to phase6 1 operation."""
    base = migrate_phase6_0a_label(
        old_label,
        legacy_review=legacy_review,
        source_run_id=source_run_id,
        recovered_at_tokens=recovered_at_tokens,
        migrated_at=migrated_at,
    )
    legacy_errors = list(
        (old_label.get("quality") or {}).get("validation_errors") or []
    )
    if legacy_errors:
        warnings = list(base["quality"].get("warning_codes") or [])
        warnings.extend(f"legacy_contract_{error}" for error in legacy_errors)
        base["quality"]["warning_codes"] = sorted(set(warnings))
        base["quality"]["needs_review"] = True
        base["quality"]["semantic_status"] = "needs_review"
        base["review"]["automatic_validation"]["status"] = "warning"
    return _finalize(
        base,
        asset_split=asset_split,
        source_run_id=source_run_id,
        source_label=old_label,
        migrated_at=migrated_at,
    )


def upgrade_phase6_1_compatible_label(
    old_label: Mapping[str, Any],
    *,
    asset_split: str,
    source_run_id: str,
    migrated_at: str | None = None,
) -> dict[str, Any]:
    """Execute the upgrade phase6 1 compatible label operation."""
    source_schema = str(old_label.get("schema_version") or "")
    if source_schema == "scenemindx_canonical_pseudo_label_v2_candidate":
        return migrate_phase6_0a_to_phase6_1(
            old_label,
            asset_split=asset_split,
            legacy_review=None,
            source_run_id=source_run_id,
            migrated_at=migrated_at,
        )
    return _finalize(
        old_label,
        asset_split=asset_split,
        source_run_id=source_run_id,
        source_label=old_label,
        migrated_at=migrated_at,
    )


def validate_phase6_1_label(
    label: Mapping[str, Any],
    *,
    verify_hash: bool = True,
) -> list[str]:
    """Validate phase6 1 label."""
    return validate_closeout_label(
        label,
        verify_hash=verify_hash,
        expected_schema_version=SCHEMA_VERSION,
    )
