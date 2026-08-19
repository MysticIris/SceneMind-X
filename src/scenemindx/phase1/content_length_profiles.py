"""Authoritative content-type input and advisory output length profiles."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


PROFILE_VERSION = "content-length-profiles-v1"
IDEAL_MIN_RATIO = Decimal("0.75")
IDEAL_MAX_RATIO = Decimal("1.30")

CONTENT_LENGTH_PROFILES: dict[str, dict[str, int]] = {
    "auto": {"default": 300, "input_min": 10, "input_max": 1000},
    "objective_description": {
        "default": 200,
        "input_min": 80,
        "input_max": 600,
    },
    "moments": {"default": 80, "input_min": 20, "input_max": 200},
    "travel_diary": {"default": 400, "input_min": 150, "input_max": 1000},
    "news_caption": {"default": 40, "input_min": 15, "input_max": 100},
    "advertisement": {"default": 35, "input_min": 10, "input_max": 100},
    "poster_title": {"default": 12, "input_min": 4, "input_max": 30},
    "poem": {"default": 80, "input_min": 40, "input_max": 200},
    "creative_story": {"default": 400, "input_min": 200, "input_max": 1000},
    "article": {"default": 800, "input_min": 300, "input_max": 1000},
}

ALIASES = {
    "story": "creative_story",
    "advertising_copy": "advertisement",
    "general_article": "article",
}


def canonical_content_type(content_type: str | None) -> str:
    """Execute the canonical content type operation."""
    key = ALIASES.get(str(content_type or "auto"), str(content_type or "auto"))
    return key if key in CONTENT_LENGTH_PROFILES else "auto"


def content_length_profile(content_type: str | None) -> dict[str, int]:
    """Execute the content length profile operation."""
    return dict(CONTENT_LENGTH_PROFILES[canonical_content_type(content_type)])


def ideal_output_window(target_length: int) -> tuple[int, int]:
    """Execute the ideal output window operation."""
    target = max(1, int(target_length))
    minimum = int((Decimal(target) * IDEAL_MIN_RATIO).to_integral_value(
        rounding=ROUND_HALF_UP
    ))
    maximum = int((Decimal(target) * IDEAL_MAX_RATIO).to_integral_value(
        rounding=ROUND_HALF_UP
    ))
    return max(1, minimum), max(1, maximum)


def _parse_length(value: Any) -> tuple[int | None, str | None]:
    if value is None:
        return None, "missing"
    if isinstance(value, bool):
        return None, "not_numeric"
    text = str(value).strip()
    if not text:
        return None, "empty"
    if len(text) > 64:
        return None, "not_numeric"
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None, "not_numeric"
    if not number.is_finite():
        return None, "not_numeric"
    rounded = int(number.to_integral_value(rounding=ROUND_HALF_UP))
    return rounded, "rounded" if number != Decimal(rounded) else None


def normalize_target_length(
    value: Any,
    content_type: str | None,
    *,
    explicit: bool | None = None,
) -> dict[str, Any]:
    """Parse, default and clamp one target against its resolved content type."""

    key = canonical_content_type(content_type)
    profile = content_length_profile(key)
    parsed, parse_reason = _parse_length(value)
    supplied = value is not None and str(value).strip() != ""
    is_explicit = supplied if explicit is None else bool(explicit)
    reasons: list[str] = []
    if parsed is None:
        target = profile["default"]
        if is_explicit and parse_reason not in {None, "missing", "empty"}:
            reasons.append("invalid_reset_to_default")
    else:
        target = parsed
        if parse_reason == "rounded":
            reasons.append("decimal_rounded")
    if target < profile["input_min"]:
        target = profile["input_min"]
        reasons.append("clamped_to_minimum")
    elif target > profile["input_max"]:
        target = profile["input_max"]
        reasons.append("clamped_to_maximum")
    adjusted = bool(reasons)
    hint = None
    if "clamped_to_maximum" in reasons and key in {"auto", "article"}:
        hint = (
            f"该内容类型最多支持{profile['input_max']}字，"
            f"目标长度已调整为{target}字。"
        )
    elif adjusted:
        hint = (
            f"已将目标长度调整为 {target} 字；"
            f"{key} 可设置 {profile['input_min']}–{profile['input_max']} 字。"
        )
    return {
        "profile_version": PROFILE_VERSION,
        "content_type": key,
        "original_value": value,
        "target": target,
        "input_min": profile["input_min"],
        "input_max": profile["input_max"],
        "default": profile["default"],
        "explicit": is_explicit,
        "adjusted": adjusted,
        "reasons": reasons,
        "public_hint": hint,
    }


def public_content_length_config() -> dict[str, Any]:
    """Execute the public content length config operation."""
    return {
        "version": PROFILE_VERSION,
        "ideal_output_ratio": {
            "minimum": float(IDEAL_MIN_RATIO),
            "maximum": float(IDEAL_MAX_RATIO),
            "gate": "advisory_warning_only",
        },
        "profiles": {
            key: dict(value) for key, value in CONTENT_LENGTH_PROFILES.items()
        },
        "aliases": dict(ALIASES),
    }
