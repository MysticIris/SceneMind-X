"""Verified OCR evidence utilities for Phase 5.3 Stage 5.

OCR output is treated as a candidate observation.  A string becomes retrieval
or answer evidence only after an independent OCR pass agrees on both text and
region under the frozen policy.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .hybrid_recall import BM25Index, character_bigram_tokens


_MATERIAL_CHAR_RE = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")
_CANONICAL_DROP_RE = re.compile(r"[^\u3400-\u9fffA-Za-z0-9]+")
_CAMERA_WATERMARK_RE = re.compile(
    r"(?:huawei|leica|dual\s*camera|shot\s*on|ai\s*camera|mate\s*\d+)",
    re.IGNORECASE,
)
_SPECIFIC_TEXT_QUESTION_RE = re.compile(
    r"(?:写着|文字|标牌|招牌|标题|字幕|内容|英文|中文|号码|型号|word|text|"
    r"sign|title|caption|number|model)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VerificationPolicy:
    """Provide verification policy behavior."""
    primary_confidence_min: float = 0.95
    verifier_confidence_min: float = 0.95
    region_iou_min: float = 0.35
    canonical_length_min: int = 2
    exclude_camera_watermarks: bool = True


def normalize_ocr_text(value: str) -> str:
    """Return a stable display form without inventing or correcting content."""

    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.split()).strip()


def canonical_ocr_text(value: str) -> str:
    """Return the strict equality key used for cross-model agreement."""

    return _CANONICAL_DROP_RE.sub("", normalize_ocr_text(value)).casefold()


def infer_script_type(value: str) -> str:
    """Execute the infer script type operation."""
    normalized = normalize_ocr_text(value)
    has_cjk = bool(re.search(r"[\u3400-\u9fff]", normalized))
    has_latin = bool(re.search(r"[A-Za-z]", normalized))
    has_digit = bool(re.search(r"\d", normalized))
    if has_cjk and has_latin:
        return "mixed_cjk_latin"
    if has_cjk:
        return "cjk"
    if has_latin and has_digit:
        return "latin_numeric"
    if has_latin:
        return "latin"
    if has_digit:
        return "numeric"
    return "other"


def classify_text_role(value: str) -> str:
    """Execute the classify text role operation."""
    return "camera_watermark" if _CAMERA_WATERMARK_RE.search(value) else "material"


def _bbox(candidate: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    raw = candidate.get("polygon")
    if raw is None:
        raw = candidate.get("box")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    if (
        len(raw) == 4
        and all(isinstance(value, (int, float)) for value in raw)
    ):
        left, top, right, bottom = map(float, raw)
    else:
        points: list[tuple[float, float]] = []
        for point in raw:
            if (
                isinstance(point, Sequence)
                and not isinstance(point, (str, bytes))
                and len(point) >= 2
                and isinstance(point[0], (int, float))
                and isinstance(point[1], (int, float))
            ):
                points.append((float(point[0]), float(point[1])))
        if not points:
            return None
        left = min(point[0] for point in points)
        top = min(point[1] for point in points)
        right = max(point[0] for point in points)
        bottom = max(point[1] for point in points)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def region_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Execute the region iou operation."""
    left_box, right_box = _bbox(left), _bbox(right)
    if left_box is None or right_box is None:
        return 0.0
    x1 = max(left_box[0], right_box[0])
    y1 = max(left_box[1], right_box[1])
    x2 = min(left_box[2], right_box[2])
    y2 = min(left_box[3], right_box[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left_box[2] - left_box[0]) * (left_box[3] - left_box[1])
    right_area = (right_box[2] - right_box[0]) * (right_box[3] - right_box[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _confidence(candidate: Mapping[str, Any]) -> float:
    value = candidate.get("confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _is_material_string(value: str, *, minimum_length: int) -> bool:
    canonical = canonical_ocr_text(value)
    return len(canonical) >= minimum_length and bool(_MATERIAL_CHAR_RE.search(canonical))


def build_verified_evidence(
    primary_candidates: Iterable[Mapping[str, Any]],
    verifier_candidates: Iterable[Mapping[str, Any]],
    *,
    policy: VerificationPolicy | None = None,
) -> dict[str, Any]:
    """Cross-check primary OCR candidates against an independent verifier."""

    policy = policy or VerificationPolicy()
    verifier_rows = [dict(row) for row in verifier_candidates]
    used_verifier_indices: set[int] = set()
    candidates: list[dict[str, Any]] = []
    verified_text: list[str] = []

    for index, raw_primary in enumerate(primary_candidates, start=1):
        primary = dict(raw_primary)
        text = normalize_ocr_text(primary.get("text", ""))
        canonical = canonical_ocr_text(text)
        role = classify_text_role(text)
        best_match: tuple[int, Mapping[str, Any], float] | None = None
        for verifier_index, verifier in enumerate(verifier_rows):
            if verifier_index in used_verifier_indices:
                continue
            if canonical_ocr_text(verifier.get("text", "")) != canonical:
                continue
            overlap = region_iou(primary, verifier)
            if best_match is None or overlap > best_match[2]:
                best_match = (verifier_index, verifier, overlap)

        reasons: list[str] = []
        if not _is_material_string(text, minimum_length=policy.canonical_length_min):
            reasons.append("non_material_or_too_short")
        if _confidence(primary) < policy.primary_confidence_min:
            reasons.append("primary_confidence_below_threshold")
        if best_match is None:
            reasons.append("no_exact_independent_text_match")
        else:
            verifier_index, verifier, overlap = best_match
            if _confidence(verifier) < policy.verifier_confidence_min:
                reasons.append("verifier_confidence_below_threshold")
            if overlap < policy.region_iou_min:
                reasons.append("region_iou_below_threshold")
        if role == "camera_watermark" and policy.exclude_camera_watermarks:
            reasons.append("excluded_camera_watermark")

        is_verified = not reasons
        evidence: dict[str, Any] = {
            "region_id": str(primary.get("region_id") or f"p{index}"),
            "text": text,
            "canonical_text": canonical,
            "script_type": infer_script_type(text),
            "role": role,
            "confidence": _confidence(primary),
            "polygon": primary.get("polygon", primary.get("box")),
            "status": "verified" if is_verified else "candidate_only",
            "verification_reasons": reasons,
            "primary_source": primary.get("source", "primary_ocr"),
            "verifier": None,
        }
        if best_match is not None:
            verifier_index, verifier, overlap = best_match
            evidence["verifier"] = {
                "region_id": verifier.get("region_id"),
                "text": normalize_ocr_text(verifier.get("text", "")),
                "confidence": _confidence(verifier),
                "polygon": verifier.get("polygon", verifier.get("box")),
                "source": verifier.get("source", "independent_ocr"),
                "region_iou": overlap,
            }
            if is_verified:
                used_verifier_indices.add(verifier_index)
        candidates.append(evidence)
        if is_verified and text not in verified_text:
            verified_text.append(text)

    return {
        "policy": {
            "primary_confidence_min": policy.primary_confidence_min,
            "verifier_confidence_min": policy.verifier_confidence_min,
            "region_iou_min": policy.region_iou_min,
            "canonical_length_min": policy.canonical_length_min,
            "exclude_camera_watermarks": policy.exclude_camera_watermarks,
        },
        "candidates": candidates,
        "verified_text": verified_text,
        "verified_text_joined": " ".join(verified_text),
        "candidate_count": len(candidates),
        "verified_count": len(verified_text),
    }


def answer_verified_text_question(
    question: str,
    verified_text: Iterable[str],
) -> dict[str, Any]:
    """Return only auditable text evidence, otherwise refuse to guess."""

    evidence = [
        normalize_ocr_text(value)
        for value in verified_text
        if _is_material_string(value, minimum_length=2)
        and classify_text_role(value) == "material"
    ]
    if not evidence:
        return {
            "status": "insufficient_verified_text_evidence",
            "answer": "当前图片没有足够的已核验文字证据，无法可靠回答。",
            "verified_text": [],
            "specific_text_question": bool(_SPECIFIC_TEXT_QUESTION_RE.search(question)),
        }
    return {
        "status": "answered_from_verified_text",
        "answer": "；".join(evidence),
        "verified_text": evidence,
        "specific_text_question": bool(_SPECIFIC_TEXT_QUESTION_RE.search(question)),
    }


def filter_specific_text_claims(
    claims: Iterable[str],
    verified_text: Iterable[str],
) -> dict[str, list[str]]:
    """Partition specific text claims by exact evidence containment."""

    evidence = {
        canonical_ocr_text(value)
        for value in verified_text
        if _is_material_string(value, minimum_length=2)
        and classify_text_role(value) == "material"
    }
    accepted: list[str] = []
    rejected: list[str] = []
    for claim in claims:
        claim_key = canonical_ocr_text(claim)
        if claim_key and any(claim_key in value or value in claim_key for value in evidence):
            accepted.append(str(claim))
        else:
            rejected.append(str(claim))
    return {"accepted": accepted, "rejected": rejected}


def search_verified_text(
    documents: Sequence[tuple[str, str]],
    query: str,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search a verified-text-only BM25 branch."""

    if not documents:
        return []
    index = BM25Index(
        documents,
        tokenizer=character_bigram_tokens,
        source="verified_text_bm25",
    )
    return index.search(query, top_k=top_k)
