"""Gate 1-D3 semantic review and OCR evidence-fusion contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


D3_RATINGS = {
    "correct",
    "partially_correct",
    "incorrect",
    "not_applicable",
    "uncertain",
}

D3_SEMANTIC_DIMENSIONS = (
    "global_scene_correctness",
    "primary_subject_coverage",
    "secondary_subject_omission",
    "activity_correctness",
    "attribute_correctness",
    "spatial_relation_correctness",
    "counting_correctness",
    "chinese_printed_text",
    "chinese_handwriting_candidates",
    "unreadable_text_abstention",
    "reflection_real_object_separation",
    "screen_printed_live_object_separation",
    "scene_inference_evidence",
    "inference_as_fact_error",
    "hallucination",
    "uncertainty_quality",
    "output_truncation",
)

D3_TEACHER_REVIEW_DIMENSIONS = (
    "global_scene",
    "primary_subject_coverage",
    "secondary_subject_coverage",
    "activity_correctness",
    "attribute_correctness",
    "spatial_relation_correctness",
    "count_correctness",
    "printed_chinese",
    "handwritten_chinese",
    "unreadable_text_rejection",
    "reflection_vs_direct_object",
    "screen_or_poster_vs_real_scene",
    "scene_evidence_coverage",
    "inference_overreach",
    "hallucination",
    "uncertainty_quality",
    "repetition",
    "truncation",
)


def new_teacher_review_record(image_id: str) -> dict[str, Any]:
    """Return an unconfirmed paired-model review record for the local UI."""

    return {
        "image_id": image_id,
        "review_status": "pending_human_review",
        "model_ratings": {
            model: {dimension: None for dimension in D3_TEACHER_REVIEW_DIMENSIONS}
            for model in ("2b", "4b")
        },
        "preferred_model": None,
        "notes": "",
        "reviewer": None,
        "reviewed_at": None,
        "human_confirmed": False,
    }


def new_human_evaluation_record(image_id: str, model_id: str, route: str) -> dict[str, Any]:
    """Return an unscored record; null ratings cannot be mistaken for results."""

    return {
        "image_id": image_id,
        "model_id": model_id,
        "route": route,
        "review_status": "pending_human_review",
        "ratings": {dimension: None for dimension in D3_SEMANTIC_DIMENSIONS},
        "notes": "",
        "reviewer": None,
        "reviewed_at": None,
    }


def fuse_text_evidence(
    visual_evidence: list[Mapping[str, Any]],
    ocr_candidates: list[Mapping[str, Any]],
    vlm_candidates: list[Mapping[str, Any]],
    scene_hypotheses: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Preserve OCR/VLM conflicts and reject unsupported high-confidence claims."""

    visual = deepcopy(visual_evidence)
    ocr = deepcopy(ocr_candidates)
    vlm = deepcopy(vlm_candidates)
    hypotheses = deepcopy(scene_hypotheses)
    conflicts: list[dict[str, Any]] = []

    ocr_by_region = {str(item["region_id"]): item for item in ocr}
    vlm_by_region = {str(item["region_id"]): item for item in vlm}
    for region_id in sorted(set(ocr_by_region) & set(vlm_by_region)):
        left = ocr_by_region[region_id]
        right = vlm_by_region[region_id]
        if left.get("text") and right.get("text") and left["text"] != right["text"]:
            conflicts.append({
                "region_id": region_id,
                "ocr_text": left["text"],
                "vlm_text": right["text"],
                "status": "unresolved",
            })

    low_confidence_regions = {
        str(item["region_id"])
        for item in ocr
        if float(item.get("confidence", 0.0)) < 0.5
        or item.get("legibility") in {"low", "unreadable"}
    }
    for hypothesis in hypotheses:
        refs = {str(ref) for ref in hypothesis.get("evidence_refs", [])}
        non_text_refs = {ref for ref in refs if not ref.startswith("ocr:")}
        text_regions = {ref.split(":", 1)[1] for ref in refs if ref.startswith("ocr:")}
        if (
            float(hypothesis.get("confidence", 0.0)) >= 0.8
            and text_regions
            and text_regions.issubset(low_confidence_regions)
            and not non_text_refs
        ):
            raise ValueError(
                "low-confidence OCR cannot be the only evidence for a high-confidence scene hypothesis"
            )
        if hypothesis.get("claim_type") not in {
            "directly_observed",
            "reasonable_inference",
            "external_common_knowledge",
            "uncertain",
        }:
            raise ValueError("scene hypothesis must use an explicit evidence-layer claim_type")

    return {
        "visual_evidence": visual,
        "ocr_candidates": ocr,
        "vlm_text_candidates": vlm,
        "scene_hypotheses": hypotheses,
        "conflicts": conflicts,
        "policy": {
            "low_confidence_text_only_high_confidence_scene_forbidden": True,
            "ocr_vlm_conflicts_preserved": True,
            "unreadable_text_may_abstain": True,
            "common_knowledge_not_image_fact": True,
        },
    }
