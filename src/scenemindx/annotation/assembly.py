"""Deterministic semantic-payload to annotation-v1.1 assembly."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .schema import AnnotationValidationError, validate_annotation
from .semantic import validate_semantic_payload


class SemanticPayloadError(ValueError):
    """Raised when a semantic payload cannot be assembled without inventing facts."""


def _ref(value: str, *, subject_count: int, text_count: int) -> str:
    if value.startswith("subject:"):
        index = int(value.split(":", 1)[1])
        if not 0 <= index < subject_count:
            raise SemanticPayloadError(f"dangling subject evidence reference: {value}")
        return f"entity:e{index + 1}"
    if value.startswith("text:"):
        index = int(value.split(":", 1)[1])
        if not 0 <= index < text_count:
            raise SemanticPayloadError(f"dangling text evidence reference: {value}")
        return f"ocr:t{index + 1}"
    if value.startswith("claim:"):
        return value
    return value


def _refs(values: list[str], *, subject_count: int, text_count: int) -> list[str]:
    return sorted({_ref(str(value), subject_count=subject_count, text_count=text_count) for value in values})


def _position(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SemanticPayloadError("position/evidence_region must be an object or null")
    bbox = value.get("bbox")
    if bbox is not None:
        if not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(x, (int, float)) for x in bbox):
            raise SemanticPayloadError("bbox must be a four-number normalized list")
        if any(float(x) < 0 or float(x) > 1 for x in bbox):
            raise SemanticPayloadError("bbox values must be normalized to [0, 1]")
        bbox = [float(x) for x in bbox]
    return {"description": value.get("description"), "bbox": bbox}


def assemble_semantic_payload(payload: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """Build v1.1 from a small semantic payload and validate it formally.

    The assembler only copies model-provided facts, assigns stable IDs, and
    inserts contract-required empty/null values. It never creates a visual
    subject, OCR string, relation, or scene hypothesis from a default.
    """

    if not isinstance(payload, Mapping) or not isinstance(context, Mapping):
        raise SemanticPayloadError("payload and context must be mappings")
    try:
        validate_semantic_payload(payload)
    except AnnotationValidationError as exc:
        raise SemanticPayloadError(str(exc)) from exc
    global_observation = payload["global_observation"]
    subjects = list(payload["subjects"])
    text_candidates = list(payload["visible_text_candidates"])
    subject_count = len(subjects)
    text_count = len(text_candidates)

    entities: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects, 1):
        entities.append({
            "entity_id": f"e{index}",
            "category": subject["category"],
            "attributes": deepcopy(subject["attributes"]),
            "approximate_count": subject["approximate_count"],
            "position": _position(subject["position"]),
            "salience": subject["salience"],
            "confidence": subject["confidence"],
            "visual_presence_type": subject["visual_presence_type"],
            "presence_confidence": subject["presence_confidence"],
            "alternative_presence_types": list(subject.get("alternative_presence_types", [])),
            "presence_evidence_refs": _refs(subject["presence_evidence_refs"], subject_count=subject_count, text_count=text_count),
        })

    relationships: list[dict[str, Any]] = []
    for relation in payload["relations"]:
        subject_index = int(relation["subject_index"])
        object_index = int(relation["object_index"])
        if not 0 <= subject_index < subject_count or not 0 <= object_index < subject_count:
            raise SemanticPayloadError("relation subject_index/object_index is out of range")
        relationships.append({
            "subject": f"e{subject_index + 1}",
            "predicate": relation["predicate"],
            "object": f"e{object_index + 1}",
            "spatial_relation": relation["spatial_relation"],
            "confidence": relation["confidence"],
            "evidence_refs": _refs(relation["evidence_refs"], subject_count=subject_count, text_count=text_count),
        })

    ocr: list[dict[str, Any]] = []
    for index, candidate in enumerate(text_candidates, 1):
        legibility = candidate["legibility"]
        selected_text = candidate["selected_text"]
        if legibility in {"low", "unreadable"} and selected_text is not None:
            raise SemanticPayloadError("low/unreadable OCR must allow selected_text=null")
        ocr.append({
            "text_region_id": f"t{index}",
            "language": candidate["language"],
            "script_type": candidate["script_type"],
            "legibility": legibility,
            "ocr_candidates": list(candidate["ocr_candidates"]),
            "selected_text": selected_text,
            "confidence": candidate["confidence"],
            "used_for_scene_inference": candidate["used_for_scene_inference"],
            "uncertainty_reason": candidate["uncertainty_reason"],
            "evidence_region": _position(candidate["evidence_region"]),
        })

    claims: list[dict[str, Any]] = []
    evidence = {"directly_observed": [], "reasonable_inference": [], "external_common_knowledge": []}
    for index, item in enumerate(payload["evidence_descriptions"], 1):
        refs = _refs(item["evidence_refs"], subject_count=subject_count, text_count=text_count)
        claim = {
            "claim_id": f"c{index}",
            "claim": item["claim"],
            "claim_type": item["claim_type"],
            "evidence_refs": refs,
            "confidence": item["confidence"],
            "conflicts": [],
            "status": "accepted" if item["claim_type"] == "directly_observed" else "uncertain",
        }
        claims.append(claim)
        if item["claim_type"] in evidence:
            evidence[item["claim_type"]].append({"claim": item["claim"], "confidence": item["confidence"], "evidence_refs": refs})

    result = {
        "schema_version": "visual_asset_annotation_v1_1",
        "image_id": context["image_id"],
        "source_split": context["source_split"],
        "source_hash": context["source_hash"],
        "model": deepcopy(context["model"]),
        "prompt_version": context["prompt_version"],
        "generation_config": deepcopy(context["generation_config"]),
        "created_at": context["created_at"],
        "concise_caption": global_observation["concise_caption"],
        "detailed_caption": global_observation["detailed_caption"],
        "scene_type": global_observation["scene_type"],
        "environment": global_observation["environment"],
        "main_subjects": [subject["category"] for subject in subjects],
        "activities": list(payload["activities"]),
        "visual_style": global_observation["visual_style"],
        "image_quality": deepcopy(global_observation["image_quality"]),
        "image_characteristics": deepcopy(context.get("image_characteristics", {"capture_style": "unknown", "scene_complexity": "unknown", "text_density": "unknown", "object_clutter": "unknown", "small_object_level": "unknown", "occlusion_level": "unknown", "motion_blur": False, "defocus_blur": False, "reflection": False, "glare": False, "ghosting": False, "low_light": False, "compression_artifact": False})),
        "entities": entities,
        "relationships": relationships,
        "ocr": ocr,
        "evidence": evidence,
        "claims": claims,
        "scene_hypotheses": deepcopy(payload["scene_hypotheses"]),
        "uncertainty": deepcopy(payload["uncertainties"]),
        "downstream": {"retrieval_tags": [], "candidate_questions": [], "candidate_answers": [], "evidence_refs": [], "safety_or_privacy_flags": []},
        "provenance": {
            "assembly_version": "semantic_payload_assembly_v1",
            "field_sources": {
                "global_observation": "semantic_payload.model",
                "entities": "semantic_payload.subjects + deterministic_ids",
                "relationships": "semantic_payload.relations + deterministic_ids",
                "ocr": "semantic_payload.visible_text_candidates + deterministic_ids",
                "claims": "semantic_payload.evidence_descriptions + deterministic_ids",
                "image_characteristics": "context.image_characteristics_or_unknown_defaults",
                "downstream": "assembly.empty_defaults",
            },
        },
    }
    try:
        from .schema_v1_1 import validate_annotation_v1_1

        validate_annotation_v1_1(result)
    except ImportError as exc:
        raise AnnotationValidationError("v1.1 validator unavailable") from exc
    return result
