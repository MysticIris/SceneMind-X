"""Explicit v1 -> v1.1 migration with conservative defaults."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .schema_v1_1 import validate_annotation_v1_1


def migrate_v1_to_v1_1(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate a frozen v1 record without inventing visual facts.

    The original v1 record is never mutated. New presence/complexity fields
    become explicit unknowns, and legacy OCR text becomes a high-level printed
    candidate only when the old record supplied non-empty text.
    """

    result: dict[str, Any] = deepcopy(dict(annotation))
    result["schema_version"] = "visual_asset_annotation_v1_1"
    result["image_characteristics"] = {
        "capture_style": "unknown", "scene_complexity": "unknown", "text_density": "unknown",
        "object_clutter": "unknown", "small_object_level": "unknown", "occlusion_level": "unknown",
        "motion_blur": False, "defocus_blur": False, "reflection": False, "glare": False,
        "ghosting": False, "low_light": False, "compression_artifact": False,
    }
    for entity in result.get("entities", []):
        entity.update({
            "visual_presence_type": "direct_object", "presence_confidence": entity.get("confidence", 0),
            "alternative_presence_types": [], "presence_evidence_refs": [f"entity:{entity['entity_id']}"],
        })
    migrated_ocr: list[dict[str, Any]] = []
    for index, item in enumerate(result.get("ocr", []), 1):
        selected = item.get("text")
        migrated_ocr.append({
            "text_region_id": f"t{index}", "language": item.get("language"), "script_type": "unknown",
            "legibility": "medium" if selected else "unreadable", "ocr_candidates": [selected] if selected else [],
            "selected_text": selected, "confidence": item.get("confidence", 0), "used_for_scene_inference": False,
            "uncertainty_reason": None if selected else "legacy v1 record had no readable text", "evidence_region": item.get("position"),
        })
    result["ocr"] = migrated_ocr
    result["claims"] = []
    for index, group in enumerate(("directly_observed", "reasonable_inference", "external_common_knowledge"), 1):
        for item in result.get("evidence", {}).get(group, []):
            result["claims"].append({
                "claim_id": f"c{len(result['claims']) + 1}", "claim": item["claim"], "claim_type": group,
                "evidence_refs": item.get("evidence_refs", []), "confidence": item.get("confidence", 0),
                "conflicts": [], "status": "accepted" if group == "directly_observed" else "uncertain",
            })
    result["scene_hypotheses"] = []
    result["provenance"] = {
        "assembly_version": "v1_to_v1_1_migration_v1",
        "field_sources": {"legacy_v1": "copied", "image_characteristics": "migration.unknown_defaults", "claims": "legacy_evidence", "ocr": "legacy_ocr"},
    }
    validate_annotation_v1_1(result)
    return result
