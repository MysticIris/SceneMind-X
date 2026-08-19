"""Evidence-aware text profiles for Phase 5.3 hybrid image retrieval."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping


P3_RETRIEVAL_FIELDS = (
    "global_observation",
    "subjects",
    "activities",
    "attributes",
    "relations",
    "scene_hypotheses",
    "evidence_descriptions",
)

NO_VISIBLE_TEXT_VALUES = {
    "",
    "无",
    "无文字",
    "无可辨识文字",
    "无可识别文字",
    "未见文字",
    "不可辨文字",
}


def canonical_json(value: Any) -> str:
    """Execute the canonical json operation."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    """Execute the sha256 text operation."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    """Normalize text."""
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).strip()


def _deduplicated_text(values: Iterable[Any]) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for raw in values:
        value = normalize_text(raw)
        if not value or value in seen:
            continue
        seen.add(value)
        parts.append(value)
    return "；".join(parts)


def _base_profile(manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    image_id = str(manifest_row["image_id"])
    return {
        "schema_version": "phase5_3_text_profile_v1",
        "asset_id": image_id,
        "image_id": image_id,
        "split": "train",
        "asset_sha256": str(manifest_row["sha256"]),
        "selection_origin": str(manifest_row["selection_origin"]),
        "selection_slice": str(manifest_row["selection_slice"]),
        "p3_facts": {},
        "caption": {
            "text": "",
            "source": "not_available",
            "status": "not_available",
        },
        "description": {
            "text": "",
            "source": "not_available",
            "status": "not_available",
        },
        "ocr_candidate": {
            "text": "",
            "source": "not_available",
            "status": "not_available",
            "included_in_retrieval_text": False,
        },
        "verified_text": {
            "items": [],
            "status": "not_available_before_phase5_3_stage5",
            "included_in_retrieval_text": False,
        },
        "scene": "",
        "subjects": "",
        "actions": "",
        "relations": "",
        "media_type": "unknown",
        "metadata": {
            "relative_path": str(manifest_row["relative_path"]),
            "width": int(manifest_row["width"]),
            "height": int(manifest_row["height"]),
            "size_bytes": int(manifest_row["size_bytes"]),
        },
    }


def profile_from_phase4b(
    manifest_row: Mapping[str, Any],
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a profile while keeping unverified visible text out of retrieval text."""

    if item.get("status") != "completed":
        raise ValueError(f"incomplete Phase 4B item: {manifest_row['image_id']}")
    analyze = item["analyze"]
    describe = item["describe"]
    facts = dict(analyze["result"]["data"]["normalized_output"])
    description = normalize_text(describe["result"]["data"]["final_output"])
    p3_facts = {
        field: normalize_text(facts.get(field, ""))
        for field in P3_RETRIEVAL_FIELDS
    }
    candidate = normalize_text(facts.get("visible_text_candidates", ""))
    if candidate in NO_VISIBLE_TEXT_VALUES:
        candidate = ""

    profile = _base_profile(manifest_row)
    profile.update(
        {
            "profile_source": "phase4b_structured_p3_v1_4_plus_description_v1",
            "profile_quality": "machine_provisional",
            "p3_facts": p3_facts,
            "caption": {
                "text": p3_facts["global_observation"],
                "source": "p3_v1_4.global_observation",
                "status": "machine_provisional",
            },
            "description": {
                "text": description,
                "source": "natural_chinese_detailed_description_v1",
                "status": "machine_provisional",
            },
            "ocr_candidate": {
                "text": candidate,
                "source": "p3_v1_4.visible_text_candidates",
                "status": "unverified_candidate" if candidate else "not_available",
                "included_in_retrieval_text": False,
            },
            "scene": p3_facts["scene_hypotheses"],
            "subjects": p3_facts["subjects"],
            "actions": p3_facts["activities"],
            "relations": p3_facts["relations"],
            "provenance": {
                "p3_prompt_id": analyze["core_prompt"]["prompt_id"],
                "p3_prompt_sha256": analyze["core_prompt"]["prompt_sha256"],
                "description_prompt_id": describe["result"]["data"]["prompt_id"],
                "description_prompt_sha256": describe["result"]["data"]["prompt_sha256"],
                "item_payload_sha256": item["item_payload_sha256"],
            },
        }
    )
    profile_text = _deduplicated_text(
        [*(p3_facts[field] for field in P3_RETRIEVAL_FIELDS), description]
    )
    profile["retrieval_text"] = profile_text
    profile["retrieval_text_sha256"] = sha256_text(profile_text)
    profile["profile_sha256"] = sha256_text(canonical_json(profile))
    return profile


def profile_from_legacy_product(
    manifest_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the frozen Product-16 text without claiming structured provenance."""

    legacy_text = normalize_text(manifest_row.get("retrieval_text", ""))
    if not legacy_text:
        raise ValueError(f"legacy product text is empty: {manifest_row['image_id']}")
    profile = _base_profile(manifest_row)
    profile.update(
        {
            "profile_source": "phase5_2a_frozen_product_index",
            "profile_quality": "machine_provisional_unstructured",
            "caption": {
                "text": legacy_text,
                "source": "phase5_2a_frozen_product_index.retrieval_text",
                "status": "machine_provisional_unstructured",
            },
            "provenance": dict(manifest_row["text_provenance"]),
            "retrieval_text": legacy_text,
            "retrieval_text_sha256": sha256_text(legacy_text),
        }
    )
    profile["profile_sha256"] = sha256_text(canonical_json(profile))
    return profile


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load jsonl."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
