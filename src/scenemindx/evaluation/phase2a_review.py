"""Phase 2A RC1 human-review records and atomic persistence."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE2A_PREFERENCES = {
    "rc1_better",
    "baseline_better",
    "equivalent",
    "both_bad",
    "not_comparable",
}
PHASE2A_RC2_PREFERENCES = {
    "rc2_better",
    "rc1_better",
    "equivalent",
    "both_bad",
    "not_comparable",
}
PHASE2A_RC4_PREFERENCES = {
    "prefer_rc3",
    "prefer_rc4",
    "approximately_equal",
    "both_bad",
}

PHASE2A_REVIEW_GUIDANCE = (
    "比较整体场景、主要/次要主体、行为、属性和空间关系是否准确完整。",
    "判断图像本身的媒介类型是否正确，并检查媒介与其内部描绘内容是否分层。",
    "检查可见文字、OCR假阳性、不可辨文字拒答、文字方向及文字是否错误传播到场景结论。",
    "检查claims是否有对应evidence，直接观察、合理推断和不确定性是否边界清楚。",
    "记录幻觉、遗漏、重复、截断、错字以及不自然的中英文混写。",
    "综合判断其是否能支持检索、事实VQA、拒答、受事实约束的内容生成和训练数据构造。",
)


def load_json(path: Path) -> dict[str, Any]:
    """Load json."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load jsonl."""
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"expected JSONL objects: {path}")
    return records


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write json atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """Write jsonl atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def new_phase2a_review_record(image_id: str) -> dict[str, Any]:
    """Execute the new phase2a review record operation."""
    return {
        "image_id": image_id,
        "preferred_version": None,
        "notes": "",
        "reviewer": None,
        "reviewed_at": None,
        "human_confirmed": False,
        "score_source": "human_draft",
    }


def new_phase2a_rc2_review_record(image_id: str) -> dict[str, Any]:
    """Execute the new phase2a rc2 review record operation."""
    return {
        "image_id": image_id,
        "preferred_version": None,
        "notes": "",
        "reviewer": None,
        "reviewed_at": None,
        "human_confirmed": False,
        "score_source": "human_draft",
    }


def validate_phase2a_review_payload(payload: Any, image_ids: set[str]) -> list[dict[str, Any]]:
    """Validate phase2a review payload."""
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("request must contain a records array")
    records = payload["records"]
    record_ids = {record.get("image_id") for record in records if isinstance(record, dict)}
    if len(records) != len(image_ids) or record_ids != image_ids:
        raise ValueError("review records must match the frozen 10-image manifest exactly")
    reviewer = str(payload.get("reviewer") or "").strip() or None
    confirmed = payload.get("confirm_all") is True
    if confirmed and reviewer is None:
        raise ValueError("reviewer is required before final confirmation")
    reviewed_at = datetime.now().astimezone().isoformat() if confirmed else None
    validated = []
    for source in records:
        if not isinstance(source, dict):
            raise ValueError("each review record must be an object")
        image_id = source["image_id"]
        preference = source.get("preferred_version")
        if preference is not None and preference not in PHASE2A_PREFERENCES:
            raise ValueError(f"{image_id}: invalid preferred_version")
        if confirmed and preference is None:
            raise ValueError(f"{image_id}: preferred_version is required before final confirmation")
        validated.append(
            {
                "image_id": image_id,
                "preferred_version": preference,
                "notes": str(source.get("notes") or ""),
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "human_confirmed": confirmed,
                "score_source": "human_confirmed" if confirmed else "human_draft",
            }
        )
    return validated


def validate_phase2a_rc2_review_payload(payload: Any, image_ids: set[str]) -> list[dict[str, Any]]:
    """Validate phase2a rc2 review payload."""
    if len(image_ids) != 4:
        raise ValueError("RC2 smoke review requires exactly four image IDs")
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("request must contain a records array")
    records = payload["records"]
    record_ids = {record.get("image_id") for record in records if isinstance(record, dict)}
    if len(records) != 4 or record_ids != image_ids:
        raise ValueError("review records must match the frozen four-image RC2 manifest exactly")
    reviewer = str(payload.get("reviewer") or "").strip() or None
    confirmed = payload.get("confirm_all") is True
    if confirmed and reviewer is None:
        raise ValueError("reviewer is required before final confirmation")
    reviewed_at = datetime.now().astimezone().isoformat() if confirmed else None
    validated = []
    for source in records:
        if not isinstance(source, dict):
            raise ValueError("each review record must be an object")
        image_id = source["image_id"]
        preference = source.get("preferred_version")
        if preference is not None and preference not in PHASE2A_RC2_PREFERENCES:
            raise ValueError(f"{image_id}: invalid preferred_version")
        if confirmed and preference is None:
            raise ValueError(f"{image_id}: preferred_version is required before final confirmation")
        validated.append(
            {
                "image_id": image_id,
                "preferred_version": preference,
                "notes": str(source.get("notes") or ""),
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "human_confirmed": confirmed,
                "score_source": "human_confirmed" if confirmed else "human_draft",
            }
        )
    return validated


def validate_phase2a_rc4_review_payload(payload: Any, image_ids: set[str]) -> list[dict[str, Any]]:
    """Validate phase2a rc4 review payload."""
    if len(image_ids) != 4:
        raise ValueError("RC4 smoke review requires exactly four image IDs")
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("request must contain a records array")
    records = payload["records"]
    record_ids = {record.get("image_id") for record in records if isinstance(record, dict)}
    if len(records) != 4 or record_ids != image_ids:
        raise ValueError("review records must match the frozen four-image RC4 manifest exactly")
    reviewer = str(payload.get("reviewer") or "").strip() or None
    confirmed = payload.get("confirm_all") is True
    if confirmed and reviewer is None:
        raise ValueError("reviewer is required before final confirmation")
    reviewed_at = datetime.now().astimezone().isoformat() if confirmed else None
    validated = []
    for source in records:
        if not isinstance(source, dict):
            raise ValueError("each review record must be an object")
        image_id = source["image_id"]
        preference = source.get("preferred_version")
        if preference is not None and preference not in PHASE2A_RC4_PREFERENCES:
            raise ValueError(f"{image_id}: invalid preferred_version")
        if confirmed and preference is None:
            raise ValueError(f"{image_id}: preferred_version is required before final confirmation")
        validated.append(
            {
                "image_id": image_id,
                "preferred_version": preference,
                "notes": str(source.get("notes") or ""),
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
                "human_confirmed": confirmed,
                "score_source": "human_confirmed" if confirmed else "human_draft",
            }
        )
    return validated
