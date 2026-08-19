"""Read-only access to the frozen Gate 1-D3 Train library."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


NOT_AVAILABLE_V1_3 = "not_available_in_v1_3"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LibraryRepository:
    """Map the frozen manifest to a controlled server-side image directory."""

    def __init__(self, manifest_path: Path, dataset_root: Path, historical_result_root: Path, ocr_result_root: Path) -> None:
        records = _load_jsonl(manifest_path)
        if any(record.get("split") != "train" for record in records):
            raise ValueError("Phase 1 library accepts Train-only manifest records")
        ids = [record["relative_path"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate image IDs in manifest")
        self.manifest_path = manifest_path
        self.dataset_root = dataset_root
        self.historical_result_root = historical_result_root
        self.ocr_result_root = ocr_result_root
        self._records = records
        self._by_id = {record["relative_path"]: record for record in records}

    def _record(self, image_id: str) -> dict[str, Any]:
        if Path(image_id).name != image_id or image_id not in self._by_id:
            raise KeyError(image_id)
        return self._by_id[image_id]

    def image_path(self, image_id: str, *, verify_hash: bool = False) -> Path:
        """Execute the image path operation."""
        record = self._record(image_id)
        path = (self.dataset_root / image_id).resolve()
        if self.dataset_root.resolve() not in path.parents:
            raise ValueError("image path escaped dataset root")
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_hash and _sha256(path).lower() != record["sha256"].lower():
            raise ValueError(f"source hash mismatch for {image_id}")
        return path

    def list_assets(self) -> list[dict[str, Any]]:
        """List assets."""
        assets: list[dict[str, Any]] = []
        for record in self._records:
            image_id = record["relative_path"]
            assets.append(
                {
                    "image_id": image_id,
                    "width": record["width"],
                    "height": record["height"],
                    "format": record["format"],
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                    "split": "train",
                    "difficulty_labels": record.get("difficulty_labels", []),
                    "image_available": (self.dataset_root / image_id).is_file(),
                    "p3_v1_3_available": (self.historical_result_root / f"{Path(image_id).stem}.json").is_file(),
                    "ocr_available": (self.ocr_result_root / f"{Path(image_id).stem}.json").is_file(),
                }
            )
        return assets

    def historical_intelligence(self, image_id: str) -> dict[str, Any]:
        """Execute the historical intelligence operation."""
        self._record(image_id)
        path = self.historical_result_root / f"{Path(image_id).stem}.json"
        if not path.is_file():
            return {"status": "not_available", "source_version": "P3 v1.3", "image_id": image_id}
        raw = json.loads(path.read_text(encoding="utf-8"))
        payload = raw.get("parsed_payload") or {}
        return {
            "status": raw.get("status", "unknown"),
            "source_version": "P3 v1.3",
            "source_path": str(path),
            "image_id": image_id,
            "global_scene": payload.get("global_observation", NOT_AVAILABLE_V1_3),
            "visual_medium": NOT_AVAILABLE_V1_3,
            "depicted_content": NOT_AVAILABLE_V1_3,
            "main_subjects": payload.get("subjects", NOT_AVAILABLE_V1_3),
            "activities": payload.get("activities", NOT_AVAILABLE_V1_3),
            "attributes": payload.get("attributes", NOT_AVAILABLE_V1_3),
            "relations": payload.get("relations", NOT_AVAILABLE_V1_3),
            "visible_text": payload.get("visible_text_candidates", NOT_AVAILABLE_V1_3),
            "direct_observations": payload.get("evidence_descriptions", NOT_AVAILABLE_V1_3),
            "inferences": payload.get("scene_hypotheses", NOT_AVAILABLE_V1_3),
            "uncertainty": payload.get("uncertainties", NOT_AVAILABLE_V1_3),
            "short_caption": NOT_AVAILABLE_V1_3,
            "dense_caption": NOT_AVAILABLE_V1_3,
            "schema_valid": raw.get("schema_valid"),
            "historical_semantic_quality_status": raw.get("semantic_quality_status"),
        }

    def retrieval_text(self, image_id: str) -> str:
        """Execute the retrieval text operation."""
        result = self.historical_intelligence(image_id)
        fields = ("global_scene", "main_subjects", "activities", "attributes", "relations", "visible_text", "direct_observations", "inferences")
        values = [str(result[field]).strip() for field in fields if result.get(field) not in {None, "", NOT_AVAILABLE_V1_3}]
        return "；".join(dict.fromkeys(values))

    def ocr_evidence(self, image_id: str) -> dict[str, Any]:
        """Execute the ocr evidence operation."""
        self._record(image_id)
        path = self.ocr_result_root / f"{Path(image_id).stem}.json"
        if not path.is_file():
            return {"status": "not_available", "candidates": []}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            "status": raw.get("status", "unknown"),
            "truth_status": "candidate_evidence_not_ground_truth",
            "model": raw.get("model", {}),
            "candidates": [
                {
                    "region_id": item.get("region_id"),
                    "text": item.get("text"),
                    "confidence": item.get("confidence"),
                    "legibility": item.get("legibility"),
                }
                for item in raw.get("candidates", [])
            ],
        }
