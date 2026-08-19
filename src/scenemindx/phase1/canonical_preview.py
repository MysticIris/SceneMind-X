"""Expose read-only Canonical previews to the product API."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scenemindx.annotation.canonical_closeout import build_two_layer_display


class CanonicalPreviewRepository:
    """Read-only Phase 6 candidate preview, isolated from production asset routing."""

    def __init__(
        self,
        project_root: Path,
        dataset_root: Path,
        manifest_path: Path | None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.dataset_root = dataset_root.resolve()
        self.manifest_path = manifest_path.resolve() if manifest_path else None
        self._items = self._load()

    def _resolve_project_path(self, value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def _load(self) -> dict[str, dict[str, Any]]:
        if self.manifest_path is None or not self.manifest_path.is_file():
            return {}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        items: dict[str, dict[str, Any]] = {}
        for raw in payload.get("items", []):
            image_id = Path(str(raw["image_id"])).name
            if image_id != str(raw["image_id"]):
                raise ValueError(f"invalid_canonical_preview_image_id:{raw['image_id']}")
            canonical_path = self._resolve_project_path(str(raw["canonical_path"]))
            label = json.loads(canonical_path.read_text(encoding="utf-8"))
            if label.get("canonical_status") != "active_candidate":
                raise ValueError(f"preview_label_not_active:{image_id}")
            if Path(label["asset"]["relative_path"]).name != image_id:
                raise ValueError(f"preview_label_image_mismatch:{image_id}")
            items[image_id] = {
                "position": int(raw["position"]),
                "image_id": image_id,
                "asset_sha256": str(label["asset"]["sha256"]),
                "selection_slices": list(raw.get("selection_slices") or []),
                "canonical_path": canonical_path,
                "label": label,
                "source_mode": str(raw.get("source_mode") or "unknown"),
            }
        return items

    def enabled(self) -> bool:
        """Execute the enabled operation."""
        return bool(self._items)

    def list_items(self) -> list[dict[str, Any]]:
        """List items."""
        return [self.public_item(item) for item in sorted(self._items.values(), key=lambda value: value["position"])]

    def public_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Execute the public item operation."""
        display = build_two_layer_display(
            item["label"],
            developer={
                "canonical_path": item["canonical_path"].relative_to(self.project_root).as_posix(),
                "source_mode": item["source_mode"],
            },
        )
        return {
            "position": item["position"],
            "image_id": item["image_id"],
            "image_url": f"/canonical-preview/{item['image_id']}/image",
            "selection_slices": item["selection_slices"],
            "two_layer": display,
            "user_human_review_status": "pending",
            "is_human_gold": False,
            "context_selectable": False,
        }

    def item(self, image_id: str) -> dict[str, Any]:
        """Execute the item operation."""
        key = Path(image_id).name
        if key != image_id or key not in self._items:
            raise KeyError(image_id)
        return self.public_item(self._items[key])

    def image_path(self, image_id: str) -> Path:
        """Execute the image path operation."""
        key = Path(image_id).name
        if key != image_id or key not in self._items:
            raise KeyError(image_id)
        item = self._items[key]
        path = (self.dataset_root / key).resolve()
        if self.dataset_root not in path.parents:
            raise KeyError(image_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["asset_sha256"]:
            raise ValueError(f"canonical_preview_sha256_mismatch:{image_id}")
        return path
