"""Read-only system Train/Val libraries for the Phase 6.1 product surface.

The repository keeps course images in their original read-only dataset roots.
Only metadata and label references live in the project.  User-owned libraries
remain in :mod:`product_store` and are deliberately not merged into either
system split.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from scenemindx.annotation.canonical_closeout import build_two_layer_display
from PIL import Image


SYSTEM_LIBRARY_IDS = frozenset({"system_train", "system_val"})
SYSTEM_RESERVED_NAMES = frozenset(
    {
        "system_train",
        "system_val",
        "train",
        "val",
        "训练图片库",
        "验证图片库",
        "系统训练图片库",
        "系统验证图片库",
    }
)


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


class SystemVisualLibraryRepository:
    """Immutable catalog for the two course split libraries."""

    def __init__(
        self,
        *,
        project_root: Path,
        train_root: Path,
        val_root: Path,
        train_manifest: Path,
        val_manifest: Path,
        catalog_path: Path,
        thumbnail_root: Path | None = None,
        train_active_manifest: Path | None = None,
        val_active_manifest: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.roots = {
            "train": train_root.resolve(),
            "val": val_root.resolve(),
        }
        self.catalog_path = catalog_path.resolve()
        self.thumbnail_root = (
            thumbnail_root.resolve()
            if thumbnail_root
            else self.project_root / "data" / "cache" / "thumbnails" / "phase6_1"
        )
        self.manifest_paths = {
            "train": train_manifest.resolve(),
            "val": val_manifest.resolve(),
        }
        self.active_manifest_paths = {
            "train": train_active_manifest.resolve()
            if train_active_manifest
            else None,
            "val": val_active_manifest.resolve() if val_active_manifest else None,
        }
        self._libraries = self._load_catalog()
        self._assets = self._load_assets()
        self._labels = self._load_labels()

    def enabled(self) -> bool:
        """Execute the enabled operation."""
        return bool(self._assets)

    def _load_catalog(self) -> dict[str, dict[str, Any]]:
        if not self.catalog_path.is_file():
            return {}
        payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        libraries: dict[str, dict[str, Any]] = {}
        for row in payload.get("libraries", []):
            library_id = str(row.get("library_id"))
            if library_id not in SYSTEM_LIBRARY_IDS:
                raise ValueError(f"unexpected_system_library:{library_id}")
            if row.get("library_type") != "system_locked" or not row.get("locked"):
                raise ValueError(f"system_library_not_locked:{library_id}")
            libraries[library_id] = dict(row)
        return libraries

    def _load_assets(self) -> dict[str, dict[str, Any]]:
        assets: dict[str, dict[str, Any]] = {}
        for split, path in self.manifest_paths.items():
            expected_library_id = f"system_{split}"
            for row in _read_jsonl(path):
                asset_id = str(row.get("asset_id"))
                if row.get("source_split") != split:
                    raise ValueError(f"system_asset_split_mismatch:{asset_id}")
                if row.get("library_id") != expected_library_id:
                    raise ValueError(f"system_asset_library_mismatch:{asset_id}")
                if asset_id in assets:
                    raise ValueError(f"duplicate_system_asset_id:{asset_id}")
                assets[asset_id] = dict(row)
        return assets

    def _load_labels(self) -> dict[str, dict[str, Any]]:
        labels: dict[str, dict[str, Any]] = {}
        active_keys: set[str] = set()
        for split, path in self.active_manifest_paths.items():
            for row in _read_jsonl(path):
                asset_id = str(row.get("primary_asset_id") or row.get("asset_id"))
                asset_ids = [
                    str(value)
                    for value in (row.get("asset_ids") or [asset_id])
                ]
                active_key = str(row.get("active_key"))
                for linked_asset_id in asset_ids:
                    if linked_asset_id not in self._assets:
                        raise ValueError(
                            f"canonical_asset_not_registered:{linked_asset_id}"
                        )
                    if self._assets[linked_asset_id]["source_split"] != split:
                        raise ValueError(
                            f"canonical_split_mismatch:{linked_asset_id}"
                        )
                if active_key in active_keys:
                    raise ValueError(f"duplicate_active_canonical:{active_key}")
                active_keys.add(active_key)
                canonical_path = _resolve_project_path(
                    self.project_root,
                    str(row["canonical_evidence"]),
                )
                label = json.loads(canonical_path.read_text(encoding="utf-8"))
                if label.get("canonical_status") != "active_candidate":
                    raise ValueError(f"canonical_not_active:{asset_id}")
                if label.get("asset", {}).get("asset_id") != asset_id:
                    raise ValueError(f"canonical_asset_identity_mismatch:{asset_id}")
                for linked_asset_id in asset_ids:
                    if label.get("asset", {}).get("sha256") != self._assets[
                        linked_asset_id
                    ].get("image_sha256"):
                        raise ValueError(
                            f"canonical_asset_sha_mismatch:{linked_asset_id}"
                        )
                    labels[linked_asset_id] = {
                        "manifest": dict(row),
                        "path": canonical_path,
                        "label": label,
                    }
        return labels

    def libraries(self) -> list[dict[str, Any]]:
        """Execute the libraries operation."""
        counts = {
            library_id: sum(
                asset.get("library_id") == library_id for asset in self._assets.values()
            )
            for library_id in SYSTEM_LIBRARY_IDS
        }
        labeled = {
            library_id: sum(
                self._assets[asset_id].get("library_id") == library_id
                for asset_id in self._labels
            )
            for library_id in SYSTEM_LIBRARY_IDS
        }
        return [
            {
                **self._libraries[library_id],
                "asset_count": counts[library_id],
                "labeled_count": labeled[library_id],
                "permissions": {
                    "browse": True,
                    "select": True,
                    "upload": False,
                    "delete_asset": False,
                    "move_asset": False,
                    "rename_library": False,
                    "delete_library": False,
                    "replace_canonical": False,
                },
            }
            for library_id in ("system_train", "system_val")
            if library_id in self._libraries
        ]

    @staticmethod
    def _real_quality_warnings(label: dict[str, Any]) -> list[str]:
        processing = {
            "punctuation_normalized",
            "migrated_from_phase6_0a",
            "safe_caption_rebuilt",
            "safe_facts_rebuilt",
        }
        return [
            str(code)
            for code in label.get("quality", {}).get("warning_codes", [])
            if str(code) not in processing
        ]

    def public_asset(self, asset: dict[str, Any]) -> dict[str, Any]:
        """Execute the public asset operation."""
        asset_id = str(asset["asset_id"])
        label_entry = self._labels.get(asset_id)
        label = label_entry["label"] if label_entry else None
        result = {
            **asset,
            "sha256": asset.get("image_sha256"),
            "image_id": asset.get("original_filename"),
            "image_url": f"/visual-assets/{asset_id}/image",
            "thumbnail_url": f"/visual-assets/{asset_id}/thumbnail",
            "detail_url": f"/visual-assets/{asset_id}",
            "locked": True,
            "library_type": "system_locked",
            "selectable": True,
            "label_status": (
                label.get("review", {}).get("review_label", "machine_provisional")
                if label
                else "pending"
            ),
            "canonical_status": label.get("canonical_status") if label else "pending",
            "needs_review": bool(
                label and label.get("quality", {}).get("needs_review")
            ),
            "automatic_validation_status": (
                label.get("review", {})
                .get("automatic_validation", {})
                .get("status")
                if label
                else "pending"
            ),
            "codex_visual_review_status": (
                label.get("review", {})
                .get("codex_visual_review", {})
                .get("status")
                if label
                else "not_reviewed"
            ),
            "user_human_review_status": (
                label.get("review", {})
                .get("user_human_review", {})
                .get("status")
                if label
                else "pending"
            ),
            "quality_warnings": self._real_quality_warnings(label) if label else [],
            "two_layer": None,
        }
        if label and label_entry:
            result["canonical_label_id"] = label.get("label_id")
            result["two_layer"] = build_two_layer_display(
                label,
                developer={
                    "canonical_path": label_entry["path"]
                    .relative_to(self.project_root)
                    .as_posix(),
                    "trace_path": label_entry["manifest"].get("trace_path"),
                    "asset_id": asset_id,
                    "sha256": asset.get("image_sha256"),
                },
            )
        return result

    def asset(self, asset_id: str) -> dict[str, Any]:
        """Execute the asset operation."""
        if asset_id not in self._assets:
            raise KeyError(asset_id)
        return self.public_asset(self._assets[asset_id])

    def model_context(self, asset_id: str) -> dict[str, Any]:
        """Return safe structured evidence without making labels the primary VLM."""
        if asset_id not in self._assets:
            raise KeyError(asset_id)
        entry = self._labels.get(asset_id)
        if entry is None:
            return {
                "facts": {},
                "ocr": {
                    "status": "not_available",
                    "truth_status": "image_only_unverified_text",
                    "candidates": [],
                },
            }
        label = entry["label"]
        text = label.get("text_evidence", {})
        return {
            "facts": dict(label.get("facts") or {}),
            "ocr": {
                "status": (
                    "available"
                    if text.get("ocr_candidates")
                    else "not_available"
                ),
                "truth_status": str(
                    text.get("verification_status")
                    or "image_only_unverified_text"
                ),
                "candidates": list(text.get("ocr_candidates") or []),
            },
        }

    def raw_asset(self, asset_id: str) -> dict[str, Any]:
        """Execute the raw asset operation."""
        if asset_id not in self._assets:
            raise KeyError(asset_id)
        return dict(self._assets[asset_id])

    def image_path(self, asset_id: str, *, verify_hash: bool = False) -> Path:
        """Execute the image path operation."""
        asset = self.raw_asset(asset_id)
        split = str(asset["source_split"])
        filename = Path(str(asset["original_filename"])).name
        if filename != asset["original_filename"]:
            raise KeyError(asset_id)
        root = self.roots[split]
        path = (root / filename).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        if verify_hash:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != asset["image_sha256"]:
                raise ValueError(f"system_asset_sha256_mismatch:{asset_id}")
        return path

    def thumbnail_path(self, asset_id: str, *, maximum: int = 320) -> Path:
        """Execute the thumbnail path operation."""
        if maximum < 64 or maximum > 1024:
            raise ValueError("invalid_thumbnail_size")
        asset = self.raw_asset(asset_id)
        split = str(asset["source_split"])
        destination = (
            self.thumbnail_root
            / split
            / f"{asset['image_sha256']}.{maximum}.webp"
        )
        if destination.is_file():
            return destination
        source = self.image_path(asset_id, verify_hash=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(
            destination.suffix + f".{os.getpid()}.tmp"
        )
        try:
            with Image.open(source) as image:
                image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert("RGB")
                image.save(temporary, format="WEBP", quality=82, method=4)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    @staticmethod
    def _search_text(item: dict[str, Any]) -> str:
        display = item.get("two_layer") or {}
        ordinary = display.get("default") or {}
        return " ".join(
            [
                str(item.get("asset_id", "")),
                str(item.get("original_filename", "")),
                str(ordinary.get("主题", "")),
                str(ordinary.get("简短描述", "")),
                " ".join(str(tag) for tag in ordinary.get("微标签", [])),
            ]
        ).casefold()

    def query(
        self,
        library_id: str,
        *,
        page: int = 1,
        page_size: int = 40,
        q: str | None = None,
        theme: str | None = None,
        micro_tag: str | None = None,
        label_status: str | None = None,
        review_status: str | None = None,
        needs_review: bool | None = None,
        sort: str = "sequence_asc",
    ) -> dict[str, Any]:
        """Execute the query operation."""
        if library_id not in SYSTEM_LIBRARY_IDS or library_id not in self._libraries:
            raise KeyError(library_id)
        if page < 1 or page_size < 1 or page_size > 100:
            raise ValueError("invalid_pagination")
        items = [
            self.public_asset(asset)
            for asset in self._assets.values()
            if asset.get("library_id") == library_id
        ]
        if q:
            needle = q.strip().casefold()
            items = [item for item in items if needle in self._search_text(item)]
        if theme:
            items = [
                item
                for item in items
                if theme.casefold()
                in str(
                    ((item.get("two_layer") or {}).get("default") or {}).get(
                        "主题", ""
                    )
                ).casefold()
            ]
        if micro_tag:
            items = [
                item
                for item in items
                if any(
                    micro_tag.casefold() in str(tag).casefold()
                    for tag in (
                        ((item.get("two_layer") or {}).get("default") or {}).get(
                            "微标签", []
                        )
                    )
                )
            ]
        if label_status:
            items = [
                item for item in items if item.get("label_status") == label_status
            ]
        if review_status:
            items = [
                item
                for item in items
                if item.get("user_human_review_status") == review_status
            ]
        if needs_review is not None:
            items = [
                item
                for item in items
                if bool(item.get("needs_review")) is needs_review
            ]
        reverse = sort.endswith("_desc")
        if sort.startswith("filename_"):
            key = lambda item: (
                0,
                int(Path(str(item["original_filename"])).stem),
            ) if Path(str(item["original_filename"])).stem.isdigit() else (
                1,
                str(item["original_filename"]).casefold(),
            )
        elif sort.startswith("updated_"):
            key = lambda item: str(item.get("registered_at", ""))
        else:
            key = lambda item: int(item.get("sequence", 0))
        items.sort(key=key, reverse=reverse)
        total = len(items)
        start = (page - 1) * page_size
        return {
            "library_id": library_id,
            "page": page,
            "page_size": page_size,
            "total": total,
            "page_count": (total + page_size - 1) // page_size,
            "items": items[start : start + page_size],
        }

    def iter_index_records(self, library_id: str) -> Iterable[dict[str, Any]]:
        """Execute the iter index records operation."""
        if library_id not in SYSTEM_LIBRARY_IDS:
            raise KeyError(library_id)
        for asset in self._assets.values():
            if asset.get("library_id") == library_id:
                yield self.public_asset(asset)
