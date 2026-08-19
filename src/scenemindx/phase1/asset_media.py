"""Authoritative public media resolution for retrieval and Chat result cards."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import quote


class AssetMediaResolver:
    """Resolve stable asset identities to safe HTTP media contracts.

    Filesystem paths stay server-side.  Retrieval labels such as SEARCH_1 are
    deliberately not accepted as asset identities.
    """

    def __init__(
        self,
        *,
        library: Any,
        system_libraries: Any,
        product: Any,
        lifecycle_registry: Any | None = None,
    ) -> None:
        self.library = library
        self.system_libraries = system_libraries
        self.product = product
        self.lifecycle_registry = lifecycle_registry

    def _lifecycle(self, value: dict[str, Any]) -> dict[str, Any]:
        if self.lifecycle_registry is None:
            return {
                "lifecycle_state": "active",
                "searchable": True,
                "lifecycle_label": "活动资产",
            }
        return dict(self.lifecycle_registry.lookup(value))

    @staticmethod
    def _urls(asset_id: str) -> tuple[str, str]:
        encoded = quote(asset_id, safe="")
        return (
            f"/assets/{encoded}/thumbnail",
            f"/assets/{encoded}/content",
        )

    def _candidate_ids(self, value: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        for raw in (value.get("asset_id"), value.get("image_id")):
            if raw is None:
                continue
            candidate = str(raw)
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _safe_display_filename(
        *values: Any,
        fallback_id: str,
    ) -> str:
        """Project an existing authoritative name without exposing a path."""

        for raw in values:
            text = str(raw or "").strip()
            if not text:
                continue
            basename = text.replace("\\", "/").rsplit("/", 1)[-1].strip()
            if basename not in {"", ".", ".."}:
                return basename
        stable = str(fallback_id or "unknown_asset").strip()
        safe = stable.replace("\\", "/").rsplit("/", 1)[-1].strip()
        return safe[:48] or "unknown_asset"

    def resolve_identity(self, value: dict[str, Any]) -> dict[str, Any]:
        """Resolve identity."""
        for candidate in self._candidate_ids(value):
            if candidate.startswith("local:"):
                local_id = candidate.removeprefix("local:")
                try:
                    item = self.product.asset(local_id)
                except KeyError:
                    continue
                thumbnail_url, content_url = self._urls(candidate)
                return {
                    "asset_id": candidate,
                    "source_asset_id": local_id,
                    "source": "local",
                    "source_type": "registered_persistent",
                    "library_id": str(item.get("library_id") or "default"),
                    "display_name": self._safe_display_filename(
                        item.get("original_filename"),
                        item.get("display_name"),
                        item.get("filename"),
                        item.get("image_id"),
                        fallback_id=local_id,
                    ),
                    "thumbnail_url": thumbnail_url,
                    "content_url": content_url,
                    "image_url": content_url,
                    "media_status": "ready",
                }
            try:
                item = self.system_libraries.asset(candidate)
                thumbnail_url, content_url = self._urls(candidate)
                return {
                    "asset_id": candidate,
                    "source_asset_id": candidate,
                    "source": "system",
                    "source_type": "system_read_only",
                    "library_id": str(item.get("library_id") or ""),
                    "display_name": self._safe_display_filename(
                        item.get("original_filename"),
                        item.get("display_name"),
                        item.get("filename"),
                        item.get("image_id"),
                        fallback_id=candidate,
                    ),
                    "thumbnail_url": thumbnail_url,
                    "content_url": content_url,
                    "image_url": content_url,
                    "media_status": "ready",
                }
            except KeyError:
                pass
            try:
                self.library.image_path(candidate)
                thumbnail_url, content_url = self._urls(candidate)
                lifecycle = self._lifecycle(
                    {
                        "asset_id": candidate,
                        "image_id": candidate,
                        "source": "frozen_library",
                    }
                )
                return {
                    "asset_id": candidate,
                    "source_asset_id": candidate,
                    "source": "library",
                    "source_type": "frozen_library",
                    "library_id": "legacy_frozen",
                    "display_name": self._safe_display_filename(
                        value.get("original_filename"),
                        value.get("display_name"),
                        value.get("filename"),
                        candidate,
                        fallback_id=candidate,
                    ),
                    "thumbnail_url": thumbnail_url,
                    "content_url": content_url,
                    "image_url": content_url,
                    "media_status": "ready",
                    "lifecycle_state": lifecycle["lifecycle_state"],
                    "searchable": lifecycle["searchable"],
                    "lifecycle_label": lifecycle["lifecycle_label"],
                }
            except (KeyError, FileNotFoundError):
                pass
            try:
                item = self.product.asset(candidate)
                public_id = f"local:{candidate}"
                thumbnail_url, content_url = self._urls(public_id)
                return {
                    "asset_id": public_id,
                    "source_asset_id": candidate,
                    "source": "local",
                    "source_type": "registered_persistent",
                    "library_id": str(item.get("library_id") or "default"),
                    "display_name": self._safe_display_filename(
                        item.get("original_filename"),
                        item.get("display_name"),
                        item.get("filename"),
                        item.get("image_id"),
                        fallback_id=candidate,
                    ),
                    "thumbnail_url": thumbnail_url,
                    "content_url": content_url,
                    "image_url": content_url,
                    "media_status": "ready",
                }
            except KeyError:
                pass
        fallback_id = str(
            value.get("asset_id") or value.get("image_id") or "unknown_asset"
        )
        return {
            "asset_id": fallback_id,
            "source_asset_id": fallback_id,
            "display_name": self._safe_display_filename(
                value.get("original_filename"),
                value.get("display_name"),
                value.get("filename"),
                fallback_id=fallback_id,
            ),
            "thumbnail_url": None,
            "content_url": None,
            "image_url": None,
            "media_status": "unresolved",
        }

    def resolve_result(self, value: dict[str, Any]) -> dict[str, Any]:
        """Resolve result."""
        item = {**value, **self.resolve_identity(value)}
        dimensions = (
            2560
            if str(item.get("retrieval_backend") or "").startswith("bailian")
            else 2048
            if item.get("retrieval_backend") == "e1"
            else None
        )
        item["index_identity"] = {
            "backend": item.get("retrieval_backend"),
            "model": item.get("model"),
            "revision": item.get("revision"),
            "dimensions": dimensions,
            "index_version": item.get("index_version"),
        }
        return item

    def media_path(
        self,
        asset_id: str,
        *,
        thumbnail: bool,
    ) -> tuple[Path, str]:
        """Execute the media path operation."""
        if asset_id.startswith("local:"):
            item = self.product.asset(asset_id.removeprefix("local:"))
            path = Path(str(item["path"])).resolve()
            return path, str(
                item.get("mime_type")
                or mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
        try:
            if thumbnail:
                path = self.system_libraries.thumbnail_path(asset_id)
            else:
                path = self.system_libraries.image_path(asset_id)
            return path.resolve(), str(
                mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
        except KeyError:
            pass
        path = self.library.image_path(asset_id).resolve()
        return path, str(
            mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
