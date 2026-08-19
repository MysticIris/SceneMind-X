"""Independent E1 search across Phase 6.1 system libraries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from scenemindx.services.e1_retrieval import FaissRetrievalIndex
from scenemindx.retrieval.candidate_contract import adaptive_topk_refill

from .multi_library import SYSTEM_LIBRARY_IDS


class SplitE1IndexRegistry:
    """Load Train and Val Faiss indices without mixing their manifests."""

    def __init__(
        self,
        root: Path | None,
        *,
        embedding: Any,
        dimensions: int = 2048,
        asset_path_resolver: Callable[[str | Path], Path] | None = None,
        asset_path_serializer: Callable[[str | Path], str] | None = None,
    ) -> None:
        self.root = root.resolve() if root else None
        self.embedding = embedding
        self.dimensions = dimensions
        self.indices: dict[str, FaissRetrievalIndex] = {}
        if self.root is not None:
            for library_id in sorted(SYSTEM_LIBRARY_IDS):
                path = self.root / library_id / "faiss"
                if (
                    (path / "index.faiss").is_file()
                    and (path / "metadata.json").is_file()
                ):
                    index = FaissRetrievalIndex(
                        path,
                        dimensions=dimensions,
                        asset_path_resolver=asset_path_resolver,
                        asset_path_serializer=asset_path_serializer,
                    )
                    if index.status()["status"] == "ready":
                        self.indices[library_id] = index

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return {
            "status": (
                "ready"
                if set(self.indices) == set(SYSTEM_LIBRARY_IDS)
                else "partial"
                if self.indices
                else "not_built"
            ),
            "root": str(self.root) if self.root else None,
            "libraries": {
                library_id: (
                    self.indices[library_id].status()
                    if library_id in self.indices
                    else {"status": "not_built", "items": 0}
                )
                for library_id in ("system_train", "system_val")
            },
            "model": getattr(self.embedding, "model_id", None),
            "model_revision": getattr(self.embedding, "model_revision", None),
            "r0_fallback_triggered": False,
        }

    def public_status(self) -> dict[str, Any]:
        """Return search-facing status without runtime filesystem paths."""

        status = self.status()
        allowed_library_keys = {
            "status",
            "items",
            "dimensions",
            "index_type",
            "index_version",
            "last_search_ms",
        }
        return {
            "status": status["status"],
            "libraries": {
                library_id: {
                    key: value
                    for key, value in library_status.items()
                    if key in allowed_library_keys
                }
                for library_id, library_status in status["libraries"].items()
            },
            "model": status["model"],
            "model_revision": status["model_revision"],
            "r0_fallback_triggered": status["r0_fallback_triggered"],
        }

    @staticmethod
    def _library_ids(values: Iterable[str]) -> list[str]:
        result = list(dict.fromkeys(str(value) for value in values))
        if not result or len(result) > 2:
            raise ValueError("one_or_two_system_libraries_required")
        if any(value not in SYSTEM_LIBRARY_IDS for value in result):
            raise ValueError("invalid_system_library_scope")
        return result

    def search(
        self,
        *,
        library_ids: Iterable[str],
        query_text: str | None = None,
        image_path: Path | None = None,
        top_k: int = 5,
        exclude_asset_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the search operation."""
        scoped = self._library_ids(library_ids)
        missing = [library_id for library_id in scoped if library_id not in self.indices]
        if missing:
            raise RuntimeError(
                "system_e1_index_not_built:" + ",".join(missing)
            )
        query = str(query_text or "").strip()
        vector, route = self.encode_query(
            query_text=query,
            image_path=image_path,
        )
        return self.search_vector(
            vector,
            library_ids=scoped,
            top_k=top_k,
            exclude_asset_ids=exclude_asset_ids,
            route=route,
        )

    def encode_query(
        self,
        *,
        query_text: str | None = None,
        image_path: Path | None = None,
    ) -> tuple[Any, str]:
        """Execute the encode query operation."""
        query = str(query_text or "").strip()
        if image_path is not None and query:
            return (
                self.embedding.encode_multimodal(image_path, query),
                "official_qwen3_vl_multimodal_joint_encoding",
            )
        if image_path is not None:
            return (
                self.embedding.encode_image(image_path),
                "official_qwen3_vl_image_encoding",
            )
        if query:
            return (
                self.embedding.encode_text(query),
                "official_qwen3_vl_text_encoding",
            )
        raise ValueError("system_retrieval_requires_text_or_image")

    def search_vector(
        self,
        vector: Any,
        *,
        library_ids: Iterable[str],
        top_k: int = 5,
        exclude_asset_ids: set[str] | None = None,
        route: str = "preencoded_query_vector",
    ) -> list[dict[str, Any]]:
        """Execute the search vector operation."""
        scoped = self._library_ids(library_ids)
        excluded = exclude_asset_ids or set()

        def fetch(fetch_n: int) -> list[dict[str, Any]]:
            candidates: list[dict[str, Any]] = []
            for library_id in scoped:
                index = self.indices[library_id]
                for item in index.search(
                    vector,
                    top_k=min(len(index.records), fetch_n),
                    unique_sha=False,
                ):
                    candidates.append(
                        {
                            **item,
                            "library_id": library_id,
                            "source_library": library_id,
                            "source_split": library_id.removeprefix(
                                "system_"
                            ),
                            "retrieval_backend": "e1",
                            "fallback_used": False,
                            "model": getattr(
                                self.embedding,
                                "model_id",
                                None,
                            ),
                            "revision": getattr(
                                self.embedding,
                                "model_revision",
                                None,
                            ),
                            "index_version": index.status().get(
                                "index_version"
                            ),
                            "route": route,
                        }
                    )
            return candidates

        results, debug = adaptive_topk_refill(
            fetch,
            requested_k=top_k,
            total_candidates=max(
                len(self.indices[library_id].records)
                for library_id in scoped
            ),
            requested_library_ids=set(scoped),
            exclude_asset_ids=excluded,
        )
        for item in results:
            item["candidate_refill"] = debug
        return results
