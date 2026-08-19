"""Auditable exact Faiss index used by Phase 5.3 candidates."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


def l2_normalize(values: Any) -> Any:
    """Execute the l2 normalize operation."""
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        denominator = float(np.linalg.norm(array))
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("cannot normalize a non-finite or zero vector")
        return array / denominator
    if array.ndim != 2:
        raise ValueError("vectors must be one- or two-dimensional")
    denominators = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or not np.isfinite(denominators).all():
        raise ValueError("vectors contain non-finite values")
    if (denominators <= 0).any():
        raise ValueError("vectors contain a zero row")
    return array / denominators


def truncate_mrl(values: Any, dimensions: int) -> Any:
    """Execute the truncate mrl operation."""
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if dimensions < 1 or dimensions > array.shape[-1]:
        raise ValueError("MRL dimensions must be within the source vector dimension")
    return l2_normalize(array[..., :dimensions])


class FaissExactIndex:
    """Persist normalized vectors in a CPU IndexFlatIP with separate metadata."""

    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        self.index_path = index_path.resolve()
        self.metadata_path = metadata_path.resolve()
        self.index: Any | None = None
        self.image_ids: list[str] = []
        self.records: list[dict[str, Any]] = []
        if self.index_path.is_file() and self.metadata_path.is_file():
            self.load()

    @staticmethod
    def _faiss() -> Any:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss_is_required_for_this_candidate") from exc
        return faiss

    def build(
        self,
        *,
        vectors: Any,
        image_ids: Iterable[str],
        records: Iterable[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the requested value."""
        import numpy as np

        if self.index_path.exists() or self.metadata_path.exists():
            raise FileExistsError("refusing to overwrite an existing candidate index")
        matrix = l2_normalize(vectors).astype(np.float32, copy=False)
        ids = [str(value) for value in image_ids]
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError("vector rows and image IDs differ")
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("image IDs must be non-empty and unique")
        metadata_records = list(records or ({"image_id": image_id} for image_id in ids))
        if len(metadata_records) != len(ids):
            raise ValueError("metadata rows and image IDs differ")

        faiss = self._faiss()
        index = faiss.IndexFlatIP(int(matrix.shape[1]))
        index.add(matrix)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_index = self.index_path.with_name(f"{self.index_path.name}.tmp.{os.getpid()}")
        temporary_metadata = self.metadata_path.with_name(
            f"{self.metadata_path.name}.tmp.{os.getpid()}"
        )
        faiss.write_index(index, str(temporary_index))
        temporary_metadata.write_text(
            json.dumps(
                {
                    "schema_version": "scenemindx_faiss_exact_index_v1",
                    "index_type": "IndexFlatIP",
                    "metric": "inner_product_on_l2_normalized_vectors",
                    "dimensions": int(matrix.shape[1]),
                    "items": len(ids),
                    "image_ids": ids,
                    "records": metadata_records,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary_index, self.index_path)
        os.replace(temporary_metadata, self.metadata_path)
        self.index = index
        self.image_ids = ids
        self.records = metadata_records
        return self.status()

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        faiss = self._faiss()
        index = faiss.read_index(str(self.index_path))
        ids = [str(value) for value in payload["image_ids"]]
        records = [dict(value) for value in payload["records"]]
        if index.ntotal != len(ids) or len(records) != len(ids):
            raise ValueError("persisted Faiss index metadata is inconsistent")
        if index.d != int(payload["dimensions"]):
            raise ValueError("persisted Faiss index dimension is inconsistent")
        self.index = index
        self.image_ids = ids
        self.records = records
        return self.status()

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return {
            "status": "ready" if self.index is not None else "not_built",
            "index_type": "IndexFlatIP",
            "items": len(self.image_ids),
            "dimensions": int(self.index.d) if self.index is not None else None,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
        }

    def search(
        self,
        query_vector: Any,
        *,
        top_k: int = 5,
        exclude_image_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the search operation."""
        import numpy as np

        if self.index is None:
            raise RuntimeError("faiss_index_not_built")
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        vector = l2_normalize(query_vector).astype(np.float32, copy=False)
        if vector.ndim != 1 or vector.shape[0] != self.index.d:
            raise ValueError("query vector dimension differs from index")
        requested = min(len(self.image_ids), top_k + int(bool(exclude_image_id)))
        scores, indices = self.index.search(vector.reshape(1, -1), requested)
        results: list[dict[str, Any]] = []
        for score, raw_index in zip(scores[0].tolist(), indices[0].tolist()):
            if raw_index < 0:
                continue
            image_id = self.image_ids[raw_index]
            if exclude_image_id and image_id == exclude_image_id:
                continue
            results.append(
                {
                    "rank": len(results) + 1,
                    "image_id": image_id,
                    "score": float(score),
                    "source": "qwen3_vl_embedding_2b+faiss_index_flat_ip",
                    "metadata": self.records[raw_index],
                }
            )
            if len(results) >= top_k:
                break
        return results
