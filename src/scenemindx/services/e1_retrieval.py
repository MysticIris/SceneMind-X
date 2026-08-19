"""Product E1 retrieval index, lifecycle management and R0 fallback routing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from scenemindx.retrieval.candidate_contract import (
    AssetLifecycleRegistry,
    adaptive_topk_refill,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FaissRetrievalIndex:
    """Persistent exact E1 product index with asset and checksum validation."""

    schema_version = "scenemindx_e1_product_index_v1"

    def __init__(
        self,
        root: Path,
        *,
        dimensions: int = 2048,
        lifecycle_path: Path | None = None,
        asset_path_resolver: Callable[[str | Path], Path] | None = None,
        asset_path_serializer: Callable[[str | Path], str] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.index_path = self.root / "index.faiss"
        self.metadata_path = self.root / "metadata.json"
        self.manifest_path = self.root / "manifest.json"
        self.checksums_path = self.root / "checksums.sha256"
        self.dimensions = dimensions
        self.index: Any | None = None
        self.records: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.last_search_ms: float | None = None
        self.last_search_debug: dict[str, Any] = {}
        self.lifecycle_registry = AssetLifecycleRegistry(lifecycle_path)
        self.asset_path_resolver = asset_path_resolver or (
            lambda value: Path(value).resolve()
        )
        self.asset_path_serializer = asset_path_serializer or (
            lambda value: str(Path(value).resolve())
        )
        self._lock = threading.RLock()
        if self.index_path.is_file() and self.metadata_path.is_file():
            try:
                self.load()
            except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError):
                self.index = None
                self.records = []
                self.metadata = {}

    @staticmethod
    def _faiss() -> Any:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss_is_required_for_e1_product_index") from exc
        return faiss

    @classmethod
    def _read_faiss_index(cls, path: Path) -> Any:
        faiss = cls._faiss()
        if hasattr(faiss, "deserialize_index"):
            import numpy as np

            payload = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            return faiss.deserialize_index(payload)
        return faiss.read_index(str(path))

    @classmethod
    def _write_faiss_index(cls, index: Any, path: Path) -> None:
        faiss = cls._faiss()
        if hasattr(faiss, "serialize_index"):
            payload = faiss.serialize_index(index)
            path.write_bytes(memoryview(payload).tobytes())
            return
        faiss.write_index(index, str(path))

    @staticmethod
    def _normalize(values: Any, dimensions: int) -> Any:
        import numpy as np

        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (dimensions,) or not np.isfinite(vector).all():
            raise ValueError("e1_embedding_vector_invalid")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("e1_embedding_vector_zero_or_nonfinite")
        return (vector / norm).astype(np.float32, copy=False)

    def _resolve_image_path(self, value: str | Path) -> Path:
        return self.asset_path_resolver(value)

    def _record(self, item: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_image_path(item["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = str(item.get("sha256") or _sha256_file(path)).lower()
        if _sha256_file(path) != digest:
            raise ValueError(f"e1_asset_sha256_mismatch:{item.get('image_id')}")
        asset_id = str(item.get("asset_id") or item["image_id"])
        return {
            "asset_id": asset_id,
            "image_id": str(item["image_id"]),
            "sha256": digest,
            "image_path": self.asset_path_serializer(path),
            "library_id": str(item.get("library_id", "default")),
            "source": str(item.get("source", "frozen_library")),
            "image_url": str(item.get("image_url", "")),
            "retrieval_text": str(item.get("retrieval_text", "")),
        }

    def canonical_records(self, items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute the canonical records operation."""
        records = [self._record(item) for item in items]
        asset_ids = [item["asset_id"] for item in records]
        if not records or len(asset_ids) != len(set(asset_ids)):
            raise ValueError("e1_product_asset_ids_must_be_nonempty_and_unique")
        return records

    def validate_metadata(self, expected_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Validate metadata."""
        if self.index is None:
            raise RuntimeError("e1_product_index_not_loaded")
        if self.index.d != self.dimensions:
            raise ValueError("e1_product_index_dimension_mismatch")
        if self.index.ntotal != len(self.records):
            raise ValueError("e1_product_index_metadata_count_mismatch")
        if self.metadata.get("schema_version") != self.schema_version:
            raise ValueError("e1_product_index_schema_mismatch")
        if expected_records is not None:
            expected_manifest = _canonical_sha256(expected_records)
            if self.metadata.get("manifest_sha256") != expected_manifest:
                raise ValueError("e1_product_index_manifest_mismatch")
        return self.status()

    def _validate_checksums(self) -> None:
        if not self.checksums_path.is_file():
            raise ValueError("e1_product_index_checksums_missing")
        expected: dict[str, str] = {}
        for line in self.checksums_path.read_text(encoding="utf-8").splitlines():
            digest, separator, name = line.partition("  ")
            if not separator:
                raise ValueError("e1_product_index_checksums_invalid")
            expected[name] = digest
        for path in (self.index_path, self.metadata_path, self.manifest_path):
            if expected.get(path.name) != _sha256_file(path):
                raise ValueError(f"e1_product_index_checksum_mismatch:{path.name}")

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        with self._lock:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            records = json.loads(self.manifest_path.read_text(encoding="utf-8"))["records"]
            index = self._read_faiss_index(self.index_path)
            self.metadata = payload
            self.records = [dict(item) for item in records]
            self.index = index
            self._validate_checksums()
            return self.validate_metadata()

    def _existing_vectors(self) -> dict[str, Any]:
        if self.index is None:
            return {}
        return {
            record["sha256"]: self.index.reconstruct(position)
            for position, record in enumerate(self.records)
        }

    def reconcile(
        self,
        items: Iterable[dict[str, Any]],
        *,
        embedding: Any,
        model: str,
        revision: str,
        identity_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the reconcile operation."""
        import numpy as np

        with self._lock:
            records = self.canonical_records(items)
            manifest_sha = _canonical_sha256(records)
            if self.index is not None and self.metadata.get("manifest_sha256") == manifest_sha:
                self.validate_metadata(records)
                return {**self.status(), "reused": len(records), "encoded": 0, "rebuilt": False}

            existing = self._existing_vectors()
            vectors: list[Any] = []
            reused = 0
            encoded = 0
            for record in records:
                vector = existing.get(record["sha256"])
                if vector is None:
                    vector = embedding.encode_image(
                        self._resolve_image_path(record["image_path"])
                    )
                    encoded += 1
                else:
                    reused += 1
                vectors.append(self._normalize(vector, self.dimensions))
            matrix = np.stack(vectors).astype(np.float32, copy=False)
            faiss = self._faiss()
            index = faiss.IndexFlatIP(self.dimensions)
            index.add(matrix)
            index_version = f"e1-product-{manifest_sha[:12]}"
            metadata = {
                "schema_version": self.schema_version,
                "index_version": index_version,
                "index_type": "IndexFlatIP",
                "metric": "inner_product_on_l2_normalized_vectors",
                "dimensions": self.dimensions,
                "items": len(records),
                "manifest_sha256": manifest_sha,
                "model": model,
                "revision": revision,
                "built_at": _now_iso(),
            }
            if identity_metadata:
                previous_created_at = (
                    self.metadata.get("created_at")
                    if isinstance(self.metadata, dict)
                    else None
                )
                metadata.update(
                    {
                        **identity_metadata,
                        "created_at": previous_created_at or _now_iso(),
                        "updated_at": _now_iso(),
                    }
                )
            manifest = {
                "schema_version": self.schema_version,
                "index_version": index_version,
                "records": records,
            }
            self.root.mkdir(parents=True, exist_ok=True)
            temporary_index = self.root / f"index.faiss.tmp.{os.getpid()}"
            temporary_metadata = self.root / f"metadata.json.tmp.{os.getpid()}"
            temporary_manifest = self.root / f"manifest.json.tmp.{os.getpid()}"
            temporary_checksums = self.root / f"checksums.sha256.tmp.{os.getpid()}"
            try:
                self._write_faiss_index(index, temporary_index)
                temporary_metadata.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                temporary_manifest.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                checksums = [
                    f"{_sha256_file(temporary_index)}  {self.index_path.name}",
                    f"{_sha256_file(temporary_metadata)}  {self.metadata_path.name}",
                    f"{_sha256_file(temporary_manifest)}  {self.manifest_path.name}",
                ]
                temporary_checksums.write_text(
                    "\n".join(checksums) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                os.replace(temporary_index, self.index_path)
                os.replace(temporary_metadata, self.metadata_path)
                os.replace(temporary_manifest, self.manifest_path)
                os.replace(temporary_checksums, self.checksums_path)
            finally:
                for path in (
                    temporary_index,
                    temporary_metadata,
                    temporary_manifest,
                    temporary_checksums,
                ):
                    if path.exists():
                        path.unlink()
            self.index = index
            self.records = records
            self.metadata = metadata
            self.validate_metadata(records)
            return {**self.status(), "reused": reused, "encoded": encoded, "rebuilt": True}

    def append_missing(
        self,
        items: Iterable[dict[str, Any]],
        *,
        embedding: Any,
        model: str,
        revision: str,
        identity_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append only previously unseen SHA identities.

        This is the user-asset lifecycle path.  It preserves existing vectors
        and order, encodes only missing content, and atomically replaces the
        persisted index files without reconstructing the full matrix.
        """

        import numpy as np

        with self._lock:
            incoming = self.canonical_records(items)
            existing_sha = {
                str(record["sha256"]) for record in self.records
            }
            missing: list[dict[str, Any]] = []
            seen_sha = set(existing_sha)
            for record in incoming:
                digest = str(record["sha256"])
                if digest in seen_sha:
                    continue
                seen_sha.add(digest)
                missing.append(record)
            if not missing:
                if self.index is not None:
                    self.validate_metadata()
                return {
                    **self.status(),
                    "reused": len(incoming),
                    "encoded": 0,
                    "appended": 0,
                    "rebuilt": False,
                }

            vectors = [
                self._normalize(
                    embedding.encode_image(
                        self._resolve_image_path(record["image_path"])
                    ),
                    self.dimensions,
                )
                for record in missing
            ]
            matrix = np.stack(vectors).astype(np.float32, copy=False)
            faiss = self._faiss()
            if self.index is None:
                index = faiss.IndexFlatIP(self.dimensions)
            elif hasattr(faiss, "clone_index"):
                index = faiss.clone_index(self.index)
            else:
                index = faiss.deserialize_index(
                    faiss.serialize_index(self.index)
                )
            index.add(matrix)

            records = [*self.records, *missing]
            manifest_sha = _canonical_sha256(records)
            index_version = f"e1-product-{manifest_sha[:12]}"
            created_at = (
                self.metadata.get("created_at")
                or self.metadata.get("built_at")
                or _now_iso()
            )
            metadata = {
                **dict(self.metadata),
                "schema_version": self.schema_version,
                "index_version": index_version,
                "index_type": "IndexFlatIP",
                "metric": "inner_product_on_l2_normalized_vectors",
                "dimensions": self.dimensions,
                "items": len(records),
                "manifest_sha256": manifest_sha,
                "model": model,
                "revision": revision,
                "created_at": created_at,
                "updated_at": _now_iso(),
                "last_append_count": len(missing),
            }
            if identity_metadata:
                metadata.update(identity_metadata)
            manifest = {
                "schema_version": self.schema_version,
                "index_version": index_version,
                "records": records,
            }
            self.root.mkdir(parents=True, exist_ok=True)
            temporary_index = self.root / f"index.faiss.tmp.{os.getpid()}"
            temporary_metadata = self.root / f"metadata.json.tmp.{os.getpid()}"
            temporary_manifest = self.root / f"manifest.json.tmp.{os.getpid()}"
            temporary_checksums = self.root / f"checksums.sha256.tmp.{os.getpid()}"
            try:
                self._write_faiss_index(index, temporary_index)
                temporary_metadata.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                temporary_manifest.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                temporary_checksums.write_text(
                    "\n".join(
                        [
                            f"{_sha256_file(temporary_index)}  {self.index_path.name}",
                            f"{_sha256_file(temporary_metadata)}  {self.metadata_path.name}",
                            f"{_sha256_file(temporary_manifest)}  {self.manifest_path.name}",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                os.replace(temporary_index, self.index_path)
                os.replace(temporary_metadata, self.metadata_path)
                os.replace(temporary_manifest, self.manifest_path)
                os.replace(temporary_checksums, self.checksums_path)
            finally:
                for path in (
                    temporary_index,
                    temporary_metadata,
                    temporary_manifest,
                    temporary_checksums,
                ):
                    if path.exists():
                        path.unlink()
            self.index = index
            self.records = records
            self.metadata = metadata
            self.validate_metadata(records)
            return {
                **self.status(),
                "reused": len(incoming) - len(missing),
                "encoded": len(missing),
                "appended": len(missing),
                "rebuilt": False,
            }

    def build(self, items: Iterable[dict[str, Any]], *, embedding: Any, model: str, revision: str) -> dict[str, Any]:
        """Build the requested value."""
        return self.reconcile(items, embedding=embedding, model=model, revision=revision)

    def rebuild(self, items: Iterable[dict[str, Any]], *, embedding: Any, model: str, revision: str) -> dict[str, Any]:
        """Execute the rebuild operation."""
        return self.reconcile(items, embedding=embedding, model=model, revision=revision)

    def add_asset(self, items: Iterable[dict[str, Any]], *, embedding: Any, model: str, revision: str) -> dict[str, Any]:
        """Execute the add asset operation."""
        return self.reconcile(items, embedding=embedding, model=model, revision=revision)

    def remove_asset(self, items: Iterable[dict[str, Any]], *, embedding: Any, model: str, revision: str) -> dict[str, Any]:
        """Remove asset."""
        return self.reconcile(items, embedding=embedding, model=model, revision=revision)

    def save(self) -> dict[str, Any]:
        """Save the requested value."""
        return self.validate_metadata()

    def search(
        self,
        vector: Any,
        *,
        top_k: int,
        exclude_asset_id: str | None = None,
        unique_sha: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute the search operation."""
        import numpy as np

        with self._lock:
            self.validate_metadata()
            query = self._normalize(vector, self.dimensions)
            requested = min(len(self.records), max(4 * top_k, 20))
            started = time.perf_counter()
            candidates: list[dict[str, Any]] = []
            history: list[dict[str, Any]] = []
            while True:
                scores, positions = self.index.search(
                    query.reshape(1, -1),
                    requested,
                )
                candidates = []
                filtered = {
                    "archived_or_inactive": 0,
                    "excluded": 0,
                    "media_unavailable": 0,
                }
                for score, position in zip(
                    scores[0].tolist(),
                    positions[0].tolist(),
                ):
                    if position < 0:
                        continue
                    record = self.records[position]
                    lifecycle = self.lifecycle_registry.lookup(record)
                    if not self.lifecycle_registry.is_searchable(record):
                        filtered["archived_or_inactive"] += 1
                        continue
                    if (
                        exclude_asset_id
                        and record["asset_id"] == exclude_asset_id
                    ):
                        filtered["excluded"] += 1
                        continue
                    image_path = str(record.get("image_path") or "").strip()
                    # Historical/unit-test records predate the persisted
                    # image_path field.  Absence means "not asserted", while
                    # an explicit path must still resolve before delivery.
                    try:
                        media_available = (
                            self._resolve_image_path(image_path).is_file()
                            if image_path
                            else True
                        )
                    except (OSError, ValueError):
                        media_available = False
                    if not media_available:
                        filtered["media_unavailable"] += 1
                        continue
                    candidates.append(
                        {
                            "asset_id": record["asset_id"],
                            "image_id": record["image_id"],
                            "sha256": str(
                                record.get("sha256")
                                or record.get("image_sha256")
                                or ""
                            ),
                            "library_id": str(
                                record.get("library_id")
                                or "legacy_frozen"
                            ),
                            "score": float(score),
                            "image_score": float(score),
                            "source": record["source"],
                            "image_url": record["image_url"],
                            "media_available": True,
                            "lifecycle_state": lifecycle[
                                "lifecycle_state"
                            ],
                            "searchable": lifecycle["searchable"],
                            "match_basis": (
                                f"{self.metadata.get('model') or 'multimodal embedding'} "
                                "cosine similarity"
                            ),
                        }
                    )
                history.append(
                    {
                        "fetch_n": requested,
                        "eligible_records": len(candidates),
                        "filtered": filtered,
                    }
                )
                if len(candidates) >= top_k or requested >= len(self.records):
                    break
                next_requested = min(len(self.records), requested * 2)
                if next_requested <= requested:
                    break
                requested = next_requested
            self.last_search_ms = (time.perf_counter() - started) * 1000.0
            candidates.sort(
                key=lambda item: (
                    -float(item["score"]),
                    str(item["asset_id"]),
                )
            )
            if unique_sha:
                by_sha: dict[str, dict[str, Any]] = {}
                for item in candidates:
                    identity = str(item.get("sha256") or item["asset_id"])
                    by_sha.setdefault(identity, item)
                candidates = list(by_sha.values())
            results = candidates[:top_k]
            self.last_search_debug = {
                "requested_k": top_k,
                "fetch_history": history,
                "rounds": len(history),
                "exhausted": (
                    len(results) < top_k
                    and requested >= len(self.records)
                ),
                "physical_ntotal": len(self.records),
                "query_embedding_count": 0,
            }
            for rank, item in enumerate(results, start=1):
                item["rank"] = rank
                item["faiss_search_ms"] = self.last_search_ms
                item["faiss_candidate_refill"] = self.last_search_debug
            return results

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        lifecycle = self.lifecycle_registry.summary(
            self.records,
            physical_ntotal=(
                int(self.index.ntotal) if self.index is not None else 0
            ),
        )
        return {
            "status": "ready" if self.index is not None else "not_built",
            "items": len(self.records),
            "dimensions": self.dimensions if self.index is not None else None,
            "index_type": "IndexFlatIP",
            "index_version": self.metadata.get("index_version"),
            "index_path": str(self.index_path),
            **lifecycle,
            "last_search_debug": self.last_search_debug,
            "metadata_path": str(self.metadata_path),
            "manifest_path": str(self.manifest_path),
            "checksums_path": str(self.checksums_path),
            "last_search_ms": self.last_search_ms,
        }


class E1RetrievalAdapter:
    """Unified product retrieval with an explicit, traceable R0 fallback."""

    def __init__(
        self,
        *,
        e1_embedding: Any,
        e1_index: FaissRetrievalIndex,
        r0: Any,
        requested_backend: str = "e1",
        fallback_backend: str = "r0",
    ) -> None:
        if requested_backend not in {"e1", "r0"}:
            raise ValueError("SCENEMINDX_RETRIEVAL_BACKEND must be e1 or r0")
        if fallback_backend != "r0":
            raise ValueError("SCENEMINDX_RETRIEVAL_FALLBACK must be r0")
        self.e1_embedding = e1_embedding
        self.e1_index = e1_index
        self.r0 = r0
        self.requested_backend = requested_backend
        self.fallback_backend = fallback_backend
        self.last_fallback_reason: str | None = None
        self.last_build: dict[str, Any] | None = None

    @property
    def embedding(self) -> Any:
        """Execute the embedding operation."""
        return self.e1_embedding if self.requested_backend == "e1" else self.r0.embedding

    @property
    def index_path(self) -> Path:
        """Execute the index path operation."""
        return self.e1_index.index_path if self.requested_backend == "e1" else self.r0.index_path

    @property
    def vectors(self) -> Any | None:
        """Execute the vectors operation."""
        if self.requested_backend == "e1":
            return self.e1_index.index
        return self.r0.vectors

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        self.load_index_only()
        if self.requested_backend == "e1":
            try:
                self.e1_embedding.load()
                self.last_fallback_reason = None
            except Exception as exc:
                self.last_fallback_reason = f"{type(exc).__name__}:{exc}"
        return self.status()

    def load_index_only(self) -> dict[str, Any]:
        """Bind persisted local indexes without probing a remote embedding service."""
        if self.r0.index_path.is_file():
            self.r0.load()
        if (
            self.requested_backend == "e1"
            and self.e1_index.index_path.is_file()
        ):
            self.e1_index.load()
        return self.cached_status()

    def build_index(self, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Build index."""
        rows = list(items)
        r0_status = self.r0.build_index(rows)
        if self.requested_backend == "r0":
            self.last_build = {"backend": "r0", "r0": r0_status}
            return self.status()
        identity = self.e1_embedding.status()
        if identity.get("status") != "ready":
            self.last_fallback_reason = f"e1_model_not_ready:{identity.get('error') or identity.get('status')}"
            self.last_build = {"backend": "r0", "fallback_reason": self.last_fallback_reason}
            return self.status()
        try:
            e1_status = self.e1_index.reconcile(
                rows,
                embedding=self.e1_embedding,
                model=str(identity.get("model")),
                revision=str(identity.get("model_revision")),
            )
            self.last_fallback_reason = None
            self.last_build = {"backend": "e1", "e1": e1_status, "r0": r0_status}
        except Exception as exc:
            self.last_fallback_reason = f"{type(exc).__name__}:{exc}"
            self.last_build = {"backend": "r0", "fallback_reason": self.last_fallback_reason}
        return self.status()

    def _decorate_r0(
        self,
        rows: list[dict[str, Any]],
        *,
        fallback_used: bool,
        reason: str | None = None,
    ) -> list[dict[str, Any]]:
        status = self.r0.embedding.status()
        return [
            {
                **row,
                "asset_id": row.get("asset_id") or row.get("image_id"),
                "retrieval_backend": "r0",
                "model": status.get("model"),
                "revision": status.get("model_revision"),
                "index_version": "r0-color-grid-v1",
                "fallback_used": fallback_used,
                "fallback_reason": reason,
            }
            for row in rows
        ]

    def _search_e1(
        self,
        encoder: Any,
        *,
        top_k: int,
        exclude_asset_id: str | None = None,
    ) -> list[dict[str, Any]]:
        vector = encoder()
        return self.search_vector_scoped(
            vector,
            top_k=top_k,
            exclude_asset_ids=(
                {exclude_asset_id} if exclude_asset_id else set()
            ),
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
                self.e1_embedding.encode_multimodal(image_path, query),
                "official_qwen3_vl_multimodal_joint_encoding",
            )
        if image_path is not None:
            return (
                self.e1_embedding.encode_image(image_path),
                "official_qwen3_vl_image_encoding",
            )
        if query:
            return (
                self.e1_embedding.encode_text(query),
                "official_qwen3_vl_text_encoding",
            )
        raise ValueError("retrieval_requires_text_or_image")

    def search_vector_scoped(
        self,
        vector: Any,
        *,
        top_k: int,
        requested_library_ids: set[str] | None = None,
        exclude_asset_ids: set[str] | None = None,
        route: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the search vector scoped operation."""
        rows, debug = adaptive_topk_refill(
            lambda fetch_n: self.e1_index.search(
                vector,
                top_k=fetch_n,
                unique_sha=False,
            ),
            requested_k=top_k,
            total_candidates=len(self.e1_index.records),
            requested_library_ids=requested_library_ids,
            exclude_asset_ids=exclude_asset_ids,
            current_library_id=(
                next(iter(requested_library_ids))
                if requested_library_ids
                and len(requested_library_ids) == 1
                else None
            ),
        )
        identity = {
            "model": getattr(self.e1_embedding, "model_id", "Qwen/Qwen3-VL-Embedding-2B"),
            "model_revision": getattr(self.e1_embedding, "model_revision", "unknown"),
        }
        embedding_latency_ms = getattr(
            self.e1_embedding,
            "_last_encode_latency_ms",
            None,
        )
        server_embedding_latency_ms = getattr(
            self.e1_embedding,
            "_last_server_latency_ms",
            None,
        )
        return [
            {
                **row,
                "retrieval_backend": "e1",
                "model": identity.get("model"),
                "revision": identity.get("model_revision"),
                "index_version": self.e1_index.metadata.get("index_version"),
                "fallback_used": False,
                "fallback_reason": None,
                "embedding_latency_ms": embedding_latency_ms,
                "server_embedding_latency_ms": server_embedding_latency_ms,
                "route": route or row.get("route"),
                "candidate_refill": debug,
            }
            for row in rows
        ]

    def search_scoped(
        self,
        *,
        query_text: str | None = None,
        image_path: Path | None = None,
        top_k: int = 5,
        requested_library_ids: set[str] | None = None,
        exclude_asset_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the search scoped operation."""
        vector, route = self.encode_query(
            query_text=query_text,
            image_path=image_path,
        )
        return self.search_vector_scoped(
            vector,
            top_k=top_k,
            requested_library_ids=requested_library_ids,
            exclude_asset_ids=exclude_asset_ids,
            route=route,
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Execute the search operation."""
        if self.requested_backend == "r0":
            return self._decorate_r0(self.r0.search(query, top_k), fallback_used=False)
        try:
            return self._search_e1(lambda: self.e1_embedding.encode_text(query), top_k=top_k)
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            self.last_fallback_reason = reason
            return self._decorate_r0(self.r0.search(query, top_k), fallback_used=True, reason=reason)

    def search_image(
        self,
        image_path: Path,
        top_k: int = 5,
        *,
        exclude_image_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the search image operation."""
        if self.requested_backend == "r0":
            return self._decorate_r0(
                self.r0.search_image(image_path, top_k, exclude_image_id=exclude_image_id),
                fallback_used=False,
            )
        try:
            return self._search_e1(
                lambda: self.e1_embedding.encode_image(image_path),
                top_k=top_k,
                exclude_asset_id=exclude_image_id,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            self.last_fallback_reason = reason
            return self._decorate_r0(
                self.r0.search_image(image_path, top_k, exclude_image_id=exclude_image_id),
                fallback_used=True,
                reason=reason,
            )

    def search_hybrid(
        self,
        image_path: Path,
        query: str,
        top_k: int = 5,
        *,
        image_weight: float = 0.55,
        text_weight: float = 0.35,
        lexical_weight: float = 0.10,
        exclude_image_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the search hybrid operation."""
        if self.requested_backend == "r0":
            return self._decorate_r0(
                self.r0.search_hybrid(
                    image_path,
                    query,
                    top_k,
                    image_weight=image_weight,
                    text_weight=text_weight,
                    lexical_weight=lexical_weight,
                    exclude_image_id=exclude_image_id,
                ),
                fallback_used=False,
            )
        try:
            rows = self._search_e1(
                lambda: self.e1_embedding.encode_multimodal(image_path, query),
                top_k=top_k,
                exclude_asset_id=exclude_image_id,
            )
            for row in rows:
                row["route"] = "official_qwen3_vl_multimodal_joint_encoding"
            return rows
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            self.last_fallback_reason = reason
            return self._decorate_r0(
                self.r0.search_hybrid(
                    image_path,
                    query,
                    top_k,
                    image_weight=image_weight,
                    text_weight=text_weight,
                    lexical_weight=lexical_weight,
                    exclude_image_id=exclude_image_id,
                ),
                fallback_used=True,
                reason=reason,
            )

    def _status_from_model(self, e1_model: dict[str, Any]) -> dict[str, Any]:
        e1_index = self.e1_index.status()
        e1_ready = e1_index["status"] == "ready" and e1_model.get("status") == "ready"
        active = "e1" if self.requested_backend == "e1" and e1_ready else "r0"
        fallback_active = self.requested_backend == "e1" and active == "r0"
        return {
            "status": "ready" if (e1_ready or self.r0.vectors is not None) else "not_built",
            "requested_backend": self.requested_backend,
            "active_backend": active,
            "retrieval_backend": active,
            "fallback_backend": self.fallback_backend,
            "fallback_active": fallback_active,
            "fallback_reason": self.last_fallback_reason if fallback_active else None,
            "items": e1_index["items"] if active == "e1" else len(self.r0.image_ids),
            "physical_vectors": (
                e1_index.get("physical_vectors", 0)
                if active == "e1"
                else len(self.r0.image_ids)
            ),
            "active_searchable_records": (
                e1_index.get("active_searchable_records", 0)
                if active == "e1"
                else len(self.r0.image_ids)
            ),
            "active_searchable_unique_sha": (
                e1_index.get("active_searchable_unique_sha", 0)
                if active == "e1"
                else len(self.r0.image_ids)
            ),
            "archived_records": (
                e1_index.get("archived_records", 0)
                if active == "e1"
                else 0
            ),
            "total_unique_sha": (
                e1_index.get("total_unique_sha", 0)
                if active == "e1"
                else len(self.r0.image_ids)
            ),
            "duplicate_sha_records": (
                e1_index.get("duplicate_sha_records", 0)
                if active == "e1"
                else 0
            ),
            "dimensions": e1_index["dimensions"] if active == "e1" else (
                int(self.r0.vectors.shape[1]) if self.r0.vectors is not None else None
            ),
            "index_type": e1_index["index_type"] if active == "e1" else "numpy_exact_cosine",
            "index_version": e1_index["index_version"] if active == "e1" else "r0-color-grid-v1",
            "index_path": e1_index["index_path"] if active == "e1" else str(self.r0.index_path),
            "embedding": e1_model if active == "e1" else self.r0.embedding.status(),
            "e1": {"model": e1_model, "index": e1_index},
            "r0": self.r0.status(),
            "last_build": self.last_build,
        }

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return self._status_from_model(self.e1_embedding.status())

    def cached_status(self) -> dict[str, Any]:
        """Execute the cached status operation."""
        reader = getattr(self.e1_embedding, "cached_status", None)
        e1_model = (
            reader()
            if callable(reader)
            else self.e1_embedding.status()
        )
        return self._status_from_model(
            dict(e1_model)
            if isinstance(e1_model, dict)
            else {"status": "error", "loaded": False}
        )
