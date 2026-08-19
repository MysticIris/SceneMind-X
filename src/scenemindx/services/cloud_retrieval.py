"""Independent Bailian 2560-d retrieval index with resumable vector caching."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scenemindx.services.cloud_image_transport import (
    CloudImageTransportPreprocessor,
)
from scenemindx.services.e1_retrieval import FaissRetrievalIndex
from scenemindx.retrieval.candidate_contract import adaptive_topk_refill


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class CachedCloudEmbedding:
    """Cache image vectors by SHA while leaving text/image queries live."""

    def __init__(
        self,
        provider: Any,
        *,
        root: Path,
        identity: dict[str, Any],
        transport_preprocessor: CloudImageTransportPreprocessor | None = None,
    ) -> None:
        self.provider = provider
        self.root = root.resolve()
        self.identity = dict(identity)
        self.identity_sha256 = _canonical_sha256(self.identity)
        self.vector_root = self.root / "vectors" / self.identity_sha256[:16]
        self.dimension = int(self.identity["dimension"])
        self.dimensions = self.dimension
        self.normalization = "l2"
        self.provider_id = "bailian"
        self.model_id = str(self.identity["model_id"])
        self.model_revision = str(
            self.identity.get("provider_model_revision") or "provider_alias"
        )
        self.transport_preprocessor = transport_preprocessor
        self.events: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def replace_provider(self, provider: Any) -> None:
        """Execute the replace provider operation."""
        with self._lock:
            self.provider = provider

    def _vector_path(self, digest: str) -> Path:
        return self.vector_root / f"{digest}.json"

    def _read_vector_payload(self, digest: str) -> dict[str, Any] | None:
        path = self._vector_path(digest)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("identity_sha256") != self.identity_sha256
                or payload.get("sha256") != digest
            ):
                return None
            vector = [float(value) for value in payload.get("vector", [])]
            if len(vector) != self.dimension or not all(
                math.isfinite(value) for value in vector
            ):
                return None
            return {**payload, "vector": vector}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _read_vector(self, digest: str) -> list[float] | None:
        payload = self._read_vector_payload(digest)
        return list(payload["vector"]) if payload is not None else None

    def vector_provenance(self, digest: str) -> dict[str, Any]:
        """Return persisted, credential-free call provenance for one vector."""
        with self._lock:
            payload = self._read_vector_payload(digest)
        if payload is None:
            return {
                "persisted": False,
                "request_count": 0,
                "retry_count": 0,
                "original_image_sent_to_bailian": False,
            }
        usage = dict(payload.get("usage") or {})
        return {
            "persisted": True,
            "request_count": 1,
            "retry_count": int(usage.get("retry_count", 0) or 0),
            "original_image_sent_to_bailian": bool(
                payload.get("transport", {}).get(
                    "original_request_attempted",
                    payload.get("transport", {}).get("transport_profile_id")
                    in {None, "default_transport_v1"},
                )
            ),
            "vector_id": payload.get("vector_id"),
            "created_at": payload.get("created_at"),
            "usage": usage,
            "transport": dict(payload.get("transport") or {}),
        }

    def encode_image(self, image_path: Path) -> list[float]:
        """Execute the encode image operation."""
        path = image_path.resolve()
        digest = _sha256_file(path)
        with self._lock:
            cached_payload = self._read_vector_payload(digest)
            if cached_payload is not None:
                persisted_usage = dict(cached_payload.get("usage") or {})
                self.events[digest] = {
                    "status": "completed",
                    "cache_hit": True,
                    # A persisted vector is proof that the original request
                    # completed. A later cache hit must not erase that
                    # provenance from the experiment manifest.
                    "api_called": True,
                    "request_count": 1,
                    "retry_count": int(
                        persisted_usage.get("retry_count", 0) or 0
                    ),
                    "vector_id": f"cloud2560:{digest[:24]}",
                    "usage": persisted_usage,
                }
                return list(cached_payload["vector"])
        if self.transport_preprocessor is None:
            vector = [float(value) for value in self.provider.encode_image(path)]
            transport = {
                "transport_profile_id": "default_transport_v1",
                "transport_profile_version": "1",
                "original_sha256": digest,
                "transport_sha256": digest,
                "original_request_attempted": True,
                "fallback_from_413": False,
                "request_count": 1,
            }
        else:
            result = self.transport_preprocessor.encode_image(
                provider=self.provider,
                image_path=path,
                original_asset_id=f"sha256:{digest}",
            )
            vector = result.vector
            transport = {
                **result.prepared.audit,
                "original_request_attempted": (
                    result.prepared.profile_id == "default_transport_v1"
                    or result.fallback_from_413
                ),
                "fallback_from_413": result.fallback_from_413,
                "request_count": result.request_count,
            }
        if len(vector) != self.dimension:
            raise ValueError("cloud_embedding_dimension_mismatch")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("cloud_embedding_zero_or_nonfinite")
        vector = [value / norm for value in vector]
        usage = (
            self.provider.usage()
            if callable(getattr(self.provider, "usage", None))
            else {}
        )
        payload = {
            "schema_version": "scenemindx_cloud_vector_cache_v1",
            "identity_sha256": self.identity_sha256,
            "sha256": digest,
            "vector_id": f"cloud2560:{digest[:24]}",
            "dimension": self.dimension,
            "normalization": self.normalization,
            "created_at": _now_iso(),
            "usage": {
                key: usage.get(key)
                for key in (
                    "image_tokens",
                    "text_tokens",
                    "total_tokens",
                    "estimated_cost_cny",
                    "retry_count",
                )
                if usage.get(key) is not None
            },
            "transport": transport,
            "vector": vector,
        }
        _atomic_json(self._vector_path(digest), payload)
        with self._lock:
            self.events[digest] = {
                "status": "completed",
                "cache_hit": False,
                "api_called": True,
                "request_count": 1,
                "retry_count": int(usage.get("retry_count", 0) or 0),
                "vector_id": payload["vector_id"],
                "usage": payload["usage"],
                "transport": transport,
            }
        return vector

    def encode_text(self, text: str) -> list[float]:
        """Execute the encode text operation."""
        return self.provider.encode_text(text)

    def encode_multimodal(self, image_path: Path, text: str) -> list[float]:
        """Execute the encode multimodal operation."""
        return self.provider.encode_multimodal(image_path, text)

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        return self.status()

    def health_check(self) -> dict[str, Any]:
        """Execute the health check operation."""
        return self.provider.health_check()

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        status = dict(self.provider.status())
        return {
            **status,
            "provider_id": self.provider_id,
            "model": self.model_id,
            "model_id": self.model_id,
            "model_revision": status.get("reported_model") or self.model_revision,
            "dimensions": self.dimension,
            "normalization": self.normalization,
            "identity_sha256": self.identity_sha256,
        }

    def usage(self) -> dict[str, Any]:
        """Execute the usage operation."""
        return (
            dict(self.provider.usage())
            if callable(getattr(self.provider, "usage", None))
            else {}
        )

    def error_mapping(self, error: Any) -> dict[str, Any]:
        """Execute the error mapping operation."""
        return (
            dict(self.provider.error_mapping(error))
            if callable(getattr(self.provider, "error_mapping", None))
            else {"category": "unknown", "code": type(error).__name__}
        )


class BailianCloudRetrieval:
    """Exact cloud index with no local-E1 or R0 fallback."""

    provider_id = "bailian"
    retrieval_backend = "bailian_cloud_e1"

    def __init__(
        self,
        *,
        embedding: Any,
        root: Path,
        identity: dict[str, Any],
        transport_preprocessor: CloudImageTransportPreprocessor | None = None,
        base_index_root: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.identity = dict(identity)
        self.embedding = CachedCloudEmbedding(
            embedding,
            root=self.root,
            identity=self.identity,
            transport_preprocessor=transport_preprocessor,
        )
        self.e1_embedding = self.embedding
        self.e1_index = FaissRetrievalIndex(
            self.root / "faiss",
            dimensions=int(self.identity["dimension"]),
        )
        self.base_index = (
            FaissRetrievalIndex(
                base_index_root.resolve(),
                dimensions=int(self.identity["dimension"]),
            )
            if base_index_root is not None
            and (base_index_root / "index.faiss").is_file()
            else None
        )
        self.last_build: dict[str, Any] | None = None
        self.last_error: dict[str, Any] | None = None
        self._lock = threading.RLock()

    @property
    def index_path(self) -> Path:
        """Execute the index path operation."""
        if self.base_index is not None and self.base_index.index is not None:
            return self.base_index.index_path
        return self.e1_index.index_path

    @property
    def vectors(self) -> Any | None:
        """Execute the vectors operation."""
        if self.base_index is not None and self.base_index.index is not None:
            return self.base_index.index
        return self.e1_index.index

    def replace_embedding(self, embedding: Any) -> None:
        """Execute the replace embedding operation."""
        self.embedding.replace_provider(embedding)

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        if self.base_index is not None and self.base_index.index_path.is_file():
            self.base_index.load()
        if self.index_path.is_file():
            if self.e1_index.index_path.is_file():
                self.e1_index.load()
        return self.status()

    def _base_sha256(self) -> set[str]:
        if self.base_index is None or self.base_index.index is None:
            return set()
        return {
            str(record.get("sha256") or "")
            for record in self.base_index.records
            if record.get("sha256")
        }

    def _user_overlay_records(self) -> list[dict[str, Any]]:
        base_sha = self._base_sha256()
        return [
            record
            for record in self.e1_index.records
            if str(record.get("source")) == "local_upload"
            and str(record.get("sha256") or "") not in base_sha
        ]

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return the authoritative base plus unique user overlay records."""

        if self.base_index is None or self.base_index.index is None:
            return list(self.e1_index.records)
        return [
            *self.base_index.records,
            *self._user_overlay_records(),
        ]

    @staticmethod
    def _deduplicate(
        existing: Iterable[dict[str, Any]],
        incoming: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_sha: dict[str, dict[str, Any]] = {}
        for item in [*existing, *incoming]:
            digest = str(item.get("sha256") or "").lower()
            if digest and digest not in by_sha:
                by_sha[digest] = dict(item)
        return list(by_sha.values())

    def reconcile(
        self,
        items: Iterable[dict[str, Any]],
        *,
        preserve_existing: bool = True,
    ) -> dict[str, Any]:
        """Execute the reconcile operation."""
        with self._lock:
            status = self.embedding.status()
            incoming = list(items)
            records = self._deduplicate(
                self.e1_index.records if preserve_existing else [],
                incoming,
            )
            existing_shas = {
                str(item.get("sha256"))
                for item in self.e1_index.records
                if item.get("sha256")
            }
            requires_encoding = any(
                str(item.get("sha256")) not in existing_shas
                for item in records
            )
            if status.get("status") != "ready" and requires_encoding:
                raise RuntimeError("cloud_embedding_connection_required")
            result = self.e1_index.reconcile(
                records,
                embedding=self.embedding,
                model=self.embedding.model_id,
                revision=str(
                    status.get("reported_model")
                    or status.get("model_revision")
                    or "provider_alias"
                ),
                identity_metadata={
                    "provider": "bailian",
                    "region": self.identity["region"],
                    "model_id": self.identity["model_id"],
                    "provider_model_revision": (
                        status.get("reported_model") or "provider_alias"
                    ),
                    "dimension": int(self.identity["dimension"]),
                    "vector_mode": self.identity["vector_mode"],
                    "normalization": self.identity["normalization"],
                    "metric": self.identity["metric"],
                    "preprocess_version": self.identity["preprocess_version"],
                    "index_schema_version": self.identity["index_schema_version"],
                    "identity_sha256": self.embedding.identity_sha256,
                },
            )
            self.last_build = {
                **result,
                "backend": self.retrieval_backend,
                "fallback_used": False,
                "existing_preserved": preserve_existing,
            }
            self.last_error = None
            return self.status()

    def build_index(self, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Build index."""
        return self.reconcile(items)

    def add_assets(self, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Execute the add assets operation."""
        with self._lock:
            status = self.embedding.status()
            incoming = self._deduplicate([], list(items))
            existing_shas = {
                str(item.get("sha256"))
                for item in self.records
                if item.get("sha256")
            }
            missing = [
                item
                for item in incoming
                if str(item.get("sha256")) not in existing_shas
            ]
            if not missing:
                self.last_build = {
                    **self.status(),
                    "backend": self.retrieval_backend,
                    "encoded": 0,
                    "appended": 0,
                    "rebuilt": False,
                    "idempotent_skip": True,
                }
                return self.status()
            if status.get("status") != "ready":
                raise RuntimeError("cloud_embedding_connection_required")
            result = self.e1_index.append_missing(
                missing,
                embedding=self.embedding,
                model=self.embedding.model_id,
                revision=str(
                    status.get("reported_model")
                    or status.get("model_revision")
                    or "provider_alias"
                ),
                identity_metadata={
                    "provider": "bailian",
                    "region": self.identity["region"],
                    "model_id": self.identity["model_id"],
                    "provider_model_revision": (
                        status.get("reported_model") or "provider_alias"
                    ),
                    "dimension": int(self.identity["dimension"]),
                    "vector_mode": self.identity["vector_mode"],
                    "normalization": self.identity["normalization"],
                    "metric": self.identity["metric"],
                    "preprocess_version": self.identity["preprocess_version"],
                    "index_schema_version": self.identity["index_schema_version"],
                    "identity_sha256": self.embedding.identity_sha256,
                },
            )
            self.last_build = {
                **result,
                "backend": self.retrieval_backend,
                "fallback_used": False,
                "existing_preserved": True,
                "incremental_append": True,
            }
            self.last_error = None
            return self.status()

    def _search(
        self,
        vector: list[float],
        *,
        top_k: int,
        exclude_asset_id: str | None = None,
    ) -> list[dict[str, Any]]:
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
    ) -> tuple[list[float], str]:
        """Execute the encode query operation."""
        query = str(query_text or "").strip()
        if image_path is not None and query:
            return (
                self.embedding.encode_multimodal(image_path, query),
                "bailian_qwen3_vl_multimodal_joint_encoding",
            )
        if image_path is not None:
            return (
                self.embedding.encode_image(image_path),
                "bailian_qwen3_vl_image_encoding",
            )
        if query:
            return (
                self.embedding.encode_text(query),
                "bailian_qwen3_vl_text_encoding",
            )
        raise ValueError("cloud_retrieval_requires_text_or_image")

    def search_vector_scoped(
        self,
        vector: list[float],
        *,
        top_k: int,
        requested_library_ids: set[str] | None = None,
        exclude_asset_ids: set[str] | None = None,
        route: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the search vector scoped operation."""
        base_ready = (
            self.base_index is not None
            and self.base_index.index is not None
        )
        user_asset_ids = {
            str(record.get("asset_id"))
            for record in self._user_overlay_records()
        }
        base_count = (
            len(self.base_index.records)
            if base_ready and self.base_index is not None
            else 0
        )
        overlay_count = len(user_asset_ids)
        base_library_ids = {
            str(record.get("library_id") or "")
            for record in (
                self.base_index.records
                if base_ready and self.base_index is not None
                else []
            )
        }
        overlay_library_ids = {
            str(record.get("library_id") or "")
            for record in self._user_overlay_records()
        }
        include_base = (
            requested_library_ids is None
            or bool(base_library_ids & requested_library_ids)
        )
        include_overlay = (
            requested_library_ids is None
            or bool(overlay_library_ids & requested_library_ids)
        )

        def fetch(fetch_n: int) -> list[dict[str, Any]]:
            candidates: list[dict[str, Any]] = []
            if include_base and base_ready and self.base_index is not None:
                candidates.extend(
                    self.base_index.search(
                        vector,
                        top_k=min(fetch_n, base_count),
                        unique_sha=False,
                    )
                )
            if (
                include_overlay
                and self.e1_index.index is not None
                and user_asset_ids
            ):
                candidates.extend(
                    row
                    for row in self.e1_index.search(
                        vector,
                        top_k=min(
                            len(self.e1_index.records),
                            fetch_n,
                        ),
                        unique_sha=False,
                    )
                    if str(row.get("asset_id")) in user_asset_ids
                )
            if (
                not base_ready
                and self.e1_index.index is not None
            ):
                candidates.extend(
                    self.e1_index.search(
                        vector,
                        top_k=min(
                            len(self.e1_index.records),
                            fetch_n,
                        ),
                        unique_sha=False,
                    )
                )
            return candidates

        rows, debug = adaptive_topk_refill(
            fetch,
            requested_k=top_k,
            total_candidates=max(
                overlay_count if include_overlay else 0,
                base_count if include_base else 0,
                (
                    len(self.e1_index.records)
                    if not base_ready
                    else 0
                ),
                0,
            ),
            requested_library_ids=requested_library_ids,
            exclude_asset_ids=exclude_asset_ids,
            current_library_id=(
                next(iter(requested_library_ids))
                if requested_library_ids
                and len(requested_library_ids) == 1
                else None
            ),
        )
        status = self.embedding.status()
        active_index = (
            self.base_index
            if self.base_index is not None and self.base_index.index is not None
            else self.e1_index
        )
        return [
            {
                **row,
                "retrieval_backend": self.retrieval_backend,
                "model": self.embedding.model_id,
                "revision": status.get("reported_model") or "provider_alias",
                "index_version": active_index.metadata.get("index_version"),
                "fallback_used": False,
                "fallback_reason": None,
                "embedding_dimension": self.embedding.dimension,
                "route": route or row.get("route"),
                "candidate_refill": debug,
            }
            for row in rows
        ]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Execute the search operation."""
        vector, route = self.encode_query(query_text=query)
        return self.search_vector_scoped(
            vector,
            top_k=top_k,
            route=route,
        )

    def search_image(
        self,
        image_path: Path,
        top_k: int = 5,
        *,
        exclude_image_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the search image operation."""
        vector, route = self.encode_query(image_path=image_path)
        return self.search_vector_scoped(
            vector,
            top_k=top_k,
            exclude_asset_ids=(
                {exclude_image_id} if exclude_image_id else set()
            ),
            route=route,
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
        del image_weight, text_weight, lexical_weight
        vector, route = self.encode_query(
            query_text=query,
            image_path=image_path,
        )
        rows = self.search_vector_scoped(
            vector,
            top_k=top_k,
            exclude_asset_ids=(
                {exclude_image_id} if exclude_image_id else set()
            ),
            route=route,
        )
        for row in rows:
            row["route"] = "bailian_qwen3_vl_multimodal_joint_encoding"
        return rows

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        overlay_index = self.e1_index.status()
        base_index = (
            self.base_index.status()
            if self.base_index is not None
            else {"status": "not_built", "items": 0}
        )
        full_base_active = (
            self.base_index is not None
            and base_index.get("status") == "ready"
        )
        base_items = (
            len(self.base_index.records)
            if full_base_active and self.base_index is not None
            else sum(
                str(record.get("source")) == "course_train_first_numeric"
                for record in self.e1_index.records
            )
        )
        user_items = len(self._user_overlay_records())
        index = (
            {
                **base_index,
                "items": base_items + user_items,
                "base_items": base_items,
                "user_items": user_items,
                "course_items": base_items,
                "active_identity": "full_train_val_plus_user_overlay",
                "user_overlay_index_path": overlay_index.get("index_path"),
            }
            if full_base_active
            else overlay_index
        )
        embedding = self.embedding.status()
        ready = (
            index.get("status") == "ready"
            and embedding.get("status") == "ready"
        )
        return {
            "status": "ready" if ready else (
                "index_ready_connection_required"
                if index.get("status") == "ready"
                else "not_built"
            ),
            "requested_backend": self.retrieval_backend,
            "active_backend": self.retrieval_backend if ready else None,
            "retrieval_backend": self.retrieval_backend,
            "fallback_backend": None,
            "fallback_active": False,
            "fallback_reason": None,
            "items": int(index.get("items", 0) or 0),
            "base_items": base_items,
            "user_items": user_items,
            "dimensions": index.get("dimensions"),
            "index_type": index.get("index_type"),
            "index_version": index.get("index_version"),
            "index_path": index.get("index_path"),
            "embedding": embedding,
            "index": index,
            "identity": {
                **self.identity,
                "identity_sha256": self.embedding.identity_sha256,
            },
            "full_base_index": base_index,
            "historical_overlay_index": overlay_index,
            "shared_by_cloud_tiers": True,
            "last_build": self.last_build,
            "last_error": self.last_error,
        }
