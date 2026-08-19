"""Environment-driven settings for local and AutoDL Phase 1 runs."""

from __future__ import annotations

import os
import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Phase1Settings:
    """Represent phase1 settings data."""
    project_root: Path
    dataset_root: Path
    manifest_path: Path
    historical_result_root: Path
    ocr_result_root: Path
    run_root: Path
    index_path: Path
    vlm_model_path: Path | None
    embedding_model_path: Path | None
    vlm_endpoint: str | None = None
    enable_vlm: bool = False
    vlm_inline_images: bool = False
    enable_embedding: bool = False
    embedding_backend: str = "transformers"
    embedding_model_id: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    embedding_model_revision: str = "pending_server_inventory"
    retrieval_backend: str = "r0"
    retrieval_fallback: str = "r0"
    e1_embedding_endpoint: str | None = None
    e1_index_root: Path | None = None
    e1_timeout_seconds: float = 30.0
    vlm_gpu_max_memory_gib: float = 14.0
    core_prompt_version: str = "p3_v1_4"
    git_commit: str = "unknown"
    canonical_preview_manifest: Path | None = None
    system_train_root: Path | None = None
    system_val_root: Path | None = None
    system_train_asset_manifest: Path | None = None
    system_val_asset_manifest: Path | None = None
    system_library_catalog: Path | None = None
    system_train_active_manifest: Path | None = None
    system_val_active_manifest: Path | None = None
    system_e1_index_root: Path | None = None
    system_thumbnail_root: Path | None = None
    bailian_credentials_path: Path | None = None

    @classmethod
    def from_env(cls, project_root: str | Path | None = None) -> "Phase1Settings":
        """Execute the from env operation."""
        root = Path(project_root or os.environ.get("SCENEMINDX_PROJECT_ROOT", Path.cwd())).resolve()

        def path_env(name: str, fallback: Path) -> Path:
            return Path(os.environ.get(name, str(fallback))).resolve()

        run_root = path_env("SCENEMINDX_RUN_ROOT", root / "runtime")
        model_value = os.environ.get("SCENEMINDX_VLM_MODEL_PATH")
        endpoint_value = os.environ.get("SCENEMINDX_VLM_ENDPOINT")
        e1_endpoint_value = os.environ.get("SCENEMINDX_E1_EMBEDDING_ENDPOINT")
        embedding_value = os.environ.get("SCENEMINDX_EMBEDDING_MODEL_PATH")
        canonical_preview_value = os.environ.get("SCENEMINDX_CANONICAL_PREVIEW_MANIFEST")
        train_active_value = os.environ.get("SCENEMINDX_SYSTEM_TRAIN_ACTIVE_MANIFEST")
        val_active_value = os.environ.get("SCENEMINDX_SYSTEM_VAL_ACTIVE_MANIFEST")
        bailian_credentials_value = os.environ.get(
            "SCENEMINDX_BAILIAN_CREDENTIALS_PATH"
        )
        e1_index_root = path_env(
            "SCENEMINDX_E1_INDEX_ROOT",
            run_root / "index" / "e1_product",
        )
        return cls(
            project_root=root,
            dataset_root=path_env("SCENEMINDX_DATASET_ROOT", root / "datasets" / "course_train"),
            manifest_path=path_env("SCENEMINDX_MANIFEST_PATH", root / "data" / "manifests" / "gate1_d3_hard_train.jsonl"),
            historical_result_root=path_env(
                "SCENEMINDX_HISTORY_ROOT",
                root / "artifacts" / "gate1" / "d3" / "semantic_compare_p3_v1_3" / "4b",
            ),
            ocr_result_root=path_env("SCENEMINDX_OCR_ROOT", root / "artifacts" / "gate1" / "d3" / "ocr"),
            run_root=run_root,
            index_path=path_env("SCENEMINDX_INDEX_PATH", run_root / "index" / "chinese_clip_index.npz"),
            vlm_model_path=Path(model_value).resolve() if model_value else None,
            embedding_model_path=Path(embedding_value).resolve() if embedding_value else None,
            vlm_endpoint=endpoint_value.rstrip("/") if endpoint_value else None,
            enable_vlm=_env_flag("SCENEMINDX_ENABLE_VLM"),
            vlm_inline_images=_env_flag("SCENEMINDX_VLM_INLINE_IMAGES"),
            enable_embedding=_env_flag("SCENEMINDX_ENABLE_EMBEDDING"),
            embedding_backend=os.environ.get("SCENEMINDX_EMBEDDING_BACKEND", "transformers"),
            embedding_model_id=os.environ.get(
                "SCENEMINDX_EMBEDDING_MODEL_ID",
                "OFA-Sys/chinese-clip-vit-base-patch16",
            ),
            embedding_model_revision=os.environ.get(
                "SCENEMINDX_EMBEDDING_MODEL_REVISION",
                "pending_server_inventory",
            ),
            retrieval_backend=os.environ.get("SCENEMINDX_RETRIEVAL_BACKEND", "r0").strip().lower(),
            retrieval_fallback=os.environ.get("SCENEMINDX_RETRIEVAL_FALLBACK", "r0").strip().lower(),
            e1_embedding_endpoint=e1_endpoint_value.rstrip("/") if e1_endpoint_value else None,
            e1_index_root=e1_index_root,
            e1_timeout_seconds=float(os.environ.get("SCENEMINDX_E1_TIMEOUT_SECONDS", "30")),
            vlm_gpu_max_memory_gib=float(os.environ.get("SCENEMINDX_VLM_GPU_MAX_GIB", "14")),
            core_prompt_version=os.environ.get("SCENEMINDX_CORE_PROMPT_VERSION", "p3_v1_4"),
            git_commit=os.environ.get("SCENEMINDX_GIT_COMMIT", "unknown"),
            canonical_preview_manifest=(
                Path(canonical_preview_value).resolve()
                if canonical_preview_value
                else None
            ),
            system_train_root=path_env(
                "SCENEMINDX_SYSTEM_TRAIN_ROOT",
                root / "datasets" / "course_train",
            ),
            system_val_root=path_env(
                "SCENEMINDX_SYSTEM_VAL_ROOT",
                root / "datasets" / "course_val",
            ),
            system_train_asset_manifest=path_env(
                "SCENEMINDX_SYSTEM_TRAIN_ASSET_MANIFEST",
                root / "data" / "manifests" / "phase6_1_train_assets.jsonl",
            ),
            system_val_asset_manifest=path_env(
                "SCENEMINDX_SYSTEM_VAL_ASSET_MANIFEST",
                root / "data" / "manifests" / "phase6_1_val_assets.jsonl",
            ),
            system_library_catalog=path_env(
                "SCENEMINDX_SYSTEM_LIBRARY_CATALOG",
                root / "data" / "manifests" / "phase6_1_system_libraries.json",
            ),
            system_train_active_manifest=(
                Path(train_active_value).resolve() if train_active_value else None
            ),
            system_val_active_manifest=(
                Path(val_active_value).resolve() if val_active_value else None
            ),
            system_e1_index_root=path_env(
                "SCENEMINDX_SYSTEM_E1_INDEX_ROOT",
                root / "data" / "indexes" / "local_e1_2048",
            ),
            system_thumbnail_root=path_env(
                "SCENEMINDX_SYSTEM_THUMBNAIL_ROOT",
                root / "data" / "cache" / "thumbnails" / "phase6_1",
            ),
            bailian_credentials_path=(
                Path(bailian_credentials_value).resolve()
                if bailian_credentials_value
                else root / ".secrets" / "bailian_credentials.csv"
            ),
        )

    def ensure_runtime_dirs(self) -> None:
        """Ensure runtime dirs."""
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.e1_index_root is not None:
            self.e1_index_root.mkdir(parents=True, exist_ok=True)

    def write_resolved_config(self) -> Path:
        """Write resolved config."""
        prompt_root = self.project_root / "prompts" / "phase1"
        prompt_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(prompt_root.glob("*.txt"))
        }
        value = {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "service_pid": os.getpid(),
            "project_root": str(self.project_root),
            "dataset_root": str(self.dataset_root),
            "manifest_path": str(self.manifest_path),
            "historical_result_root": str(self.historical_result_root),
            "ocr_result_root": str(self.ocr_result_root),
            "run_root": str(self.run_root),
            "index_path": str(self.index_path),
            "canonical_preview_manifest": (
                str(self.canonical_preview_manifest)
                if self.canonical_preview_manifest
                else None
            ),
            "system_train_root": str(self.system_train_root) if self.system_train_root else None,
            "system_val_root": str(self.system_val_root) if self.system_val_root else None,
            "system_train_asset_manifest": (
                str(self.system_train_asset_manifest)
                if self.system_train_asset_manifest
                else None
            ),
            "system_val_asset_manifest": (
                str(self.system_val_asset_manifest)
                if self.system_val_asset_manifest
                else None
            ),
            "system_library_catalog": (
                str(self.system_library_catalog)
                if self.system_library_catalog
                else None
            ),
            "system_train_active_manifest": (
                str(self.system_train_active_manifest)
                if self.system_train_active_manifest
                else None
            ),
            "system_val_active_manifest": (
                str(self.system_val_active_manifest)
                if self.system_val_active_manifest
                else None
            ),
            "system_e1_index_root": (
                str(self.system_e1_index_root)
                if self.system_e1_index_root
                else None
            ),
            "system_thumbnail_root": (
                str(self.system_thumbnail_root)
                if self.system_thumbnail_root
                else None
            ),
            # Never persist a secret path or credential value in the resolved
            # runtime configuration.  This boolean is enough for diagnostics.
            "bailian_course_credentials_file_present": bool(
                self.bailian_credentials_path
                and self.bailian_credentials_path.is_file()
            ),
            "vlm_model_path": str(self.vlm_model_path) if self.vlm_model_path else None,
            "vlm_endpoint": self.vlm_endpoint,
            "embedding_model_path": str(self.embedding_model_path) if self.embedding_model_path else None,
            "enable_vlm": self.enable_vlm,
            "vlm_inline_images": self.vlm_inline_images,
            "enable_embedding": self.enable_embedding,
            "embedding_backend": self.embedding_backend,
            "embedding_model_id": self.embedding_model_id,
            "embedding_model_revision": self.embedding_model_revision,
            "retrieval_backend": self.retrieval_backend,
            "retrieval_fallback": self.retrieval_fallback,
            "e1_embedding_endpoint": self.e1_embedding_endpoint,
            "e1_index_root": str(self.e1_index_root) if self.e1_index_root else None,
            "e1_timeout_seconds": self.e1_timeout_seconds,
            "vlm_gpu_max_memory_gib": self.vlm_gpu_max_memory_gib,
            "core_prompt_version": self.core_prompt_version,
            "git_commit": self.git_commit,
            "api_bind_host": "127.0.0.1",
            "api_port": int(os.environ["SCENEMINDX_API_PORT"]) if os.environ.get("SCENEMINDX_API_PORT") else None,
            "prompt_sha256": prompt_hashes,
        }
        path = self.run_root / "config_resolved.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=self.run_root, suffix=".tmp") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
        return path
