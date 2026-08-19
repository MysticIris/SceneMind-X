"""On-demand local Qwen model preflight and embedding runtime."""

from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any


VLM_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
VLM_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EMBEDDING_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
EMBEDDING_REVISION = "cda4398c9bbfb3a644105446a2793692a8da5ea1"
EMBEDDING_WEIGHTS_SHA256 = (
    "c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1"
)
EMBEDDING_SOURCE_COMMIT = "393e2978d27852b0d0230d6994f37f9c15bed73c"
EMBEDDING_DIMENSION = 2048
EMBEDDING_INSTRUCTION = "Retrieve images relevant to the user's query."
LOCAL_INDEX_IDENTITY_SCHEMA = "scenemindx_local_embedding_identity_v2"

# Exact frozen model measurements in the accepted unified server environment.
OBSERVED_VLM_PEAK_BYTES = 8_875_651_584
OBSERVED_EMBEDDING_PEAK_BYTES = 4_672_253_952
RECOMMENDED_FREE_VRAM_BYTES = 16 * 1024**3
CONSERVATIVE_FREE_VRAM_BYTES = 18 * 1024**3

LOCAL_NOT_INSTALLED_MESSAGE = (
    "本地模型接入能力已准备完成，但当前设备尚未安装模型权重，且显存不足以"
    "稳定加载完整双模型。建议继续使用百炼云端模式。"
)

INDEX_IDENTITY_FIELDS = [
    "provider_runtime",
    "model_id",
    "model_revision",
    "weights_sha256",
    "source_commit",
    "image_preprocess",
    "text_preprocess",
    "dimension",
    "normalization",
    "metric",
    "serialization_schema",
]


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return "<external-path>"


def _model_files_present(path: Path, *, embedding: bool = False) -> bool:
    if not path.is_dir():
        return False
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not embedding:
        required.add("preprocessor_config.json")
    return all((path / name).is_file() for name in required) and bool(
        list(path.glob("*.safetensors"))
    )


def _dependency_state() -> dict[str, Any]:
    modules = {
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "safetensors": "safetensors",
        "qwen-vl-utils": "qwen_vl_utils",
        "numpy": "numpy",
        "Pillow": "PIL",
        "faiss-cpu": "faiss",
    }
    result: dict[str, Any] = {}
    for distribution, module in modules.items():
        installed = importlib.util.find_spec(module) is not None
        try:
            version = (
                importlib.metadata.version(distribution)
                if installed
                else None
            )
        except importlib.metadata.PackageNotFoundError:
            version = None
        result[distribution] = {
            "installed": installed,
            "version": version,
        }
    return result


def _manifest_state(
    *,
    project_root: Path,
    manifest_path: Path | None,
) -> dict[str, Any]:
    path = manifest_path or (
        project_root / "configs" / "providers" / "local_models_manifest.json"
    )
    if not path.is_file():
        return {
            "status": "missing",
            "relative_path": _relative(path, project_root),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {
            "status": "invalid",
            "relative_path": _relative(path, project_root),
        }
    return {
        "status": "ready",
        "relative_path": _relative(path, project_root),
        "schema_version": payload.get("schema_version"),
        "models": sorted((payload.get("models") or {}).keys()),
    }


def local_model_preflight(
    *,
    vlm_path: Path,
    embedding_path: Path,
    embedding_source_path: Path,
    disk_root: Path,
    manifest_path: Path | None = None,
    local_index_root: Path | None = None,
) -> dict[str, Any]:
    """Return public hardware/runtime facts without loading model weights."""

    project_root = disk_root.resolve()
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda or "not_available")
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            free_vram, total_vram = torch.cuda.mem_get_info()
            device_name = torch.cuda.get_device_name(0)
            device_count = int(torch.cuda.device_count())
        else:
            free_vram, total_vram = 0, 0
            device_name = None
            device_count = 0
    except Exception:
        torch_version = "unavailable"
        cuda_version = "unavailable"
        cuda_available = False
        free_vram, total_vram = 0, 0
        device_name = None
        device_count = 0

    disk = shutil.disk_usage(project_root)
    vlm_present = _model_files_present(vlm_path)
    embedding_present = _model_files_present(embedding_path, embedding=True)
    source_file = (
        embedding_source_path / "src" / "models" / "qwen3_vl_embedding.py"
    )
    source_present = source_file.is_file()
    dependencies = _dependency_state()
    manifest = _manifest_state(
        project_root=project_root,
        manifest_path=manifest_path,
    )
    index_root = local_index_root or (
        project_root / "data" / "indexes" / "local_e1_2048"
    )
    index_scopes = {
        scope: {
            "status": (
                "prepared"
                if (index_root / scope / "faiss" / "index.faiss").is_file()
                and (index_root / scope / "faiss" / "identity.json").is_file()
                else "missing"
            ),
            "relative_path": _relative(
                index_root / scope / "faiss",
                project_root,
            ),
        }
        for scope in ("product", "system_train", "system_val")
    }

    hard_blockers: list[dict[str, str]] = []
    if not cuda_available:
        hard_blockers.append(
            {"code": "LOCAL_CUDA_UNAVAILABLE", "label": "CUDA 不可用"}
        )
    if not vlm_present:
        hard_blockers.append(
            {
                "code": "LOCAL_VLM_WEIGHTS_MISSING",
                "label": f"缺少 {VLM_MODEL_ID} 本地权重",
            }
        )
    if not embedding_present:
        hard_blockers.append(
            {
                "code": "LOCAL_EMBEDDING_WEIGHTS_MISSING",
                "label": f"缺少 {EMBEDDING_MODEL_ID} 本地权重",
            }
        )
    if not source_present:
        hard_blockers.append(
            {
                "code": "LOCAL_EMBEDDING_SOURCE_MISSING",
                "label": "缺少冻结的 Qwen3-VL-Embedding 推理源码",
            }
        )
    if manifest["status"] != "ready":
        hard_blockers.append(
            {
                "code": "LOCAL_MODEL_MANIFEST_INVALID",
                "label": "本地模型清单缺失或无效",
            }
        )

    warnings: list[str] = []
    if not cuda_available:
        warnings.append("未检测到可用 CUDA。")
    if free_vram < RECOMMENDED_FREE_VRAM_BYTES:
        warnings.append("当前可用显存低于双模型稳定加载建议值 16 GiB。")
    if not vlm_present or not embedding_present:
        warnings.append("本地模型权重尚未安装；系统不会自动下载。")
    if not source_present:
        warnings.append("冻结 Embedding 推理源码缺失。")

    historical_sum = OBSERVED_VLM_PEAK_BYTES + OBSERVED_EMBEDDING_PEAK_BYTES
    can_attempt = bool(
        cuda_available
        and vlm_present
        and embedding_present
        and source_present
        and manifest["status"] == "ready"
    )
    recommended = bool(
        can_attempt and free_vram >= RECOMMENDED_FREE_VRAM_BYTES
    )
    conclusion = (
        LOCAL_NOT_INSTALLED_MESSAGE
        if not vlm_present or not embedding_present
        else (
            "本地模型文件已就绪，但当前可用显存低于稳定加载建议值。"
            if free_vram < RECOMMENDED_FREE_VRAM_BYTES
            else "本地双模型文件、推理源码和硬件前置条件均已就绪。"
        )
    )
    return {
        "status": "ready_to_attempt" if can_attempt else "attention_required",
        "load_started": False,
        "cuda_available": cuda_available,
        "torch_version": torch_version,
        "torch_cuda_version": cuda_version,
        "gpu": {
            "name": device_name,
            "device_count": device_count,
            "total_vram_bytes": int(total_vram),
            "free_vram_bytes": int(free_vram),
        },
        "disk": {"relative_root": ".", "free_bytes": int(disk.free)},
        "weights": {
            "vlm": {
                "model_id": VLM_MODEL_ID,
                "relative_path": _relative(vlm_path, project_root),
                "present": vlm_present,
            },
            "embedding": {
                "model_id": EMBEDDING_MODEL_ID,
                "relative_path": _relative(embedding_path, project_root),
                "present": embedding_present,
                "source_relative_path": _relative(
                    embedding_source_path,
                    project_root,
                ),
                "source_present": source_present,
            },
        },
        "manifest": manifest,
        "dependencies": dependencies,
        "indexes": {
            "root": _relative(index_root, project_root),
            "scopes": index_scopes,
            "all_prepared": all(
                value["status"] == "prepared"
                for value in index_scopes.values()
            ),
        },
        "historical_vram_evidence": {
            "vlm_peak_bytes": OBSERVED_VLM_PEAK_BYTES,
            "embedding_peak_bytes": OBSERVED_EMBEDDING_PEAK_BYTES,
            "sum_bytes": historical_sum,
            "recommended_free_vram_bytes": RECOMMENDED_FREE_VRAM_BYTES,
            "conservative_free_vram_bytes": CONSERVATIVE_FREE_VRAM_BYTES,
            "basis": (
                "exact frozen model measurements in the unified server environment"
            ),
        },
        "can_attempt": can_attempt,
        "recommended": recommended,
        "warnings": warnings,
        "hard_blockers": hard_blockers,
        "low_vram_requires_override": bool(
            can_attempt and free_vram < RECOMMENDED_FREE_VRAM_BYTES
        ),
        "diagnostic_summary": {
            "cuda": "available" if cuda_available else "unavailable",
            "gpu_name": device_name,
            "total_vram_gib": round(float(total_vram) / 1024**3, 2),
            "free_vram_gib": round(float(free_vram) / 1024**3, 2),
            "vlm_weights": "found" if vlm_present else "missing",
            "embedding_weights": "found" if embedding_present else "missing",
            "embedding_source": "found" if source_present else "missing",
            "model_manifest": manifest["status"],
            "standard_indexes": (
                "prepared"
                if all(
                    value["status"] == "prepared"
                    for value in index_scopes.values()
                )
                else "incomplete"
            ),
            "historical_dual_peak_gib": round(historical_sum / 1024**3, 2),
            "recommended_free_vram_gib": 16,
            "recommended_gpu_vram_gib": 24,
        },
        "conclusion": conclusion,
        "recommended_model_paths": {
            "vlm": _relative(vlm_path, project_root),
            "embedding": _relative(embedding_path, project_root),
            "embedding_source": _relative(
                embedding_source_path,
                project_root,
            ),
        },
        "automatic_download": False,
        "quantization_changed": False,
        "index_reuse_policy": {
            "schema_version": LOCAL_INDEX_IDENTITY_SCHEMA,
            "requires_exact_identity": True,
            "required_fields": INDEX_IDENTITY_FIELDS,
            "self_hosted_reuse_allowed": False,
            "conclusion": (
                "本地标准索引只在全部 11 项身份字段精确匹配时只读复用；"
                "不会复用服务器映射句柄，也不会自动重新编码。"
            ),
        },
    }


def local_index_reuse_contract(
    *,
    preflight: dict[str, Any],
    index_path: Path,
) -> dict[str, Any]:
    """Decide reuse from the full frozen identity, never from model name."""

    policy = dict(preflight.get("index_reuse_policy") or {})
    required = list(policy.get("required_fields") or INDEX_IDENTITY_FIELDS)
    identity_path = index_path / "identity.json"
    reasons: list[str] = []
    identity: dict[str, Any] = {}
    if identity_path.is_file():
        try:
            value = json.loads(identity_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                identity = value
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append("INDEX_IDENTITY_INVALID")
    else:
        reasons.append("INDEX_IDENTITY_MISSING")

    missing = [field for field in required if not identity.get(field)]
    if missing:
        reasons.append("INDEX_IDENTITY_FIELDS_INCOMPLETE")
    expected = {
        "provider_runtime": "local_in_process",
        "model_id": EMBEDDING_MODEL_ID,
        "model_revision": EMBEDDING_REVISION,
        "weights_sha256": EMBEDDING_WEIGHTS_SHA256,
        "source_commit": EMBEDDING_SOURCE_COMMIT,
        "dimension": EMBEDDING_DIMENSION,
        "normalization": "l2",
        "metric": "inner_product",
    }
    mismatches = [
        field
        for field, expected_value in expected.items()
        if identity.get(field) != expected_value
    ]
    if mismatches:
        reasons.append("INDEX_IDENTITY_MISMATCH")
    index_exists = (
        (index_path / "index.faiss").is_file()
        and (index_path / "manifest.json").is_file()
        and (index_path / "metadata.json").is_file()
    )
    if not index_exists:
        reasons.append("INDEX_FILES_MISSING")

    identity_match = not reasons and not missing and not mismatches
    runtime_ready = bool(preflight.get("can_attempt"))
    if not runtime_ready:
        reasons.append("LOCAL_RUNTIME_INCOMPLETE")
    reusable = bool(identity_match and runtime_ready)
    return {
        "schema_version": LOCAL_INDEX_IDENTITY_SCHEMA,
        "reusable": reusable,
        "reusable_when_runtime_ready": identity_match,
        "identity_match": identity_match,
        "index_exists": index_exists,
        "identity_present": identity_path.is_file(),
        "required_fields": required,
        "missing_fields": missing,
        "mismatched_fields": mismatches,
        "reason_codes": list(dict.fromkeys(reasons)),
        "provider_identity": "local",
        "self_hosted_index_reused": False,
        "conclusion": (
            "本地标准索引身份已精确匹配；安装权重并通过 Loader 前置检查后可直接复用。"
            if identity_match and not runtime_ready
            else (
                "本地标准索引身份与运行时精确匹配，可只读加载。"
                if reusable
                else "本地索引身份未获完整证明，不会静默复用或重新编码。"
            )
        ),
    }


class LocalQwenVLEmbeddingService:
    """In-process local E1 runtime using the audited pinned upstream source."""

    model_id = EMBEDDING_MODEL_ID
    model_revision = EMBEDDING_REVISION
    dimension = EMBEDDING_DIMENSION
    dimensions = EMBEDDING_DIMENSION
    normalization = "l2"

    def __init__(self, model_path: Path, source_path: Path) -> None:
        self.model_path = model_path.resolve()
        self.source_path = source_path.resolve()
        self.embedder: Any | None = None
        self.load_seconds: float | None = None
        self.calls = 0
        self.last_latency_ms: float | None = None
        self.load_peak_vram_bytes: int | None = None

    def _embedder_class(self) -> type:
        module_path = (
            self.source_path / "src" / "models" / "qwen3_vl_embedding.py"
        )
        if not module_path.is_file():
            raise FileNotFoundError("local_qwen3_vl_embedding_source_missing")
        spec = importlib.util.spec_from_file_location(
            "scenemindx_local_qwen3_vl_embedding",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("local_embedding_source_load_failed")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.Qwen3VLEmbedder

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        if self.embedder is not None:
            return self.status()
        if not _model_files_present(self.model_path, embedding=True):
            raise FileNotFoundError("local_embedding_weights_missing")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("local_embedding_cuda_required")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.embedder = self._embedder_class()(
            str(self.model_path),
            max_length=8192,
            min_pixels=4096,
            max_pixels=1843200,
            default_instruction=EMBEDDING_INSTRUCTION,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.load_seconds = time.perf_counter() - started
        self.load_peak_vram_bytes = int(torch.cuda.max_memory_allocated())
        return self.status()

    def _encode(self, payload: dict[str, Any]) -> list[float]:
        if self.embedder is None:
            raise RuntimeError("local_embedding_not_loaded")
        import numpy as np

        started = time.perf_counter()
        values = self.embedder.process([payload], normalize=True)
        vector = (
            values[0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        norm = float(np.linalg.norm(vector))
        if (
            vector.shape != (self.dimension,)
            or not np.isfinite(vector).all()
            or not math.isfinite(norm)
            or norm <= 0
        ):
            raise RuntimeError("local_embedding_vector_health_failed")
        vector = (vector / norm).astype(np.float32, copy=False)
        self.calls += 1
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return vector.tolist()

    def encode_text(self, text: str) -> list[float]:
        """Execute the encode text operation."""
        if not text.strip():
            raise ValueError("embedding_text_must_not_be_empty")
        return self._encode(
            {"text": text, "instruction": EMBEDDING_INSTRUCTION}
        )

    def encode_image(self, image_path: Path) -> list[float]:
        """Execute the encode image operation."""
        return self._encode({"image": str(image_path.resolve())})

    def encode_multimodal(
        self,
        image_path: Path,
        text: str,
    ) -> list[float]:
        """Execute the encode multimodal operation."""
        return self._encode(
            {
                "image": str(image_path.resolve()),
                "text": text,
                "instruction": EMBEDDING_INSTRUCTION,
            }
        )

    def unload(self) -> dict[str, Any]:
        """Execute the unload operation."""
        self.embedder = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return self.status()

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return {
            "status": "ready" if self.embedder is not None else "not_loaded",
            "loaded": self.embedder is not None,
            "provider_id": "local",
            "model": self.model_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "source_commit": EMBEDDING_SOURCE_COMMIT,
            "dimensions": self.dimension,
            "normalization": self.normalization,
            "device": "cuda",
            "load_seconds": self.load_seconds,
            "calls": self.calls,
            "last_latency_ms": self.last_latency_ms,
            "peak_vram_bytes": self.load_peak_vram_bytes,
        }
