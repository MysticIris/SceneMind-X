"""Legacy Phase 7.4 local-model implementation retained for audit only."""

from __future__ import annotations

import gc
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
EMBEDDING_SOURCE_COMMIT = "393e2978d27852b0d0230d6994f37f9c15bed73c"
EMBEDDING_DIMENSION = 2048
EMBEDDING_INSTRUCTION = "Retrieve images relevant to the user's query."
LOCAL_INDEX_IDENTITY_SCHEMA = "scenemindx_local_embedding_identity_v1"

# Accepted historical remote measurements on the exact frozen models.
OBSERVED_VLM_PEAK_BYTES = 8_875_651_584
OBSERVED_EMBEDDING_PEAK_BYTES = 4_670_996_480
RECOMMENDED_FREE_VRAM_BYTES = 16 * 1024**3
CONSERVATIVE_FREE_VRAM_BYTES = 18 * 1024**3


def _model_files_present(path: Path, *, embedding: bool = False) -> bool:
    if not path.is_dir():
        return False
    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if not embedding:
        required.add("preprocessor_config.json")
    weights = list(path.glob("*.safetensors"))
    return all((path / name).is_file() for name in required) and bool(weights)


def local_model_preflight(
    *,
    vlm_path: Path,
    embedding_path: Path,
    embedding_source_path: Path,
    disk_root: Path,
) -> dict[str, Any]:
    """Return public hardware facts without loading weights."""

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
    disk = shutil.disk_usage(disk_root)
    vlm_present = _model_files_present(vlm_path)
    embedding_present = _model_files_present(
        embedding_path,
        embedding=True,
    )
    source_file = (
        embedding_source_path
        / "src"
        / "models"
        / "qwen3_vl_embedding.py"
    )
    source_present = source_file.is_file()
    warnings: list[str] = []
    if not cuda_available:
        warnings.append("未检测到可用 CUDA。")
    if free_vram < RECOMMENDED_FREE_VRAM_BYTES:
        warnings.append(
            "当前可用显存低于基于历史双模型峰值给出的 16 GiB 推荐值。"
        )
    if not vlm_present:
        warnings.append(f"缺少本地模型：{VLM_MODEL_ID}。")
    if not embedding_present:
        warnings.append(f"缺少本地模型：{EMBEDDING_MODEL_ID}。")
    if not source_present:
        warnings.append(
            "缺少冻结的 Qwen3-VL-Embedding 推理源码；不会自动下载。"
        )
    hard_blockers: list[dict[str, str]] = []
    if not cuda_available:
        hard_blockers.append(
            {
                "code": "LOCAL_CUDA_UNAVAILABLE",
                "label": "CUDA 不可用",
            }
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
    free_gib = float(free_vram) / 1024**3
    historical_sum_gib = float(
        OBSERVED_VLM_PEAK_BYTES + OBSERVED_EMBEDDING_PEAK_BYTES
    ) / 1024**3
    missing_labels = [
        item["label"]
        for item in hard_blockers
        if item["code"] != "LOCAL_CUDA_UNAVAILABLE"
    ]
    conclusion_parts = [
        "当前电脑暂时无法完整加载本地模型。"
        if hard_blockers
        or free_vram < RECOMMENDED_FREE_VRAM_BYTES
        else "当前环境具备本地双模型加载条件。",
    ]
    if cuda_available:
        conclusion_parts.append(
            f"检测到可用显存约 {free_gib:.1f} GiB，"
            f"双模型历史峰值合计约 {historical_sum_gib:.2f} GiB。"
        )
    if missing_labels:
        conclusion_parts.append("同时" + "、".join(missing_labels) + "。")
    if hard_blockers or free_vram < RECOMMENDED_FREE_VRAM_BYTES:
        conclusion_parts.append(
            "建议使用百炼云端模式或服务器映射课程演示模式。"
        )
    return {
        "status": "ready_to_attempt" if not warnings else "attention_required",
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
        "disk": {
            "root": str(disk_root.resolve()),
            "free_bytes": int(disk.free),
        },
        "weights": {
            "vlm": {
                "model_id": VLM_MODEL_ID,
                "path": str(vlm_path.resolve()),
                "present": vlm_present,
            },
            "embedding": {
                "model_id": EMBEDDING_MODEL_ID,
                "path": str(embedding_path.resolve()),
                "present": embedding_present,
                "source_path": str(embedding_source_path.resolve()),
                "source_present": source_present,
            },
        },
        "historical_vram_evidence": {
            "vlm_peak_bytes": OBSERVED_VLM_PEAK_BYTES,
            "embedding_peak_bytes": OBSERVED_EMBEDDING_PEAK_BYTES,
            "sum_bytes": (
                OBSERVED_VLM_PEAK_BYTES + OBSERVED_EMBEDDING_PEAK_BYTES
            ),
            "recommended_free_vram_bytes": RECOMMENDED_FREE_VRAM_BYTES,
            "conservative_free_vram_bytes": CONSERVATIVE_FREE_VRAM_BYTES,
            "basis": "accepted independent runtime peaks plus coexistence margin",
        },
        "can_attempt": bool(
            cuda_available
            and vlm_present
            and embedding_present
            and source_present
        ),
        "recommended": bool(
            cuda_available
            and free_vram >= RECOMMENDED_FREE_VRAM_BYTES
            and vlm_present
            and embedding_present
            and source_present
        ),
        "warnings": warnings,
        "hard_blockers": hard_blockers,
        "low_vram_requires_override": bool(
            cuda_available
            and free_vram < RECOMMENDED_FREE_VRAM_BYTES
        ),
        "diagnostic_summary": {
            "cuda": "available" if cuda_available else "unavailable",
            "gpu_name": device_name,
            "total_vram_gib": round(float(total_vram) / 1024**3, 2),
            "free_vram_gib": round(free_gib, 2),
            "vlm_weights": "found" if vlm_present else "missing",
            "embedding_weights": (
                "found" if embedding_present else "missing"
            ),
            "embedding_source": (
                "found" if source_present else "missing"
            ),
            "historical_dual_peak_gib": round(historical_sum_gib, 2),
            "recommended_free_vram_gib": 16,
            "recommended_gpu_vram_gib": 24,
        },
        "conclusion": "".join(conclusion_parts),
        "recommended_model_paths": {
            "vlm": str(vlm_path.resolve()),
            "embedding": str(embedding_path.resolve()),
            "embedding_source": str(embedding_source_path.resolve()),
        },
        "automatic_download": False,
        "quantization_changed": False,
        "index_reuse_policy": {
            "schema_version": LOCAL_INDEX_IDENTITY_SCHEMA,
            "requires_exact_identity": True,
            "required_fields": [
                "model_id",
                "model_revision",
                "weights_sha256",
                "source_commit",
                "preprocess_version",
                "dimension",
                "normalization",
                "metric",
                "index_schema_version",
                "serialization_version",
                "provider_identity",
            ],
            "self_hosted_reuse_allowed": False,
            "conclusion": (
                "当前本地 Embedding 权重或冻结源码不完整，不能证明与服务器映射索引身份一致；"
                "本地与服务器索引必须保持独立。"
                if not embedding_present or not source_present
                else "仅在索引身份文件全部字段精确一致时允许复用。"
            ),
        },
    }


def local_index_reuse_contract(
    *,
    preflight: dict[str, Any],
    index_path: Path,
) -> dict[str, Any]:
    """Decide reuse from a complete persisted identity, never from model name."""

    policy = dict(preflight.get("index_reuse_policy") or {})
    required = list(policy.get("required_fields") or [])
    identity_path = index_path / "identity.json"
    reasons: list[str] = []
    identity: dict[str, Any] = {}
    if not preflight.get("can_attempt"):
        reasons.append("LOCAL_RUNTIME_INCOMPLETE")
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
        "model_id": EMBEDDING_MODEL_ID,
        "model_revision": EMBEDDING_REVISION,
        "source_commit": EMBEDDING_SOURCE_COMMIT,
        "dimension": EMBEDDING_DIMENSION,
        "normalization": "l2",
        "metric": "inner_product",
        "provider_identity": "local",
    }
    mismatches = [
        field
        for field, expected_value in expected.items()
        if identity.get(field) != expected_value
    ]
    if mismatches:
        reasons.append("INDEX_IDENTITY_MISMATCH")
    reusable = not reasons and not missing and not mismatches
    return {
        "schema_version": LOCAL_INDEX_IDENTITY_SCHEMA,
        "reusable": reusable,
        "index_exists": (index_path / "index.faiss").is_file(),
        "identity_present": identity_path.is_file(),
        "required_fields": required,
        "missing_fields": missing,
        "mismatched_fields": mismatches,
        "reason_codes": reasons,
        "provider_identity": "local",
        "self_hosted_index_reused": False,
        "conclusion": (
            "本地索引身份合同精确匹配，可以只读复用。"
            if reusable
            else "本地索引身份未获精确证明；不会复用服务器映射索引。"
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
            self.source_path
            / "src"
            / "models"
            / "qwen3_vl_embedding.py"
        )
        if not module_path.is_file():
            raise FileNotFoundError(
                "local_qwen3_vl_embedding_source_missing"
            )
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
