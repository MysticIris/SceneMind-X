"""Atomic per-request execution traces."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .resources import memory_snapshot


def now_iso() -> str:
    """Execute the now iso operation."""
    return datetime.now(timezone.utc).astimezone().isoformat()


def gpu_snapshot() -> dict[str, Any]:
    """Execute the gpu snapshot operation."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": False}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        persisted = {key: item for key, item in value.items() if not key.startswith("_")}
        json.dump(persisted, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


class TraceStore:
    """Provide trace store behavior."""
    def __init__(self, root: Path, git_commit: str) -> None:
        self.root = root
        self.git_commit = git_commit

    def start(
        self,
        task_type: str,
        image_ids: list[str],
        *,
        model: str | None,
        model_revision: str | None,
        prompt_version: str,
        schema_version: str,
        services: list[str],
        prompt_sha256: str | None = None,
        input_facts_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Execute the start operation."""
        request_id = str(uuid.uuid4())
        trace = {
            "request_id": request_id,
            "task_type": task_type,
            "image_ids": image_ids,
            "model": model,
            "model_revision": model_revision,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "input_facts_sha256": input_facts_sha256,
            "schema_version": schema_version,
            "services": services,
            "started_at": now_iso(),
            "ended_at": None,
            "latency_seconds": None,
            "gpu_start": gpu_snapshot(),
            "gpu_end": None,
            "memory_start": memory_snapshot(),
            "memory_end": None,
            "status": "running",
            "error_type": None,
            "error": None,
            "output_path": None,
            "git_commit": self.git_commit,
            "_started_perf_counter": time.perf_counter(),
        }
        _write_json_atomic(self.root / f"{request_id}.json", trace)
        return trace

    def finish(self, trace: dict[str, Any], *, status: str, output_path: str | None = None, error: Exception | None = None) -> dict[str, Any]:
        """Execute the finish operation."""
        started = float(trace.pop("_started_perf_counter"))
        trace.update(
            {
                "ended_at": now_iso(),
                "latency_seconds": time.perf_counter() - started,
                "gpu_end": gpu_snapshot(),
                "memory_end": memory_snapshot(),
                "status": status,
                "error_type": type(error).__name__ if error else None,
                "error": str(error) if error else None,
                "output_path": output_path,
            }
        )
        _write_json_atomic(self.root / f"{trace['request_id']}.json", trace)
        return trace

    def get(self, request_id: str) -> dict[str, Any]:
        """Return the requested value."""
        if not request_id or any(character not in "0123456789abcdef-" for character in request_id.lower()):
            raise KeyError(request_id)
        path = self.root / f"{request_id}.json"
        if not path.is_file():
            raise KeyError(request_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def annotate(self, request_id: str, **fields: Any) -> dict[str, Any]:
        """Attach bounded post-validation metadata to an existing trace."""

        trace = self.get(request_id)
        trace.update(fields)
        _write_json_atomic(self.root / f"{request_id}.json", trace)
        return trace
