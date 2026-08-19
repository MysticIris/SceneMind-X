"""Lightweight process, host-memory and GPU sampling for one API run."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def memory_snapshot() -> dict[str, Any]:
    """Execute the memory snapshot operation."""
    process = psutil.Process()
    memory = process.memory_info()
    full = process.memory_full_info()
    system = psutil.virtual_memory()
    swap = psutil.swap_memory()
    values: dict[str, Any] = {
        "process_pid": process.pid,
        "process_rss_bytes": int(memory.rss),
        "process_vms_bytes": int(memory.vms),
        "process_uss_bytes": int(getattr(full, "uss", memory.rss)),
        "system_total_bytes": int(system.total),
        "system_available_bytes": int(system.available),
        "system_used_bytes": int(system.used),
        "system_percent": float(system.percent),
        "swap_total_bytes": int(swap.total),
        "swap_used_bytes": int(swap.used),
    }
    for name in ("peak_wset", "peak_pagefile", "private"):
        if hasattr(memory, name):
            values[f"process_{name}_bytes"] = int(getattr(memory, name))
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        values["process_peak_rss_bytes"] = peak * 1024
    except (ImportError, AttributeError):
        if "process_peak_wset_bytes" in values:
            values["process_peak_rss_bytes"] = values["process_peak_wset_bytes"]
    return values


def gpu_memory_snapshot() -> dict[str, Any]:
    """Execute the gpu memory snapshot operation."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "free_bytes": int(free),
                "total_bytes": int(total),
            }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": False}


class RuntimeResourceMonitor:
    """Provide runtime resource monitor behavior."""
    def __init__(self, metrics_root: Path, interval_seconds: float = 2.0) -> None:
        self.metrics_root = metrics_root
        self.interval_seconds = interval_seconds
        self.samples_path = metrics_root / "resources.jsonl"
        self.summary_path = metrics_root / "resource_summary.json"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []

    def _summary(self, *, status: str) -> dict[str, Any]:
        memory_rows = [sample["memory"] for sample in self._samples]
        gpu_rows = [sample["gpu"] for sample in self._samples if sample["gpu"].get("available")]
        return {
            "status": status,
            "sample_count": len(self._samples),
            "interval_seconds": self.interval_seconds,
            "started_at": self._samples[0]["sampled_at"],
            "latest_sampled_at": self._samples[-1]["sampled_at"],
            "ended_at": self._samples[-1]["sampled_at"] if status == "stopped" else None,
            "process_rss_peak_bytes": max(row["process_rss_bytes"] for row in memory_rows),
            "process_uss_peak_bytes": max(row["process_uss_bytes"] for row in memory_rows),
            "system_used_peak_bytes": max(row["system_used_bytes"] for row in memory_rows),
            "system_available_min_bytes": min(row["system_available_bytes"] for row in memory_rows),
            "swap_used_peak_bytes": max(row["swap_used_bytes"] for row in memory_rows),
            "gpu_allocated_peak_bytes": max((row["allocated_bytes"] for row in gpu_rows), default=None),
            "gpu_reserved_peak_bytes": max((row["reserved_bytes"] for row in gpu_rows), default=None),
            "gpu_total_bytes": gpu_rows[0]["total_bytes"] if gpu_rows else None,
        }

    def _write_summary(self, *, status: str) -> dict[str, Any]:
        summary = self._summary(status=status)
        _write_json_atomic(self.summary_path, summary)
        return summary

    def _sample(self) -> dict[str, Any]:
        sample = {"sampled_at": _now_iso(), "memory": memory_snapshot(), "gpu": gpu_memory_snapshot()}
        self.metrics_root.mkdir(parents=True, exist_ok=True)
        with self.samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._samples.append(sample)
        self._write_summary(status="running")
        return sample

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        """Execute the start operation."""
        if self._thread is not None:
            return
        self._sample()
        self._thread = threading.Thread(target=self._run, name="phase1-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """Execute the stop operation."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self._sample()
        return self._write_summary(status="stopped")
