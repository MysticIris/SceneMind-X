"""Durable and strictly bounded user-confirmed index backfill tasks."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .product_store import now_iso


MAX_TASK_ASSETS = 64
MAX_ATTEMPTS_PER_ASSET = 3
ALLOWED_SCOPES = {"asset", "library", "all_user_assets"}
TERMINAL_TASK_STATES = {
    "SUCCESS", "CANCELLED", "BILLING_STOPPED", "HARD_FAILED",
    "IDENTITY_MISMATCH",
}
HARD_STOP_CATEGORIES = {
    "billing_or_quota": "BILLING_STOPPED",
    "invalid_api_key": "HARD_FAILED",
    "permission": "HARD_FAILED",
}


class ProviderIndexBackfillManager:
    """Checkpoint after every asset without changing Provider identity."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._workers: dict[str, threading.Thread] = {}
        for path in self.root.glob("*.json"):
            task = self._read_path(path)
            if task and task.get("status") in {"RUNNING", "ENCODING"}:
                task["status"] = "PENDING"
                task["interrupted"] = True
                self._write(task)

    def _path(self, task_id: str) -> Path:
        return self.root / f"{task_id}.json"

    @staticmethod
    def _read_path(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _write(self, task: dict[str, Any]) -> dict[str, Any]:
        task["updated_at"] = now_iso()
        path = self._path(str(task["task_id"]))
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
        ) as handle:
            json.dump(task, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
        return task

    @staticmethod
    def _counts(task: dict[str, Any]) -> dict[str, int]:
        states = [str(item.get("status")) for item in task.get("items", [])]
        return {
            "total": len(states),
            "completed": sum(
                state in {"SUCCESS", "SKIPPED_EXISTING"} for state in states
            ),
            "skipped": states.count("SKIPPED_EXISTING"),
            "failed": sum(
                state in {"RETRYABLE_FAILED", "HARD_FAILED"} for state in states
            ),
            "remaining": sum(
                state in {"PENDING", "ENCODING", "RETRYABLE_FAILED"}
                for state in states
            ),
        }

    def create(
        self,
        *,
        asset_ids: list[str],
        identity: dict[str, Any],
        scope: str,
        metadata: dict[str, Any],
        confirmed_by_user: bool,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """Create the requested value."""
        unique_ids = list(dict.fromkeys(str(value) for value in asset_ids))
        normalized_operation_id = str(operation_id or "").strip() or str(
            uuid.uuid4()
        )
        if not confirmed_by_user:
            raise ValueError("explicit_user_confirmation_required")
        if scope not in ALLOWED_SCOPES:
            raise ValueError("invalid_backfill_scope")
        if not unique_ids or len(unique_ids) > MAX_TASK_ASSETS:
            raise ValueError("backfill_batch_must_contain_1_to_64_assets")
        if len(normalized_operation_id) > 128:
            raise ValueError("operation_id_too_long")
        with self._lock:
            for path in self.root.glob("*.json"):
                existing = self._read_path(path)
                if (
                    existing
                    and existing.get("operation_id") == normalized_operation_id
                    and existing.get("scope") == scope
                    and existing.get("provider_identity_sha256")
                    == identity.get("identity_sha256")
                ):
                    existing["progress"] = self._counts(existing)
                    return existing
        task = {
            "schema_version": "scenemindx_provider_index_backfill_task_v1",
            "task_id": str(uuid.uuid4()),
            "operation_id": normalized_operation_id,
            "status": "PENDING",
            "scope": scope,
            "provider_identity": dict(identity),
            "provider_identity_sha256": identity.get("identity_sha256"),
            "metadata": {**dict(metadata), "confirmed_by_user": True},
            "cancel_requested": False,
            "current_asset_id": None,
            "items": [
                {"asset_id": value, "status": "PENDING", "attempts": 0}
                for value in unique_ids
            ],
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        task["progress"] = self._counts(task)
        with self._lock:
            return dict(self._write(task))

    def get(self, task_id: str) -> dict[str, Any]:
        """Return the requested value."""
        with self._lock:
            task = self._read_path(self._path(task_id))
            if task is None:
                raise KeyError(task_id)
            task["progress"] = self._counts(task)
            return task

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        """List the requested value."""
        tasks = [
            task for path in self.root.glob("*.json")
            if (task := self._read_path(path)) is not None
        ]
        tasks.sort(key=lambda value: str(value.get("created_at")), reverse=True)
        for task in tasks:
            task["progress"] = self._counts(task)
        return tasks[:limit]

    def cancel(self, task_id: str) -> dict[str, Any]:
        """Execute the cancel operation."""
        with self._lock:
            task = self.get(task_id)
            if task.get("status") not in TERMINAL_TASK_STATES:
                task["cancel_requested"] = True
                self._write(task)
            return task

    def start(
        self,
        task_id: str,
        *,
        runner: Callable[[str], dict[str, Any]],
        current_identity_sha256: Callable[[], str | None],
    ) -> dict[str, Any]:
        """Execute the start operation."""
        with self._lock:
            task = self.get(task_id)
            worker = self._workers.get(task_id)
            if worker is not None and worker.is_alive():
                return task
            if task.get("status") == "SUCCESS":
                return task
            if task.get("status") in {"BILLING_STOPPED", "HARD_FAILED"}:
                for item in task.get("items", []):
                    if item.get("status") in {
                        "HARD_FAILED", "RETRYABLE_FAILED"
                    }:
                        item["status"] = "PENDING"
                        item["attempts"] = 0
                        item["resumed_at"] = now_iso()
            task["cancel_requested"] = False
            task["status"] = "RUNNING"
            self._write(task)
            worker = threading.Thread(
                target=self._run,
                args=(task_id, runner, current_identity_sha256),
                daemon=True,
                name=f"provider-index-backfill-{task_id[:8]}",
            )
            self._workers[task_id] = worker
            worker.start()
            return task

    def _run(
        self,
        task_id: str,
        runner: Callable[[str], dict[str, Any]],
        current_identity_sha256: Callable[[], str | None],
    ) -> None:
        while True:
            with self._lock:
                task = self.get(task_id)
                if task.get("cancel_requested"):
                    task["status"] = "CANCELLED"
                    task["current_asset_id"] = None
                    self._write(task)
                    return
                if current_identity_sha256() != task.get(
                    "provider_identity_sha256"
                ):
                    task["status"] = "IDENTITY_MISMATCH"
                    task["current_asset_id"] = None
                    self._write(task)
                    return
                item = next((
                    value for value in task.get("items", [])
                    if value.get("status") == "PENDING"
                    or (
                        value.get("status") == "RETRYABLE_FAILED"
                        and int(value.get("attempts", 0))
                        < MAX_ATTEMPTS_PER_ASSET
                    )
                ), None)
                if item is None:
                    exhausted = any(
                        value.get("status") == "RETRYABLE_FAILED"
                        for value in task.get("items", [])
                    )
                    task["status"] = "HARD_FAILED" if exhausted else "SUCCESS"
                    task["current_asset_id"] = None
                    self._write(task)
                    return
                attempts = int(item.get("attempts", 0))
                if attempts:
                    time.sleep(0.25 * (2 ** (attempts - 1)))
                item["status"] = "ENCODING"
                item["attempts"] = attempts + 1
                item["started_at"] = now_iso()
                task["current_asset_id"] = item["asset_id"]
                self._write(task)
            try:
                result = runner(str(item["asset_id"]))
            except Exception as exc:
                result = {
                    "status": "failed",
                    "failure": {
                        "category": "unknown",
                        "code": type(exc).__name__,
                        "retryable": False,
                        "stop_retries": True,
                        "public_message": "索引补齐未完成，已保留图片和已有进度。",
                    },
                }
            with self._lock:
                task = self.get(task_id)
                current = next(
                    value for value in task["items"]
                    if value["asset_id"] == item["asset_id"]
                )
                failure = dict(result.get("failure") or {})
                if result.get("status") == "completed":
                    current["status"] = (
                        "SKIPPED_EXISTING"
                        if int(result.get("encoded", 0) or 0) == 0
                        else "SUCCESS"
                    )
                elif result.get("status") == "waiting_for_ready_provider":
                    current["status"] = "HARD_FAILED"
                    failure = {
                        "category": "provider_not_ready",
                        "code": "PROVIDER_NOT_READY",
                        "retryable": False,
                        "stop_retries": True,
                        "public_message": "当前 Embedding Provider 尚未就绪。",
                    }
                else:
                    current["status"] = (
                        "RETRYABLE_FAILED"
                        if failure.get("retryable")
                        else "HARD_FAILED"
                    )
                current["result"] = {
                    "status": result.get("status"),
                    "encoded": int(result.get("encoded", 0) or 0),
                    "appended": int(result.get("appended", 0) or 0),
                    "rebuilt": bool(result.get("rebuilt", False)),
                }
                current["failure"] = failure or None
                current["finished_at"] = now_iso()
                task["current_asset_id"] = None
                stop_state = HARD_STOP_CATEGORIES.get(
                    str(failure.get("category"))
                )
                if stop_state or failure.get("stop_retries"):
                    task["status"] = stop_state or "HARD_FAILED"
                    task["public_message"] = failure.get("public_message")
                    self._write(task)
                    return
                self._write(task)
