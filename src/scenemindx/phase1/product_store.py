"""Small durable store for the Phase 5 product skeleton.

The store is deliberately JSON based: it keeps the demo portable, auditable,
and free of a new database dependency while retaining records across restarts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .multi_library import SYSTEM_LIBRARY_IDS, SYSTEM_RESERVED_NAMES
from .session_ledger import (
    append_binding_event,
    binding_snapshot,
    ensure_ledger,
    image_index,
    project_conversation_state,
    validate_model_bindings,
)

DEFAULT_LIBRARY_ID = "default"
LEGACY_DEFAULT_LIBRARY_IDS = frozenset({"local"})


def now_iso() -> str:
    """Execute the now iso operation."""
    return datetime.now(timezone.utc).astimezone().isoformat()


_FOCUS_UNSET = object()
CHAT_IMAGE_SLOT_LIMIT = 5


def next_vacant_image_label(
    occupied_labels: Iterable[str],
    *,
    limit: int = CHAT_IMAGE_SLOT_LIMIT,
) -> str:
    """Return the smallest unoccupied Chat IMG slot.

    Removed, unlocked bindings are deliberately not passed as occupied.  Their
    ledger events remain historical evidence, while the bounded active slot can
    be reused without renumbering any other active binding.
    """

    occupied_indices = {
        image_index(str(label))
        for label in occupied_labels
    }
    for index in range(1, limit + 1):
        if index not in occupied_indices:
            return f"IMG_{index}"
    raise ValueError("chat_context_maximum_is_5_images")


class WorkspaceVersionConflict(RuntimeError):
    """Raised when a client tries to replace a newer workspace snapshot."""


class LibraryMutationForbidden(ValueError):
    """Raised when a normal product route targets a system-locked library."""

    def __init__(self, public_message: str) -> None:
        super().__init__("system_library_locked")
        self.public_message = public_message


class ProductStore:
    """Provide product store behavior."""
    def __init__(
        self,
        root: Path,
        *,
        assets_root: Path | None = None,
        session_assets_root: Path | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets_root = (assets_root or root / "uploads").resolve()
        self.assets_root.mkdir(parents=True, exist_ok=True)
        self.session_assets_root = (
            session_assets_root or root / "session_uploads"
        ).resolve()
        self.session_assets_root.mkdir(parents=True, exist_ok=True)
        self._paths = {
            "tasks": root / "tasks.json",
            "feedback": root / "feedback.json",
            "sessions": root / "vqa_sessions.json",
            "libraries": root / "libraries.json",
            "assets": root / "assets.json",
            "session_assets": root / "session_assets.json",
            "analyses": root / "local_analyses.json",
            "workspaces": root / "function_workspaces.json",
        }
        self._workspace_store_lock = threading.RLock()
        self._workspace_locks = {
            kind: threading.RLock()
            for kind in ("generation", "retrieval", "compare")
        }
        self._session_store_lock = threading.RLock()
        self._migrate_legacy_managed_assets()

    def _read(self, kind: str) -> list[dict[str, Any]]:
        path = self._paths[kind]
        if not path.is_file():
            return []
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return value if isinstance(value, list) else []

    def _write(self, kind: str, values: list[dict[str, Any]]) -> None:
        path = self._paths[kind]
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
            json.dump(values, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)

    def _migrate_legacy_managed_assets(self) -> None:
        """Copy legacy product uploads into the configured managed store.

        The old files are deliberately retained.  This makes the migration
        recoverable while ensuring active records no longer depend on the
        previous runtime-specific upload directory.
        """

        legacy_root = (self.root / "uploads").resolve()
        if legacy_root == self.assets_root or not legacy_root.is_dir():
            return
        values = self._read("assets")
        changed = False
        for item in values:
            source = Path(str(item.get("path") or ""))
            if not source.is_file():
                continue
            resolved_source = source.resolve()
            if not resolved_source.is_relative_to(legacy_root):
                continue
            digest = str(item.get("sha256") or "").lower()
            if len(digest) != 64:
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
            extension = (
                str(item.get("original_extension") or "").lower()
                or source.suffix.lower()
                or ".bin"
            )
            destination = (
                self.assets_root
                / digest[:2]
                / digest
                / f"{digest}{extension}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
            item.update(
                path=str(destination),
                storage_key=destination.relative_to(
                    self.assets_root.parent
                ).as_posix(),
                sha256=digest,
                original_filename=(
                    item.get("original_filename")
                    or item.get("image_id")
                    or source.name
                ),
                original_extension=extension,
                storage_migrated_at=now_iso(),
                legacy_source_retained=True,
            )
            changed = True
        if changed:
            self._write("assets", values)

    def bootstrap_default_assets(
        self,
        *,
        project_root: Path,
    ) -> int:
        """Register packaged public assets for a fresh runtime.

        Runtime JSON is intentionally not versioned.  A fresh public clone
        therefore reconstructs the Default Library from the immutable image
        files shipped under ``data/user_assets``.  Existing runtime records
        always win, so this never rewrites a user's library state.
        """

        if self._read("assets"):
            return 0
        supported_extensions = {
            ".avif",
            ".bmp",
            ".gif",
            ".jpeg",
            ".jpg",
            ".png",
            ".webp",
        }
        values: list[dict[str, Any]] = []
        seen_digests: set[str] = set()
        for source in sorted(self.assets_root.rglob("*")):
            if (
                not source.is_file()
                or source.suffix.lower() not in supported_extensions
            ):
                continue
            content = source.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest in seen_digests:
                continue
            seen_digests.add(digest)
            width = None
            height = None
            try:
                from PIL import Image

                with Image.open(source) as image:
                    width, height = image.size
            except Exception:
                pass
            safe_name = source.name
            values.append(
                {
                    "asset_id": digest[:16],
                    "library_id": DEFAULT_LIBRARY_ID,
                    "image_id": safe_name,
                    "sha256": digest,
                    "path": source.relative_to(project_root).as_posix(),
                    "storage_key": source.relative_to(
                        self.assets_root.parent
                    ).as_posix(),
                    "original_filename": safe_name,
                    "original_extension": source.suffix.lower(),
                    "mime_type": mimetypes.guess_type(safe_name)[0]
                    or "application/octet-stream",
                    "bytes": len(content),
                    "width": width,
                    "height": height,
                    "import_status": "completed",
                    "created_at": now_iso(),
                }
            )
        if values:
            self._write("assets", values)
        return len(values)

    def create_task(self, task_type: str, total: int, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create task."""
        task = {
            "task_id": str(uuid.uuid4()),
            "task_type": task_type,
            "status": "queued",
            "total": total,
            "completed": 0,
            "failed": 0,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "metadata": metadata or {},
            "errors": [],
        }
        values = self._read("tasks")
        values.append(task)
        self._write("tasks", values)
        return task

    def update_task(self, task_id: str, **changes: Any) -> dict[str, Any]:
        """Update task."""
        values = self._read("tasks")
        for task in values:
            if task["task_id"] == task_id:
                task.update(changes, updated_at=now_iso())
                self._write("tasks", values)
                return task
        raise KeyError(task_id)

    def tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        """Execute the tasks operation."""
        return list(reversed(self._read("tasks")))[:limit]

    def add_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the add feedback operation."""
        item = {"feedback_id": str(uuid.uuid4()), "created_at": now_iso(), "status": "queued", **payload}
        values = self._read("feedback")
        values.append(item)
        self._write("feedback", values)
        return item

    def update_feedback(self, feedback_id: str, status: str) -> dict[str, Any]:
        """Update feedback."""
        if status not in {"accepted", "rejected", "queued"}:
            raise ValueError("invalid_feedback_status")
        values = self._read("feedback")
        for item in values:
            if item["feedback_id"] == feedback_id:
                item.update(status=status, reviewed_at=now_iso())
                self._write("feedback", values)
                return item
        raise KeyError(feedback_id)

    def feedback(self, limit: int = 100) -> list[dict[str, Any]]:
        """Execute the feedback operation."""
        return list(reversed(self._read("feedback")))[:limit]

    def save_session(self, session: dict[str, Any]) -> dict[str, Any]:
        """Save session."""
        session = self.normalize_session(session)
        values = self._read("sessions")
        existing = next((item for item in values if item.get("conversation_id") == session.get("conversation_id")), None)
        if existing is None:
            values.append(session)
        else:
            existing.update(session, updated_at=now_iso())
        self._write("sessions", values)
        return session

    def get_session(self, conversation_id: str) -> dict[str, Any] | None:
        """Return session."""
        item = next((item for item in self._read("sessions") if item.get("conversation_id") == conversation_id), None)
        return self.normalize_session(item) if item is not None else None

    def sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        """Execute the sessions operation."""
        return [self.normalize_session(item) for item in reversed(self._read("sessions"))][:limit]

    def delete_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Remove one conversation and its explicitly session-scoped uploads."""

        values = self._read("sessions")
        target = next(
            (
                item
                for item in values
                if str(item.get("conversation_id")) == conversation_id
            ),
            None,
        )
        if target is None:
            raise KeyError(conversation_id)
        remaining = [
            item
            for item in values
            if str(item.get("conversation_id")) != conversation_id
        ]
        self._write("sessions", remaining)
        session_assets = self._read("session_assets")
        removed_session_assets = [
            item
            for item in session_assets
            if item.get("conversation_id") == conversation_id
        ]
        for item in removed_session_assets:
            path = Path(str(item.get("path", "")))
            if (
                path.is_file()
                and self.session_assets_root.resolve() in path.resolve().parents
            ):
                path.unlink()
        self._write(
            "session_assets",
            [
                item
                for item in session_assets
                if item.get("conversation_id") != conversation_id
            ],
        )
        return {
            "conversation_id": conversation_id,
            "deleted": True,
            "deleted_at": now_iso(),
            "removed_message_count": len(target.get("messages", [])),
            "removed_active_asset_binding_count": len(
                target.get("active_assets", [])
            ),
            "remaining_conversation_count": len(remaining),
            "library_assets_modified": False,
            "retrieval_index_modified": False,
            "historical_traces_modified": False,
            "removed_session_asset_count": len(removed_session_assets),
        }

    @staticmethod
    def normalize_session(session: dict[str, Any]) -> dict[str, Any]:
        """Keep legacy single-image sessions readable under the Phase 5.2 model."""

        value = dict(session)
        legacy_asset_id = value.get("asset_id")
        if "active_assets" not in value:
            value["active_assets"] = (
                [
                    {
                        "ref": f"library:{legacy_asset_id}",
                        "source": "library",
                        "asset_id": legacy_asset_id,
                        "image_id": legacy_asset_id,
                        "order": 1,
                    }
                ]
                if legacy_asset_id
                else []
            )
        defaults: dict[str, Any] = {
            "title": "未命名对话",
            "selected_asset_refs": [item.get("ref") for item in value["active_assets"] if item.get("ref")],
            "messages": [],
            "context_summary": "",
            "tool_calls": [],
            "tool_results": [],
            "generated_content": [],
            "retrieval_results": [],
            "comparison_results": [],
            "ranking_results": [],
        }
        for key, default in defaults.items():
            value.setdefault(key, default)
        chat_state = value.get("chat_state") if isinstance(value.get("chat_state"), dict) else {}
        bindings = [
            dict(item)
            for item in chat_state.get("asset_bindings", [])
            if isinstance(item, dict) and item.get("image_label")
        ]
        by_ref = {
            str(item.get("ref")): item
            for item in bindings
            if item.get("ref")
        }
        locked_labels = {
            str(label)
            for label in chat_state.get("locked_image_labels", [])
            if str(label).startswith("IMG_")
        }
        lock_records = {
            str(label): dict(record)
            for label, record in dict(
                chat_state.get("image_lock_records", {})
            ).items()
            if isinstance(record, dict)
        }
        if "locked_image_labels" not in chat_state and value["messages"]:
            visual_tasks = {
                "generate",
                "generate_content_from_images",
                "rank",
                "compare",
                "compare_or_rank_images",
                "recommend",
                "retrieve",
                "search_images",
                "describe",
                "explain",
                "vqa",
            }
            used_refs = {
                str(ref)
                for message in value["messages"]
                if str(message.get("task_type") or "") in visual_tasks
                for ref in message.get("asset_refs", [])
            }
            for ref in used_refs:
                binding = by_ref.get(ref)
                if binding is not None:
                    label = str(binding["image_label"])
                    locked_labels.add(label)
                    lock_records.setdefault(
                        label,
                        {
                            "locked_at": value.get("updated_at") or now_iso(),
                            "first_use_kind": "legacy_visual_turn",
                            "trace_id": None,
                        },
                    )
        active_refs = {
            str(item.get("ref"))
            for item in value["active_assets"]
            if item.get("ref")
        }
        historical_bindings = [
            {
                **item,
                "status": "removed",
                "active": False,
                "locked": (
                    str(item.get("image_label")) in locked_labels
                ),
                "image_sha256": str(
                    item.get("image_sha256")
                    or item.get("sha256")
                    or ""
                ),
                "deactivated_at": (
                    item.get("deactivated_at")
                    or item.get("removed_at")
                    or value.get("updated_at")
                ),
            }
            for item in bindings
            if str(item.get("ref")) not in active_refs
        ]
        occupied_labels = {
            str(item["image_label"])
            for item in bindings
            if (
                str(item.get("ref")) in active_refs
                or str(item.get("image_label")) in locked_labels
            )
        }
        normalized_active = []
        normalized_bindings = list(historical_bindings)
        for asset in value["active_assets"]:
            item = dict(asset)
            ref = str(item.get("ref") or "")
            binding = by_ref.get(ref)
            existing_label = (
                str(binding.get("image_label"))
                if binding is not None
                else None
            )
            if existing_label is None:
                image_label = next_vacant_image_label(occupied_labels)
                normalized_bindings = [
                    historical
                    for historical in normalized_bindings
                    if not (
                        str(historical.get("image_label")) == image_label
                        and not bool(historical.get("locked"))
                        and historical.get("status") != "active"
                    )
                ]
            else:
                image_label = existing_label
            occupied_labels.add(image_label)
            binding = {
                **(binding or {}),
                **item,
                "image_label": image_label,
                "status": "active",
                "active": True,
                "locked": image_label in locked_labels,
                "image_sha256": str(
                    item.get("image_sha256")
                    or item.get("sha256")
                    or (binding or {}).get("image_sha256")
                    or (binding or {}).get("sha256")
                    or ""
                ),
                "introduced_turn_id": (
                    (binding or {}).get("introduced_turn_id")
                ),
                "introduced_at": (
                    (binding or {}).get("introduced_at")
                    or (binding or {}).get("added_at")
                    or value.get("created_at")
                    or now_iso()
                ),
                "added_at": (
                    (binding or {}).get("added_at")
                    or value.get("created_at")
                    or now_iso()
                ),
                "removed_at": None,
                "deactivated_at": None,
            }
            normalized_bindings.append(binding)
            item["image_label"] = binding["image_label"]
            item["order"] = int(str(binding["image_label"]).split("_")[-1])
            item["locked"] = bool(binding["locked"])
            normalized_active.append(item)
        bindings = normalized_bindings
        value["active_assets"] = normalized_active
        focus = chat_state.get("current_focus_label")
        active_labels = [str(item["image_label"]) for item in normalized_active]
        if focus not in active_labels:
            focus = None
        summary = chat_state.get("summary") if isinstance(chat_state.get("summary"), dict) else {}
        summary.setdefault("confirmed_facts", [])
        summary.setdefault("asset_notes", {})
        summary.setdefault("current_goal", "")
        summary.setdefault("unresolved_questions", [])
        value["chat_state"] = {
            "schema_version": "scenemindx_single_thread_multiturn_chat_state_v5",
            "asset_bindings": sorted(
                bindings,
                key=lambda item: int(str(item["image_label"]).split("_")[-1]),
            ),
            "current_focus_label": focus,
            "locked_image_labels": sorted(
                locked_labels,
                key=lambda label: int(label.split("_")[-1]),
            ),
            "image_lock_records": lock_records,
            "recent_mentioned_labels": list(chat_state.get("recent_mentioned_labels", [])),
            "newly_added_labels": list(chat_state.get("newly_added_labels", [])),
            "removed_labels": list(chat_state.get("removed_labels", [])),
            "summary": summary,
            "last_asset_change": chat_state.get("last_asset_change"),
            "last_tool_call": chat_state.get("last_tool_call"),
            "last_search_results": list(
                chat_state.get("last_search_results", [])
            ),
            "selected_tool_images": list(
                chat_state.get("selected_tool_images", [])
            ),
            "tool_result_image_mapping": dict(
                chat_state.get("tool_result_image_mapping", {})
            ),
            "current_tool_goal": chat_state.get("current_tool_goal"),
            "pending_tool_action": chat_state.get("pending_tool_action"),
            "tool_error": chat_state.get("tool_error"),
            "tool_trace_id": chat_state.get("tool_trace_id"),
            "last_decision_type": chat_state.get("last_decision_type"),
            "last_selected_images": list(
                chat_state.get("last_selected_images") or []
            ),
            "last_ranking": [
                dict(item)
                for item in (chat_state.get("last_ranking") or [])
                if isinstance(item, dict)
            ],
            "last_comparison_criterion": str(
                chat_state.get("last_comparison_criterion") or ""
            ),
            "last_decision_reasons": dict(
                chat_state.get("last_decision_reasons") or {}
            ),
            "last_decision_public_answer": str(
                chat_state.get("last_decision_public_answer") or ""
            ),
            "last_decision_tool_trace_id": chat_state.get(
                "last_decision_tool_trace_id"
            ),
            "last_decision_turn_id": chat_state.get(
                "last_decision_turn_id"
            ),
            "last_decision_origin_turn_id": chat_state.get(
                "last_decision_origin_turn_id"
            ),
            "last_decision_image_scope": list(
                chat_state.get("last_decision_image_scope") or []
            ),
            "turn_states": [
                dict(item)
                for item in (chat_state.get("turn_states") or [])[-32:]
                if isinstance(item, dict)
            ],
            "last_completed_turn": (
                dict(chat_state["last_completed_turn"])
                if isinstance(
                    chat_state.get("last_completed_turn"), dict
                )
                else None
            ),
            "last_visual_answer_turn": (
                dict(chat_state["last_visual_answer_turn"])
                if isinstance(
                    chat_state.get("last_visual_answer_turn"), dict
                )
                else None
            ),
            "last_decision_turn": (
                dict(chat_state["last_decision_turn"])
                if isinstance(
                    chat_state.get("last_decision_turn"), dict
                )
                else None
            ),
            "last_generation_turn": (
                dict(chat_state["last_generation_turn"])
                if isinstance(
                    chat_state.get("last_generation_turn"), dict
                )
                else None
            ),
            "last_search_turn": (
                dict(chat_state["last_search_turn"])
                if isinstance(
                    chat_state.get("last_search_turn"), dict
                )
                else None
            ),
            "last_tool_turn": (
                dict(chat_state["last_tool_turn"])
                if isinstance(
                    chat_state.get("last_tool_turn"), dict
                )
                else None
            ),
            "last_explainable_turn": (
                dict(chat_state["last_explainable_turn"])
                if isinstance(
                    chat_state.get("last_explainable_turn"), dict
                )
                else None
            ),
            "active_task": chat_state.get("active_task"),
            "active_target_images": list(
                chat_state.get("active_target_images") or []
            ),
            "pending_clarification": (
                dict(chat_state["pending_clarification"])
                if isinstance(
                    chat_state.get("pending_clarification"), dict
                )
                else None
            ),
            "discourse_focus_stack": [
                dict(item)
                for item in (
                    chat_state.get("discourse_focus_stack") or []
                )[-8:]
                if isinstance(item, dict)
            ],
            "active_task_frame": (
                dict(chat_state["active_task_frame"])
                if isinstance(chat_state.get("active_task_frame"), dict)
                else None
            ),
            "last_completed_task_frame": (
                dict(chat_state["last_completed_task_frame"])
                if isinstance(
                    chat_state.get("last_completed_task_frame"), dict
                )
                else None
            ),
            "latest_compaction_entry_id": chat_state.get(
                "latest_compaction_entry_id"
            ),
        }
        value["selected_asset_refs"] = [item.get("ref") for item in normalized_active if item.get("ref")]
        ensure_ledger(
            value,
            bindings,
            created_at=str(value.get("created_at") or now_iso()),
        )
        snapshot = binding_snapshot(
            bindings,
            active_order=[
                str(item["image_label"]) for item in normalized_active
            ],
            display_order=[
                str(item["image_label"]) for item in normalized_active
            ],
        )
        project_conversation_state(value, snapshot=snapshot)
        return value

    def create_conversation(
        self,
        *,
        title: str | None = None,
        active_assets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create conversation."""
        created = now_iso()
        session = self.normalize_session(
            {
                "conversation_id": str(uuid.uuid4()),
                "title": (title or "新对话").strip() or "新对话",
                "active_assets": list(active_assets or []),
                "messages": [],
                "created_at": created,
                "updated_at": created,
            }
        )
        return self.save_session(session)

    @staticmethod
    def _workspace_default(kind: str) -> dict[str, Any]:
        if kind not in {"generation", "retrieval", "compare"}:
            raise ValueError(f"invalid_workspace_kind:{kind}")
        created = now_iso()
        return {
            "workspace_id": f"{kind}-default",
            "workspace_kind": kind,
            "workspace_type": (
                "compare_rank" if kind == "compare" else kind
            ),
            "version": 0,
            "next_ordinal": 1,
            "selected_assets": [],
            "local_order": [],
            "local_options": {},
            "last_result": None,
            "applied_operations": [],
            "call_source": "standalone_workspace",
            "created_at": created,
            "updated_at": created,
        }

    def workspace(self, kind: str) -> dict[str, Any]:
        """Execute the workspace operation."""
        if kind not in self._workspace_locks:
            raise ValueError(f"invalid_workspace_kind:{kind}")
        with self._workspace_store_lock:
            values = self._read("workspaces")
            item = next(
                (
                    value
                    for value in values
                    if value.get("workspace_kind") == kind
                ),
                None,
            )
            current = (
                dict(item)
                if item is not None
                else self._workspace_default(kind)
            )
        current.setdefault(
            "workspace_type",
            "compare_rank" if kind == "compare" else kind,
        )
        current.setdefault("version", 0)
        current.setdefault("next_ordinal", 1)
        current.setdefault("applied_operations", [])
        assets = [dict(item) for item in current.get("selected_assets", [])]
        next_ordinal = int(current.get("next_ordinal") or 1)
        for index, asset in enumerate(assets, start=1):
            if not asset.get("ordinal"):
                asset["ordinal"] = index
            asset.setdefault("added_at", current.get("created_at") or now_iso())
            asset.setdefault(
                "image_sha256",
                asset.get("sha256") or "",
            )
            next_ordinal = max(next_ordinal, int(asset["ordinal"]) + 1)
        current["selected_assets"] = assets
        current["local_order"] = [
            str(item.get("ref"))
            for item in assets
            if item.get("ref")
        ]
        current["next_ordinal"] = next_ordinal
        return current

    def workspaces(self) -> list[dict[str, Any]]:
        """Execute the workspaces operation."""
        return [self.workspace(kind) for kind in ("generation", "retrieval", "compare")]

    def save_workspace(
        self,
        kind: str,
        *,
        selected_assets: list[dict[str, Any]],
        local_options: dict[str, Any] | None = None,
        last_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save workspace."""
        with self._workspace_locks[kind], self._workspace_store_lock:
            current = self.workspace(kind)
            refs = [str(item["ref"]) for item in selected_assets]
            if len(refs) != len(set(refs)):
                raise ValueError("duplicate_workspace_asset_ref")
            previous = {
                str(item["ref"]): dict(item)
                for item in current.get("selected_assets", [])
            }
            next_ordinal = int(current.get("next_ordinal") or 1)
            normalized = []
            for item in selected_assets:
                asset = dict(item)
                existing = previous.get(str(asset["ref"]))
                if existing:
                    asset.setdefault("ordinal", existing.get("ordinal"))
                    asset.setdefault("added_at", existing.get("added_at"))
                else:
                    asset.setdefault("ordinal", next_ordinal)
                    asset.setdefault("added_at", now_iso())
                    next_ordinal += 1
                asset.setdefault(
                    "image_sha256",
                    asset.get("sha256") or "",
                )
                normalized.append(asset)
            current.update(
                selected_assets=normalized,
                local_order=refs,
                local_options=dict(local_options or {}),
                last_result=last_result,
                next_ordinal=next_ordinal,
                version=int(current.get("version") or 0) + 1,
                call_source="standalone_workspace",
                updated_at=now_iso(),
            )
            values = [
                value
                for value in self._read("workspaces")
                if value.get("workspace_kind") != kind
            ]
            values.append(current)
            self._write("workspaces", values)
            return current

    def apply_workspace_operation(
        self,
        kind: str,
        *,
        operation_id: str,
        workspace_id: str,
        action: str,
        selected_assets: list[dict[str, Any]] | None = None,
        asset: dict[str, Any] | None = None,
        asset_ref: str | None = None,
        direction: int = 0,
        client_sequence: int = 0,
        expected_version: int | None = None,
        local_options: dict[str, Any] | None = None,
        last_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply one idempotent, versioned selection operation atomically."""

        if not operation_id.strip():
            raise ValueError("workspace_operation_id_required")
        if action not in {
            "add",
            "add_many",
            "remove",
            "clear",
            "replace",
            "move",
        }:
            raise ValueError("invalid_workspace_operation")
        with self._workspace_locks[kind], self._workspace_store_lock:
            current = self.workspace(kind)
            if workspace_id != current["workspace_id"]:
                raise WorkspaceVersionConflict("workspace_id_mismatch")
            applied = [
                dict(item)
                for item in current.get("applied_operations", [])
            ]
            replay = next(
                (
                    item
                    for item in applied
                    if item.get("operation_id") == operation_id
                ),
                None,
            )
            if replay is not None:
                return {
                    **current,
                    "operation": {
                        **replay,
                        "idempotent_replay": True,
                    },
                }
            version = int(current.get("version") or 0)
            if expected_version is not None and expected_version != version:
                raise WorkspaceVersionConflict(
                    f"workspace_version_conflict:{version}"
                )
            values = [
                dict(item)
                for item in current.get("selected_assets", [])
            ]
            changed = False
            if action == "add":
                if asset is None or not asset.get("ref"):
                    raise ValueError("workspace_add_asset_required")
                ref = str(asset["ref"])
                if not any(str(item.get("ref")) == ref for item in values):
                    if len(values) >= 5:
                        raise ValueError("workspace_maximum_is_5_images")
                    item = dict(asset)
                    item["ordinal"] = int(
                        current.get("next_ordinal") or 1
                    )
                    item["added_at"] = now_iso()
                    item["image_sha256"] = str(
                        item.get("image_sha256")
                        or item.get("sha256")
                        or ""
                    )
                    values.append(item)
                    current["next_ordinal"] = item["ordinal"] + 1
                    changed = True
            elif action == "add_many":
                incoming = [dict(item) for item in selected_assets or []]
                incoming_refs = [
                    str(item.get("ref") or "") for item in incoming
                ]
                if (
                    any(not ref for ref in incoming_refs)
                    or len(incoming_refs) != len(set(incoming_refs))
                ):
                    raise ValueError("duplicate_workspace_asset_ref")
                existing_refs = {
                    str(item.get("ref") or "") for item in values
                }
                missing = [
                    item
                    for item in incoming
                    if str(item.get("ref") or "") not in existing_refs
                ]
                if len(values) + len(missing) > 5:
                    raise ValueError("workspace_maximum_is_5_images")
                for asset in missing:
                    asset["ordinal"] = int(
                        current.get("next_ordinal") or 1
                    )
                    asset["added_at"] = now_iso()
                    asset["image_sha256"] = str(
                        asset.get("image_sha256")
                        or asset.get("sha256")
                        or ""
                    )
                    values.append(asset)
                    current["next_ordinal"] = asset["ordinal"] + 1
                    existing_refs.add(str(asset["ref"]))
                changed = bool(missing)
            elif action == "remove":
                before = len(values)
                values = [
                    item
                    for item in values
                    if str(item.get("ref")) != str(asset_ref or "")
                ]
                changed = len(values) != before
            elif action == "clear":
                changed = bool(values)
                values = []
            elif action == "replace":
                incoming = [dict(item) for item in selected_assets or []]
                refs = [str(item.get("ref") or "") for item in incoming]
                if (
                    any(not ref for ref in refs)
                    or len(refs) != len(set(refs))
                ):
                    raise ValueError("duplicate_workspace_asset_ref")
                if len(incoming) > 5:
                    raise ValueError("workspace_maximum_is_5_images")
                previous = {
                    str(item.get("ref")): item
                    for item in values
                }
                normalized = []
                for item in incoming:
                    ref = str(item["ref"])
                    old = previous.get(ref)
                    if old:
                        item.setdefault("ordinal", old.get("ordinal"))
                        item.setdefault("added_at", old.get("added_at"))
                    else:
                        item["ordinal"] = int(
                            current.get("next_ordinal") or 1
                        )
                        item["added_at"] = now_iso()
                        current["next_ordinal"] = item["ordinal"] + 1
                    item["image_sha256"] = str(
                        item.get("image_sha256")
                        or item.get("sha256")
                        or ""
                    )
                    normalized.append(item)
                changed = [
                    str(item.get("ref")) for item in values
                ] != refs
                values = normalized
            else:
                ref = str(asset_ref or "")
                source = next(
                    (
                        index
                        for index, item in enumerate(values)
                        if str(item.get("ref")) == ref
                    ),
                    -1,
                )
                target = source + int(direction)
                if (
                    source >= 0
                    and 0 <= target < len(values)
                    and target != source
                ):
                    values[source], values[target] = (
                        values[target],
                        values[source],
                    )
                    changed = True
            if changed:
                version += 1
            current.update(
                selected_assets=values,
                local_order=[
                    str(item["ref"]) for item in values
                ],
                version=version,
                local_options=(
                    dict(local_options)
                    if local_options is not None
                    else dict(current.get("local_options") or {})
                ),
                last_result=(
                    last_result
                    if last_result is not None
                    else current.get("last_result")
                ),
                updated_at=now_iso(),
            )
            operation = {
                "operation_id": operation_id,
                "workspace_id": workspace_id,
                "asset_id": (
                    str((asset or {}).get("asset_id") or asset_ref or "")
                ),
                "action": action,
                "client_sequence": int(client_sequence),
                "expected_version": expected_version,
                "result_version": version,
                "changed": changed,
                "idempotent_replay": False,
                "applied_at": now_iso(),
            }
            current["applied_operations"] = (applied + [operation])[-256:]
            records = [
                value
                for value in self._read("workspaces")
                if value.get("workspace_kind") != kind
            ]
            records.append(current)
            self._write("workspaces", records)
            return {**current, "operation": operation}

    def _update_conversation_assets_unlocked(
        self,
        conversation_id: str,
        active_assets: list[dict[str, Any]],
        *,
        focus_image_label: str | None | object = _FOCUS_UNSET,
    ) -> dict[str, Any]:
        session = self.get_session(conversation_id)
        if session is None:
            raise KeyError(conversation_id)
        chat_state = session["chat_state"]
        bindings = [dict(item) for item in chat_state["asset_bindings"]]
        by_ref = {
            str(item.get("ref")): item
            for item in bindings
            if item.get("ref")
        }
        locked_labels = set(chat_state.get("locked_image_labels", []))
        requested_refs = [str(item["ref"]) for item in active_assets]
        previous_refs = [str(item["ref"]) for item in session["active_assets"]]
        removed_locked = [
            str(item["image_label"])
            for item in bindings
            if item.get("status") == "active"
            and str(item.get("image_label")) in locked_labels
            and str(item.get("ref")) not in requested_refs
        ]
        if removed_locked:
            raise ValueError(
                "locked_image_cannot_be_removed:"
                + ",".join(
                    sorted(
                        removed_locked,
                        key=lambda label: int(label.split("_")[-1]),
                    )
                )
            )
        added_labels: list[str] = []
        removed_labels = [
            str(item["image_label"])
            for item in bindings
            if item.get("status") == "active"
            and str(item.get("ref")) not in requested_refs
        ]
        previous_labels_by_ref = {
            str(item.get("ref")): str(item.get("image_label"))
            for item in bindings
            if item.get("ref")
        }
        changed_at = now_iso()
        preserved_bindings = [
            {
                **dict(item),
                "status": "removed",
                "active": False,
                "removed_at": (
                    item.get("removed_at")
                    or (
                        changed_at
                        if item.get("status") == "active"
                        else None
                    )
                ),
                "deactivated_at": (
                    item.get("deactivated_at")
                    or (
                        changed_at
                        if item.get("status") == "active"
                        else None
                    )
                ),
            }
            for item in bindings
            if str(item.get("ref")) not in requested_refs
        ]
        removed_event_bindings = {
            str(item["image_label"]): dict(item)
            for item in preserved_bindings
            if str(item.get("image_label")) in removed_labels
        }
        occupied_labels = {
            str(item.get("image_label"))
            for item in bindings
            if (
                item.get("image_label")
                and (
                    str(item.get("ref")) in requested_refs
                    or str(item.get("image_label")) in locked_labels
                )
            )
        }
        normalized_active = []
        normalized_bindings = preserved_bindings
        binding_events: list[tuple[dict[str, Any], str]] = []
        assigned_vacant_labels: list[str] = []
        reused_vacant_labels: list[str] = []
        for asset in active_assets:
            item = dict(asset)
            ref = str(item["ref"])
            binding = by_ref.get(ref)
            existing_label = (
                str(binding.get("image_label"))
                if binding is not None
                else None
            )
            if existing_label is None:
                image_label = next_vacant_image_label(occupied_labels)
                assigned_vacant_labels.append(image_label)
                vacated_owner_exists = any(
                    str(historical.get("image_label")) == image_label
                    and not bool(historical.get("locked"))
                    and historical.get("status") != "active"
                    for historical in normalized_bindings
                )
                normalized_bindings = [
                    historical
                    for historical in normalized_bindings
                    if not (
                        str(historical.get("image_label")) == image_label
                        and not bool(historical.get("locked"))
                        and historical.get("status") != "active"
                    )
                ]
                if vacated_owner_exists:
                    reused_vacant_labels.append(image_label)
            else:
                image_label = existing_label
            occupied_labels.add(image_label)
            action = (
                "bind"
                if binding is None
                else "reactivate"
                if binding.get("status") != "active"
                else "reorder"
            )
            binding = {
                **(binding or {}),
                **item,
                "image_label": image_label,
                "status": "active",
                "active": True,
                "locked": image_label in locked_labels,
                "image_sha256": str(
                    item.get("image_sha256")
                    or item.get("sha256")
                    or (binding or {}).get("image_sha256")
                    or (binding or {}).get("sha256")
                    or ""
                ),
                "introduced_turn_id": (
                    (binding or {}).get("introduced_turn_id")
                ),
                "introduced_at": (
                    (binding or {}).get("introduced_at")
                    or (binding or {}).get("added_at")
                    or changed_at
                ),
                "added_at": (
                    (binding or {}).get("added_at")
                    or changed_at
                ),
                "removed_at": None,
                "deactivated_at": None,
            }
            normalized_bindings.append(binding)
            if ref not in previous_refs:
                added_labels.append(image_label)
                binding_events.append((binding, action))
            item["image_label"] = binding["image_label"]
            item["order"] = int(str(binding["image_label"]).split("_")[-1])
            item["locked"] = bool(binding["locked"])
            normalized_active.append(item)
        bindings = normalized_bindings
        active_labels = [str(item["image_label"]) for item in normalized_active]
        if len(active_labels) != len(set(active_labels)):
            raise ValueError("duplicate_active_image_label_assignment")
        if focus_image_label is not _FOCUS_UNSET:
            if focus_image_label is not None and focus_image_label not in active_labels:
                raise ValueError("focus_image_label_not_active")
            focus = focus_image_label
        else:
            previous_focus = chat_state.get("current_focus_label")
            old_focus_ref = next(
                (
                    ref
                    for ref, label in previous_labels_by_ref.items()
                    if label == previous_focus
                ),
                None,
            )
            if old_focus_ref is not None:
                focus = next(
                    (
                        str(item["image_label"])
                        for item in normalized_active
                        if str(item["ref"]) == old_focus_ref
                    ),
                    None,
                )
            else:
                focus = (
                    previous_focus
                    if previous_focus in active_labels
                    else None
                )
        session["active_assets"] = normalized_active
        session["selected_asset_refs"] = requested_refs
        current_labels_by_ref = {
            str(item["ref"]): str(item["image_label"])
            for item in normalized_active
        }
        label_remap = {
            old_label: current_labels_by_ref[ref]
            for ref, old_label in previous_labels_by_ref.items()
            if ref in current_labels_by_ref
        }
        chat_state["recent_mentioned_labels"] = [
            label_remap.get(str(label), str(label))
            for label in chat_state.get("recent_mentioned_labels", [])
            if label_remap.get(str(label), str(label)) in active_labels
        ]
        chat_state["asset_bindings"] = sorted(
            bindings,
            key=lambda item: int(str(item["image_label"]).split("_")[-1]),
        )
        chat_state["current_focus_label"] = focus
        chat_state["locked_image_labels"] = sorted(
            locked_labels,
            key=lambda label: int(label.split("_")[-1]),
        )
        if requested_refs != previous_refs:
            chat_state["newly_added_labels"] = added_labels
            chat_state["removed_labels"] = removed_labels
            chat_state["last_asset_change"] = {
                "added_labels": added_labels,
                "removed_labels": removed_labels,
                "assigned_vacant_labels": assigned_vacant_labels,
                "reused_vacant_labels": reused_vacant_labels,
                "changed_at": now_iso(),
            }
        session["chat_state"] = chat_state
        ensure_ledger(
            session,
            bindings,
            created_at=str(session.get("created_at") or changed_at),
        )
        for label in removed_labels:
            removed = removed_event_bindings.get(label)
            if removed is not None:
                append_binding_event(
                    session,
                    removed,
                    action="deactivate",
                    turn_id=None,
                    created_at=changed_at,
                )
        for binding, action in binding_events:
            append_binding_event(
                session,
                binding,
                action=action,
                turn_id=None,
                created_at=changed_at,
            )
        snapshot = binding_snapshot(
            bindings,
            active_order=active_labels,
            display_order=active_labels,
        )
        project_conversation_state(session, snapshot=snapshot)
        session["updated_at"] = now_iso()
        return self.save_session(session)

    def update_conversation_assets(
        self,
        conversation_id: str,
        active_assets: list[dict[str, Any]],
        *,
        focus_image_label: str | None | object = _FOCUS_UNSET,
        operation_id: str | None = None,
        workspace_id: str | None = None,
        client_sequence: int | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        """Serialize one conversation's authoritative image selection."""

        with self._session_store_lock:
            current = self.get_session(conversation_id)
            if current is None:
                raise KeyError(conversation_id)
            expected_workspace_id = f"chat:{conversation_id}"
            current["workspace_id"] = expected_workspace_id
            current.setdefault("selection_version", 0)
            current.setdefault("selection_operations", [])
            if workspace_id not in (None, "", expected_workspace_id):
                raise WorkspaceVersionConflict(
                    "chat_workspace_id_mismatch"
                )
            if operation_id:
                replay = next(
                    (
                        item
                        for item in current["selection_operations"]
                        if item.get("operation_id") == operation_id
                    ),
                    None,
                )
                if replay is not None:
                    current["selection_operation"] = {
                        **replay,
                        "idempotent_replay": True,
                    }
                    return current
            version = int(current.get("selection_version") or 0)
            if (
                expected_version is not None
                and int(expected_version) != version
            ):
                raise WorkspaceVersionConflict(
                    f"chat_selection_version_conflict:{version}"
                )
            before = [
                str(item.get("ref"))
                for item in current.get("active_assets", [])
            ]
            requested = [
                str(item.get("ref"))
                for item in active_assets
            ]
            saved = self._update_conversation_assets_unlocked(
                conversation_id,
                active_assets,
                focus_image_label=focus_image_label,
            )
            changed = before != requested
            if changed:
                version += 1
            saved["workspace_id"] = expected_workspace_id
            saved["selection_version"] = version
            operations = list(
                current.get("selection_operations", [])
            )
            operation = {
                "operation_id": operation_id,
                "workspace_id": expected_workspace_id,
                "client_sequence": client_sequence,
                "expected_version": expected_version,
                "result_version": version,
                "changed": changed,
                "idempotent_replay": False,
                "applied_at": now_iso(),
            }
            if operation_id:
                operations.append(operation)
            saved["selection_operations"] = operations[-256:]
            saved["selection_operation"] = operation
            return self.save_session(saved)

    @staticmethod
    def lock_session_images(
        session: dict[str, Any],
        *,
        asset_refs: Iterable[str],
        use_kind: str,
        trace_id: str | None = None,
    ) -> list[str]:
        """Apply an already validated lock mutation to active bindings."""

        refs = {str(ref) for ref in asset_refs}
        if not refs:
            return []
        chat_state = session["chat_state"]
        bindings = [
            dict(item)
            for item in chat_state.get("asset_bindings", [])
        ]
        locked_labels = set(chat_state.get("locked_image_labels", []))
        lock_records = dict(chat_state.get("image_lock_records", {}))
        newly_locked: list[str] = []
        for binding in bindings:
            if (
                binding.get("status") != "active"
                or str(binding.get("ref")) not in refs
            ):
                continue
            label = str(binding["image_label"])
            if label not in locked_labels:
                newly_locked.append(label)
            locked_labels.add(label)
            binding["locked"] = True
            lock_records.setdefault(
                label,
                {
                    "locked_at": now_iso(),
                    "first_use_kind": use_kind,
                    "trace_id": trace_id,
                },
            )
        for asset in session.get("active_assets", []):
            if str(asset.get("ref")) in refs:
                asset["locked"] = True
        chat_state["asset_bindings"] = bindings
        chat_state["locked_image_labels"] = sorted(
            locked_labels,
            key=lambda label: int(label.split("_")[-1]),
        )
        chat_state["image_lock_records"] = lock_records
        session["chat_state"] = chat_state
        for binding in bindings:
            if str(binding.get("image_label")) in newly_locked:
                append_binding_event(
                    session,
                    binding,
                    action="lock",
                    turn_id=None,
                    created_at=now_iso(),
                )
        snapshot = binding_snapshot(
            bindings,
            active_order=[
                str(item.get("image_label"))
                for item in session.get("active_assets", [])
                if item.get("image_label")
            ],
            display_order=[
                str(item.get("image_label"))
                for item in session.get("active_assets", [])
                if item.get("image_label")
            ],
        )
        project_conversation_state(session, snapshot=snapshot)
        return sorted(
            newly_locked,
            key=lambda label: int(label.split("_")[-1]),
        )

    def commit_referenced_image_scope(
        self,
        session: dict[str, Any],
        *,
        assets: list[dict[str, Any]],
        image_labels: list[str],
        use_kind: str,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate and durably lock an explicit visual-task image scope.

        This is the single authoritative pre-tool commit used by visual
        business routes.  It rejects missing, inactive, reordered or
        SHA-mismatched canonical bindings before changing any lock state.
        """

        labels = [str(label) for label in image_labels]
        if not labels or any(
            not label.startswith("IMG_") for label in labels
        ):
            raise ValueError("referenced_image_scope_requires_active_img_labels")
        validation = validate_model_bindings(session, assets, labels)
        refs = [str(item.get("ref") or "") for item in assets]
        if any(not ref for ref in refs):
            raise ValueError("referenced_image_scope_asset_ref_missing")
        newly_locked = self.lock_session_images(
            session,
            asset_refs=refs,
            use_kind=use_kind,
            trace_id=trace_id,
        )
        session["updated_at"] = now_iso()
        self.save_session(session)
        return {
            "status": "committed",
            "use_kind": use_kind,
            "image_labels": labels,
            "asset_refs": refs,
            "newly_locked": newly_locked,
            "binding_validation": validation,
        }

    def libraries(self) -> list[dict[str, Any]]:
        """Execute the libraries operation."""
        values = self._read("libraries")
        if not values:
            values = [
                {
                    "library_id": "default",
                    "name": "默认图片库",
                    "display_name": "默认图片库",
                    "library_type": "user_custom",
                    "locked": False,
                    "owner": "local_user",
                    "description": "",
                    "created_at": now_iso(),
                }
            ]
            self._write("libraries", values)
        changed = False
        for item in values:
            if item.get("library_id") in SYSTEM_LIBRARY_IDS:
                raise ValueError("system_library_must_not_be_user_store_record")
            defaults = {
                "display_name": item.get("name", "未命名图片库"),
                "library_type": "user_custom",
                "locked": False,
                "owner": "local_user",
                "description": "",
            }
            for key, default in defaults.items():
                if key not in item:
                    item[key] = default
                    changed = True
        if changed:
            self._write("libraries", values)
        return values

    @staticmethod
    def _validate_custom_library_name(name: str) -> str:
        value = name.strip()
        if not value:
            raise ValueError("library_name_required")
        if value.casefold() in {name.casefold() for name in SYSTEM_RESERVED_NAMES}:
            raise ValueError("reserved_system_library_name")
        return value

    def assert_library_mutable(
        self,
        library_id: str,
        operation: str,
    ) -> dict[str, Any]:
        """Central normal-user guard for every persistent-library mutation."""

        if library_id in LEGACY_DEFAULT_LIBRARY_IDS:
            library_id = DEFAULT_LIBRARY_ID
        if library_id in SYSTEM_LIBRARY_IDS:
            raise LibraryMutationForbidden(
                "系统训练/验证图片库为只读，不能执行该操作。"
            )
        library = next(
            (
                item
                for item in self.libraries()
                if item.get("library_id") == library_id
            ),
            None,
        )
        if library is None:
            raise KeyError(library_id)
        if (
            library.get("locked")
            or library.get("library_type") != "user_custom"
        ):
            raise LibraryMutationForbidden(
                "当前图片库为只读，不能执行该操作。"
            )
        return {
            "library": library,
            "operation": operation,
            "mutable": True,
        }

    def create_library(
        self,
        name: str,
        *,
        description: str = "",
        owner: str = "local_user",
    ) -> dict[str, Any]:
        """Create library."""
        display_name = self._validate_custom_library_name(name)
        item = {
            "library_id": str(uuid.uuid4()),
            "name": display_name,
            "display_name": display_name,
            "library_type": "user_custom",
            "locked": False,
            "owner": owner,
            "description": description.strip(),
            "index_version": None,
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        values = self.libraries()
        values.append(item)
        self._write("libraries", values)
        return item

    def rename_library(self, library_id: str, name: str) -> dict[str, Any]:
        """Execute the rename library operation."""
        self.assert_library_mutable(library_id, "rename_library")
        display_name = self._validate_custom_library_name(name)
        values = self.libraries()
        for item in values:
            if item["library_id"] == library_id:
                item.update(
                    name=display_name,
                    display_name=display_name,
                    updated_at=now_iso(),
                )
                self._write("libraries", values)
                return item
        raise KeyError(library_id)

    def delete_library(self, library_id: str) -> dict[str, Any]:
        """Delete library."""
        self.assert_library_mutable(library_id, "delete_library")
        values = self.libraries()
        item = next(
            (value for value in values if value.get("library_id") == library_id),
            None,
        )
        if item is None:
            raise KeyError(library_id)
        asset_ids = [
            asset["asset_id"]
            for asset in self.assets(library_id)
        ]
        for asset_id in asset_ids:
            self.delete_asset(asset_id)
        self._write(
            "libraries",
            [
                value
                for value in values
                if value.get("library_id") != library_id
            ],
        )
        return {
            "library_id": library_id,
            "deleted": True,
            "deleted_asset_count": len(asset_ids),
            "system_libraries_modified": False,
        }

    def assets(self, library_id: str | None = None) -> list[dict[str, Any]]:
        """Execute the assets operation."""
        values = self._read("assets")
        if library_id is None:
            return values
        if library_id in {DEFAULT_LIBRARY_ID, *LEGACY_DEFAULT_LIBRARY_IDS}:
            return [
                item
                for item in values
                if item.get("library_id")
                in {DEFAULT_LIBRARY_ID, *LEGACY_DEFAULT_LIBRARY_IDS}
            ]
        return [
            item
            for item in values
            if item.get("library_id") == library_id
        ]

    def asset(self, asset_id: str) -> dict[str, Any]:
        """Execute the asset operation."""
        item = next((value for value in self.assets() if value.get("asset_id") == asset_id), None)
        if item is None:
            raise KeyError(asset_id)
        analysis = next((value for value in self._read("analyses") if value.get("asset_id") == asset_id), None)
        return {**item, "analysis": analysis, "analysis_status": analysis.get("status") if analysis else "not_started", "index_status": "not_indexed"}

    def delete_asset(self, asset_id: str) -> dict[str, Any]:
        """Delete asset."""
        values = self._read("assets")
        item = next((value for value in values if value.get("asset_id") == asset_id), None)
        if item is None:
            raise KeyError(asset_id)
        self.assert_library_mutable(
            str(item.get("library_id") or DEFAULT_LIBRARY_ID),
            "delete_asset",
        )
        path = Path(item["path"])
        shared_path = any(
            value.get("asset_id") != asset_id
            and Path(str(value.get("path") or "")).resolve()
            == path.resolve()
            for value in values
        )
        if (
            path.is_file()
            and not shared_path
            and self.assets_root.resolve() in path.resolve().parents
        ):
            path.unlink()
        values = [value for value in values if value.get("asset_id") != asset_id]
        self._write("assets", values)
        analyses = [value for value in self._read("analyses") if value.get("asset_id") != asset_id]
        self._write("analyses", analyses)
        return {"asset_id": asset_id, "deleted": True, "source_dataset_modified": False, "index_updated": "not_applicable_not_indexed"}

    def move_asset(self, asset_id: str, target_library_id: str) -> dict[str, Any]:
        """Execute the move asset operation."""
        self.assert_library_mutable(target_library_id, "move_asset_target")
        values = self._read("assets")
        for item in values:
            if item.get("asset_id") == asset_id:
                if item.get("library_id") in SYSTEM_LIBRARY_IDS:
                    raise LibraryMutationForbidden(
                        "系统图片库资产为只读，不能移动。"
                    )
                self.assert_library_mutable(
                    str(item.get("library_id") or DEFAULT_LIBRARY_ID),
                    "move_asset_source",
                )
                item["library_id"] = target_library_id
                item["updated_at"] = now_iso()
                self._write("assets", values)
                return dict(item)
        raise KeyError(asset_id)

    def save_analysis(self, asset_id: str, value: dict[str, Any]) -> dict[str, Any]:
        """Save analysis."""
        self.asset(asset_id)
        item = {"asset_id": asset_id, "updated_at": now_iso(), **value}
        values = [existing for existing in self._read("analyses") if existing.get("asset_id") != asset_id]
        values.append(item)
        self._write("analyses", values)
        return item

    def save_canonical_analysis(
        self,
        asset_id: str,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """Save canonical analysis."""
        self.asset(asset_id)
        existing = next(
            (
                item
                for item in self._read("analyses")
                if item.get("asset_id") == asset_id
            ),
            {},
        )
        item = {
            **dict(existing),
            "asset_id": asset_id,
            "status": value.get("status", "completed"),
            "canonical": dict(value),
            "updated_at": now_iso(),
        }
        item.pop("canonical_failure", None)
        item.pop("canonical_last_attempt", None)
        values = [
            current
            for current in self._read("analyses")
            if current.get("asset_id") != asset_id
        ]
        values.append(item)
        self._write("analyses", values)
        return item

    def save_canonical_failure(
        self,
        asset_id: str,
        failure: dict[str, Any],
    ) -> dict[str, Any]:
        """Save canonical failure."""
        self.asset(asset_id)
        existing = next(
            (
                item
                for item in self._read("analyses")
                if item.get("asset_id") == asset_id
            ),
            {},
        )
        preserved_canonical = (
            dict(existing.get("canonical") or {})
            if isinstance(existing.get("canonical"), dict)
            else None
        )
        item = {
            **dict(existing),
            "asset_id": asset_id,
            "status": "completed" if preserved_canonical else "failed",
            "canonical_failure": dict(failure),
            "updated_at": now_iso(),
        }
        if preserved_canonical:
            item["canonical"] = preserved_canonical
            item["canonical_last_attempt"] = dict(failure)
        values = [
            current
            for current in self._read("analyses")
            if current.get("asset_id") != asset_id
        ]
        values.append(item)
        self._write("analyses", values)
        return item

    def import_bytes(self, name: str, content: bytes, *, library_id: str = "default") -> dict[str, Any]:
        """Import bytes."""
        if library_id in LEGACY_DEFAULT_LIBRARY_IDS:
            library_id = DEFAULT_LIBRARY_ID
        self.assert_library_mutable(library_id, "import_asset")
        safe_name = Path(name).name or "upload.bin"
        digest = hashlib.sha256(content).hexdigest()
        values = self._read("assets")
        duplicate = next(
            (
                existing
                for existing in values
                if existing.get("sha256") == digest
                and existing.get("library_id") == library_id
            ),
            None,
        )
        if duplicate:
            return {**duplicate, "duplicate": True}
        shared_content = next(
            (
                existing
                for existing in values
                if existing.get("sha256") == digest
                and Path(str(existing.get("path") or "")).is_file()
                and Path(str(existing.get("path") or "")).resolve()
                .is_relative_to(self.assets_root)
            ),
            None,
        )
        asset_dir = self.assets_root / digest[:2] / digest
        asset_dir.mkdir(parents=True, exist_ok=True)
        extension = Path(safe_name).suffix.lower() or ".bin"
        destination = (
            Path(str(shared_content["path"]))
            if shared_content is not None
            else asset_dir / f"{digest}{extension}"
        )
        if shared_content is None and not destination.exists():
            destination.write_bytes(content)
        width = None
        height = None
        try:
            from PIL import Image

            with Image.open(destination) as image:
                width, height = image.size
        except Exception:
            pass
        asset_id = digest[:16]
        if any(item.get("asset_id") == asset_id for item in values):
            library_suffix = hashlib.sha256(
                library_id.encode("utf-8")
            ).hexdigest()[:4]
            asset_id = f"{digest[:12]}{library_suffix}"
        item = {
            "asset_id": asset_id,
            "library_id": library_id,
            "image_id": safe_name,
            "sha256": digest,
            "path": str(destination),
            "storage_key": destination.relative_to(
                self.assets_root.parent
            ).as_posix(),
            "original_filename": safe_name,
            "original_extension": extension,
            "mime_type": mimetypes.guess_type(safe_name)[0] or "application/octet-stream",
            "bytes": len(content),
            "width": width,
            "height": height,
            "import_status": "completed",
            "created_at": now_iso(),
        }

        values.append(item)
        self._write("assets", values)
        return {**item, "duplicate": False}

    def session_assets(
        self,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the session assets operation."""
        values = self._read("session_assets")
        return (
            values
            if conversation_id is None
            else [
                item
                for item in values
                if item.get("conversation_id") == conversation_id
            ]
        )

    def session_asset(self, asset_id: str, conversation_id: str) -> dict[str, Any]:
        """Execute the session asset operation."""
        item = next(
            (
                value
                for value in self.session_assets(conversation_id)
                if value.get("asset_id") == asset_id
            ),
            None,
        )
        if item is None:
            raise KeyError(asset_id)
        return dict(item)

    def import_session_bytes(
        self,
        name: str,
        content: bytes,
        *,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Import session bytes."""
        if not conversation_id.strip():
            raise ValueError("conversation_id_required")
        safe_name = Path(name).name or "upload.bin"
        digest = hashlib.sha256(content).hexdigest()
        asset_id = f"session-{uuid.uuid4().hex}"
        asset_dir = self.session_assets_root / conversation_id / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        destination = asset_dir / safe_name
        destination.write_bytes(content)
        item = {
            "asset_id": asset_id,
            "conversation_id": conversation_id,
            "library_id": None,
            "library_type": "session_temporary",
            "persistent": False,
            "image_id": safe_name,
            "sha256": digest,
            "path": str(destination),
            "storage_key": destination.relative_to(
                self.session_assets_root
            ).as_posix(),
            "original_filename": safe_name,
            "original_extension": Path(safe_name).suffix.lower(),
            "mime_type": mimetypes.guess_type(safe_name)[0]
            or "application/octet-stream",
            "bytes": len(content),
            "import_status": "completed",
            "created_at": now_iso(),
        }
        values = self._read("session_assets")
        duplicate = next(
            (
                existing
                for existing in values
                if existing.get("conversation_id") == conversation_id
                and existing.get("sha256") == digest
            ),
            None,
        )
        if duplicate:
            destination.unlink(missing_ok=True)
            return {**duplicate, "duplicate": True}
        values.append(item)
        self._write("session_assets", values)
        return {**item, "duplicate": False}

    def save_session_canonical(
        self,
        asset_id: str,
        *,
        conversation_id: str,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """Save session canonical."""
        values = self._read("session_assets")
        for item in values:
            if (
                item.get("asset_id") == asset_id
                and item.get("conversation_id") == conversation_id
            ):
                item["canonical"] = dict(value)
                item["canonical_status"] = value.get(
                    "status",
                    "completed",
                )
                item.pop("canonical_failure", None)
                item["updated_at"] = now_iso()
                self._write("session_assets", values)
                return dict(item)
        raise KeyError(asset_id)

    def save_session_canonical_failure(
        self,
        asset_id: str,
        *,
        conversation_id: str,
        failure: dict[str, Any],
    ) -> dict[str, Any]:
        """Save session canonical failure."""
        values = self._read("session_assets")
        for item in values:
            if (
                item.get("asset_id") == asset_id
                and item.get("conversation_id") == conversation_id
            ):
                item["canonical_failure"] = dict(failure)
                if not isinstance(item.get("canonical"), dict):
                    item["canonical_status"] = "failed"
                item["updated_at"] = now_iso()
                self._write("session_assets", values)
                return dict(item)
        raise KeyError(asset_id)

    def persist_session_asset(
        self,
        asset_id: str,
        *,
        conversation_id: str,
        library_id: str,
    ) -> dict[str, Any]:
        """Execute the persist session asset operation."""
        temporary = self.session_asset(asset_id, conversation_id)
        content = Path(str(temporary["path"])).read_bytes()
        persistent = self.import_bytes(
            str(
                temporary.get("original_filename")
                or temporary.get("image_id")
                or "upload.bin"
            ),
            content,
            library_id=library_id,
        )
        return {
            "persistent_asset": persistent,
            "temporary_asset_id": asset_id,
            "conversation_id": conversation_id,
            "temporary_retained_until_session_cleanup": True,
        }

    def import_session_base64(
        self,
        name: str,
        encoded: str,
        *,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Import session base64."""
        try:
            content = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("invalid_base64_upload") from exc
        return self.import_session_bytes(
            name,
            content,
            conversation_id=conversation_id,
        )

    def delete_session_asset(
        self,
        asset_id: str,
        *,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Delete session asset."""
        values = self._read("session_assets")
        item = next(
            (
                value
                for value in values
                if value.get("asset_id") == asset_id
                and value.get("conversation_id") == conversation_id
            ),
            None,
        )
        if item is None:
            raise KeyError(asset_id)
        path = Path(str(item["path"]))
        if (
            path.is_file()
            and self.session_assets_root.resolve() in path.resolve().parents
        ):
            path.unlink()
        self._write(
            "session_assets",
            [value for value in values if value is not item],
        )
        return {
            "asset_id": asset_id,
            "conversation_id": conversation_id,
            "deleted": True,
            "persistent_library_modified": False,
            "system_libraries_modified": False,
        }

    def import_base64(self, name: str, encoded: str, *, library_id: str = "default") -> dict[str, Any]:
        """Import base64."""
        try:
            content = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("invalid_base64_upload") from exc
        return self.import_bytes(name, content, library_id=library_id)

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        tasks = self.tasks()
        return {
            "storage": "ready",
            "tasks": {"total": len(tasks), "running": sum(item.get("status") == "running" for item in tasks), "failed": sum(item.get("status") == "failed" for item in tasks)},
            "feedback_queue": len(self.feedback()),
            "vqa_sessions": len(self.sessions()),
            "upload_assets": len(list(self.assets_root.rglob("*"))),
            "libraries": len(self.libraries()),
            "imported_asset_records": len(self.assets()),
            "session_asset_records": len(self.session_assets()),
            "function_workspaces": len(self.workspaces()),
        }
