"""Phase 5.4-I append-only session ledger and deterministic projections.

The ledger is deliberately small and JSON-native.  It borrows the separation
between durable entries and per-call context projection, but does not depend on
or copy any external agent framework.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


LEDGER_SCHEMA = "scenemindx_session_ledger_v1"
BINDING_SCHEMA = "scenemindx_canonical_image_bindings_v1"
CONVERSATION_STATE_SCHEMA = "scenemindx_single_thread_multiturn_chat_state_v6"
CONTEXT_PROJECTION_SCHEMA = "scenemindx_context_projection_v1"
COMPACTION_TURN_THRESHOLD = 6
RETAIN_RECENT_TURNS = 3


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Execute the file sha256 operation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_index(label: str) -> int:
    """Execute the image index operation."""
    try:
        return int(str(label).split("_", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def binding_snapshot(
    bindings: Iterable[dict[str, Any]],
    *,
    active_order: Iterable[str] | None = None,
    display_order: Iterable[str] | None = None,
    model_send_order: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Execute the binding snapshot operation."""
    normalized: dict[str, dict[str, Any]] = {}
    for raw in sorted(
        (dict(item) for item in bindings if item.get("image_label")),
        key=lambda item: image_index(str(item["image_label"])),
    ):
        label = str(raw["image_label"])
        normalized[label] = {
            "asset_id": str(
                raw.get("asset_id")
                or raw.get("source_asset_id")
                or raw.get("image_id")
                or ""
            ),
            "ref": str(raw.get("ref") or ""),
            "image_sha256": str(
                raw.get("image_sha256") or raw.get("sha256") or ""
            ),
            "introduced_turn_id": raw.get("introduced_turn_id"),
            "locked": bool(raw.get("locked")),
            "active": raw.get("status", "active") == "active",
            "source": str(raw.get("source") or ""),
            "introduced_at": raw.get("introduced_at")
            or raw.get("added_at"),
            "deactivated_at": raw.get("deactivated_at")
            or raw.get("removed_at"),
        }
    active = list(active_order or [])
    if not active:
        active = [
            label
            for label, item in normalized.items()
            if item["active"]
        ]
    display = list(display_order or active)
    payload = {
        "binding_version": BINDING_SCHEMA,
        "next_image_index": max(
            [image_index(label) for label in normalized] or [0]
        )
        + 1,
        "bindings": normalized,
        "active_order": active,
        "display_order": display,
        "model_send_order": list(model_send_order or []),
    }
    payload["binding_snapshot_sha256"] = _canonical_sha256(payload)
    return payload


def _new_ledger() -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA,
        "next_sequence": 1,
        "entries": [],
        "last_entry_id": None,
        "migration_version": 1,
        "compaction": {
            "latest_entry_id": None,
            "threshold": COMPACTION_TURN_THRESHOLD,
            "retain_recent_turns": RETAIN_RECENT_TURNS,
        },
    }


def append_entry(
    session: dict[str, Any],
    entry_type: str,
    *,
    turn_id: str | None,
    created_at: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Execute the append entry operation."""
    ledger = session.setdefault("session_ledger", _new_ledger())
    if ledger.get("schema_version") != LEDGER_SCHEMA:
        raise ValueError("unsupported_session_ledger_schema")
    sequence = int(ledger.get("next_sequence") or 1)
    entry = {
        "entry_id": f"LE_{sequence:06d}",
        "sequence": sequence,
        "parent_entry_id": ledger.get("last_entry_id"),
        "entry_type": entry_type,
        "turn_id": turn_id,
        "created_at": created_at,
        "payload": payload,
    }
    ledger.setdefault("entries", []).append(entry)
    ledger["last_entry_id"] = entry["entry_id"]
    ledger["next_sequence"] = sequence + 1
    session["session_ledger"] = ledger
    return entry


def ensure_ledger(
    session: dict[str, Any],
    bindings: Iterable[dict[str, Any]],
    *,
    created_at: str,
) -> dict[str, Any]:
    """Create a one-time migration ledger without altering public history."""

    ledger = session.get("session_ledger")
    if isinstance(ledger, dict) and ledger.get("schema_version") == LEDGER_SCHEMA:
        ledger.setdefault("entries", [])
        ledger.setdefault("next_sequence", len(ledger["entries"]) + 1)
        ledger.setdefault(
            "last_entry_id",
            ledger["entries"][-1]["entry_id"]
            if ledger["entries"]
            else None,
        )
        ledger.setdefault(
            "compaction",
            {
                "latest_entry_id": None,
                "threshold": COMPACTION_TURN_THRESHOLD,
                "retain_recent_turns": RETAIN_RECENT_TURNS,
            },
        )
        session["session_ledger"] = ledger
        return ledger

    session["session_ledger"] = _new_ledger()
    for binding in sorted(
        (dict(item) for item in bindings if item.get("image_label")),
        key=lambda item: image_index(str(item["image_label"])),
    ):
        append_entry(
            session,
            "image_binding",
            turn_id=binding.get("introduced_turn_id"),
            created_at=str(
                binding.get("introduced_at")
                or binding.get("added_at")
                or created_at
            ),
            payload={
                "image_label": str(binding["image_label"]),
                "asset_id": str(
                    binding.get("asset_id")
                    or binding.get("source_asset_id")
                    or binding.get("image_id")
                    or ""
                ),
                "ref": str(binding.get("ref") or ""),
                "image_sha256": str(
                    binding.get("image_sha256")
                    or binding.get("sha256")
                    or ""
                ),
                "action": (
                    "bind"
                    if binding.get("status", "active") == "active"
                    else "deactivate"
                ),
                "source": str(binding.get("source") or ""),
                "locked": bool(binding.get("locked")),
            },
        )
    for message in session.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        entry_type = (
            "user_message" if role == "user" else "assistant_public"
        )
        content = message.get("content")
        if isinstance(content, dict):
            content = (
                content.get("display_text")
                or content.get("public_answer")
                or content.get("answer")
                or ""
            )
        append_entry(
            session,
            entry_type,
            turn_id=str(message.get("message_id") or "") or None,
            created_at=str(message.get("created_at") or created_at),
            payload=(
                {"original_text": str(content or "")}
                if role == "user"
                else {
                    "public_answer": str(content or ""),
                    "task_type": str(message.get("task_type") or ""),
                    "referenced_images": list(
                        message.get("image_references")
                        or message.get("asset_refs")
                        or []
                    ),
                    "answer_provenance": "legacy_public_history",
                }
            ),
        )
    return session["session_ledger"]


def append_binding_event(
    session: dict[str, Any],
    binding: dict[str, Any],
    *,
    action: str,
    turn_id: str | None,
    created_at: str,
) -> dict[str, Any]:
    """Execute the append binding event operation."""
    return append_entry(
        session,
        "image_binding",
        turn_id=turn_id,
        created_at=created_at,
        payload={
            "image_label": str(binding["image_label"]),
            "asset_id": str(
                binding.get("asset_id")
                or binding.get("source_asset_id")
                or binding.get("image_id")
                or ""
            ),
            "ref": str(binding.get("ref") or ""),
            "image_sha256": str(
                binding.get("image_sha256")
                or binding.get("sha256")
                or ""
            ),
            "action": action,
            "source": str(binding.get("source") or ""),
            "locked": bool(binding.get("locked")),
        },
    )


def append_completed_turn(
    session: dict[str, Any],
    *,
    user_message_id: str,
    assistant_message_id: str,
    question: str,
    public_answer: str,
    task_type: str,
    image_labels: list[str],
    answer_provenance: str,
    tool_call: dict[str, Any],
    tool_result: dict[str, Any],
    dialogue_state: dict[str, Any],
    created_at: str,
) -> list[dict[str, Any]]:
    """Execute the append completed turn operation."""
    entries = [
        append_entry(
            session,
            "user_message",
            turn_id=user_message_id,
            created_at=created_at,
            payload={"original_text": question},
        ),
        append_entry(
            session,
            "tool_call",
            turn_id=assistant_message_id,
            created_at=created_at,
            payload={
                "tool_name": str(tool_call.get("tool_name") or ""),
                "arguments": {
                    "asset_refs": list(tool_call.get("asset_refs") or [])
                },
                "original_user_request": question,
                "referenced_images": image_labels,
            },
        ),
        append_entry(
            session,
            "tool_result",
            turn_id=assistant_message_id,
            created_at=created_at,
            payload={
                "selected": list(tool_result.get("selected") or []),
                "ranking": list(tool_result.get("ranking") or []),
                "search_results": list(
                    tool_result.get("search_results") or []
                ),
                "generated_text": str(
                    tool_result.get("generated_text") or ""
                ),
                "technical_result_reference": tool_call.get("trace_id"),
            },
        ),
        append_entry(
            session,
            "assistant_public",
            turn_id=assistant_message_id,
            created_at=created_at,
            payload={
                "public_answer": public_answer,
                "task_type": task_type,
                "referenced_images": image_labels,
                "answer_provenance": answer_provenance,
            },
        ),
        append_entry(
            session,
            "dialogue_state",
            turn_id=assistant_message_id,
            created_at=created_at,
            payload=dialogue_state,
        ),
    ]
    maybe_compact(session, created_at=created_at)
    return entries


def maybe_compact(
    session: dict[str, Any],
    *,
    created_at: str,
    force: bool = False,
) -> dict[str, Any] | None:
    """Execute the maybe compact operation."""
    ledger = session.get("session_ledger") or {}
    entries = list(ledger.get("entries") or [])
    last_compaction_index = max(
        (
            index
            for index, entry in enumerate(entries)
            if entry.get("entry_type") == "compaction"
        ),
        default=-1,
    )
    new_dialogue = [
        entry
        for entry in entries[last_compaction_index + 1 :]
        if entry.get("entry_type") == "dialogue_state"
    ]
    threshold = int(
        ledger.get("compaction", {}).get(
            "threshold", COMPACTION_TURN_THRESHOLD
        )
    )
    if not force and len(new_dialogue) < threshold:
        return None
    dialogue_entries = [
        entry
        for entry in entries
        if entry.get("entry_type") == "dialogue_state"
    ]
    retained_dialogue = dialogue_entries[-RETAIN_RECENT_TURNS:]
    retained_ids: list[str] = []
    if retained_dialogue:
        first_dialogue_sequence = int(retained_dialogue[0]["sequence"])
        first_user_sequence = max(
            (
                int(entry["sequence"])
                for entry in entries
                if entry.get("entry_type") == "user_message"
                and int(entry["sequence"]) < first_dialogue_sequence
            ),
            default=first_dialogue_sequence,
        )
        retained_ids = [
            str(entry["entry_id"])
            for entry in entries
            if int(entry["sequence"]) >= first_user_sequence
        ]
    compacted = [
        entry
        for entry in entries
        if entry.get("entry_id") not in retained_ids
        and entry.get("entry_type") != "compaction"
    ]
    task_counts: dict[str, int] = {}
    old_turn_ids: list[str] = []
    for entry in compacted:
        if entry.get("entry_type") != "dialogue_state":
            continue
        payload = entry.get("payload") or {}
        task = str(payload.get("active_task") or "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
        if entry.get("turn_id"):
            old_turn_ids.append(str(entry["turn_id"]))
    binding = (
        session.get("chat_state", {}).get("canonical_image_bindings")
        or {}
    )
    entry = append_entry(
        session,
        "compaction",
        turn_id=None,
        created_at=created_at,
        payload={
            "compacted_entry_ids": [
                str(item["entry_id"]) for item in compacted
            ],
            "retained_tail_entry_ids": retained_ids,
            "structured_summary": {
                "older_turn_ids": old_turn_ids,
                "task_counts": task_counts,
                "summary_kind": "deterministic_business_state",
            },
            "canonical_image_bindings": binding,
            "immutable_binding_snapshot_sha256": binding.get(
                "binding_snapshot_sha256"
            ),
            "source_entry_ids": [
                str(item["entry_id"]) for item in entries
            ],
        },
    )
    session["session_ledger"]["compaction"]["latest_entry_id"] = entry[
        "entry_id"
    ]
    return entry


def project_conversation_state(
    session: dict[str, Any],
    *,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Execute the project conversation state operation."""
    state = session.setdefault("chat_state", {})
    ledger = session.get("session_ledger") or {}
    latest_dialogue = next(
        (
            entry
            for entry in reversed(ledger.get("entries") or [])
            if entry.get("entry_type") == "dialogue_state"
        ),
        None,
    )
    dialogue_payload = (
        dict(latest_dialogue.get("payload") or {})
        if latest_dialogue
        else {}
    )
    state.setdefault(
        "schema_version",
        "scenemindx_single_thread_multiturn_chat_state_v5",
    )
    state["ledger_projection_schema_version"] = (
        CONVERSATION_STATE_SCHEMA
    )
    state["ledger_schema_version"] = LEDGER_SCHEMA
    state["ledger_last_entry_id"] = ledger.get("last_entry_id")
    state["canonical_image_bindings"] = snapshot
    state["active_task_frame"] = dialogue_payload.get("task_frame")
    state["last_completed_task_frame"] = (
        dialogue_payload.get("task_frame")
        or state.get("last_completed_task_frame")
    )
    state["latest_compaction_entry_id"] = (
        ledger.get("compaction", {}).get("latest_entry_id")
    )
    state["context_projection_version"] = CONTEXT_PROJECTION_SCHEMA
    session["chat_state"] = state
    return state


def context_projection(
    session: dict[str, Any],
    *,
    original_user_message: str,
    dialogue_act: str | None,
    referenced_turn_id: str | None,
    current_image_scope: list[str],
    standalone_request: str | None,
) -> dict[str, Any]:
    """Execute the context projection operation."""
    ledger = session.get("session_ledger") or {}
    entries = list(ledger.get("entries") or [])
    selected_entries: list[dict[str, Any]] = []
    if referenced_turn_id:
        selected_entries.extend(
            entry
            for entry in entries
            if str(entry.get("turn_id")) == referenced_turn_id
        )
    dialogue_entries = [
        entry
        for entry in entries
        if entry.get("entry_type") == "dialogue_state"
    ]
    recent_turn_ids = [
        str(entry.get("turn_id"))
        for entry in dialogue_entries[-RETAIN_RECENT_TURNS:]
        if entry.get("turn_id")
    ]
    selected_entries.extend(
        entry
        for entry in entries
        if str(entry.get("turn_id")) in recent_turn_ids
        and entry.get("entry_type")
        in {"user_message", "assistant_public", "tool_result", "dialogue_state"}
    )
    latest_compaction = next(
        (
            entry
            for entry in reversed(entries)
            if entry.get("entry_type") == "compaction"
        ),
        None,
    )
    if latest_compaction:
        selected_entries.append(latest_compaction)
    deduped = {
        str(entry["entry_id"]): entry for entry in selected_entries
    }
    selected_entries = sorted(
        deduped.values(), key=lambda entry: int(entry["sequence"])
    )
    state = session.get("chat_state") or {}
    task_frame = (
        state.get("active_task_frame")
        or state.get("last_completed_task_frame")
    )
    snapshot = state.get("canonical_image_bindings") or {}
    referenced_turn = next(
        (
            entry.get("payload")
            for entry in selected_entries
            if referenced_turn_id
            and str(entry.get("turn_id")) == referenced_turn_id
            and entry.get("entry_type") == "assistant_public"
        ),
        None,
    )
    sections = {
        "Stable Policy": (
            "只使用当前会话权威图片绑定和相关公开历史；不得把内部状态暴露给用户。"
        ),
        "Canonical Image Bindings": [
            {
                "image_label": label,
                "active": item.get("active"),
                "locked": item.get("locked"),
            }
            for label, item in (snapshot.get("bindings") or {}).items()
            if item.get("active")
        ],
        "Referenced Previous Turn": referenced_turn,
        "Relevant Tool/Decision State": task_frame,
        "Current Image Scope": current_image_scope,
        "Original User Message": original_user_message,
        "Optional Standalone Rewrite": standalone_request,
    }
    return {
        "schema_version": CONTEXT_PROJECTION_SCHEMA,
        "sections": sections,
        "context_projection_entry_ids": [
            str(entry["entry_id"]) for entry in selected_entries
        ],
        "compaction_entry_id": (
            latest_compaction.get("entry_id")
            if latest_compaction
            else None
        ),
    }


def render_context_projection(projection: dict[str, Any]) -> str:
    """Render the seven partitions without exposing backend asset identity."""

    sections = projection.get("sections") or {}
    lines: list[str] = []
    for name in (
        "Stable Policy",
        "Canonical Image Bindings",
        "Referenced Previous Turn",
        "Relevant Tool/Decision State",
        "Current Image Scope",
        "Original User Message",
        "Optional Standalone Rewrite",
    ):
        value = sections.get(name)
        if value in (None, "", [], {}):
            continue
        if name == "Canonical Image Bindings":
            rendered = "；".join(
                f"{item.get('image_label')}（"
                f"{'有效' if item.get('active') else '失效'}，"
                f"{'已锁定' if item.get('locked') else '未锁定'}）"
                for item in value
            )
        elif name == "Relevant Tool/Decision State":
            rendered = (
                f"任务={value.get('task_type') or '未指定'}；"
                f"动作={value.get('action') or '未指定'}；"
                f"标准={value.get('criterion') or '未指定'}；"
                f"k={value.get('k') or 1}；"
                f"目标={ '、'.join(value.get('target_images') or []) }"
            )
        elif name == "Referenced Previous Turn":
            rendered = str(value.get("public_answer") or "")
        elif isinstance(value, list):
            rendered = "、".join(str(item) for item in value)
        else:
            rendered = str(value)
        lines.append(f"[{name}]\n{rendered}")
    return "\n\n".join(lines)


def validate_model_bindings(
    session: dict[str, Any],
    assets: list[dict[str, Any]],
    requested_labels: list[str],
) -> dict[str, Any]:
    """Validate model bindings."""
    snapshot = (
        session.get("chat_state", {}).get("canonical_image_bindings")
        or {}
    )
    bindings = snapshot.get("bindings") or {}
    if requested_labels != [
        str(item.get("image_label")) for item in assets
    ]:
        raise ValueError("canonical_binding_label_order_mismatch")
    actual_asset_ids: list[str] = []
    actual_sha: list[str] = []
    image_block_order: list[dict[str, str]] = []
    for label, asset in zip(requested_labels, assets):
        binding = bindings.get(label)
        if not binding or not binding.get("active"):
            raise ValueError(f"canonical_binding_missing_or_inactive:{label}")
        asset_id = str(
            asset.get("asset_id")
            or asset.get("source_asset_id")
            or asset.get("image_id")
            or ""
        )
        expected_asset_id = str(binding.get("asset_id") or "")
        if asset_id != expected_asset_id:
            raise ValueError(f"canonical_binding_asset_mismatch:{label}")
        path = Path(str(asset.get("path") or ""))
        if not path.is_file():
            raise ValueError(f"canonical_binding_image_path_missing:{label}")
        sha = file_sha256(path)
        expected_sha = str(binding.get("image_sha256") or "")
        if not expected_sha or sha != expected_sha:
            raise ValueError(f"canonical_binding_sha_mismatch:{label}")
        actual_asset_ids.append(asset_id)
        actual_sha.append(sha)
        image_block_order.append(
            {"image_label": label, "asset_id": asset_id, "sha256": sha}
        )
    return {
        "canonical_image_bindings": snapshot,
        "binding_snapshot_sha256": snapshot.get(
            "binding_snapshot_sha256"
        ),
        "actual_asset_ids": actual_asset_ids,
        "actual_image_sha256": actual_sha,
        "image_block_order": image_block_order,
        "model_send_order": requested_labels,
    }
