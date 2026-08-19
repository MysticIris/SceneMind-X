"""Shared retrieval candidate lifecycle, eligibility and bounded refill contract."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable


ARCHIVED_STATES = {"archived", "legacy_archived"}
INACTIVE_STATES = {"inactive", "deleted", "tombstone"}


class AssetLifecycleRegistry:
    """Read one authoritative, reversible asset lifecycle registry.

    The registry is deliberately separate from the immutable Faiss mapping.
    It changes search eligibility without deleting an asset, mapping row,
    media file or vector.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path.resolve() if path is not None else None

    def _payload(self) -> dict[str, Any]:
        if self.path is None or not self.path.is_file():
            return {"records": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {"records": []}
        return payload if isinstance(payload, dict) else {"records": []}

    def records(self) -> list[dict[str, Any]]:
        """Execute the records operation."""
        records = self._payload().get("records", [])
        return [dict(item) for item in records if isinstance(item, dict)]

    def lookup(self, record: dict[str, Any] | str) -> dict[str, Any]:
        """Execute the lookup operation."""
        asset_id = (
            str(record)
            if isinstance(record, str)
            else str(record.get("asset_id") or record.get("image_id") or "")
        )
        source = (
            ""
            if isinstance(record, str)
            else str(record.get("source") or "")
        )
        for item in self.records():
            if str(item.get("asset_id") or "") != asset_id:
                continue
            required_source = str(item.get("source") or "")
            if required_source and source and required_source != source:
                continue
            state = str(item.get("lifecycle_state") or "active")
            searchable = bool(
                item.get(
                    "searchable",
                    state not in ARCHIVED_STATES | INACTIVE_STATES,
                )
            )
            return {
                **item,
                "lifecycle_state": state,
                "searchable": searchable,
                "lifecycle_label": (
                    "历史封存资产"
                    if state in ARCHIVED_STATES
                    else "不可用资产"
                    if not searchable
                    else "活动资产"
                ),
            }
        return {
            "asset_id": asset_id,
            "lifecycle_state": "active",
            "searchable": True,
            "lifecycle_label": "活动资产",
        }

    def is_searchable(self, record: dict[str, Any]) -> bool:
        """Execute the is searchable operation."""
        state = self.lookup(record)
        return bool(state["searchable"]) and str(
            state["lifecycle_state"]
        ) not in ARCHIVED_STATES | INACTIVE_STATES

    def summary(
        self,
        records: Iterable[dict[str, Any]],
        *,
        physical_ntotal: int | None = None,
    ) -> dict[str, Any]:
        """Execute the summary operation."""
        values = [dict(item) for item in records]
        active = [item for item in values if self.is_searchable(item)]
        archived = [
            item
            for item in values
            if str(self.lookup(item)["lifecycle_state"]) in ARCHIVED_STATES
        ]
        all_sha = [
            str(item.get("sha256") or item.get("image_sha256") or "")
            for item in values
            if item.get("sha256") or item.get("image_sha256")
        ]
        active_sha = [
            str(item.get("sha256") or item.get("image_sha256") or "")
            for item in active
            if item.get("sha256") or item.get("image_sha256")
        ]
        counts = Counter(all_sha)
        return {
            "physical_vectors": int(
                len(values) if physical_ntotal is None else physical_ntotal
            ),
            "asset_records": len(values),
            "active_searchable_records": len(active),
            "active_searchable_unique_sha": len(set(active_sha)),
            "archived_records": len(archived),
            "total_unique_sha": len(set(all_sha)),
            "duplicate_sha_records": sum(
                max(0, count - 1) for count in counts.values()
            ),
            "lifecycle_registry_configured": bool(
                self.path is not None and self.path.is_file()
            ),
        }


def effective_library_id(item: dict[str, Any]) -> str:
    """Execute the effective library id operation."""
    if str(item.get("source") or "") == "frozen_library":
        return "legacy_frozen"
    return str(item.get("library_id") or "default")


def candidate_identity(item: dict[str, Any]) -> str:
    """Execute the candidate identity operation."""
    return str(
        item.get("sha256")
        or item.get("image_sha256")
        or (
            f"{effective_library_id(item)}:"
            f"{item.get('asset_id') or item.get('image_id') or ''}"
        )
    )


def candidate_asset_id(item: dict[str, Any]) -> str:
    """Execute the candidate asset id operation."""
    return str(item.get("asset_id") or item.get("image_id") or "")


def candidate_eligibility(
    item: dict[str, Any],
    *,
    requested_library_ids: set[str] | None = None,
    exclude_asset_ids: set[str] | None = None,
) -> tuple[bool, str]:
    """Execute the candidate eligibility operation."""
    asset_id = candidate_asset_id(item)
    if not asset_id:
        return False, "missing_asset_id"
    state = str(item.get("lifecycle_state") or "active")
    if state in ARCHIVED_STATES:
        return False, "archived"
    if state in INACTIVE_STATES or item.get("active") is False:
        return False, "inactive"
    if item.get("searchable") is False:
        return False, "not_searchable"
    if item.get("media_available") is False or item.get("media_status") in {
        "missing",
        "unresolved",
    }:
        return False, "media_unavailable"
    if exclude_asset_ids and asset_id in exclude_asset_ids:
        return False, "excluded"
    if (
        requested_library_ids is not None
        and effective_library_id(item) not in requested_library_ids
    ):
        return False, "out_of_scope"
    return True, "eligible"


def _representative_key(
    item: dict[str, Any],
    *,
    current_library_id: str | None,
) -> tuple[Any, ...]:
    library_id = effective_library_id(item)
    source = str(item.get("source") or "")
    state = str(item.get("lifecycle_state") or "active")
    return (
        0 if current_library_id and library_id == current_library_id else 1,
        0 if source != "frozen_library" else 1,
        0 if state not in ARCHIVED_STATES else 1,
        0 if item.get("media_available", True) else 1,
        -float(item.get("score") or 0.0),
        str(item.get("created_at") or item.get("indexed_at") or ""),
        candidate_asset_id(item),
    )


def select_unique_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    requested_k: int,
    requested_library_ids: set[str] | None = None,
    exclude_asset_ids: set[str] | None = None,
    current_library_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Execute the select unique candidates operation."""
    counters: Counter[str] = Counter()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        item = dict(raw)
        eligible, reason = candidate_eligibility(
            item,
            requested_library_ids=requested_library_ids,
            exclude_asset_ids=exclude_asset_ids,
        )
        if not eligible:
            counters[reason] += 1
            continue
        grouped.setdefault(candidate_identity(item), []).append(item)
    selected = []
    for group in grouped.values():
        if len(group) > 1:
            counters["duplicate_sha"] += len(group) - 1
        selected.append(
            min(
                group,
                key=lambda item: _representative_key(
                    item,
                    current_library_id=current_library_id,
                ),
            )
        )
    selected.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            effective_library_id(item),
            candidate_asset_id(item),
        )
    )
    return selected[:requested_k], dict(counters)


def adaptive_topk_refill(
    fetch_candidates: Callable[[int], list[dict[str, Any]]],
    *,
    requested_k: int,
    total_candidates: int,
    requested_library_ids: set[str] | None = None,
    exclude_asset_ids: set[str] | None = None,
    current_library_id: str | None = None,
    initial_fetch: int | None = None,
    safety_cap: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch a bounded candidate pool until K unique eligible rows or exhaustion."""

    if requested_k < 1 or requested_k > 100:
        raise ValueError("requested_top_k_must_be_between_1_and_100")
    total = max(0, int(total_candidates))
    cap = total if safety_cap is None else min(total, max(1, safety_cap))
    fetch_n = min(cap, max(int(initial_fetch or 0), 4 * requested_k, 20))
    history: list[dict[str, Any]] = []
    previous_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    stop_reason = "candidate_exhausted"

    if total <= 0:
        return [], {
            "requested_k": requested_k,
            "rounds": 0,
            "fetch_history": [],
            "exhausted": True,
            "stop_reason": "empty_index",
            "final_unique_count": 0,
            "query_embedding_count": 1,
        }

    while True:
        rows = list(fetch_candidates(fetch_n))
        selected, counters = select_unique_candidates(
            rows,
            requested_k=requested_k,
            requested_library_ids=requested_library_ids,
            exclude_asset_ids=exclude_asset_ids,
            current_library_id=current_library_id,
        )
        asset_ids = {candidate_asset_id(item) for item in rows}
        new_candidates = len(asset_ids - previous_ids)
        history.append(
            {
                "fetch_n": fetch_n,
                "returned_candidates": len(rows),
                "new_candidates": new_candidates,
                "eligible_unique": len(selected),
                "filtered": counters,
            }
        )
        if len(selected) >= requested_k:
            stop_reason = "requested_k_fulfilled"
            break
        if fetch_n >= cap or len(rows) < fetch_n:
            stop_reason = "candidate_exhausted"
            break
        next_fetch = min(cap, fetch_n * 2)
        if next_fetch <= fetch_n:
            stop_reason = "safety_cap_reached"
            break
        if previous_ids and not new_candidates:
            stop_reason = "no_new_candidates"
            break
        previous_ids = asset_ids
        fetch_n = next_fetch

    exhausted = stop_reason in {
        "empty_index",
        "candidate_exhausted",
        "no_new_candidates",
        "safety_cap_reached",
    } and len(selected) < requested_k
    debug = {
        "requested_k": requested_k,
        "rounds": len(history),
        "fetch_history": history,
        "exhausted": exhausted,
        "stop_reason": stop_reason,
        "final_unique_count": len(selected),
        "query_embedding_count": 1,
    }
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
        item["candidate_refill"] = debug
    return selected, debug
