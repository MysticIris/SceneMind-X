"""Contracts and deterministic ranking helpers for Phase 5.3 reranking."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def validate_candidate_pools(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_query_ids: Sequence[str],
    expected_pool_size: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """Validate and return the frozen Stage 3 primary candidate pools."""

    expected = [str(value) for value in expected_query_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected query IDs contain duplicates")
    pools: dict[str, list[dict[str, Any]]] = {}
    for raw_row in rows:
        query_id = str(raw_row["query_id"])
        if query_id in pools:
            raise ValueError(f"duplicate candidate pool: {query_id}")
        candidates = [dict(value) for value in raw_row["primary"]]
        candidate_ids = [str(value["image_id"]) for value in candidates]
        if len(candidates) != expected_pool_size:
            raise ValueError(
                f"{query_id}: expected {expected_pool_size} candidates, "
                f"found {len(candidates)}"
            )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"{query_id}: duplicate candidate image IDs")
        query_image = raw_row.get("query_image")
        if query_image and str(query_image) in candidate_ids:
            raise ValueError(f"{query_id}: query image leaked into candidate pool")
        for rank, candidate in enumerate(candidates, start=1):
            candidate["image_id"] = str(candidate["image_id"])
            candidate["original_rank"] = rank
        pools[query_id] = candidates
    if set(pools) != set(expected):
        raise ValueError(
            "candidate pool query IDs differ: "
            f"missing={sorted(set(expected) - set(pools))}, "
            f"extra={sorted(set(pools) - set(expected))}"
        )
    return pools


def rerank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Sort by reranker score with a frozen, deterministic tie-break."""

    if len(candidates) != len(scores):
        raise ValueError("candidate and score counts differ")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fallback_rank, (candidate, raw_score) in enumerate(
        zip(candidates, scores, strict=True),
        start=1,
    ):
        image_id = str(candidate["image_id"])
        if image_id in seen:
            raise ValueError(f"duplicate candidate image ID: {image_id}")
        seen.add(image_id)
        score = float(raw_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid reranker score for {image_id}: {score}")
        original_rank = int(candidate.get("original_rank", fallback_rank))
        rows.append(
            {
                **dict(candidate),
                "image_id": image_id,
                "reranker_score": score,
                "original_rank": original_rank,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["reranker_score"]),
            int(row["original_rank"]),
            str(row["image_id"]),
        )
    )
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        rows = rows[:top_k]
    return [
        {**row, "rank": rank, "source": "qwen3_vl_reranker_2b"}
        for rank, row in enumerate(rows, start=1)
    ]


def rerank_at_depth(
    candidates: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    depth: int,
    output_top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rerank the first ``depth`` candidates and preserve the remaining order."""

    if not 1 <= depth <= len(candidates):
        raise ValueError("depth must fall within the candidate pool")
    reranked = rerank_candidates(candidates[:depth], scores[:depth])
    remaining = [
        {
            **dict(candidate),
            "image_id": str(candidate["image_id"]),
            "original_rank": int(candidate.get("original_rank", rank)),
            "reranker_score": None,
            "source": "stage3_primary_order",
        }
        for rank, candidate in enumerate(candidates[depth:], start=depth + 1)
    ]
    merged = [*reranked, *remaining]
    return [
        {**row, "rank": rank}
        for rank, row in enumerate(merged[:output_top_k], start=1)
    ]


def top1_accuracy(
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    cases: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate frozen comparison groups with one or more accepted best assets."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        accepted = {str(value) for value in case["accepted_top1"]}
        ranking = list(rankings.get(case_id, ()))
        predicted = str(ranking[0]["image_id"]) if ranking else None
        rows.append(
            {
                "case_id": case_id,
                "predicted_top1": predicted,
                "accepted_top1": sorted(accepted),
                "correct": predicted in accepted,
            }
        )
    return {
        "cases": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": (
            sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "rows": rows,
    }
