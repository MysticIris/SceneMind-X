"""Reproducible retrieval benchmark helpers for Phase 5.3.

The benchmark deliberately keeps relevance judgments outside model outputs.
Only Train assets are accepted, and every query is evaluated against the same
frozen manifest and graded human review.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


RELEVANCE_GRADES = {
    "irrelevant": 0,
    "weakly_relevant": 1,
    "relevant": 2,
    "strongly_relevant": 3,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load jsonl."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def normalize_grade(value: Any) -> int:
    """Normalize grade."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a relevance grade")
    if isinstance(value, int) and 0 <= value <= 3:
        return value
    if isinstance(value, str) and value in RELEVANCE_GRADES:
        return RELEVANCE_GRADES[value]
    raise ValueError(f"unsupported relevance grade: {value!r}")


def validate_benchmark(
    queries: Iterable[Mapping[str, Any]],
    *,
    available_image_ids: set[str],
) -> list[dict[str, Any]]:
    """Validate benchmark."""
    required = {
        "query_id",
        "query_text",
        "query_image",
        "expected_assets",
        "graded_relevance",
        "query_type",
        "evidence",
        "reviewer_note",
    }
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in queries:
        row = dict(raw)
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{row.get('query_id', '<unknown>')}: missing fields {missing}")
        query_id = str(row["query_id"])
        if query_id in seen_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen_ids.add(query_id)
        mode = str(row.get("mode", "text"))
        if mode not in {"text", "image", "hybrid"}:
            raise ValueError(f"{query_id}: unsupported mode {mode}")
        query_image = row["query_image"]
        if mode in {"image", "hybrid"}:
            if not query_image or str(query_image) not in available_image_ids:
                raise ValueError(f"{query_id}: query_image is not in the frozen manifest")
        elif query_image not in (None, ""):
            raise ValueError(f"{query_id}: text query must not declare query_image")
        expected = [str(value) for value in row["expected_assets"]]
        if not expected:
            raise ValueError(f"{query_id}: expected_assets must not be empty")
        missing_assets = sorted(set(expected) - available_image_ids)
        if missing_assets:
            raise ValueError(f"{query_id}: assets absent from manifest: {missing_assets}")
        grades = {
            str(image_id): normalize_grade(value)
            for image_id, value in dict(row["graded_relevance"]).items()
        }
        if set(expected) != set(grades):
            raise ValueError(f"{query_id}: expected_assets and graded_relevance keys differ")
        if max(grades.values(), default=0) < 2:
            raise ValueError(f"{query_id}: at least one asset must have grade >= 2")
        row["query_id"] = query_id
        row["mode"] = mode
        row["query_text"] = str(row["query_text"])
        row["query_image"] = str(query_image) if query_image else None
        row["expected_assets"] = expected
        row["graded_relevance"] = grades
        validated.append(row)
    if not validated:
        raise ValueError("benchmark must contain at least one query")
    return validated


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _query_metrics(query: Mapping[str, Any], results: list[Mapping[str, Any]]) -> dict[str, Any]:
    grades_by_id = dict(query["graded_relevance"])
    result_grades = [grades_by_id.get(str(row["image_id"]), 0) for row in results[:5]]
    padded = [*result_grades, *([0] * max(0, 5 - len(result_grades)))]
    first_relevant_rank = next(
        (rank for rank, grade in enumerate(result_grades, start=1) if grade >= 2),
        None,
    )
    ideal = sorted(grades_by_id.values(), reverse=True)[:5]
    ideal_dcg = _dcg(ideal)
    return {
        "recall_at_1": float(bool(result_grades and result_grades[0] >= 2)),
        "recall_at_5": float(any(grade >= 2 for grade in result_grades)),
        "mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "ndcg_at_5": _dcg(padded[:5]) / ideal_dcg if ideal_dcg else 0.0,
        "top5_manual_relevance_rate": sum(grade >= 1 for grade in padded[:5]) / 5.0,
        "top5_strict_relevance_rate": sum(grade >= 2 for grade in padded[:5]) / 5.0,
        "first_relevant_rank": first_relevant_rank,
        "result_grades": padded[:5],
    }


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"queries": 0}
    metrics = (
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
        "top5_manual_relevance_rate",
        "top5_strict_relevance_rate",
    )
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "queries": len(rows),
        **{
            name: statistics.fmean(float(row["metrics"][name]) for row in rows)
            for name in metrics
        },
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies),
        },
    }


def evaluate_retrieval(
    *,
    service: Any,
    queries: Iterable[Mapping[str, Any]],
    image_paths: Mapping[str, Path],
    top_k: int = 5,
    require_relevant_in_index: bool = False,
) -> dict[str, Any]:
    """Execute the evaluate retrieval operation."""
    available = set(service.image_ids)
    query_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for query in queries:
        grades_in_index = {
            image_id: grade
            for image_id, grade in dict(query["graded_relevance"]).items()
            if image_id in available
            and not (
                query["mode"] in {"image", "hybrid"}
                and image_id == query.get("query_image")
            )
        }
        if require_relevant_in_index and max(grades_in_index.values(), default=0) < 2:
            skipped.append(
                {
                    "query_id": str(query["query_id"]),
                    "reason": "no_grade_2_or_3_target_in_evaluation_index",
                }
            )
            continue
        evaluation_query = {**query, "graded_relevance": grades_in_index}
        started = time.perf_counter()
        if query["mode"] == "text":
            results = service.search(str(query["query_text"]), top_k=top_k)
        elif query["mode"] == "image":
            results = service.search_image(
                image_paths[str(query["query_image"])],
                top_k=top_k,
                exclude_image_id=str(query["query_image"]),
            )
        else:
            results = service.search_hybrid(
                image_paths[str(query["query_image"])],
                str(query["query_text"]),
                top_k=top_k,
                exclude_image_id=str(query["query_image"]),
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        query_rows.append(
            {
                "query_id": query["query_id"],
                "mode": query["mode"],
                "query_type": query["query_type"],
                "query_text": query["query_text"],
                "query_image": query["query_image"],
                "latency_ms": latency_ms,
                "results": results,
                "metrics": _query_metrics(evaluation_query, results),
            }
        )
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        by_mode[str(row["mode"])].append(row)
        by_type[str(row["query_type"])].append(row)
    return {
        "metric_contract": {
            "relevant_threshold": "grade >= 2",
            "recall_at_k": "query hit rate: at least one grade >= 2 asset appears in top-k",
            "mrr": "reciprocal rank of the first grade >= 2 asset",
            "ndcg_at_5": "graded DCG using gain 2^grade-1, normalized by all judged positives",
            "top5_manual_relevance_rate": "fraction of five slots with grade >= 1; missing slots count 0",
            "unjudged_assets": "implicitly grade 0 (irrelevant)",
        },
        "overall": _aggregate(query_rows),
        "by_mode": {key: _aggregate(value) for key, value in sorted(by_mode.items())},
        "by_query_type": {key: _aggregate(value) for key, value in sorted(by_type.items())},
        "queries": query_rows,
        "skipped_queries": skipped,
    }


def evaluate_ranked_results(
    *,
    queries: Iterable[Mapping[str, Any]],
    results_by_query_id: Mapping[str, list[Mapping[str, Any]]],
    latency_ms_by_query_id: Mapping[str, float] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Evaluate externally produced rankings with the frozen benchmark contract."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    query_rows: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    for query in queries:
        query_id = str(query["query_id"])
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen_query_ids.add(query_id)
        if query_id not in results_by_query_id:
            raise ValueError(f"missing results for query: {query_id}")
        raw_results = list(results_by_query_id[query_id])
        results = [dict(row) for row in raw_results[:top_k]]
        result_ids = [str(row["image_id"]) for row in results]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError(f"duplicate result image_id for query: {query_id}")
        query_image = str(query["query_image"]) if query.get("query_image") else None
        if query["mode"] in {"image", "hybrid"} and query_image in result_ids:
            raise ValueError(f"query asset leaked into results: {query_id}")
        grades = {
            str(image_id): int(grade)
            for image_id, grade in dict(query["graded_relevance"]).items()
            if image_id != query_image
        }
        evaluation_query = {**query, "graded_relevance": grades}
        query_rows.append(
            {
                "query_id": query_id,
                "mode": query["mode"],
                "query_type": query["query_type"],
                "query_text": query["query_text"],
                "query_image": query_image,
                "latency_ms": float((latency_ms_by_query_id or {}).get(query_id, 0.0)),
                "results": results,
                "metrics": _query_metrics(evaluation_query, results),
            }
        )
    extra = sorted(set(results_by_query_id) - seen_query_ids)
    if extra:
        raise ValueError(f"results contain unknown queries: {extra}")
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        by_mode[str(row["mode"])].append(row)
        by_type[str(row["query_type"])].append(row)
    return {
        "metric_contract": {
            "relevant_threshold": "grade >= 2",
            "recall_at_k": "query hit rate: at least one grade >= 2 asset appears in top-k",
            "mrr": "reciprocal rank of the first grade >= 2 asset",
            "ndcg_at_5": "graded DCG using gain 2^grade-1, normalized by all judged positives",
            "top5_manual_relevance_rate": "fraction of five slots with grade >= 1; missing slots count 0",
            "unjudged_assets": "implicitly grade 0 (irrelevant)",
        },
        "overall": _aggregate(query_rows),
        "by_mode": {key: _aggregate(value) for key, value in sorted(by_mode.items())},
        "by_query_type": {key: _aggregate(value) for key, value in sorted(by_type.items())},
        "queries": query_rows,
        "skipped_queries": [],
    }
