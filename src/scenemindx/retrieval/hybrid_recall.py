"""Small-corpus BM25 and score-fusion helpers for Phase 5.3 Stage 3."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[\u3400-\u9fff]+|[a-z0-9]+", re.IGNORECASE)


def normalize_search_text(value: str) -> str:
    """Normalize search text."""
    return unicodedata.normalize("NFKC", value).lower().strip()


def character_bigram_tokens(value: str) -> list[str]:
    """Execute the character bigram tokens operation."""
    tokens: list[str] = []
    for segment in TOKEN_RE.findall(normalize_search_text(value)):
        if re.fullmatch(r"[\u3400-\u9fff]+", segment):
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        else:
            tokens.append(segment)
    return tokens


def jieba_search_tokens(value: str) -> list[str]:
    """Execute the jieba search tokens operation."""
    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError("jieba_0_42_1_is_required_for_this_ablation") from exc
    normalized = normalize_search_text(value)
    return [
        token
        for token in (part.strip() for part in jieba.lcut_for_search(normalized, HMM=True))
        if token and TOKEN_RE.search(token)
    ]


@dataclass(frozen=True)
class BM25Parameters:
    """Provide b m25 parameters behavior."""
    k1: float = 1.2
    b: float = 0.75


class BM25Index:
    """Provide b m25 index behavior."""
    def __init__(
        self,
        documents: Sequence[tuple[str, str]],
        *,
        tokenizer: Callable[[str], list[str]],
        parameters: BM25Parameters | None = None,
        source: str,
    ) -> None:
        if not documents:
            raise ValueError("BM25 requires at least one document")
        self.parameters = parameters or BM25Parameters()
        self.tokenizer = tokenizer
        self.source = source
        self.image_ids = [str(image_id) for image_id, _ in documents]
        if len(self.image_ids) != len(set(self.image_ids)):
            raise ValueError("BM25 document IDs must be unique")
        self.term_frequencies = [Counter(tokenizer(text)) for _, text in documents]
        self.lengths = [sum(row.values()) for row in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths)
        document_frequency: Counter[str] = Counter()
        for row in self.term_frequencies:
            document_frequency.update(row.keys())
        count = len(documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score_all(self, query: str) -> list[tuple[str, float]]:
        """Execute the score all operation."""
        query_terms = self.tokenizer(query)
        if not query_terms:
            return []
        k1 = self.parameters.k1
        b = self.parameters.b
        scores: list[tuple[str, float]] = []
        for image_id, frequencies, length in zip(
            self.image_ids,
            self.term_frequencies,
            self.lengths,
        ):
            score = 0.0
            norm = 1.0 - b + b * (length / self.average_length if self.average_length else 0.0)
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                score += self.idf.get(term, 0.0) * (
                    frequency * (k1 + 1.0) / (frequency + k1 * norm)
                )
            if score > 0:
                scores.append((image_id, score))
        return sorted(scores, key=lambda row: (-row[1], row[0]))

    def search(
        self,
        query: str,
        *,
        top_k: int,
        exclude_image_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the search operation."""
        results: list[dict[str, Any]] = []
        for image_id, score in self.score_all(query):
            if exclude_image_id and image_id == exclude_image_id:
                continue
            results.append(
                {
                    "rank": len(results) + 1,
                    "image_id": image_id,
                    "score": float(score),
                    "source": self.source,
                }
            )
            if len(results) >= top_k:
                break
        return results


def _validated_branch(
    rows: Iterable[Mapping[str, Any]],
    *,
    exclude_image_id: str | None,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in rows:
        image_id = str(raw["image_id"])
        if image_id in seen or (exclude_image_id and image_id == exclude_image_id):
            continue
        seen.add(image_id)
        validated.append(dict(raw))
    return validated


def reciprocal_rank_fusion(
    branches: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    top_k: int,
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
    exclude_image_id: str | None = None,
) -> list[dict[str, Any]]:
    """Execute the reciprocal rank fusion operation."""
    if top_k < 1 or rrf_k < 1:
        raise ValueError("top_k and rrf_k must be positive")
    totals: defaultdict[str, float] = defaultdict(float)
    contributions: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for branch, raw_rows in branches.items():
        weight = float((weights or {}).get(branch, 1.0))
        if weight < 0:
            raise ValueError("fusion weights must be non-negative")
        rows = _validated_branch(raw_rows, exclude_image_id=exclude_image_id)
        for rank, row in enumerate(rows, start=1):
            image_id = str(row["image_id"])
            contribution = weight / (rrf_k + rank)
            totals[image_id] += contribution
            contributions[image_id][branch] = contribution
    ordered = sorted(totals, key=lambda image_id: (-totals[image_id], image_id))[:top_k]
    return [
        {
            "rank": rank,
            "image_id": image_id,
            "score": float(totals[image_id]),
            "source": "rrf",
            "branch_contributions": contributions[image_id],
        }
        for rank, image_id in enumerate(ordered, start=1)
    ]


def normalized_weighted_fusion(
    branches: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    top_k: int,
    weights: Mapping[str, float] | None = None,
    exclude_image_id: str | None = None,
) -> list[dict[str, Any]]:
    """Execute the normalized weighted fusion operation."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    totals: defaultdict[str, float] = defaultdict(float)
    contributions: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for branch, raw_rows in branches.items():
        weight = float((weights or {}).get(branch, 1.0))
        if weight < 0:
            raise ValueError("fusion weights must be non-negative")
        rows = _validated_branch(raw_rows, exclude_image_id=exclude_image_id)
        if not rows:
            continue
        scores = [float(row["score"]) for row in rows]
        minimum, maximum = min(scores), max(scores)
        for row, score in zip(rows, scores):
            normalized = 1.0 if math.isclose(minimum, maximum) else (score - minimum) / (
                maximum - minimum
            )
            image_id = str(row["image_id"])
            contribution = weight * normalized
            totals[image_id] += contribution
            contributions[image_id][branch] = contribution
    ordered = sorted(totals, key=lambda image_id: (-totals[image_id], image_id))[:top_k]
    return [
        {
            "rank": rank,
            "image_id": image_id,
            "score": float(totals[image_id]),
            "source": "normalized_weighted_fusion",
            "branch_contributions": contributions[image_id],
        }
        for rank, image_id in enumerate(ordered, start=1)
    ]
