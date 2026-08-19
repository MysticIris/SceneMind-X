"""Small persistent cosine-similarity index for the Phase 1 library."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .contracts import EmbeddingService


class RetrievalService:
    """Provide retrieval service behavior."""
    def __init__(self, embedding: EmbeddingService, index_path: Path) -> None:
        self.embedding = embedding
        self.index_path = index_path
        self.image_ids: list[str] = []
        self.texts: list[str] = []
        self.sources: list[str] = []
        self.image_urls: list[str] = []
        self.vectors: Any | None = None
        if index_path.is_file():
            self.load()

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return {
            "status": "ready" if self.vectors is not None else "not_built",
            "items": len(self.image_ids),
            "index_path": str(self.index_path),
            "embedding": self.embedding.status(),
            "text_scoring": "unicode_bigram_jaccard+query_character_coverage+exact_phrase_overlap",
        }

    def build_index(self, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Build index."""
        import numpy as np

        rows = list(items)
        if not rows:
            raise ValueError("cannot build an empty retrieval index")
        ids: list[str] = []
        texts: list[str] = []
        vectors: list[list[float]] = []
        sources: list[str] = []
        image_urls: list[str] = []
        for row in rows:
            ids.append(str(row["image_id"]))
            texts.append(str(row.get("retrieval_text", "")))
            vectors.append(self.embedding.encode_image(Path(row["image_path"])))
            sources.append(str(row.get("source", "frozen_library")))
            image_urls.append(str(row.get("image_url", f"/library/{row['image_id']}/image")))
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError("embedding service returned an invalid matrix")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=self.index_path.parent, suffix=".npz") as handle:
            temporary = Path(handle.name)
        try:
            np.savez_compressed(
                temporary,
                image_ids=np.asarray(ids),
                texts=np.asarray(texts),
                vectors=matrix,
                sources=np.asarray(sources),
                image_urls=np.asarray(image_urls),
            )
            os.replace(temporary, self.index_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.image_ids = ids
        self.texts = texts
        self.vectors = matrix
        self.sources = sources
        self.image_urls = image_urls
        return self.status()

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        import numpy as np

        with np.load(self.index_path, allow_pickle=False) as value:
            self.image_ids = [str(item) for item in value["image_ids"].tolist()]
            self.texts = [str(item) for item in value["texts"].tolist()]
            self.vectors = value["vectors"].astype(np.float32)
            self.sources = [str(item) for item in value["sources"].tolist()] if "sources" in value else ["frozen_library"] * len(self.image_ids)
            self.image_urls = [str(item) for item in value["image_urls"].tolist()] if "image_urls" in value else [f"/library/{item}/image" for item in self.image_ids]
        if len(self.image_ids) != len(self.texts) or len(self.image_ids) != len(self.vectors):
            raise ValueError("persisted retrieval index is inconsistent")
        return self.status()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Execute the search operation."""
        import numpy as np

        if self.vectors is None:
            raise RuntimeError("retrieval_index_not_built")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        scores = np.asarray([self._text_similarity(query, value) for value in self.texts], dtype=np.float32)
        order = sorted(
            range(len(scores)),
            key=lambda index: (
                -float(scores[index]),
                self.image_ids[index],
            ),
        )[: min(top_k, len(scores))]
        return [
            {
                "rank": rank + 1,
                "image_id": self.image_ids[int(index)],
                "score": float(scores[int(index)]),
                "match_basis": self.texts[int(index)],
                "source": self.sources[int(index)],
                "image_url": self.image_urls[int(index)],
                "text_score": float(scores[int(index)]),
            }
            for rank, index in enumerate(order)
        ]

    def search_image(self, image_path: Path, top_k: int = 5, *, exclude_image_id: str | None = None) -> list[dict[str, Any]]:
        """Execute the search image operation."""
        import numpy as np

        if self.vectors is None:
            raise RuntimeError("retrieval_index_not_built")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        query_vector = np.asarray(self.embedding.encode_image(image_path), dtype=np.float32)
        if query_vector.shape != (self.vectors.shape[1],):
            raise ValueError("query image and index embedding dimensions differ")
        scores = self.vectors @ query_vector
        order = sorted(
            range(len(scores)),
            key=lambda index: (
                -float(scores[index]),
                self.image_ids[index],
            ),
        )
        results = []
        for index in order:
            image_id = self.image_ids[int(index)]
            if exclude_image_id and image_id == exclude_image_id:
                continue
            results.append({"rank": len(results) + 1, "image_id": image_id, "score": float(scores[int(index)]), "image_score": float(scores[int(index)]), "match_basis": self.texts[int(index)], "source": self.sources[int(index)], "image_url": self.image_urls[int(index)]})
            if len(results) >= top_k:
                break
        return results

    def search_hybrid(self, image_path: Path, query: str, top_k: int = 5, *, image_weight: float = 0.55, text_weight: float = 0.35, lexical_weight: float = 0.10, exclude_image_id: str | None = None) -> list[dict[str, Any]]:
        """Execute the search hybrid operation."""
        import numpy as np

        if self.vectors is None:
            raise RuntimeError("retrieval_index_not_built")
        weights = np.asarray([image_weight, text_weight, lexical_weight], dtype=np.float32)
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-5):
            raise ValueError("hybrid search weights must be non-negative and sum to 1")
        image_vector = np.asarray(self.embedding.encode_image(image_path), dtype=np.float32)
        image_scores = self.vectors @ image_vector
        text_scores = np.asarray([self._text_similarity(query, value) for value in self.texts], dtype=np.float32)
        lexical_scores = np.asarray([self._lexical_score(query, value) for value in self.texts], dtype=np.float32)
        final_scores = image_weight * image_scores + text_weight * text_scores + lexical_weight * lexical_scores
        order = sorted(
            range(len(final_scores)),
            key=lambda index: (
                -float(final_scores[index]),
                self.image_ids[index],
            ),
        )
        results = []
        for index in order:
            image_id = self.image_ids[int(index)]
            if exclude_image_id and image_id == exclude_image_id:
                continue
            results.append({
                "rank": len(results) + 1,
                "image_id": image_id,
                "image_score": float(image_scores[int(index)]),
                "text_score": float(text_scores[int(index)]),
                "lexical_score": float(lexical_scores[int(index)]),
                "final_score": float(final_scores[int(index)]),
                "score": float(final_scores[int(index)]),
                "weights": {"image": image_weight, "text": text_weight, "lexical": lexical_weight},
                "route": "deterministic_weighted_fusion",
                "fallback": False,
                "match_basis": self.texts[int(index)],
                "source": self.sources[int(index)],
                "image_url": self.image_urls[int(index)],
            })
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _units(value: str) -> set[str]:
        compact = re.sub(r"\s+", "", value.lower())
        if len(compact) < 2:
            return {compact} if compact else set()
        return {compact[index : index + 2] for index in range(len(compact) - 1)}

    @staticmethod
    def _characters(value: str) -> set[str]:
        return {character for character in value.lower() if character.isalnum()}

    @classmethod
    def _text_similarity(cls, query: str, text: str) -> float:
        query_units = cls._units(query)
        text_units = cls._units(text)
        query_characters = cls._characters(query)
        text_characters = cls._characters(text)
        if not query_characters or not text_characters:
            return 0.0
        bigram_score = (
            len(query_units & text_units) / len(query_units | text_units)
            if query_units and text_units
            else 0.0
        )
        character_coverage = len(query_characters & text_characters) / len(query_characters)
        compact_query = re.sub(r"\s+", "", query.lower())
        compact_text = re.sub(r"\s+", "", text.lower())
        exact_phrase = 1.0 if compact_query and compact_query in compact_text else 0.0
        return 0.55 * bigram_score + 0.30 * character_coverage + 0.15 * exact_phrase

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        normalized = text.lower()
        terms = [term for term in re.split(r"[\s，。！？、；：]+", query.lower()) if term]
        return sum(term in normalized for term in terms) / max(1, len(terms))

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Expose the future reranker boundary without fabricating a score."""

        if not query.strip():
            raise ValueError("query must not be empty")
        return {
            "status": "not_implemented",
            "reason": "reranker_training_is_out_of_scope_for_phase1",
            "query": query,
            "candidates": candidates,
        }
