"""Task-aware visible-text confidence and post-filter repair.

This module intentionally does not change any persisted Canonical schema.  It
only projects the evidence already available to each product task.  Canonical
uses a binary public decision (retain clear text or remove unreliable text),
while Chat and factual generation may expose a middle, explicitly qualified
confidence tier.  Creative generation may use readable text as inspiration
without claiming that invented wording is an observed image fact.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


CANONICAL = "canonical"
CHAT = "chat"
GENERATION_CREATIVE = "generation_creative"
GENERATION_FACTUAL = "generation_factual"
OBJECTIVE_DESCRIPTION = "objective_description"
NEWS_CAPTION = "news_caption"
COMPARE = "compare"
RANK = "rank"

CREATIVE_CONTENT_TYPES = {
    "moments",
    "travel_diary",
    "advertisement",
    "poster_title",
    "poem",
    "story",
    "creative_story",
    "article",
}
FACTUAL_CONTENT_TYPES = {"objective_description", "news_caption"}

_TECHNICAL_TOKEN_RE = re.compile(
    r"^(?=.{2,80}$)(?=.*[A-Za-z0-9])"
    r"[A-Za-z0-9][A-Za-z0-9_+./:#×() \-–—]*[A-Za-z0-9).]$"
)
_CHINESE_READABLE_RE = re.compile(
    r"^[\u3400-\u9fffA-Za-z0-9][\u3400-\u9fffA-Za-z0-9_+./:#×() \-–—]{1,79}$"
)
_ONLY_PUNCTUATION_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_BROKEN_EDGE_RE = re.compile(r"(?:^[-–—，。、；：]+|[-–—，、；：]+$)")
_REPEATED_NOISE_RE = re.compile(r"(.{1,3})\1{3,}")
_PLACEHOLDER_RE = re.compile(
    r"(?:某某|未知|待定|不详|无法辨认|[Xx?？*]{2,}|\bunknown\b|\bn/?a\b)",
    re.IGNORECASE,
)
_UNCERTAINTY_RE = re.compile(
    r"(?:疑似|可能|类似|或许|似乎|大概|推测|看起来像|无法确认|不确定)"
)
_FACT_ASSERTION_RE = re.compile(
    r"(?:图(?:中|上)|画面(?:中|上)|图片(?:中|上)|屏幕(?:中|上)|"
    r"标签|标牌|标题|文字|字幕|型号|编号)"
    r".{0,16}(?:写着|写有|显示|标为|标注为|内容为|是)"
)


@dataclass(frozen=True)
class TextCandidate:
    """Provide text candidate behavior."""
    text: str
    confidence: str
    numeric_confidence: float | None
    source_count: int
    sources: tuple[str, ...]


def normalize_text(value: Any) -> str:
    """Normalize text."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.replace("\x00", "").split()).strip()


def content_type_task_mode(content_type: str | None) -> str:
    """Execute the content type task mode operation."""
    key = normalize_text(content_type).lower()
    if key in CREATIVE_CONTENT_TYPES:
        return GENERATION_CREATIVE
    if key == "news_caption":
        return NEWS_CAPTION
    if key == "objective_description":
        return OBJECTIVE_DESCRIPTION
    return GENERATION_FACTUAL


def _candidate_quality(
    text: str,
    *,
    presence: str,
    numeric_confidence: float | None,
    source_count: int,
) -> str:
    if (
        not text
        or len(text) > 80
        or "\ufffd" in text
        or _ONLY_PUNCTUATION_RE.fullmatch(text)
        or _BROKEN_EDGE_RE.search(text)
        or _REPEATED_NOISE_RE.search(text)
        or _PLACEHOLDER_RE.search(text)
    ):
        return "low"
    if numeric_confidence is not None:
        if numeric_confidence >= 0.88:
            return "high"
        if numeric_confidence < 0.45:
            return "low"
    if source_count >= 2:
        return "high"
    if presence == "present_readable" and (
        _TECHNICAL_TOKEN_RE.fullmatch(text)
        or _CHINESE_READABLE_RE.fullmatch(text)
    ):
        return "high"
    if presence in {"present_unreadable", "none"}:
        return "low"
    if _TECHNICAL_TOKEN_RE.fullmatch(text) or _CHINESE_READABLE_RE.fullmatch(text):
        return "medium"
    return "low"


def collect_text_candidates(
    value: Mapping[str, Any] | None,
    *,
    default_presence: str | None = None,
) -> list[TextCandidate]:
    """Merge candidate sources before applying any bounded output budget."""

    source = dict(value or {})
    presence = normalize_text(source.get("presence") or default_presence or "uncertain")
    collected: dict[str, dict[str, Any]] = {}

    def add(raw: Any, source_name: str) -> None:
        if isinstance(raw, Mapping):
            text = normalize_text(raw.get("text") or raw.get("value"))
            raw_confidence = raw.get("confidence")
        else:
            text = normalize_text(raw)
            raw_confidence = None
        if not text:
            return
        key = text.casefold()
        row = collected.setdefault(
            key,
            {"text": text, "confidences": [], "sources": []},
        )
        if source_name not in row["sources"]:
            row["sources"].append(source_name)
        if isinstance(raw_confidence, (int, float)) and not isinstance(
            raw_confidence, bool
        ):
            row["confidences"].append(float(raw_confidence))

    for key in ("visual_candidates", "ocr_candidates", "candidate_text"):
        raw_values = source.get(key, [])
        if not isinstance(raw_values, list):
            raw_values = [raw_values]
        for raw in raw_values:
            add(raw, key)

    result: list[TextCandidate] = []
    for row in collected.values():
        numeric = max(row["confidences"]) if row["confidences"] else None
        confidence = _candidate_quality(
            row["text"],
            presence=presence,
            numeric_confidence=numeric,
            source_count=len(row["sources"]),
        )
        result.append(
            TextCandidate(
                text=row["text"],
                confidence=confidence,
                numeric_confidence=numeric,
                source_count=len(row["sources"]),
                sources=tuple(row["sources"]),
            )
        )
    return result


def candidate_partition(
    value: Mapping[str, Any] | None,
    *,
    default_presence: str | None = None,
) -> tuple[list[str], list[str], list[TextCandidate]]:
    """Execute the candidate partition operation."""
    records = collect_text_candidates(value, default_presence=default_presence)
    retained = [item.text for item in records if item.confidence == "high"]
    blocked = [item.text for item in records if item.confidence != "high"]
    return retained, blocked, records


def candidate_is_allowed(
    candidate: TextCandidate,
    *,
    task_mode: str,
    qualified: bool = False,
) -> bool:
    """Execute the candidate is allowed operation."""
    if task_mode == CANONICAL:
        return candidate.confidence == "high"
    if task_mode == GENERATION_CREATIVE:
        return candidate.confidence in {"high", "medium"}
    if task_mode == CHAT:
        return candidate.confidence == "high" or (
            candidate.confidence == "medium" and qualified
        )
    return candidate.confidence == "high" or (
        candidate.confidence == "medium" and qualified
    )


def is_qualified_statement(text: str) -> bool:
    """Execute the is qualified statement operation."""
    return bool(_UNCERTAINTY_RE.search(normalize_text(text)))


def is_image_text_fact_assertion(text: str) -> bool:
    """Execute the is image text fact assertion operation."""
    return bool(_FACT_ASSERTION_RE.search(normalize_text(text)))


def _compact_identifier_ranges(values: list[str]) -> list[str]:
    """Compact consecutive technical labels before the schema's item cap."""

    groups: dict[str, list[tuple[int, str, int]]] = {}
    for index, value in enumerate(values):
        match = re.fullmatch(r"([A-Za-z_]+)(\d+)", value)
        if not match:
            continue
        prefix, number = match.group(1), int(match.group(2))
        groups.setdefault(prefix.casefold(), []).append((index, prefix, number))
    replacements: dict[int, str] = {}
    skipped: set[int] = set()
    for entries in groups.values():
        ordered = sorted(entries, key=lambda item: item[2])
        numbers = [number for _, _, number in ordered]
        if len(numbers) < 3 or numbers != list(range(numbers[0], numbers[-1] + 1)):
            continue
        first_index = min(index for index, _, _ in entries)
        prefix = next(prefix for index, prefix, _ in entries if index == first_index)
        replacements[first_index] = f"{prefix}{numbers[0]}–{prefix}{numbers[-1]}"
        skipped.update(index for index, _, _ in entries if index != first_index)
    result: list[str] = []
    for index, value in enumerate(values):
        if index in skipped:
            continue
        result.append(replacements.get(index, value))
    return result


def prioritize_candidate_values(
    values: Iterable[Any],
    *,
    maximum_items: int,
    maximum_length: int,
) -> list[str]:
    """Deduplicate, compact ranges, then spend the existing bounded budget."""

    unique: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = normalize_text(raw)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        unique.append(text[:maximum_length])
    compacted = _compact_identifier_ranges(unique)

    def score(item: tuple[int, str]) -> tuple[int, int]:
        index, text = item
        technical = bool(_TECHNICAL_TOKEN_RE.fullmatch(text))
        information = len(set(text.casefold()))
        return (2 if technical else 1 if information >= 4 else 0, -index)

    ranked = sorted(enumerate(compacted), key=score, reverse=True)
    selected_indices = {index for index, _ in ranked[:maximum_items]}
    return [
        text
        for index, text in enumerate(compacted)
        if index in selected_indices
    ][:maximum_items]


def repair_filtered_text(value: Any, *, sentence: bool = False) -> str:
    """Repair punctuation damage without inventing any removed text."""

    text = normalize_text(value)
    if not text:
        return ""
    text = re.sub(r"\(\s*\)|（\s*）|\[\s*\]|【\s*】", "", text)
    text = re.sub(r"([、，；：])\s*(?:、|，|；|：)+", r"\1", text)
    text = re.sub(r"(描绘|展示|包括|包含|涉及)了\s*[、，]\s*", r"\1了", text)
    text = re.sub(
        r"(描绘|展示|包括|包含|涉及)了\s*等(?=模块|组件|内容)",
        r"\1了多个",
        text,
    )
    text = re.sub(r"(?:标记|标注)为\s*[-–—、，和至到0-9\s]+\)", "", text)
    text = re.sub(r"(?:标记|标注)为\s*[-–—、，和至到0-9\s]+）", "", text)
    text = re.sub(r"(?:标记|标注)为\s*[-–—、，和至到0-9\s]+(?=[，。；])", "", text)
    if text.count("(") > text.count(")"):
        text = text.replace("(", "")
    if text.count(")") > text.count("("):
        text = text.replace(")", "")
    text = re.sub(r"\b([A-Za-z_]+\d+)\s*[-–—]\s*\1\b", r"\1", text)
    text = re.sub(r"\s+([，。；：！？、])", r"\1", text)
    text = re.sub(r"([，。；：！？、])\s+", r"\1", text)
    text = re.sub(r"[、，；：]+([。！？])", r"\1", text)
    text = re.sub(r"([、，；：！？。])\1+", r"\1", text)
    text = text.strip(" 、，；：-–—")
    if re.search(r"[\u3400-\u9fff]", text):
        text = (
            text.replace(",", "，")
            .replace(";", "；")
            .replace(":", "：")
            .replace("?", "？")
            .replace("!", "！")
        )
    if sentence and text and text[-1] not in "。！？；":
        text += "。"
    return text
