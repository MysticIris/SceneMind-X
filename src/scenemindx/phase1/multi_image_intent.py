"""Deterministic content-type resolution for Phase 5.2-D generation."""

from __future__ import annotations

import re
from typing import Any


CONTENT_TYPE_ALIASES = {
    "story": "creative_story",
}

CONTENT_TYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "creative_story",
        (
            r"(?:编|講|讲|写|寫|创作|創作|虚构|虛構|构思|構思).{0,10}(?:故事|情节|情節)",
            r"(?:故事|童话|童話|小说|小說).{0,6}(?:创作|創作|模式)",
        ),
    ),
    (
        "moments",
        (
            r"朋友圈",
            r"(?:社交平台|动态|動態).{0,8}(?:文案|配文)",
        ),
    ),
    (
        "travel_diary",
        (
            r"旅行日记",
            r"旅行日記",
            r"游记",
            r"遊記",
        ),
    ),
    (
        "news_caption",
        (
            r"新闻图注",
            r"新聞圖注",
            r"(?:新闻|新聞).{0,6}(?:配图|配圖|图注|圖注)",
        ),
    ),
    (
        "advertisement",
        (
            r"广告文案",
            r"廣告文案",
            r"(?:营销|營銷|宣传|宣傳).{0,6}文案",
        ),
    ),
    (
        "poster_title",
        (
            r"海报标题",
            r"海報標題",
        ),
    ),
    (
        "poem",
        (
            r"(?:写|寫|创作|創作).{0,6}(?:诗|詩)",
            r"诗歌",
            r"詩歌",
        ),
    ),
    (
        "objective_description",
        (
            r"(?:客观|客觀).{0,6}(?:描述|说明|說明)",
            r"(?:描述|说明|說明).{0,8}(?:图片|圖片|画面|畫面|这几张|這幾張)",
            r"(?:图片|圖片|画面|畫面).{0,8}(?:有什么|有什麼|内容|內容)",
        ),
    ),
    (
        "article",
        (
            r"(?:写|寫|整理).{0,6}(?:文章|短文|介绍|介紹)",
            r"(?:一段|一篇).{0,6}(?:介绍|介紹|文章|短文)",
        ),
    ),
)

NON_GENERATION_PATTERNS = (
    r"(?:比较|比較|对比|對比).{0,12}(?:图片|圖片|这几张|這幾張|哪张|哪張)",
    r"(?:哪张|哪張|哪一张|哪一張).{0,10}(?:更|最|适合|適合|好)",
    r"(?:排序|排名|从好到差|從好到差)",
    r"(?:只选|只選|推荐|推薦).{0,8}(?:一张|一張|哪张|哪張)",
)


def _normalized(value: str | None) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def infer_content_type(natural_language_request: str | None) -> dict[str, Any]:
    """Infer one task type without inspecting images or calling a model."""

    normalized = _normalized(natural_language_request)
    if not normalized:
        return {
            "content_type": "objective_description",
            "matched_rule": "empty_request_default",
            "route_hint": None,
        }
    for pattern in NON_GENERATION_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return {
                "content_type": None,
                "matched_rule": pattern,
                "route_hint": "compare_rank",
            }
    for content_type, patterns in CONTENT_TYPE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return {
                    "content_type": content_type,
                    "matched_rule": pattern,
                    "route_hint": None,
                }
    return {
        "content_type": "objective_description",
        "matched_rule": "no_specific_intent_default",
        "route_hint": None,
    }


def resolve_content_type(
    *,
    requested_content_type: str,
    natural_language_request: str | None,
    content_type_source: str,
    content_type_user_selected: bool,
) -> dict[str, Any]:
    """Resolve explicit selection, untouched defaults, and natural language."""

    requested = CONTENT_TYPE_ALIASES.get(
        str(requested_content_type),
        str(requested_content_type),
    )
    inferred = infer_content_type(natural_language_request)
    explicitly_selected = bool(content_type_user_selected) or (
        content_type_source == "explicit_user_selection"
    )

    if requested == "auto":
        resolved = inferred["content_type"]
        resolution_source = "auto_inferred"
    elif explicitly_selected:
        resolved = requested
        resolution_source = "explicit_user_selection"
    elif natural_language_request and inferred["content_type"] is not None:
        resolved = inferred["content_type"]
        resolution_source = "natural_language_over_default"
    else:
        resolved = requested
        resolution_source = (
            "default_value" if content_type_source == "default_value" else content_type_source
        )

    route_hint = (
        inferred["route_hint"]
        if requested == "auto" or not explicitly_selected
        else None
    )
    conflict = bool(
        explicitly_selected
        and inferred["content_type"] is not None
        and inferred["matched_rule"]
        not in {"empty_request_default", "no_specific_intent_default"}
        and inferred["content_type"] != resolved
    )
    return {
        "requested_content_type": requested,
        "resolved_content_type": resolved,
        "content_type_source": resolution_source,
        "content_type_user_selected": explicitly_selected,
        "natural_language_request": str(natural_language_request or "").strip(),
        "inferred_content_type": inferred["content_type"],
        "matched_rule": inferred["matched_rule"],
        "route_hint": route_hint,
        "explicit_natural_language_conflict": conflict,
        "resolution_warning": (
            "显式内容类型优先于自然语言中的不同类型要求。"
            if conflict
            else None
        ),
    }
