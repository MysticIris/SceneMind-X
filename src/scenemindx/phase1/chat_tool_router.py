"""Bounded Phase 5.4 Chat tool routing.

The router exposes three business tools only.  Deterministic current-turn
rules establish a safe proposal; an optional Qwen router decision may refine
ambiguous parameters, but backend validation and the two-step ceiling remain
authoritative.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .content_length_profiles import normalize_target_length
from .multiturn_chat import local_image_reference_matches


ROUTER_CANDIDATE_ID = "SCENEMINDX_CHAT_TOOL_ROUTER_V1_2_CANDIDATE"
ROUTER_PROMPT_ID = "phase5_4d_chat_tool_router_v1_2"
DIRECT_CHAT_CANDIDATE_ID = "SCENEMINDX_DIRECT_CHAT_V1_CANDIDATE"
DIRECT_CHAT_PROMPT_ID = "phase5_4d_direct_chat_v1"
ALLOWED_TOOLS = {
    "generate_content_from_images",
    "search_images",
    "compare_or_rank_images",
}
ALLOWED_ACTIONS = {"direct_answer", "tool_call", "clarification"}
SEARCH_MODES = {"auto", "text", "image", "hybrid"}
COMPARE_ACTIONS = {"compare", "select", "rank", "auto"}
CONTENT_TYPES = {
    "auto",
    "objective_description",
    "moments",
    "creative_story",
    "article",
    "travel_diary",
    "news_caption",
    "advertisement",
    "poster_title",
    "poem",
}
COLLECTION_REFERENCE_TOKENS = (
    "这几张图",
    "这几张图片",
    "这几张",
    "这些图",
    "这些图片",
    "这些",
    "这几个",
    "全部图片",
    "全部",
    "全都",
    "所有图片",
    "前面这些图",
    "前面这些图片",
    "前面这些",
    "上面这些图",
    "上面这些图片",
    "上面这些",
    "当前这些图",
    "当前这些图片",
    "当前这些",
)
ORDINALS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChatToolRouterCandidate:
    """Load and hash-check the independent Router Candidate."""

    def __init__(self, project_root: Path) -> None:
        self.root = (
            project_root
            / "prompts"
            / "phase5_4d"
            / "chat_tool_router_v1_2_candidate"
        )
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("candidate_id") != ROUTER_CANDIDATE_ID:
            raise ValueError("phase5_4_router_candidate_identity_mismatch")
        spec = dict(self.manifest["prompt"])
        self.prompt_path = project_root / str(spec["file"])
        if _sha256(self.prompt_path) != str(spec["raw_sha256"]):
            raise ValueError("phase5_4_router_candidate_sha256_mismatch")
        self.text = self.prompt_path.read_text(encoding="utf-8")
        direct_spec = dict(self.manifest["direct_chat_prompt"])
        self.direct_chat_path = project_root / str(direct_spec["file"])
        if _sha256(self.direct_chat_path) != str(direct_spec["raw_sha256"]):
            raise ValueError("phase5_4d_direct_chat_candidate_sha256_mismatch")
        self.direct_chat_text = self.direct_chat_path.read_text(encoding="utf-8")

    def identity(self, name: str = "router") -> dict[str, Any]:
        """Execute the identity operation."""
        if name == "direct_chat":
            spec = dict(self.manifest["direct_chat_prompt"])
            return {
                "candidate_id": DIRECT_CHAT_CANDIDATE_ID,
                "prompt_id": DIRECT_CHAT_PROMPT_ID,
                "prompt_sha256": str(spec["raw_sha256"]),
                "status": str(self.manifest["status"]),
                "iteration": 1,
                "tool_count": 0,
                "max_business_steps": 0,
            }
        return {
            "candidate_id": ROUTER_CANDIDATE_ID,
            "prompt_id": ROUTER_PROMPT_ID,
            "prompt_sha256": str(self.manifest["prompt"]["raw_sha256"]),
            "status": str(self.manifest["status"]),
            "iteration": int(self.manifest["iteration"]),
            "tool_count": 3,
            "max_business_steps": 2,
        }

    def render(self, values: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Render the requested value."""
        prompt = self.text
        for key, value in values.items():
            rendered = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            )
            prompt = prompt.replace("{{" + key + "}}", rendered)
        unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", prompt)))
        if unresolved:
            raise ValueError(
                "phase5_4_router_unresolved_placeholders:"
                + ",".join(unresolved)
            )
        return prompt, self.identity()

    def render_direct_chat(
        self,
        current_user_turn: str,
    ) -> tuple[str, dict[str, Any]]:
        """Render direct chat."""
        prompt = self.direct_chat_text.replace(
            "{{CURRENT_USER_TURN}}",
            current_user_turn,
        )
        unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", prompt)))
        if unresolved:
            raise ValueError(
                "phase5_4d_direct_chat_unresolved_placeholders:"
                + ",".join(unresolved)
            )
        return prompt, self.identity("direct_chat")


def _normalized(message: str) -> str:
    return re.sub(r"\s+", "", message).strip()


def _exact_identity_query(message: str) -> bool:
    """Match the single user-visible direct semantic answer contract."""

    normalized = message.strip()
    if normalized.endswith(("？", "?", "。")):
        normalized = normalized[:-1].rstrip()
    return normalized == "你是什么模型"


def _search_import_semantic_remainder(message: str) -> str | None:
    """Extract the optional second step of the bounded import-then-chat form."""

    matched = re.search(
        r"(?:，|,|。|；|;)?\s*(?:然后|再|并且)\s*(?P<remainder>.+?)\s*$",
        message.strip(),
    )
    if matched is None:
        return None
    remainder = matched.group("remainder").strip()
    return remainder or None


def detect_system_utility(
    message: str,
    *,
    search_labels: list[str] | None = None,
) -> dict[str, Any] | None:
    """Resolve bounded non-business utilities before visual-tool routing."""

    if _exact_identity_query(message):
        return {"name": "current_model_identity", "arguments": {}}

    normalized = _normalized(message)

    if not re.search(r"(?:加入|添加|放进|放入)(?:当前)?(?:会话)?上下文", normalized):
        return None
    available = list(search_labels or [])
    labels: list[str] = []
    for value in re.findall(
        r"(?<![A-Z0-9_])SEARCH_([1-5])(?!\d)",
        normalized,
        flags=re.IGNORECASE,
    ):
        label = f"SEARCH_{int(value)}"
        if label in available and label not in labels:
            labels.append(label)
    for value in re.findall(r"第?([一二两三四五1-5])张", normalized):
        index = ORDINALS.get(value, int(value) if value.isdigit() else 0)
        label = f"SEARCH_{index}"
        if label in available and label not in labels:
            labels.append(label)
    if labels:
        arguments: dict[str, Any] = {"search_labels": labels}
        semantic_remainder = _search_import_semantic_remainder(message)
        if semantic_remainder is not None:
            arguments["semantic_remainder"] = semantic_remainder
        return {
            "name": "add_search_results_to_context",
            "arguments": arguments,
        }
    return None


def visual_evidence_required(message: str) -> bool:
    """Execute the visual evidence required operation."""
    normalized = _normalized(message)
    return bool(
        re.search(
            r"(?:第\s*[一二两三四五1-5]\s*(?:张|幅)(?:图|图片)?"
            r"|(?:图|图片)\s*[一二两三四五1-5]"
            r"|[一二两三四五]\s*号(?:图|图片)?"
            r"|(?<![A-Z])image\s*[1-5](?!\d)"
            r"|IMG_[1-9]\d*)",
            normalized,
            flags=re.IGNORECASE,
        )
    ) or any(
        token in normalized
        for token in (
            "图中",
            "图片里",
            "画面中",
            "这张图",
            "那张图",
            "那幅图",
            "两张图",
            "三张图",
            "这些图",
            "这几张",
            "这些图片",
            "全部图片",
            "全部图",
            "所有图片",
            "所有图",
            "当前所有图片",
            "当前所有图",
            "前面这些",
            "上面这些",
            "当前这些",
            "看图",
            "哪张图",
            "照片里",
            "前一张",
            "上一张",
            "它和",
            "它的",
            "它怎么样",
            "概括一下它",
            "分别说明",
            "分别描述",
        )
    )


def visual_groundable_intent(
    message: str,
    *,
    active_image_count: int = 0,
    has_visual_history: bool = False,
) -> bool:
    """Recognize natural visual turns without forcing every turn onto images.

    Explicit image references always win. Implicit defaults are intentionally
    limited to deictic/visual language or compact property questions when one
    authoritative active image exists. Named external-world questions remain
    ordinary text turns.
    """

    if visual_evidence_required(message):
        return True
    normalized = _normalized(message)
    if not normalized:
        return False
    if any(
        token in normalized
        for token in (
            "这里",
            "上面",
            "里面",
            "画面",
            "照片",
            "看着",
            "这个场景",
            "这个设计",
            "可见内容",
            "可读文字",
        )
    ):
        return True
    if active_image_count == 1 and re.fullmatch(
        r"(?:今天天气|天气|气温|光线|氛围|构图|画质|颜色|设计)(?:感觉)?"
        r"(?:怎么样|如何|是什么|有什么特点)[？?。]?$",
        normalized,
    ):
        return True
    if active_image_count == 1 and re.fullmatch(
        r"(?:这是什么|有什么值得注意的?|为什么会这样|再具体一点|"
        r"再详细一点|详细说一下|具体说说)[？?。]?$",
        normalized,
    ):
        return True
    if has_visual_history and re.fullmatch(
        r"(?:为什么|为什么会这样|再具体一点|再详细一点|详细说一下|具体说说)"
        r"[？?。]?$",
        normalized,
    ):
        return True
    return False


def _content_type(message: str) -> str:
    rules = (
        ("creative_story", ("故事", "童话", "小说")),
        ("moments", ("朋友圈", "动态文案")),
        ("travel_diary", ("旅行日记", "旅行日記", "游记", "遊記")),
        ("news_caption", ("新闻配文", "新闻图注", "新聞配文", "新聞圖注")),
        ("advertisement", ("广告文案", "廣告文案", "营销文案", "宣传文案")),
        ("poster_title", ("海报标题", "海報標題")),
        ("poem", ("诗歌", "写诗", "寫詩")),
        ("article", ("文章", "短文")),
        ("objective_description", ("客观描述", "客觀描述", "说明画面")),
    )
    for content_type, tokens in rules:
        if any(token in message for token in tokens):
            return content_type
    return "auto"


def _target_length(message: str, content_type: str) -> int:
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(?:字|字符)", message)
    return int(
        normalize_target_length(
            match.group(1) if match else None,
            content_type,
            explicit=bool(match),
        )["target"]
    )


def _target_length_source(message: str) -> str:
    return (
        "user_explicit"
        if re.search(r"[+-]?\d+(?:\.\d+)?\s*(?:字|字符)", message)
        else "profile_default"
    )


def _explicit_image_refs(
    message: str,
    *,
    active_labels: list[str],
    search_labels: list[str],
) -> list[str]:
    refs: list[str] = []

    def add(label: str) -> None:
        if label not in refs:
            refs.append(label)

    ordinals = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
    }
    search_context_spans: list[tuple[int, int]] = []
    for match in re.finditer(
        r"(?:刚才|上次)?(?:搜到|搜索到|找到|检索到|搜索结果|检索结果)"
        r"(?:里|中|中的|的)?(?:第\s*)?"
        r"([一二两三四五]|[1-5])\s*(?:张|个)?",
        message,
    ):
        search_context_spans.append((match.start(), match.end()))
        raw = match.group(1)
        index = ordinals.get(raw, int(raw) if raw.isdigit() else 0)
        label = f"SEARCH_{index}"
        if label in search_labels:
            add(label)
    for match in re.finditer(
        r"(?<![A-Z0-9_])SEARCH_([1-9]\d*)(?!\d)",
        message,
        flags=re.IGNORECASE,
    ):
        label = f"SEARCH_{int(match.group(1))}"
        if label in search_labels:
            add(label)
    for match in local_image_reference_matches(message):
        if any(
            match["start"] < end and match["end"] > start
            for start, end in search_context_spans
        ):
            continue
        add(str(match["label"]))
    compact_enumerated_set = bool(
        re.search(r"(?:图[一二两三四五]){2,}", message)
    )
    search_top = re.search(
        r"(?:刚才|上次)?(?:搜到|搜索到|找到|检索到|搜索结果|检索结果)"
        r"(?:里|中|中的|的)?前([一二两三四五1-5])张",
        message,
    )
    if search_top:
        raw = search_top.group(1)
        count = ordinals.get(raw, int(raw) if raw.isdigit() else 0)
        for label in search_labels[:count]:
            add(label)
    if not refs and (
        any(token in message for token in COLLECTION_REFERENCE_TOKENS)
        or compact_enumerated_set
    ):
        refs.extend(active_labels)
    return refs


def _references_previous_search(message: str, search_labels: list[str]) -> bool:
    if not search_labels:
        return False
    return any(
        token in message
        for token in (
            "刚才搜索",
            "刚才搜到",
            "刚才找到",
            "刚才检索",
            "搜索结果",
            "检索结果",
            "搜到的",
            "找到的",
        )
    )


def _generation_marked(message: str) -> bool:
    return any(
        token in message
        for token in (
            "写个",
            "写一",
            "生成",
            "编一个",
            "编个",
            "朋友圈",
            "旅行日记",
            "游记",
            "图注",
            "配文",
            "广告文案",
            "海报标题",
            "诗歌",
            "故事",
            "文章",
        )
    )


def _search_marked(message: str) -> bool:
    return bool(
        re.search(r"(?:找|搜).{0,12}(?:图|图片)|(?:图|图片).{0,12}(?:找|搜)", message)
    ) or any(
        token in message
        for token in (
            "检索",
            "搜索",
            "搜图",
            "找图",
            "找几张",
            "只根据文字找",
            "只按这句话找",
            "只用这张图找",
            "用这张图和这句话找",
            "图库里",
            "相似图片",
            "类似图片",
            "相似图",
        )
    )


def _compare_marked(message: str) -> bool:
    collection_context = any(
        token in message
        for token in (
            *COLLECTION_REFERENCE_TOKENS,
            "这三张",
            "这两张",
            "这几张里",
            "这些里面",
        )
    )
    return bool(
        re.search(
            r"哪(?:[一二两三四五1-5])?(?:张|个|些).{0,24}"
            r"(?:适合|合适|更好|最好|更像|像|符合|推荐|选)"
            r"|(?:第一张|第二张|IMG_[1-9]\d*).{0,16}(?:还是|对比|比较)"
            r"|(?:帮我|给我)?(?:挑|选)(?:出|一下|一个|一张|[一二两三四五1-5]张)"
            r"|(?:按|按照).{1,24}(?:排一下|排序|排名|从高到低排)"
            r"|(?:用|选)哪(?:一)?张.{0,16}(?:更好|合适|适合)",
            message,
            flags=re.IGNORECASE,
        )
    ) or any(
        token in message
        for token in (
            "比较",
            "对比",
            "哪张更",
            "哪张最",
            "哪张好",
            "哪两张",
            "哪些更",
            "哪个更",
            "只选",
            "选一张",
            "选两张",
            "挑一张",
            "挑两张",
            "帮我挑",
            "从这几张里选",
            "排序",
            "排名",
            "排一下",
            "从高到低排",
            "作为封面",
            "当封面",
            "做封面",
            "作为配图",
            "当配图",
            "朋友圈首图",
            "旅游照片",
            "更适合",
            "更符合",
        )
    ) or (
        collection_context
        and any(
            token in message
            for token in (
                "推荐",
                "选出",
                "挑",
                "首图",
                "封面",
                "配图",
                "哪个好",
                "哪个较好",
                "哪张好",
                "哪张较好",
                "哪一个好",
                "哪一个较好",
            )
        )
    ) or bool(
        re.search(
            r"(?:它|当前这张|刚加入的这张).{0,6}(?:和|与).{0,6}"
            r"(?:前一张|上一张).{0,8}(?:区别|差异)",
            message,
        )
    )


def _search_query(message: str) -> str:
    query = re.split(
        r"(?:，|,|；|;)?(?:再|然后|接着).{0,8}(?:用|拿).{0,12}(?:写|生成|编)",
        message,
        maxsplit=1,
    )[0]
    query = re.sub(
        r"^(?:请|麻烦)?(?:帮我)?(?:在图库里|从图库里)?"
        r"(?:检索|搜索|搜|找)(?:几张|一些|一下)?",
        "",
        query,
    )
    query = re.sub(r"(?:图片|图)$", "", query)
    return query.strip(" ，,。？?!！") or message.strip()


def _search_mode(message: str, image_refs: list[str]) -> str:
    if any(token in message for token in ("只根据文字", "只按这句话", "只用文字")):
        return "text"
    if any(token in message for token in ("只用这张图", "只根据这张图", "只按图片")):
        return "image"
    if re.search(r"用第?[一二两三四五1-5]张图(?:片)?找(?:相似|类似)", message):
        return "image"
    if image_refs and re.search(
        r"(?:用|拿).{0,16}(?:找|搜)(?:相似|类似)(?:图|图片)?",
        message,
    ):
        return "image"
    if any(token in message for token in ("图和这句话", "图片和文字", "图文")):
        return "hybrid"
    return "hybrid" if image_refs else "text"


def _compare_action(message: str) -> str:
    if any(
        token in message
        for token in (
            "排序",
            "排名",
            "从好到差",
            "从高到低",
            "完整顺序",
            "完整排序",
            "排一下",
        )
    ):
        return "rank"
    if any(
        token in message
        for token in (
            "只选",
            "选一张",
            "选一个",
            "选出",
            "推荐一张",
            "哪张最",
            "哪一张最",
            "哪张更",
            "哪一张更",
            "哪两张",
            "哪些更",
            "哪个更",
            "哪个好",
            "哪个较好",
            "哪张好",
            "哪张较好",
            "哪一个好",
            "哪一个较好",
            "帮我挑",
            "挑一张",
            "挑两张",
            "推荐",
            "用哪张",
            "更适合当",
            "更适合做",
            "作为封面",
            "当封面",
            "作为配图",
            "朋友圈首图",
        )
    ) or re.search(
        r"(?:选|挑|推荐|取)(?:出|给我|一下)?[一二两三四五1-5]张",
        message,
    ):
        return "select"
    if any(token in message for token in ("比较", "对比")):
        return "compare"
    return "auto"


def _selection_count(message: str) -> int:
    top_k = re.search(
        r"\btop\s*([1-5])\b",
        message,
        flags=re.IGNORECASE,
    )
    if top_k:
        return int(top_k.group(1))
    patterns = (
        r"(?:选|挑|推荐|取|要)(?:出|给我|一下)?([一二两三四五1-5])张",
        r"哪([一二两三四五1-5])张",
        r"前([一二两三四五1-5])张",
    )
    for pattern in patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        raw = match.group(1)
        return ORDINALS.get(raw, int(raw) if raw.isdigit() else 1)
    return 1


def _comparison_criterion(message: str) -> str:
    """Keep a supplied daily-use criterion; leave generic selection empty."""

    compact = message.strip(" ，,。？?!！")
    criterion_terms = (
        "旅游",
        "旅行",
        "朋友圈",
        "首图",
        "封面",
        "配图",
        "节日",
        "节庆",
        "庆祝",
        "氛围",
        "表情包",
        "宣传",
        "广告",
        "构图",
        "清晰",
        "色彩",
        "主题",
        "上课",
        "课堂",
        "课程",
        "教学",
        "学习",
        "出现的概率",
        "出现概率",
        "概率",
        "传播",
        "吸引",
        "点击",
        "专业",
        "正式",
        "更像",
        "更符合",
    )
    if any(token in compact for token in criterion_terms):
        return compact
    explicit = re.search(
        r"(?:按|按照|依据|以)([^，。；？?]{1,36})(?:排|排序|排名|选|挑|比较)",
        compact,
    )
    return explicit.group(1).strip() if explicit else ""


def deterministic_tool_plan(
    message: str,
    *,
    active_labels: list[str],
    search_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Create a current-turn-first, schema-shaped proposal."""

    normalized = _normalized(message)
    search_labels = list(search_labels or [])
    explicit_refs = _explicit_image_refs(
        normalized,
        active_labels=active_labels,
        search_labels=search_labels,
    )
    has_search = _search_marked(normalized)
    has_generate = _generation_marked(normalized)
    has_compare = _compare_marked(normalized)
    previous_search = _references_previous_search(
        normalized,
        search_labels,
    )
    default_refs = (
        explicit_refs
        or (list(search_labels) if previous_search else list(active_labels))
    )
    # An existing search-result mention is a reference, not a request to
    # search again. The explicit current-turn compare/select/rank verb wins.
    if previous_search and explicit_refs and (
        has_compare or not has_generate
    ):
        has_search = False
    steps: list[dict[str, Any]] = []

    if has_compare:
        action = _compare_action(normalized)
        select_count = _selection_count(normalized)
        steps.append(
            {
                "tool_name": "compare_or_rank_images",
                "arguments": {
                    "image_refs": default_refs,
                    "action": action,
                    "criterion": _comparison_criterion(message),
                    "scenario": message.strip(),
                    "select_count": select_count,
                    "natural_language_request": message.strip(),
                    "original_user_request": message.strip(),
                },
            }
        )
    elif has_search:
        mode = _search_mode(normalized, default_refs)
        search_refs = [] if mode == "text" else default_refs
        if re.search(r"验证(?:图片)?库|system[_\s-]*val|\bval\b", normalized):
            library_scope = "system_val"
        elif re.search(r"训练(?:图片)?库|system[_\s-]*train|\btrain\b", normalized):
            library_scope = "system_train"
        else:
            library_scope = "all_libraries"
        steps.append(
            {
                "tool_name": "search_images",
                "arguments": {
                    "mode": mode,
                    "text_query": (
                        None if mode == "image" else _search_query(message)
                    ),
                    "image_refs": search_refs,
                    "top_k": 5,
                    "exclude_query_images": True,
                    "library_scope": library_scope,
                },
            }
        )
        if has_generate and re.search(r"再|然后|接着|随后", normalized):
            content_type = _content_type(normalized)
            count_match = re.search(r"前([一二两三四五1-5])张", normalized)
            count = 3
            if count_match:
                raw = count_match.group(1)
                count = {
                    "一": 1,
                    "二": 2,
                    "两": 2,
                    "三": 3,
                    "四": 4,
                    "五": 5,
                }.get(raw, int(raw) if raw.isdigit() else 3)
            steps.append(
                {
                    "tool_name": "generate_content_from_images",
                    "arguments": {
                        "image_refs": [
                            f"SEARCH_{index}" for index in range(1, count + 1)
                        ],
                        "content_type": content_type,
                        "natural_language_request": message.strip(),
                        "style": "自然",
                        "audience": "普通读者",
                        "target_length": _target_length(
                            normalized, content_type
                        ),
                        "target_length_source": _target_length_source(
                            normalized
                        ),
                        "organization": "input_order",
                        "importance": [],
                        "use_all_context_images": False,
                    },
                }
            )
    elif has_generate:
        content_type = _content_type(normalized)
        steps.append(
            {
                "tool_name": "generate_content_from_images",
                "arguments": {
                    "image_refs": default_refs,
                    "content_type": content_type,
                    "natural_language_request": message.strip(),
                    "style": "自然",
                    "audience": "普通读者",
                    "target_length": _target_length(normalized, content_type),
                    "target_length_source": _target_length_source(normalized),
                    "organization": "input_order",
                    "importance": [],
                    "use_all_context_images": not bool(explicit_refs),
                },
            }
        )

    if not steps:
        return {
            "action": "direct_answer",
            "tool_name": None,
            "arguments": {"image_refs": explicit_refs},
            "steps": [],
            "reason_code": "no_business_tool_required",
            "confidence": "high",
            "clarification": None,
            "source": "deterministic_current_turn",
        }
    first = steps[0]
    return {
        "action": "tool_call",
        "tool_name": first["tool_name"],
        "arguments": first["arguments"],
        "steps": steps[:2],
        "reason_code": (
            "explicit_two_step_search_then_generate"
            if len(steps) == 2
            else "explicit_current_turn_tool_intent"
        ),
        "confidence": "high",
        "clarification": None,
        "source": "rule_based_proposal",
    }


def _extract_json_object(raw_output: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_output):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw_output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def validate_router_decision(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate router decision."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return None, ["router_root_must_be_object"]
    action = str(value.get("action") or "")
    if action not in ALLOWED_ACTIONS:
        errors.append("invalid_router_action")
    tool_name = value.get("tool_name")
    if action == "tool_call" and tool_name not in ALLOWED_TOOLS:
        errors.append("invalid_router_tool_name")
    if action != "tool_call" and tool_name not in (None, ""):
        errors.append("non_tool_action_must_not_name_tool")
    arguments = value.get("arguments")
    if not isinstance(arguments, dict):
        errors.append("router_arguments_must_be_object")
        arguments = {}
    if tool_name == "search_images":
        if arguments.get("mode") not in SEARCH_MODES:
            errors.append("invalid_search_mode")
        top_k = arguments.get("top_k", 5)
        if not isinstance(top_k, int) or not 1 <= top_k <= 5:
            errors.append("invalid_search_top_k")
    elif tool_name == "generate_content_from_images":
        if arguments.get("content_type", "auto") not in CONTENT_TYPES:
            errors.append("invalid_generation_content_type")
        target = arguments.get("target_length", 180)
        if not isinstance(target, int) or not 4 <= target <= 2000:
            errors.append("invalid_generation_target_length")
    elif tool_name == "compare_or_rank_images":
        if arguments.get("action", "auto") not in COMPARE_ACTIONS:
            errors.append("invalid_compare_action")
        select_count = arguments.get("select_count", 1)
        if not isinstance(select_count, int) or not 1 <= select_count <= 5:
            errors.append("invalid_compare_select_count")
    image_refs = arguments.get("image_refs", [])
    if not isinstance(image_refs, list) or any(
        not isinstance(item, str)
        or re.fullmatch(r"(?:IMG|SEARCH)_[1-9]\d*", item) is None
        for item in image_refs
    ):
        errors.append("invalid_router_image_refs")
    if errors:
        return None, errors
    normalized = {
        "action": action,
        "tool_name": tool_name or None,
        "arguments": arguments,
        "reason_code": str(value.get("reason_code") or "model_router"),
        "confidence": str(value.get("confidence") or "medium"),
        "clarification": (
            str(value["clarification"]).strip()
            if value.get("clarification")
            else None
        ),
        "steps": (
            [
                {"tool_name": tool_name, "arguments": arguments}
            ]
            if action == "tool_call"
            else []
        ),
        "source": "qwen_router",
    }
    return normalized, []


def validate_router_decision_against_state(
    decision: dict[str, Any],
    *,
    active_labels: list[str],
    search_labels: list[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Apply state-aware schema checks before any business tool executes."""

    validated, errors = validate_router_decision(decision)
    if validated is None:
        return None, errors
    if validated["action"] != "tool_call":
        return validated, []
    arguments = dict(validated["arguments"])
    refs = [str(item) for item in arguments.get("image_refs", [])]
    allowed = set(active_labels) | set(search_labels)
    unknown = [label for label in refs if label not in allowed]
    if unknown:
        errors.append("unknown_router_image_refs:" + ",".join(unknown))
    tool_name = validated["tool_name"]
    if tool_name == "generate_content_from_images" and not refs:
        errors.append("generation_requires_resolved_images")
    if tool_name == "compare_or_rank_images" and len(refs) < 2:
        errors.append("compare_requires_two_resolved_images")
    if (
        tool_name == "compare_or_rank_images"
        and arguments.get("action") == "select"
        and int(arguments.get("select_count", 1)) > len(refs)
    ):
        errors.append("selection_count_exceeds_resolved_images")
    if errors:
        return None, errors
    validated["arguments"] = arguments
    validated["steps"] = [
        {"tool_name": tool_name, "arguments": arguments}
    ]
    return validated, []


def parse_router_output(raw_output: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse router output."""
    payload = _extract_json_object(raw_output)
    return validate_router_decision(payload)


def parse_direct_chat_output(raw_output: str) -> tuple[str | None, bool]:
    """Extract one safe public answer and never expose a raw JSON envelope."""

    def public_answer(value: str) -> str | None:
        answer = " ".join(value.split())
        if not answer:
            return None
        internal_markers = (
            "raw_output",
            "validator",
            "schema_version",
            "prompt_id",
            "trace_id",
            "tool_name",
            "内部合同",
            "验证器",
            "原始输出",
        )
        if any(marker.lower() in answer.lower() for marker in internal_markers):
            return None
        return answer

    payload = _extract_json_object(raw_output)
    if isinstance(payload, dict):
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            cleaned = public_answer(answer)
            if cleaned:
                return cleaned, bool(payload.get("needs_clarification", False))
        return None, False
    text = raw_output.strip()
    if not text or text.startswith(("{", "[", "```")):
        return None, False
    return public_answer(text), False


def merge_router_decisions(
    deterministic: dict[str, Any],
    model: dict[str, Any] | None,
) -> dict[str, Any]:
    """Current-turn explicit rules win; Qwen refines only non-explicit turns."""

    if deterministic.get("action") == "tool_call":
        merged = dict(deterministic)
        agreed = bool(
            model
            and model.get("action") == "tool_call"
            and model.get("tool_name") == deterministic.get("tool_name")
        )
        if agreed:
            deterministic_arguments = dict(
                deterministic.get("arguments") or {}
            )
            model_arguments = dict(model.get("arguments") or {})
            if deterministic.get("tool_name") == "compare_or_rank_images":
                # The rule layer owns action, references and K.  The model may
                # recover a semantic criterion which the lightweight rule
                # proposal could not isolate, but it must never rewrite the
                # user's original instruction.
                model_criterion = str(
                    model_arguments.get("criterion") or ""
                ).strip()
                if (
                    not str(
                        deterministic_arguments.get("criterion") or ""
                    ).strip()
                    and model_criterion
                ):
                    deterministic_arguments["criterion"] = model_criterion
                original = str(
                    deterministic_arguments.get("original_user_request")
                    or deterministic_arguments.get(
                        "natural_language_request"
                    )
                    or ""
                ).strip()
                deterministic_arguments["natural_language_request"] = (
                    original
                )
                deterministic_arguments["original_user_request"] = original
            merged["arguments"] = deterministic_arguments
            merged["steps"] = [
                {
                    "tool_name": deterministic.get("tool_name"),
                    "arguments": deterministic_arguments,
                }
            ]
        return {
            **merged,
            "model_agreed": agreed,
            "model_proposal": model,
        }
    if model is not None:
        merged = {
            **model,
            "steps": list(model.get("steps", []))[:2],
            "model_agreed": True,
            "deterministic_proposal": deterministic,
        }
        deterministic_refs = list(
            (deterministic.get("arguments") or {}).get(
                "image_refs",
                [],
            )
        )
        if (
            model.get("action") == "direct_answer"
            and deterministic_refs
        ):
            merged["arguments"] = {
                **dict(model.get("arguments") or {}),
                "image_refs": deterministic_refs,
            }
        return merged
    return {**deterministic, "model_agreed": None, "model_proposal": None}
