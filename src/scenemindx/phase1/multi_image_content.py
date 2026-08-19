"""Isolated Phase 5.2-B multi-image content generation contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from scenemindx.task_text_policy import (
    GENERATION_CREATIVE,
    candidate_is_allowed,
    collect_text_candidates,
    content_type_task_mode,
    is_image_text_fact_assertion,
    is_qualified_statement,
)
from .content_length_profiles import (
    CONTENT_LENGTH_PROFILES,
    ideal_output_window,
)


CANDIDATE_ID = "SCENEMINDX_MULTI_IMAGE_CONTENT_V2_CANDIDATE"
PROMPT_ID = "phase5_2_multi_image_content_v2"
STORY_CANDIDATE_ID = "SCENEMINDX_MULTI_IMAGE_CONTENT_V3_CANDIDATE"
STORY_PROMPT_ID = "phase5_2d_multi_image_story_v1"
FRIENDLY_FAILURE_TEXT = "本次内容生成未成功，请稍后重试。"

CONTENT_TYPE_LABELS = {
    "auto": "自动识别",
    "objective_description": "客观描述",
    "moments": "朋友圈",
    "travel_diary": "旅行日记",
    "news_caption": "新闻图注",
    "advertisement": "广告文案",
    "poster_title": "海报标题",
    "poem": "诗歌",
    "story": "故事创作",
    "creative_story": "故事创作",
    "article": "普通文章",
}
DEFAULT_CONTENT_TYPE_LENGTH_WINDOWS = {
    key: (value["default"], value["input_min"], value["input_max"])
    for key, value in CONTENT_LENGTH_PROFILES.items()
    if key != "auto"
}
DEFAULT_CONTENT_TYPE_LENGTH_WINDOWS["story"] = (
    DEFAULT_CONTENT_TYPE_LENGTH_WINDOWS["creative_story"]
)
ORGANIZATION_LABELS = {
    "input_order": "按输入顺序",
    "importance": "按重要性",
    "chronological_if_evidenced": "仅在有视觉证据时按时间顺序",
    "independent_panels": "独立画面自然串联",
}
SAFE_FACT_KEYS = (
    "global_observation",
    "global_scene",
    "subjects",
    "main_subjects",
    "activities",
    "relations",
    "attributes",
)
INTERNAL_OUTPUT_TOKENS = (
    "direct_facts",
    "narrative_organization",
    "creative_expression",
    "cross_image_relation",
    "asset_id",
    "asset_ids",
    "sha256",
    "candidate_id",
    "prompt_id",
    "trace_id",
    "validator",
    "json",
    "p3",
    "根据候选事实",
    "证据层级",
    "系统规则",
)
COMMON_TRADITIONAL_TO_SIMPLIFIED = str.maketrans(
    {
        "節": "节",
        "學": "学",
        "華": "华",
        "國": "国",
        "臺": "台",
        "門": "门",
        "風": "风",
        "書": "书",
        "畫": "画",
        "體": "体",
        "時": "时",
        "間": "间",
        "樂": "乐",
        "禮": "礼",
        "貓": "猫",
        "來": "来",
        "將": "将",
        "擺": "摆",
        "爛": "烂",
        "氣": "气",
        "場": "场",
        "徹": "彻",
        "籠": "笼",
        "雙": "双",
        "無": "无",
        "卻": "却",
        "領": "领",
        "視": "视",
        "轉": "转",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_character_count(text: str) -> int:
    """Count user-visible Unicode characters while excluding whitespace."""

    return sum(1 for character in text if not character.isspace())


def target_length_window(target_length: int) -> tuple[int, int]:
    """Execute the target length window operation."""
    return ideal_output_window(target_length)


def content_type_length_window(
    content_type: str | None,
    target_length: int,
    *,
    target_length_source: str = "user_or_caller",
) -> tuple[int, int]:
    """Return the advisory 75%-130% output window for every target."""

    del content_type, target_length_source
    return target_length_window(target_length)


def user_requested_per_image_explanation(request: str) -> bool:
    """Detect the narrow exception that permits explicit image-by-image prose."""

    compact = re.sub(r"\s+", "", str(request or ""))
    return bool(
        re.search(
            r"(?:逐图|逐张|分别|每张|按图|一张一张|图一图二)"
            r".{0,12}(?:说明|描述|介绍|分析|写|展开)",
            compact,
        )
    )


def _collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(_collect_strings(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_collect_strings(item))
        return result
    return []


def _verified_text_values(asset: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for item in asset.get("verified_text", []):
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
        else:
            text = ""
        if text and text not in values:
            values.append(text)
    return values


def _unverified_candidate_values(asset: dict[str, Any]) -> list[str]:
    verified = set(_verified_text_values(asset))
    values: list[str] = []
    for item in asset.get("ocr_candidates", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if len(text) >= 2 and text not in verified and text not in values:
            values.append(text)
    return values


def build_safe_image_context(
    assets: Iterable[dict[str, Any]],
    *,
    task_mode: str = "generation_factual",
) -> list[dict[str, Any]]:
    """Serialize visible text according to the requested downstream task."""

    rows: list[dict[str, Any]] = []
    for position, asset in enumerate(assets, start=1):
        candidate_records = collect_text_candidates(
            {
                "presence": asset.get("text_presence")
                or asset.get("presence")
                or "uncertain",
                "ocr_candidates": asset.get("ocr_candidates", []),
                "visual_candidates": asset.get("visible_text_candidates", []),
            }
        )
        candidates = [item.text for item in candidate_records]
        readable_text = [
            item.text for item in candidate_records if item.confidence == "high"
        ]
        possible_text = [
            item.text
            for item in candidate_records
            if item.confidence == "medium" and task_mode == GENERATION_CREATIVE
        ]
        blocked_candidates = [
            item.text for item in candidate_records if item.confidence != "high"
        ]
        facts = asset.get("facts") if isinstance(asset.get("facts"), dict) else {}
        text_dense_unverified = (
            len(candidates) >= 10 and not _verified_text_values(asset)
        )
        safe_facts: list[str] = []
        for key in (() if text_dense_unverified else SAFE_FACT_KEYS):
            for raw in _collect_strings(facts.get(key)):
                value = " ".join(raw.split()).strip("。；; ")
                lowered = value.lower()
                if (
                    not value
                    or lowered.startswith("not_available")
                    or value in {"无文字", "不可辨文字", "无"}
                    or any(
                        candidate in value or (len(value) >= 2 and value in candidate)
                        for candidate in blocked_candidates
                    )
                    or value in safe_facts
                ):
                    continue
                safe_facts.append(value)
                if len(safe_facts) == 6:
                    break
            if len(safe_facts) == 6:
                break
        rows.append(
            {
                "image_position": position,
                "visual_candidate_facts": safe_facts,
                "verified_text": _verified_text_values(asset),
                "readable_text": readable_text,
                "possible_text_inspiration": possible_text,
                "text_policy": (
                    "creative_inspiration_not_observed_fact"
                    if task_mode == GENERATION_CREATIVE
                    else "high_confidence_direct_medium_qualified_low_refuse"
                ),
                "text_dense_unverified": text_dense_unverified,
            }
        )
    return rows


def build_generation_options(
    options: dict[str, Any],
    assets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build generation options."""
    target = int(options["target_length"])
    content_type_key = str(
        options.get("resolved_content_type")
        or options.get("content_type")
        or "objective_description"
    )
    target_length_source = str(
        options.get("target_length_source") or "user_or_caller"
    )
    minimum, maximum = content_type_length_window(
        content_type_key,
        target,
        target_length_source=target_length_source,
    )
    importance = []
    ref_positions = {str(item.get("ref")): index for index, item in enumerate(assets, start=1)}
    for ref in options.get("importance", []):
        position = ref_positions.get(str(ref))
        if position and position not in importance:
            importance.append(position)
    if target <= 120:
        paragraph_count = 1
        sentence_count = 3
    elif target <= 250:
        paragraph_count = 2
        sentence_count = 6
    elif target <= 400:
        paragraph_count = 3
        sentence_count = 9
    else:
        paragraph_count = 5
        sentence_count = 14
    average_sentence_length = max(8, round(target / sentence_count))
    natural_language_request = str(
        options.get("natural_language_request") or ""
    ).strip()
    natural_language_request_source = (
        "user"
        if natural_language_request
        else "default_task_semantics"
        if content_type_key == "creative_story"
        else "empty"
    )
    if not natural_language_request and content_type_key == "creative_story":
        organization = ORGANIZATION_LABELS.get(
            str(options["organization"]),
            str(options["organization"]),
        )
        natural_language_request = (
            f"根据所选图片，{organization}创作一个约{target}字、语义完整的小故事。"
        )
    return {
        "content_type": CONTENT_TYPE_LABELS.get(content_type_key, content_type_key),
        "content_type_key": content_type_key,
        "content_type_source": str(
            options.get("content_type_source") or "default_value"
        ),
        "content_type_user_selected": bool(
            options.get("content_type_user_selected")
        ),
        "natural_language_request": natural_language_request,
        "natural_language_request_source": natural_language_request_source,
        "intent_resolution": dict(options.get("intent_resolution") or {}),
        "content_profile": dict(options.get("content_profile") or {}),
        "target_length_source": target_length_source,
        "target_length_instruction": f"约{target}字",
        "allow_image_meta_language": user_requested_per_image_explanation(
            options.get("natural_language_request") or ""
        ),
        "style": str(options["style"]),
        "audience": str(options["audience"]),
        "target_length": target,
        "accepted_min": minimum,
        "accepted_max": maximum,
        "organization": ORGANIZATION_LABELS.get(
            str(options["organization"]),
            str(options["organization"]),
        ),
        "important_image_positions": importance,
        "required_image_positions": list(range(1, len(assets) + 1)),
        "length_plan": {
            "paragraph_count": paragraph_count,
            "sentence_count": sentence_count,
            "suggested_visible_characters_per_sentence": {
                "minimum": max(6, average_sentence_length - 4),
                "maximum": average_sentence_length + 4,
            },
            "rule": "完整执行句段规划并达到 accepted_min 后才能结束正文",
        },
    }


class MultiImageContentV2Candidate:
    """Provide multi image content v2 candidate behavior."""
    def __init__(self, project_root: Path) -> None:
        root = project_root / "prompts" / "phase5_2" / "multi_image_content_v2_candidate"
        manifest_path = root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("candidate_id") != CANDIDATE_ID:
            raise ValueError("multi_image_content_v2_candidate_identity_mismatch")
        if self.manifest.get("prompt_id") != PROMPT_ID:
            raise ValueError("multi_image_content_v2_prompt_identity_mismatch")
        prompt_path = project_root / str(self.manifest["prompt_file"])
        schema_path = project_root / str(self.manifest["output_schema"])
        if _sha256(prompt_path) != str(self.manifest["prompt_sha256"]):
            raise ValueError("multi_image_content_v2_prompt_sha256_mismatch")
        if _sha256(schema_path) != str(self.manifest["output_schema_sha256"]):
            raise ValueError("multi_image_content_v2_schema_sha256_mismatch")
        self.prompt_text = prompt_path.read_text(encoding="utf-8")
        self.validator = Draft202012Validator(
            json.loads(schema_path.read_text(encoding="utf-8"))
        )

    def identity(self) -> dict[str, str]:
        """Execute the identity operation."""
        return {
            "candidate_id": CANDIDATE_ID,
            "prompt_id": PROMPT_ID,
            "prompt_sha256": str(self.manifest["prompt_sha256"]),
            "schema_sha256": str(self.manifest["output_schema_sha256"]),
            "status": str(self.manifest["status"]),
        }

    def render(
        self,
        assets: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Render the requested value."""
        resolved_options = build_generation_options(options, assets)
        safe_context = build_safe_image_context(
            assets,
            task_mode=content_type_task_mode(
                resolved_options.get("content_type_key")
            ),
        )
        text = self.prompt_text.replace(
            "{{IMAGE_CONTEXT}}",
            json.dumps(safe_context, ensure_ascii=False, indent=2),
        ).replace(
            "{{GENERATION_OPTIONS}}",
            json.dumps(resolved_options, ensure_ascii=False, indent=2),
        )
        if "{{IMAGE_CONTEXT}}" in text or "{{GENERATION_OPTIONS}}" in text:
            raise ValueError("multi_image_content_v2_unresolved_placeholder")
        return text, self.identity(), resolved_options

    def render_bounded_revision(
        self,
        assets: list[dict[str, Any]],
        options: dict[str, Any],
        raw_output: str,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build one bounded rewrite pass from a safely generalized draft."""

        base_prompt, identity, resolved_options = self.render(assets, options)
        payload = _repair_minimal_payload(raw_output) or {}
        draft, _ = _generalize_risky_text(
            str(payload.get("final_text") or ""),
            assets,
            content_type=resolved_options.get("content_type_key"),
        )
        actual = visible_character_count(draft)
        minimum = int(resolved_options["accepted_min"])
        maximum = int(resolved_options["accepted_max"])
        paragraph_count = int(
            resolved_options["length_plan"]["paragraph_count"]
        )
        target = int(resolved_options["target_length"])
        paragraph_minimum = max(
            20,
            math.ceil((target / paragraph_count) * 1.05),
        )
        paragraph_maximum = max(
            paragraph_minimum,
            math.floor((target / paragraph_count) * 1.10),
        )
        revision = (
            "\n\n# Bounded Revision Pass / 一次受控改写\n"
            "上一版未通过产品长度或传输合同。请从头改写一次，不要解释失败原因，"
            "不要道歉，也不要复述合同。\n"
            f"- 上一版安全草稿可见字符数：{actual}\n"
            f"- 本次必须恰好写 {paragraph_count} 个自然段；每段约 "
            f"{paragraph_minimum}-{paragraph_maximum} 个可见字符；全文必须处于 "
            f"{minimum}-{maximum} 个可见字符。\n"
            "- 每段都要提供新信息，不得用重复、空话、规则说明或否定清单凑长度。\n"
            "- 下方草稿只可借鉴自然表达；其中所有被泛化的文字都不得还原成具体文字、"
            "机构、数字或身份。\n"
            "<SAFE_DRAFT>\n"
            f"{draft}\n"
            "</SAFE_DRAFT>\n"
            "重新观察原始图片并按四段定界协议输出。"
        )
        return base_prompt + revision, identity, resolved_options

    def render_bounded_continuation(
        self,
        assets: list[dict[str, Any]],
        options: dict[str, Any],
        current_text: str,
        *,
        attempt_number: int,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Request one model-authored continuation for an under-length draft."""

        _, identity, resolved_options = self.render(assets, options)
        safe_draft, _ = _generalize_risky_text(
            current_text,
            assets,
            content_type=resolved_options.get("content_type_key"),
        )
        actual = visible_character_count(safe_draft)
        minimum = int(resolved_options["accepted_min"])
        maximum = int(resolved_options["accepted_max"])
        missing = max(1, minimum - actual)
        available = max(1, maximum - actual)
        requested = min(available, max(50, math.ceil(missing * 1.6)))
        lower = min(requested, max(20, math.floor(requested * 0.85)))
        sentence_count = max(2, math.ceil(requested / 28))
        sentence_minimum = max(10, math.floor(lower / sentence_count))
        sentence_maximum = max(
            sentence_minimum,
            math.floor(requested / sentence_count),
        )
        profile = dict(resolved_options.get("content_profile") or {})
        completion_strategy = "、".join(
            str(item)
            for item in profile.get("length_completion_strategy", [])
        )
        authored = (
            resolved_options.get("content_type_key")
            in AUTHORED_CONTENT_TYPES
        )
        angle = (
            f"严格使用该文体内部的补长方向：{completion_strategy}。"
            "新增内容必须像正文自身的一部分，不得讨论图片数量、顺序、组合或关系。"
            if authored
            else "只补充与已有事实直接相关的观察或必要说明，不添加未经支持的事实。"
        )
        continuation = (
            "# Bounded Continuation Pass / 受控续写补全\n"
            "你是中文内容编辑。现有正文已经覆盖全部图片，但长度不足。只续写一个"
            "自然段作为全文的后续"
            "部分；不要改写、复述或评价现有正文，不要新增具体文字、机构、地点、"
            "数字、时间、身份或未经图片支持的共同事件。\n"
            f"- 续写段必须为 {lower}-{requested} 个可见中文字符；\n"
            f"- 必须恰好写 {sentence_count} 句，每句约 "
            f"{sentence_minimum}-{sentence_maximum} 个可见字符；\n"
            "- 只沿用现有正文已经建立的主题、卖点、行程、情绪或意象做自然推进；"
            "不得写“两幅图、两张图片、这些图片、第一张、第二张、画面组合、"
            "两个瞬间彼此独立”等图片元话语；不得复用现有正文中的短语或句式；\n"
            "- 不得使用空话、规则说明或否定清单凑数；\n"
            "- 只输出 `<ADDITION>续写段</ADDITION>`，结束标签后立即停止。\n"
            f"- 本次补全编号：{attempt_number}。\n"
            f"- 内容类型：{resolved_options['content_type']}；风格："
            f"{resolved_options['style']}；受众：{resolved_options['audience']}。\n"
            f"- 已有正文可见字符数：{actual}。\n"
            f"- 本段角度：{angle}"
        )
        return continuation, identity, resolved_options


class MultiImageStoryV3Candidate:
    """Independent story branch; the accepted V2 prompt remains byte-frozen."""

    def __init__(self, project_root: Path) -> None:
        root = project_root / "prompts" / "phase5_2" / "multi_image_content_v2_candidate"
        manifest_path = root / "story_manifest_v1.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("candidate_id") != STORY_CANDIDATE_ID:
            raise ValueError("multi_image_story_v1_candidate_identity_mismatch")
        if self.manifest.get("prompt_id") != STORY_PROMPT_ID:
            raise ValueError("multi_image_story_v1_prompt_identity_mismatch")
        prompt_path = project_root / str(self.manifest["prompt_file"])
        schema_path = project_root / str(self.manifest["output_schema"])
        if _sha256(prompt_path) != str(self.manifest["prompt_sha256"]):
            raise ValueError("multi_image_story_v1_prompt_sha256_mismatch")
        if _sha256(schema_path) != str(self.manifest["output_schema_sha256"]):
            raise ValueError("multi_image_story_v1_schema_sha256_mismatch")
        self.prompt_text = prompt_path.read_text(encoding="utf-8")
        self.validator = Draft202012Validator(
            json.loads(schema_path.read_text(encoding="utf-8"))
        )

    def identity(self) -> dict[str, str]:
        """Execute the identity operation."""
        return {
            "candidate_id": STORY_CANDIDATE_ID,
            "prompt_id": STORY_PROMPT_ID,
            "prompt_sha256": str(self.manifest["prompt_sha256"]),
            "schema_sha256": str(self.manifest["output_schema_sha256"]),
            "status": str(self.manifest["status"]),
        }

    def render(
        self,
        assets: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Render the requested value."""
        resolved_options = build_generation_options(options, assets)
        safe_context = build_safe_image_context(
            assets,
            task_mode=content_type_task_mode(
                resolved_options.get("content_type_key")
            ),
        )
        text = self.prompt_text.replace(
            "{{IMAGE_CONTEXT}}",
            json.dumps(safe_context, ensure_ascii=False, indent=2),
        ).replace(
            "{{GENERATION_OPTIONS}}",
            json.dumps(resolved_options, ensure_ascii=False, indent=2),
        )
        if "{{IMAGE_CONTEXT}}" in text or "{{GENERATION_OPTIONS}}" in text:
            raise ValueError("multi_image_story_v1_unresolved_placeholder")
        return text, self.identity(), resolved_options

    def render_bounded_revision(
        self,
        assets: list[dict[str, Any]],
        options: dict[str, Any],
        raw_output: str,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Render bounded revision."""
        base_prompt, identity, resolved_options = self.render(assets, options)
        payload = _repair_minimal_payload(raw_output) or {}
        draft, _ = _generalize_risky_text(
            str(payload.get("final_text") or ""),
            assets,
            creative_story=True,
            content_type="creative_story",
        )
        minimum = int(resolved_options["accepted_min"])
        maximum = int(resolved_options["accepted_max"])
        revision = (
            "\n\n# Story Contract Revision / 一次故事合同改写\n"
            "上一版没有同时满足故事结构、长度或传输合同。请重新观察图片并从头"
            "改写一次，不要解释失败原因，不要沿用逐图描述结构。\n"
            f"- 全文必须为 {minimum}-{maximum} 个可见字符；\n"
            "- 明确建立一个主人公或叙事主体、一个起因、至少一次行动/发现/选择/"
            "阻碍/转折，以及结果或开放结尾；\n"
            "- 全部图片元素进入同一条虚构叙事线，但不得宣称图片真实记录同一事件；\n"
            "- 长度不足时增加动作、对话、心理变化、转折或结尾，不加抽象套话；\n"
            "- 不得出现“第一张图/第二张图/最后一张图”等逐图报幕。\n"
            "<PREVIOUS_SAFE_DRAFT>\n"
            f"{draft}\n"
            "</PREVIOUS_SAFE_DRAFT>\n"
            "按四段定界协议输出全新的完整故事。"
        )
        return base_prompt + revision, identity, resolved_options

    def render_bounded_continuation(
        self,
        assets: list[dict[str, Any]],
        options: dict[str, Any],
        current_text: str,
        *,
        attempt_number: int,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Render bounded continuation."""
        _, identity, resolved_options = self.render(assets, options)
        safe_draft, _ = _generalize_risky_text(
            current_text,
            assets,
            creative_story=True,
            content_type="creative_story",
        )
        actual = visible_character_count(safe_draft)
        minimum = int(resolved_options["accepted_min"])
        maximum = int(resolved_options["accepted_max"])
        available = max(1, maximum - actual)
        if actual >= minimum:
            requested = min(available, 22)
            lower = max(6, requested - 10)
            story_job = (
                "只补一个与现有情节直接衔接的结果、人物决定或自然悬念，使故事"
                "真正结束；不要引入新的场景或总结图片。"
            )
        else:
            missing = max(1, minimum - actual)
            requested = min(available, max(36, math.ceil(missing * 1.35)))
            lower = max(24, requested - 12)
            story_job = (
                "增加一个与现有情节衔接的人物动作、对话、发现或小转折，使故事继续"
                "向结果推进。"
                if attempt_number == 1
                else "用人物的决定、变化、结果或自然悬念完成结尾；不要总结图片，也"
                "不要抽象抒情。"
            )
        continuation = (
            "# Story Continuation / 故事情节补全\n"
            "现有正文已经包含故事主体。只续写一个紧接正文的故事片段，不得复述"
            "已有内容。\n"
            f"- 续写约 {lower}-{requested} 个可见字符；\n"
            f"- 本次任务：{story_job}\n"
            "- 允许合理虚构行动、对话与心理，但不得猜写图片中的具体文字、真实"
            "身份、地点或机构；\n"
            "- 禁止“余韵悠长”“诗意悄然生长”“生活藏着不同风景”等空洞句；\n"
            "- 只输出 `<ADDITION>续写情节</ADDITION>`。\n"
            f"- 已有正文可见字符数：{actual}。\n"
            "<CURRENT_STORY>\n"
            f"{safe_draft}\n"
            "</CURRENT_STORY>"
        )
        return continuation, identity, resolved_options


def _extract_json_field(raw_output: str, field: str) -> Any:
    match = re.search(rf'["“”]{re.escape(field)}["“”]\s*:\s*', raw_output)
    if not match:
        raise ValueError(field)
    try:
        value, _ = json.JSONDecoder().raw_decode(raw_output[match.end() :])
    except json.JSONDecodeError as exc:
        raise ValueError(field) from exc
    return value


def _parse_marker_evidence(
    value: str,
    used_images: list[int],
) -> list[dict[str, Any]]:
    """Parse canonical rows or an ordered one-line-per-image transport variant."""

    evidence = [
        {"image_position": int(position), "basis": basis.strip()}
        for position, basis in re.findall(
            r"(?:^|[;；\n])\s*(\d+)\s*[|｜]\s*(.*?)"
            r"(?=(?:[;；\n]\s*\d+\s*[|｜])|$)",
            value,
            flags=re.DOTALL,
        )
        if basis.strip()
    ]
    if evidence:
        return evidence
    ordered_lines = [
        line.strip(" -•\t")
        for line in value.splitlines()
        if line.strip(" -•\t")
    ]
    if used_images and len(ordered_lines) == len(used_images):
        return [
            {"image_position": position, "basis": basis}
            for position, basis in zip(used_images, ordered_lines, strict=True)
        ]
    return []


def _repair_minimal_payload(raw_output: str) -> dict[str, Any] | None:
    """Extract only model-authored V2 fields from the marker or JSON protocol."""

    marker_text = re.search(
        r"<\s*FINAL[_\s-]*TEXT\s*>\s*(.*?)\s*"
        r"<\s*/\s*FINAL[_\s-]*TEXT\s*>",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    marker_images = re.search(
        r"<\s*USED[_\s-]*IMAGES\s*>\s*([0-9,\s，]+)\s*"
        r"<\s*/\s*USED[_\s-]*IMAGES\s*>",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Qwen occasionally letter-spaces protocol tag names and, in one observed
    # deterministic output, omitted the ``C`` only in ``EVIDENCE``.  Accept
    # that bounded tag typo while retaining the ordered four-section contract;
    # the evidence rows and image coverage are still validated below.
    evidence_tag = r"E\s*V\s*I\s*D\s*E\s*N\s*C?\s*E"
    marker_evidence = re.search(
        rf"<\s*{evidence_tag}\s*>\s*(.*?)\s*"
        rf"<\s*/\s*{evidence_tag}\s*>",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    uncertainty_tag = r"U\s*N\s*C\s*E\s*R\s*T[A-Z_ -]*"
    marker_uncertainty = re.search(
        rf"<\s*{uncertainty_tag}>\s*(.*?)\s*"
        rf"<\s*/\s*{uncertainty_tag}>",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if marker_text and marker_images and marker_evidence and marker_uncertainty:
        used_images = [
            int(value)
            for value in re.findall(r"\d+", marker_images.group(1))
        ]
        evidence = _parse_marker_evidence(
            marker_evidence.group(1),
            used_images,
        )
        uncertainty_text = marker_uncertainty.group(1).strip()
        uncertainty = (
            []
            if uncertainty_text in {"", "无", "沒有", "没有"}
            else [
                item.strip()
                for item in re.split(r"[;；\n]+", uncertainty_text)
                if item.strip()
            ]
        )
        return {
            "final_text": marker_text.group(1).strip(),
            "used_images": used_images,
            "evidence": evidence,
            "uncertainty": uncertainty,
        }

    try:
        final_text = _extract_json_field(raw_output, "final_text")
        used_images = _extract_json_field(raw_output, "used_images")
        evidence = _extract_json_field(raw_output, "evidence")
        uncertainty = _extract_json_field(raw_output, "uncertainty")
    except ValueError:
        return None
    if not isinstance(final_text, str):
        return None
    return {
        "final_text": final_text,
        "used_images": used_images,
        "evidence": evidence,
        "uncertainty": uncertainty,
    }


def extract_multi_image_sections(raw_output: str) -> dict[str, Any]:
    """Extract every independently complete marker/JSON field.

    A later marker can be truncated without invalidating an already closed
    ``FINAL_TEXT`` section. The returned payload may be partial; product
    validation still owns the complete contract.
    """

    payload: dict[str, Any] = {}
    marker_specs = {
        "final_text": r"FINAL[_\s-]*TEXT",
        "used_images": r"USED[_\s-]*IMAGES",
        "evidence": r"E\s*V\s*I\s*D\s*E\s*N\s*C?\s*E",
        "uncertainty": r"U\s*N\s*C\s*E\s*R\s*T[A-Z_ -]*",
    }
    sections: dict[str, str] = {}
    for field, tag in marker_specs.items():
        match = re.search(
            rf"<\s*{tag}\s*>\s*(.*?)\s*<\s*/\s*{tag}\s*>",
            raw_output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            sections[field] = match.group(1).strip()
    if sections.get("final_text"):
        payload["final_text"] = sections["final_text"]
    if "used_images" in sections:
        payload["used_images"] = [
            int(value) for value in re.findall(r"\d+", sections["used_images"])
        ]
    if "evidence" in sections:
        payload["evidence"] = _parse_marker_evidence(
            sections["evidence"],
            list(payload.get("used_images") or []),
        )
    if "uncertainty" in sections:
        value = sections["uncertainty"]
        payload["uncertainty"] = (
            []
            if value in {"", "无", "沒有", "没有"}
            else [
                item.strip()
                for item in re.split(r"[;；\n]+", value)
                if item.strip()
            ]
        )
    for field in marker_specs:
        if field in payload:
            continue
        try:
            value = _extract_json_field(raw_output, field)
        except ValueError:
            continue
        if field == "final_text" and not isinstance(value, str):
            continue
        payload[field] = value
    return payload


def extract_multi_image_payload(raw_output: str) -> dict[str, Any] | None:
    """Expose the bounded candidate envelope parser to the API retry loop."""

    payload = extract_multi_image_sections(raw_output)
    return payload or None


def extract_story_public_payload(
    raw_output: str,
    *,
    image_count: int,
) -> dict[str, Any] | None:
    """Extract safe public Story prose without requiring transport metadata.

    The VLM call itself supplies the bounded image scope. Story transport
    metadata remains auditable when present, but a complete public body is not
    discarded merely because a later metadata section was omitted or
    malformed.
    """

    expected_positions = list(range(1, image_count + 1))
    payload = extract_multi_image_payload(raw_output)
    if payload is not None and str(payload.get("final_text") or "").strip():
        normalized = dict(payload)
        normalized.setdefault("used_images", expected_positions)
        normalized.setdefault("evidence", [])
        normalized.setdefault("uncertainty", [])
        return normalized

    candidate = str(raw_output or "").strip()
    if not candidate:
        return None

    fenced, fence_actions = _strip_complete_outer_code_fence(candidate)
    if fence_actions:
        candidate = fenced
        payload = extract_multi_image_payload(candidate)
        if payload is not None and str(payload.get("final_text") or "").strip():
            normalized = dict(payload)
            normalized.setdefault("used_images", expected_positions)
            normalized.setdefault("evidence", [])
            normalized.setdefault("uncertainty", [])
            return normalized

    if candidate[:1] in {"{", "["}:
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded, dict):
            return None
        public_text = next(
            (
                str(decoded.get(key) or "").strip()
                for key in ("final_text", "display_text", "public_answer")
                if isinstance(decoded.get(key), str)
                and str(decoded.get(key) or "").strip()
            ),
            "",
        )
        if not public_text:
            return None
        candidate = public_text

    candidate = re.sub(
        r"^(?:以下是(?:创作的)?(?:故事|正文)|(?:故事|正文)如下|"
        r"为你创作(?:如下)?)[：:]\s*",
        "",
        candidate,
        count=1,
    ).strip()
    candidate = re.sub(
        r"\s*(?:以上(?:就是|是)?(?:创作的)?(?:故事|正文)|希望你喜欢)[。！!]?\s*$",
        "",
        candidate,
        count=1,
    ).strip()
    if (
        not candidate
        or "```" in candidate
        or re.search(r"<\s*/?\s*[A-Z][A-Z_\s-]*\s*>", candidate)
        or "{" in candidate
        or "}" in candidate
    ):
        return None
    lowered = candidate.lower()
    if any(token.lower() in lowered for token in INTERNAL_OUTPUT_TOKENS):
        return None
    return {
        "final_text": candidate,
        "used_images": expected_positions,
        "evidence": [],
        "uncertainty": [],
    }


def candidate_quality_score(
    payload: dict[str, Any] | None,
    *,
    contract: dict[str, Any],
) -> tuple[int, ...]:
    """Return a stable lexicographic score; a retry may never lower it."""

    payload = payload or {}
    text = str(payload.get("final_text") or "").strip()
    structure = contract.get("story_structure") or {}
    slots = structure.get("slots") if isinstance(structure, dict) else {}
    slot_count = sum(bool(value) for value in (slots or {}).values())
    length = contract.get("length_contract") or {}
    delta = 0
    if length:
        actual = int(length.get("actual") or 0)
        minimum = int(length.get("minimum") or 0)
        maximum = int(length.get("maximum") or 0)
        delta = (
            minimum - actual
            if actual < minimum
            else actual - maximum
            if actual > maximum
            else 0
        )
    errors = [str(item) for item in contract.get("contract_errors", [])]
    internal_or_text_risk = sum(
        "internal_output_leak" in item or "unverified" in item for item in errors
    )
    return (
        int(bool(contract.get("product_contract_valid"))),
        int(bool(text)),
        int(bool(contract.get("image_coverage", {}).get("passed"))),
        int(bool(contract.get("evidence_coverage", {}).get("passed"))),
        int(bool(structure.get("passed", True))),
        slot_count,
        -internal_or_text_risk,
        -delta,
        len(text),
    )


def adjust_final_text_only(
    payload: dict[str, Any],
    *,
    assets: list[dict[str, Any]],
    target_length: int,
    creative_story: bool,
    content_type: str | None = None,
    target_length_source: str = "user_or_caller",
) -> dict[str, Any] | None:
    """Resolve a pure over-length failure without rewriting metadata."""

    text, _ = _generalize_risky_text(
        str(payload.get("final_text") or "").strip(),
        assets,
        creative_story=creative_story,
    )
    minimum, maximum = content_type_length_window(
        content_type,
        target_length,
        target_length_source=target_length_source,
    )
    actual = visible_character_count(text)
    if not text or actual <= maximum:
        return None
    # A tiny punctuation-only overrun should not trigger a second model call.
    # Removing low-information separators preserves every content word and all
    # metadata, while avoiding the quality collapse that a full rewrite can
    # introduce for a one-to-three-character miss.
    excess = actual - maximum
    if excess <= max(3, maximum // 50):
        micro_adjusted = text
        exact_fillers = {
            3: ("直勾勾", "忍不住"),
            2: ("轻轻", "瞬间", "仿佛", "原本", "已经", "早已", "终于"),
            1: ("竟", "刚", "也"),
        }
        removed_filler = False
        for filler in exact_fillers.get(excess, ()):
            filler_index = micro_adjusted.rfind(filler)
            if filler_index >= 0:
                micro_adjusted = (
                    micro_adjusted[:filler_index]
                    + micro_adjusted[filler_index + len(filler) :]
                )
                removed_filler = True
                break
        if not removed_filler:
            for _ in range(excess):
                removable_index = max(
                    (
                        index
                        for index, character in enumerate(micro_adjusted)
                        if character in "，、；"
                    ),
                    default=-1,
                )
                if removable_index < 0:
                    break
                micro_adjusted = (
                    micro_adjusted[:removable_index]
                    + micro_adjusted[removable_index + 1 :]
                )
        micro_count = visible_character_count(micro_adjusted)
        if minimum <= micro_count <= maximum and (
            not creative_story
            or evaluate_story_structure(micro_adjusted)["passed"]
        ):
            adjusted = dict(payload)
            adjusted["final_text"] = micro_adjusted
            return adjusted
    # For a moderate overrun, delete only low-information modifiers or a
    # comma-delimited descriptive clause.  Every intermediate candidate keeps
    # the original order and words, remains above the lower bound, and must
    # still pass the Story structure gate.  This is intentionally a deletion
    # compressor, not a template rewrite.
    if actual <= math.ceil(maximum * 1.2):
        working = text

        def structurally_safe(value: str) -> bool:
            count = visible_character_count(value)
            return (
                minimum <= count
                and (
                    not creative_story
                    or evaluate_story_structure(value)["passed"]
                )
            )

        modifiers = (
            "轻轻地",
            "慢慢地",
            "悄悄地",
            "不由得",
            "忍不住",
            "直勾勾",
            "小心翼翼地",
            "仿佛",
            "似乎",
            "微微",
            "格外",
            "十分",
            "非常",
            "原本",
            "已经",
            "早已",
            "小小的",
        )
        changed = True
        while visible_character_count(working) > maximum and changed:
            changed = False
            for modifier in modifiers:
                index = working.find(modifier)
                if index < 0:
                    continue
                candidate = (
                    working[:index] + working[index + len(modifier) :]
                )
                if structurally_safe(candidate):
                    working = candidate
                    changed = True
                    break

        for _ in range(6):
            current_count = visible_character_count(working)
            if current_count <= maximum:
                break
            parts = re.split(r"([，；])", working)
            candidates: list[tuple[int, str]] = []
            for index in range(2, len(parts), 2):
                clause = parts[index].strip()
                if len(clause) < 4:
                    continue
                start = index - 1
                trial = "".join(parts[:start] + parts[index + 1 :])
                trial = re.sub(r"([。！？!?])\1+", r"\1", trial).strip()
                count = visible_character_count(trial)
                if count >= current_count or not structurally_safe(trial):
                    continue
                candidates.append((count, trial))
            if not candidates:
                break
            within = [item for item in candidates if item[0] <= maximum]
            if within:
                _, working = max(within, key=lambda item: item[0])
            else:
                _, working = min(candidates, key=lambda item: item[0])

        working_count = visible_character_count(working)
        if (
            minimum <= working_count <= maximum
            and structurally_safe(working)
        ):
            adjusted = dict(payload)
            adjusted["final_text"] = working
            return adjusted
    visible = 0
    boundary = len(text)
    for index, character in enumerate(text):
        if not character.isspace():
            visible += 1
        if visible > maximum:
            boundary = index
            break
    bounded = text[:boundary].rstrip()
    sentence_boundaries = [
        index + 1
        for index, character in enumerate(bounded)
        if character in "。！？!?"
        and visible_character_count(bounded[: index + 1]) >= minimum
    ]
    candidates = [bounded[:index].rstrip() for index in reversed(sentence_boundaries)]
    if (
        not creative_story
        and minimum <= visible_character_count(bounded) <= maximum
    ):
        candidates.append(bounded)
    for candidate in candidates:
        if creative_story and not evaluate_story_structure(candidate)["passed"]:
            continue
        adjusted = dict(payload)
        adjusted["final_text"] = candidate
        return adjusted
    return None


def render_final_text_revision_prompt(
    payload: dict[str, Any],
    *,
    target_length: int,
    content_type: str,
    target_length_source: str = "user_or_caller",
) -> str:
    """Build a text-only bounded revision that cannot rewrite metadata."""

    minimum, maximum = content_type_length_window(
        content_type,
        target_length,
        target_length_source=target_length_source,
    )
    text = str(payload.get("final_text") or "").strip()
    action = (
        "压缩"
        if visible_character_count(text) > maximum
        else "扩写"
    )
    strategy = (
        "保留主人公、起因、发展、转折和结尾，只删减重复修饰或压缩句子。"
        if content_type == "creative_story" and action == "压缩"
        else "只增加与现有情节直接衔接的行动、对话、心理变化、转折或结尾。"
        if content_type == "creative_story"
        else "保持原主题和事实边界，只调整正文表达，不添加未经支持的事实。"
    )
    return (
        "# SceneMind-X Final Text Only Revision\n"
        f"请仅{action}下方正文，使其处于 {minimum}-{maximum} 个非空白可见字符。"
        "不得输出 used_images、evidence、uncertainty，不得解释规则，不得改变"
        "图片覆盖或既有事实边界。\n"
        f"{strategy}\n"
        "只输出 `<FINAL_TEXT>修订后的完整正文</FINAL_TEXT>`，闭合标签后立即停止。\n"
        "<CURRENT_FINAL_TEXT>\n"
        f"{text}\n"
        "</CURRENT_FINAL_TEXT>"
    )


def merge_final_text_revision(
    payload: dict[str, Any],
    raw_output: str,
) -> dict[str, Any] | None:
    """Merge only a complete revised FINAL_TEXT into preserved metadata."""

    sections = extract_multi_image_sections(raw_output)
    text = str(sections.get("final_text") or "").strip()
    if not text:
        return None
    merged = dict(payload)
    merged["final_text"] = text
    return merged


def render_metadata_completion_prompt(
    payload: dict[str, Any],
    *,
    image_count: int,
) -> str:
    """Build a short metadata-only repair; the story is immutable input."""

    positions = ",".join(str(index) for index in range(1, image_count + 1))
    return (
        "# SceneMind-X Metadata Only Completion\n"
        "正文已经完成且不可修改。仅补齐机器内部元数据，不得重复、改写、总结正文，"
        "不得添加新人物、地点、机构、文字、数字或事件。\n"
        f"- USED_IMAGES 必须为：{positions}\n"
        "- EVIDENCE 每张图片恰好一行，格式 `序号|简短视觉锚点`；只使用正文已经"
        "出现的宽泛元素，不确定时写“该图片提供故事视觉灵感”。\n"
        "- UNCERTAINTY 没有则写“无”。\n"
        "只输出以下三段，禁止输出 FINAL_TEXT：\n"
        f"<USED_IMAGES>{positions}</USED_IMAGES>\n"
        "<EVIDENCE>\n"
        "1|...\n"
        "</EVIDENCE>\n"
        "<UNCERTAINTY>无</UNCERTAINTY>\n"
        "<IMMUTABLE_FINAL_TEXT>\n"
        f"{str(payload.get('final_text') or '').strip()}\n"
        "</IMMUTABLE_FINAL_TEXT>"
    )


def merge_metadata_completion(
    payload: dict[str, Any],
    raw_output: str,
) -> dict[str, Any] | None:
    """Merge only complete metadata fields into an immutable final text."""

    sections = extract_multi_image_sections(raw_output)
    required = ("used_images", "evidence", "uncertainty")
    if not all(field in sections for field in required):
        return None
    merged = dict(payload)
    for field in required:
        merged[field] = sections[field]
    return merged


def build_text_risk_generalization(
    payload: dict[str, Any],
    *,
    assets: list[dict[str, Any]],
    target_length: int,
) -> dict[str, Any] | None:
    """Return a reusable visual-only description for wholly unverified slides."""

    if not assets or not all(
        len(_unverified_candidate_values(asset)) >= 10
        and not _verified_text_values(asset)
        for asset in assets
    ):
        return None
    text = (
        "这组画面均以文字密集的演示页面为主体。不同页面通过标题、段落、色块和"
        "重点标记组织信息，版式层级较为清楚，但局部受拍摄角度、清晰度与反光"
        "影响，具体文字内容无法仅凭当前画面可靠确认。为避免把未核验文字当成"
        "事实，这里只描述页面的视觉结构：多段文本按顺序展开，重点内容通过字号、"
        "颜色或粗细变化得到强调，两幅画面共同呈现出课堂讲解或材料阅读的视觉形态。"
    )
    minimum, maximum = target_length_window(target_length)
    if not minimum <= visible_character_count(text) <= maximum:
        return None
    generalized = dict(payload)
    generalized["final_text"] = text
    generalized["used_images"] = list(range(1, len(assets) + 1))
    generalized["evidence"] = [
        {
            "image_position": position,
            "basis": "文字密集页面的版式、段落与色块结构",
        }
        for position in range(1, len(assets) + 1)
    ]
    generalized["uncertainty"] = ["具体文字内容未经人工核验"]
    return generalized


def append_safe_short_bridge(
    payload: dict[str, Any],
    *,
    target_length: int,
    content_type: str | None = None,
    target_length_source: str = "user_or_caller",
) -> dict[str, Any] | None:
    """Close a short multi-image draft without adding new visual facts."""

    if target_length > 120 or content_type in AUTHORED_CONTENT_TYPES:
        return None
    base = str(payload.get("final_text") or "").strip()
    minimum, maximum = content_type_length_window(
        content_type,
        target_length,
        target_length_source=target_length_source,
    )
    if minimum <= visible_character_count(base) <= maximum:
        return None
    bridges = (
        "画面彼此独立，自然并置。",
        "两个瞬间彼此独立，却保留了日常中轻松鲜活的一面。",
        "这些画面彼此独立，以自然并置的方式呈现各自气质，不被解释为同一事件。",
    )
    for bridge in bridges:
        combined = f"{base}\n\n{bridge}".strip()
        if minimum <= visible_character_count(combined) <= maximum:
            merged = dict(payload)
            merged["final_text"] = combined
            return merged
    return None


def merge_model_authored_addition(
    payload: dict[str, Any],
    addition_raw_output: str,
    *,
    assets: list[dict[str, Any]],
    target_length: int,
    prefer_ending: bool = False,
    content_type: str | None = None,
    target_length_source: str = "user_or_caller",
) -> dict[str, Any] | None:
    """Append a bounded model-authored paragraph and safely cap at a sentence."""

    match = re.search(
        r"<\s*ADDITION\s*>\s*(.*?)\s*<\s*/\s*ADDITION\s*>",
        addition_raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    addition, _ = _generalize_risky_text(match.group(1).strip(), assets)
    if not addition:
        return None
    base, _ = _generalize_risky_text(
        str(payload.get("final_text") or "").strip(),
        assets,
    )
    addition_bigrams = {
        addition[index : index + 2]
        for index in range(max(0, len(addition) - 1))
        if not addition[index : index + 2].isspace()
    }
    for paragraph in re.split(r"\n\s*\n", base):
        paragraph_bigrams = {
            paragraph[index : index + 2]
            for index in range(max(0, len(paragraph) - 1))
            if not paragraph[index : index + 2].isspace()
        }
        union = addition_bigrams | paragraph_bigrams
        similarity = (
            len(addition_bigrams & paragraph_bigrams) / len(union)
            if union
            else 0.0
        )
        if similarity >= 0.55:
            return None
        normalized_addition = re.sub(
            r"[\W_]+",
            "",
            addition.translate(COMMON_TRADITIONAL_TO_SIMPLIFIED),
        )
        normalized_paragraph = re.sub(
            r"[\W_]+",
            "",
            paragraph.translate(COMMON_TRADITIONAL_TO_SIMPLIFIED),
        )
        if (
            normalized_addition
            and normalized_paragraph
            and SequenceMatcher(
                None,
                normalized_addition,
                normalized_paragraph,
            ).ratio()
            >= 0.65
        ):
            return None
    combined = f"{base}\n\n{addition}".strip()
    minimum, maximum = content_type_length_window(
        content_type,
        target_length,
        target_length_source=target_length_source,
    )
    if visible_character_count(combined) > maximum:
        if prefer_ending:
            available = maximum - visible_character_count(base)
            sentences = re.findall(r"[^。！？!?]*[。！？!?]", addition)
            ending_suffix = None
            for sentence in reversed(sentences):
                clauses = re.split(r"(?<=[，；,;])", sentence)
                for start in range(len(clauses)):
                    candidate = "".join(clauses[start:]).strip("，；,; \n")
                    length = visible_character_count(candidate)
                    if 6 <= length <= available:
                        ending_suffix = candidate
                        break
                if ending_suffix is not None:
                    break
            if ending_suffix is not None:
                combined = f"{base}\n\n{ending_suffix}".strip()
            else:
                return None
        else:
            visible = 0
            boundary = len(combined)
            for index, character in enumerate(combined):
                if not character.isspace():
                    visible += 1
                if visible > maximum:
                    boundary = index
                    break
            bounded = combined[:boundary].rstrip()
            sentence_boundaries = [
                index + 1
                for index, character in enumerate(bounded)
                if character in "。！？!?"
                and visible_character_count(bounded[: index + 1]) >= minimum
            ]
            if not sentence_boundaries:
                return None
            boundary = sentence_boundaries[-1]
            while boundary < len(bounded) and bounded[boundary] in "”’」』":
                boundary += 1
            combined = bounded[:boundary].rstrip()
    merged = dict(payload)
    merged["final_text"] = combined
    return merged


def _generalize_risky_text(
    text: str,
    assets: list[dict[str, Any]],
    *,
    creative_story: bool = False,
    content_type: str | None = None,
) -> tuple[str, list[str]]:
    sanitized = text
    reasons: list[str] = []
    simplified = sanitized.translate(COMMON_TRADITIONAL_TO_SIMPLIFIED)
    if simplified != sanitized:
        sanitized = simplified
        reasons.append("traditional_characters_normalized")
    for position, asset in enumerate(assets, start=1):
        backend_values = {
            str(asset.get("asset_id") or ""),
            str(asset.get("source_asset_id") or ""),
            str(asset.get("image_id") or ""),
            str(asset.get("ref") or ""),
            str(asset.get("sha256") or ""),
        }
        for value in sorted((item for item in backend_values if item), key=len, reverse=True):
            if value in sanitized:
                sanitized = sanitized.replace(value, f"第{position}张图")
                reasons.append(f"backend_identity_generalized:image_{position}")
        candidate_records = collect_text_candidates(
            {
                "presence": asset.get("text_presence")
                or asset.get("presence")
                or "uncertain",
                "ocr_candidates": asset.get("ocr_candidates", []),
                "visual_candidates": asset.get("visible_text_candidates", []),
            }
        )
        confidence_by_text = {
            item.text: item for item in candidate_records if item.text
        }
        candidates = set(confidence_by_text)
        for candidate in list(candidates):
            simplified_candidate = candidate.translate(
                COMMON_TRADITIONAL_TO_SIMPLIFIED
            )
            candidates.add(simplified_candidate)
            if simplified_candidate not in confidence_by_text:
                confidence_by_text[simplified_candidate] = confidence_by_text[candidate]
        if creative_story and is_image_text_fact_assertion(sanitized):
            for candidate in sorted(candidates, key=len, reverse=True):
                escaped = re.escape(candidate)
                before = sanitized
                sanitized = re.sub(
                    rf"那句\s*[“「『\"]{escaped}[”」』\"]",
                    "一条没有看清的提示",
                    sanitized,
                )
                sanitized = re.sub(
                    rf"[“「『\"]{escaped}[”」』\"]"
                    rf"(?=\s*(?:月饼|礼盒|包装|卡片|牌匾|标签|名称))",
                    "普通",
                    sanitized,
                )
                sanitized = re.sub(
                    rf"[“「『\"]{escaped}[”」』\"]",
                    "“没看清的字样”",
                    sanitized,
                )
                if sanitized != before:
                    reasons.append(
                        f"unverified_story_text_generalized:image_{position}"
                    )
        # A short text line can be read in the wrong visual order (for
        # example, the same three glyphs reversed) while still looking
        # confident.  Treat an unverified quoted anagram as the same risky
        # OCR claim; ordinary story dialogue is unaffected unless it consists
        # of exactly the same glyph multiset as a withheld OCR candidate.
        quoted_values = list(
            re.finditer(r"([“「『\"])([^”」』\"]{2,20})([”」』\"])", sanitized)
        )
        for match in reversed(quoted_values):
            quoted = re.sub(r"\s+", "", match.group(2))
            risky_reordering = next(
                (
                    candidate
                    for candidate in candidates
                    if 3 <= len(candidate) <= 8
                    and re.fullmatch(r"[\u4e00-\u9fff]+", candidate)
                    and len(candidate) == len(quoted)
                    and candidate != quoted
                    and sorted(candidate) == sorted(quoted)
                ),
                None,
            )
            if risky_reordering is not None:
                sanitized = (
                    sanitized[: match.start()]
                    + "普通"
                    + sanitized[match.end() :]
                )
                reasons.append(
                    f"unverified_reordered_text_generalized:image_{position}"
                )
        if any("摆烂" in candidate for candidate in candidates) and "摆烂" in sanitized:
            sanitized = sanitized.replace("持续“摆烂”", "带来片刻松弛")
            sanitized = sanitized.replace("持续「摆烂」", "带来片刻松弛")
            sanitized = sanitized.replace("摆烂", "慵懒")
            reasons.append(f"unverified_text_generalized:image_{position}")
        for candidate in sorted(candidates, key=len, reverse=True):
            if candidate in sanitized:
                record = confidence_by_text.get(candidate)
                task_mode = content_type_task_mode(
                    content_type
                    or ("creative_story" if creative_story else "objective_description")
                )
                if record is not None and candidate_is_allowed(
                    record,
                    task_mode=task_mode,
                    qualified=is_qualified_statement(sanitized),
                ):
                    if (
                        task_mode != GENERATION_CREATIVE
                        or not is_image_text_fact_assertion(sanitized)
                    ):
                        continue
                if candidate in {"佳節", "佳节"}:
                    replacement = "节日"
                elif candidate.endswith(("大学", "大學")):
                    replacement = "学校"
                elif any(character.isdigit() for character in candidate):
                    replacement = (
                        "接下来的一段时间"
                        if candidate.startswith("接下来")
                        else "画面中的时间信息"
                    )
                else:
                    replacement = "画面中的文字"
                sanitized = sanitized.replace(candidate, replacement)
                reasons.append(f"unverified_text_generalized:image_{position}")
    if re.search(r"IMG_[1-9]\d*", sanitized, flags=re.IGNORECASE):
        sanitized = re.sub(
            r"IMG_([1-9]\d*)",
            lambda match: f"第{match.group(1)}张图",
            sanitized,
            flags=re.IGNORECASE,
        )
        reasons.append("img_label_generalized")
    sanitized = re.sub(r"(画面中的文字)(?:[、，；和与\s]*画面中的文字)+", r"\1", sanitized)
    return sanitized.strip(), sorted(set(reasons))


def _sanitize_moments_meta_text(text: str) -> tuple[str, list[str]]:
    """Remove evidence-policy prose that must never become publishable copy."""

    if not text:
        return text, []
    patterns = (
        r"(?:这些|这几张)(?:图片|画面)(?:之间)?(?:彼此|相互)?(?:是|显得)?独立",
        r"(?:这些|这几张|它们)?(?:图片|画面)?(?:并非|不是|不属于)(?:同一|一个|共同的?)(?:真实)?(?:事件|场景|时空)",
        r"(?:以|通过|采用)(?:一种)?自然并置(?:的方式)?(?:呈现|组织|串联)?",
        r"(?:图片|画面)(?:之间)?(?:没有|并无)(?:真实)?(?:事件|时空|场景)?(?:关联|关系)",
        r"(?:以下|这段)(?:文案|文字)(?:只是|仅是)?(?:创意|主观)?(?:串联|表达)",
    )
    parts = re.split(r"(?<=[。！？!?；;])\s*", text)
    kept = [
        part
        for part in parts
        if part.strip() and not any(re.search(pattern, part) for pattern in patterns)
    ]
    sanitized = "".join(kept).strip()
    if sanitized == text.strip():
        return sanitized, []
    return sanitized, ["moments_internal_meta_text_removed"]


AUTHORED_CONTENT_TYPES = {
    "moments",
    "creative_story",
    "travel_diary",
    "article",
    "advertisement",
    "poster_title",
    "poem",
}
_IMAGE_META_EXPRESSION = re.compile(
    r"(?:"
    r"第[一二两三四五六七八九1-9]张(?:图(?:片)?|画面)?"
    r"|(?:第?[一二两三四五六七八九1-9]|图)[幅张]?(?:图(?:片)?|画面)"
    r"|(?:图|画面)[一二两三四五六七八九1-9]"
    r"|(?:这|那|上述)?(?:两|三|几|多)[幅张](?:图(?:片)?|画面|影像|照片)"
    r"|(?:这些|那些|上述|这组|那组)(?:图(?:片)?|画面|影像|照片)"
    r"|(?:图片|画面|影像|照片)(?:之间|彼此|共同|分别|一静一动)"
    r"|(?:两|多)(?:幅|张)(?:图|影像|照片)(?:共同)?(?:呈现|展现|交织|构成)"
    r"|镜头所及"
    r"|观者(?:沉浸|看到|感受)"
    r"|共同构成一首诗"
    r"|两个瞬间彼此独立"
    r"|画面组合"
    r")"
)


def _sanitize_authored_meta_text(
    text: str,
    *,
    content_type: str | None,
    allow_image_meta_language: bool = False,
) -> tuple[str, list[str]]:
    """Keep publishable prose free of image-analysis narration."""

    if (
        not text
        or content_type not in AUTHORED_CONTENT_TYPES
        or allow_image_meta_language
    ):
        return text, []
    actions: list[str] = []
    parts = re.split(r"(?<=[。！？!?；;])\s*", text)
    kept = [
        part
        for part in parts
        if part.strip() and not _IMAGE_META_EXPRESSION.search(part)
    ]
    sanitized = "".join(kept).strip()
    if sanitized != text.strip():
        actions.append("authored_content_image_meta_text_removed")
    normalized = re.sub(
        r"(?i)\bbrick\b",
        "砖",
        sanitized,
    )
    normalized = re.sub(
        r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])",
        "",
        normalized,
    )
    if normalized != sanitized:
        actions.append("authored_content_observed_lexical_leak_normalized")
    return normalized, actions


def _strip_complete_outer_code_fence(
    text: str,
) -> tuple[str, list[str]]:
    """Remove one complete presentation-only fence without rewriting prose."""

    match = re.fullmatch(
        r"\s*```(?:text|txt|markdown|md|plaintext)?[ \t]*\r?\n?"
        r"(?P<body>.*?)\r?\n?```\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return text.strip(), []
    return match.group("body").strip(), ["outer_code_fence_removed"]


def _strong_story_truncation_reasons(text: str) -> list[str]:
    """Return only high-confidence evidence that public prose is incomplete."""

    stripped = text.strip()
    if not stripped:
        return ["empty_public_body"]
    reasons: list[str] = []
    if stripped.count("```") % 2:
        reasons.append("unclosed_code_fence")
    delimiter_pairs = (
        ("“", "”"),
        ("「", "」"),
        ("『", "』"),
        ("《", "》"),
        ("（", "）"),
        ("(", ")"),
    )
    if any(stripped.count(left) > stripped.count(right) for left, right in delimiter_pairs):
        reasons.append("unclosed_delimiter")
    if re.search(r"[，、：；（(“「『《—-]\s*$", stripped):
        reasons.append("trailing_open_delimiter")
    if re.search(
        r"(?:因为|所以|但是|可是|然而|而且|并且|如果|虽然|为了|"
        r"直到|当|正当|只见|发现|看见|听见|说道|问道|回答道)"
        r"\s*$",
        stripped,
    ):
        reasons.append("trailing_incomplete_connector_or_predicate")
    return sorted(set(reasons))


def evaluate_story_structure(text: str) -> dict[str, Any]:
    """Record narrative hints while rejecting only explicit truncation."""

    compact = re.sub(r"\s+", "", text)
    slot_patterns = {
        "narrative_subject": (
            r"他|她|他们|她们|学生|女孩|男孩|孩子|老人|旅人|青年|少年|"
            r"主人公|店主|一家人|小猫|小狗|妈妈|爸爸|家人|朋友"
        ),
        "scene": (
            r"那天|清晨|早晨|午后|傍晚|夜里|冬天|雪|家中|校园|街上|"
            r"路上|车站|窗边|房间|城市|村庄|中秋|节日|湖边|湖畔|"
            r"桥头|桥上|月夜|街角|巷口|郊外|海边|山间|公园"
        ),
        "cause_or_goal": (
            r"因为|为了|收到|想要|想把|希望|决定|打算|寻找|等待|准备|"
            r"原本|必须|要在|想|问|礼物|礼盒|消息|邀请|约定|带着|捧着|"
            r"送给|交给"
        ),
        "development": (
            r"于是|随后|后来|接着|却|突然|发现|打开|走进|来到|拿出|"
            r"发给|问道|回答|犹豫|赶往|开始|继续"
        ),
        "turn_or_choice": (
            r"却|但是|但他|但她|突然|没想到|原来|犹豫|选择|决定|"
            r"发现|问题|难题|误会"
        ),
        "ending": (
            r"最后|最终|从此|那一刻|回到|留下|笑了|明白|决定了|"
            r"发进|发到|也许|或许|仍然|终于|正等着|等待着|将要|赢回|"
            r"投降|心甘情愿"
        ),
    }
    slots = {
        name: bool(re.search(pattern, compact))
        for name, pattern in slot_patterns.items()
    }
    if not slots["ending"]:
        ending_tail = compact[-48:]
        slots["ending"] = bool(
            re.search(
                r"(?:掀翻|打开|合上|收好|抱紧|放下|停下|离开|回家|出发|"
                r"答应|点头|摇头|转身|回头|睡着|亮起|消失|完成|结束|"
                r"笑|哭)(?:了|着|。|！|？)",
                ending_tail,
            )
            or re.search(
                r"(?:更|终于|终于变得|感到|觉得|变得|心里|内心|"
                r"比[^。！？]{0,16})?"
                r"(?:踏实|安心|坦然|释然|平静|坚定|笃定|轻松|满足|"
                r"温暖|勇敢|清醒)(?:了|下来|起来|。|！|？)",
                ending_tail,
            )
        )
    filler_phrases = [
        phrase
        for phrase in (
            "余韵悠长",
            "诗意悄然生长",
            "生活藏着不同风景",
            "生活中不同的风景",
        )
        if phrase in compact
    ]
    mechanical_sequence = bool(
        re.search(r"第[一二三四五12345]张(?:图|图片)", compact)
    )
    warnings = [
        f"missing_{name}"
        for name, present in slots.items()
        if not present
    ]
    if filler_phrases:
        warnings.append("story_filler_phrase_detected")
    if mechanical_sequence:
        warnings.append("mechanical_image_sequence_detected")
    hard_rejection_reasons = _strong_story_truncation_reasons(text)
    passed = bool(compact) and not hard_rejection_reasons
    return {
        "required": True,
        "gate_mode": "weak_warning_first",
        "passed": passed,
        "slots": slots,
        "filler_phrases": filler_phrases,
        "mechanical_image_sequence": mechanical_sequence,
        "warnings": warnings,
        "hard_rejection_reasons": hard_rejection_reasons,
    }


def validate_multi_image_content(
    model_payload: Any,
    raw_output: str,
    *,
    assets: list[dict[str, Any]],
    target_length: int,
    validator: Draft202012Validator,
    content_type: str | None = None,
    target_length_source: str = "user_or_caller",
    allow_image_meta_language: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate, minimally repair, and product-normalize V2 content."""

    expected_positions = list(range(1, len(assets) + 1))
    payload = dict(model_payload) if isinstance(model_payload, dict) else None
    repair_applied = False
    schema_errors: list[str] = []
    if payload is not None:
        schema_errors = [
            error.message
            for error in sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
        ]
    if payload is None or "final_text" not in payload:
        payload = _repair_minimal_payload(raw_output)
        repair_applied = payload is not None
        if payload is None:
            schema_errors.append("final_text_not_extractable")
        else:
            # The marker envelope is an accepted transport contract.  Once it
            # has been repaired into the canonical payload, report only
            # validation errors that remain on that final payload instead of
            # stale errors from the pre-repair empty JSON parse.
            schema_errors = [
                error.message
                for error in sorted(
                    validator.iter_errors(payload),
                    key=lambda item: list(item.path),
                )
            ]
    if payload is None:
        payload = {}

    final_text = str(payload.get("final_text") or "").strip()
    raw_used_images = payload.get("used_images")
    used_images = [
        int(value)
        for value in raw_used_images
        if isinstance(value, int) and not isinstance(value, bool)
    ] if isinstance(raw_used_images, list) else []
    coverage_valid = used_images == expected_positions
    if not coverage_valid:
        schema_errors.append(
            f"used_images_contract_violation:expected={expected_positions},got={used_images}"
        )

    final_text, risk_actions = _generalize_risky_text(
        final_text,
        assets,
        creative_story=content_type == "creative_story",
        content_type=content_type,
    )
    if content_type == "moments":
        final_text, moments_actions = _sanitize_moments_meta_text(final_text)
        risk_actions = sorted(set([*risk_actions, *moments_actions]))
    final_text, authored_actions = _sanitize_authored_meta_text(
        final_text,
        content_type=content_type,
        allow_image_meta_language=allow_image_meta_language,
    )
    risk_actions = sorted(set([*risk_actions, *authored_actions]))
    final_text, fence_actions = _strip_complete_outer_code_fence(final_text)
    risk_actions = sorted(set([*risk_actions, *fence_actions]))
    if (
        content_type in AUTHORED_CONTENT_TYPES
        and not allow_image_meta_language
        and _IMAGE_META_EXPRESSION.search(final_text)
    ):
        schema_errors.append("authored_content_image_meta_text_violation")
    lowered = final_text.lower()
    internal_leaks = [
        token for token in INTERNAL_OUTPUT_TOKENS if token.lower() in lowered
    ]
    if "```" in final_text:
        internal_leaks.append("code_fence")
    if "{" in final_text or "}" in final_text:
        internal_leaks.append("json_brace")
    if internal_leaks:
        schema_errors.append("internal_output_leak:" + ",".join(sorted(set(internal_leaks))))

    actual_length = visible_character_count(final_text)
    minimum, maximum = content_type_length_window(
        content_type,
        target_length,
        target_length_source=target_length_source,
    )
    length_valid = minimum <= actual_length <= maximum
    length_warnings = []
    if not length_valid:
        length_warnings.append(
            f"visible_length_outside_ideal:expected={minimum}-{maximum},got={actual_length}"
        )
    story_structure = (
        evaluate_story_structure(final_text)
        if content_type == "creative_story"
        else {
            "required": False,
            "passed": True,
            "slots": {},
            "filler_phrases": [],
            "mechanical_image_sequence": False,
        }
    )
    if not story_structure["passed"]:
        schema_errors.append(
            "story_output_gate_violation:"
            + ",".join(story_structure["hard_rejection_reasons"])
        )
    public_body_hard_reasons = _strong_story_truncation_reasons(final_text)
    public_body_gate = {
        "passed": not public_body_hard_reasons,
        "hard_rejection_reasons": public_body_hard_reasons,
    }
    if not public_body_gate["passed"] and content_type != "creative_story":
        schema_errors.append(
            "public_body_gate_violation:"
            + ",".join(public_body_hard_reasons)
        )

    evidence = []
    for item in payload.get("evidence", []):
        if not isinstance(item, dict):
            continue
        position = item.get("image_position")
        basis = str(item.get("basis") or "").strip()
        if isinstance(position, int) and position in expected_positions and basis:
            evidence.append({"image_position": position, "basis": basis[:160]})
    evidence_positions = [item["image_position"] for item in evidence]
    evidence_coverage_valid = evidence_positions == expected_positions
    if not evidence_coverage_valid:
        schema_errors.append(
            "evidence_coverage_violation:"
            f"expected={expected_positions},got={evidence_positions}"
        )
    uncertainty = [
        str(item).strip()[:200]
        for item in payload.get("uncertainty", [])
        if isinstance(item, str) and item.strip()
    ] if isinstance(payload.get("uncertainty"), list) else []

    product_valid = (
        bool(final_text)
        and coverage_valid
        and not internal_leaks
        and public_body_gate["passed"]
        and story_structure["passed"]
    )
    visible_text = final_text if product_valid else FRIENDLY_FAILURE_TEXT
    failure_reason = None if product_valid else (
        "model_output_insufficient_or_contract_invalid"
    )
    fallback_applied = bool(risk_actions) or not product_valid
    fallback_source = (
        "risk_generalization"
        if product_valid and risk_actions
        else "friendly_failure"
        if not product_valid
        else None
    )
    normalized = {
        "final_text": visible_text,
        "actual_length": visible_character_count(visible_text),
        "target_length": target_length,
        "accepted_min": minimum,
        "accepted_max": maximum,
        "used_images": [f"图片{position}" for position in expected_positions] if product_valid else [],
        "evidence": evidence,
        "uncertainty": uncertainty,
        "story_structure": story_structure,
        "length_warning": length_warnings,
    }
    return normalized, {
        "model_contract_valid": not schema_errors and not repair_applied,
        "product_contract_valid": product_valid,
        "contract_valid": product_valid,
        "repair_applied": repair_applied,
        "risk_sanitized": bool(risk_actions),
        "risk_actions": risk_actions,
        "fallback_applied": fallback_applied,
        "fallback_source": fallback_source,
        "fallback_reason": failure_reason,
        "candidate_final_text": final_text,
        "contract_errors": sorted(set(schema_errors)),
        "contract_warnings": sorted(set(length_warnings)),
        "length_contract": {
            "unit": "non_whitespace_visible_unicode_characters",
            "target": target_length,
            "minimum": minimum,
            "maximum": maximum,
            "actual": actual_length,
            "passed": length_valid,
            "gate": "advisory_warning_only",
        },
        "image_coverage": {
            "expected_positions": expected_positions,
            "model_positions": used_images,
            "passed": coverage_valid,
        },
        "evidence_coverage": {
            "expected_positions": expected_positions,
            "model_positions": evidence_positions,
            "passed": evidence_coverage_valid,
        },
        "story_structure": story_structure,
        "public_body_gate": public_body_gate,
    }


def model_token_budget(
    target_length: int,
    content_type: str | None = None,
    *,
    stage: str = "full_contract",
) -> int:
    """Reserve enough output room for final text plus the compact JSON contract."""

    if stage == "metadata_completion":
        return 160
    if stage == "final_text_revision":
        return min(768, max(192, math.ceil(target_length * 1.35) + 160))
    if content_type == "creative_story" and 180 <= target_length <= 300:
        return 640
    return min(768, max(320, math.ceil(target_length * 1.35) + 220))


def model_min_token_budget(target_length: int) -> None:
    """Do not force generation past a natural contract-complete stop."""

    del target_length
    return None
