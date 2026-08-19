"""Finalize and validate reusable Canonical visual annotations."""

from __future__ import annotations

import copy
import difflib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .canonical_label import canonical_json, sha256_text
from scenemindx.task_text_policy import (
    candidate_partition,
    is_qualified_statement,
    prioritize_candidate_values,
    repair_filtered_text,
)


SCHEMA_VERSION = "scenemindx_canonical_pseudo_label_v2_1_candidate"
MIGRATION_VERSION = "phase6_0b_canonical_safety_migration_v1"
VALIDATOR_VERSION = "phase6_0b_canonical_safety_validator_v1"
CAPTION_COMPOSER_VERSION = "phase6_0b_safe_facts_caption_composer_v1"

FIELD_DISPLAY_NAMES = {
    "theme": "主题",
    "short_description": "简短描述",
    "micro_tags": "微标签",
    "subjects": "主要主体",
    "actions": "动作",
    "attributes": "显著属性",
    "relations": "空间关系",
    "direct_observations": "直接观察",
    "cautious_inferences": "谨慎推断",
    "text_evidence": "文字证据",
    "uncertainties": "不确定信息",
    "ocr_status": "OCR状态",
    "verification_status": "文字验证状态",
    "semantic_status": "质量状态",
    "warning_codes": "质量提醒",
}

STATUS_DISPLAY_NAMES = {
    "pass": "检查通过",
    "warning": "存在提醒",
    "needs_review": "需要复核",
    "failed": "生成失败",
    "candidate": "候选文字，尚未验证",
    "verified": "已验证",
    "conflicted": "证据存在冲突",
    "none": "未检测到",
    "pending": "待审核",
    "approved": "已通过",
    "rejected": "已拒绝",
    "edited": "人工已编辑",
    "minor": "存在轻微提醒",
    "major": "存在明显问题",
    "critical": "存在严重问题",
    "not_reviewed": "尚未复核",
    "machine_provisional": "机器暂定",
}

WARNING_DISPLAY_NAMES = {
    "migrated_from_phase6_0a": "由上一版本兼容迁移",
    "text_candidate_redacted": "候选文字已从公开字段降级或移除",
    "text_evidence_conflicted": "文字证据存在冲突，公开描述已保守降级",
    "safe_facts_rebuilt": "安全事实已按直接观察重新构建",
    "safe_caption_rebuilt": "安全回退描述已由安全事实重新组合",
    "micro_tags_below_recommended": "微标签少于建议数量",
    "codex_visual_minor": "机器视觉复核发现轻微提醒",
    "codex_visual_major": "机器视觉复核发现明显问题",
    "punctuation_normalized": "公开中文标点已规范化",
    "recovered_at_768_tokens": "首次输出截断，已使用768-token重试恢复",
    "recovered_at_extended_tokens": "完整结构输出截断，已使用扩展输出预算恢复",
    "recovered_with_minimal_contract": "完整合同仍失败，已使用最小安全合同恢复",
}

_INTERNAL_RE = re.compile(
    r"(?:prompt|validator|trace|schema|json|asset_id|sha256|canonical|"
    r"模型指令|系统提示|内部字段)",
    re.IGNORECASE,
)
_SPECIFIC_TEXT_RE = re.compile(
    r"(?:写着|写有|印有|显示为|标题为|标牌为|字幕为|号码为|型号为|"
    r"机构名为|地点名为|商品名为|人物名为|[“「『‘'][^”」』’']{2,80}[”」』’'])"
)
_UNCERTAIN_RE = re.compile(r"(?:可能|似乎|看起来|或许|推测|大概|疑似|无法确认|不确定)")
_TERMINAL_RE = re.compile(r"[。！？；]+$")
_CLAUSE_RE = re.compile(r"(?<=[，。；！？])")
_MEDIUM_CN = {
    "photograph": "照片",
    "screenshot": "屏幕截图",
    "illustration": "插画",
    "poster_document": "海报或文档",
    "product_packaging": "产品包装图",
    "mixed": "混合媒介图片",
    "uncertain": "图片",
}


def _text(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", "").split()).strip()


def _unique(values: Iterable[Any], *, maximum: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _text(raw)
        key = unicodedata.normalize("NFKC", value).casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
        if maximum is not None and len(result) >= maximum:
            break
    return result


def normalize_chinese_punctuation(value: Any, *, sentence: bool) -> str:
    """Normalize chinese punctuation."""
    text = _text(value)
    for source, target in {",": "，", ";": "；", ":": "：", "?": "？", "!": "！"}.items():
        text = text.replace(source, target)
    text = re.sub(r"\s*([，。；：！？])\s*", r"\1", text)
    text = re.sub(r"([，。；：！？])\1+", r"\1", text)
    text = re.sub(r"[，；]+([。！？])", r"\1", text)
    if text and sentence and not _TERMINAL_RE.search(text):
        text += "。"
    return text


def _match_key(value: str) -> str:
    return re.sub(r"[\s，。；：！？、,.!?;:'\"“”‘’「」『』()（）\[\]【】_-]", "", value).casefold()


def _candidate_values(text_evidence: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("visual_candidates", "ocr_candidates", "candidate_text"):
        raw = text_evidence.get(key, [])
        values.extend(raw if isinstance(raw, list) else [raw])
    return _unique(values, maximum=24)


def _mentions_candidate(value: str, candidates: Iterable[str]) -> bool:
    value_key = _match_key(value)
    if not value_key:
        return False
    for candidate in candidates:
        candidate_key = _match_key(candidate)
        if len(candidate_key) >= 2 and (candidate_key in value_key or value_key in candidate_key):
            return True
        if (
            len(value_key) >= 4
            and len(candidate_key) >= 4
            and difflib.SequenceMatcher(None, value_key, candidate_key).find_longest_match().size
            >= 4
        ):
            return True
    return False


def _redact(
    value: Any,
    blocked_candidates: list[str],
    retained_candidates: list[str],
    *,
    whole: bool,
) -> tuple[str, bool]:
    text = _text(value)
    mentions_retained = _mentions_candidate(text, retained_candidates)
    unsafe = bool(
        _mentions_candidate(text, blocked_candidates)
        or is_qualified_statement(text)
        or (_SPECIFIC_TEXT_RE.search(text) and not mentions_retained)
    )
    if not unsafe:
        return text, False
    if whole:
        return "", True
    kept = [
        clause
        for clause in _CLAUSE_RE.split(text)
        if clause.strip()
        and not is_qualified_statement(clause)
        and not _mentions_candidate(clause, blocked_candidates)
        and (
            not _SPECIFIC_TEXT_RE.search(clause)
            or _mentions_candidate(clause, retained_candidates)
        )
    ]
    return repair_filtered_text("".join(kept), sentence=False), True


def _safe_direct(
    value: Any,
    blocked_candidates: list[str],
    retained_candidates: list[str],
    do_not_assert: list[str],
) -> bool:
    text = _text(value)
    if not text or _INTERNAL_RE.search(text):
        return False
    if _UNCERTAIN_RE.search(text) or _mentions_candidate(text, blocked_candidates):
        return False
    if _SPECIFIC_TEXT_RE.search(text) and not _mentions_candidate(
        text, retained_candidates
    ):
        return False
    key = _match_key(text)
    return not any(_match_key(item) and _match_key(item) in key for item in do_not_assert)


def compose_safe_caption(safe_facts: Iterable[str]) -> str:
    """Execute the compose safe caption operation."""
    fragments = [
        _TERMINAL_RE.sub("", normalize_chinese_punctuation(item, sentence=False))
        for item in _unique(safe_facts, maximum=3)
    ]
    fragments = [item for item in fragments if item]
    return "；".join(fragments[:2]) + "。" if fragments else ""


def safe_caption_matches_facts(caption: str, safe_facts: Iterable[str]) -> bool:
    """Execute the safe caption matches facts operation."""
    return normalize_chinese_punctuation(caption, sentence=True) == compose_safe_caption(safe_facts)


@dataclass(frozen=True)
class CloseoutPublicBodyDecision:
    """Provide closeout public body decision behavior."""
    text: str
    source: str
    applied: bool


def _valid_public_body(value: Any) -> str:
    text = _text(value)
    if not text or len(text) < 4:
        return ""
    if text[:1] in "[{" or _INTERNAL_RE.search(text):
        return ""
    return normalize_chinese_punctuation(text, sentence=True)


def select_closeout_public_body(
    model_body: Any,
    label: Mapping[str, Any] | None,
    *,
    safe_recovery_body: Any = None,
) -> CloseoutPublicBodyDecision:
    """Apply the frozen public-delivery priority without overriding valid output."""

    valid_model_body = _valid_public_body(model_body)
    if valid_model_body:
        return CloseoutPublicBodyDecision(valid_model_body, "model", False)
    valid_recovery_body = _valid_public_body(safe_recovery_body)
    if valid_recovery_body:
        return CloseoutPublicBodyDecision(
            valid_recovery_body,
            "safe_recovery_model",
            True,
        )
    fallback = dict((label or {}).get("fallback") or {})
    caption = _valid_public_body(fallback.get("safe_caption"))
    if caption and safe_caption_matches_facts(caption, fallback.get("safe_facts") or []):
        return CloseoutPublicBodyDecision(
            caption,
            "canonical_pseudo_label_safe_caption",
            True,
        )
    composed = compose_safe_caption(fallback.get("safe_facts") or [])
    if composed:
        return CloseoutPublicBodyDecision(composed, "composed_safe_facts", True)
    return CloseoutPublicBodyDecision(
        "暂时无法生成可靠的图片说明，请稍后重试。",
        "friendly_failure",
        True,
    )


def _legacy_review(value: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str], bool]:
    legacy = dict(value or {})
    # Absence of a real review record must never be promoted to a visual pass.
    # Historical Phase 6.0 records explicitly carry severity="none" when they
    # were actually inspected and no issue was found; full-batch labels pass
    # ``None`` and therefore remain honestly not_reviewed.
    severity = str(
        legacy.get("severity")
        or ("not_reviewed" if value is None else "none")
    )
    codex_status = {
        "none": "pass",
        "minor": "minor",
        "major": "major",
        "critical": "critical",
    }.get(severity, "not_reviewed")
    warnings: list[str] = []
    if severity == "minor":
        warnings.append("codex_visual_minor")
    elif severity in {"major", "critical"}:
        warnings.append("codex_visual_major")
    return (
        {
            "automatic_validation": {"status": "pass"},
            "codex_visual_review": {
                "status": codex_status,
                "notes": _unique([legacy.get("notes")], maximum=4),
            },
            "user_human_review": {"status": "pending"},
            "review_label": "machine_provisional",
            "gold_status": False,
        },
        warnings,
        severity in {"minor", "major", "critical"},
    )


def _public_values(label: Mapping[str, Any]) -> list[str]:
    """Return fields eligible for ordinary-user delivery.

    ``fallback.do_not_assert`` is an internal safety constraint, not public
    copy. Including it here rejects labels precisely because the guard list
    retains the unverified claims that public fields have already removed.
    """

    display = label.get("display", {})
    facts = label.get("facts", {})
    evidence = label.get("evidence", {})
    fallback = label.get("fallback", {})
    return [
        _text(value)
        for value in [
            display.get("theme"),
            display.get("short_description"),
            *(display.get("micro_tags") or []),
            facts.get("scene"),
            *(facts.get("subjects") or []),
            *(facts.get("actions") or []),
            *(facts.get("attributes") or []),
            *(facts.get("relations") or []),
            *(evidence.get("direct_observations") or []),
            fallback.get("safe_caption"),
            *(fallback.get("safe_facts") or []),
        ]
        if _text(value)
    ]


def migrate_phase6_0a_label(
    old_label: Mapping[str, Any],
    *,
    legacy_review: Mapping[str, Any] | None,
    source_run_id: str,
    recovered_at_tokens: int | None = None,
    migrated_at: str | None = None,
) -> dict[str, Any]:
    """Execute the migrate phase6 0a label operation."""
    old = copy.deepcopy(dict(old_label))
    display = copy.deepcopy(dict(old.get("display") or {}))
    facts = copy.deepcopy(dict(old.get("facts") or {}))
    evidence = copy.deepcopy(dict(old.get("evidence") or {}))
    old_text = copy.deepcopy(dict(old.get("text_evidence") or {}))
    candidates = _candidate_values(old_text)
    retained_candidates, blocked_candidates, _ = candidate_partition(old_text)
    verified_text = _unique(old_text.get("verified_text") or [], maximum=12)
    verified_source = _text(old_text.get("verified_text_source"))
    verified = bool(verified_text and verified_source and verified_source != "not_available")
    conflicted = str(old_text.get("verification_status") or "") == "conflicted"
    if conflicted:
        retained_candidates = []
        blocked_candidates = list(candidates)
    presence = str(old_text.get("presence") or "none")
    verification_status = (
        "conflicted"
        if conflicted
        else "verified"
        if verified
        else "candidate"
        if candidates or presence != "none"
        else "none"
    )
    warnings = ["migrated_from_phase6_0a", "safe_facts_rebuilt", "safe_caption_rebuilt"]
    actions: list[str] = []
    if verification_status != "verified":
        for container, key in (
            (display, "micro_tags"),
            (facts, "subjects"),
            (facts, "actions"),
            (facts, "attributes"),
            (facts, "relations"),
            (evidence, "direct_observations"),
        ):
            cleaned_values = []
            for item in container.get(key, []) or []:
                cleaned, removed = _redact(
                    item,
                    blocked_candidates,
                    retained_candidates,
                    whole=True,
                )
                if removed:
                    actions.append(f"removed_unverified_text:{key}")
                if cleaned:
                    cleaned_values.append(cleaned)
            container[key] = _unique(cleaned_values)
        for key in ("theme", "short_description"):
            cleaned, removed = _redact(
                display.get(key),
                blocked_candidates,
                retained_candidates,
                whole=key == "theme",
            )
            if removed:
                actions.append(f"removed_unverified_text:display.{key}")
            display[key] = cleaned
        if actions:
            warnings.append("text_candidate_redacted")

    facts["subjects"] = _unique(facts.get("subjects") or [], maximum=6)
    facts["actions"] = _unique(facts.get("actions") or [], maximum=5)
    facts["attributes"] = _unique(facts.get("attributes") or [], maximum=6)
    facts["relations"] = _unique(facts.get("relations") or [], maximum=5)
    evidence["cautious_inferences"] = _unique(evidence.get("cautious_inferences") or [], maximum=4)
    evidence["uncertainties"] = _unique(evidence.get("uncertainties") or [], maximum=4)
    do_not_assert = _unique(
        [
            *evidence["cautious_inferences"],
            *evidence["uncertainties"],
            *(
                ["存在清晰度不足或不完整的候选文字"]
                if blocked_candidates and not verified
                else []
            ),
            *(["存在冲突的文字内容"] if conflicted else []),
        ],
        maximum=24,
    )
    direct = [
        normalize_chinese_punctuation(item, sentence=True)
        for item in evidence.get("direct_observations", []) or []
        if _safe_direct(
            item,
            blocked_candidates,
            retained_candidates,
            do_not_assert,
        )
    ]
    direct = _unique(direct, maximum=8)
    if not direct:
        safe_subjects = [
            item
            for item in facts["subjects"]
            if _safe_direct(
                item,
                blocked_candidates,
                retained_candidates,
                do_not_assert,
            )
        ]
        if safe_subjects:
            direct = [
                normalize_chinese_punctuation(
                    "画面中可见" + "和".join(safe_subjects[:2]),
                    sentence=True,
                )
            ]
        else:
            direct = [f"这是一张{_MEDIUM_CN.get(str(facts.get('visual_medium')), '图片')}。"]
    evidence["direct_observations"] = direct
    safe_facts = direct[:3]
    safe_caption = compose_safe_caption(safe_facts)

    if not display.get("theme"):
        display["theme"] = "与".join(facts["subjects"][:2]) or _MEDIUM_CN.get(
            str(facts.get("visual_medium")),
            "图片内容",
        )
    if not display.get("short_description"):
        display["short_description"] = safe_caption
    display["theme"] = normalize_chinese_punctuation(display["theme"], sentence=False).strip("。")
    display["short_description"] = normalize_chinese_punctuation(
        repair_filtered_text(display["short_description"], sentence=False),
        sentence=True,
    )
    display["micro_tags"] = [
        normalize_chinese_punctuation(item, sentence=False).strip("，。；：！？")
        for item in _unique(display.get("micro_tags") or [], maximum=8)
    ]
    facts["scene"] = normalize_chinese_punctuation(facts.get("scene"), sentence=False)
    for key in ("subjects", "actions", "attributes"):
        facts[key] = [
            normalize_chinese_punctuation(item, sentence=False).strip("，。；：！？")
            for item in facts[key]
        ]
    facts["relations"] = [
        normalize_chinese_punctuation(item, sentence=True) for item in facts["relations"]
    ]
    evidence["cautious_inferences"] = [
        normalize_chinese_punctuation(item, sentence=True)
        for item in evidence["cautious_inferences"]
    ]
    evidence["uncertainties"] = [
        normalize_chinese_punctuation(item, sentence=True) for item in evidence["uncertainties"]
    ]
    do_not_assert = [
        normalize_chinese_punctuation(item, sentence=True) for item in do_not_assert
    ]
    warnings.append("punctuation_normalized")
    if len(display["micro_tags"]) < 3:
        warnings.append("micro_tags_below_recommended")
    if verification_status == "conflicted":
        warnings.append("text_evidence_conflicted")
    if recovered_at_tokens == 768:
        warnings.append("recovered_at_768_tokens")
    elif recovered_at_tokens is not None and recovered_at_tokens >= 1024:
        warnings.append("recovered_at_extended_tokens")
    elif recovered_at_tokens is not None:
        warnings.append("recovered_with_minimal_contract")

    review, review_warnings, review_needs = _legacy_review(legacy_review)
    warnings.extend(review_warnings)
    needs_review = review_needs or verification_status == "conflicted"
    semantic_status = "needs_review" if needs_review else "warning" if warnings else "pass"
    review["automatic_validation"]["status"] = "warning" if warnings else "pass"
    asset_sha = str(old.get("asset", {}).get("sha256") or "").lower()
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "label_id": f"cpl_{sha256_text(f'{asset_sha}:{SCHEMA_VERSION}')[:24]}",
        "canonical_status": "active_candidate",
        "supersedes": [str(old.get("label_id"))] if old.get("label_id") else [],
        "asset": copy.deepcopy(old.get("asset") or {}),
        "display": display,
        "facts": facts,
        "evidence": evidence,
        "text_evidence": {
            "presence": presence,
            "visual_candidates": _unique(old_text.get("visual_candidates") or [], maximum=12),
            "ocr_candidates": _unique(old_text.get("ocr_candidates") or [], maximum=12),
            "candidate_text": candidates,
            "confidence": old_text.get("confidence"),
            "verification_status": verification_status,
            "verified_text": verified_text if verified else [],
            "verified_text_source": verified_source if verified else "not_available",
        },
        "fallback": {
            "safe_caption": safe_caption,
            "safe_facts": safe_facts,
            "do_not_assert": do_not_assert,
            "caption_composer": CAPTION_COMPOSER_VERSION,
            "policy": "use_only_when_public_model_body_invalid",
            "must_not_override_valid_model_output": True,
        },
        "quality": {
            "semantic_status": semantic_status,
            "schema_valid": True,
            "warning_codes": sorted(set(warnings)),
            "needs_review": needs_review,
            "validation_errors": [],
            "sanitization_applied": bool(actions),
            "sanitization_actions": sorted(set(actions)),
        },
        "review": review,
        "provenance": {
            **copy.deepcopy(old.get("provenance") or {}),
            "source_run_id": source_run_id,
            "migrated_at": migrated_at or datetime.now(timezone.utc).isoformat(),
            "migration_version": MIGRATION_VERSION,
            "validator": VALIDATOR_VERSION,
            "source_schema_version": str(old.get("schema_version") or ""),
            "source_label_id": str(old.get("label_id") or ""),
            "recovered_at_tokens": recovered_at_tokens,
        },
    }
    errors = validate_closeout_label(canonical, verify_hash=False)
    canonical["quality"]["validation_errors"] = errors
    canonical["quality"]["schema_valid"] = not errors
    if errors:
        canonical["quality"]["semantic_status"] = "failed"
        canonical["quality"]["needs_review"] = True
        canonical["canonical_status"] = "inactive"
        canonical["review"]["automatic_validation"]["status"] = "failed"
    canonical["canonical_sha256"] = sha256_text(canonical_json(canonical))
    return canonical


def validate_closeout_label(
    label: Mapping[str, Any],
    *,
    verify_hash: bool = True,
    expected_schema_version: str = SCHEMA_VERSION,
) -> list[str]:
    """Validate closeout label."""
    errors: list[str] = []
    if label.get("schema_version") != expected_schema_version:
        errors.append("schema_version_mismatch")
    fallback = label.get("fallback", {})
    safe_facts = fallback.get("safe_facts", [])
    if not safe_facts:
        errors.append("safe_facts_empty")
    if not safe_caption_matches_facts(str(fallback.get("safe_caption") or ""), safe_facts):
        errors.append("safe_caption_not_deterministic_from_safe_facts")
    text_evidence = label.get("text_evidence", {})
    candidates = _candidate_values(text_evidence)
    retained_candidates, blocked_candidates, _ = candidate_partition(text_evidence)
    verification = str(text_evidence.get("verification_status") or "")
    if verification != "verified" and any(
        _mentions_candidate(value, blocked_candidates)
        or (
            _SPECIFIC_TEXT_RE.search(value)
            and not _mentions_candidate(value, retained_candidates)
        )
        for value in _public_values(label)
    ):
        errors.append("unverified_text_in_public_field")
    if any(is_qualified_statement(value) for value in _public_values(label)):
        errors.append("uncertain_text_in_canonical_public_field")
    if verification == "verified" and (
        not text_evidence.get("verified_text")
        or text_evidence.get("verified_text_source") in {None, "", "not_available"}
    ):
        errors.append("verified_text_without_source")
    if verification == "conflicted" and not label.get("quality", {}).get("needs_review"):
        errors.append("conflicted_text_without_review")
    if any(_INTERNAL_RE.search(value) for value in _public_values(label)):
        errors.append("internal_term_in_public_field")
    if any(_UNCERTAIN_RE.search(str(value)) for value in safe_facts):
        errors.append("uncertain_statement_in_safe_facts")
    review = label.get("review", {})
    if review.get("user_human_review", {}).get("status") != "pending":
        errors.append("user_human_review_not_pending")
    if review.get("gold_status") is not False:
        errors.append("gold_status_not_false")
    if review.get("review_label") != "machine_provisional":
        errors.append("review_label_not_machine_provisional")
    if fallback.get("must_not_override_valid_model_output") is not True:
        errors.append("fallback_override_policy_invalid")
    if verify_hash:
        expected = str(label.get("canonical_sha256") or "")
        value = copy.deepcopy(dict(label))
        value.pop("canonical_sha256", None)
        if expected != sha256_text(canonical_json(value)):
            errors.append("canonical_sha256_mismatch")
    return sorted(set(errors))


def is_truncation_failure(
    *,
    raw_output: str,
    finish_reason: str | None,
    error: str | None,
) -> bool:
    """Execute the is truncation failure operation."""
    reason = str(finish_reason or "").casefold()
    if reason in {"length", "max_new_tokens", "max_tokens"}:
        return True
    raw = str(raw_output or "")
    if raw.count("```") % 2 == 1:
        return True
    if raw and (raw.count("{") > raw.count("}") or raw.count("[") > raw.count("]")):
        return True
    return bool(
        "model_output_did_not_contain_json_object" in str(error or "")
        and raw.rstrip().endswith((",", ":", "\\", '"'))
    )


def public_failure_display() -> dict[str, Any]:
    """Execute the public failure display operation."""
    return {
        "categories": ["待复核"],
        "default": {
            "主题": "暂未生成标注",
            "简短描述": "该图片的自动标注尚未完成，等待重新处理或人工复核。",
            "微标签": [],
            "当前状态": "需要复核",
        },
        "details": {"质量状态": "需要复核", "用户人工审核": "待审核"},
        "developer": {},
        "is_active_canonical": False,
    }


def _display(value: Any) -> Any:
    if value is None:
        return "未提供"
    if isinstance(value, list):
        return value if value else "暂无"
    if isinstance(value, bool):
        return "是" if value else "否"
    text = _text(value)
    return STATUS_DISPLAY_NAMES.get(text, text or "暂无")


def derive_public_category_chips(
    label: Mapping[str, Any] | None,
) -> list[str]:
    """Project frozen semantics into the Phase 6 public browsing taxonomy.

    This is a read-only display projection. It derives labels from the frozen
    Canonical payload instead of persisting a parallel category field or
    changing the Canonical Schema.
    """

    if not label:
        return ["待复核"]
    facts = label.get("facts", {})
    display = label.get("display", {})
    text_evidence = label.get("text_evidence", {})
    quality = label.get("quality", {})
    medium = str(facts.get("visual_medium") or "")
    values = [
        str(facts.get("scene") or ""),
        str(display.get("theme") or ""),
        str(display.get("short_description") or ""),
        *[str(item) for item in facts.get("subjects", []) or []],
        *[str(item) for item in facts.get("actions", []) or []],
        *[str(item) for item in facts.get("attributes", []) or []],
        *[str(item) for item in facts.get("relations", []) or []],
        *[str(item) for item in display.get("micro_tags", []) or []],
    ]
    corpus = " ".join(values).casefold()
    subjects = _unique(facts.get("subjects") or [])
    actions = _unique(facts.get("actions") or [])
    relations = _unique(facts.get("relations") or [])
    categories: list[str] = []

    def add(value: str) -> None:
        if value not in categories:
            categories.append(value)

    expression_tokens = (
        "表情",
        "呆萌",
        "微笑",
        "笑容",
        "哭泣",
        "愤怒",
        "惊讶",
        "困惑",
        "开心",
        "悲伤",
    )
    has_expression = any(token in corpus for token in expression_tokens)
    is_meme = any(
        token in corpus for token in ("表情包", "梗图", "斗图", "meme")
    ) or (
        medium == "illustration"
        and has_expression
        and any(token in corpus for token in ("问号", "感叹号", "墨镜", "卡通"))
    )
    is_presentation = (
        medium in {"screenshot", "poster_document", "mixed"}
        and any(
            token in corpus
            for token in ("演示文稿", "幻灯片", "教学幻灯片", "课件", "ppt")
        )
    )
    candidates = [
        *list(text_evidence.get("visual_candidates") or []),
        *list(text_evidence.get("ocr_candidates") or []),
        *list(text_evidence.get("candidate_text") or []),
        *list(text_evidence.get("verified_text") or []),
    ]
    is_text_dense = len(candidates) >= 4 or any(
        token in corpus
        for token in ("文字密集", "大段文字", "文档排版", "正文段落", "表格")
    )
    is_pure_text = "纯文字" in corpus or (
        medium == "poster_document"
        and is_text_dense
        and subjects
        and all(
            any(
                token in subject.casefold()
                for token in (
                    "文字",
                    "文本",
                    "段落",
                    "标题",
                    "列表",
                    "表格",
                    "文档",
                    "页面",
                )
            )
            for subject in subjects
        )
    )

    if is_meme:
        add("表情包")
    if is_presentation:
        add("PPT")
    if is_pure_text:
        add("纯文字图片")
    if any(
        token in corpus
        for token in (
            "人物",
            "人群",
            "男人",
            "女人",
            "男性",
            "女性",
            "儿童",
            "学生",
            "讲师",
            "行人",
            "人像",
            "person",
        )
    ):
        add("人物")
    if any(
        token in corpus
        for token in (
            "动物",
            "猫",
            "狗",
            "鸟",
            "熊",
            "马",
            "animal",
        )
    ):
        add("动物")
    if any(
        token in corpus
        for token in (
            "风景",
            "山",
            "湖",
            "海",
            "树林",
            "天空",
            "自然",
        )
    ):
        add("风景")
    if has_expression:
        add("表情类")
    if (
        len(subjects) >= 3
        and len(subjects) + len(actions) + len(relations) >= 6
    ) or any(
        token in corpus for token in ("复杂场景", "多人场景", "拥挤场景", "多主体")
    ):
        add("复杂场景")
    if any(
        token in corpus
        for token in ("建筑", "大楼", "房屋", "街景", "桥", "塔")
    ):
        add("建筑")
    if any(
        token in corpus
        for token in ("商品", "包装", "产品", "瓶", "盒", "品牌")
    ):
        add("商品")
    if is_text_dense and not is_pure_text:
        add("文字密集")
    if any(token in corpus for token in ("手写", "笔记", "签名")):
        add("手写")
    warning_corpus = " ".join(
        str(value) for value in quality.get("warning_codes", []) or []
    ).casefold()
    if any(
        token in f"{corpus} {warning_corpus}"
        for token in (
            "blur",
            "low_quality",
            "decode",
            "模糊",
            "低质量",
            "低清",
            "失焦",
            "噪点",
            "像素化",
            "过曝",
            "曝光不足",
        )
    ):
        add("低质量图片")

    if medium == "photograph":
        add("普通图片")
    elif medium == "screenshot" and not is_presentation:
        add("屏幕图")
    elif medium == "illustration" and not is_meme:
        add("插画")
    elif medium == "poster_document" and not is_pure_text:
        add("文字图片")
    elif medium == "product_packaging":
        add("产品图片")
    elif medium == "mixed":
        add("混合媒介")

    if (
        text_evidence.get("presence") == "none"
        and not candidates
    ):
        add("无文字图片")
    if quality.get("needs_review"):
        add("待复核")
    return categories[:8] or ["普通图片"]


def build_two_layer_display(
    label: Mapping[str, Any] | None,
    *,
    developer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build two layer display."""
    if label is None:
        result = public_failure_display()
        result["developer"] = dict(developer or {})
        return result
    display = label.get("display", {})
    facts = label.get("facts", {})
    evidence = label.get("evidence", {})
    text = label.get("text_evidence", {})
    quality = label.get("quality", {})
    review = label.get("review", {})
    candidates = _unique(text.get("candidate_text") or [])
    projected_description = str(display.get("short_description") or "")
    projection_actions: list[str] = []
    retained_candidates, _, _ = candidate_partition(text)
    if re.search(r"(?:描绘|展示|包括|包含|涉及)了\s*[、，]\s*等", projected_description):
        module_names = [
            item
            for item in retained_candidates
            if len(item) >= 3 and not re.fullmatch(r"[A-Za-z_]+\d+", item)
        ][:3]
        if module_names:
            projected_description = re.sub(
                r"((?:描绘|展示|包括|包含|涉及)了)\s*[、，]\s*(等)",
                lambda match: (
                    match.group(1)
                    + "、".join(module_names)
                    + match.group(2)
                ),
                projected_description,
                count=1,
            )
            projection_actions.append("restored_clear_module_names")
    if re.search(
        r"[（(]\s*(?:标记|标注)为\s*[-–—、，和至到0-9\s]+\s*[）)]",
        projected_description,
    ):
        compact_candidates = prioritize_candidate_values(
            retained_candidates,
            maximum_items=6,
            maximum_length=40,
        )
        compact_ranges = [
            item for item in compact_candidates if "–" in item or "-" in item
        ]
        replacement = (
            f"（标记为 {' 和 '.join(compact_ranges)}）"
            if compact_ranges
            else ""
        )
        projected_description = re.sub(
            r"[（(]\s*(?:标记|标注)为\s*[-–—、，和至到0-9\s]+\s*[）)]",
            replacement,
            projected_description,
            count=1,
        )
        projection_actions.append("repaired_broken_identifier_enumeration")
    repaired_description = repair_filtered_text(
        projected_description,
        sentence=True,
    )
    if repaired_description != str(display.get("short_description") or ""):
        projection_actions.append("post_filter_grammar_repaired")
    text_summary = (
        "尚未验证：" + "；".join(candidates)
        if candidates and text.get("verification_status") != "verified"
        else "已验证：" + "；".join(text.get("verified_text") or [])
        if text.get("verification_status") == "verified"
        else "暂无"
    )
    warnings = _unique(
        WARNING_DISPLAY_NAMES.get(str(code), "存在一项内部质量提醒")
        for code in quality.get("warning_codes", [])
    )
    details = {
        "主要主体": _display(facts.get("subjects")),
        "动作": _display(facts.get("actions")),
        "显著属性": _display(facts.get("attributes")),
        "空间关系": _display(facts.get("relations")),
        "直接观察": _display(evidence.get("direct_observations")),
        "谨慎推断": _display(evidence.get("cautious_inferences")),
        "文字证据": text_summary,
        "不确定信息": _display(evidence.get("uncertainties")),
        "OCR状态": "存在OCR候选" if text.get("ocr_candidates") else "未接入独立OCR核验",
        "文字验证状态": _display(text.get("verification_status")),
        "质量状态": _display(quality.get("semantic_status")),
        "质量提醒": warnings or "暂无",
        "自动检查": _display(review.get("automatic_validation", {}).get("status")),
        "机器视觉复核": _display(review.get("codex_visual_review", {}).get("status")),
        "用户人工审核": _display(review.get("user_human_review", {}).get("status")),
        "标签状态": _display(review.get("review_label")),
    }
    developer_info = {
        "asset_id": label.get("asset", {}).get("asset_id"),
        "sha256": label.get("asset", {}).get("sha256"),
        "schema_version": label.get("schema_version"),
        "prompt_id": label.get("provenance", {}).get("prompt_id"),
        "prompt_sha256": label.get("provenance", {}).get("prompt_sha256"),
        "model": label.get("provenance", {}).get("model"),
        "model_revision": label.get("provenance", {}).get("model_revision"),
        "warning_codes": quality.get("warning_codes", []),
        "task_text_projection_actions": sorted(set(projection_actions)),
        **dict(developer or {}),
    }
    return {
        "categories": derive_public_category_chips(label),
        "default": {
            "主题": _display(display.get("theme")),
            "简短描述": _display(repaired_description),
            "微标签": display.get("micro_tags") or [],
            "当前状态": _display(quality.get("semantic_status")),
        },
        "details": details,
        "developer": developer_info,
        "is_active_canonical": label.get("canonical_status") == "active_candidate",
    }
