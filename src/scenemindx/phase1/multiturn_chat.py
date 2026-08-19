"""Phase 5.2-A single-thread multimodal multi-turn chat contracts.

This module deliberately owns only the Chat path.  It keeps stable public
``IMG_n`` labels separate from private asset identities, resolves references
deterministically, builds bounded conversation context, and validates model
output before backend asset metadata is attached.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from scenemindx.task_text_policy import (
    CHAT,
    candidate_is_allowed,
    collect_text_candidates,
    is_qualified_statement,
)
from .conversational_response import clean_public_answer, infer_current_turn_state


CHAT_CANDIDATE_ID = "SCENEMINDX_MULTITURN_CHAT_V2_CANDIDATE"
CHAT_PROMPT_ID = "phase5_2a_multiturn_chat_v2"
_LABEL_PATTERN = re.compile(
    r"(?<![A-Z0-9_])IMG_([1-9]\d*)(?!\d)",
    re.IGNORECASE,
)
_ORDINALS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
_LOCAL_REFERENCE_PATTERNS = (
    (
        re.compile(
            r"第\s*([一二两三四五]|[1-9]\d*)\s*(?:张|幅)(?:\s*(?:图|图片))?"
        ),
        "explicit_ordinal",
    ),
    (
        re.compile(
            r"(?:图|图片)\s*([一二两三四五]|[1-9]\d*)",
            re.IGNORECASE,
        ),
        "explicit_numbered_image",
    ),
    (
        re.compile(r"([一二两三四五])\s*号(?:\s*(?:图|图片))?"),
        "explicit_numbered_image",
    ),
    (
        re.compile(
            r"(?<![A-Z])image\s*([1-9]\d*)(?!\d)",
            re.IGNORECASE,
        ),
        "explicit_numbered_image",
    ),
)
_COLLECTION_REFERENCE_TOKENS = (
    "三张图",
    "三张图片",
    "四张图",
    "四张图片",
    "五张图",
    "五张图片",
    "这几张图",
    "这几张图片",
    "这几张",
    "这些图",
    "这些图片",
    "这些",
    "这几个",
    "全部图片",
    "全部图",
    "全部",
    "全都",
    "所有图",
    "所有图片",
    "当前所有图",
    "当前所有图片",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MultiturnChatPromptCandidate:
    """Load and hash-check the bounded Phase 5.2-A Chat-only candidate."""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root / "prompts" / "phase5_2" / "multiturn_chat_v2_candidate"
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("candidate_id") != CHAT_CANDIDATE_ID:
            raise ValueError("phase5_2a_chat_candidate_identity_mismatch")
        prompt_spec = dict(self.manifest["prompt"])
        self.prompt_path = project_root / str(prompt_spec["file"])
        actual = _sha256(self.prompt_path)
        if actual != str(prompt_spec["raw_sha256"]):
            raise ValueError("phase5_2a_chat_candidate_sha256_mismatch")
        self.text = self.prompt_path.read_text(encoding="utf-8")

    def identity(self) -> dict[str, str]:
        """Execute the identity operation."""
        return {
            "candidate_id": CHAT_CANDIDATE_ID,
            "prompt_id": CHAT_PROMPT_ID,
            "prompt_sha256": str(self.manifest["prompt"]["raw_sha256"]),
            "status": str(self.manifest["status"]),
            "iteration": str(self.manifest["iteration"]),
        }

    def render(self, values: dict[str, Any]) -> tuple[str, dict[str, str]]:
        """Render the requested value."""
        text = self.text
        for key, value in values.items():
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
            text = text.replace("{{" + key + "}}", rendered)
        unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text)))
        if unresolved:
            raise ValueError(f"phase5_2a_unresolved_prompt_placeholders:{','.join(unresolved)}")
        return text, self.identity()


def _bindings(session: dict[str, Any], *, active_only: bool = False) -> list[dict[str, Any]]:
    chat_state = session.get("chat_state") if isinstance(session.get("chat_state"), dict) else {}
    values = chat_state.get("asset_bindings") if isinstance(chat_state.get("asset_bindings"), list) else []
    result = [dict(item) for item in values if isinstance(item, dict) and item.get("image_label")]
    if active_only:
        result = [item for item in result if item.get("status") == "active"]
    return sorted(result, key=lambda item: int(str(item["image_label"]).split("_")[-1]))


def _label_for_ref(bindings: Iterable[dict[str, Any]]) -> dict[str, str]:
    return {str(item.get("ref")): str(item["image_label"]) for item in bindings if item.get("ref")}


def local_image_reference_matches(text: str) -> list[dict[str, Any]]:
    """Return canonical conversation-local ``IMG_n`` references in text order."""

    matches: list[dict[str, Any]] = []
    occupied: set[tuple[int, int, str]] = set()
    for match in _LABEL_PATTERN.finditer(text):
        label = f"IMG_{int(match.group(1))}"
        key = (match.start(), match.end(), label)
        occupied.add(key)
        matches.append(
            {
                "source": match.group(0),
                "label": label,
                "reason": "explicit_img_label",
                "start": match.start(),
                "end": match.end(),
            }
        )
    for pattern, reason in _LOCAL_REFERENCE_PATTERNS:
        for match in pattern.finditer(text):
            raw_index = match.group(1)
            index = (
                _ORDINALS[raw_index]
                if raw_index in _ORDINALS
                else int(raw_index)
            )
            label = f"IMG_{index}"
            key = (match.start(), match.end(), label)
            if key in occupied:
                continue
            occupied.add(key)
            matches.append(
                {
                    "source": match.group(0),
                    "label": label,
                    "reason": reason,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return sorted(matches, key=lambda item: (item["start"], item["end"]))


def _semantic_reference_descriptor(text: str) -> str:
    match = re.search(
        r"(?:^|[，。！？、\s])([^，。！？、\s]{1,16})(?:那张|那幅)(?:图|图片)?",
        text,
    )
    if not match:
        return ""
    descriptor = match.group(1)
    for prefix in (
        "请帮我看看",
        "帮我看看",
        "请描述一下",
        "描述一下",
        "请看看",
        "看看",
        "请说说",
        "说说",
        "关于",
    ):
        if descriptor.startswith(prefix):
            descriptor = descriptor[len(prefix) :]
            break
    return descriptor.strip()


def _semantic_aliases(descriptor: str) -> set[str]:
    aliases = {descriptor}
    if descriptor.endswith("咪") and len(descriptor) >= 2:
        aliases.update({descriptor[:-1], descriptor[:-1] + "猫"})
    if descriptor in {"猫咪", "猫猫"}:
        aliases.update({"猫", "猫咪", "猫猫"})
    if descriptor in {"狗头", "狗头人"}:
        aliases.update({"狗头", "狗头面具"})
    return {item for item in aliases if item}


def _semantic_binding_score(
    binding: dict[str, Any],
    descriptor: str,
) -> tuple[int, list[str]]:
    facts = binding.get("facts") if isinstance(binding.get("facts"), dict) else {}
    aliases = _semantic_aliases(descriptor)
    weighted_fields = (
        ("global_observation", 4),
        ("global_scene", 4),
        ("subjects", 4),
        ("main_subjects", 4),
        ("attributes", 2),
        ("direct_observations", 2),
        ("activities", 1),
        ("relations", 1),
        ("evidence_descriptions", 1),
    )
    score = 0
    evidence_fields: list[str] = []
    for field, weight in weighted_fields:
        value = facts.get(field)
        if isinstance(value, list):
            compact = " ".join(str(item) for item in value)
        elif isinstance(value, str):
            compact = value
        else:
            continue
        if any(alias in compact for alias in aliases):
            score += weight
            evidence_fields.append(field)
    return score, evidence_fields


def _resolve_semantic_reference(
    text: str,
    active_bindings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    descriptor = _semantic_reference_descriptor(text)
    if not descriptor:
        return None
    candidates = []
    for binding in active_bindings:
        score, fields = _semantic_binding_score(binding, descriptor)
        if score:
            candidates.append(
                {
                    "label": str(binding["image_label"]),
                    "score": score,
                    "evidence_fields": fields,
                }
            )
    candidates.sort(key=lambda item: (-item["score"], item["label"]))
    if not candidates:
        return {
            "descriptor": descriptor,
            "status": "not_found",
            "candidates": [],
        }
    top = candidates[0]
    runner_up = candidates[1]["score"] if len(candidates) > 1 else 0
    # Require corroboration across at least two trusted fact fields. This
    # prevents one isolated pseudo-annotation token from forcing a binding.
    trusted = {
        "global_observation",
        "global_scene",
        "subjects",
        "main_subjects",
        "attributes",
        "direct_observations",
    }
    corroborated_fields = trusted.intersection(top["evidence_fields"])
    if (
        top["score"] >= 6
        and len(corroborated_fields) >= 2
        and top["score"] - runner_up >= 2
    ):
        return {
            "descriptor": descriptor,
            "status": "resolved",
            "label": top["label"],
            "score": top["score"],
            "evidence_fields": sorted(corroborated_fields),
            "candidates": candidates,
        }
    return {
        "descriptor": descriptor,
        "status": "ambiguous",
        "candidates": candidates,
    }


def resolve_image_references(question: str, session: dict[str, Any]) -> dict[str, Any]:
    """Resolve Chinese conversational references against stable ``IMG_n`` labels."""

    raw = question.strip()
    all_bindings = _bindings(session)
    active_bindings = _bindings(session, active_only=True)
    by_label = {str(item["image_label"]): item for item in all_bindings}
    active_labels = [str(item["image_label"]) for item in active_bindings]
    active_set = set(active_labels)
    chat_state = session.get("chat_state") if isinstance(session.get("chat_state"), dict) else {}
    focus = str(chat_state.get("current_focus_label") or "")
    if focus not in active_set:
        focus = ""
    focus_before_resolution = focus or None
    newly_added = [
        str(label)
        for label in chat_state.get("newly_added_labels", [])
        if str(label) in active_set
    ]
    recent = [
        str(label)
        for label in chat_state.get("recent_mentioned_labels", [])
        if str(label) in active_set
    ]
    selected: list[str] = []
    reasons: dict[str, list[str]] = {}
    errors: list[str] = []
    substitutions: list[tuple[str, str]] = []
    semantic_resolution: dict[str, Any] | None = None

    def add(label: str, reason: str) -> None:
        normalized = label.upper()
        binding = by_label.get(normalized)
        if binding is None:
            errors.append(f"unknown_image_label:{normalized}")
            return
        if normalized not in active_set:
            errors.append(f"removed_image_label:{normalized}")
            return
        if normalized not in selected:
            selected.append(normalized)
        reasons.setdefault(normalized, []).append(reason)

    for match in local_image_reference_matches(raw):
        add(match["label"], match["reason"])
        substitutions.append((match["source"], match["label"]))

    if "最后一张" in raw and active_labels:
        label = active_labels[-1]
        add(label, "last_active_image")
        substitutions.append(("最后一张", label))
    if any(token in raw for token in ("左边那张", "最左边", "左侧那张")) and active_labels:
        add(active_labels[0], "leftmost_active_image")
        for token in ("左边那张", "最左边", "左侧那张"):
            substitutions.append((token, active_labels[0]))
    if any(token in raw for token in ("右边那张", "最右边", "右侧那张")) and active_labels:
        add(active_labels[-1], "rightmost_active_image")
        for token in ("右边那张", "最右边", "右侧那张"):
            substitutions.append((token, active_labels[-1]))

    newest = newly_added[-1] if newly_added else (active_labels[-1] if active_labels else "")
    if any(token in raw for token in ("新加入", "刚加入", "新添加", "刚添加")) and newest:
        add(newest, "newly_added_image")
        for token in ("新加入的图片", "刚加入的图片", "新添加的图片", "刚添加的图片"):
            substitutions.append((token, newest))

    explicit_reference_present = any(
        reason in {
            "explicit_img_label",
            "explicit_ordinal",
            "explicit_numbered_image",
        }
        for values in reasons.values()
        for reason in values
    )
    if not explicit_reference_present:
        semantic_resolution = _resolve_semantic_reference(
            raw,
            active_bindings,
        )
        if semantic_resolution:
            if semantic_resolution["status"] == "resolved":
                label = str(semantic_resolution["label"])
                add(label, "semantic_reference_high_confidence")
                substitutions.append(
                    (
                        f"{semantic_resolution['descriptor']}那张图",
                        label,
                    )
                )
                explicit_reference_present = True
            elif semantic_resolution["status"] == "not_found":
                errors.append(
                    "semantic_reference_not_found:"
                    + str(semantic_resolution["descriptor"])
                )
            else:
                errors.append(
                    "ambiguous_semantic_reference:"
                    + str(semantic_resolution["descriptor"])
                )
    compact = re.sub(r"\s+", "", raw)
    compact_enumerated_set = bool(
        re.search(r"(?:图[一二两三四五12345]){2,}", compact)
    )
    counted_collection_match = re.search(
        r"(?:这|当前)?([三四五345])张(?:图|图片)?",
        compact,
    )
    counted_collection = False
    if counted_collection_match:
        requested_count = _ORDINALS[
            counted_collection_match.group(1)
        ]
        if len(active_labels) == requested_count:
            counted_collection = True
        else:
            errors.append(
                "counted_collection_scope_mismatch:"
                f"{requested_count}:{len(active_labels)}"
            )
    all_request = any(
        token in raw
        for token in (
            *_COLLECTION_REFERENCE_TOKENS,
            "每张图",
            "逐张",
            "分别说明",
        )
    ) or compact_enumerated_set or counted_collection
    decision_all_request = any(
        token in raw
        for token in (
            "哪张",
            "哪一个",
            "更适合",
            "更推荐",
            "选一张",
            "只选一",
            "排序",
            "排名",
            "从好到差",
            "从高到低",
        )
    )
    pair_request = any(token in raw for token in ("这两张", "两张图", "两张图片"))
    plural_pronoun = any(token in raw for token in ("它们", "前面几张"))
    explicit_image_scope: str | None = None
    scope_resolution_reason: str | None = None
    if explicit_reference_present:
        explicit_image_scope = (
            "explicit_single"
            if len(selected) == 1
            else "explicit_multiple"
        )
        scope_resolution_reason = "explicit_image_reference"
    if not explicit_reference_present and (all_request or decision_all_request):
        for label in active_labels:
            add(label, "all_active_images_for_current_task")
        explicit_image_scope = "explicit_collection"
        scope_resolution_reason = (
            "explicit_collection_all_active"
            if all_request
            else "decision_requires_all_active"
        )
    elif not explicit_reference_present and pair_request:
        pair = recent[-2:] if len(recent) >= 2 else active_labels if len(active_labels) == 2 else []
        if len(pair) != 2:
            errors.append("ambiguous_pair_reference")
        else:
            for label in pair:
                add(label, "two_image_reference")
            substitutions.append(("这两张", " 与 ".join(pair)))
    elif not explicit_reference_present and plural_pronoun:
        plural = recent if len(recent) >= 2 else active_labels if len(active_labels) <= 2 else []
        if len(plural) < 2:
            errors.append("ambiguous_plural_reference")
        else:
            for label in plural:
                add(label, "plural_pronoun")
            substitutions.append(("它们", "、".join(plural)))

    singular_pronoun = re.search(r"(?<!它)它(?!们)|这张|当前图片|当前这张", raw) is not None
    anchor = ""
    if selected:
        anchor = selected[-1]
    elif newest and len(newly_added) == 1:
        anchor = newest
    elif focus:
        anchor = focus
    elif recent:
        anchor = recent[-1]
    elif len(active_labels) == 1:
        anchor = active_labels[0]

    if singular_pronoun:
        if not anchor:
            errors.append("ambiguous_singular_reference")
        else:
            add(anchor, "singular_pronoun")
            for token in ("当前这张", "当前图片", "这张", "它"):
                substitutions.append((token, anchor))

    if any(token in raw for token in ("前一张", "上一张", "之前那张", "前一幅")):
        if not anchor:
            errors.append("previous_image_without_anchor")
        else:
            anchor_index = active_labels.index(anchor)
            if anchor_index == 0:
                errors.append("previous_image_not_available")
            else:
                previous = active_labels[anchor_index - 1]
                add(previous, "previous_relative_to_anchor")
                for token in ("前一张", "上一张", "之前那张", "前一幅"):
                    substitutions.append((token, previous))

    if not selected and not errors:
        if len(newly_added) == 1:
            add(newly_added[0], "newly_added_default")
            scope_resolution_reason = "single_newly_added_image"
        elif focus:
            add(focus, "current_focus")
            scope_resolution_reason = "current_focus"
        elif recent:
            add(recent[-1], "recently_used_image")
            scope_resolution_reason = "recently_used_image"
        elif len(active_labels) == 1:
            add(active_labels[0], "single_active_image")
            scope_resolution_reason = "single_active_image"
        elif not active_labels:
            errors.append("no_active_image")
        else:
            errors.append("ambiguous_unqualified_reference")

    resolved = raw
    for source, target in sorted(set(substitutions), key=lambda item: len(item[0]), reverse=True):
        resolved = resolved.replace(source, target)
    for match in _LABEL_PATTERN.finditer(resolved):
        resolved = resolved.replace(match.group(0), f"IMG_{int(match.group(1))}")

    clarification = None
    if errors:
        removed = next((item.split(":", 1)[1] for item in errors if item.startswith("removed_image_label:")), None)
        unknown = next((item.split(":", 1)[1] for item in errors if item.startswith("unknown_image_label:")), None)
        if removed:
            clarification = f"当前会话中不存在 {removed} 的有效图片；它已从当前会话移除，不能继续作为“第几张”引用。请重新加入图片或改用当前仍在会话中的 IMG 标签。"
        elif unknown:
            clarification = f"当前会话中不存在 {unknown}。请使用界面显示的有效 IMG 标签。"
        elif "no_active_image" in errors:
            clarification = "当前会话没有可用图片，请先加入 1–5 张图片。"
        elif next(
            (
                item
                for item in errors
                if item.startswith("semantic_reference_not_found:")
            ),
            None,
        ):
            descriptor = next(
                item.split(":", 1)[1]
                for item in errors
                if item.startswith("semantic_reference_not_found:")
            )
            clarification = (
                f"我还不能可靠确定“{descriptor}”指的是哪张图。"
                "请用当前会话中的 IMG_n 指定一次。"
            )
        elif next(
            (
                item
                for item in errors
                if item.startswith("ambiguous_semantic_reference:")
            ),
            None,
        ):
            descriptor = next(
                item.split(":", 1)[1]
                for item in errors
                if item.startswith("ambiguous_semantic_reference:")
            )
            clarification = (
                f"当前有不止一张图片可能符合“{descriptor}”。"
                "请用 IMG_n 指定一次。"
            )
        else:
            clarification = "当前有多张图片，但指代不明确。请指定 IMG_n，或先在图片条中设置当前焦点。"

    focus_applied = bool(
        focus
        and focus in selected
        and any(
            reason in {"current_focus", "singular_pronoun"}
            for reason in reasons.get(focus, [])
        )
    )
    if scope_resolution_reason is None and selected:
        scope_resolution_reason = next(
            iter(reasons.get(selected[0], [])),
            "resolved_by_current_turn",
        )
    return {
        "raw_question": raw,
        "original_user_message": raw,
        "resolved_question": resolved,
        "selected_image_labels": selected if not errors else [],
        "resolved_image_refs": selected if not errors else [],
        "selection_reasons": reasons,
        "explicit_image_scope": explicit_image_scope,
        "focus_before_resolution": focus_before_resolution,
        "focus_applied": focus_applied,
        "scope_resolution_reason": scope_resolution_reason,
        "requires_clarification": bool(errors),
        "resolution_errors": errors,
        "clarification": clarification,
        "active_image_labels": active_labels,
        "current_focus_label": focus or None,
        "newly_added_labels": newly_added,
        "semantic_reference_resolution": semantic_resolution,
    }


_DECISION_FOLLOW_UP_TYPES = {
    "compare",
    "select",
    "select_one",
    "select_top_k",
    "top_k",
    "rank",
    "rank_all",
    "recommendation",
    "recommend",
}
_DECISION_EXPLAIN_TOKENS = (
    "为什么",
    "理由是什么",
    "什么原因",
    "具体说说原因",
    "怎么说",
    "为什么这么判断",
    "为什么这样判断",
)
_DECISION_EXPAND_TOKENS = (
    "再详细一点",
    "再具体一点",
    "展开说说",
    "详细说说",
    "具体讲讲",
)
_DECISION_REJECTED_TOKENS = (
    "其他两张为什么不选",
    "另外两张为什么不选",
    "其余两张为什么不选",
    "其他的为什么不选",
    "为什么不选其他",
)


def resolve_decision_follow_up(
    question: str,
    session: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind a short discourse follow-up only to the immediately prior decision."""

    raw = question.strip()
    compact = re.sub(r"\s+", "", raw)
    explicit_matches = local_image_reference_matches(raw)
    chat_state = (
        session.get("chat_state")
        if isinstance(session.get("chat_state"), dict)
        else {}
    )
    last_type = str(chat_state.get("last_decision_type") or "")
    last_turn_id = str(chat_state.get("last_decision_turn_id") or "")
    messages = list(session.get("messages", []))
    latest_assistant = next(
        (
            item
            for item in reversed(messages)
            if item.get("role") == "assistant"
        ),
        None,
    )
    immediate = bool(
        latest_assistant
        and str(latest_assistant.get("message_id") or "") == last_turn_id
    )
    # Decision follow-up classification is valid only for an authoritative,
    # immediately preceding decision.  Generic first-turn wording such as
    # “为什么……” is an ordinary semantic request, not a missing-decision
    # clarification.
    if last_type not in _DECISION_FOLLOW_UP_TYPES:
        return None
    rejected = any(token in compact for token in _DECISION_REJECTED_TOKENS)
    expand = any(token in compact for token in _DECISION_EXPAND_TOKENS)
    explain = any(token in compact for token in _DECISION_EXPLAIN_TOKENS)
    alternative = bool(
        explicit_matches
        and str(chat_state.get("last_decision_type") or "")
        and any(
            token in compact
            for token in ("呢", "怎么样", "如何", "为什么", "怎么")
        )
    )
    if not any((rejected, expand, explain, alternative)):
        return None

    if not immediate:
        return {
            "detected": True,
            "bound": False,
            "follow_up_type": (
                "rejected_alternatives"
                if rejected
                else "expand_reason"
                if expand
                else "alternative_image"
                if alternative
                else "explain_reason"
            ),
            "requires_clarification": True,
            "clarification": "你是想追问哪一条比较、推荐或排序结论的原因？请先指出那条结论。",
            "selected_image_labels": [],
            "resolution_errors": [
                "no_immediately_previous_completed_decision"
            ],
        }

    active_labels = {
        str(item["image_label"])
        for item in _bindings(session, active_only=True)
    }
    selected_before = [
        str(label)
        for label in chat_state.get("last_selected_images", [])
        if str(label) in active_labels
    ]
    decision_scope = [
        str(label)
        for label in chat_state.get("last_decision_image_scope", [])
        if str(label) in active_labels
    ]
    ranking_labels = [
        str(item.get("image_label"))
        for item in chat_state.get("last_ranking", [])
        if isinstance(item, dict)
        and str(item.get("image_label")) in active_labels
    ]
    if rejected:
        labels = [
            label
            for label in (decision_scope or ranking_labels)
            if label not in selected_before
        ]
        follow_up_type = "rejected_alternatives"
    elif alternative:
        labels = [
            str(item["label"])
            for item in explicit_matches
            if str(item["label"]) in active_labels
        ]
        follow_up_type = "alternative_image"
    else:
        labels = selected_before or ranking_labels[:1] or decision_scope
        follow_up_type = "expand_reason" if expand else "explain_reason"
    labels = list(dict.fromkeys(labels))
    if not labels:
        return {
            "detected": True,
            "bound": False,
            "follow_up_type": follow_up_type,
            "requires_clarification": True,
            "clarification": "上一条结论涉及的图片已不可用，请重新指定当前会话中的 IMG_n。",
            "selected_image_labels": [],
            "resolution_errors": ["previous_decision_images_not_available"],
        }
    return {
        "detected": True,
        "bound": True,
        "follow_up_type": follow_up_type,
        "requires_clarification": False,
        "clarification": None,
        "selected_image_labels": labels,
        "resolution_errors": [],
        "referenced_decision": {
            "decision_type": last_type,
            "selected_images": selected_before,
            "ranking": list(chat_state.get("last_ranking", [])),
            "criterion": str(
                chat_state.get("last_comparison_criterion") or ""
            ),
            "reasons": copy.deepcopy(
                chat_state.get("last_decision_reasons", {})
            ),
            "public_answer": str(
                chat_state.get("last_decision_public_answer") or ""
            ),
            "tool_trace_id": chat_state.get(
                "last_decision_tool_trace_id"
            ),
            "origin_turn_id": chat_state.get(
                "last_decision_origin_turn_id"
            )
            or last_turn_id,
        },
    }


def _message_answer(content: Any) -> str:
    return clean_public_answer(content)


def _complete_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for message in messages:
        if message.get("role") == "user":
            pending = message
        elif message.get("role") == "assistant" and pending is not None:
            pairs.append({"user": pending, "assistant": message})
            pending = None
    return pairs


def build_multiturn_context(
    session: dict[str, Any],
    resolution: dict[str, Any],
    selected_assets: list[dict[str, Any]],
    *,
    decision_follow_up: dict[str, Any] | None = None,
    conversation_follow_up: dict[str, Any] | None = None,
    recent_pair_limit: int = 5,
    summary_fact_limit: int = 12,
) -> dict[str, Any]:
    """Build four bounded layers: current turn, recent pairs, summary, state."""

    all_bindings = _bindings(session)
    ref_labels = _label_for_ref(all_bindings)
    selected_refs = {str(item.get("ref")) for item in selected_assets}
    pairs = _complete_pairs(list(session.get("messages", [])))
    relevant_pairs = []
    for pair in pairs:
        refs = set(pair["user"].get("asset_refs", [])) | set(pair["assistant"].get("asset_refs", []))
        if not refs or refs & selected_refs:
            relevant_pairs.append(pair)
    recent_pairs = relevant_pairs[-recent_pair_limit:]
    old_pairs = relevant_pairs[:-recent_pair_limit]
    recent_messages: list[dict[str, str]] = []
    recent_index: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(recent_pairs, start=max(1, len(relevant_pairs) - len(recent_pairs) + 1)):
        user = pair["user"]
        assistant = pair["assistant"]
        labels = [
            ref_labels[ref]
            for ref in user.get("asset_refs", [])
            if ref in ref_labels
        ]
        user_text = str(user.get("resolved_question") or user.get("content") or "").strip()
        assistant_text = _message_answer(assistant.get("content"))
        if not user_text or not assistant_text:
            continue
        recent_messages.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )
        recent_index.append({"pair": pair_index, "image_references": labels})

    chat_state = session.get("chat_state") if isinstance(session.get("chat_state"), dict) else {}
    summary = copy.deepcopy(chat_state.get("summary") if isinstance(chat_state.get("summary"), dict) else {})
    summary.setdefault("confirmed_facts", [])
    summary.setdefault("asset_notes", {})
    summary.setdefault("current_goal", "")
    summary.setdefault("unresolved_questions", [])
    for pair in old_pairs:
        assistant_text = _message_answer(pair["assistant"].get("content"))
        if not assistant_text:
            continue
        refs = [
            ref_labels[ref]
            for ref in pair["user"].get("asset_refs", [])
            if ref in ref_labels
        ]
        note = assistant_text[:260]
        for label in refs:
            summary["asset_notes"][label] = note
        if note not in summary["confirmed_facts"]:
            summary["confirmed_facts"].append(note)
    summary["confirmed_facts"] = summary["confirmed_facts"][-summary_fact_limit:]

    selected_context = []
    for item in selected_assets:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        compact_facts = {
            key: facts.get(key)
            for key in (
                "global_observation",
                "global_scene",
                "subjects",
                "main_subjects",
                "activities",
                "relations",
                "attributes",
                "visible_text_candidates",
                "visible_text",
                "uncertainties",
                "uncertainty",
                "evidence_descriptions",
            )
            if facts.get(key) not in (None, "", [], {})
        }
        text_candidates = collect_text_candidates(
            {
                "presence": item.get("text_presence")
                or item.get("presence")
                or "uncertain",
                "ocr_candidates": item.get("ocr_candidates", []),
                "visual_candidates": item.get("visible_text_candidates", []),
            }
        )
        selected_context.append(
            {
                "image_label": item["image_label"],
                "direct_or_candidate_facts": compact_facts,
                "verified_text": item.get("verified_text", []),
                "readable_text": [
                    candidate.text
                    for candidate in text_candidates
                    if candidate.confidence == "high"
                ],
                "possible_text": [
                    candidate.text
                    for candidate in text_candidates
                    if candidate.confidence == "medium"
                ],
                "text_confidence_policy": (
                    "high_direct_medium_must_say_suspected_low_refuse"
                ),
                "evidence_truth_status": item.get("evidence_truth_status", "image_only_unverified_text"),
            }
        )

    state_variables = infer_current_turn_state(
        str(resolution["raw_question"]),
        resolution,
        previous_goal=str(summary.get("current_goal") or ""),
        confirmed_facts=list(summary["confirmed_facts"]),
    )
    state_variables.update(
        {
        "all_active_image_labels": resolution["active_image_labels"],
        "recent_mentioned_labels": list(chat_state.get("recent_mentioned_labels", [])),
        "newly_added_labels": resolution["newly_added_labels"],
        "unresolved_questions": summary["unresolved_questions"],
        }
    )
    if decision_follow_up and decision_follow_up.get("bound"):
        state_variables.update(
            {
                "current_intent": "explain",
                "requested_action": "explain_previous_decision",
                "decision_follow_up_type": decision_follow_up[
                    "follow_up_type"
                ],
                "previous_decision": copy.deepcopy(
                    decision_follow_up["referenced_decision"]
                ),
            }
        )
    if conversation_follow_up and conversation_follow_up.get("bound"):
        state_variables.update(
            {
                "current_intent": (
                    "explain"
                    if conversation_follow_up["follow_up_type"]
                    in {
                        "explain_previous_answer",
                        "explain_previous_decision",
                        "show_visual_evidence",
                        "justify_previous_claim",
                        "express_uncertainty_about_previous_answer",
                    }
                    else str(
                        conversation_follow_up.get(
                            "inherited_task_type"
                        )
                        or state_variables["current_intent"]
                    )
                ),
                "requested_action": conversation_follow_up[
                    "follow_up_type"
                ],
                "follow_up_type": conversation_follow_up[
                    "follow_up_type"
                ],
                "referenced_turn_id": conversation_follow_up.get(
                    "referenced_turn_id"
                ),
                "inherited_task_type": conversation_follow_up.get(
                    "inherited_task_type"
                ),
                "inherited_image_refs": list(
                    conversation_follow_up.get(
                        "inherited_image_refs", []
                    )
                ),
                "previous_turn_snapshot": copy.deepcopy(
                    conversation_follow_up.get(
                        "previous_turn_snapshot", {}
                    )
                ),
            }
        )
    char_count = sum(len(item["content"]) for item in recent_messages) + len(json.dumps(summary, ensure_ascii=False))
    return {
        "current_turn": {
            "raw_question": resolution["raw_question"],
            "resolved_question": resolution["resolved_question"],
            "rewritten_standalone_request": (
                conversation_follow_up.get("standalone_request")
                if conversation_follow_up
                and conversation_follow_up.get("bound")
                else None
            ),
            "selected_image_labels": resolution["selected_image_labels"],
            "current_intent": state_variables["current_intent"],
            "requested_action": state_variables["requested_action"],
        },
        "recent_messages": recent_messages,
        "recent_message_index": recent_index,
        "conversation_summary": summary,
        "state_variables": state_variables,
        "previous_turn_snapshot": (
            copy.deepcopy(
                conversation_follow_up.get(
                    "previous_turn_snapshot", {}
                )
            )
            if conversation_follow_up
            else {}
        ),
        "current_conversation_state": {
            "active_task": chat_state.get("active_task"),
            "active_target_images": list(
                chat_state.get("active_target_images", [])
            ),
            "pending_clarification": copy.deepcopy(
                chat_state.get("pending_clarification")
            ),
            "discourse_focus_stack": copy.deepcopy(
                chat_state.get("discourse_focus_stack", [])[-8:]
            ),
        },
        "selected_asset_context": selected_context,
        "pruned_complete_pair_count": len(old_pairs),
        "recent_complete_pair_count": len(recent_pairs),
        "approx_context_characters": char_count,
        "context_policy": (
            "stable_policy + previous_turn_snapshot + "
            "current_conversation_state + current_user_message_last + "
            "optional_standalone_rewrite"
        ),
    }


def validate_chat_model_output(
    model_payload: Any,
    *,
    expected_labels: list[str],
    selected_assets: list[dict[str, Any]],
    all_bindings: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the model-only IMG contract and attach backend identities later."""

    errors: list[str] = []
    if not isinstance(model_payload, dict):
        return None, ["payload_not_object"]
    allowed = {"answer", "image_references", "evidence", "uncertainty"}
    unexpected = sorted(set(model_payload) - allowed)
    if unexpected:
        errors.append(f"unexpected_fields:{','.join(unexpected)}")
    answer = model_payload.get("answer")
    references = model_payload.get("image_references")
    evidence = model_payload.get("evidence")
    uncertainty = model_payload.get("uncertainty")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("answer_empty")
    if not isinstance(references, list) or not all(isinstance(item, str) for item in references):
        errors.append("image_references_not_string_list")
        reference_labels: list[str] = []
    else:
        reference_labels = [item.upper() for item in references]
        if len(set(reference_labels)) != len(reference_labels):
            errors.append("duplicate_image_reference")
    known_labels = {str(item["image_label"]) for item in all_bindings}
    active_selected = {str(item["image_label"]) for item in selected_assets}
    unknown = sorted(set(reference_labels) - known_labels)
    outside = sorted(set(reference_labels) - active_selected)
    if unknown:
        errors.append(f"unknown_image_reference:{','.join(unknown)}")
    elif outside:
        errors.append(f"unselected_or_removed_image_reference:{','.join(outside)}")
    if set(reference_labels) != set(expected_labels):
        errors.append("resolved_reference_coverage_mismatch")
    if not isinstance(evidence, list) or not all(isinstance(item, str) and item.strip() for item in evidence):
        errors.append("evidence_not_nonempty_string_list")
    if not isinstance(uncertainty, list) or not all(isinstance(item, str) for item in uncertainty):
        errors.append("uncertainty_not_string_list")

    serialized = json.dumps(model_payload, ensure_ascii=False)
    leaked = []
    for binding in all_bindings:
        for key in ("asset_id", "source_asset_id", "ref", "sha256", "image_id"):
            value = str(binding.get(key) or "")
            if value and value not in known_labels and value in serialized:
                leaked.append(key)
    if leaked:
        errors.append(f"backend_identity_leak:{','.join(sorted(set(leaked)))}")
    # OCR safety applies to public answer values, not JSON field names.
    # Otherwise a short candidate such as ``ac`` falsely matches
    # ``action_completed`` in a structured response.
    public_text = "\n".join(
        [
            str(answer or ""),
            *[
                str(item)
                for item in (
                    evidence if isinstance(evidence, list) else []
                )
            ],
            *[
                str(item)
                for item in (
                    uncertainty
                    if isinstance(uncertainty, list)
                    else []
                )
            ],
        ]
    )
    unverified_text_claims = []
    generic_text_tokens = {"文字", "文本", "招牌", "牌匾", "不可辨文字", "无法辨认"}
    for asset in selected_assets:
        if asset.get("verified_text"):
            continue
        candidate_records = collect_text_candidates(
            {
                "presence": asset.get("text_presence")
                or asset.get("presence")
                or "uncertain",
                "ocr_candidates": asset.get("ocr_candidates", []),
                "visual_candidates": asset.get("visible_text_candidates", []),
            }
        )
        for candidate in candidate_records:
            text = candidate.text
            if (
                len(text) >= 2
                and text not in generic_text_tokens
                and text in public_text
                and not candidate_is_allowed(
                    candidate,
                    task_mode=CHAT,
                    qualified=is_qualified_statement(public_text),
                )
            ):
                unverified_text_claims.append(str(asset["image_label"]))
                break
    if unverified_text_claims:
        errors.append(
            "unverified_text_claim:"
            + ",".join(sorted(set(unverified_text_claims)))
        )
    if errors:
        return None, errors

    asset_map = {str(item["image_label"]): item for item in selected_assets}
    normalized = {
        "answer": answer.strip(),
        "image_references": [
            {
                "image_label": label,
                "asset": {
                    "ref": asset_map[label]["ref"],
                    "asset_id": asset_map[label]["asset_id"],
                    "source": asset_map[label]["source"],
                    "sha256": asset_map[label]["sha256"],
                    "image_url": asset_map[label]["image_url"],
                },
            }
            for label in reference_labels
        ],
        "evidence": [item.strip() for item in evidence],
        "uncertainty": [item.strip() for item in uncertainty],
        "answer_source": "qwen3_vl_multiturn_chat_v2",
    }
    return normalized, []


def deterministic_chat_fallback(
    selected_assets: list[dict[str, Any]],
    *,
    intent: str,
    contract_errors: list[str],
) -> dict[str, Any]:
    """Produce a traceable product answer without inventing new visual facts."""

    fact_keys = ("global_observation", "global_scene", "subjects", "main_subjects", "activities", "relations", "attributes")
    clauses: list[str] = []
    evidence: list[str] = []
    references = []
    for item in selected_assets:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        blocked_text = {
            str(candidate.get("text") or "").strip()
            for candidate in item.get("ocr_candidates", [])
            if isinstance(candidate, dict)
            and len(str(candidate.get("text") or "").strip()) >= 2
        } if not item.get("verified_text") else set()
        snippets = []
        for key in fact_keys:
            value = facts.get(key)
            if not isinstance(value, str):
                continue
            compact = " ".join(value.split()).strip("。；; ")
            if any(text in compact for text in blocked_text):
                continue
            if compact and not compact.startswith("not_available") and compact not in snippets:
                snippets.append(compact)
            if len(snippets) == 3:
                break
        label = str(item["image_label"])
        if snippets:
            clauses.append(f"{label}：{'；'.join(snippets)}")
            evidence.extend(f"{label}：{snippet}" for snippet in snippets)
        references.append(
            {
                "image_label": label,
                "asset": {
                    "ref": item["ref"],
                    "asset_id": item["asset_id"],
                    "source": item["source"],
                    "sha256": item["sha256"],
                    "image_url": item["image_url"],
                },
            }
        )
    if clauses and intent == "compare":
        answer = "模型回答未通过 IMG 合同校验。根据已有 P3 候选事实，可确认各图核心内容为：" + "；".join(clauses) + "。"
    elif clauses:
        answer = "模型回答未通过 IMG 合同校验。根据已有 P3 候选事实，" + "；".join(clauses) + "。"
    else:
        answer = "模型回答未通过 IMG 合同校验，当前也没有足够的既有结构化事实；请先分析图片后再试。"
    return {
        "answer": answer,
        "image_references": references,
        "evidence": evidence or ["当前没有可用的结构化视觉事实。"],
        "uncertainty": [
            "这是可追踪的既有 P3 候选事实降级，不是新的模型语义结论。",
            f"模型合同错误：{'; '.join(contract_errors)}",
        ],
        "answer_source": "deterministic_existing_p3_fact_fallback",
    }


def update_chat_state_after_turn(
    session: dict[str, Any],
    *,
    question: str,
    selected_labels: list[str],
    answer: str,
    requires_clarification: bool,
) -> None:
    """Persist only bounded state variables; never promote model text silently."""

    chat_state = session.get("chat_state") if isinstance(session.get("chat_state"), dict) else {}
    summary = chat_state.get("summary") if isinstance(chat_state.get("summary"), dict) else {}
    summary.setdefault("confirmed_facts", [])
    summary.setdefault("asset_notes", {})
    summary.setdefault("unresolved_questions", [])
    summary["current_goal"] = question.strip()[:300]
    for label in selected_labels:
        if answer:
            summary["asset_notes"][label] = answer.strip()[:320]
    if requires_clarification:
        item = question.strip()[:300]
        if item and item not in summary["unresolved_questions"]:
            summary["unresolved_questions"].append(item)
    else:
        summary["unresolved_questions"] = [
            item for item in summary["unresolved_questions"] if item != question.strip()[:300]
        ]
    normalized_question = "".join(question.split())
    if normalized_question in {"对", "是的", "没错", "正确", "就是这样"}:
        previous_messages = list(session.get("messages", []))
        previous_answer = ""
        for message in reversed(previous_messages):
            if message.get("role") == "assistant":
                previous_answer = _message_answer(message.get("content"))
                if previous_answer:
                    break
        if previous_answer and previous_answer not in summary["confirmed_facts"]:
            summary["confirmed_facts"].append(previous_answer[:320])
    summary["confirmed_facts"] = summary["confirmed_facts"][-12:]
    summary["unresolved_questions"] = summary["unresolved_questions"][-8:]
    chat_state["summary"] = summary
    chat_state["recent_mentioned_labels"] = list(selected_labels)
    chat_state["newly_added_labels"] = []
    session["chat_state"] = chat_state


def update_decision_state_after_turn(
    session: dict[str, Any],
    *,
    response: dict[str, Any],
    image_scope: list[str],
    assistant_message_id: str,
) -> None:
    """Persist a completed compare/select/rank decision for one-turn follow-ups."""

    action = str(response.get("action") or "")
    if action not in {"compare", "select", "rank"}:
        return
    if response.get("product_contract_valid") is not True:
        return
    answer_payload = (
        response.get("answer")
        if isinstance(response.get("answer"), dict)
        else {}
    )
    if answer_payload.get("needs_clarification") is True:
        return
    selected = [
        str(item.get("image_label"))
        for item in response.get("selected", [])
        if isinstance(item, dict) and item.get("image_label")
    ]
    ranking = [
        {
            "image_label": str(item.get("image_label")),
            "rank": item.get("rank"),
            "reason": str(item.get("reason") or "").strip(),
        }
        for item in response.get("ranking", [])
        if isinstance(item, dict) and item.get("image_label")
    ]
    public_answer = str(
        response.get("public_answer")
        or answer_payload.get("public_answer")
        or answer_payload.get("answer")
        or response.get("display_text")
        or ""
    ).strip()
    reasons: dict[str, list[str]] = {}
    for item in ranking:
        if item["reason"]:
            reasons.setdefault(item["image_label"], []).append(
                item["reason"]
            )
    for item in response.get("selected", []):
        if not isinstance(item, dict) or not item.get("image_label"):
            continue
        reason = str(item.get("reason") or "").strip()
        if reason:
            reasons.setdefault(str(item["image_label"]), []).append(reason)
    for evidence in answer_payload.get("evidence", []):
        text = str(evidence).strip()
        match = re.match(r"(IMG_[1-9]\d*)[：:]\s*(.+)", text)
        if match:
            reasons.setdefault(match.group(1), []).append(match.group(2))
    chat_state = (
        session.get("chat_state")
        if isinstance(session.get("chat_state"), dict)
        else {}
    )
    decision_type = (
        "top_k"
        if action == "select" and len(selected) > 1
        else "select"
        if action == "select"
        else "rank"
        if action == "rank"
        else "compare"
    )
    chat_state.update(
        {
            "last_decision_type": decision_type,
            "last_selected_images": selected,
            "last_ranking": ranking,
            "last_comparison_criterion": str(
                response.get("criterion") or ""
            ).strip(),
            "last_decision_reasons": reasons,
            "last_decision_public_answer": public_answer,
            "last_decision_tool_trace_id": response.get("request_id"),
            "last_decision_turn_id": assistant_message_id,
            "last_decision_origin_turn_id": assistant_message_id,
            "last_decision_image_scope": list(
                dict.fromkeys(str(label) for label in image_scope)
            ),
        }
    )
    session["chat_state"] = chat_state


def continue_decision_state_after_follow_up(
    session: dict[str, Any],
    *,
    assistant_message_id: str,
) -> None:
    """Advance only the discourse-chain tail; keep the original decision intact."""

    chat_state = (
        session.get("chat_state")
        if isinstance(session.get("chat_state"), dict)
        else {}
    )
    if chat_state.get("last_decision_type"):
        chat_state["last_decision_turn_id"] = assistant_message_id
        session["chat_state"] = chat_state
