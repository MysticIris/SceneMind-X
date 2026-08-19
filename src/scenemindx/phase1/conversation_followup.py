"""General, auditable conversation follow-up resolution for Phase 5.4-H.

The deterministic layer owns high-confidence dialogue acts and state
inheritance.  The optional model rewriter is deliberately narrower: it may
only turn an already-detected contextual request into a standalone request.
It never answers the user, inspects images, or mutates conversation state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .multiturn_chat import local_image_reference_matches


REWRITER_CANDIDATE_ID = (
    "SCENEMINDX_CONTEXTUAL_QUERY_REWRITER_V1_CANDIDATE"
)
REWRITER_PROMPT_ID = "phase5_4h_contextual_query_rewriter_v1"

CONTINUATION_ACTS = {
    "same_task_same_target",
    "same_task_new_target",
    "continue_previous_answer",
    "elaborate_previous_answer",
    "summarize_previous_answer",
    "shorten_previous_answer",
    "rewrite_previous_answer",
    "give_example_for_previous_answer",
    "translate_previous_answer",
}
REASON_ACTS = {
    "explain_previous_answer",
    "explain_previous_decision",
    "explain_unselected_image",
    "show_visual_evidence",
    "justify_previous_claim",
    "express_uncertainty_about_previous_answer",
}
TARGET_ACTS = {
    "switch_image_target",
    "same_task_new_targets",
    "targets_and_action_substitution",
    "add_image_target",
    "remove_image_target",
    "compare_with_previous_image",
    "ask_about_alternative_image",
    "ask_about_remaining_images",
    "criterion_substitution",
    "action_substitution",
}
CORRECTION_ACTS = {
    "correct_previous_assumption",
    "reject_previous_answer",
    "clarify_user_intent",
    "ambiguous_follow_up",
    "unresolvable_follow_up",
}
FOLLOW_UP_ACTS = (
    CONTINUATION_ACTS | REASON_ACTS | TARGET_ACTS | CORRECTION_ACTS
)
TEXT_TRANSFORM_ACTS = {
    "summarize_previous_answer",
    "shorten_previous_answer",
    "rewrite_previous_answer",
    "give_example_for_previous_answer",
    "translate_previous_answer",
}
DECISION_TASKS = {
    "compare",
    "select",
    "select_one",
    "select_top_k",
    "top_k",
    "rank",
    "rank_all",
    "recommend",
    "recommendation",
    "compare_or_rank_images",
}
VISUAL_TASKS = {
    "vqa",
    "describe",
    "visual_summary",
    "visual_description",
    "visual_commonality",
    "compare",
    "select",
    "rank",
    "recommend",
    "explain",
    "generate",
    "generate_content_from_images",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContextualQueryRewriterCandidate:
    """Load and hash-check the isolated L1 query rewriter candidate."""

    def __init__(self, project_root: Path) -> None:
        self.root = (
            project_root
            / "prompts"
            / "phase5_4h"
            / "contextual_query_rewriter_v1_candidate"
        )
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("candidate_id") != REWRITER_CANDIDATE_ID:
            raise ValueError("phase5_4h_rewriter_identity_mismatch")
        spec = dict(self.manifest["prompt"])
        self.prompt_path = project_root / str(spec["file"])
        if _sha256(self.prompt_path) != str(spec["raw_sha256"]):
            raise ValueError("phase5_4h_rewriter_sha256_mismatch")
        self.text = self.prompt_path.read_text(encoding="utf-8")

    def identity(self) -> dict[str, Any]:
        """Execute the identity operation."""
        return {
            "candidate_id": REWRITER_CANDIDATE_ID,
            "prompt_id": REWRITER_PROMPT_ID,
            "prompt_sha256": str(
                self.manifest["prompt"]["raw_sha256"]
            ),
            "status": str(self.manifest["status"]),
            "max_new_tokens": int(
                self.manifest["generation"]["max_new_tokens"]
            ),
        }

    def render(
        self,
        *,
        current_user_message: str,
        recent_clean_pairs: list[dict[str, str]],
        relevant_turn_state: dict[str, Any],
        current_reference_mapping: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Render the requested value."""
        values = {
            "CURRENT_USER_MESSAGE": current_user_message,
            "RECENT_CLEAN_PAIRS": recent_clean_pairs[-3:],
            "RELEVANT_TURN_STATE": relevant_turn_state,
            "CURRENT_REFERENCE_MAPPING": current_reference_mapping,
        }
        prompt = self.text
        for key, value in values.items():
            rendered = (
                value
                if isinstance(value, str)
                else json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            prompt = prompt.replace("{{" + key + "}}", rendered)
        unresolved = re.findall(r"\{\{[A-Z0-9_]+\}\}", prompt)
        if unresolved:
            raise ValueError(
                "phase5_4h_rewriter_unresolved_placeholders:"
                + ",".join(sorted(set(unresolved)))
            )
        return prompt, self.identity()


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


def validate_contextual_rewrite(
    value: Any,
    *,
    allowed_turn_ids: set[str],
    allowed_image_refs: set[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate contextual rewrite."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return None, ["rewriter_root_not_object"]
    expected = {
        "standalone_request",
        "follow_up_type",
        "referenced_turn_id",
        "inherited_task_type",
        "inherited_image_refs",
        "confidence",
        "needs_clarification",
        "reason_code",
    }
    if set(value) != expected:
        errors.append("rewriter_fields_mismatch")
    follow_up_type = str(value.get("follow_up_type") or "")
    if follow_up_type not in FOLLOW_UP_ACTS:
        errors.append("rewriter_follow_up_type_invalid")
    confidence = str(value.get("confidence") or "")
    if confidence not in {"high", "medium", "low"}:
        errors.append("rewriter_confidence_invalid")
    needs_clarification = value.get("needs_clarification")
    if not isinstance(needs_clarification, bool):
        errors.append("rewriter_clarification_not_boolean")
    turn_id = value.get("referenced_turn_id")
    if turn_id is not None and str(turn_id) not in allowed_turn_ids:
        errors.append("rewriter_unknown_turn")
    image_refs = value.get("inherited_image_refs")
    if not isinstance(image_refs, list) or any(
        not isinstance(item, str) for item in image_refs
    ):
        errors.append("rewriter_image_refs_not_string_list")
        image_refs = []
    unknown = [
        str(item)
        for item in image_refs
        if str(item) not in allowed_image_refs
    ]
    if unknown:
        errors.append("rewriter_unknown_or_removed_image")
    standalone = str(value.get("standalone_request") or "").strip()
    if needs_clarification is False and not standalone:
        errors.append("rewriter_standalone_request_empty")
    if needs_clarification is True and standalone:
        errors.append("rewriter_clarification_must_not_rewrite")
    forbidden = (
        "prompt",
        "validator",
        "trace",
        "schema",
        "raw_output",
        "tool_name",
    )
    serialized = json.dumps(value, ensure_ascii=False).lower()
    if any(token in serialized for token in forbidden):
        errors.append("rewriter_internal_language_leak")
    if errors:
        return None, errors
    return {
        "standalone_request": standalone,
        "follow_up_type": follow_up_type,
        "referenced_turn_id": str(turn_id) if turn_id else None,
        "inherited_task_type": (
            str(value["inherited_task_type"])
            if value.get("inherited_task_type")
            else None
        ),
        "inherited_image_refs": list(dict.fromkeys(image_refs)),
        "confidence": confidence,
        "needs_clarification": needs_clarification,
        "reason_code": str(value.get("reason_code") or ""),
    }, []


def parse_contextual_rewrite(
    raw_output: str,
    *,
    allowed_turn_ids: set[str],
    allowed_image_refs: set[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse contextual rewrite."""
    return validate_contextual_rewrite(
        _extract_json_object(raw_output),
        allowed_turn_ids=allowed_turn_ids,
        allowed_image_refs=allowed_image_refs,
    )


def _public_text(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(
            text for item in value if (text := _public_text(item))
        )
    if not isinstance(value, dict):
        return ""
    for key in (
        "display_text",
        "public_answer",
        "answer",
        "final_text",
        "content",
    ):
        if key in value:
            text = _public_text(value.get(key))
            if text:
                return text
    return ""


def _answer_contract(response: dict[str, Any]) -> dict[str, Any]:
    answer = response.get("answer")
    return answer if isinstance(answer, dict) else {}


def _task_type(intent: str, question: str) -> str:
    compact = re.sub(r"\s+", "", question)
    if intent in {"rank", "compare", "recommend", "select"}:
        return intent
    if intent in {"generate", "generate_content_from_images"}:
        return "content_generation"
    if intent in {"retrieve", "search_images"}:
        return "search"
    if intent in {"system_utility", "direct_chat", "text_chat"}:
        return intent
    if any(token in compact for token in ("概括", "总结", "简述")):
        return "visual_summary"
    if any(token in compact for token in ("详细描述", "描述", "画面")):
        return "visual_description"
    if any(token in compact for token in ("共同点", "共性")):
        return "visual_commonality"
    if intent == "vqa":
        return "visual_vqa"
    return intent or "vqa"


def _new_dialogue_act(intent: str, task_type: str) -> str:
    if task_type == "search":
        return "new_search_task"
    if task_type == "content_generation":
        return "new_generation_task"
    if task_type in DECISION_TASKS:
        return "new_compare_task"
    if task_type == "system_utility":
        return "new_tool_task"
    if task_type in VISUAL_TASKS or task_type.startswith("visual_"):
        return "new_visual_task"
    return "new_direct_text_task"


def update_general_turn_state(
    session: dict[str, Any],
    *,
    user_message_id: str,
    assistant_message_id: str,
    question: str,
    intent: str,
    assets: list[dict[str, Any]],
    response: dict[str, Any],
    follow_up: dict[str, Any] | None,
    created_at: str,
) -> dict[str, Any]:
    """Persist a bounded completed-turn record and derived conversation state."""

    chat_state = session.setdefault("chat_state", {})
    asset_labels = [
        str(item.get("image_label"))
        for item in assets
        if item.get("image_label")
    ]
    labels = (
        [
            str(item)
            for item in follow_up.get(
                "selected_image_labels", []
            )
        ]
        if follow_up and follow_up.get("bound")
        else asset_labels
    )
    task_type = (
        str(follow_up.get("inherited_task_type") or intent)
        if follow_up and follow_up.get("bound")
        else _task_type(intent, question)
    )
    dialogue_act = (
        str(follow_up.get("follow_up_type"))
        if follow_up and follow_up.get("detected")
        else _new_dialogue_act(intent, task_type)
    )
    answer = _public_text(response)
    contract = _answer_contract(response)
    evidence = [
        _public_text(item)
        for item in (
            contract.get("evidence")
            or response.get("evidence")
            or []
        )
        if _public_text(item)
    ][:8]
    uncertainty = [
        _public_text(item)
        for item in (
            contract.get("uncertainty")
            or response.get("uncertainty")
            or []
        )
        if _public_text(item)
    ][:8]
    claims = [
        item.strip()
        for item in re.split(r"(?<=[。！？!?])", answer)
        if item.strip()
    ][:5]
    inherited_frame = (
        dict(follow_up.get("inherited_task_frame") or {})
        if follow_up
        else {}
    )
    action = str(
        response.get("action")
        or inherited_frame.get("action")
        or ""
    )
    selected_images = list(
        contract.get("recommended_images")
        or response.get("selected_images")
        or chat_state.get("last_selected_images")
        or []
    )
    ranking = list(
        contract.get("ranking")
        or response.get("ranking")
        or (
            chat_state.get("last_ranking")
            if action in {"select", "rank", "compare"}
            else []
        )
        or []
    )
    completed = str(response.get("status") or "") not in {
        "failed",
        "refused",
        "tool_failed",
    }
    follow_up_eligible = bool(
        completed
        and answer
        and str(response.get("status") or "")
        != "clarification_required"
    )
    criterion = str(
        response.get("criterion")
        or inherited_frame.get("criterion")
        or chat_state.get("last_comparison_criterion")
        or ""
    ).strip()
    select_count = int(
        inherited_frame.get("k")
        or (
            len(selected_images)
            if action == "select" and selected_images
            else 1
        )
    )
    task_frame = {
        "task_type": (
            "compare_select"
            if action in {"compare", "select", "rank"}
            or task_type in DECISION_TASKS
            else task_type
        ),
        "action": action or None,
        "criterion": criterion,
        "k": select_count,
        "target_images": labels,
        "output_style": (
            str(inherited_frame.get("output_style") or "")
            or (
                "明确选择并简要说明理由"
                if action == "select"
                else "完整排序并简要说明理由"
                if action == "rank"
                else "比较并说明可见依据"
                if action == "compare"
                else str(
                    follow_up.get("answer_style") or "default"
                )
                if follow_up
                else "default"
            )
        ),
        "completed": completed,
        "source_turn_id": assistant_message_id,
    }
    record = {
        "turn_id": assistant_message_id,
        "user_turn_id": user_message_id,
        "user_message": question,
        "public_assistant_answer": answer,
        "task_type": task_type,
        "dialogue_act": dialogue_act,
        "tool_name": (
            intent
            if intent
            not in {"direct_chat", "text_chat", "clarification"}
            else None
        ),
        "tool_action": action or None,
        "resolved_image_refs": labels,
        "primary_target_images": labels,
        "image_scope_type": (
            str(
                response.get("reference_resolution", {}).get(
                    "explicit_image_scope"
                )
                or ""
            )
            or (
                "single"
                if len(labels) == 1
                else "collection"
                if labels
                else "none"
            )
        ),
        "selected_images": selected_images,
        "ranking": copy.deepcopy(ranking),
        "criterion": criterion,
        "task_frame": task_frame,
        "generated_content_type": (
            response.get("resolved_options", {}).get(
                "resolved_content_type"
            )
            if isinstance(response.get("resolved_options"), dict)
            else None
        ),
        "answer_subject": (
            str(follow_up.get("answer_subject") or "")
            if follow_up
            else task_type
        )
        or task_type,
        "answer_claims": claims,
        "answer_evidence_summary": evidence,
        "answer_uncertainty": uncertainty,
        "answer_style": (
            str(follow_up.get("answer_style") or "")
            if follow_up
            else "default"
        )
        or "default",
        "answer_length": len(answer),
        "follow_up_eligible": follow_up_eligible,
        "referenced_turn_id": (
            follow_up.get("referenced_turn_id") if follow_up else None
        ),
        "created_at": created_at,
    }
    turn_states = [
        dict(item)
        for item in chat_state.get("turn_states", [])
        if isinstance(item, dict)
    ]
    turn_states.append(record)
    chat_state["turn_states"] = turn_states[-32:]
    chat_state["last_completed_turn"] = record
    chat_state["last_completed_task_frame"] = task_frame
    chat_state["active_task_frame"] = task_frame
    if labels and completed:
        chat_state["last_visual_answer_turn"] = record
    if task_type in DECISION_TASKS or action in {
        "compare",
        "select",
        "rank",
    }:
        chat_state["last_decision_turn"] = record
    if task_type == "content_generation":
        chat_state["last_generation_turn"] = record
    if task_type == "search":
        chat_state["last_search_turn"] = record
    if record["tool_name"]:
        chat_state["last_tool_turn"] = record
    if follow_up_eligible:
        chat_state["last_explainable_turn"] = record
    if str(response.get("status") or "") == "clarification_required":
        chat_state["pending_clarification"] = {
            "turn_id": assistant_message_id,
            "question": answer,
            "created_at": created_at,
        }
    else:
        chat_state["pending_clarification"] = None
    if completed and task_type not in {
        "direct_chat",
        "system_utility",
        "clarification",
    }:
        chat_state["active_task"] = task_type
    if labels and completed:
        chat_state["active_target_images"] = labels
    focus_stack = [
        dict(item)
        for item in chat_state.get("discourse_focus_stack", [])
        if isinstance(item, dict)
    ]
    if follow_up_eligible:
        focus_stack.append(
            {
                "turn_id": assistant_message_id,
                "task_type": task_type,
                "dialogue_act": dialogue_act,
                "target_images": labels,
                "answer_subject": record["answer_subject"],
                "created_at": created_at,
            }
        )
    chat_state["discourse_focus_stack"] = focus_stack[-8:]
    session["chat_state"] = chat_state
    return record


def _active_labels(session: dict[str, Any]) -> list[str]:
    state = (
        session.get("chat_state")
        if isinstance(session.get("chat_state"), dict)
        else {}
    )
    return [
        str(item.get("image_label"))
        for item in state.get("asset_bindings", [])
        if item.get("status", "active") == "active"
        and item.get("image_label")
    ]


def _latest_turn(session: dict[str, Any]) -> dict[str, Any] | None:
    state = (
        session.get("chat_state")
        if isinstance(session.get("chat_state"), dict)
        else {}
    )
    value = state.get("last_completed_turn")
    if not isinstance(value, dict) or not value.get("turn_id"):
        return None
    return value


def _is_strong_new_task(message: str) -> bool:
    compact = re.sub(r"\s+", "", message)
    if any(
        token in compact
        for token in (
            "现在几点",
            "当前几点",
            "搜索",
            "检索",
            "搜一下",
            "写一个故事",
            "写一篇",
            "生成一段",
            "比较当前",
            "对比当前",
        )
    ):
        return True
    if any(
        token in compact
        for token in ("排序", "选一张", "选出", "推荐一张")
    ) and not compact.startswith(("为什么", "那", "如果")):
        return True
    return False


def _explicit_labels(message: str, active: set[str]) -> list[str]:
    labels = [
        str(item["label"])
        for item in local_image_reference_matches(message)
        if str(item.get("label")) in active
    ]
    return list(dict.fromkeys(labels))


def _clarification(
    *,
    follow_up_type: str,
    reason_code: str,
    text: str,
    referenced_turn_id: str | None = None,
) -> dict[str, Any]:
    return {
        "detected": True,
        "bound": False,
        "follow_up_type": follow_up_type,
        "referenced_turn_id": referenced_turn_id,
        "inherited_task_type": None,
        "inherited_image_refs": [],
        "selected_image_labels": [],
        "standalone_request": "",
        "confidence": "high",
        "needs_clarification": True,
        "requires_clarification": True,
        "clarification": text,
        "reason_code": reason_code,
        "resolution_errors": [reason_code],
        "execution_mode": "clarification",
        "rewriter_level": "L2",
    }


def _operation_for(task_type: str, prior_question: str) -> str:
    if task_type == "visual_summary" or any(
        token in prior_question for token in ("概括", "总结", "简述")
    ):
        return "概括"
    if task_type == "visual_description" or "描述" in prior_question:
        return "详细描述"
    if task_type == "visual_commonality" or "共同点" in prior_question:
        return "说明共同点"
    return "回答关于"


def _bound(
    *,
    follow_up_type: str,
    previous: dict[str, Any],
    labels: list[str],
    standalone_request: str,
    execution_mode: str,
    reason_code: str,
    answer_style: str = "default",
) -> dict[str, Any]:
    previous_frame = (
        dict(previous.get("task_frame") or {})
        if isinstance(previous.get("task_frame"), dict)
        else {
            "task_type": str(previous.get("task_type") or "vqa"),
            "action": previous.get("tool_action"),
            "criterion": str(previous.get("criterion") or ""),
            "k": (
                len(previous.get("selected_images") or [])
                or 1
            ),
            "target_images": list(
                previous.get("primary_target_images") or []
            ),
            "output_style": str(
                previous.get("answer_style") or "default"
            ),
            "completed": True,
            "source_turn_id": str(previous.get("turn_id") or ""),
        }
    )
    inherited_frame = {
        **previous_frame,
        "target_images": list(labels),
        "source_turn_id": str(
            previous_frame.get("source_turn_id")
            or previous.get("turn_id")
            or ""
        ),
    }
    return {
        "detected": True,
        "bound": True,
        "follow_up_type": follow_up_type,
        "referenced_turn_id": str(previous.get("turn_id") or ""),
        "inherited_task_type": str(previous.get("task_type") or "vqa"),
        "inherited_image_refs": labels,
        "selected_image_labels": labels,
        "standalone_request": standalone_request,
        "confidence": "high",
        "needs_clarification": False,
        "requires_clarification": False,
        "clarification": None,
        "reason_code": reason_code,
        "resolution_errors": [],
        "execution_mode": execution_mode,
        "rewriter_level": "L0",
        "previous_turn_snapshot": copy.deepcopy(previous),
        "answer_subject": str(
            previous.get("answer_subject")
            or previous.get("task_type")
            or ""
        ),
        "answer_style": answer_style,
        "inherited_task_frame": inherited_frame,
    }


def resolve_general_follow_up(
    message: str,
    session: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve high-confidence follow-ups before tool or decision routing."""

    raw = message.strip()
    compact = re.sub(r"\s+", "", raw)
    if not compact:
        return None
    if _is_strong_new_task(raw):
        return None
    active_labels = _active_labels(session)
    active = set(active_labels)
    explicit = _explicit_labels(raw, active)
    if "最后一张" in compact and active_labels:
        explicit = list(dict.fromkeys([*explicit, active_labels[-1]]))
    first_two_subset = "前两张" in compact and len(active_labels) >= 2
    if first_two_subset:
        explicit = active_labels[:2]
    all_matches = local_image_reference_matches(raw)
    missing_explicit = [
        str(item.get("label"))
        for item in all_matches
        if str(item.get("label")) not in active
    ]
    previous = _latest_turn(session)

    # Follow-up wording is meaningful only when an authoritative previous
    # target exists.  First-turn questions such as “为什么这么暗” remain new
    # semantic requests; the normal Reference Resolver may still bind their
    # visual scope independently.
    if previous is None:
        if missing_explicit:
            return _clarification(
                follow_up_type="unresolvable_follow_up",
                reason_code="referenced_image_removed_or_unknown",
                text=(
                    "你提到的图片当前已不可用，请重新指定会话中仍有效的 IMG_n。"
                ),
            )
        return None

    explain = any(
        token in compact
        for token in (
            "为什么",
            "什么原因",
            "依据是什么",
            "怎么看出来",
            "为什么这么说",
            "为什么这样判断",
        )
    )
    elaborate = any(
        token in compact
        for token in (
            "详细一点",
            "更详细",
            "展开说",
            "具体一点",
            "具体讲",
        )
    )
    shorten = any(
        token in compact
        for token in ("简短一点", "短一点", "精简一下")
    )
    rewrite = any(
        token in compact
        for token in (
            "通俗",
            "换个说法",
            "换种说法",
            "重新表述",
            "改写",
        )
    )
    translate = any(
        token in compact
        for token in ("翻译成", "译成", "英文版", "英语版")
    )
    example = any(
        token in compact
        for token in ("举个例子", "给个例子", "来个例子")
    )
    continue_answer = compact in {
        "继续",
        "继续写",
        "接着说",
        "继续说",
        "往下写",
    }
    remaining = any(
        token in compact
        for token in (
            "其余两张",
            "其他两张",
            "剩下的",
            "其余图片",
            "其他的为什么不选",
        )
    )
    correction = compact.startswith(
        ("不是这个意思", "不对", "我说的是", "我的意思是")
    )
    clarify_intent = compact in {
        "重新来。",
        "重新来",
    }
    vague = compact in {
        "那个呢？",
        "那个呢",
        "你再说说。",
        "你再说说",
    }
    model_rewrite_cue = any(
        token in compact
        for token in (
            "照刚才的方式",
            "按刚才的要求",
            "接着刚才那个",
            "和刚才一样处理",
        )
    )
    target_switch = bool(
        previous
        and (all_matches or first_two_subset)
        and (
            compact.startswith(
                ("那", "换成", "再说")
            )
            or any(
                token in compact
                for token in ("怎么样", "呢", "再讲", "再说一次")
            )
        )
    )
    unselected_explanation = bool(
        previous
        and explain
        and all_matches
        and any(token in compact for token in ("没选", "未选", "不选"))
        and (
            str(previous.get("task_type")) in DECISION_TASKS
            or str(previous.get("tool_action") or "")
            in {"rank", "select", "compare", "top_k"}
        )
    )
    add_target = bool(
        previous
        and
        all_matches
        and any(token in compact for token in ("再加", "加上"))
    )
    remove_target = bool(
        previous
        and
        all_matches
        and any(
            token in compact
            for token in ("不看", "去掉", "排除", "只讲", "只说")
        )
    )
    compare_previous = bool(
        all_matches
        and any(token in compact for token in ("相比", "比较", "不同"))
        and any(token in compact for token in ("上一张", "前一张"))
    )
    criterion_change = bool(
        previous
        and (
            str(previous.get("task_type")) in DECISION_TASKS
            or str(previous.get("tool_action") or "")
            in {"rank", "select", "compare"}
        )
        and compact.startswith(("如果只看", "如果更看重", "如果按"))
    )
    action_change = bool(
        previous
        and (
            str(previous.get("task_type")) in DECISION_TASKS
            or str(previous.get("tool_action") or "")
            in {"rank", "select", "compare"}
        )
        and any(
            token in compact
            for token in (
                "完整排序",
                "完整排一下",
                "只选一张",
                "选两张",
                "选出两张",
            )
        )
    )
    detected = any(
        (
            explain,
            elaborate,
            shorten,
            rewrite,
            translate,
            example,
            continue_answer,
            remaining,
            correction,
            clarify_intent,
            vague,
            model_rewrite_cue,
            target_switch,
            add_target,
            remove_target,
            compare_previous,
            criterion_change,
            action_change,
            unselected_explanation,
        )
    )
    if not detected:
        return None
    if previous is None or not previous.get("follow_up_eligible"):
        follow_type = (
            "ambiguous_follow_up"
            if not missing_explicit
            else "unresolvable_follow_up"
        )
        return _clarification(
            follow_up_type=follow_type,
            reason_code="no_explainable_previous_turn",
            text=(
                "你想追问或继续哪一条内容？请先指出上一条回答或要看的图片。"
            ),
        )
    previous_id = str(previous.get("turn_id") or "")
    previous_labels = [
        str(item)
        for item in previous.get("primary_target_images", [])
        if str(item) in active
    ]
    if missing_explicit:
        return _clarification(
            follow_up_type="unresolvable_follow_up",
            reason_code="referenced_image_removed_or_unknown",
            text=(
                "你提到的图片当前已不可用，请重新指定会话中仍有效的 IMG_n。"
            ),
            referenced_turn_id=previous_id,
        )
    if clarify_intent:
        return _clarification(
            follow_up_type="clarify_user_intent",
            reason_code="restart_intent_requires_scope",
            text=(
                "你想重新执行上一任务，还是清空它并开始一个新任务？"
            ),
            referenced_turn_id=previous_id,
        )
    if vague:
        return _clarification(
            follow_up_type="ambiguous_follow_up",
            reason_code="vague_follow_up_has_multiple_interpretations",
            text=(
                "你是想让我扩写上一条回答、解释依据，还是改看另一张图片？"
            ),
            referenced_turn_id=previous_id,
        )

    prior_question = str(previous.get("user_message") or "")
    prior_answer = str(previous.get("public_assistant_answer") or "")
    task_type = str(previous.get("task_type") or "vqa")
    operation = _operation_for(task_type, prior_question)

    if correction:
        labels = explicit or previous_labels
        return _bound(
            follow_up_type="correct_previous_assumption",
            previous=previous,
            labels=labels,
            standalone_request=raw,
            execution_mode="visual" if labels else "text",
            reason_code="explicit_user_correction",
        )
    if remaining:
        decision_scope = [
            str(item)
            for item in (
                session.get("chat_state", {}).get(
                    "last_decision_image_scope", []
                )
            )
            if str(item) in active
        ]
        selected = {
            str(item)
            for item in session.get("chat_state", {}).get(
                "last_selected_images", []
            )
        }
        labels = [item for item in decision_scope if item not in selected]
        if not labels:
            return _clarification(
                follow_up_type="unresolvable_follow_up",
                reason_code="no_remaining_decision_images",
                text="上一条选择没有可继续说明的其余图片，请重新指定 IMG_n。",
                referenced_turn_id=previous_id,
            )
        return _bound(
            follow_up_type="ask_about_remaining_images",
            previous=previous,
            labels=labels,
            standalone_request=(
                "请结合可见内容说明上一轮未选择 "
                + "、".join(labels)
                + " 的原因，并保持上一轮选择标准不变。"
            ),
            execution_mode="visual",
            reason_code="remaining_images_from_previous_decision",
        )
    if compare_previous:
        labels = list(
            dict.fromkeys([*previous_labels[-1:], *explicit])
        )
        if len(labels) < 2:
            return _clarification(
                follow_up_type="ambiguous_follow_up",
                reason_code="comparison_missing_previous_target",
                text="你想把当前图片和哪一张先前图片比较？",
                referenced_turn_id=previous_id,
            )
        return _bound(
            follow_up_type="compare_with_previous_image",
            previous=previous,
            labels=labels,
            standalone_request=(
                "请比较 "
                + " 和 ".join(labels)
                + "，说明它们的可见差异。"
            ),
            execution_mode="visual",
            reason_code="explicit_compare_with_previous_target",
        )
    if add_target:
        labels = list(dict.fromkeys([*previous_labels, *explicit]))
        return _bound(
            follow_up_type="add_image_target",
            previous=previous,
            labels=labels,
            standalone_request=(
                f"请对 {'、'.join(labels)} 继续执行上一任务："
                f"{prior_question}"
            ),
            execution_mode="visual",
            reason_code="explicit_target_addition",
        )
    if remove_target:
        labels = explicit
        return _bound(
            follow_up_type="remove_image_target",
            previous=previous,
            labels=labels,
            standalone_request=(
                f"请只对 {'、'.join(labels)} 继续执行上一任务："
                f"{prior_question}"
            ),
            execution_mode="visual",
            reason_code="explicit_target_removal",
        )
    if unselected_explanation:
        labels = explicit
        result = _bound(
            follow_up_type="explain_unselected_image",
            previous=previous,
            labels=labels,
            standalone_request=(
                "请结合可见内容说明上一轮为什么没有选择 "
                + "、".join(labels)
                + "，并保持上一轮选择标准不变。"
            ),
            execution_mode="visual",
            reason_code="explicit_unselected_image_explanation",
        )
        result["inherited_task_frame"]["target_images"] = labels
        return result
    if explain and explicit and explicit != previous_labels:
        result = _bound(
            follow_up_type="switch_image_target",
            previous=previous,
            labels=explicit,
            standalone_request=raw,
            execution_mode="visual",
            reason_code="explicit_image_reference_overrides_previous_answer",
        )
        result["inherited_task_frame"]["target_images"] = explicit
        return result
    if target_switch:
        labels = explicit
        decision_frame = (
            dict(previous.get("task_frame") or {})
            if isinstance(previous.get("task_frame"), dict)
            else {}
        )
        is_decision = (
            task_type in DECISION_TASKS
            or str(previous.get("tool_action") or "")
            in {"compare", "select", "rank", "top_k"}
            or str(decision_frame.get("action") or "")
            in {"compare", "select", "rank", "top_k"}
        )
        action = str(
            decision_frame.get("action")
            or previous.get("tool_action")
            or "compare"
        )
        criterion = str(
            decision_frame.get("criterion")
            or previous.get("criterion")
            or session.get("chat_state", {}).get(
                "last_comparison_criterion"
            )
            or ""
        ).strip()
        k = int(decision_frame.get("k") or 1)
        if is_decision:
            if action == "select":
                standalone = (
                    f"在当前会话的 {' 和 '.join(labels)} 中，选择"
                    f"{'一张' if k == 1 else f'{k}张'}"
                    + (f"{criterion}的图片" if criterion else "图片")
                    + "，并简要说明理由。"
                )
            elif action == "rank":
                standalone = (
                    f"请按{criterion or '上一轮标准'}对 "
                    f"{'、'.join(labels)} 完整排序，并简要说明理由。"
                )
            else:
                standalone = (
                    f"请按{criterion or '上一轮标准'}比较 "
                    f"{'、'.join(labels)}，并说明可见依据。"
                )
        else:
            standalone = f"请{operation} {'、'.join(labels)}。"
        combined_action_change = bool(
            action_change and explicit and "排序" in compact
        )
        if combined_action_change:
            action = "rank"
            k = len(labels)
            standalone = (
                f"请按{criterion or '上一轮标准'}对 "
                f"{'、'.join(labels)} 完整排序，并简要说明理由。"
            )
        nondecision_same_task = bool(
            not is_decision
            and re.fullmatch(
                r"图(?:[一二三四五六七八九十]|\d+)(?:图片)?呢[？?。]?",
                compact,
            )
        )
        follow_type = (
            "targets_and_action_substitution"
            if combined_action_change
            else "same_task_new_targets"
            if (is_decision and len(labels) >= 2)
            or nondecision_same_task
            else "ask_about_alternative_image"
            if is_decision
            else "switch_image_target"
        )
        result = _bound(
            follow_up_type=follow_type,
            previous=previous,
            labels=labels,
            standalone_request=standalone,
            execution_mode="visual",
            reason_code="explicit_same_task_target_substitution",
        )
        result["follow_up_type"] = follow_type
        result["inherited_task_frame"].update(
            {
                "task_type": (
                    "compare_select"
                    if is_decision
                    else result["inherited_task_frame"].get(
                        "task_type"
                    )
                ),
                "action": action if is_decision else None,
                "criterion": criterion,
                "k": k,
                "target_images": labels,
            }
        )
        return result
    if criterion_change:
        labels = previous_labels
        if compact.startswith("如果更看重"):
            criterion = "更适合" + compact[len("如果更看重") :]
        else:
            criterion = re.sub(
                r"^如果(?:只看|按)",
                "",
                compact,
            )
        criterion = re.sub(r"[呢？?。]+$", "", criterion).strip()
        result = _bound(
            follow_up_type="criterion_substitution",
            previous=previous,
            labels=labels,
            standalone_request=(
                f"请按{criterion}对 {'、'.join(labels)} "
                "重新执行上一轮任务。"
            ),
            execution_mode="visual",
            reason_code="explicit_criterion_change_same_task",
        )
        result["inherited_task_frame"]["criterion"] = criterion
        return result
    if action_change:
        labels = explicit or previous_labels
        previous_frame = (
            dict(previous.get("task_frame") or {})
            if isinstance(previous.get("task_frame"), dict)
            else {}
        )
        criterion = str(
            previous_frame.get("criterion")
            or previous.get("criterion")
            or session.get("chat_state", {}).get(
                "last_comparison_criterion"
            )
            or ""
        ).strip()
        if any(token in compact for token in ("排序", "排一下")):
            action, k, style = "rank", len(labels), "完整排序并简要说明理由"
        elif "两张" in compact:
            action, k, style = "top_k", 2, "明确选择两张并简要说明理由"
        else:
            action, k, style = "select", 1, "明确选择并简要说明理由"
        result = _bound(
            follow_up_type=(
                "targets_and_action_substitution"
                if explicit
                and any(token in compact for token in ("排序", "排一下"))
                else "action_substitution"
            ),
            previous=previous,
            labels=labels,
            standalone_request=(
                f"请按{criterion or '上一轮标准'}对 "
                f"{'、'.join(labels)}"
                + (
                    "完整排序，并简要说明理由。"
                    if action == "rank"
                    else f"选择{k}张，并简要说明理由。"
                )
            ),
            execution_mode="visual",
            reason_code="explicit_action_change_same_task",
        )
        result["inherited_task_frame"].update(
            {
                "task_type": "compare_select",
                "action": action,
                "criterion": criterion,
                "k": k,
                "target_images": labels,
                "output_style": style,
            }
        )
        return result
    if explain:
        is_decision = task_type in DECISION_TASKS
        if is_decision and (
            str(previous.get("tool_action") or "") == "select"
            or task_type in {
            "select",
            "select_one",
            "select_top_k",
            "top_k",
            "recommend",
            "recommendation",
            }
        ):
            labels = [
                str(item)
                for item in session.get("chat_state", {}).get(
                    "last_selected_images", []
                )
                if str(item) in active
            ] or previous_labels
        else:
            labels = previous_labels
        follow_type = (
            "explain_previous_decision"
            if is_decision
            else (
                "justify_previous_claim"
                if "依据是什么" in compact
                else "explain_previous_answer"
                if "为什么这么说" in compact
                and "怎么看出来" not in compact
                else "show_visual_evidence"
                if labels
                else "explain_previous_answer"
            )
        )
        return _bound(
            follow_up_type=follow_type,
            previous=previous,
            labels=labels,
            standalone_request=(
                "请结合 "
                + ("、".join(labels) if labels else "上一轮公开回答")
                + " 解释上一回答的依据，并明确可见证据与不确定性。"
                + f"上一回答是：{prior_answer}"
            ),
            execution_mode="visual" if labels else "text_transform",
            reason_code=(
                "immediate_previous_decision_explanation"
                if is_decision
                else "immediate_previous_answer_evidence"
            ),
        )
    if elaborate:
        labels = previous_labels
        return _bound(
            follow_up_type="elaborate_previous_answer",
            previous=previous,
            labels=labels,
            standalone_request=(
                f"请更详细地回答上一问题“{prior_question}”"
                + (
                    f"，继续只使用 {'、'.join(labels)}。"
                    if labels
                    else "。"
                )
            ),
            execution_mode="visual" if labels else "text_transform",
            reason_code="immediate_previous_answer_elaboration",
            answer_style="detailed",
        )
    transform_type = (
        "shorten_previous_answer"
        if shorten
        else "rewrite_previous_answer"
        if rewrite
        else "translate_previous_answer"
        if translate
        else "give_example_for_previous_answer"
        if example
        else None
    )
    if transform_type:
        instruction = {
            "shorten_previous_answer": "在不改变含义的前提下精简",
            "rewrite_previous_answer": "改写为更通俗自然的说法",
            "translate_previous_answer": raw,
            "give_example_for_previous_answer": "给出一个帮助理解的例子",
        }[transform_type]
        return _bound(
            follow_up_type=transform_type,
            previous=previous,
            labels=previous_labels,
            standalone_request=(
                f"{instruction}以下上一轮公开回答：{prior_answer}"
            ),
            execution_mode="text_transform",
            reason_code="previous_public_answer_transform",
        )
    if continue_answer:
        labels = previous_labels
        return _bound(
            follow_up_type="continue_previous_answer",
            previous=previous,
            labels=labels,
            standalone_request=(
                f"请继续上一轮任务“{prior_question}”，从上一回答结尾自然续写，"
                f"不要重复已完成内容。上一回答：{prior_answer}"
            ),
            execution_mode="visual" if labels else "text_transform",
            reason_code="continue_immediate_previous_answer",
        )
    return {
        "detected": True,
        "bound": False,
        "follow_up_type": "ambiguous_follow_up",
        "referenced_turn_id": previous_id,
        "inherited_task_type": task_type,
        "inherited_image_refs": previous_labels,
        "selected_image_labels": previous_labels,
        "standalone_request": "",
        "confidence": "medium",
        "needs_clarification": False,
        "requires_clarification": False,
        "clarification": None,
        "reason_code": "context_dependency_requires_l1_rewrite",
        "resolution_errors": [],
        "execution_mode": "model_rewrite",
        "rewriter_level": "L1",
        "previous_turn_snapshot": copy.deepcopy(previous),
    }


def clean_rewriter_inputs(
    session: dict[str, Any],
    follow_up: dict[str, Any],
) -> dict[str, Any]:
    """Return only public pairs, relevant turn state and stable mappings."""

    messages = list(session.get("messages", []))
    pairs: list[dict[str, str]] = []
    pending: str | None = None
    for item in messages:
        role = item.get("role")
        if role == "user":
            pending = _public_text(item.get("content"))
        elif role == "assistant" and pending:
            answer = _public_text(item.get("content"))
            if answer:
                pairs.append({"user": pending, "assistant": answer})
            pending = None
    state = session.get("chat_state", {})
    turns = [
        {
            key: item.get(key)
            for key in (
                "turn_id",
                "user_message",
                "public_assistant_answer",
                "task_type",
                "dialogue_act",
                "primary_target_images",
                "answer_subject",
                "follow_up_eligible",
            )
        }
        for item in state.get("turn_states", [])[-3:]
        if isinstance(item, dict)
    ]
    mappings = {
        "active_images": _active_labels(session),
        "search_results": sorted(
            str(label)
            for label in state.get("tool_result_image_mapping", {})
            if str(label).startswith("SEARCH_")
        ),
    }
    return {
        "recent_clean_pairs": pairs[-3:],
        "relevant_turn_state": {
            "candidate_turns": turns,
            "active_task": state.get("active_task"),
            "active_target_images": state.get(
                "active_target_images", []
            ),
            "pending_clarification": state.get(
                "pending_clarification"
            ),
            "deterministic_detection": {
                key: follow_up.get(key)
                for key in (
                    "follow_up_type",
                    "referenced_turn_id",
                    "inherited_task_type",
                    "inherited_image_refs",
                    "reason_code",
                )
            },
        },
        "current_reference_mapping": mappings,
        "allowed_turn_ids": {
            str(item.get("turn_id"))
            for item in turns
            if item.get("turn_id")
        },
        "allowed_image_refs": set(
            mappings["active_images"] + mappings["search_results"]
        ),
    }
