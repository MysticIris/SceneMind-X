"""Phase 5.2-C conversation policy, current-turn state and response recovery.

The public answer and the technical recovery trace are deliberately separate.
No function in this module is allowed to turn a failed comparison, selection
or ranking request into a list of internal P3 facts.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from scenemindx.task_text_policy import (
    COMPARE,
    candidate_is_allowed,
    collect_text_candidates,
    is_qualified_statement,
)


POLICY_CANDIDATE_ID = "SCENEMINDX_CONVERSATION_POLICY_V1_CANDIDATE"
CHAT_CANDIDATE_ID = "SCENEMINDX_MULTITURN_CHAT_V3_CANDIDATE"
COMPARE_RANK_PARENT_CANDIDATE_ID = "SCENEMINDX_COMPARE_RANK_V2_CANDIDATE"
COMPARE_RANK_V4_PARENT_CANDIDATE_ID = "SCENEMINDX_COMPARE_RANK_V3_CANDIDATE"
COMPARE_RANK_CANDIDATE_ID = "SCENEMINDX_COMPARE_RANK_V4_CANDIDATE"
COMMON_FIELDS = {
    "public_answer",
    "intent",
    "action_completed",
    "recommended_images",
    "ranking",
    "image_references",
    "evidence",
    "uncertainty",
    "needs_clarification",
}
INTERNAL_LANGUAGE = re.compile(
    r"(P3|FALLBACK|Validator|Trace(?: ID)?|Prompt SHA|Schema|合同(?:校验)?|"
    r"模型回答未通过|候选事实|asset[_ ]?id|sha-?256|后端)",
    re.IGNORECASE,
)
_LABEL = re.compile(r"\bIMG[\s_-]?([1-9]\d*)\b", re.IGNORECASE)
_ORDINAL = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ConversationalResponsePromptCandidate:
    """Load and hash-check the bounded Phase 5.2-C candidate family."""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root / "prompts" / "phase5_2" / "conversational_response_v1_candidate"
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("candidate_id") != POLICY_CANDIDATE_ID:
            raise ValueError("phase5_2c_policy_candidate_identity_mismatch")
        if self.manifest.get("chat_candidate_id") != CHAT_CANDIDATE_ID:
            raise ValueError("phase5_2c_chat_candidate_identity_mismatch")
        if (
            self.manifest.get("compare_rank_candidate_id")
            != COMPARE_RANK_PARENT_CANDIDATE_ID
        ):
            raise ValueError("phase5_2c_compare_rank_candidate_identity_mismatch")
        self.files: dict[str, dict[str, Any]] = {}
        for name, raw_spec in dict(self.manifest["files"]).items():
            spec = dict(raw_spec)
            path = project_root / str(spec["file"])
            if _sha256(path) != str(spec["raw_sha256"]):
                raise ValueError(f"phase5_2c_candidate_sha256_mismatch:{name}")
            self.files[name] = {**spec, "path": path, "text": path.read_text(encoding="utf-8")}
        compare_root = (
            project_root
            / "prompts"
            / "phase5_4d"
            / "compare_rank_v4_candidate"
        )
        compare_manifest = json.loads(
            (compare_root / "manifest.json").read_text(encoding="utf-8")
        )
        if compare_manifest.get("candidate_id") != COMPARE_RANK_CANDIDATE_ID:
            raise ValueError("phase5_4b_compare_rank_candidate_identity_mismatch")
        if (
            compare_manifest.get("parent_candidate_id")
            != COMPARE_RANK_V4_PARENT_CANDIDATE_ID
        ):
            raise ValueError("phase5_4b_compare_rank_parent_mismatch")
        compare_spec = dict(compare_manifest["prompt"])
        compare_path = project_root / str(compare_spec["file"])
        if _sha256(compare_path) != str(compare_spec["raw_sha256"]):
            raise ValueError("phase5_4b_compare_rank_candidate_sha256_mismatch")
        self.files["compare_rank"] = {
            **compare_spec,
            "path": compare_path,
            "text": compare_path.read_text(encoding="utf-8"),
        }
        retry_spec = dict(compare_manifest["retry_prompt"])
        retry_path = project_root / str(retry_spec["file"])
        if _sha256(retry_path) != str(retry_spec["raw_sha256"]):
            raise ValueError("phase5_4d_compare_rank_retry_sha256_mismatch")
        self.files["compare_rank_retry"] = {
            **retry_spec,
            "path": retry_path,
            "text": retry_path.read_text(encoding="utf-8"),
        }
        self.compare_manifest = compare_manifest

    def identity(self, name: str) -> dict[str, Any]:
        """Execute the identity operation."""
        spec = self.files[name]
        candidate_id = {
            "conversation_policy": POLICY_CANDIDATE_ID,
            "multiturn_chat": CHAT_CANDIDATE_ID,
            "compare_rank": COMPARE_RANK_CANDIDATE_ID,
            "compare_rank_retry": COMPARE_RANK_CANDIDATE_ID,
        }.get(name, POLICY_CANDIDATE_ID)
        return {
            "candidate_id": candidate_id,
            "prompt_id": str(spec.get("prompt_id") or "phase5_2c_conversation_policy_v1"),
            "prompt_sha256": str(spec["raw_sha256"]),
            "status": str(
                self.compare_manifest["status"]
                if name in {"compare_rank", "compare_rank_retry"}
                else self.manifest["status"]
            ),
            "iteration": int(
                self.compare_manifest["iteration"]
                if name in {"compare_rank", "compare_rank_retry"}
                else self.manifest["iteration"]
            ),
        }

    def system_prompt(self) -> tuple[str, dict[str, Any]]:
        """Execute the system prompt operation."""
        text = (
            str(self.files["conversation_policy"]["text"]).strip()
            + "\n\n"
            + str(self.files["examples"]["text"]).strip()
        )
        return text, self.identity("conversation_policy")

    def render(self, name: str, values: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Render the requested value."""
        text = str(self.files[name]["text"])
        for key, value in values.items():
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
            text = text.replace("{{" + key + "}}", rendered)
        unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text)))
        if unresolved:
            raise ValueError(f"phase5_2c_unresolved_prompt_placeholders:{','.join(unresolved)}")
        return text, self.identity(name)


def infer_current_turn_state(
    question: str,
    resolution: dict[str, Any],
    *,
    previous_goal: str = "",
    confirmed_facts: list[str] | None = None,
) -> dict[str, Any]:
    """Build explicit state from the latest user turn, never from the old goal."""

    raw = " ".join(question.split())
    action_text = raw
    for marker in ("现在只回答", "只回答", "现在只问", "只问"):
        if marker in action_text:
            action_text = action_text.split(marker, 1)[1]
            break
    compact = "".join(action_text.lower().split())
    continue_signals = ("继续", "接着", "沿用", "按刚才", "照上一个", "续写", "还是刚才")
    previous_goal_active = any(token in compact for token in continue_signals)
    ranking_required = any(
        token in compact
        for token in (
            "排序",
            "排名",
            "从好到差",
            "从高到低",
            "完整顺序",
            "完整排序",
            "排一下",
        )
    )
    selection_required = any(
        token in compact
        for token in (
            "只选一",
            "选一个",
            "选一张",
            "选两张",
            "挑一张",
            "挑两张",
            "帮我挑",
            "选出",
            "哪张",
            "哪两张",
            "哪些更",
            "哪一个",
            "哪个更",
            "更适合",
            "更推荐",
            "推荐一张",
            "推荐",
            "最佳",
            "最好",
        )
    )
    compare_required = any(token in compact for token in ("比较", "对比", "差异", "区别", "共同点"))
    generation_required = any(
        token in compact
        for token in ("生成", "写一", "写个", "文案", "朋友圈", "游记", "图注", "标题", "诗", "故事", "广告")
    )
    retrieval_required = any(token in compact for token in ("检索", "搜索", "搜图", "找图", "图片库", "相似图片"))
    if retrieval_required:
        current_intent, requested_action = "retrieve", "retrieve"
    elif generation_required and not (ranking_required or selection_required or compare_required):
        current_intent, requested_action = "generate", "generate"
    elif ranking_required:
        current_intent, requested_action = "rank", "rank_all"
    elif selection_required:
        selection_count_match = re.search(
            r"(?:选|挑|推荐|哪)(?:出|给我|一下)?([一二两三四五1-5])张",
            compact,
        )
        selection_count = (
            _ORDINAL[selection_count_match.group(1)]
            if selection_count_match
            else 1
        )
        current_intent = "recommend"
        requested_action = (
            "select_top_k" if selection_count > 1 else "select_one"
        )
    elif compare_required:
        current_intent, requested_action = "compare", "compare"
    elif any(token in compact for token in ("描述", "说说画面", "画面里")):
        current_intent, requested_action = "describe", "describe"
    elif any(token in compact for token in ("解释", "说明", "为什么")):
        current_intent, requested_action = "explain", "explain"
    else:
        current_intent, requested_action = "vqa", "answer"

    criterion = ""
    for pattern in (
        r"(?:按|按照|依据|以)([^，。；？?]{1,32})(?:排序|排名|比较|选择|选|判断)",
        r"只比较([^，。；？?]{1,32})",
        r"(?:主要|更)考虑([^，。；？?]{1,32})",
    ):
        match = re.search(pattern, action_text)
        if match:
            criterion = match.group(1).strip()
            break
    if not criterion and (ranking_required or selection_required or compare_required):
        criterion = "画面表达清晰度（默认标准）"

    constraints = []
    if "只选一" in compact or "选一个" in compact or "选一张" in compact:
        constraints.append("exactly_one_recommendation")
    if ranking_required:
        constraints.append("complete_ranking")
    if any(token in compact for token in ("简短", "简要", "直接")):
        constraints.append("concise_answer")
    if "不要" in raw:
        constraints.append("honor_explicit_negative_constraint")

    if not selection_required:
        selection_count = 1

    return {
        "current_intent": current_intent,
        "requested_action": requested_action,
        "decision_required": selection_required or ranking_required,
        "ranking_required": ranking_required,
        "selection_count": selection_count,
        "comparison_criterion": criterion,
        "current_user_constraints": constraints,
        "resolved_images": list(resolution.get("selected_image_labels", [])),
        "current_focus": resolution.get("current_focus_label"),
        "current_goal": raw,
        "previous_goal": previous_goal,
        "previous_goal_active": previous_goal_active,
        "confirmed_facts": list(confirmed_facts or []),
        "unresolved_ambiguity": list(resolution.get("resolution_errors", [])),
    }


def clean_public_answer(content: Any) -> str:
    """Return only safe user-visible text for history and summary."""

    if isinstance(content, str):
        answer = content
    elif not isinstance(content, dict):
        return ""
    else:
        value = content.get("answer")
        if isinstance(value, str):
            answer = value
        elif isinstance(value, dict):
            answer = str(value.get("public_answer") or value.get("answer") or "")
        else:
            answer = str(content.get("public_answer") or "")
    compact = " ".join(answer.split())
    return "" if not compact or INTERNAL_LANGUAGE.search(compact) else compact


def _extract_object(raw: Any) -> tuple[dict[str, Any] | None, bool, list[str]]:
    if isinstance(raw, dict):
        return dict(raw), False, []
    if not isinstance(raw, str) or not raw.strip():
        return None, False, ["payload_not_object"]
    text = raw.strip()
    repaired = False
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        repaired = True
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, repaired, ["json_object_not_found"]
    if start or end != len(text) - 1:
        repaired = True
    object_text = text[start : end + 1]
    try:
        value = json.loads(object_text)
    except json.JSONDecodeError as exc:
        without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", object_text)
        if without_trailing_commas == object_text:
            return None, repaired, [f"json_decode_error:{exc.msg}"]
        try:
            value = json.loads(without_trailing_commas)
            repaired = True
        except json.JSONDecodeError as retry_exc:
            return None, repaired, [f"json_decode_error:{retry_exc.msg}"]
    if not isinstance(value, dict):
        return None, repaired, ["payload_not_object"]
    return value, repaired, []


def _label_value(value: Any, allowed: set[str]) -> tuple[str | None, bool]:
    if not isinstance(value, str):
        return None, False
    raw = value.strip().upper()
    direct = re.fullmatch(r"IMG_([1-9]\d*)", raw)
    if direct:
        label = f"IMG_{int(direct.group(1))}"
        return (label, False) if label in allowed else (label, False)
    alias = _LABEL.fullmatch(raw)
    if alias:
        return f"IMG_{int(alias.group(1))}", True
    ordinal = re.fullmatch(r"第([一二两三四五12345])张", value.strip())
    if ordinal:
        return f"IMG_{_ORDINAL[ordinal.group(1)]}", True
    return None, False


def _list_value(value: Any) -> tuple[list[Any], bool]:
    if value is None:
        return [], True
    if isinstance(value, list):
        return value, False
    return [value], True


_PUBLIC_REASON_CUES = (
    "因为",
    "理由",
    "依据",
    "主要是",
    "原因",
    "更贴合",
    "更符合",
    "差异在",
    "差异",
    "不同",
    "区别",
    "共同点",
    "相比",
    "考虑到",
)


def _has_public_decision_reason(text: str) -> bool:
    return any(token in text for token in _PUBLIC_REASON_CUES)


def deterministic_contract_repair(
    raw: Any,
    *,
    state: dict[str, Any],
    allowed_labels: list[str],
) -> tuple[dict[str, Any] | None, list[str], bool]:
    """Level 0/1 parse and uniquely normalize without adding visual semantics."""

    payload, repaired, parse_errors = _extract_object(raw)
    if payload is None:
        return None, parse_errors, repaired
    stripped = {str(key).strip(): value for key, value in payload.items()}
    if set(stripped) != set(payload):
        repaired = True
    aliases = {
        "answer": "public_answer",
        "best_reason": "public_answer",
        "best_image": "recommended_images",
        "best_image_label": "recommended_images",
        "best_asset_id": "recommended_images",
        "refs": "image_references",
        "asset_references": "image_references",
        "recommendations": "recommended_images",
    }
    normalized_keys: dict[str, Any] = {}
    for key, value in stripped.items():
        target = aliases.get(key, key)
        repaired = repaired or target != key
        if target not in normalized_keys:
            normalized_keys[target] = value
    payload = normalized_keys
    allowed = set(allowed_labels)
    public_answer = payload.get("public_answer")
    if not isinstance(public_answer, str) or not public_answer.strip():
        return None, ["public_answer_empty"], repaired
    public_answer = " ".join(public_answer.split())

    recommendations_raw, changed = _list_value(payload.get("recommended_images"))
    repaired = repaired or changed
    recommendations = []
    for value in recommendations_raw:
        label, label_changed = _label_value(value, allowed)
        repaired = repaired or label_changed
        if label and label not in recommendations:
            recommendations.append(label)

    refs_raw, changed = _list_value(payload.get("image_references"))
    repaired = repaired or changed
    references = []
    for value in refs_raw:
        if isinstance(value, dict):
            value = value.get("image_label") or value.get("label")
            repaired = True
        label, label_changed = _label_value(value, allowed)
        repaired = repaired or label_changed
        if label and label not in references:
            references.append(label)
    answer_labels = []
    for match in _LABEL.finditer(public_answer):
        label = f"IMG_{int(match.group(1))}"
        if label not in answer_labels:
            answer_labels.append(label)
        if match.group(0).upper() != label:
            public_answer = public_answer.replace(match.group(0), label)
            repaired = True
    for match in re.finditer(r"第([一二两三四五12345])张", public_answer):
        label = f"IMG_{_ORDINAL[match.group(1)]}"
        if label in allowed and label not in answer_labels:
            answer_labels.append(label)
    if not references and answer_labels:
        references = answer_labels
        repaired = True
    if (
        not recommendations
        and state["requested_action"] in {"select_one", "select_top_k"}
        and len(answer_labels)
        == max(1, int(state.get("selection_count") or 1))
    ):
        recommendations = answer_labels
        repaired = True

    ranking_raw, changed = _list_value(payload.get("ranking"))
    repaired = repaired or changed
    ranking = []
    for index, item in enumerate(ranking_raw, start=1):
        if isinstance(item, str):
            label, label_changed = _label_value(item, allowed)
            repaired = True or label_changed
            if label:
                ranking.append({"image_label": label, "rank": index, "reason": ""})
        elif isinstance(item, dict):
            label_value = item.get("image_label") or item.get("label") or item.get("image")
            label, label_changed = _label_value(label_value, allowed)
            repaired = repaired or label_changed or "image_label" not in item
            rank = item.get("rank", index)
            if isinstance(rank, str) and rank.isdigit():
                rank = int(rank)
                repaired = True
            if label:
                ranking.append(
                    {
                        "image_label": label,
                        "rank": rank,
                        "reason": str(item.get("reason") or "").strip(),
                    }
                )

    evidence, changed = _list_value(payload.get("evidence"))
    repaired = repaired or changed
    normalized_evidence = [
        str(item).strip() for item in evidence if str(item).strip()
    ]
    uncertainty, changed = _list_value(payload.get("uncertainty"))
    repaired = repaired or changed
    needs_clarification = payload.get("needs_clarification", False)
    if not isinstance(needs_clarification, bool):
        needs_clarification = str(needs_clarification).strip().lower() in {"true", "1", "yes"}
        repaired = True
    action_completed = payload.get("action_completed")
    if not isinstance(action_completed, bool):
        action_completed = bool(public_answer) and not needs_clarification
        repaired = True
    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        intent = state["current_intent"]
        repaired = True

    if (
        state["requested_action"] in {"select_one", "select_top_k"}
        and recommendations
        and not ranking
    ):
        ranking = [
            {
                "image_label": label,
                "rank": index,
                "reason": "",
            }
            for index, label in enumerate(recommendations, start=1)
        ]
        repaired = True

    decision_action = state["requested_action"]
    if (
        decision_action
        in {"select_one", "select_top_k", "rank_all", "compare"}
        and not needs_clarification
        and not _has_public_decision_reason(public_answer)
    ):
        reason_candidates = [
            str(item.get("reason") or "").strip()
            for item in ranking
            if isinstance(item, dict)
            and str(item.get("reason") or "").strip()
        ]
        reason_candidates.extend(normalized_evidence)
        if reason_candidates:
            reason = reason_candidates[0].strip("。；; ")
            prefix = (
                "理由是"
                if decision_action in {"select_one", "select_top_k"}
                else "主要依据是"
            )
            public_answer = (
                public_answer.rstrip("。；; ")
                + f"。{prefix}{reason}。"
            )
            repaired = True

    return {
        "public_answer": public_answer,
        "intent": intent.strip(),
        "action_completed": action_completed,
        "recommended_images": recommendations,
        "ranking": ranking,
        "image_references": references,
        "evidence": normalized_evidence,
        "uncertainty": [str(item).strip() for item in uncertainty if str(item).strip()],
        "needs_clarification": needs_clarification,
    }, [], repaired


def validate_common_response(
    payload: dict[str, Any],
    *,
    state: dict[str, Any],
    allowed_labels: list[str],
    selected_assets: list[dict[str, Any]],
    all_bindings: list[dict[str, Any]],
) -> list[str]:
    """Validate semantics, completion, references, safety and identity isolation."""

    errors: list[str] = []
    if set(payload) != COMMON_FIELDS:
        errors.append("common_contract_fields_mismatch")
    allowed = set(allowed_labels)
    references = list(payload.get("image_references", []))
    recommendations = list(payload.get("recommended_images", []))
    ranking = list(payload.get("ranking", []))
    referenced = references + recommendations + [
        str(item.get("image_label")) for item in ranking if isinstance(item, dict)
    ]
    unknown = sorted({label for label in referenced if label not in allowed})
    if unknown:
        errors.append(f"unknown_image_reference:{','.join(unknown)}")
    if len(references) != len(set(references)):
        errors.append("duplicate_image_reference")
    action = state["requested_action"]
    if action in {"select_one", "select_top_k"}:
        selection_count = max(
            1,
            min(int(state.get("selection_count") or 1), len(allowed)),
        )
        if len(recommendations) != selection_count:
            errors.append(
                f"selection_requires_exact_count:{selection_count}"
            )
        elif any(label not in references for label in recommendations):
            errors.append("recommended_image_missing_from_references")
        ranking_labels = [
            str(item.get("image_label"))
            for item in ranking
            if isinstance(item, dict)
        ]
        ranking_positions = [
            item.get("rank")
            for item in ranking
            if isinstance(item, dict)
        ]
        if ranking_labels != recommendations:
            errors.append("selection_ranking_order_mismatch")
        if ranking_positions != list(range(1, selection_count + 1)):
            errors.append("selection_ranking_positions_incomplete")
    if action == "rank_all":
        labels = [str(item.get("image_label")) for item in ranking if isinstance(item, dict)]
        ranks = [item.get("rank") for item in ranking if isinstance(item, dict)]
        if set(labels) != allowed or len(labels) != len(allowed):
            errors.append("ranking_image_coverage_incomplete")
        if set(ranks) != set(range(1, len(allowed) + 1)):
            errors.append("ranking_positions_incomplete")
    if action == "compare" and set(references) != allowed:
        errors.append("compare_image_coverage_incomplete")
    if not payload.get("action_completed") and not payload.get("needs_clarification"):
        errors.append("incomplete_action_without_clarification")
    public_answer = str(payload.get("public_answer") or "")
    if (
        action in {"select_one", "select_top_k", "rank_all", "compare"}
        and not payload.get("needs_clarification")
        and not _has_public_decision_reason(public_answer)
    ):
        errors.append("public_answer_missing_decision_reason")
    if INTERNAL_LANGUAGE.search(public_answer):
        errors.append("internal_language_leak")
    serialized = json.dumps(payload, ensure_ascii=False)
    known_labels = {str(item.get("image_label")) for item in all_bindings}
    for binding in all_bindings:
        for key in ("asset_id", "source_asset_id", "ref", "sha256", "image_id"):
            value = str(binding.get(key) or "")
            if value and value not in known_labels and value in serialized:
                errors.append(f"backend_identity_leak:{key}")
    # Scan only user-facing values.  Scanning serialized JSON keys caused
    # short OCR candidates such as ``ac`` to match ``action_completed`` and
    # reject an otherwise valid visual answer.
    public_text = "\n".join(
        [
            str(payload.get("public_answer") or ""),
            *[
                str(item)
                for item in payload.get("evidence", [])
            ],
            *[
                str(item)
                for item in payload.get("uncertainty", [])
            ],
            *[
                str(item.get("reason") or "")
                for item in payload.get("ranking", [])
                if isinstance(item, dict)
            ],
        ]
    )
    generic_text = {"文字", "文本", "招牌", "牌匾", "不可辨文字", "无法辨认"}
    for asset in selected_assets:
        if asset.get("verified_text"):
            continue
        for candidate in collect_text_candidates(
            {
                "presence": asset.get("text_presence")
                or asset.get("presence")
                or "uncertain",
                "ocr_candidates": asset.get("ocr_candidates", []),
                "visual_candidates": asset.get(
                    "visible_text_candidates", []
                ),
            }
        ):
            text = candidate.text
            if (
                len(text) >= 2
                and text not in generic_text
                and text in public_text
                and not candidate_is_allowed(
                    candidate,
                    task_mode=COMPARE,
                    qualified=is_qualified_statement(public_text),
                )
            ):
                errors.append(f"unverified_text_claim:{asset.get('image_label')}")
                break
    return sorted(set(errors))


def attach_public_assets(
    payload: dict[str, Any],
    selected_assets: list[dict[str, Any]],
    *,
    answer_source: str,
) -> dict[str, Any]:
    """Attach backend identities only after the model-facing contract passed."""

    by_label = {str(item["image_label"]): item for item in selected_assets}
    result = dict(payload)
    result["image_references"] = [
        {
            "image_label": label,
            "asset": {
                "ref": by_label[label]["ref"],
                "asset_id": by_label[label]["asset_id"],
                "source": by_label[label]["source"],
                "sha256": by_label[label]["sha256"],
                "image_url": by_label[label]["image_url"],
            },
        }
        for label in payload["image_references"]
        if label in by_label
    ]
    result["answer"] = result["public_answer"]
    result["answer_source"] = answer_source
    return result


def safe_asset_facts(selected_assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Execute the safe asset facts operation."""
    keys = ("global_observation", "global_scene", "subjects", "main_subjects", "activities", "relations", "attributes")
    rows = []
    for asset in selected_assets:
        facts = asset.get("facts") if isinstance(asset.get("facts"), dict) else {}
        blocked = {
            item.text
            for item in collect_text_candidates(
                {
                    "presence": asset.get("text_presence")
                    or asset.get("presence")
                    or "uncertain",
                    "ocr_candidates": asset.get("ocr_candidates", []),
                    "visual_candidates": asset.get(
                        "visible_text_candidates", []
                    ),
                }
            )
            if item.confidence != "high" and not asset.get("verified_text")
        }
        snippets = []
        for key in keys:
            value = facts.get(key)
            if not isinstance(value, str):
                continue
            compact = " ".join(value.split()).strip("。；; ")
            if (
                compact
                and not compact.startswith("not_available")
                and not any(text and text in compact for text in blocked)
                and compact not in snippets
            ):
                snippets.append(compact)
            if len(snippets) == 2:
                break
        rows.append({"image_label": asset["image_label"], "safe_facts": snippets})
    return rows


_CRITERION_SEMANTIC_TERMS = {
    "travel": {
        "criterion": ("旅游", "旅行", "风景", "游记", "景点", "度假"),
        "positive": (
            "风景", "自然", "山", "湖", "海", "天空", "云", "草地", "树林",
            "建筑", "街道", "城市", "户外", "远景", "景观",
        ),
        "negative": (
            "示波器", "波形", "仪器", "屏幕", "图表", "电路", "文档", "表格",
        ),
    },
    "holiday": {
        "criterion": ("节日", "节假日", "庆祝", "庆典", "新年", "春节"),
        "positive": (
            "灯笼", "烟花", "彩灯", "装饰", "礼物", "红色", "聚会", "庆祝",
            "节日", "笑容", "人群", "热闹",
        ),
        "negative": ("示波器", "仪器", "图表", "文档", "表格", "电路"),
    },
    "social": {
        "criterion": ("朋友圈", "首图", "封面", "社交", "分享"),
        "positive": (
            "风景", "人物", "自然", "天空", "建筑", "花", "食物", "色彩",
            "笑容", "户外", "城市", "氛围",
        ),
        "negative": ("示波器", "波形", "表格", "文档", "电路", "截图"),
    },
}


def _grounded_decision_rows(
    rows: list[dict[str, Any]],
    *,
    criterion: str,
) -> list[dict[str, Any]]:
    """Rank existing safe facts for a daily-use criterion without new claims."""

    compact_criterion = "".join(criterion.lower().split())
    semantic = next(
        (
            value
            for value in _CRITERION_SEMANTIC_TERMS.values()
            if any(token in compact_criterion for token in value["criterion"])
        ),
        None,
    )
    ranked: list[dict[str, Any]] = []
    for original_index, row in enumerate(rows):
        facts = "；".join(row["safe_facts"])
        compact_facts = "".join(facts.lower().split())
        score = min(len(row["safe_facts"]), 3) * 0.05
        matched: list[str] = []
        if semantic is not None:
            for token in semantic["positive"]:
                if token in compact_facts:
                    score += 1.0
                    matched.append(token)
            for token in semantic["negative"]:
                if token in compact_facts:
                    score -= 1.0
        else:
            criterion_tokens = {
                token
                for token in re.findall(r"[\u4e00-\u9fff]{2,6}", compact_criterion)
                if token not in {"图片", "选择", "排序", "适合", "作为", "哪张"}
            }
            for token in criterion_tokens:
                if token in compact_facts:
                    score += 1.0
                    matched.append(token)
        reason = (
            f"可见内容与“{criterion}”的用途更贴合"
            if matched
            else (
                f"画面主体和场景与“{criterion}”更贴合"
                if facts
                else "当前缺少可确认的画面事实"
            )
        )
        ranked.append(
            {
                "image_label": row["image_label"],
                "score": score,
                "reason": reason,
                "fact": row["safe_facts"][0] if row["safe_facts"] else "",
                "original_index": original_index,
            }
        )
    return sorted(
        ranked,
        key=lambda item: (-item["score"], item["original_index"]),
    )


def task_preserving_fallback(
    selected_assets: list[dict[str, Any]],
    *,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Level 4: preserve the current task or ask one honest clarification."""

    labels = [str(item["image_label"]) for item in selected_assets]
    rows = safe_asset_facts(selected_assets)
    action = state["requested_action"]
    raw_criterion = str(state.get("comparison_criterion") or "")
    criterion = (
        ""
        if "（默认标准）" in raw_criterion
        else raw_criterion.strip()
    )
    snippets = {row["image_label"]: row["safe_facts"] for row in rows}
    references: list[str] = []
    evidence: list[str] = []
    recommended_images: list[str] = []
    ranking: list[dict[str, Any]] = []
    if action == "compare" and len(labels) >= 2 and all(snippets[label] for label in labels):
        clauses = [f"{label} 主要呈现{snippets[label][0]}" for label in labels]
        public = "从目前可确认的画面内容看，" + "；".join(clauses) + "。"
        public += "它们的主要差异在于主体或场景内容；若要判断哪张更好，还需要明确评价标准。"
        references = labels
        evidence = [f"{label}：{snippets[label][0]}" for label in labels]
        completed, clarification = True, False
    elif (
        action in {"select_one", "select_top_k", "rank_all"}
        and criterion
        and any(snippets.values())
    ):
        decision_criterion = criterion or str(state.get("current_goal") or "画面用途")
        ranked_rows = _grounded_decision_rows(rows, criterion=decision_criterion)
        if action in {"select_one", "select_top_k"}:
            requested_count = max(
                1,
                min(int(state.get("selection_count") or 1), len(ranked_rows)),
            )
            chosen = ranked_rows[:requested_count]
            references = [item["image_label"] for item in chosen]
            evidence = [
                f"{item['image_label']}：{item['fact']}"
                for item in chosen
                if item["fact"]
            ]
            recommended_images = references
            ranking = [
                {
                    "image_label": item["image_label"],
                    "rank": index,
                    "reason": item["reason"],
                }
                for index, item in enumerate(chosen, start=1)
            ]
            if requested_count == 1:
                public = (
                    f"更推荐 {chosen[0]['image_label']}，"
                    f"{chosen[0]['reason']}。"
                )
            else:
                public = (
                    "更推荐"
                    + "、".join(item["image_label"] for item in chosen)
                    + "，它们的画面主体和场景与用途更贴合。"
                )
        else:
            references = [item["image_label"] for item in ranked_rows]
            evidence = [
                f"{item['image_label']}：{item['fact']}"
                for item in ranked_rows
                if item["fact"]
            ]
            recommended_images = []
            ranking = [
                {
                    "image_label": item["image_label"],
                    "rank": index,
                    "reason": item["reason"],
                }
                for index, item in enumerate(ranked_rows, start=1)
            ]
            public = (
                "完整排序为："
                + "、".join(
                    f"{index}. {item['image_label']}"
                    for index, item in enumerate(ranked_rows, start=1)
                )
                + "。排序依据是各图可见主体和场景与用途的匹配度。"
            )
        completed, clarification = True, False
    elif action in {"select_one", "select_top_k", "rank_all"}:
        public = (
            "请补充你想按什么用途或标准来判断，"
            "例如旅游配图、节日庆祝配图或朋友圈首图。"
        )
        completed, clarification = False, True
    elif (
        len(labels) >= 2
        and all(snippets.get(label) for label in labels)
    ):
        observations = [
            str(snippets[label][0]).rstrip("。")
            for label in labels
        ]
        if any(
            token in str(state.get("current_goal") or "")
            for token in ("共同点", "相同点", "共性")
        ):
            public = (
                "这些画面都有清晰可辨的主体，"
                "但题材和表达方式并不完全相同："
                + "；".join(observations)
                + "。"
            )
        elif any(
            token in str(state.get("current_goal") or "")
            for token in ("整体", "感觉", "氛围", "表达")
        ):
            public = (
                "整体看，这组图片的内容跨度较大，"
                "既有"
                + "，也有".join(observations)
                + "，因此呈现出多样而非单一的视觉气质。"
            )
        else:
            public = (
                "结合这些画面，可以确认的主要内容是："
                + "；".join(observations)
                + "。"
            )
        references = labels
        evidence = [
            f"{label}：{snippets[label][0]}"
            for label in labels
        ]
        completed, clarification = True, False
    elif labels and snippets.get(labels[0]):
        facts = snippets[labels[0]]
        goal = str(state.get("current_goal") or "")
        if any(token in goal for token in ("作用", "功能", "用途")):
            public = (
                f"从画面可确认的是，{facts[0]}。"
                "至于它的具体作用，当前画面没有提供足够信息，"
                "因此不宜进一步猜测。"
            )
        elif len(facts) > 1:
            public = (
                f"画面主要呈现{facts[0]}，"
                f"还能看到{facts[1]}。"
            )
        else:
            public = f"画面主要呈现{facts[0]}。"
        references = [labels[0]]
        evidence = [f"{labels[0]}：{snippets[labels[0]][0]}"]
        completed, clarification = True, False
    else:
        public = "目前没有足够的可靠画面信息来回答这个问题。请重新选择图片后再试。"
        completed, clarification = False, True
    return {
        "public_answer": public,
        "intent": state["current_intent"],
        "action_completed": completed,
        "recommended_images": recommended_images,
        "ranking": ranking,
        "image_references": references,
        "evidence": evidence,
        "uncertainty": ["仅使用当前可确认的画面信息。"],
        "needs_clarification": clarification,
    }
