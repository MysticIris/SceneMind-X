"""Phase 5.2 course-first prompt, context, routing, and contract helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CANDIDATE_ID = "SCENEMINDX_PHASE5_2_COURSE_PROMPT_CANDIDATE_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CoursePromptCandidate:
    """Provide course prompt candidate behavior."""
    def __init__(self, project_root: Path) -> None:
        self.root = project_root / "prompts" / "phase5_2" / "course_prompt_candidate_v1"
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("candidate_id") != CANDIDATE_ID:
            raise ValueError("phase5_2_candidate_identity_mismatch")
        self.prompts: dict[str, dict[str, Any]] = {}
        for spec in self.manifest.get("prompts", []):
            prompt_id = str(spec["prompt_id"])
            path = project_root / str(spec["file"])
            actual = _sha256(path)
            if actual != str(spec["raw_sha256"]):
                raise ValueError(f"phase5_2_candidate_sha256_mismatch:{prompt_id}")
            self.prompts[prompt_id] = {**spec, "path": path, "text": path.read_text(encoding="utf-8")}

    def identity(self, prompt_id: str) -> dict[str, str]:
        """Execute the identity operation."""
        spec = self.prompts[prompt_id]
        return {
            "candidate_id": CANDIDATE_ID,
            "prompt_id": prompt_id,
            "prompt_sha256": str(spec["raw_sha256"]),
            "status": str(self.manifest["status"]),
        }

    def render(self, prompt_id: str, values: dict[str, Any]) -> tuple[str, dict[str, str]]:
        """Render the requested value."""
        text = str(self.prompts[prompt_id]["text"])
        for key, value in values.items():
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
            text = text.replace("{{" + key + "}}", rendered)
        unresolved = [token for token in ("{{ASSET_CONTEXT}}", "{{CONTEXT_SUMMARY}}", "{{RECENT_HISTORY}}", "{{CURRENT_QUESTION}}", "{{GENERATION_OPTIONS}}", "{{CRITERION}}") if token in text]
        if unresolved:
            raise ValueError(f"phase5_2_unresolved_prompt_placeholders:{','.join(unresolved)}")
        return text, self.identity(prompt_id)


def classify_intent(message: str) -> str:
    """Execute the classify intent operation."""
    normalized = "".join(message.lower().split())
    for marker in ("现在只回答", "只回答", "现在只问", "只问"):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1]
            break
    rules = (
        ("rank", ("排序", "排名", "从好到差", "从高到低", "完整顺序")),
        ("recommend", ("最匹配", "最佳", "最好", "哪张最", "哪张更", "哪一个", "选出一张", "选一张", "只选一", "推荐一张")),
        ("compare", ("比较", "对比", "共同点", "差异", "区别")),
        ("retrieve", ("检索", "搜索", "搜图", "找图", "图片库", "相似图片", "相似的图")),
        ("generate", ("生成", "写一", "写个", "文案", "朋友圈", "游记", "图注", "标题", "诗", "故事", "广告")),
    )
    for intent, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return intent
    return "vqa"


def build_context_plan(
    session: dict[str, Any],
    current_question: str,
    active_assets: list[dict[str, Any]],
    *,
    recent_limit: int = 8,
    summary_limit: int = 1600,
) -> dict[str, Any]:
    """Build context plan."""
    active_refs = {item["ref"] for item in active_assets}
    messages = list(session.get("messages", []))
    relevant: list[dict[str, Any]] = []
    for message in messages:
        refs = set(message.get("asset_refs", []))
        if not refs or refs & active_refs:
            relevant.append(message)
    recent = relevant[-recent_limit:]
    old = relevant[:-recent_limit]
    existing_summary = str(session.get("context_summary", "")).strip()
    summary_parts = [existing_summary] if existing_summary else []
    for message in old:
        content = message.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        compact = " ".join(content.split())
        summary_parts.append(f"{message.get('role', 'unknown')}:{compact[:220]}")
    context_summary = " | ".join(part for part in summary_parts if part)[-summary_limit:]
    return {
        "current_question": current_question,
        "active_asset_refs": [item["ref"] for item in active_assets],
        "recent_messages": recent,
        "context_summary": context_summary,
        "structured_tool_results": list(session.get("tool_results", []))[-3:],
        "pruned_message_count": len(messages) - len(recent),
        "current_question_repeated_near_output_contract": True,
    }


def compact_asset_context(assets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Execute the compact asset context operation."""
    result = []
    for item in assets:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        evidence = {
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
        result.append(
            {
                "image_position": item["order"],
                "asset_id": item["asset_id"],
                "ref": item["ref"],
                "source": item["source"],
                "sha256": item["sha256"],
                "direct_or_candidate_facts": evidence,
                "verified_text": item.get("verified_text", []),
                "ocr_candidates": item.get("ocr_candidates", []),
            }
        )
    return result


def normalize_course_chat_answer(
    model_payload: Any,
    assets: list[dict[str, Any]],
    *,
    intent: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the model answer and transparently fall back to stored P3 facts."""

    asset_ids = [str(item["asset_id"]) for item in assets]
    errors: list[str] = []
    if not isinstance(model_payload, dict):
        errors.append("answer_payload_not_object")
    else:
        answer = model_payload.get("answer")
        references = model_payload.get("asset_references")
        if not isinstance(answer, str) or not answer.strip():
            errors.append("answer_empty")
        if not isinstance(references, list) or not references:
            errors.append("asset_references_empty")
        else:
            reference_ids = [
                str(item.get("asset_id"))
                for item in references
                if isinstance(item, dict) and item.get("asset_id") is not None
            ]
            if any(asset_id not in asset_ids for asset_id in reference_ids):
                errors.append("unknown_asset_reference")
            if intent == "compare" and set(reference_ids) != set(asset_ids):
                errors.append("compare_asset_coverage_incomplete")
        if not isinstance(model_payload.get("refused"), bool):
            errors.append("refused_flag_missing")
    if not errors:
        return dict(model_payload), {
            "model_contract_valid": True,
            "product_contract_valid": True,
            "fallback_applied": False,
            "fallback_source": None,
            "contract_errors": [],
        }

    references: list[dict[str, Any]] = []
    clauses: list[str] = []
    fact_keys = (
        "global_observation",
        "global_scene",
        "subjects",
        "main_subjects",
        "activities",
        "relations",
        "attributes",
    )
    for item in assets:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        evidence: list[str] = []
        for key in fact_keys:
            value = facts.get(key)
            if not isinstance(value, str):
                continue
            value = " ".join(value.split()).strip("。；; ")
            if not value or value.startswith("not_available") or value in evidence:
                continue
            evidence.append(value)
            if len(evidence) == 3:
                break
        if evidence:
            clauses.append(f"{item['asset_id']}：{'；'.join(evidence)}")
        references.append(
            {
                "asset_id": str(item["asset_id"]),
                "evidence": evidence or ["当前没有可用的结构化视觉事实"],
                "evidence_status": "existing_p3_candidate_not_ground_truth",
            }
        )
    has_evidence = any(item["evidence"][0] != "当前没有可用的结构化视觉事实" for item in references)
    if intent == "compare" and len(clauses) > 1:
        answer = "根据现有 P3 候选事实，各图核心场景如下：" + "；".join(clauses) + "。"
    elif clauses:
        answer = "根据现有 P3 候选事实，" + "；".join(clauses) + "。"
    else:
        answer = "远端模型未给出有效回答，且当前没有可用的结构化视觉事实；请先完成图片分析后再试。"
    fallback = {
        "answer": answer,
        "asset_references": references,
        "uncertainty": [
            "远端模型返回的回答合同无效；当前内容由已有 P3 候选事实确定性降级生成，不是新的模型质量结论。"
        ],
        "refused": not has_evidence,
        "answer_source": "deterministic_existing_p3_fact_fallback",
    }
    return fallback, {
        "model_contract_valid": False,
        "product_contract_valid": True,
        "fallback_applied": True,
        "fallback_source": "existing_p3_candidate_facts",
        "contract_errors": errors,
    }


def normalize_course_generation(
    model_payload: Any,
    assets: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize interface-only generation fields while preserving raw model data."""

    asset_ids = [str(item["asset_id"]) for item in assets]
    payload = dict(model_payload) if isinstance(model_payload, dict) else {}
    model_valid, model_errors = validate_asset_coverage(payload, asset_ids)
    content = payload.get("content")
    fallback_applied = False
    fallback_source: str | None = None
    if not isinstance(content, str) or not content.strip():
        fallback, _ = normalize_course_chat_answer({}, assets, intent="vqa")
        content = fallback["answer"]
        fallback_applied = True
        fallback_source = "existing_p3_candidate_facts"
        model_errors.append("content_empty")

    raw_facts = payload.get("direct_facts")
    fact_rows = raw_facts if isinstance(raw_facts, list) else []
    direct_facts = []
    for index, asset_id in enumerate(asset_ids):
        source = fact_rows[index] if index < len(fact_rows) and isinstance(fact_rows[index], dict) else {}
        facts = source.get("facts")
        direct_facts.append(
            {
                "asset_id": asset_id,
                "facts": [str(item) for item in facts] if isinstance(facts, list) else [],
            }
        )
    relation = payload.get("cross_image_relation", payload.get("__cross_image_relation", "independent"))
    relation_map = {"独立": "independent", "共同事件": "evidenced_shared_event", "创意串联": "creative_sequence"}
    relation = relation_map.get(str(relation), str(relation))
    if relation not in {"independent", "evidenced_shared_event", "creative_sequence"}:
        relation = "independent"
    normalized = {
        "content": str(content).strip(),
        "direct_facts": direct_facts,
        "narrative_organization": payload.get(
            "narrative_organization",
            payload.get("narrative_organizaton", []),
        ),
        "creative_expression": payload.get("creative_expression", []),
        "unknowns": payload.get("unknowns", payload.get("_unknowns", [])),
        "cross_image_relation": relation,
        "asset_ids": asset_ids,
    }
    product_valid = bool(normalized["content"]) and normalized["asset_ids"] == asset_ids
    normalization_applied = normalized != payload
    return normalized, {
        "model_contract_valid": model_valid and not model_errors,
        "product_contract_valid": product_valid,
        "normalization_applied": normalization_applied,
        "fallback_applied": fallback_applied,
        "fallback_source": fallback_source,
        "model_contract_errors": model_errors,
    }


def validate_asset_coverage(payload: dict[str, Any], asset_ids: list[str]) -> tuple[bool, list[str]]:
    """Validate asset coverage."""
    output_ids = payload.get("asset_ids")
    if not isinstance(output_ids, list):
        return False, ["asset_ids_missing"]
    normalized = [str(value) for value in output_ids]
    errors = []
    if normalized != asset_ids:
        errors.append(f"asset_ids_contract_violation:expected={asset_ids},got={normalized}")
    return not errors, errors


def validate_ranking(payload: dict[str, Any], asset_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate ranking."""
    ranking = payload.get("ranking")
    errors: list[str] = []
    if not isinstance(ranking, list) or not all(isinstance(item, dict) for item in ranking):
        return [], ["ranking_missing_or_invalid"]
    ids = [str(item.get("asset_id")) for item in ranking]
    ranks = [item.get("rank") for item in ranking]
    if len(ranking) != len(asset_ids) or set(ids) != set(asset_ids) or len(ids) != len(set(ids)):
        errors.append(f"ranking_asset_contract_violation:expected={asset_ids},got={ids}")
    if set(ranks) != set(range(1, len(asset_ids) + 1)):
        errors.append(f"ranking_rank_contract_violation:got={ranks}")
    for item in ranking:
        score = item.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= float(score) <= 100:
            errors.append(f"ranking_score_contract_violation:{item.get('asset_id')}={score}")
    ordered = sorted(ranking, key=lambda item: item.get("rank", 10**9))
    if ordered and payload.get("best_asset_id") != ordered[0].get("asset_id"):
        errors.append("best_asset_id_contract_violation")
    return ordered, errors
