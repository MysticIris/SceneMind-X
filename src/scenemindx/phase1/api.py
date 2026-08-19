"""FastAPI gateway for SceneMind-X Studio Phase 1."""

from __future__ import annotations

from contextlib import asynccontextmanager
import copy
from datetime import datetime
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
import threading
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field

from scenemindx.services.embedding import (
    ChineseCLIPEmbeddingService,
    DeterministicVisualEmbeddingService,
    DisabledEmbeddingService,
    ModelScopeChineseCLIPEmbeddingService,
)
from scenemindx.services.e1_retrieval import E1RetrievalAdapter, FaissRetrievalIndex
from scenemindx.services.cloud_retrieval import BailianCloudRetrieval
from scenemindx.services.cloud_image_transport import (
    CloudImageTransportPreprocessor,
    CloudTransportBudgetExceeded,
)
from scenemindx.services.local_models import (
    LocalQwenVLEmbeddingService,
    local_index_reuse_contract,
    local_model_preflight,
)
from scenemindx.services.portable_assets import PortableAssetResolver
from scenemindx.services.remote_embedding import RemoteQwenVLEmbeddingService
from scenemindx.services.retrieval import RetrievalService
from scenemindx.services.vlm import DisabledVLMService, PersistentQwen3VLService
from scenemindx.services.remote_vlm import RemoteVLMService
from scenemindx.retrieval.candidate_contract import (
    adaptive_topk_refill,
    candidate_eligibility,
)
from scenemindx.services.bailian import (
    BailianEmbeddingProvider,
    BailianProviderFailure,
    BailianVLMProvider,
    credentials_from_user,
    load_course_credentials,
)

from .library import LibraryRepository
from .asset_media import AssetMediaResolver
from .provider_index_backfill import ProviderIndexBackfillManager
from .multi_library import SYSTEM_LIBRARY_IDS, SystemVisualLibraryRepository
from .orchestrator import Orchestrator
from .product_store import (
    LibraryMutationForbidden,
    ProductStore,
    WorkspaceVersionConflict,
    now_iso,
)
from .split_retrieval import SplitE1IndexRegistry
from .course import (
    CoursePromptCandidate,
    build_context_plan,
    classify_intent,
    compact_asset_context,
    normalize_course_chat_answer,
    normalize_course_generation,
    validate_ranking,
)
from .canonical_preview import CanonicalPreviewRepository
from .conversational_response import (
    ConversationalResponsePromptCandidate,
    attach_public_assets,
    deterministic_contract_repair,
    infer_current_turn_state,
    safe_asset_facts,
    task_preserving_fallback,
    validate_common_response,
)

from .multiturn_chat import (
    CHAT_PROMPT_ID,
    MultiturnChatPromptCandidate,
    build_multiturn_context,
    continue_decision_state_after_follow_up,
    deterministic_chat_fallback,
    resolve_decision_follow_up,
    resolve_image_references,
    update_chat_state_after_turn,
    update_decision_state_after_turn,
    validate_chat_model_output,
)
from .conversation_followup import (
    ContextualQueryRewriterCandidate,
    clean_rewriter_inputs,
    parse_contextual_rewrite,
    resolve_general_follow_up,
    update_general_turn_state,
)
from .session_ledger import (
    append_completed_turn,
    context_projection,
    render_context_projection,
    validate_model_bindings,
)
from .multi_image_content import (
    FRIENDLY_FAILURE_TEXT,
    MultiImageContentV2Candidate,
    MultiImageStoryV3Candidate,
    adjust_final_text_only,
    append_safe_short_bridge,
    build_text_risk_generalization,
    candidate_quality_score,
    extract_multi_image_payload,
    extract_story_public_payload,
    merge_final_text_revision,
    merge_metadata_completion,
    merge_model_authored_addition,
    model_min_token_budget,
    model_token_budget,
    render_final_text_revision_prompt,
    render_metadata_completion_prompt,
    validate_multi_image_content,
    visible_character_count,
)
from .multi_image_intent import resolve_content_type
from .chat_tool_router import (
    ChatToolRouterCandidate,
    detect_system_utility,
    deterministic_tool_plan,
    merge_router_decisions,
    parse_direct_chat_output,
    parse_router_output,
    validate_router_decision_against_state,
    visual_evidence_required,
    visual_groundable_intent,
)
from .content_profiles import ContentProfileRegistry
from .content_length_profiles import (
    ideal_output_window,
    normalize_target_length,
    public_content_length_config,
)
from .resources import RuntimeResourceMonitor
from .settings import Phase1Settings
from .tracing import TraceStore
from .provider_access import (
    CapabilityUnavailable,
    ProviderError,
    ProviderManager,
    ProviderRetrievalProxy,
    ProviderSwitchBusy,
    ProviderVLMProxy,
    SelfHostedEmbeddingProvider,
    SelfHostedVLMProvider,
    LocalEmbeddingProvider,
    LocalVLMProvider,
    classify_bailian_error,
)
from scenemindx.annotation.canonical_closeout import (
    build_two_layer_display,
    compose_safe_caption,
    is_truncation_failure,
)
from scenemindx.annotation.canonical_label import (
    build_canonical_label,
    extract_json_object,
    normalize_model_payload,
    sha256_text,
)
from scenemindx.annotation.phase6_1_canonical import (
    migrate_phase6_0a_to_phase6_1,
    normalize_recovery_payload,
    upgrade_phase6_1_compatible_label,
    validate_phase6_1_label,
    validate_recovery_payload_safety,
)

BAILIAN_VALIDATION_TTL_SECONDS = max(
    1,
    int(os.environ.get("SCENEMINDX_BAILIAN_VALIDATION_TTL_SECONDS", str(15 * 60))),
)


PUBLIC_TEXT_FAILURE = "本次生成未成功，请稍后重试。"


class CanonicalGenerationError(RuntimeError):
    """Provide canonical generation error behavior."""
    def __init__(self, failure: dict[str, Any]) -> None:
        super().__init__(str(failure["public_message"]))
        self.failure = dict(failure)


def _canonical_failure(
    exc: Exception,
    *,
    request_id: str,
    retry_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    base = {
        "status": "failed",
        "request_id": request_id,
        "attempt_count": len(retry_trace),
        "model_called": bool(retry_trace),
    }
    if isinstance(exc, BailianProviderFailure):
        return {**base, **exc.error.as_dict(), "http_status": exc.http_status}
    if isinstance(exc, CapabilityUnavailable):
        credential = dict(exc.snapshot.get("credential") or {})
        if not credential.get("configured"):
            return {
                **base,
                "category": "missing_credentials",
                "code": "CREDENTIAL_NOT_CONFIGURED",
                "retryable": False,
                "stop_retries": True,
                "public_message": "当前模型凭据尚未配置，请先在模型接入中输入 API Key 或切换可用模式。",
            }
        return {
            **base,
            "category": "waiting_for_provider",
            "code": "PROVIDER_NOT_READY",
            "retryable": True,
            "stop_retries": True,
            "public_message": "当前视觉模型尚未连接，图片和已有结果均已保留。请恢复连接后重新尝试。",
        }
    if isinstance(exc, CloudTransportBudgetExceeded):
        return {
            **base,
            "category": "request_too_large",
            "code": exc.code,
            "retryable": False,
            "stop_retries": True,
            "public_message": exc.public_message,
        }
    last_attempt = retry_trace[-1] if retry_trace else {}
    error_text = str(exc)
    if last_attempt.get("explicit_truncation"):
        return {
            **base,
            "category": "model_output_incomplete",
            "code": "CANONICAL_OUTPUT_TRUNCATED",
            "retryable": True,
            "stop_retries": True,
            "public_message": "本次伪标注输出不完整，暂未保存，可重新尝试。",
        }
    if "model_output_did_not_contain_json_object" in error_text:
        return {
            **base,
            "category": "structured_parse_failed",
            "code": "CANONICAL_JSON_NOT_FOUND",
            "retryable": True,
            "stop_retries": True,
            "public_message": "当前图片的结构化结果未能完整解析，暂未保存，可重新尝试。",
        }
    if "canonical_validation_failed" in error_text:
        return {
            **base,
            "category": "safety_validation_failed",
            "code": "CANONICAL_SAFETY_VALIDATION_FAILED",
            "retryable": True,
            "stop_retries": True,
            "public_message": "本次结构化结果未通过安全检查，暂未保存，可重新尝试。",
        }
    if isinstance(exc, (FileNotFoundError, OSError)):
        return {
            **base,
            "category": "image_processing_failed",
            "code": "CANONICAL_IMAGE_PROCESSING_FAILED",
            "retryable": False,
            "stop_retries": True,
            "public_message": "当前图片未能完成安全读取或处理，请确认图片仍可访问后重新尝试。",
        }
    if isinstance(exc, TimeoutError) or "timeout" in error_text.casefold():
        return {
            **base,
            "category": "network_unavailable",
            "code": "CANONICAL_REQUEST_TIMEOUT",
            "retryable": True,
            "stop_retries": True,
            "public_message": "模型请求超时，图片和已有结果均已保留，请稍后重新尝试。",
        }
    return {
        **base,
        "category": "failed",
        "code": "CANONICAL_GENERATION_FAILED",
        "retryable": True,
        "stop_retries": True,
        "public_message": "本次 Canonical 标注未完成，图片和已有结果均已保留，可以重新尝试。",
    }


def _plain_public_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text[:1] in {"{", "["}:
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, (dict, list)):
            return None
    return text


def extract_display_text(
    payload: Any,
    *,
    fallback: str = PUBLIC_TEXT_FAILURE,
) -> str:
    """Execute the extract display text operation."""
    direct = _plain_public_text(payload)
    if direct is not None:
        return direct
    if not isinstance(payload, dict):
        return fallback

    for key in ("display_text", "final_text", "public_answer"):
        direct = _plain_public_text(payload.get(key))
        if direct is not None:
            return direct

    for container_key in ("answer", "content", "public_result"):
        container = payload.get(container_key)
        direct = _plain_public_text(container)
        if direct is not None:
            return direct
        if isinstance(container, dict):
            for key in ("display_text", "final_text", "public_answer", "answer", "content"):
                direct = _plain_public_text(container.get(key))
                if direct is not None:
                    return direct

    result = payload.get("result")
    if isinstance(result, dict):
        data = result.get("data")
        if isinstance(data, dict):
            direct = _plain_public_text(data.get("final_output"))
            if direct is not None:
                return direct
            parsed = data.get("parsed_output")
            if isinstance(parsed, dict):
                for key in ("display_text", "final_text", "public_answer", "answer"):
                    direct = _plain_public_text(parsed.get(key))
                    if direct is not None:
                        return direct

    results = payload.get("results")
    if isinstance(results, list):
        mode = _plain_public_text(payload.get("mode")) or "retrieval"
        return f"已返回 {len(results)} 个检索结果（{mode}）。"

    ranking = payload.get("ranking")
    if isinstance(ranking, list):
        return f"已完成 {len(ranking)} 张图片排序，最佳结果已标出。"

    return fallback


class AnalyzeRequest(BaseModel):
    """Represent analyze request data."""
    image_id: str
    prompt_version: Literal["p3_v1_4", "p3_v1_3"] | None = None


class SearchRequest(BaseModel):
    """Represent search request data."""
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=5)


class ImageSearchRequest(BaseModel):
    """Represent image search request data."""
    image_id: str | None = None
    local_asset_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=5)
    exclude_self: bool = True


class HybridSearchRequest(ImageSearchRequest):
    """Represent hybrid search request data."""
    query: str = Field(min_length=1, max_length=500)
    image_weight: float = Field(default=0.55, ge=0, le=1)
    text_weight: float = Field(default=0.35, ge=0, le=1)
    lexical_weight: float = Field(default=0.10, ge=0, le=1)


class VQARequest(BaseModel):
    """Represent v q a request data."""
    image_id: str
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = None


class LocalVQARequest(BaseModel):
    """Represent local v q a request data."""
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = None


class GenerateRequest(BaseModel):
    """Represent generate request data."""
    image_ids: list[str] = Field(min_length=1, max_length=4)
    tone: str = Field(default="客观", max_length=80)
    audience: str = Field(default="普通读者", max_length=120)
    length: int = Field(default=120, ge=20, le=500)
    style: str = Field(default="正式", max_length=80)


class DescribeRequest(BaseModel):
    """Represent describe request data."""
    image_id: str
    length: int = Field(default=180, ge=150, le=350)
    style: str = Field(default="自然客观", max_length=80)


class ImportFile(BaseModel):
    """Provide import file behavior."""
    name: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1)


class ImportRequest(BaseModel):
    """Represent import request data."""
    library_id: str = Field(default="default", min_length=1, max_length=80)
    files: list[ImportFile] = Field(min_length=1, max_length=50)


class LibraryRequest(BaseModel):
    """Represent library request data."""
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)


class MoveAssetRequest(BaseModel):
    """Represent move asset request data."""
    target_library_id: str = Field(min_length=1, max_length=80)


class SessionImportRequest(BaseModel):
    """Represent session import request data."""
    conversation_id: str = Field(min_length=1, max_length=120)
    files: list[ImportFile] = Field(min_length=1, max_length=5)


class PersistSessionAssetRequest(BaseModel):
    """Represent persist session asset request data."""
    conversation_id: str = Field(min_length=1, max_length=120)
    library_id: str = Field(min_length=1, max_length=80)


class ComparisonRequest(BaseModel):
    """Represent comparison request data."""
    asset_ids: list[str] = Field(min_length=2, max_length=4)
    instruction: str = Field(default="比较这些图片的共同点与差异", max_length=500)
    dimensions: list[str] = Field(default_factory=lambda: ["scene", "subjects", "activities", "verified_text"])


class RankingRequest(BaseModel):
    """Represent ranking request data."""
    asset_ids: list[str] = Field(min_length=2, max_length=20)
    instruction: str = Field(min_length=1, max_length=500)


class FeedbackRequest(BaseModel):
    """Represent feedback request data."""
    function_id: str = Field(min_length=1, max_length=80)
    asset_ids: list[str] = Field(default_factory=list, max_length=20)
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    rating: Literal["correct", "partially_correct", "incorrect", "missing", "hallucination"]
    error_type: str | None = Field(default=None, max_length=120)
    correction: str | None = Field(default=None, max_length=4000)
    trace_id: str | None = None


class CourseAssetRefRequest(BaseModel):
    """Represent course asset ref request data."""
    source: Literal["library", "local", "system", "session"]
    asset_id: str = Field(min_length=1, max_length=255)
    conversation_id: str | None = Field(default=None, max_length=120)


class CourseConversationRequest(BaseModel):
    """Represent course conversation request data."""
    title: str | None = Field(default=None, max_length=120)
    asset_refs: list[CourseAssetRefRequest] = Field(default_factory=list, max_length=5)


class CourseConversationAssetsRequest(BaseModel):
    """Represent course conversation assets request data."""
    asset_refs: list[CourseAssetRefRequest] = Field(default_factory=list, max_length=5)
    focus_image_label: str | None = Field(default=None, pattern=r"^IMG_[1-9]\d*$")
    operation_id: str | None = Field(default=None, max_length=120)
    workspace_id: str | None = Field(default=None, max_length=120)
    client_sequence: int | None = Field(default=None, ge=0)
    expected_version: int | None = Field(default=None, ge=0)


class CourseChatRequest(BaseModel):
    """Represent course chat request data."""
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    asset_refs: list[CourseAssetRefRequest] | None = Field(default=None, max_length=5)


class CourseGenerateRequest(BaseModel):
    """Represent course generate request data."""
    asset_refs: list[CourseAssetRefRequest] = Field(min_length=1, max_length=5)
    content_type: Literal[
        "auto",
        "objective_description",
        "moments",
        "travel_diary",
        "news_caption",
        "advertisement",
        "poster_title",
        "poem",
        "story",
        "creative_story",
        "article",
    ] = "auto"
    natural_language_request: str = Field(default="", max_length=1000)
    content_type_source: Literal[
        "auto_inferred",
        "default_value",
        "explicit_user_selection",
    ] = "default_value"
    content_type_user_selected: bool = False
    target_length: Any = None
    style: str = Field(default="自然客观", max_length=80)
    audience: str = Field(default="普通读者", max_length=120)
    organization: Literal["input_order", "importance", "chronological_if_evidenced", "independent_panels"] = "input_order"
    importance: list[str] = Field(default_factory=list, max_length=5)
    conversation_id: str | None = None
    call_source: Literal["standalone_workspace", "chat_tool_call"] = (
        "standalone_workspace"
    )
    workspace_id: str | None = Field(default=None, max_length=120)


class CourseRetrieveRequest(BaseModel):
    """Represent course retrieve request data."""
    query_text: str | None = Field(default=None, max_length=500)
    query_asset_ref: CourseAssetRefRequest | None = None
    query_asset_refs: list[CourseAssetRefRequest] = Field(
        default_factory=list,
        max_length=5,
    )
    top_k: int = Field(default=5, ge=1, le=5)
    exclude_query_images: bool = True
    library_scope: Literal[
        "current_library",
        "system_train",
        "system_val",
        "all_libraries",
    ] = "all_libraries"
    current_library_id: str | None = Field(default=None, max_length=80)
    library_ids: list[str] = Field(
        default_factory=list,
        max_length=50,
    )
    call_source: Literal["standalone_workspace", "chat_tool_call"] = (
        "standalone_workspace"
    )
    workspace_id: str | None = Field(default=None, max_length=120)


class VisualLibrarySearchRequest(BaseModel):
    """Represent visual library search request data."""
    query_text: str | None = Field(default=None, max_length=500)
    query_asset_ref: CourseAssetRefRequest | None = None
    library_ids: list[Literal["system_train", "system_val"]] = Field(
        min_length=1,
        max_length=2,
    )
    top_k: int = Field(default=5, ge=1, le=5)
    exclude_query_image: bool = True


class CourseRankRequest(BaseModel):
    """Represent course rank request data."""
    criterion: str = Field(min_length=1, max_length=1000)
    asset_refs: list[CourseAssetRefRequest] = Field(min_length=3, max_length=5)
    conversation_id: str | None = None
    call_source: Literal["standalone_workspace", "chat_tool_call"] = (
        "standalone_workspace"
    )
    workspace_id: str | None = Field(default=None, max_length=120)


class CourseCompareRequest(BaseModel):
    """Represent course compare request data."""
    criterion: str = Field(min_length=1, max_length=1000)
    scenario: str = Field(default="", max_length=500)
    asset_refs: list[CourseAssetRefRequest] = Field(min_length=2, max_length=5)
    action: Literal["compare", "select", "rank"] = "compare"
    select_count: int = Field(default=1, ge=1, le=5)
    call_source: Literal["standalone_workspace", "chat_tool_call"] = (
        "standalone_workspace"
    )
    workspace_id: str | None = Field(default=None, max_length=120)


class FunctionWorkspaceUpdateRequest(BaseModel):
    """Represent function workspace update request data."""
    selected_assets: list[CourseAssetRefRequest] = Field(
        default_factory=list,
        max_length=5,
    )
    local_options: dict[str, Any] = Field(default_factory=dict)
    last_result: dict[str, Any] | None = None


class FunctionWorkspaceOperationRequest(BaseModel):
    """Represent function workspace operation request data."""
    operation_id: str = Field(min_length=1, max_length=120)
    workspace_id: str = Field(min_length=1, max_length=120)
    action: Literal[
        "add",
        "add_many",
        "remove",
        "clear",
        "replace",
        "move",
    ]
    asset: CourseAssetRefRequest | None = None
    asset_ref: str | None = Field(default=None, max_length=300)
    selected_assets: list[CourseAssetRefRequest] = Field(
        default_factory=list,
        max_length=5,
    )
    direction: Literal[-1, 0, 1] = 0
    client_sequence: int = Field(default=0, ge=0)
    expected_version: int | None = Field(default=None, ge=0)
    local_options: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None


class CourseSearchResultImportRequest(BaseModel):
    """Represent course search result import request data."""
    search_labels: list[str] = Field(min_length=1, max_length=5)
    destination: Literal[
        "chat_context",
        "generation_workspace",
        "compare_workspace",
    ]


class CourseSearchResultSelectionRequest(BaseModel):
    """Represent course search result selection request data."""
    search_labels: list[str] = Field(default_factory=list, max_length=5)


class ProviderSelectionRequest(BaseModel):
    """Represent provider selection request data."""
    mode: Literal["no_model", "bailian", "local", "self_hosted"]
    cloud_tier: Literal["standard", "high_quality"] = "standard"
    credential_source: Literal["user_session", "course_default"] = "course_default"
    region: Literal["cn-beijing"] = "cn-beijing"
    api_host_override: str | None = Field(default=None, max_length=500)
    endpoint_mode: Literal["shared", "workspace", "custom"] = "shared"
    workspace_id: str | None = Field(default=None, max_length=128)


class ProviderCredentialRequest(BaseModel):
    """Represent provider credential request data."""
    api_key: str = Field(min_length=12, max_length=500)
    region: Literal["cn-beijing"] = "cn-beijing"
    api_host: str | None = Field(default=None, max_length=500)
    endpoint_mode: Literal["shared", "workspace", "custom"] = "shared"
    workspace_id: str | None = Field(default=None, max_length=128)
    only_this_session: bool = True


class ProviderConnectionTestRequest(BaseModel):
    """Represent provider connection test request data."""
    cloud_tier: Literal["standard", "high_quality"] = "standard"
    credential_source: Literal["user_session", "course_default"] = "course_default"


class CloudIndexBuildRequest(BaseModel):
    """Represent cloud index build request data."""
    target_base_items: Literal[10] = 10


class CloudIndexAssetRequest(BaseModel):
    """Represent cloud index asset request data."""
    asset_ids: list[str] = Field(min_length=1, max_length=50)


class LocalProviderLoadRequest(BaseModel):
    """Represent local provider load request data."""
    force_low_vram_attempt: bool = False


class ProviderIndexBackfillRequest(BaseModel):
    """Represent provider index backfill request data."""
    scope: Literal["asset", "library", "all_user_assets"]
    asset_id: str | None = None
    library_id: str | None = None
    confirmed_by_user: bool = False
    operation_id: str | None = Field(default=None, max_length=128)


def create_app(
    settings: Phase1Settings | None = None,
    *,
    vlm_service: Any | None = None,
    embedding_service: Any | None = None,
    e1_embedding_service: Any | None = None,
) -> FastAPI:
    """Create app."""
    settings = settings or Phase1Settings.from_env()
    injected_vlm_service = vlm_service is not None
    settings.ensure_runtime_dirs()
    settings.write_resolved_config()
    library = LibraryRepository(settings.manifest_path, settings.dataset_root, settings.historical_result_root, settings.ocr_result_root)

    if vlm_service is None:
        if settings.enable_vlm:
            if settings.vlm_endpoint:
                vlm_service = RemoteVLMService(
                    settings.vlm_endpoint,
                    settings.project_root / "prompts" / "phase1",
                    core_registry_path=settings.project_root / "prompts" / "gate1" / "p3_registry.json",
                    core_prompt_version=settings.core_prompt_version,
                    inline_images=settings.vlm_inline_images,
                )
            elif settings.vlm_model_path is None:
                raise ValueError("SCENEMINDX_VLM_MODEL_PATH is required when VLM is enabled")
            else:
                vlm_service = PersistentQwen3VLService(
                    settings.vlm_model_path,
                    settings.project_root / "prompts" / "phase1",
                    gpu_max_memory_gib=settings.vlm_gpu_max_memory_gib,
                    core_registry_path=settings.project_root / "prompts" / "gate1" / "p3_registry.json",
                    core_prompt_version=settings.core_prompt_version,
                )
        else:
            vlm_service = DisabledVLMService()
    if embedding_service is None:
        if settings.enable_embedding:
            embedding_kwargs = {
                "model_id": settings.embedding_model_id,
                "model_revision": settings.embedding_model_revision,
            }
            if settings.embedding_backend == "deterministic_baseline":
                embedding_service = DeterministicVisualEmbeddingService()
            elif settings.embedding_model_path is None:
                raise ValueError("SCENEMINDX_EMBEDDING_MODEL_PATH is required when neural embedding is enabled")
            elif settings.embedding_backend == "transformers":
                embedding_service = ChineseCLIPEmbeddingService(
                    settings.embedding_model_path,
                    **embedding_kwargs,
                )
            elif settings.embedding_backend == "modelscope_iic":
                embedding_service = ModelScopeChineseCLIPEmbeddingService(
                    settings.embedding_model_path,
                    **embedding_kwargs,
                )
            else:
                raise ValueError(f"unsupported embedding backend: {settings.embedding_backend}")
        else:
            embedding_service = DisabledEmbeddingService()

    r0_retrieval = RetrievalService(embedding_service, settings.index_path)
    if settings.retrieval_backend == "e1":
        if e1_embedding_service is None:
            if not settings.e1_embedding_endpoint:
                raise ValueError(
                    "SCENEMINDX_E1_EMBEDDING_ENDPOINT is required when "
                    "SCENEMINDX_RETRIEVAL_BACKEND=e1"
                )
            e1_embedding_service = RemoteQwenVLEmbeddingService(
                settings.e1_embedding_endpoint,
                model_revision="cda4398c9bbfb3a644105446a2793692a8da5ea1",
                timeout_seconds=settings.e1_timeout_seconds,
            )
        retrieval = E1RetrievalAdapter(
            e1_embedding=e1_embedding_service,
            e1_index=FaissRetrievalIndex(
                settings.e1_index_root or settings.run_root / "index" / "e1_product",
                dimensions=2048,
                lifecycle_path=(
                    settings.project_root
                    / "artifacts"
                    / "phase7_4c_legacy_archival_top5"
                    / "archived_legacy_assets.json"
                ),
            ),
            r0=r0_retrieval,
            requested_backend=settings.retrieval_backend,
            fallback_backend=settings.retrieval_fallback,
        )
    else:
        retrieval = r0_retrieval
    legacy_retrieval = retrieval
    traces = TraceStore(settings.run_root / "traces", settings.git_commit)
    orchestrator = Orchestrator(library, vlm_service, retrieval, traces, settings.run_root / "outputs")
    legacy_orchestrator = Orchestrator(
        library,
        vlm_service,
        legacy_retrieval,
        traces,
        settings.run_root / "outputs",
    )
    managed_user_assets_root = (
        settings.project_root / "data" / "user_assets"
        if settings.run_root.resolve().is_relative_to(
            settings.project_root.resolve()
        )
        else settings.run_root / "product" / "uploads"
    )
    product = ProductStore(
        settings.run_root / "product",
        assets_root=managed_user_assets_root,
        session_assets_root=(
            settings.run_root / "product" / "session_uploads"
        ),
    )
    if managed_user_assets_root == settings.project_root / "data" / "user_assets":
        product.bootstrap_default_assets(project_root=settings.project_root)
    provider_index_backfill = ProviderIndexBackfillManager(
        product.root / "provider_index_backfill"
    )
    self_hosted_vlm = SelfHostedVLMProvider(vlm_service)
    self_hosted_embedding = (
        SelfHostedEmbeddingProvider(e1_embedding_service)
        if e1_embedding_service is not None
        else None
    )
    provider_manager = ProviderManager(
        state_path=settings.run_root / "product" / "provider_access.json",
        profiles_path=(
            settings.project_root
            / "configs"
            / "providers"
            / "phase7_0_provider_profiles.json"
        ),
        migrate_self_hosted=bool(
            injected_vlm_service
            or (
                settings.enable_vlm
                and settings.vlm_endpoint
                and settings.retrieval_backend == "e1"
                and settings.e1_embedding_endpoint
            )
        ),
        legacy_vlm=self_hosted_vlm,
        legacy_retrieval=retrieval,
        legacy_embedding=self_hosted_embedding,
    )
    course_credentials_path = (
        settings.bailian_credentials_path
        or settings.project_root / ".secrets" / "bailian_credentials.csv"
    ).resolve()

    def course_default_credentials_available() -> bool:
        try:
            load_course_credentials(course_credentials_path)
        except (FileNotFoundError, OSError, ValueError):
            return False
        return True

    provider_manager.set_course_default_credentials_available(
        course_default_credentials_available()
    )
    bailian_runtime: dict[str, Any] = {}
    cloud_retrieval: BailianCloudRetrieval | None = None
    cloud_index_root = Path(
        os.environ.get(
            "SCENEMINDX_CLOUD_INDEX_ROOT",
            str(
                (
                    settings.project_root
                    / "data"
                    / "indexes"
                    / "cloud"
                    / "bailian_qwen3_vl_embedding_2560"
                )
                if settings.run_root.resolve().is_relative_to(
                    settings.project_root.resolve()
                )
                else (
                    settings.run_root
                    / "index"
                    / "cloud"
                    / "bailian_qwen3_vl_embedding_2560"
                )
            ),
        )
    ).resolve()
    cloud_full_index_root = Path(
        os.environ.get(
            "SCENEMINDX_CLOUD_FULL_INDEX_ROOT",
            str(
                (
                    settings.project_root
                    / "data"
                    / "indexes"
                    / "cloud"
                    / "bailian_qwen3_vl_embedding_2560"
                    / "full_train_val"
                    / "faiss"
                )
                if settings.run_root.resolve().is_relative_to(
                    settings.project_root.resolve()
                )
                else (
                    settings.run_root
                    / "index"
                    / "cloud"
                    / "bailian_qwen3_vl_embedding_2560"
                    / "full_train_val"
                    / "faiss"
                )
            ),
        )
    ).resolve()
    cloud_transport_cache_root = Path(
        os.environ.get(
            "SCENEMINDX_CLOUD_TRANSPORT_CACHE_ROOT",
            str(
                (
                    settings.project_root
                    / "data"
                    / "cache"
                    / "cloud_transport"
                )
                if settings.run_root.resolve().is_relative_to(
                    settings.project_root.resolve()
                )
                else settings.run_root / "cache" / "cloud_transport"
            ),
        )
    ).resolve()
    cloud_transport_preprocessor = CloudImageTransportPreprocessor(
        config_path=(
            settings.project_root
            / "configs"
            / "providers"
            / "cloud_image_transport_profiles.json"
        ),
        cache_root=cloud_transport_cache_root,
    )
    cloud_index_identity = {
        "provider": "bailian",
        "region": "cn-beijing",
        "model_id": "qwen3-vl-embedding",
        "provider_model_revision": "provider_alias",
        "dimension": 2560,
        "vector_mode": "independent",
        "normalization": "l2",
        "metric": "inner_product",
        "preprocess_version": "scenemindx_cloud_image_v1",
        "index_schema_version": "v1",
    }
    cloud_first10_manifest_path = (
        settings.project_root
        / "data"
        / "manifests"
        / "phase7_0_cloud_index_train_first10_v1.json"
        if settings.run_root.resolve().is_relative_to(
            settings.project_root.resolve()
        )
        else (
            settings.run_root
            / "manifests"
            / "phase7_0_cloud_index_train_first10_v1.json"
        )
    )
    local_vlm_path = (
        settings.project_root / "models" / "local" / "qwen3-vl-4b-instruct"
    )
    local_embedding_path = (
        settings.project_root
        / "models"
        / "local"
        / "qwen3-vl-embedding-2b"
    )
    local_embedding_source_path = (
        settings.project_root / "vendor" / "qwen3_vl_embedding_frozen"
    )
    local_index_root = (
        settings.project_root / "data" / "indexes" / "local_e1_2048"
    )
    local_product_index_path = local_index_root / "product" / "faiss"
    local_system_index_root = local_index_root
    local_runtime: dict[str, Any] = {}
    bailian_reconnect_lock = threading.RLock()

    def _bailian_validation_expired() -> bool:
        value = bailian_runtime.get("last_validated_at")
        if not value:
            # A fresh process has never validated its newly-created clients.
            # Treat that NOT_TESTED state as requiring the same single,
            # bounded request-time check as an expired (STALE) client.
            return True
        try:
            validated_at = datetime.fromisoformat(str(value))
            now = datetime.now(
                validated_at.tzinfo or ZoneInfo("Asia/Shanghai")
            )
            expired = (
                now - validated_at
            ).total_seconds() >= BAILIAN_VALIDATION_TTL_SECONDS
        except (TypeError, ValueError):
            expired = True
        if expired:
            bailian_runtime["stale"] = True
        return expired

    def _close_bailian_runtime_clients() -> int:
        """Dispose process-local Bailian clients without touching product state."""
        services = [
            bailian_runtime.get("vlm"),
            bailian_runtime.get("embedding"),
            *(bailian_runtime.get("vlm_by_tier") or {}).values(),
        ]
        closed = 0
        seen: set[int] = set()
        for service in services:
            if service is None or id(service) in seen:
                continue
            seen.add(id(service))
            close = getattr(service, "close", None)
            if callable(close):
                close()
                closed += 1
        provider_manager.remove_runtime("bailian")
        bailian_runtime["vlm"] = None
        bailian_runtime["embedding"] = None
        bailian_runtime["vlm_by_tier"] = {}
        bailian_runtime["clients_closed"] = closed
        return closed

    def _active_bailian_credentials() -> Any:
        if provider_manager.credential_source == "user_session":
            value = provider_manager.user_session_credentials()
            if not value:
                raise ValueError("user_session_api_key_required")
            return credentials_from_user(
                api_key=value["api_key"],
                region=value["region"],
                api_host=value.get("api_host"),
                workspace_id=value.get("workspace_id"),
                endpoint_mode=value.get("endpoint_mode", "shared"),
            )
        return load_course_credentials(course_credentials_path)

    def _bailian_config_identities() -> tuple[str, str]:
        credentials = _active_bailian_credentials()
        profiles = provider_manager.profiles["profiles"]["bailian"]
        embedding = profiles["embedding"]
        shared = {
            "schema_version": "scenemindx_bailian_config_identity_v1",
            "provider": "bailian",
            "region": credentials.region,
            "openai_base": credentials.openai_base,
            "dashscope_base": credentials.dashscope_base,
            "workspace_scope": credentials.workspace_scope,
            "credential_source": provider_manager.credential_source,
            "credential_session_revision": (
                provider_manager.session_credential_revision
                if provider_manager.credential_source == "user_session"
                else 0
            ),
            "embedding_model": embedding["model_id"],
            "embedding_dimension": int(embedding["dimension"]),
        }
        embedding_identity = hashlib.sha256(
            json.dumps(
                shared,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        full = {
            **shared,
            "cloud_tier": provider_manager.cloud_tier,
            "vlm_model": profiles[provider_manager.cloud_tier][
                "vlm_model_id"
            ],
        }
        return (
            hashlib.sha256(
                json.dumps(
                    full,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            embedding_identity,
        )

    def ensure_bailian_runtime(*, force_new: bool = False) -> tuple[Any, Any]:
        nonlocal cloud_retrieval
        previous_runtime = dict(bailian_runtime)
        if force_new:
            _close_bailian_runtime_clients()
        tier = provider_manager.cloud_tier
        credential_source = provider_manager.credential_source
        config_identity, embedding_config_identity = (
            _bailian_config_identities()
        )
        if (
            not force_new
            and bailian_runtime.get("config_identity") == config_identity
            and bailian_runtime.get("vlm") is not None
            and bailian_runtime.get("embedding") is not None
        ):
            return bailian_runtime["vlm"], bailian_runtime["embedding"]
        cloud_profiles = provider_manager.profiles["profiles"]["bailian"]
        same_embedding_config = (
            bailian_runtime.get("embedding_config_identity")
            == embedding_config_identity
        )
        vlm_by_tier = (
            dict(bailian_runtime.get("vlm_by_tier") or {})
            if same_embedding_config and not force_new
            else {}
        )
        vlm = vlm_by_tier.get(tier)
        if vlm is None:
            vlm = BailianVLMProvider(
                credentials=_active_bailian_credentials,
                model_id=cloud_profiles[tier]["vlm_model_id"],
                prompt_root=settings.project_root / "prompts" / "phase1",
                capability_profile=cloud_profiles[tier],
                core_registry_path=(
                    settings.project_root / "prompts" / "gate1" / "p3_registry.json"
                ),
                core_prompt_version=settings.core_prompt_version,
                error_sink=provider_manager.set_error,
            )
        cloud_embedding = (
            bailian_runtime.get("embedding")
            if same_embedding_config and not force_new
            else None
        )
        if cloud_embedding is None:
            cloud_embedding = BailianEmbeddingProvider(
                credentials=_active_bailian_credentials,
                model_id=cloud_profiles["embedding"]["model_id"],
                dimension=int(cloud_profiles["embedding"]["dimension"]),
                error_sink=provider_manager.set_error,
            )
        vlm_by_tier[tier] = vlm
        bailian_runtime.clear()
        bailian_runtime.update(
            {
                "tier": tier,
                "credential_source": credential_source,
                "config_identity": config_identity,
                "embedding_config_identity": embedding_config_identity,
                "vlm": vlm,
                "vlm_by_tier": vlm_by_tier,
                "embedding": cloud_embedding,
                "validation_in_progress": False,
                "stale": bool(previous_runtime.get("stale", False)),
                "validated_embedding_identity": previous_runtime.get(
                    "validated_embedding_identity"
                ),
                "validated_vlm_identities": list(
                    previous_runtime.get("validated_vlm_identities") or []
                ),
                "last_validation_automatic": previous_runtime.get(
                    "last_validation_automatic"
                ),
                "last_validated_at": previous_runtime.get(
                    "last_validated_at"
                ),
            }
        )
        if cloud_retrieval is None:
            cloud_retrieval = BailianCloudRetrieval(
                embedding=cloud_embedding,
                root=cloud_index_root,
                identity=cloud_index_identity,
                transport_preprocessor=cloud_transport_preprocessor,
                base_index_root=cloud_full_index_root,
            )
        else:
            cloud_retrieval.replace_embedding(cloud_embedding)
        provider_manager.install_runtime(
            "bailian",
            vlm=vlm,
            retrieval=cloud_retrieval,
            embedding=cloud_embedding,
        )
        return vlm, cloud_embedding

    def current_local_preflight() -> dict[str, Any]:
        return local_model_preflight(
            vlm_path=local_vlm_path,
            embedding_path=local_embedding_path,
            embedding_source_path=local_embedding_source_path,
            disk_root=settings.project_root,
            manifest_path=(
                settings.project_root
                / "configs"
                / "providers"
                / "local_models_manifest.json"
            ),
            local_index_root=local_index_root,
        )

    def cleanup_local_runtime() -> None:
        for key in ("embedding_provider", "vlm_provider"):
            service = local_runtime.get(key)
            if service is not None:
                try:
                    service.unload()
                except Exception:
                    pass
        local_runtime.clear()
        provider_manager.remove_runtime("local")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    provider_vlm = ProviderVLMProxy(
        provider_manager,
        transport_preprocessor=(
            None
            if injected_vlm_service
            else cloud_transport_preprocessor
        ),
    )
    provider_retrieval = ProviderRetrievalProxy(provider_manager)
    orchestrator.vlm = provider_vlm
    orchestrator.retrieval = provider_retrieval
    vlm_service = provider_vlm
    retrieval = provider_retrieval
    if (
        provider_manager.mode == "bailian"
        and provider_manager.snapshot_cached()["credential"]["configured"]
        and (
            provider_manager.credential_source != "user_session"
            or provider_manager.has_user_session_credentials()
        )
    ):
        ensure_bailian_runtime(force_new=True)
    course_train_root = (
        settings.system_train_root
        or settings.project_root / "datasets" / "course_train"
    )
    course_val_root = (
        settings.system_val_root
        or settings.project_root / "datasets" / "course_val"
    )
    portable_asset_resolver = PortableAssetResolver(
        project_root=settings.project_root,
        train_root=course_train_root,
        val_root=course_val_root,
    )
    system_libraries = SystemVisualLibraryRepository(
        project_root=settings.project_root,
        train_root=course_train_root,
        val_root=course_val_root,
        train_manifest=settings.system_train_asset_manifest
        or settings.project_root / "data" / "manifests" / "phase6_1_train_assets.jsonl",
        val_manifest=settings.system_val_asset_manifest
        or settings.project_root / "data" / "manifests" / "phase6_1_val_assets.jsonl",
        catalog_path=settings.system_library_catalog
        or settings.project_root / "data" / "manifests" / "phase6_1_system_libraries.json",
        thumbnail_root=settings.system_thumbnail_root,
        train_active_manifest=settings.system_train_active_manifest,
        val_active_manifest=settings.system_val_active_manifest,
    )
    def system_library_asset_count(library_id: str) -> int:
        return next(
            (
                int(item.get("asset_count", 0) or 0)
                for item in system_libraries.libraries()
                if item.get("library_id") == library_id
            ),
            0,
        )

    def prepared_index_item_count(index_root: Path) -> int:
        metadata_path = index_root / "metadata.json"
        try:
            return int(json.loads(metadata_path.read_text(encoding="utf-8")).get("items", 0) or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 0

    split_retrieval = (
        SplitE1IndexRegistry(
            settings.system_e1_index_root,
            embedding=e1_embedding_service,
        )
        if e1_embedding_service is not None
        else None
    )

    def active_split_retrieval() -> SplitE1IndexRegistry | None:
        if provider_manager.mode == "local":
            return local_runtime.get("split_retrieval")
        return split_retrieval
    legacy_lifecycle_registry = getattr(
        getattr(legacy_retrieval, "e1_index", None),
        "lifecycle_registry",
        None,
    )
    asset_media = AssetMediaResolver(
        library=library,
        system_libraries=system_libraries,
        product=product,
        lifecycle_registry=legacy_lifecycle_registry,
    )

    def provider_snapshot(*, cached_only: bool = False) -> dict[str, Any]:
        snapshot = provider_manager.snapshot(cached_only=cached_only)
        mode = snapshot.get("mode")
        if mode == "bailian":
            _bailian_validation_expired()
            try:
                config_identity, embedding_identity = (
                    _bailian_config_identities()
                )
            except (ValueError, FileNotFoundError):
                config_identity = None
                embedding_identity = None
            if not snapshot.get("credential", {}).get("configured"):
                connection_state = "UNCONFIGURED"
            elif snapshot.get("errors"):
                connection_state = (
                    snapshot.get("connection_state_override")
                    or "ERROR"
                )
            elif bailian_runtime.get("validation_in_progress"):
                connection_state = "CONNECTING"
            elif bailian_runtime.get("stale"):
                connection_state = "STALE"
            elif (
                snapshot.get("capabilities", {}).get("vlm")
                and snapshot.get("capabilities", {}).get("embedding")
            ):
                connection_state = "READY"
            elif (
                snapshot.get("capabilities", {}).get("vlm")
                or snapshot.get("capabilities", {}).get("embedding")
            ):
                connection_state = "PARTIAL_READY"
            else:
                connection_state = "NOT_TESTED"
            snapshot["connection_state"] = connection_state
            snapshot["connection_transition"] = bailian_runtime.get(
                "transition"
            ) or snapshot.get("connection_transition")
            snapshot["last_validated_at"] = bailian_runtime.get(
                "last_validated_at"
            )
            snapshot["config_identity"] = {
                "schema_version": "scenemindx_bailian_config_identity_v1",
                "sha256": config_identity,
                "embedding_sha256": embedding_identity,
                "contains_api_key": False,
                "stable_within_credential_session": True,
            }
        elif mode == "self_hosted":
            product_status = (
                (
                    legacy_retrieval.cached_status()
                    if callable(
                        getattr(legacy_retrieval, "cached_status", None)
                    )
                    else legacy_retrieval.status()
                )
                if cached_only
                else legacy_retrieval.status()
            )
            if (
                getattr(legacy_retrieval, "requested_backend", None) == "e1"
                and getattr(legacy_retrieval, "e1_index", None) is not None
            ):
                # Provider inventory describes the persisted product index,
                # even when query execution temporarily falls back because
                # the remote embedding worker is offline.
                product_status = {
                    **product_status,
                    **legacy_retrieval.e1_index.status(),
                }
            system_status = (
                split_retrieval.public_status()
                if split_retrieval is not None
                else {
                    "status": "not_built",
                    "libraries": {
                        "system_train": {"status": "not_built", "items": 0},
                        "system_val": {"status": "not_built", "items": 0},
                    },
                }
            )
            train = system_status["libraries"].get(
                "system_train",
                {"status": "not_built", "items": 0},
            )
            val = system_status["libraries"].get(
                "system_val",
                {"status": "not_built", "items": 0},
            )
            snapshot["connection_state"] = (
                snapshot.get("connection_state_override")
                or (
                    "READY"
                    if snapshot.get("state") == "READY"
                    else snapshot.get("state", "ERROR")
                )
            )
            snapshot["index_scopes"] = {
                "product": {
                    "label": "产品/自定义资产",
                    "status": product_status.get("status"),
                    "items": int(
                        product_status.get(
                            "active_searchable_unique_sha",
                            product_status.get("items", 0),
                        )
                        or 0
                    ),
                    "active_unique_images": int(
                        product_status.get(
                            "active_searchable_unique_sha",
                            0,
                        )
                        or 0
                    ),
                    "active_records": int(
                        product_status.get(
                            "active_searchable_records",
                            0,
                        )
                        or 0
                    ),
                    "archived_records": int(
                        product_status.get("archived_records", 0)
                        or 0
                    ),
                    "physical_vectors": int(
                        product_status.get(
                            "physical_vectors",
                            product_status.get("items", 0),
                        )
                        or 0
                    ),
                    "total_unique_sha": int(
                        product_status.get("total_unique_sha", 0)
                        or 0
                    ),
                    "duplicate_sha_records": int(
                        product_status.get("duplicate_sha_records", 0)
                        or 0
                    ),
                    "dimensions": product_status.get("dimensions"),
                    "index_version": product_status.get("index_version"),
                },
                "system_train": {
                    "label": "系统 Train 索引",
                    **train,
                },
                "system_val": {
                    "label": "系统 Val 索引",
                    **val,
                },
                "scope_policy": (
                    "current_product_and_system_splits_are_independent"
                ),
            }
        elif mode == "local":
            local_product = local_runtime.get("retrieval")
            local_splits = local_runtime.get("split_retrieval")
            product_status = (
                local_product.e1_index.status()
                if local_product is not None
                and getattr(local_product, "e1_index", None) is not None
                else {
                    "status": "prepared_not_loaded"
                    if (local_product_index_path / "index.faiss").is_file()
                    else "not_built",
                    "items": (
                        prepared_index_item_count(local_product_index_path)
                        if (local_product_index_path / "index.faiss").is_file()
                        else 0
                    ),
                    "dimensions": 2048
                    if (local_product_index_path / "index.faiss").is_file()
                    else None,
                }
            )
            system_status = (
                local_splits.public_status()
                if local_splits is not None
                else {
                    "status": "prepared_not_loaded",
                    "libraries": {
                        "system_train": {
                            "status": "prepared_not_loaded",
                            "items": system_library_asset_count("system_train"),
                            "dimensions": 2048,
                        },
                        "system_val": {
                            "status": "prepared_not_loaded",
                            "items": system_library_asset_count("system_val"),
                            "dimensions": 2048,
                        },
                    },
                }
            )
            snapshot["connection_state"] = (
                "READY"
                if local_product is not None and local_splits is not None
                else snapshot.get("state", "NOT_LOADED")
            )
            snapshot["index_scopes"] = {
                "product": {
                    "label": "产品/自定义资产",
                    "status": product_status.get("status"),
                    "items": int(product_status.get("items", 0) or 0),
                    "dimensions": product_status.get("dimensions"),
                    "index_version": product_status.get("index_version"),
                },
                "system_train": {
                    "label": "系统 Train 索引",
                    **system_status["libraries"]["system_train"],
                },
                "system_val": {
                    "label": "系统 Val 索引",
                    **system_status["libraries"]["system_val"],
                },
                "scope_policy": (
                    "local_product_and_system_splits_are_independent"
                ),
            }
        else:
            snapshot["connection_state"] = snapshot.get("state")
        return snapshot

    def current_model_identity_answer() -> str:
        """Render the one allowed direct semantic answer from runtime state."""

        snapshot = provider_snapshot(cached_only=True)
        mode = str(snapshot.get("mode") or "no_model")
        mode_label = str(snapshot.get("mode_label") or mode)
        state = str(
            snapshot.get("connection_state")
            or snapshot.get("state")
            or "UNKNOWN"
        )
        model_id = str(
            (snapshot.get("vlm") or {}).get("model_id")
            or (snapshot.get("vlm") or {}).get("model")
            or ""
        ).strip()
        canonical_mode = {
            "bailian": "Bailian",
            "self_hosted": "Self-hosted",
            "local": "Local",
            "no_model": "No-model",
        }.get(mode, mode)
        provider_text = (
            canonical_mode
            if not mode_label or mode_label == mode
            else f"{canonical_mode}（{mode_label}）"
        )
        if mode == "no_model" or not model_id:
            return f"当前未启用视觉语言模型；Provider 为 {provider_text}，状态为 {state}。"
        tier = str(snapshot.get("cloud_tier") or "").strip()
        tier_text = f" / {tier}" if mode == "bailian" and tier else ""
        return (
            f"当前视觉语言模型是 {model_id}，通过 {provider_text}{tier_text} "
            f"Provider 运行，状态为 {state}。"
        )
    canonical_preview = CanonicalPreviewRepository(
        settings.project_root,
        settings.dataset_root,
        settings.canonical_preview_manifest,
    )
    course_candidate = CoursePromptCandidate(settings.project_root)
    multiturn_chat_candidate = MultiturnChatPromptCandidate(settings.project_root)
    conversational_response_candidate = ConversationalResponsePromptCandidate(settings.project_root)
    multi_image_content_candidate = MultiImageContentV2Candidate(settings.project_root)
    multi_image_story_candidate = MultiImageStoryV3Candidate(settings.project_root)
    chat_tool_router_candidate = ChatToolRouterCandidate(settings.project_root)
    contextual_rewriter_candidate = ContextualQueryRewriterCandidate(
        settings.project_root
    )
    content_profile_registry = ContentProfileRegistry(settings.project_root)
    library_assets_by_id = {item["image_id"]: item for item in library.list_assets()}
    canonical_manifest_path = (
        settings.project_root
        / "prompts"
        / "phase6_0a"
        / "canonical_pseudo_label_v2_candidate"
        / "manifest.json"
    )
    canonical_manifest = json.loads(
        canonical_manifest_path.read_text(encoding="utf-8")
    )
    canonical_prompt_path = (
        settings.project_root / canonical_manifest["prompt_file"]
    )
    canonical_prompt_bytes = canonical_prompt_path.read_bytes()
    canonical_prompt_sha = hashlib.sha256(
        canonical_prompt_bytes
    ).hexdigest()
    if canonical_prompt_sha != canonical_manifest["prompt_sha256"]:
        raise ValueError("canonical_prompt_sha256_mismatch")
    canonical_prompt = canonical_prompt_bytes.decode("utf-8")
    canonical_recovery_manifest_path = (
        settings.project_root
        / "prompts"
        / "phase6_0b"
        / "canonical_recovery_v1"
        / "manifest.json"
    )
    canonical_recovery_manifest = json.loads(
        canonical_recovery_manifest_path.read_text(encoding="utf-8")
    )
    canonical_recovery_prompt_path = (
        settings.project_root
        / canonical_recovery_manifest["prompt_path"]
    )
    canonical_recovery_prompt_bytes = (
        canonical_recovery_prompt_path.read_bytes()
    )
    canonical_recovery_prompt_sha = hashlib.sha256(
        canonical_recovery_prompt_bytes
    ).hexdigest()
    if (
        canonical_recovery_prompt_sha
        != canonical_recovery_manifest["prompt_sha256"]
    ):
        raise ValueError("canonical_recovery_prompt_sha256_mismatch")
    canonical_recovery_prompt = canonical_recovery_prompt_bytes.decode(
        "utf-8"
    )
    canonical_recovery_schema = json.loads(
        (
            settings.project_root
            / canonical_recovery_manifest["response_schema_path"]
        ).read_text(encoding="utf-8")
    )
    canonical_recovery_validator = Draft202012Validator(
        canonical_recovery_schema
    )
    canonical_schema_path = (
        settings.project_root
        / "data"
        / "schemas"
        / "scenemindx_canonical_pseudo_label_v2_1_1_candidate.schema.json"
    )
    product_scope_schema = json.loads(
        canonical_schema_path.read_text(encoding="utf-8")
    )
    product_scope_schema["$id"] = (
        "https://scenemind-x.local/schemas/"
        "phase6_3_product_scope_canonical_adapter_v1.json"
    )
    product_scope_schema["title"] = (
        "Phase 6.3 product-scope adapter for the frozen Canonical record"
    )
    product_scope_schema["properties"]["asset"]["properties"]["split"][
        "enum"
    ] = ["train", "val", "user_custom", "session_temporary"]
    canonical_schema_validator = Draft202012Validator(
        product_scope_schema
    )
    persistent_canonical_root = managed_user_assets_root / "canonical"
    persistent_canonical_root.mkdir(parents=True, exist_ok=True)

    def _write_json_atomic(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(
            path.suffix + f".{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _build_product_recovery_legacy(
        payload: dict[str, Any],
        *,
        item: dict[str, Any],
        asset_scope: str,
        result: Any,
        raw_output: str,
        run_id: str,
    ) -> dict[str, Any]:
        safe_facts = list(payload["fallback"]["safe_facts"])
        verification = str(
            payload["text_evidence"]["verification_status"]
        )
        old_payload = {
            "payload_version": "phase6_0a_model_payload_v1",
            "display": dict(payload["display"]),
            "facts": {
                "visual_medium": "uncertain",
                "scene": "",
                "subjects": [],
                "actions": [],
                "attributes": [],
                "relations": [],
            },
            "evidence": {
                "direct_observations": safe_facts,
                "cautious_inferences": [],
                "uncertainties": list(
                    payload["evidence_boundary"]["uncertainty"]
                ),
            },
            "text_evidence": {
                "presence": (
                    "none" if verification == "none" else "uncertain"
                ),
                "visual_candidates": [],
            },
            "fallback": {
                "safe_caption": compose_safe_caption(safe_facts),
                "safe_facts": safe_facts,
            },
        }
        normalized = normalize_model_payload(old_payload)
        legacy = build_canonical_label(
            normalized,
            asset_id=str(item["asset_id"]),
            asset_sha256=str(item["sha256"]).lower(),
            relative_path=str(
                item.get("storage_key")
                or item.get("image_id")
                or item["asset_id"]
            ),
            asset_split=asset_scope,
            prompt_sha256=canonical_recovery_prompt_sha,
            model=str(
                result.model
                or canonical_recovery_manifest["model"]
            ),
            model_revision=str(
                result.model_revision
                or canonical_recovery_manifest["model_revision"]
            ),
            raw_output_sha256=sha256_text(raw_output),
            source_run_id=run_id,
        )
        legacy["text_evidence"]["verification_status"] = verification
        return legacy

    def generate_product_canonical(
        item: dict[str, Any],
        *,
        asset_scope: Literal["user_custom", "session_temporary"],
        evidence_root: Path,
    ) -> dict[str, Any]:
        asset_id = str(item["asset_id"])
        image_path = Path(str(item["path"]))
        image_sha = str(item["sha256"]).lower()
        trace = traces.start(
            "generate_product_canonical",
            [asset_id],
            model=vlm_service.status().get("model"),
            model_revision=vlm_service.status().get("model_revision"),
            prompt_version=canonical_manifest["prompt_id"],
            schema_version=(
                "scenemindx_canonical_pseudo_label_v2_1_1_candidate"
            ),
            services=["product_store", "vlm", "canonical_validator"],
        )
        request_id = str(trace["request_id"])
        run_id = f"phase6_3_online_{request_id}"
        retry_trace: list[dict[str, Any]] = []
        last_error: Exception | None = None

        def accept_legacy(
            legacy: dict[str, Any],
            *,
            recovered_at_tokens: int | None,
            recovery_mode: str | None,
        ) -> dict[str, Any]:
            canonical = migrate_phase6_0a_to_phase6_1(
                legacy,
                asset_split=asset_scope,
                legacy_review=None,
                source_run_id=run_id,
                recovered_at_tokens=recovered_at_tokens,
            )
            contract_errors = validate_phase6_1_label(canonical)
            schema_errors = sorted(
                error.message
                for error in canonical_schema_validator.iter_errors(
                    canonical
                )
            )
            if contract_errors or schema_errors:
                raise ValueError(
                    "canonical_validation_failed:"
                    + ",".join([*contract_errors, *schema_errors])
                )
            canonical_path = (
                persistent_canonical_root
                / image_sha[:2]
                / f"{image_sha}.json"
                if asset_scope == "user_custom"
                else evidence_root / "canonical.json"
            )
            _write_json_atomic(canonical_path, canonical)
            finished = traces.finish(trace, status="success")
            return {
                "status": "completed",
                "canonical": canonical,
                "two_layer": build_two_layer_display(
                    canonical,
                    developer={
                        "canonical_path": (
                            canonical_path.relative_to(
                                settings.project_root
                            ).as_posix()
                            if settings.project_root
                            in canonical_path.resolve().parents
                            else canonical_path.name
                        ),
                        "trace_id": request_id,
                        "source_scope": asset_scope,
                        "recovery_mode": recovery_mode,
                    },
                ),
                "canonical_path": str(canonical_path),
                "trace_id": request_id,
                "request_id": finished["request_id"],
                "retry_trace": retry_trace,
                "model_called": True,
                "recovered_at_tokens": recovered_at_tokens,
                "recovery_mode": recovery_mode,
            }

        try:
            requested_tokens = 512
            for attempt_number in range(1, 3):
                result = vlm_service.run_course_prompt(
                    [image_path],
                    canonical_prompt,
                    prompt_id=canonical_manifest["prompt_id"],
                    prompt_sha256=canonical_prompt_sha,
                    max_new_tokens=requested_tokens,
                )
                raw_output = str(result.data.get("raw_output") or "")
                effective_tokens = int(
                    result.data.get("max_new_tokens")
                    or requested_tokens
                )
                evidence_path = (
                    evidence_root
                    / request_id
                    / (
                        f"raw_{requested_tokens}.json"
                        if attempt_number == 1
                        else f"raw_retry_{requested_tokens}.json"
                    )
                )
                _write_json_atomic(
                    evidence_path,
                    {
                        "asset_id": asset_id,
                        "asset_sha256": image_sha,
                        "scope": asset_scope,
                        "requested_max_new_tokens": requested_tokens,
                        "effective_max_new_tokens": effective_tokens,
                        "result": result.as_dict(),
                        "raw_output_sha256": sha256_text(raw_output),
                    },
                )
                try:
                    if result.status != "success":
                        raise RuntimeError(
                            result.error or "model_result_not_success"
                        )
                    payload = normalize_model_payload(
                        extract_json_object(raw_output)
                    )
                    legacy = build_canonical_label(
                        payload,
                        asset_id=asset_id,
                        asset_sha256=image_sha,
                        relative_path=str(
                            item.get("storage_key")
                            or item.get("image_id")
                            or asset_id
                        ),
                        asset_split=asset_scope,
                        prompt_sha256=canonical_prompt_sha,
                        model=str(
                            result.model or canonical_manifest["model"]
                        ),
                        model_revision=str(
                            result.model_revision
                            or canonical_manifest["model_revision"]
                        ),
                        raw_output_sha256=sha256_text(raw_output),
                        source_run_id=run_id,
                    )
                    return accept_legacy(
                        legacy,
                        recovered_at_tokens=(
                            effective_tokens
                            if attempt_number > 1
                            else None
                        ),
                        recovery_mode=(
                            "expanded_full_contract"
                            if attempt_number > 1
                            else None
                        ),
                    )
                except Exception as exc:
                    last_error = exc
                    explicit_truncation = is_truncation_failure(
                        raw_output=raw_output,
                        finish_reason=result.data.get("finish_reason"),
                        error=str(exc),
                    )
                    retry_trace.append(
                        {
                            "stage": "full_contract",
                            "attempt": attempt_number,
                            "requested_max_new_tokens": requested_tokens,
                            "effective_max_new_tokens": effective_tokens,
                            "status": "failed",
                            "explicit_truncation": explicit_truncation,
                            "error": f"{type(exc).__name__}:{exc}",
                            "raw_evidence": str(evidence_path),
                        }
                    )
                    if attempt_number == 2 or not explicit_truncation:
                        break
                    requested_tokens = min(
                        1536,
                        max(768, effective_tokens + 512),
                    )

            should_recover = bool(
                retry_trace
                and (
                    retry_trace[-1].get("explicit_truncation")
                    or "model_output_did_not_contain_json_object"
                    in str(last_error or "")
                )
            )
            if should_recover:
                recovery_result = vlm_service.run_course_prompt(
                    [image_path],
                    canonical_recovery_prompt,
                    prompt_id=canonical_recovery_manifest["prompt_id"],
                    prompt_sha256=canonical_recovery_prompt_sha,
                    max_new_tokens=int(
                        canonical_recovery_manifest["max_new_tokens"]
                    ),
                )
                recovery_raw = str(
                    recovery_result.data.get("raw_output") or ""
                )
                recovery_evidence = (
                    evidence_root
                    / request_id
                    / "raw_minimal_recovery.json"
                )
                _write_json_atomic(
                    recovery_evidence,
                    {
                        "asset_id": asset_id,
                        "asset_sha256": image_sha,
                        "scope": asset_scope,
                        "stage": "minimal_recovery",
                        "requested_max_new_tokens": int(
                            canonical_recovery_manifest[
                                "max_new_tokens"
                            ]
                        ),
                        "result": recovery_result.as_dict(),
                        "raw_output_sha256": sha256_text(recovery_raw),
                    },
                )
                try:
                    if recovery_result.status != "success":
                        raise RuntimeError(
                            recovery_result.error
                            or "recovery_model_result_not_success"
                        )
                    recovery_payload = normalize_recovery_payload(
                        extract_json_object(recovery_raw)
                    )
                    recovery_schema_errors = sorted(
                        error.message
                        for error in canonical_recovery_validator.iter_errors(
                            recovery_payload
                        )
                    )
                    recovery_safety_errors = (
                        validate_recovery_payload_safety(
                            recovery_payload
                        )
                    )
                    if (
                        recovery_schema_errors
                        or recovery_safety_errors
                    ):
                        raise ValueError(
                            "canonical_recovery_validation_failed:"
                            + ",".join(
                                [
                                    *recovery_schema_errors,
                                    *recovery_safety_errors,
                                ]
                            )
                        )
                    legacy = _build_product_recovery_legacy(
                        recovery_payload,
                        item=item,
                        asset_scope=asset_scope,
                        result=recovery_result,
                        raw_output=recovery_raw,
                        run_id=run_id,
                    )
                    retry_trace.append(
                        {
                            "stage": "minimal_recovery",
                            "status": "accepted",
                            "requested_max_new_tokens": int(
                                canonical_recovery_manifest[
                                    "max_new_tokens"
                                ]
                            ),
                            "effective_max_new_tokens": int(
                                recovery_result.data.get(
                                    "max_new_tokens"
                                )
                                or canonical_recovery_manifest[
                                    "max_new_tokens"
                                ]
                            ),
                            "raw_evidence": str(recovery_evidence),
                        }
                    )
                    return accept_legacy(
                        legacy,
                        recovered_at_tokens=int(
                            canonical_recovery_manifest[
                                "max_new_tokens"
                            ]
                        ),
                        recovery_mode="minimal_safe_contract",
                    )
                except Exception as exc:
                    last_error = exc
                    retry_trace.append(
                        {
                            "stage": "minimal_recovery",
                            "status": "failed",
                            "error": f"{type(exc).__name__}:{exc}",
                            "raw_evidence": str(recovery_evidence),
                        }
                    )
            raise last_error or RuntimeError(
                "canonical_generation_failed"
            )
        except Exception as exc:
            traces.finish(trace, status="failed", error=exc)
            raise CanonicalGenerationError(
                _canonical_failure(
                    exc,
                    request_id=request_id,
                    retry_trace=retry_trace,
                )
            ) from exc

    def resolve_course_assets(refs: list[CourseAssetRefRequest]) -> list[dict[str, Any]]:
        if len(refs) > 5:
            raise ValueError("course_context_maximum_is_5_images")
        resolved: list[dict[str, Any]] = []
        seen: set[str] = set()
        for order, ref in enumerate(refs, start=1):
            stable_ref = f"{ref.source}:{ref.asset_id}"
            if stable_ref in seen:
                raise ValueError(f"duplicate_course_asset_ref:{stable_ref}")
            seen.add(stable_ref)
            if ref.source == "library":
                metadata = library_assets_by_id.get(ref.asset_id)
                if metadata is None:
                    raise KeyError(ref.asset_id)
                image_path = library.image_path(ref.asset_id, verify_hash=True)
                facts = orchestrator.current_facts(ref.asset_id)
                ocr = library.ocr_evidence(ref.asset_id)
                public_asset_id = ref.asset_id
                image_url = f"/library/{ref.asset_id}/image"
                lifecycle = (
                    legacy_lifecycle_registry.lookup(
                        {
                            "asset_id": ref.asset_id,
                            "image_id": ref.asset_id,
                            "source": "frozen_library",
                        }
                    )
                    if legacy_lifecycle_registry is not None
                    else {
                        "lifecycle_state": "active",
                        "searchable": True,
                        "lifecycle_label": "活动资产",
                    }
                )
            elif ref.source == "system":
                metadata = system_libraries.asset(ref.asset_id)
                image_path = system_libraries.image_path(ref.asset_id, verify_hash=True)
                model_context = system_libraries.model_context(ref.asset_id)
                facts = model_context["facts"]
                ocr = model_context["ocr"]
                public_asset_id = ref.asset_id
                image_url = f"/visual-assets/{ref.asset_id}/image"
                lifecycle = {
                    "lifecycle_state": "active",
                    "searchable": True,
                    "lifecycle_label": "活动资产",
                }
            elif ref.source == "session":
                if not ref.conversation_id:
                    raise ValueError("session_asset_conversation_id_required")
                metadata = product.session_asset(
                    ref.asset_id,
                    ref.conversation_id,
                )
                image_path = Path(metadata["path"])
                canonical_record = metadata.get("canonical")
                canonical_label = (
                    canonical_record.get("canonical")
                    if isinstance(canonical_record, dict)
                    else None
                )
                facts = (
                    {
                        **dict(canonical_label.get("facts") or {}),
                        "safe_facts": list(
                            (canonical_label.get("fallback") or {}).get(
                                "safe_facts"
                            )
                            or []
                        ),
                        "safe_caption": str(
                            (canonical_label.get("fallback") or {}).get(
                                "safe_caption"
                            )
                            or ""
                        ),
                    }
                    if isinstance(canonical_label, dict)
                    else {}
                )
                ocr = {
                    "status": "not_available",
                    "truth_status": "image_only_unverified_text",
                    "candidates": [],
                }
                public_asset_id = stable_ref
                image_url = (
                    f"/session-assets/{ref.asset_id}/image"
                    f"?conversation_id={ref.conversation_id}"
                )
                lifecycle = {
                    "lifecycle_state": "active",
                    "searchable": True,
                    "lifecycle_label": "活动资产",
                }
            else:
                metadata = product.asset(ref.asset_id)
                image_path = Path(metadata["path"])
                analysis = metadata.get("analysis") or {}
                data = analysis.get("result", {}).get("data", {}) if isinstance(analysis, dict) else {}
                canonical_record = (
                    analysis.get("canonical")
                    if isinstance(analysis, dict)
                    else None
                )
                canonical_label = (
                    canonical_record.get("canonical")
                    if isinstance(canonical_record, dict)
                    else None
                )
                facts = (
                    {
                        **dict(canonical_label.get("facts") or {}),
                        "safe_facts": list(
                            (canonical_label.get("fallback") or {}).get(
                                "safe_facts"
                            )
                            or []
                        ),
                        "safe_caption": str(
                            (canonical_label.get("fallback") or {}).get(
                                "safe_caption"
                            )
                            or ""
                        ),
                    }
                    if isinstance(canonical_label, dict)
                    else data.get("normalized_output")
                    or data.get("parsed_output")
                    or {}
                )
                ocr = {"status": "not_available", "truth_status": "image_only_unverified_text", "candidates": []}
                public_asset_id = stable_ref
                image_url = f"/local-assets/{ref.asset_id}/image"
                lifecycle = {
                    "lifecycle_state": "active",
                    "searchable": True,
                    "lifecycle_label": "活动资产",
                }
            resolved.append(
                {
                    "ref": stable_ref,
                    "source": ref.source,
                    "source_asset_id": ref.asset_id,
                    "asset_id": public_asset_id,
                    "image_id": str(metadata.get("image_id", ref.asset_id)),
                    "library_id": metadata.get("library_id"),
                    "order": order,
                    "sha256": str(metadata["sha256"]),
                    "image_url": image_url,
                    "thumbnail_url": str(
                        metadata.get("thumbnail_url") or image_url
                    ),
                    "path": image_path,
                    "facts": facts,
                    "verified_text": [],
                    "ocr_candidates": ocr.get("candidates", []),
                    "evidence_truth_status": ocr.get("truth_status", "image_only_unverified_text"),
                    "lifecycle_state": lifecycle["lifecycle_state"],
                    "searchable": lifecycle["searchable"],
                    "lifecycle_label": lifecycle["lifecycle_label"],
                }
            )
        return resolved

    def public_course_assets(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{key: value for key, value in item.items() if key != "path"} for item in values]

    def session_refs(session: dict[str, Any]) -> list[CourseAssetRefRequest]:
        refs = []
        for item in session.get("active_assets", []):
            source = str(item.get("source", "library"))
            source_asset_id = item.get("source_asset_id") or item.get("image_id") or item.get("asset_id")
            if source_asset_id:
                refs.append(
                    CourseAssetRefRequest(
                        source=source,
                        asset_id=str(source_asset_id).removeprefix("local:"),
                        conversation_id=(
                            str(session.get("conversation_id"))
                            if source == "session"
                            else None
                        ),
                    )
                )
        return refs

    def course_prompt_call(
        *,
        prompt_id: str,
        prompt: str,
        identity: dict[str, str],
        assets: list[dict[str, Any]],
        task_type: str,
        max_new_tokens: int,
        image_labels: list[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        trace = traces.start(
            task_type,
            [item["asset_id"] for item in assets],
            model=vlm_service.status().get("model"),
            model_revision=vlm_service.status().get("model_revision"),
            prompt_version=prompt_id,
            prompt_sha256=identity["prompt_sha256"],
            schema_version=f"{task_type}_v1",
            services=["product_store", "course_context", "vlm"],
        )
        try:
            result = vlm_service.run_course_prompt(
                [item["path"] for item in assets],
                prompt,
                prompt_id=prompt_id,
                prompt_sha256=identity["prompt_sha256"],
                max_new_tokens=max_new_tokens,
                image_labels=image_labels,
            )
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            finished = traces.finish(trace, status="success")
            return result.as_dict(), finished
        except Exception as exc:
            traces.finish(trace, status="failed", error=exc)
            raise

    def resolve_session_assets_by_label(
        session: dict[str, Any],
        image_labels: list[str],
    ) -> list[dict[str, Any]]:
        active_by_label = {
            str(item.get("image_label")): item
            for item in session.get("active_assets", [])
            if item.get("image_label")
        }
        refs = []
        for label in image_labels:
            item = active_by_label.get(label)
            if item is None:
                raise ValueError(f"resolved_image_label_not_active:{label}")
            source_asset_id = item.get("source_asset_id") or item.get("image_id") or item.get("asset_id")
            refs.append(
                CourseAssetRefRequest(
                    source=str(item.get("source", "library")),
                    asset_id=str(source_asset_id).removeprefix("local:"),
                    conversation_id=(
                        str(session.get("conversation_id"))
                        if str(item.get("source", "library")) == "session"
                        else None
                    ),
                )
            )
        resolved = resolve_course_assets(refs)
        for item, label in zip(resolved, image_labels):
            item["image_label"] = label
        validate_model_bindings(
            session,
            resolved,
            image_labels,
        )
        return resolved

    def commit_referenced_image_scope(
        session: dict[str, Any],
        assets: list[dict[str, Any]],
        *,
        use_kind: str,
        confidence: str = "high",
        requires_clarification: bool = False,
        trace_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Commit a validated visual scope before the business tool runs."""

        if requires_clarification or confidence != "high":
            return None
        labels = [
            str(item.get("image_label") or "")
            for item in assets
        ]
        if (
            not labels
            or any(not label.startswith("IMG_") for label in labels)
        ):
            return None
        return run_or_http_error(
            product.commit_referenced_image_scope,
            session,
            assets=assets,
            image_labels=labels,
            use_kind=use_kind,
            trace_id=trace_id,
        )

    def canonicalize_session_assets(
        session: dict[str, Any],
        assets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        label_by_ref = {
            str(item.get("ref")): str(item.get("image_label"))
            for item in session.get("active_assets", [])
            if item.get("ref") and item.get("image_label")
        }
        labels = [
            label_by_ref.get(str(item.get("ref")), "")
            for item in assets
        ]
        if any(not label for label in labels):
            raise ValueError(
                "referenced_image_scope_not_in_conversation"
            )
        return resolve_session_assets_by_label(session, labels)

    def multiturn_chat_call(
        *,
        identity: dict[str, str],
        system_prompt: str,
        history_messages: list[dict[str, str]],
        current_prompt: str,
        assets: list[dict[str, Any]],
        context_trace: dict[str, Any],
        max_new_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        image_labels = [str(item["image_label"]) for item in assets]
        used_transport_labels = {
            label
            for label in image_labels
            if re.fullmatch(r"IMG_[1-9]\d*", label)
        }
        used_transport_labels.update(
            match.upper()
            for match in re.findall(
                r"(?<![A-Z0-9_])IMG_[1-9]\d*(?!\d)",
                (
                    current_prompt
                    + "\n"
                    + json.dumps(
                        history_messages,
                        ensure_ascii=False,
                    )
                ),
                flags=re.IGNORECASE,
            )
        )
        model_label_map: dict[str, str] = {}
        next_transport_index = 1
        for label in image_labels:
            if re.fullmatch(r"IMG_[1-9]\d*", label):
                model_label_map[label] = label
                continue
            while (
                f"IMG_{next_transport_index}"
                in used_transport_labels
            ):
                next_transport_index += 1
            transport_label = f"IMG_{next_transport_index}"
            model_label_map[label] = transport_label
            used_transport_labels.add(transport_label)
            next_transport_index += 1
        transport_labels = [
            model_label_map[label]
            for label in image_labels
        ]

        def replace_labels(value: Any, mapping: dict[str, str]) -> Any:
            if isinstance(value, str):
                result = value
                for source, target in sorted(
                    mapping.items(),
                    key=lambda item: len(item[0]),
                    reverse=True,
                ):
                    result = result.replace(source, target)
                return result
            if isinstance(value, list):
                return [
                    replace_labels(item, mapping)
                    for item in value
                ]
            if isinstance(value, dict):
                return {
                    key: replace_labels(item, mapping)
                    for key, item in value.items()
                }
            return value

        model_prompt = replace_labels(
            current_prompt,
            model_label_map,
        )
        model_history = replace_labels(
            history_messages,
            model_label_map,
        )
        context_trace["model_transport_label_map"] = dict(
            model_label_map
        )
        context_trace["model_transport_alias_applied"] = any(
            source != target
            for source, target in model_label_map.items()
        )
        trace = traces.start(
            "course_multiturn_chat",
            image_labels,
            model=vlm_service.status().get("model"),
            model_revision=vlm_service.status().get("model_revision"),
            prompt_version=identity["prompt_id"],
            prompt_sha256=identity["prompt_sha256"],
            schema_version="phase5_2a_multiturn_chat_response_v2",
            services=["product_store", "reference_resolver", "context_builder", "vlm"],
        )
        trace["chat_context"] = context_trace
        try:
            if hasattr(vlm_service, "run_multiturn_chat"):
                result = vlm_service.run_multiturn_chat(
                    [item["path"] for item in assets],
                    transport_labels,
                    system_prompt=system_prompt,
                    history_messages=model_history,
                    current_prompt=model_prompt,
                    prompt_id=identity["prompt_id"],
                    prompt_sha256=identity["prompt_sha256"],
                    max_new_tokens=max_new_tokens,
                )
            else:
                compatibility_prompt = (
                    f"{system_prompt}\n\n<current_user_turn>\n{current_prompt}\n</current_user_turn>"
                )
                result = vlm_service.run_course_prompt(
                    [item["path"] for item in assets],
                    replace_labels(
                        compatibility_prompt,
                        model_label_map,
                    ),
                    prompt_id=identity["prompt_id"],
                    prompt_sha256=identity["prompt_sha256"],
                    max_new_tokens=max_new_tokens,
                )
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            payload = result.as_dict()
            inverse_label_map = {
                transport: public
                for public, transport in model_label_map.items()
                if public != transport
            }
            if inverse_label_map:
                payload["data"] = replace_labels(
                    payload.get("data", {}),
                    inverse_label_map,
                )
            finished = traces.finish(trace, status="success")
            return payload, finished
        except Exception as exc:
            traces.finish(trace, status="failed", error=exc)
            raise

    def conversation_repair_call(
        *,
        prompt: str,
        identity: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        trace = traces.start(
            "course_conversation_contract_repair",
            [],
            model=vlm_service.status().get("model"),
            model_revision=vlm_service.status().get("model_revision"),
            prompt_version=identity["prompt_id"],
            prompt_sha256=identity["prompt_sha256"],
            schema_version="phase5_2c_conversational_response_v1",
            services=["conversation_validator", "vlm_text_repair"],
        )
        try:
            if not hasattr(vlm_service, "run_text_repair"):
                raise RuntimeError("conversation_text_repair_not_supported")
            result = vlm_service.run_text_repair(
                prompt,
                prompt_id=identity["prompt_id"],
                prompt_sha256=identity["prompt_sha256"],
                max_new_tokens=384,
            )
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            finished = traces.finish(trace, status="success")
            return result.as_dict(), finished
        except Exception as exc:
            traces.finish(trace, status="failed", error=exc)
            raise

    def direct_chat_payload(message: str) -> dict[str, Any]:
        """Answer a non-visual turn without sending conversation images."""

        prompt, identity = chat_tool_router_candidate.render_direct_chat(
            message
        )
        trace = traces.start(
            "course_direct_text_chat",
            [],
            model=vlm_service.status().get("model"),
            model_revision=vlm_service.status().get("model_revision"),
            prompt_version=identity["prompt_id"],
            prompt_sha256=identity["prompt_sha256"],
            schema_version="phase5_4d_direct_chat_v1",
            services=["chat_tool_router", "vlm_text_only"],
        )
        try:
            if not hasattr(vlm_service, "run_text_repair"):
                raise RuntimeError("direct_text_chat_not_supported")
            result = vlm_service.run_text_repair(
                prompt,
                prompt_id=identity["prompt_id"],
                prompt_sha256=identity["prompt_sha256"],
                max_new_tokens=220,
            )
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            raw_output = str(
                result.as_dict().get("data", {}).get("raw_output") or ""
            )
            answer, needs_clarification = parse_direct_chat_output(
                raw_output
            )
            if not answer:
                raise ValueError("direct_chat_answer_missing")
            finished = traces.finish(trace, status="success")
            return {
                "status": (
                    "clarification_required"
                    if needs_clarification
                    else "success"
                ),
                "answer": answer,
                "intent": "direct_chat",
                "model_called": True,
                "visual_model_called": False,
                "prompt_candidate": identity,
                "request_id": finished["request_id"],
                "trace": finished,
            }
        except Exception as exc:
            finished = traces.finish(trace, status="failed", error=exc)
            compact = "".join(message.lower().split())
            equation = re.search(
                r"[xｘXＸ]([+-])(-?\d+(?:\.\d+)?)=(-?\d+(?:\.\d+)?)",
                compact,
            )
            if equation:
                operator, offset_raw, total_raw = equation.groups()
                offset = float(offset_raw)
                total = float(total_raw)
                value = total - offset if operator == "+" else total + offset
                shown = int(value) if value.is_integer() else value
                answer = f"X 等于 {shown}。"
            elif any(
                token in compact
                for token in ("烧开水", "替我开门", "替我拿", "帮我拿")
            ):
                answer = (
                    "我不能替你执行现实中的物理操作。"
                    "如果要烧水，请使用安全的烧水设备并留意防烫。"
                )
            else:
                answer = (
                    "我理解这是普通文字问题，但这次文本回答没有稳定返回。"
                    "请换一种说法，我会继续直接回答。"
                )
            return {
                "status": "success",
                "answer": answer,
                "intent": "direct_chat",
                "model_called": False,
                "visual_model_called": False,
                "fallback_applied": True,
                "fallback_reason": type(exc).__name__,
                "request_id": finished["request_id"],
                "trace": finished,
            }

    def contextual_rewrite_call(
        message: str,
        session: dict[str, Any],
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        """Run L1 only after deterministic context dependence is known."""

        clean_inputs = clean_rewriter_inputs(session, detection)
        prompt, identity = contextual_rewriter_candidate.render(
            current_user_message=message,
            recent_clean_pairs=clean_inputs["recent_clean_pairs"],
            relevant_turn_state=clean_inputs[
                "relevant_turn_state"
            ],
            current_reference_mapping=clean_inputs[
                "current_reference_mapping"
            ],
        )
        trace = traces.start(
            "course_contextual_query_rewrite",
            [],
            model=vlm_service.status().get("model"),
            model_revision=vlm_service.status().get(
                "model_revision"
            ),
            prompt_version=identity["prompt_id"],
            prompt_sha256=identity["prompt_sha256"],
            schema_version="phase5_4h_contextual_rewrite_v1",
            services=["conversation_state", "vlm_text_repair"],
        )
        try:
            if not hasattr(vlm_service, "run_text_repair"):
                raise RuntimeError(
                    "contextual_rewriter_not_supported"
                )
            result = vlm_service.run_text_repair(
                prompt,
                prompt_id=identity["prompt_id"],
                prompt_sha256=identity["prompt_sha256"],
                max_new_tokens=identity["max_new_tokens"],
            )
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            raw_output = str(
                result.as_dict().get("data", {}).get(
                    "raw_output"
                )
                or ""
            )
            rewrite, errors = parse_contextual_rewrite(
                raw_output,
                allowed_turn_ids=clean_inputs["allowed_turn_ids"],
                allowed_image_refs=clean_inputs[
                    "allowed_image_refs"
                ],
            )
            if rewrite is None:
                raise ValueError(
                    "contextual_rewriter_contract_failed:"
                    + ",".join(errors)
                )
            finished = traces.finish(trace, status="success")
            return {
                **detection,
                **rewrite,
                "detected": True,
                "bound": not rewrite["needs_clarification"],
                "selected_image_labels": list(
                    rewrite["inherited_image_refs"]
                ),
                "requires_clarification": bool(
                    rewrite["needs_clarification"]
                ),
                "clarification": (
                    "这句话可能指向不止一个任务或目标。你想继续哪一条回答？"
                    if rewrite["needs_clarification"]
                    else None
                ),
                "execution_mode": (
                    "clarification"
                    if rewrite["needs_clarification"]
                    else (
                        "visual"
                        if rewrite["inherited_image_refs"]
                        else "text_transform"
                    )
                ),
                "rewriter_level": "L1",
                "rewriter_model_called": True,
                "rewriter_trace_id": finished["request_id"],
                "rewriter_prompt_candidate": identity,
                "resolution_errors": [],
            }
        except Exception as exc:
            finished = traces.finish(trace, status="failed", error=exc)
            return {
                **detection,
                "bound": False,
                "standalone_request": "",
                "confidence": "low",
                "needs_clarification": True,
                "requires_clarification": True,
                "clarification": (
                    "我还不能唯一确定你想继续哪一条回答，请指出任务或图片。"
                ),
                "execution_mode": "clarification",
                "rewriter_level": "L2",
                "rewriter_model_called": True,
                "rewriter_trace_id": finished["request_id"],
                "rewriter_prompt_candidate": identity,
                "resolution_errors": [
                    "contextual_rewriter_failed"
                ],
            }

    def normalize_conversational_payload(
        raw_payload: Any,
        *,
        state: dict[str, Any],
        assets: list[dict[str, Any]],
        all_bindings: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, list[str], bool]:
        labels = [str(item["image_label"]) for item in assets]
        normalized, parse_errors, repaired = deterministic_contract_repair(
            raw_payload,
            state=state,
            allowed_labels=labels,
        )
        if normalized is None:
            return None, parse_errors, repaired
        errors = validate_common_response(
            normalized,
            state=state,
            allowed_labels=labels,
            selected_assets=assets,
            all_bindings=all_bindings,
        )
        return normalized if not errors else None, errors, repaired

    def conversational_model_payload(result: dict[str, Any]) -> Any:
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        raw = data.get("raw_output")
        parsed = data.get("parsed_output")
        if isinstance(parsed, dict) and parsed and str(raw or "").strip() in {"", "{}"}:
            return parsed
        return raw if raw not in (None, "") else parsed

    def local_index_items() -> list[dict[str, Any]]:
        rows = []
        for item in product.assets():
            detail = product.asset(item["asset_id"])
            analysis = detail.get("analysis") or {}
            data = analysis.get("result", {}).get("data", {}) if isinstance(analysis, dict) else {}
            canonical_record = (
                analysis.get("canonical")
                if isinstance(analysis, dict)
                else None
            )
            canonical_label = (
                canonical_record.get("canonical")
                if isinstance(canonical_record, dict)
                else None
            )
            facts = (
                canonical_label
                if isinstance(canonical_label, dict)
                else data.get("normalized_output")
                or data.get("parsed_output")
                or {}
            )
            if isinstance(canonical_label, dict):
                display = canonical_label.get("display", {})
                semantic = canonical_label.get("facts", {})
                retrieval_text = "；".join(
                    str(value)
                    for value in [
                        display.get("theme"),
                        display.get("short_description"),
                        *(display.get("micro_tags") or []),
                        semantic.get("scene"),
                        *(semantic.get("subjects") or []),
                        *(semantic.get("actions") or []),
                        *(semantic.get("attributes") or []),
                        *(semantic.get("relations") or []),
                    ]
                    if value
                )
            else:
                retrieval_text = (
                    orchestrator.facts_text(facts)
                    if isinstance(facts, dict)
                    else ""
                )
            rows.append(
                {
                    "image_id": f"local:{item['asset_id']}",
                    "asset_id": f"local:{item['asset_id']}",
                    "sha256": item["sha256"],
                    "library_id": item.get("library_id", "default"),
                    "image_path": item["path"],
                    "retrieval_text": retrieval_text,
                    "source": "local_upload",
                    "image_url": f"/local-assets/{item['asset_id']}/image",
                }
            )
        return rows

    def cloud_train_items(target_count: int) -> list[dict[str, Any]]:
        """Freeze Train assets by numeric filename order, never lexical order."""

        candidates: list[dict[str, Any]] = []
        page = 1
        while len(candidates) < target_count:
            payload = system_libraries.query(
                "system_train",
                page=page,
                page_size=100,
                sort="filename_asc",
            )
            if not payload["items"]:
                break
            for item in payload["items"]:
                stem = Path(str(item["original_filename"])).stem
                if not stem.isdigit():
                    continue
                candidates.append(item)
                if len(candidates) == target_count:
                    break
            page += 1
        if len(candidates) != target_count:
            raise ValueError("cloud_train_numeric_scope_incomplete")
        candidates.sort(
            key=lambda item: int(Path(str(item["original_filename"])).stem)
        )
        return [
            {
                "asset_id": str(item["asset_id"]),
                "image_id": str(item["asset_id"]),
                "sha256": str(item["sha256"]),
                "library_id": "system_train",
                "image_path": str(
                    system_libraries.image_path(
                        str(item["asset_id"]),
                        verify_hash=True,
                    )
                ),
                "retrieval_text": "",
                "source": "course_train_first_numeric",
                "image_url": str(item["image_url"]),
                "source_split": "train",
                "numeric_id": int(Path(str(item["original_filename"])).stem),
            }
            for item in candidates
        ]

    def cloud_user_items(asset_ids: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for asset_id in asset_ids:
            item = product.asset(asset_id)
            rows.append(
                {
                    "asset_id": f"local:{asset_id}",
                    "image_id": f"local:{asset_id}",
                    "sha256": str(item["sha256"]),
                    "library_id": str(item.get("library_id") or "default"),
                    "image_path": str(item["path"]),
                    "retrieval_text": "",
                    "source": "local_upload",
                    "image_url": f"/local-assets/{asset_id}/image",
                    "source_split": "user_custom",
                }
            )
        return rows

    cloud_user_state_path = cloud_index_root / "user_asset_states.json"

    def read_cloud_user_states() -> dict[str, Any]:
        if not cloud_user_state_path.is_file():
            return {
                "schema_version": "scenemindx_cloud_user_asset_states_v1",
                "items": {},
            }
        try:
            value = json.loads(
                cloud_user_state_path.read_text(encoding="utf-8")
            )
            if not isinstance(value.get("items"), dict):
                raise ValueError("cloud_user_state_items_invalid")
            return value
        except (OSError, ValueError, json.JSONDecodeError):
            return {
                "schema_version": "scenemindx_cloud_user_asset_states_v1",
                "items": {},
            }

    def set_cloud_user_state(
        items: list[dict[str, Any]],
        status: str,
        *,
        failure: dict[str, Any] | None = None,
    ) -> None:
        value = read_cloud_user_states()
        for item in items:
            asset_id = str(item["asset_id"]).removeprefix("local:")
            provenance = (
                cloud_retrieval.embedding.vector_provenance(
                    str(item["sha256"])
                )
                if cloud_retrieval is not None
                else {}
            )
            value["items"][asset_id] = {
                "asset_id": asset_id,
                "sha256": str(item["sha256"]),
                "status": status,
                "index_identity_sha256": (
                    cloud_retrieval.embedding.identity_sha256
                    if cloud_retrieval is not None
                    else None
                ),
                "failure": failure,
                "transport": dict(provenance.get("transport") or {}),
                "updated_at": now_iso(),
            }
        value["updated_at"] = now_iso()
        _write_json_atomic(cloud_user_state_path, value)

    def cloud_user_asset_status(item: dict[str, Any]) -> dict[str, Any]:
        if cloud_retrieval is None:
            return {
                "status": "cloud_index_not_built",
                "label": "未建立云索引",
            }
        metadata_identity = cloud_retrieval.e1_index.metadata.get(
            "identity_sha256"
        )
        if (
            metadata_identity
            and metadata_identity
            != cloud_retrieval.embedding.identity_sha256
        ):
            return {
                "status": "configuration_mismatch",
                "label": "配置不匹配，需要重新编码",
            }
        digest = str(item.get("sha256") or "")
        if any(
            str(record.get("sha256")) == digest
            for record in cloud_retrieval.e1_index.records
        ):
            return {
                "status": "completed",
                "label": "云端索引已完成",
                "identity_sha256": cloud_retrieval.embedding.identity_sha256,
            }
        state = read_cloud_user_states()["items"].get(
            str(item.get("asset_id")),
            {},
        )
        if state:
            return {
                "status": state.get("status", "not_indexed"),
                "label": {
                    "preparing_cloud_image": "正在准备云端图片",
                    "oversized_preparing": "图片尺寸较大，正在生成兼容传输版本",
                    "indexing": "正在准备云端图片并建立索引",
                    "failed": "云端请求失败",
                    "waiting_for_api_key": "等待 API Key",
                }.get(state.get("status"), "未进入云索引"),
                "failure": state.get("failure"),
                "transport": dict(state.get("transport") or {}),
            }
        if cloud_retrieval.embedding.status().get("status") != "ready":
            return {
                "status": "waiting_for_api_key",
                "label": "等待 API Key",
            }
        return {
            "status": "not_indexed",
            "label": "未进入云索引",
        }

    def write_cloud_index_manifests(
        *,
        requested_base_count: int,
        failure: dict[str, Any] | None = None,
    ) -> None:
        if cloud_retrieval is None:
            return
        first10 = cloud_train_items(10)
        indexed_by_sha = {
            str(item.get("sha256")): position
            for position, item in enumerate(
                cloud_retrieval.e1_index.records,
                start=1,
            )
        }
        events = cloud_retrieval.embedding.events
        manifest_path = cloud_first10_manifest_path
        previous_assets: dict[str, dict[str, Any]] = {}
        if manifest_path.is_file():
            try:
                previous_payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                previous_assets = {
                    str(item.get("sha256")): dict(item)
                    for item in previous_payload.get("assets", [])
                    if isinstance(item, dict) and item.get("sha256")
                }
            except (OSError, ValueError, json.JSONDecodeError):
                previous_assets = {}
        assets = []
        for item in first10:
            digest = str(item["sha256"])
            event = events.get(digest, {})
            previous = previous_assets.get(digest, {})
            persisted = cloud_retrieval.embedding.vector_provenance(digest)
            completed = digest in indexed_by_sha
            assets.append(
                {
                    "split": "train",
                    "asset_id": item["asset_id"],
                    "sha256": digest,
                    "numeric_id": item["numeric_id"],
                    "embedding_status": (
                        "completed" if completed else event.get("status", "pending")
                    ),
                    "vector_id": (
                        event.get("vector_id")
                        or (
                            f"cloud2560:{digest[:24]}"
                            if completed
                            else None
                        )
                    ),
                    "request_count": (
                        int(event.get("request_count", 0) or 0)
                        if event
                        else max(
                            int(previous.get("request_count", 0) or 0),
                            int(persisted.get("request_count", 0) or 0),
                        )
                    ),
                    "retry_count": (
                        int(event.get("retry_count", 0) or 0)
                        if event
                        else max(
                            int(previous.get("retry_count", 0) or 0),
                            int(persisted.get("retry_count", 0) or 0),
                        )
                    ),
                    "original_image_sent_to_bailian": bool(
                        event.get("api_called")
                        if event
                        else (
                            previous.get(
                                "original_image_sent_to_bailian",
                                False,
                            )
                            or persisted.get(
                                "original_image_sent_to_bailian",
                                False,
                            )
                        )
                    ),
                }
            )
        status = cloud_retrieval.status()
        manifest = {
            "schema_version": "phase7_0_cloud_index_train_first10_v1",
            "provider": "bailian",
            "region": "cn-beijing",
            "model_id": "qwen3-vl-embedding",
            "dimension": 2560,
            "vector_mode": "independent",
            "normalization": "l2",
            "metric": "inner_product",
            "preprocess_version": "scenemindx_cloud_image_v1",
            "index_schema_version": "v1",
            "index_identity_sha256": (
                cloud_retrieval.embedding.identity_sha256
            ),
            "requested_base_count": requested_base_count,
            "frozen_first10_count": 10,
            "train_count": 10,
            "val_count": 0,
            "test_count": 0,
            "blind_count": 0,
            "assets": assets,
            "index": {
                "status": status["status"],
                "items": status["items"],
                "base_items": status["base_items"],
                "user_items": status["user_items"],
                "index_version": status["index_version"],
                "index_path": status["index_path"],
            },
            "failure": failure,
            "updated_at": now_iso(),
        }
        _write_json_atomic(
            manifest_path,
            manifest,
        )
        _write_json_atomic(
            cloud_index_root / "identity.json",
            {
                "schema_version": "scenemindx_cloud_index_identity_v1",
                **cloud_index_identity,
                "identity_sha256": cloud_retrieval.embedding.identity_sha256,
                "index": manifest["index"],
                "asset_manifest_sha256": (
                    cloud_retrieval.e1_index.metadata.get("manifest_sha256")
                ),
                "created_at": (
                    cloud_retrieval.e1_index.metadata.get("created_at")
                    or now_iso()
                ),
                "updated_at": now_iso(),
                "contains_sensitive_credentials": False,
                "contains_image_payloads": False,
            },
        )

    def safe_cloud_failure(exc: Exception) -> dict[str, Any]:
        if isinstance(exc, BailianProviderFailure):
            return exc.public_dict()
        return {
            "category": "cloud_index_error",
            "code": type(exc).__name__,
            "retryable": False,
            "stop_retries": True,
            "public_message": (
                "云端索引更新未完成，原图片与已有索引均已保留。"
                "请检查模型接入状态后重试。"
            ),
        }

    def rebuild_all_index() -> dict[str, Any]:
        return legacy_orchestrator.build_index(local_index_items())

    def _active_local_user_retrieval() -> Any | None:
        if provider_manager.mode == "self_hosted":
            return legacy_retrieval
        if provider_manager.mode == "local":
            return local_runtime.get("retrieval")
        return None

    def incremental_index_user_assets(
        asset_ids: list[str],
    ) -> dict[str, Any]:
        """Index only missing persistent user assets for the active provider."""

        ordered_ids = list(dict.fromkeys(str(value) for value in asset_ids))
        if not ordered_ids:
            return {
                "status": "not_applicable",
                "encoded": 0,
                "appended": 0,
                "rebuilt": False,
            }
        if provider_manager.mode == "bailian":
            snapshot = provider_snapshot()
            if (
                cloud_retrieval is None
                or snapshot.get("connection_state") != "READY"
                or int(cloud_retrieval.status().get("base_items", 0) or 0)
                <= 0
            ):
                return {
                    "status": "waiting_for_ready_provider",
                    "provider": "bailian",
                    "encoded": 0,
                    "appended": 0,
                    "rebuilt": False,
                }
            rows = cloud_user_items(ordered_ids)
            set_cloud_user_state(rows, "indexing")
            try:
                status = cloud_retrieval.add_assets(rows)
                set_cloud_user_state(rows, "completed")
                write_cloud_index_manifests(requested_base_count=10)
                return {
                    "status": "completed",
                    "provider": "bailian",
                    "index": status,
                    "encoded": int(
                        (cloud_retrieval.last_build or {}).get("encoded", 0)
                        or 0
                    ),
                    "appended": int(
                        (cloud_retrieval.last_build or {}).get("appended", 0)
                        or 0
                    ),
                    "rebuilt": False,
                }
            except Exception as exc:
                failure = safe_cloud_failure(exc)
                set_cloud_user_state(rows, "failed", failure=failure)
                return {
                    "status": "failed",
                    "provider": "bailian",
                    "failure": failure,
                    "encoded": 0,
                    "appended": 0,
                    "rebuilt": False,
                }
        adapter = _active_local_user_retrieval()
        if (
            provider_manager.mode not in {"self_hosted", "local"}
            or adapter is None
            or not hasattr(adapter, "e1_index")
        ):
            return {
                "status": "waiting_for_ready_provider",
                "provider": provider_manager.mode,
                "encoded": 0,
                "appended": 0,
                "rebuilt": False,
            }
        identity = adapter.e1_embedding.status()
        if identity.get("status") != "ready":
            return {
                "status": "waiting_for_ready_provider",
                "provider": provider_manager.mode,
                "encoded": 0,
                "appended": 0,
                "rebuilt": False,
            }
        requested = set(ordered_ids)
        rows = [
            item
            for item in local_index_items()
            if str(item["asset_id"]).removeprefix("local:") in requested
        ]
        status = adapter.e1_index.append_missing(
            rows,
            embedding=adapter.e1_embedding,
            model=str(identity.get("model") or "Qwen3-VL-Embedding-2B"),
            revision=str(
                identity.get("model_revision")
                or identity.get("revision")
                or "provider_runtime"
            ),
        )
        return {
            "status": "completed",
            "provider": provider_manager.mode,
            "index": status,
            "encoded": int(status.get("encoded", 0) or 0),
            "appended": int(status.get("appended", 0) or 0),
            "rebuilt": False,
        }

    def reuse_cached_cloud_user_vectors() -> dict[str, Any]:
        """Attach only already-cached user vectors after a cloud switch.

        This method performs no new provider request.  Assets without a vector
        for the active 2560-d identity remain explicitly pending.
        """

        if (
            provider_manager.mode != "bailian"
            or cloud_retrieval is None
            or provider_snapshot().get("connection_state") != "READY"
        ):
            return {
                "status": "not_applicable",
                "cached_candidates": 0,
                "pending": len(product.assets()),
                "new_api_calls": 0,
            }
        cached_ids: list[str] = []
        pending = 0
        for item in product.assets():
            provenance = cloud_retrieval.embedding.vector_provenance(
                str(item.get("sha256") or "")
            )
            if provenance.get("persisted"):
                cached_ids.append(str(item["asset_id"]))
            else:
                pending += 1
        update = (
            incremental_index_user_assets(cached_ids)
            if cached_ids
            else {
                "status": "completed",
                "encoded": 0,
                "appended": 0,
                "rebuilt": False,
            }
        )
        return {
            **update,
            "cached_candidates": len(cached_ids),
            "pending": pending,
            "new_api_calls": 0,
            "cache_only": True,
        }

    def local_user_asset_status(item: dict[str, Any]) -> dict[str, Any]:
        digest = str(item.get("sha256") or "")
        candidates = [
            legacy_retrieval,
            local_runtime.get("retrieval"),
        ]
        for adapter in candidates:
            index = getattr(adapter, "e1_index", None)
            if index is not None and any(
                str(record.get("sha256") or "") == digest
                for record in index.records
            ):
                return {
                    "status": "completed",
                    "label": "本地 E1 索引已完成",
                    "dimensions": 2048,
                    "index_version": index.metadata.get("index_version"),
                }
        return {
            "status": "not_indexed",
            "label": "未进入本地 E1 索引",
            "dimensions": 2048,
        }

    def asset_lifecycle_status(item: dict[str, Any]) -> dict[str, Any]:
        canonical = (
            (item.get("analysis") or {}).get("canonical")
            if isinstance(item.get("analysis"), dict)
            else None
        )
        return {
            "canonical": {
                "status": (
                    "completed"
                    if isinstance(canonical, dict)
                    and canonical.get("status") == "completed"
                    else "not_started"
                ),
                "required_for_embedding": False,
            },
            "local_e1": local_user_asset_status(item),
            "cloud_e1": cloud_user_asset_status(item),
            "active_provider": active_provider_asset_status(item),
        }

    def active_embedding_identity() -> dict[str, Any]:
        snapshot = provider_snapshot()
        mode = str(snapshot.get("mode") or "no_model")
        ready = (
            snapshot.get("connection_state") == "READY"
            and bool((snapshot.get("capabilities") or {}).get("embedding"))
        )
        if mode == "bailian" and cloud_retrieval is not None:
            identity = {
                **dict(cloud_retrieval.identity),
                "provider_id": "bailian",
                "transport_profile_version": (
                    "default_transport_v1@1;"
                    "oversized_image_downscale_v1@1"
                ),
            }
            identity["identity_sha256"] = hashlib.sha256(
                json.dumps(
                    identity, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            return {
                **identity,
                "mode": mode,
                "mode_label": "百炼云端",
                "ready": ready,
                "credential_source": snapshot.get("credential_source"),
                "index_identity_sha256": (
                    cloud_retrieval.embedding.identity_sha256
                ),
            }
        adapter = _active_local_user_retrieval()
        embedding = (
            adapter.e1_embedding.status()
            if adapter is not None and hasattr(adapter, "e1_embedding")
            else {}
        )
        index = (
            adapter.e1_index
            if adapter is not None and hasattr(adapter, "e1_index")
            else None
        )
        identity = {
            "provider_id": mode,
            "region": "private_runtime",
            "model_id": str(
                embedding.get("model")
                or embedding.get("model_id")
                or "Qwen3-VL-Embedding-2B"
            ),
            "dimension": int(
                embedding.get("dimensions")
                or embedding.get("dimension")
                or 2048
            ),
            "preprocess_version": str(
                (index.metadata if index is not None else {}).get(
                    "preprocess_version", "qwen3_vl_embedding_official_v1"
                )
            ),
            "transport_profile_version": (
                "self_hosted_private_transport_v1"
                if mode == "self_hosted"
                else "local_file_transport_v1"
            ),
            "normalization": str(
                embedding.get("normalization") or "l2"
            ),
            "metric": "inner_product",
            "index_schema_version": str(
                (index.metadata if index is not None else {}).get(
                    "schema_version", "scenemindx_e1_index_v1"
                )
            ),
        }
        identity["identity_sha256"] = hashlib.sha256(
            json.dumps(
                identity, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            **identity,
            "mode": mode,
            "mode_label": {
                "self_hosted": "服务器映射",
                "local": "当前电脑本地加载",
                "no_model": "暂不接入模型",
            }.get(mode, mode),
            "ready": ready and embedding.get("status") == "ready",
            "credential_source": None,
            "index_identity_sha256": (
                (index.metadata if index is not None else {}).get(
                    "identity_sha256"
                )
            ),
        }

    def active_provider_asset_status(
        item: dict[str, Any],
    ) -> dict[str, Any]:
        identity = active_embedding_identity()
        base = {
            "provider_mode": identity["mode"],
            "provider_label": identity["mode_label"],
            "identity_sha256": identity["identity_sha256"],
            "model_id": identity["model_id"],
            "dimensions": identity["dimension"],
        }
        if not identity.get("ready"):
            return {
                **base,
                "status": "waiting_for_provider",
                "label": "等待模型",
                "actionable": False,
            }
        if identity["mode"] == "bailian":
            value = cloud_user_asset_status(item)
            status = str(value.get("status"))
            mapped = {
                "completed": ("indexed", "已索引", False),
                "indexing": ("indexing", "建立中", False),
                "preparing_cloud_image": ("indexing", "建立中", False),
                "oversized_preparing": ("indexing", "建立中", False),
                "failed": ("failed", "索引失败", True),
                "configuration_mismatch": (
                    "identity_mismatch", "需要重新建立", True
                ),
            }.get(status, ("pending", "待补齐", True))
            return {
                **base,
                **value,
                "status": mapped[0],
                "label": mapped[1],
                "actionable": mapped[2],
            }
        adapter = _active_local_user_retrieval()
        index = getattr(adapter, "e1_index", None)
        digest = str(item.get("sha256") or "")
        if index is not None and any(
            str(record.get("sha256") or "") == digest
            for record in index.records
        ):
            return {
                **base,
                "status": "indexed",
                "label": "已索引",
                "actionable": False,
                "index_version": index.metadata.get("index_version"),
            }
        return {
            **base,
            "status": "pending",
            "label": "待补齐",
            "actionable": True,
        }

    def provider_index_coverage(
        *,
        scope: str = "all_user_assets",
        asset_id: str | None = None,
        library_id: str | None = None,
    ) -> dict[str, Any]:
        assets = product.assets()
        if scope == "asset":
            assets = [
                item for item in assets
                if str(item.get("asset_id")) == str(asset_id)
            ]
        elif scope == "library":
            assets = [
                item for item in assets
                if str(item.get("library_id") or "default")
                == str(library_id or "default")
            ]
        elif scope != "all_user_assets":
            raise ValueError("invalid_backfill_scope")
        statuses = [
            {
                "asset_id": str(item["asset_id"]),
                "library_id": str(item.get("library_id") or "default"),
                "image_id": str(item.get("image_id") or item["asset_id"]),
                "index": active_provider_asset_status(item),
            }
            for item in assets
        ]
        counts = {
            "total": len(statuses),
            "indexed": sum(
                item["index"]["status"] == "indexed" for item in statuses
            ),
            "pending": sum(
                item["index"]["status"] == "pending" for item in statuses
            ),
            "failed": sum(
                item["index"]["status"] == "failed" for item in statuses
            ),
            "identity_mismatch": sum(
                item["index"]["status"] == "identity_mismatch"
                for item in statuses
            ),
            "indexing": sum(
                item["index"]["status"] == "indexing" for item in statuses
            ),
            "waiting": sum(
                item["index"]["status"] == "waiting_for_provider"
                for item in statuses
            ),
        }
        missing = [
            item["asset_id"] for item in statuses
            if item["index"]["status"]
            in {"pending", "failed", "identity_mismatch"}
        ]
        identity = active_embedding_identity()
        is_cloud = identity["mode"] == "bailian"
        return {
            "schema_version": "scenemindx_provider_index_coverage_v1",
            "scope": scope,
            "identity": identity,
            "counts": counts,
            "missing_asset_ids": missing,
            "items": statuses,
            "requires_confirmation": bool(missing),
            "confirmation": {
                "external_processing": is_cloud,
                "destination": (
                    "阿里云百炼" if is_cloud else identity["mode_label"]
                ),
                "credential_source": identity.get("credential_source"),
                "estimated_images": len(missing),
                "estimated_cost_cny": (
                    {
                        "minimum": round(len(missing) * 0.001, 4),
                        "maximum": round(len(missing) * 0.005, 4),
                        "basis": "基于既有小规模调用的非承诺估算",
                    }
                    if is_cloud else None
                ),
                "estimated_seconds": {
                    "minimum": len(missing) * (1 if is_cloud else 2),
                    "maximum": len(missing) * (8 if is_cloud else 12),
                },
                "max_assets_per_task": 64,
                "completed_items_not_reencoded": True,
            },
            "course_index": {
                "train": f"{system_library_asset_count('system_train')}/{system_library_asset_count('system_train')}",
                "val": f"{system_library_asset_count('system_val')}/{system_library_asset_count('system_val')}",
            },
            "temporary_assets_included": False,
        }

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        monitor = RuntimeResourceMonitor(settings.run_root / "metrics")
        monitor.start()
        try:
            if settings.enable_embedding and hasattr(embedding_service, "load"):
                embedding_service.load()
            if (
                provider_manager.mode in {"self_hosted", "local"}
                and
                settings.enable_embedding
                and hasattr(legacy_retrieval, "load_index_only")
                and Path(legacy_retrieval.index_path).is_file()
            ):
                legacy_retrieval.load_index_only()
            yield
        finally:
            monitor.stop()

    app = FastAPI(title="SceneMind-X Studio", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.library = library
    app.state.orchestrator = orchestrator
    app.state.product = product
    app.state.system_libraries = system_libraries
    app.state.split_retrieval = split_retrieval
    app.state.canonical_preview = canonical_preview
    app.state.course_candidate = course_candidate
    app.state.multiturn_chat_candidate = multiturn_chat_candidate
    app.state.conversational_response_candidate = conversational_response_candidate
    app.state.contextual_rewriter_candidate = contextual_rewriter_candidate
    app.state.provider_manager = provider_manager
    app.state.bailian_runtime = bailian_runtime
    app.state.provider_retrieval = provider_retrieval
    app.state.asset_media = asset_media
    app.state.provider_index_backfill = provider_index_backfill

    vlm_paths = {
        "/analyze",
        "/vqa",
        "/generate",
        "/describe",
        "/compare",
        "/rank",
        "/course/generate",
        "/course/rank",
        "/course/compare",
        "/course/chat",
    }
    vlm_path_suffixes = {
        "/canonical-label",
        "/analyze",
        "/vqa",
    }
    retrieval_paths = {
        "/search",
        "/search/image",
        "/search/hybrid",
        "/course/retrieve",
        "/visual-libraries/search",
    }
    embedding_paths = {"/index/rebuild"}

    def _required_provider_capability(path: str, method: str) -> str | None:
        if method not in {"POST", "PUT", "PATCH"}:
            return None
        if path in vlm_paths or any(
            path.endswith(suffix) for suffix in vlm_path_suffixes
        ):
            return "vlm"
        if path in retrieval_paths:
            return "retrieval"
        if path in embedding_paths:
            return "embedding"
        return None

    @app.middleware("http")
    async def provider_capability_gate(request: Request, call_next: Any) -> Any:
        capability = _required_provider_capability(
            request.url.path, request.method.upper()
        )
        if capability == "vlm" and request.url.path == "/course/chat":
            try:
                chat_body = await request.json()
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                chat_body = {}
            utility = detect_system_utility(
                str(chat_body.get("message") or "")
            )
            if utility and utility.get("name") == "current_model_identity":
                capability = None
        if capability is None:
            return await call_next(request)
        try:
            # A real capability request is the bounded authority for lazy
            # Bailian revalidation. Perform it before the snapshot gate so a
            # freshly restarted process can recover from NOT_TESTED/STALE.
            provider_manager.prepare_for_request(capability)
            provider_manager.require(capability)
        except CapabilityUnavailable as exc:
            missing_bailian_credential = (
                exc.snapshot.get("mode") == "bailian"
                and not exc.snapshot.get("credential", {}).get("configured")
            )
            return JSONResponse(
                status_code=409,
                content={
                    "detail": (
                        "当前未配置百炼 API Key。请在“模型接入”中输入自己的"
                        " API Key；已有图片、工作区和历史结果仍可继续浏览。"
                        if missing_bailian_credential
                        else (
                            "当前尚未接入可用模型。请先在顶部“模型接入”中选择"
                            "本地、云端或服务器演示模式。"
                        )
                    ),
                    "code": (
                        "BAILIAN_CREDENTIALS_REQUIRED"
                        if missing_bailian_credential
                        else "PROVIDER_CAPABILITY_UNAVAILABLE"
                    ),
                    "capability": exc.capability,
                    "provider": exc.snapshot,
                },
            )
        with provider_manager.request_scope():
            return await call_next(request)

    web_root = settings.project_root / "apps" / "web"
    if web_root.is_dir():
        app.mount("/static", StaticFiles(directory=web_root), name="static")

    @app.get("/")
    def root() -> FileResponse:
        index = web_root / "course.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="web_frontend_not_built")
        return FileResponse(index)

    @app.get("/providers/access")
    def provider_access() -> dict[str, Any]:
        # Page bootstrap is an observation, not an explicit provider probe.
        # Manual test/revalidate routes retain bounded active validation.
        return provider_snapshot(cached_only=True)

    @app.get("/providers/profiles")
    def provider_profiles() -> dict[str, Any]:
        return provider_manager.profiles

    @app.get("/provider-index/coverage")
    def provider_index_coverage_route(
        scope: Literal["asset", "library", "all_user_assets"] = (
            "all_user_assets"
        ),
        asset_id: str | None = None,
        library_id: str | None = None,
    ) -> dict[str, Any]:
        if scope == "asset" and not asset_id:
            raise HTTPException(status_code=422, detail="asset_id_required")
        return provider_index_coverage(
            scope=scope, asset_id=asset_id, library_id=library_id
        )

    @app.get("/provider-index/tasks")
    def provider_index_tasks() -> dict[str, Any]:
        return {"items": provider_index_backfill.list()}

    @app.get("/provider-index/tasks/{task_id}")
    def provider_index_task(task_id: str) -> dict[str, Any]:
        try:
            return provider_index_backfill.get(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="backfill_task_not_found")

    @app.post("/provider-index/tasks")
    def create_provider_index_task(
        payload: ProviderIndexBackfillRequest,
    ) -> dict[str, Any]:
        coverage = provider_index_coverage(
            scope=payload.scope,
            asset_id=payload.asset_id,
            library_id=payload.library_id,
        )
        identity = coverage["identity"]
        if not identity.get("ready"):
            raise HTTPException(
                status_code=409,
                detail="当前 Embedding Provider 尚未就绪，请先接入并测试模型。",
            )
        missing = list(coverage["missing_asset_ids"])
        if not missing:
            return {
                "status": "no_action",
                "message": (
                    "当前范围内所有持久图片均已具备当前模式的检索索引，"
                    "无需补齐。"
                ),
                "coverage": coverage,
                "task": None,
            }
        try:
            task = provider_index_backfill.create(
                asset_ids=missing,
                identity=identity,
                scope=payload.scope,
                metadata={
                    "library_id": payload.library_id,
                    "asset_id": payload.asset_id,
                    "confirmation": coverage["confirmation"],
                },
                confirmed_by_user=payload.confirmed_by_user,
                operation_id=payload.operation_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        task = provider_index_backfill.start(
            str(task["task_id"]),
            runner=lambda asset_id: incremental_index_user_assets([asset_id]),
            current_identity_sha256=lambda: active_embedding_identity().get(
                "identity_sha256"
            ),
        )
        return {
            "status": "started",
            "task_id": task["task_id"],
            "coverage": coverage,
            "task": task,
        }

    @app.post("/provider-index/tasks/{task_id}/cancel")
    def cancel_provider_index_task(task_id: str) -> dict[str, Any]:
        try:
            return provider_index_backfill.cancel(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="backfill_task_not_found")

    @app.post("/provider-index/tasks/{task_id}/resume")
    def resume_provider_index_task(task_id: str) -> dict[str, Any]:
        try:
            return provider_index_backfill.start(
                task_id,
                runner=lambda asset_id: incremental_index_user_assets(
                    [asset_id]
                ),
                current_identity_sha256=lambda: active_embedding_identity().get(
                    "identity_sha256"
                ),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="backfill_task_not_found")

    @app.put("/providers/access/selection")
    def select_provider(payload: ProviderSelectionRequest) -> dict[str, Any]:
        def attach_cached_lifecycle(
            snapshot: dict[str, Any],
        ) -> dict[str, Any]:
            if snapshot.get("connection_state") != "READY":
                return snapshot
            result = {
                **snapshot, "index_coverage": provider_index_coverage()
            }
            if snapshot.get("mode") == "bailian":
                result["asset_lifecycle"] = reuse_cached_cloud_user_vectors()
            return result
        try:
            # Provider selection must not inherit the latency of a remote
            # health probe.  The Bailian switched-in branch below performs its
            # own explicit bounded validation; every other branch returns the
            # last truthful cached projection.
            previous = provider_snapshot(cached_only=True)
            if (
                previous.get("mode") == "bailian"
                and payload.mode != "bailian"
            ):
                bailian_runtime["stale"] = True
                bailian_runtime["transition"] = "SWITCHING"
                _close_bailian_runtime_clients()
            provider_manager.select(**payload.model_dump())
            if payload.mode == "bailian":
                if payload.credential_source == "course_default":
                    provider_manager.set_course_default_credentials_available(
                        course_default_credentials_available()
                    )
                    if not provider_manager.snapshot_cached()["credential"][
                        "configured"
                    ]:
                        _close_bailian_runtime_clients()
                        bailian_runtime["stale"] = False
                        bailian_runtime["transition"] = (
                            "BAILIAN_CREDENTIALS_REQUIRED"
                        )
                        return provider_snapshot(cached_only=True)
                if (
                    payload.credential_source == "user_session"
                    and not provider_manager.has_user_session_credentials()
                ):
                    bailian_runtime["stale"] = True
                    return provider_snapshot(cached_only=True)
                config_identity, embedding_identity = (
                    _bailian_config_identities()
                )
                switched_in = previous.get("mode") != "bailian"
                config_changed = (
                    bailian_runtime.get("config_identity")
                    != config_identity
                )
                embedding_changed = (
                    bailian_runtime.get("embedding_config_identity")
                    != embedding_identity
                )
                embedding_was_validated = (
                    bailian_runtime.get("validated_embedding_identity")
                    == embedding_identity
                )
                changing_only_cloud_vlm = bool(
                    not switched_in
                    and config_changed
                    and not embedding_changed
                    and embedding_was_validated
                )
                if switched_in or _bailian_validation_expired():
                    validated = _automatic_bailian_reconnect(
                        ProviderConnectionTestRequest(
                            cloud_tier=payload.cloud_tier,
                            credential_source=payload.credential_source,
                        )
                    )
                    return attach_cached_lifecycle(validated["provider"])
                if changing_only_cloud_vlm:
                    validated = _execute_bailian_connection_test(
                        ProviderConnectionTestRequest(
                            cloud_tier=payload.cloud_tier,
                            credential_source=payload.credential_source,
                        ),
                        force_new=False,
                        validate_embedding=False,
                        automatic=True,
                    )
                    return attach_cached_lifecycle(validated["provider"])
                ensure_bailian_runtime(force_new=False)
            return attach_cached_lifecycle(
                provider_snapshot(cached_only=True)
            )
        except ProviderSwitchBusy:
            raise HTTPException(
                status_code=409,
                detail="当前仍有模型请求正在运行，请等待完成后再切换。",
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post("/providers/access/credentials")
    def save_provider_credentials(
        payload: ProviderCredentialRequest,
    ) -> dict[str, Any]:
        if not payload.only_this_session:
            raise HTTPException(
                status_code=400,
                detail="本阶段只支持在当前后端运行会话中保存 API Key。",
            )
        try:
            provider_manager.set_user_session_credentials(
                api_key=payload.api_key,
                region=payload.region,
                api_host=payload.api_host,
                workspace_id=payload.workspace_id,
                endpoint_mode=payload.endpoint_mode,
            )
            bailian_runtime["stale"] = True
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {
            "configured": True,
            "credential_source": "user_session",
            "process_session_only": True,
            "revealable": False,
            "masked_summary": (
                payload.api_key[:3] + "****" + payload.api_key[-4:]
            ),
            "region": payload.region,
            "endpoint_mode": payload.endpoint_mode,
        }

    @app.delete("/providers/access/credentials")
    def clear_provider_credentials() -> dict[str, Any]:
        provider_manager.clear_user_session_credentials()
        bailian_runtime["stale"] = True
        return {
            "configured": False,
            "credential_source": "user_session",
            "process_session_only": True,
            "revealable": False,
        }

    def _execute_bailian_connection_test(
        payload: ProviderConnectionTestRequest,
        *,
        force_new: bool = True,
        validate_embedding: bool = True,
        automatic: bool = False,
    ) -> dict[str, Any]:
        bailian_runtime["validation_in_progress"] = True
        provider_manager.set_connection_state(
            "RECONNECTING" if automatic else "CONNECTING",
            transition="BAILIAN_REVALIDATING",
        )
        try:
            current = provider_manager.snapshot()
            if (
                current["mode"] != "bailian"
                or current["cloud_tier"] != payload.cloud_tier
                or current["credential_source"] != payload.credential_source
            ):
                provider_manager.select(
                    mode="bailian",
                    cloud_tier=payload.cloud_tier,
                    credential_source=payload.credential_source,
                    region="cn-beijing",
                )
            if (
                payload.credential_source == "user_session"
                and not provider_manager.has_user_session_credentials()
            ):
                provider_manager.set_connection_state(
                    "UNCONFIGURED",
                    transition="BAILIAN_CREDENTIALS_REQUIRED",
                )
                bailian_runtime["stale"] = False
                raise HTTPException(
                    status_code=400,
                    detail="请先输入自己的 API Key，再测试连接。",
                )
            if payload.credential_source == "course_default":
                provider_manager.set_course_default_credentials_available(
                    course_default_credentials_available()
                )
                if not provider_manager.snapshot_cached()["credential"][
                    "configured"
                ]:
                    provider_manager.set_connection_state(
                        "UNCONFIGURED",
                        transition="BAILIAN_CREDENTIALS_REQUIRED",
                    )
                    bailian_runtime["stale"] = False
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "课程演示默认 API Key 当前不可用。"
                            "请输入自己的阿里云百炼 API Key 后重新测试连接。"
                        ),
                    )
            provider_manager.clear_errors()
            cloud_vlm, cloud_embedding = ensure_bailian_runtime(
                force_new=force_new
            )
        except ProviderSwitchBusy:
            bailian_runtime["validation_in_progress"] = False
            bailian_runtime["stale"] = True
            raise HTTPException(
                status_code=409,
                detail="当前仍有模型请求正在运行，请等待完成后再测试连接。",
            ) from None
        except HTTPException:
            bailian_runtime["validation_in_progress"] = False
            bailian_runtime["stale"] = True
            raise
        except (ValueError, FileNotFoundError) as exc:
            mapped = classify_bailian_error(
                http_status=401,
                code=type(exc).__name__,
                message=str(exc),
                credential_source=payload.credential_source,
            )
            provider_manager.set_error("credentials", mapped)
            bailian_runtime["validation_in_progress"] = False
            bailian_runtime["stale"] = True
            raise HTTPException(
                status_code=400,
                detail=mapped.public_message,
            ) from None

        results: dict[str, Any] = {}
        stop_all = False
        try:
            vlm_health = cloud_vlm.health_check()
            vlm_ok = bool(
                str(vlm_health.get("status", "")).lower() == "ready"
                and vlm_health.get("identity_verified") is True
                and vlm_health.get("response_nonempty") is True
            )
            if not vlm_ok:
                identity_error = ProviderError(
                    "model_identity",
                    str(
                        vlm_health.get("error_code")
                        or "MODEL_IDENTITY_OR_RESPONSE_CHECK_FAILED"
                    ),
                    False,
                    True,
                    "百炼视觉模型连接已建立，但模型身份或响应校验未通过。请检查所选模型与业务空间权限。",
                )
                provider_manager.set_error("vlm", identity_error)
            results["vlm"] = {
                "success": vlm_ok,
                **vlm_health,
                "usage": cloud_vlm.usage(),
                **({} if vlm_ok else identity_error.as_dict()),
            }
        except BailianProviderFailure as exc:
            results["vlm"] = {"success": False, **exc.public_dict()}
            stop_all = exc.error.stop_retries
        except Exception as exc:
            mapped = classify_bailian_error(
                http_status=0,
                code=type(exc).__name__,
                message=str(exc),
                credential_source=payload.credential_source,
            )
            provider_manager.set_error("vlm", mapped)
            results["vlm"] = {"success": False, **mapped.as_dict()}
            stop_all = mapped.stop_retries

        if stop_all:
            results["embedding"] = {
                "success": False,
                "skipped": True,
                "code": "SKIPPED_AFTER_HARD_STOP",
                "public_message": "凭据或账户错误已触发停止保护，未继续发送 Embedding 请求。",
            }
        elif not validate_embedding:
            embedding_health = cloud_embedding.status()
            embedding_ok = bool(
                str(embedding_health.get("status", "")).lower() == "ready"
                and int(
                    embedding_health.get("dimensions", 0) or 0
                )
                == 2560
            )
            results["embedding"] = {
                "success": embedding_ok,
                **embedding_health,
                "dimension_verified": embedding_ok,
                "reused_without_request": True,
                "usage": cloud_embedding.usage(),
            }
        else:
            try:
                embedding_health = cloud_embedding.health_check()
                embedding_ok = bool(
                    str(embedding_health.get("status", "")).lower() == "ready"
                    and embedding_health.get("dimension_verified") is True
                )
                if not embedding_ok:
                    dimension_error = ProviderError(
                        "embedding_contract",
                        str(
                            embedding_health.get("error_code")
                            or "EMBEDDING_DIMENSION_CHECK_FAILED"
                        ),
                        False,
                        True,
                        "百炼 Embedding 连接已建立，但 2560 维向量合同校验未通过。",
                    )
                    provider_manager.set_error("embedding", dimension_error)
                results["embedding"] = {
                    "success": embedding_ok,
                    **embedding_health,
                    "usage": cloud_embedding.usage(),
                    **({} if embedding_ok else dimension_error.as_dict()),
                }
            except BailianProviderFailure as exc:
                results["embedding"] = {
                    "success": False,
                    **exc.public_dict(),
                }
            except Exception as exc:
                mapped = classify_bailian_error(
                    http_status=0,
                    code=type(exc).__name__,
                    message=str(exc),
                    credential_source=payload.credential_source,
                )
                provider_manager.set_error("embedding", mapped)
                results["embedding"] = {
                    "success": False,
                    **mapped.as_dict(),
                }
        all_ready = bool(
            results.get("vlm", {}).get("success")
            and results.get("embedding", {}).get("success")
        )
        config_identity, embedding_identity = _bailian_config_identities()
        if results.get("embedding", {}).get("success"):
            bailian_runtime["validated_embedding_identity"] = (
                embedding_identity
            )
        if results.get("vlm", {}).get("success"):
            validated_vlm = set(
                bailian_runtime.get("validated_vlm_identities") or []
            )
            validated_vlm.add(config_identity)
            bailian_runtime["validated_vlm_identities"] = sorted(
                validated_vlm
            )
        bailian_runtime["validation_in_progress"] = False
        bailian_runtime["stale"] = not all_ready
        bailian_runtime["transition"] = (
            "BAILIAN_READY" if all_ready else "BAILIAN_STALE"
        )
        bailian_runtime["last_validation_automatic"] = automatic
        bailian_runtime["last_validated_at"] = now_iso()
        if all_ready:
            provider_manager.set_connection_state(
                "READY",
                transition="BAILIAN_READY",
            )
        return {
            "provider": provider_snapshot(),
            "results": results,
            "hard_stop": stop_all,
            "automatic": automatic,
            "embedding_request_skipped": not validate_embedding,
        }

    def _automatic_bailian_reconnect(
        payload: ProviderConnectionTestRequest,
    ) -> dict[str, Any]:
        """Create fresh clients and retry once only for recoverable failures."""
        with bailian_reconnect_lock:
            bailian_runtime["transition"] = "BAILIAN_CONNECTING"
            first = _execute_bailian_connection_test(
                payload,
                force_new=True,
                validate_embedding=True,
                automatic=True,
            )
            first["reconnect_attempts"] = 1
            if first["provider"].get("connection_state") == "READY":
                bailian_runtime["transition"] = "BAILIAN_READY"
                return first
            failures = [
                value for value in first.get("results", {}).values()
                if isinstance(value, dict) and not value.get("success")
            ]
            retryable = bool(
                not first.get("hard_stop")
                and any(
                    value.get("retryable")
                    and not value.get("stop_retries")
                    for value in failures
                )
            )
            if not retryable:
                bailian_runtime["transition"] = "BAILIAN_STALE"
                return first
            second = _execute_bailian_connection_test(
                payload,
                force_new=True,
                validate_embedding=True,
                automatic=True,
            )
            second["reconnect_attempts"] = 2
            bailian_runtime["transition"] = (
                "BAILIAN_READY"
                if second["provider"].get("connection_state") == "READY"
                else "BAILIAN_STALE"
            )
            return second

    def _lazy_revalidate_bailian(capability: str) -> None:
        if provider_manager.mode != "bailian":
            return
        if not _bailian_validation_expired():
            return
        snapshot = provider_manager.snapshot()
        # Missing credentials are a hard local precondition, not a
        # reconnectable network state. Let the existing capability gate emit
        # its friendly response without constructing clients or leaking an
        # internal HTTPException.
        if not snapshot.get("credential", {}).get("configured"):
            return
        # A request-time guard is intentionally one bounded validation, not
        # the two-attempt manual reconnect workflow. VLM requests do not pay
        # for an unrelated Embedding probe; the validated capability itself
        # is the gate authority.
        with bailian_reconnect_lock:
            result = _execute_bailian_connection_test(
                ProviderConnectionTestRequest(
                    cloud_tier=provider_manager.cloud_tier,
                    credential_source=provider_manager.credential_source,
                ),
                force_new=True,
                validate_embedding=capability != "vlm",
                automatic=True,
            )
        capability_ready = bool(
            result.get("provider", {})
            .get("capabilities", {})
            .get(capability)
        )
        if not capability_ready:
            raise CapabilityUnavailable(capability, result["provider"])
        bailian_runtime["stale"] = False
        bailian_runtime["transition"] = (
            f"BAILIAN_{capability.upper()}_READY"
        )
        bailian_runtime["last_validation_automatic"] = True
        bailian_runtime["last_validated_at"] = now_iso()

    provider_manager.set_before_request_hook(_lazy_revalidate_bailian)

    @app.post("/providers/access/preflight")
    def preflight_active_provider(
        capability: Literal["vlm", "embedding", "retrieval"] = "vlm",
    ) -> dict[str, Any]:
        """Run at most one strict capability check before a user request.

        READY snapshots return without network I/O. Recoverable stale states
        use the same provider validation contract as the capability gate.
        """

        before = provider_snapshot(cached_only=True)
        if (
            before.get("connection_state") == "READY"
            and before.get("capabilities", {}).get(capability)
        ):
            return {
                "success": True,
                "provider": before,
                "capability": capability,
                "network_request_count": 0,
                "validation_cycle_count": 0,
                "recovered": False,
            }
        if provider_manager.mode == "bailian":
            try:
                _lazy_revalidate_bailian(capability)
            except CapabilityUnavailable as exc:
                return {
                    "success": False,
                    "provider": exc.snapshot,
                    "capability": capability,
                    "network_request_count_upper_bound": 3,
                    "validation_cycle_count": 1,
                    "recovered": False,
                }
            after = provider_snapshot(cached_only=True)
            return {
                "success": bool(
                    after.get("capabilities", {}).get(capability)
                ),
                "provider": after,
                "capability": capability,
                "network_request_count_upper_bound": 3,
                "validation_cycle_count": 1,
                "recovered": True,
            }
        if provider_manager.mode == "self_hosted":
            result = test_self_hosted_provider()
            return {
                "success": bool(
                    result.get("provider", {})
                    .get("capabilities", {})
                    .get(capability)
                ),
                "provider": result["provider"],
                "capability": capability,
                "network_request_count": 0,
                "validation_cycle_count": 1,
                "recovered": True,
            }
        after = provider_snapshot(cached_only=True)
        return {
            "success": bool(
                after.get("capabilities", {}).get(capability)
            ),
            "provider": after,
            "capability": capability,
            "network_request_count": 0,
            "validation_cycle_count": 0,
            "recovered": False,
        }

    @app.post("/providers/access/test")
    def test_provider_connection(
        payload: ProviderConnectionTestRequest,
    ) -> dict[str, Any]:
        return _execute_bailian_connection_test(payload)

    @app.post("/providers/access/revalidate")
    def revalidate_active_provider() -> dict[str, Any]:
        """Run one explicit bounded validation; status polling never calls models."""
        if provider_manager.mode == "bailian":
            with bailian_reconnect_lock:
                result = _execute_bailian_connection_test(
                    ProviderConnectionTestRequest(
                        cloud_tier=provider_manager.cloud_tier,
                        credential_source=provider_manager.credential_source,
                    ),
                    force_new=True,
                    validate_embedding=True,
                    automatic=True,
                )
                result["trigger"] = "manual_revalidate"
                result["reconnect_attempts"] = 1
                result["paid_periodic_probe"] = False
                result["image_request_count"] = 0
                return result
        if provider_manager.mode == "self_hosted":
            return test_self_hosted_provider()
        if provider_manager.mode == "local":
            return {
                "success": provider_snapshot().get("state") == "READY",
                "trigger": "manual_revalidate",
                "provider": provider_snapshot(),
                "preflight": current_local_preflight(),
                "model_load_started": False,
            }
        return {
            "success": True,
            "trigger": "manual_revalidate",
            "provider": provider_snapshot(),
            "model_request_count": 0,
        }

    @app.get("/providers/access/cloud-index")
    def cloud_index_status() -> dict[str, Any]:
        full_checkpoint_path = cloud_full_index_root.parent / "checkpoint.json"
        train_total = system_library_asset_count("system_train")
        val_total = system_library_asset_count("system_val")
        course_total = train_total + val_total
        full_coverage: dict[str, Any] = {
            "status": "empty_public_release" if course_total == 0 else "not_built",
            "train_completed": 0,
            "train_total": train_total,
            "val_completed": 0,
            "val_total": val_total,
            "total_completed": 0,
            "total": course_total,
            "faiss_ntotal": 0,
        }
        if full_checkpoint_path.is_file():
            try:
                checkpoint = json.loads(
                    full_checkpoint_path.read_text(encoding="utf-8")
                )
                summary = dict(checkpoint.get("summary") or {})
                full_coverage = {
                    **full_coverage,
                    "status": (
                        "ready"
                        if course_total > 0
                        and summary.get("total_completed") == course_total
                        and checkpoint.get("faiss_ntotal") == course_total
                        else "partial"
                    ),
                    "train_completed": int(
                        summary.get("train_completed", 0)
                    ),
                    "val_completed": int(summary.get("val_completed", 0)),
                    "total_completed": int(
                        summary.get("total_completed", 0)
                    ),
                    "faiss_ntotal": int(
                        checkpoint.get("faiss_ntotal", 0)
                    ),
                    "dimension": 2560,
                    "shared_by_cloud_tiers": True,
                }
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if cloud_retrieval is None:
            ready_on_disk = full_coverage["status"] == "ready"
            return {
                "status": (
                    "index_ready_connection_required"
                    if ready_on_disk
                    else "not_configured"
                ),
                "items": int(full_coverage["total_completed"]),
                "base_items": int(full_coverage["total_completed"]),
                "user_items": 0,
                "dimensions": 2560,
                "scope_message": (
                    "当前完整云端检索索引："
                    f"Train {full_coverage['train_completed']}/{train_total} · "
                    f"Val {full_coverage['val_completed']}/{val_total} · "
                    "用户增量资产 0 项 · "
                    f"总计 {full_coverage['total_completed']} 项"
                ),
                "full_course_coverage": full_coverage,
            }
        status = cloud_retrieval.status()
        return {
            **status,
            "full_course_coverage": full_coverage,
            "scope_message": (
                "当前完整云端检索索引："
                f"Train {full_coverage['train_completed']}/{train_total} · "
                f"Val {full_coverage['val_completed']}/{val_total} · "
                f"用户增量资产 {status['user_items']} 项 · "
                f"总计 {status['items']} 项"
            ),
        }

    @app.post("/providers/access/cloud-index/base")
    def build_cloud_base_index(
        payload: CloudIndexBuildRequest,
    ) -> dict[str, Any]:
        full_status = cloud_index_status().get("full_course_coverage", {})
        if (
            full_status.get("status") == "ready"
            and int(full_status.get("total", 0) or 0) > 0
            and int(full_status.get("faiss_ntotal", 0) or 0)
            == int(full_status.get("total", 0) or 0)
        ):
            raise HTTPException(
                status_code=409,
                detail="完整云索引已经建立并注册，无需再次建立 Train 前十索引。",
            )
        if provider_manager.mode != "bailian":
            raise HTTPException(
                status_code=409,
                detail="请先选择百炼云端模型并完成 Embedding 连接测试。",
            )
        try:
            ensure_bailian_runtime()
            if cloud_retrieval is None:
                raise RuntimeError("cloud_retrieval_not_configured")
            if cloud_retrieval.embedding.status().get("status") != "ready":
                raise RuntimeError("cloud_embedding_connection_required")
            rows = cloud_train_items(10)
            status = cloud_retrieval.reconcile(rows, preserve_existing=True)
            write_cloud_index_manifests(requested_base_count=10)
            return {
                "success": True,
                "hard_stop": False,
                "provider": provider_snapshot(),
                "index": status,
                "scope_message": (
                    f"当前云端检索索引：{status['base_items']}项基础资产 + "
                    f"{status['user_items']}项用户资产"
                ),
            }
        except Exception as exc:
            failure = safe_cloud_failure(exc)
            write_cloud_index_manifests(
                requested_base_count=10,
                failure=failure,
            )
            return {
                "success": False,
                "hard_stop": bool(failure.get("stop_retries")),
                "failure": failure,
                "provider": provider_snapshot(),
                "index": (
                    cloud_retrieval.status()
                    if cloud_retrieval is not None
                    else {"status": "not_configured", "items": 0}
                ),
            }

    @app.post("/providers/access/cloud-index/assets")
    def add_cloud_index_assets(
        payload: CloudIndexAssetRequest,
    ) -> dict[str, Any]:
        if provider_manager.mode != "bailian":
            raise HTTPException(
                status_code=409,
                detail="请先选择百炼云端模型并完成 Embedding 连接测试。",
            )
        try:
            ensure_bailian_runtime()
            if cloud_retrieval is None:
                raise RuntimeError("cloud_retrieval_not_configured")
            if int(cloud_retrieval.status().get("base_items", 0) or 0) <= 0:
                raise RuntimeError("cloud_base_index_required")
            cloud_rows = cloud_user_items(payload.asset_ids)
            set_cloud_user_state(cloud_rows, "indexing")
            status = cloud_retrieval.add_assets(cloud_rows)
            set_cloud_user_state(cloud_rows, "completed")
            write_cloud_index_manifests(requested_base_count=10)
            return {
                "success": True,
                "hard_stop": False,
                "provider": provider_snapshot(),
                "index": status,
                "scope_message": (
                    f"当前云端检索索引：{status['base_items']}项基础资产 + "
                    f"{status['user_items']}项用户资产"
                ),
            }
        except Exception as exc:
            failure = safe_cloud_failure(exc)
            try:
                set_cloud_user_state(
                    cloud_user_items(payload.asset_ids),
                    "failed",
                    failure=failure,
                )
            except Exception:
                pass
            return {
                "success": False,
                "hard_stop": bool(failure.get("stop_retries")),
                "failure": failure,
                "provider": provider_snapshot(),
                "index": (
                    cloud_retrieval.status()
                    if cloud_retrieval is not None
                    else {"status": "not_configured", "items": 0}
                ),
            }

    @app.get("/providers/access/local/preflight")
    def local_provider_preflight() -> dict[str, Any]:
        preflight = current_local_preflight()
        return {
            **preflight,
            "index_contract": local_index_reuse_contract(
                preflight=preflight,
                index_path=local_product_index_path,
            ),
        }

    @app.get("/providers/access/local/index-contract")
    def local_provider_index_contract() -> dict[str, Any]:
        preflight = current_local_preflight()
        return local_index_reuse_contract(
            preflight=preflight,
            index_path=local_product_index_path,
        )

    @app.post("/providers/access/local/unload")
    def unload_local_provider() -> dict[str, Any]:
        cleanup_local_runtime()
        provider_manager.clear_errors()
        provider_manager.set_connection_state(
            None,
            transition="LOCAL_UNLOADED",
        )
        return {
            "success": True,
            "cleanup_completed": True,
            "backend_alive": True,
            "provider": provider_snapshot(),
        }

    @app.post("/providers/access/local/load")
    def load_local_provider(
        payload: LocalProviderLoadRequest,
    ) -> dict[str, Any]:
        if provider_manager.mode != "local":
            try:
                provider_manager.select(mode="local")
            except ProviderSwitchBusy:
                raise HTTPException(
                    status_code=409,
                    detail="当前仍有模型请求正在运行，请等待完成后再加载本地模型。",
                ) from None
        preflight = current_local_preflight()
        if not preflight["can_attempt"]:
            cleanup_local_runtime()
            blockers = list(preflight.get("hard_blockers") or [])
            blocker_codes = [
                str(item.get("code"))
                for item in blockers
                if item.get("code")
            ]
            primary_code = (
                blocker_codes[0]
                if blocker_codes
                else "LOCAL_PREFLIGHT_BLOCKED"
            )
            error = ProviderError(
                "local_preflight",
                primary_code,
                False,
                True,
                str(preflight.get("conclusion")),
            )
            provider_manager.set_error("local", error)
            return {
                "success": False,
                "stage": "preflight",
                "load_started": False,
                "requires_confirmation": False,
                "preflight": preflight,
                "failure": {
                    **error.as_dict(),
                    "reason_codes": blocker_codes,
                    "missing_models": [
                        item.get("label")
                        for item in blockers
                        if "WEIGHTS_MISSING" in str(item.get("code"))
                    ],
                    "recommended_model_paths": preflight.get(
                        "recommended_model_paths"
                    ),
                },
                "provider": provider_snapshot(),
                "backend_alive": True,
            }
        if not preflight["recommended"] and not payload.force_low_vram_attempt:
            return {
                "success": False,
                "stage": "preflight",
                "load_started": False,
                "requires_confirmation": True,
                "preflight": preflight,
                "failure": {
                    "category": "low_vram_warning",
                    "code": "LOCAL_FREE_VRAM_BELOW_RECOMMENDED",
                    "public_message": str(preflight.get("conclusion")),
                },
                "provider": provider_snapshot(),
                "backend_alive": True,
            }
        cleanup_local_runtime()
        stages = [
            {"name": "检查环境", "status": "completed"},
            {"name": "加载 VLM", "status": "pending"},
            {"name": "加载 Embedding", "status": "pending"},
            {"name": "检查本地索引", "status": "pending"},
        ]
        try:
            local_vlm_service = PersistentQwen3VLService(
                local_vlm_path,
                settings.project_root / "prompts" / "phase1",
                core_registry_path=(
                    settings.project_root
                    / "prompts"
                    / "gate1"
                    / "p3_registry.json"
                ),
                core_prompt_version=settings.core_prompt_version,
            )
            local_vlm_provider = LocalVLMProvider(local_vlm_service)
            stages[1]["status"] = "running"
            local_vlm_provider.load()
            stages[1]["status"] = "completed"

            local_embedding_service = LocalQwenVLEmbeddingService(
                local_embedding_path,
                local_embedding_source_path,
            )
            local_embedding_provider = LocalEmbeddingProvider(
                local_embedding_service
            )
            stages[2]["status"] = "running"
            local_embedding_provider.load()
            stages[2]["status"] = "completed"

            local_retrieval: Any | None = None
            stages[3]["status"] = "running"
            index_contract = local_index_reuse_contract(
                preflight=preflight,
                index_path=local_product_index_path,
            )
            if index_contract["reusable"]:
                local_retrieval = E1RetrievalAdapter(
                    e1_embedding=local_embedding_provider,
                    e1_index=FaissRetrievalIndex(
                        local_product_index_path,
                        dimensions=2048,
                        lifecycle_path=(
                            local_index_root / "product" / "lifecycle.json"
                        ),
                        asset_path_resolver=portable_asset_resolver.resolve,
                        asset_path_serializer=portable_asset_resolver.serialize,
                    ),
                    r0=r0_retrieval,
                    requested_backend="e1",
                    fallback_backend="r0",
                )
                local_retrieval.load()
                if (
                    local_retrieval.status().get("active_backend")
                    != "e1"
                ):
                    local_retrieval = None
                    stages[3]["status"] = "not_built"
                else:
                    stages[3]["status"] = "completed"
            else:
                stages[3]["status"] = "not_built"
            local_split_retrieval = SplitE1IndexRegistry(
                local_system_index_root,
                embedding=local_embedding_provider,
                asset_path_resolver=portable_asset_resolver.resolve,
                asset_path_serializer=portable_asset_resolver.serialize,
            )
            if local_split_retrieval.status().get("status") != "ready":
                raise RuntimeError("local_system_indices_not_ready")
            local_runtime.update(
                {
                    "vlm_provider": local_vlm_provider,
                    "embedding_provider": local_embedding_provider,
                    "retrieval": local_retrieval,
                    "split_retrieval": local_split_retrieval,
                }
            )
            provider_manager.clear_errors()
            provider_manager.install_runtime(
                "local",
                vlm=local_vlm_provider,
                retrieval=local_retrieval,
                embedding=local_embedding_provider,
            )
            snapshot = provider_snapshot()
            return {
                "success": True,
                "stage": "completed",
                "preflight": preflight,
                "stages": stages,
                "provider": snapshot,
                "backend_alive": True,
                "automatic_download": False,
                "quantization_changed": False,
                "index_contract": index_contract,
            }
        except Exception as exc:
            cleanup_local_runtime()
            lowered = str(exc).lower()
            if (
                "out of memory" in lowered
                or "insufficient free gpu memory" in lowered
            ):
                category = "cuda_oom"
                code = "LOCAL_CUDA_OUT_OF_MEMORY"
            elif isinstance(exc, FileNotFoundError):
                category = "model_weights_missing"
                code = "LOCAL_MODEL_FILE_NOT_FOUND"
            elif "cuda" in lowered:
                category = "cuda_runtime"
                code = "LOCAL_CUDA_INITIALIZATION_FAILED"
            else:
                category = "model_initialization"
                code = "LOCAL_MODEL_INITIALIZATION_FAILED"
            public = (
                "本地模型加载失败。当前可用显存不足以同时运行 "
                "Qwen3-VL-4B 与 Qwen3-VL-Embedding-2B。"
                "你可以改用百炼云端模式、服务器演示模式，"
                "或关闭其他占用显存的程序后重试。"
                if category == "cuda_oom"
                else (
                    "本地模型加载未完成，已清理本次加载对象；"
                    "百炼云端和服务器映射仍可继续使用。"
                    if category == "model_initialization"
                    else (
                        "本地模型文件在加载过程中缺失；系统没有自动下载。"
                        "请按诊断中建议目录补齐冻结权重后重试。"
                        if category == "model_weights_missing"
                        else (
                            "本地 CUDA 初始化失败，已清理本次加载对象；"
                            "请检查驱动、Torch CUDA 版本和当前 GPU 状态。"
                        )
                    )
                )
            )
            error = ProviderError(
                category,
                code,
                False,
                True,
                public,
            )
            provider_manager.set_error("local", error)
            for stage in stages:
                if stage["status"] == "running":
                    stage["status"] = "failed"
                elif stage["status"] == "pending":
                    stage["status"] = "skipped"
            return {
                "success": False,
                "stage": "load",
                "load_started": True,
                "preflight": preflight,
                "stages": stages,
                "failure": error.as_dict(),
                "provider": provider_snapshot(),
                "backend_alive": True,
                "cleanup_completed": True,
                "automatic_download": False,
                "quantization_changed": False,
            }

    @app.post("/providers/access/self-hosted/test")
    def test_self_hosted_provider() -> dict[str, Any]:
        try:
            if provider_manager.mode != "self_hosted":
                provider_manager.select(mode="self_hosted")
            provider_manager.clear_errors()
            provider_manager.set_connection_state(
                "RECONNECTING",
                transition="SELF_HOSTED_REVALIDATING",
            )
        except ProviderSwitchBusy:
            raise HTTPException(
                status_code=409,
                detail="当前仍有模型请求正在运行，请等待完成后再检查服务器映射。",
            ) from None
        try:
            vlm_health = self_hosted_vlm.health_check()
        except Exception as exc:
            vlm_health = {
                "status": "unavailable",
                "loaded": False,
                "error_code": type(exc).__name__,
            }
        try:
            embedding_health = (
                self_hosted_embedding.health_check()
                if self_hosted_embedding is not None
                else {
                    "status": "not_configured",
                    "dimension_verified": False,
                }
            )
        except Exception as exc:
            embedding_health = {
                "status": "unavailable",
                "loaded": False,
                "dimension_verified": False,
                "error_code": type(exc).__name__,
            }
        vlm_ok = (
            vlm_health.get("status") == "ready"
            and vlm_health.get("loaded") is True
        )
        embedding_ok = (
            embedding_health.get("status") == "ready"
            and embedding_health.get("dimension_verified") is True
            and int(embedding_health.get("dimensions", 0) or 0) == 2048
        )
        if not vlm_ok:
            provider_manager.set_error(
                "vlm",
                ProviderError(
                    "self_hosted_unavailable",
                    "SELF_HOSTED_VLM_UNAVAILABLE",
                    True,
                    False,
                    "服务器演示 VLM 或本地 SSH 隧道暂不可用，请检查作者演示环境后重试。",
                ),
            )
        if not embedding_ok:
            provider_manager.set_error(
                "embedding",
                ProviderError(
                    "self_hosted_unavailable",
                    "SELF_HOSTED_EMBEDDING_UNAVAILABLE",
                    True,
                    False,
                    "服务器演示 Embedding 或本地 SSH 隧道暂不可用，请检查作者演示环境后重试。",
                ),
            )
        if vlm_ok and embedding_ok:
            provider_manager.set_connection_state(
                "READY",
                transition="SELF_HOSTED_READY",
            )
        elif vlm_ok or embedding_ok:
            provider_manager.set_connection_state(
                "PARTIAL_READY",
                transition="SELF_HOSTED_PARTIAL_READY",
            )
        else:
            provider_manager.set_connection_state(
                "OFFLINE",
                transition="SELF_HOSTED_OFFLINE",
            )
        public_message = (
            "服务器映射当前可用。"
            if vlm_ok and embedding_ok
            else (
                "服务器映射当前部分可用；VLM 与 Embedding 已分别显示状态，"
                "不可用的一侧不会影响图片、会话和已有索引。"
                if vlm_ok or embedding_ok
                else (
                    "服务器映射当前不可用。请确认课程演示服务器与 SSH 隧道已启动，"
                    "或切换到百炼云端模型。您的图片、会话和本地索引不会受到影响。"
                )
            )
        )
        return {
            "success": vlm_ok and embedding_ok,
            "public_message": public_message,
            "tunnel_status": (
                "ready"
                if vlm_ok and embedding_ok
                else "partial"
                if vlm_ok or embedding_ok
                else "unavailable"
            ),
            "vlm": {
                "success": vlm_ok,
                "status": vlm_health.get("status"),
                "model": vlm_health.get("model"),
                "model_revision": vlm_health.get("model_revision"),
                "error_code": vlm_health.get("error_code"),
            },
            "embedding": {
                "success": embedding_ok,
                "status": embedding_health.get("status"),
                "model": embedding_health.get("model"),
                "model_revision": embedding_health.get("model_revision"),
                "dimensions": embedding_health.get("dimensions"),
                "error_code": embedding_health.get("error_code"),
            },
            "provider": provider_snapshot(),
            "course_demo_only": True,
            "server_address_exposed": False,
            "ssh_credentials_exposed": False,
            "remote_process_restarted": False,
            "tunnel_restarted": False,
        }

    def cached_runtime_status() -> dict[str, Any]:
        return {
            "vlm": (
                vlm_service.cached_status()
                if callable(getattr(vlm_service, "cached_status", None))
                else {"status": "not_validated", "loaded": False}
            ),
            "retrieval": (
                retrieval.cached_status()
                if callable(getattr(retrieval, "cached_status", None))
                else {"status": "not_validated", "loaded": False}
            ),
            "provider": provider_snapshot(cached_only=True),
        }

    @app.get("/health/live")
    def health_live() -> dict[str, Any]:
        return {
            "status": "ok",
            "liveness": "alive",
            "git_commit": settings.git_commit,
        }

    @app.get("/health/ready")
    def health_ready() -> dict[str, Any]:
        runtime = cached_runtime_status()
        provider_state = str(runtime["provider"].get("state", "CONNECTING"))
        readiness = "ready" if provider_state == "READY" else "degraded"
        return {
            "status": readiness,
            "readiness": readiness,
            "provider_state": provider_state,
            **runtime,
            "git_commit": settings.git_commit,
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        runtime = cached_runtime_status()
        provider_state = str(runtime["provider"].get("state", "CONNECTING"))
        return {
            "status": "ok",
            "liveness": "alive",
            "readiness": "ready" if provider_state == "READY" else "degraded",
            "library_items": len(library.list_assets()),
            "vlm": runtime["vlm"],
            "retrieval": runtime["retrieval"],
            "product": product.status(),
            "provider": runtime["provider"],
            "git_commit": settings.git_commit,
        }

    @app.get("/library")
    def list_library() -> dict[str, Any]:
        assets = library.list_assets()
        for asset in assets:
            asset["image_url"] = f"/library/{asset['image_id']}/image"
            lifecycle = (
                legacy_lifecycle_registry.lookup(
                    {
                        "asset_id": asset["image_id"],
                        "image_id": asset["image_id"],
                        "source": "frozen_library",
                    }
                )
                if legacy_lifecycle_registry is not None
                else {
                    "lifecycle_state": "active",
                    "searchable": True,
                    "lifecycle_label": "活动资产",
                }
            )
            asset.update(
                lifecycle_state=lifecycle["lifecycle_state"],
                searchable=lifecycle["searchable"],
                lifecycle_label=lifecycle["lifecycle_label"],
            )
        return {"count": len(assets), "items": assets}

    @app.get("/canonical-preview")
    def canonical_preview_list() -> dict[str, Any]:
        return {
            "enabled": canonical_preview.enabled(),
            "mode": "candidate_preview_only",
            "context_selectable": False,
            "count": len(canonical_preview.list_items()),
            "items": canonical_preview.list_items(),
        }

    @app.get("/canonical-preview/{image_id}")
    def canonical_preview_detail(image_id: str) -> dict[str, Any]:
        try:
            return canonical_preview.item(image_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail="canonical_preview_item_not_found",
            ) from None

    @app.get("/canonical-preview/{image_id}/image")
    def canonical_preview_image(image_id: str) -> FileResponse:
        try:
            return FileResponse(canonical_preview.image_path(image_id))
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail="canonical_preview_item_not_found",
            ) from None
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="canonical_preview_image_not_available",
            ) from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @app.get("/library/{image_id}")
    def library_detail(image_id: str) -> dict[str, Any]:
        try:
            return {
                "image_id": image_id,
                "intelligence": library.historical_intelligence(image_id),
                "ocr": library.ocr_evidence(image_id),
                "lifecycle": (
                    legacy_lifecycle_registry.lookup(
                        {
                            "asset_id": image_id,
                            "image_id": image_id,
                            "source": "frozen_library",
                        }
                    )
                    if legacy_lifecycle_registry is not None
                    else {
                        "lifecycle_state": "active",
                        "searchable": True,
                        "lifecycle_label": "活动资产",
                    }
                ),
            }
        except KeyError:
            raise HTTPException(status_code=404, detail="image_not_in_frozen_library") from None

    @app.get("/library/{image_id}/image")
    def library_image(image_id: str) -> FileResponse:
        try:
            return FileResponse(library.image_path(image_id, verify_hash=True))
        except KeyError:
            raise HTTPException(status_code=404, detail="image_not_in_frozen_library") from None
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="image_file_not_available") from None

    def run_or_http_error(function: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except LibraryMutationForbidden as exc:
            raise HTTPException(
                status_code=403,
                detail=exc.public_message,
            ) from None
        except WorkspaceVersionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown_image:{exc.args[0]}") from None
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise HTTPException(status_code=503 if "disabled" in str(exc) or "not_built" in str(exc) else 400, detail=str(exc)) from None

    @app.post("/analyze")
    def analyze(request: AnalyzeRequest) -> dict[str, Any]:
        return run_or_http_error(orchestrator.analyze, request.image_id, request.prompt_version)

    @app.post("/index/rebuild")
    def rebuild_index() -> dict[str, Any]:
        if provider_manager.mode == "bailian":
            raise HTTPException(
                status_code=409,
                detail="百炼模式使用独立云索引，请在“模型接入”中建立 Train 前十云索引。",
            )
        return run_or_http_error(rebuild_all_index)

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        return run_or_http_error(orchestrator.search, request.query, request.top_k)

    @app.post("/search/image")
    def search_image(request: ImageSearchRequest) -> dict[str, Any]:
        if bool(request.image_id) == bool(request.local_asset_id):
            raise HTTPException(status_code=422, detail="provide_exactly_one_query_image")
        query_id = request.image_id or request.local_asset_id or ""
        if request.image_id:
            image_path = run_or_http_error(library.image_path, request.image_id, verify_hash=True)
            exclude_id = request.image_id if request.exclude_self else None
        else:
            local_item = run_or_http_error(product.asset, request.local_asset_id or "")
            image_path = Path(local_item["path"])
            exclude_id = f"local:{request.local_asset_id}" if request.exclude_self else None
        trace = traces.start("image_search", [query_id], model=retrieval.embedding.status().get("model"), model_revision=retrieval.embedding.status().get("model_revision"), prompt_version="not_applicable", schema_version="phase5_image_search_v1", services=["library", "embedding", "retrieval"])
        try:
            results = retrieval.search_image(image_path, request.top_k, exclude_image_id=exclude_id)
            if results:
                trace["retrieval"] = {
                    key: results[0].get(key)
                    for key in (
                        "retrieval_backend",
                        "model",
                        "revision",
                        "index_version",
                        "fallback_used",
                        "fallback_reason",
                    )
                }
            finished = traces.finish(trace, status="success", output_path=str(retrieval.index_path))
            return {"query_asset_id": query_id, "query_source": "frozen_library" if request.image_id else "local_upload", "results": results, "same_image_excluded": bool(exclude_id), "request_id": finished["request_id"], "trace": finished}
        except Exception as exc:
            traces.finish(trace, status="failed", error=exc)
            return run_or_http_error(lambda: (_ for _ in ()).throw(exc))

    @app.post("/search/hybrid")
    def search_hybrid(request: HybridSearchRequest) -> dict[str, Any]:
        if bool(request.image_id) == bool(request.local_asset_id):
            raise HTTPException(status_code=422, detail="provide_exactly_one_query_image")
        query_id = request.image_id or request.local_asset_id or ""
        if request.image_id:
            image_path = run_or_http_error(library.image_path, request.image_id, verify_hash=True)
            exclude_id = request.image_id if request.exclude_self else None
        else:
            local_item = run_or_http_error(product.asset, request.local_asset_id or "")
            image_path = Path(local_item["path"])
            exclude_id = f"local:{request.local_asset_id}" if request.exclude_self else None
        trace = traces.start("hybrid_search", [query_id], model=retrieval.embedding.status().get("model"), model_revision=retrieval.embedding.status().get("model_revision"), prompt_version="not_applicable", schema_version="phase5_hybrid_search_v1", services=["library", "embedding", "retrieval", "lexical"])
        try:
            results = retrieval.search_hybrid(image_path, request.query, request.top_k, image_weight=request.image_weight, text_weight=request.text_weight, lexical_weight=request.lexical_weight, exclude_image_id=exclude_id)
            if results:
                trace["retrieval"] = {
                    key: results[0].get(key)
                    for key in (
                        "retrieval_backend",
                        "model",
                        "revision",
                        "index_version",
                        "fallback_used",
                        "fallback_reason",
                    )
                }
            finished = traces.finish(trace, status="success", output_path=str(retrieval.index_path))
            return {"query_asset_id": query_id, "query_source": "frozen_library" if request.image_id else "local_upload", "query_text": request.query, "results": results, "same_image_excluded": bool(exclude_id), "request_id": finished["request_id"], "trace": finished}
        except Exception as exc:
            traces.finish(trace, status="failed", error=exc)
            return run_or_http_error(lambda: (_ for _ in ()).throw(exc))

    @app.post("/vqa")
    def vqa(request: VQARequest) -> dict[str, Any]:
        existing = product.get_session(request.conversation_id) if request.conversation_id else None
        if existing is not None and existing.get("asset_id") != request.image_id:
            raise HTTPException(status_code=409, detail="conversation_asset_mismatch")
        history = list(existing.get("messages", []))[-12:] if existing else []
        result = run_or_http_error(orchestrator.answer_question, request.image_id, request.question, history)
        conversation_id = request.conversation_id if hasattr(request, "conversation_id") and request.conversation_id else str(uuid.uuid4())
        session = existing or {
            "conversation_id": conversation_id,
            "asset_id": request.image_id,
            "messages": [],
            "created_at": now_iso(),
        }
        session["messages"].extend([
            {"role": "user", "content": request.question, "created_at": now_iso()},
            {"role": "assistant", "content": result["result"]["data"].get("parsed_output", {}), "trace_id": result["request_id"], "created_at": now_iso()},
        ])
        session["updated_at"] = now_iso()
        product.save_session(session)
        return {**result, "conversation_id": conversation_id, "messages": session["messages"]}

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        return run_or_http_error(
            orchestrator.generate_content,
            request.image_ids,
            {"tone": request.tone, "audience": request.audience, "length": request.length, "style": request.style},
        )

    @app.post("/describe")
    def describe(request: DescribeRequest) -> dict[str, Any]:
        payload = run_or_http_error(
            orchestrator.describe_image,
            request.image_id,
            {"length": request.length, "style": request.style},
        )
        return {
            **payload,
            "display_text": extract_display_text(
                payload,
                fallback="本次图片描述未成功，请稍后重试。",
            ),
        }

    @app.get("/trace/{request_id}")
    def trace(request_id: str) -> dict[str, Any]:
        try:
            return traces.get(request_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="trace_not_found") from None

    @app.post("/imports")
    def import_assets(request: ImportRequest) -> dict[str, Any]:
        run_or_http_error(
            product.assert_library_mutable,
            request.library_id,
            "batch_import",
        )
        task = product.create_task("batch_import", len(request.files), metadata={"library_id": request.library_id})
        product.update_task(task["task_id"], status="running")
        imported: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for item in request.files:
            try:
                imported.append(product.import_base64(item.name, item.content_base64, library_id=request.library_id))
            except ValueError as exc:
                errors.append({"name": item.name, "error": str(exc)})
        final = product.update_task(task["task_id"], status="completed" if not errors else "partial", completed=len(imported), failed=len(errors), errors=errors)
        index_update = (
            incremental_index_user_assets(
                [str(item["asset_id"]) for item in imported]
            )
            if imported
            else {
                "status": "not_applicable",
                "encoded": 0,
                "appended": 0,
                "rebuilt": False,
            }
        )
        index = index_update.get("index") or legacy_retrieval.status()
        cloud_index_update: dict[str, Any] = {
            "status": "not_applicable",
            "message": "当前未使用已就绪的百炼云端索引。",
        }
        if imported and provider_manager.mode == "bailian":
            cloud_index_update = dict(index_update)
            cloud_index_update["message"] = {
                "completed": "云端用户资产索引已增量更新。",
                "waiting_for_ready_provider": (
                    "图片已持久保存；完成百炼 Embedding 连接后可补齐云端索引。"
                ),
                "failed": "图片已持久保存；云端索引更新未完成。",
            }.get(str(index_update.get("status")), "图片已持久保存。")
        return {
            "task": final,
            "assets": imported,
            "errors": errors,
            "index": index,
            "index_update": index_update,
            "cloud_index_update": cloud_index_update,
        }

    @app.get("/tasks")
    def tasks() -> dict[str, Any]:
        items = product.tasks()
        return {"count": len(items), "items": items}

    @app.get("/libraries")
    def libraries() -> dict[str, Any]:
        custom = product.libraries()
        current_split = active_split_retrieval()
        split_index_status = (
            current_split.status().get("libraries", {})
            if current_split is not None
            else {}
        )
        custom_counts = {
            item["library_id"]: len(product.assets(item["library_id"]))
            for item in custom
        }
        items = [
            {
                **item,
                "index_version": (
                    split_index_status.get(item["library_id"], {}).get(
                        "index_version"
                    )
                ),
                "embedding_status": split_index_status.get(
                    item["library_id"],
                    {},
                ).get("status", "not_built"),
            }
            for item in system_libraries.libraries()
        ] + [
            {
                **item,
                "asset_count": custom_counts[item["library_id"]],
                "labeled_count": sum(
                    bool(
                        product.asset(asset["asset_id"]).get(
                            "analysis_status"
                        )
                        == "completed"
                    )
                    for asset in product.assets(item["library_id"])
                ),
                "permissions": {
                    "browse": True,
                    "select": True,
                    "upload": True,
                    "delete_asset": True,
                    "move_asset": True,
                    "rename_library": True,
                    "delete_library": True,
                    "replace_canonical": False,
                },
            }
            for item in custom
        ]
        return {"count": len(items), "items": items}

    @app.post("/libraries")
    def create_library(request: LibraryRequest) -> dict[str, Any]:
        return run_or_http_error(
            product.create_library,
            request.name,
            description=request.description,
        )

    @app.patch("/libraries/{library_id}")
    def rename_library(library_id: str, request: LibraryRequest) -> dict[str, Any]:
        run_or_http_error(
            product.assert_library_mutable,
            library_id,
            "rename_library",
        )
        return run_or_http_error(product.rename_library, library_id, request.name)

    @app.delete("/libraries/{library_id}")
    def delete_library(library_id: str) -> dict[str, Any]:
        run_or_http_error(
            product.assert_library_mutable,
            library_id,
            "delete_library",
        )
        return run_or_http_error(product.delete_library, library_id)

    @app.get("/visual-libraries")
    def visual_libraries() -> dict[str, Any]:
        return libraries()

    @app.get("/visual-libraries/{library_id}/assets")
    def visual_library_assets(
        library_id: str,
        page: int = 1,
        page_size: int = 40,
        q: str | None = None,
        theme: str | None = None,
        micro_tag: str | None = None,
        label_status: str | None = None,
        review_status: str | None = None,
        needs_review: bool | None = None,
        sort: str = "sequence_asc",
    ) -> dict[str, Any]:
        if library_id in {"system_train", "system_val"}:
            return run_or_http_error(
                system_libraries.query,
                library_id,
                page=page,
                page_size=page_size,
                q=q,
                theme=theme,
                micro_tag=micro_tag,
                label_status=label_status,
                review_status=review_status,
                needs_review=needs_review,
                sort=sort,
            )
        values = []
        for raw_item in product.assets(library_id):
            detail = product.asset(raw_item["asset_id"])
            canonical_record = (
                (detail.get("analysis") or {}).get("canonical")
                if isinstance(detail.get("analysis"), dict)
                else None
            )
            two_layer = (
                canonical_record.get("two_layer")
                if isinstance(canonical_record, dict)
                else None
            )
            values.append(
                {
                    **raw_item,
                    "analysis_status": detail.get("analysis_status"),
                    "label_status": (
                        "machine_provisional"
                        if canonical_record
                        and canonical_record.get("status") == "completed"
                        else "pending"
                    ),
                    "two_layer": two_layer,
                }
            )
        custom_library = next(
            (
                item
                for item in product.libraries()
                if item.get("library_id") == library_id
            ),
            None,
        )
        if custom_library is None:
            raise HTTPException(status_code=404, detail="library_not_found")
        needle = (q or "").strip().casefold()
        if needle:
            values = [
                item
                for item in values
                if needle
                in " ".join(
                    [
                        str(item.get("asset_id", "")),
                        str(item.get("image_id", "")),
                        str(
                            (
                                (item.get("two_layer") or {}).get(
                                    "default"
                                )
                                or {}
                            ).get("主题", "")
                        ),
                        str(
                            (
                                (item.get("two_layer") or {}).get(
                                    "default"
                                )
                                or {}
                            ).get("简短描述", "")
                        ),
                        " ".join(
                            str(value)
                            for value in (
                                (
                                    (item.get("two_layer") or {}).get(
                                        "default"
                                    )
                                    or {}
                                ).get("微标签", [])
                                or []
                            )
                        ),
                    ]
                ).casefold()
            ]
        if label_status:
            values = [
                item
                for item in values
                if item.get("label_status") == label_status
            ]
        values.sort(
            key=lambda item: (
                str(item.get("image_id", "")).casefold(),
                str(item.get("asset_id", "")),
            ),
            reverse=sort.endswith("_desc"),
        )
        total = len(values)
        start = (page - 1) * page_size
        return {
            "library_id": library_id,
            "page": page,
            "page_size": page_size,
            "total": total,
            "page_count": (total + page_size - 1) // page_size,
            "items": [
                {
                    **item,
                    "image_url": f"/local-assets/{item['asset_id']}/image",
                    "cloud_index": cloud_user_asset_status(item),
                    "index_lifecycle": asset_lifecycle_status(
                        product.asset(str(item["asset_id"]))
                    ),
                    "locked": False,
                    "library_type": "user_custom",
                    "selectable": True,
                }
                for item in values[start : start + page_size]
            ],
        }

    @app.get("/visual-assets/{asset_id}")
    def visual_asset_detail(asset_id: str) -> dict[str, Any]:
        return run_or_http_error(system_libraries.asset, asset_id)

    @app.post("/visual-libraries/search")
    def search_visual_libraries(
        request: VisualLibrarySearchRequest,
    ) -> dict[str, Any]:
        current_split = active_split_retrieval()
        if provider_manager.mode != "bailian" and current_split is None:
            raise HTTPException(
                status_code=503,
                detail="system_e1_retrieval_disabled",
            )
        query_asset = (
            run_or_http_error(
                resolve_course_assets,
                [request.query_asset_ref],
            )[0]
            if request.query_asset_ref
            else None
        )
        excluded = (
            {str(query_asset["asset_id"])}
            if query_asset and request.exclude_query_image
            else set()
        )
        if provider_manager.mode == "bailian":
            results = run_or_http_error(
                search_persistent_libraries,
                requested_library_ids=request.library_ids,
                query_text=request.query_text,
                image_path=query_asset["path"] if query_asset else None,
                top_k=request.top_k,
                exclude_asset_ids=excluded,
            )
            backend = "bailian_cloud_e1"
            public_status = retrieval.status()
        else:
            results = run_or_http_error(
                current_split.search,
                library_ids=request.library_ids,
                query_text=request.query_text,
                image_path=query_asset["path"] if query_asset else None,
                top_k=request.top_k,
                exclude_asset_ids=excluded,
            )
            results = _decorate_retrieval_sources(results)
            backend = "e1"
            public_status = current_split.public_status()
        return {
            "mode": (
                "hybrid"
                if request.query_text and query_asset
                else "image"
                if query_asset
                else "text"
            ),
            "library_ids": request.library_ids,
            "results": results,
            "top_k": request.top_k,
            "retrieval_backend": backend,
            "fallback_used": False,
            "r0_fallback_triggered": False,
            "status": public_status,
        }

    @app.get("/visual-assets/{asset_id}/image")
    def visual_asset_image(asset_id: str) -> FileResponse:
        path = run_or_http_error(system_libraries.image_path, asset_id)
        return FileResponse(path)

    @app.get("/visual-assets/{asset_id}/thumbnail")
    def visual_asset_thumbnail(asset_id: str) -> FileResponse:
        path = run_or_http_error(system_libraries.thumbnail_path, asset_id)
        return FileResponse(path, media_type="image/webp")

    @app.delete("/visual-assets/{asset_id}")
    def delete_system_asset(asset_id: str) -> dict[str, Any]:
        run_or_http_error(system_libraries.asset, asset_id)
        raise HTTPException(
            status_code=403,
            detail="系统训练/验证图片库资产为只读，不能删除。",
        )

    @app.post("/visual-assets/{asset_id}/move")
    def move_system_asset(
        asset_id: str,
        request: MoveAssetRequest,
    ) -> dict[str, Any]:
        del request
        run_or_http_error(system_libraries.asset, asset_id)
        raise HTTPException(
            status_code=403,
            detail="系统训练/验证图片库资产为只读，不能移动。",
        )

    @app.get("/local-assets")
    def local_assets(library_id: str | None = None) -> dict[str, Any]:
        items = product.assets(library_id)
        for item in items:
            item["image_url"] = f"/local-assets/{item['asset_id']}/image"
            item["cloud_index"] = cloud_user_asset_status(item)
            item["index_lifecycle"] = asset_lifecycle_status(
                product.asset(str(item["asset_id"]))
            )
        return {"count": len(items), "items": items}

    @app.get("/local-assets/{asset_id}/image")
    def local_asset_image(asset_id: str) -> FileResponse:
        item = next((value for value in product.assets() if value.get("asset_id") == asset_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail="local_asset_not_found")
        path = Path(item["path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="local_asset_file_missing")
        return FileResponse(path)

    @app.get("/assets/{asset_id:path}/content")
    def authoritative_asset_content(asset_id: str) -> FileResponse:
        try:
            path, media_type = asset_media.media_path(
                asset_id,
                thumbnail=False,
            )
        except (KeyError, FileNotFoundError):
            raise HTTPException(
                status_code=404,
                detail="asset_media_not_found",
            ) from None
        return FileResponse(path, media_type=media_type)

    @app.get("/assets/{asset_id:path}/thumbnail")
    def authoritative_asset_thumbnail(asset_id: str) -> FileResponse:
        try:
            path, media_type = asset_media.media_path(
                asset_id,
                thumbnail=True,
            )
        except (KeyError, FileNotFoundError):
            raise HTTPException(
                status_code=404,
                detail="asset_media_not_found",
            ) from None
        return FileResponse(path, media_type=media_type)

    @app.get("/local-assets/{asset_id}")
    def local_asset_detail(asset_id: str) -> dict[str, Any]:
        item = run_or_http_error(product.asset, asset_id)
        item["cloud_index"] = cloud_user_asset_status(item)
        item["index_lifecycle"] = asset_lifecycle_status(item)
        canonical = (
            (item.get("analysis") or {}).get("canonical")
            if isinstance(item.get("analysis"), dict)
            else None
        )
        if isinstance(canonical, dict):
            canonical_label = canonical.get("canonical")
            stored_layer = canonical.get("two_layer")
            item["two_layer"] = (
                build_two_layer_display(
                    canonical_label,
                    developer=(
                        stored_layer.get("developer")
                        if isinstance(stored_layer, dict)
                        and isinstance(stored_layer.get("developer"), dict)
                        else None
                    ),
                )
                if isinstance(canonical_label, dict)
                else stored_layer
            )
            item["canonical_status"] = canonical.get("status")
        return item

    @app.delete("/local-assets/{asset_id}")
    def delete_local_asset(asset_id: str) -> dict[str, Any]:
        existing = run_or_http_error(product.asset, asset_id)
        result = run_or_http_error(product.delete_asset, asset_id)
        if (
            settings.enable_embedding
            and provider_manager.mode in {"self_hosted", "local"}
        ):
            result["index"] = run_or_http_error(rebuild_all_index)
            result["index_updated"] = True
        elif (
            provider_manager.mode == "bailian"
            and cloud_retrieval is not None
            and int(cloud_retrieval.status().get("base_items", 0) or 0) > 0
        ):
            remaining = [
                item
                for item in cloud_retrieval.e1_index.records
                if (
                    str(item.get("asset_id")) != f"local:{asset_id}"
                    and str(item.get("sha256")) != str(existing.get("sha256"))
                )
            ]
            result["index"] = cloud_retrieval.reconcile(
                remaining,
                preserve_existing=False,
            )
            result["index_updated"] = True
        else:
            result["index"] = legacy_retrieval.status()
            result["index_updated"] = False
        return result

    @app.post("/local-assets/{asset_id}/move")
    def move_local_asset(
        asset_id: str,
        request: MoveAssetRequest,
    ) -> dict[str, Any]:
        return run_or_http_error(
            product.move_asset,
            asset_id,
            request.target_library_id,
        )

    @app.post("/session-assets/import")
    def import_session_assets(request: SessionImportRequest) -> dict[str, Any]:
        imported = [
            run_or_http_error(
                product.import_session_base64,
                item.name,
                item.content_base64,
                conversation_id=request.conversation_id,
            )
            for item in request.files
        ]
        return {
            "conversation_id": request.conversation_id,
            "persistent": False,
            "count": len(imported),
            "items": [
                {
                    **item,
                    "image_url": (
                        f"/session-assets/{item['asset_id']}/image"
                        f"?conversation_id={request.conversation_id}"
                    ),
                }
                for item in imported
            ],
        }

    @app.get("/session-assets/{asset_id}")
    def session_asset_detail(
        asset_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        item = run_or_http_error(
            product.session_asset,
            asset_id,
            conversation_id,
        )
        canonical = item.get("canonical")
        if isinstance(canonical, dict):
            item["two_layer"] = canonical.get("two_layer")
        return item

    @app.post("/session-assets/{asset_id}/persist")
    def persist_session_asset(
        asset_id: str,
        request: PersistSessionAssetRequest,
    ) -> dict[str, Any]:
        temporary = run_or_http_error(
            product.session_asset,
            asset_id,
            request.conversation_id,
        )
        result = run_or_http_error(
            product.persist_session_asset,
            asset_id,
            conversation_id=request.conversation_id,
            library_id=request.library_id,
        )
        canonical_record = temporary.get("canonical")
        canonical_label = (
            canonical_record.get("canonical")
            if isinstance(canonical_record, dict)
            else None
        )
        if isinstance(canonical_label, dict):
            persistent = result["persistent_asset"]
            rebound_source = copy.deepcopy(canonical_label)
            rebound_source["asset"].update(
                asset_id=persistent["asset_id"],
                sha256=persistent["sha256"],
                split="user_custom",
                relative_path=persistent.get("storage_key")
                or persistent.get("image_id"),
            )
            rebound_source.pop("canonical_sha256", None)
            rebound = upgrade_phase6_1_compatible_label(
                rebound_source,
                asset_split="user_custom",
                source_run_id=(
                    "phase6_3_session_to_persistent_"
                    + str(uuid.uuid4())
                ),
            )
            errors = validate_phase6_1_label(rebound)
            schema_errors = sorted(
                error.message
                for error in canonical_schema_validator.iter_errors(
                    rebound
                )
            )
            if errors or schema_errors:
                raise HTTPException(
                    status_code=422,
                    detail="canonical_migration_failed",
                )
            canonical_path = (
                persistent_canonical_root
                / str(persistent["sha256"])[:2]
                / f"{persistent['sha256']}.json"
            )
            _write_json_atomic(canonical_path, rebound)
            migrated_record = {
                "status": "completed",
                "canonical": rebound,
                "two_layer": build_two_layer_display(
                    rebound,
                    developer={
                        "canonical_path": canonical_path.relative_to(
                            settings.project_root
                        ).as_posix(),
                        "source_scope": "user_custom",
                        "migrated_from_session_asset": asset_id,
                    },
                ),
                "canonical_path": str(canonical_path),
                "model_called": False,
                "migration_source": "session_temporary",
            }
            product.save_canonical_analysis(
                str(persistent["asset_id"]),
                migrated_record,
            )
            result["canonical_migrated"] = True
        persistent = result["persistent_asset"]
        index_update = incremental_index_user_assets(
            [str(persistent["asset_id"])]
        )
        result["index_update"] = index_update
        if provider_manager.mode == "bailian":
            result["cloud_index_update"] = index_update
        elif index_update.get("index"):
            result["index"] = index_update["index"]
        return result

    @app.get("/session-assets/{asset_id}/image")
    def session_asset_image(
        asset_id: str,
        conversation_id: str,
    ) -> FileResponse:
        item = run_or_http_error(
            product.session_asset,
            asset_id,
            conversation_id,
        )
        return FileResponse(Path(item["path"]))

    @app.delete("/session-assets/{asset_id}")
    def delete_session_asset(
        asset_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        return run_or_http_error(
            product.delete_session_asset,
            asset_id,
            conversation_id=conversation_id,
        )

    @app.post("/local-assets/{asset_id}/canonical-label")
    def canonical_label_local_asset(asset_id: str) -> dict[str, Any]:
        item = run_or_http_error(product.asset, asset_id)
        run_or_http_error(
            product.assert_library_mutable,
            str(item.get("library_id") or "default"),
            "generate_canonical_label",
        )
        try:
            result = generate_product_canonical(
                item,
                asset_scope="user_custom",
                evidence_root=(
                    managed_user_assets_root
                    / "canonical_evidence"
                    / str(item["sha256"])[:2]
                    / str(item["sha256"])
                ),
            )
        except CanonicalGenerationError as exc:
            product.save_canonical_failure(asset_id, exc.failure)
            raise HTTPException(
                status_code=503
                if exc.failure.get("category")
                in {
                    "billing_or_quota",
                    "invalid_api_key",
                    "permission",
                    "region_endpoint_mismatch",
                    "waiting_for_provider",
                    "missing_credentials",
                    "network_unavailable",
                    "rate_limit",
                    "service_unavailable",
                }
                else 422,
                detail=exc.failure["public_message"],
            ) from None
        stored = product.save_canonical_analysis(asset_id, result)
        index_update = incremental_index_user_assets([asset_id])
        result["index_update"] = index_update
        if provider_manager.mode == "bailian":
            result["cloud_index_update"] = index_update
        elif index_update.get("index"):
            result["index"] = index_update["index"]
        return {
            "asset": product.asset(asset_id),
            "analysis": stored,
            **result,
        }

    @app.post("/session-assets/{asset_id}/canonical-label")
    def canonical_label_session_asset(
        asset_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        item = run_or_http_error(
            product.session_asset,
            asset_id,
            conversation_id,
        )
        try:
            result = generate_product_canonical(
                item,
                asset_scope="session_temporary",
                evidence_root=(
                    product.session_assets_root
                    / conversation_id
                    / asset_id
                    / "canonical_evidence"
                ),
            )
        except CanonicalGenerationError as exc:
            product.save_session_canonical_failure(
                asset_id,
                conversation_id=conversation_id,
                failure=exc.failure,
            )
            raise HTTPException(
                status_code=503
                if exc.failure.get("category")
                in {
                    "billing_or_quota",
                    "invalid_api_key",
                    "permission",
                    "region_endpoint_mismatch",
                    "waiting_for_provider",
                    "missing_credentials",
                    "network_unavailable",
                    "rate_limit",
                    "service_unavailable",
                }
                else 422,
                detail=exc.failure["public_message"],
            ) from None
        stored = product.save_session_canonical(
            asset_id,
            conversation_id=conversation_id,
            value=result,
        )
        return {
            "asset": stored,
            **result,
            "persistent_index_modified": False,
        }

    @app.post("/local-assets/{asset_id}/analyze")
    def analyze_local_asset(asset_id: str) -> dict[str, Any]:
        item = run_or_http_error(product.asset, asset_id)
        trace = traces.start("analyze_local_asset", [asset_id], model=vlm_service.status().get("model"), model_revision=vlm_service.status().get("model_revision"), prompt_version="P3 v1.4", schema_version="phase5_local_analysis_v1", services=["product_store", "vlm"])
        try:
            result = vlm_service.analyze_image(Path(item["path"]))
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            stored = product.save_analysis(asset_id, {"status": "completed", "result": result.as_dict(), "trace_id": trace["request_id"]})
            stored["index_update"] = incremental_index_user_assets(
                [asset_id]
            )
            finished = traces.finish(trace, status="success")
            return {"asset": product.asset(asset_id), "analysis": stored, "request_id": finished["request_id"], "trace": finished}
        except Exception as exc:
            product.save_analysis(asset_id, {"status": "failed", "error": str(exc), "trace_id": trace["request_id"]})
            traces.finish(trace, status="failed", error=exc)
            return run_or_http_error(lambda: (_ for _ in ()).throw(exc))

    @app.post("/local-assets/{asset_id}/vqa")
    def vqa_local_asset(asset_id: str, request: LocalVQARequest) -> dict[str, Any]:
        item = run_or_http_error(product.asset, asset_id)
        trace = traces.start(
            "vqa_local_asset",
            [asset_id],
            model=vlm_service.status().get("model"),
            model_revision=vlm_service.status().get("model_revision"),
            prompt_version="phase1_mvp_vqa_v1",
            schema_version="phase5_local_vqa_v1",
            services=["product_store", "vlm"],
        )
        try:
            evidence = {"source": "local_upload", "verified_text": [], "truth_status": "image_only_unverified_text"}
            existing = product.get_session(request.conversation_id) if request.conversation_id else None
            if existing is not None and existing.get("asset_id") != asset_id:
                raise HTTPException(status_code=409, detail="conversation_asset_mismatch")
            if existing:
                evidence["conversation_history"] = list(existing.get("messages", []))[-12:]
            result = vlm_service.answer_question(Path(item["path"]), request.question, evidence)
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            conversation_id = request.conversation_id or str(uuid.uuid4())
            session = existing or {
                "conversation_id": conversation_id,
                "asset_id": asset_id,
                "messages": [],
                "created_at": now_iso(),
            }
            session["messages"].extend([
                {"role": "user", "content": request.question, "created_at": now_iso()},
                {"role": "assistant", "content": result.data.get("parsed_output", {}), "trace_id": trace["request_id"], "created_at": now_iso()},
            ])
            session["updated_at"] = now_iso()
            product.save_session(session)
            finished = traces.finish(trace, status="success")
            return {
                "asset_id": asset_id,
                "question": request.question,
                "evidence": evidence,
                "result": result.as_dict(),
                "conversation_id": conversation_id,
                "messages": session["messages"],
                "request_id": finished["request_id"],
                "trace": finished,
            }
        except Exception as exc:
            traces.finish(trace, status="failed", error=exc)
            return run_or_http_error(lambda: (_ for _ in ()).throw(exc))

    @app.get("/conversations")
    def conversations() -> dict[str, Any]:
        items = product.sessions()
        return {"count": len(items), "items": items}

    @app.get("/conversations/{conversation_id}")
    def conversation(conversation_id: str) -> dict[str, Any]:
        item = product.get_session(conversation_id)
        if item is None:
            raise HTTPException(status_code=404, detail="conversation_not_found")
        return item

    @app.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict[str, Any]:
        try:
            return product.delete_conversation(conversation_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail="conversation_not_found",
            ) from None

    def retrieval_library_catalog() -> dict[str, dict[str, Any]]:
        catalog = {
            str(item["library_id"]): {
                "library_id": str(item["library_id"]),
                "library_name": str(
                    item.get("display_name")
                    or item.get("name")
                    or item["library_id"]
                ),
                "source_type": "system_locked",
            }
            for item in system_libraries.libraries()
        }
        catalog.update(
            {
                str(item["library_id"]): {
                    "library_id": str(item["library_id"]),
                    "library_name": str(
                        item.get("display_name")
                        or item.get("name")
                        or item["library_id"]
                    ),
                    "source_type": "user_custom",
                }
                for item in product.libraries()
                if not item.get("deleted")
            }
        )
        # The frozen 16-image product library remains a persistent,
        # searchable compatibility library, but is not confused with a user
        # custom library that happens to use the historical "default" id.
        catalog["legacy_frozen"] = {
            "library_id": "legacy_frozen",
            "library_name": "课程历史示例图片库",
            "source_type": "registered_persistent",
        }
        return catalog

    def resolve_retrieval_library_ids(
        *,
        library_scope: str,
        current_library_id: str | None,
        explicit_library_ids: list[str] | None,
    ) -> tuple[str, list[str]]:
        catalog = retrieval_library_catalog()
        explicit = list(
            dict.fromkeys(
                str(item)
                for item in (explicit_library_ids or [])
                if str(item)
            )
        )
        if explicit:
            unknown = [item for item in explicit if item not in catalog]
            if unknown:
                raise ValueError(
                    "unknown_retrieval_library:"
                    + ",".join(unknown)
                )
            return "explicit_libraries", explicit
        if library_scope == "system_train":
            return library_scope, ["system_train"]
        if library_scope == "system_val":
            return library_scope, ["system_val"]
        if library_scope == "current_library":
            selected = str(current_library_id or "").strip()
            if not selected or selected not in catalog:
                raise ValueError("current_retrieval_library_required")
            return library_scope, [selected]
        if library_scope != "all_libraries":
            raise ValueError("invalid_retrieval_library_scope")
        if provider_manager.mode == "bailian":
            cloud_service = provider_manager.retrieval_service()
            cloud_records = list(
                getattr(cloud_service, "records", [])
            )
            return library_scope, list(
                dict.fromkeys(
                    str(item.get("library_id"))
                    for item in cloud_records
                    if item.get("library_id") in catalog
                )
            )
        current_split = active_split_retrieval()
        return library_scope, [
            library_id
            for library_id in catalog
            if (
                library_id not in SYSTEM_LIBRARY_IDS
                or (
                    current_split is not None
                    and library_id in current_split.indices
                )
            )
        ]

    def resolve_chat_retrieval_scope(
        message: str,
        router_arguments: dict[str, Any],
    ) -> tuple[str, list[str]]:
        normalized = message.casefold()
        custom_matches = []
        for item in product.libraries():
            name = str(
                item.get("display_name")
                or item.get("name")
                or ""
            ).strip()
            if name and name.casefold() in normalized:
                custom_matches.append(
                    (len(name), str(item["library_id"]))
                )
        if custom_matches:
            custom_matches.sort(reverse=True)
            return "explicit_libraries", [custom_matches[0][1]]
        scope = str(
            router_arguments.get("library_scope")
            or "all_libraries"
        )
        if scope not in {
            "system_train",
            "system_val",
            "all_libraries",
        }:
            scope = "all_libraries"
        return scope, []

    @staticmethod
    def _effective_product_library_id(item: dict[str, Any]) -> str:
        return (
            "legacy_frozen"
            if item.get("source") == "frozen_library"
            else str(item.get("library_id") or "default")
        )

    def _decorate_retrieval_sources(
        values: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        catalog = retrieval_library_catalog()
        decorated = []
        for value in values:
            item = dict(value)
            library_id = (
                str(item.get("library_id"))
                if item.get("library_id") in SYSTEM_LIBRARY_IDS
                else _effective_product_library_id(item)
            )
            library = catalog.get(
                library_id,
                {
                    "library_name": library_id,
                    "source_type": "registered_persistent",
                },
            )
            item.update(
                library_id=library_id,
                library_name=library["library_name"],
                source_type=library["source_type"],
                source_library=library_id,
            )
            decorated.append(asset_media.resolve_result(item))
        return decorated

    def _deduplicate_scoped_results(
        values: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            _decorate_retrieval_sources(values),
            key=lambda item: (
                -float(item.get("score") or 0.0),
                str(item.get("library_id") or ""),
                str(item.get("asset_id") or item.get("image_id") or ""),
            ),
        )
        by_identity: dict[str, dict[str, Any]] = {}
        for item in ordered:
            eligible, _ = candidate_eligibility(item)
            if not eligible:
                continue
            identity = str(
                item.get("sha256")
                or (
                    f"{item.get('library_id')}:"
                    f"{item.get('asset_id') or item.get('image_id')}"
                )
            )
            if identity in by_identity:
                duplicate = by_identity[identity]
                sources = list(duplicate.get("duplicate_sources") or [])
                sources.append(
                    {
                        "library_id": item.get("library_id"),
                        "library_name": item.get("library_name"),
                        "asset_id": (
                            item.get("asset_id")
                            or item.get("image_id")
                        ),
                    }
                )
                duplicate["duplicate_sources"] = sources
                duplicate["duplicate_sha_collapsed"] = True
                continue
            by_identity[identity] = item
        return list(by_identity.values())[:top_k]

    def search_persistent_libraries(
        *,
        requested_library_ids: list[str],
        query_text: str | None,
        image_path: Path | None,
        requested_k: int,
        exclude_asset_ids: set[str],
    ) -> list[dict[str, Any]]:
        if not hasattr(retrieval, "encode_query"):
            candidate_k = requested_k + len(exclude_asset_ids)
            if image_path is not None and query_text:
                raw_rows = retrieval.search_hybrid(
                    image_path,
                    query_text,
                    candidate_k,
                    exclude_image_id=None,
                )
            elif image_path is not None:
                raw_rows = retrieval.search_image(
                    image_path,
                    candidate_k,
                    exclude_image_id=None,
                )
            elif query_text:
                raw_rows = retrieval.search(query_text, candidate_k)
            else:
                raise ValueError("course_retrieval_requires_text_or_image")
            candidates = []
            for raw in raw_rows:
                item = dict(raw)
                effective = _effective_product_library_id(item)
                if effective == "legacy_frozen":
                    lifecycle = (
                        legacy_lifecycle_registry.lookup(item)
                        if legacy_lifecycle_registry is not None
                        else {
                            "lifecycle_state": "active",
                            "searchable": True,
                        }
                    )
                    item.update(
                        lifecycle_state=lifecycle["lifecycle_state"],
                        searchable=lifecycle["searchable"],
                    )
                asset_id = str(
                    item.get("asset_id")
                    or item.get("image_id")
                    or ""
                )
                if (
                    effective in requested_library_ids
                    and asset_id not in exclude_asset_ids
                ):
                    candidates.append(item)
            return _deduplicate_scoped_results(
                candidates,
                top_k=requested_k,
            )
        vector, route = retrieval.encode_query(
            query_text=query_text,
            image_path=image_path,
        )
        if provider_manager.mode == "bailian":
            candidates = retrieval.search_vector_scoped(
                vector,
                top_k=requested_k,
                requested_library_ids=set(requested_library_ids),
                exclude_asset_ids=exclude_asset_ids,
                route=route,
            )
            return _deduplicate_scoped_results(
                candidates,
                top_k=requested_k,
            )
        system_ids = [
            item
            for item in requested_library_ids
            if item in SYSTEM_LIBRARY_IDS
        ]
        product_ids = [
            item
            for item in requested_library_ids
            if item not in SYSTEM_LIBRARY_IDS
        ]
        current_split = active_split_retrieval()
        if system_ids and current_split is None:
            raise RuntimeError("system_e1_retrieval_disabled")

        product_total = len(
            getattr(
                getattr(retrieval, "e1_index", None),
                "records",
                [],
            )
        )
        system_totals = [
            len(current_split.indices[library_id].records)
            for library_id in system_ids
        ] if current_split is not None else []

        def fetch(fetch_n: int) -> list[dict[str, Any]]:
            candidates: list[dict[str, Any]] = []
            if system_ids and current_split is not None:
                candidates.extend(
                    current_split.search_vector(
                        vector,
                        library_ids=system_ids,
                        top_k=min(fetch_n, 100),
                        exclude_asset_ids=exclude_asset_ids,
                        route=route,
                    )
                )
            if product_ids:
                candidates.extend(
                    retrieval.search_vector_scoped(
                        vector,
                        top_k=min(fetch_n, 100),
                        requested_library_ids=set(product_ids),
                        exclude_asset_ids=exclude_asset_ids,
                        route=route,
                    )
                )
            return candidates

        candidates, debug = adaptive_topk_refill(
            fetch,
            requested_k=requested_k,
            total_candidates=max([product_total, *system_totals, 0]),
            requested_library_ids=set(requested_library_ids),
            exclude_asset_ids=exclude_asset_ids,
            current_library_id=(
                requested_library_ids[0]
                if len(requested_library_ids) == 1
                else None
            ),
        )
        decorated = _deduplicate_scoped_results(
            candidates,
            top_k=requested_k,
        )
        for item in decorated:
            item["candidate_refill"] = debug
        return decorated

    def course_retrieve_payload(
        query_text: str | None,
        query_asset_ref: CourseAssetRefRequest | None,
        *,
        query_asset_refs: list[CourseAssetRefRequest] | None = None,
        top_k: int = 5,
        exclude_query_images: bool = True,
        library_scope: str = "all_libraries",
        current_library_id: str | None = None,
        library_ids: list[str] | None = None,
        call_source: str = "standalone_workspace",
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        public_top_k = int(top_k)
        if public_top_k < 1 or public_top_k > 5:
            raise ValueError("public_top_k_must_be_between_1_and_5")
        query = (query_text or "").strip()
        requested_refs = list(query_asset_refs or [])
        if query_asset_ref is not None:
            requested_refs.insert(0, query_asset_ref)
        unique_refs: list[CourseAssetRefRequest] = []
        seen_query_refs: set[str] = set()
        for ref in requested_refs:
            stable_ref = f"{ref.source}:{ref.asset_id}"
            if stable_ref not in seen_query_refs:
                seen_query_refs.add(stable_ref)
                unique_refs.append(ref)
        if not query and not unique_refs:
            raise ValueError("course_retrieval_requires_text_or_image")
        query_assets = resolve_course_assets(unique_refs) if unique_refs else []
        query_asset = query_assets[0] if query_assets else None
        if query and query_asset:
            mode = "hybrid"
            schema_version = "phase5_2_course_hybrid_retrieval_v1"
        elif query_asset:
            mode = "image"
            schema_version = "phase5_2_course_image_retrieval_v1"
        else:
            mode = "text"
            schema_version = "phase5_2_course_text_retrieval_v1"
        trace = traces.start(
            "course_retrieve",
            [str(item["asset_id"]) for item in query_assets],
            model=retrieval.embedding.status().get("model"),
            model_revision=retrieval.embedding.status().get("model_revision"),
            prompt_version="not_applicable",
            schema_version=schema_version,
            services=["library", "embedding", "retrieval"],
        )
        try:
            candidate_refills: list[dict[str, Any]] = []
            internal_candidate_k: int | None = None
            resolved_library_scope, scoped_library_ids = (
                resolve_retrieval_library_ids(
                    library_scope=library_scope,
                    current_library_id=current_library_id,
                    explicit_library_ids=library_ids,
                )
            )
            if scoped_library_ids:
                excluded = (
                    {str(item["asset_id"]) for item in query_assets}
                    if exclude_query_images
                    else set()
                )
                if len(query_assets) <= 1:
                    results = search_persistent_libraries(
                        requested_library_ids=scoped_library_ids,
                        query_text=query or None,
                        image_path=(
                            query_assets[0]["path"] if query_assets else None
                        ),
                        requested_k=public_top_k,
                        exclude_asset_ids=excluded,
                    )
                    if results and isinstance(
                        results[0].get("candidate_refill"),
                        dict,
                    ):
                        candidate_refills.append(
                            dict(results[0]["candidate_refill"])
                        )
                else:
                    fused: dict[str, dict[str, Any]] = {}
                    for query_index, item in enumerate(query_assets, start=1):
                        candidates = search_persistent_libraries(
                            requested_library_ids=scoped_library_ids,
                            query_text=query or None,
                            image_path=item["path"],
                            requested_k=public_top_k,
                            exclude_asset_ids=excluded,
                        )
                        if candidates and isinstance(
                            candidates[0].get("candidate_refill"),
                            dict,
                        ):
                            candidate_refills.append(
                                dict(candidates[0]["candidate_refill"])
                            )
                        for candidate_rank, candidate in enumerate(
                            candidates,
                            start=1,
                        ):
                            asset_id = str(
                                candidate.get("asset_id")
                                or candidate.get("image_id")
                                or ""
                            )
                            if not asset_id:
                                continue
                            row = fused.setdefault(
                                asset_id,
                                {
                                    **candidate,
                                    "_rrf_score": 0.0,
                                    "_query_hits": [],
                                },
                            )
                            row["_rrf_score"] += 1.0 / (60 + candidate_rank)
                            row["_query_hits"].append(query_index)
                    results = sorted(
                        fused.values(),
                        key=lambda item: (
                            -float(item["_rrf_score"]),
                            str(
                                item.get("asset_id")
                                or item.get("image_id")
                                or ""
                            ),
                        ),
                    )[:public_top_k]
                    for item in results:
                        item["multi_query_rrf_score"] = round(
                            float(item.pop("_rrf_score")),
                            8,
                        )
                        item["query_image_positions"] = item.pop("_query_hits")
                        item["ordering_basis"] = (
                            "multi_query_rrf_desc_then_asset_id"
                        )
            elif mode == "text":
                results = retrieval.search(query, public_top_k)
            else:
                per_query: list[list[dict[str, Any]]] = []
                internal_candidate_k = min(
                    100,
                    public_top_k + len(query_assets),
                )
                for item in query_assets:
                    if mode == "image":
                        per_query.append(
                            retrieval.search_image(
                                item["path"],
                                internal_candidate_k,
                                exclude_image_id=None,
                            )
                        )
                    else:
                        per_query.append(
                            retrieval.search_hybrid(
                                item["path"],
                                query,
                                internal_candidate_k,
                                exclude_image_id=None,
                            )
                        )
                excluded = (
                    {str(item["asset_id"]) for item in query_assets}
                    if exclude_query_images
                    else set()
                )
                if len(per_query) == 1:
                    # A single-image search has one authoritative similarity
                    # score. Preserve it directly instead of rewriting order
                    # through multi-query RRF.
                    candidates = []
                    for candidate in per_query[0]:
                        image_id = str(
                            candidate.get("image_id")
                            or candidate.get("asset_id")
                            or ""
                        )
                        if image_id and image_id not in excluded:
                            candidates.append(dict(candidate))
                    results = sorted(
                        candidates,
                        key=lambda item: (
                            -float(item.get("score") or 0.0),
                            str(
                                item.get("image_id")
                                or item.get("asset_id")
                                or ""
                            ),
                        ),
                    )[:public_top_k]
                    for item in results:
                        item["query_image_positions"] = [1]
                        item["ordering_basis"] = (
                            "similarity_score_desc_then_asset_id"
                        )
                else:
                    fused: dict[str, dict[str, Any]] = {}
                    for query_index, candidates in enumerate(
                        per_query,
                        start=1,
                    ):
                        for rank, candidate in enumerate(
                            candidates,
                            start=1,
                        ):
                            image_id = str(
                                candidate.get("image_id")
                                or candidate.get("asset_id")
                                or ""
                            )
                            if not image_id or image_id in excluded:
                                continue
                            row = fused.setdefault(
                                image_id,
                                {
                                    **candidate,
                                    "_rrf_score": 0.0,
                                    "_query_hits": [],
                                },
                            )
                            row["_rrf_score"] += 1.0 / (60 + rank)
                            row["_query_hits"].append(query_index)
                    results = sorted(
                        fused.values(),
                        key=lambda item: (
                            -float(item["_rrf_score"]),
                            str(item.get("image_id") or ""),
                        ),
                    )[:public_top_k]
                    for item in results:
                        item["multi_query_rrf_score"] = round(
                            float(item.pop("_rrf_score")),
                            8,
                        )
                        item["query_image_positions"] = item.pop(
                            "_query_hits"
                        )
                        item["ordering_basis"] = (
                            "multi_query_rrf_desc_then_asset_id"
                        )
            # Per-query ranks are no longer meaningful after multi-image RRF
            # fusion. Publish one stable, gap-free rank for the final list
            # while retaining the backend's original position for audit.
            for final_rank, item in enumerate(results, start=1):
                source_rank = item.get("rank")
                if source_rank is not None and source_rank != final_rank:
                    item["source_rank"] = source_rank
                item["rank"] = final_rank
            first = results[0] if results else {}
            backend = str(first.get("retrieval_backend") or retrieval.status().get("active_backend") or "r0")
            fallback_used = bool(first.get("fallback_used", False))
            baseline = (
                "Qwen3-VL-Embedding-2B + Faiss"
                if backend in {"e1", "bailian_cloud_e1"}
                else "Fallback: color-grid-v1"
                if fallback_used
                else "color-grid-v1"
            )
            for item in results:
                item["reason"] = str(item.get("match_basis") or item.get("route") or "基线相似度命中")[:240]
                item["baseline_label"] = baseline
                item.setdefault("retrieval_backend", backend)
                item.setdefault("fallback_used", fallback_used)
                item.setdefault(
                    "index_version",
                    first.get("index_version")
                    or retrieval.status().get("index_version"),
                )
            retrieval_contract = {
                "public_top_k": public_top_k,
                "returned_count": len(results),
                "query_input_count": max(1, len(query_assets)),
                "query_embedding_count": max(1, len(query_assets)),
                "refill_reencoded_query": False,
                "internal_fetch_histories": [
                    list(item.get("fetch_history") or [])
                    for item in candidate_refills
                ],
                "adaptive_refill_rounds": [
                    int(item.get("rounds") or 0)
                    for item in candidate_refills
                ],
                "internal_candidate_k": internal_candidate_k,
                "exclude_query_images": bool(exclude_query_images),
                "excluded_asset_ids": (
                    sorted(str(item["asset_id"]) for item in query_assets)
                    if exclude_query_images
                    else []
                ),
                "library_scope": resolved_library_scope,
                "library_ids": list(scoped_library_ids),
                "provider_mode": provider_manager.mode,
                "retrieval_backend": backend,
                "embedding_dimension": (
                    first.get("embedding_dimension")
                    or retrieval.status().get("embedding_dimension")
                    or retrieval.status().get("dimensions")
                ),
                "routes": sorted(
                    {
                        str(item.get("route"))
                        for item in results
                        if item.get("route")
                    }
                ),
                "status": "success",
                "error_origin": None,
            }
            trace["retrieval"] = {
                "retrieval_backend": backend,
                "model": first.get("model"),
                "revision": first.get("revision"),
                "index_version": first.get("index_version"),
                "fallback_used": fallback_used,
                "fallback_reason": first.get("fallback_reason"),
                "ordering_basis": first.get("ordering_basis")
                or "similarity_score_desc_then_asset_id",
                "visible_scores_non_increasing": all(
                    float(left.get("score") or 0.0)
                    >= float(right.get("score") or 0.0)
                    for left, right in zip(results, results[1:])
                )
                if len(query_assets) <= 1
                else None,
                "public_contract": retrieval_contract,
            }
            finished = traces.finish(trace, status="success", output_path=str(retrieval.index_path))
            return {
                "mode": mode,
                "query_text": query or None,
                "query_asset": public_course_assets(query_assets)[0] if query_assets else None,
                "query_assets": public_course_assets(query_assets),
                "query_image_count": len(query_assets),
                "library_scope": resolved_library_scope,
                "library_ids": scoped_library_ids,
                "top_k": public_top_k,
                "results": results,
                "display_text": f"已返回 {len(results)} 个检索结果（{mode}）。",
                "baseline_label": baseline,
                "retrieval_backend": backend,
                "model": first.get("model"),
                "revision": first.get("revision"),
                "index_version": first.get("index_version"),
                "fallback_used": fallback_used,
                "fallback_reason": first.get("fallback_reason"),
                "retrieval_contract": retrieval_contract,
                "reranker": "NOT IMPLEMENTED",
                "call_source": call_source,
                "workspace_id": workspace_id,
                "request_id": finished["request_id"],
                "trace": finished,
            }
        except Exception as exc:
            failed_trace = traces.finish(trace, status="failed", error=exc)
            if call_source == "chat_tool_call":
                return {
                    "status": "tool_failed",
                    "mode": mode,
                    "query_text": query or None,
                    "query_asset": (
                        public_course_assets(query_assets)[0]
                        if query_assets
                        else None
                    ),
                    "query_assets": public_course_assets(query_assets),
                    "query_image_count": len(query_assets),
                    "top_k": public_top_k,
                    "results": [],
                    "display_text": (
                        "这次图片检索没有成功，我没有返回虚假的结果。"
                        "你可以稍后重试，或改用另一种检索方式。"
                    ),
                    "tool_error": type(exc).__name__,
                    "call_source": call_source,
                    "workspace_id": workspace_id,
                    "request_id": failed_trace["request_id"],
                    "trace": failed_trace,
                }
            raise

    def course_generate_payload(
        assets: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        content_type_key = str(
            options.get("resolved_content_type")
            or options.get("content_type")
            or "objective_description"
        )
        options = dict(options)
        target_source = str(
            options.get("target_length_source")
            or (
                "profile_default"
                if options.get("target_length") is None
                else "user_or_caller"
            )
        )
        target_adjustment = normalize_target_length(
            options.get("target_length"),
            content_type_key,
            explicit=target_source not in {"profile_default", "default_value"},
        )
        options["target_length"] = int(target_adjustment["target"])
        options["target_length_source"] = target_source
        options["target_length_adjustment"] = target_adjustment
        profile = content_profile_registry.get(content_type_key)
        ideal_minimum, ideal_maximum = ideal_output_window(
            options["target_length"]
        )
        profile["default_length"] = {
            "target": options["target_length"],
            "minimum": ideal_minimum,
            "maximum": ideal_maximum,
        }
        profile["input_length_bounds"] = {
            "minimum": target_adjustment["input_min"],
            "maximum": target_adjustment["input_max"],
            "default": target_adjustment["default"],
        }
        profile.pop("hard_maximum", None)
        options["content_profile"] = profile
        active_content_candidate = (
            multi_image_story_candidate
            if content_type_key == "creative_story"
            else multi_image_content_candidate
        )
        prompt, identity, resolved_options = active_content_candidate.render(
            assets,
            options,
        )
        prompt_id = identity["prompt_id"]
        trace = traces.start(
            "course_generate",
            [item["asset_id"] for item in assets],
            model=vlm_service.status().get("model"),
            model_revision=vlm_service.status().get("model_revision"),
            prompt_version=prompt_id,
            prompt_sha256=identity["prompt_sha256"],
            schema_version=(
                "phase5_2d_multi_image_story_v1"
                if content_type_key == "creative_story"
                else "phase5_2b_multi_image_content_v2"
            ),
            services=[
                "product_store",
                (
                    "multi_image_story_v1"
                    if content_type_key == "creative_story"
                    else "multi_image_content_v2"
                ),
                "vlm",
            ],
        )
        bounded_revision_attempted = False
        bounded_completion_attempts = 0
        try:
            target_length = int(options["target_length"])
            full_token_budget = model_token_budget(
                target_length,
                content_type_key,
            )
            service_result = vlm_service.run_course_prompt(
                [item["path"] for item in assets],
                prompt,
                prompt_id=prompt_id,
                prompt_sha256=identity["prompt_sha256"],
                max_new_tokens=full_token_budget,
                image_labels=[f"图片{index}" for index in range(1, len(assets) + 1)],
                min_new_tokens=model_min_token_budget(target_length),
            )
            if service_result.status != "success":
                raise RuntimeError(service_result.error or service_result.status)
            result = service_result.as_dict()
            result["data"]["candidate_id"] = identity["candidate_id"]
            candidate_history: list[dict[str, Any]] = []
            best_payload: dict[str, Any] | None = None
            best_normalized: dict[str, Any] | None = None
            best_contract: dict[str, Any] | None = None
            best_score: tuple[int, ...] | None = None
            best_phase: str | None = None

            def consider_candidate(
                record: dict[str, Any],
                *,
                phase: str,
                max_new_tokens: int | None,
                payload_override: dict[str, Any] | None = None,
                dynamic_prompt_sha256: str | None = None,
            ) -> None:
                nonlocal best_payload, best_normalized, best_contract
                nonlocal best_score, best_phase
                data = record.get("data") if isinstance(record.get("data"), dict) else {}
                raw_output = str(data.get("raw_output") or "")
                payload = payload_override or (
                    extract_story_public_payload(
                        raw_output,
                        image_count=len(assets),
                    )
                    if content_type_key == "creative_story"
                    else extract_multi_image_payload(raw_output)
                )
                if payload is None and isinstance(
                    data.get("parsed_output"), dict
                ):
                    payload = dict(data["parsed_output"])
                normalized_candidate, contract_candidate = validate_multi_image_content(
                    payload,
                    raw_output if payload_override is None else json.dumps(
                        payload_override,
                        ensure_ascii=False,
                    ),
                    assets=assets,
                    target_length=target_length,
                    validator=active_content_candidate.validator,
                    content_type=content_type_key,
                    target_length_source=str(
                        resolved_options.get("target_length_source")
                        or "user_or_caller"
                    ),
                    allow_image_meta_language=bool(
                        resolved_options.get("allow_image_meta_language")
                    ),
                )
                if payload is not None and contract_candidate.get("candidate_final_text"):
                    payload = dict(payload)
                    payload["final_text"] = contract_candidate["candidate_final_text"]
                score = candidate_quality_score(
                    payload,
                    contract=contract_candidate,
                )
                promoted = best_score is None or score > best_score
                rejection_reason = None
                if promoted:
                    best_payload = payload
                    best_normalized = normalized_candidate
                    best_contract = contract_candidate
                    best_score = score
                    best_phase = phase
                else:
                    rejection_reason = "candidate_not_better_than_best"
                output_tokens = data.get("output_tokens")
                max_hit = (
                    isinstance(output_tokens, int)
                    and isinstance(max_new_tokens, int)
                    and output_tokens >= max_new_tokens
                )
                candidate_history.append(
                    {
                        "phase": phase,
                        "status": record.get("status"),
                        "parseable": payload is not None,
                        "has_final_text": bool(
                            payload and str(payload.get("final_text") or "").strip()
                        ),
                        "score": list(score),
                        "promoted": promoted,
                        "rejection_reason": rejection_reason,
                        "contract_errors": contract_candidate["contract_errors"],
                        "input_tokens": data.get("input_tokens"),
                        "output_tokens": output_tokens,
                        "finish_reason": data.get("finish_reason"),
                        "max_new_tokens": max_new_tokens,
                        "max_new_tokens_hit": bool(
                            data.get("max_new_tokens_hit", max_hit)
                        ),
                        "latency_seconds": record.get("latency_seconds"),
                        "dynamic_prompt_sha256": dynamic_prompt_sha256,
                        "raw_output": str(data.get("raw_output") or ""),
                    }
                )

            consider_candidate(
                result,
                phase="initial_full_contract",
                max_new_tokens=full_token_budget,
                dynamic_prompt_sha256=hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
            )
            assert best_normalized is not None and best_contract is not None

            if best_payload is not None and best_payload.get("final_text"):
                missing_metadata = []
                if not best_contract["image_coverage"]["passed"]:
                    missing_metadata.append("used_images")
                if missing_metadata:
                    metadata_prompt = render_metadata_completion_prompt(
                        best_payload,
                        image_count=len(assets),
                    )
                    metadata_sha = hashlib.sha256(
                        metadata_prompt.encode("utf-8")
                    ).hexdigest()
                    metadata_budget = model_token_budget(
                        target_length,
                        content_type_key,
                        stage="metadata_completion",
                    )
                    metadata_result = vlm_service.run_course_prompt(
                        [item["path"] for item in assets],
                        metadata_prompt,
                        prompt_id=f"{prompt_id}_metadata_completion",
                        prompt_sha256=metadata_sha,
                        max_new_tokens=metadata_budget,
                        image_labels=[
                            f"图片{index}"
                            for index in range(1, len(assets) + 1)
                        ],
                        min_new_tokens=None,
                    )
                    metadata_record = metadata_result.as_dict()
                    merged_metadata = (
                        merge_metadata_completion(
                            best_payload,
                            str(metadata_record.get("data", {}).get("raw_output") or ""),
                        )
                        if metadata_result.status == "success"
                        else None
                    )
                    consider_candidate(
                        metadata_record,
                        phase="metadata_completion",
                        max_new_tokens=metadata_budget,
                        payload_override=merged_metadata,
                        dynamic_prompt_sha256=metadata_sha,
                    )

            assert best_contract is not None
            contract_errors = [
                str(item) for item in best_contract["contract_errors"]
            ]
            story_public_text_only_invalid = (
                content_type_key == "creative_story"
                and best_payload is not None
                and bool(best_payload.get("final_text"))
                and best_contract["image_coverage"]["passed"]
                and best_contract["story_structure"]["passed"]
                and not any(
                    str(error).startswith("internal_output_leak")
                    or str(error).startswith("story_output_gate_violation")
                    or str(error).startswith("authored_content_image_meta_text_violation")
                    for error in contract_errors
                )
                and not best_contract["length_contract"]["passed"]
            )
            only_length_invalid = (
                best_payload is not None
                and bool(best_payload.get("final_text"))
                and best_contract["image_coverage"]["passed"]
                and best_contract["evidence_coverage"]["passed"]
                and best_contract["story_structure"]["passed"]
                and bool(contract_errors)
                and all(
                    error.startswith("visible_length_contract_violation")
                    for error in contract_errors
                )
            ) or story_public_text_only_invalid
            if (
                not best_contract["product_contract_valid"]
                and
                only_length_invalid
                and best_contract["length_contract"]["actual"]
                > best_contract["length_contract"]["maximum"]
            ):
                adjusted = adjust_final_text_only(
                    best_payload,
                    assets=assets,
                    target_length=target_length,
                    creative_story=content_type_key == "creative_story",
                    content_type=content_type_key,
                    target_length_source=str(
                        resolved_options.get("target_length_source")
                        or "user_or_caller"
                    ),
                )
                if adjusted is not None:
                    consider_candidate(
                        {
                            "status": "success",
                            "data": {
                                "raw_output": json.dumps(
                                    adjusted,
                                    ensure_ascii=False,
                                ),
                                "output_tokens": 0,
                                "finish_reason": "deterministic_sentence_boundary",
                                "max_new_tokens_hit": False,
                            },
                            "latency_seconds": 0.0,
                        },
                        phase="deterministic_final_text_compression",
                        max_new_tokens=0,
                        payload_override=adjusted,
                        dynamic_prompt_sha256=None,
                    )

            assert best_contract is not None
            if (
                not best_contract["product_contract_valid"]
                and only_length_invalid
                and best_payload is not None
            ):
                text_prompt = render_final_text_revision_prompt(
                    best_payload,
                    target_length=target_length,
                    content_type=content_type_key,
                    target_length_source=str(
                        resolved_options.get("target_length_source")
                        or "user_or_caller"
                    ),
                )
                text_sha = hashlib.sha256(
                    text_prompt.encode("utf-8")
                ).hexdigest()
                text_budget = model_token_budget(
                    target_length,
                    content_type_key,
                    stage="final_text_revision",
                )
                text_result = vlm_service.run_course_prompt(
                    [item["path"] for item in assets],
                    text_prompt,
                    prompt_id=f"{prompt_id}_final_text_revision",
                    prompt_sha256=text_sha,
                    max_new_tokens=text_budget,
                    image_labels=[
                        f"图片{index}" for index in range(1, len(assets) + 1)
                    ],
                    min_new_tokens=None,
                )
                text_record = text_result.as_dict()
                merged_text = (
                    merge_final_text_revision(
                        best_payload,
                        str(text_record.get("data", {}).get("raw_output") or ""),
                    )
                    if text_result.status == "success"
                    else None
                )
                consider_candidate(
                    text_record,
                    phase="final_text_only_revision",
                    max_new_tokens=text_budget,
                    payload_override=merged_text,
                    dynamic_prompt_sha256=text_sha,
                )

            assert best_contract is not None
            if (
                not best_contract["product_contract_valid"]
            ):
                bounded_revision_attempted = True
                revision_prompt, _, _ = active_content_candidate.render_bounded_revision(
                    assets,
                    options,
                    str(result["data"].get("raw_output") or ""),
                )
                revision_sha = hashlib.sha256(
                    revision_prompt.encode("utf-8")
                ).hexdigest()
                revision_result = vlm_service.run_course_prompt(
                    [item["path"] for item in assets],
                    revision_prompt,
                    prompt_id=prompt_id,
                    prompt_sha256=identity["prompt_sha256"],
                    max_new_tokens=full_token_budget,
                    image_labels=[
                        f"图片{index}" for index in range(1, len(assets) + 1)
                    ],
                    min_new_tokens=None,
                )
                revision_record = revision_result.as_dict()
                consider_candidate(
                    revision_record,
                    phase="bounded_full_contract_revision",
                    max_new_tokens=full_token_budget,
                    dynamic_prompt_sha256=revision_sha,
                )

            assert best_normalized is not None and best_contract is not None
            post_revision_errors = [
                str(item) for item in best_contract["contract_errors"]
            ]
            post_revision_story_public_text_only_invalid = (
                content_type_key == "creative_story"
                and best_payload is not None
                and bool(best_payload.get("final_text"))
                and best_contract["image_coverage"]["passed"]
                and best_contract["story_structure"]["passed"]
                and not any(
                    str(error).startswith("internal_output_leak")
                    or str(error).startswith("story_output_gate_violation")
                    or str(error).startswith("authored_content_image_meta_text_violation")
                    for error in post_revision_errors
                )
                and not best_contract["length_contract"]["passed"]
            )
            post_revision_only_length_invalid = (
                best_payload is not None
                and bool(best_payload.get("final_text"))
                and best_contract["image_coverage"]["passed"]
                and best_contract["evidence_coverage"]["passed"]
                and best_contract["story_structure"]["passed"]
                and bool(post_revision_errors)
                and all(
                    error.startswith("visible_length_contract_violation")
                    for error in post_revision_errors
                )
            ) or post_revision_story_public_text_only_invalid
            if (
                not best_contract["product_contract_valid"]
                and
                post_revision_only_length_invalid
                and best_contract["length_contract"]["actual"]
                > best_contract["length_contract"]["maximum"]
                and best_payload is not None
            ):
                adjusted = adjust_final_text_only(
                    best_payload,
                    assets=assets,
                    target_length=target_length,
                    creative_story=content_type_key == "creative_story",
                    content_type=content_type_key,
                    target_length_source=str(
                        resolved_options.get("target_length_source")
                        or "user_or_caller"
                    ),
                )
                if adjusted is not None:
                    consider_candidate(
                        {
                            "status": "success",
                            "data": {
                                "raw_output": json.dumps(
                                    adjusted,
                                    ensure_ascii=False,
                                ),
                                "output_tokens": 0,
                                "finish_reason": (
                                    "deterministic_punctuation_or_sentence_boundary"
                                ),
                                "max_new_tokens_hit": False,
                            },
                            "latency_seconds": 0.0,
                        },
                        phase="post_revision_deterministic_compression",
                        max_new_tokens=0,
                        payload_override=adjusted,
                        dynamic_prompt_sha256=None,
                    )

            assert best_normalized is not None and best_contract is not None
            normalized = best_normalized
            contract = best_contract
            if best_phase in {
                "deterministic_final_text_compression",
                "post_revision_deterministic_compression",
            }:
                contract["model_contract_valid"] = False
                contract["repair_applied"] = True
                contract["fallback_applied"] = True
                contract["fallback_source"] = "target_length_safe_compression"
                contract["fallback_reason"] = None
            current_payload = best_payload
            if current_payload is not None:
                text_risk_payload = (
                    None
                    if content_type_key == "creative_story"
                    else build_text_risk_generalization(
                        current_payload,
                        assets=assets,
                        target_length=int(options["target_length"]),
                    )
                )
                short_bridge_payload = (
                    append_safe_short_bridge(
                        current_payload,
                        target_length=int(options["target_length"]),
                        content_type=content_type_key,
                        target_length_source=str(
                            resolved_options.get("target_length_source")
                            or "user_or_caller"
                        ),
                    )
                    if text_risk_payload is None
                    and content_type_key != "creative_story"
                    else None
                )
                fallback_payload = (
                    text_risk_payload or short_bridge_payload
                )
                if fallback_payload is not None:
                    normalized, contract = validate_multi_image_content(
                        fallback_payload,
                        json.dumps(fallback_payload, ensure_ascii=False),
                        assets=assets,
                        target_length=target_length,
                        validator=active_content_candidate.validator,
                        content_type=content_type_key,
                        target_length_source=str(
                            resolved_options.get("target_length_source")
                            or "user_or_caller"
                        ),
                        allow_image_meta_language=bool(
                            resolved_options.get("allow_image_meta_language")
                        ),
                    )
                    contract["model_contract_valid"] = False
                    contract["repair_applied"] = True
                    contract["fallback_applied"] = True
                    contract["risk_sanitized"] = bool(text_risk_payload)
                    contract["fallback_source"] = (
                        "unverified_text_visual_generalization"
                        if text_risk_payload is not None
                        else "safe_short_multiview_bridge"
                    )
                    current_payload = fallback_payload
                    best_payload = fallback_payload
                    best_normalized = normalized
                    best_contract = contract
                    best_phase = "deterministic_nonstory_fallback"
            continuation_records: list[dict[str, Any]] = []
            while (
                not contract["product_contract_valid"]
                and bounded_completion_attempts < 2
                and current_payload is not None
                and (
                    contract["length_contract"]["actual"]
                    < contract["length_contract"]["minimum"]
                    or (
                        content_type_key == "creative_story"
                        and not contract["story_structure"]["slots"].get(
                            "ending",
                            False,
                        )
                        and contract["length_contract"]["actual"]
                        <= contract["length_contract"]["maximum"] - 8
                    )
                )
                and contract["image_coverage"]["passed"]
                and contract["evidence_coverage"]["passed"]
            ):
                needs_story_ending = (
                    content_type_key == "creative_story"
                    and contract["length_contract"]["actual"]
                    >= contract["length_contract"]["minimum"]
                    and not contract["story_structure"]["slots"].get(
                        "ending",
                        False,
                    )
                )
                continuation_prompt, _, _ = (
                    active_content_candidate.render_bounded_continuation(
                        assets,
                        options,
                        str(current_payload.get("final_text") or ""),
                        attempt_number=bounded_completion_attempts + 1,
                    )
                )
                continuation_result = vlm_service.run_course_prompt(
                    [item["path"] for item in assets],
                    continuation_prompt,
                    prompt_id=prompt_id,
                    prompt_sha256=identity["prompt_sha256"],
                    max_new_tokens=320,
                    image_labels=[
                        f"图片{index}"
                        for index in range(1, len(assets) + 1)
                    ],
                    min_new_tokens=None,
                )
                bounded_completion_attempts += 1
                continuation_record = continuation_result.as_dict()
                continuation_records.append(continuation_record)
                if continuation_result.status != "success":
                    break
                merged = merge_model_authored_addition(
                    current_payload,
                    str(continuation_record["data"].get("raw_output") or ""),
                    assets=assets,
                    target_length=target_length,
                    prefer_ending=needs_story_ending,
                    content_type=content_type_key,
                    target_length_source=str(
                        resolved_options.get("target_length_source")
                        or "user_or_caller"
                    ),
                )
                if merged is None:
                    consider_candidate(
                        continuation_record,
                        phase=f"bounded_continuation_{bounded_completion_attempts}",
                        max_new_tokens=320,
                        dynamic_prompt_sha256=hashlib.sha256(
                            continuation_prompt.encode("utf-8")
                        ).hexdigest(),
                    )
                    break
                previous_risk_sanitized = contract["risk_sanitized"]
                consider_candidate(
                    continuation_record,
                    phase=f"bounded_continuation_{bounded_completion_attempts}",
                    max_new_tokens=320,
                    payload_override=merged,
                    dynamic_prompt_sha256=hashlib.sha256(
                        continuation_prompt.encode("utf-8")
                    ).hexdigest(),
                )
                assert best_normalized is not None and best_contract is not None
                current_payload = best_payload
                normalized = best_normalized
                contract = best_contract
                contract["model_contract_valid"] = False
                contract["repair_applied"] = True
                contract["fallback_applied"] = True
                contract["fallback_source"] = (
                    "model_authored_bounded_completion"
                    if contract["product_contract_valid"]
                    else "friendly_failure"
                )
                contract["risk_sanitized"] = (
                    contract["risk_sanitized"] or previous_risk_sanitized
                )
            if (
                not contract["product_contract_valid"]
                and best_payload is not None
                and bool(best_payload.get("final_text"))
                and contract["image_coverage"]["passed"]
                and contract["evidence_coverage"]["passed"]
                and contract["story_structure"]["passed"]
                and contract["contract_errors"]
                and all(
                    str(error).startswith(
                        "visible_length_contract_violation"
                    )
                    for error in contract["contract_errors"]
                )
            ):
                normalized = {
                    "final_text": str(best_payload["final_text"]).strip(),
                    "actual_length": visible_character_count(
                        str(best_payload["final_text"])
                    ),
                    "target_length": target_length,
                    "accepted_min": contract["length_contract"]["minimum"],
                    "accepted_max": contract["length_contract"]["maximum"],
                    "used_images": [
                        f"图片{position}"
                        for position in contract["image_coverage"][
                            "expected_positions"
                        ]
                    ],
                    "evidence": list(best_payload.get("evidence") or []),
                    "uncertainty": list(
                        best_payload.get("uncertainty") or []
                    ),
                    "story_structure": contract["story_structure"],
                }
                contract["fallback_applied"] = True
                contract["fallback_source"] = "last_parseable_payload"
                contract["fallback_reason"] = (
                    "length_only_revision_failed_preserved_safe_candidate"
                )
            result["data"]["candidate_history"] = candidate_history
            result["data"]["selected_candidate_phase"] = best_phase
            result["data"]["last_parseable_payload"] = best_payload
            result["data"]["best_valid_candidate"] = (
                best_payload
                if contract["product_contract_valid"]
                else None
            )
            if continuation_records or best_payload is not None:
                result["data"]["bounded_completion_attempts"] = (
                    continuation_records
                )
                result["data"]["normalized_merged_payload"] = best_payload
            finished = traces.finish(trace, status="success")
        except Exception as exc:
            finished = traces.finish(trace, status="failed", error=exc)
            minimum = resolved_options["accepted_min"]
            maximum = resolved_options["accepted_max"]
            normalized = {
                "final_text": FRIENDLY_FAILURE_TEXT,
                "actual_length": visible_character_count(FRIENDLY_FAILURE_TEXT),
                "target_length": int(options["target_length"]),
                "accepted_min": minimum,
                "accepted_max": maximum,
                "used_images": [],
                "evidence": [],
                "uncertainty": [],
            }
            contract = {
                "model_contract_valid": False,
                "product_contract_valid": False,
                "contract_valid": False,
                "repair_applied": False,
                "risk_sanitized": False,
                "risk_actions": [],
                "fallback_applied": True,
                "fallback_source": "friendly_failure",
                "fallback_reason": "model_unavailable_or_generation_failed",
                "contract_errors": ["model_unavailable_or_generation_failed"],
                "length_contract": {
                    "unit": "non_whitespace_visible_unicode_characters",
                    "target": int(options["target_length"]),
                    "minimum": minimum,
                    "maximum": maximum,
                    "actual": 0,
                    "passed": False,
                },
                "image_coverage": {
                    "expected_positions": list(range(1, len(assets) + 1)),
                    "model_positions": [],
                    "passed": False,
                },
            }
            result = {
                "status": "failed",
                "data": {
                    "raw_output": None,
                    "parsed_output": None,
                    "prompt_id": prompt_id,
                    "prompt_sha256": identity["prompt_sha256"],
                    "candidate_id": identity["candidate_id"],
                },
                "error": "model_unavailable_or_generation_failed",
                "model": vlm_service.status().get("model"),
                "model_revision": vlm_service.status().get("model_revision"),
            }
        finished = traces.annotate(
            finished["request_id"],
            generation_validation={
                "candidate_id": identity["candidate_id"],
                "prompt_id": prompt_id,
                "prompt_sha256": identity["prompt_sha256"],
                "target_length": int(options["target_length"]),
                "actual_length": contract["length_contract"]["actual"],
                "length_contract_passed": contract["length_contract"]["passed"],
                "image_coverage_passed": contract["image_coverage"]["passed"],
                "model_contract_valid": contract["model_contract_valid"],
                "product_contract_valid": contract["product_contract_valid"],
                "repair_applied": contract["repair_applied"],
                "risk_sanitized": contract["risk_sanitized"],
                "fallback_source": contract["fallback_source"],
                "image_labels_adjacent_to_visual_blocks": True,
                "min_new_tokens": model_min_token_budget(int(options["target_length"])),
                "initial_max_new_tokens": model_token_budget(
                    int(options["target_length"]),
                    content_type_key,
                ),
                "bounded_revision_attempted": bounded_revision_attempted,
                "bounded_completion_attempts": bounded_completion_attempts,
                "selected_candidate_phase": result.get("data", {}).get(
                    "selected_candidate_phase"
                ),
                "candidate_history": result.get("data", {}).get(
                    "candidate_history",
                    [],
                ),
                "contract_errors": contract["contract_errors"],
            },
        )
        public_contract = {
            key: value
            for key, value in contract.items()
            if key != "candidate_final_text"
        }
        return {
            "asset_refs": [item["ref"] for item in assets],
            "options": options,
            "resolved_options": resolved_options,
            "target_length_adjustment": options["target_length_adjustment"],
            "public_hint": options["target_length_adjustment"].get(
                "public_hint"
            ),
            "result": result,
            "final_text": normalized["final_text"],
            "display_text": normalized["final_text"],
            "content": normalized,
            "prompt_candidate": identity,
            **public_contract,
            "baseline_label": (
                "multi_image_story_v1_candidate"
                if content_type_key == "creative_story"
                else "multi_image_content_v2_candidate"
            ),
            "request_id": finished["request_id"],
            "trace": finished,
            "call_source": str(
                options.get("call_source") or "standalone_workspace"
            ),
            "workspace_id": options.get("workspace_id"),
        }

    def course_rank_payload(
        assets: list[dict[str, Any]],
        criterion: str,
        *,
        action: str = "rank",
        scenario: str = "",
        select_count: int = 1,
        original_user_request: str | None = None,
        call_source: str = "standalone_workspace",
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        labeled_assets = [
            {
                **item,
                "image_label": str(
                    item.get("image_label") or f"IMG_{index}"
                ),
            }
            for index, item in enumerate(assets, start=1)
        ]
        labels = [str(item["image_label"]) for item in labeled_assets]
        resolution = {
            "selected_image_labels": labels,
            "current_focus_label": None,
            "resolution_errors": [],
        }
        action_instruction = {
            "compare": "比较这些图片并说明关键差异",
            "select": f"选择最合适的 {select_count} 张图片",
            "rank": "将这些图片从高到低完整排序",
        }.get(action, "将这些图片从高到低完整排序")
        scenario_suffix = f"，使用场景是：{scenario}" if scenario.strip() else ""
        forwarded_user_instruction = str(
            original_user_request or ""
        ).strip()
        current_question = (
            forwarded_user_instruction
            or f"按{criterion}{action_instruction}{scenario_suffix}"
        )
        state = infer_current_turn_state(current_question, resolution)
        state["requested_action"] = {
            "compare": "compare",
            "select": (
                "select_top_k"
                if max(1, min(select_count, len(labels))) > 1
                else "select_one"
            ),
            "rank": "rank_all",
        }.get(action, "rank_all")
        state["comparison_criterion"] = (
            criterion.strip() or "画面表达清晰度（默认标准）"
        )
        state["selection_count"] = max(
            1,
            min(select_count, len(labels)),
        )
        policy, _ = conversational_response_candidate.system_prompt()
        prompt, identity = conversational_response_candidate.render(
            "compare_rank",
            {
                "CONVERSATION_POLICY": policy,
                "CURRENT_STATE": state,
                "SELECTED_ASSET_CONTEXT": safe_asset_facts(labeled_assets),
                "CURRENT_TASK": {
                    "question": current_question,
                    "original_user_request": forwarded_user_instruction,
                    "forwarded_user_instruction": current_question,
                    "criterion": criterion,
                    "required_images": labels,
                    "required_action": action,
                    "select_count": select_count,
                    "scenario": scenario,
                },
            },
        )
        finished: dict[str, Any] | None = None
        initial_call_failed = False
        try:
            result, finished = course_prompt_call(
                prompt_id=identity["prompt_id"],
                prompt=prompt,
                identity=identity,
                assets=labeled_assets,
                task_type="course_rank",
                max_new_tokens=512,
                image_labels=labels,
            )
        except Exception as exc:
            initial_call_failed = True
            result = {
                "status": "failed",
                "data": {"raw_output": None, "parsed_output": None},
                "error": f"initial_model_call_failed:{type(exc).__name__}",
            }
        attempts: list[dict[str, Any]] = []
        common, errors, repaired = normalize_conversational_payload(
            conversational_model_payload(result),
            state=state,
            assets=labeled_assets,
            all_bindings=labeled_assets,
        )
        level = "deterministic_repair" if common is not None and repaired else "direct"
        attempts.append({"level": level, "passed": common is not None, "errors": errors})

        raw_text = str(result["data"].get("raw_output") or "")
        semantic_candidate = any(token in raw_text for token in ("推荐", "排序", "第一", "第1", "IMG"))
        unsafe = any(
            error.startswith(("unknown_", "backend_identity_leak", "unverified_text_claim", "internal_language_leak"))
            for error in errors
        )
        if common is None and semantic_candidate and not unsafe:
            repair_prompt, repair_identity = conversational_response_candidate.render(
                "contract_repair",
                {"CURRENT_STATE": state, "RAW_ANSWER": raw_text},
            )
            try:
                repair_result, _ = conversation_repair_call(prompt=repair_prompt, identity=repair_identity)
                common, repair_errors, _ = normalize_conversational_payload(
                    conversational_model_payload(repair_result),
                    state=state,
                    assets=labeled_assets,
                    all_bindings=labeled_assets,
                )
                errors = repair_errors
                attempts.append({"level": "constrained_repair", "passed": common is not None, "errors": errors})
                if common is not None:
                    level = "constrained_repair"
            except Exception as exc:
                errors = [*errors, f"constrained_repair_failed:{type(exc).__name__}"]
                attempts.append({"level": "constrained_repair", "passed": False, "errors": errors})

        if common is None and not initial_call_failed:
            retry_prompt, retry_identity = conversational_response_candidate.render(
                "compare_rank_retry",
                {
                    "CONVERSATION_POLICY": policy,
                    "CURRENT_STATE": state,
                    "SAFE_FACTS": safe_asset_facts(labeled_assets),
                    "CURRENT_TASK": {
                        "question": current_question,
                        "original_user_request": forwarded_user_instruction,
                        "forwarded_user_instruction": current_question,
                        "criterion": criterion,
                        "required_images": labels,
                        "required_action": action,
                        "select_count": select_count,
                        "scenario": scenario,
                    },
                },
            )
            try:
                retry_result, _ = course_prompt_call(
                    prompt_id=retry_identity["prompt_id"],
                    prompt=retry_prompt,
                    identity=retry_identity,
                    assets=labeled_assets,
                    task_type="course_rank_retry",
                    max_new_tokens=512,
                    image_labels=labels,
                )
                common, retry_errors, _ = normalize_conversational_payload(
                    conversational_model_payload(retry_result),
                    state=state,
                    assets=labeled_assets,
                    all_bindings=labeled_assets,
                )
                errors = retry_errors
                attempts.append({"level": "task_preserving_retry", "passed": common is not None, "errors": errors})
                if common is not None:
                    level = "task_preserving_retry"
            except Exception as exc:
                errors = [*errors, f"task_retry_failed:{type(exc).__name__}"]
                attempts.append({"level": "task_preserving_retry", "passed": False, "errors": errors})

        if common is None:
            common = task_preserving_fallback(labeled_assets, state=state)
            level = "fallback"
            attempts.append({"level": "fallback", "passed": True, "errors": errors})
        annotated_trace = (
            traces.annotate(
                finished["request_id"],
                original_user_message=forwarded_user_instruction,
                resolved_image_refs=labels,
                router_action=(
                    "tool_call"
                    if call_source == "chat_tool_call"
                    else "standalone_workspace"
                ),
                router_tool_name="compare_or_rank_images",
                extracted_action=action,
                extracted_k=(
                    max(1, min(select_count, len(labels)))
                    if action == "select"
                    else len(labels)
                    if action == "rank"
                    else None
                ),
                forwarded_user_instruction=current_question,
                clarification_required=bool(
                    common.get("needs_clarification")
                ),
                clarification_reason=(
                    "comparison_criterion_missing"
                    if common.get("needs_clarification")
                    and not criterion.strip()
                    else None
                ),
                model_messages_summary={
                    "prompt_id": identity["prompt_id"],
                    "prompt_sha256": identity["prompt_sha256"],
                    "forwarded_instruction_sha256": hashlib.sha256(
                        current_question.encode("utf-8")
                    ).hexdigest(),
                    "image_count": len(labels),
                },
            )
            if finished is not None
            else None
        )
        public_contract = attach_public_assets(common, labeled_assets, answer_source=f"phase5_2c_{level}")
        by_label = {str(item["image_label"]): item for item in labeled_assets}
        ranking = [
            {
                "asset_id": str(by_label[item["image_label"]]["asset_id"]),
                "image_label": item["image_label"],
                "rank": item["rank"],
                "reason": item["reason"],
            }
            for item in common["ranking"]
            if item["image_label"] in by_label
        ]
        recommended = [
            {
                "asset_id": str(by_label[label]["asset_id"]),
                "image_label": label,
                "reason": next(
                    (
                        item
                        for item in common["evidence"]
                        if str(item).startswith(f"{label}：")
                    ),
                    common["public_answer"],
                ),
            }
            for label in common["recommended_images"]
            if label in by_label
        ]
        best = ranking[0] if ranking else recommended[0] if recommended else {}
        model_contract_valid = level in {"direct", "deterministic_repair", "constrained_repair", "task_preserving_retry"}
        return {
            "criterion": criterion,
            "scenario": scenario,
            "action": action,
            "select_count": select_count,
            "asset_refs": [item["ref"] for item in assets],
            "ranking": ranking,
            "selected": recommended if action == "select" else [],
            "best_asset_id": best.get("asset_id"),
            "best_reason": best.get("reason", ""),
            "answer": public_contract,
            "public_answer": common["public_answer"],
            "display_text": common["public_answer"],
            "uncertainty": common["uncertainty"],
            "result": result,
            "prompt_candidate": identity,
            "model_contract_valid": model_contract_valid,
            "product_contract_valid": True,
            "fallback_applied": level == "fallback",
            "repair_level": level,
            "repair_attempts": attempts,
            "model_contract_errors": errors,
            "baseline_label": "qwen3_vl_phase5_2c_prompt_ranking_not_trained_reranker",
            "reranker": "NOT IMPLEMENTED",
            "request_id": finished["request_id"] if finished else None,
            "trace": annotated_trace or finished,
            "call_source": call_source,
            "workspace_id": workspace_id,
        }

    def route_chat_tools(
        session: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        chat_state = (
            session.get("chat_state")
            if isinstance(session.get("chat_state"), dict)
            else {}
        )
        active_labels = [
            str(item["image_label"])
            for item in session.get("active_assets", [])
            if item.get("image_label")
        ]
        mapping = (
            chat_state.get("tool_result_image_mapping")
            if isinstance(chat_state.get("tool_result_image_mapping"), dict)
            else {}
        )
        search_labels = sorted(
            (
                str(label)
                for label in mapping
                if str(label).startswith("SEARCH_")
            ),
            key=lambda label: int(label.split("_")[-1]),
        )
        deterministic = deterministic_tool_plan(
            message,
            active_labels=active_labels,
            search_labels=search_labels,
        )
        prompt, identity = chat_tool_router_candidate.render(
            {
                "CURRENT_USER_TURN": message,
                "CURRENT_STATE": {
                    "active_image_labels": active_labels,
                    "available_search_result_labels": search_labels,
                    "last_tool_call": chat_state.get("last_tool_call"),
                    "current_tool_goal": chat_state.get("current_tool_goal"),
                    "pending_tool_action": chat_state.get("pending_tool_action"),
                },
                "DETERMINISTIC_PROPOSAL": {
                    key: deterministic.get(key)
                    for key in (
                        "action",
                        "tool_name",
                        "arguments",
                        "reason_code",
                    )
                },
            }
        )
        router_trace = traces.start(
            "course_chat_tool_router",
            [],
            model=vlm_service.status().get("model"),
            model_revision=vlm_service.status().get("model_revision"),
            prompt_version=identity["prompt_id"],
            prompt_sha256=identity["prompt_sha256"],
            schema_version="phase5_4d_chat_tool_router_v1_2",
            services=["deterministic_router", "qwen_text_router", "router_validator"],
        )
        model_decision: dict[str, Any] | None = None
        model_errors: list[str] = []
        raw_output = ""
        model_called = False
        if hasattr(vlm_service, "run_text_repair"):
            try:
                model_called = True
                model_result = vlm_service.run_text_repair(
                    prompt,
                    prompt_id=identity["prompt_id"],
                    prompt_sha256=identity["prompt_sha256"],
                    max_new_tokens=220,
                )
                raw_output = str(
                    model_result.as_dict().get("data", {}).get("raw_output")
                    or ""
                )
                if model_result.status == "success":
                    model_decision, model_errors = parse_router_output(raw_output)
                else:
                    model_errors = [
                        f"router_model_status:{model_result.status}"
                    ]
            except Exception as exc:
                model_errors = [f"router_model_failed:{type(exc).__name__}"]
        merged = merge_router_decisions(deterministic, model_decision)
        merged_steps = list(merged.get("steps") or [])[:2]
        state_validated, state_errors = validate_router_decision_against_state(
            merged,
            active_labels=active_labels,
            search_labels=search_labels,
        )
        if state_validated is None:
            deterministic_validated, deterministic_state_errors = (
                validate_router_decision_against_state(
                    deterministic,
                    active_labels=active_labels,
                    search_labels=search_labels,
                )
            )
            if deterministic_validated is not None:
                merged = {
                    **deterministic_validated,
                    "steps": list(deterministic.get("steps") or [])[:2],
                    "source": "rule_based_proposal_after_model_rejection",
                    "model_agreed": False,
                    "model_proposal": model_decision,
                }
            else:
                merged = {
                    "action": "clarification",
                    "tool_name": None,
                    "arguments": {},
                    "steps": [],
                    "reason_code": "router_state_validation_failed",
                    "confidence": "high",
                    "clarification": (
                        "我还无法确定要使用哪些当前图片，请补充图片或说明选择范围。"
                    ),
                    "source": "backend_state_validation",
                    "model_agreed": False,
                    "model_proposal": model_decision,
                }
                state_errors = [
                    *state_errors,
                    *deterministic_state_errors,
                ]
        else:
            merged = {
                **merged,
                "arguments": state_validated["arguments"],
                "steps": merged_steps,
            }
        merged["candidate"] = identity
        merged["model_called"] = model_called
        merged["model_contract_errors"] = [
            *model_errors,
            *state_errors,
        ]
        finished = traces.finish(router_trace, status="success")
        annotated = traces.annotate(
            finished["request_id"],
            deterministic_proposal=deterministic,
            raw_model_output=raw_output,
            parsed_model_output=model_decision,
            model_contract_errors=[*model_errors, *state_errors],
            final_router_decision=merged,
            max_business_steps=2,
        )
        merged["router_trace_id"] = annotated["request_id"]
        return merged

    def resolve_tool_image_refs(
        session: dict[str, Any],
        labels: list[str],
    ) -> list[dict[str, Any]]:
        chat_state = (
            session.get("chat_state")
            if isinstance(session.get("chat_state"), dict)
            else {}
        )
        active = {
            str(item["image_label"]): item
            for item in session.get("active_assets", [])
            if item.get("image_label")
        }
        mapping = (
            chat_state.get("tool_result_image_mapping")
            if isinstance(chat_state.get("tool_result_image_mapping"), dict)
            else {}
        )
        refs: list[CourseAssetRefRequest] = []
        resolved_labels: list[str] = []
        for label in labels:
            item = active.get(label) or mapping.get(label)
            if not isinstance(item, dict):
                raise ValueError(f"tool_image_reference_not_available:{label}")
            source = str(item.get("source") or "library")
            source_asset_id = (
                item.get("source_asset_id")
                or item.get("image_id")
                or item.get("asset_id")
            )
            if source_asset_id is None:
                raise ValueError(f"tool_image_reference_missing_identity:{label}")
            refs.append(
                CourseAssetRefRequest(
                    source=(
                        source
                        if source in {"local", "system", "session"}
                        else "library"
                    ),
                    asset_id=str(source_asset_id).removeprefix("local:"),
                    conversation_id=(
                        session.get("conversation_id")
                        if source == "session"
                        else None
                    ),
                )
            )
            resolved_labels.append(label)
        assets = resolve_course_assets(refs)
        for item, label in zip(assets, resolved_labels):
            item["image_label"] = label
        return assets

    def bind_chat_search_results(
        session: dict[str, Any],
        response: dict[str, Any],
    ) -> dict[str, Any]:
        if response.get("status") == "tool_failed":
            chat_state = session["chat_state"]
            chat_state["tool_error"] = response.get("tool_error")
            chat_state["last_search_results"] = []
            chat_state["selected_tool_images"] = []
            session["chat_state"] = chat_state
            return dict(response)
        public_response = dict(response)
        public_results: list[dict[str, Any]] = []
        mapping: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(response.get("results", []), start=1):
            item = asset_media.resolve_result(dict(raw))
            label = f"SEARCH_{index}"
            source_library = str(item.get("source_library") or "")
            source = str(item.get("source") or "library")
            source_asset_id = (
                item.get("source_asset_id")
                or item.get("asset_id")
                or item.get("image_id")
            )
            if source not in {"local", "system", "session", "library"}:
                source = (
                    "system"
                    if source_library in {"system_train", "system_val"}
                    else "local"
                    if str(source_asset_id).startswith("local:")
                    else "library"
                )
            if source == "local":
                source_asset_id = str(source_asset_id).removeprefix("local:")
            mapping[label] = {
                "tool_image_label": label,
                "source": source,
                "source_asset_id": source_asset_id,
                "asset_id": item.get("asset_id"),
                "image_url": item.get("image_url"),
                "thumbnail_url": item.get("thumbnail_url"),
                "content_url": item.get("content_url"),
                "media_status": item.get("media_status"),
                "rank": index,
                "score": item.get("score"),
                "retrieval_backend": item.get("retrieval_backend"),
                "fallback_used": bool(item.get("fallback_used", False)),
                "index_version": item.get("index_version"),
                "index_identity": item.get("index_identity"),
                "source_library": source_library or None,
                "ordering_basis": item.get("ordering_basis"),
            }
            item["tool_image_label"] = label
            item.pop("image_id", None)
            item.pop("sha256", None)
            item.pop("image_path", None)
            public_results.append(item)
        public_response["results"] = public_results
        public_response["image_cards"] = public_results
        public_response["display_text"] = (
            f"为你找到 {len(public_results)} 张图片。"
            "可以直接选择卡片继续生成、比较或排序，也可以把需要的图片加入当前会话。"
        )
        chat_state = session["chat_state"]
        chat_state["last_search_results"] = [
            {
                "tool_image_label": item["tool_image_label"],
                "rank": item.get("rank"),
                "asset_id": item.get("asset_id"),
                "image_url": item.get("image_url"),
                "thumbnail_url": item.get("thumbnail_url"),
                "content_url": item.get("content_url"),
                "media_status": item.get("media_status"),
                "score": item.get("score"),
                "retrieval_backend": item.get("retrieval_backend"),
                "fallback_used": bool(item.get("fallback_used", False)),
                "index_version": item.get("index_version"),
                "index_identity": item.get("index_identity"),
                "ordering_basis": item.get("ordering_basis"),
            }
            for item in public_results
        ]
        chat_state["tool_result_image_mapping"] = mapping
        chat_state["selected_tool_images"] = []
        session["chat_state"] = chat_state
        return public_response

    def import_chat_search_results(
        session: dict[str, Any],
        labels: list[str],
        destination: str,
    ) -> dict[str, Any]:
        normalized_labels = list(
            dict.fromkeys(str(label).upper() for label in labels)
        )
        if any(
            not label.startswith("SEARCH_")
            for label in normalized_labels
        ):
            raise ValueError("search_result_label_invalid")
        imported_assets = resolve_tool_image_refs(
            session,
            normalized_labels,
        )
        if destination == "chat_context":
            current_assets = resolve_course_assets(session_refs(session))
            by_ref = {
                str(item["ref"]): item
                for item in current_assets
            }
            for item in imported_assets:
                by_ref.setdefault(str(item["ref"]), item)
            merged = list(by_ref.values())
            if len(merged) > 5:
                raise ValueError("chat_context_maximum_is_5_images")
            saved = product.update_conversation_assets(
                session["conversation_id"],
                public_course_assets(merged),
            )
            label_by_ref = {
                str(item["ref"]): str(item["image_label"])
                for item in saved.get("active_assets", [])
                if item.get("ref") and item.get("image_label")
            }
            imported = [
                {
                    "search_label": label,
                    "image_label": label_by_ref.get(str(item["ref"])),
                }
                for label, item in zip(
                    normalized_labels,
                    imported_assets,
                )
            ]
            return {
                "destination": destination,
                "display_text": (
                    "已加入当前会话："
                    + "、".join(
                        str(item["image_label"])
                        for item in imported
                        if item.get("image_label")
                    )
                    + "。"
                ),
                "imported": imported,
                "session": saved,
            }

        workspace_kind = (
            "generation"
            if destination == "generation_workspace"
            else "compare"
        )
        workspace = product.workspace(workspace_kind)
        current_refs = [
            CourseAssetRefRequest(
                source=str(item.get("source") or "library"),
                asset_id=str(
                    item.get("source_asset_id")
                    or item.get("image_id")
                    or item.get("asset_id")
                ).removeprefix("local:"),
            )
            for item in workspace.get("selected_assets", [])
        ]
        current_assets = resolve_course_assets(current_refs)
        by_ref = {
            str(item["ref"]): item
            for item in current_assets
        }
        for item in imported_assets:
            by_ref.setdefault(str(item["ref"]), item)
        merged = list(by_ref.values())
        if len(merged) > 5:
            raise ValueError("workspace_maximum_is_5_images")
        saved_workspace = product.save_workspace(
            workspace_kind,
            selected_assets=public_course_assets(merged),
            local_options=dict(workspace.get("local_options") or {}),
            last_result=workspace.get("last_result"),
        )
        return {
            "destination": destination,
            "display_text": (
                f"已将 {len(imported_assets)} 张搜索结果加入"
                f"{'生成' if workspace_kind == 'generation' else '比较排序'}工作区。"
            ),
            "imported": [
                {"search_label": label}
                for label in normalized_labels
            ],
            "workspace": saved_workspace,
        }

    def course_compare_tool_payload(
        assets: list[dict[str, Any]],
        instruction: str,
    ) -> dict[str, Any]:
        trace = traces.start(
            "course_compare_tool",
            [str(item["image_label"]) for item in assets],
            model=vlm_service.status().get("model"),
            model_revision=vlm_service.status().get("model_revision"),
            prompt_version="phase5_2c_compare_rank_v2",
            schema_version="phase5_4_compare_tool_v1",
            services=["chat_tool_router", "vlm_compare"],
        )
        try:
            result = vlm_service.compare_images(
                [item["path"] for item in assets],
                instruction,
            )
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            parsed = result.data.get("parsed_output")
            summary = parsed if parsed not in (None, {}, []) else result.data.get(
                "raw_output"
            )
            finished = traces.finish(trace, status="success")
            return {
                "status": "success",
                "action": "compare",
                "answer": summary,
                "result": result.as_dict(),
                "display_text": extract_display_text(
                    {"answer": summary},
                    fallback="图片比较已完成，详细依据保留在技术记录中。",
                ),
                "request_id": finished["request_id"],
                "trace": finished,
            }
        except Exception as exc:
            traces.finish(trace, status="failed", error=exc)
            raise

    def save_course_turn(
        session: dict[str, Any],
        *,
        question: str,
        intent: str,
        assets: list[dict[str, Any]],
        response: dict[str, Any],
        bucket: str | None = None,
        reference_resolution: dict[str, Any] | None = None,
        decision_follow_up: dict[str, Any] | None = None,
        conversation_follow_up: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timestamp = now_iso()
        refs = [item["ref"] for item in assets]
        request_id = response.get("request_id")
        tool_call_id = str(uuid.uuid4())
        tool_call = {
            "tool_call_id": tool_call_id,
            "tool_name": intent,
            "asset_refs": refs,
            "status": "completed",
            "trace_id": request_id,
            "created_at": timestamp,
        }
        tool_result = {
            "tool_call_id": tool_call_id,
            "tool_name": intent,
            "status": "completed",
            "trace_id": request_id,
            "result": response,
            "created_at": timestamp,
        }
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        session["messages"].extend(
            [
                {
                    "message_id": user_message_id,
                    "role": "user",
                    "content": question,
                    "asset_refs": refs,
                    "image_references": (
                        reference_resolution.get("selected_image_labels", [])
                        if reference_resolution
                        else []
                    ),
                    "resolved_question": (
                        reference_resolution.get("resolved_question")
                        if reference_resolution
                        else question
                    ),
                    "task_type": intent,
                    "created_at": timestamp,
                },
                {
                    "message_id": assistant_message_id,
                    "role": "assistant",
                    "content": response,
                    "asset_refs": refs,
                    "task_type": intent,
                    "trace_id": request_id,
                    "created_at": now_iso(),
                },
            ]
        )
        session["tool_calls"].append(tool_call)
        session["tool_results"].append(tool_result)
        if bucket:
            session[bucket].append(response)
        if len(session["messages"]) == 2:
            session["title"] = question.strip()[:48]
        if reference_resolution is not None:
            answer_payload = response.get("answer")
            if isinstance(answer_payload, dict):
                answer_text = str(answer_payload.get("answer") or "")
            else:
                answer_text = str(answer_payload or "")
            update_chat_state_after_turn(
                session,
                question=question,
                selected_labels=list(reference_resolution.get("selected_image_labels", [])),
                answer=answer_text,
                requires_clarification=bool(reference_resolution.get("requires_clarification")),
            )
        if str(response.get("action") or "") in {
            "compare",
            "select",
            "rank",
        }:
            update_decision_state_after_turn(
                session,
                response=response,
                image_scope=(
                    list(
                        reference_resolution.get(
                            "selected_image_labels",
                            [],
                        )
                    )
                    if reference_resolution
                    else [
                        str(item.get("image_label"))
                        for item in assets
                        if item.get("image_label")
                    ]
                ),
                assistant_message_id=assistant_message_id,
            )
        elif decision_follow_up and decision_follow_up.get("bound"):
            continue_decision_state_after_follow_up(
                session,
                assistant_message_id=assistant_message_id,
            )
        turn_state = update_general_turn_state(
            session,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            question=question,
            intent=intent,
            assets=assets,
            response=response,
            follow_up=conversation_follow_up,
            created_at=timestamp,
        )
        answer_payload = (
            response.get("answer")
            if isinstance(response.get("answer"), dict)
            else {}
        )
        public_answer = extract_display_text(
            response,
            fallback="",
        )
        append_completed_turn(
            session,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            question=question,
            public_answer=public_answer,
            task_type=str(turn_state.get("task_type") or intent),
            image_labels=list(
                turn_state.get("primary_target_images") or []
            ),
            answer_provenance=str(
                answer_payload.get("answer_source")
                or response.get("fallback_source")
                or (
                    "model_direct"
                    if response.get("model_called") is not False
                    else "deterministic"
                )
            ),
            tool_call=tool_call,
            tool_result=response,
            dialogue_state={
                "dialogue_act": turn_state.get("dialogue_act"),
                "active_task": turn_state.get("task_type"),
                "discourse_focus": turn_state.get(
                    "answer_subject"
                ),
                "target_images": list(
                    turn_state.get("primary_target_images") or []
                ),
                "inherited_turn": turn_state.get(
                    "referenced_turn_id"
                ),
                "criterion": turn_state.get("criterion"),
                "action": turn_state.get("tool_action"),
                "k": (
                    turn_state.get("task_frame", {}).get("k")
                ),
                "output_style": (
                    turn_state.get("task_frame", {}).get(
                        "output_style"
                    )
                ),
                "task_frame": turn_state.get("task_frame"),
            },
            created_at=timestamp,
        )
        session["updated_at"] = now_iso()
        return product.save_session(session)

    @app.post("/course/conversations")
    def create_course_conversation(request: CourseConversationRequest) -> dict[str, Any]:
        assets = resolve_course_assets(request.asset_refs)
        return product.create_conversation(title=request.title, active_assets=public_course_assets(assets))

    @app.patch("/course/conversations/{conversation_id}/assets")
    def update_course_conversation_assets(
        conversation_id: str,
        request: CourseConversationAssetsRequest,
    ) -> dict[str, Any]:
        assets = resolve_course_assets(request.asset_refs)
        focus_kwargs = (
            {"focus_image_label": request.focus_image_label}
            if "focus_image_label" in request.model_fields_set
            else {}
        )
        return run_or_http_error(
            product.update_conversation_assets,
            conversation_id,
            public_course_assets(assets),
            operation_id=request.operation_id,
            workspace_id=request.workspace_id,
            client_sequence=request.client_sequence,
            expected_version=request.expected_version,
            **focus_kwargs,
        )

    @app.post(
        "/course/conversations/{conversation_id}/search-results/import"
    )
    def import_course_search_results(
        conversation_id: str,
        request: CourseSearchResultImportRequest,
    ) -> dict[str, Any]:
        session = product.get_session(conversation_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="conversation_not_found",
            )
        return run_or_http_error(
            import_chat_search_results,
            session,
            request.search_labels,
            request.destination,
        )

    @app.put(
        "/course/conversations/{conversation_id}/search-results/selection"
    )
    def select_course_search_results(
        conversation_id: str,
        request: CourseSearchResultSelectionRequest,
    ) -> dict[str, Any]:
        session = product.get_session(conversation_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="conversation_not_found",
            )
        mapping = session["chat_state"].get(
            "tool_result_image_mapping",
            {},
        )
        labels = list(
            dict.fromkeys(
                str(label).upper()
                for label in request.search_labels
            )
        )
        if any(label not in mapping for label in labels):
            raise HTTPException(
                status_code=409,
                detail="search_result_not_available",
            )
        session["chat_state"]["selected_tool_images"] = labels
        saved = product.save_session(session)
        return {
            "selected_search_labels": labels,
            "conversation_id": saved["conversation_id"],
        }

    @app.get("/course/workspaces")
    def course_workspaces() -> dict[str, Any]:
        return {
            "call_source": "standalone_workspace",
            "items": product.workspaces(),
        }

    @app.get("/course/workspaces/{kind}")
    def course_workspace(
        kind: Literal["generation", "retrieval", "compare"],
    ) -> dict[str, Any]:
        return product.workspace(kind)

    @app.put("/course/workspaces/{kind}")
    def update_course_workspace(
        kind: Literal["generation", "retrieval", "compare"],
        request: FunctionWorkspaceUpdateRequest,
    ) -> dict[str, Any]:
        assets = run_or_http_error(resolve_course_assets, request.selected_assets)
        return run_or_http_error(
            product.save_workspace,
            kind,
            selected_assets=public_course_assets(assets),
            local_options=request.local_options,
            last_result=request.last_result,
        )

    @app.post("/course/workspaces/{kind}/operations")
    def operate_course_workspace(
        kind: Literal["generation", "retrieval", "compare"],
        request: FunctionWorkspaceOperationRequest,
    ) -> dict[str, Any]:
        resolved_asset = (
            run_or_http_error(resolve_course_assets, [request.asset])[0]
            if request.asset is not None
            else None
        )
        resolved_assets = (
            run_or_http_error(
                resolve_course_assets,
                request.selected_assets,
            )
            if request.selected_assets
            else []
        )
        return run_or_http_error(
            product.apply_workspace_operation,
            kind,
            operation_id=request.operation_id,
            workspace_id=request.workspace_id,
            action=request.action,
            asset=(
                public_course_assets([resolved_asset])[0]
                if resolved_asset is not None
                else None
            ),
            asset_ref=request.asset_ref,
            selected_assets=public_course_assets(resolved_assets),
            direction=request.direction,
            client_sequence=request.client_sequence,
            expected_version=request.expected_version,
            local_options=request.local_options,
            last_result=request.last_result,
        )

    @app.post("/course/retrieve")
    def course_retrieve(request: CourseRetrieveRequest) -> dict[str, Any]:
        return run_or_http_error(
            course_retrieve_payload,
            request.query_text,
            request.query_asset_ref,
            query_asset_refs=request.query_asset_refs,
            top_k=request.top_k,
            exclude_query_images=request.exclude_query_images,
            library_scope=request.library_scope,
            current_library_id=request.current_library_id,
            library_ids=request.library_ids,
            call_source=request.call_source,
            workspace_id=request.workspace_id,
        )

    @app.get("/course/content-length-profiles")
    def course_content_length_profiles() -> dict[str, Any]:
        return public_content_length_config()

    @app.post("/course/generate")
    def course_generate(request: CourseGenerateRequest) -> dict[str, Any]:
        assets = run_or_http_error(resolve_course_assets, request.asset_refs)
        session = None
        if request.conversation_id:
            session = product.get_session(request.conversation_id)
            if session is None:
                raise HTTPException(
                    status_code=404,
                    detail="conversation_not_found",
                )
            assets = run_or_http_error(
                canonicalize_session_assets,
                session,
                assets,
            )
            commit_referenced_image_scope(
                session,
                assets,
                use_kind="generate_content_from_images",
            )
        intent_resolution = resolve_content_type(
            requested_content_type=request.content_type,
            natural_language_request=request.natural_language_request,
            content_type_source=request.content_type_source,
            content_type_user_selected=request.content_type_user_selected,
        )
        if intent_resolution["route_hint"] == "compare_rank":
            raise HTTPException(
                status_code=409,
                detail=(
                    "当前请求属于图片比较、选择或排序，请使用“比较与排序”功能。"
                ),
            )
        options = {
            "content_type": request.content_type,
            "resolved_content_type": intent_resolution["resolved_content_type"],
            "content_type_source": intent_resolution["content_type_source"],
            "content_type_user_selected": intent_resolution[
                "content_type_user_selected"
            ],
            "natural_language_request": request.natural_language_request,
            "intent_resolution": intent_resolution,
            "target_length": request.target_length,
            "style": request.style,
            "audience": request.audience,
            "organization": request.organization,
            "importance": request.importance,
            "call_source": request.call_source,
            "workspace_id": request.workspace_id,
        }
        payload = run_or_http_error(course_generate_payload, assets, options)
        if session is not None:
            save_course_turn(
                session,
                question=(
                    request.natural_language_request.strip()
                    or f"生成 {intent_resolution['resolved_content_type']}"
                ),
                intent="generate",
                assets=assets,
                response=payload,
                bucket="generated_content",
            )
        return payload

    @app.post("/course/rank")
    def course_rank(request: CourseRankRequest) -> dict[str, Any]:
        assets = run_or_http_error(resolve_course_assets, request.asset_refs)
        session = None
        if request.conversation_id:
            session = product.get_session(request.conversation_id)
            if session is None:
                raise HTTPException(
                    status_code=404,
                    detail="conversation_not_found",
                )
            assets = run_or_http_error(
                canonicalize_session_assets,
                session,
                assets,
            )
            commit_referenced_image_scope(
                session,
                assets,
                use_kind="compare_or_rank_images",
            )
        payload = run_or_http_error(
            course_rank_payload,
            assets,
            request.criterion,
            call_source=request.call_source,
            workspace_id=request.workspace_id,
        )
        if session is not None:
            save_course_turn(
                session,
                question=request.criterion,
                intent="rank",
                assets=assets,
                response=payload,
                bucket="ranking_results",
            )
        return payload

    @app.post("/course/compare")
    def course_compare(request: CourseCompareRequest) -> dict[str, Any]:
        assets = run_or_http_error(resolve_course_assets, request.asset_refs)
        return run_or_http_error(
            course_rank_payload,
            assets,
            request.criterion,
            action=request.action,
            scenario=request.scenario,
            select_count=min(request.select_count, len(assets)),
            call_source=request.call_source,
            workspace_id=request.workspace_id,
        )

    @app.post("/course/chat")
    def course_chat(request: CourseChatRequest) -> dict[str, Any]:
        session = product.get_session(request.conversation_id) if request.conversation_id else None
        if request.conversation_id and session is None:
            raise HTTPException(status_code=404, detail="conversation_not_found")
        if session is None:
            session = product.create_conversation(title=request.message)
        early_utility = detect_system_utility(request.message)
        if (
            early_utility is not None
            and early_utility.get("name") == "current_model_identity"
        ):
            response = {
                "status": "success",
                "answer": current_model_identity_answer(),
                "intent": "system_utility",
                "model_called": False,
                "visual_model_called": False,
                "request_id": None,
            }
            response["display_text"] = extract_display_text(response)
            response["tool_router"] = {
                "action": "direct_answer",
                "tool_name": None,
                "arguments": {},
                "steps": [],
                "reason_code": "system_utility:current_model_identity",
                "confidence": "high",
                "model_called": False,
                "model_agreed": None,
                "model_contract_errors": [],
                "router_trace_id": None,
                "candidate": None,
                "system_utility": early_utility,
            }
            saved = save_course_turn(
                session,
                question=request.message,
                intent="system_utility",
                assets=[],
                response=response,
            )
            return {
                "conversation_id": saved["conversation_id"],
                "intent": "system_utility",
                "tool_router": response["tool_router"],
                "active_assets": saved["active_assets"],
                "response": response,
                "display_text": response["display_text"],
                "messages": saved["messages"],
                "context_summary": saved["context_summary"],
                "tool_calls": saved["tool_calls"],
                "tool_results": saved["tool_results"],
                "chat_state": saved["chat_state"],
                "request_id": None,
            }
        if request.asset_refs is not None:
            selected = run_or_http_error(resolve_course_assets, request.asset_refs)
            session = product.update_conversation_assets(session["conversation_id"], public_course_assets(selected))
        else:
            selected = run_or_http_error(resolve_course_assets, session_refs(session))
        conversation_follow_up = resolve_general_follow_up(
            request.message,
            session,
        )
        if (
            conversation_follow_up
            and conversation_follow_up.get("execution_mode")
            == "model_rewrite"
        ):
            conversation_follow_up = contextual_rewrite_call(
                request.message,
                session,
                conversation_follow_up,
            )
        decision_follow_up = (
            None
            if conversation_follow_up is not None
            else resolve_decision_follow_up(
                request.message,
                session,
            )
        )
        available_search_labels = sorted(
            (
                str(label)
                for label in session["chat_state"].get(
                    "tool_result_image_mapping",
                    {},
                )
                if str(label).startswith("SEARCH_")
            ),
            key=lambda label: int(label.split("_")[-1]),
        )
        system_utility = detect_system_utility(
            request.message,
            search_labels=available_search_labels,
        )
        if (
            conversation_follow_up
            and conversation_follow_up.get(
                "requires_clarification"
            )
        ):
            router_decision = {
                "action": "clarification",
                "tool_name": None,
                "arguments": {},
                "steps": [],
                "reason_code": (
                    "conversation_follow_up:"
                    + str(
                        conversation_follow_up.get(
                            "reason_code"
                        )
                    )
                ),
                "confidence": conversation_follow_up.get(
                    "confidence", "high"
                ),
                "model_called": bool(
                    conversation_follow_up.get(
                        "rewriter_model_called"
                    )
                ),
                "model_agreed": None,
                "model_contract_errors": list(
                    conversation_follow_up.get(
                        "resolution_errors", []
                    )
                ),
                "router_trace_id": (
                    conversation_follow_up.get(
                        "rewriter_trace_id"
                    )
                ),
                "candidate": conversation_follow_up.get(
                    "rewriter_prompt_candidate"
                ),
                "clarification": conversation_follow_up[
                    "clarification"
                ],
                "system_utility": None,
            }
        elif (
            conversation_follow_up
            and conversation_follow_up.get("bound")
            and str(
                (
                    conversation_follow_up.get(
                        "inherited_task_frame"
                    )
                    or {}
                ).get("action")
                or ""
            )
            in {"compare", "select", "rank", "top_k"}
            and str(
                conversation_follow_up.get("follow_up_type")
                or ""
            )
            in {
                "same_task_new_targets",
                "criterion_substitution",
                "action_substitution",
                "targets_and_action_substitution",
            }
        ):
            inherited_frame = dict(
                conversation_follow_up[
                    "inherited_task_frame"
                ]
            )
            router_decision = {
                "action": "tool_call",
                "tool_name": "compare_or_rank_images",
                "arguments": {
                    "image_refs": list(
                        conversation_follow_up[
                            "selected_image_labels"
                        ]
                    ),
                    "action": (
                        "select"
                        if inherited_frame["action"] == "top_k"
                        else inherited_frame["action"]
                    ),
                    "criterion": str(
                        inherited_frame.get("criterion") or ""
                    ),
                    "select_count": int(
                        inherited_frame.get("k") or 1
                    ),
                    "scenario": str(
                        inherited_frame.get("output_style") or ""
                    ),
                },
                "steps": [],
                "reason_code": (
                    "conversation_follow_up:task_frame_inheritance"
                ),
                "confidence": "high",
                "model_called": False,
                "model_agreed": None,
                "model_contract_errors": [],
                "router_trace_id": None,
                "candidate": None,
                "system_utility": None,
            }
        elif (
            conversation_follow_up
            and conversation_follow_up.get("bound")
        ):
            router_decision = {
                "action": "direct_answer",
                "tool_name": None,
                "arguments": {
                    "image_refs": conversation_follow_up[
                        "selected_image_labels"
                    ],
                },
                "steps": [],
                "reason_code": (
                    "conversation_follow_up:"
                    + str(
                        conversation_follow_up[
                            "follow_up_type"
                        ]
                    )
                ),
                "confidence": conversation_follow_up.get(
                    "confidence", "high"
                ),
                "model_called": bool(
                    conversation_follow_up.get(
                        "rewriter_model_called"
                    )
                ),
                "model_agreed": None,
                "model_contract_errors": [],
                "router_trace_id": (
                    conversation_follow_up.get(
                        "rewriter_trace_id"
                    )
                ),
                "candidate": conversation_follow_up.get(
                    "rewriter_prompt_candidate"
                ),
                "system_utility": None,
            }
        elif (
            decision_follow_up
            and decision_follow_up.get("requires_clarification")
        ):
            router_decision = {
                "action": "clarification",
                "tool_name": None,
                "arguments": {},
                "steps": [],
                "reason_code": "decision_follow_up_without_immediate_decision",
                "confidence": "high",
                "model_called": False,
                "model_agreed": None,
                "model_contract_errors": [],
                "router_trace_id": None,
                "candidate": None,
                "clarification": decision_follow_up["clarification"],
                "system_utility": None,
            }
        elif decision_follow_up and decision_follow_up.get("bound"):
            router_decision = {
                "action": "direct_answer",
                "tool_name": None,
                "arguments": {
                    "image_refs": decision_follow_up[
                        "selected_image_labels"
                    ],
                },
                "steps": [],
                "reason_code": (
                    "decision_follow_up:"
                    + str(decision_follow_up["follow_up_type"])
                ),
                "confidence": "high",
                "model_called": False,
                "model_agreed": None,
                "model_contract_errors": [],
                "router_trace_id": None,
                "candidate": None,
                "system_utility": None,
            }
        elif system_utility is not None:
            router_decision = {
                "action": "direct_answer",
                "tool_name": None,
                "arguments": {},
                "steps": [],
                "reason_code": f"system_utility:{system_utility['name']}",
                "confidence": "high",
                "model_called": False,
                "model_agreed": None,
                "model_contract_errors": [],
                "router_trace_id": None,
                "candidate": None,
                "system_utility": system_utility,
            }
        else:
            router_decision = route_chat_tools(session, request.message)
        active_image_count = len(
            [
                item
                for item in session.get("active_assets", [])
                if item.get("image_label")
            ]
        )
        unique_image_business_guard = deterministic_tool_plan(
            request.message,
            active_labels=[
                str(item["image_label"])
                for item in session.get("active_assets", [])
                if item.get("image_label")
            ],
            search_labels=[],
        )
        if (
            active_image_count == 1
            and router_decision.get("action") == "clarification"
            and unique_image_business_guard.get("action") == "direct_answer"
            and unique_image_business_guard.get("tool_name") is None
            and unique_image_business_guard.get("reason_code")
            == "no_business_tool_required"
            and system_utility is None
            and conversation_follow_up is None
            and decision_follow_up is None
        ):
            # The deterministic router's finite contract, rather than the
            # model's free-form reason text, establishes that this is ordinary
            # semantic chat and not a missing business-tool parameter. With
            # one authoritative image, let the multimodal answer model resolve
            # the meaning. Contract and state clarifications retain their
            # explicit deterministic/follow-up routes.
            router_decision = {
                "action": "direct_answer",
                "tool_name": None,
                "arguments": {"image_refs": []},
                "steps": [],
                "reason_code": "unique_image_semantic_default",
                "confidence": "high",
                "model_called": bool(router_decision.get("model_called")),
                "model_agreed": router_decision.get("model_agreed"),
                "model_contract_errors": list(
                    router_decision.get("model_contract_errors") or []
                ),
                "router_trace_id": router_decision.get("router_trace_id"),
                "candidate": router_decision.get("candidate"),
                "system_utility": None,
            }
        routed_tool = (
            str(router_decision.get("tool_name"))
            if router_decision.get("action") == "tool_call"
            else None
        )
        router_arguments = dict(
            router_decision.get("arguments") or {}
        )
        if (
            conversation_follow_up
            and conversation_follow_up.get("bound")
            and routed_tool != "compare_or_rank_images"
        ):
            execution_mode = conversation_follow_up.get(
                "execution_mode"
            )
            intent = (
                "text_transform"
                if execution_mode == "text_transform"
                else (
                    "explain"
                    if conversation_follow_up[
                        "follow_up_type"
                    ]
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
                        or "vqa"
                    )
                )
            )
        elif decision_follow_up and decision_follow_up.get("bound"):
            intent = "explain"
        elif routed_tool == "generate_content_from_images":
            intent = "generate"
        elif routed_tool == "search_images":
            intent = "retrieve"
        elif routed_tool == "compare_or_rank_images":
            router_compare_action = str(
                router_arguments.get("action") or "auto"
            )
            intent = (
                "rank"
                if router_compare_action == "rank"
                else "recommend"
                if router_compare_action == "select"
                else "compare"
            )
        else:
            intent = classify_intent(request.message)
        routed_image_refs = [
            str(item) for item in router_arguments.get("image_refs", [])
        ]
        if (
            routed_tool is None
            and any(
                label.startswith("SEARCH_")
                for label in routed_image_refs
            )
        ):
            # “搜索结果第二张是什么” references an existing result; it is
            # an open visual question, not a request to run retrieval again.
            intent = "vqa"
        router_reference_resolution: dict[str, Any] | None = None
        if routed_tool == "compare_or_rank_images":
            router_reference_resolution = resolve_image_references(
                request.message,
                session,
            )
            resolved_labels = list(
                router_reference_resolution.get(
                    "selected_image_labels",
                    [],
                )
            )
            if (
                not router_reference_resolution.get(
                    "requires_clarification",
                    False,
                )
                and len(resolved_labels) >= 2
            ):
                routed_image_refs = resolved_labels
        if routed_image_refs and (
            routed_tool is not None
            or any(
                label.startswith("SEARCH_")
                for label in routed_image_refs
            )
        ):
            selected = run_or_http_error(
                resolve_tool_image_refs,
                session,
                routed_image_refs,
            )
        legacy_context_plan = build_context_plan(session, request.message, public_course_assets(selected))
        session["context_summary"] = legacy_context_plan["context_summary"]
        reference_resolution: dict[str, Any] | None = None
        referenced_scope_commit: dict[str, Any] | None = None

        if (
            system_utility is not None
            and system_utility["name"]
            == "add_search_results_to_context"
        ):
            promotion = run_or_http_error(
                import_chat_search_results,
                session,
                list(
                    system_utility["arguments"].get(
                        "search_labels",
                        [],
                    )
                ),
                "chat_context",
            )
            session = promotion["session"]
            selected = run_or_http_error(
                resolve_course_assets,
                session_refs(session),
            )
            semantic_remainder = str(
                system_utility["arguments"].get(
                    "semantic_remainder", ""
                )
            ).strip()
            if semantic_remainder:
                continued = course_chat(
                    CourseChatRequest(
                        conversation_id=session["conversation_id"],
                        message=semantic_remainder,
                    )
                )
                continued["compound_action"] = {
                    "name": "add_search_results_to_context",
                    "imported": promotion["imported"],
                    "display_text": promotion["display_text"],
                    "semantic_remainder": semantic_remainder,
                    "step_count": 2,
                }
                return continued
            response = {
                "status": "success",
                "answer": promotion["display_text"],
                "imported": promotion["imported"],
                "intent": "system_utility",
                "model_called": False,
                "request_id": None,
            }
            intent = "system_utility"
            bucket = None
        elif router_decision.get("action") == "clarification":
            response = {
                "status": "clarification_required",
                "answer": (
                    router_decision.get("clarification")
                    or "我还缺少一个必要条件，请说明要使用哪些图片或希望执行什么操作。"
                ),
                "request_id": None,
                "model_called": bool(router_decision.get("model_called")),
            }
            intent = "clarification"
            bucket = None
        elif intent == "retrieve":
            search_mode = str(router_arguments.get("mode") or "auto")
            query_asset_refs = (
                [
                    CourseAssetRefRequest(
                        source=item["source"],
                        asset_id=item["source_asset_id"],
                    )
                    for item in selected
                ]
                if search_mode != "text"
                else []
            )
            query_text = (
                None
                if search_mode == "image"
                else str(
                    router_arguments.get("text_query")
                    or request.message
                ).strip()
            )
            if search_mode in {"image", "hybrid"} and not query_asset_refs:
                response = {
                    "status": "clarification_required",
                    "answer": "请指定当前会话中的图片，或改为只根据文字搜索。",
                    "request_id": None,
                    "model_called": bool(router_decision.get("model_called")),
                }
                intent = "clarification"
                bucket = None
            else:
                chat_library_scope, chat_library_ids = (
                    resolve_chat_retrieval_scope(
                        request.message,
                        router_arguments,
                    )
                )
                referenced_scope_commit = (
                    commit_referenced_image_scope(
                        session,
                        selected,
                        use_kind="search_images",
                        confidence=str(
                            router_decision.get("confidence") or ""
                        ),
                    )
                )
                raw_search_response = run_or_http_error(
                    course_retrieve_payload,
                    query_text,
                    None,
                    query_asset_refs=query_asset_refs,
                    top_k=int(router_arguments.get("top_k", 5)),
                    exclude_query_images=bool(
                        router_arguments.get("exclude_query_images", True)
                    ),
                    library_scope=chat_library_scope,
                    library_ids=chat_library_ids,
                    call_source="chat_tool_call",
                )
                search_response = bind_chat_search_results(
                    session,
                    raw_search_response,
                )
                steps = (
                    []
                    if search_response.get("status") == "tool_failed"
                    else list(router_decision.get("steps") or [])
                )
                if (
                    len(steps) == 2
                    and steps[1].get("tool_name")
                    == "generate_content_from_images"
                ):
                    generation_arguments = dict(
                        steps[1].get("arguments") or {}
                    )
                    generation_assets = run_or_http_error(
                        resolve_tool_image_refs,
                        session,
                        [
                            str(item)
                            for item in generation_arguments.get(
                                "image_refs", []
                            )
                        ],
                    )
                    if not generation_assets:
                        raise HTTPException(
                            status_code=409,
                            detail="search_then_generate_has_no_bound_results",
                        )
                    intent_resolution = resolve_content_type(
                        requested_content_type="auto",
                        natural_language_request=request.message,
                        content_type_source="auto_inferred",
                        content_type_user_selected=False,
                    )
                    generation_response = run_or_http_error(
                        course_generate_payload,
                        generation_assets,
                        {
                            **generation_arguments,
                            "resolved_content_type": intent_resolution[
                                "resolved_content_type"
                            ],
                            "content_type_source": intent_resolution[
                                "content_type_source"
                            ],
                            "content_type_user_selected": (
                                False
                            ),
                            "intent_resolution": intent_resolution,
                            "call_source": "chat_tool_call",
                        },
                    )
                    response = {
                        "status": "success",
                        "tool_chain": [
                            {
                                "tool_name": "search_images",
                                "result": search_response,
                            },
                            {
                                "tool_name": "generate_content_from_images",
                                "result": generation_response,
                            },
                        ],
                        "search_results": search_response["results"],
                        "image_cards": search_response["image_cards"],
                        "generated_content": generation_response,
                        "display_text": extract_display_text(
                            generation_response
                        ),
                        "request_id": generation_response.get("request_id"),
                    }
                    selected = generation_assets
                    intent = "generate"
                    bucket = "generated_content"
                else:
                    response = search_response
                    bucket = "retrieval_results"
        elif (
            conversation_follow_up
            and conversation_follow_up.get("bound")
            and conversation_follow_up.get("execution_mode")
            == "text_transform"
        ):
            response = direct_chat_payload(
                str(
                    conversation_follow_up.get(
                        "standalone_request"
                    )
                    or request.message
                )
            )
            response.update(
                {
                    "intent": "text_transform",
                    "contextual_follow_up": (
                        conversation_follow_up
                    ),
                    "context_trace": {
                        "original_user_message": request.message,
                        "rewritten_standalone_request": (
                            conversation_follow_up.get(
                                "standalone_request"
                            )
                        ),
                        "follow_up_type": (
                            conversation_follow_up.get(
                                "follow_up_type"
                            )
                        ),
                        "referenced_turn_id": (
                            conversation_follow_up.get(
                                "referenced_turn_id"
                            )
                        ),
                        "resolved_image_refs": list(
                            conversation_follow_up.get(
                                "selected_image_labels", []
                            )
                        ),
                        "actual_images_sent_to_model": [],
                        "visual_model_called": False,
                        "rewriter_level": (
                            conversation_follow_up.get(
                                "rewriter_level"
                            )
                        ),
                    },
                }
            )
            intent = "text_transform"
            bucket = None
        elif (
            router_decision.get("action") == "direct_answer"
            and not (
                decision_follow_up
                and decision_follow_up.get("bound")
            )
            and not (
                conversation_follow_up
                and conversation_follow_up.get("bound")
            )
            and active_image_count != 1
            and not visual_groundable_intent(
                request.message,
                active_image_count=len(
                    [
                        item
                        for item in session.get("active_assets", [])
                        if item.get("image_label")
                    ]
                ),
                has_visual_history=bool(
                    session.get("chat_state", {}).get(
                        "last_visual_answer_turn"
                    )
                ),
            )
        ):
            response = direct_chat_payload(request.message)
            intent = "direct_chat"
            bucket = None
        elif not selected:
            response = {
                "status": "help_without_image",
                "answer": "当前会话没有图片。我可以介绍功能或进行文本检索；如需视觉问答、生成、比较或排序，请先从图片库或本地导入中加入 1-5 张图片。",
                "intent": "help",
                "refused": True,
                "reason": "visual_evidence_missing",
                "request_id": None,
            }
            intent = "help"
            bucket = None
        elif intent == "generate":
            resolved_router_content_type = "auto"
            intent_resolution = resolve_content_type(
                requested_content_type=resolved_router_content_type,
                natural_language_request=request.message,
                content_type_source="auto_inferred",
                content_type_user_selected=False,
            )
            referenced_scope_commit = (
                commit_referenced_image_scope(
                    session,
                    selected,
                    use_kind="generate_content_from_images",
                    confidence=str(
                        router_decision.get("confidence") or ""
                    ),
                )
            )
            response = run_or_http_error(
                course_generate_payload,
                selected,
                {
                    **router_arguments,
                    "content_type": resolved_router_content_type,
                    "resolved_content_type": intent_resolution[
                        "resolved_content_type"
                    ],
                    "content_type_source": intent_resolution[
                        "content_type_source"
                    ],
                    "content_type_user_selected": False,
                    "target_length": router_arguments.get("target_length"),
                    "target_length_source": str(
                        router_arguments.get("target_length_source")
                        or "profile_default"
                    ),
                    "style": str(
                        router_arguments.get("style") or "自然"
                    ),
                    "audience": str(
                        router_arguments.get("audience") or "普通读者"
                    ),
                    "organization": str(
                        router_arguments.get("organization")
                        or "input_order"
                    ),
                    "importance": list(
                        router_arguments.get("importance") or []
                    ),
                    "natural_language_request": request.message,
                    "intent_resolution": intent_resolution,
                    "call_source": "chat_tool_call",
                },
            )
            bucket = "generated_content"
        elif (
            routed_tool == "compare_or_rank_images"
            or intent == "rank"
        ):
            if routed_tool == "compare_or_rank_images":
                reference_resolution = (
                    router_reference_resolution
                    or resolve_image_references(
                        request.message,
                        session,
                    )
                )
            compare_action = str(
                router_arguments.get("action")
                or ("rank" if intent == "rank" else "compare")
            )
            if compare_action not in {"compare", "select", "rank"}:
                compare_action = (
                    "select"
                    if intent == "recommend"
                    else "rank"
                    if intent == "rank"
                    else "compare"
                )
            criterion = str(
                router_arguments.get("criterion") or ""
            ).strip()
            if (
                compare_action in {"select", "rank"}
                and not criterion
            ):
                response = {
                    "status": "clarification_required",
                    "answer": (
                        "你更看重哪种标准？例如旅游感、氛围感，"
                        "或是否适合做朋友圈首图。"
                    ),
                    "intent": intent,
                    "action_completed": False,
                    "needs_clarification": True,
                    "model_called": False,
                    "request_id": None,
                }
            else:
                selected_labels = [
                    str(item.get("image_label"))
                    for item in selected
                    if item.get("image_label")
                ]
                referenced_scope_commit = (
                    commit_referenced_image_scope(
                        session,
                        selected,
                        use_kind="compare_or_rank_images",
                        confidence=str(
                            router_decision.get(
                                "confidence"
                            )
                            or ""
                        ),
                    )
                )
                binding_validation = (
                    referenced_scope_commit.get(
                        "binding_validation"
                    )
                    if referenced_scope_commit
                    else None
                )
                response = run_or_http_error(
                    course_rank_payload,
                    selected,
                    criterion or request.message,
                    action=compare_action,
                    scenario=str(router_arguments.get("scenario") or ""),
                    select_count=int(router_arguments.get("select_count") or 1),
                    original_user_request=request.message,
                    call_source="chat_tool_call",
                )
                if reference_resolution is not None:
                    response["reference_resolution"] = (
                        reference_resolution
                    )
                if binding_validation is not None:
                    response["canonical_binding_trace"] = (
                        binding_validation
                    )
                    response["inherited_task_frame"] = (
                        conversation_follow_up.get(
                            "inherited_task_frame"
                        )
                        if conversation_follow_up
                        else None
                    )
            bucket = "ranking_results"
        else:
            routed_reference_assets = (
                bool(routed_image_refs)
                and any(
                    label.startswith("SEARCH_")
                    for label in routed_image_refs
                )
            )
            if (
                conversation_follow_up
                and conversation_follow_up.get("bound")
            ):
                follow_up_labels = list(
                    conversation_follow_up[
                        "selected_image_labels"
                    ]
                )
                reference_resolution = {
                    "raw_question": request.message,
                    "original_user_message": request.message,
                    "resolved_question": str(
                        conversation_follow_up.get(
                            "standalone_request"
                        )
                        or request.message
                    ),
                    "selected_image_labels": follow_up_labels,
                    "resolved_image_refs": follow_up_labels,
                    "selection_reasons": {
                        label: [
                            "general_conversation_follow_up:"
                            + str(
                                conversation_follow_up[
                                    "follow_up_type"
                                ]
                            )
                        ]
                        for label in follow_up_labels
                    },
                    "active_image_labels": [
                        str(item["image_label"])
                        for item in session.get(
                            "active_assets", []
                        )
                        if item.get("image_label")
                    ],
                    "newly_added_labels": [],
                    "current_focus_label": session[
                        "chat_state"
                    ].get("current_focus_label"),
                    "explicit_image_scope": (
                        "explicit_single"
                        if len(follow_up_labels) == 1
                        and conversation_follow_up[
                            "follow_up_type"
                        ]
                        in {
                            "switch_image_target",
                            "ask_about_alternative_image",
                        }
                        else "previous_turn"
                    ),
                    "focus_before_resolution": session[
                        "chat_state"
                    ].get("current_focus_label"),
                    "focus_applied": False,
                    "scope_resolution_reason": (
                        "general_conversation_follow_up:"
                        + str(
                            conversation_follow_up[
                                "follow_up_type"
                            ]
                        )
                    ),
                    "requires_clarification": False,
                    "clarification": None,
                    "resolution_errors": [],
                }
            elif decision_follow_up and decision_follow_up.get("bound"):
                follow_up_labels = list(
                    decision_follow_up["selected_image_labels"]
                )
                reference_resolution = {
                    "raw_question": request.message,
                    "original_user_message": request.message,
                    "resolved_question": request.message,
                    "selected_image_labels": follow_up_labels,
                    "resolved_image_refs": follow_up_labels,
                    "selection_reasons": {
                        label: [
                            "immediate_previous_decision_follow_up"
                        ]
                        for label in follow_up_labels
                    },
                    "active_image_labels": [
                        str(item["image_label"])
                        for item in session.get("active_assets", [])
                        if item.get("image_label")
                    ],
                    "newly_added_labels": [],
                    "current_focus_label": session[
                        "chat_state"
                    ].get("current_focus_label"),
                    "explicit_image_scope": (
                        "explicit_single"
                        if decision_follow_up["follow_up_type"]
                        == "alternative_image"
                        and len(follow_up_labels) == 1
                        else "previous_decision"
                    ),
                    "focus_before_resolution": session[
                        "chat_state"
                    ].get("current_focus_label"),
                    "focus_applied": False,
                    "scope_resolution_reason": (
                        "immediate_previous_decision:"
                        + str(decision_follow_up["follow_up_type"])
                    ),
                    "requires_clarification": False,
                    "clarification": None,
                    "resolution_errors": [],
                }
            else:
                reference_resolution = (
                {
                    "raw_question": request.message,
                    "resolved_question": request.message,
                    "selected_image_labels": routed_image_refs,
                    "selection_reasons": {
                        label: ["stable_tool_result_reference"]
                        for label in routed_image_refs
                    },
                    "active_image_labels": [
                        *[
                            str(item["image_label"])
                            for item in session.get("active_assets", [])
                            if item.get("image_label")
                        ],
                        *routed_image_refs,
                    ],
                    "newly_added_labels": [],
                    "current_focus_label": None,
                    "requires_clarification": False,
                    "clarification": None,
                    "resolution_errors": [],
                }
                if routed_reference_assets
                else resolve_image_references(request.message, session)
                )
            if reference_resolution["requires_clarification"]:
                selected = []
                response = {
                    "status": "clarification_required",
                    "intent": intent,
                    "answer": {
                        "answer": reference_resolution["clarification"],
                        "image_references": [],
                        "evidence": [],
                        "uncertainty": reference_resolution["resolution_errors"],
                        "answer_source": "deterministic_reference_clarification",
                    },
                    "model_contract_valid": None,
                    "product_contract_valid": True,
                    "fallback_applied": False,
                    "fallback_source": None,
                    "contract_errors": [],
                    "reference_resolution": reference_resolution,
                    "request_id": None,
                    "model_called": False,
                }
            elif not routed_reference_assets:
                selected = run_or_http_error(
                    resolve_session_assets_by_label,
                    session,
                    reference_resolution["selected_image_labels"],
                )
            if (
                not reference_resolution["requires_clarification"]
                and intent == "compare"
                and len(selected) < 2
            ):
                response = {
                    "status": "refused",
                    "answer": {
                        "answer": "多图比较至少需要 2 张明确相关的图片，请指定或补充 IMG 标签。",
                        "image_references": [],
                        "evidence": [],
                        "uncertainty": ["compare_requires_at_least_2_resolved_images"],
                        "answer_source": "deterministic_reference_clarification",
                    },
                    "model_contract_valid": None,
                    "product_contract_valid": True,
                    "fallback_applied": False,
                    "fallback_source": None,
                    "contract_errors": [],
                    "reference_resolution": reference_resolution,
                    "request_id": None,
                    "model_called": False,
                }
            elif not reference_resolution["requires_clarification"]:
                referenced_scope_commit = (
                    commit_referenced_image_scope(
                        session,
                        selected,
                        use_kind=str(intent),
                    )
                )
                context_plan = build_multiturn_context(
                    session,
                    reference_resolution,
                    selected,
                    decision_follow_up=decision_follow_up,
                    conversation_follow_up=conversation_follow_up,
                )
                state = context_plan["state_variables"]
                projection = context_projection(
                    session,
                    original_user_message=request.message,
                    dialogue_act=(
                        str(
                            conversation_follow_up.get(
                                "follow_up_type"
                            )
                        )
                        if conversation_follow_up
                        else None
                    ),
                    referenced_turn_id=(
                        str(
                            conversation_follow_up.get(
                                "referenced_turn_id"
                            )
                        )
                        if conversation_follow_up
                        and conversation_follow_up.get(
                            "referenced_turn_id"
                        )
                        else None
                    ),
                    current_image_scope=list(
                        reference_resolution[
                            "selected_image_labels"
                        ]
                    ),
                    standalone_request=(
                        str(
                            conversation_follow_up.get(
                                "standalone_request"
                            )
                        )
                        if conversation_follow_up
                        and conversation_follow_up.get("bound")
                        else None
                    ),
                )
                selected_transport_labels = [
                    str(item["image_label"])
                    for item in selected
                ]
                if selected_transport_labels and all(
                    label.startswith("IMG_")
                    for label in selected_transport_labels
                ):
                    binding_validation = validate_model_bindings(
                        session,
                        selected,
                        selected_transport_labels,
                    )
                else:
                    binding_validation = {
                        "canonical_image_bindings": session[
                            "chat_state"
                        ].get("canonical_image_bindings"),
                        "binding_snapshot_sha256": session[
                            "chat_state"
                        ].get(
                            "canonical_image_bindings", {}
                        ).get("binding_snapshot_sha256"),
                        "actual_asset_ids": [
                            str(item.get("asset_id") or "")
                            for item in selected
                        ],
                        "actual_image_sha256": [
                            str(item.get("sha256") or "")
                            for item in selected
                        ],
                        "image_block_order": [
                            {
                                "image_label": str(
                                    item.get("image_label") or ""
                                ),
                                "asset_id": str(
                                    item.get("asset_id") or ""
                                ),
                                "sha256": str(
                                    item.get("sha256") or ""
                                ),
                            }
                            for item in selected
                        ],
                        "model_send_order": (
                            selected_transport_labels
                        ),
                    }
                validation_bindings = list(
                    session["chat_state"]["asset_bindings"]
                )
                known_validation_labels = {
                    str(item.get("image_label"))
                    for item in validation_bindings
                }
                validation_bindings.extend(
                    item
                    for item in selected
                    if str(item.get("image_label"))
                    not in known_validation_labels
                )
                system_prompt, policy_identity = conversational_response_candidate.system_prompt()
                current_prompt, identity = conversational_response_candidate.render(
                    "multiturn_chat",
                    {
                        "RAW_QUESTION": reference_resolution["raw_question"],
                        "RESOLVED_QUESTION": reference_resolution["resolved_question"],
                        "REQUIRED_IMAGES": reference_resolution["selected_image_labels"],
                        "SELECTED_ASSET_CONTEXT": context_plan["selected_asset_context"],
                        "CURRENT_STATE": state,
                    },
                )
                current_prompt = (
                    current_prompt
                    + "\n\n<context_projection>\n"
                    + render_context_projection(projection)
                    + "\n</context_projection>"
                )
                image_map = {
                    item["image_label"]: {
                        "ref": item["ref"],
                        "asset_id": item["asset_id"],
                        "sha256": item["sha256"],
                        "source": item["source"],
                    }
                    for item in selected
                }
                legacy_decision_follow_up_trace = decision_follow_up
                if (
                    legacy_decision_follow_up_trace is None
                    and conversation_follow_up
                    and conversation_follow_up.get("bound")
                    and str(
                        conversation_follow_up.get(
                            "inherited_task_type"
                        )
                    )
                    in {
                        "compare",
                        "select",
                        "rank",
                        "recommend",
                        "compare_or_rank_images",
                    }
                ):
                    legacy_type = {
                        "explain_previous_decision": "explain_reason",
                        "elaborate_previous_answer": "expand_reason",
                        "ask_about_alternative_image": "alternative_image",
                        "ask_about_remaining_images": (
                            "rejected_alternatives"
                        ),
                    }.get(
                        str(
                            conversation_follow_up.get(
                                "follow_up_type"
                            )
                        )
                    )
                    if legacy_type:
                        legacy_decision_follow_up_trace = {
                            "detected": True,
                            "bound": True,
                            "follow_up_type": legacy_type,
                            "selected_image_labels": list(
                                conversation_follow_up.get(
                                    "selected_image_labels", []
                                )
                            ),
                            "requires_clarification": False,
                            "compatibility_source": (
                                "phase5_4h_general_follow_up"
                            ),
                        }
                context_trace = {
                    "raw_question": reference_resolution["raw_question"],
                    "original_user_message": reference_resolution[
                        "raw_question"
                    ],
                    "resolved_question": reference_resolution["resolved_question"],
                    "all_active_image_labels": reference_resolution["active_image_labels"],
                    "selected_image_labels": reference_resolution["selected_image_labels"],
                    "resolved_image_refs": reference_resolution[
                        "selected_image_labels"
                    ],
                    "selection_reasons": reference_resolution["selection_reasons"],
                    "explicit_image_scope": reference_resolution.get(
                        "explicit_image_scope"
                    ),
                    "focus_before_resolution": reference_resolution.get(
                        "focus_before_resolution"
                    ),
                    "focus_applied": bool(
                        reference_resolution.get("focus_applied")
                    ),
                    "scope_resolution_reason": reference_resolution.get(
                        "scope_resolution_reason"
                    ),
                    "actual_images_sent_to_model": [
                        str(item["image_label"]) for item in selected
                    ],
                    "decision_follow_up": (
                        legacy_decision_follow_up_trace
                    ),
                    "conversation_follow_up": (
                        conversation_follow_up
                    ),
                    "rewritten_standalone_request": (
                        conversation_follow_up.get(
                            "standalone_request"
                        )
                        if conversation_follow_up
                        and conversation_follow_up.get("bound")
                        else None
                    ),
                    "backend_image_map": image_map,
                    "canonical_image_bindings": (
                        binding_validation[
                            "canonical_image_bindings"
                        ]
                    ),
                    "binding_snapshot_sha256": (
                        binding_validation[
                            "binding_snapshot_sha256"
                        ]
                    ),
                    "actual_asset_ids": binding_validation[
                        "actual_asset_ids"
                    ],
                    "actual_image_sha256": binding_validation[
                        "actual_image_sha256"
                    ],
                    "image_block_order": binding_validation[
                        "image_block_order"
                    ],
                    "model_send_order": binding_validation[
                        "model_send_order"
                    ],
                    "context_projection_entry_ids": projection[
                        "context_projection_entry_ids"
                    ],
                    "compaction_entry_id": projection[
                        "compaction_entry_id"
                    ],
                    "original_images_sent": [
                        {
                            "image_label": item["image_label"],
                            "sha256": item["sha256"],
                            "source": item["source"],
                        }
                        for item in selected
                    ],
                    "visual_model_called": True,
                    "verbatim_user_question_forwarded": True,
                    "summary_used": bool(
                        context_plan["conversation_summary"].get("confirmed_facts")
                        or context_plan["conversation_summary"].get("asset_notes")
                    ),
                    "recent_complete_pair_count": context_plan["recent_complete_pair_count"],
                    "pruned_complete_pair_count": context_plan["pruned_complete_pair_count"],
                    "history_messages_sent": context_plan["recent_messages"],
                    "current_turn_state": state,
                    "message_organization": "system + bounded role history + current IMG_n visual user turn",
                    "visual_id_strategy": "explicit stable IMG_n label adjacent to each visual block",
                    "policy_identity": policy_identity,
                    "final_messages_contract": [
                        "system: stable conversation policy and frozen behavior examples",
                        "user/assistant: clean bounded public history only",
                        "user: adjacent IMG_n visual blocks plus explicit current state and current task",
                    ],
                }
                result: dict[str, Any] = {
                    "status": "failed",
                    "data": {"raw_output": None, "parsed_output": None},
                    "error": None,
                }
                finished: dict[str, Any] | None = None
                attempts: list[dict[str, Any]] = []
                common_answer: dict[str, Any] | None = None
                model_errors: list[str] = []
                level = "fallback"
                try:
                    result, finished = multiturn_chat_call(
                        identity=identity,
                        system_prompt=system_prompt,
                        history_messages=context_plan["recent_messages"],
                        current_prompt=current_prompt,
                        assets=selected,
                        context_trace=context_trace,
                        max_new_tokens=512,
                    )
                    common_answer, model_errors, repaired = normalize_conversational_payload(
                        conversational_model_payload(result),
                        state=state,
                        assets=selected,
                        all_bindings=validation_bindings,
                    )
                    level = "deterministic_repair" if common_answer is not None and repaired else "direct"
                    attempts.append({"level": level, "passed": common_answer is not None, "errors": model_errors})
                except Exception as exc:
                    result["error"] = f"initial_model_call_failed:{type(exc).__name__}"
                    model_errors = [result["error"]]
                    attempts.append({"level": "direct", "passed": False, "errors": model_errors})

                raw_text = str(result["data"].get("raw_output") or "")
                semantic_candidate = any(
                    token in raw_text for token in ("推荐", "更适合", "区别", "差异", "排序", "IMG", "第")
                )
                unsafe = any(
                    error.startswith(
                        ("unknown_", "backend_identity_leak", "unverified_text_claim", "internal_language_leak")
                    )
                    for error in model_errors
                )
                if common_answer is None and semantic_candidate and not unsafe:
                    repair_prompt, repair_identity = conversational_response_candidate.render(
                        "contract_repair",
                        {"CURRENT_STATE": state, "RAW_ANSWER": raw_text},
                    )
                    try:
                        repair_result, _ = conversation_repair_call(
                            prompt=repair_prompt,
                            identity=repair_identity,
                        )
                        common_answer, model_errors, _ = normalize_conversational_payload(
                            conversational_model_payload(repair_result),
                            state=state,
                            assets=selected,
                            all_bindings=validation_bindings,
                        )
                        attempts.append(
                            {"level": "constrained_repair", "passed": common_answer is not None, "errors": model_errors}
                        )
                        if common_answer is not None:
                            level = "constrained_repair"
                    except Exception as exc:
                        model_errors = [*model_errors, f"constrained_repair_failed:{type(exc).__name__}"]
                        attempts.append({"level": "constrained_repair", "passed": False, "errors": model_errors})

                if common_answer is None and result.get("error") is None:
                    retry_prompt, retry_identity = conversational_response_candidate.render(
                        "task_retry",
                        {
                            "CONVERSATION_POLICY": system_prompt,
                            "CURRENT_STATE": state,
                            "SAFE_FACTS": safe_asset_facts(selected),
                            "CURRENT_TASK": {
                                "raw_question": reference_resolution["raw_question"],
                                "resolved_question": reference_resolution["resolved_question"],
                                "required_images": reference_resolution["selected_image_labels"],
                                "required_action": state["requested_action"],
                                "criterion": state["comparison_criterion"],
                            },
                        },
                    )
                    try:
                        retry_result, _ = multiturn_chat_call(
                            identity=retry_identity,
                            system_prompt=system_prompt,
                            history_messages=[],
                            current_prompt=retry_prompt,
                            assets=selected,
                            context_trace={**context_trace, "retry": "task_preserving_no_history"},
                            max_new_tokens=512,
                        )
                        common_answer, model_errors, _ = normalize_conversational_payload(
                            conversational_model_payload(retry_result),
                            state=state,
                            assets=selected,
                            all_bindings=validation_bindings,
                        )
                        attempts.append(
                            {"level": "task_preserving_retry", "passed": common_answer is not None, "errors": model_errors}
                        )
                        if common_answer is not None:
                            level = "task_preserving_retry"
                    except Exception as exc:
                        model_errors = [*model_errors, f"task_retry_failed:{type(exc).__name__}"]
                        attempts.append({"level": "task_preserving_retry", "passed": False, "errors": model_errors})

                if common_answer is None:
                    common_answer = task_preserving_fallback(selected, state=state)
                    level = "fallback"
                    attempts.append({"level": "fallback", "passed": True, "errors": model_errors})
                normalized_answer = attach_public_assets(
                    common_answer,
                    selected,
                    answer_source=f"phase5_2c_{level}",
                )
                model_contract_valid = level != "fallback"
                annotated = (
                    traces.annotate(
                        finished["request_id"],
                        raw_model_output=result["data"].get("raw_output"),
                        parsed_model_output=result["data"].get("parsed_output"),
                        model_contract_valid=model_contract_valid,
                        model_contract_errors=model_errors,
                        repair_level=level,
                        repair_attempts=attempts,
                        fallback_applied=level == "fallback",
                        fallback_grounding_source=(
                            "p3_safe_facts_auxiliary"
                            if level == "fallback"
                            else None
                        ),
                        visual_model_called=True,
                        original_images_sent=[
                            {
                                "image_label": item["image_label"],
                                "sha256": item["sha256"],
                            }
                            for item in selected
                        ],
                        original_user_message=reference_resolution[
                            "raw_question"
                        ],
                        explicit_image_scope=reference_resolution.get(
                            "explicit_image_scope"
                        ),
                        resolved_image_refs=reference_resolution[
                            "selected_image_labels"
                        ],
                        focus_before_resolution=reference_resolution.get(
                            "focus_before_resolution"
                        ),
                        focus_applied=bool(
                            reference_resolution.get("focus_applied")
                        ),
                        scope_resolution_reason=reference_resolution.get(
                            "scope_resolution_reason"
                        ),
                        actual_images_sent_to_model=[
                            str(item["image_label"])
                            for item in selected
                        ],
                        decision_follow_up=decision_follow_up,
                        router_action=router_decision.get("action"),
                        router_tool_name=router_decision.get("tool_name"),
                        extracted_action=state["requested_action"],
                        extracted_k=state.get("selection_count"),
                        forwarded_user_instruction=reference_resolution[
                            "raw_question"
                        ],
                        clarification_required=bool(
                            common_answer.get("needs_clarification")
                        ),
                        clarification_reason=(
                            "model_or_reference_requires_clarification"
                            if common_answer.get("needs_clarification")
                            else None
                        ),
                        model_messages_summary={
                            "prompt_id": identity["prompt_id"],
                            "prompt_sha256": identity[
                                "prompt_sha256"
                            ],
                            "forwarded_instruction_sha256": (
                                hashlib.sha256(
                                    reference_resolution[
                                        "raw_question"
                                    ].encode("utf-8")
                                ).hexdigest()
                            ),
                            "image_count": len(selected),
                        },
                    )
                    if finished is not None
                    else None
                )
                response = {
                    "status": result["status"] if model_contract_valid else "partial_success",
                    "intent": intent,
                    "result": result,
                    "answer": normalized_answer,
                    "model_contract_valid": model_contract_valid,
                    "product_contract_valid": True,
                    "fallback_applied": level == "fallback",
                    "fallback_source": None if model_contract_valid else "task_preserving_safe_fallback",
                    "repair_level": level,
                    "repair_attempts": attempts,
                    "contract_errors": model_errors,
                    "prompt_candidate": identity,
                    "conversation_policy": policy_identity,
                    "context_plan": context_plan,
                    "reference_resolution": reference_resolution,
                    "context_trace": context_trace,
                    "model_called": True,
                    "request_id": finished["request_id"] if finished else None,
                    "trace": annotated,
                }
            bucket = "comparison_results" if intent in {"compare", "recommend"} else None

        if referenced_scope_commit is not None:
            response["referenced_scope_commit"] = (
                referenced_scope_commit
            )
        response["display_text"] = extract_display_text(response)
        response["tool_router"] = {
            "action": router_decision.get("action"),
            "tool_name": router_decision.get("tool_name"),
            "arguments": router_decision.get("arguments"),
            "steps": router_decision.get("steps"),
            "reason_code": router_decision.get("reason_code"),
            "confidence": router_decision.get("confidence"),
            "model_called": router_decision.get("model_called"),
            "model_agreed": router_decision.get("model_agreed"),
            "model_contract_errors": router_decision.get(
                "model_contract_errors"
            ),
            "router_trace_id": router_decision.get("router_trace_id"),
            "candidate": router_decision.get("candidate"),
            "system_utility": router_decision.get("system_utility"),
        }
        chat_state = session["chat_state"]
        if routed_tool:
            chat_state["last_tool_call"] = {
                "tool_name": routed_tool,
                "arguments": router_arguments,
                "step_count": len(
                    list(router_decision.get("steps") or [])
                ),
                "status": response.get("status"),
                "created_at": now_iso(),
            }
            if routed_tool != "search_images":
                chat_state["selected_tool_images"] = [
                    str(item.get("image_label"))
                    for item in selected
                    if item.get("image_label")
                ]
            chat_state["current_tool_goal"] = request.message.strip()
            chat_state["pending_tool_action"] = None
            chat_state["tool_error"] = (
                None
                if response.get("status")
                not in {
                    "failed",
                    "tool_failed",
                    "refused",
                    "clarification_required",
                }
                else str(
                    response.get("tool_error")
                    or response.get("reason")
                    or response.get("answer")
                    or response.get("status")
                )
            )
            chat_state["tool_trace_id"] = (
                response.get("request_id")
                or router_decision.get("router_trace_id")
            )
        session["chat_state"] = chat_state
        turn_assets = selected
        if intent in {
            "system_utility",
            "direct_chat",
            "unsupported_non_visual",
            "text_chat",
            "text_transform",
            "help",
            "clarification",
        }:
            turn_assets = []
        if (
            routed_tool == "search_images"
            and str(router_arguments.get("mode") or "auto") == "text"
        ):
            turn_assets = []
        saved = save_course_turn(
            session,
            question=request.message,
            intent=routed_tool or intent,
            assets=turn_assets,
            response=response,
            bucket=bucket,
            reference_resolution=reference_resolution,
            decision_follow_up=decision_follow_up,
            conversation_follow_up=conversation_follow_up,
        )
        return {
            "conversation_id": saved["conversation_id"],
            "intent": intent,
            "tool_router": response["tool_router"],
            "active_assets": saved["active_assets"],
            "response": response,
            "display_text": response["display_text"],
            "messages": saved["messages"],
            "context_summary": saved["context_summary"],
            "tool_calls": saved["tool_calls"],
            "tool_results": saved["tool_results"],
            "chat_state": saved["chat_state"],
            "request_id": response.get("request_id"),
        }

    def facts_for(image_id: str) -> dict[str, Any]:
        try:
            return orchestrator.current_facts(image_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown_image:{image_id}") from None

    @app.post("/compare")
    def compare(request: ComparisonRequest) -> dict[str, Any]:
        rows = [{"asset_id": image_id, "facts": facts_for(image_id)} for image_id in request.asset_ids]
        trace = traces.start("compare", request.asset_ids, model=vlm_service.status().get("model"), model_revision=vlm_service.status().get("model_revision"), prompt_version="phase1_compare_v1", schema_version="phase5_comparison_v2", services=["library", "vlm"])
        try:
            paths = [library.image_path(image_id, verify_hash=True) for image_id in request.asset_ids]
            result = vlm_service.compare_images(paths, request.instruction)
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            finished = traces.finish(trace, status="success")
            summary = result.data.get("parsed_output") or result.data.get("raw_output")
            return {
                "asset_ids": request.asset_ids,
                "instruction": request.instruction,
                "dimensions": request.dimensions,
                "rows": rows,
                "comparison": result.as_dict(),
                "summary": summary,
                "display_text": extract_display_text(
                    {"answer": summary},
                    fallback="图片比较已完成；详细结构保留在技术记录中。",
                ),
                "baseline": "real_multi_image_vlm",
                "request_id": finished["request_id"],
                "trace": finished,
            }
        except Exception as exc:
            traces.finish(trace, status="failed", error=exc)
            return run_or_http_error(lambda: (_ for _ in ()).throw(exc))

    @app.post("/rank")
    def rank(request: RankingRequest) -> dict[str, Any]:
        trace = traces.start("rank", request.asset_ids, model=vlm_service.status().get("model"), model_revision=vlm_service.status().get("model_revision"), prompt_version="phase1_compare_v1", schema_version="phase5_ranking_v2", services=["library", "vlm"])
        try:
            paths = [library.image_path(image_id, verify_hash=True) for image_id in request.asset_ids]
            result = vlm_service.compare_images(paths, f"RANK_CONTRACT: {request.instruction}")
            if result.status != "success":
                raise RuntimeError(result.error or result.status)
            parsed = result.data.get("parsed_output") or {}
            ranking = parsed.get("ranking") if isinstance(parsed, dict) else None
            if isinstance(ranking, dict):
                ranking = [ranking]
            ranking_ids = [item.get("asset_id") for item in ranking] if isinstance(ranking, list) and all(isinstance(item, dict) for item in ranking) else []
            ranks = [item.get("rank") for item in ranking] if isinstance(ranking, list) and all(isinstance(item, dict) for item in ranking) else []
            if (
                not isinstance(ranking, list)
                or len(ranking) != len(request.asset_ids)
                or len(set(ranking_ids)) != len(request.asset_ids)
                or set(ranking_ids) != set(request.asset_ids)
                or set(ranks) != set(range(1, len(request.asset_ids) + 1))
            ):
                raise HTTPException(
                    status_code=502,
                    detail=f"ranking_asset_contract_violation: expected={request.asset_ids}, got_ids={ranking_ids}, got_ranks={ranks}",
                )
            ranking = sorted(ranking, key=lambda item: item["rank"])
            for item in ranking:
                facts = facts_for(item["asset_id"])
                observation = facts.get("global_observation") or facts.get("global_scene") or facts.get("subjects") or "已记录核心视觉事实"
                if not item.get("reason"):
                    item["reason"] = f"依据当前核心事实：{observation}"
                    item["reason_source"] = "current_core_facts"
            finished = traces.finish(trace, status="success")
            ranking_text = "；".join(
                f"{item['rank']}. {item['asset_id']}：{item.get('reason') or '未提供理由'}"
                for item in ranking
            )
            return {
                "asset_ids": request.asset_ids,
                "instruction": request.instruction,
                "ranking": ranking,
                "display_text": ranking_text,
                "result": result.as_dict(),
                "baseline": "real_multi_image_vlm",
                "request_id": finished["request_id"],
                "trace": finished,
            }
        except Exception as exc:
            traces.finish(trace, status="failed", error=exc)
            return run_or_http_error(lambda: (_ for _ in ()).throw(exc))

    @app.post("/feedback")
    def add_feedback(request: FeedbackRequest) -> dict[str, Any]:
        item = product.add_feedback(request.model_dump())
        return {"status": "queued_for_human_review", "feedback": item, "training_applied": False}

    @app.get("/feedback")
    def feedback() -> dict[str, Any]:
        items = product.feedback()
        return {"count": len(items), "items": items, "automatic_training": False}

    @app.patch("/feedback/{feedback_id}/{status}")
    def review_feedback(feedback_id: str, status: Literal["accepted", "rejected", "queued"]) -> dict[str, Any]:
        return run_or_http_error(product.update_feedback, feedback_id, status)

    @app.get("/training-queue")
    def training_queue() -> dict[str, Any]:
        category = {
            "description": "candidate_description_correction",
            "ocr": "candidate_ocr_correction",
            "vqa": "candidate_vqa_correction",
            "search": "candidate_search_preference",
            "compare": "candidate_comparison_preference",
            "generation": "candidate_generation_preference",
        }
        items = []
        for feedback_item in product.feedback():
            function_id = str(feedback_item.get("function_id", ""))
            queue_type = next((value for key, value in category.items() if key in function_id), "candidate_general_review")
            items.append({**feedback_item, "queue_type": queue_type, "train_only": True, "sensitive_information_review": "pending", "automatic_training": False})
        return {"count": len(items), "items": items, "automatic_training": False, "gold_write": False}

    @app.get("/history")
    def history() -> dict[str, Any]:
        trace_root = settings.run_root / "traces"
        items = []
        for path in sorted(trace_root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:100]:
            try:
                items.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return {"count": len(items), "items": items}

    @app.get("/exports/{kind}")
    def export_records(kind: Literal["history", "feedback", "tasks"], format: Literal["json", "markdown"] = "json") -> Any:
        if kind == "history":
            values = history()["items"]
        elif kind == "feedback":
            values = product.feedback()
        else:
            values = product.tasks()
        if format == "json":
            return {"kind": kind, "count": len(values), "items": values, "exported_at": now_iso()}
        lines = [f"# SceneMind-X {kind.title()} Export", "", f"Exported: {now_iso()}", ""]
        for item in values:
            lines.extend([f"## {item.get('request_id') or item.get('feedback_id') or item.get('task_id') or 'record'}", "", "```json", json.dumps(item, ensure_ascii=False, indent=2), "```", ""])
        return PlainTextResponse("\n".join(lines), media_type="text/markdown; charset=utf-8")

    @app.get("/system/status")
    def system_status() -> dict[str, Any]:
        runtime = cached_runtime_status()
        retrieval_status = runtime["retrieval"]
        active_retriever = (
            retrieval_status.get("active_backend")
            or retrieval_status.get("retrieval_backend")
            or ("disabled" if provider_manager.mode == "no_model" else "not_ready")
        )
        if active_retriever == "bailian_cloud_e1":
            retriever_label = "qwen3-vl-embedding@2560 + 独立云 Faiss"
            retrieval_limitation = (
                "当前公开版云端检索覆盖已进入独立索引的"
                "Default/用户资产；课程 Train/Validation 图片不随公开版提供。"
            )
        elif active_retriever == "e1":
            retriever_label = "Qwen3-VL-Embedding-2B + Faiss"
            retrieval_limitation = (
                "E1 是冻结预训练多模态 Embedding + CPU Faiss；"
                "未接入 R2/R3/R4/E2，也不是训练后的 Reranker。"
            )
        elif active_retriever == "disabled":
            retriever_label = "未接入模型"
            retrieval_limitation = "当前未接入模型，新的实时检索已禁用。"
        elif retrieval_status.get("fallback_active"):
            retriever_label = "Fallback: color-grid-v1"
            retrieval_limitation = (
                "当前检索使用 R0 color-grid-v1 回退，"
                "不代表已完成 Retriever/Reranker 训练。"
            )
        else:
            retriever_label = "检索尚未就绪"
            retrieval_limitation = "当前 Embedding 或匹配索引尚未就绪。"
        return {
            "api": "ready",
            "database": product.status()["storage"],
            "vlm": runtime["vlm"],
            "retrieval": retrieval_status,
            "system_libraries": {
                "items": system_libraries.libraries(),
                "e1_indices": (
                    active_split_retrieval().status()
                    if active_split_retrieval() is not None
                    else {"status": "disabled"}
                ),
            },
            "product": product.status(),
            "known_limitations": [retrieval_limitation],
            "default_retriever": active_retriever,
            "retriever_label": retriever_label,
            "production_prompt": "NATURAL_CHINESE_DETAILED_DESCRIPTION_V1",
            "frozen_prompt_suite": "SCENEMINDX_PROMPT_SUITE_V1",
            "course_prompt_candidate": course_candidate.identity("phase5_2_multimodal_chat_v1"),
            "multiturn_chat_candidate": {
                **conversational_response_candidate.identity("multiturn_chat"),
                "default_chat_route": True,
                "promotion_decision": "CONVERSATIONAL_RESPONSE_V1_ACCEPTED",
            },
            "conversation_policy_candidate": conversational_response_candidate.identity(
                "conversation_policy"
            ),
            "compare_rank_candidate": {
                **conversational_response_candidate.identity("compare_rank"),
                "default_compare_rank_route": True,
            },
            "multi_image_content_candidate": multi_image_content_candidate.identity(),
            "multi_image_story_candidate": multi_image_story_candidate.identity(),
            "chat_tool_router_candidate": chat_tool_router_candidate.identity(),
            "direct_chat_candidate": chat_tool_router_candidate.identity(
                "direct_chat"
            ),
            "content_profile_registry": content_profile_registry.status(),
            "content_length_profiles": public_content_length_config(),
            "chat_business_tools": {
                "whitelist": [
                    "generate_content_from_images",
                    "search_images",
                    "compare_or_rank_images",
                ],
                "search_shape": "one_tool_with_mode",
                "max_business_steps": 2,
            },
            "multi_image_intent_resolution": {
                "default": "auto",
                "explicit_selection_precedence": True,
                "natural_language_over_untouched_default": True,
            },
            "user_text_output": {
                "public_field": "display_text",
                "main_region_accepts_objects": False,
                "technical_details_default_collapsed": True,
                "story_chat_route_uses_auto_intent": True,
            },
            "conversation_delete": {
                "api": "DELETE /conversations/{conversation_id}",
                "physical_session_removal": True,
                "library_assets_modified": False,
                "retrieval_index_modified": False,
            },
            "course_phase": "PHASE 6.1 FULL TRAIN/VAL CANONICAL LABELING AND MULTI-LIBRARY",
            "training": False,
            "val_read": True,
            "test_blind_read": False,
            "external_api_used": False,
        }

    return app
