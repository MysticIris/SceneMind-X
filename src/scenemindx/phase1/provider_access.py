"""Phase 7.0 provider selection, capability state and safe error contracts.

This module deliberately keeps credentials out of the persisted provider
selection. User-supplied keys live only in ``ProviderManager`` process memory;
the course-demo key is loaded only when a Bailian adapter needs it.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Iterator, Literal

from scenemindx.services.cloud_image_transport import (
    CloudImageTransportPreprocessor,
    PreparedCloudImageBatch,
)


ProviderMode = Literal["no_model", "bailian", "local", "self_hosted"]
CloudTier = Literal["standard", "high_quality"]
CredentialSource = Literal["user_session", "course_default"]


class ProviderState(str, Enum):
    """Represent provider state data."""
    UNCONFIGURED = "UNCONFIGURED"
    NO_MODEL = "NO_MODEL"
    CONNECTING = "CONNECTING"
    RECONNECTING = "RECONNECTING"
    READY = "READY"
    PARTIAL_READY = "PARTIAL_READY"
    STALE = "STALE"
    OFFLINE = "OFFLINE"
    INVALID_KEY = "INVALID_KEY"
    BILLING_BLOCKED = "BILLING_BLOCKED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    REGION_ENDPOINT_MISMATCH = "REGION_ENDPOINT_MISMATCH"
    RATE_LIMITED = "RATE_LIMITED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    ERROR = "ERROR"
    SWITCHING = "SWITCHING"


class CapabilityUnavailable(RuntimeError):
    """Provide capability unavailable behavior."""
    def __init__(self, capability: str, snapshot: dict[str, Any]) -> None:
        super().__init__(f"provider_capability_unavailable:{capability}")
        self.capability = capability
        self.snapshot = snapshot


class ProviderSwitchBusy(RuntimeError):
    """Provide provider switch busy behavior."""
    pass


@dataclass(frozen=True)
class ProviderError:
    """Provide provider error behavior."""
    category: str
    code: str
    retryable: bool
    stop_retries: bool
    public_message: str

    def as_dict(self) -> dict[str, Any]:
        """Execute the as dict operation."""
        return {
            "category": self.category,
            "code": self.code,
            "retryable": self.retryable,
            "stop_retries": self.stop_retries,
            "public_message": self.public_message,
        }


_BILLING_TERMS = (
    "arrearage",
    "insufficient balance",
    "balance not enough",
    "billing",
    "payment",
    "free quota",
    "quota exceeded",
    "insufficient_quota",
    "allocationquota",
    "current quota",
    "余额不足",
    "账户欠费",
    "欠费",
    "免费额度",
    "额度不足",
)
_KEY_TERMS = (
    "invalidapikey",
    "invalid_api_key",
    "invalid api key",
    "api key invalid",
)
_PERMISSION_TERMS = (
    "accessdenied",
    "access_denied",
    "not authorized",
    "permission",
    "model access denied",
    "workspace access denied",
    "unsupported model",
)
_RATE_LIMIT_TERMS = (
    "throttling.ratequota",
    "limitrequests",
    "limit_requests",
    "burst_rate",
    "rate limit",
    "too many requests",
)
_OFFLINE_TERMS = (
    "connection refused",
    "connection reset",
    "network is unreachable",
    "no route to host",
    "name or service not known",
    "temporary failure in name resolution",
    "getaddrinfo failed",
    "nodename nor servname",
    "failed to establish a new connection",
    "dns",
    "offline",
)
_REGION_ENDPOINT_TERMS = (
    "region mismatch",
    "endpoint mismatch",
    "workspace and base url",
    "workspace.baseurl",
    "invalid endpoint",
)


_ERROR_STATE_BY_CATEGORY = {
    "network_offline": ProviderState.OFFLINE.value,
    "network_timeout": ProviderState.NETWORK_TIMEOUT.value,
    "invalid_api_key": ProviderState.INVALID_KEY.value,
    "billing_or_quota": ProviderState.BILLING_BLOCKED.value,
    "permission": ProviderState.PERMISSION_DENIED.value,
    "region_endpoint_mismatch": ProviderState.REGION_ENDPOINT_MISMATCH.value,
    "rate_limit": ProviderState.RATE_LIMITED.value,
    "service_unavailable": ProviderState.SERVICE_UNAVAILABLE.value,
}


def classify_bailian_error(
    *,
    http_status: int,
    code: str | None,
    message: str | None,
    credential_source: str,
) -> ProviderError:
    """Map a provider failure without returning the sensitive raw body."""

    safe_code = re.sub(r"[^A-Za-z0-9_.-]", "", str(code or "UNKNOWN"))[:96] or "UNKNOWN"
    combined = f"{safe_code} {message or ''}".lower()
    if any(term in combined for term in _BILLING_TERMS) or http_status == 402:
        if credential_source == "course_default":
            public = (
                "课程演示默认 API Key 当前余额或可用额度不足，暂时无法继续调用模型。"
                "您可以稍后重试，或切换为自己的阿里云百炼 API Key。"
            )
        else:
            public = (
                "当前阿里云百炼账户余额或可用额度不足，暂时无法继续调用模型。"
                "请前往阿里云费用中心充值或检查免费额度，完成后重新测试连接。"
            )
        return ProviderError("billing_or_quota", safe_code, False, True, public)
    if any(term in combined for term in _KEY_TERMS) or http_status == 401:
        return ProviderError(
            "invalid_api_key",
            safe_code,
            False,
            True,
            "当前 API Key 无效或与 API Host 不匹配，请重新输入或切换凭据后测试连接。",
        )
    if any(term in combined for term in _REGION_ENDPOINT_TERMS):
        return ProviderError(
            "region_endpoint_mismatch",
            safe_code,
            False,
            True,
            "当前地域、业务空间或 API Host 不匹配，请检查高级连接设置后重新测试。",
        )
    if any(term in combined for term in _PERMISSION_TERMS) or http_status == 403:
        return ProviderError(
            "permission",
            safe_code,
            False,
            True,
            "当前凭据没有所选模型或业务空间的访问权限，请检查模型权限、地域和 API Host。",
        )
    if http_status == 429 or any(term in combined for term in _RATE_LIMIT_TERMS):
        return ProviderError(
            "rate_limit",
            safe_code,
            True,
            False,
            "阿里云百炼当前请求较多，已停止本次请求。请稍后重试。",
        )
    if http_status == 413 or any(
        term in combined
        for term in (
            "requesttoolarge",
            "requestentitytoolarge",
            "request too large",
            "request entity too large",
            "payload too large",
        )
    ):
        return ProviderError(
            "request_too_large",
            safe_code,
            False,
            True,
            "图片请求体超过云端限制，系统将使用兼容传输版本有限重试；原图不会被修改。",
        )
    if http_status == 0 and any(
        term in combined for term in ("timeout", "timed out", "超时")
    ):
        return ProviderError(
            "network_timeout",
            safe_code,
            True,
            False,
            "连接阿里云百炼超时，请检查网络与 API Host 后重试。",
        )
    if http_status == 0 and any(term in combined for term in _OFFLINE_TERMS):
        return ProviderError(
            "network_offline",
            safe_code,
            True,
            False,
            "当前无法连接阿里云百炼，可能是网络断开、DNS 解析失败或连接被拒绝。已有图片、对话和索引不会丢失；网络恢复后请点击“重新连接”。",
        )
    if any(term in combined for term in ("unsupported image", "image format", "decoder")):
        return ProviderError(
            "unsupported_image",
            safe_code,
            False,
            True,
            "当前图片格式不受模型支持，请转换为常用 JPG、PNG 或 WebP 格式后重试。",
        )
    if http_status in {500, 502, 503, 504} or any(
        term in combined for term in ("model unavailable", "modelservingerror")
    ):
        return ProviderError(
            "service_unavailable",
            safe_code,
            True,
            False,
            "阿里云百炼或当前模型暂时不可用，请稍后重试。",
        )
    return ProviderError(
        "unknown",
        safe_code,
        False,
        True,
        "模型调用失败，但暂时无法确认是额度、权限还是服务问题。请核对脱敏错误码后重试连接。",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class ProviderManager:
    """Own the persisted public selection and process-private credentials."""

    schema_version = "scenemindx_provider_access_v1"

    def __init__(
        self,
        *,
        state_path: Path,
        profiles_path: Path,
        migrate_self_hosted: bool,
        legacy_vlm: Any,
        legacy_retrieval: Any,
        legacy_embedding: Any | None,
    ) -> None:
        self.state_path = state_path.resolve()
        self.profiles_path = profiles_path.resolve()
        self.profiles = json.loads(self.profiles_path.read_text(encoding="utf-8"))
        self._legacy_vlm = legacy_vlm
        self._legacy_retrieval = legacy_retrieval
        self._legacy_embedding = legacy_embedding
        self._vlm_by_mode: dict[str, Any] = {"self_hosted": legacy_vlm}
        self._retrieval_by_mode: dict[str, Any] = {"self_hosted": legacy_retrieval}
        self._embedding_by_mode: dict[str, Any] = {}
        if legacy_embedding is not None:
            self._embedding_by_mode["self_hosted"] = legacy_embedding
        self._session_credentials: dict[str, str] | None = None
        self._session_credential_revision = 0
        self._lock = threading.RLock()
        self._active_requests = 0
        self._last_errors: dict[str, dict[str, Any]] = {}
        self._connection_state: str | None = None
        self._connection_transition: str | None = None
        self._before_request_hook: Any | None = None
        self._course_default_credentials_available = True
        self._selection = self._load_or_initialize(migrate_self_hosted)

    def _load_or_initialize(self, migrate_self_hosted: bool) -> dict[str, Any]:
        if self.state_path.is_file():
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if value.get("schema_version") != self.schema_version:
                raise ValueError("provider_access_schema_mismatch")
            return value
        mode: ProviderMode = "self_hosted" if migrate_self_hosted else "no_model"
        value = {
            "schema_version": self.schema_version,
            "mode": mode,
            "cloud_tier": "standard",
            "credential_source": "course_default",
            "region": "cn-beijing",
            "api_host_override": None,
            "selection_required": not migrate_self_hosted,
            "migrated_from_legacy_runtime": bool(migrate_self_hosted),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        _atomic_json(self.state_path, value)
        return value

    @property
    def mode(self) -> str:
        """Execute the mode operation."""
        return str(self._selection["mode"])

    @property
    def cloud_tier(self) -> str:
        """Execute the cloud tier operation."""
        return str(self._selection.get("cloud_tier", "standard"))

    @property
    def credential_source(self) -> str:
        """Execute the credential source operation."""
        return str(self._selection.get("credential_source", "course_default"))

    @property
    def active_requests(self) -> int:
        """Execute the active requests operation."""
        with self._lock:
            return self._active_requests

    def set_user_session_credentials(
        self,
        *,
        api_key: str,
        region: str,
        api_host: str | None,
        workspace_id: str | None = None,
        endpoint_mode: str = "shared",
    ) -> None:
        """Set user session credentials."""
        key = api_key
        if (
            not key.startswith("sk-")
            or len(key) < 12
            or any(character.isspace() for character in key)
        ):
            raise ValueError("invalid_api_key_shape")
        if region != "cn-beijing":
            raise ValueError("unsupported_region")
        host = (api_host or "").strip().rstrip("/")
        if host and not host.startswith("https://"):
            raise ValueError("api_host_must_use_https")
        with self._lock:
            self._session_credential_revision += 1
            self._session_credentials = {
                "api_key": key,
                "region": region,
                "api_host": host,
                "workspace_id": (workspace_id or "").strip(),
                "endpoint_mode": endpoint_mode,
            }

    def has_user_session_credentials(self) -> bool:
        """Execute the has user session credentials operation."""
        with self._lock:
            return bool(self._session_credentials)

    def set_course_default_credentials_available(self, available: bool) -> None:
        """Project whether the private course credential can actually be read.

        The path and credential value remain outside the public provider
        snapshot; only availability is projected to UI capability state.
        """
        with self._lock:
            self._course_default_credentials_available = bool(available)

    @property
    def session_credential_revision(self) -> int:
        """Execute the session credential revision operation."""
        with self._lock:
            return self._session_credential_revision

    def user_session_credentials(self) -> dict[str, str] | None:
        """Execute the user session credentials operation."""
        with self._lock:
            return dict(self._session_credentials) if self._session_credentials else None

    def clear_user_session_credentials(self) -> None:
        """Execute the clear user session credentials operation."""
        with self._lock:
            self._session_credential_revision += 1
            self._session_credentials = None

    def install_runtime(
        self,
        mode: str,
        *,
        vlm: Any | None = None,
        retrieval: Any | None = None,
        embedding: Any | None = None,
    ) -> None:
        """Execute the install runtime operation."""
        with self._lock:
            if vlm is not None:
                self._vlm_by_mode[mode] = vlm
            if retrieval is not None:
                self._retrieval_by_mode[mode] = retrieval
            if embedding is not None:
                self._embedding_by_mode[mode] = embedding

    def remove_runtime(self, mode: str) -> None:
        """Remove runtime."""
        if mode == "self_hosted":
            return
        with self._lock:
            self._vlm_by_mode.pop(mode, None)
            self._retrieval_by_mode.pop(mode, None)
            self._embedding_by_mode.pop(mode, None)

    def set_error(self, capability: str, error: ProviderError) -> None:
        """Set error."""
        with self._lock:
            self._last_errors[capability] = error.as_dict()
            self._connection_state = _ERROR_STATE_BY_CATEGORY.get(
                error.category,
                ProviderState.ERROR.value,
            )
            self._connection_transition = f"FAILED:{error.code}"

    def clear_errors(self) -> None:
        """Execute the clear errors operation."""
        with self._lock:
            self._last_errors = {}

    def set_connection_state(
        self,
        state: str | ProviderState | None,
        *,
        transition: str | None = None,
    ) -> None:
        """Set connection state."""
        value = state.value if isinstance(state, ProviderState) else state
        if value is not None and value not in {item.value for item in ProviderState}:
            raise ValueError("invalid_provider_connection_state")
        with self._lock:
            self._connection_state = value
            self._connection_transition = transition

    def set_before_request_hook(self, hook: Any | None) -> None:
        """Install one process-local lazy validation hook.

        The hook is intentionally never persisted and runs before a real
        capability request, not from status polling.
        """
        with self._lock:
            self._before_request_hook = hook

    def prepare_for_request(self, capability: str) -> None:
        """Execute the prepare for request operation."""
        with self._lock:
            hook = self._before_request_hook
        if hook is not None:
            hook(capability)

    def select(
        self,
        *,
        mode: ProviderMode,
        cloud_tier: CloudTier = "standard",
        credential_source: CredentialSource = "course_default",
        region: str = "cn-beijing",
        api_host_override: str | None = None,
        endpoint_mode: str = "shared",
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute the select operation."""
        if mode not in {"no_model", "bailian", "local", "self_hosted"}:
            raise ValueError("unsupported_provider_mode")
        if cloud_tier not in {"standard", "high_quality"}:
            raise ValueError("unsupported_cloud_tier")
        if credential_source not in {"user_session", "course_default"}:
            raise ValueError("unsupported_credential_source")
        if region != "cn-beijing":
            raise ValueError("unsupported_region")
        if endpoint_mode not in {"shared", "workspace", "custom"}:
            raise ValueError("unsupported_endpoint_mode")
        override = (api_host_override or "").strip().rstrip("/") or None
        if override and not override.startswith("https://"):
            raise ValueError("api_host_must_use_https")
        with self._lock:
            if self._active_requests:
                raise ProviderSwitchBusy("provider_switch_wait_for_active_requests")
            self._selection = {
                **self._selection,
                "mode": mode,
                "cloud_tier": cloud_tier,
                "credential_source": credential_source,
                "region": region,
                "api_host_override": override,
                "endpoint_mode": endpoint_mode,
                "workspace_configured": bool((workspace_id or "").strip()),
                "selection_required": False,
                "updated_at": _now_iso(),
            }
            self._last_errors = {}
            self._connection_state = None
            self._connection_transition = None
            _atomic_json(self.state_path, self._selection)
        # Selecting a provider is a state transition, not an implicit
        # connectivity test.  Keep the response bounded by projecting the
        # services' last cached health; explicit connection tests and the
        # first real capability request remain the authorities that may probe
        # a dependency.
        return self.snapshot_cached()

    @contextmanager
    def request_scope(self) -> Iterator[None]:
        """Execute the request scope operation."""
        with self._lock:
            self._active_requests += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_requests = max(0, self._active_requests - 1)

    def _safe_status(self, service: Any | None) -> dict[str, Any]:
        if service is None:
            return {"status": "not_configured", "loaded": False}
        try:
            value = service.status()
        except Exception as exc:  # provider boundary must not crash status UI
            return {
                "status": "error",
                "loaded": False,
                "error_code": type(exc).__name__,
            }
        return dict(value) if isinstance(value, dict) else {"status": "error"}

    def _safe_cached_status(self, service: Any | None) -> dict[str, Any]:
        if service is None:
            return {"status": "not_configured", "loaded": False}
        reader = getattr(service, "cached_status", None)
        if not callable(reader):
            return self._safe_status(service)
        try:
            value = reader()
        except Exception as exc:  # cached health must remain a safe local read
            return {
                "status": "error",
                "loaded": False,
                "error_code": type(exc).__name__,
            }
        return dict(value) if isinstance(value, dict) else {"status": "error"}

    @staticmethod
    def _ready(value: dict[str, Any]) -> bool:
        return str(value.get("status", "")).lower() in {
            "ready",
            "success",
            "loaded",
        }

    def vlm_service(self) -> Any | None:
        """Execute the vlm service operation."""
        return self._vlm_by_mode.get(self.mode)

    def retrieval_service(self) -> Any | None:
        """Execute the retrieval service operation."""
        return self._retrieval_by_mode.get(self.mode)

    def embedding_service(self) -> Any | None:
        """Execute the embedding service operation."""
        return self._embedding_by_mode.get(self.mode)

    def require(self, capability: str) -> None:
        """Execute the require operation."""
        snapshot = self.snapshot()
        if not snapshot["capabilities"].get(capability, False):
            raise CapabilityUnavailable(capability, snapshot)

    def snapshot(self, *, cached_only: bool = False) -> dict[str, Any]:
        """Execute the snapshot operation."""
        with self._lock:
            mode = self.mode
            status_reader = (
                self._safe_cached_status if cached_only else self._safe_status
            )
            if mode == "no_model":
                vlm = {"status": "disabled", "loaded": False}
                embedding = {"status": "disabled", "loaded": False}
                index = {"status": "disabled", "items": 0}
            else:
                vlm = status_reader(self._vlm_by_mode.get(mode))
                retrieval = status_reader(self._retrieval_by_mode.get(mode))
                embedding = (
                    dict(retrieval.get("embedding", {}))
                    if isinstance(retrieval.get("embedding"), dict)
                    else status_reader(self._embedding_by_mode.get(mode))
                )
                index = (
                    {
                        **dict(retrieval.get("index", {})),
                        **{
                            key: retrieval.get(key)
                            for key in ("base_items", "user_items")
                            if retrieval.get(key) is not None
                        },
                    }
                    if isinstance(retrieval.get("index"), dict)
                    else dict(retrieval)
                )
            credential_configured = (
                mode != "bailian"
                or (
                    self.credential_source == "course_default"
                    and self._course_default_credentials_available
                )
                or self.has_user_session_credentials()
            )
            credential_error = (
                mode == "bailian" and "credentials" in self._last_errors
            )
            vlm_ready = (
                self._ready(vlm)
                and "vlm" not in self._last_errors
                and not credential_error
                and credential_configured
            )
            embedding_ready = (
                self._ready(embedding)
                and "embedding" not in self._last_errors
                and not credential_error
                and credential_configured
            )
            index_ready = self._ready(index) and int(index.get("items", 0) or 0) > 0
            if mode == "no_model":
                state = ProviderState.NO_MODEL.value
            elif mode == "bailian" and not credential_configured:
                state = ProviderState.UNCONFIGURED.value
            elif vlm_ready and embedding_ready and index_ready:
                state = ProviderState.READY.value
            elif vlm_ready or (embedding_ready and index_ready):
                state = ProviderState.PARTIAL_READY.value
            elif self._last_errors:
                state = (
                    ProviderState.PARTIAL_READY.value
                    if vlm_ready or (embedding_ready and index_ready)
                    else ProviderState.ERROR.value
                )
            else:
                state = ProviderState.CONNECTING.value
            cloud = self.profiles["profiles"]["bailian"]
            selected_cloud = cloud[self.cloud_tier]
            return {
                "schema_version": self.schema_version,
                "state": state,
                "mode": mode,
                "mode_label": self.profiles["modes"][mode]["label"],
                "selection_required": bool(self._selection.get("selection_required")),
                "migrated_from_legacy_runtime": bool(
                    self._selection.get("migrated_from_legacy_runtime")
                ),
                "cloud_tier": self.cloud_tier,
                "credential_source": self.credential_source,
                "credential": {
                    "configured": credential_configured,
                    "course_default_available": (
                        self._course_default_credentials_available
                    ),
                    "revealable": False,
                    "process_session_only": self.credential_source == "user_session",
                },
                "region": self._selection.get("region", "cn-beijing"),
                "api_host_override_configured": bool(
                    self._selection.get("api_host_override")
                ),
                "endpoint_mode": self._selection.get(
                    "endpoint_mode", "shared"
                ),
                "workspace_configured": bool(
                    self._selection.get("workspace_configured")
                ),
                "vlm": {
                    **vlm,
                    "model_id": (
                        selected_cloud["vlm_model_id"]
                        if mode == "bailian"
                        else vlm.get("model")
                    ),
                },
                "embedding": {
                    **embedding,
                    "model_id": (
                        cloud["embedding"]["model_id"]
                        if mode == "bailian"
                        else embedding.get("model")
                    ),
                    "dimensions": (
                        cloud["embedding"]["dimension"]
                        if mode == "bailian"
                        else embedding.get("dimensions")
                    ),
                },
                "index": index,
                "index_availability": {
                    "exists": bool(
                        self._ready(index)
                        and int(index.get("items", 0) or 0) > 0
                    ),
                    "status": index.get("status", "not_configured"),
                    "query_vector_available": embedding_ready,
                    "preserved_while_offline": True,
                },
                "capabilities": {
                    "vlm": vlm_ready,
                    "embedding": embedding_ready,
                    "retrieval": embedding_ready and index_ready,
                },
                "active_requests": self._active_requests,
                "errors": dict(self._last_errors),
                "connection_state_override": self._connection_state,
                "connection_transition": self._connection_transition,
                "rerank": {
                    "connected": False,
                    "visible_in_standard_ui": False,
                },
                "updated_at": self._selection.get("updated_at"),
            }

    def snapshot_cached(self) -> dict[str, Any]:
        """Build a provider snapshot without dependency probes or network I/O."""
        return self.snapshot(cached_only=True)


class ProviderVLMProxy:
    """Late-bind every VLM call so business routes never branch on model IDs."""

    def __init__(
        self,
        manager: ProviderManager,
        *,
        transport_preprocessor: CloudImageTransportPreprocessor | None = None,
    ) -> None:
        self.manager = manager
        self.transport_preprocessor = transport_preprocessor

    def _service(self) -> Any:
        self.manager.prepare_for_request("vlm")
        service = self.manager.vlm_service()
        if service is None:
            raise CapabilityUnavailable("vlm", self.manager.snapshot())
        return service

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        service = self.manager.vlm_service()
        if service is None:
            return {"status": "disabled", "loaded": False}
        return self.manager._safe_status(service)

    def cached_status(self) -> dict[str, Any]:
        """Execute the cached status operation."""
        service = self.manager.vlm_service()
        if service is None:
            return {"status": "disabled", "loaded": False}
        return self.manager._safe_cached_status(service)

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        service = self.manager.vlm_service()
        if service is None or not hasattr(service, "load"):
            return self.status()
        return service.load()

    def unload(self) -> dict[str, Any]:
        """Execute the unload operation."""
        service = self.manager.vlm_service()
        if service is None or not hasattr(service, "unload"):
            return self.status()
        return service.unload()

    @staticmethod
    def _attach_transport(
        result: Any,
        prepared: PreparedCloudImageBatch | None,
        *,
        request_count: int,
        fallback_from_request_too_large: bool,
    ) -> Any:
        if prepared is None:
            return result
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            data["image_transport"] = {
                **prepared.public_audit(),
                "request_count": request_count,
                "fallback_from_request_too_large": (
                    fallback_from_request_too_large
                ),
            }
        return result

    def _call_with_transport(
        self,
        image_paths: list[Path] | tuple[Path, ...],
        invoke: Any,
    ) -> Any:
        service = self._service()
        originals = tuple(Path(path).resolve() for path in image_paths)
        preprocessor = self.transport_preprocessor
        if preprocessor is None or not originals:
            return invoke(service, originals)
        provider_id = str(
            getattr(service, "provider_id", "default") or "default"
        )
        model_id = (
            str(getattr(service, "model_id", "") or "").casefold()
            if provider_id == "bailian"
            else ""
        )
        provider_profile = (
            "bailian_high_quality"
            if provider_id == "bailian" and "qwen3.7" in model_id
            else "bailian_standard"
            if provider_id == "bailian"
            else "self_hosted"
            if provider_id == "self_hosted"
            else "local"
            if provider_id == "local"
            else "default"
        )
        prepared = preprocessor.prepare_batch(
            originals,
            provider_profile=provider_profile,
        )
        request_count = 0
        attempted_signatures: set[tuple[str, ...]] = set()
        while True:
            signature = tuple(
                item.profile_id for item in prepared.items
            )
            attempted_signatures.add(signature)
            request_count += 1
            try:
                result = invoke(service, prepared.request_paths)
                return self._attach_transport(
                    result,
                    prepared,
                    request_count=request_count,
                    fallback_from_request_too_large=request_count > 1,
                )
            except Exception as exc:
                if not preprocessor.is_request_too_large(exc):
                    raise
                fallback: PreparedCloudImageBatch | None = None
                for minimum_level in range(
                    1,
                    preprocessor.max_compatible_level(
                        provider_profile=provider_profile,
                    )
                    + 1,
                ):
                    candidate = preprocessor.prepare_batch(
                        originals,
                        force_oversized=True,
                        fallback_reason="provider_request_too_large",
                        provider_profile=provider_profile,
                        minimum_compatible_level=minimum_level,
                    )
                    candidate_signature = tuple(
                        item.profile_id for item in candidate.items
                    )
                    if candidate_signature not in attempted_signatures:
                        fallback = candidate
                        break
                if fallback is None:
                    setattr(exc, "compatible_transport_attempted", True)
                    raise
                prepared = fallback

    def run_course_prompt(
        self,
        image_paths: Any,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        """Execute the run course prompt operation."""
        return self._call_with_transport(
            tuple(image_paths),
            lambda service, paths: service.run_course_prompt(
                paths,
                prompt,
                **kwargs,
            ),
        )

    def run_multiturn_chat(
        self,
        image_paths: Any,
        image_labels: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute the run multiturn chat operation."""
        return self._call_with_transport(
            tuple(image_paths),
            lambda service, paths: service.run_multiturn_chat(
                paths,
                image_labels,
                **kwargs,
            ),
        )

    def analyze_image(
        self,
        image_path: Path,
        prompt_version: str | None = None,
    ) -> Any:
        """Execute the analyze image operation."""
        return self._call_with_transport(
            (image_path,),
            lambda service, paths: service.analyze_image(
                paths[0],
                prompt_version,
            ),
        )

    def describe_image(
        self,
        image_path: Path,
        core_facts: dict[str, Any],
        options: dict[str, Any],
    ) -> Any:
        """Execute the describe image operation."""
        return self._call_with_transport(
            (image_path,),
            lambda service, paths: service.describe_image(
                paths[0],
                core_facts,
                options,
            ),
        )

    def answer_question(
        self,
        image_path: Path,
        question: str,
        evidence: dict[str, Any],
    ) -> Any:
        """Execute the answer question operation."""
        return self._call_with_transport(
            (image_path,),
            lambda service, paths: service.answer_question(
                paths[0],
                question,
                evidence,
            ),
        )

    def generate_content(
        self,
        image_paths: Any,
        facts: Any,
        options: dict[str, Any],
    ) -> Any:
        """Execute the generate content operation."""
        return self._call_with_transport(
            tuple(image_paths),
            lambda service, paths: service.generate_content(
                paths,
                facts,
                options,
            ),
        )

    def compare_images(
        self,
        image_paths: Any,
        instruction: str | None = None,
    ) -> Any:
        """Execute the compare images operation."""
        return self._call_with_transport(
            tuple(image_paths),
            lambda service, paths: service.compare_images(
                paths,
                instruction,
            ),
        )

    def generate(self, request: dict[str, Any]) -> Any:
        """Execute the generate operation."""
        paths = tuple(Path(value) for value in request.get("image_paths", []))
        return self._call_with_transport(
            paths,
            lambda service, prepared_paths: service.generate(
                {
                    **request,
                    "image_paths": [str(path) for path in prepared_paths],
                }
            ),
        )

    def generate_structured(self, request: dict[str, Any]) -> Any:
        """Execute the generate structured operation."""
        paths = tuple(Path(value) for value in request.get("image_paths", []))
        return self._call_with_transport(
            paths,
            lambda service, prepared_paths: service.generate_structured(
                {
                    **request,
                    "image_paths": [str(path) for path in prepared_paths],
                }
            ),
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._service(), name)


class ProviderRetrievalProxy:
    """Late-bind retrieval while keeping every provider index isolated."""

    def __init__(self, manager: ProviderManager) -> None:
        self.manager = manager

    def _service(self, *, require_ready: bool = False) -> Any:
        if require_ready:
            self.manager.prepare_for_request("retrieval")
            self.manager.require("retrieval")
        service = self.manager.retrieval_service()
        if service is None:
            raise CapabilityUnavailable("retrieval", self.manager.snapshot())
        return service

    @property
    def embedding(self) -> Any:
        """Execute the embedding operation."""
        return self._service().embedding

    @property
    def index_path(self) -> Path:
        """Execute the index path operation."""
        return Path(self._service().index_path)

    @property
    def vectors(self) -> Any | None:
        """Execute the vectors operation."""
        service = self.manager.retrieval_service()
        return getattr(service, "vectors", None) if service is not None else None

    @property
    def e1_index(self) -> Any:
        """Execute the e1 index operation."""
        return self._service().e1_index

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        service = self.manager.retrieval_service()
        if service is None:
            state = (
                "disabled"
                if self.manager.mode == "no_model"
                else "not_configured"
            )
            return {
                "status": state,
                "items": 0,
                "fallback_active": False,
            }
        return dict(service.status())

    def cached_status(self) -> dict[str, Any]:
        """Execute the cached status operation."""
        service = self.manager.retrieval_service()
        if service is None:
            state = (
                "disabled"
                if self.manager.mode == "no_model"
                else "not_configured"
            )
            return {
                "status": state,
                "items": 0,
                "fallback_active": False,
            }
        return self.manager._safe_cached_status(service)

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        return dict(self._service().load())

    def build_index(self, items: Any) -> dict[str, Any]:
        """Build index."""
        return dict(self._service().build_index(items))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Execute the search operation."""
        return list(self._service(require_ready=True).search(query, top_k))

    def search_image(
        self,
        image_path: Path,
        top_k: int = 5,
        *,
        exclude_image_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the search image operation."""
        return list(
            self._service(require_ready=True).search_image(
                image_path,
                top_k,
                exclude_image_id=exclude_image_id,
            )
        )

    def search_hybrid(
        self,
        image_path: Path,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute the search hybrid operation."""
        return list(
            self._service(require_ready=True).search_hybrid(
                image_path,
                query,
                top_k,
                **kwargs,
            )
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._service(), name)


class ServiceVLMProviderAdapter:
    """Expose an existing local/HTTP VLM through the Phase 7 provider contract."""

    provider_id = "service"

    def __init__(
        self,
        service: Any,
        *,
        provider_id: str,
        capability_profile: dict[str, Any] | None = None,
    ) -> None:
        self.service = service
        self.provider_id = provider_id
        self._capability_profile = dict(capability_profile or {})

    @property
    def model_id(self) -> str:
        """Execute the model id operation."""
        status = self.status()
        return str(
            status.get("model")
            or getattr(self.service, "model_id", "unknown")
        )

    def health_check(self) -> dict[str, Any]:
        """Execute the health check operation."""
        status = self.status()
        return {
            **status,
            "identity_verified": bool(status.get("model")),
        }

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        value = self.service.status()
        return dict(value) if isinstance(value, dict) else {"status": "error"}

    def cached_status(self) -> dict[str, Any]:
        """Execute the cached status operation."""
        reader = getattr(self.service, "cached_status", None)
        if not callable(reader):
            return self.status()
        value = reader()
        return dict(value) if isinstance(value, dict) else {"status": "error"}

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        if callable(getattr(self.service, "load", None)):
            value = self.service.load()
            return dict(value) if isinstance(value, dict) else self.status()
        return self.status()

    def unload(self) -> dict[str, Any]:
        """Execute the unload operation."""
        if callable(getattr(self.service, "unload", None)):
            value = self.service.unload()
            return dict(value) if isinstance(value, dict) else self.status()
        if callable(getattr(self.service, "unload_model", None)):
            self.service.unload_model()
        return self.status()

    def generate(self, request: dict[str, Any]) -> Any:
        """Execute the generate operation."""
        if callable(getattr(self.service, "generate", None)):
            return self.service.generate(request)
        raise NotImplementedError("generic_generate_not_exposed_by_service")

    def generate_structured(self, request: dict[str, Any]) -> Any:
        """Execute the generate structured operation."""
        if callable(getattr(self.service, "generate_structured", None)):
            return self.service.generate_structured(request)
        return self.generate(request)

    def capability_profile(self) -> dict[str, Any]:
        """Execute the capability profile operation."""
        return dict(self._capability_profile)

    def usage(self) -> dict[str, Any]:
        """Execute the usage operation."""
        status = self.status()
        return {
            key: status.get(key)
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if status.get(key) is not None
        }

    def latency(self) -> dict[str, Any]:
        """Execute the latency operation."""
        status = self.status()
        return {
            key: status.get(key)
            for key in (
                "latency_seconds",
                "last_latency_seconds",
                "last_request_latency_ms",
            )
            if status.get(key) is not None
        }

    @staticmethod
    def error_mapping(error: Any) -> dict[str, Any]:
        """Execute the error mapping operation."""
        return {
            "category": (
                "cuda_oom"
                if "out of memory" in str(error).lower()
                else "service_error"
            ),
            "code": type(error).__name__,
        }

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.service, name)


class LocalVLMProvider(ServiceVLMProviderAdapter):
    """Provide local v l m provider behavior."""
    def __init__(self, service: Any) -> None:
        super().__init__(
            service,
            provider_id="local",
            capability_profile={"execution": "current_computer_cuda"},
        )


class SelfHostedVLMProvider(ServiceVLMProviderAdapter):
    """Provide self hosted v l m provider behavior."""
    def __init__(self, service: Any) -> None:
        super().__init__(
            service,
            provider_id="self_hosted",
            capability_profile={"execution": "private_server_tunnel"},
        )


class ServiceEmbeddingProviderAdapter:
    """Expose an existing embedding runtime through the provider contract."""

    provider_id = "service"
    normalization = "l2"

    def __init__(self, service: Any, *, provider_id: str) -> None:
        self.service = service
        self.provider_id = provider_id

    @property
    def model_id(self) -> str:
        """Execute the model id operation."""
        return str(
            self.status().get("model")
            or getattr(self.service, "model_id", "unknown")
        )

    @property
    def dimension(self) -> int:
        """Execute the dimension operation."""
        return int(
            self.status().get("dimensions")
            or getattr(self.service, "dimension", 0)
        )

    def health_check(self) -> dict[str, Any]:
        """Execute the health check operation."""
        status = self.status()
        return {
            **status,
            "dimension_verified": self.dimension > 0,
        }

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        value = self.service.status()
        return dict(value) if isinstance(value, dict) else {"status": "error"}

    def cached_status(self) -> dict[str, Any]:
        """Execute the cached status operation."""
        reader = getattr(self.service, "cached_status", None)
        if not callable(reader):
            return self.status()
        value = reader()
        return dict(value) if isinstance(value, dict) else {"status": "error"}

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        if callable(getattr(self.service, "load", None)):
            value = self.service.load()
            return dict(value) if isinstance(value, dict) else self.status()
        return self.status()

    def unload(self) -> dict[str, Any]:
        """Execute the unload operation."""
        if callable(getattr(self.service, "unload", None)):
            value = self.service.unload()
            return dict(value) if isinstance(value, dict) else self.status()
        if callable(getattr(self.service, "unload_model", None)):
            self.service.unload_model()
        return self.status()

    def encode_text(self, text: str) -> Any:
        """Execute the encode text operation."""
        return self.service.encode_text(text)

    def encode_image(self, image_path: Path) -> Any:
        """Execute the encode image operation."""
        return self.service.encode_image(image_path)

    def encode_multimodal(self, image_path: Path, text: str) -> Any:
        """Execute the encode multimodal operation."""
        return self.service.encode_multimodal(image_path, text)

    def usage(self) -> dict[str, Any]:
        """Execute the usage operation."""
        return {}

    @staticmethod
    def error_mapping(error: Any) -> dict[str, Any]:
        """Execute the error mapping operation."""
        return {
            "category": (
                "cuda_oom"
                if "out of memory" in str(error).lower()
                else "service_error"
            ),
            "code": type(error).__name__,
        }


class LocalEmbeddingProvider(ServiceEmbeddingProviderAdapter):
    """Provide local embedding provider behavior."""
    def __init__(self, service: Any) -> None:
        super().__init__(service, provider_id="local")


class SelfHostedEmbeddingProvider(ServiceEmbeddingProviderAdapter):
    """Provide self hosted embedding provider behavior."""
    def __init__(self, service: Any) -> None:
        super().__init__(service, provider_id="self_hosted")
