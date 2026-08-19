"""Alibaba Cloud Model Studio providers for the Phase 7 product boundary.

Secrets and image Data URIs exist only for the duration of an HTTP request.
Only safe usage, latency, model identity and classified error metadata are
retained in memory.
"""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import time
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator

from scenemindx.phase1.provider_access import ProviderError, classify_bailian_error
from scenemindx.prompts.loader import (
    CorePromptRegistry,
    load_core_prompt_registry,
    load_prompt_manifest,
)

from .contracts import ServiceResult
from .vlm import (
    _extract_json_object,
    _normalize_plain_text,
    _normalize_simplified_chinese,
    _parse_rank_positions,
    _strip_description_meta_text,
)


EMBEDDING_PATH = (
    "/services/embeddings/multimodal-embedding/multimodal-embedding"
)
DEFAULT_OPENAI_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/api/v1"
RETRIEVAL_INSTRUCT = (
    "Retrieve the most relevant images from a personal visual library for the "
    "given query. Prioritize semantic intent, subject, scene, medium, layout, "
    "intended use, and explicit exclusions."
)


@dataclass(frozen=True)
class BailianCredentials:
    """Provide bailian credentials behavior."""
    api_key: str
    openai_base: str
    dashscope_base: str
    region: str
    source: str
    workspace_scope: str = "default_workspace"


class BailianProviderFailure(RuntimeError):
    """Provide bailian provider failure behavior."""
    def __init__(
        self,
        *,
        error: ProviderError,
        http_status: int,
        model_id: str,
    ) -> None:
        super().__init__(error.public_message)
        self.error = error
        self.http_status = http_status
        self.model_id = model_id

    def public_dict(self) -> dict[str, Any]:
        """Execute the public dict operation."""
        return {
            **self.error.as_dict(),
            "http_status": self.http_status,
            "model_id": self.model_id,
        }


def _normalize_hosts(api_host: str | None, region: str) -> tuple[str, str]:
    if region != "cn-beijing":
        raise ValueError("unsupported_region")
    value = (api_host or DEFAULT_OPENAI_BASE).strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("api_host_must_use_https")
    hostname = parsed.hostname.lower()
    if any(
        marker in hostname
        for marker in ("dashscope-intl", "dashscope-us", "ap-southeast", "virginia")
    ):
        raise ValueError("api_host_region_mismatch")
    if value.endswith("/compatible-mode/v1"):
        return value, value[: -len("/compatible-mode/v1")] + "/api/v1"
    if value.endswith("/api/v1"):
        return value[: -len("/api/v1")] + "/compatible-mode/v1", value
    if value.endswith("/chat/completions"):
        openai = value[: -len("/chat/completions")]
        return openai, openai[: -len("/compatible-mode/v1")] + "/api/v1"
    return value + "/compatible-mode/v1", value + "/api/v1"


def credentials_from_user(
    *,
    api_key: str,
    region: str,
    api_host: str | None,
    workspace_id: str | None = None,
    endpoint_mode: str = "shared",
) -> BailianCredentials:
    """Execute the credentials from user operation."""
    key = api_key
    if (
        not key.startswith("sk-")
        or len(key) < 12
        or any(character.isspace() for character in key)
    ):
        raise ValueError("invalid_api_key_shape")
    if endpoint_mode not in {"shared", "workspace", "custom"}:
        raise ValueError("unsupported_endpoint_mode")
    host = api_host
    scope = "user_session_default_workspace"
    if endpoint_mode == "workspace":
        workspace = (workspace_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", workspace):
            raise ValueError("invalid_workspace_id")
        host = f"https://{workspace}.dashscope.aliyuncs.com"
        scope = (
            "user_workspace:"
            + hashlib.sha256(workspace.encode("utf-8")).hexdigest()[:16]
        )
    elif endpoint_mode == "custom" and not api_host:
        raise ValueError("custom_api_host_required")
    openai_base, dashscope_base = _normalize_hosts(host, region)
    return BailianCredentials(
        api_key=key,
        openai_base=openai_base,
        dashscope_base=dashscope_base,
        region=region,
        source="user_session",
        workspace_scope=scope,
    )


def load_course_credentials(path: Path) -> BailianCredentials:
    """Load course credentials."""
    if not path.is_file():
        raise FileNotFoundError("course_demo_credentials_missing")
    pairs: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0].strip():
                pairs[row[0].strip().lower()] = row[1].strip()
    api_key = (
        pairs.get("apikey")
        or pairs.get("api_key")
        or pairs.get("dashscope_api_key")
        or ""
    )
    openai_base = (
        pairs.get("openaicompatible")
        or pairs.get("openai_compatible")
        or DEFAULT_OPENAI_BASE
    )
    dashscope_base = (
        pairs.get("dashscope")
        or pairs.get("apihost")
        or DEFAULT_DASHSCOPE_BASE
    )
    if not api_key.startswith("sk-") or len(api_key) < 12:
        raise ValueError("course_demo_api_key_invalid")
    openai = urlparse(openai_base)
    dashscope = urlparse(dashscope_base)
    if (
        openai.scheme != "https"
        or dashscope.scheme != "https"
        or openai.hostname != dashscope.hostname
    ):
        raise ValueError("course_demo_api_host_mismatch")
    region = "cn-beijing"
    if any(
        marker in (openai.hostname or "").lower()
        for marker in ("dashscope-intl", "dashscope-us", "ap-southeast", "virginia")
    ):
        raise ValueError("course_demo_region_mismatch")
    workspace_reference = (
        pairs.get("workspaceid")
        or pairs.get("workspacename")
        or "course_default_workspace"
    )
    workspace_scope = (
        "course_workspace:"
        + hashlib.sha256(
            workspace_reference.encode("utf-8")
        ).hexdigest()[:16]
    )
    return BailianCredentials(
        api_key=api_key,
        openai_base=openai_base.rstrip("/"),
        dashscope_base=dashscope_base.rstrip("/"),
        region=region,
        source="course_default",
        workspace_scope=workspace_scope,
    )


def _safe_error(raw: bytes) -> tuple[str, str]:
    try:
        value = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return "UNKNOWN", "request rejected"
    if not isinstance(value, dict):
        return "UNKNOWN", "request rejected"
    error = value.get("error")
    if isinstance(error, dict):
        return str(error.get("code", "UNKNOWN")), str(
            error.get("message", "request rejected")
        )[:1000]
    return str(value.get("code", "UNKNOWN")), str(
        value.get("message", "request rejected")
    )[:1000]


class _BailianHTTP:
    """Provide bailian h t t p behavior."""
    def __init__(
        self,
        *,
        credentials: Callable[[], BailianCredentials],
        model_id: str,
        timeout_seconds: float,
        error_sink: Callable[[str, ProviderError], None] | None,
    ) -> None:
        self._credentials = credentials
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.error_sink = error_sink
        self.last_retry_count = 0
        self._closed = False

    def close(self) -> None:
        """Execute the close operation."""
        self._closed = True

    def request(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        capability: str,
        retries: int = 2,
    ) -> tuple[dict[str, Any], float, int]:
        """Execute the request operation."""
        if self._closed:
            raise RuntimeError("bailian_http_client_closed")
        credentials = self._credentials()
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credentials.api_key}",
        }
        attempt = 0
        while True:
            started = time.perf_counter()
            request = Request(url, data=body, headers=headers, method="POST")
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                    status = int(response.status)
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("provider_response_root_not_object")
                self.last_retry_count = attempt
                return value, time.perf_counter() - started, status
            except HTTPError as exc:
                status = int(exc.code)
                code, message = _safe_error(exc.read())
                mapped = classify_bailian_error(
                    http_status=status,
                    code=code,
                    message=message,
                    credential_source=credentials.source,
                )
            except (URLError, TimeoutError, OSError) as exc:
                status = 0
                mapped = classify_bailian_error(
                    http_status=0,
                    code=type(exc).__name__,
                    message=str(exc),
                    credential_source=credentials.source,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                status = 0
                mapped = classify_bailian_error(
                    http_status=0,
                    code=type(exc).__name__,
                    message="provider response could not be decoded",
                    credential_source=credentials.source,
                )
            if mapped.retryable and attempt < retries:
                time.sleep(0.5 * (2**attempt))
                attempt += 1
                continue
            if self.error_sink:
                self.error_sink(capability, mapped)
            self.last_retry_count = attempt
            raise BailianProviderFailure(
                error=mapped,
                http_status=status,
                model_id=self.model_id,
            )


def _image_data_uri(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return (
        f"data:{media_type};base64,"
        f"{base64.b64encode(path.read_bytes()).decode('ascii')}"
    )


def _message_content(response: dict[str, Any]) -> tuple[str, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        content = "".join(parts)
    return str(content), choice.get("finish_reason")


class BailianVLMProvider:
    """Provide bailian v l m provider behavior."""
    provider_id = "bailian"

    def __init__(
        self,
        *,
        credentials: Callable[[], BailianCredentials],
        model_id: str,
        prompt_root: Path,
        capability_profile: dict[str, Any],
        core_registry_path: Path,
        core_prompt_version: str = "p3_v1_4",
        timeout_seconds: float = 180.0,
        error_sink: Callable[[str, ProviderError], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.prompt_root = prompt_root.resolve()
        self._profile = dict(capability_profile)
        self.core_registry: CorePromptRegistry = load_core_prompt_registry(
            core_registry_path
        )
        self.core_prompt_version = core_prompt_version
        self.task_prompts = load_prompt_manifest(self.prompt_root / "registry.json")
        schema_path = (
            self.prompt_root.parents[1]
            / "data"
            / "schemas"
            / "gate1_d3_semantic_review_payload_p3_v1_1.schema.json"
        )
        self.core_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self._credentials = credentials
        self._http = _BailianHTTP(
            credentials=credentials,
            model_id=model_id,
            timeout_seconds=timeout_seconds,
            error_sink=error_sink,
        )
        self._status: dict[str, Any] = {
            "status": "not_tested",
            "loaded": False,
            "provider_id": self.provider_id,
            "model": model_id,
            "model_id": model_id,
        }
        self._last_usage: dict[str, Any] = {}
        self._last_latency: dict[str, Any] = {}

    def _chat_url(self) -> str:
        base = self._credentials().openai_base
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    def _call(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> tuple[str, dict[str, Any], float, str | None, str | None]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
            "stream": False,
        }
        if self.model_id.startswith(("qwen3.6-", "qwen3.7-")):
            payload["enable_thinking"] = False
        response, latency, status = self._http.request(
            url=self._chat_url(),
            payload=payload,
            capability="vlm",
        )
        text, finish_reason = _message_content(response)
        if not text.strip():
            raise RuntimeError("bailian_vlm_empty_response")
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        self._last_usage = {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "estimated_cost_cny": self._estimate_cost(usage),
            "retry_count": self._http.last_retry_count,
        }
        self._last_latency = {"seconds": latency}
        reported_model = (
            str(response.get("model")) if response.get("model") is not None else None
        )
        self._status = {
            "status": "ready",
            "loaded": True,
            "provider_id": self.provider_id,
            "model": self.model_id,
            "model_id": self.model_id,
            "reported_model": reported_model,
            "http_status": status,
            "last_latency_seconds": latency,
            "core_prompt": self.prompt_identity(),
        }
        return text, self._last_usage, latency, finish_reason, reported_model

    def _estimate_cost(self, usage: dict[str, Any]) -> float:
        input_rate, output_rate = (
            (1.2, 7.2) if self.model_id == "qwen3.6-flash" else (2.0, 8.0)
        )
        return round(
            int(usage.get("prompt_tokens", 0) or 0) * input_rate / 1_000_000
            + int(usage.get("completion_tokens", 0) or 0)
            * output_rate
            / 1_000_000,
            8,
        )

    def _messages(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        image_labels: Sequence[str] | None = None,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for item in history_messages or []:
            role = str(item.get("role", ""))
            text = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not text:
                raise ValueError("invalid_multiturn_chat_history_message")
            messages.append({"role": role, "content": text})
        content: list[dict[str, Any]] = []
        for index, path in enumerate(image_paths):
            if image_labels is not None:
                content.append({"type": "text", "text": f"{image_labels[index]}："})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_uri(path)},
                }
            )
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _generate_raw(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        max_new_tokens: int,
        image_labels: Sequence[str] | None = None,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, str]] | None = None,
    ) -> ServiceResult:
        if image_labels is not None and len(image_paths) != len(image_labels):
            raise ValueError("image_label_count_mismatch")
        text, usage, latency, finish_reason, reported_model = self._call(
            messages=self._messages(
                image_paths,
                prompt,
                image_labels=image_labels,
                system_prompt=system_prompt,
                history_messages=history_messages,
            ),
            max_tokens=max_new_tokens,
        )
        output_tokens = int(usage.get("output_tokens", 0))
        return ServiceResult(
            status="success",
            data={
                "raw_output": text,
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": output_tokens,
                "finish_reason": finish_reason,
                "max_new_tokens": max_new_tokens,
                "max_new_tokens_hit": finish_reason in {"length", "max_tokens"},
                "usage": usage,
                "request_model_id": self.model_id,
                "reported_model": reported_model,
            },
            model=self.model_id,
            model_revision=reported_model,
            latency_seconds=latency,
        )

    def health_check(self) -> dict[str, Any]:
        """Execute the health check operation."""
        text, _, latency, _, reported_model = self._call(
            messages=[
                {
                    "role": "user",
                    "content": "SceneMind-X 连接测试：只回复 OK。",
                }
            ],
            max_tokens=8,
        )
        identity_ok = bool(
            not reported_model
            or reported_model == self.model_id
            or reported_model.startswith(self.model_id + "-")
        )
        if not identity_ok:
            self._status = {
                **self._status,
                "status": "error",
                "loaded": False,
                "error_code": "REPORTED_MODEL_MISMATCH",
            }
        return {
            **self._status,
            "identity_verified": identity_ok,
            "response_nonempty": bool(text.strip()),
            "latency_seconds": latency,
        }

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return dict(self._status)

    def cached_status(self) -> dict[str, Any]:
        """Execute the cached status operation."""
        return self.status()

    def close(self) -> None:
        """Execute the close operation."""
        self._http.close()
        self._status = {
            **self._status,
            "status": "closed",
            "loaded": False,
        }

    def capability_profile(self) -> dict[str, Any]:
        """Execute the capability profile operation."""
        return dict(self._profile)

    def usage(self) -> dict[str, Any]:
        """Execute the usage operation."""
        return dict(self._last_usage)

    def latency(self) -> dict[str, Any]:
        """Execute the latency operation."""
        return dict(self._last_latency)

    def error_mapping(self, error: Any) -> dict[str, Any]:
        """Execute the error mapping operation."""
        if isinstance(error, BailianProviderFailure):
            return error.public_dict()
        return {"category": "unknown", "code": type(error).__name__}

    def generate(self, request: dict[str, Any]) -> ServiceResult:
        """Execute the generate operation."""
        return self._generate_raw(
            [Path(value) for value in request.get("image_paths", [])],
            str(request["prompt"]),
            max_new_tokens=int(request.get("max_new_tokens", 512)),
            image_labels=request.get("image_labels"),
        )

    def generate_structured(self, request: dict[str, Any]) -> ServiceResult:
        """Execute the generate structured operation."""
        raw = self.generate(request)
        parsed = _extract_json_object(str(raw.data["raw_output"]))
        return ServiceResult(
            status=raw.status,
            data={**raw.data, "parsed_output": parsed},
            model=raw.model,
            model_revision=raw.model_revision,
            latency_seconds=raw.latency_seconds,
        )

    def prompt_identity(self, prompt_version: str | None = None) -> dict[str, str]:
        """Execute the prompt identity operation."""
        prompt_id = prompt_version or self.core_prompt_version
        prompt = self.core_registry.prompts[prompt_id]
        return {
            "prompt_id": prompt.prompt_id,
            "prompt_version": prompt.version,
            "prompt_sha256": prompt.bundle_sha256,
        }

    def task_prompt_identity(self, prompt_id: str) -> dict[str, str]:
        """Execute the task prompt identity operation."""
        prompt = self.task_prompts[prompt_id]
        return {
            "prompt_id": prompt.prompt_id,
            "prompt_version": prompt.version,
            "prompt_sha256": prompt.sha256,
        }

    def analyze_image(
        self, image_path: Path, prompt_version: str | None = None
    ) -> ServiceResult:
        """Execute the analyze image operation."""
        prompt_id = prompt_version or self.core_prompt_version
        identity = self.prompt_identity(prompt_id)
        core_prompt = self.core_registry.prompts[prompt_id]
        normalized: dict[str, str] = {}
        stages = []
        errors = []
        total_latency = 0.0
        for stage in core_prompt.stages:
            result = self._generate_raw(
                [image_path],
                stage.prompt.text,
                max_new_tokens=stage.max_new_tokens,
            )
            raw_output = str(result.data["raw_output"])
            parsed = None
            stage_error = None
            try:
                parsed = _extract_json_object(raw_output)
            except ValueError as exc:
                if len(stage.fields) == 1 and raw_output.strip():
                    normalized[stage.fields[0]] = (
                        raw_output.replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                    stage_error = f"raw_text_fallback:{exc}"
                else:
                    stage_error = str(exc)
            if parsed is not None:
                if set(parsed) != set(stage.fields) or any(
                    not isinstance(parsed.get(field), str)
                    for field in stage.fields
                ):
                    stage_error = "stage_field_contract_violation"
                else:
                    normalized.update(
                        {field: parsed[field] for field in stage.fields}
                    )
            if stage_error:
                errors.append(f"{stage.stage_id}:{stage_error}")
            total_latency += float(result.latency_seconds or 0)
            stages.append(
                {
                    "stage_id": stage.stage_id,
                    "prompt_sha256": stage.prompt.sha256,
                    "raw_output": raw_output,
                    "parsed_output": parsed,
                    "output_tokens": result.data.get("output_tokens"),
                    "latency_seconds": result.latency_seconds,
                    "stage_error": stage_error,
                }
            )
        schema_errors = [
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in sorted(
                Draft202012Validator(self.core_schema).iter_errors(normalized),
                key=lambda item: list(item.absolute_path),
            )
        ]
        return ServiceResult(
            status="success",
            data={
                **identity,
                "raw_output": "\n\n".join(
                    f"[{stage['stage_id']}]\n{stage['raw_output']}"
                    for stage in stages
                ),
                "normalized_output": normalized,
                "parsed_output": normalized,
                "stage_outputs": stages,
                "parse_error": "; ".join(errors) if errors else None,
                "schema_valid": not schema_errors,
                "schema_errors": schema_errors,
                "output_tokens": sum(
                    int(stage.get("output_tokens") or 0) for stage in stages
                ),
            },
            model=self.model_id,
            model_revision=self._status.get("reported_model"),
            latency_seconds=total_latency,
        )

    def describe_image(
        self,
        image_path: Path,
        core_facts: dict[str, Any],
        options: dict[str, Any],
    ) -> ServiceResult:
        """Execute the describe image operation."""
        spec = self.task_prompts["natural_chinese_detailed_description_v1"]
        prompt = spec.text.format(
            facts=json.dumps(core_facts, ensure_ascii=False),
            options=json.dumps(options, ensure_ascii=False),
        )
        raw = self._generate_raw([image_path], prompt, max_new_tokens=512)
        content, meta_removed = _strip_description_meta_text(
            str(raw.data["raw_output"])
        )
        normalized = _normalize_plain_text(content)
        final_output, converted = _normalize_simplified_chinese(normalized)
        return ServiceResult(
            status=raw.status,
            data={
                **raw.data,
                "prompt_id": spec.prompt_id,
                "prompt_version": spec.version,
                "prompt_sha256": spec.sha256,
                "final_output": final_output,
                "simplified_conversion_applied": converted,
                "prohibited_meta_text_removed": meta_removed,
            },
            model=raw.model,
            model_revision=raw.model_revision,
            latency_seconds=raw.latency_seconds,
        )

    def answer_question(
        self, image_path: Path, question: str, evidence: dict[str, Any]
    ) -> ServiceResult:
        """Execute the answer question operation."""
        prompt = (self.prompt_root / "vqa_v1.txt").read_text(
            encoding="utf-8"
        ).strip().format(
            question=question,
            evidence=json.dumps(evidence, ensure_ascii=False),
        )
        raw = self._generate_raw([image_path], prompt, max_new_tokens=320)
        return self._with_parsed(raw)

    def generate_content(
        self,
        image_paths: Sequence[Path],
        facts: Sequence[dict[str, Any]],
        options: dict[str, Any],
    ) -> ServiceResult:
        """Execute the generate content operation."""
        prompt = (self.prompt_root / "content_v1.txt").read_text(
            encoding="utf-8"
        ).strip().format(
            facts=json.dumps(list(facts), ensure_ascii=False),
            options=json.dumps(options, ensure_ascii=False),
        )
        return self._with_parsed(
            self._generate_raw(image_paths, prompt, max_new_tokens=384)
        )

    def compare_images(
        self,
        image_paths: Sequence[Path],
        instruction: str | None = None,
    ) -> ServiceResult:
        """Execute the compare images operation."""
        if len(image_paths) < 2:
            raise ValueError("compare_images_requires_two_images")
        if instruction and instruction.startswith("RANK_CONTRACT:"):
            user_instruction = instruction.removeprefix("RANK_CONTRACT:").strip()
            prompt = (
                f"将 {len(image_paths)} 张图按用户标准从优到劣排序。"
                f"只输出 1 到 {len(image_paths)} 的全部整数，以英文逗号分隔；"
                f"不得输出其他文本。标准：{user_instruction}"
            )
            raw = self._generate_raw(
                image_paths, prompt, max_new_tokens=32
            )
            positions = _parse_rank_positions(
                str(raw.data["raw_output"]), len(image_paths)
            )
            if positions is None:
                return ServiceResult(
                    status="invalid_output",
                    data={**raw.data, "parsed_output": None},
                    error="ranking_position_contract_violation",
                    model=raw.model,
                    model_revision=raw.model_revision,
                    latency_seconds=raw.latency_seconds,
                )
            parsed = {
                "ranking": [
                    {
                        "asset_id": image_paths[position - 1].name,
                        "rank": rank,
                        "reason": "",
                    }
                    for rank, position in enumerate(positions, start=1)
                ]
            }
            return ServiceResult(
                status="success",
                data={**raw.data, "parsed_output": parsed},
                model=raw.model,
                model_revision=raw.model_revision,
                latency_seconds=raw.latency_seconds,
            )
        prompt = (self.prompt_root / "compare_v1.txt").read_text(
            encoding="utf-8"
        ).strip()
        if instruction:
            prompt += f"\n\n用户要求：{instruction}"
        return self._with_parsed(
            self._generate_raw(image_paths, prompt, max_new_tokens=384)
        )

    def _with_parsed(self, raw: ServiceResult) -> ServiceResult:
        try:
            parsed = _extract_json_object(str(raw.data["raw_output"]))
            error = None
        except ValueError as exc:
            parsed = {}
            error = str(exc)
        return ServiceResult(
            status=raw.status,
            data={**raw.data, "parsed_output": parsed, "parse_error": error},
            model=raw.model,
            model_revision=raw.model_revision,
            latency_seconds=raw.latency_seconds,
        )

    def run_course_prompt(
        self,
        image_paths: Sequence[Path],
        prompt: str,
        *,
        prompt_id: str,
        prompt_sha256: str,
        max_new_tokens: int = 512,
        image_labels: Sequence[str] | None = None,
        min_new_tokens: int | None = None,
    ) -> ServiceResult:
        """Execute the run course prompt operation."""
        if not 1 <= len(image_paths) <= 5:
            raise ValueError("course_prompt_requires_1_to_5_images")
        if self.model_id == "qwen3.7-plus" and (
            prompt_id.startswith("phase6_0")
            or "canonical" in prompt_id.lower()
        ):
            max_new_tokens = max(768, max_new_tokens)
        raw = self._generate_raw(
            image_paths,
            prompt,
            max_new_tokens=max_new_tokens,
            image_labels=image_labels,
        )
        parsed = self._with_parsed(raw)
        return ServiceResult(
            status=parsed.status,
            data={
                **parsed.data,
                "prompt_id": prompt_id,
                "prompt_sha256": prompt_sha256,
                "image_labels": list(image_labels) if image_labels else None,
                "min_new_tokens": min_new_tokens,
                "provider_generation_profile": self._profile.get(
                    "generation_policy"
                ),
            },
            model=parsed.model,
            model_revision=parsed.model_revision,
            latency_seconds=parsed.latency_seconds,
        )

    def run_text_repair(
        self,
        prompt: str,
        *,
        prompt_id: str,
        prompt_sha256: str,
        max_new_tokens: int = 384,
    ) -> ServiceResult:
        """Execute the run text repair operation."""
        raw = self._generate_raw([], prompt, max_new_tokens=max_new_tokens)
        parsed = self._with_parsed(raw)
        return ServiceResult(
            status=parsed.status,
            data={
                **parsed.data,
                "prompt_id": prompt_id,
                "prompt_sha256": prompt_sha256,
            },
            model=parsed.model,
            model_revision=parsed.model_revision,
            latency_seconds=parsed.latency_seconds,
        )

    def run_multiturn_chat(
        self,
        image_paths: Sequence[Path],
        image_labels: Sequence[str],
        *,
        system_prompt: str,
        history_messages: Sequence[dict[str, str]],
        current_prompt: str,
        prompt_id: str,
        prompt_sha256: str,
        max_new_tokens: int = 512,
    ) -> ServiceResult:
        """Execute the run multiturn chat operation."""
        raw = self._generate_raw(
            image_paths,
            current_prompt,
            max_new_tokens=max_new_tokens,
            image_labels=image_labels,
            system_prompt=system_prompt,
            history_messages=history_messages,
        )
        parsed = self._with_parsed(raw)
        return ServiceResult(
            status=parsed.status,
            data={
                **parsed.data,
                "prompt_id": prompt_id,
                "prompt_sha256": prompt_sha256,
                "image_labels": list(image_labels),
                "history_message_count": len(history_messages),
                "candidate_id": "SCENEMINDX_MULTITURN_CHAT_V2_CANDIDATE",
            },
            model=parsed.model,
            model_revision=parsed.model_revision,
            latency_seconds=parsed.latency_seconds,
        )


class BailianEmbeddingProvider:
    """Provide bailian embedding provider behavior."""
    provider_id = "bailian"
    normalization = "l2"

    def __init__(
        self,
        *,
        credentials: Callable[[], BailianCredentials],
        model_id: str = "qwen3-vl-embedding",
        dimension: int = 2560,
        timeout_seconds: float = 90.0,
        error_sink: Callable[[str, ProviderError], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self._credentials = credentials
        self._http = _BailianHTTP(
            credentials=credentials,
            model_id=model_id,
            timeout_seconds=timeout_seconds,
            error_sink=error_sink,
        )
        self._status: dict[str, Any] = {
            "status": "not_tested",
            "loaded": False,
            "provider_id": self.provider_id,
            "model": model_id,
            "model_id": model_id,
            "dimensions": dimension,
            "normalization": self.normalization,
        }
        self._last_usage: dict[str, Any] = {}

    def _url(self) -> str:
        base = self._credentials().dashscope_base
        return base if base.endswith(EMBEDDING_PATH) else base + EMBEDDING_PATH

    def _encode(self, contents: list[dict[str, Any]]) -> list[list[float]]:
        payload = {
            "model": self.model_id,
            "input": {"contents": contents},
            "parameters": {
                "dimension": self.dimension,
                "enable_fusion": False,
                "instruct": RETRIEVAL_INSTRUCT,
            },
        }
        response, latency, status = self._http.request(
            url=self._url(),
            payload=payload,
            capability="embedding",
        )
        output = response.get("output") if isinstance(response.get("output"), dict) else {}
        embeddings = (
            output.get("embeddings")
            if isinstance(output.get("embeddings"), list)
            else []
        )
        ordered = sorted(
            (item for item in embeddings if isinstance(item, dict)),
            key=lambda item: int(item.get("index", 0)),
        )
        vectors = [
            [float(value) for value in item.get("embedding", [])]
            for item in ordered
        ]
        if len(vectors) != len(contents) or any(
            len(vector) != self.dimension for vector in vectors
        ):
            raise RuntimeError("bailian_embedding_dimension_mismatch")
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        image_tokens = int(usage.get("image_tokens", 0) or 0)
        text_tokens = int(usage.get("input_tokens", 0) or 0)
        self._last_usage = {
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "estimated_cost_cny": round(
                image_tokens * 1.8 / 1_000_000
                + text_tokens * 0.7 / 1_000_000,
                8,
            ),
            "retry_count": self._http.last_retry_count,
        }
        self._status = {
            "status": "ready",
            "loaded": True,
            "provider_id": self.provider_id,
            "model": self.model_id,
            "model_id": self.model_id,
            "reported_model": response.get("model") or self.model_id,
            "dimensions": self.dimension,
            "normalization": self.normalization,
            "http_status": status,
            "last_latency_seconds": latency,
        }
        return vectors

    def health_check(self) -> dict[str, Any]:
        """Execute the health check operation."""
        vector = self.encode_text("SceneMind-X multimodal retrieval connection test")
        return {
            **self._status,
            "dimension_verified": len(vector) == self.dimension,
        }

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return dict(self._status)

    def cached_status(self) -> dict[str, Any]:
        """Execute the cached status operation."""
        return self.status()

    def close(self) -> None:
        """Execute the close operation."""
        self._http.close()
        self._status = {
            **self._status,
            "status": "closed",
            "loaded": False,
        }

    def encode_text(self, text: str) -> list[float]:
        """Execute the encode text operation."""
        if not text.strip():
            raise ValueError("embedding_text_must_not_be_empty")
        return self._encode([{"text": text}])[0]

    def encode_image(self, image_path: Path) -> list[float]:
        """Execute the encode image operation."""
        return self._encode([{"image": _image_data_uri(image_path)}])[0]

    def encode_multimodal(self, image_path: Path, text: str) -> list[float]:
        # Product retrieval uses independent vectors. This method returns the
        # provider's fused representation only when explicitly requested.
        """Execute the encode multimodal operation."""
        payload = {
            "model": self.model_id,
            "input": {
                "contents": [
                    {"image": _image_data_uri(image_path)},
                    {"text": text},
                ]
            },
            "parameters": {
                "dimension": self.dimension,
                "enable_fusion": True,
                "instruct": RETRIEVAL_INSTRUCT,
            },
        }
        response, _, _ = self._http.request(
            url=self._url(),
            payload=payload,
            capability="embedding",
        )
        output = response.get("output") if isinstance(response.get("output"), dict) else {}
        embeddings = output.get("embeddings", [])
        if not isinstance(embeddings, list) or not embeddings:
            raise RuntimeError("bailian_embedding_missing_fusion_vector")
        vector = [float(value) for value in embeddings[0].get("embedding", [])]
        if len(vector) != self.dimension:
            raise RuntimeError("bailian_embedding_dimension_mismatch")
        return vector

    def usage(self) -> dict[str, Any]:
        """Execute the usage operation."""
        return dict(self._last_usage)

    def error_mapping(self, error: Any) -> dict[str, Any]:
        """Execute the error mapping operation."""
        if isinstance(error, BailianProviderFailure):
            return error.public_dict()
        return {"category": "unknown", "code": type(error).__name__}
