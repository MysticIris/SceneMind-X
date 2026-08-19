"""HTTP adapter for a user-controlled SceneMind-X VLM runtime."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scenemindx.prompts.loader import CorePromptRegistry, load_core_prompt_registry, load_prompt_manifest

from .contracts import ServiceResult


class RemoteVLMService:
    """Use the existing VLM API without loading model weights locally.

    The remote runtime receives image IDs, so this adapter intentionally only
    supports files whose basename is present in the remote frozen library.
    This keeps the transport contract explicit and prevents silently sending
    an arbitrary local asset to an unknown endpoint.
    """

    def __init__(
        self,
        endpoint: str,
        prompt_root: Path,
        *,
        core_registry_path: Path | None = None,
        core_prompt_version: str = "p3_v1_4",
        inline_images: bool = False,
        timeout_seconds: float = 180.0,
        health_timeout_seconds: float = 5.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.prompt_root = prompt_root.resolve()
        self.timeout_seconds = timeout_seconds
        self.health_timeout_seconds = max(0.5, float(health_timeout_seconds))
        self.inline_images = inline_images
        registry_path = core_registry_path or self.prompt_root.parent / "gate1" / "p3_registry.json"
        self.core_registry: CorePromptRegistry = load_core_prompt_registry(registry_path)
        self.core_prompt_version = core_prompt_version
        self.task_prompts = load_prompt_manifest(self.prompt_root / "registry.json")
        self._last_status: dict[str, Any] | None = None

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(f"{self.endpoint}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(
                request,
                timeout=(
                    self.timeout_seconds
                    if timeout_seconds is None
                    else timeout_seconds
                ),
            ) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            detail = getattr(exc, "reason", None) or str(exc)
            raise RuntimeError(f"remote_vlm_unreachable:{detail}") from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("remote_vlm_invalid_json") from exc
        if not isinstance(value, dict):
            raise RuntimeError("remote_vlm_invalid_response")
        return value

    @staticmethod
    def _result(response: dict[str, Any]) -> ServiceResult:
        value = response.get("result")
        if not isinstance(value, dict):
            raise RuntimeError("remote_vlm_missing_result")
        return ServiceResult(
            status=str(value.get("status", "error")),
            data=value.get("data") if isinstance(value.get("data"), dict) else {},
            error=value.get("error"),
            model=value.get("model"),
            model_revision=value.get("model_revision"),
            latency_seconds=value.get("latency_seconds"),
            peak_vram_bytes=value.get("peak_vram_bytes"),
        )

    @staticmethod
    def _image_id(image_path: Path) -> str:
        image_id = image_path.name
        if not image_id:
            raise ValueError("remote_vlm_image_id_missing")
        return image_id

    def _image_payload(self, image_path: Path) -> dict[str, Any]:
        payload: dict[str, Any] = {"image_id": self._image_id(image_path)}
        if not self.inline_images:
            return payload
        content = image_path.read_bytes()
        payload.update(
            {
                "content_base64": base64.b64encode(content).decode("ascii"),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        return payload

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        try:
            health = self._request(
                "GET",
                "/health",
                timeout_seconds=self.health_timeout_seconds,
            )
            remote = health.get("vlm") if isinstance(health.get("vlm"), dict) else {}
            value = {
                "status": remote.get("status", "unavailable"),
                "loaded": bool(remote.get("loaded", False)),
                "model": remote.get("model"),
                "model_revision": remote.get("model_revision"),
                "endpoint": self.endpoint,
                "inline_images": self.inline_images,
                "core_prompt": self.prompt_identity(),
                "remote_library_items": health.get("library_items"),
            }
        except RuntimeError as exc:
            value = {
                "status": "unavailable",
                "loaded": False,
                "endpoint": self.endpoint,
                "inline_images": self.inline_images,
                "error": str(exc),
                "core_prompt": self.prompt_identity(),
            }
        self._last_status = value
        return value

    def cached_status(self) -> dict[str, Any]:
        """Return the last observed status without performing network I/O."""
        if self._last_status is not None:
            return dict(self._last_status)
        return {
            "status": "not_validated",
            "loaded": False,
            "model": None,
            "model_revision": None,
            "endpoint": self.endpoint,
            "inline_images": self.inline_images,
            "core_prompt": self.prompt_identity(),
            "remote_library_items": None,
        }

    def prompt_identity(self, prompt_version: str | None = None) -> dict[str, str]:
        """Execute the prompt identity operation."""
        prompt_id = prompt_version or self.core_prompt_version
        try:
            prompt = self.core_registry.prompts[prompt_id]
        except KeyError:
            raise ValueError(f"unknown core Prompt version: {prompt_id}") from None
        return {"prompt_id": prompt.prompt_id, "prompt_version": prompt.version, "prompt_sha256": prompt.bundle_sha256}

    def task_prompt_identity(self, prompt_id: str) -> dict[str, str]:
        """Execute the task prompt identity operation."""
        try:
            prompt = self.task_prompts[prompt_id]
        except KeyError:
            raise ValueError(f"unknown task Prompt: {prompt_id}") from None
        return {"prompt_id": prompt.prompt_id, "prompt_version": prompt.version, "prompt_sha256": prompt.sha256}

    def analyze_image(self, image_path: Path, prompt_version: str | None = None) -> ServiceResult:
        """Execute the analyze image operation."""
        return self._result(self._request("POST", "/analyze", {**self._image_payload(image_path), "prompt_version": prompt_version}))

    def answer_question(self, image_path: Path, question: str, evidence: dict[str, Any]) -> ServiceResult:
        """Execute the answer question operation."""
        return self._result(self._request("POST", "/vqa", {**self._image_payload(image_path), "question": question, "evidence": evidence}))

    def describe_image(self, image_path: Path, core_facts: dict[str, Any], options: dict[str, Any]) -> ServiceResult:
        """Execute the describe image operation."""
        return self._result(self._request("POST", "/describe", {**self._image_payload(image_path), "core_facts": core_facts, **options}))

    def generate_content(self, image_paths: Sequence[Path], facts: Sequence[dict[str, Any]], options: dict[str, Any]) -> ServiceResult:
        """Execute the generate content operation."""
        payload = {"images": [self._image_payload(path) for path in image_paths], "facts": list(facts), **options}
        return self._result(self._request("POST", "/generate", payload))

    def compare_images(self, image_paths: Sequence[Path], instruction: str | None = None) -> ServiceResult:
        """Execute the compare images operation."""
        return self._result(
            self._request(
                "POST",
                "/compare",
                {
                    "images": [self._image_payload(path) for path in image_paths],
                    "instruction": instruction or "比较这些图片的共同点与差异",
                },
            )
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
        if image_labels is not None and len(image_paths) != len(image_labels):
            raise ValueError("course_prompt_image_label_count_mismatch")
        payload = {
            "images": [self._image_payload(path) for path in image_paths],
            "prompt": prompt,
            "prompt_id": prompt_id,
            "prompt_sha256": prompt_sha256,
            "max_new_tokens": max_new_tokens,
        }
        if image_labels is not None:
            payload["image_labels"] = list(image_labels)
        if min_new_tokens is not None:
            payload["min_new_tokens"] = min_new_tokens
        return self._result(
            self._request(
                "POST",
                "/course-prompt",
                payload,
            )
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
        if len(image_paths) != len(image_labels):
            raise ValueError("multiturn_chat_image_label_count_mismatch")
        return self._result(
            self._request(
                "POST",
                "/multiturn-chat",
                {
                    "images": [self._image_payload(path) for path in image_paths],
                    "image_labels": list(image_labels),
                    "system_prompt": system_prompt,
                    "history_messages": list(history_messages),
                    "current_prompt": current_prompt,
                    "prompt_id": prompt_id,
                    "prompt_sha256": prompt_sha256,
                    "max_new_tokens": max_new_tokens,
                },
            )
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
        return self._result(
            self._request(
                "POST",
                "/conversation-repair",
                {
                    "prompt": prompt,
                    "prompt_id": prompt_id,
                    "prompt_sha256": prompt_sha256,
                    "max_new_tokens": max_new_tokens,
                },
            )
        )
