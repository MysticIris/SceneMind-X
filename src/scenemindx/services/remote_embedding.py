"""HTTP adapter for the private Qwen3-VL-Embedding-2B runtime."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class RemoteQwenVLEmbeddingService:
    """Encode product queries through the user-controlled private runtime."""

    model_id = "Qwen/Qwen3-VL-Embedding-2B"

    def __init__(
        self,
        endpoint: str,
        *,
        model_revision: str,
        timeout_seconds: float = 30.0,
        health_timeout_seconds: float = 5.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_revision = model_revision
        self.timeout_seconds = timeout_seconds
        self.health_timeout_seconds = max(0.5, float(health_timeout_seconds))
        self._last_status: dict[str, Any] | None = None
        self._last_latency_ms: float | None = None
        self._last_encode_latency_ms: float | None = None
        self._last_server_latency_ms: float | None = None

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
        started = time.perf_counter()
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
            raise RuntimeError(f"remote_e1_unreachable:{detail}") from exc
        self._last_latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("remote_e1_invalid_json") from exc
        if not isinstance(value, dict):
            raise RuntimeError("remote_e1_invalid_response")
        return value

    @staticmethod
    def _image_payload(image_path: Path) -> dict[str, Any]:
        content = image_path.read_bytes()
        if not content:
            raise ValueError("e1_query_image_empty")
        if len(content) > 20 * 1024 * 1024:
            raise ValueError("e1_query_image_too_large")
        return {
            "image_id": image_path.name,
            "content_base64": base64.b64encode(content).decode("ascii"),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    def _vector(self, value: dict[str, Any]) -> list[float]:
        vector = value.get("vector")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("remote_e1_missing_vector")
        try:
            result = [float(item) for item in vector]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("remote_e1_invalid_vector") from exc
        dimensions = value.get("dimensions")
        if dimensions is not None and int(dimensions) != len(result):
            raise RuntimeError("remote_e1_vector_dimension_mismatch")
        self._last_encode_latency_ms = self._last_latency_ms
        server_latency = value.get("latency_ms")
        self._last_server_latency_ms = (
            float(server_latency) if server_latency is not None else None
        )
        return result

    def load(self) -> dict[str, Any]:
        """Load the requested value."""
        status = self.status()
        if status.get("status") != "ready":
            raise RuntimeError(f"remote_e1_not_ready:{status.get('status')}")
        return status

    def unload_model(self) -> None:
        # The private worker is process-owned. Product shutdown must not stop it.
        """Execute the unload model operation."""
        return None

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        try:
            value = self._request(
                "GET",
                "/health",
                timeout_seconds=self.health_timeout_seconds,
            )
            remote = value.get("embedding") if isinstance(value.get("embedding"), dict) else {}
            status = {
                "status": remote.get("status", "unavailable"),
                "loaded": bool(remote.get("loaded", False)),
                "backend": "qwen3_vl_embedding_2b_remote",
                "model": remote.get("model", self.model_id),
                "model_revision": remote.get("model_revision", self.model_revision),
                "dimensions": remote.get("dimensions"),
                "device": remote.get("device"),
                "load_seconds": remote.get("load_seconds"),
                "peak_vram_bytes": remote.get("peak_vram_bytes"),
                "last_request_latency_ms": self._last_latency_ms,
                "last_encode_latency_ms": self._last_encode_latency_ms,
                "last_server_encode_latency_ms": self._last_server_latency_ms,
            }
            self._last_status = status
            return status
        except RuntimeError as exc:
            return {
                "status": "unavailable",
                "loaded": False,
                "backend": "qwen3_vl_embedding_2b_remote",
                "model": self.model_id,
                "model_revision": self.model_revision,
                "dimensions": 2048,
                "device": "remote_cuda",
                "error": str(exc),
                "last_known": self._last_status,
                "last_encode_latency_ms": self._last_encode_latency_ms,
                "last_server_encode_latency_ms": self._last_server_latency_ms,
            }

    def cached_status(self) -> dict[str, Any]:
        """Return the last observed status without contacting the remote worker."""
        if self._last_status is not None:
            return dict(self._last_status)
        return {
            "status": "not_validated",
            "loaded": False,
            "backend": "qwen3_vl_embedding_2b_remote",
            "model": self.model_id,
            "model_revision": self.model_revision,
            "dimensions": None,
            "last_request_latency_ms": self._last_latency_ms,
            "last_encode_latency_ms": self._last_encode_latency_ms,
            "last_server_encode_latency_ms": self._last_server_latency_ms,
        }

    def health(self) -> dict[str, Any]:
        """Execute the health operation."""
        return self.status()

    def encode_text(self, text: str) -> list[float]:
        """Execute the encode text operation."""
        if not text.strip():
            raise ValueError("e1_text_query_required")
        return self._vector(self._request("POST", "/embed/text", {"text": text}))

    def encode_image(self, image_path: Path) -> list[float]:
        """Execute the encode image operation."""
        return self._vector(self._request("POST", "/embed/image", self._image_payload(image_path)))

    def encode_multimodal(self, image_path: Path, text: str) -> list[float]:
        """Execute the encode multimodal operation."""
        if not text.strip():
            raise ValueError("e1_multimodal_text_required")
        payload = self._image_payload(image_path)
        payload["text"] = text
        return self._vector(self._request("POST", "/embed/multimodal", payload))
