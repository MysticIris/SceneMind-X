"""Service contracts shared by the API and server workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class ServiceResult:
    """Represent service result data."""
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    model: str | None = None
    model_revision: str | None = None
    latency_seconds: float | None = None
    peak_vram_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Execute the as dict operation."""
        return {
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "model": self.model,
            "model_revision": self.model_revision,
            "latency_seconds": self.latency_seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
        }


class VLMService(Protocol):
    """Provide v l m service behavior."""
    def status(self) -> dict[str, Any]: ...

    def analyze_image(self, image_path: Path, prompt_version: str | None = None) -> ServiceResult: ...

    def describe_image(self, image_path: Path, core_facts: dict[str, Any], options: dict[str, Any]) -> ServiceResult: ...

    def task_prompt_identity(self, prompt_id: str) -> dict[str, str]: ...

    def answer_question(self, image_path: Path, question: str, evidence: dict[str, Any]) -> ServiceResult: ...

    def generate_content(
        self,
        image_paths: Sequence[Path],
        facts: Sequence[dict[str, Any]],
        options: dict[str, Any],
    ) -> ServiceResult: ...


class VLMProvider(VLMService, Protocol):
    """Provider identity/capability surface above the frozen task contract."""

    @property
    def provider_id(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def health_check(self) -> dict[str, Any]: ...

    def generate(self, request: dict[str, Any]) -> ServiceResult: ...

    def generate_structured(self, request: dict[str, Any]) -> ServiceResult: ...

    def capability_profile(self) -> dict[str, Any]: ...

    def usage(self) -> dict[str, Any]: ...

    def latency(self) -> dict[str, Any]: ...

    def error_mapping(self, error: Any) -> dict[str, Any]: ...

    def compare_images(self, image_paths: Sequence[Path], instruction: str | None = None) -> ServiceResult: ...

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
    ) -> ServiceResult: ...

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
    ) -> ServiceResult: ...


class EmbeddingService(Protocol):
    """Provide embedding service behavior."""
    def status(self) -> dict[str, Any]: ...

    def encode_image(self, image_path: Path) -> list[float]: ...

    def encode_text(self, text: str) -> list[float]: ...

    def encode_multimodal(self, image_path: Path, text: str) -> list[float]: ...


class EmbeddingProvider(EmbeddingService, Protocol):
    """Provider identity/capability surface for matching index identities."""

    @property
    def provider_id(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def normalization(self) -> str: ...

    def health_check(self) -> dict[str, Any]: ...

    def usage(self) -> dict[str, Any]: ...

    def error_mapping(self, error: Any) -> dict[str, Any]: ...


class OCRService(Protocol):
    """Provide o c r service behavior."""
    def recognize(self, image_path: Path) -> ServiceResult: ...


class GroundingService(Protocol):
    """Provide grounding service behavior."""
    def locate(self, image_path: Path, query: str) -> ServiceResult: ...


class SegmentationService(Protocol):
    """Provide segmentation service behavior."""
    def segment(self, image_path: Path, query: str) -> ServiceResult: ...


class DisabledOCRService:
    """Provide disabled o c r service behavior."""
    def recognize(self, image_path: Path) -> ServiceResult:
        """Execute the recognize operation."""
        return ServiceResult(status="disabled", error="ocr_service_disabled")


class DisabledGroundingService:
    """Provide disabled grounding service behavior."""
    def locate(self, image_path: Path, query: str) -> ServiceResult:
        """Execute the locate operation."""
        return ServiceResult(status="not_implemented", error="grounding_not_implemented")


class DisabledSegmentationService:
    """Provide disabled segmentation service behavior."""
    def segment(self, image_path: Path, query: str) -> ServiceResult:
        """Execute the segment operation."""
        return ServiceResult(status="not_implemented", error="segmentation_not_implemented")
