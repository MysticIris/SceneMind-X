"""Structured-decoding boundary with explicit unavailable/failure handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ConstraintDecoderError(RuntimeError):
    """Raised when a constrained decoder is unavailable or rejects generation."""


@dataclass(frozen=True)
class DecoderResult:
    """Represent decoder result data."""
    raw_text: str
    constrained: bool
    decoder_name: str
    decoder_version: str | None
    failure: str | None = None


class ConstraintDecoder(Protocol):
    """Provide constraint decoder behavior."""
    name: str
    version: str

    def generate(self, image_path: str, prompt: str, schema: dict[str, Any]) -> DecoderResult:
        """Execute the generate operation."""
        ...


def require_audited_decoder(decoder: ConstraintDecoder | None) -> ConstraintDecoder:
    """Execute the require audited decoder operation."""
    if decoder is None:
        raise ConstraintDecoderError("no audited constrained decoder is installed; use semantic payload assembly")
    return decoder
