"""Replaceable service contracts for the Phase 1 application."""

from .contracts import (
    EmbeddingService,
    GroundingService,
    OCRService,
    SegmentationService,
    ServiceResult,
    VLMService,
)

__all__ = [
    "EmbeddingService",
    "GroundingService",
    "OCRService",
    "SegmentationService",
    "ServiceResult",
    "VLMService",
]
