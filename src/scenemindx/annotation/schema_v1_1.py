"""Draft v1.1 validator; v1 remains validated by the historical validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schema import AnnotationValidationError


SCHEMA_V1_1_PATH = Path(__file__).resolve().parents[3] / "data" / "schemas" / "visual_asset_annotation_v1_1.schema.json"


def load_annotation_schema_v1_1(path: str | Path = SCHEMA_V1_1_PATH) -> dict[str, Any]:
    """Load annotation schema v1 1."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnnotationValidationError("v1.1 schema root must be an object")
    return value


def validate_annotation_v1_1(annotation: Mapping[str, Any], path: str | Path = SCHEMA_V1_1_PATH) -> None:
    """Validate annotation v1 1."""
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise AnnotationValidationError("v1.1 requires the formal jsonschema backend") from exc
    errors = sorted(
        Draft202012Validator(load_annotation_schema_v1_1(path), format_checker=FormatChecker()).iter_errors(annotation),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        details = "; ".join(f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}" for error in errors)
        raise AnnotationValidationError(details)
