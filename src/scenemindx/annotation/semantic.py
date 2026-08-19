"""Formal validator for the model-facing semantic payload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schema import AnnotationValidationError


SEMANTIC_PAYLOAD_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "data" / "schemas" / "semantic_payload_v1.schema.json"


def validate_semantic_payload(payload: Mapping[str, Any], path: str | Path = SEMANTIC_PAYLOAD_SCHEMA_PATH) -> None:
    """Validate semantic payload."""
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise AnnotationValidationError("semantic payload requires the formal jsonschema backend") from exc
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload), key=lambda error: [str(part) for part in error.absolute_path])
    if errors:
        details = "; ".join(f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}" for error in errors)
        raise AnnotationValidationError(details)
