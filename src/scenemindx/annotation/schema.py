"""Load and validate the frozen visual asset annotation schema."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "schemas"
    / "visual_asset_annotation_v1.schema.json"
)


class AnnotationValidationError(ValueError):
    """Raised when an annotation does not satisfy the frozen schema."""


def load_annotation_schema(path: str | Path = SCHEMA_PATH) -> dict[str, Any]:
    """Read the schema from disk without changing it."""

    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise AnnotationValidationError("annotation schema root must be an object")
    return schema


def validate_annotation(
    annotation: Mapping[str, Any], path: str | Path = SCHEMA_PATH
) -> None:
    """Validate one annotation and report all deterministic schema errors."""

    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError:
        _validate_without_optional_dependency(annotation)
        return

    validator = Draft202012Validator(
        load_annotation_schema(path), format_checker=FormatChecker()
    )
    errors = sorted(
        validator.iter_errors(annotation),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: "
            f"{error.message}"
            for error in errors
        )
        raise AnnotationValidationError(details)


def _validate_without_optional_dependency(annotation: Mapping[str, Any]) -> None:
    """Validate the frozen contract with the standard library when jsonschema is absent."""

    if not isinstance(annotation, dict):
        raise AnnotationValidationError("<root>: annotation must be an object")
    required = set(load_annotation_schema()["required"])
    missing = required - set(annotation)
    if missing:
        raise AnnotationValidationError(f"<root>: missing required fields {sorted(missing)}")
    unexpected = set(annotation) - required
    if unexpected:
        raise AnnotationValidationError(f"<root>: unexpected fields {sorted(unexpected)}")
    if annotation["schema_version"] != "visual_asset_annotation_v1":
        raise AnnotationValidationError("schema_version: invalid frozen schema identity")
    if annotation["source_split"] not in {"train", "val"}:
        raise AnnotationValidationError("source_split: expected train or val")
    if not re.fullmatch(r"[0-9a-f]{64}", annotation["source_hash"]):
        raise AnnotationValidationError("source_hash: expected lowercase SHA-256")
    if not re.fullmatch(r"P[0-9]+(?:\.[0-9]+)?", annotation["prompt_version"]):
        raise AnnotationValidationError("prompt_version: invalid version")
    try:
        datetime.fromisoformat(str(annotation["created_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise AnnotationValidationError("created_at: expected ISO-8601 date-time") from exc

    model = annotation["model"]
    if not isinstance(model, dict) or set(model) != {"name", "version", "path", "license"}:
        raise AnnotationValidationError("model: expected the four traceability fields")
    generation = annotation["generation_config"]
    generation_required = {"seed", "temperature", "top_p", "max_tokens"}
    if not isinstance(generation, dict) or not generation_required.issubset(generation):
        raise AnnotationValidationError("generation_config: missing reproducibility fields")
    if generation["temperature"] < 0 or not 0 < generation["top_p"] <= 1 or generation["max_tokens"] < 1:
        raise AnnotationValidationError("generation_config: invalid sampling bounds")
    if not isinstance(annotation["main_subjects"], list) or not isinstance(annotation["activities"], list):
        raise AnnotationValidationError("main_subjects/activities: expected arrays")
    quality = annotation["image_quality"]
    if not isinstance(quality, dict) or quality.get("overall") not in {"high", "medium", "low", "unknown"}:
        raise AnnotationValidationError("image_quality: invalid overall value")
    for section in ("evidence", "uncertainty", "downstream"):
        if not isinstance(annotation[section], dict):
            raise AnnotationValidationError(f"{section}: expected an object")
    for entity in annotation["entities"]:
        if not isinstance(entity, dict) or not re.fullmatch(r"e[0-9]+", str(entity.get("entity_id", ""))):
            raise AnnotationValidationError("entities: invalid entity_id")
        if not 0 <= entity.get("confidence", -1) <= 1:
            raise AnnotationValidationError("entities: confidence must be in [0, 1]")
    for item in annotation["ocr"]:
        if not isinstance(item, dict) or not item.get("text") or not 0 <= item.get("confidence", -1) <= 1:
            raise AnnotationValidationError("ocr: invalid item")
