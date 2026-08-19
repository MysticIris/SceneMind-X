"""Course dataset registry validation and purpose-based split access guards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


class DatasetRegistryError(ValueError):
    """Base error for invalid registry or manifest state."""


class AccessDeniedError(DatasetRegistryError):
    """Raised when a purpose is not authorized for a split."""


class DatasetUnavailableError(DatasetRegistryError):
    """Raised when a registered split or source file is unavailable."""


class DatasetIntegrityError(DatasetRegistryError):
    """Raised when manifest or file integrity validation fails."""


@dataclass(frozen=True)
class ManifestRecord:
    """Provide manifest record behavior."""
    dataset_version: str
    split: str
    image_id: str
    relative_path: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    format: str
    integrity_status: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ManifestRecord":
        """Execute the from mapping operation."""
        required = {field for field in cls.__dataclass_fields__}
        missing = required - value.keys()
        if missing:
            raise DatasetIntegrityError(
                f"manifest record missing fields: {sorted(missing)}"
            )
        try:
            record = cls(**{field: value[field] for field in required})
        except TypeError as exc:
            raise DatasetIntegrityError(f"invalid manifest record: {exc}") from exc
        if record.size_bytes < 1 or record.width < 1 or record.height < 1:
            raise DatasetIntegrityError(
                f"invalid image dimensions or size for {record.image_id}"
            )
        if len(record.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in record.sha256
        ):
            raise DatasetIntegrityError(f"invalid SHA-256 for {record.image_id}")
        if record.integrity_status != "ok":
            raise DatasetIntegrityError(
                f"non-ok integrity status for {record.image_id}: "
                f"{record.integrity_status}"
            )
        _validate_relative_path(record.relative_path, record.split)
        return record


def _validate_relative_path(relative_path: str, split: str) -> PurePosixPath:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or "\\" in relative_path or ".." in path.parts:
        raise DatasetIntegrityError(f"unsafe relative path: {relative_path}")
    if len(path.parts) < 2 or path.parts[0] != split:
        raise DatasetIntegrityError(
            f"manifest split/path mismatch: split={split}, path={relative_path}"
        )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, expected_split: str) -> tuple[ManifestRecord, ...]:
    """Load manifest."""
    if not path.is_file():
        raise DatasetUnavailableError(f"manifest does not exist: {path}")
    records: list[ManifestRecord] = []
    image_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetIntegrityError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise DatasetIntegrityError(
                    f"manifest record must be an object at {path}:{line_number}"
                )
            record = ManifestRecord.from_mapping(value)
            if record.split != expected_split:
                raise DatasetIntegrityError(
                    f"unexpected split {record.split} in {expected_split} manifest"
                )
            if record.image_id in image_ids:
                raise DatasetIntegrityError(f"duplicate image_id: {record.image_id}")
            image_ids.add(record.image_id)
            records.append(record)
    return tuple(records)


class DatasetRegistry:
    """Validated registry with fail-closed, purpose-specific split access."""

    def __init__(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.path = path.resolve()
        self.base_dir = self.path.parent
        self.payload = dict(payload)
        self.dataset_version = str(payload.get("dataset_version", ""))
        self.splits = payload.get("splits")
        self.access_policy = payload.get("access_policy")
        if not self.dataset_version or not isinstance(self.splits, dict):
            raise DatasetRegistryError("registry requires dataset_version and splits")
        if not isinstance(self.access_policy, dict):
            raise DatasetRegistryError("registry requires access_policy")

    @classmethod
    def load(cls, path: str | Path) -> "DatasetRegistry":
        """Load the requested value."""
        resolved = Path(path)
        if not resolved.is_file():
            raise DatasetUnavailableError(f"registry does not exist: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise DatasetRegistryError("registry root must be an object")
        return cls(resolved, payload)

    def authorize(
        self,
        *,
        purpose: str,
        split: str,
        explicit_model_selection_authorization: bool = False,
    ) -> None:
        """Execute the authorize operation."""
        split_entry = self.splits.get(split)
        if not isinstance(split_entry, dict):
            raise DatasetUnavailableError(f"split is not registered: {split}")
        status = split_entry.get("status")
        if status != "available":
            raise DatasetUnavailableError(f"split {split} is {status}")
        rule = self.access_policy.get(purpose)
        if not isinstance(rule, dict):
            raise AccessDeniedError(f"unknown or denied dataset purpose: {purpose}")
        allowed_splits = rule.get("allowed_splits", [])
        if split not in allowed_splits:
            raise AccessDeniedError(f"purpose {purpose} cannot access split {split}")
        if rule.get("requires_explicit_authorization") and not (
            purpose == "model_selection" and explicit_model_selection_authorization
        ):
            raise AccessDeniedError(
                f"purpose {purpose} requires explicit later authorization"
            )

    def records_for(
        self,
        *,
        purpose: str,
        split: str,
        roots: Mapping[str, str | Path],
        explicit_model_selection_authorization: bool = False,
        verify_files: bool = True,
    ) -> tuple[ManifestRecord, ...]:
        """Execute the records for operation."""
        self.authorize(
            purpose=purpose,
            split=split,
            explicit_model_selection_authorization=(
                explicit_model_selection_authorization
            ),
        )
        split_entry = self.splits[split]
        manifest_value = split_entry.get("manifest")
        if not isinstance(manifest_value, str):
            raise DatasetUnavailableError(f"split {split} has no manifest")
        manifest_path = (self.base_dir / manifest_value).resolve()
        data_root = self.base_dir.parent.resolve()
        if not manifest_path.is_relative_to(data_root):
            raise DatasetIntegrityError(
                f"manifest path escapes registry data root: {manifest_value}"
            )
        expected_manifest_sha = split_entry.get("manifest_sha256")
        if expected_manifest_sha and _sha256_file(manifest_path) != expected_manifest_sha:
            raise DatasetIntegrityError(f"manifest SHA-256 mismatch for split {split}")
        records = load_manifest(manifest_path, split)
        expected_count = split_entry.get("image_count")
        if expected_count != len(records):
            raise DatasetIntegrityError(
                f"split {split} count mismatch: registry={expected_count}, "
                f"manifest={len(records)}"
            )
        if any(record.dataset_version != self.dataset_version for record in records):
            raise DatasetIntegrityError("manifest dataset_version mismatch")
        expected_bytes = split_entry.get("total_bytes")
        actual_bytes = sum(record.size_bytes for record in records)
        if expected_bytes is not None and expected_bytes != actual_bytes:
            raise DatasetIntegrityError(
                f"split {split} byte total mismatch: registry={expected_bytes}, "
                f"manifest={actual_bytes}"
            )
        if verify_files:
            self.verify_files(records, split=split, roots=roots)
        return records

    @staticmethod
    def verify_files(
        records: Iterable[ManifestRecord],
        *,
        split: str,
        roots: Mapping[str, str | Path],
    ) -> None:
        """Execute the verify files operation."""
        root_value = roots.get(split)
        if root_value is None:
            raise DatasetUnavailableError(f"no source root configured for split {split}")
        root = Path(root_value).resolve()
        if not root.is_dir():
            raise DatasetUnavailableError(f"source root does not exist: {root}")
        for record in records:
            relative = _validate_relative_path(record.relative_path, split)
            candidate = root.joinpath(*relative.parts[1:]).resolve()
            if not candidate.is_relative_to(root):
                raise DatasetIntegrityError(f"path escapes source root: {relative}")
            if not candidate.is_file():
                raise DatasetUnavailableError(f"dataset file does not exist: {relative}")
            if candidate.stat().st_size != record.size_bytes:
                raise DatasetIntegrityError(f"size mismatch: {relative}")
            if _sha256_file(candidate) != record.sha256:
                raise DatasetIntegrityError(f"SHA-256 mismatch: {relative}")
