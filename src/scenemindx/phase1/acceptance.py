"""Verify the controlled server-side copy of the frozen Phase 1 library."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class LibraryAcceptanceError(ValueError):
    """Raised when a server-side library copy differs from its frozen manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise LibraryAcceptanceError("frozen manifest is empty")
    return records


def verify_library_copy(manifest_path: Path, dataset_root: Path) -> dict[str, Any]:
    """Check the complete controlled copy without reading any source data directory."""

    manifest_path = manifest_path.resolve()
    dataset_root = dataset_root.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    records = _load_manifest(manifest_path)
    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        image_id = str(record.get("relative_path", ""))
        if Path(image_id).name != image_id or not image_id:
            raise LibraryAcceptanceError(f"unsafe image ID in manifest: {image_id!r}")
        if record.get("split") != "train":
            raise LibraryAcceptanceError(f"non-Train record in manifest: {image_id}")
        if image_id in expected:
            raise LibraryAcceptanceError(f"duplicate image ID in manifest: {image_id}")
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise LibraryAcceptanceError(f"missing SHA-256 in manifest: {image_id}")
        expected[image_id] = record

    actual = {
        path.relative_to(dataset_root).as_posix(): path
        for path in dataset_root.rglob("*")
        if path.is_file()
    }
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatches: list[dict[str, str]] = []
    verified: list[dict[str, Any]] = []
    for image_id in sorted(set(expected) & set(actual)):
        actual_hash = _sha256(actual[image_id])
        expected_hash = str(expected[image_id]["sha256"]).lower()
        if actual_hash.lower() != expected_hash:
            mismatches.append(
                {
                    "image_id": image_id,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                }
            )
        else:
            verified.append(
                {
                    "image_id": image_id,
                    "sha256": actual_hash,
                    "bytes": actual[image_id].stat().st_size,
                }
            )

    summary = {
        "status": "accepted" if not (missing or unexpected or mismatches) else "rejected",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "dataset_root": str(dataset_root),
        "expected_count": len(expected),
        "actual_file_count": len(actual),
        "verified_count": len(verified),
        "missing": missing,
        "unexpected": unexpected,
        "hash_mismatches": mismatches,
        "verified_files": verified,
    }
    if summary["status"] != "accepted":
        raise LibraryAcceptanceError(json.dumps(summary, ensure_ascii=False))
    return summary
