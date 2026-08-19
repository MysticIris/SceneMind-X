"""Deterministic transport-only preprocessing for Bailian image requests.

The original managed asset and the semantic embedding identity never change.
Only an oversized request receives a cached transport derivative. Image
payloads remain in memory inside the provider and are never persisted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from PIL import Image, ImageOps


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class PreparedCloudImage:
    """Provide prepared cloud image behavior."""
    request_path: Path
    profile_id: str
    profile_version: str
    cache_hit: bool
    audit: dict[str, Any]


@dataclass(frozen=True)
class PreparedCloudImageBatch:
    """Provide prepared cloud image batch behavior."""
    request_paths: tuple[Path, ...]
    items: tuple[PreparedCloudImage, ...]
    fallback_reason: str | None
    original_preserved: bool
    provider_profile: str = "default"
    transport_budget_bytes: int = 0
    estimated_request_body_bytes: int = 0

    @property
    def used_compatible_transport(self) -> bool:
        """Execute the used compatible transport operation."""
        return any(
            item.profile_id != "default_transport_v1"
            for item in self.items
        )

    def public_audit(self) -> dict[str, Any]:
        """Execute the public audit operation."""
        return {
            "item_count": len(self.items),
            "used_compatible_transport": self.used_compatible_transport,
            "profiles": [item.profile_id for item in self.items],
            "fallback_reason": self.fallback_reason,
            "original_preserved": self.original_preserved,
            "provider_profile": self.provider_profile,
            "transport_budget_bytes": self.transport_budget_bytes,
            "estimated_request_body_bytes": (
                self.estimated_request_body_bytes
            ),
            "request_bytes": sum(
                int(item.audit.get("transport_file_size", 0) or 0)
                for item in self.items
            ),
        }


@dataclass(frozen=True)
class CloudTransportEmbeddingResult:
    """Represent cloud transport embedding result data."""
    vector: list[float]
    prepared: PreparedCloudImage
    request_count: int
    fallback_from_413: bool


class CloudTransportBudgetExceeded(RuntimeError):
    """No configured compatibility level can satisfy the transport budget."""

    code = "CLOUD_TRANSPORT_BUDGET_EXHAUSTED"
    public_message = (
        "图片在有界兼容处理后仍超过当前模型的传输限制，未发送模型请求；"
        "原图和已有结果均已保留。"
    )

    def __init__(
        self,
        *,
        reason: str,
        provider_profile: str,
        transport_budget_bytes: int,
        attempts: list[dict[str, Any]],
    ) -> None:
        super().__init__(self.public_message)
        self.reason = reason
        self.provider_profile = provider_profile
        self.transport_budget_bytes = int(transport_budget_bytes)
        self.attempts = [dict(item) for item in attempts]


class CloudImageTransportPreprocessor:
    """Select and materialize a deterministic cloud transport profile."""

    def __init__(self, *, config_path: Path, cache_root: Path) -> None:
        self.config_path = config_path.resolve()
        self.cache_root = cache_root.resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        if self.config.get("schema_version") != (
            "scenemindx_cloud_image_transport_profiles_v1"
        ):
            raise ValueError("cloud_transport_config_schema_mismatch")
        self.default_profile_id = str(self.config["default_profile_id"])
        self.oversized_profile_id = str(self.config["oversized_profile_id"])
        self.profiles = dict(self.config["profiles"])
        self.compatible_profile_ids = tuple(
            str(value)
            for value in self.config.get(
                "compatible_profile_ids",
                [self.oversized_profile_id],
            )
        )
        self.provider_profiles = dict(
            self.config.get(
                "provider_profiles",
                {
                    "default": {
                        "compatible_profile_ids": list(
                            self.compatible_profile_ids
                        ),
                        "transport_budget_bytes": self.config[
                            "preflight"
                        ]["max_estimated_request_body_bytes"],
                    }
                },
            )
        )
        self.preflight = dict(self.config["preflight"])
        self.fallback = dict(self.config["fallback"])
        if (
            self.default_profile_id not in self.profiles
            or self.oversized_profile_id not in self.profiles
            or not self.compatible_profile_ids
            or any(
                profile_id not in self.profiles
                for profile_id in self.compatible_profile_ids
            )
        ):
            raise ValueError("cloud_transport_profile_missing")
        configured_levels = [
            int(
                self.profiles[profile_id].get(
                    "compatibility_level",
                    index,
                )
            )
            for index, profile_id in enumerate(
                self.compatible_profile_ids,
                start=1,
            )
        ]
        if configured_levels != list(range(1, len(configured_levels) + 1)):
            raise ValueError("cloud_transport_profile_levels_invalid")

    def provider_settings(self, provider_profile: str) -> dict[str, Any]:
        """Resolve a provider override without inventing provider limits."""

        requested = (
            provider_profile
            if provider_profile in self.provider_profiles
            else "default"
        )

        def resolve(
            profile_id: str,
            stack: tuple[str, ...] = (),
        ) -> dict[str, Any]:
            if profile_id in stack:
                raise ValueError("cloud_transport_provider_inheritance_cycle")
            value = dict(self.provider_profiles.get(profile_id, {}))
            inherited = value.pop("inherits", None)
            base = (
                resolve(str(inherited), (*stack, profile_id))
                if inherited
                else {}
            )
            return {**base, **value}

        settings = resolve(requested)
        profile_ids = tuple(
            str(value)
            for value in settings.get(
                "compatible_profile_ids",
                self.compatible_profile_ids,
            )
        )
        if not profile_ids or any(
            profile_id not in self.profiles
            for profile_id in profile_ids
        ):
            raise ValueError("cloud_transport_provider_profiles_invalid")
        return {
            **settings,
            "provider_profile": requested,
            "compatible_profile_ids": profile_ids,
            "transport_budget_bytes": int(
                settings.get(
                    "transport_budget_bytes",
                    self.preflight["max_estimated_request_body_bytes"],
                )
            ),
            "request_overhead_bytes": int(
                settings.get(
                    "request_overhead_bytes",
                    self.preflight["request_overhead_bytes"],
                )
            ),
        }

    def profile_level(
        self,
        profile_id: str,
        *,
        provider_profile: str = "default",
    ) -> int:
        """Execute the profile level operation."""
        if profile_id == self.default_profile_id:
            return 0
        profile_ids = self.provider_settings(provider_profile)[
            "compatible_profile_ids"
        ]
        try:
            return profile_ids.index(profile_id) + 1
        except ValueError:
            return len(profile_ids) + 1

    def max_compatible_level(
        self,
        *,
        provider_profile: str = "default",
    ) -> int:
        """Execute the max compatible level operation."""
        return len(
            self.provider_settings(provider_profile)[
                "compatible_profile_ids"
            ]
        )

    @staticmethod
    def _estimated_request_body_bytes(source_bytes: int, overhead: int) -> int:
        base64_bytes = 4 * ((int(source_bytes) + 2) // 3)
        return base64_bytes + int(overhead)

    def inspect(self, image_path: Path) -> dict[str, Any]:
        """Execute the inspect operation."""
        path = image_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        source_bytes = path.stat().st_size
        digest = _sha256_file(path)
        with Image.open(path) as image:
            width, height = image.size
            encoded_format = str(image.format or "UNKNOWN").upper()
            mode = str(image.mode)
            orientation = image.getexif().get(274)
        long_edge = max(width, height)
        short_edge = min(width, height)
        pixels = width * height
        estimated = self._estimated_request_body_bytes(
            source_bytes,
            int(self.preflight["request_overhead_bytes"]),
        )
        triggers = {
            "estimated_request_body_bytes": estimated
            > int(self.preflight["max_estimated_request_body_bytes"]),
            "source_file_bytes": source_bytes
            > int(self.preflight["max_source_file_bytes"]),
            "long_edge": long_edge
            > int(self.preflight["max_original_long_edge"]),
            "short_edge": short_edge
            > int(self.preflight["max_original_short_edge"]),
            "pixel_count": pixels
            > int(self.preflight["max_original_pixels"]),
        }
        return {
            "original_sha256": digest,
            "original_width": width,
            "original_height": height,
            "original_long_edge": long_edge,
            "original_short_edge": short_edge,
            "original_pixels": pixels,
            "original_file_size": source_bytes,
            "original_format": encoded_format,
            "original_mode": mode,
            "original_exif_orientation": orientation,
            "estimated_request_body_bytes": estimated,
            "preflight_triggers": triggers,
            "preflight_oversized": any(triggers.values()),
        }

    def _cache_paths(
        self, *, original_sha256: str, profile_id: str, profile_version: str
    ) -> tuple[Path, Path]:
        root = self.cache_root / profile_id / f"v{profile_version}"
        return (
            root / original_sha256[:2] / f"{original_sha256}.jpg",
            root / original_sha256[:2] / f"{original_sha256}.json",
        )

    def _validated_cached(
        self,
        *,
        image_path: Path,
        metadata_path: Path,
        original_sha256: str,
        profile_id: str,
        profile_version: str,
    ) -> dict[str, Any] | None:
        if not image_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if (
            metadata.get("original_sha256") != original_sha256
            or metadata.get("transport_profile_id") != profile_id
            or str(metadata.get("transport_profile_version")) != profile_version
            or metadata.get("transport_sha256") != _sha256_file(image_path)
            or int(metadata.get("transport_file_size", -1))
            != image_path.stat().st_size
        ):
            return None
        return metadata

    @staticmethod
    def _rgb(image: Image.Image, background: str) -> Image.Image:
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            color = background.lstrip("#")
            if len(color) != 6:
                raise ValueError("cloud_transport_background_invalid")
            canvas = Image.new(
                "RGBA",
                rgba.size,
                (
                    int(color[0:2], 16),
                    int(color[2:4], 16),
                    int(color[4:6], 16),
                    255,
                ),
            )
            canvas.alpha_composite(rgba)
            return canvas.convert("RGB")
        return image.convert("RGB")

    @staticmethod
    def _target_size(
        width: int, height: int, *, max_long_edge: int, max_pixels: int
    ) -> tuple[int, int]:
        scale = min(
            1.0,
            max_long_edge / max(width, height),
            math.sqrt(max_pixels / (width * height)),
        )
        return (
            max(1, int(math.floor(width * scale))),
            max(1, int(math.floor(height * scale))),
        )

    def _prepare_profile(
        self,
        path: Path,
        *,
        original: dict[str, Any],
        original_asset_id: str,
        profile_id: str,
        fallback_reason: str | None = None,
    ) -> PreparedCloudImage:
        profile = dict(self.profiles[profile_id])
        profile_version = str(profile["profile_version"])
        if profile_id == self.default_profile_id:
            transport_size = int(original["original_file_size"])
            audit = {
                **original,
                "schema_version": "scenemindx_cloud_transport_audit_v1",
                "original_asset_id": original_asset_id,
                "transport_profile_id": profile_id,
                "transport_profile_version": profile_version,
                "transport_sha256": original["original_sha256"],
                "transport_width": original["original_width"],
                "transport_height": original["original_height"],
                "transport_pixels": original["original_pixels"],
                "transport_file_size": transport_size,
                "transport_base64_bytes": (
                    4 * ((transport_size + 2) // 3)
                ),
                "transport_format": original["original_format"],
                "cache_hit": True,
                "fallback_reason": fallback_reason,
                "original_preserved": True,
                "created_at": None,
            }
            return PreparedCloudImage(
                request_path=path,
                profile_id=profile_id,
                profile_version=profile_version,
                cache_hit=True,
                audit=audit,
            )

        cache_path, metadata_path = self._cache_paths(
            original_sha256=original["original_sha256"],
            profile_id=profile_id,
            profile_version=profile_version,
        )
        cached = self._validated_cached(
            image_path=cache_path,
            metadata_path=metadata_path,
            original_sha256=original["original_sha256"],
            profile_id=profile_id,
            profile_version=profile_version,
        )
        if cached is not None:
            return PreparedCloudImage(
                request_path=cache_path,
                profile_id=profile_id,
                profile_version=profile_version,
                cache_hit=True,
                audit={**cached, "cache_hit": True, "fallback_reason": fallback_reason},
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(
            f"{cache_path.name}.tmp.{os.getpid()}"
        )
        try:
            with Image.open(path) as source:
                oriented = ImageOps.exif_transpose(source)
                rgb = self._rgb(
                    oriented,
                    str(profile["transparency_background"]),
                )
                target = self._target_size(
                    *rgb.size,
                    max_long_edge=int(profile["max_long_edge"]),
                    max_pixels=int(profile["max_pixels"]),
                )
                derived = (
                    rgb
                    if rgb.size == target
                    else rgb.resize(target, Image.Resampling.LANCZOS)
                )
                derived.save(
                    temporary,
                    format="JPEG",
                    quality=int(profile["jpeg_quality"]),
                    subsampling=int(profile["jpeg_subsampling"]),
                    optimize=bool(profile["jpeg_optimize"]),
                    progressive=bool(profile["jpeg_progressive"]),
                    exif=b"",
                )
            transport_sha256 = _sha256_file(temporary)
            os.replace(temporary, cache_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        if _sha256_file(path) != original["original_sha256"]:
            raise ValueError("cloud_transport_original_sha_changed")
        with Image.open(cache_path) as transported:
            transport_width, transport_height = transported.size
            transport_format = str(transported.format or "UNKNOWN").upper()
        metadata = {
            **original,
            "schema_version": "scenemindx_cloud_transport_audit_v1",
            "original_asset_id": original_asset_id,
            "transport_profile_id": profile_id,
            "transport_profile_version": profile_version,
            "transport_sha256": transport_sha256,
            "transport_width": transport_width,
            "transport_height": transport_height,
            "transport_pixels": transport_width * transport_height,
            "transport_file_size": cache_path.stat().st_size,
            "transport_estimated_request_body_bytes": (
                self._estimated_request_body_bytes(
                    cache_path.stat().st_size,
                    int(self.preflight["request_overhead_bytes"]),
                )
            ),
            "transport_format": transport_format,
            "profile": profile,
            "cache_hit": False,
            "fallback_reason": fallback_reason,
            "original_preserved": True,
            "created_at": _now_iso(),
            "contains_image_payload": False,
        }
        _atomic_json(metadata_path, metadata)
        return PreparedCloudImage(
            request_path=cache_path,
            profile_id=profile_id,
            profile_version=profile_version,
            cache_hit=False,
            audit=metadata,
        )

    def _post_encode_validation(
        self,
        prepared: PreparedCloudImage,
        *,
        provider_profile: str,
    ) -> PreparedCloudImage:
        settings = self.provider_settings(provider_profile)
        profile = dict(self.profiles[prepared.profile_id])
        request_path = prepared.request_path.resolve()
        transport_size = request_path.stat().st_size
        with Image.open(request_path) as transported:
            width, height = transported.size
            transport_format = str(
                transported.format or "UNKNOWN"
            ).upper()
        pixels = width * height
        base64_bytes = 4 * ((transport_size + 2) // 3)
        estimated_body = (
            base64_bytes + int(settings["request_overhead_bytes"])
        )
        if prepared.profile_id == self.default_profile_id:
            max_long_edge = None
            max_pixels = None
            max_encoded_bytes = max(
                0,
                (
                    (
                        int(settings["transport_budget_bytes"])
                        - int(settings["request_overhead_bytes"])
                    )
                    * 3
                )
                // 4,
            )
            expected_format = None
        else:
            max_long_edge = int(profile["max_long_edge"])
            max_pixels = int(profile["max_pixels"])
            max_encoded_bytes = int(
                profile.get(
                    "max_encoded_bytes",
                    max(
                        0,
                        (
                            (
                                int(settings["transport_budget_bytes"])
                                - int(settings["request_overhead_bytes"])
                            )
                            * 3
                        )
                        // 4,
                    ),
                )
            )
            expected_format = str(
                profile.get("output_format") or "JPEG"
            ).upper()
        checks = {
            "positive_dimensions": width > 0 and height > 0,
            "max_long_edge": (
                True
                if max_long_edge is None
                else max(width, height) <= max_long_edge
            ),
            "max_pixels": (
                True if max_pixels is None else pixels <= max_pixels
            ),
            "encoded_bytes": transport_size <= max_encoded_bytes,
            "estimated_base64_body": (
                estimated_body
                <= int(settings["transport_budget_bytes"])
            ),
            "output_format": (
                True
                if expected_format is None
                else transport_format == expected_format
            ),
        }
        validation = {
            "passed": all(checks.values()),
            "checks": checks,
            "provider_profile": settings["provider_profile"],
            "compatibility_level": self.profile_level(
                prepared.profile_id,
                provider_profile=provider_profile,
            ),
            "transport_budget_bytes": int(
                settings["transport_budget_bytes"]
            ),
            "max_encoded_bytes": max_encoded_bytes,
            "actual_encoded_bytes": transport_size,
            "estimated_base64_bytes": base64_bytes,
            "estimated_request_body_bytes": estimated_body,
            "actual_width": width,
            "actual_height": height,
            "actual_pixels": pixels,
        }
        return PreparedCloudImage(
            request_path=request_path,
            profile_id=prepared.profile_id,
            profile_version=prepared.profile_version,
            cache_hit=prepared.cache_hit,
            audit={
                **prepared.audit,
                "transport_width": width,
                "transport_height": height,
                "transport_pixels": pixels,
                "transport_file_size": transport_size,
                "transport_base64_bytes": base64_bytes,
                "transport_estimated_request_body_bytes": estimated_body,
                "transport_format": transport_format,
                "provider_profile": settings["provider_profile"],
                "compatibility_level": validation[
                    "compatibility_level"
                ],
                "post_encode_validation": validation,
            },
        )

    def prepare(
        self,
        image_path: Path,
        *,
        original_asset_id: str,
        force_oversized: bool = False,
        fallback_reason: str | None = None,
        provider_profile: str = "default",
        minimum_compatible_level: int = 1,
    ) -> PreparedCloudImage:
        """Return only a transport input that passes post-encode validation."""

        path = image_path.resolve()
        original = self.inspect(path)
        settings = self.provider_settings(provider_profile)
        profile_ids = settings["compatible_profile_ids"]
        attempts: list[dict[str, Any]] = []

        if not force_oversized and not original["preflight_oversized"]:
            passthrough = self._post_encode_validation(
                self._prepare_profile(
                    path,
                    original=original,
                    original_asset_id=original_asset_id,
                    profile_id=self.default_profile_id,
                    fallback_reason=fallback_reason,
                ),
                provider_profile=provider_profile,
            )
            validation = dict(
                passthrough.audit.get("post_encode_validation") or {}
            )
            if validation.get("passed"):
                return passthrough
            attempts.append(
                {
                    "profile_id": passthrough.profile_id,
                    "compatibility_level": 0,
                    "validation": validation,
                }
            )

        start_level = max(1, int(minimum_compatible_level))
        for level, profile_id in enumerate(profile_ids, start=1):
            if level < start_level:
                continue
            candidate = self._post_encode_validation(
                self._prepare_profile(
                    path,
                    original=original,
                    original_asset_id=original_asset_id,
                    profile_id=profile_id,
                    fallback_reason=fallback_reason,
                ),
                provider_profile=provider_profile,
            )
            validation = dict(
                candidate.audit.get("post_encode_validation") or {}
            )
            attempts.append(
                {
                    "profile_id": profile_id,
                    "compatibility_level": level,
                    "validation": validation,
                }
            )
            if validation.get("passed"):
                return candidate

        raise CloudTransportBudgetExceeded(
            reason="single_image_post_encode_budget_exhausted",
            provider_profile=str(settings["provider_profile"]),
            transport_budget_bytes=int(
                settings["transport_budget_bytes"]
            ),
            attempts=attempts,
        )

    def prepare_batch(
        self,
        image_paths: list[Path] | tuple[Path, ...],
        *,
        force_oversized: bool = False,
        fallback_reason: str | None = None,
        provider_profile: str = "default",
        minimum_compatible_level: int = 1,
    ) -> PreparedCloudImageBatch:
        """Prepare one bounded request without changing any source asset.

        Per-image limits are applied first. If the combined inline payload
        would still exceed the configured request-body ceiling, the largest
        remaining originals are converted one by one until the batch fits.
        This matters for multi-image VLM requests where every individual file
        may be valid while their combined Base64 body is not.
        """

        settings = self.provider_settings(provider_profile)
        originals = [Path(path).resolve() for path in image_paths]
        prepared = [
            self.prepare(
                path,
                original_asset_id=f"inference_input_{index}",
                force_oversized=force_oversized,
                fallback_reason=fallback_reason,
                provider_profile=provider_profile,
                minimum_compatible_level=minimum_compatible_level,
            )
            for index, path in enumerate(originals, start=1)
        ]
        limit = int(settings["transport_budget_bytes"])
        overhead = int(settings["request_overhead_bytes"])

        def estimated_total() -> int:
            return overhead + sum(
                int(
                    item.audit.get("transport_base64_bytes")
                    or (
                        4
                        * (
                            (
                                int(
                                    item.audit.get(
                                        "transport_file_size",
                                        0,
                                    )
                                    or 0
                                )
                                + 2
                            )
                            // 3
                        )
                    )
                )
                for item in prepared
            )

        batch_attempts: list[dict[str, Any]] = []
        max_level = len(settings["compatible_profile_ids"])
        while estimated_total() > limit:
            candidates = [
                (
                    -int(
                        item.audit.get("transport_file_size", 0) or 0
                    ),
                    index,
                    self.profile_level(
                        item.profile_id,
                        provider_profile=provider_profile,
                    ),
                )
                for index, item in enumerate(prepared)
                if self.profile_level(
                    item.profile_id,
                    provider_profile=provider_profile,
                )
                < max_level
            ]
            candidates.sort()
            if not candidates:
                raise CloudTransportBudgetExceeded(
                    reason="batch_post_encode_budget_exhausted",
                    provider_profile=str(settings["provider_profile"]),
                    transport_budget_bytes=limit,
                    attempts=[
                        *batch_attempts,
                        {
                            "estimated_request_body_bytes": (
                                estimated_total()
                            ),
                            "profiles": [
                                item.profile_id for item in prepared
                            ],
                        },
                    ],
                )
            _, index, current_level = candidates[0]
            before_total = estimated_total()
            prepared[index] = self.prepare(
                originals[index],
                original_asset_id=f"inference_input_{index + 1}",
                force_oversized=True,
                fallback_reason=(
                    fallback_reason
                    or "combined_request_body_preflight"
                ),
                provider_profile=provider_profile,
                minimum_compatible_level=current_level + 1,
            )
            batch_attempts.append(
                {
                    "asset_position": index + 1,
                    "from_level": current_level,
                    "to_level": self.profile_level(
                        prepared[index].profile_id,
                        provider_profile=provider_profile,
                    ),
                    "before_estimated_request_body_bytes": before_total,
                    "after_estimated_request_body_bytes": estimated_total(),
                }
            )

        total = estimated_total()
        batch_validation = {
            "passed": total <= limit,
            "estimated_request_body_bytes": total,
            "transport_budget_bytes": limit,
            "provider_profile": settings["provider_profile"],
            "attempts": batch_attempts,
        }
        prepared = [
            PreparedCloudImage(
                request_path=item.request_path,
                profile_id=item.profile_id,
                profile_version=item.profile_version,
                cache_hit=item.cache_hit,
                audit={
                    **item.audit,
                    "batch_validation": batch_validation,
                },
            )
            for item in prepared
        ]
        return PreparedCloudImageBatch(
            request_paths=tuple(item.request_path for item in prepared),
            items=tuple(prepared),
            fallback_reason=fallback_reason,
            original_preserved=True,
            provider_profile=str(settings["provider_profile"]),
            transport_budget_bytes=limit,
            estimated_request_body_bytes=total,
        )

    def is_request_too_large(self, error: Exception) -> bool:
        """Execute the is request too large operation."""
        http_status = int(getattr(error, "http_status", 0) or 0)
        provider_error = getattr(error, "error", None)
        code = str(getattr(provider_error, "code", "") or "")
        message = str(getattr(provider_error, "public_message", "") or "")
        combined = f"{code} {message} {error}".lower()
        return (
            http_status in {int(value) for value in self.fallback["http_statuses"]}
            or code.lower()
            in {str(value).lower() for value in self.fallback["codes"]}
            or any(
                str(term).lower() in combined
                for term in self.fallback["message_terms"]
            )
        )

    def encode_image(
        self,
        *,
        provider: Any,
        image_path: Path,
        original_asset_id: str,
        status_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> CloudTransportEmbeddingResult:
        """Execute the encode image operation."""
        prepared = self.prepare(
            image_path,
            original_asset_id=original_asset_id,
        )
        if status_sink:
            status_sink(
                "oversized_preparing"
                if prepared.profile_id == self.oversized_profile_id
                else "preparing_cloud_image",
                prepared.audit,
            )
        request_count = 0
        attempted_paths: set[Path] = set()
        while True:
            request_count += 1
            attempted_paths.add(prepared.request_path)
            try:
                vector = [
                    float(value)
                    for value in provider.encode_image(
                        prepared.request_path
                    )
                ]
                return CloudTransportEmbeddingResult(
                    vector=vector,
                    prepared=prepared,
                    request_count=request_count,
                    fallback_from_413=request_count > 1,
                )
            except Exception as exc:
                if not self.is_request_too_large(exc):
                    raise
                next_prepared: PreparedCloudImage | None = None
                current_level = self.profile_level(prepared.profile_id)
                for next_level in range(
                    max(1, current_level + 1),
                    self.max_compatible_level() + 1,
                ):
                    candidate = self.prepare(
                        image_path,
                        original_asset_id=original_asset_id,
                        force_oversized=True,
                        fallback_reason="http_413_request_too_large",
                        minimum_compatible_level=next_level,
                    )
                    if candidate.request_path not in attempted_paths:
                        next_prepared = candidate
                        break
                if next_prepared is None:
                    setattr(exc, "compatible_transport_attempted", True)
                    raise
                prepared = next_prepared
                if status_sink:
                    status_sink("oversized_preparing", prepared.audit)
