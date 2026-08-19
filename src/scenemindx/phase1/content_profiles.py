"""Versioned Phase 5.4 content-type Prompt profile registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


BASE_CANDIDATE_ID = "SCENEMINDX_CONTENT_TYPE_PROFILES_V1_CANDIDATE"
BASE_PROFILE_VERSION = "1.0.0"
MOMENTS_CANDIDATE_ID = "SCENEMINDX_CONTENT_TYPE_PROFILES_V1_1_CANDIDATE"
MOMENTS_PROFILE_VERSION = "1.1.0"
AUTHORED_CANDIDATE_ID = "SCENEMINDX_CONTENT_TYPE_PROFILES_V1_2_CANDIDATE"
AUTHORED_PROFILE_VERSION = "1.2.0"
CANDIDATE_ID = "SCENEMINDX_CONTENT_TYPE_PROFILES_V1_3_CANDIDATE"
PROFILE_VERSION = "1.3.0"
REQUIRED_FIELDS = {
    "task_identity",
    "objective",
    "required_structure",
    "tone_style",
    "evidence_boundary",
    "expected_use",
    "default_length",
    "length_completion_strategy",
    "forbidden_filler",
    "output_behavior",
    "few_shot",
    "final_self_check",
}
ALIASES = {"story": "creative_story", "auto": "objective_description"}


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ContentProfileRegistry:
    """Provide content profile registry behavior."""
    def __init__(self, project_root: Path) -> None:
        base_path = (
            project_root
            / "prompts"
            / "phase5_4"
            / "content_type_profiles_v1_candidate"
            / "profiles.json"
        )
        moments_override_path = (
            project_root
            / "prompts"
            / "phase5_4a"
            / "moments_profile_v1_1_candidate"
            / "profile.json"
        )
        authored_override_path = (
            project_root
            / "prompts"
            / "phase5_4b"
            / "authored_content_profiles_v1_2_candidate"
            / "profiles.json"
        )
        override_path = (
            project_root
            / "prompts"
            / "phase5_4c"
            / "content_type_profiles_v1_3_candidate"
            / "profiles.json"
        )
        base_payload = json.loads(base_path.read_text(encoding="utf-8"))
        if base_payload.get("candidate_id") != BASE_CANDIDATE_ID:
            raise ValueError("content_profile_candidate_identity_mismatch")
        if base_payload.get("version") != BASE_PROFILE_VERSION:
            raise ValueError("content_profile_version_mismatch")
        profiles = base_payload.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("content_profile_registry_empty")
        profiles = {name: dict(profile) for name, profile in profiles.items()}
        moments_override = json.loads(
            moments_override_path.read_text(encoding="utf-8")
        )
        if moments_override.get("candidate_id") != MOMENTS_CANDIDATE_ID:
            raise ValueError("content_profile_override_identity_mismatch")
        if moments_override.get("version") != MOMENTS_PROFILE_VERSION:
            raise ValueError("content_profile_override_version_mismatch")
        profile_key = str(moments_override.get("profile_key") or "")
        if profile_key != "moments":
            raise ValueError("content_profile_override_key_mismatch")
        profiles[profile_key] = dict(moments_override.get("profile") or {})
        authored_override_payload = json.loads(
            authored_override_path.read_text(encoding="utf-8")
        )
        if authored_override_payload.get("candidate_id") != AUTHORED_CANDIDATE_ID:
            raise ValueError("content_profile_authored_candidate_identity_mismatch")
        if authored_override_payload.get("version") != AUTHORED_PROFILE_VERSION:
            raise ValueError("content_profile_authored_version_mismatch")
        if authored_override_payload.get("parent_candidate_id") != MOMENTS_CANDIDATE_ID:
            raise ValueError("content_profile_authored_parent_mismatch")
        authored_profiles = authored_override_payload.get("profiles")
        if not isinstance(authored_profiles, dict) or not authored_profiles:
            raise ValueError("content_profile_authored_overrides_empty")
        protected = {"objective_description", "news_caption"}
        if protected & set(authored_profiles):
            raise ValueError("content_profile_objective_type_must_not_be_overridden")
        for name, profile in authored_profiles.items():
            if name not in profiles:
                raise ValueError(f"content_profile_authored_unknown:{name}")
            profiles[name] = dict(profile)
        override_payload = json.loads(override_path.read_text(encoding="utf-8"))
        if override_payload.get("candidate_id") != CANDIDATE_ID:
            raise ValueError("content_profile_candidate_v1_3_identity_mismatch")
        if override_payload.get("version") != PROFILE_VERSION:
            raise ValueError("content_profile_candidate_v1_3_version_mismatch")
        if override_payload.get("parent_candidate_id") != AUTHORED_CANDIDATE_ID:
            raise ValueError("content_profile_candidate_v1_3_parent_mismatch")
        profile_overrides = override_payload.get("profiles")
        if not isinstance(profile_overrides, dict) or not profile_overrides:
            raise ValueError("content_profile_candidate_v1_3_overrides_empty")
        if protected & set(profile_overrides):
            raise ValueError("content_profile_objective_type_must_not_be_overridden")
        for name, partial in profile_overrides.items():
            if name not in profiles or not isinstance(partial, dict):
                raise ValueError(f"content_profile_candidate_v1_3_unknown:{name}")
            profiles[name] = {**profiles[name], **partial}
        for name, profile in profiles.items():
            if not isinstance(profile, dict):
                raise ValueError(f"content_profile_invalid:{name}")
            missing = REQUIRED_FIELDS - set(profile)
            if missing:
                raise ValueError(
                    f"content_profile_missing_fields:{name}:{','.join(sorted(missing))}"
                )
            length = profile["default_length"]
            if (
                not isinstance(length, dict)
                or not int(length["minimum"]) <= int(length["target"]) <= int(length["maximum"])
            ):
                raise ValueError(f"content_profile_invalid_default_length:{name}")
        payload = {
            "candidate_id": CANDIDATE_ID,
            "version": PROFILE_VERSION,
            "status": override_payload["status"],
            "parent_candidate_id": AUTHORED_CANDIDATE_ID,
            "base_registry_sha256": _canonical_json_sha256(base_payload),
            "moments_override": moments_override,
            "authored_override": authored_override_payload,
            "override": override_payload,
            "profiles": profiles,
        }
        self.path = override_path
        self.payload = payload
        self.profiles = profiles
        self.sha256 = _canonical_json_sha256(payload)

    def canonical_name(self, content_type: str) -> str:
        """Execute the canonical name operation."""
        key = ALIASES.get(str(content_type), str(content_type))
        if key not in self.profiles:
            raise ValueError(f"content_profile_unknown:{content_type}")
        return key

    def get(self, content_type: str) -> dict[str, Any]:
        """Return the requested value."""
        key = self.canonical_name(content_type)
        profile = dict(self.profiles[key])
        profile["content_type_key"] = key
        profile["version"] = PROFILE_VERSION
        profile["registry_sha256"] = self.sha256
        return profile

    def default_target(self, content_type: str) -> int:
        """Execute the default target operation."""
        return int(self.get(content_type)["default_length"]["target"])

    def status(self) -> dict[str, Any]:
        """Execute the status operation."""
        return {
            "candidate_id": CANDIDATE_ID,
            "version": PROFILE_VERSION,
            "status": self.payload["status"],
            "registry_sha256": self.sha256,
            "profile_count": len(self.profiles),
            "content_types": sorted(self.profiles),
        }
