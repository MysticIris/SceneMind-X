"""Versioned prompt loading with SHA-256 integrity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSpec:
    """Provide prompt spec behavior."""
    prompt_id: str
    version: str
    path: Path
    sha256: str
    text: str


@dataclass(frozen=True)
class PromptBundleSpec:
    """Provide prompt bundle spec behavior."""
    prompt_id: str
    version: str
    status: str
    system: PromptSpec | None
    user: PromptSpec | None
    contract: PromptSpec | None
    payload_skeleton: PromptSpec | None
    modules: tuple[PromptSpec, ...] = ()


@dataclass(frozen=True)
class CorePromptStageSpec:
    """Provide core prompt stage spec behavior."""
    stage_id: str
    prompt: PromptSpec
    max_new_tokens: int
    fields: tuple[str, ...]


@dataclass(frozen=True)
class CorePromptSpec:
    """Provide core prompt spec behavior."""
    prompt_id: str
    version: str
    status: str
    bundle_sha256: str
    stages: tuple[CorePromptStageSpec, ...]


@dataclass(frozen=True)
class CorePromptRegistry:
    """Provide core prompt registry behavior."""
    default_prompt: str
    prompts: dict[str, CorePromptSpec]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_prompt(prompt_path: str | Path, *, prompt_id: str, version: str) -> PromptSpec:
    """Load prompt."""
    path = Path(prompt_path).resolve()
    payload = path.read_bytes()
    return PromptSpec(
        prompt_id=prompt_id,
        version=version,
        path=path,
        sha256=_sha256(payload),
        text=payload.decode("utf-8").strip(),
    )


def load_prompt_manifest(path: str | Path) -> dict[str, PromptSpec]:
    """Load prompt manifest."""
    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompts: dict[str, PromptSpec] = {}
    for prompt_id, item in raw["prompts"].items():
        prompt_path = manifest_path.parent / item["file"]
        spec = load_prompt(prompt_path, prompt_id=prompt_id, version=item["version"])
        if spec.sha256.lower() != str(item["sha256"]).lower():
            raise ValueError(f"prompt SHA-256 mismatch: {prompt_id}")
        prompts[prompt_id] = spec
    return prompts


def load_core_prompt_registry(path: str | Path) -> CorePromptRegistry:
    """Load core prompt registry."""
    registry_path = Path(path).resolve()
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    prompts: dict[str, CorePromptSpec] = {}
    for prompt_id, item in raw["prompts"].items():
        stages: list[CorePromptStageSpec] = []
        digest_parts: list[bytes] = []
        for stage in item["stages"]:
            spec = load_prompt(
                registry_path.parent / stage["file"],
                prompt_id=f"{prompt_id}:{stage['id']}",
                version=item["version"],
            )
            if spec.sha256.lower() != str(stage["sha256"]).lower():
                raise ValueError(f"prompt SHA-256 mismatch: {prompt_id}:{stage['id']}")
            digest_parts.append(stage["id"].encode("utf-8") + b"\0" + spec.sha256.encode("ascii"))
            stages.append(
                CorePromptStageSpec(
                    stage_id=stage["id"],
                    prompt=spec,
                    max_new_tokens=int(stage["max_new_tokens"]),
                    fields=tuple(stage["fields"]),
                )
            )
        bundle_sha256 = _sha256(b"\0".join(digest_parts))
        if bundle_sha256.lower() != str(item["bundle_sha256"]).lower():
            raise ValueError(f"core Prompt bundle SHA-256 mismatch: {prompt_id}")
        prompts[prompt_id] = CorePromptSpec(
            prompt_id=prompt_id,
            version=item["version"],
            status=item["status"],
            bundle_sha256=bundle_sha256,
            stages=tuple(stages),
        )
    default_prompt = str(raw["default_prompt"])
    if default_prompt not in prompts:
        raise ValueError(f"unknown default core Prompt: {default_prompt}")
    return CorePromptRegistry(default_prompt=default_prompt, prompts=prompts)


def load_prompt_bundle_registry(path: str | Path) -> dict[str, PromptBundleSpec]:
    """Load the D3/Phase 2A registry without merging legacy and RC1 namespaces."""

    registry_path = Path(path).resolve()
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    bundles: dict[str, PromptBundleSpec] = {}
    for prompt_id, item in raw["prompts"].items():
        components: dict[str, PromptSpec | None] = {}
        for key in ("system", "user", "contract", "payload_skeleton"):
            component = item.get(key)
            if component is None:
                components[key] = None
                continue
            spec = load_prompt(
                registry_path.parent / component["file"],
                prompt_id=f"{prompt_id}:{key}",
                version=item["version"],
            )
            if spec.sha256.lower() != str(component["sha256"]).lower():
                raise ValueError(f"prompt SHA-256 mismatch: {prompt_id}:{key}")
            components[key] = spec
        modules: list[PromptSpec] = []
        for module_name, component in item.get("modules", {}).items():
            spec = load_prompt(
                registry_path.parent / component["file"],
                prompt_id=f"{prompt_id}:module:{module_name}",
                version=item["version"],
            )
            if spec.sha256.lower() != str(component["sha256"]).lower():
                raise ValueError(f"prompt SHA-256 mismatch: {prompt_id}:module:{module_name}")
            modules.append(spec)
        bundles[prompt_id] = PromptBundleSpec(
            prompt_id=prompt_id,
            version=item["version"],
            status=item["status"],
            system=components["system"],
            user=components["user"],
            contract=components["contract"],
            payload_skeleton=components["payload_skeleton"],
            modules=tuple(modules),
        )
    return bundles
