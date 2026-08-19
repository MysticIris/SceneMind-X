"""Load and verify the modular Phase 2B prompt registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Phase2bPromptModule:
    """Provide phase2b prompt module behavior."""
    path: Path
    relative_path: str
    sha256: str
    text: str


@dataclass(frozen=True)
class Phase2bPromptStage:
    """Provide phase2b prompt stage behavior."""
    name: str
    version: str
    status: str
    modules: tuple[Phase2bPromptModule, ...]

    def module(self, filename: str) -> Phase2bPromptModule:
        """Execute the module operation."""
        matches = [item for item in self.modules if Path(item.relative_path).name == filename]
        if len(matches) != 1:
            raise KeyError(f"expected one {filename} module in {self.name}")
        return matches[0]


def load_phase2b_prompt_registry(path: str | Path) -> dict[str, Phase2bPromptStage]:
    """Load phase2b prompt registry."""
    registry_path = Path(path).resolve()
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    if raw.get("registry_version") != "scenemindx_phase2b_prompt_registry_v1":
        raise ValueError("unexpected Phase 2B prompt registry version")
    stages: dict[str, Phase2bPromptStage] = {}
    for name, item in raw["stages"].items():
        modules: list[Phase2bPromptModule] = []
        for spec in item["modules"]:
            module_path = (registry_path.parent / spec["file"]).resolve()
            module_path.relative_to(registry_path.parent)
            payload = module_path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest.lower() != str(spec["sha256"]).lower():
                raise ValueError(f"Phase 2B prompt SHA-256 mismatch: {name}:{spec['file']}")
            modules.append(Phase2bPromptModule(
                path=module_path,
                relative_path=spec["file"],
                sha256=digest,
                text=payload.decode("utf-8").strip(),
            ))
        stages[name] = Phase2bPromptStage(
            name=name,
            version=item["version"],
            status=item["status"],
            modules=tuple(modules),
        )
    if set(stages) != {"media_router", "semantic_extractor"}:
        raise ValueError("Phase 2B registry must contain exactly two stages")
    return stages
