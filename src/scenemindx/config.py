"""Small, dependency-free configuration loader for Phase 0."""

from __future__ import annotations

import json
import os
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a project configuration is invalid."""


def _read_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix == ".toml":
            value = tomllib.load(handle)
        elif suffix in {".json", ".yaml", ".yml"}:
            # Phase 0 YAML files use JSON syntax, which is valid YAML 1.2.
            value = json.loads(handle.read().decode("utf-8"))
        else:
            raise ConfigError(f"unsupported config extension: {suffix}")
    if not isinstance(value, dict):
        raise ConfigError(f"config root must be an object: {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config, apply an optional relative `extends`, and validate boundaries."""

    config_path = Path(path).resolve()
    current = _read_config(config_path)
    parent_ref = current.pop("extends", None)
    if parent_ref:
        parent_path = (config_path.parent / str(parent_ref)).resolve()
        current = _deep_merge(load_config(parent_path), current)

    paths = current.get("paths", {})
    train_dir = Path(paths.get("train_dir", "")).resolve()
    val_dir = Path(paths.get("val_dir", "")).resolve()
    project_root = Path(paths.get("project_root", "")).resolve()
    if train_dir == val_dir:
        raise ConfigError("Train and Val paths must differ")
    for output_key in ("derived_dir", "output_dir", "experiment_dir"):
        output = Path(paths.get(output_key, project_root)).resolve()
        if output == train_dir or output == val_dir:
            raise ConfigError(f"{output_key} may not target a source split")
        if project_root not in (output, *output.parents):
            raise ConfigError(f"{output_key} must stay inside project_root")

    split_policy = current.get("data", {}).get("split_policy", {})
    if split_policy.get("val_for_training") is not False:
        raise ConfigError("data.split_policy.val_for_training must be false")
    if split_policy.get("val_mix_with_train") is not False:
        raise ConfigError("data.split_policy.val_mix_with_train must be false")
    if split_policy.get("val_training_index_allowed") is not False:
        raise ConfigError(
            "data.split_policy.val_training_index_allowed must be false"
        )
    if split_policy.get("val_independent_index_allowed") is not True:
        raise ConfigError(
            "data.split_policy.val_independent_index_allowed must be true"
        )
    if split_policy.get("val_hard_negative_mining_for_training") is not False:
        raise ConfigError(
            "data.split_policy.val_hard_negative_mining_for_training must be false"
        )
    if split_policy.get("val_generated_samples_return_to_training") is not False:
        raise ConfigError(
            "data.split_policy.val_generated_samples_return_to_training must be false"
        )
    if split_policy.get("hidden_test_for_tuning") is not False:
        raise ConfigError("data.split_policy.hidden_test_for_tuning must be false")

    current["runtime"] = {
        "config_path": str(config_path),
        "environment": os.environ.get("SCENEMINDX_ENV", "local"),
    }
    return current
