from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_project_path(config_path or PROJECT_ROOT / "configs" / "config.yaml")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def get_path(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing config key: {'.'.join(keys)}")
        value = value[key]
    return resolve_project_path(value)
