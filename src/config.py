from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Configuration loaded from env vars with optional YAML defaults."""

    app_env: str = Field("local", env="APP_ENV")
    data_dir: str = Field("data", env="DATA_DIR")
    raw_dir: str = Field("data/raw", env="RAW_DIR")
    processed_dir: str = Field("data/processed", env="PROCESSED_DIR")
    model_dir: str = Field("data/models", env="MODEL_DIR")
    artifact_dir: str = Field("data/artifacts", env="ARTIFACT_DIR")
    tracking_uri: str = Field("http://localhost:5000", env="TRACKING_URI")
    database_url: str = Field("", env="DATABASE_URL")
    s3_bucket: str = Field("", env="S3_BUCKET")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def _read_yaml_defaults(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        parsed = yaml.safe_load(fh) or {}
    return parsed.get("defaults", {})


@lru_cache(maxsize=1)
def get_settings(config_path: str = "configs/config.yaml") -> Settings:
    defaults = _read_yaml_defaults(Path(config_path))
    return Settings(**defaults)


def as_paths(settings: Settings) -> Dict[str, Path]:
    """Resolve key data paths to absolute Paths."""

    base = Path.cwd()
    return {
        "data_dir": (base / settings.data_dir).resolve(),
        "raw_dir": (base / settings.raw_dir).resolve(),
        "processed_dir": (base / settings.processed_dir).resolve(),
        "model_dir": (base / settings.model_dir).resolve(),
        "artifact_dir": (base / settings.artifact_dir).resolve(),
    }
