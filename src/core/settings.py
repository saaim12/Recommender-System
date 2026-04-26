from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=False, alias="LOG_JSON")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_key: str | None = Field(default=None, alias="API_KEY")

    model_path: str = Field(default="data/models/recommender.pkl", alias="MODEL_PATH")
    config_path: str = Field(default="configs/config.yaml", alias="CONFIG_PATH")

    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_backend: str = Field(default="memory", alias="CACHE_BACKEND")
    cache_ttl_seconds: int = Field(default=300, alias="CACHE_TTL_SECONDS")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    mlflow_tracking_uri: str | None = Field(default=None, alias="MLFLOW_TRACKING_URI")

    @property
    def resolved_model_path(self) -> Path:
        return Path(self.model_path).resolve()


settings = AppSettings()
