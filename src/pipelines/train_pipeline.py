from __future__ import annotations

from pathlib import Path

from src.training.train import train_recommender


def run_pipeline(config_path: str | Path | None = None) -> dict[str, float | int]:
    return train_recommender(config_path=config_path)
