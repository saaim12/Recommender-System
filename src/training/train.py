from __future__ import annotations

import json
from pathlib import Path

from src.data_ingestion.loader import load_raw_data, split_train_test_per_user
from src.evaluation.metrics import evaluate_holdout
from src.training.model import ContentBasedRecommender
from src.utils.config import get_path, load_config


def train_recommender(config_path: str | Path | None = None) -> dict[str, float | int]:
    config = load_config(config_path)
    raw_data = load_raw_data(get_path(config, "paths", "raw_data"))
    train_ratings, holdout_ratings = split_train_test_per_user(
        raw_data.ratings,
        test_size=float(config.get("training", {}).get("test_size", 0.2)),
        min_interactions=5,
    )

    model = ContentBasedRecommender.fit(raw_data, train_ratings=train_ratings)
    evaluation = evaluate_holdout(model, holdout_ratings, k=10)

    model_path = get_path(config, "paths", "models") / "recommender.pkl"
    metrics_path = get_path(config, "paths", "models") / "metrics.json"
    model.save(model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "precision_at_k": evaluation.precision_at_k,
                "recall_at_k": evaluation.recall_at_k,
                "hit_rate_at_k": evaluation.hit_rate_at_k,
                "coverage": evaluation.coverage,
                "evaluated_users": evaluation.evaluated_users,
            },
            handle,
            indent=2,
        )

    return {
        "precision_at_k": evaluation.precision_at_k,
        "recall_at_k": evaluation.recall_at_k,
        "hit_rate_at_k": evaluation.hit_rate_at_k,
        "coverage": evaluation.coverage,
        "evaluated_users": evaluation.evaluated_users,
    }


def main() -> None:
    metrics = train_recommender()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
