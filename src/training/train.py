from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.core.logging import configure_logging
from src.core.settings import settings
from src.data_ingestion.loader import load_raw_data, split_train_test_per_user
from src.evaluation.metrics import evaluate_holdout
from src.training.model import ContentBasedRecommender
from src.utils.config import get_path, load_config


logger = logging.getLogger(__name__)


def train_recommender(config_path: str | Path | None = None) -> dict[str, float | int]:
    configure_logging(level=settings.log_level, use_json=settings.log_json)
    config = load_config(config_path)

    tracking_uri = settings.mlflow_tracking_uri or config.get("tracking", {}).get("mlflow_tracking_uri")
    if tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = str(tracking_uri)

    logger.info("Starting training pipeline")
    raw_data = load_raw_data(get_path(config, "paths", "raw_data"))
    logger.info(
        "Source summary movies=%s ratings=%s tags=%s links=%s",
        len(raw_data.movies),
        len(raw_data.ratings),
        len(raw_data.tags),
        len(raw_data.links),
    )
    train_ratings, holdout_ratings = split_train_test_per_user(
        raw_data.ratings,
        test_size=float(config.get("training", {}).get("test_size", 0.2)),
        min_interactions=5,
    )
    logger.info(
        "Dataset split train_rows=%s holdout_rows=%s train_users=%s holdout_users=%s",
        len(train_ratings),
        len(holdout_ratings),
        train_ratings["userId"].nunique() if not train_ratings.empty else 0,
        holdout_ratings["userId"].nunique() if not holdout_ratings.empty else 0,
    )

    training_config = config.get("training", {})
    model = ContentBasedRecommender.fit(
        raw_data,
        train_ratings=train_ratings,
        factorization_latent_dim=int(training_config.get("factorization_latent_dim", 16)),
        factorization_epochs=int(training_config.get("factorization_epochs", 4)),
        factorization_sample_rows=int(training_config.get("factorization_sample_rows", 250000)),
    )
    logger.info(
        "Model fit complete catalog_rows=%s feature_dim=%s users=%s",
        len(model.catalog),
        model.item_bundle.matrix.shape[1],
        len(model.user_history),
    )
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

    logger.info("Training complete model_path=%s metrics_path=%s", model_path, metrics_path)
    logger.info(
        "Final metrics precision_at_k=%.4f recall_at_k=%.4f hit_rate_at_k=%.4f coverage=%.4f evaluated_users=%s",
        evaluation.precision_at_k,
        evaluation.recall_at_k,
        evaluation.hit_rate_at_k,
        evaluation.coverage,
        evaluation.evaluated_users,
    )

    return {
        "precision_at_k": evaluation.precision_at_k,
        "recall_at_k": evaluation.recall_at_k,
        "hit_rate_at_k": evaluation.hit_rate_at_k,
        "coverage": evaluation.coverage,
        "evaluated_users": evaluation.evaluated_users,
    }


def main() -> None:
    try:
        metrics = train_recommender()
        print(json.dumps(metrics, indent=2))
    except Exception:
        logger.exception("Training failed")
        raise


if __name__ == "__main__":
    main()
