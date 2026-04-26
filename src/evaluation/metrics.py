from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.training.model import ContentBasedRecommender


logger = logging.getLogger(__name__)


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0 or not recommended:
        return 0.0
    hits = sum(1 for movie_id in recommended[:k] if movie_id in relevant)
    return hits / min(k, len(recommended))


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for movie_id in recommended[:k] if movie_id in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    return float(any(movie_id in relevant for movie_id in recommended[:k]))


@dataclass(slots=True)
class EvaluationResult:
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    coverage: float
    evaluated_users: int


def evaluate_holdout(
    model: ContentBasedRecommender,
    holdout_ratings: pd.DataFrame,
    *,
    k: int = 10,
    positive_threshold: float = 4.0,
) -> EvaluationResult:
    if holdout_ratings.empty:
        logger.warning("Holdout ratings are empty; skipping evaluation")
        return EvaluationResult(0.0, 0.0, 0.0, 0.0, 0)

    logger.info(
        "Starting holdout evaluation rows=%s users=%s k=%s positive_threshold=%s",
        len(holdout_ratings),
        holdout_ratings["userId"].nunique() if "userId" in holdout_ratings.columns else 0,
        k,
        positive_threshold,
    )
    precision_values: list[float] = []
    recall_values: list[float] = []
    hit_values: list[float] = []
    recommended_items: set[int] = set()

    user_groups = list(holdout_ratings.groupby("userId"))
    for index, (user_id, group) in enumerate(user_groups, start=1):
        relevant_items = set(group.loc[group["rating"] >= positive_threshold, "movieId"].astype(int).tolist())
        if not relevant_items:
            continue

        recommendations = model.recommend(int(user_id), top_n=k)
        recommended_ids = recommendations["movieId"].astype(int).tolist() if not recommendations.empty else []
        recommended_items.update(recommended_ids)

        precision_values.append(precision_at_k(recommended_ids, relevant_items, k))
        recall_values.append(recall_at_k(recommended_ids, relevant_items, k))
        hit_values.append(hit_rate_at_k(recommended_ids, relevant_items, k))
        if index == 1 or index % 100 == 0 or index == len(user_groups):
            logger.info("Evaluation progress processed_users=%s total_users=%s", index, len(user_groups))

    catalog_size = max(len(model.catalog), 1)
    coverage = len(recommended_items) / catalog_size
    evaluated_users = len(precision_values)

    if evaluated_users == 0:
        logger.warning("No evaluable users found in holdout set")
        return EvaluationResult(0.0, 0.0, 0.0, coverage, 0)

    result = EvaluationResult(
        precision_at_k=sum(precision_values) / evaluated_users,
        recall_at_k=sum(recall_values) / evaluated_users,
        hit_rate_at_k=sum(hit_values) / evaluated_users,
        coverage=coverage,
        evaluated_users=evaluated_users,
    )
    logger.info(
        "Evaluation complete precision_at_k=%.4f recall_at_k=%.4f hit_rate_at_k=%.4f coverage=%.4f evaluated_users=%s",
        result.precision_at_k,
        result.recall_at_k,
        result.hit_rate_at_k,
        result.coverage,
        result.evaluated_users,
    )
    return result
