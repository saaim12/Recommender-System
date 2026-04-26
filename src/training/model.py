from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.data_ingestion.loader import RawData, build_catalog
from src.features.engineer import ItemFeatureBundle, build_item_features, build_user_profile, normalize_scores


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ContentBasedRecommender:
    catalog: pd.DataFrame
    item_bundle: ItemFeatureBundle
    user_history: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    popularity_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    item_similarity: dict[int, dict[int, float]] = field(default_factory=dict)
    global_bias: float = 0.0
    user_bias: dict[int, float] = field(default_factory=dict)
    item_bias: dict[int, float] = field(default_factory=dict)
    user_factors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    item_factors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    factor_user_index: dict[int, int] = field(default_factory=dict)
    factor_item_index: dict[int, int] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        raw_data: RawData,
        train_ratings: pd.DataFrame | None = None,
        *,
        factorization_latent_dim: int | None = None,
        factorization_epochs: int | None = None,
        factorization_sample_rows: int | None = None,
    ) -> "ContentBasedRecommender":
        logger.info("Fitting recommender model")
        catalog = build_catalog(raw_data)
        item_bundle = build_item_features(catalog)
        ratings = train_ratings if train_ratings is not None else raw_data.ratings
        training_rows = len(ratings)
        logger.info(
            "Training ratings ready rows=%s users=%s items=%s feature_dim=%s",
            training_rows,
            ratings["userId"].nunique() if not ratings.empty and "userId" in ratings.columns else 0,
            ratings["movieId"].nunique() if not ratings.empty and "movieId" in ratings.columns else 0,
            item_bundle.matrix.shape[1],
        )

        user_history: dict[int, list[tuple[int, float]]] = {}
        grouped_users = list(ratings.groupby("userId"))
        logger.info("Building user histories users=%s", len(grouped_users))
        for index, (user_id, group) in enumerate(grouped_users, start=1):
            user_history[int(user_id)] = [
                (int(movie_id), float(rating)) for movie_id, rating in group[["movieId", "rating"]].itertuples(index=False, name=None)
            ]
            if index == 1 or index % 500 == 0 or index == len(grouped_users):
                logger.info("User history progress processed=%s total=%s", index, len(grouped_users))

        popularity_source = catalog.reindex(columns=["rating_count", "avg_rating"], fill_value=0).fillna(0)
        popularity = np.log1p(popularity_source["rating_count"].to_numpy(dtype=float)) + popularity_source["avg_rating"].to_numpy(dtype=float)
        popularity_scores = normalize_scores(popularity)
        logger.info("Popularity scores computed items=%s", len(popularity_scores))
        item_similarity = cls._build_item_similarity(ratings)
        similarity_pairs = sum(len(neighbors) for neighbors in item_similarity.values())
        logger.info("Item similarity built items=%s directed_pairs=%s", len(item_similarity), similarity_pairs)

        logger.info("Starting iterative factorization training")
        factorization_kwargs = {}
        if factorization_latent_dim is not None:
            factorization_kwargs["latent_dim"] = factorization_latent_dim
        if factorization_epochs is not None:
            factorization_kwargs["epochs"] = factorization_epochs
        if factorization_sample_rows is not None:
            factorization_kwargs["max_rows"] = factorization_sample_rows

        global_bias, user_bias, item_bias, user_factors, item_factors, factor_user_index, factor_item_index = cls._train_matrix_factorization(
            ratings,
            **factorization_kwargs,
        )
        logger.info(
            "Iterative training complete latent_dim=%s users=%s items=%s",
            user_factors.shape[1] if user_factors.size else 0,
            user_factors.shape[0],
            item_factors.shape[0],
        )
        return cls(
            catalog=catalog,
            item_bundle=item_bundle,
            user_history=user_history,
            popularity_scores=popularity_scores,
            item_similarity=item_similarity,
            global_bias=global_bias,
            user_bias=user_bias,
            item_bias=item_bias,
            user_factors=user_factors,
            item_factors=item_factors,
            factor_user_index=factor_user_index,
            factor_item_index=factor_item_index,
        )

    @staticmethod
    def _train_matrix_factorization(
        ratings: pd.DataFrame,
        *,
        latent_dim: int = 16,
        epochs: int = 4,
        learning_rate: float = 0.01,
        regularization: float = 0.02,
        max_rows: int = 250_000,
        random_state: int = 42,
    ) -> tuple[float, dict[int, float], dict[int, float], np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
        if ratings.empty:
            logger.warning("No ratings provided for factorization training")
            return 0.0, {}, {}, np.zeros((0, latent_dim), dtype=float), np.zeros((0, latent_dim), dtype=float), {}, {}

        working = ratings[["userId", "movieId", "rating"]].dropna().copy()
        if len(working) > max_rows:
            logger.info("Sampling ratings for factorization training original_rows=%s sample_rows=%s", len(working), max_rows)
            working = working.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        else:
            working = working.reset_index(drop=True)

        user_ids = sorted(working["userId"].astype(int).unique().tolist())
        item_ids = sorted(working["movieId"].astype(int).unique().tolist())
        user_index = {user_id: index for index, user_id in enumerate(user_ids)}
        item_index = {item_id: index for index, item_id in enumerate(item_ids)}

        user_factors = np.random.default_rng(random_state).normal(0, 0.1, size=(len(user_ids), latent_dim))
        item_factors = np.random.default_rng(random_state + 1).normal(0, 0.1, size=(len(item_ids), latent_dim))
        user_bias = {user_id: 0.0 for user_id in user_ids}
        item_bias = {item_id: 0.0 for item_id in item_ids}
        global_bias = float(working["rating"].mean())

        interactions = list(working[["userId", "movieId", "rating"]].itertuples(index=False, name=None))
        logger.info(
            "Factorization setup interactions=%s users=%s items=%s latent_dim=%s epochs=%s",
            len(interactions),
            len(user_ids),
            len(item_ids),
            latent_dim,
            epochs,
        )

        rng = np.random.default_rng(random_state)
        for epoch in range(1, epochs + 1):
            rng.shuffle(interactions)
            squared_error = 0.0
            total = 0
            for batch_start in range(0, len(interactions), 100_000):
                batch = interactions[batch_start : batch_start + 100_000]
                for user_id, movie_id, rating in batch:
                    u_idx = user_index[int(user_id)]
                    i_idx = item_index[int(movie_id)]
                    prediction = global_bias + user_bias[int(user_id)] + item_bias[int(movie_id)] + float(
                        np.dot(user_factors[u_idx], item_factors[i_idx])
                    )
                    error = float(rating) - prediction
                    squared_error += error * error
                    total += 1

                    user_bias[int(user_id)] += learning_rate * (error - regularization * user_bias[int(user_id)])
                    item_bias[int(movie_id)] += learning_rate * (error - regularization * item_bias[int(movie_id)])

                    user_vector = user_factors[u_idx].copy()
                    item_vector = item_factors[i_idx].copy()
                    user_factors[u_idx] += learning_rate * (error * item_vector - regularization * user_vector)
                    item_factors[i_idx] += learning_rate * (error * user_vector - regularization * item_vector)

                batch_loss = squared_error / max(total, 1)
                logger.info(
                    "Factorization batch epoch=%s rows_processed=%s batch_size=%s running_mse=%.6f",
                    epoch,
                    total,
                    len(batch),
                    batch_loss,
                )

            epoch_loss = squared_error / max(total, 1)
            rmse = float(np.sqrt(epoch_loss))
            logger.info(
                "Epoch complete epoch=%s/%s rows=%s mse=%.6f rmse=%.6f",
                epoch,
                epochs,
                total,
                epoch_loss,
                rmse,
            )

        return global_bias, user_bias, item_bias, user_factors, item_factors, user_index, item_index

    @staticmethod
    def _build_item_similarity(ratings: pd.DataFrame) -> dict[int, dict[int, float]]:
        if ratings.empty:
            return {}

        positive = ratings[ratings["rating"] >= 4.0]
        if positive.empty:
            return {}

        item_counts: dict[int, int] = {}
        pair_counts: dict[tuple[int, int], int] = {}

        positive_groups = list(positive.groupby("userId"))
        logger.info("Building collaborative item similarity positive_users=%s positive_rows=%s", len(positive_groups), len(positive))
        for index, (_, group) in enumerate(positive_groups, start=1):
            items = sorted(set(group["movieId"].astype(int).tolist()))
            for item in items:
                item_counts[item] = item_counts.get(item, 0) + 1
            for idx, left in enumerate(items):
                for right in items[idx + 1 :]:
                    key = (left, right)
                    pair_counts[key] = pair_counts.get(key, 0) + 1
            if index == 1 or index % 500 == 0 or index == len(positive_groups):
                logger.info("Collaborative progress processed_users=%s total_users=%s", index, len(positive_groups))

        similarities: dict[int, dict[int, float]] = {}
        for (left, right), count in pair_counts.items():
            denom = np.sqrt(item_counts[left] * item_counts[right])
            if denom <= 0:
                continue
            score = float(count / denom)
            similarities.setdefault(left, {})[right] = score
            similarities.setdefault(right, {})[left] = score
        logger.info("Collaborative similarity stats pair_counts=%s items=%s", len(pair_counts), len(item_counts))
        return similarities

    @property
    def movie_ids(self) -> list[int]:
        return self.catalog["movieId"].astype(int).tolist()

    def _score_profile(self, profile: np.ndarray) -> np.ndarray:
        if profile.size == 0:
            return np.zeros(self.item_bundle.matrix.shape[0], dtype=float)

        scores = self.item_bundle.matrix @ profile
        norms = np.linalg.norm(self.item_bundle.matrix, axis=1)
        denominator = np.clip(norms, 1e-8, None)
        scores = scores / denominator
        if self.popularity_scores.size == scores.size:
            scores = 0.85 * scores + 0.15 * self.popularity_scores
        return scores

    def _factorization_scores(self, user_id: int, history: list[tuple[int, float]]) -> np.ndarray:
        if self.user_factors.size == 0 or self.item_factors.size == 0:
            return np.zeros(self.item_bundle.matrix.shape[0], dtype=float)

        factor_user_index = self.factor_user_index.get(int(user_id))
        if factor_user_index is None or factor_user_index >= self.user_factors.shape[0]:
            return np.zeros(self.item_bundle.matrix.shape[0], dtype=float)

        latent_user = self.user_factors[factor_user_index]

        scores = np.zeros(self.item_bundle.matrix.shape[0], dtype=float)
        for movie_id, index in self.item_bundle.movie_index.items():
            factor_item_index = self.factor_item_index.get(int(movie_id))
            if factor_item_index is None or factor_item_index >= self.item_factors.shape[0]:
                continue
            scores[index] = self.global_bias + self.item_bias.get(movie_id, 0.0) + float(
                np.dot(latent_user, self.item_factors[factor_item_index])
            )
        return normalize_scores(scores)

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        history = self.user_history.get(int(user_id), [])
        if not history:
            logger.info("No user history found user_id=%s returning popular recommendations", user_id)
            return self._popular_recommendations(top_n)

        profile = build_user_profile(history, self.item_bundle)
        content_scores = self._score_profile(profile)
        collaborative_scores = self._collaborative_scores(history)
        factorization_scores = self._factorization_scores(user_id, history)
        scores = 0.55 * content_scores + 0.20 * collaborative_scores + 0.25 * factorization_scores
        watched_movie_ids = {movie_id for movie_id, _ in history}
        logger.info(
            "Ranking recommendations user_id=%s history_size=%s candidate_items=%s score_components=content+collaborative+factorization",
            user_id,
            len(history),
            len(self.catalog),
        )

        recommendations = self.catalog.copy()
        recommendations = recommendations[~recommendations["movieId"].astype(int).isin(watched_movie_ids)].copy()
        if recommendations.empty:
            return recommendations

        recommendations["score"] = [scores[self.item_bundle.movie_index[int(movie_id)]] for movie_id in recommendations["movieId"].astype(int)]
        recommendations = recommendations.sort_values(["score", "rating_count", "avg_rating"], ascending=False)
        logger.info("Recommendations generated user_id=%s top_n=%s returned=%s", user_id, top_n, min(top_n, len(recommendations)))
        return recommendations[["movieId", "title", "genres", "score"]].head(top_n).reset_index(drop=True)

    def _collaborative_scores(self, history: list[tuple[int, float]]) -> np.ndarray:
        if not history or not self.item_similarity:
            return np.zeros(self.item_bundle.matrix.shape[0], dtype=float)

        score_map: dict[int, float] = {}
        for movie_id, rating in history:
            neighbors = self.item_similarity.get(int(movie_id), {})
            if not neighbors:
                continue
            weight = max(float(rating), 0.1) / 5.0
            for neighbor_id, similarity in neighbors.items():
                score_map[neighbor_id] = score_map.get(neighbor_id, 0.0) + similarity * weight

        scores = np.zeros(self.item_bundle.matrix.shape[0], dtype=float)
        for movie_id, score in score_map.items():
            index = self.item_bundle.movie_index.get(int(movie_id))
            if index is not None:
                scores[index] = score
        return normalize_scores(scores)

    def similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        movie_index = self.item_bundle.movie_index.get(int(movie_id))
        if movie_index is None:
            logger.warning("Movie not found movie_id=%s returning popular recommendations", movie_id)
            return self._popular_recommendations(top_n)

        target = self.item_bundle.matrix[movie_index]
        target_norm = np.linalg.norm(target)
        if np.isclose(target_norm, 0):
            logger.warning("Movie has zero vector movie_id=%s returning popular recommendations", movie_id)
            return self._popular_recommendations(top_n)

        similarities = (self.item_bundle.matrix @ target) / np.clip(np.linalg.norm(self.item_bundle.matrix, axis=1) * target_norm, 1e-8, None)
        recommendations = self.catalog.copy()
        recommendations["score"] = similarities
        recommendations = recommendations[recommendations["movieId"].astype(int) != int(movie_id)]
        recommendations = recommendations.sort_values(["score", "rating_count", "avg_rating"], ascending=False)
        logger.info("Similarity results generated movie_id=%s top_n=%s returned=%s", movie_id, top_n, min(top_n, len(recommendations)))
        return recommendations[["movieId", "title", "genres", "score"]].head(top_n).reset_index(drop=True)

    def _popular_recommendations(self, top_n: int) -> pd.DataFrame:
        recommendations = self.catalog.copy()
        if self.popularity_scores.size == len(recommendations):
            recommendations["score"] = self.popularity_scores
        else:
            recommendations["score"] = recommendations.get("avg_rating", 0)
        recommendations = recommendations.sort_values(["score", "rating_count", "avg_rating"], ascending=False)
        return recommendations[["movieId", "title", "genres", "score"]].head(top_n).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)
        logger.info("Model saved path=%s bytes=%s", target, target.stat().st_size)

    @classmethod
    def load(cls, path: str | Path) -> "ContentBasedRecommender":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("Loaded artifact is not a ContentBasedRecommender")
        return model
