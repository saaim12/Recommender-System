from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.data_ingestion.loader import RawData, build_catalog
from src.features.engineer import ItemFeatureBundle, build_item_features, build_user_profile, normalize_scores


@dataclass(slots=True)
class ContentBasedRecommender:
    catalog: pd.DataFrame
    item_bundle: ItemFeatureBundle
    user_history: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    popularity_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    @classmethod
    def fit(cls, raw_data: RawData, train_ratings: pd.DataFrame | None = None) -> "ContentBasedRecommender":
        catalog = build_catalog(raw_data)
        item_bundle = build_item_features(catalog)
        ratings = train_ratings if train_ratings is not None else raw_data.ratings

        user_history: dict[int, list[tuple[int, float]]] = {}
        for user_id, group in ratings.groupby("userId"):
            user_history[int(user_id)] = [
                (int(movie_id), float(rating)) for movie_id, rating in group[["movieId", "rating"]].itertuples(index=False, name=None)
            ]

        popularity_source = catalog.reindex(columns=["rating_count", "avg_rating"], fill_value=0).fillna(0)
        popularity = np.log1p(popularity_source["rating_count"].to_numpy(dtype=float)) + popularity_source["avg_rating"].to_numpy(dtype=float)
        popularity_scores = normalize_scores(popularity)
        return cls(catalog=catalog, item_bundle=item_bundle, user_history=user_history, popularity_scores=popularity_scores)

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

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        history = self.user_history.get(int(user_id), [])
        if not history:
            return self._popular_recommendations(top_n)

        profile = build_user_profile(history, self.item_bundle)
        scores = self._score_profile(profile)
        watched_movie_ids = {movie_id for movie_id, _ in history}

        recommendations = self.catalog.copy()
        recommendations = recommendations[~recommendations["movieId"].astype(int).isin(watched_movie_ids)].copy()
        if recommendations.empty:
            return recommendations

        recommendations["score"] = [scores[self.item_bundle.movie_index[int(movie_id)]] for movie_id in recommendations["movieId"].astype(int)]
        recommendations = recommendations.sort_values(["score", "rating_count", "avg_rating"], ascending=False)
        return recommendations[["movieId", "title", "genres", "score"]].head(top_n).reset_index(drop=True)

    def similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        movie_index = self.item_bundle.movie_index.get(int(movie_id))
        if movie_index is None:
            return self._popular_recommendations(top_n)

        target = self.item_bundle.matrix[movie_index]
        target_norm = np.linalg.norm(target)
        if np.isclose(target_norm, 0):
            return self._popular_recommendations(top_n)

        similarities = (self.item_bundle.matrix @ target) / np.clip(np.linalg.norm(self.item_bundle.matrix, axis=1) * target_norm, 1e-8, None)
        recommendations = self.catalog.copy()
        recommendations["score"] = similarities
        recommendations = recommendations[recommendations["movieId"].astype(int) != int(movie_id)]
        recommendations = recommendations.sort_values(["score", "rating_count", "avg_rating"], ascending=False)
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

    @classmethod
    def load(cls, path: str | Path) -> "ContentBasedRecommender":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("Loaded artifact is not a ContentBasedRecommender")
        return model
