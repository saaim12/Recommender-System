from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer


NUMERIC_FEATURES = ["rating_count", "avg_rating", "rating_std", "user_count", "tag_count", "distinct_tag_count"]


def split_genres(genres: str) -> list[str]:
    if not isinstance(genres, str):
        return []
    cleaned = genres.strip()
    if not cleaned or cleaned == "(no genres listed)":
        return []
    return [genre for genre in cleaned.split("|") if genre]


@dataclass(slots=True)
class ItemFeatureBundle:
    matrix: np.ndarray
    feature_names: list[str]
    movie_index: dict[int, int]


def build_item_features(catalog: pd.DataFrame) -> ItemFeatureBundle:
    frame = catalog.copy().reset_index(drop=True)
    frame["genres"] = frame["genres"].fillna("(no genres listed)")

    genre_lists = frame["genres"].apply(split_genres).tolist()
    mlb = MultiLabelBinarizer(sparse_output=False)
    if genre_lists and any(genre_lists):
        genre_matrix = mlb.fit_transform(genre_lists)
        genre_names = [f"genre:{name}" for name in mlb.classes_]
    else:
        genre_matrix = np.zeros((len(frame), 0), dtype=float)
        genre_names = []

    numeric_frame = frame.reindex(columns=NUMERIC_FEATURES, fill_value=0).fillna(0)
    scaler = MinMaxScaler()
    numeric_matrix = scaler.fit_transform(numeric_frame) if len(numeric_frame.columns) else np.zeros((len(frame), 0), dtype=float)
    numeric_names = [f"numeric:{name}" for name in numeric_frame.columns]

    if genre_matrix.size and numeric_matrix.size:
        matrix = np.hstack([genre_matrix.astype(float), numeric_matrix.astype(float)])
    elif genre_matrix.size:
        matrix = genre_matrix.astype(float)
    else:
        matrix = numeric_matrix.astype(float)

    movie_index = {int(movie_id): index for index, movie_id in enumerate(frame["movieId"].astype(int).tolist())}
    feature_names = genre_names + numeric_names
    return ItemFeatureBundle(matrix=matrix, feature_names=feature_names, movie_index=movie_index)


def build_user_profile(
    rated_movies: list[tuple[int, float]],
    item_bundle: ItemFeatureBundle,
) -> np.ndarray:
    if not rated_movies:
        return np.zeros(item_bundle.matrix.shape[1], dtype=float)

    indices: list[int] = []
    weights: list[float] = []
    for movie_id, rating in rated_movies:
        movie_index = item_bundle.movie_index.get(int(movie_id))
        if movie_index is None:
            continue
        indices.append(movie_index)
        weights.append(max(float(rating), 0.1))

    if not indices:
        return np.zeros(item_bundle.matrix.shape[1], dtype=float)

    feature_block = item_bundle.matrix[indices]
    weight_array = np.asarray(weights, dtype=float)
    profile = np.average(feature_block, axis=0, weights=weight_array)
    norm = np.linalg.norm(profile)
    if norm == 0:
        return profile
    return profile / norm


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    minimum = float(np.min(scores))
    maximum = float(np.max(scores))
    if np.isclose(minimum, maximum):
        return np.zeros_like(scores, dtype=float)
    return (scores - minimum) / (maximum - minimum)
