from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.config import resolve_project_path


@dataclass(slots=True)
class RawData:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame


def _read_csv(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(path, usecols=usecols)


def load_raw_data(raw_data_dir: str | Path) -> RawData:
    base_dir = resolve_project_path(raw_data_dir)
    movies = _read_csv(base_dir / "movies.csv")
    ratings = _read_csv(base_dir / "ratings.csv")
    tags = _read_csv(base_dir / "tags.csv")
    links = _read_csv(base_dir / "links.csv")
    return RawData(movies=movies, ratings=ratings, tags=tags, links=links)


def aggregate_movie_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    if ratings.empty:
        return pd.DataFrame(columns=["movieId", "rating_count", "avg_rating", "rating_std", "user_count"])

    grouped = ratings.groupby("movieId")
    summary = grouped.agg(
        rating_count=("rating", "size"),
        avg_rating=("rating", "mean"),
        rating_std=("rating", "std"),
        user_count=("userId", pd.Series.nunique),
    ).reset_index()
    summary["rating_std"] = summary["rating_std"].fillna(0.0)
    return summary


def aggregate_movie_tags(tags: pd.DataFrame) -> pd.DataFrame:
    if tags.empty:
        return pd.DataFrame(columns=["movieId", "tag_count", "distinct_tag_count", "tag_text"])

    tag_frame = tags.copy()
    tag_frame["tag"] = tag_frame["tag"].astype(str).str.strip().str.lower()
    grouped = tag_frame.groupby("movieId")
    summary = grouped.agg(
        tag_count=("tag", "size"),
        distinct_tag_count=("tag", pd.Series.nunique),
        tag_text=("tag", lambda values: " | ".join(sorted(set(values.dropna())))),
    ).reset_index()
    return summary


def build_catalog(raw_data: RawData) -> pd.DataFrame:
    movies = raw_data.movies.copy()
    ratings_summary = aggregate_movie_ratings(raw_data.ratings)
    tags_summary = aggregate_movie_tags(raw_data.tags)

    catalog = movies.merge(raw_data.links, on="movieId", how="left")
    catalog = catalog.merge(ratings_summary, on="movieId", how="left")
    catalog = catalog.merge(tags_summary, on="movieId", how="left")

    for column in ["rating_count", "avg_rating", "rating_std", "user_count", "tag_count", "distinct_tag_count"]:
        if column not in catalog:
            catalog[column] = 0
        catalog[column] = catalog[column].fillna(0)

    catalog["genres"] = catalog["genres"].fillna("(no genres listed)")
    catalog["title"] = catalog["title"].fillna("")
    return catalog


def split_train_test_per_user(
    ratings: pd.DataFrame,
    *,
    test_size: float = 0.2,
    min_interactions: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ratings.empty:
        return ratings.copy(), ratings.copy()

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    sort_columns = [column for column in ["timestamp", "movieId"] if column in ratings.columns]

    for _, group in ratings.groupby("userId"):
        ordered = group.sort_values(sort_columns) if sort_columns else group.copy()
        if len(ordered) < min_interactions:
            train_parts.append(ordered)
            continue

        test_count = max(1, int(round(len(ordered) * test_size)))
        if test_count >= len(ordered):
            test_count = 1
        train_parts.append(ordered.iloc[:-test_count])
        test_parts.append(ordered.iloc[-test_count:])

    train_frame = pd.concat(train_parts, ignore_index=True) if train_parts else ratings.iloc[0:0].copy()
    test_frame = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train_frame, test_frame
