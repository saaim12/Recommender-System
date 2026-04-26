from __future__ import annotations

import pandas as pd

from src.data_ingestion.loader import RawData, build_catalog, split_train_test_per_user
from src.evaluation.metrics import evaluate_holdout
from src.training.model import ContentBasedRecommender


def build_sample_raw_data() -> RawData:
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": ["Toy Story", "Jumanji", "Heat", "Sabrina"],
            "genres": ["Adventure|Animation|Children|Comedy|Fantasy", "Adventure|Children|Fantasy", "Action|Crime|Thriller", "Comedy|Romance"],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 2, 3, 1, 3, 4],
            "rating": [5.0, 4.0, 1.0, 4.0, 5.0, 2.0],
            "timestamp": [1, 2, 3, 1, 2, 3],
        }
    )
    tags = pd.DataFrame({"userId": [1], "movieId": [1], "tag": ["classic"], "timestamp": [1]})
    links = pd.DataFrame({"movieId": [1, 2, 3, 4], "imdbId": [1, 2, 3, 4], "tmdbId": [11, 22, 33, 44]})
    return RawData(movies=movies, ratings=ratings, tags=tags, links=links)


def test_catalog_builds_and_recommendations_exclude_history() -> None:
    raw_data = build_sample_raw_data()
    catalog = build_catalog(raw_data)
    assert len(catalog) == 4

    train_ratings, holdout_ratings = split_train_test_per_user(raw_data.ratings, test_size=0.33, min_interactions=2)
    model = ContentBasedRecommender.fit(raw_data, train_ratings=train_ratings)
    recommendations = model.recommend(1, top_n=3)

    watched = set(train_ratings.loc[train_ratings["userId"] == 1, "movieId"].tolist())
    assert not set(recommendations["movieId"]).intersection(watched)
    assert not recommendations.empty

    evaluation = evaluate_holdout(model, holdout_ratings, k=2)
    assert evaluation.evaluated_users >= 1


def test_similarity_endpoint_logic_returns_data() -> None:
    raw_data = build_sample_raw_data()
    model = ContentBasedRecommender.fit(raw_data)
    similar = model.similar_movies(1, top_n=2)
    assert len(similar) == 2
    assert 1 not in set(similar["movieId"].tolist())
