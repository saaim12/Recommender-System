from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.training.model import ContentBasedRecommender
from src.utils.config import get_path, load_config


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    top_n: int = Field(default=10, ge=1, le=100)


class SimilarMoviesRequest(BaseModel):
    movie_id: int = Field(..., ge=0)
    top_n: int = Field(default=10, ge=1, le=100)


def _load_model(config_path: str | Path | None = None) -> ContentBasedRecommender | None:
    config = load_config(config_path)
    model_path = get_path(config, "paths", "models") / "recommender.pkl"
    if not model_path.exists():
        return None
    return ContentBasedRecommender.load(model_path)


def create_app(config_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Recommender System API", version="1.0.0")
    app.state.model = _load_model(config_path)

    @app.get("/health")
    def health() -> dict[str, object]:
        model_loaded = app.state.model is not None
        return {"status": "ok", "model_loaded": model_loaded}

    @app.post("/recommend")
    def recommend(request: RecommendationRequest) -> dict[str, object]:
        model = app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")
        recommendations = model.recommend(request.user_id, top_n=request.top_n)
        return {"user_id": request.user_id, "recommendations": recommendations.to_dict(orient="records")}

    @app.post("/similar")
    def similar_movies(request: SimilarMoviesRequest) -> dict[str, object]:
        model = app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")
        recommendations = model.similar_movies(request.movie_id, top_n=request.top_n)
        return {"movie_id": request.movie_id, "recommendations": recommendations.to_dict(orient="records")}

    return app


app = create_app()
