from __future__ import annotations

from pathlib import Path
import logging

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.core.logging import configure_logging
from src.core.settings import settings
from src.serving.middleware import register_middlewares
from src.serving.service import RecommendationService


logger = logging.getLogger(__name__)


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    top_n: int = Field(default=10, ge=1, le=100)


class SimilarMoviesRequest(BaseModel):
    movie_id: int = Field(..., ge=0)
    top_n: int = Field(default=10, ge=1, le=100)


def _authorize(x_api_key: str | None) -> None:
    configured_api_key = settings.api_key
    if not configured_api_key:
        return
    if not x_api_key or x_api_key != configured_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def create_app(config_path: str | Path | None = None) -> FastAPI:
    configure_logging(level=settings.log_level, use_json=settings.log_json)
    app = FastAPI(title="Recommender System API", version="1.1.0")
    register_middlewares(app)

    service = RecommendationService(settings)
    service.load_model(config_path)
    app.state.service = service

    @app.post("/reload-model")
    def reload_model(x_api_key: str | None = Header(default=None)) -> dict[str, object]:
        _authorize(x_api_key)
        model = service.load_model(config_path)
        return {"reloaded": model is not None, "model_loaded": model is not None}

    @app.get("/health")
    def health() -> dict[str, object]:
        return service.health()

    @app.post("/recommend")
    def recommend(request: RecommendationRequest, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
        _authorize(x_api_key)
        try:
            recommendations = service.recommend(request.user_id, top_n=request.top_n)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"user_id": request.user_id, "recommendations": recommendations}

    @app.post("/similar")
    def similar_movies(request: SimilarMoviesRequest, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
        _authorize(x_api_key)
        try:
            recommendations = service.similar(request.movie_id, top_n=request.top_n)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"movie_id": request.movie_id, "recommendations": recommendations}

    logger.info("API app created with env=%s", settings.app_env)

    return app


app = create_app()
