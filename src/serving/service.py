from __future__ import annotations

import logging
from pathlib import Path

from src.core.settings import AppSettings
from src.serving.cache import CacheBackend, InMemoryCache, RedisCache
from src.training.model import ContentBasedRecommender
from src.utils.config import get_path, load_config


logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.model: ContentBasedRecommender | None = None
        self.cache: CacheBackend | None = self._build_cache()

    def _build_cache(self) -> CacheBackend | None:
        if not self.settings.cache_enabled:
            return None
        if self.settings.cache_backend.lower() == "redis":
            try:
                return RedisCache(self.settings.redis_url)
            except Exception:
                logger.warning("Redis cache init failed; falling back to in-memory cache.", exc_info=True)
                return InMemoryCache()
        return InMemoryCache()

    def load_model(self, config_path: str | Path | None = None) -> ContentBasedRecommender | None:
        config = load_config(config_path or self.settings.config_path)
        model_path = get_path(config, "paths", "models") / "recommender.pkl"
        if not model_path.exists():
            logger.warning("Model artifact not found at %s", model_path)
            self.model = None
            return None
        self.model = ContentBasedRecommender.load(model_path)
        logger.info("Model loaded from %s", model_path)
        return self.model

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "model_loaded": self.model is not None,
            "cache_enabled": self.cache is not None,
            "cache_backend": self.settings.cache_backend if self.cache is not None else "disabled",
        }

    def _get_cached(self, key: str) -> list[dict[str, object]] | None:
        if self.cache is None:
            return None
        payload = self.cache.get(key)
        if payload is None:
            return None
        return payload

    def _set_cached(self, key: str, payload: list[dict[str, object]]) -> None:
        if self.cache is None:
            return
        self.cache.set(key, payload, ttl_seconds=self.settings.cache_ttl_seconds)

    def recommend(self, user_id: int, top_n: int) -> list[dict[str, object]]:
        if self.model is None:
            raise RuntimeError("Model artifact not loaded.")
        cache_key = f"recommend:{user_id}:{top_n}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        frame = self.model.recommend(user_id, top_n=top_n)
        payload = frame.to_dict(orient="records")
        self._set_cached(cache_key, payload)
        return payload

    def similar(self, movie_id: int, top_n: int) -> list[dict[str, object]]:
        if self.model is None:
            raise RuntimeError("Model artifact not loaded.")
        cache_key = f"similar:{movie_id}:{top_n}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        frame = self.model.similar_movies(movie_id, top_n=top_n)
        payload = frame.to_dict(orient="records")
        self._set_cached(cache_key, payload)
        return payload
