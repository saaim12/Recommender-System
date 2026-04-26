from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

try:
    import redis
except Exception:  # pragma: no cover - optional dependency path in constrained envs
    redis = None


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        value = self._store.get(key)
        if value is None:
            return None
        expires_at, payload = value
        if expires_at < time.time():
            self._store.pop(key, None)
            return None
        return payload

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self._store[key] = (time.time() + ttl_seconds, value)


class RedisCache(CacheBackend):
    def __init__(self, redis_url: str) -> None:
        if redis is None:
            raise RuntimeError("Redis dependency is not available.")
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        raw = self.client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self.client.set(key, json.dumps(value), ex=ttl_seconds)
