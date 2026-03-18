"""Redis state management for live training metrics and environment state."""

from __future__ import annotations

import json
from typing import Any

import redis

from config import RedisConfig, DEFAULT_CONFIG


class RedisStateManager:
    def __init__(self, config: RedisConfig | None = None):
        self.cfg = config or DEFAULT_CONFIG.redis
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=self.cfg.host,
                port=self.cfg.port,
                db=self.cfg.db,
                decode_responses=True,
            )
        return self._client

    def is_connected(self) -> bool:
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False

    # --- Environment state ---

    def set_env_state(self, state: dict[str, Any]) -> None:
        self.client.hset(self.cfg.state_key, mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in state.items()
        })

    def get_env_state(self) -> dict[str, str]:
        return self.client.hgetall(self.cfg.state_key)

    # --- Training metrics ---

    def push_metrics(self, metrics: dict[str, Any]) -> None:
        self.client.rpush(self.cfg.metrics_key, json.dumps(metrics))

    def get_metrics(self, start: int = 0, end: int = -1) -> list[dict]:
        raw = self.client.lrange(self.cfg.metrics_key, start, end)
        return [json.loads(m) for m in raw]

    def clear_metrics(self) -> None:
        self.client.delete(self.cfg.metrics_key)

    # --- Config ---

    def set_config(self, config: dict[str, Any]) -> None:
        self.client.set(self.cfg.config_key, json.dumps(config))

    def get_config(self) -> dict[str, Any] | None:
        raw = self.client.get(self.cfg.config_key)
        return json.loads(raw) if raw else None

    # --- Training status ---

    def set_status(self, status: str) -> None:
        self.client.set(self.cfg.status_key, status)

    def get_status(self) -> str:
        return self.client.get(self.cfg.status_key) or "idle"
