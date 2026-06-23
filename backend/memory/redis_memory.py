"""Redis-backed conversation memory with in-memory fallback."""

import json
import time
from typing import Optional

from loguru import logger

from backend.config import get_settings


class ConversationMemory:
    """
    Unified memory interface: Redis when available, else in-process dict.
    Persists chat history and supports session cache.
    """

    def __init__(self, max_messages: int = 50, session_timeout: int = 3600):
        self.settings = get_settings()
        self.max_messages = max_messages
        self.session_timeout = session_timeout
        self._local: dict[str, dict] = {}
        self._redis = None
        self._use_redis = False
        self._connect_redis()
        logger.info(
            f"ConversationMemory initialized | backend={'redis' if self._use_redis else 'memory'} | "
            f"max_messages={max_messages}"
        )

    def _connect_redis(self):
        if not self.settings.redis_enabled:
            return
        try:
            import redis

            self._redis = redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._redis.ping()
            self._use_redis = True
            logger.info(f"Redis connected: {self.settings.redis_url}")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory fallback: {e}")
            self._redis = None
            self._use_redis = False

    def _key(self, session_id: str) -> str:
        return f"campusgpt:chat:{session_id}"

    def _cache_key(self, session_id: str) -> str:
        return f"campusgpt:cache:{session_id}"

    def get_history(self, session_id: str) -> list[dict]:
        if self._use_redis and self._redis:
            raw = self._redis.get(self._key(session_id))
            if raw:
                return json.loads(raw)
            return []
        self._ensure_session(session_id)
        return self._local[session_id]["messages"]

    def add_message(self, session_id: str, role: str, content: str):
        history = self.get_history(session_id)
        history.append({"role": role, "content": content, "timestamp": time.time()})
        if len(history) > self.max_messages:
            history = history[-self.max_messages :]
        self._save(session_id, history)

    def add_exchange(self, session_id: str, question: str, answer: str):
        self.add_message(session_id, "user", question)
        self.add_message(session_id, "assistant", answer)

    def clear_session(self, session_id: str):
        if self._use_redis and self._redis:
            self._redis.delete(self._key(session_id))
            self._redis.delete(self._cache_key(session_id))
        elif session_id in self._local:
            self._local[session_id]["messages"] = []
        logger.info(f"Cleared session: {session_id}")

    def export_chat(self, session_id: str) -> str:
        history = self.get_history(session_id)
        lines = [f"=== Chat Session: {session_id} ===\n"]
        for msg in history:
            role = msg.get("role", "user").upper()
            lines.append(f"\n[{role}]\n{msg.get('content', '')}\n")
        return "\n".join(lines)

    def set_cache(self, session_id: str, data: dict, ttl: int = 3600):
        if self._use_redis and self._redis:
            self._redis.setex(self._cache_key(session_id), ttl, json.dumps(data))

    def get_cache(self, session_id: str) -> Optional[dict]:
        if self._use_redis and self._redis:
            raw = self._redis.get(self._cache_key(session_id))
            return json.loads(raw) if raw else None
        return None

    def _save(self, session_id: str, messages: list[dict]):
        if self._use_redis and self._redis:
            self._redis.setex(
                self._key(session_id),
                self.session_timeout,
                json.dumps(messages),
            )
        else:
            self._ensure_session(session_id)
            self._local[session_id]["messages"] = messages
            self._local[session_id]["last_active"] = time.time()

    def _ensure_session(self, session_id: str):
        if session_id not in self._local:
            self._local[session_id] = {
                "messages": [],
                "created_at": time.time(),
                "last_active": time.time(),
            }
