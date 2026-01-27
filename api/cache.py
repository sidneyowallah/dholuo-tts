import redis
import json
import hashlib
from typing import Optional, Any
import logging
from api.config import settings

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Connected to Redis successfully.")
        except redis.ConnectionError as e:
            logger.warning(f"Failed to connect to Redis: {e}. Caching will be disabled.")
            self.redis_client = None

    def generate_key(self, prefix: str, data: Any) -> str:
        """Generate a unique cache key based on data."""
        data_str = json.dumps(data, sort_keys=True)
        hashed = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}:{hashed}"

    def get(self, key: str) -> Optional[Any]:
        if not self.redis_client:
            return None
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        if not self.redis_client:
            return
        try:
            self.redis_client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def is_connected(self) -> bool:
        return self.redis_client is not None

cache = CacheService()
