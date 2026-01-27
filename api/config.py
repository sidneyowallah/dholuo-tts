import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Dholuo TTS API"
    
    # Model Paths
    MODEL_PATH: str = "models/luo-tts/female/checkpoint_180000.pth"
    CONFIG_PATH: str = "models/luo-tts/female/config.json"
    
    # Cache Settings (Redis)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    CACHE_TTL_AUDIO: int = 86400  # 24 hours
    CACHE_TTL_IPA: int = 604800  # 7 days
    
    # TTS Defaults
    DEFAULT_SPEED: float = 1.0
    DEFAULT_FORMAT: str = "wav"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    class Config:
        case_sensitive = True

settings = Settings()
