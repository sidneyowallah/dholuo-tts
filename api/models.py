from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict

class SynthesizeRequest(BaseModel):
    text: str = Field(..., max_length=1000, description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Voice ID or name")
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    format: Optional[str] = Field("wav", pattern="^(wav|mp3|ogg)$", description="Audio format")
    return_ipa: Optional[bool] = Field(False, description="Whether to return IPA transcription")

class SynthesizeResponse(BaseModel):
    audio: str = Field(..., description="Base64 encoded audio or URL")
    duration: float = Field(..., description="Audio duration in seconds")
    ipa_text: Optional[str] = Field(None, description="IPA transcription if requested")
    cached: bool = Field(False, description="Whether result was served from cache")
    processing_time: float = Field(..., description="Processing time in seconds")

class BatchSynthesizeRequest(BaseModel):
    texts: List[str] = Field(..., max_items=50, description="List of texts to synthesize")
    voice: Optional[str] = Field(None)
    callback_url: Optional[str] = Field(None, description="Webhook URL for results")

class BatchJobResponse(BaseModel):
    job_id: str
    status: str
    total_items: int

class TagRequest(BaseModel):
    text: str = Field(..., max_length=5000)

class TagResponse(BaseModel):
    tagged_pairs: List[Union[tuple, List[str]]] = Field(..., description="List of (word, tag) pairs")

class PhonemizeRequest(BaseModel):
    text: str = Field(..., max_length=5000)

class PhonemizeResponse(BaseModel):
    ipa_text: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool
    version: str

class MetricsResponse(BaseModel):
    request_count: int
    average_latency: float
    cache_hit_rate: float
