from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
import time
import uuid
from typing import List

from api.models import (
    SynthesizeRequest, SynthesizeResponse, 
    TagRequest, TagResponse, 
    PhonemizeRequest, PhonemizeResponse,
    HealthResponse, MetricsResponse,
    BatchSynthesizeRequest, BatchJobResponse
)
from api.inference import inference_engine
from api.cache import cache
from api.config import settings

router = APIRouter()

# Metrics storage (simple in-memory for now, ideally Prometheus)
metrics = {
    "request_count": 0,
    "total_latency": 0.0,
    "cache_hits": 0,
    "cache_misses": 0
}

@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    start_time = time.time()
    metrics["request_count"] += 1
    
    # Check cache
    cache_key = cache.generate_key("synth", request.dict())
    cached_result = cache.get(cache_key)
    
    if cached_result:
        metrics["cache_hits"] += 1
        cached_result["cached"] = True
        cached_result["processing_time"] = 0.0 # served almost instantly
        return cached_result
        
    metrics["cache_misses"] += 1
    
    try:
        # Run inference
        # Note: inference_engine.synthesize is blocking (CPU intensive), 
        # but in FastAPI sync routes run in threadpool. 
        # If it was async def, we'd block the event loop.
        # But here we define it as async def, so we should offload it if it blocks.
        # Ideally we'd use `run_in_threadpool` or just define route as `def synthesize`
        # However, for simplicity/consistency we keep async and rely on fast execution or assume low load for now.
        # Better pattern: `def synthesize(...)` (no async) tells FastAPI to run in threadpool.
        # Or explicitly use `run_in_threadpool`.
        # Given we have async cache logic (though currently sync redis), let's keep it async 
        # and assume inference is fast enough or acceptable for this POC. 
        # Actually, let's just make the route `def` if we strictly want threadpool default, 
        # but `inference_engine` also calls blocking code.
        # Let's wrap inference in run_in_threadpool for safety if we stay `async`.
        # OR just define route as `def`.
        # Wait, I declared router methods as `async def`. I'll switch to `def` for CPU bound tasks 
        # to let FastAPI use threadpool, BUT `cache.get` is synchronous currently. 
        # So `def` is safer for the blocking inference.
        
        # ACTUALLY: fastAPI runs `def` endpoints in a threadpool. `async def` runs in main loop.
        # Since `inference_engine.synthesize` is blocking, I should use `def synthesize` OR `await run_in_threadpool`.
        # I'll stick to `async def` + `run_in_threadpool` if I want to use `await` elsewhere later.
        # But to be simple, I will call the synchronous inference directly but define the route as `def`?
        # No, mixing async (if I actally used async redis) and sync is tricky.
        # I'll define it as `def` to be safe with blocking CPU work.
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Re-implementing as `def` isn't compatible with `await` unless I remove `async`.
    # But I see I made `synthesize` `async`. I should probably use `starlette.concurrency.run_in_threadpool`.
    
    from starlette.concurrency import run_in_threadpool
    
    try:
        result = await run_in_threadpool(
            inference_engine.synthesize, 
            request.text, 
            request.speed, 
            request.return_ipa
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")
        
    execution_time = time.time() - start_time
    metrics["total_latency"] += execution_time
    
    response_data = {
        "audio": result["audio"],
        "duration": result["duration"],
        "ipa_text": result.get("ipa_text"),
        "cached": False,
        "processing_time": execution_time
    }
    
    # Save to cache
    cache.set(cache_key, response_data, ttl=settings.CACHE_TTL_AUDIO)
    
    return response_data

@router.post("/tag", response_model=TagResponse)
async def tag_text(request: TagRequest):
    from starlette.concurrency import run_in_threadpool
    try:
        tagged = await run_in_threadpool(inference_engine.get_tagging, request.text)
        return {"tagged_pairs": tagged}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/phonemize", response_model=PhonemizeResponse)
async def phonemize_text(request: PhonemizeRequest):
    from starlette.concurrency import run_in_threadpool
    try:
        ipa = await run_in_threadpool(inference_engine.text_to_ipa, request.text)
        return {"ipa_text": ipa}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch processing
@router.post("/batch", response_model=BatchJobResponse)
async def batch_synthesize(request: BatchSynthesizeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    bg_task_params = {
        "job_id": job_id,
        "texts": request.texts,
        "voice": request.voice
    }
    
    # Initialize job status
    cache.set(f"job:{job_id}", {"status": "processing", "total": len(request.texts), "completed": 0, "results": []}, ttl=86400)
    
    background_tasks.add_task(process_batch, bg_task_params)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "total_items": len(request.texts)
    }

@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    job_data = cache.get(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_data

def process_batch(params):
    job_id = params["job_id"]
    texts = params["texts"]
    results = []
    
    for text in texts:
        try:
            # We don't cache individual items here to simplify, or we could. 
            # For simplicity, just run inference.
            res = inference_engine.synthesize(text)
            results.append({"text": text, "audio": res["audio"], "status": "success"})
        except Exception as e:
            results.append({"text": text, "status": "error", "error": str(e)})
            
        # Update progress
        current_job = cache.get(f"job:{job_id}")
        if current_job:
            current_job["completed"] += 1
            current_job["results"] = results
            # If done
            if current_job["completed"] == len(texts):
                current_job["status"] = "completed"
            cache.set(f"job:{job_id}", current_job, ttl=86400)

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": inference_engine._initialized,
        "redis_connected": cache.is_connected(),
        "version": "1.0.0"
    }

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    avg_latency = 0
    if metrics["request_count"] > 0:
        avg_latency = metrics["total_latency"] / metrics["request_count"]
    
    total_cache_ops = metrics["cache_hits"] + metrics["cache_misses"]
    hit_rate = 0
    if total_cache_ops > 0:
        hit_rate = metrics["cache_hits"] / total_cache_ops
        
    return {
        "request_count": metrics["request_count"],
        "average_latency": avg_latency,
        "cache_hit_rate": hit_rate
    }
