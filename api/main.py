from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from api.routes import router as api_router
from api.config import settings
from api.inference import inference_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Dholuo TTS API...")
    # Trigger model loading if not already initialized (it is lazy loaded in singleton, but good to warm up)
    # inference_engine is instantiated at module level in routes/inference, 
    # but we can access it here to log status
    if inference_engine._initialized:
        logger.info("Inference engine is ready.")
    else:
        logger.info("Inference engine initializing...")
        
    yield
    # Shutdown
    logger.info("Shutting down Dholuo TTS API...")
    if hasattr(inference_engine, "redis_client") and inference_engine.redis_client:
         inference_engine.redis_client.close()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "*" # Configure properly for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "Welcome to Dholuo TTS API", "docs": "/docs"}
