"""FastAPI application entry point for RAG Email Reply Generator."""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.middleware.error_handler import ErrorHandlerMiddleware
from backend.models import HealthResponse
from backend.routers import dataset_router, email_router, search_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Global pipeline instance (initialized at startup)
_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    global _pipeline

    logger.info("Starting RAG Email Reply Generator...")
    logger.info("LLM Provider: %s", settings.llm_provider)
    logger.info("Vector DB: %s", settings.vector_db_type)

    # Initialize the RAG pipeline
    from rag_pipeline.pipeline import create_pipeline

    _pipeline = create_pipeline(settings)
    app.state.pipeline = _pipeline

    logger.info("RAG pipeline initialized successfully.")

    yield

    logger.info("Shutting down RAG Email Reply Generator...")
    _pipeline = None


app = FastAPI(
    title="RAG Email Reply Generator",
    description="Production-ready Generative Email Reply Generator using RAG with LLMs",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom error handling middleware
app.add_middleware(ErrorHandlerMiddleware)

# Include routers
app.include_router(email_router.router, prefix="/api/v1", tags=["email"])
app.include_router(dataset_router.router, prefix="/api/v1", tags=["dataset"])
app.include_router(search_router.router, prefix="/api/v1", tags=["search"])


@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_db=settings.vector_db_type,
        llm_provider=settings.llm_provider,
    )
