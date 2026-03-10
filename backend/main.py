"""
FastAPI backend for RAG Email Reply Generator.
Provides endpoints for email reply generation, dataset upload, and context search.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.dataset_loader import DatasetLoader
from backend.embeddings import EmbeddingGenerator
from backend.rag_pipeline import RAGPipeline
from backend.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_vector_store: Optional[ChromaVectorStore] = None
_rag_pipeline: Optional[RAGPipeline] = None


def _get_vector_store() -> ChromaVectorStore:
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialised")
    return _vector_store


def _get_rag_pipeline() -> RAGPipeline:
    if _rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialised")
    return _rag_pipeline


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[misc]
    global _vector_store, _rag_pipeline  # noqa: PLW0603

    logger.info("Starting up RAG Email Reply Generator …")

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    collection_name = os.getenv("COLLECTION_NAME", "email_collection")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    dataset_path = os.getenv("DATASET_PATH", "dataset/sample_emails.json")

    embedding_gen = EmbeddingGenerator(model_name=embedding_model)
    _vector_store = ChromaVectorStore(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_generator=embedding_gen,
    )
    _rag_pipeline = RAGPipeline(
        vector_store=_vector_store,
        embedding_generator=embedding_gen,
    )

    # Load sample emails on startup
    try:
        emails = DatasetLoader.load_sample_emails(filepath=dataset_path)
        if emails:
            docs, metas, ids = DatasetLoader.prepare_for_vectorstore(emails)
            count = _vector_store.add_documents(docs, metas, ids)
            logger.info("Loaded %d sample emails into vector store", count)
        else:
            logger.warning("No sample emails loaded (file missing or empty)")
    except Exception as exc:
        logger.error("Failed to load sample emails: %s", exc)

    logger.info("Startup complete")
    yield
    logger.info("Shutting down …")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Email Reply Generator",
    description="Generate context-aware email replies using Retrieval-Augmented Generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EmailRequest(BaseModel):
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body text")
    sender: Optional[str] = Field(None, description="Sender name or email")
    tone: str = Field("formal", description="Desired reply tone: formal, friendly, professional")


class EmailReply(BaseModel):
    reply_text: str
    confidence_score: float
    intent: str
    retrieved_contexts: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class DatasetUploadRequest(BaseModel):
    emails: List[Dict[str, Any]] = Field(..., description="List of email dicts")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total_results: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "running", "service": "RAG Email Reply Generator"}


@app.get("/api/v1/health")
async def health() -> Dict[str, Any]:
    """Detailed health check with vector store statistics."""
    try:
        store = _get_vector_store()
        stats = store.get_collection_stats()
        return {
            "status": "healthy",
            "service": "RAG Email Reply Generator",
            "vector_store": stats,
        }
    except HTTPException:
        return {"status": "degraded", "service": "RAG Email Reply Generator"}
    except Exception as exc:
        logger.error("Health check error: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}


@app.post("/api/v1/generate_reply", response_model=EmailReply)
async def generate_reply(request: EmailRequest) -> EmailReply:
    """Generate a context-aware email reply using the RAG pipeline."""
    pipeline = _get_rag_pipeline()
    try:
        result = pipeline.generate_reply(
            subject=request.subject,
            body=request.body,
            sender=request.sender or "User",
            tone=request.tone,
        )
        return EmailReply(**result)
    except Exception as exc:
        logger.error("Error generating reply: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/upload_dataset")
async def upload_dataset(request: DatasetUploadRequest) -> Dict[str, Any]:
    """Process and store a batch of emails in the vector store."""
    store = _get_vector_store()
    try:
        docs, metas, ids = DatasetLoader.prepare_for_vectorstore(request.emails)
        count = store.add_documents(docs, metas, ids)
        return {
            "status": "success",
            "emails_processed": count,
            "message": f"Successfully added {count} emails to the vector store",
        }
    except Exception as exc:
        logger.error("Error uploading dataset: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/search_context", response_model=SearchResponse)
async def search_context(request: SearchRequest) -> SearchResponse:
    """Search the vector store for emails similar to the query."""
    store = _get_vector_store()
    try:
        results = store.similarity_search(request.query, top_k=request.top_k)
        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
        )
    except Exception as exc:
        logger.error("Error searching context: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
