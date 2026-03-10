"""Semantic context search router."""
import logging

from fastapi import APIRouter, HTTPException, Request

from backend.models import RetrievedContext, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search_context", response_model=SearchResponse)
async def search_context(request: Request, search_request: SearchRequest) -> SearchResponse:
    """
    Perform semantic similarity search against the knowledge base.

    Args:
        request: FastAPI request object containing app state.
        search_request: Search query and parameters.

    Returns:
        SearchResponse with ranked context documents.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    if not search_request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    logger.info(
        "Searching context for query: '%s', top_k=%d",
        search_request.query[:100],
        search_request.top_k,
    )

    try:
        results = pipeline.retriever.retrieve(
            query=search_request.query,
            top_k=search_request.top_k,
            filters=search_request.filters,
        )
        contexts = [
            RetrievedContext(
                content=doc,
                similarity_score=score,
                source=meta.get("source_type") if meta else None,
                metadata=meta,
            )
            for doc, score, meta in results
        ]
        return SearchResponse(
            results=contexts,
            query=search_request.query,
            total_results=len(contexts),
        )
    except Exception as exc:
        logger.error("Search failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Search failed: {str(exc)}"
        ) from exc
