"""Document retrieval logic for the RAG pipeline."""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EmailRetriever:
    """
    Retrieves relevant email documents from the vector store
    based on query similarity.
    """

    def __init__(self, vector_store: Any, relevance_threshold: float = 0.3) -> None:
        """
        Initialize the retriever.

        Args:
            vector_store: A ChromaVectorStore or FAISSVectorStore instance.
            relevance_threshold: Minimum similarity score for a result to be included.
        """
        self.vector_store = vector_store
        self.relevance_threshold = relevance_threshold
        logger.info(
            "EmailRetriever initialized with threshold=%.2f", relevance_threshold
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Retrieve the most relevant documents for a query.

        Args:
            query: Query string to search for.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of (document_text, similarity_score, metadata) tuples,
            filtered by relevance_threshold and sorted by score descending.
        """
        if not query.strip():
            logger.warning("Empty query provided to retriever")
            return []

        try:
            raw_results = self.vector_store.similarity_search(
                query_text=query, top_k=top_k
            )
        except Exception as exc:
            logger.error("Vector store search failed: %s", exc, exc_info=True)
            return []

        # Filter by relevance threshold
        filtered = [
            (doc, score, meta)
            for doc, score, meta in raw_results
            if score >= self.relevance_threshold
        ]

        # Sort by score descending
        filtered.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            "Retrieved %d/%d results above threshold %.2f",
            len(filtered),
            len(raw_results),
            self.relevance_threshold,
        )
        return filtered
