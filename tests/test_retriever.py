"""Tests for the EmailRetriever module."""
from unittest.mock import MagicMock

import pytest

from rag_pipeline.retriever import EmailRetriever


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock()
    store.similarity_search.return_value = [
        ("Document about refunds", 0.85, {"source_type": "customer_service"}),
        ("Document about returns", 0.70, {"source_type": "customer_service"}),
        ("Unrelated document", 0.20, {"source_type": "other"}),
    ]
    return store


@pytest.fixture
def retriever(mock_vector_store):
    return EmailRetriever(vector_store=mock_vector_store, relevance_threshold=0.3)


def test_retrieve_returns_filtered_results(retriever, mock_vector_store):
    """Test that retriever filters by relevance threshold."""
    results = retriever.retrieve("refund policy", top_k=5)
    # Score 0.20 should be filtered out
    assert len(results) == 2
    for doc, score, meta in results:
        assert score >= 0.3


def test_retrieve_sorted_by_score(retriever, mock_vector_store):
    """Test that results are sorted by score descending."""
    results = retriever.retrieve("refund policy", top_k=5)
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_empty_query(retriever, mock_vector_store):
    """Test that empty query returns empty results."""
    results = retriever.retrieve("")
    assert results == []
    mock_vector_store.similarity_search.assert_not_called()


def test_retrieve_calls_vector_store(retriever, mock_vector_store):
    """Test that retriever delegates to vector store."""
    retriever.retrieve("test query", top_k=3)
    mock_vector_store.similarity_search.assert_called_once_with(
        query_text="test query", top_k=3
    )


def test_retrieve_handles_vector_store_error(retriever, mock_vector_store):
    """Test that retriever handles vector store errors gracefully."""
    mock_vector_store.similarity_search.side_effect = Exception("DB error")
    results = retriever.retrieve("test query")
    assert results == []


def test_retrieve_all_below_threshold():
    """Test retriever returns empty when all results are below threshold."""
    store = MagicMock()
    store.similarity_search.return_value = [
        ("Low relevance doc", 0.1, {}),
        ("Another low doc", 0.15, {}),
    ]
    retriever = EmailRetriever(vector_store=store, relevance_threshold=0.5)
    results = retriever.retrieve("query")
    assert results == []
