"""Tests for FastAPI endpoints using TestClient."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_pipeline():
    """Create a mock RAG pipeline."""
    from backend.models import EmailReply, RetrievedContext

    pipeline = MagicMock()
    pipeline.generate_reply = AsyncMock(
        return_value=EmailReply(
            reply_text="Dear Customer,\n\nThank you.\n\nBest regards,\nSupport",
            confidence_score=0.85,
            intent="request",
            retrieved_contexts=[
                RetrievedContext(
                    content="Sample context",
                    similarity_score=0.80,
                    source="test",
                    metadata={"source_type": "test"},
                )
            ],
            metadata={"tone": "formal", "elapsed_ms": 100, "entities": {}},
        )
    )
    pipeline.retriever = MagicMock()
    pipeline.retriever.vector_store = MagicMock()
    pipeline.retriever.vector_store.add_documents = MagicMock()
    pipeline.retriever.retrieve = MagicMock(
        return_value=[
            ("Sample context", 0.80, {"source_type": "test"}),
        ]
    )
    return pipeline


@pytest.fixture
def app(mock_pipeline):
    """Create test app with mock pipeline."""
    from backend.main import app

    app.state.pipeline = mock_pipeline
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "vector_db" in data


def test_generate_reply_success(client):
    """Test successful email reply generation."""
    response = client.post(
        "/api/v1/generate_reply",
        json={
            "subject": "Product inquiry",
            "body": "I would like to know more about your pricing.",
            "sender": "customer@example.com",
            "tone": "formal",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply_text" in data
    assert "confidence_score" in data
    assert "intent" in data
    assert "retrieved_contexts" in data


def test_generate_reply_missing_fields(client):
    """Test that missing required fields return validation error."""
    response = client.post(
        "/api/v1/generate_reply",
        json={"subject": "Test"},  # Missing 'body'
    )
    assert response.status_code == 422


def test_upload_dataset_success(client):
    """Test successful dataset upload."""
    response = client.post(
        "/api/v1/upload_dataset",
        json={
            "emails": [
                {
                    "subject": "Test email",
                    "body": "Test body content",
                    "sender": "test@example.com",
                    "recipient": "support@example.com",
                    "intent": "inquiry",
                    "reply": "Thank you for your inquiry.",
                }
            ],
            "source_type": "manual",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "processed_count" in data
    assert data["processed_count"] >= 0


def test_upload_dataset_empty(client):
    """Test upload with empty email list."""
    response = client.post(
        "/api/v1/upload_dataset",
        json={"emails": [], "source_type": "manual"},
    )
    assert response.status_code == 400


def test_search_context_success(client):
    """Test successful context search."""
    response = client.post(
        "/api/v1/search_context",
        json={"query": "refund policy", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "query" in data
    assert data["query"] == "refund policy"
    assert "total_results" in data


def test_search_context_empty_query(client):
    """Test search with empty query returns 400."""
    response = client.post(
        "/api/v1/search_context",
        json={"query": "", "top_k": 5},
    )
    assert response.status_code == 400


def test_api_docs_accessible(client):
    """Test that API docs are accessible."""
    response = client.get("/api/v1/docs")
    assert response.status_code == 200
