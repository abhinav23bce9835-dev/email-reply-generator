"""
Test suite for the FastAPI backend endpoints.
"""

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App fixture — import after setting env vars so startup uses a temp directory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Create a TestClient with a temporary ChromaDB directory."""
    import os

    tmp_dir = str(tmp_path_factory.mktemp("chroma"))
    os.environ["CHROMA_PERSIST_DIR"] = tmp_dir
    os.environ["COLLECTION_NAME"] = "test_email_collection"

    # Lazy import so env vars are set before the module-level lifespan runs
    from backend.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_check(client):
    """GET / should return 200 with running status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "RAG Email Reply Generator" in data["service"]


def test_health_detailed(client):
    """GET /api/v1/health should return status field."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_generate_reply(client):
    """POST /api/v1/generate_reply should return a valid EmailReply."""
    payload = {
        "subject": "Question about pricing",
        "body": "Hi, I would like to know more about your enterprise pricing plans.",
        "sender": "client@example.com",
        "tone": "formal",
    }
    response = client.post("/api/v1/generate_reply", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply_text" in data
    assert isinstance(data["reply_text"], str)
    assert len(data["reply_text"]) > 0
    assert "confidence_score" in data
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert "intent" in data
    assert "retrieved_contexts" in data
    assert "metadata" in data


def test_generate_reply_different_tones(client):
    """POST /api/v1/generate_reply should work with all supported tones."""
    for tone in ("formal", "friendly", "professional"):
        payload = {
            "subject": "Meeting request",
            "body": "Can we schedule a call to discuss the project?",
            "tone": tone,
        }
        response = client.post("/api/v1/generate_reply", json=payload)
        assert response.status_code == 200, f"Failed for tone: {tone}"
        data = response.json()
        assert len(data["reply_text"]) > 0


def test_generate_reply_missing_body(client):
    """POST /api/v1/generate_reply with missing required field should return 422."""
    payload = {"subject": "Test subject"}
    response = client.post("/api/v1/generate_reply", json=payload)
    assert response.status_code == 422


def test_search_context(client):
    """POST /api/v1/search_context should return SearchResponse."""
    payload = {"query": "pricing inquiry", "top_k": 3}
    response = client.post("/api/v1/search_context", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "query" in data
    assert data["query"] == "pricing inquiry"
    assert "total_results" in data
    assert isinstance(data["results"], list)


def test_upload_dataset(client):
    """POST /api/v1/upload_dataset should process and store emails."""
    payload = {
        "emails": [
            {
                "id": "test_001",
                "subject": "Test meeting request",
                "body": "Can we schedule a call next week?",
                "sender": "test@example.com",
                "recipient": "team@example.com",
                "intent": "scheduling",
            },
            {
                "id": "test_002",
                "subject": "Bug report: login fails",
                "body": "The login button is not working since the latest update.",
                "sender": "user@example.com",
                "recipient": "support@example.com",
                "intent": "complaint",
            },
        ]
    }
    response = client.post("/api/v1/upload_dataset", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["emails_processed"] == 2
