"""Tests for ChromaDB vector store operations."""
import os
import tempfile

import pytest

from vector_db.embeddings import EmbeddingGenerator


class MockEmbeddingGenerator:
    """Simple mock embedding generator that avoids loading real ML models."""

    embedding_dimension = 384

    def embed_text(self, text: str) -> list:
        rng = hash(text) % 1000
        vec = [float((rng + i) % 100) / 100.0 for i in range(384)]
        norm = sum(v ** 2 for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list) -> list:
        return [self.embed_text(t) for t in texts]


@pytest.fixture
def embedding_gen():
    """Create a simple mock embedding generator for tests."""
    return MockEmbeddingGenerator()


@pytest.fixture
def chroma_store(embedding_gen):
    """Create a temporary ChromaDB store for testing."""
    try:
        from vector_db.chroma_store import ChromaVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaVectorStore(
                persist_directory=tmpdir,
                collection_name="test_collection",
                embedding_generator=embedding_gen,
            )
            yield store
    except ImportError:
        pytest.skip("chromadb not installed")


def test_add_and_search_documents(chroma_store):
    """Test adding documents and performing similarity search."""
    docs = [
        "Refund policy: full refund within 30 days",
        "Technical support for software issues",
        "HR vacation request process",
    ]
    metadatas = [{"source": "policy"}, {"source": "tech"}, {"source": "hr"}]
    ids = ["doc_1", "doc_2", "doc_3"]

    chroma_store.add_documents(documents=docs, metadatas=metadatas, ids=ids)

    results = chroma_store.similarity_search("refund money back", top_k=2)
    assert len(results) >= 1
    assert all(isinstance(score, float) for _, score, _ in results)
    assert all(0.0 <= score <= 1.0 for _, score, _ in results)


def test_empty_collection_returns_no_results(chroma_store):
    """Test that searching empty collection returns empty results."""
    results = chroma_store.similarity_search("query", top_k=5)
    assert results == []


def test_get_collection_stats(chroma_store):
    """Test collection statistics."""
    chroma_store.add_documents(["test doc"], [{"src": "test"}], ["id_1"])
    stats = chroma_store.get_collection_stats()
    assert "document_count" in stats
    assert stats["document_count"] >= 1


def test_delete_collection(chroma_store):
    """Test collection deletion resets document count."""
    chroma_store.add_documents(
        ["doc 1", "doc 2"], [{"src": "test"}] * 2, ["a", "b"]
    )
    chroma_store.delete_collection()
    stats = chroma_store.get_collection_stats()
    assert stats["document_count"] == 0


def test_add_documents_without_ids(chroma_store):
    """Test that documents can be added without explicit IDs."""
    chroma_store.add_documents(
        documents=["Auto-ID document"],
        metadatas=[{"key": "value"}],
    )
    stats = chroma_store.get_collection_stats()
    assert stats["document_count"] >= 1
