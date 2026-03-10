"""FAISS vector store implementation."""
import json
import logging
import os
import uuid
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Vector store backed by FAISS for high-performance similarity search.
    Uses IndexFlatIP (inner product) with normalized vectors for cosine similarity.
    Stores metadata separately in a JSON-serializable dict.
    """

    def __init__(
        self,
        embedding_generator: Any,
        index_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the FAISS vector store.

        Args:
            embedding_generator: EmbeddingGenerator for creating vectors.
            index_path: Optional path to persist/load the FAISS index.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu package required. Run: pip install faiss-cpu") from exc

        self.embedding_generator = embedding_generator
        self.index_path = index_path

        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._ids: list[str] = []
        self._index: Any = None

        if index_path and os.path.exists(f"{index_path}.index"):
            self.load_index(index_path)
        else:
            dim = embedding_generator.embedding_dimension
            import faiss as faiss_lib

            self._index = faiss_lib.IndexFlatIP(dim)

        logger.info("FAISSVectorStore initialized (dim=%d)", self._index.d)

    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> None:
        """
        Add documents to the FAISS index.

        Args:
            documents: List of document text strings.
            metadatas: Optional list of metadata dicts.
            ids: Optional list of unique IDs.
        """
        if not documents:
            return

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        embeddings = self.embedding_generator.embed_batch(documents)
        vectors = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity (inner product on normalized = cosine)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = vectors / norms

        self._index.add(vectors)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)
        self._ids.extend(ids)

        logger.info("Added %d documents to FAISS index (total: %d)", len(documents), self._index.ntotal)

        if self.index_path:
            self.save_index(self.index_path)

    def similarity_search(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Perform cosine similarity search using FAISS.

        Args:
            query_text: Query text to search for.
            top_k: Number of top results to return.

        Returns:
            List of (document_text, similarity_score, metadata) tuples.
        """
        if self._index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []

        query_vec = np.array(
            [self.embedding_generator.embed_text(query_text)], dtype=np.float32
        )
        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            results.append((
                self._documents[idx],
                round(float(score), 4),
                self._metadatas[idx],
            ))

        return results

    def save_index(self, path: str) -> None:
        """
        Save FAISS index and metadata to disk.

        Args:
            path: Base path for saving (extensions added automatically).
        """
        import faiss

        faiss.write_index(self._index, f"{path}.index")
        with open(f"{path}.meta", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                    "ids": self._ids,
                },
                f,
            )
        logger.info("FAISS index saved to '%s'", path)

    def load_index(self, path: str) -> None:
        """
        Load FAISS index and metadata from disk.

        Args:
            path: Base path to load from.
        """
        import faiss

        self._index = faiss.read_index(f"{path}.index")
        if os.path.exists(f"{path}.meta"):
            with open(f"{path}.meta", "r", encoding="utf-8") as f:
                data = json.load(f)
                self._documents = data.get("documents", [])
                self._metadatas = data.get("metadatas", [])
                self._ids = data.get("ids", [])
        logger.info("FAISS index loaded from '%s' (%d docs)", path, self._index.ntotal)

    def delete_collection(self) -> None:
        """Reset the FAISS index and clear all stored documents."""
        import faiss

        dim = self._index.d
        self._index = faiss.IndexFlatIP(dim)
        self._documents = []
        self._metadatas = []
        self._ids = []
        logger.info("FAISS index reset")

    def get_collection_stats(self) -> dict[str, Any]:
        """Return statistics about the FAISS index."""
        return {
            "document_count": self._index.ntotal,
            "embedding_dimension": self._index.d,
            "index_path": self.index_path,
        }
