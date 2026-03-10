"""ChromaDB vector store implementation."""
import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Vector store backed by ChromaDB with persistent storage.
    Provides document addition, similarity search, and management.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        collection_name: str = "email_knowledge_base",
        embedding_generator: Optional[Any] = None,
    ) -> None:
        """
        Initialize the ChromaDB vector store.

        Args:
            persist_directory: Path to persist ChromaDB data on disk.
            collection_name: Name of the ChromaDB collection.
            embedding_generator: EmbeddingGenerator instance for creating vectors.
        """
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise ImportError(
                "chromadb package required. Run: pip install chromadb"
            ) from exc

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_generator = embedding_generator

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore initialized: dir=%s, collection=%s",
            persist_directory,
            collection_name,
        )

    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document text strings.
            metadatas: Optional list of metadata dicts for each document.
            ids: Optional list of unique IDs. Auto-generated if not provided.
        """
        if not documents:
            return

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        else:
            # ChromaDB requires non-empty metadata dicts
            metadatas = [m if m else {"source": "unknown"} for m in metadatas]

        # Generate embeddings
        if self.embedding_generator:
            embeddings = self.embedding_generator.embed_batch(documents)
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

        logger.info("Added %d documents to ChromaDB collection '%s'", len(documents), self.collection_name)

    def similarity_search(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Perform cosine similarity search against stored documents.

        Args:
            query_text: Query text to search for.
            top_k: Number of top results to return.

        Returns:
            List of (document_text, similarity_score, metadata) tuples.
        """
        if self.collection.count() == 0:
            logger.warning("ChromaDB collection '%s' is empty", self.collection_name)
            return []

        # Generate query embedding
        if self.embedding_generator:
            query_embedding = self.embedding_generator.embed_text(query_text)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "distances", "metadatas"],
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "distances", "metadatas"],
            )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        # ChromaDB returns distances; convert to similarity scores
        # For cosine space: score = 1 - distance
        output = []
        for doc, dist, meta in zip(documents, distances, metadatas):
            similarity = max(0.0, 1.0 - dist)
            output.append((doc, round(similarity, 4), meta or {}))

        return output

    def delete_collection(self) -> None:
        """Delete the entire ChromaDB collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Deleted and recreated ChromaDB collection '%s'", self.collection_name)

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the current collection.

        Returns:
            Dictionary with count and collection name.
        """
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.persist_directory,
        }
