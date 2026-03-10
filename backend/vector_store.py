"""
ChromaDB vector store implementation for email document storage and retrieval.
"""

import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Manages a ChromaDB collection for storing and retrieving email embeddings."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_generator: Any,
    ) -> None:
        """
        Initialise the ChromaDB client and collection.

        Args:
            persist_directory: Path to persist the ChromaDB data.
            collection_name: Name of the ChromaDB collection.
            embedding_generator: An EmbeddingGenerator instance.
        """
        import chromadb

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_generator = embedding_generator

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore initialised — collection '%s' at '%s'",
            collection_name,
            persist_directory,
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> int:
        """
        Add documents with metadata to the collection.

        Args:
            documents: List of text documents.
            metadatas: Corresponding list of metadata dicts.
            ids: Unique string IDs for each document.

        Returns:
            Number of documents added.
        """
        if not documents:
            logger.warning("add_documents called with empty list")
            return 0

        try:
            embeddings = self.embedding_generator.embed_batch(documents)
            self._collection.upsert(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info("Added/updated %d documents in collection", len(documents))
            return len(documents)
        except Exception as exc:
            logger.error("Error adding documents: %s", exc)
            raise

    def similarity_search(
        self, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the collection for documents similar to the query.

        Args:
            query_text: The query string to search for.
            top_k: Number of top results to return.

        Returns:
            List of dicts, each with 'content', 'score', and 'metadata'.
        """
        try:
            query_embedding = self.embedding_generator.embed_text(query_text)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(self._collection.count(), 1)),
                include=["documents", "distances", "metadatas"],
            )
            output: List[Dict[str, Any]] = []
            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            for doc, dist, meta in zip(docs, distances, metas):
                # Convert cosine distance to similarity score
                score = float(1 - dist)
                output.append({"content": doc, "score": score, "metadata": meta})
            return output
        except Exception as exc:
            logger.error("Error during similarity search: %s", exc)
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return basic statistics about the collection.

        Returns:
            Dict with document count and collection name.
        """
        try:
            count = self._collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
            }
        except Exception as exc:
            logger.error("Error getting collection stats: %s", exc)
            return {"collection_name": self.collection_name, "document_count": 0}

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self._client.delete_collection(self.collection_name)
            logger.info("Deleted collection '%s'", self.collection_name)
        except Exception as exc:
            logger.error("Error deleting collection: %s", exc)
            raise

    def clear_and_recreate(self) -> None:
        """Clear all documents and recreate the collection."""
        try:
            self.delete_collection()
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Collection '%s' cleared and recreated", self.collection_name)
        except Exception as exc:
            logger.error("Error clearing collection: %s", exc)
            raise
