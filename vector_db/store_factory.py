"""Factory pattern to create the appropriate vector store."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_vector_store(settings: Any, embedding_generator: Any) -> Any:
    """
    Create and return the appropriate vector store based on configuration.

    Args:
        settings: Application settings with VECTOR_DB_TYPE field.
        embedding_generator: EmbeddingGenerator instance for vector creation.

    Returns:
        ChromaVectorStore or FAISSVectorStore depending on settings.

    Raises:
        ValueError: If an unsupported vector DB type is specified.
    """
    db_type = settings.vector_db_type.lower()
    logger.info("Creating vector store: type=%s", db_type)

    if db_type == "chroma":
        from vector_db.chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.collection_name,
            embedding_generator=embedding_generator,
        )
    elif db_type == "faiss":
        from vector_db.faiss_store import FAISSVectorStore

        return FAISSVectorStore(
            embedding_generator=embedding_generator,
            index_path=f"./faiss_data/{settings.collection_name}",
        )
    else:
        raise ValueError(
            f"Unsupported vector DB type: '{db_type}'. Choose 'chroma' or 'faiss'."
        )
