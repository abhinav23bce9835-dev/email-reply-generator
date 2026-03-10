"""Embedding generation using Sentence Transformers or OpenAI."""
import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates text embeddings using Sentence Transformers (default)
    or OpenAI embeddings (optional).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the embedding generator (model loaded lazily).

        Args:
            model_name: Sentence Transformers model name or OpenAI embedding model name.
            use_openai: Whether to use OpenAI embeddings instead of local model.
            openai_api_key: OpenAI API key (required if use_openai=True).
        """
        self.model_name = model_name
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        self._model: Any = None
        logger.info(
            "EmbeddingGenerator configured: model=%s, use_openai=%s",
            model_name,
            use_openai,
        )

    def _load_model(self) -> None:
        """Lazily load the embedding model."""
        if self._model is not None:
            return

        if self.use_openai:
            try:
                import openai

                self._model = openai.OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI embedding client loaded")
            except ImportError as exc:
                raise ImportError("openai package required for OpenAI embeddings") from exc
        else:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                logger.info("SentenceTransformer model loaded: %s", self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers package required. Run: pip install sentence-transformers"
                ) from exc

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            Embedding as a list of floats.
        """
        self._load_model()

        if self.use_openai:
            response = self._model.embeddings.create(
                input=[text], model=self.model_name
            )
            return response.data[0].embedding

        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        self._load_model()

        if self.use_openai:
            response = self._model.embeddings.create(
                input=texts, model=self.model_name
            )
            return [item.embedding for item in response.data]

        embeddings = self._model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings.tolist()

    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        self._load_model()
        if self.use_openai:
            # Dimension depends on the model:
            # text-embedding-3-small: 1536, text-embedding-ada-002: 1536,
            # text-embedding-3-large: 3072. Default assumes small/ada.
            return 1536
        return self._model.get_sentence_embedding_dimension()
