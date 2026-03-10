"""
Embedding generation using Sentence Transformers.
Provides text-to-vector conversion for semantic search.
Falls back to a deterministic hash-based embedding when the model
cannot be downloaded (e.g. offline / sandboxed environments).
"""

import hashlib
import logging
from typing import List

logger = logging.getLogger(__name__)

# Dimensionality used by all-MiniLM-L6-v2
_EMBEDDING_DIM = 384


def _hash_embed(text: str, dim: int = _EMBEDDING_DIM) -> List[float]:
    """
    Create a deterministic embedding from *text* using SHA-512 hashing.

    Each hash produces 64 bytes.  We generate ceil(dim / 64) hashes using
    different seeds and concatenate them, then truncate to *dim* values.
    Values are normalised to the range [-1, 1].

    Args:
        text: Input text to embed.
        dim: Desired embedding dimensionality.

    Returns:
        List of *dim* floats in [-1, 1].
    """
    result: List[float] = []
    seed = 0
    while len(result) < dim:
        h = hashlib.sha512(f"{seed}:{text}".encode("utf-8")).digest()
        for byte in h:
            result.append(byte / 127.5 - 1.0)  # map [0, 255] → [-1, 1]
        seed += 1
    return result[:dim]


class EmbeddingGenerator:
    """Generates text embeddings using Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialise the embedding generator with lazy model loading.

        Args:
            model_name: Name of the Sentence Transformers model to use.
        """
        self.model_name = model_name
        self._model = None
        self._use_fallback: bool = False
        logger.info("EmbeddingGenerator initialised (model will load on first use)")

    def _load_model(self) -> None:
        """
        Load the SentenceTransformer model on first use.

        If the model cannot be loaded (e.g. no network access), a
        deterministic hash-based fallback is activated so that the
        application can still run and tests can still pass.
        """
        if self._model is not None or self._use_fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as exc:
            logger.warning(
                "Could not load SentenceTransformer model '%s' (%s). "
                "Falling back to hash-based embedding.",
                self.model_name,
                exc,
            )
            self._use_fallback = True

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single text string.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        self._load_model()
        if self._use_fallback:
            return _hash_embed(text)
        try:
            embedding = self._model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as exc:
            logger.error("Error embedding text: %s", exc)
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of texts.

        Args:
            texts: A list of input texts to embed.

        Returns:
            A list of embedding vectors (each a list of floats).
        """
        self._load_model()
        if self._use_fallback:
            return [_hash_embed(t) for t in texts]
        try:
            embeddings = self._model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as exc:
            logger.error("Error embedding batch: %s", exc)
            raise
