"""Embeddings provider implementations.

This module contains concrete implementations of the EmbeddingsProvider interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EmbeddingsProvider(ABC):
    """Abstract base class for embeddings providers.

    Implementations must provide methods that return native Python floats to
    avoid dtype issues with downstream vector stores.

    Methods:
        embed_documents: Embed multiple documents and return list of vectors.
        embed_query: Embed a single query and return a single vector.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents.

        Args:
            texts: List of text documents to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        raise NotImplementedError


class SentenceTransformerEmbeddings(EmbeddingsProvider):
    """sentence-transformers based provider.

    Attributes:
        model_name: SentenceTransformer model identifier.
        device: Device string ("cpu" or "cuda"). Defaults to "cpu".
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - runtime check
            raise RuntimeError(
                "sentence-transformers required. Install with: "
                "poetry run pip install sentence-transformers"
            ) from exc

        logger.info("Loading embeddings model: %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.device = device

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents and return native float lists.

        Uses the sentence-transformers v5.x asymmetric document encoder when
        available, falling back to ``encode`` for older versions.
        """
        import numpy as np  # local import to keep import time small

        # Prefer modern asymmetric API when present
        encode_docs = getattr(self.model, "encode_document", None)
        if callable(encode_docs):
            vectors = encode_docs(texts, show_progress_bar=False, convert_to_numpy=True)
        else:  # pragma: no cover - backward compatibility path
            vectors = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        if isinstance(vectors, np.ndarray):
            return np.asarray(vectors).astype(float).tolist()
        return [[float(x) for x in v] for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query and return a vector as list[float].

        Uses the sentence-transformers v5.x asymmetric query encoder when
        available, falling back to ``encode`` for older versions.
        """
        import numpy as np

        encode_query = getattr(self.model, "encode_query", None)
        if callable(encode_query):
            vec = encode_query([text], show_progress_bar=False, convert_to_numpy=True)[0]
        else:  # pragma: no cover - backward compatibility path
            vec = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]

        if isinstance(vec, np.ndarray):
            return np.asarray(vec).astype(float).tolist()
        return [float(x) for x in vec]


__all__ = ["EmbeddingsProvider", "SentenceTransformerEmbeddings"]
