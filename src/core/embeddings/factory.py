"""Embeddings factory for creating provider instances.

This module provides a factory pattern for creating embeddings providers
with sensible defaults.
"""

from __future__ import annotations

from .provider import EmbeddingsProvider, SentenceTransformerEmbeddings


class EmbeddingsFactory:
    """Factory for creating embeddings providers.

    Defaults are conservative (CPU + MiniLM) to minimize resource usage.
    """

    @staticmethod
    def create_provider(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu") -> EmbeddingsProvider:
        """Create an embeddings provider instance.

        Args:
            model_name: Model identifier for sentence-transformers.
            device: Device to run on ("cpu" or "cuda").

        Returns:
            Configured embeddings provider.
        """
        return SentenceTransformerEmbeddings(model_name=model_name, device=device)


__all__ = ["EmbeddingsFactory"]
