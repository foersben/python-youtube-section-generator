"""Embeddings providers for the RAG system.

This module defines a small, typed embeddings provider API and a factory for
creating providers. It defaults to sentence-transformers on CPU and returns
native Python floats (lists) for compatibility with ChromaDB and LangChain.

Usage:
    from src.core.embeddings import EmbeddingsFactory
    provider = EmbeddingsFactory.create_provider(device="cpu")
    vectors = provider.embed_documents(["text1", "text2"])
"""

from __future__ import annotations

from .provider import EmbeddingsProvider, SentenceTransformerEmbeddings
from .factory import EmbeddingsFactory

__all__ = [
    "EmbeddingsProvider",
    "SentenceTransformerEmbeddings",
    "EmbeddingsFactory",
]
