"""Disk-backed cache helpers for embeddings.

This module provides a tiny JSON-based cache for storing and retrieving
embeddings vectors for transcript chunks. The cache is keyed by a stable
hash that includes video ID, model name, device, and chunking params so
that changes invalidate old entries automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _safe_hash(parts: list[str]) -> str:
    """Return a short, stable hash for a list of string parts."""
    data = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(data).hexdigest()[:16]


def build_embedding_cache_key(
    video_id: str,
    model_name: str,
    device: str,
    chunk_size: int,
    chunk_overlap: int,
    extra_tag: str | None = None,
) -> str:
    """Build a deterministic cache key for a segmentation run.

    Args:
        video_id: YouTube video identifier.
        model_name: Embedding model identifier.
        device: Device string (e.g. "cpu").
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap characters.
        extra_tag: Optional extra tag, for future versioning.

    Returns:
        Short hash string that uniquely identifies this configuration.
    """
    parts = [
        video_id,
        model_name,
        device,
        str(chunk_size),
        str(chunk_overlap),
    ]
    if extra_tag:
        parts.append(extra_tag)
    return _safe_hash(parts)


def load_cached_embeddings(cache_dir: Path, cache_key: str) -> list[list[float]] | None:
    """Load cached embeddings if available.

    Args:
        cache_dir: Directory where cache files live.
        cache_key: Cache key as returned by ``build_embedding_cache_key``.

    Returns:
        List of embedding vectors or ``None`` when cache is missing/invalid.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.json"
        if not cache_path.exists():
            return None
        with cache_path.open("r", encoding="utf-8") as f:
            data: Any = json.load(f)
        if not isinstance(data, list):
            return None
        # Perform a very lightweight structural check: list[list[number-like]]
        vectors: list[list[float]] = []
        for row in data:
            if not isinstance(row, list):
                return None
            try:
                vectors.append([float(x) for x in row])
            except (TypeError, ValueError):
                return None
        logger.debug("Loaded %d cached embedding vectors from %s", len(vectors), cache_path)
        return vectors
    except Exception:
        logger.debug("Failed to load embedding cache", exc_info=True)
        return None


def save_cached_embeddings(
    cache_dir: Path,
    cache_key: str,
    embeddings: list[list[float]],
) -> None:
    """Persist embeddings to disk as JSON.

    Errors are logged but never raised to callers; caching is best-effort.

    Args:
        cache_dir: Directory where cache files live.
        cache_key: Cache key as returned by ``build_embedding_cache_key``.
        embeddings: Embedding matrix to persist.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.json"
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(embeddings, f)
        logger.debug("Saved %d embedding vectors to %s", len(embeddings), cache_path)
    except Exception:
        logger.debug("Failed to save embedding cache", exc_info=True)


__all__ = [
    "build_embedding_cache_key",
    "load_cached_embeddings",
    "save_cached_embeddings",
]

