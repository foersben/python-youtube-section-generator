"""Semantic segmentation helpers for transcript processing.

This module provides small utilities to chunk a transcript into overlapping
text windows, compute embeddings for each chunk, and select coarse section
boundaries based on adjacent-chunk semantic similarity.
"""

from __future__ import annotations

import logging
import os
import time
from math import floor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.core.embeddings.cache import (
    build_embedding_cache_key,
    load_cached_embeddings,
    save_cached_embeddings,
)
from src.core.embeddings.factory import EmbeddingsFactory

logger = logging.getLogger(__name__)


def chunk_transcript_text(
    transcript: list[dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[dict[str, Any]]:
    """Create overlapping text chunks from a transcript.

    Args:
        transcript: List of transcript segments where each segment contains
            at least 'start' (float seconds) and 'text' (string).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        List of chunk dicts with keys: 'start' (float) and 'text' (str).
    """
    text = ""
    offsets: list[tuple[int, float]] = []
    # Build a single long text with markers mapping char offset -> timestamp
    for seg in transcript:
        seg_text = str(seg.get("text", ""))
        start = float(seg.get("start", 0.0))
        offsets.append((len(text), start))
        # separate segments with a space to avoid accidental token join
        if text:
            text += " "
        text += seg_text

    if not text:
        return []

    chunks: list[dict[str, Any]] = []
    L = len(text)
    size = int(chunk_size)
    overlap = int(chunk_overlap)
    pos = 0
    while pos < L:
        end = min(pos + size, L)
        chunk_text = text[pos:end].strip()
        # estimate start timestamp for the chunk using nearest offset
        chunk_start = 0.0
        for off_ts in reversed(offsets):
            off, ts = off_ts
            if off <= pos:
                chunk_start = ts
                break
        chunks.append({"start": float(chunk_start), "text": chunk_text})
        if end == L:
            break
        pos = max(0, end - overlap)
    return chunks


def select_coarse_sections(
    transcript: list[dict[str, Any]],
    target_sections: int = 10,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str | None = None,
    device: str = "cpu",
    video_id: str | None = None,
    metrics_callback: Callable[[dict[str, float]], None] | None = None,
) -> list[dict[str, Any]]:
    """Select coarse section boundaries from a transcript.

    The algorithm:
    1. Chunk the transcript into overlapping windows.
    2. Embed each chunk using Sentence-Transformers provider.
    3. Compute adjacent-chunk similarities and pick the (target_sections - 1)
       lowest-similarity boundaries as candidate breaks.
    4. Return selected sections as start timestamps and aggregated text.

    Args:
        transcript: Transcript list of segments.
        target_sections: Desired number of sections to return.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.
        embedding_model: Optional identifier for the embeddings model; if
            None the factory default is used.
        device: Device for embeddings provider (default: 'cpu').
        video_id: Optional video identifier used for disk cache keying.
        metrics_callback: Optional callback that receives a dict of timing
            metrics (e.g. batch_avg_sec, batch_p95_sec).

    Returns:
        List of section dicts: [{'start': float, 'text': str}, ...]
    """
    if target_sections <= 0:
        raise ValueError("target_sections must be > 0")

    chunks = chunk_transcript_text(transcript, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return []

    # If we have fewer chunks than requested sections, fallback to chunk starts
    if len(chunks) <= target_sections:
        return [{"start": c["start"], "text": c["text"]} for c in chunks]

    texts = [c["text"] for c in chunks]

    model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
    provider = EmbeddingsFactory.create_provider(model_name=model_name, device=device)

    # Derive a cache key when we have a video_id; otherwise caching is disabled.
    cache_dir = Path(os.getenv("SEGMENTATION_CACHE_DIR", ".cache/embeddings"))
    cache_key = (
        build_embedding_cache_key(
            video_id=video_id or "unknown",
            model_name=model_name,
            device=device,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extra_tag="segmentation-v1",
        )
        if video_id is not None
        else None
    )

    cached_embeddings: list[list[float]] | None = None
    if cache_key is not None:
        cached_embeddings = load_cached_embeddings(cache_dir, cache_key)

    embeddings: list[list[float]] = []
    batch_timings: list[float] = []

    try:
        if cached_embeddings is not None and len(cached_embeddings) == len(texts):
            embeddings = cached_embeddings
            logger.info(
                "Using cached embeddings for segmentation (chunks=%d, model=%s)",
                len(embeddings),
                model_name,
            )
        else:
            try:
                batch_size = int(os.getenv("SEGMENTATION_EMBED_BATCH_SIZE", "32"))
            except Exception:
                batch_size = 32
            batch_size = max(1, batch_size)

            if batch_size <= 1 or len(texts) <= batch_size:
                start_t = time.perf_counter()
                embeddings = provider.embed_documents(texts)
                batch_timings.append(time.perf_counter() - start_t)
            else:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    start_t = time.perf_counter()
                    batch_emb = provider.embed_documents(batch_texts)
                    batch_timings.append(time.perf_counter() - start_t)
                    embeddings.extend(batch_emb)

            if cache_key is not None and embeddings:
                save_cached_embeddings(cache_dir, cache_key, embeddings)

    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to compute embeddings for segmentation: %s", e)
        total = len(transcript)
        step = max(1, floor(total / target_sections))
        result: list[dict[str, Any]] = []
        # Build explicit list of candidate indices and take the first N
        indices = list(range(0, total, step))[:target_sections]
        for i in indices:
            seg = transcript[i]
            result.append({"start": float(seg.get("start", 0.0)), "text": seg.get("text", "")})
        return result

    if batch_timings:
        # Basic batch telemetry
        times = np.asarray(batch_timings, dtype=float)
        avg = float(times.mean())
        p95 = float(np.percentile(times, 95))
        logger.info(
            "Segmentation embeddings computed in %d batch(es); avg=%.3fs, p95=%.3fs",
            len(batch_timings),
            avg,
            p95,
        )
        if metrics_callback is not None:
            metrics_callback(
                {
                    "batch_count": float(len(batch_timings)),
                    "batch_avg_sec": avg,
                    "batch_p95_sec": p95,
                }
            )

    vecs = np.asarray(embeddings, dtype=float)
    sims: list[float] = []
    for i in range(len(vecs) - 1):
        a = vecs[i].reshape(1, -1)
        b = vecs[i + 1].reshape(1, -1)
        sim = float(cosine_similarity(a, b)[0, 0])
        sims.append(sim)

    n_breaks = max(1, target_sections - 1)
    candidate_indices = sorted(range(len(sims)), key=lambda i: sims[i])

    selected_breaks: list[int] = []
    min_distance = max(1, int(len(chunks) / (target_sections * 2)))
    for idx in candidate_indices:
        if not selected_breaks:
            selected_breaks.append(idx)
        else:
            if all(abs(idx - b) >= min_distance for b in selected_breaks):
                selected_breaks.append(idx)
        if len(selected_breaks) >= n_breaks:
            break

    if len(selected_breaks) < n_breaks:
        for idx in candidate_indices:
            if idx not in selected_breaks:
                selected_breaks.append(idx)
            if len(selected_breaks) >= n_breaks:
                break

    selected_breaks = sorted(selected_breaks)

    boundaries = [0]
    for b in selected_breaks:
        boundaries.append(b + 1)
    boundaries.append(len(chunks))

    sections: list[dict[str, Any]] = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        seg_text = "\n".join(c["text"] for c in chunks[start_idx:end_idx])
        seg_start = (
            float(chunks[start_idx]["start"]) if chunks[start_idx].get("start") is not None else 0.0
        )
        sections.append({"start": seg_start, "text": seg_text})

    if len(sections) > target_sections:
        while len(sections) > target_sections:
            a = sections.pop()
            b = sections.pop()
            merged = {"start": b["start"], "text": b["text"] + "\n" + a["text"]}
            sections.append(merged)

    return sections
