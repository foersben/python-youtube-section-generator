"""Semantic segmentation helpers for transcript processing.

This module provides small utilities to chunk a transcript into overlapping
text windows, compute embeddings for each chunk, and select coarse section
boundaries based on adjacent-chunk semantic similarity.
"""

from __future__ import annotations

import logging
from math import floor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    offsets: list[float] = []
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
        for off, ts in reversed(offsets):
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
        return [ {"start": c["start"], "text": c["text"]} for c in chunks ]

    # Prepare texts
    texts = [c["text"] for c in chunks]

    # Embeddings
    provider = EmbeddingsFactory.create_provider(model_name=embedding_model or "sentence-transformers/all-MiniLM-L6-v2", device=device)
    try:
        embeddings = provider.embed_documents(texts)
    except Exception as e:
        logger.exception("Failed to compute embeddings for segmentation: %s", e)
        # As a robust fallback, evenly space sections across transcript
        total = len(transcript)
        step = max(1, floor(total / target_sections))
        result: list[dict[str, Any]] = []
        for i in range(0, total, step)[:target_sections]:
            seg = transcript[i]
            result.append({"start": float(seg.get("start", 0.0)), "text": seg.get("text", "")})
        return result

    # Convert to numpy array
    vecs = np.asarray(embeddings, dtype=float)
    # compute adjacent similarities
    sims: list[float] = []
    for i in range(len(vecs) - 1):
        a = vecs[i].reshape(1, -1)
        b = vecs[i + 1].reshape(1, -1)
        sim = float(cosine_similarity(a, b)[0, 0])
        sims.append(sim)

    # pick boundaries with lowest similarity (where topic shift is strongest)
    n_breaks = max(1, target_sections - 1)
    # indices in range(len(chunks)-1)
    candidate_indices = sorted(range(len(sims)), key=lambda i: sims[i])

    # ensure we pick well-spread indices: greedily select lowest-sim indices while keeping distance
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

    # if not enough breaks selected due to spacing, fill from remaining candidates
    if len(selected_breaks) < n_breaks:
        for idx in candidate_indices:
            if idx not in selected_breaks:
                selected_breaks.append(idx)
            if len(selected_breaks) >= n_breaks:
                break

    selected_breaks = sorted(selected_breaks)

    # Build sections by splitting chunks at the selected break points
    boundaries = [0]
    for b in selected_breaks:
        boundaries.append(b + 1)
    boundaries.append(len(chunks))

    sections: list[dict[str, Any]] = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        # aggregate text and choose earliest start timestamp
        seg_text = "\n".join(c["text"] for c in chunks[start_idx:end_idx])
        seg_start = float(chunks[start_idx]["start"]) if chunks[start_idx].get("start") is not None else 0.0
        sections.append({"start": seg_start, "text": seg_text})

    # If we produced more sections than desired (rare), merge/slice to target_sections
    if len(sections) > target_sections:
        # merge the last ones
        while len(sections) > target_sections:
            # merge last two
            a = sections.pop()
            b = sections.pop()
            merged = {"start": b["start"], "text": b["text"] + "\n" + a["text"]}
            sections.append(merged)

    return sections

