"""Tests for semantic segmentation helpers."""

from __future__ import annotations

from src.core.retrieval.segmentation import (
    chunk_transcript_text,
    select_coarse_sections,
)


def make_dummy_transcript(n: int = 60) -> list[dict]:
    """Create a dummy transcript with n segments and repetitive content to simulate topic shifts."""
    segments = []
    for i in range(n):
        start = i * 10.0
        # alternate topics every 10 segments
        topic = f"topic_{(i // 10) % 6}"
        text = f"This is {topic} segment number {i}. It contains example text."
        segments.append({"start": start, "text": text})
    return segments


def test_chunk_transcript_text():
    tr = make_dummy_transcript(20)
    chunks = chunk_transcript_text(tr, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 0
    # ensure chunk starts are non-decreasing
    starts = [c["start"] for c in chunks]
    assert all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1))


def test_select_coarse_sections():
    tr = make_dummy_transcript(60)
    sections = select_coarse_sections(tr, target_sections=6, chunk_size=300, chunk_overlap=80)
    assert len(sections) == 6
    # starts should be increasing
    starts = [s["start"] for s in sections]
    assert all(starts[i] < starts[i + 1] for i in range(len(starts) - 1))
