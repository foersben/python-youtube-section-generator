"""Formatting utilities for sections and transcripts."""

from __future__ import annotations
from typing import Any

def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or H:MM:SS format."""
    total_seconds = int(round(seconds))
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def format_sections_for_youtube(sections: list[dict[str, Any]]) -> str:
    """Format a list of sections for YouTube timestamps text output."""
    lines: list[str] = []
    for s in sections:
        if not isinstance(s, dict):
            continue
        start = s.get("start", 0.0)
        title = str(s.get("title", "")).strip()

        try:
            timestamp = _seconds_to_timestamp(float(start))
        except Exception:
            timestamp = "00:00"

        lines.append(f"{timestamp} - {title}")
    return "\n".join(lines)

def format_transcript_for_display(transcript: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for seg in transcript:
        start = seg.get("start", 0.0)
        text = seg.get("text", "")
        parts.append(f"[{float(start):.1f}s] {text}")
    return "\n".join(parts)
