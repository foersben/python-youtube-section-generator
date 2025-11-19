"""Formatting utilities for sections and transcripts.

Provides a minimal, well-documented implementation used by the web
application and tests. Kept intentionally small and dependency-free so it
can be safely maintained here.
"""

from __future__ import annotations

from typing import Any


def format_sections_for_youtube(sections: list[dict[str, Any]]) -> str:
    """Format a list of sections for YouTube timestamps text output.

    Args:
        sections: List of section dicts. Each dict should contain at least
            the keys 'start' (float seconds) and 'title' (str).

    Returns:
        A multi-line string where each line is of the form "{start}s - {title}".

    Raises:
        ValueError: When a section lacks required keys.
    """
    lines: list[str] = []
    for s in sections:
        if not isinstance(s, dict):
            raise ValueError("Section entry must be a dict")
        if "start" not in s or "title" not in s:
            raise ValueError("Section dict must contain 'start' and 'title'")
        try:
            start = float(s["start"])
        except Exception:
            # Use `from None` to avoid masking unrelated exceptions as the cause
            raise ValueError("Section 'start' must be numeric") from None
        title = str(s["title"]).strip()
        lines.append(f"{start:.1f}s - {title}")
    return "\n".join(lines)


def format_transcript_for_display(transcript: list[dict[str, Any]]) -> str:
    """Formats transcript data for human-readable display.

    Args:
        transcript: List of transcript segments as dictionaries. Each dictionary
            is expected to contain keys 'start' and 'text'.

    Returns:
        A formatted string with one transcript segment per line.
    """
    parts: list[str] = []
    for seg in transcript:
        if not isinstance(seg, dict):
            continue
        start = seg.get("start")
        text = seg.get("text", "")
        try:
            start_f = float(start) if start is not None else 0.0
        except Exception:
            start_f = 0.0
        parts.append(f"[{start_f:.1f}s] {text}")
    return "\n".join(parts)
