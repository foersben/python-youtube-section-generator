"""YouTube transcript extraction utilities.

This package provides utilities for extracting and processing YouTube transcripts.
"""

from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

from src.utils import file_io

logger = logging.getLogger(__name__)


def _convert_transcript_to_dict(transcript_data) -> list[dict[str, Any]]:
    """Convert youtube-transcript-api result to serializable list of dicts.

    Args:
        transcript_data: Raw transcript data from youtube-transcript-api.

    Returns:
        List of dictionaries each containing 'text', 'start', and 'duration'.
    """
    segments = []
    for segment in transcript_data:
        try:
            text = segment.text
            start = segment.start
            duration = getattr(segment, "duration", 0.0)
        except Exception:
            # Fallback if segment is a dict
            text = segment.get("text")
            start = segment.get("start")
            duration = segment.get("duration", 0.0)

        segments.append({"text": str(text), "start": float(start), "duration": float(duration)})
    return segments


def extract_transcript(video_id: str, output_file: str | None = None, translate_to: str | None = None) -> list[dict[str, Any]]:
    """Extract a YouTube transcript and optionally save or translate it.

    Args:
        video_id: YouTube video ID (11-character ID or URL accepted by the library).
        output_file: Optional path to save the transcript as JSON.
        translate_to: Optional language code to translate to.

    Returns:
        List of transcript segments as dictionaries.

    Raises:
        Exception: If transcript extraction fails.
    """
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        transcripts = list(transcript_list)

        transcript = next((s for s in transcripts if not s.is_generated), transcripts[0])

        if translate_to:
            transcript_data = transcript.translate(translate_to).fetch()
        else:
            transcript_data = transcript.fetch()

        serializable_data = _convert_transcript_to_dict(transcript_data)

        logger.info("Transcript Details:")
        try:
            logger.info(f"- Video ID: {transcript.video_id}")
            logger.info(f"- Language: {transcript.language} ({transcript.language_code})")
            logger.info(f"- Generated: {'Yes' if transcript.is_generated else 'No'}")
        except Exception:
            pass

        if output_file:
            file_io.write_to_file(serializable_data, output_file)
            logger.info(f"Transcript saved to {output_file}")

        return serializable_data

    except Exception as e:
        logger.error(f"Error retrieving transcript: {e}")
        raise


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL or return ID if already provided.

    Args:
        url: YouTube URL or video ID.

    Returns:
        11-character YouTube video ID.

    Raises:
        ValueError: If no valid video ID found.
    """
    import re

    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL or ID")


__all__ = ["extract_transcript", "extract_video_id", "_convert_transcript_to_dict"]
