"""Transcript extraction package.

This package provides utilities for extracting and processing YouTube transcripts.
"""

from __future__ import annotations

from .extractor import extract_transcript, extract_video_id, _convert_transcript_to_dict

__all__ = ["extract_transcript", "extract_video_id", "_convert_transcript_to_dict"]
