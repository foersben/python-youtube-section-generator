"""YouTube transcript extraction utilities.

This package provides utilities for extracting and processing YouTube transcripts.
"""

from __future__ import annotations

import logging
import os
from typing import Any

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


def extract_transcript(
    video_id: str,
    output_file: str | None = None,
    translate_to: str | None = None,
    refine_with_llm: bool = None,
) -> list[dict[str, Any]]:
    """Extract a YouTube transcript for processing and track original language.

    The pipeline always prefers an English transcript for processing when
    available (or falls back to the original-language transcript if English
    cannot be fetched or translated). The original language of the video is
    recorded in the first transcript segment under ``original_language_code``
    so that section titles can later be translated back to the original
    language at the end of the processing pipeline.

    Args:
        video_id: YouTube video ID (11-character ID or URL accepted by the
            library).
        output_file: Optional path to save the transcript as JSON.
        translate_to: Deprecated for transcript content.
        refine_with_llm: Whether to use LLM for intelligent transcript
            refinement. If None, uses REFINE_TRANSCRIPTS env var (default:
            True).

    Returns:
        List of transcript segments as dictionaries.

    Raises:
        Exception: If transcript extraction fails.
    """
    try:
        ytt_api = YouTubeTranscriptApi()

        # Retrieve available transcripts to inspect languages
        transcript_list = ytt_api.list(video_id)
        transcripts = list(transcript_list)
        if not transcripts:
            raise RuntimeError("No transcripts available for video")

        # Prefer manually created transcript when available (original)
        original_transcript = next((s for s in transcripts if not s.is_generated), transcripts[0])
        original_lang_code = getattr(original_transcript, "language_code", "") or ""

        # 1) Try to fetch an English transcript directly (preferred for processing)
        fetched = None
        try:
            try:
                fetched_en = ytt_api.fetch(video_id, languages=["en"])
                fetched = fetched_en
                logger.info("Using fetched English transcript for processing")
            except Exception:
                fetched = None

            # 2) If fetch didn't return English, try to find or translate
            if fetched is None:
                processing_transcript = None
                try:
                    processing_transcript = transcript_list.find_transcript(["en"])
                    logger.info("Using native English transcript for processing")
                    fetched = processing_transcript.fetch()
                except Exception:
                    processing_transcript = None

                # 3) If no English transcript found, try translating the original transcript to English
                if fetched is None:
                    if getattr(original_transcript, "is_translatable", False):
                        try:
                            logger.info("Translating original transcript %s -> en for processing", original_lang_code or "unknown")
                            translated_transcript = original_transcript.translate("en")
                            fetched = translated_transcript.fetch()
                        except Exception as exc:
                            logger.warning("Failed to translate transcript to English: %s", exc)
                            fetched = None

                # 4) If still no fetched transcript, fall back to the original transcript
                if fetched is None:
                    logger.warning("Falling back to original transcript language '%s' for processing", original_lang_code or "unknown")
                    fetched = original_transcript.fetch()

        except Exception as e:
            logger.error("Unexpected error while selecting transcript: %s", e, exc_info=True)
            # Last resort: use original transcript
            try:
                fetched = original_transcript.fetch()
            except Exception:
                raise

        # Convert fetched transcript into serializable segments
        serializable_data = _convert_transcript_to_dict(fetched)

        logger.info("Transcript Details (processing):")
        try:
            logger.info(f"- Original Language: {original_transcript.language} ({original_lang_code})")
            # fetched may be a FetchedTranscript or a Transcript-like object
            proc_lang = getattr(fetched, "language", None) or getattr(fetched, "language_code", None) or "unknown"
            proc_code = getattr(fetched, "language_code", None) or getattr(fetched, "language", None) or ""
            logger.info(f"- Processing Language: {proc_lang} ({proc_code})")
            # For fetched objects, check is_generated flag when available
            is_gen = getattr(fetched, "is_generated", None)
            if is_gen is not None:
                logger.info(f"- Generated: {'Yes' if is_gen else 'No'}")
        except Exception:
            pass

        # Attach original language metadata to the first segment for downstream
        # use so that later stages (e.g., section title translation) can
        # translate English titles back to the original language if desired.
        if serializable_data:
            serializable_data[0]["original_language_code"] = original_lang_code or getattr(fetched, "language_code", "")

        # Apply LLM-based refinement if enabled
        if refine_with_llm is None:
            refine_with_llm = os.getenv("REFINE_TRANSCRIPTS", "true").lower() == "true"

        if refine_with_llm:
            try:
                from src.core.services.transcript_refinement import TranscriptRefinementService

                logger.info("Refining transcript with LLM...")
                refinement_service = TranscriptRefinementService()
                serializable_data = refinement_service.refine_transcript_batch(
                    serializable_data
                )
                logger.info("Transcript refinement complete")
            except Exception as e:
                logger.warning(
                    f"Failed to refine transcript with LLM: {e}. Using raw transcript."
                )

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
