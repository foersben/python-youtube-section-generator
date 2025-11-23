"""YouTube transcript extraction utilities.

This package provides utilities for extracting and processing YouTube transcripts.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from youtube_transcript_api import FetchedTranscript, YouTubeTranscriptApi

from src.core.config import config as _config
from src.utils import file_io

logger = logging.getLogger(__name__)


@runtime_checkable
class TranscriptObject(Protocol):
    text: Any
    start: Any
    duration: Any


def _convert_transcript_to_dict(
        transcript_data: Sequence[dict[str, Any] | TranscriptObject]
) -> list[dict[str, Any]]:
    """Convert youtube-transcript-api result to serializable list of dicts."""
    segments: list[dict[str, Any]] = []
    for segment in transcript_data:
        if isinstance(segment, dict):
            text = segment.get("text")
            start = segment.get("start", 0.0)
            duration = segment.get("duration", 0.0)
        else:
            text = getattr(segment, "text", None)
            start = getattr(segment, "start", 0.0)
            duration = getattr(segment, "duration", 0.0)

        text_str = str(text) if text is not None else ""
        try:
            start_f = float(start or 0.0)
        except Exception:
            start_f = 0.0
        try:
            duration_f = float(duration or 0.0)
        except Exception:
            duration_f = 0.0

        segments.append({"text": text_str, "start": start_f, "duration": duration_f})
    return segments


def extract_transcript(
        video_id: str,
        output_file: str | None = None,
        translate_to: str | None = None,
        refine_with_llm: bool | None = None,
) -> list[dict[str, Any]]:
    """Extract a YouTube transcript for processing and track original language.

    Strategy:
    1. Detect TRUE original language (from metadata).
    2. Fetch ENGLISH transcript for processing (native or translated).
    3. Attach original language metadata for final title translation.
    """

    try:
        ytt_api = YouTubeTranscriptApi()

        # 1. List all available transcripts to find the ORIGINAL language
        try:
            transcript_list = ytt_api.list(video_id)
        except Exception as e:
            # If listing fails (e.g. cookies required), we might fallback, but usually fatal
            raise RuntimeError(f"Could not list transcripts for {video_id}: {e}") from e

        # Find the most "original" transcript
        # Priority: Manually created (not generated) > First available
        manually_created = [t for t in transcript_list if not t.is_generated]
        if manually_created:
            original_transcript_obj = manually_created[0]
        else:
            original_transcript_obj = list(transcript_list)[0]

        true_original_lang = original_transcript_obj.language_code
        logger.info(f"Detected Original Video Language: {true_original_lang}")

        # 2. Fetch English for Processing
        # We always want 'en' for the LLM to work best.
        fetched_transcript = None

        # Try fetching 'en' directly (this handles native English OR auto-translate if available)
        try:
            fetched_transcript = transcript_list.find_transcript(['en'])
        except Exception:
            # If 'en' not directly available, try translating the original
            if original_transcript_obj.is_translatable:
                try:
                    fetched_transcript = original_transcript_obj.translate('en')
                except Exception as e:
                    logger.warning(f"YouTube translation to 'en' failed: {e}")

        # 3. Fallback: If we simply can't get English from YouTube, take the original
        # and let the external DeepL/Local fallback handle it later.
        if not fetched_transcript:
            logger.warning("Could not fetch English transcript from YouTube. Using original.")
            fetched_transcript = original_transcript_obj

        fetched_data = fetched_transcript.fetch()

        # --- External Translation Fallback (DeepL / Local) ---
        # If the fetched transcript is NOT English (e.g. YouTube translate failed),
        # we must translate it now so the pipeline receives English.
        current_lang_code = fetched_transcript.language_code
        if not current_lang_code.startswith('en'):
            logger.info(f"Transcript is in '{current_lang_code}', applying external translation to EN...")
            fetched_data = _apply_external_translation(fetched_data, target_lang="en")

        # Convert to dicts
        serializable_data = _convert_transcript_to_dict(fetched_data)

        # 4. Attach Metadata
        # This is crucial: We store the TRUE original language we found in step 1.
        if serializable_data:
            serializable_data[0]["original_language_code"] = true_original_lang
            # Also store the language we are actually processing in
            serializable_data[0]["processing_language_code"] = "en"

        # Optional LLM Refinement
        if refine_with_llm is None:
            refine_with_llm = os.getenv("REFINE_TRANSCRIPTS", "true").lower() == "true"

        if refine_with_llm:
            try:
                from src.core.services.transcript_refinement import TranscriptRefinementService
                logger.info("Refining transcript with LLM...")
                refinement_service = TranscriptRefinementService()
                serializable_data = refinement_service.refine_transcript_batch(serializable_data)
            except Exception as e:
                logger.warning(f"Failed to refine transcript with LLM: {e}")

        if output_file:
            file_io.write_to_file(serializable_data, output_file)
            logger.info(f"Transcript saved to {output_file}")

        return serializable_data

    except Exception as e:
        logger.error(f"Error retrieving transcript: {e}")
        raise


def _apply_external_translation(segments: list[dict] | list[Any], target_lang: str) -> list[dict]:
    """Helper to translate segments using DeepL or Local LLM."""
    # 1. Prepare text list
    texts = []
    for s in segments:
        if isinstance(s, dict): texts.append(s.get("text", ""))
        else: texts.append(getattr(s, "text", ""))

    translated_texts = None

    # 2. Try DeepL
    deepl_key = os.getenv("DEEPL_API_KEY")
    if deepl_key:
        try:
            from src.core.services.translation import DeepLAdapter
            deepl = DeepLAdapter(deepl_key)
            translated_texts = deepl.translate_batch(texts, target_lang.upper())
        except Exception as e:
            logger.warning(f"DeepL translation failed: {e}")

    # 3. Try Local LLM
    if not translated_texts:
        try:
            from src.core.services.translation import LlamaCppTranslator
            translator = LlamaCppTranslator()
            # Batch to avoid overhead
            translated_texts = translator.translate_batch(texts, target_lang.upper())
        except Exception as e:
            logger.warning(f"Local LLM translation failed: {e}")

    # 4. Reconstruct segments
    if translated_texts and len(translated_texts) == len(segments):
        new_segments = []
        for i, orig in enumerate(segments):
            # Normalize origin to dict for easier handling
            if isinstance(orig, dict):
                start = orig.get("start", 0.0)
                dur = orig.get("duration", 0.0)
            else:
                start = getattr(orig, "start", 0.0)
                dur = getattr(orig, "duration", 0.0)

            new_segments.append({
                "text": translated_texts[i],
                "start": start,
                "duration": dur
            })
        return new_segments

    return segments # Return original if failed


def extract_video_id(url: str) -> str:
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
