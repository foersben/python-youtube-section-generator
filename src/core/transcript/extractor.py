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
    """Protocol describing a transcript-like object returned by some fetch methods.

    Implementations (e.g., items returned by transcript_list.fetch()) typically
    expose attributes `text`, `start`, and `duration`. We only model the
    attributes this module reads.
    """

    text: Any
    start: Any
    duration: Any


def _convert_transcript_to_dict(
    transcript_data: Sequence[dict[str, Any] | TranscriptObject]
) -> list[dict[str, Any]]:
    """Convert youtube-transcript-api result to serializable list of dicts.

    Args:
        transcript_data: Raw transcript data from youtube-transcript-api. Elements
            may be mapping-like dicts or objects exposing attributes `text`,
            `start`, and `duration`.

    Returns:
        List of dictionaries each containing 'text', 'start', and 'duration'.
    """

    segments: list[dict[str, Any]] = []
    for segment in transcript_data:
        # Handle mapping-like segments (preferred) first
        if isinstance(segment, dict):
            text = segment.get("text")
            start = segment.get("start", 0.0)
            duration = segment.get("duration", 0.0)
        else:
            # Treat as an object implementing TranscriptObject protocol
            # Use getattr with defaults to be defensive against odd shapes
            text = getattr(segment, "text", None)
            start = getattr(segment, "start", 0.0)
            duration = getattr(segment, "duration", 0.0)

        # Safe coercions to avoid float(None)
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

        # Helper: attempt to fetch a transcript object in desired languages, using
        # YouTube's native features (including translate()) before falling back
        # to external translation services.
        def _fetch_preferred_transcript(
            desired_langs: list[str],
        ) -> tuple[FetchedTranscript, str | None | Any] | tuple[FetchedTranscript, str | Any]:
            """Return (fetched_segments, language_code) using YT translate when possible.

            Strategy:
            1. Try YouTubeTranscriptApi.fetch with languages preference.
            2. Try transcript_list.find_transcript(desired_langs) and fetch.
            3. If not found, and original is translatable, call original_transcript.translate(target)
               (YouTube's server-side translate feature) and fetch.
            4. If still not available, fall back to returning the original fetched segments.

            Args:
                desired_langs: List of desired language codes in order of preference.

            Returns:
                Tuple of (fetched_segments, language_code).

            Raises:
                Exception: If no transcript could be fetched.
            """

            # 1) Try direct fetch with preferred languages (may return native or translated)
            try:
                fetched_try = ytt_api.fetch(video_id, languages=desired_langs)
                if fetched_try:
                    logger.info("Using YouTubeTranscriptApi.fetch with languages=%s", desired_langs)
                    return fetched_try, (getattr(fetched_try, "language_code", None) or "")
            except Exception:
                pass

            # 2) Try transcript_list.find_transcript which returns a Transcript object
            try:
                t_obj = transcript_list.find_transcript(desired_langs)
                if t_obj:
                    try:
                        fetched = t_obj.fetch()
                        logger.info(
                            "Using transcript_list.find_transcript result for languages=%s",
                            desired_langs,
                        )
                        return fetched, (getattr(t_obj, "language_code", None) or "")
                    except Exception:
                        logger.debug(
                            "Found transcript object but fetch() failed for %s", desired_langs
                        )
            except Exception:
                pass

            # 3) Try translating the original transcript via YouTube's translate() if possible
            try:
                if getattr(original_transcript, "is_translatable", False):
                    # Attempt translation to the first desired language
                    target = desired_langs[0]
                    logger.info(
                        "Attempting server-side translate() of original transcript -> %s", target
                    )
                    try:
                        translated = original_transcript.translate(target)
                        fetched = translated.fetch()
                        logger.info("Used server-side transcript.translate(%s)", target)
                        return fetched, (getattr(translated, "language_code", None) or target)
                    except Exception as exc:
                        logger.warning("Server-side translate() failed: %s", exc)
            except Exception:
                pass

            # 4) Last-resort: return original transcript fetched
            try:
                orig_fetched = original_transcript.fetch()
                logger.info(
                    "Falling back to original transcript language '%s'",
                    original_lang_code or "unknown",
                )
                return orig_fetched, original_lang_code
            except Exception:
                raise

        # Use desired processing language(s). If translate_to provided, accept
        # either a single language string or a list-like comma-separated string.
        if translate_to:
            if isinstance(translate_to, (list, tuple)):
                desired = list(translate_to)
            else:
                # allow comma-separated strings like 'de,en'
                desired = [s.strip() for s in str(translate_to).split(",") if s.strip()]
        else:
            # default: prefer English for processing
            desired = ["en"]
        fetched, fetched_lang = _fetch_preferred_transcript(desired)

        # If no English transcript could be produced by YouTube server-side translate,
        # use DeepL (if available) as a fallback by translating each segment.
        if not fetched or (fetched_lang and fetched_lang.lower().startswith("en") is False):
            # fetched may contain segments but not in English. If it's already English, keep.
            lang_is_en = False
            try:
                if fetched and (
                    (
                        getattr(fetched, "language", "")
                        or getattr(fetched, "language_code", "")
                        or ""
                    )
                    .lower()
                    .startswith("en")
                ):
                    lang_is_en = True
            except Exception:
                lang_is_en = False

            if not lang_is_en:
                # Try DeepL if API key present; else try local Llama translator
                deepl_key = os.getenv("DEEPL_API_KEY")
                translated_segments = None
                # Setup cache directory for translations
                cache_dir = Path(_config.project_root) / ".cache" / "translations"
                cache_dir.mkdir(parents=True, exist_ok=True)

                def _cache_path_for(video_id: str, target: str) -> Path:
                    safe_vid = str(video_id).replace("/", "_")
                    return cache_dir / f"{safe_vid}_{target.lower()}.json"

                if deepl_key:
                    try:
                        from src.core.services.translation import (
                            DeepLAdapter,
                            TranslationQuotaExceeded,
                        )

                        logger.info(
                            "Attempting DeepL fallback translation of transcript to English (batched)"
                        )
                        deepl = DeepLAdapter(deepl_key)
                        # Translate in batch to reduce API calls and avoid quick quota exhaustion
                        original_segments = fetched or original_transcript.fetch()
                        texts = [
                            str(
                                seg.get("text")
                                if isinstance(seg, dict)
                                else getattr(seg, "text", "")
                            )
                            for seg in original_segments
                        ]

                        # Check cache first
                        cache_path = _cache_path_for(video_id, "en")
                        translated_texts = None
                        try:
                            if cache_path.exists():
                                cached = file_io.read_json_file(str(cache_path))
                                # cached format: { 'ts': epoch, 'segments': [ {text, start, duration}, ... ] }
                                if isinstance(cached, dict) and "segments" in cached:
                                    segments = cached.get("segments", [])
                                    if isinstance(segments, list) and len(segments) == len(texts):
                                        # check TTL
                                        ttl = int(
                                            os.getenv("DEEPL_CACHE_TTL_SECONDS", "604800")
                                        )  # default 7 days
                                        ts = int(cached.get("ts", 0))
                                        if ts and (int(time.time()) - ts) <= ttl:
                                            logger.info(
                                                "Using cached translated transcript from %s",
                                                cache_path,
                                            )
                                            translated_texts = [
                                                (
                                                    str(item.get("text", ""))
                                                    if isinstance(item, dict)
                                                    else str(item)
                                                )
                                                for item in segments
                                            ]
                                        else:
                                            logger.debug(
                                                "Translation cache expired or invalid for %s",
                                                cache_path,
                                            )
                        except Exception:
                            logger.debug("Failed to read translation cache %s", cache_path)

                        if translated_texts is None:
                            try:
                                translated_texts = deepl.translate_batch(texts, "EN")
                            except TranslationQuotaExceeded as tqe:
                                logger.error(
                                    "DeepL quota exceeded during batched translate: %s", tqe
                                )
                                translated_texts = None

                        if translated_texts:
                            translated_segments = []
                            for seg, new_text in zip(
                                original_segments, translated_texts, strict=False
                            ):
                                # Coerce start/duration defensively and narrow types for mypy
                                start_val = (
                                    seg.get("start")
                                    if isinstance(seg, dict)
                                    else getattr(seg, "start", 0.0)
                                )
                                if isinstance(start_val, (int, float, str)):
                                    try:
                                        start = float(start_val)
                                    except Exception:
                                        start = 0.0
                                else:
                                    start = 0.0

                                duration_val = (
                                    seg.get("duration")
                                    if isinstance(seg, dict)
                                    else getattr(seg, "duration", 0.0)
                                )
                                if isinstance(duration_val, (int, float, str)):
                                    try:
                                        duration = float(duration_val)
                                    except Exception:
                                        duration = 0.0
                                else:
                                    duration = 0.0

                                translated_segments.append(
                                    {
                                        "text": str(new_text),
                                        "start": start,
                                        "duration": duration,
                                    }
                                )
                            fetched = translated_segments
                            fetched_lang = "en"
                            # persist to cache
                            try:
                                file_io.write_to_file(
                                    {"ts": int(time.time()), "segments": translated_segments},
                                    str(cache_path),
                                )
                            except Exception:
                                logger.debug("Failed to write translation cache %s", cache_path)
                        else:
                            logger.warning("DeepL batch translation was not available or failed.")
                    except Exception as exc:
                        logger.warning("DeepL fallback failed: %s", exc)

                if not fetched or (fetched_lang and not fetched_lang.lower().startswith("en")):
                    # Last fallback: try local LlamaCpp-based translator if configured
                    try:
                        from src.core.services.translation import LlamaCppTranslator

                        logger.info(
                            "Attempting local LlamaCpp fallback translation of transcript to English"
                        )
                        translator = LlamaCppTranslator()
                        original_segments = fetched or original_transcript.fetch()
                        translated_segments = []
                        for seg in original_segments:
                            text = (
                                seg.get("text")
                                if isinstance(seg, dict)
                                else getattr(seg, "text", "")
                            )
                            try:
                                new_text = translator.translate(str(text), "EN")
                            except Exception:
                                new_text = str(text)
                            # Narrow types to satisfy mypy and avoid float(None)
                            s_val = (
                                seg.get("start")
                                if isinstance(seg, dict)
                                else getattr(seg, "start", 0.0)
                            )
                            if isinstance(s_val, (int, float, str)):
                                try:
                                    start = float(s_val)
                                except Exception:
                                    start = 0.0
                            else:
                                start = 0.0

                            d_val = (
                                seg.get("duration")
                                if isinstance(seg, dict)
                                else getattr(seg, "duration", 0.0)
                            )
                            if isinstance(d_val, (int, float, str)):
                                try:
                                    duration = float(d_val)
                                except Exception:
                                    duration = 0.0
                            else:
                                duration = 0.0

                            translated_segments.append(
                                {
                                    "text": str(new_text),
                                    "start": start,
                                    "duration": duration,
                                }
                            )
                        fetched = translated_segments
                        fetched_lang = "en"
                    except Exception as exc:
                        logger.warning("Local Llama translation fallback failed: %s", exc)

        # Convert fetched transcript into serializable segments
        serializable_data = _convert_transcript_to_dict(fetched)

        logger.info("Transcript Details (processing):")
        try:
            logger.info(
                f"- Original Language: {original_transcript.language} ({original_lang_code})"
            )
            # fetched may be a FetchedTranscript or a Transcript-like object
            proc_lang = (
                getattr(fetched, "language", None)
                or getattr(fetched, "language_code", None)
                or "unknown"
            )
            proc_code = (
                getattr(fetched, "language_code", None) or getattr(fetched, "language", None) or ""
            )
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
            serializable_data[0]["original_language_code"] = original_lang_code or getattr(
                fetched, "language_code", ""
            )

        # Apply LLM-based refinement if enabled
        if refine_with_llm is None:
            refine_with_llm = os.getenv("REFINE_TRANSCRIPTS", "true").lower() == "true"

        if refine_with_llm:
            try:
                from src.core.services.transcript_refinement import (
                    TranscriptRefinementService,
                )

                logger.info("Refining transcript with LLM...")
                refinement_service = TranscriptRefinementService()
                serializable_data = refinement_service.refine_transcript_batch(serializable_data)
                logger.info("Transcript refinement complete")
            except Exception as e:
                logger.warning(f"Failed to refine transcript with LLM: {e}. Using raw transcript.")

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
