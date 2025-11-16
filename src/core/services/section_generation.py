"""Section generation service (Facade).

Contains the SectionGenerationService class that selects between RAG
and direct LLM strategies and returns typed Section objects.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langdetect import detect as _lang_detect

from src.core.config import config
from src.core.llm import LLMFactory
from src.core.models import Section, SectionGenerationConfig, TranscriptSegment
from src.core.retrieval import RAGSystem
from src.core.services.translation import DeepLAdapter, TranslationProvider
from src.core.services.pipeline import PipelineContext, build_pipeline
from src.core.services.pipeline_stages import FinalTitleTranslationStage

logger = logging.getLogger(__name__)


class SectionGenerationService:
    """Service for generating video sections (Facade pattern).

    Provides simplified interface for section generation, hiding the
    complexity of choosing between RAG and direct LLM approaches.
    """

    def __init__(self) -> None:
        """Initialize section generation service."""
        self.llm_provider = None
        self.rag_system = None

    def generate_sections(
        self,
        transcript: list[dict[str, Any]] | list[TranscriptSegment],
        video_id: str | None = None,
        generation_config: SectionGenerationConfig | None = None,
    ) -> list[Section]:
        """Generate sections from transcript.

        Args:
            transcript: Transcript segments (dicts or TranscriptSegment objects).
            video_id: Video identifier (generated if None).
            generation_config: Generation configuration (uses defaults if None).

        Returns:
            List of Section objects.
        """
        # Normalize input
        transcript_dicts = self._normalize_transcript(transcript)
        vid_id = video_id or self._generate_video_id(transcript_dicts)
        gen_config = generation_config or SectionGenerationConfig()

        # Calculate duration
        total_duration = max(
            seg["start"] + seg.get("duration", 0) for seg in transcript_dicts
        )

        # Optional: translation pipeline (German -> English -> German)
        source_lang = self._detect_language(transcript_dicts)
        use_translation = (os.getenv("USE_TRANSLATION", "true").lower() == "true")
        translator = self._get_translator()
        translated_for_generation: list[dict[str, Any]] | None = None
        if use_translation and translator is not None and source_lang and source_lang.lower() not in ("en", "eng"):
            logger.info("Translating transcript from %s to EN-US for generation...", source_lang)
            translated_for_generation = self._translate_transcript_segments(
                transcript_dicts, translator=translator, target_lang="EN-US", source_lang=source_lang
            )

        working_transcript = translated_for_generation or transcript_dicts

        # Decide strategy: RAG or direct
        use_rag = config.should_use_rag(total_duration)
        logger.info(
            f"Generating sections (duration={total_duration:.0f}s, use_rag={use_rag}, pipeline={config.pipeline_strategy})"
        )

        original_lang_code = self._get_original_lang_code(transcript_dicts, source_lang)

        if use_rag:
            sections_data = self._generate_with_rag(working_transcript, vid_id, gen_config)
            sections = [Section.from_dict(s) for s in sections_data]

            pipeline_context = PipelineContext(transcript=working_transcript, sections=list(sections))
            pipeline_context.metadata["generation_config"] = gen_config
            if original_lang_code:
                pipeline_context.metadata["original_language_code"] = original_lang_code
            if translator is not None:
                pipeline_context.metadata["translation_provider"] = translator

            translation_stage = FinalTitleTranslationStage()
            logger.info("Running pipeline stage: %s", translation_stage.name)
            translation_stage.run(pipeline_context)
            sections = pipeline_context.sections or sections
        else:
            sections_data = self._generate_direct(working_transcript, gen_config)
            sections = [Section.from_dict(s) for s in sections_data]

            pipeline_context = PipelineContext(transcript=working_transcript, sections=list(sections))
            pipeline_context.metadata["generation_config"] = gen_config
            if self.llm_provider is not None:
                pipeline_context.metadata["llm_provider"] = self.llm_provider
            if original_lang_code:
                pipeline_context.metadata["original_language_code"] = original_lang_code
            if translator is not None:
                pipeline_context.metadata["translation_provider"] = translator

            stages = build_pipeline(pipeline_context)
            for stage in stages:
                logger.info("Running pipeline stage: %s", getattr(stage, "name", stage.__class__.__name__))
                stage.run(pipeline_context)

            sections = pipeline_context.sections or pipeline_context.metadata.get("enriched_sections", sections)

        # Apply validation and cleanup
        sections = self._validate_and_clean(list(sections), transcript_dicts)

        logger.info(
            f"\u2705 Generated {len(sections)} sections (RAG={use_rag}, Pipeline={'RAG' if use_rag else config.pipeline_strategy})"
        )
        return sections

    def _generate_with_rag(
        self,
        transcript: list[dict[str, Any]],
        video_id: str,
        config_obj: SectionGenerationConfig,
    ) -> list[dict[str, Any]]:
        """Generate sections using RAG approach."""
        if self.rag_system is None:
            self.rag_system = RAGSystem()
        num_sections = (config_obj.min_sections + config_obj.max_sections) // 2
        sections = self.rag_system.generate_sections(
            transcript=transcript,
            video_id=video_id,
            num_sections=num_sections,
            hierarchical=config_obj.use_hierarchical,
        )
        return sections

    def _generate_direct(
        self,
        transcript: list[dict[str, Any]],
        config_obj: SectionGenerationConfig,
    ) -> list[dict[str, Any]]:
        """Generate sections using direct LLM approach."""
        if self.llm_provider is None:
            self.llm_provider = LLMFactory.create_provider()
        num_sections = (config_obj.min_sections + config_obj.max_sections) // 2
        sections = self.llm_provider.generate_sections(
            transcript=transcript, num_sections=num_sections, max_retries=3
        )
        return sections

    def _normalize_transcript(
        self, transcript: list[dict[str, Any]] | list[TranscriptSegment]
    ) -> list[dict[str, Any]]:
        """Normalize transcript to dict format.

        Accepts lists of dicts, TranscriptSegment objects (with .to_dict()),
        or arbitrary objects with a .to_dict() or __dict__ mapping.
        """
        if not transcript:
            raise ValueError("Transcript cannot be empty")

        normalized: list[dict[str, Any]] = []
        for seg in transcript:
            if isinstance(seg, dict):
                normalized.append(seg)
                continue
            # Domain object with to_dict
            to_dict = getattr(seg, "to_dict", None)
            if callable(to_dict):
                try:
                    normalized.append(to_dict())
                    continue
                except Exception:
                    # fallthrough to other attempts
                    pass
            # Fallback: dataclass or object with __dict__
            if hasattr(seg, "__dict__"):
                try:
                    normalized.append({k: v for k, v in vars(seg).items()})
                    continue
                except Exception:
                    pass

            raise ValueError("Transcript items must be dict-like or provide to_dict()")

        return normalized

    def _generate_video_id(self, transcript: list[dict[str, Any]]) -> str:
        """Generate video ID from transcript hash."""
        import hashlib

        text = "".join(seg["text"] for seg in transcript[:100])
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _validate_and_clean(
        self, sections: list[Section], transcript: list[dict[str, Any]]
    ) -> list[Section]:
        """Validate and clean sections."""
        max_time = max(seg["start"] + seg.get("duration", 0) for seg in transcript)
        valid_sections = []
        for section in sections:
            # Validate timestamp
            if 0 <= section.start <= max_time:
                # Clean title
                section.title = self._clean_title(section.title)
                if len(section.title) >= 3:
                    valid_sections.append(section)
        # Sort by timestamp
        valid_sections.sort(key=lambda s: s.start)
        return valid_sections

    def _clean_title(self, title: str) -> str:
        """Clean section title."""
        import re
        # Remove prefixes
        prefixes = ["response:", "answer:", "title:", "section:", "===" , "_____", "---"]
        title_lower = title.lower()
        for prefix in prefixes:
            if title_lower.startswith(prefix):
                title = title[len(prefix):].strip()
                title_lower = title.lower()
        # Remove special chars
        title = title.strip('"\'`.,;:!?-–—')
        while title and title[0] in '_=-~*#@[](){}':
            title = title[1:].strip()
        # Remove patterns
        title = re.sub(r'\bsection at \d+s?\b', '', title, flags=re.IGNORECASE).strip()
        # Capitalize
        if title and not title[0].isupper():
            title = title[0].upper() + title[1:]
        return title

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.rag_system:
            self.rag_system.cleanup()

    # ---------------- Translation helpers ----------------
    def _get_translator(self) -> TranslationProvider | None:
        api_key = os.getenv("DEEPL_API_KEY")

        # Try DeepL first (preferred)
        if api_key:
            try:
                translator = DeepLAdapter(api_key)
                logger.info("Using DeepL for translation")
                return translator
            except Exception as e:
                logger.error("Failed to initialize DeepL: %s; falling back to local LLM translator", e, exc_info=True)
        else:
            logger.info("DEEPL_API_KEY not set; using local LLM translator as fallback")

        # Fallback to local LlamaCpp translator
        try:
            from src.core.services.translation import LlamaCppTranslator
            translator = LlamaCppTranslator()
            logger.info("Using local LlamaCpp translator (fallback mode)")
            return translator
        except Exception as e:
            logger.error(
                "❌ CRITICAL: Failed to initialize local translator: %s\n"
                "This will cause poor title quality for non-English transcripts!\n"
                "Fix: Ensure DEEPL_API_KEY is set or LOCAL_MODEL_PATH points to a valid model.",
                e,
                exc_info=True
            )
            return None

    def _detect_language(self, transcript: list[dict[str, Any]]) -> str | None:
        try:
            sample = " ".join(seg.get("text", "") for seg in transcript[:50])
            if not sample.strip():
                return None
            lang = _lang_detect(sample)
            return lang
        except Exception:
            return None

    def _translate_transcript_segments(
        self,
        transcript: list[dict[str, Any]],
        translator: TranslationProvider,
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[dict[str, Any]]:
        """Translate transcript segments efficiently using batching.

        Instead of translating each segment individually (1000+ API calls),
        we batch segments into larger chunks (10-50 API calls) while preserving
        timestamps and context.
        """
        if not transcript:
            return []

        # Batch segments into chunks (max 5000 chars each for DeepL free tier)
        MAX_BATCH_SIZE = 4500  # Leave buffer for special chars
        batches: list[tuple[str, list[int]]] = []  # (combined_text, segment_indices)

        current_batch = []
        current_indices = []
        current_size = 0

        for idx, seg in enumerate(transcript):
            text = seg.get("text", "").strip()
            if not text:
                continue

            # Add segment marker for splitting later (use unique delimiter)
            # Format: [SEG_123]actual text content
            marked_text = f"[SEG_{idx}]{text}"
            text_size = len(marked_text) + 2  # +2 for newline

            # Start new batch if adding this would exceed limit
            if current_size + text_size > MAX_BATCH_SIZE and current_batch:
                batches.append(("\n".join(current_batch), current_indices))
                current_batch = []
                current_indices = []
                current_size = 0

            current_batch.append(marked_text)
            current_indices.append(idx)
            current_size += text_size

        # Add final batch
        if current_batch:
            batches.append(("\n".join(current_batch), current_indices))

        logger.info(
            "Batched %d segments into %d chunks (avg %.0f segments/chunk)",
            len(transcript),
            len(batches),
            len(transcript) / max(len(batches), 1)
        )

        # Translate batches
        translated_map: dict[int, str] = {}
        for batch_idx, (batch_text, indices) in enumerate(batches):
            try:
                logger.info("Translating batch %d/%d (%d segments)", batch_idx + 1, len(batches), len(indices))
                translated_batch = translator.translate(
                    batch_text,
                    target_lang=target_lang,
                    source_lang=source_lang
                )

                # Split translated batch back into segments using markers
                import re
                # Extract segments: [SEG_123]translated text
                pattern = r'\[SEG_(\d+)\]([^\[]*?)(?=\[SEG_|\Z)'
                matches = re.findall(pattern, translated_batch, re.DOTALL)

                for seg_id_str, translated_text in matches:
                    seg_id = int(seg_id_str)
                    translated_map[seg_id] = translated_text.strip()

            except Exception as e:
                logger.warning("Batch translation failed for batch %d: %s; keeping originals", batch_idx + 1, e)
                # Fallback: keep original text for this batch
                for idx in indices:
                    if idx not in translated_map:
                        translated_map[idx] = transcript[idx].get("text", "")

        # Reconstruct transcript with translated texts
        result: list[dict[str, Any]] = []
        for idx, seg in enumerate(transcript):
            new_seg = dict(seg)
            new_seg["text"] = translated_map.get(idx, seg.get("text", ""))
            result.append(new_seg)

        logger.info("✅ Translation complete: %d segments translated in %d batches", len(result), len(batches))
        return result

    def _get_original_lang_code(
        self,
        transcript_dicts: list[dict[str, Any]],
        source_lang: str | None,
    ) -> str | None:
        """Helper to extract original language code for later translation."""
        if transcript_dicts and transcript_dicts[0].get("original_language_code"):
            return transcript_dicts[0].get("original_language_code")
        return source_lang

__all__ = ["SectionGenerationService"]
