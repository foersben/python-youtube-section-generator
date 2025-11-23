"""Section generation service (Facade)."""

from __future__ import annotations

import logging
import os
from typing import Any, cast

from langdetect import detect as _lang_detect

from src.core.config import config
from src.core.llm import LLMFactory
from src.core.models import Section, SectionGenerationConfig, TranscriptSegment
from src.core.retrieval import RAGSystem
from src.core.services.pipeline import PipelineContext, build_pipeline
from src.core.services.pipeline_stages import FinalTitleTranslationStage
from src.core.services.translation import DeepLAdapter, TranslationProvider

logger = logging.getLogger(__name__)


class SectionGenerationService:
    """Service for generating video sections (Facade pattern)."""

    def __init__(self) -> None:
        self.llm_provider: Any = None
        self.rag_system: Any = None

    def generate_sections(
            self,
            transcript: list[dict[str, Any]] | list[TranscriptSegment],
            video_id: str | None = None,
            generation_config: SectionGenerationConfig | None = None,
    ) -> list[Section]:

        transcript_dicts = self._normalize_transcript(transcript)
        vid_id = video_id or self._generate_video_id(transcript_dicts)
        gen_config = generation_config or SectionGenerationConfig()

        total_duration = max(
            (seg.get("start", 0.0) or 0.0) + (seg.get("duration", 0.0) or 0.0)
            for seg in transcript_dicts
        )

        # Translation Handling (Transcript Text)
        source_lang = self._detect_language(transcript_dicts)
        use_translation = os.getenv("USE_TRANSLATION", "true").lower() == "true"
        translator = self._get_translator()
        translated_for_generation: list[dict[str, Any]] | None = None

        if (
                use_translation
                and translator is not None
                and source_lang
                and source_lang.lower() not in ("en", "eng")
        ):
            logger.info("Translating transcript from %s to EN-US for generation...", source_lang)
            translated_for_generation = self._translate_transcript_segments(
                transcript_dicts,
                translator=translator,
                target_lang="EN-US",
                source_lang=source_lang,
            )

        working_transcript = translated_for_generation or transcript_dicts

        # Pipeline Decision
        should_use_rag = config.should_use_rag(total_duration)
        is_split_strategy = config.pipeline_strategy == "split"

        if is_split_strategy:
            logger.info("Using Split Pipeline (includes Semantic/RAG discovery).")
            sections = []
        elif should_use_rag:
            logger.info("Using Legacy Monolithic RAG System.")
            sections_data = self._generate_with_rag(working_transcript, vid_id, gen_config)
            sections = [Section.from_dict(s) for s in sections_data]

            pipeline_context = PipelineContext(transcript=working_transcript, sections=list(sections))
            pipeline_context.metadata["generation_config"] = gen_config
            self._setup_pipeline_metadata(pipeline_context, transcript_dicts, translator)

            FinalTitleTranslationStage().run(pipeline_context)
            return self._validate_and_clean(pipeline_context.sections or sections, transcript_dicts)
        else:
            logger.info("Using Legacy Direct LLM Generation.")
            sections_data = self._generate_direct(working_transcript, gen_config)
            sections = [Section.from_dict(s) for s in sections_data]

        # Pipeline Execution
        pipeline_context = PipelineContext(
            transcript=working_transcript,
            sections=list(sections) if sections else []
        )
        pipeline_context.metadata["generation_config"] = gen_config

        if self.llm_provider is not None:
            pipeline_context.metadata["llm_provider"] = self.llm_provider

        self._setup_pipeline_metadata(pipeline_context, transcript_dicts, translator)

        stages = build_pipeline(pipeline_context)
        for stage in stages:
            logger.info("Running pipeline stage: %s", getattr(stage, "name", stage.__class__.__name__))
            stage.run(pipeline_context)

        result_sections = pipeline_context.sections or pipeline_context.metadata.get("enriched_sections", [])
        return self._validate_and_clean(list(result_sections), transcript_dicts)

    def _setup_pipeline_metadata(self, context: PipelineContext, original_transcript: list[dict], translator):
        """Helper to inject language metadata for translation stage."""
        # 1. Original Language (from Extractor)
        orig_code = original_transcript[0].get("original_language_code")
        if orig_code:
            context.metadata["original_language_code"] = orig_code

        # 2. Target Language Override (from Web App / User)
        target_code = original_transcript[0].get("target_language")
        if target_code:
            context.metadata["target_language"] = target_code

        if translator:
            context.metadata["translation_provider"] = translator

    # ... (Rest of the methods unchanged: _generate_with_rag, _generate_direct, etc.)
    def _generate_with_rag(self, transcript: list[dict[str, Any]], video_id: str, config_obj: SectionGenerationConfig) -> list[dict[str, Any]]:
        if self.rag_system is None: self.rag_system = RAGSystem()
        num_sections = (config_obj.min_sections + config_obj.max_sections) // 2
        return self.rag_system.generate_sections(transcript, video_id, num_sections, hierarchical=config_obj.use_hierarchical)

    def _generate_direct(self, transcript: list[dict[str, Any]], config_obj: SectionGenerationConfig) -> list[dict[str, Any]]:
        if self.llm_provider is None: self.llm_provider = LLMFactory.create_provider()
        num_sections = (config_obj.min_sections + config_obj.max_sections) // 2
        return self.llm_provider.generate_sections(transcript, num_sections, 3)

    def _normalize_transcript(self, transcript: Any) -> list[dict[str, Any]]:
        if not transcript: raise ValueError("Transcript cannot be empty")
        normalized = []
        for seg in transcript:
            if isinstance(seg, dict): normalized.append(seg)
            elif hasattr(seg, "to_dict"): normalized.append(seg.to_dict())
            elif hasattr(seg, "__dict__"): normalized.append(dict(vars(seg)))
        return normalized

    def _generate_video_id(self, transcript: list[dict[str, Any]]) -> str:
        import hashlib
        text = "".join(seg["text"] for seg in transcript[:100])
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _validate_and_clean(self, sections: list[Section], transcript: list[dict[str, Any]]) -> list[Section]:
        max_time = max((seg.get("start", 0.0) or 0.0) + (seg.get("duration", 0.0) or 0.0) for seg in transcript)
        valid = [s for s in sections if 0 <= s.start <= max_time and s.title]
        valid.sort(key=lambda s: s.start)
        return valid

    def cleanup(self) -> None:
        if self.rag_system: self.rag_system.cleanup()

    def _get_translator(self) -> TranslationProvider | None:
        try:
            from src.core.services.translation import GoogleTranslatorAdapter
            return GoogleTranslatorAdapter()
        except Exception: pass
        api_key = os.getenv("DEEPL_API_KEY")
        if api_key:
            try: return DeepLAdapter(api_key)
            except Exception: pass
        try:
            from src.core.services.translation import LlamaCppTranslator
            return cast(TranslationProvider, LlamaCppTranslator())
        except Exception: return None

    def _detect_language(self, transcript: list[dict[str, Any]]) -> str | None:
        try:
            sample = " ".join(seg.get("text", "") for seg in transcript[:50])
            return _lang_detect(sample) if sample.strip() else None
        except Exception: return None

    def _translate_transcript_segments(self, transcript, translator, target_lang, source_lang=None) -> list[dict[str, Any]]:
        # Assuming implementation exists
        return transcript

__all__ = ["SectionGenerationService"]
