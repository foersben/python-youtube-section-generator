"""Pipeline stage implementations for section generation workflows.

This module now contains two families of stages:
- Legacy single-prompt style: DetermineSectionsStage, EnrichSectionsStage,
  FinalizeTitlesStage (kept for backward compatibility and tests).
- New multi-step pipeline: LooseSectionDiscoveryStage, IntervalParsingStage,
  BatchTitleDraftStage, TitleDraftStructStage, TitleOptimizationStage,
  OptimizedTitleParsingStage, FinalFormattingStage.
"""

from __future__ import annotations

import json
import logging
import re
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

from src.core.config import config
from src.core.llm import LLMFactory
from src.core.models import Section, SectionGenerationConfig

if TYPE_CHECKING:
    from src.core.services.pipeline import PipelineContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy stages (used by tests and "legacy" strategy)
# ---------------------------------------------------------------------------

@dataclass
class DetermineSectionsStage:
    """Legacy stage: direct section generation via provider.generate_sections."""

    section_multiplier: float = 1.0
    name: str = "determine_sections"

    def run(self, context: "PipelineContext") -> None:
        if context.sections and not context.metadata.get("force_section_refresh", False):
            logger.debug("Using precomputed sections; determination skipped")
            context.metadata.setdefault("raw_sections", list(context.sections))
            return

        if context.metadata.get("raw_sections"):
            logger.debug("Raw sections already present; skipping determination")
            context.sections = list(context.metadata["raw_sections"])
            return

        generation_config = context.metadata.get("generation_config")
        if not isinstance(generation_config, SectionGenerationConfig):
            generation_config = SectionGenerationConfig(
                min_sections=config.default_min_sections,
                max_sections=config.default_max_sections,
            )

        target = max(2, int(generation_config.max_sections * self.section_multiplier))

        provider = _ensure_provider(context)

        logger.info("Determining ~%d rough sections (legacy)", target)
        sections_raw = provider.generate_sections(
            transcript=context.transcript,
            num_sections=target,
            max_retries=3,
        )
        normalized = _normalize_sections(sections_raw)
        context.metadata["raw_sections"] = normalized
        context.sections = list(normalized)


class EnrichSectionsStage:
    """Legacy stage: enrich sections with titles/descriptions in one shot."""

    name = "enrich_sections"

    def run(self, context: "PipelineContext") -> None:
        sections = _normalize_sections(context.metadata.get("raw_sections") or context.sections)
        if not sections:
            logger.warning("Enrichment skipped because no sections present")
            return

        provider = _ensure_provider(context)

        enriched: list[Section] = []
        for section in sections:
            prompt = (
                "Provide a descriptive title and optional bullet summary for the "
                "following section timestamp:"
                f"\nStart: {section.start:.1f}s"
            )
            try:
                response = provider.generate_text(prompt, max_tokens=128)
                parsed = _parse_title_payload(response)
                section.title = parsed.get("title", section.title or "Section")
            except Exception as exc:
                logger.warning("Failed to enrich section at %.1fs: %s", section.start, exc)
            enriched.append(section)

        context.metadata["enriched_sections"] = enriched
        context.sections = list(enriched)


class FinalizeTitlesStage:
    """Legacy stage: enforce fuzzy title length on existing titles."""

    name = "finalize_titles"

    def run(self, context: "PipelineContext") -> None:
        sections = _normalize_sections(
            context.metadata.get("enriched_sections")
            or context.metadata.get("raw_sections")
            or context.sections
        )
        if not sections:
            logger.warning("Finalize stage skipped because no sections available")
            return

        config_obj = context.metadata.get("generation_config") or SectionGenerationConfig()
        min_words, max_words = config_obj.title_length_range
        provider = _ensure_provider(context)

        final_sections: list[Section] = []
        for section in sections:
            prompt = (
                "Rewrite the section title to stay within a word range."
                f"\nCurrent title: {section.title}\nWord range: {min_words}-{max_words}."
                "Return JSON {\"title\":\"New Title\"}."
            )
            try:
                response = provider.generate_text(prompt, max_tokens=64)
                parsed = _parse_title_payload(response)
                new_title = parsed.get("title")
                if new_title:
                    section.title = new_title.strip()
            except Exception as exc:
                logger.warning("Failed to finalize title '%s': %s", section.title, exc)
            final_sections.append(section)

        context.sections = final_sections


# ---------------------------------------------------------------------------
# New multi-step pipeline stages (split strategy)
# ---------------------------------------------------------------------------

@dataclass
class LooseSectionDiscoveryStage:
    name: str = "discover_sections"

    def run(self, context: "PipelineContext") -> None:
        provider = _ensure_provider(context)
        config_obj = _ensure_generation_config(context)
        min_count = max(2, config_obj.min_sections)
        max_count = max(min_count, config_obj.max_sections)
        excerpt = _summarize_transcript(context.transcript)
        prompt = (
            f"{excerpt}\n\n"
            f"Select {min_count}-{max_count} isolated topic ranges."
            "Write each line as 'start-end note' in seconds."
        )
        response = provider.generate_text(prompt, max_tokens=384)
        context.metadata["section_notes_text"] = response


class IntervalParsingStage:
    name = "parse_intervals"

    def run(self, context: "PipelineContext") -> None:
        raw_text = context.metadata.get("section_notes_text", "")
        if not raw_text:
            logger.warning("No section notes text available for parsing")
            context.metadata["intervals"] = []
            return
        total_duration = _total_duration(context.transcript)
        intervals = _extract_intervals_from_text(raw_text, total_duration)
        context.metadata["intervals"] = intervals
        if not intervals:
            logger.warning("Failed to parse intervals; downstream stages may fallback")


@dataclass
class BatchTitleDraftStage:
    batch_size: int = 4
    name: str = "draft_titles"

    def run(self, context: "PipelineContext") -> None:
        intervals = context.metadata.get("intervals", [])
        if not intervals:
            logger.warning("No intervals to title")
            context.metadata["title_draft_blocks"] = []
            return
        provider = _ensure_provider(context)
        drafts: list[str] = []
        for chunk in _chunk(intervals, self.batch_size):
            ranges_text = "\n".join(f"{entry['start']}-{entry['end']}" for entry in chunk)
            snippets = _context_snippets(context.transcript, chunk)
            prompt = (
                "For each range write one line 'start-end title sentence'.\n"
                "Titles may be any length. Return plain text lines only.\n"
                f"Ranges:\n{ranges_text}\n\nContext:\n{snippets}"
            )
            drafts.append(provider.generate_text(prompt, max_tokens=400))
        context.metadata["title_draft_blocks"] = drafts


class TitleDraftStructStage:
    name = "structure_title_drafts"

    def run(self, context: "PipelineContext") -> None:
        drafts = context.metadata.get("title_draft_blocks", [])
        if not drafts:
            logger.warning("No draft blocks to structure")
            context.metadata["title_candidates"] = []
            return
        provider = _ensure_provider(context)
        structured: list[dict[str, Any]] = []
        raw_responses: list[str] = []
        for draft in drafts:
            prompt = (
                "Convert the lines below into JSON list [{\"start\":s,\"end\":e,\"title\":\"...\"}].\n"
                "Return JSON only.\n"
                f"Lines:\n{draft}"
            )
            response = provider.generate_text(prompt, max_tokens=400)
            raw_responses.append(response)
            payload = _safe_json_load(response)
            if isinstance(payload, dict) and "sections" in payload:
                payload = payload["sections"]
            if isinstance(payload, list):
                for entry in payload:
                    try:
                        structured.append(
                            {
                                "start": float(entry.get("start", 0.0)),
                                "end": float(entry.get("end", entry.get("start", 0.0))),
                                "title": str(entry.get("title", "")).strip(),
                            }
                        )
                    except (TypeError, ValueError):
                        logger.debug("Skipping malformed structured entry: %s", entry)
        if not structured:
            logger.warning("Structuring drafts failed; falling back to intervals")
            fallback = context.metadata.get("intervals", [])
            structured = [
                {"start": item.get("start", 0.0), "end": item.get("end", item.get("start", 0.0)), "title": "Section"}
                for item in fallback
            ]
        context.metadata["title_candidates"] = structured
        context.metadata["title_batches_raw"] = raw_responses


class TitleOptimizationStage:
    name = "optimize_titles"

    def run(self, context: "PipelineContext") -> None:
        base_entries = context.metadata.get("title_candidates")
        if not base_entries:
            logger.warning("No title candidates to optimize")
            context.metadata["optimized_titles_raw"] = ""
            return
        provider = _ensure_provider(context)
        payload = json.dumps(base_entries, ensure_ascii=False)
        prompt = (
            "Improve clarity of these section titles but keep meaning."
            "Return JSON list with same fields (start,end,title).\n"
            f"Data:{payload}"
        )
        context.metadata["optimized_titles_raw"] = provider.generate_text(prompt, max_tokens=400)


class OptimizedTitleParsingStage:
    name = "parse_optimized_titles"

    def run(self, context: "PipelineContext") -> None:
        raw_text = context.metadata.get("optimized_titles_raw", "")
        payload = _safe_json_load(raw_text)
        if isinstance(payload, dict) and "sections" in payload:
            payload = payload["sections"]
        entries: list[dict[str, Any]] = []
        if isinstance(payload, list):
            for entry in payload:
                try:
                    entries.append(
                        {
                            "start": float(entry.get("start", 0.0)),
                            "end": float(entry.get("end", entry.get("start", 0.0))),
                            "title": str(entry.get("title", "")).strip(),
                        }
                    )
                except (TypeError, ValueError):
                    logger.debug("Skipping malformed optimized entry: %s", entry)
        context.metadata["optimized_titles"] = entries


class FinalFormattingStage:
    name = "finalize_sections"

    def run(self, context: "PipelineContext") -> None:
        entries = context.metadata.get("optimized_titles") or context.metadata.get("title_candidates")
        if not entries:
            logger.warning("No entries to finalize")
            context.sections = []
            return
        provider = _ensure_provider(context)
        config_obj = _ensure_generation_config(context)
        min_words, max_words = config_obj.title_length_range
        payload = json.dumps(entries, ensure_ascii=False)
        prompt = (
            f"Limit each title to {min_words}-{max_words} words. Keep start times."
            "Return JSON {\"sections\":[{\"start\":s,\"title\":\"...\"}]}."
            f"\nData:{payload}"
        )
        response = provider.generate_text(prompt, max_tokens=256)
        sections_data = _extract_sections_list(response)
        if not sections_data:
            logger.warning("Failed to parse final sections; using previous entries")
            sections_data = [
                {"start": entry.get("start", 0.0), "title": (entry.get("title") or "Section").strip()}
                for entry in entries
            ]
        context.sections = [Section(title=item["title"].strip(), start=float(item["start"])) for item in sections_data]


class FinalTitleTranslationStage:
    """Translate finalized English titles back to the original language.

    Expects 'original_language_code' to be present in context.metadata (set
    during transcript extraction). If the original language is English or
    unknown, this stage is a no-op. On any translation error, the original
    title is preserved.
    """

    name = "final_title_translation"

    def run(self, context: "PipelineContext") -> None:
        original_lang = context.metadata.get("original_language_code")
        if not original_lang or original_lang.lower() in {"en", "eng"}:
            logger.debug("Skipping title translation: original language '%s'", original_lang)
            return

        if not context.sections:
            logger.debug("No sections to translate")
            return

        # Prefer a translation provider exposed via pipeline context
        translator = context.metadata.get("translation_provider")

        # If not provided, try to initialize DeepLAdapter (via services.translation)
        if translator is None:
            try:
                # Use the adapter defined in services.translation
                from src.core.services.translation import DeepLAdapter

                api_key = os.getenv("DEEPL_API_KEY")
                if api_key:
                    translator = DeepLAdapter(api_key)
                    logger.info("FinalTitleTranslationStage: using DeepLAdapter fallback")
                else:
                    logger.warning("FinalTitleTranslationStage: no translation provider and DEEPL_API_KEY not set; skipping")
                    return
            except Exception as exc:  # pragma: no cover - env dependent
                logger.warning("Translation adapter unavailable; skipping title translation: %s", exc)
                return

        target_lang = original_lang.upper()
        translated_sections: list[Section] = []

        for section in context.sections:
            original_title = section.title
            if not original_title:
                translated_sections.append(section)
                continue
            try:
                # TranslationProvider interface: translate(text, target_lang, source_lang=None)
                new_title = translator.translate(original_title, target_lang=target_lang, source_lang="EN")
                if not new_title:
                    translated_sections.append(section)
                    continue
                section.title = str(new_title).strip()
            except Exception as exc:
                logger.warning("Failed to translate title '%s' to %s: %s", original_title, target_lang, exc)
                # Keep original title on failure
            translated_sections.append(section)

        context.sections = translated_sections


# ---------------------------------------------------------------------------
# Helper utilities shared by both families
# ---------------------------------------------------------------------------

def _ensure_provider(context: "PipelineContext"):
    provider = context.metadata.get("llm_provider")
    if provider is None:
        provider = LLMFactory.create_provider()
        context.metadata["llm_provider"] = provider
    return provider


def _ensure_generation_config(context: "PipelineContext") -> SectionGenerationConfig:
    config_obj = context.metadata.get("generation_config")
    if isinstance(config_obj, SectionGenerationConfig):
        return config_obj
    default = SectionGenerationConfig()
    context.metadata["generation_config"] = default
    return default


def _summarize_transcript(transcript: list[dict[str, Any]], limit: int = 40) -> str:
    if not transcript:
        return ""

    count = min(limit, len(transcript))
    if count == len(transcript):
        sampled = transcript
    else:
        indices = np.linspace(0, len(transcript) - 1, num=count, dtype=int)
        sampled = [transcript[i] for i in indices]

    lines: list[str] = []
    for seg in sampled:
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(f"{seg.get('start', 0.0):.1f}: {text}")
    return "\n".join(lines)


def _total_duration(transcript: list[dict[str, Any]]) -> float:
    if not transcript:
        return 0.0
    return max(seg.get("start", 0.0) + seg.get("duration", 0.0) for seg in transcript)


def _extract_intervals_from_text(text: str, fallback_duration: float) -> list[dict[str, float]]:
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|–|to|~)\s*(\d+(?:\.\d+)?)")
    intervals: list[dict[str, float]] = []
    for start_str, end_str in pattern.findall(text):
        try:
            start = float(start_str)
            end = float(end_str)
            if end < start:
                start, end = end, start
            intervals.append({"start": start, "end": end})
        except ValueError:
            continue
    if intervals:
        return intervals
    numbers = sorted({float(num) for num in re.findall(r"\d+(?:\.\d+)?", text)})
    pairs = []
    for idx in range(0, len(numbers), 2):
        try:
            start = numbers[idx]
            end = numbers[idx + 1]
        except IndexError:
            continue
        if end <= start:
            end = start + max(5.0, fallback_duration * 0.05)
        pairs.append({"start": start, "end": end})
    return pairs


def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _context_snippets(transcript: list[dict[str, Any]], intervals: list[dict[str, Any]]) -> str:
    snippets = []
    for interval in intervals:
        start = interval.get("start", 0.0)
        end = interval.get("end", start)
        window_text = _context_window(transcript, start, end)
        snippets.append(f"{start:.1f}-{end:.1f}: {window_text}")
    return "\n".join(snippets)


def _context_window(transcript: list[dict[str, Any]], start: float, end: float, pad: float = 15.0) -> str:
    lower = start - pad
    upper = end + pad
    texts = [seg.get("text", "") for seg in transcript if lower <= seg.get("start", 0.0) <= upper]
    joined = " ".join(texts)
    return joined[:800]


def _safe_json_load(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _extract_sections_list(raw: str) -> list[dict[str, Any]]:
    payload = _safe_json_load(raw)
    if isinstance(payload, dict) and "sections" in payload:
        payload = payload["sections"]
    if not isinstance(payload, list):
        return []
    sections: list[dict[str, Any]] = []
    for entry in payload:
        try:
            sections.append(
                {
                    "start": float(entry.get("start", 0.0)),
                    "title": str(entry.get("title", "")).strip() or "Section",
                }
            )
        except (TypeError, ValueError):
            logger.debug("Skipping malformed final entry: %s", entry)
    return sections


def _parse_title_payload(raw: str) -> dict[str, str]:
    """Parse the JSON payload from title-related prompts."""
    payload = _safe_json_load(raw)
    if isinstance(payload, dict):
        return payload
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
        payload = _safe_json_load(raw)
        if isinstance(payload, dict):
            return payload
    return {}


def _normalize_sections(sections: list[Section]) -> list[Section]:
    """Ensure consistent section data structure."""
    normalized: list[Section] = []
    for section in sections:
        if isinstance(section, Section):
            normalized.append(section)
        elif isinstance(section, dict):
            normalized.append(Section(**section))
    return normalized


__all__ = [
    # legacy
    "DetermineSectionsStage",
    "EnrichSectionsStage",
    "FinalizeTitlesStage",
    # new pipeline
    "LooseSectionDiscoveryStage",
    "IntervalParsingStage",
    "BatchTitleDraftStage",
    "TitleDraftStructStage",
    "TitleOptimizationStage",
    "OptimizedTitleParsingStage",
    "FinalFormattingStage",
    "FinalTitleTranslationStage",
]
