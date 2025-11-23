"""Pipeline stage implementations for section generation workflows."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.core.config import config
from src.core.llm import LLMFactory
from src.core.models import Section, SectionGenerationConfig
from src.core.retrieval.segmentation import select_coarse_sections

if TYPE_CHECKING:
    from src.core.services.pipeline import PipelineContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# New Stepwise Pipeline Stages (Granular -> Draft -> Consolidate)
# ---------------------------------------------------------------------------


@dataclass
class GranularSectionDiscoveryStage:
    """Step 1: Discover sections (Oversampling)."""
    name: str = "discover_granular_sections"

    def run(self, context: PipelineContext) -> None:
        config_obj = _ensure_generation_config(context)

        # COMPROMISE: Target 2.0x (instead of 3.0x) to save processing time.
        # This finds enough puzzle pieces to merge without overloading the LLM.
        target_count = min(40, int(config_obj.max_sections * 2.0))

        logger.info("Step 1: Discovering ~%d granular puzzle pieces...", target_count)

        semantic_sections = []
        try:
            semantic_sections = select_coarse_sections(
                transcript=context.transcript,
                target_sections=target_count,
                chunk_size=1000,
                chunk_overlap=200,
                device=config.embeddings_device,
            )
        except Exception as e:
            logger.error("Semantic segmentation failed: %s", e)

        total_duration = _total_duration(context.transcript)
        intervals: list[dict[str, float]] = []

        if semantic_sections and len(semantic_sections) >= config_obj.min_sections:
            semantic_sections.sort(key=lambda x: x.get("start", 0.0))
            for i, sec in enumerate(semantic_sections):
                start = float(sec.get("start", 0.0))
                if i < len(semantic_sections) - 1:
                    end = float(semantic_sections[i+1].get("start", total_duration))
                else:
                    end = total_duration

                if end <= start + 5.0:
                    end = start + 20.0
                intervals.append({"start": start, "end": end})
        else:
            logger.warning("Segmentation yielded too few results. Using uniform fallback.")
            step = total_duration / max(5, target_count)
            for i in range(int(target_count)):
                intervals.append({"start": step * i, "end": step * (i + 1)})

        context.metadata["intervals"] = intervals
        logger.info("Found %d granular intervals.", len(intervals))


class IntervalParsingStage:
    name = "parse_intervals"
    def run(self, context: PipelineContext) -> None:
        if not context.metadata.get("intervals"):
            context.metadata["intervals"] = []


@dataclass
class BatchTitleDraftStage:
    """Step 2: Draft titles for all granular sections."""
    batch_size: int = 3
    name: str = "draft_granular_titles"

    def run(self, context: PipelineContext) -> None:
        intervals = context.metadata.get("intervals", [])
        if not intervals:
            return

        provider = _ensure_provider(context)
        batches = _chunk(intervals, self.batch_size)
        draft_sections: list[dict[str, Any]] = []

        logger.info("Step 2: Drafting titles for %d intervals...", len(intervals))

        for i, chunk in enumerate(batches):
            snippets = _context_snippets(context.transcript, chunk, char_limit=1200)

            # IMPROVED PROMPT: Journalism-style headlines
            prompt = (
                "Task: Write a concise, engaging headline (4-6 WORDS) for each numbered section.\n"
                "Format: Numbered list (1., 2., 3.).\n"
                "Style: Journalism headlines. Focus on the 'Why', 'How', or 'Result'.\n"
                "Avoid: 'Intro', 'Summary', 'Discussion about...'.\n"
                "Example: 'Achieving Microfoam With Oat Milk'\n\n"
                f"Context:\n{snippets}\n\nHeadlines:"
            )

            try:
                response = provider.generate_text(prompt, max_tokens=256)
                lines = [l.strip() for l in response.strip().split('\n') if l.strip()]

                chunk_idx = 0
                for line in lines:
                    if chunk_idx >= len(chunk): break

                    # Cleanup
                    clean_title = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                    clean_title = re.sub(r'\s*[\(\[]?(\d+:\d+|\d+s)[\)\]]?\s*$', '', clean_title).strip()

                    if len(clean_title) > 2:
                        draft_sections.append({
                            "start": chunk[chunk_idx]["start"],
                            "end": chunk[chunk_idx]["end"],
                            "title": clean_title
                        })
                        chunk_idx += 1

                while chunk_idx < len(chunk):
                    draft_sections.append({
                        "start": chunk[chunk_idx]["start"],
                        "end": chunk[chunk_idx]["end"],
                        "title": "Key Topic Section"
                    })
                    chunk_idx += 1

            except Exception as e:
                logger.error(f"Batch {i} drafting failed: {e}")
                for c in chunk:
                    draft_sections.append({"start": c["start"], "end": c["end"], "title": "Key Topic Section"})

        context.metadata["draft_sections"] = draft_sections
        logger.info("Drafted %d raw sections.", len(draft_sections))


@dataclass
class SectionConsolidationStage:
    """
    Step 3: The "Puzzle" Step (Merge & Filter).
    OPTIMIZED: Uses title-based consolidation (Fast) with dynamic prompts.
    """
    name: str = "consolidate_sections"

    def run(self, context: PipelineContext) -> None:
        drafts = context.metadata.get("draft_sections", [])
        if not drafts:
            return

        config_obj = _ensure_generation_config(context)
        min_sections = config_obj.min_sections
        max_sections = config_obj.max_sections
        min_w, max_w = config_obj.title_length_range

        provider = _ensure_provider(context)
        total_duration = _total_duration(context.transcript)

        # 1. Structure Classification (Heuristic)
        draft_text_blob = " ".join([d["title"].lower() for d in drafts])
        is_qa_format = "question" in draft_text_blob or "answer" in draft_text_blob
        is_long_form = total_duration > 180.0

        # 2. Dynamic Prompt
        structure_guide = ""
        if is_qa_format:
            logger.info("Detected Q&A/Interview structure.")
            structure_guide = (
                "Structure: Q&A Session.\n"
                "- Group by Interaction: Question + Answer = One Chapter.\n"
                "- Titles: The core question or topic discussed.\n"
            )
        elif is_long_form:
            logger.info("Detected Standard Long-Form structure.")
            structure_guide = (
                "Structure: Narrative Arc.\n"
                "- Intro: 'Introduction to [Topic]'.\n"
                "- Body: Merge sequential technical steps into broader themes.\n"
                "- Outro: 'Conclusion' or 'Final Thoughts' (only if present).\n"
            )
        else:
            logger.info("Detected Short-Form structure.")
            structure_guide = "Structure: Short Clip. Merge into logical flow."

        timeline_text = "\n".join([f"- {d['start']:.1f}s: {d['title']}" for d in drafts])

        logger.info("Step 3: Consolidating %d drafts -> %d-%d final chapters.", len(drafts), min_sections, max_sections)

        prompt = (
            f"I have a timeline of {len(drafts)} raw segments.\n"
            f"Consolidate them into exactly {min_sections} to {max_sections} broader chapters.\n\n"
            f"{structure_guide}\n"
            f"Constraint: Titles MUST be {min_w}-{max_w} words long.\n"
            "Constraint: Do NOT include end times or durations in titles.\n"
            "Output Format: Line-by-line 'StartSeconds - New Title'\n\n"
            f"Raw Timeline:\n{timeline_text}\n\nMerged Chapters:"
        )

        final_candidates = []
        try:
            response = provider.generate_text(prompt, max_tokens=512)
            lines = response.strip().split('\n')
            for line in lines:
                # Parse: "123.4 - Title"
                match = re.search(r'(\d+(?:\.\d+)?)\s*[sS]?\s*[:-]\s*(.+)', line)
                if match:
                    try:
                        t_start = float(match.group(1))
                        t_title = match.group(2).strip()

                        # Filter generic single-word titles for long-form content
                        if is_long_form and len(t_title.split()) < 2:
                            if t_title.lower() == "introduction": t_title = "Introduction to Topic"
                            elif t_title.lower() == "conclusion": t_title = "Final Conclusion"

                        if t_title:
                            final_candidates.append({"start": t_start, "title": t_title})
                    except ValueError:
                        continue

        except Exception as e:
            logger.error(f"Consolidation LLM call failed: {e}")

        # 3. Deterministic Fallback (Shortest Segment Merge)
        # This ensures we honor the max_sections limit even if the LLM ignores it.
        work_list = final_candidates if (final_candidates and len(final_candidates) >= min_sections) else drafts

        work_list.sort(key=lambda x: x["start"])
        for i in range(len(work_list)):
            next_start = work_list[i+1]["start"] if i < len(work_list)-1 else total_duration
            work_list[i]["duration"] = next_start - work_list[i]["start"]

        while len(work_list) > max_sections:
            min_idx = -1
            min_dur = float('inf')
            for i, item in enumerate(work_list):
                if i == 0 and item["start"] == 0.0: continue
                if item["duration"] < min_dur:
                    min_dur = item["duration"]
                    min_idx = i

            if min_idx == -1: min_idx = 1

            if min_idx > 0:
                work_list[min_idx - 1]["duration"] += work_list[min_idx]["duration"]
                del work_list[min_idx]
            else:
                if len(work_list) > 1:
                    work_list[0]["duration"] += work_list[1]["duration"]
                    del work_list[1]
                else:
                    break

        context.metadata["optimized_titles"] = work_list
        logger.info("Finalized %d sections.", len(work_list))


class FinalFormattingStage:
    """Validates and finalizes sections, stripping any leaked timestamps."""
    name = "finalize_sections"

    def run(self, context: PipelineContext) -> None:
        entries = context.metadata.get("optimized_titles") or context.metadata.get("draft_sections")
        if not entries:
            context.sections = []
            return

        final_sections = []
        seen_starts = set()
        entries.sort(key=lambda x: x.get("start", 0.0))

        for entry in entries:
            title = entry.get("title", "Section").strip()
            start = float(entry.get("start", 0.0))

            if start in seen_starts: continue
            seen_starts.add(start)

            # Aggressive Cleanup
            title = re.sub(r'^[\s:\.\-]*', '', title)
            title = re.sub(r'^\[\d+(\.\d+)?[sS]?\]\s*[:\-]?\s*', '', title)
            title = re.sub(r'^\(\d+(\.\d+)?[sS]?\)\s*[:\-]?\s*', '', title)
            title = re.sub(r'^\d+(\.\d+)?[sS]?\s*[:-]\s*', '', title)
            title = re.sub(r'[\s-]*\d+(\.\d+)?[sS]?\s*$', '', title)

            title = title.strip("\"' ")

            if title and title[0].islower():
                title = title[0].upper() + title[1:]

            final_sections.append(Section(start=start, title=title))

        context.sections = final_sections


# ---------------------------------------------------------------------------
# Helper utilities (unchanged)
# ---------------------------------------------------------------------------

def _ensure_provider(context: PipelineContext):
    provider = context.metadata.get("llm_provider")
    if provider is None:
        provider = LLMFactory.create_provider()
        context.metadata["llm_provider"] = provider
    return provider

def _ensure_generation_config(context: PipelineContext) -> SectionGenerationConfig:
    config_obj = context.metadata.get("generation_config")
    if isinstance(config_obj, SectionGenerationConfig):
        return config_obj
    default = SectionGenerationConfig()
    context.metadata["generation_config"] = default
    return default

def _total_duration(transcript: list[dict[str, Any]]) -> float:
    if not transcript: return 0.0
    return max(seg.get("start", 0.0) + seg.get("duration", 0.0) for seg in transcript)

def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]

def _context_snippets(transcript: list[dict[str, Any]], intervals: list[dict[str, Any]], char_limit: int = 1200) -> str:
    snippets = []
    for interval in intervals:
        start = interval.get("start", 0.0)
        end = interval.get("end", start)
        window_text = _context_window(transcript, start, end, char_limit=char_limit)
        snippets.append(f"[{start:.1f}s]: {window_text}")
    return "\n".join(snippets)

def _context_window(
        transcript: list[dict[str, Any]], start: float, end: float, pad: float = 15.0, char_limit: int = 1200
) -> str:
    lower = start - pad
    upper = end + pad
    texts = [seg.get("text", "") for seg in transcript if lower <= seg.get("start", 0.0) <= upper]
    joined = " ".join(texts)
    return joined[:char_limit]

def _safe_json_load(raw: str) -> Any:
    raw = raw.strip()
    if not raw: return None
    try: return json.loads(raw)
    except json.JSONDecodeError: return None

# ---------------------------------------------------------------------------
# Legacy Exports
# ---------------------------------------------------------------------------

@dataclass
class DetermineSectionsStage:
    section_multiplier: float = 1.0
    name: str = "determine_sections"
    def run(self, context: PipelineContext) -> None: pass

class EnrichSectionsStage:
    name = "enrich_sections"
    def run(self, context: PipelineContext) -> None: pass

class FinalizeTitlesStage:
    name = "finalize_titles"
    def run(self, context: PipelineContext) -> None: pass

class TitleDraftStructStage:
    name = "structure_title_drafts"
    def run(self, context: PipelineContext) -> None: pass

class TitleOptimizationStage:
    name = "optimize_titles"
    def run(self, context: PipelineContext) -> None: pass

class OptimizedTitleParsingStage:
    name = "parse_optimized_titles"
    def run(self, context: PipelineContext) -> None: pass

LooseSectionDiscoveryStage = GranularSectionDiscoveryStage
UnifiedSectionGenerationStage = BatchTitleDraftStage

class FinalTitleTranslationStage:
    name = "final_title_translation"
    def run(self, context: PipelineContext) -> None:
        # Priority: User Target (from UI) > Original Lang (from Extractor)
        user_target = context.metadata.get("target_language")
        original_lang = context.metadata.get("original_language_code")

        target_lang = user_target or original_lang

        if not target_lang or target_lang.lower() in {"en", "eng"}:
            logger.debug("Skipping translation (Target is English)")
            return
        if not context.sections:
            return

        translator = context.metadata.get("translation_provider")
        if translator is None:
            try:
                from src.core.services.translation import DeepLAdapter
                api_key = os.getenv("DEEPL_API_KEY")
                if api_key: translator = DeepLAdapter(api_key)
            except Exception: return
        if not translator:
            return

        logger.info("Translating final titles to: %s", target_lang)
        target_lang_code = target_lang.upper()
        translated_sections = []
        for section in context.sections:
            try:
                new_title = translator.translate(section.title, target_lang=target_lang_code, source_lang="EN")
                if new_title: section.title = str(new_title).strip()
            except Exception: pass
            translated_sections.append(section)
        context.sections = translated_sections

__all__ = [
    "DetermineSectionsStage",
    "EnrichSectionsStage",
    "FinalizeTitlesStage",
    "LooseSectionDiscoveryStage",
    "IntervalParsingStage",
    "UnifiedSectionGenerationStage",
    "BatchTitleDraftStage",
    "SectionConsolidationStage",
    "TitleDraftStructStage",
    "TitleOptimizationStage",
    "OptimizedTitleParsingStage",
    "FinalFormattingStage",
    "FinalTitleTranslationStage",
]
