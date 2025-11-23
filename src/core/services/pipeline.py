"""Pipeline strategy orchestration for section generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.core.config import config
from src.core.models import Section

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Shared mutable state flowing through pipeline stages."""
    transcript: list[dict[str, Any]]
    sections: list[Section] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineStage(Protocol):
    name: str
    def run(self, context: PipelineContext) -> None: ...


class PipelineStrategy(Protocol):
    def build(self, context: PipelineContext) -> list[PipelineStage]: ...


class DefaultPipelineStrategy:
    """Legacy single-stage behavior."""
    def build(self, context: PipelineContext) -> list[PipelineStage]:
        from src.core.services.pipeline_stages import DetermineSectionsStage
        overshoot_ratio = max(config.pipeline_section_overshoot, 0.0)
        return [DetermineSectionsStage(section_multiplier=1.0 + overshoot_ratio)]


class SplitPromptPipelineStrategy:
    """
    New Stepwise Pipeline:
    1. Discovery (Granular)
    2. Drafting (Lean)
    3. Consolidation (Merge & Refine)
    4. Formatting & Translation
    """
    def build(self, context: PipelineContext) -> list[PipelineStage]:
        from src.core.services import pipeline_stages as stages

        return [
            stages.GranularSectionDiscoveryStage(),
            stages.IntervalParsingStage(),
            stages.BatchTitleDraftStage(),      # Drafting
            stages.SectionConsolidationStage(), # The "Puzzle" Merger
            stages.FinalFormattingStage(),
            stages.FinalTitleTranslationStage(),
        ]


STRATEGY_REGISTRY: dict[str, type[PipelineStrategy]] = {
    "legacy": DefaultPipelineStrategy,
    "split": SplitPromptPipelineStrategy,
}


def build_pipeline(context: PipelineContext) -> list[PipelineStage]:
    strategy_name = config.pipeline_strategy
    strategy_cls = STRATEGY_REGISTRY.get(strategy_name, DefaultPipelineStrategy)
    if strategy_cls is DefaultPipelineStrategy and strategy_name not in STRATEGY_REGISTRY:
        logger.warning("Unknown pipeline strategy '%s'; falling back to legacy", strategy_name)
    strategy = strategy_cls()
    stages = strategy.build(context)
    logger.info("Using pipeline strategy '%s' with %d stages", strategy_name, len(stages))
    return stages


__all__ = [
    "PipelineContext",
    "PipelineStage",
    "build_pipeline",
]
