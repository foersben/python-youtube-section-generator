"""Pipeline strategy orchestration for section generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, Any

from src.core.config import config
from src.core.models import Section

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Shared mutable state flowing through pipeline stages.

    Attributes:
        transcript: Normalized transcript segments.
        sections: Intermediate list of sections shared across stages.
        metadata: Arbitrary data stash for cross-stage coordination.
    """

    transcript: list[dict[str, Any]]
    sections: list[Section] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineStage(Protocol):
    """Executable unit in the processing pipeline."""

    name: str

    def run(self, context: PipelineContext) -> None:
        """Execute stage logic and mutate pipeline context."""
        ...


class PipelineStrategy(Protocol):
    """Factory for composing stage sequences based on strategy name."""

    def build(self, context: PipelineContext) -> list[PipelineStage]:
        ...


class DefaultPipelineStrategy:
    """Legacy single-stage behavior returning existing section results."""

    def build(self, context: PipelineContext) -> list[PipelineStage]:
        from src.core.services.pipeline_stages import DetermineSectionsStage

        overshoot_ratio = max(config.pipeline_section_overshoot, 0.0)
        return [DetermineSectionsStage(section_multiplier=1.0 + overshoot_ratio)]


class SplitPromptPipelineStrategy:
    """Pipeline that splits section detection, enrichment, and tightening."""

    def build(self, context: PipelineContext) -> list[PipelineStage]:
        from src.core.services import pipeline_stages as stages

        return [
            stages.LooseSectionDiscoveryStage(),
            stages.IntervalParsingStage(),
            stages.BatchTitleDraftStage(),
            stages.TitleDraftStructStage(),
            stages.TitleOptimizationStage(),
            stages.OptimizedTitleParsingStage(),
            stages.FinalFormattingStage(),
            stages.FinalTitleTranslationStage(),
        ]


STRATEGY_REGISTRY: dict[str, type[PipelineStrategy]] = {
    "legacy": DefaultPipelineStrategy,
    "split": SplitPromptPipelineStrategy,
}


def build_pipeline(context: PipelineContext) -> list[PipelineStage]:
    """Build stage sequence based on configured pipeline strategy."""

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
