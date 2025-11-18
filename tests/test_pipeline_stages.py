"""Unit tests for pipeline stage behavior."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.core.models import Section, SectionGenerationConfig
from src.core.services import pipeline_stages
from src.core.services.pipeline import PipelineContext


class DummyProvider:
    def __init__(self, sections: list[dict[str, Any]]) -> None:
        self.sections = sections

    def generate_sections(
        self, transcript: list[dict[str, Any]], num_sections: int, max_retries: int = 3
    ) -> list[dict[str, Any]]:
        return self.sections

    def generate_text(
        self, prompt: str, max_tokens: int = 256, temperature: float | None = None
    ) -> str:
        return json.dumps({"title": "Refined Title"})


@pytest.fixture()
def transcript() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "text": "Intro"},
        {"start": 10.0, "text": "Middle"},
        {"start": 20.0, "text": "End"},
    ]


def test_determine_stage_populates_sections(
    monkeypatch: pytest.MonkeyPatch, transcript: list[dict[str, Any]]
) -> None:
    stage = pipeline_stages.DetermineSectionsStage()
    ctx = PipelineContext(transcript=transcript)
    dummy_sections = [{"title": "A", "start": 0.0}]
    monkeypatch.setitem(ctx.metadata, "llm_provider", DummyProvider(dummy_sections))

    stage.run(ctx)

    assert ctx.sections


def test_enrich_stage_uses_existing_sections(
    monkeypatch: pytest.MonkeyPatch, transcript: list[dict[str, Any]]
) -> None:
    stage = pipeline_stages.EnrichSectionsStage()
    ctx = PipelineContext(transcript=transcript)
    ctx.metadata["raw_sections"] = [Section(title="A", start=0.0)]
    monkeypatch.setitem(ctx.metadata, "llm_provider", DummyProvider([]))

    stage.run(ctx)

    assert ctx.sections[0].title == "Refined Title"


def test_finalize_stage_polishes_titles(
    monkeypatch: pytest.MonkeyPatch, transcript: list[dict[str, Any]]
) -> None:
    stage = pipeline_stages.FinalizeTitlesStage()
    ctx = PipelineContext(transcript=transcript)
    ctx.metadata["enriched_sections"] = [Section(title="rough title", start=0.0)]
    ctx.metadata["generation_config"] = SectionGenerationConfig(
        min_title_words=2, max_title_words=4
    )
    monkeypatch.setitem(ctx.metadata, "llm_provider", DummyProvider([]))

    stage.run(ctx)

    # Provider returns 'Refined Title' (capital T); align expectation with provider behavior
    assert ctx.sections[0].title == "Refined Title"
