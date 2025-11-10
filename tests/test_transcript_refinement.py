"""Tests for transcript refinement service."""

import pytest
from unittest.mock import Mock, patch

from src.core.services.transcript_refinement import TranscriptRefinementService


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    with patch("src.core.services.transcript_refinement.LLMProviderFactory.create") as mock_factory:
        mock_provider = Mock()
        mock_factory.return_value = mock_provider
        yield mock_provider


@pytest.fixture
def sample_segments():
    """Sample transcript segments for testing."""
    return [
        {"text": "um so like I think that uh we should start", "start": 0.0, "duration": 3.5},
        {"text": "the the project today you know", "start": 3.5, "duration": 2.0},
        {"text": "and uh make sure äh everything is ready hmm", "start": 5.5, "duration": 3.0},
    ]


def test_refinement_service_initialization(mock_llm_provider):
    """Test that the refinement service initializes correctly."""
    service = TranscriptRefinementService()
    assert service.llm_provider is not None


def test_refine_single_segment(mock_llm_provider):
    """Test refining a single transcript segment."""
    mock_llm_provider.generate.return_value = "I think we should start"

    service = TranscriptRefinementService()
    result = service.refine_transcript_segment("um so like I think that uh we should start")

    assert result == "I think we should start"
    assert mock_llm_provider.generate.called


def test_refine_segment_with_context(mock_llm_provider):
    """Test refining a segment with surrounding context."""
    mock_llm_provider.generate.return_value = "the project today"

    service = TranscriptRefinementService()
    result = service.refine_transcript_segment(
        "the the project today you know", context="We should start"
    )

    assert result == "the project today"
    assert mock_llm_provider.generate.called


def test_refine_batch_preserves_timestamps(mock_llm_provider, sample_segments):
    """Test that batch refinement preserves timestamps."""
    mock_llm_provider.generate.return_value = "I think we should start the project today and make sure everything is ready"

    service = TranscriptRefinementService()
    result = service.refine_transcript_batch(sample_segments, batch_size=3)

    assert len(result) == len(sample_segments)
    for i, segment in enumerate(result):
        assert segment["start"] == sample_segments[i]["start"]
        assert segment["duration"] == sample_segments[i]["duration"]


def test_refine_batch_cleans_text(mock_llm_provider, sample_segments):
    """Test that batch refinement cleans the text content."""
    mock_llm_provider.generate.return_value = "I think we should start the project today and make sure everything is ready"

    service = TranscriptRefinementService()
    result = service.refine_transcript_batch(sample_segments, batch_size=3)

    # Check that filler words are removed
    combined_text = " ".join(s["text"] for s in result)
    assert "um" not in combined_text.lower()
    assert "uh" not in combined_text.lower()
    assert "like" not in combined_text.lower()


def test_refinement_failure_returns_original(mock_llm_provider, sample_segments):
    """Test that refinement failure returns original text."""
    mock_llm_provider.generate.side_effect = Exception("LLM error")

    service = TranscriptRefinementService()
    result = service.refine_transcript_batch(sample_segments, batch_size=3)

    # Should return original segments on failure
    assert len(result) == len(sample_segments)
    assert result[0]["text"] == sample_segments[0]["text"]


def test_empty_segments_handling(mock_llm_provider):
    """Test handling of empty segment list."""
    service = TranscriptRefinementService()
    result = service.refine_transcript_batch([])

    assert result == []


def test_single_segment_batch(mock_llm_provider):
    """Test refinement of a single segment as a batch."""
    mock_llm_provider.generate.return_value = "We should start"

    segment = [{"text": "um we should start", "start": 0.0, "duration": 2.0}]

    service = TranscriptRefinementService()
    result = service.refine_transcript_batch(segment, batch_size=1)

    assert len(result) == 1
    assert "um" not in result[0]["text"].lower()


def test_batch_context_building(mock_llm_provider, sample_segments):
    """Test that batch refinement uses surrounding context."""
    service = TranscriptRefinementService()

    # Mock the generate method to capture the prompt
    prompts_captured = []

    def capture_prompt(prompt, **kwargs):
        prompts_captured.append(prompt)
        return "refined text"

    mock_llm_provider.generate.side_effect = capture_prompt

    service.refine_transcript_batch(sample_segments, batch_size=2)

    # Check that context was included in the prompt
    assert len(prompts_captured) > 0
    assert any("context" in prompt.lower() for prompt in prompts_captured)

