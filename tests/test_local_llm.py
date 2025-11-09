"""Tests for local LLM client functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch


class TestLocalLLMClient:
    """Test suite for LocalLLMClient."""

    @pytest.fixture
    def mock_model_components(self):
        """Mock transformers components to avoid loading actual model."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        mock_model = MagicMock()

        return mock_tokenizer, mock_model

    def test_init_with_cuda_available(self, mock_model_components):
        """Test LocalLLMClient initialization when CUDA is available."""
        mock_tokenizer, mock_model = mock_model_components

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "src.core.local_llm_client.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "src.core.local_llm_client.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
        ):
            from src.core.local_llm_client import LocalLLMClient

            client = LocalLLMClient()

            assert client.device == "cuda"
            assert client.model_name == "microsoft/Phi-3-mini-4k-instruct"

    def test_init_cpu_fallback(self, mock_model_components):
        """Test LocalLLMClient initialization falls back to CPU."""
        mock_tokenizer, mock_model = mock_model_components

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "src.core.local_llm_client.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "src.core.local_llm_client.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
        ):
            from src.core.local_llm_client import LocalLLMClient

            client = LocalLLMClient()

            assert client.device == "cpu"

    def test_generate_sections_with_empty_transcript_raises_error(self, mock_model_components):
        """Test that empty transcript raises ValueError."""
        mock_tokenizer, mock_model = mock_model_components

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "src.core.local_llm_client.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "src.core.local_llm_client.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
        ):
            from src.core.local_llm_client import LocalLLMClient

            client = LocalLLMClient()

            with pytest.raises(ValueError, match="Transcript cannot be empty"):
                client.generate_sections(transcript=[])

    def test_validate_sections_rejects_invalid_format(self, mock_model_components):
        """Test validation rejects sections with invalid format."""
        mock_tokenizer, mock_model = mock_model_components

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "src.core.local_llm_client.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "src.core.local_llm_client.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
        ):
            from src.core.local_llm_client import LocalLLMClient

            client = LocalLLMClient()

            # Missing 'title' key
            invalid_sections = [{"start": 0.0}]
            transcript = [{"start": 0.0, "text": "test", "duration": 5.0}]

            result = client._validate_sections(invalid_sections, transcript)
            assert result is False

    def test_validate_sections_rejects_out_of_range_timestamps(self, mock_model_components):
        """Test validation rejects timestamps beyond transcript duration."""
        mock_tokenizer, mock_model = mock_model_components

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "src.core.local_llm_client.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "src.core.local_llm_client.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
        ):
            from src.core.local_llm_client import LocalLLMClient

            client = LocalLLMClient()

            sections = [
                {"title": "Intro", "start": 0.0},
                {"title": "Beyond", "start": 999.0},  # Beyond transcript
            ]
            transcript = [{"start": 0.0, "text": "test", "duration": 10.0}]

            result = client._validate_sections(sections, transcript)
            assert result is False

    def test_extract_json_from_valid_response(self, mock_model_components):
        """Test JSON extraction from LLM response."""
        mock_tokenizer, mock_model = mock_model_components

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "src.core.local_llm_client.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "src.core.local_llm_client.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
        ):
            from src.core.local_llm_client import LocalLLMClient

            client = LocalLLMClient()

            response = """Here is the JSON:
            [
                {"title": "Introduction", "start": 0.0},
                {"title": "Main Content", "start": 10.5}
            ]
            That's the result."""

            sections = client._extract_json(response)

            assert len(sections) == 2
            assert sections[0]["title"] == "Introduction"
            assert sections[1]["start"] == 10.5

    def test_extract_json_raises_error_when_no_json(self, mock_model_components):
        """Test JSON extraction raises error when no JSON found."""
        mock_tokenizer, mock_model = mock_model_components

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "src.core.local_llm_client.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch(
                "src.core.local_llm_client.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
        ):
            from src.core.local_llm_client import LocalLLMClient

            client = LocalLLMClient()

            response = "No JSON here, just text"

            with pytest.raises(ValueError, match="No JSON array found"):
                client._extract_json(response)
