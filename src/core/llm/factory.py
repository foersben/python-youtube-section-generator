"""LLM Factory for creating appropriate provider instances.

Implements Factory pattern for LLM provider instantiation.
"""

import logging
from pathlib import Path
from typing import Any

from src.core.config import config
from src.core.llm.base import LLMProvider
from src.core.llm.gemini_provider import GeminiProvider
from src.core.llm.local_provider import LocalLLMProvider

logger = logging.getLogger(__name__)


class NoOpProvider(LLMProvider):
    """Fallback provider that raises a clear error on use when no provider is configured."""

    def generate_sections(
        self,
        transcript: list[dict[str, Any]],
        num_sections: int = 5,
        max_retries: int = 3,
    ) -> list[dict[str, Any]]:
        raise RuntimeError(
            "No LLM provider configured. Set USE_LOCAL_LLM=true and provide a local model, or set GOOGLE_API_KEY for Gemini API."
        )

    def generate_text(
        self, prompt: str, max_tokens: int = 256, temperature: float | None = None
    ) -> str:
        raise RuntimeError("No LLM provider configured. See configuration.")

    def get_info(self) -> dict[str, Any]:
        return {"provider": "noop", "model": None, "device": "none"}


class SimpleHeuristicProvider(LLMProvider):
    """Lightweight heuristic provider used as a safe default when no LLM is configured.

    This provider divides the transcript into evenly spaced sections and uses
    the first few words of each section as a title. It is CPU-only and does
    not depend on external models; it provides predictable behaviour for the
    webapp when no LLM is available.
    """

    def generate_sections(
        self,
        transcript: list[dict[str, Any]],
        num_sections: int = 5,
        max_retries: int = 3,
    ) -> list[dict[str, Any]]:
        if not transcript:
            raise ValueError("Transcript cannot be empty")

        # Compute total duration
        total_duration = max(seg.get("start", 0) + seg.get("duration", 0) for seg in transcript)
        # If total_duration is zero, fallback to using number of segments
        if total_duration <= 0:
            # Use segment starts as section starts
            starts = [seg.get("start", 0) for seg in transcript]
            titles = [seg.get("text", "").split(".")[0][:50] for seg in transcript]
            return [
                {"title": t or f"Section {i+1}", "start": float(s)}
                for i, (t, s) in enumerate(zip(titles, starts))
            ][:num_sections]

        interval = total_duration / num_sections
        sections = []
        for i in range(num_sections):
            start_time = round(i * interval, 1)
            # Find nearest transcript segment
            candidate = min(transcript, key=lambda s: abs(s.get("start", 0) - start_time))
            title = candidate.get("text", "").strip()
            # Use first sentence or first 6 words as title
            if "." in title:
                title = title.split(".")[0]
            title_words = title.split()
            title_excerpt = " ".join(title_words[:6]) if title_words else f"Section {i+1}"
            sections.append({"title": title_excerpt, "start": float(start_time)})
        return sections

    def generate_text(
        self, prompt: str, max_tokens: int = 256, temperature: float | None = None
    ) -> str:
        # Very simple echo-like behavior for text generation (not used in production)
        return prompt[:max_tokens]

    def get_info(self) -> dict[str, Any]:
        return {"provider": "heuristic", "model": "simple-heuristic", "device": "cpu"}


class LLMFactory:
    """Factory for creating LLM provider instances (Factory pattern)."""

    @staticmethod
    def create_provider() -> LLMProvider:
        """Create appropriate LLM provider based on configuration.

        Returns:
            Configured LLM provider instance.

        Raises:
            RuntimeError: If provider cannot be created.
        """
        if config.use_local_llm:
            try:
                return LLMFactory.create_local_provider()
            except Exception:
                logger.exception("Failed to create local provider, falling back to NoOpProvider")
                return SimpleHeuristicProvider()
        else:
            try:
                return LLMFactory.create_gemini_provider()
            except Exception:
                logger.exception("Failed to create Gemini provider, falling back to NoOpProvider")
                return SimpleHeuristicProvider()

    @staticmethod
    def create_local_provider(
        model_path: str | Path | None = None,
        temperature: float | None = None,
    ) -> LocalLLMProvider:
        """Create local LLM provider.

        Args:
            model_path: Path to model (None = use config).
            temperature: Sampling temperature (None = use config).

        Returns:
            Configured LocalLLMProvider instance.
        """
        path = model_path or config.local_model_path
        temp = temperature if temperature is not None else config.llm_temperature

        logger.info(f"Creating local LLM provider: {path}")

        return LocalLLMProvider(
            model_path=path,
            n_ctx=config.llm_context_size,
            temperature=temp,
        )

    @staticmethod
    def create_gemini_provider(
        api_key: str | None = None,
        temperature: float | None = None,
    ) -> GeminiProvider:
        """Create Gemini API provider.

        Args:
            api_key: API key (None = use config).
            temperature: Sampling temperature (None = use config).

        Returns:
            Configured GeminiProvider instance.
        """
        key = api_key or config.google_api_key
        if not key:
            raise ValueError("Google API key not configured")

        temp = temperature if temperature is not None else config.llm_temperature

        logger.info("Creating Gemini API provider")

        return GeminiProvider(
            api_key=key,
            temperature=temp,
        )


__all__ = ["LLMFactory"]
