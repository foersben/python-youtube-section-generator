"""Adapter moved from src.core.gemini_client.

Preserves legacy GeminiService API under adapters package.
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)


def _get_gemini_provider(
    api_key_env: str | None = None, model: str | None = None, temperature: float = 0.0
):
    from src.core.llm.gemini_provider import GeminiProvider

    # Use env var via config internally; pass-through kept for compatibility
    return GeminiProvider(api_key=None, model=model, temperature=temperature)


class GeminiService:
    """Deprecated adapter preserving older Gemini client API.

    For new code use `src.core.llm.GeminiProvider` or `LLMFactory.create_gemini_provider()`.
    """

    def __init__(self, api_key_env: str = "GOOGLE_API_KEY", model: str | None = None):
        warnings.warn(
            "GeminiService adapter is deprecated; use src.core.llm.GeminiProvider instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._provider = _get_gemini_provider(api_key_env=api_key_env, model=model)
        logger.info("Initialized GeminiService adapter")

    def generate(
        self,
        prompt: str,
        cache_contents: list[str],
        system_instruction: str = "",
        thinking_budget: int = 0,
        temperature: float = 0.0,
        retry_delay: float = 5.0,
        max_delay: float = 60.0,
    ) -> str:
        """Generate content via Gemini by delegating to the new provider."""
        # The new provider's API differs; flattening for compatibility
        return self._provider.generate_text(prompt=prompt, max_tokens=2048, temperature=temperature)


__all__ = ["GeminiService"]
