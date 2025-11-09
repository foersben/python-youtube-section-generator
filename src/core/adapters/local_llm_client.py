"""Adapter moved from src.core.local_llm_client.

This file preserves the legacy API but lives under `src.core.adapters`.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid heavy dependencies at module import time
def _get_local_provider(*, model_path: str | Path | None = None, n_ctx: int = 4096, n_threads: int | None = None, temperature: float = 0.2):
    import importlib

    mod = importlib.import_module("src.core.llm.local_provider")
    LocalLLMProvider = getattr(mod, "LocalLLMProvider")
    return LocalLLMProvider(model_path=model_path or None, n_ctx=n_ctx, n_threads=n_threads, temperature=temperature)


class LocalLLMClient:
    """Deprecated adapter for older LocalLLMClient users.

    This adapter maintains the older method names but delegates
    to the refactored LocalLLMProvider. Use `src.core.llm.LocalLLMProvider`
    or `src.core.llm.LLMFactory` directly in new code.
    """

    def __init__(self, model_path: str | None = None, n_ctx: int = 4096, n_threads: int | None = None, temperature: float = 0.2) -> None:
        warnings.warn(
            "LocalLLMClient is deprecated; use src.core.llm.LocalLLMProvider or LLMFactory instead",
            DeprecationWarning,
        )
        self._provider = _get_local_provider(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, temperature=temperature)
        logger.info("Initialized LocalLLMClient adapter")

    def generate_sections(self, transcript: list[dict[str, Any]], num_sections: int = 5, max_retries: int = 3) -> list[dict[str, Any]]:
        """Generate sections by delegating to the refactored provider."""
        return self._provider.generate_sections(transcript=transcript, num_sections=num_sections, max_retries=max_retries)

    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float | None = None) -> str:
        """Generate text by delegating to the refactored provider."""
        return self._provider.generate_text(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

    def get_info(self) -> dict[str, Any]:
        return self._provider.get_info()


__all__ = ["LocalLLMClient"]
