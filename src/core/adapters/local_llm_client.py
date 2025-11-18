"""Adapter moved from src.core.local_llm_client.

This file preserves the legacy API but lives under `src.core.adapters`.
"""

from __future__ import annotations

import importlib
import logging
import warnings
from pathlib import Path
from typing import Any

try:
    from filelock import FileLock
except Exception as e:
    raise RuntimeError(
        "filelock is required for serializing local model loads. Run `poetry install` to install project dependencies (filelock)."
    ) from e

logger = logging.getLogger(__name__)


# Lazy import to avoid heavy dependencies at module import time
def _get_local_provider(
    *,
    model_path: str | Path | None = None,
    n_ctx: int = 4096,
    n_threads: int | None = None,
    temperature: float = 0.2,
):
    """Create and return a LocalLLMProvider while serializing model loads across processes.

    Use `filelock.FileLock` to serialize model loads across processes. The lock
    file is placed next to the model file (default: `models/.llm_load.lock`).
    """

    # Determine lock directory (model parent or ./models)
    try:
        lock_dir = Path(model_path).parent if model_path else Path.cwd() / "models"
    except Exception:
        lock_dir = Path.cwd() / "models"

    lock_dir.mkdir(parents=True, exist_ok=True)
    lockfile = lock_dir / ".llm_load.lock"

    # Always use FileLock (we raised earlier if filelock missing)
    lock = FileLock(str(lockfile))
    logger.info(f"Waiting for local LLM filelock: {lockfile}")
    with lock:
        logger.info(f"Acquired local LLM filelock: {lockfile}")
        mod = importlib.import_module("src.core.llm.local_provider")
        LocalLLMProvider = mod.LocalLLMProvider
        provider = LocalLLMProvider(
            model_path=model_path or None, n_ctx=n_ctx, n_threads=n_threads, temperature=temperature
        )
        logger.info(f"Releasing local LLM filelock: {lockfile}")
        return provider


class LocalLLMClient:
    """Deprecated adapter for older LocalLLMClient users.

    This adapter maintains the older method names but delegates
    to the refactored LocalLLMProvider. Use `src.core.llm.LocalLLMProvider`
    or `src.core.llm.LLMFactory` directly in new code.
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        temperature: float = 0.2,
    ) -> None:
        warnings.warn(
            "LocalLLMClient is deprecated; use src.core.llm.LocalLLMProvider or LLMFactory instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._provider = _get_local_provider(
            model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, temperature=temperature
        )
        logger.info("Initialized LocalLLMClient adapter")

    def generate_sections(
        self, transcript: list[dict[str, Any]], num_sections: int = 5, max_retries: int = 3
    ) -> list[dict[str, Any]]:
        """Generate sections by delegating to the refactored provider."""
        return self._provider.generate_sections(
            transcript=transcript, num_sections=num_sections, max_retries=max_retries
        )

    def generate_text(
        self, prompt: str, max_tokens: int = 512, temperature: float | None = None
    ) -> str:
        """Generate text by delegating to the refactored provider."""
        return self._provider.generate_text(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature
        )

    def get_info(self) -> dict[str, Any]:
        return self._provider.get_info()


__all__ = ["LocalLLMClient"]
