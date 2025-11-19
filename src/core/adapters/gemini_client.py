"""Deprecated Gemini adapter shim.

The old `GeminiService` adapter has been pruned. This module keeps a
compatibility shim which raises a clear error if instantiated.

Please use `src.core.llm.gemini_provider.GeminiProvider` or the LLM factory.
"""

from __future__ import annotations

import warnings


class GeminiService:  # pragma: no cover - compatibility shim
    """Compatibility shim for removed GeminiService adapter.

    Instantiating this class raises a RuntimeError with guidance on the
    new API. We keep a shim to avoid ImportError on import but prevent use.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GeminiService has been removed. Use src.core.llm.GeminiProvider or LLMFactory instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise RuntimeError(
            "GeminiService adapter removed: please use src.core.llm.GeminiProvider(api_key) "
            "or LLMFactory.create_provider(...) to obtain a modern LLM provider."
        )


__all__ = ["GeminiService"]
