"""Deprecated LocalLLMClient adapter shim.

The old `LocalLLMClient` adapter has been pruned. This module keeps a
compatibility shim which raises a clear error if instantiated.

Please use `src.core.llm.local_provider.LocalLLMProvider` or
`src.core.llm.factory.LLMFactory` directly.
"""

from __future__ import annotations

import warnings


class LocalLLMClient:  # pragma: no cover - compatibility shim
    """Compatibility shim for removed LocalLLMClient adapter.

    Instantiating this class raises a RuntimeError with guidance on the
    new API. We keep a shim to avoid ImportError on import but prevent use.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LocalLLMClient has been removed. Use src.core.llm.LocalLLMProvider or LLMFactory instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise RuntimeError(
            "LocalLLMClient adapter removed: please use src.core.llm.LocalLLMProvider(model_path=...) "
            "or LLMFactory.create_provider(...) to obtain a modern LLM provider."
        )


__all__ = ["LocalLLMClient"]
