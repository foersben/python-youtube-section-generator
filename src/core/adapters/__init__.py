"""Adapter re-exports.

This module re-exports legacy adapter classes for backward
compatibility and discoverability.

Keep this thin: adapters themselves are small deprecated shims
that delegate to the refactored services/providers in
`src.core.services` and `src.core.llm`.
"""

from .deepl_client import DeepLClient
from .gemini_client import GeminiService
from .local_llm_client import LocalLLMClient

__all__ = ["DeepLClient", "GeminiService", "LocalLLMClient"]
