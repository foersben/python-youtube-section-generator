"""Adapter moved from src.core.deepl_client.

Preserves DeepL compatibility helper and client under adapters.
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)


def _get_deepl_adapter(api_key: str | None = None):
    # Lazy import to avoid import-time errors when deepl isn't installed
    import importlib

    mod = importlib.import_module("src.core.services.translation")
    deepl_adapter_cls = mod.DeepLAdapter
    return deepl_adapter_cls(api_key=api_key or None)


class DeepLClient:
    """Deprecated adapter for older code using deepl_client.py.

    Use `src.core.services.DeepLAdapter` directly in new code.
    """

    def __init__(self, api_key: str | None = None):
        warnings.warn(
            "DeepLClient adapter is deprecated; use src.core.services.DeepLAdapter instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._adapter = _get_deepl_adapter(api_key=api_key)
        logger.info("Initialized DeepLClient adapter")

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        return self._adapter.translate(text=text, target_lang=target_lang, source_lang=source_lang)


__all__ = ["DeepLClient"]


# Backwards-compatible helper function expected by older modules
def translate_text(text: str, target_lang: str, source_lang: str | None = None) -> str:
    """Translate text using DeepL adapter (compatibility helper).

    Args:
        text: Text to translate.
        target_lang: Target language code (e.g., 'EN', 'DE').
        source_lang: Optional source language code.

    Returns:
        Translated text as string.
    """
    adapter = _get_deepl_adapter(api_key=None)
    return adapter.translate(text=text, target_lang=target_lang, source_lang=source_lang)
