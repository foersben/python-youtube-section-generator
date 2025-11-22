"""Utilities for loading language-specific stopword lists.

This module prefers the public `stopwordsiso` corpus when available and falls
back to bundled text resources. Results are cached per language code.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from src.core.config import config as _config

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from stopwordsiso import stopwords as stopwordsiso
except Exception:  # pragma: no cover - best-effort import
    stopwordsiso = None

_RESOURCE_DIR = Path(_config.project_root) / "src" / "resources" / "stopwords"


def _load_resource_stopwords(language_code: str) -> set[str]:
    path = _RESOURCE_DIR / f"{language_code.lower()}.txt"
    if not path.exists():
        return set()
    try:
        data = path.read_text(encoding="utf-8")
    except Exception:  # pragma: no cover - IO guard
        logger.debug("Failed to read stopword resource for %s", language_code, exc_info=True)
        return set()
    return {line.strip().lower() for line in data.splitlines() if line.strip() and not line.startswith("#")}


@lru_cache(maxsize=64)
def get_stopwords(language_code: str) -> set[str]:
    """Return a lowercase stopword set for the requested language code.

    Args:
        language_code: ISO language code such as "en" or "de".

    Returns:
        Set of lowercase stopwords. Empty set when unavailable.
    """

    lang = (language_code or "").lower()
    if not lang:
        lang = "en"

    if stopwordsiso is not None:
        try:
            words = stopwordsiso(lang)
        except KeyError:
            words = set()
        except Exception:  # pragma: no cover - guard optional package behavior
            logger.debug("stopwordsiso lookup failed for %s", lang, exc_info=True)
            words = set()
        if words:
            return {w.lower() for w in words}

    fallback = _load_resource_stopwords(lang)
    if fallback:
        return fallback

    if lang != "en":
        logger.debug("Falling back to English stopwords for %s", lang)
        return get_stopwords("en")
    return set()


__all__ = ["get_stopwords"]

