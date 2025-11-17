"""Translation service adapters.

Provides adapters for external translation APIs (DeepL, etc.) and a local
LlamaCpp-based fallback that uses the configured GGUF model to translate
texts when DeepL is unavailable.
"""

from __future__ import annotations

import logging
import math
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from src.core.config import config as _config
from src.utils.file_io import read_json_file, write_to_file

logger = logging.getLogger(__name__)


class TranslationQuotaExceeded(Exception):
    """Raised when an external translation provider reports quota exceeded."""


class TranslationProvider(ABC):
    """Abstract base class for translation providers."""

    @abstractmethod
    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        """Translate text to target language.

        Args:
            text: Text to translate.
            target_lang: Target language code.
            source_lang: Source language code (None = auto-detect).

        Returns:
            Translated text.
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self) -> dict[str, str]:
        """Get provider information."""
        raise NotImplementedError


class DeepLAdapter(TranslationProvider):
    """DeepL translation API adapter.

    Adapts DeepL API to our TranslationProvider interface.
    """

    def __init__(self, api_key: str):
        """Initialize DeepL adapter.

        Args:
            api_key: DeepL API key.

        Raises:
            ValueError: If API key is empty.
            RuntimeError: If deepl package not installed.
        """
        if not api_key:
            raise ValueError("DeepL API key cannot be empty")

        try:
            import deepl  # type: ignore

            self.deepl = deepl
        except ImportError:
            raise RuntimeError("deepl package not installed. " "Install with: poetry add deepl")

        self.translator = self.deepl.Translator(api_key)
        logger.info("Initialized DeepL translator")

        # simple in-memory flag to avoid re-hitting DeepL after quota errors
        # store timestamp in environment to persist across processes if desired
        self._quota_disabled_until = 0
        # persisted quota file location
        self._cache_dir = Path(_config.project_root) / ".cache" / "translations"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._quota_file = self._cache_dir / "deepl_quota.json"

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        """Translate text using DeepL.

        Args:
            text: Text to translate.
            target_lang: Target language code (e.g., 'EN-US', 'EN-GB', 'DE', 'FR').
            source_lang: Source language code (None = auto-detect).

        Returns:
            Translated text.

        Raises:
            RuntimeError: If translation fails.
        """
        # Normalize English variants for DeepL API compatibility
        target_normalized = target_lang.upper()
        if target_normalized == "EN":
            target_normalized = "EN-US"  # Default to US English

        source_normalized = None
        if source_lang:
            source_normalized = source_lang.upper()
            if source_normalized == "EN":
                source_normalized = "EN-US"

        # respect short-circuit if quota hit recently
        import time

        # reload persisted quota if present
        try:
            if self._quota_file.exists():
                q = read_json_file(str(self._quota_file))
                self._quota_disabled_until = int(q.get("quota_disabled_until", 0))
        except Exception:
            pass

        if self._quota_disabled_until and time.time() < self._quota_disabled_until:
            raise RuntimeError("DeepL temporarily disabled due to previous quota errors")

        try:
            result = self.translator.translate_text(
                text,
                target_lang=target_normalized,
                source_lang=source_normalized,
            )
            return str(result)

        except Exception as e:
            msg = str(e)
            logger.error(f"DeepL translation failed: {e}")
            # Detect quota exceeded messages and set a short disable period
            if "Quota" in msg or "Quota exceeded" in msg or "456" in msg:
                # determine cooldown from env or default to 1 hour
                cooldown = int(os.getenv("DEEPL_COOLDOWN_SECONDS", "3600"))
                self._quota_disabled_until = int(time.time()) + cooldown
                # persist
                try:
                    write_to_file(
                        {"quota_disabled_until": self._quota_disabled_until}, str(self._quota_file)
                    )
                except Exception:
                    logger.debug("Failed to persist DeepL quota file")
                raise TranslationQuotaExceeded(f"DeepL quota exceeded: {msg}") from e
            # Rate limit 429 - raise transient exception to be possibly retried elsewhere
            if "Too many requests" in msg or "429" in msg:
                raise RuntimeError("DeepL rate limit / server busy: " + msg) from e

            raise RuntimeError(f"Translation failed: {e}") from e

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
        separator: str = "\n<<<SEG>>>\n",
    ) -> list[str]:
        """Translate a list of texts in one call by joining with a separator.

        This reduces the number of HTTP calls and helps avoid hitting quotas.

        Args:
            texts: list of strings to translate
            target_lang: target language code
            source_lang: optional source language
            separator: unique separator string to join texts

        Returns:
            list of translated strings in same order
        """
        if not texts:
            return []

        # Location for caching and quota persistence
        cache_dir = Path(_config.project_root) / ".cache" / "translations"
        cache_dir.mkdir(parents=True, exist_ok=True)
        quota_file = cache_dir / "deepl_quota.json"

        # Load persisted quota info if present
        now = int(time.time())
        quota_disabled_until = 0
        try:
            if quota_file.exists():
                data = read_json_file(str(quota_file))
                quota_disabled_until = int(data.get("quota_disabled_until", 0))
                # reflect to in-memory state
                self._quota_disabled_until = quota_disabled_until
        except Exception:
            quota_disabled_until = self._quota_disabled_until

        if quota_disabled_until and now < quota_disabled_until:
            raise TranslationQuotaExceeded(
                "DeepL quota previously exceeded; disabled until %d" % quota_disabled_until
            )

        # Chunk inputs to respect max chars and max segments per batch
        MAX_CHARS = 10_000
        MAX_SEGMENTS = 100

        results: list[str] = []

        # Build batches
        i = 0
        n = len(texts)
        while i < n:
            # expand window to include up to MAX_SEGMENTS but also respect char limit
            j = i
            chars = 0
            while (
                j < n
                and (j - i) < MAX_SEGMENTS
                and (chars + (len(texts[j]) if texts[j] else 0)) <= MAX_CHARS
            ):
                chars += len(texts[j]) if texts[j] else 0
                j += 1

            if j == i:
                # single segment too large; force at least one
                j = min(i + 1, n)

            batch = texts[i:j]

            joined = separator.join(batch)
            try:
                translated_joined = self.translate(joined, target_lang, source_lang)
            except TranslationQuotaExceeded as tqe:
                # persist the quota disable and re-raise
                # determine cooldown from env
                cooldown = int(os.getenv("DEEPL_COOLDOWN_SECONDS", "3600"))
                quota_disabled_until = int(time.time()) + cooldown
                try:
                    write_to_file({"quota_disabled_until": quota_disabled_until}, str(quota_file))
                except Exception:
                    logger.debug("Failed to persist DeepL quota file")
                raise

            # split back
            parts = translated_joined.split(separator)
            if len(parts) != len(batch):
                logger.warning(
                    "DeepL batch translation returned %d parts for %d inputs; falling back to per-segment translate for that batch",
                    len(parts),
                    len(batch),
                )
                # fallback: translate each segment individually (this may hit quota)
                for t in batch:
                    results.append(self.translate(t, target_lang, source_lang))
            else:
                results.extend(parts)

            i = j

        return results

    def get_info(self) -> dict[str, str]:
        """Get provider information."""
        return {
            "provider": "deepl",
            "api": "DeepL API",
        }


class LlamaCppTranslator(TranslationProvider):
    """Local translator using llama.cpp via LangChain LlamaCpp.

    Uses the local GGUF model specified by LOCAL_MODEL_PATH. Designed as a
    fallback when DeepL is unavailable or disabled. This is slower and less
    accurate than DeepL but avoids external API calls.
    """

    def __init__(self, model_path: str | None = None, temperature: float = 0.1) -> None:
        try:
            from langchain_community.llms import LlamaCpp  # type: ignore
        except Exception as e:  # pragma: no cover - runtime env check
            raise RuntimeError(
                f"LangChain LlamaCpp not available: {e}. Install langchain-community and llama-cpp-python."
            ) from e

        if model_path is None:
            model_path = os.getenv("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
        if not os.path.isabs(model_path):
            from src.core.config import config as _config

            model_path = str(Path(_config.project_root) / model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Local model not found at {model_path}")

        self.model_path = model_path
        import multiprocessing

        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_ctx=2048,
            n_threads=multiprocessing.cpu_count(),
            temperature=temperature,
            max_tokens=256,
            top_p=0.9,
            verbose=False,
        )
        logger.info("Initialized LlamaCppTranslator with %s", Path(self.model_path).name)

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        # Normalize language code to readable name for prompt
        lang_map = {
            "EN": "English",
            "EN-US": "English",
            "EN-GB": "English",
            "DE": "German",
            "FR": "French",
            "ES": "Spanish",
            "IT": "Italian",
            "PT": "Portuguese",
        }
        tgt = lang_map.get(target_lang.upper(), target_lang)
        src = lang_map.get((source_lang or "").upper(), "the source language")
        prompt = (
            f"Translate the following {src} text into {tgt}. "
            "Return only the translation, no commentary.\n\n"
            f"Text: {text}\n"
        )
        try:
            result = self.llm.invoke(prompt)  # type: ignore[attr-defined]
            return str(result).strip()
        except Exception as e:
            logger.warning("Local LLM translation failed: %s", e)
            return text

    def get_info(self) -> dict[str, str]:
        return {"provider": "llama.cpp", "model": Path(self.model_path).name}


__all__ = ["TranslationProvider", "DeepLAdapter", "LlamaCppTranslator"]
