"""Translation service adapters.

Provides adapters for external translation APIs (DeepL, etc.) and a local
LlamaCpp-based fallback that uses the configured GGUF model to translate
texts when DeepL is unavailable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


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
            raise RuntimeError(
                "deepl package not installed. "
                "Install with: poetry add deepl"
            )

        self.translator = self.deepl.Translator(api_key)
        logger.info("Initialized DeepL translator")

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

        try:
            result = self.translator.translate_text(
                text,
                target_lang=target_normalized,
                source_lang=source_normalized,
            )
            return str(result)

        except Exception as e:
            logger.error(f"DeepL translation failed: {e}")
            raise RuntimeError(f"Translation failed: {e}") from e

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
