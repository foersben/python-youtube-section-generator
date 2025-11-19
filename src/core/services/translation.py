"""Translation service adapters.

Provides adapters for external translation APIs (DeepL, etc.) and a local
LlamaCpp-based fallback that uses the configured GGUF model to translate
texts when DeepL is unavailable.
"""

from __future__ import annotations

import logging
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
            import deepl
        except ImportError as err:
            raise RuntimeError(
                "deepl package not installed. Install with: poetry add deepl"
            ) from err

        self.deepl = deepl
        self.translator = self.deepl.Translator(api_key)
        logger.info("Initialized DeepL translator")

        # simple in-memory flag to avoid re-hitting DeepL after quota errors
        # store timestamp in environment to persist across processes if desired
        self._quota_disabled_until = 0
        # persisted quota file location
        self._cache_dir = Path(_config.project_root) / ".cache" / "translations"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._quota_file = self._cache_dir / "deepl_quota.json"

        # persistent translation cache
        self._cache_file = self._cache_dir / "deepl_cache.json"
        # use filelock to avoid concurrent writers
        try:
            from filelock import FileLock

            self._cache_lock = FileLock(str(self._cache_file) + ".lock")
        except Exception:
            self._cache_lock = None

        # load cache lazily
        self._cache: dict[str, dict] | None = None

    def _load_cache(self) -> dict[str, dict]:
        if self._cache is not None:
            return self._cache
        data: dict[str, dict] = {}
        try:
            if self._cache_file.exists():
                data = read_json_file(str(self._cache_file)) or {}
        except Exception:
            logger.debug("Failed to read DeepL cache file; starting fresh")
            data = {}
        self._cache = data
        return data

    def _save_cache(self) -> None:
        if self._cache is None:
            return
        try:
            if self._cache_lock is not None:
                with self._cache_lock:
                    write_to_file(self._cache, str(self._cache_file))
            else:
                write_to_file(self._cache, str(self._cache_file))
        except Exception:
            logger.exception("Failed to persist DeepL translation cache")

    def _cache_key(self, text: str, target: str, source: str | None) -> str:
        import hashlib

        src = source or ""
        key = hashlib.sha256((src + "::" + target + "::" + text).encode("utf-8"))
        return key.hexdigest()

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        """Translate text using DeepL with cache lookup.

        Args:
            text: Text to translate.
            target_lang: Target language code (e.g., 'EN-US', 'EN-GB', 'DE', 'FR').
            source_lang: Source language code (None = auto-detect).

        Returns:
            Translated text.

        Raises:
            RuntimeError: If translation fails.
        """
        # Short-circuit empty
        if not text:
            return ""

        # Normalize English variants for DeepL API compatibility
        target_normalized = target_lang.upper()
        if target_normalized == "EN":
            target_normalized = "EN-US"  # Default to US English

        source_normalized = None
        if source_lang:
            source_normalized = source_lang.upper()
            if source_normalized == "EN":
                source_normalized = "EN-US"

        # check cache
        try:
            cache = self._load_cache()
            key = self._cache_key(text, target_normalized, source_normalized)
            entry = cache.get(key)
            if entry:
                # Optional TTL support
                ttl = int(os.getenv("TRANSLATION_CACHE_TTL_SECONDS", str(7 * 24 * 3600)))
                ts = int(entry.get("ts", 0))
                if ttl <= 0 or (int(time.time()) - ts) <= ttl:
                    logger.debug("DeepL cache hit for key %s", key[:8])
                    return entry.get("translated", "")
                else:
                    # expired
                    logger.debug("DeepL cache expired for key %s", key[:8])
                    cache.pop(key, None)
        except Exception:
            logger.debug("DeepL cache lookup failed; proceeding to API")

        # respect short-circuit if quota hit recently
        # 'time' is imported at module level; do not re-import here to avoid shadowing

        # reload persisted quota if present
        try:
            if self._quota_file.exists():
                q = read_json_file(str(self._quota_file))
                self._quota_disabled_until = int(q.get("quota_disabled_until", 0))
        except Exception:
            pass

        if self._quota_disabled_until and time.time() < self._quota_disabled_until:
            raise RuntimeError("DeepL temporarily disabled due to previous quota errors")

        # Exponential backoff retry parameters
        max_retries = int(os.getenv("DEEPL_MAX_RETRIES", "5"))
        backoff_base = float(os.getenv("DEEPL_BACKOFF_BASE", "0.5"))
        backoff_max = float(os.getenv("DEEPL_BACKOFF_MAX", "8"))

        import random

        attempt = 0
        while True:
            try:
                result = self.translator.translate_text(
                    text,
                    target_lang=target_normalized,
                    source_lang=source_normalized,
                )
                translated = str(result)

                # persist to cache
                try:
                    cache = self._load_cache()
                    key = self._cache_key(text, target_normalized, source_normalized)
                    cache[key] = {"translated": translated, "ts": int(time.time())}
                    self._save_cache()
                except Exception:
                    logger.debug("Failed to write to DeepL cache")

                return translated

            except Exception as e:
                msg = str(e)
                logger.error(f"DeepL translation failed (attempt {attempt}): {e}")
                # Detect quota exceeded messages and set a short disable period
                if "Quota" in msg or "Quota exceeded" in msg or "456" in msg:
                    # determine cooldown from env or default to 1 hour
                    cooldown = int(os.getenv("DEEPL_COOLDOWN_SECONDS", "3600"))
                    self._quota_disabled_until = int(time.time()) + cooldown
                    # persist
                    try:
                        write_to_file(
                            {"quota_disabled_until": self._quota_disabled_until},
                            str(self._quota_file),
                        )
                    except Exception:
                        logger.debug("Failed to persist DeepL quota file")
                    raise TranslationQuotaExceeded(f"DeepL quota exceeded: {msg}") from e

                # Rate limit / transient server busy (retryable)
                if "Too many requests" in msg or "429" in msg or "rate limit" in msg.lower():
                    if attempt >= max_retries:
                        logger.error("DeepL retries exhausted (%d); last error: %s", attempt, msg)
                        raise RuntimeError("DeepL rate limit / server busy: " + msg) from e
                    # backoff with jitter
                    backoff = min(backoff_max, backoff_base * (2**attempt))
                    jitter = random.uniform(0, backoff * 0.1)
                    sleep_for = backoff + jitter
                    logger.info(
                        "DeepL transient error, retrying in %.2fs (attempt %d/%d)",
                        sleep_for,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(sleep_for)
                    attempt += 1
                    continue

                # Non-retryable error
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
        quota_file = self._quota_file

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

        # load cache once
        cache = self._load_cache()

        # Determine which items need translation and which are cached
        to_translate_indices: list[int] = []
        cached_map: dict[int, str] = {}
        for idx, txt in enumerate(texts):
            try:
                key = self._cache_key(
                    txt, target_lang.upper(), (source_lang or "").upper() if source_lang else None
                )
                entry = cache.get(key)
                if entry:
                    ttl = int(os.getenv("TRANSLATION_CACHE_TTL_SECONDS", str(7 * 24 * 3600)))
                    ts = int(entry.get("ts", 0))
                    if ttl <= 0 or (int(time.time()) - ts) <= ttl:
                        cached_map[idx] = entry.get("translated", "")
                        continue
                    else:
                        # expired
                        cache.pop(key, None)
                to_translate_indices.append(idx)
            except Exception:
                to_translate_indices.append(idx)

        # If everything cached, return in order
        if not to_translate_indices:
            return [cached_map[i] for i in range(len(texts))]

        # Build batches over the indices needing translation
        i = 0
        indices = to_translate_indices
        n = len(indices)
        while i < n:
            j = i
            chars = 0
            cnt = 0
            batch_indices: list[int] = []
            while j < n and cnt < MAX_SEGMENTS:
                idx = indices[j]
                t = texts[idx] or ""
                tlen = len(t)
                if cnt > 0 and (chars + tlen) > MAX_CHARS:
                    break
                batch_indices.append(idx)
                chars += tlen
                cnt += 1
                j += 1

            # ensure at least one
            if not batch_indices:
                batch_indices = [indices[i]]
                j = i + 1

            batch_texts = [texts[k] for k in batch_indices]
            joined = separator.join(batch_texts)
            try:
                translated_joined = self.translate(joined, target_lang, source_lang)
            except TranslationQuotaExceeded:
                # persist the quota disable and re-raise
                cooldown = int(os.getenv("DEEPL_COOLDOWN_SECONDS", "3600"))
                quota_disabled_until = int(time.time()) + cooldown
                try:
                    write_to_file({"quota_disabled_until": quota_disabled_until}, str(quota_file))
                except Exception:
                    logger.debug("Failed to persist DeepL quota file")
                raise

            # split back
            parts = translated_joined.split(separator)
            if len(parts) != len(batch_texts):
                logger.warning(
                    "DeepL batch translation returned %d parts for %d inputs; falling back to per-segment translate for that batch",
                    len(parts),
                    len(batch_texts),
                )
                for k in batch_indices:
                    translated_single = self.translate(texts[k], target_lang, source_lang)
                    cache[
                        self._cache_key(
                            texts[k],
                            target_lang.upper(),
                            (source_lang or "").upper() if source_lang else None,
                        )
                    ] = {
                        "translated": translated_single,
                        "ts": int(time.time()),
                    }
                    results.append(translated_single)
            else:
                for off, k in enumerate(batch_indices):
                    translated_part = parts[off]
                    cache[
                        self._cache_key(
                            texts[k],
                            target_lang.upper(),
                            (source_lang or "").upper() if source_lang else None,
                        )
                    ] = {
                        "translated": translated_part,
                        "ts": int(time.time()),
                    }
                    results.append(translated_part)

            # advance
            i = j

        # fill in cached results for other indices in order
        final: list[str] = []
        for idx in range(len(texts)):
            if idx in cached_map:
                final.append(cached_map[idx])
            else:
                # pop from results in the same sequence
                final.append(results.pop(0))

        # persist cache
        try:
            self._save_cache()
        except Exception:
            logger.debug("Failed to persist DeepL cache after batch")

        return final

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

        # Allow environment overrides for n_ctx and n_batch to utilize larger RAM
        n_ctx = int(os.getenv("LOCAL_MODEL_N_CTX", "4096"))
        n_batch = int(os.getenv("LOCAL_MODEL_N_BATCH", "512"))

        # enforce sensible minimums to avoid llama.cpp warnings
        min_batch = int(os.getenv("LOCAL_MODEL_MIN_BATCH", "64"))
        if n_batch < min_batch:
            logger.info(
                "LOCAL_MODEL_N_BATCH (%d) is less than minimum required (%d); increasing to %d",
                n_batch,
                min_batch,
                min_batch,
            )
            n_batch = min_batch

        # Optional auto-detect of model n_ctx_train to fully utilize model capacity
        auto_detect = os.getenv("LOCAL_MODEL_AUTO_DETECT_CTX", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        detected_n_ctx: int | None = None
        if auto_detect:
            try:
                # try to use llama-cpp-python directly to inspect the model
                from llama_cpp import Llama as LlamaCPP  # type: ignore

                logger.debug("Probing model for n_ctx_train using llama_cpp")
                try:
                    probe = LlamaCPP(
                        model_path=self.model_path, n_ctx=1, n_threads=1, verbose=False
                    )
                    # llama_cpp exposes n_ctx_train via method or attribute
                    if hasattr(probe, "n_ctx_train"):
                        try:
                            detected_n_ctx = int(probe.n_ctx_train())
                        except Exception:
                            detected_n_ctx = None
                    elif hasattr(probe, "_model") and hasattr(probe._model, "n_ctx_train"):
                        try:
                            detected_n_ctx = int(probe._model.n_ctx_train())
                        except Exception:
                            detected_n_ctx = None
                finally:
                    try:
                        # ensure we free resources
                        del probe
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("Auto-detect of model n_ctx_train failed: %s", e)

        # If auto-detect succeeded, decide whether to adopt the model's trained n_ctx.
        # Soft-cap heuristic controls:
        # - LOCAL_MODEL_MAX_CTX: if >0, cap detected_n_ctx to this value
        # - LOCAL_MODEL_FORCE_ADOPT_CTX: if true, force adoption regardless of RAM
        # - LOCAL_MODEL_RAM_MULTIPLE: multiple of model file size to require as free RAM (default 1.5)
        if detected_n_ctx:
            try:
                max_ctx_env = int(os.getenv("LOCAL_MODEL_MAX_CTX", "0"))
            except Exception:
                max_ctx_env = 0
            force_adopt = os.getenv("LOCAL_MODEL_FORCE_ADOPT_CTX", "false").lower() in (
                "1",
                "true",
                "yes",
            )

            # compute model file size and available RAM
            model_size_bytes = 0
            try:
                model_size_bytes = int(Path(self.model_path).stat().st_size)
            except Exception:
                model_size_bytes = 0

            available_ram = 0
            try:
                import psutil

                available_ram = int(psutil.virtual_memory().available)
            except Exception:
                # fallback to /proc/meminfo on Linux
                try:
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if line.startswith("MemAvailable:"):
                                parts = line.split()
                                available_ram = int(parts[1]) * 1024
                                break
                except Exception:
                    available_ram = 0

            ram_multiple = float(os.getenv("LOCAL_MODEL_RAM_MULTIPLE", "1.5"))

            # decide adoption
            adopt_ctx = detected_n_ctx
            if max_ctx_env and max_ctx_env > 0:
                if detected_n_ctx > max_ctx_env:
                    logger.info(
                        "Detected n_ctx_train=%d capped by LOCAL_MODEL_MAX_CTX=%d",
                        detected_n_ctx,
                        max_ctx_env,
                    )
                adopt_ctx = min(detected_n_ctx, max_ctx_env)

            if not force_adopt and model_size_bytes > 0 and available_ram > 0:
                required = int(model_size_bytes * ram_multiple)
                if available_ram < required:
                    # Not enough available RAM to safely adopt full context
                    logger.warning(
                        "Insufficient free RAM to adopt detected n_ctx_train=%d (model_size=%d bytes, available_ram=%d bytes, required=%d). Keeping configured n_ctx=%d. Set LOCAL_MODEL_FORCE_ADOPT_CTX=true to override or increase system RAM.",
                        detected_n_ctx,
                        model_size_bytes,
                        available_ram,
                        required,
                        n_ctx,
                    )
                    # do not adopt
                    adopt_ctx = n_ctx

            # If adoption changes n_ctx, log and set
            if adopt_ctx != n_ctx:
                logger.info(
                    "Setting runtime n_ctx to %d (detected %d, configured %d)",
                    adopt_ctx,
                    detected_n_ctx,
                    n_ctx,
                )
                n_ctx = adopt_ctx

        # create LlamaCpp with the computed parameters
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_threads=multiprocessing.cpu_count(),
            n_batch=n_batch,
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
