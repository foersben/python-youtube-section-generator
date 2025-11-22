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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, cast

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
            target_lang: Target language code (None = auto-detect).

        Returns:
            Translated text.
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self) -> dict[str, str]:
        """Get provider information."""
        raise NotImplementedError


class GoogleTranslatorAdapter(TranslationProvider):
    """Adapter for deep-translator's GoogleTranslator.

    This adapter is intentionally lightweight and lazy-imports deep_translator at
    use time so it doesn't create a hard dependency for users who don't need it.
    """

    def __init__(self) -> None:
        try:
            # imported lazily to avoid hard dependency
            from deep_translator import GoogleTranslator  # type: ignore

            self._gt_cls = GoogleTranslator
            logger.info("GoogleTranslatorAdapter initialized")
        except Exception as e:
            logger.error("deep-translator not available: %s", e)
            raise RuntimeError("deep-translator (GoogleTranslator) not installed") from e

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
        if not text:
            return ""
        # map EN-US -> en etc.
        tgt = target_lang.split("-")[0].lower()
        src = source_lang.split("-")[0].lower() if source_lang else "auto"
        try:
            gt_cls = self._gt_cls
            if gt_cls is None:
                raise RuntimeError("deep-translator (GoogleTranslator) not available")
            translator = gt_cls(source=src, target=tgt)  # type: ignore[call-arg]
            return cast(Any, translator).translate(text)
        except Exception as e:
            logger.debug("GoogleTranslatorAdapter translate failed: %s", e)
            raise

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
        separator: str = "\n<<<SEG>>>\n",
    ) -> list[str]:
        if not texts:
            return []
        tgt = target_lang.split("-")[0].lower()
        src = source_lang.split("-")[0].lower() if source_lang else "auto"
        try:
            gt_cls = self._gt_cls
            if gt_cls is None:
                return []
            translator = gt_cls(source=src, target=tgt)  # type: ignore[call-arg]
            # deep-translator doesn't provide a bulk translate API that accepts list,
            # so we map sequentially. Use ThreadPoolExecutor for parallelism if many items.
            from concurrent.futures import ThreadPoolExecutor

            def _t(t: str) -> str:
                try:
                    return translator.translate(t)
                except Exception:
                    return ""

            with ThreadPoolExecutor(max_workers=min(8, len(texts))) as ex:
                return list(ex.map(_t, texts))
        except Exception as e:
            logger.debug("GoogleTranslatorAdapter batch failed: %s", e)
            raise

    def get_info(self) -> dict[str, str]:
        return {"provider": "google", "api": "deep-translator GoogleTranslator"}


class DeepLAdapter(TranslationProvider):
    """DeepL translation API adapter with graceful fallback to deep-translator (Google).

    Behavior:
    - If the `deepl` package is installed and usable, it will be used.
    - If `deepl` is not available or returns quota/server errors, the adapter will try
      to fall back to `deep-translator`'s GoogleTranslator (if installed).
    - Translation results are cached to `.cache/translations/deepl_cache.json`.
    - When a DeepL quota error is detected, a cooldown timestamp is persisted to
      `.cache/translations/deepl_quota.json` and DeepL calls are suppressed until expiry.
    """

    MAX_BATCH_CHARS = 10_000
    MAX_BATCH_SEGMENTS = 100

    def __init__(self, api_key: str | None):
        """Initialize the adapter (lazy).

        Args:
            api_key: DeepL API key (may be None if you only want fallback behavior).
        """
        # store API key but do not import deepl eagerly
        self._api_key = api_key
        # lazy imports and clients - annotate types to satisfy static checks
        self.deepl: Any | None = None
        self.translator: Any | None = None
        self._deepl_available: bool | None = None

        # deep-translator (Google) lazy availability flag
        self._deep_translator_available: bool | None = None
        self._gt_cls: Any | None = None

        # simple in-memory flag to avoid re-hitting DeepL after quota errors
        self._quota_disabled_until = 0
        # persisted files
        self._cache_dir = Path(_config.project_root) / ".cache" / "translations"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._quota_file = self._cache_dir / "deepl_quota.json"
        self._cache_file = self._cache_dir / "deepl_cache.json"

        # file lock for cache writing
        self._cache_lock: Any | None = None
        try:
            from filelock import FileLock

            self._cache_lock = FileLock(str(self._cache_file) + ".lock")
        except Exception:
            self._cache_lock = None

        # lazy cache memory
        self._cache: dict[str, dict] | None = None
        # precomputed batch cache for lists -> list[str]
        self._batch_cache: dict[str, list[str]] = {}

    def _ensure_deepl_client(self) -> bool:
        """Attempt to import and initialize the official `deepl` package.

        Returns True if available and initialized, False otherwise.
        """
        if self._deepl_available is not None:
            return bool(self._deepl_available)

        if not self._api_key:
            self._deepl_available = False
            return False

        try:
            import deepl  # type: ignore

            self.deepl = deepl
            try:
                self.translator = self.deepl.Translator(self._api_key)
            except Exception:
                # cannot instantiate translator -> mark unavailable
                logger.debug("deepl package present but Translator init failed", exc_info=True)
                self._deepl_available = False
                return False

            self._deepl_available = True
            logger.info("DeepL client initialized (lazy)")
            return True
        except Exception:
            logger.debug("deepl package not available; will use fallback if present", exc_info=True)
            self._deepl_available = False
            return False

    def _ensure_deep_translator(self) -> bool:
        """Ensure deep-translator's GoogleTranslator is importable and cached."""
        if self._deep_translator_available is not None:
            return bool(self._deep_translator_available)
        try:
            from deep_translator import GoogleTranslator  # type: ignore

            self._gt_cls = GoogleTranslator
            self._deep_translator_available = True
            logger.debug("deep-translator (GoogleTranslator) available")
            return True
        except Exception:
            logger.debug("deep-translator not available", exc_info=True)
            self._deep_translator_available = False
            return False

    def _deep_translate_single(
        self, text: str, target_lang: str, source_lang: str | None
    ) -> str | None:
        """Translate a single text using deep-translator Google fallback.

        Returns translated text or None if not available/failed.
        """
        if not self._ensure_deep_translator():
            return None
        # deep-translator GoogleTranslator expects simple 'en'/'de' codes
        tgt = target_lang.split("-")[0].lower()
        src = source_lang.split("-")[0].lower() if source_lang else "auto"
        try:
            gt_cls = self._gt_cls
            if gt_cls is None:
                return None
            translator = gt_cls(source=src, target=tgt)  # type: ignore[call-arg]
            return translator.translate(text)
        except Exception:
            logger.debug("deep-translator single translation failed", exc_info=True)
            return None

    def _deep_translate_batch(
        self, texts: list[str], target_lang: str, source_lang: str | None
    ) -> list[str] | None:
        """Translate a list of texts using deep-translator in parallel.

        Returns list[str] or None if deep-translator not available.
        """
        if not texts:
            return []
        if not self._ensure_deep_translator():
            return None
        tgt = target_lang.split("-")[0].lower()
        src = source_lang.split("-")[0].lower() if source_lang else "auto"
        try:
            gt_cls = self._gt_cls
            if gt_cls is None:
                return None
            translator = gt_cls(source=src, target=tgt)  # type: ignore[call-arg]

            def _t(t: str) -> str:
                try:
                    return translator.translate(t)
                except Exception:
                    return ""

            with ThreadPoolExecutor(max_workers=min(8, len(texts))) as ex:
                return list(ex.map(_t, texts))
        except Exception:
            logger.debug("deep-translator batch failed", exc_info=True)
            return None

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

    def _load_quota_state(self) -> int:
        if self._quota_file.exists():
            try:
                payload = read_json_file(str(self._quota_file)) or {}
                return int(payload.get("quota_disabled_until", 0))
            except Exception:
                logger.debug("Failed to read quota cache", exc_info=True)
        return 0

    def _persist_quota_state(self, disabled_until: int) -> None:
        data = {"quota_disabled_until": disabled_until, "ts": int(time.time())}
        try:
            write_to_file(data, str(self._quota_file))
        except Exception:
            logger.debug("Failed to persist DeepL quota state", exc_info=True)

    def _persist_quota_cooldown(self) -> None:
        cooldown = int(os.getenv("DEEPL_COOLDOWN_SECONDS", "3600"))
        disabled_until = int(time.time()) + cooldown
        self._quota_disabled_until = disabled_until
        self._persist_quota_state(disabled_until)

    def _cache_key(self, text: str, target: str, source: str | None) -> str:
        import hashlib

        src_str = str(source or "")
        target_str = str(target)
        text_str = str(text)
        key = hashlib.sha256((src_str + "::" + target_str + "::" + text_str).encode("utf-8"))
        return cast(str, key.hexdigest())

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
            # deep-translator fallback attempt if available
            fb = self._deep_translate_single(text, target_normalized, source_normalized)
            if fb is not None:
                return fb
            raise RuntimeError("DeepL temporarily disabled due to previous quota errors")

        # Try Deepl first (if available)
        if self._ensure_deepl_client():
            # retry loop + backoff is unchanged from original implementation
            max_retries = int(os.getenv("DEEPL_MAX_RETRIES", "5"))
            backoff_base = float(os.getenv("DEEPL_BACKOFF_BASE", "0.5"))
            backoff_max = float(os.getenv("DEEPL_BACKOFF_MAX", "8"))
            import random

            attempt = 0
            while True:
                try:
                    tr = self.translator
                    if tr is None:
                        raise RuntimeError("DeepL translator not initialized")
                    res = cast(Any, tr).translate_text(
                        text,
                        target_lang=target_normalized,
                        source_lang=source_normalized,
                    )
                    translated = str(res)
                    # cache and return
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
                    # deep-translator fallback
                    deep_fallback = self._deep_translate_single(
                        text, target_normalized, source_normalized
                    )
                    if deep_fallback is not None:
                        try:
                            cache = self._load_cache()
                            key = self._cache_key(text, target_normalized, source_normalized)
                            cache[key] = {"translated": deep_fallback, "ts": int(time.time())}
                            self._save_cache()
                        except Exception:
                            logger.debug("Failed to write deep-translator fallback to DeepL cache")
                        return deep_fallback
                    if "Quota" in msg or "Quota exceeded" in msg or "456" in msg:
                        self._persist_quota_cooldown()
                        raise TranslationQuotaExceeded(f"DeepL quota exceeded: {msg}") from e
                    if "Too many requests" in msg or "429" in msg or "rate limit" in msg.lower():
                        if attempt >= max_retries:
                            logger.error(
                                "DeepL retries exhausted (%d); last error: %s", attempt, msg
                            )
                            raise RuntimeError("DeepL rate limit / server busy: " + msg) from e
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
                    raise RuntimeError(f"Translation failed: {e}") from e

        # If deepl not available, fallback to deep-translator single
        fb = self._deep_translate_single(text, target_normalized, source_normalized)
        if fb is not None:
            # persist
            try:
                cache = self._load_cache()
                key = self._cache_key(text, target_normalized, source_normalized)
                cache[key] = {"translated": fb, "ts": int(time.time())}
                self._save_cache()
            except Exception:
                logger.debug("Failed to write deep-translator fallback to DeepL cache")
            return fb

        raise RuntimeError("No translation provider available")

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

        # quick cache lookup to skip already translated entries
        cache = self._load_cache()
        results: list[str] = [""] * len(texts)
        to_translate_idx: list[int] = []
        to_translate_texts: list[str] = []
        key_map: dict[int, str] = {}
        for i, t in enumerate(texts):
            try:
                k = self._cache_key(t, target_lang.upper(), (source_lang or None))
                key_map[i] = k
                ent = cache.get(k)
                if ent:
                    results[i] = ent.get("translated", "")
                else:
                    to_translate_idx.append(i)
                    to_translate_texts.append(t)
            except Exception:
                to_translate_idx.append(i)
                to_translate_texts.append(t)

        if not to_translate_texts:
            return results

        # Chunk to respect both MAX_BATCH_CHARS and MAX_BATCH_SEGMENTS
        batches: list[list[tuple[int, str]]] = []
        cur: list[tuple[int, str]] = []
        cur_chars = 0
        for idx, s in zip(to_translate_idx, to_translate_texts, strict=False):
            s_len = len(s)
            if (len(cur) >= self.MAX_BATCH_SEGMENTS) or (
                cur_chars + s_len > self.MAX_BATCH_CHARS and cur
            ):
                batches.append(cur)
                cur = []
                cur_chars = 0
            cur.append((idx, s))
            cur_chars += s_len
        if cur:
            batches.append(cur)

        # Try DeepL batch if available, otherwise deep-translator per-item
        translated_map: dict[int, str] = {}
        for batch in batches:
            idxs = [p[0] for p in batch]
            segs = [p[1] for p in batch]
            # try deepL joined translation first if available
            if self._ensure_deepl_client():
                joiner = separator
                joined = joiner.join(segs)
                try:
                    tr = self.translator
                    if tr is None:
                        raise RuntimeError("DeepL translator not initialized")
                    res = cast(Any, tr).translate_text(
                        joined, target_lang=target_lang.upper(), source_lang=(source_lang or None)
                    )
                    # split
                    out = str(res).split(joiner)
                    if len(out) == len(segs):
                        for i, o in zip(idxs, out, strict=False):
                            translated_map[i] = o
                            # persist
                            try:
                                k = self._cache_key(
                                    texts[i], target_lang.upper(), (source_lang or None)
                                )
                                cache[k] = {"translated": o, "ts": int(time.time())}
                            except Exception:
                                logger.debug("Failed to cache DeepL batch result")
                    else:
                        # fallback to deep-translator for this batch
                        fb_out = self._deep_translate_batch(segs, target_lang, source_lang)
                        if fb_out is None:
                            raise RuntimeError("No fallback translator available for batch")
                        for i, o in zip(idxs, fb_out, strict=False):
                            translated_map[i] = o
                            try:
                                k = self._cache_key(
                                    texts[i], target_lang.upper(), (source_lang or None)
                                )
                                cache[k] = {"translated": o, "ts": int(time.time())}
                            except Exception:
                                logger.debug("Failed to cache deep-translator batch result")
                except Exception as e:
                    logger.debug("DeepL batch failed, trying fallback: %s", e)
                    fb_out = self._deep_translate_batch(segs, target_lang, source_lang)
                    if fb_out is None:
                        # mark quota if message contains quota words
                        msg = str(e)
                        if "Quota" in msg or "Quota exceeded" in msg or "456" in msg:
                            self._persist_quota_cooldown()
                            raise TranslationQuotaExceeded(msg) from e
                        raise
                    for i, o in zip(idxs, fb_out, strict=False):
                        translated_map[i] = o
                        try:
                            k = self._cache_key(
                                texts[i], target_lang.upper(), (source_lang or None)
                            )
                            cache[k] = {"translated": o, "ts": int(time.time())}
                        except Exception:
                            logger.debug("Failed to cache deep-translator fallback result")
            else:
                fb_out = self._deep_translate_batch(segs, target_lang, source_lang)
                if fb_out is None:
                    raise RuntimeError("No translation provider available for batch")
                for i, o in zip(idxs, fb_out, strict=False):
                    translated_map[i] = o
                    try:
                        k = self._cache_key(texts[i], target_lang.upper(), (source_lang or None))
                        cache[k] = {"translated": o, "ts": int(time.time())}
                    except Exception:
                        logger.debug("Failed to cache deep-translator batch result")

        # persist cache once per call
        try:
            self._save_cache()
        except Exception:
            logger.debug("Failed to persist cache after batch translation")

        for i in range(len(texts)):
            if results[i]:
                continue
            results[i] = translated_map.get(i, "")

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

    _cache_dir = Path(_config.project_root) / ".cache" / "local_llm"
    _cache_dir.mkdir(parents=True, exist_ok=True)

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
        self._translation_cache_path = self._cache_dir / "translation_cache.json"
        self._title_cache_path = self._cache_dir / "title_cache.json"
        self._cache_lock = None
        try:
            from filelock import FileLock

            self._cache_lock = FileLock(str(self._translation_cache_path) + ".lock")
        except Exception:
            self._cache_lock = None
        self._translation_cache: dict[str, Any] = {}

    def _load_translation_cache(self) -> dict[str, Any]:
        if self._translation_cache:
            return self._translation_cache
        try:
            if self._translation_cache_path.exists():
                self._translation_cache = read_json_file(str(self._translation_cache_path)) or {}
        except Exception:
            logger.debug("Failed to read local LLM translation cache", exc_info=True)
            self._translation_cache = {}
        return self._translation_cache

    def _save_translation_cache(self) -> None:
        if not self._translation_cache:
            return
        payload = self._translation_cache
        try:
            if self._cache_lock:
                with self._cache_lock:
                    write_to_file(payload, str(self._translation_cache_path))
            else:
                write_to_file(payload, str(self._translation_cache_path))
        except Exception:
            logger.debug("Failed to persist local LLM translation cache", exc_info=True)

    def _translation_cache_key(self, text: str, target: str, source: str | None) -> str:
        import hashlib

        src = str(source or "")
        tgt = target.upper()
        key = hashlib.sha256(f"{src}::{tgt}::{text}".encode("utf-8")).hexdigest()
        return key

    def translate(self, text: str, target_lang: str, source_lang: str | None = None) -> str:
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
        cache_ttl = int(os.getenv("LOCAL_LLM_TRANSLATION_CACHE_TTL", str(7 * 24 * 3600)))
        cache = self._load_translation_cache()
        key = self._translation_cache_key(text, target_lang, source_lang)
        if cache_ttl > 0:
            entry = cache.get(key)
            if entry:
                ts = int(entry.get("ts", 0))
                if ts and (int(time.time()) - ts) <= cache_ttl:
                    return entry.get("translated", text)

        prompt = (
            f"Translate the following {src} text into {tgt}. "
            "Return only the translation, no commentary.\n\n"
            f"Text: {text}\n"
        )
        # Run the local LLM translation with a timeout to avoid blocking the
        # web request indefinitely. Timeout is configurable via
        # LOCAL_MODEL_TRANSLATION_TIMEOUT (seconds).
        timeout = float(os.getenv("LOCAL_MODEL_TRANSLATION_TIMEOUT", "300"))

        def _invoke() -> str:
            try:
                res = self.llm.invoke(prompt)  # type: ignore[attr-defined]
                return str(res).strip()
            except Exception as e:  # pragma: no cover - runtime guard
                logger.exception("Local LLM invoke error: %s", e)
                raise

        # Use a short-lived thread pool to enforce the timeout.
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_invoke)
                translated = fut.result(timeout=timeout)
                cache[key] = {"translated": translated, "ts": int(time.time())}
                self._save_translation_cache()
                return translated
        except FuturesTimeoutError:
            logger.warning(
                "Local LLM translation timed out after %.1fs; returning original text",
                timeout,
            )
            return text
        except Exception as e:  # pragma: no cover - fallback behaviour
            logger.warning("Local LLM translation failed: %s", e)
            return text

    def get_info(self) -> dict[str, str]:
        return {"provider": "llama.cpp", "model": Path(self.model_path).name}

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
        separator: str = "\n<<<SEG>>>\n",
    ) -> list[str]:
        """Batch translate using local LlamaCpp by joining texts with a separator.

        This reduces the overhead of invoking the model per segment and lets the
        model translate multiple segments in a single prompt. The underlying
        translate() call already enforces a timeout.
        """
        if not texts:
            return []

        MAX_CHARS = 10_000
        MAX_SEGMENTS = 100

        # Load local LLM translation cache and prepare TTL
        cache = self._load_translation_cache()
        ttl = int(os.getenv("LOCAL_LLM_TRANSLATION_CACHE_TTL", str(7 * 24 * 3600)))
        n = len(texts)
        results: list[str] = [""] * n

        # Pre-fill results from cache when available and not expired
        cached_indices: set[int] = set()
        now_ts = int(time.time())
        for idx, t in enumerate(texts):
            try:
                k = self._translation_cache_key(t, target_lang, source_lang)
                ent = cache.get(k)
                if ent:
                    ts = int(ent.get("ts", 0))
                    if ttl <= 0 or (now_ts - ts) <= ttl:
                        results[idx] = ent.get("translated", "")
                        cached_indices.add(idx)
            except Exception:
                # On any cache error, treat as cache miss
                continue

        i = 0
        while i < n:
            # Advance i to the next index that isn't already cached
            while i < n and i in cached_indices:
                i += 1
            if i >= n:
                break

            j = i
            chars = 0
            cnt = 0
            batch_indices: list[int] = []
            while j < n and cnt < MAX_SEGMENTS:
                # Skip already-cached indices
                if j in cached_indices:
                    j += 1
                    continue
                t = texts[j] or ""
                tlen = len(t)
                if cnt > 0 and (chars + tlen) > MAX_CHARS:
                    break
                batch_indices.append(j)
                chars += tlen
                cnt += 1
                j += 1

            if not batch_indices:
                batch_indices = [i]
                j = i + 1

            batch_texts = [texts[k] for k in batch_indices]
            joined = separator.join(batch_texts)
            try:
                translated_joined = self.translate(joined, target_lang, source_lang)
            except Exception as e:
                logger.warning("Llama batch translate failed, falling back to per-item: %s", e)
                # Fallback: translate individually
                for t in batch_texts:
                    results.append(self.translate(t, target_lang, source_lang))
                i = j
                continue

            parts = translated_joined.split(separator)
            if len(parts) != len(batch_texts):
                logger.warning(
                    "Llama batch translation returned %d parts for %d inputs; falling back to per-segment translate",
                    len(parts),
                    len(batch_texts),
                )
                for t in batch_texts:
                    results.append(self.translate(t, target_lang, source_lang))
            else:
                for part in parts:
                    results.append(part)

            i = j

        return results


__all__ = ["TranslationProvider", "DeepLAdapter", "LlamaCppTranslator"]
