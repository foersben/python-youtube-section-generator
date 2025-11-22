"""RAG system for transcript processing.

This module provides TranscriptRAG which indexes long transcripts into a
Chroma vector store and generates sections (flat or hierarchical). It uses
sentence-transformers for embeddings (CPU-only) and llama.cpp (via
LangChain's LlamaCpp) for refining main section titles using a local GGUF
model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add missing imports used later in the module
from src.core.embeddings.factory import EmbeddingsFactory
from src.core.utils.stopwords import get_stopwords

# Type for optional segmentation function imported from `src.core.retrieval.segmentation`
SegmentationFn = Callable[..., list[dict[str, Any]]]

select_coarse_sections: Optional[SegmentationFn] = None

logger = logging.getLogger(__name__)


class TranscriptRAG:
    """RAG system for processing long video transcripts.

    - Splits transcript text into chunks
    - Indexes chunks in ChromaDB with sentence-transformers embeddings
    - Generates main anchors and subsections
    - Refines main section titles using local llama.cpp model
    """

    def __init__(
        self,
        model_path: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temperature: float = 0.3,
    ) -> None:
        # Try to import langchain text splitter; provide a small local fallback
        # so RAG can still operate (with reduced features) when langchain
        # isn't installed or was upgraded with breaking changes.
        try:
            # Preferred import (works with most langchain releases)
            from langchain.text_splitter import (
                RecursiveCharacterTextSplitter,  # type: ignore
            )

        except Exception:
            # Try several alternative package layouts that changed across releases
            imported = False
            # 1) Newer split out package name used by some ecosystems
            try:
                from langchain_text_splitters import (
                    RecursiveCharacterTextSplitter,  # type: ignore
                )

                imported = True
            except Exception:
                pass

            # 2) Alternate singular package name
            if not imported:
                try:
                    from langchain_text_splitter import (
                        RecursiveCharacterTextSplitter,  # type: ignore
                    )

                    imported = True
                except Exception:
                    pass

            # 3) Fallback to CharacterTextSplitter under langchain (older names)
            if not imported:
                try:
                    from langchain.text_splitter import (
                        CharacterTextSplitter as _CTS,  # type: ignore
                    )

                    class _FallbackRecursiveCharacterTextSplitter(_CTS):  # type: ignore
                        """Thin compatibility wrapper mapping to CharacterTextSplitter API."""

                        pass

                    RecursiveCharacterTextSplitter = _FallbackRecursiveCharacterTextSplitter
                    imported = True
                except Exception:
                    imported = False

            if not imported:
                # Fallback: provide a minimal local splitter implementation
                logger.warning(
                    "langchain text splitter not available; using local fallback splitter. "
                    "To enable full RAG splitting features install a compatible text-splitter package, e.g. 'pip install langchain' or 'pip install langchain_text_splitters'."
                )

                class _FallbackLocalRecursiveCharacterTextSplitter:
                    """Minimal fallback splitter that creates overlapping chunks.

                    This implements a tiny subset of LangChain's API used in this
                    project: constructor signature and `create_documents(texts, metadatas)`.
                    It returns simple objects with `page_content` and `metadata`.
                    """

                    def __init__(
                        self,
                        chunk_size: int = 1000,
                        chunk_overlap: int = 200,
                        separators=None,
                        length_function=None,
                    ) -> None:
                        self.chunk_size = int(chunk_size)
                        self.chunk_overlap = int(chunk_overlap)

                    def _split_text(self, text: str) -> list[str]:
                        if not text:
                            return []
                        size = self.chunk_size
                        overlap = self.chunk_overlap
                        if size <= 0:
                            return [text]
                        chunks: list[str] = []
                        start = 0
                        L = len(text)
                        while start < L:
                            end = min(start + size, L)
                            chunks.append(text[start:end])
                            if end == L:
                                break
                            start = max(0, end - overlap)
                        return chunks

                    def create_documents(
                        self, texts: list[str], metadatas: list[dict[str, Any] | None] | None = None
                    ):
                        docs: list[object] = []
                        if metadatas is None:
                            metadatas = [None] * len(texts)
                        for text, meta in zip(texts, metadatas, strict=False):
                            for chunk in self._split_text(text):
                                # create a lightweight document-like object
                                docs.append(
                                    type("Doc", (), {"page_content": chunk, "metadata": meta})()
                                )
                        return docs

                RecursiveCharacterTextSplitter = _FallbackLocalRecursiveCharacterTextSplitter

        # Try to import LlamaCpp from langchain_community; if missing we let the
        # import error surface when attempting to use Llama functionality later.
        try:
            from langchain_community.llms import LlamaCpp  # type: ignore
        except Exception as e:
            # The local LLM provider is required for title refinement; fail early
            raise RuntimeError(
                "Required packages for local RAG LLM not installed: langchain-community (LlamaCpp). "
                "Install with: poetry add langchain langchain-community\n"
                f"Original error: {e}"
            ) from e

        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = max(0.0, min(1.0, temperature))
        self.detected_language = "en"  # Default, will be set during indexing

        # Model path resolution (GGUF)
        if model_path is None:
            model_path = os.getenv("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
        if not os.path.isabs(model_path):
            from src.core.config import config as _config

            model_path = str(Path(_config.project_root) / model_path)
        self.model_path = model_path
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Download with: ./scripts/download_model.sh"
            )

        logger.info("Initializing RAG system (model=%s)", Path(self.model_path).name)

        # Text splitter (may be a third-party or our local fallback)
        from typing import Any as _Any

        self.text_splitter: _Any = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        # Embeddings (CPU)
        logger.info("Loading embeddings (sentence-transformers/all-MiniLM-L6-v2) on CPU")
        self.embeddings = EmbeddingsFactory.create_provider(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )

        # Local LLM (llama.cpp via LangChain) - optimized for concise title generation
        import multiprocessing

        n_threads = multiprocessing.cpu_count()
        # Allow auto-detection of model n_ctx (trusting host RAM) when enabled
        auto_detect_ctx = os.getenv("LOCAL_MODEL_AUTO_DETECT_CTX", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        llm_kwargs: dict[str, object] = {
            "model_path": self.model_path,
            "n_threads": n_threads,
            "temperature": 0.05,
            "max_tokens": 64,
            "top_p": 0.85,
            "top_k": 20,
            "repeat_penalty": 1.1,
            "verbose": False,
        }
        if not auto_detect_ctx:
            # Keep legacy safe default if user does not want to auto-detect
            llm_kwargs["n_ctx"] = int(os.getenv("LOCAL_MODEL_N_CTX", "4096"))

        self.llm = LlamaCpp(**llm_kwargs)  # type: ignore[arg-type]

        # Concurrency limiter for LLM invocations to avoid spawning many
        # simultaneous heavy model calls that overload CPU and memory.
        max_concurrency = int(os.getenv("LOCAL_MODEL_MAX_CONCURRENCY", "1"))
        self._llm_semaphore = threading.BoundedSemaphore(value=max_concurrency)

        # Cache directory for title refinements
        self._title_cache_dir = Path(os.getenv("RAG_TITLE_CACHE_DIR", ".cache/rag_titles"))
        self._title_cache_dir.mkdir(parents=True, exist_ok=True)

        self.vectorstore: _Any = None
        self.chroma_client: Any | None = None
        self.current_video_id: str | None = None
        logger.info("✅ RAG system initialized")

    # ----------------------- Internal helpers -----------------------
    def _get_video_hash(self, video_id: str) -> str:
        # Prefer stability without type checker warnings
        return hashlib.md5(memoryview(video_id.encode("utf-8"))).hexdigest()[:12]

    def _create_transcript_text(self, transcript: list[dict[str, Any]]) -> str:
        lines = []
        for seg in transcript:
            timestamp = f"[{seg['start']:.1f}s]"
            lines.append(f"{timestamp} {seg.get('text', '').strip()}")
        return "\n".join(lines)

    def _aggregate_context(
        self,
        transcript: list[dict[str, Any]],
        ts: float,
        window: float = 60.0,
        max_chars: int = 2000,
    ) -> str:
        """Aggregate transcript text around a timestamp, plus vector hits."""
        nearby = [seg for seg in transcript if abs(seg.get("start", 0.0) - ts) <= window]
        if not nearby:
            nearby = [min(transcript, key=lambda seg: abs(seg.get("start", 0.0) - ts))]
        window_text = " ".join(str(s.get("text", "")) for s in nearby).strip()

        vec_text = ""
        if getattr(self, "vectorstore", None) is not None and window_text:
            try:
                query_text = window_text[:512]
                docs = self.vectorstore.similarity_search(query=query_text, k=3)
                vec_text = " ".join(getattr(d, "page_content", str(d)) for d in docs)
            except Exception:
                vec_text = ""

        combined = f"{window_text}\n{vec_text}".strip()
        combined = re.sub(r"\s+", " ", combined)
        return combined[:max_chars]

    def _safe_llm_invoke(self, prompt: str, timeout: int | None = None) -> str | None:
        """Invoke the local LLM with a timeout.

        Two modes are supported:
        - Process mode (recommended): spawn a child process that creates its own
          Llama instance and runs the prompt. If the child doesn't return within
          `timeout` seconds it is terminated. Enable with env var
          LOCAL_MODEL_USE_PROCESS=true (default: false).

        - Thread mode (legacy): run self.llm.invoke in a thread and wait with a
          timeout. This cannot kill underlying C work but avoids blocking the
          request thread; kept as fallback.

        Returns the raw string result or None on timeout/error.
        """
        use_process = os.getenv("LOCAL_MODEL_USE_PROCESS", "false").lower() in (
            "1",
            "true",
            "yes",
        )

        # Configure default timeout from environment if not provided
        if timeout is None:
            try:
                timeout = int(os.getenv("LOCAL_MODEL_LLM_TIMEOUT", "30"))
            except Exception:
                timeout = 30

        # Acquire concurrency semaphore to limit parallel LLM work
        acquired = self._llm_semaphore.acquire(timeout=0)
        if not acquired:
            # No slot available; wait a little and try to acquire with timeout
            logger.debug("LLM concurrency limit reached; waiting for a slot")
            if not self._llm_semaphore.acquire(timeout=timeout):
                logger.warning("LLM concurrency wait timed out after %ds", timeout)
                return None

        try:
            if use_process:
                # Use spawn context for better isolation
                ctx = mp.get_context("spawn")

                def _worker(pipe_conn, model_path, prompt_text):
                    """Child process target: create its own Llama and invoke prompt.

                    We import inside the child to avoid pickling the parent LLM.
                    """
                    try:
                        # Import in child
                        from llama_cpp import Llama

                        llm_child = Llama(model_path=model_path, verbose=False)
                        try:
                            res = llm_child.invoke(prompt_text)  # type: ignore[attr-defined]
                        except Exception as e:
                            pipe_conn.send({"error": str(e)})
                        else:
                            pipe_conn.send({"result": str(res)})
                    except Exception as e:
                        try:
                            pipe_conn.send({"error": f"Child error: {e}"})
                        except Exception:
                            pass
                    finally:
                        try:
                            pipe_conn.close()
                        except Exception:
                            pass

                parent_conn, child_conn = mp.Pipe()
                p = ctx.Process(target=_worker, args=(child_conn, self.model_path, prompt))
                p.start()
                try:
                    if parent_conn.poll(timeout):
                        payload = parent_conn.recv()
                        if not payload:
                            return None
                        if "result" in payload:
                            return str(payload["result"]) if payload["result"] is not None else None
                        # error case
                        logger.debug("LLM child returned error: %s", payload.get("error"))
                        return None
                    else:
                        logger.warning("LLM invoke timed out after %ds", timeout)
                        try:
                            p.terminate()
                        except Exception:
                            logger.debug("Failed to terminate child process", exc_info=True)
                        return None
                finally:
                    try:
                        parent_conn.close()
                    except Exception:
                        pass
                    p.join(timeout=1)
            else:
                # Thread-based fallback (cannot kill underlying C work)
                from concurrent.futures import ThreadPoolExecutor, TimeoutError

                def _call():
                    try:
                        return self.llm.invoke(prompt)  # type: ignore[attr-defined]
                    except Exception as e:
                        logger.debug("LLM invoke raised: %s", e, exc_info=True)
                        raise

                with ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_call)
                    try:
                        res = fut.result(timeout=timeout)
                        return str(res)
                    except TimeoutError:
                        logger.warning("LLM invoke timed out after %ds", timeout)
                        return None
        finally:
            try:
                self._llm_semaphore.release()
            except Exception:
                pass

    # ----------------------- Title cache helpers -----------------------
    def _title_cache_path(self, video_hash: str) -> Path:
        return self._title_cache_dir / f"{video_hash}.json"

    def _load_title_cache(self, video_hash: str) -> dict[str, str]:
        path = self._title_cache_path(video_hash)
        try:
            if not path.exists():
                return {}
            with path.open("r", encoding="utf-8") as f:
                data: Any = json.load(f)
            if not isinstance(data, dict):
                return {}
            cache: dict[str, str] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str):
                    cache[k] = v
            logger.debug("Loaded %d title cache entries for %s", len(cache), video_hash)
            return cache
        except Exception:
            logger.debug("Failed to load title cache", exc_info=True)
            return {}

    def _save_title_cache(self, video_hash: str, cache: dict[str, str]) -> None:
        path = self._title_cache_path(video_hash)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.debug("Saved %d title cache entries for %s", len(cache), video_hash)
        except Exception:
            logger.debug("Failed to save title cache", exc_info=True)

    def _title_cache_key(self, start_ts: float, context: str) -> str:
        payload = f"{start_ts:.1f}|{context[:512]}".encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _refine_title_with_llm_cached(
        self,
        video_id: str,
        start_ts: float,
        context_window: str,
    ) -> str | None:
        """Refine a title with LLM using a small disk cache.

        If the cached value exists it is returned without calling the LLM.
        On timeout or error, ``None`` is returned and callers are expected to
        fall back to heuristic titles.
        """
        video_hash = self._get_video_hash(video_id)
        cache = self._load_title_cache(video_hash)
        key = self._title_cache_key(start_ts, context_window)

        if key in cache:
            title = cache[key]
            logger.info(
                "[cache] Reusing cached title at %.1fs for video %s -> %s",
                start_ts,
                video_id,
                title,
            )
            return title

        prompt = (
            "You are an assistant that creates short, punchy section titles \n"
            "for long YouTube transcripts. Given the following context from a \n"
            "video transcript, return a concise 3-6 word title that captures \n"
            "the main idea. Do not include timestamps or quotes.\n\n"
            f"Context:\n{context_window}\n\nTitle:"
        )

        raw = self._safe_llm_invoke(prompt)
        if raw is None:
            logger.warning("LLM title refinement timed out or failed at %.1fs", start_ts)
            return None

        title = str(raw).strip().splitlines()[0].strip().strip("-:•·")
        if not title:
            return None

        cache[key] = title
        self._save_title_cache(video_hash, cache)
        logger.info("[llama.cpp] Main title refined at %.1fs -> %s", start_ts, title)
        return title

    # ----------------------- Indexing -----------------------
    def index_transcript(self, transcript: list[dict[str, Any]], video_id: str) -> None:
        from langchain_community.vectorstores import Chroma

        logger.info("Indexing transcript for video %s", video_id)
        logger.info("Total segments: %d", len(transcript))

        total_duration = max(seg["start"] + seg.get("duration", 0) for seg in transcript)
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        logger.info("Video duration: %dh %dm (%.0fs)", hours, minutes, total_duration)

        full_text = self._create_transcript_text(transcript)
        logger.info("Full text length: %s characters", f"{len(full_text):,}")

        # Detect transcript language for language-aware fallback
        try:
            from langdetect import detect

            sample_text = " ".join(seg.get("text", "") for seg in transcript[:50])
            self.detected_language = detect(sample_text) if sample_text.strip() else "en"
            logger.info("Detected transcript language: %s", self.detected_language)
        except Exception as e:
            logger.warning("Language detection failed: %s; defaulting to English", e)
            self.detected_language = "en"

        # create_documents signature differs between implementations; ignore strict typing here
        chunks = self.text_splitter.create_documents(  # type: ignore[call-arg]
            texts=[full_text],
            metadatas=[
                {
                    "video_id": video_id,
                    "total_segments": len(transcript),
                    "language": self.detected_language,
                }
            ],
        )
        logger.info("Created %d chunks for indexing", len(chunks))

        video_hash = self._get_video_hash(video_id)
        # Use configured Chroma persistence directory from AppConfig
        from src.core.config.config import AppConfig

        cfg = AppConfig()
        persist_directory = str(cfg.chromadb_dir / video_hash)

        logger.info("Creating vector store in %s", persist_directory)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=cast(Any, self.embeddings),
            persist_directory=persist_directory,
            collection_name=f"transcript_{video_hash}",
        )
        self.current_video_id = video_id
        logger.info("\u2705 Transcript indexed successfully")

    # ----------------------- Generation -----------------------
    def _clean_title_tokens(self, text: str, detected_lang: str | None = None) -> str:
        """Extract clean title using language-aware heuristics.

        Args:
            text: Text snippet to extract title from.
            detected_lang: Optional language code (e.g., 'de', 'en'). Auto-detected if None.

        Returns:
            Cleaned title string.
        """
        t = text.strip().strip("\"'`.,;:!?-–—")

        # Remove timestamps and numeric-only tokens
        t = re.sub(r"\b\d{1,2}:\d{2}\b", "", t)
        t = re.sub(r"\b\d{2,4}s\b", "", t)
        t = re.sub(r"\[\d+(:\d+)*s\]", "", t)

        # Detect language if not provided
        if detected_lang is None:
            try:
                from langdetect import detect

                detected_lang = detect(t)
            except Exception:
                detected_lang = "en"  # Default to English

        # Language-specific stopwords: use centralized utility
        try:
            stopwords_set = get_stopwords(detected_lang)
        except Exception:
            stopwords_set = set()

        # Tokenize and filter
        tokens = re.findall(r"\b[\wÄÖÜäöüß]+\b", t)
        filtered: list[str] = []
        for w in tokens:
            w_clean = w.strip().lower()
            if w_clean in stopwords_set:
                continue
            # Keep short words only if they look meaningful
            if len(w_clean) < 3:
                continue

            # Language-specific capitalization logic
            if detected_lang == "de":
                # German: All nouns are capitalized - strong signal for important words
                if w[0].isupper():
                    filtered.append(w_clean)
                elif len(filtered) < 1:  # Allow first word even if lowercase
                    filtered.append(w_clean)
            else:
                # English/other: Prefer capitalized (proper nouns), but allow lowercase if few words
                if w[0].isupper() or len(filtered) < 2:
                    filtered.append(w_clean)

            if len(filtered) >= 5:  # Limit to 5 words max
                break

        if not filtered:
            # Desperate fallback: extract any capitalized words
            alpha = re.findall(r"[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,3}", text)
            if alpha:
                return alpha[0].strip()
            # Last resort: get some alphabetic words
            alpha = re.findall(r"[A-Za-zÄÖÜäöüß]{4,}(?:\s+[A-Za-zÄÖÜäöüß]{4,}){0,3}", text)
            return alpha[0].strip() if alpha else "Section"

        title = " ".join(filtered[:5])  # Max 5 words

        # Capitalize first word if needed
        if title and not title[0].isupper():
            title = title[0].upper() + title[1:]

        return title if title else "Section"

    def _generate_flat_sections(
        self,
        transcript: list[dict[str, Any]],
        total_duration: float,
        num_sections: int,
        retrieval_k: int,
    ) -> list[dict[str, Any]]:
        if not transcript:
            return []
        try:
            interval = total_duration / (num_sections + 1)
        except Exception:
            interval = max((seg.get("start", 0) for seg in transcript), default=1.0)
        sample_times = [interval * (i + 1) for i in range(num_sections)]

        sections: list[dict[str, Any]] = []
        use_vectorstore = self.vectorstore is not None

        for idx, sample_time in enumerate(sample_times):
            logger.info(
                "Generating flat section %d/%d at ~%ds", idx + 1, num_sections, int(sample_time)
            )
            nearest = min(transcript, key=lambda s: abs(s.get("start", 0) - sample_time))
            candidate_title = self._clean_title_tokens(
                nearest.get("text", ""), self.detected_language
            )
            candidate_start = float(nearest.get("start", 0.0))

            if use_vectorstore:
                try:
                    docs = self.vectorstore.similarity_search(
                        query=f"Context around {int(sample_time)} seconds",
                        k=min(max(1, retrieval_k), 10),
                    )
                    if docs:
                        doc_text = getattr(docs[0], "page_content", None) or str(docs[0])
                        candidate_title = self._clean_title_tokens(doc_text, self.detected_language)
                except Exception:
                    logger.debug("Vectorstore retrieval failed", exc_info=True)

            sections.append({"title": candidate_title, "start": round(candidate_start, 1)})

        return sections

    def _find_semantic_boundaries(
        self, transcript: list[dict[str, Any]], distance_percentile: int = 25
    ) -> list[dict[str, Any]]:
        """
        Finds topic boundaries by calculating embedding similarity between
        adjacent transcript segments.

        Args:
            transcript: The full transcript.
            distance_percentile: The percentile of similarity scores to use
                as the "break" threshold. A lower value (e.g., 25) means
                we only break on very large, obvious topic shifts.

        Returns:
            A list of transcript segments that are the *start* of a new
            semantic section.
        """

        logger.info("Finding semantic boundaries...")
        if not transcript:
            return []

        # 1. Get texts and embeddings for all segments
        texts = [seg.get("text", "").strip() for seg in transcript]
        # Filter out empty segments which can break the chain
        valid_segments = [
            (i, seg) for i, (seg, text) in enumerate(zip(transcript, texts, strict=False)) if text
        ]
        if not valid_segments:
            logger.warning("No valid text segments found in transcript.")
            return []

        original_indices = [i for i, seg in valid_segments]
        valid_texts = [seg.get("text", "").strip() for i, seg in valid_segments]

        embeddings = self.embeddings.embed_documents(valid_texts)

        if len(embeddings) < 2:
            return [transcript[0]]  # Not enough text to compare

        # 2. Calculate cosine similarity between adjacent (N-1) segments
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)

        # 3. Find the "dips" (topic breaks)
        # We use a percentile as a threshold. Any similarity gap in the
        # bottom Nth percentile is a potential new section.
        threshold = np.percentile(similarities, distance_percentile)
        logger.info(f"Semantic similarity threshold (p{distance_percentile}): {threshold:.3f}")

        # Find indices where similarity drops below the threshold
        boundary_indices = [0]  # Always include the first segment
        for i, sim in enumerate(similarities):
            if sim < threshold:
                logger.info(f"Topic break detected at segment {i + 1} (Sim: {sim:.3f})")
                boundary_indices.append(i + 1)

        # De-duplicate and get the actual transcript segments
        unique_segment_indices = sorted(set(boundary_indices))

        # Map back to original transcript indices
        original_boundary_segments = [
            transcript[original_indices[idx]] for idx in unique_segment_indices
        ]

        logger.info(f"Found {len(original_boundary_segments)} semantic sections.")
        return original_boundary_segments

    def _generate_hierarchical_sections(
        self,
        transcript: list[dict[str, Any]],
        total_duration: float,
        main_sections_count: int,  # retained for compatibility, not strictly used
        subsections_per_main: int,
        retrieval_k: int,
    ) -> list[dict[str, Any]]:
        """Generate hierarchical sections using semantic boundaries and refined prompts."""
        # Prefer segmentation-based coarse selection when available and enabled.
        use_seg = os.getenv("RAG_USE_SEGMENTATION", "true").lower() in ("1", "true", "yes")
        main_sections: list[dict[str, Any]] = []
        if use_seg and select_coarse_sections is not None:
            try:
                segs = select_coarse_sections(
                    transcript,
                    target_sections=max(3, min(5, main_sections_count)),
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                # select_coarse_sections returns [{'start': float, 'title': str}, ...]
                if segs:
                    # Map coarse sections to nearest original transcript segment objects
                    for s in segs:
                        nearest = min(
                            transcript,
                            key=lambda sg: abs(sg.get("start", 0.0) - float(s.get("start", 0.0))),
                        )
                        main_sections.append(nearest)
            except Exception:
                logger.debug("Segmentation-based selection failed; falling back", exc_info=True)

        if not main_sections:
            main_sections = self._find_semantic_boundaries(transcript, distance_percentile=25)
        if not main_sections:
            return []

        sections: list[dict[str, Any]] = []
        use_llm_titles = os.getenv("USE_LLM_TITLES", "true").lower() == "true"
        vectorstore_available = getattr(self, "vectorstore", None) is not None

        main_sections = self._select_main_sections(
            main_sections, max(3, min(5, main_sections_count))
        )

        for idx, main_seg in enumerate(main_sections):
            main_start = float(main_seg.get("start", 0.0))
            context_window = self._aggregate_context(
                transcript, ts=main_start, window=90.0, max_chars=2000
            )

            main_title = self._clean_title_tokens(main_seg.get("text", ""), self.detected_language)
            if (
                use_llm_titles
                and vectorstore_available
                and context_window
                and self.current_video_id
            ):
                try:
                    cached = self._refine_title_with_llm_cached(
                        video_id=self.current_video_id,
                        start_ts=main_start,
                        context_window=context_window,
                    )
                    if cached is not None:
                        main_title = cached
                except Exception:
                    logger.warning(
                        "Main title refinement failed at %.1fs; using heuristic",
                        main_start,
                    )

            sections.append({"title": main_title, "start": main_start, "level": 0})

            # Determine chunk boundaries for subsections
            if idx + 1 < len(main_sections):
                next_start = float(main_sections[idx + 1].get("start", total_duration))
            else:
                next_start = total_duration

            window_start = main_start
            window_end = max(window_start + 1.0, next_start)

            if subsections_per_main <= 0:
                continue

            chunk_segments = [
                seg for seg in transcript if window_start <= seg.get("start", 0.0) < window_end
            ]

            subtopics = self._find_subtopics_in_chunk(chunk_segments, subsections_per_main)
            if not subtopics:
                subtopics = self._fallback_time_subsections(
                    transcript, window_start, window_end, subsections_per_main
                )

            for sub in subtopics:
                sub_start = float(sub.get("start", window_start))
                if sub_start <= window_start + 5.0:  # avoid duplicates right after main title
                    continue
                if sub_start > window_end:
                    sub_start = window_end
                sections.append(
                    {
                        "title": sub.get("title", "Section"),
                        "start": round(sub_start, 1),
                        "level": 1,
                    }
                )

        return sections

    def generate_sections(
        self,
        transcript: list[dict[str, Any]],
        video_id: str,
        num_sections: int = 10,
        retrieval_k: int = 5,
        hierarchical: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate sections using RAG approach with optional hierarchical structure.

        Args:
            transcript: Full transcript segments.
            video_id: Video identifier.
            num_sections: Target total number of sections (including subsections).
            retrieval_k: Number of chunks to retrieve per query.
            hierarchical: If True, creates main sections + subsections.

        Returns:
            List of section dicts with 'title', 'start', and optional 'level'.
        """
        if self.current_video_id != video_id or self.vectorstore is None:
            self.index_transcript(transcript, video_id)

        logger.info("Generating sections using RAG (hierarchical=%s)", hierarchical)

        total_duration = max(seg["start"] + seg.get("duration", 0) for seg in transcript)
        if hierarchical:
            main_sections_count = max(3, min(5, num_sections // 3))
            subsections_per_main = max(
                2, (num_sections - main_sections_count) // main_sections_count
            )
            sections = self._generate_hierarchical_sections(
                transcript, total_duration, main_sections_count, subsections_per_main, retrieval_k
            )
        else:
            sections = self._generate_flat_sections(
                transcript, total_duration, num_sections, retrieval_k
            )

        # Final cleanup
        clean_sections: list[dict[str, Any]] = []
        for s in sections:
            title = str(s.get("title", "")).strip().strip("\"'`.,;:!?-–—")
            if not title or len(title) < 3:
                title = "Section"
            start_val = s.get("start", 0.0)
            start_f = float(start_val or 0.0)
            clean_entry: dict[str, Any] = {"title": title, "start": start_f}
            if "level" in s:
                clean_entry["level"] = s["level"]
            clean_sections.append(clean_entry)

        return clean_sections

    # ----------------------- Cleanup -----------------------
    def cleanup(self) -> None:
        """Best-effort cleanup of heavy resources held by the RAG system.

        This method attempts to persist/close the vectorstore and chroma client
        if present, release the local LLM reference (calling .close() if
        available), and reset runtime state. All operations are non-fatal and
        failures are logged at DEBUG level.
        """
        logger.info("Cleaning up RAG resources...")
        try:
            # Vectorstore persist/close if available
            if getattr(self, "vectorstore", None) is not None:
                try:
                    if hasattr(self.vectorstore, "persist"):
                        try:
                            self.vectorstore.persist()
                        except Exception:
                            logger.debug("Vectorstore.persist() failed", exc_info=True)
                    if hasattr(self.vectorstore, "close"):
                        try:
                            self.vectorstore.close()
                        except Exception:
                            logger.debug("Vectorstore.close() failed", exc_info=True)
                except Exception:
                    logger.debug("Vectorstore cleanup failed", exc_info=True)

            # Chromadb client shutdown (best-effort)
            if getattr(self, "chroma_client", None) is not None:
                try:
                    cc = cast(Any, self.chroma_client)
                    if hasattr(cc, "close"):
                        try:
                            cc.close()
                        except Exception:
                            logger.debug("chroma_client.close() failed", exc_info=True)
                except Exception:
                    logger.debug("Chroma client cleanup failed", exc_info=True)

            # Release heavy resources and reset state
            try:
                self.vectorstore = None
            except Exception:
                logger.debug("Failed to clear vectorstore reference", exc_info=True)

            try:
                if getattr(self, "llm", None) is not None:
                    if hasattr(self.llm, "close"):
                        try:
                            self.llm.close()
                        except Exception:
                            logger.debug("llm.close() failed", exc_info=True)
                    self.llm = None
            except Exception:
                logger.debug("Failed to cleanup local LLM", exc_info=True)

            try:
                self.current_video_id = None
            except Exception:
                pass

        except Exception:
            logger.debug("RAG cleanup encountered an error", exc_info=True)

        logger.info("RAG cleanup complete")

    def _select_main_sections(
        self, sections: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        """Choose up to `limit` main sections from detected boundaries.

        Simple heuristic: if there are fewer than limit, return all. Otherwise
        evenly sample across the detected sections to produce `limit` entries.
        """
        if not sections:
            return []
        if len(sections) <= limit:
            return sections
        # Evenly pick indices
        step = max(1, len(sections) // limit)
        picked = [sections[i] for i in range(0, len(sections), step)][:limit]
        return picked

    def _find_subtopics_in_chunk(
        self, chunk_segments: list[dict[str, Any]], count: int
    ) -> list[dict[str, Any]]:
        """Extract simple subtopics from a list of transcript segments.

        This is a lightweight fallback that picks `count` candidate starts by
        sampling the chunk and generating heuristic titles.
        """
        if not chunk_segments:
            return []
        n = len(chunk_segments)
        if n <= count:
            candidates = chunk_segments
        else:
            step = max(1, n // count)
            candidates = [chunk_segments[i] for i in range(0, n, step)][:count]
        results: list[dict[str, Any]] = []
        for seg in candidates:
            t = str(seg.get("text", ""))
            title = self._clean_title_tokens(t, self.detected_language)
            results.append({"start": float(seg.get("start", 0.0)), "title": title})
        return results

    def _fallback_time_subsections(
        self, transcript: list[dict[str, Any]], window_start: float, window_end: float, count: int
    ) -> list[dict[str, Any]]:
        """Fallback that picks subsections evenly in time between window_start and window_end."""
        if window_end <= window_start:
            return []
        duration = window_end - window_start
        step = duration / (count + 1)
        starts = [window_start + step * (i + 1) for i in range(count)]
        results: list[dict[str, Any]] = []
        for s in starts:
            # pick nearest segment
            nearest = min(transcript, key=lambda sg: abs(sg.get("start", 0.0) - s))
            title = self._clean_title_tokens(str(nearest.get("text", "")), self.detected_language)
            results.append({"start": float(nearest.get("start", 0.0)), "title": title})
        return results
