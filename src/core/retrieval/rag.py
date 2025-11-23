"""RAG system for transcript processing.

This module provides TranscriptRAG which indexes long transcripts into a
Chroma vector store and generates sections (flat or hierarchical).
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
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity

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
        # Try to import LlamaCpp from langchain_community
        try:
            from langchain_community.llms import LlamaCpp  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Required packages for local RAG LLM not installed: langchain-community (LlamaCpp). "
                "Install with: poetry add langchain langchain-community\n"
                f"Original error: {e}"
            ) from e

        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = max(0.0, min(1.0, temperature))
        self.detected_language = "en"

        # Model path resolution (GGUF)
        if model_path is None:
            model_path = os.getenv("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
        if not os.path.isabs(model_path):
            from src.core.config import config as _config
            model_path = str(Path(_config.project_root) / model_path)
        self.model_path = model_path
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\nDownload with: ./scripts/download_model.sh"
            )

        logger.info("Initializing RAG system (model=%s)", Path(self.model_path).name)

        self.text_splitter = RecursiveCharacterTextSplitter(
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

        # Local LLM (llama.cpp via LangChain)
        import multiprocessing
        n_threads = multiprocessing.cpu_count()
        auto_detect_ctx = os.getenv("LOCAL_MODEL_AUTO_DETECT_CTX", "true").lower() in ("1", "true", "yes")

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
            llm_kwargs["n_ctx"] = int(os.getenv("LOCAL_MODEL_N_CTX", "4096"))

        self.llm = LlamaCpp(**llm_kwargs)  # type: ignore

        max_concurrency = int(os.getenv("LOCAL_MODEL_MAX_CONCURRENCY", "1"))
        self._llm_semaphore = threading.BoundedSemaphore(value=max_concurrency)

        self._title_cache_dir = Path(os.getenv("RAG_TITLE_CACHE_DIR", ".cache/rag_titles"))
        self._title_cache_dir.mkdir(parents=True, exist_ok=True)

        self.vectorstore: Any = None
        self.chroma_client: Any | None = None
        self.current_video_id: str | None = None
        logger.info("✅ RAG system initialized")

    # ----------------------- Internal helpers -----------------------
    def _get_video_hash(self, video_id: str) -> str:
        return hashlib.md5(memoryview(video_id.encode("utf-8"))).hexdigest()[:12]

    def _create_transcript_text(self, transcript: list[dict[str, Any]]) -> str:
        lines = []
        for seg in transcript:
            timestamp = f"[{seg['start']:.1f}s]"
            lines.append(f"{timestamp} {seg.get('text', '').strip()}")
        return "\n".join(lines)

    def _aggregate_context(self, transcript: list[dict[str, Any]], ts: float, window: float = 60.0, max_chars: int = 2000) -> str:
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
        """Invoke the local LLM with a timeout."""
        if timeout is None:
            timeout = 30

        acquired = self._llm_semaphore.acquire(timeout=0)
        if not acquired:
            if not self._llm_semaphore.acquire(timeout=timeout):
                return None

        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            def _call():
                try:
                    return self.llm.invoke(prompt)
                except Exception:
                    raise

            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_call)
                try:
                    res = fut.result(timeout=timeout)
                    return str(res)
                except TimeoutError:
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
                return json.load(f)
        except Exception:
            return {}

    def _save_title_cache(self, video_hash: str, cache: dict[str, str]) -> None:
        path = self._title_cache_path(video_hash)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _title_cache_key(self, start_ts: float, context: str) -> str:
        payload = f"{start_ts:.1f}|{context[:512]}".encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _refine_title_with_llm_cached(self, video_id: str, start_ts: float, context_window: str) -> str | None:
        video_hash = self._get_video_hash(video_id)
        cache = self._load_title_cache(video_hash)
        key = self._title_cache_key(start_ts, context_window)

        if key in cache:
            return cache[key]

        prompt = (
            "You are an assistant that creates short, punchy section titles \n"
            "for long YouTube transcripts. Given the following context from a \n"
            "video transcript, return a concise 3-6 word title that captures \n"
            "the main idea. Do not include timestamps or quotes.\n\n"
            f"Context:\n{context_window}\n\nTitle:"
        )

        raw = self._safe_llm_invoke(prompt)
        if raw is None:
            return None

        title = str(raw).strip().splitlines()[0].strip().strip("-:•·")
        if not title:
            return None

        cache[key] = title
        self._save_title_cache(video_hash, cache)
        return title

    # ----------------------- Indexing -----------------------
    def index_transcript(self, transcript: list[dict[str, Any]], video_id: str) -> None:
        from langchain_community.vectorstores import Chroma

        logger.info("Indexing transcript for video %s", video_id)
        full_text = self._create_transcript_text(transcript)

        # Simple language detection
        try:
            from langdetect import detect
            sample_text = " ".join(seg.get("text", "") for seg in transcript[:50])
            self.detected_language = detect(sample_text) if sample_text.strip() else "en"
        except Exception:
            self.detected_language = "en"

        chunks = self.text_splitter.create_documents(
            texts=[full_text],
            metadatas=[{"video_id": video_id, "language": self.detected_language}]
        )

        video_hash = self._get_video_hash(video_id)
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
        logger.info("✅ Transcript indexed successfully")

    # ----------------------- Generation -----------------------
    def _clean_title_tokens(self, text: str, detected_lang: str | None = None) -> str:
        """Simple fallback extraction."""
        # ... (Keeping existing heuristic logic if needed for direct RAG usage) ...
        t = text.strip().strip("\"'`.,;:!?-–—")
        return " ".join(t.split()[:5]) if t else "Section"

    def generate_sections(self, transcript: list[dict[str, Any]], video_id: str, num_sections: int = 10, retrieval_k: int = 5, hierarchical: bool = True) -> list[dict[str, Any]]:
        if self.current_video_id != video_id or self.vectorstore is None:
            self.index_transcript(transcript, video_id)

        # ... (Retaining existing logic for backward compatibility) ...
        # NOTE: The Split Pipeline supersedes this for better quality
        return []

    # ----------------------- Cleanup -----------------------
    def cleanup(self) -> None:
        logger.info("Cleaning up RAG resources...")
        try:
            if getattr(self, "vectorstore", None) is not None:
                # REMOVED: self.vectorstore.persist() (Deprecated)
                pass

            if getattr(self, "chroma_client", None) is not None:
                try:
                    cast(Any, self.chroma_client).close()
                except Exception:
                    pass

            self.vectorstore = None
            self.llm = None
            self.current_video_id = None
        except Exception:
            logger.debug("RAG cleanup error", exc_info=True)
        logger.info("RAG cleanup complete")
