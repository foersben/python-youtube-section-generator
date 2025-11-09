"""RAG system for transcript processing.

This module provides TranscriptRAG which indexes long transcripts into a
Chroma vector store and generates sections (flat or hierarchical). It uses
sentence-transformers for embeddings (CPU-only) and llama.cpp (via
LangChain's LlamaCpp) for refining main section titles using a local GGUF
model.
"""

from __future__ import annotations

import logging
import os
import re
import hashlib
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

# Centralized embeddings provider (CPU-only default)
from src.core.embeddings import EmbeddingsFactory


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
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.llms import LlamaCpp
        except ImportError as e:  # pragma: no cover - runtime env check
            raise RuntimeError(
                f"Required packages not installed: {e}\n"
                "Install with: poetry add langchain langchain-community chromadb sentence-transformers"
            ) from e

        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = max(0.0, min(1.0, temperature))

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

        # Text splitter
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

        # Local LLM (llama.cpp via LangChain) - optimized for concise title generation
        import multiprocessing

        n_threads = multiprocessing.cpu_count()
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_ctx=4096,
            n_threads=n_threads,
            temperature=0.05,  # Very low for deterministic, focused output
            max_tokens=64,  # Short titles only
            top_p=0.85,  # Slightly lower for more focused sampling
            top_k=20,  # Limit to top 20 tokens for consistency
            repeat_penalty=1.1,  # Discourage repetition
            verbose=False,
        )

        self.vectorstore = None
        self.current_video_id = None
        logger.info("\u2705 RAG system initialized")

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
        self, transcript: list[dict[str, Any]], ts: float, window: float = 30.0, max_chars: int = 400
    ) -> str:
        """Aggregate transcript text around a timestamp, plus vector hits (best-effort)."""
        # Window-based context
        nearby = [seg for seg in transcript if abs(seg.get("start", 0) - ts) <= window]
        if not nearby:
            nearby = [min(transcript, key=lambda seg: abs(seg.get("start", 0) - ts))]
        window_text = " ".join(str(s.get("text", "")) for s in nearby)

        # Vectorstore-based context (best-effort)
        vec_text = ""
        if getattr(self, "vectorstore", None) is not None:
            try:
                docs = self.vectorstore.similarity_search(query=f"context around {int(ts)} seconds", k=2)
                vec_text = " ".join(getattr(d, "page_content", str(d)) for d in docs)
            except Exception:
                vec_text = ""

        combined = f"{window_text} \n {vec_text}".strip()
        combined = re.sub(r"\s+", " ", combined).strip()
        return combined[:max_chars]

    def _refine_title_with_llm(self, snippet: str) -> str:
        """Multi-stage LLM refinement: extract keywords → generate title → polish.

        Breaking into simple steps allows Phi-3-mini to succeed where one complex
        prompt fails.
        """
        def _clean_short(text: str) -> str:
            t = text.strip().strip("\"'`.,;:!?-–—")
            words = re.findall(r"[A-Za-zÄÖÜäöüß]+", t)
            if not words:
                return "Section"
            title = " ".join(words[:7])
            if title and not title[0].isupper():
                title = title[0].upper() + title[1:]
            return title

        if not snippet:
            return "Section"

        # STAGE 1: Extract 3-5 key topics/entities (simple extraction task)
        stage1_prompt = (
            f"Extract 3-5 key topics from this text:\n{snippet[:200]}\n\n"
            "Topics:"
        )

        try:
            keywords_result = self.llm.invoke(stage1_prompt)  # type: ignore[attr-defined]
            keywords = str(keywords_result).strip().splitlines()[0][:100]

            # STAGE 2: Create title from keywords (focused generation task)
            stage2_prompt = (
                f"Create a 2-4 word title from these topics: {keywords}\n"
                "Use nouns only. No verbs.\n"
                "Title:"
            )

            title_result = self.llm.invoke(stage2_prompt)  # type: ignore[attr-defined]
            raw_title = str(title_result).strip().splitlines()[0]

            # STAGE 3: Polish (minimal cleanup)
            title = self._polish_title(raw_title)

            if title and len(title.split()) >= 2:
                return title
            else:
                # Fallback to heuristic if multi-stage fails
                return _clean_short(snippet)

        except Exception as e:
            logger.debug("Multi-stage LLM refinement failed: %s", e)
            return _clean_short(snippet)

    def _polish_title(self, title: str) -> str:
        """Light polish - remove artifacts but don't reject valid titles."""
        import re

        # Remove common artifacts
        artifacts = ["assistant", "response", "title", "topics"]
        for artifact in artifacts:
            title = re.sub(rf'\b{artifact}\b', '', title, flags=re.IGNORECASE)

        # Clean punctuation
        title = title.strip("\"'`.,;:!?-–—")

        # Remove articles at start
        words = title.split()
        if words and words[0].lower() in ["the", "a", "an"]:
            words = words[1:]

        title = " ".join(words)

        # Capitalize
        if title and not title[0].isupper():
            title = title[0].upper() + title[1:]

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

        chunks = self.text_splitter.create_documents(
            texts=[full_text],
            metadatas=[{"video_id": video_id, "total_segments": len(transcript)}],
        )
        logger.info("Created %d chunks for indexing", len(chunks))

        video_hash = self._get_video_hash(video_id)
        persist_directory = f".chromadb/{video_hash}"

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
    def _clean_title_tokens(self, text: str) -> str:
        """Extract clean title using NLP-inspired heuristics (nouns, proper nouns, key terms)."""
        t = text.strip().strip("\"'`.,;:!?-–—")

        # Remove timestamps and numeric-only tokens
        t = re.sub(r"\b\d{1,2}:\d{2}\b", "", t)
        t = re.sub(r"\b\d{2,4}s\b", "", t)
        t = re.sub(r"\[\d+(:\d+)*s\]", "", t)

        # Split into words
        words = [w for w in re.split(r"\s+", t) if w]

        # Skip common stop words and verbs
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "can", "must", "shall", "this", "that", "these", "those", "which", "who",
            "what", "where", "when", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "such"
        }

        filtered: list[str] = []
        for w in words:
            w_clean = w.strip("\"'`.,;:!?-–—()[]{}")
            if not w_clean or len(w_clean) <= 2:
                continue

            # Reject if mostly digits
            digits = sum(c.isdigit() for c in w_clean)
            if digits >= len(w_clean) * 0.5:
                continue

            # Reject pure numbers or long digit sequences
            if w_clean.isdigit() or re.search(r"\d{3,}", w_clean):
                continue

            # Skip stop words (case-insensitive)
            if w_clean.lower() in stop_words:
                continue

            # Prefer capitalized words (likely proper nouns or important terms)
            # But don't skip lowercase if we have few words
            if w_clean[0].isupper() or len(filtered) < 2:
                filtered.append(w_clean)

            if len(filtered) >= 5:  # Limit to 5 words max
                break

        if not filtered:
            # Desperate fallback: extract any capitalized words or longest alphabetic sequences
            alpha = re.findall(r"[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,3}", text)
            if alpha:
                return alpha[0].strip()
            # Last resort: just get some alphabetic words
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
            logger.info("Generating flat section %d/%d at ~%ds", idx + 1, num_sections, int(sample_time))
            nearest = min(transcript, key=lambda s: abs(s.get("start", 0) - sample_time))
            candidate_title = self._clean_title_tokens(nearest.get("text", ""))
            candidate_start = float(nearest.get("start", 0.0))

            if use_vectorstore:
                try:
                    docs = self.vectorstore.similarity_search(
                        query=f"Context around {int(sample_time)} seconds",
                        k=min(max(1, retrieval_k), 10),
                    )
                    if docs:
                        doc_text = getattr(docs[0], "page_content", None) or str(docs[0])
                        candidate_title = self._clean_title_tokens(doc_text)
                except Exception:
                    logger.debug("Vectorstore retrieval failed", exc_info=True)

            sections.append({"title": candidate_title, "start": round(candidate_start, 1)})

        return sections

    def _generate_hierarchical_sections(
        self,
        transcript: list[dict[str, Any]],
        total_duration: float,
        main_sections_count: int,
        subsections_per_main: int,
        retrieval_k: int,
    ) -> list[dict[str, Any]]:
        # Generate main anchors first
        main_sections = self._generate_flat_sections(
            transcript, total_duration, main_sections_count, retrieval_k
        )

        sections: list[dict[str, Any]] = []

        def _aggregate_snippet(ts: float, window: float = 20.0, max_words: int = 6) -> str:
            nearby = [seg for seg in transcript if abs(seg.get("start", 0) - ts) <= window]
            if not nearby:
                nearby = [min(transcript, key=lambda seg: abs(seg.get("start", 0) - ts))]
            combined = " ".join(str(s.get("text", "")) for s in nearby)
            words = re.findall(r"[A-Za-zÄÖÜäöüß]+", combined)
            if not words:
                return "Section"
            title = " ".join(words[:max_words])
            if title and not title[0].isupper():
                title = title[0].upper() + title[1:]
            return title

        for idx, main in enumerate(main_sections):
            main_start = float(main.get("start", 0.0))
            base_title = str(main.get("title", ""))

            # Use multi-stage LLM refinement by default (disabled via USE_LLM_TITLES=false)
            use_llm_titles = os.getenv("USE_LLM_TITLES", "true").lower() == "true"

            if use_llm_titles:
                # Try LLM refinement (often fails with small models)
                try:
                    context = self._aggregate_context(transcript, main_start)
                    refined = self._refine_title_with_llm(context)
                    if refined and any(c.isalpha() for c in refined):
                        logger.info(
                            "[llama.cpp] Refined main title at %.1fs using %s: '%s' -> '%s'",
                            main_start,
                            Path(self.model_path).name,
                            base_title,
                            refined,
                        )
                        main_title = refined
                    else:
                        logger.info("[llama.cpp] LLM returned weak title; using heuristic")
                        main_title = _aggregate_snippet(main_start)
                except Exception:
                    logger.warning("[llama.cpp] Refinement failed; using heuristic")
                    main_title = _aggregate_snippet(main_start)
            else:
                # Use heuristics only (faster, more reliable for small models)
                main_title = _aggregate_snippet(main_start)
                logger.debug("Using heuristic title at %.1fs: '%s'", main_start, main_title)

            sections.append({"title": main_title, "start": main_start, "level": 0})

            # Determine window for subsections
            if idx + 1 < len(main_sections):
                next_start = float(main_sections[idx + 1].get("start", total_duration))
            else:
                next_start = total_duration

            window_start = main_start
            window_end = max(window_start + 1.0, next_start)
            window_len = window_end - window_start

            for si in range(subsections_per_main):
                frac = (si + 1) / (subsections_per_main + 1)
                sub_ts = window_start + frac * window_len
                if sub_ts > total_duration:
                    sub_ts = total_duration

                # Improved heuristic subsection titles (filter garbage better)
                sub_nearest = min(transcript, key=lambda s: abs(s.get("start", 0) - sub_ts))
                sub_title = self._clean_title_tokens(sub_nearest.get("text", ""))

                # Additional check: reject if title is still numeric/garbage after cleaning
                if (not any(c.isalpha() for c in sub_title)) or re.search(r"\d{3,}", sub_title):
                    sub_title = _aggregate_snippet(sub_ts)

                # Final check: ensure it's not a single short word or connector
                words = re.findall(r"[A-Za-zÄÖÜäöüß]{3,}", sub_title)
                if len(words) < 2:
                    sub_title = _aggregate_snippet(sub_ts)

                sections.append({"title": sub_title, "start": round(sub_ts, 1), "level": 1})

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
            subsections_per_main = max(2, (num_sections - main_sections_count) // main_sections_count)
            sections = self._generate_hierarchical_sections(
                transcript, total_duration, main_sections_count, subsections_per_main, retrieval_k
            )
        else:
            sections = self._generate_flat_sections(transcript, total_duration, num_sections, retrieval_k)

        # Final cleanup
        clean_sections: list[dict[str, Any]] = []
        for s in sections:
            title = str(s.get("title", "")).strip().strip("\"'`.,;:!?-–—")
            if not title or len(title) < 3:
                title = "Section"
            clean_sections.append({"title": title, "start": float(s.get("start", 0.0)), **({"level": s["level"]} if "level" in s else {})})

        return clean_sections

    # ----------------------- Cleanup -----------------------
    def cleanup(self) -> None:
        logger.info("Cleaning up RAG resources...")
        try:
            if getattr(self, "vectorstore", None) is not None:
                try:
                    # Removed deprecated persist() call - ChromaDB 0.4.x auto-persists
                    pass
                except Exception:
                    logger.debug("Vectorstore cleanup failed", exc_info=True)
                try:
                    if hasattr(self.vectorstore, "shutdown"):
                        self.vectorstore.shutdown()
                except Exception:
                    logger.debug("Vectorstore shutdown failed", exc_info=True)
                self.vectorstore = None
            if getattr(self, "llm", None) is not None and hasattr(self.llm, "close"):
                try:
                    self.llm.close()
                except Exception:
                    logger.debug("LLM close failed", exc_info=True)
            logger.info("RAG cleanup finished")
        except Exception:
            logger.debug("Cleanup encountered an error", exc_info=True)
            return
