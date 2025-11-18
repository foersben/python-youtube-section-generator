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
import os
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.core.embeddings import EmbeddingsFactory

logger = logging.getLogger(__name__)

# Centralized embeddings provider (CPU-only default)


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
            # Some langchain releases may reorganize modules; try common alternatives
            try:
                from langchain.text_splitter import (
                    CharacterTextSplitter as _CTS,  # type: ignore
                )

                class _FallbackRecursiveCharacterTextSplitter(_CTS):  # type: ignore
                    """Thin compatibility wrapper mapping to CharacterTextSplitter API."""

                    pass

                RecursiveCharacterTextSplitter = _FallbackRecursiveCharacterTextSplitter

            except Exception:
                # Fallback: provide a minimal local splitter implementation
                logger.warning(
                    "langchain.text_splitter not available; using local fallback splitter. "
                    "Install 'langchain' for full functionality."
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

        self.vectorstore: _Any = None
        self.current_video_id: str | None = None
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

    def _refine_title_with_llm(self, snippet: str) -> str:
        """Generate a concise section title from a transcript snippet."""
        snippet = (snippet or "").strip()
        if not snippet:
            return "Section"

        context_snippet = snippet[:1500]
        prompt = (
            "You write table-of-contents headings for long-form videos. "
            "Given the transcript excerpt below, produce one descriptive title (3-10 words). "
            "It must be a standalone phrase without prefixes or numbering.\n\n"
            f'Excerpt:\n"...{context_snippet}..."\n\nTitle:'
        )
        try:
            result = self.llm.invoke(prompt)  # type: ignore[attr-defined]
            raw_title = str(result).strip().splitlines()[0]
            polished = self._polish_title(raw_title)
            if len(polished.split()) >= 2:
                return polished
        except Exception as exc:
            logger.error("LLM title refinement failed: %s", exc, exc_info=True)

        return self._clean_title_tokens(snippet, self.detected_language)

    def _polish_title(self, title: str) -> str:
        """Remove artifacts and normalize capitalization."""
        title = title.strip().strip("\"'`.,;:!?-–—")
        artifacts = [
            "assistant",
            "response",
            "title",
            "topics",
            "here is",
            "here's",
            "sure",
            "of course",
            "intro",
            "section",
        ]
        for artifact in artifacts:
            title = re.sub(rf"\b{artifact}\b:?", "", title, flags=re.IGNORECASE)

        words = title.split()
        if words and words[0].lower() in {"the", "a", "an"}:
            words = words[1:]
        title = " ".join(words).strip()

        if title and not title[0].isupper():
            title = title[0].upper() + title[1:]

        return title or "Section"

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

        # Language-specific stopwords
        stopwords_by_lang = {
            "en": {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "up",
                "about",
                "into",
                "through",
                "during",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "must",
                "shall",
                "this",
                "that",
                "these",
                "those",
                "which",
                "who",
                "what",
                "where",
                "when",
                "why",
                "how",
                "all",
                "each",
                "every",
                "both",
                "few",
                "more",
                "most",
                "other",
                "some",
                "such",
            },
            "de": {
                # German stopwords
                "der",
                "die",
                "das",
                "den",
                "dem",
                "des",
                "ein",
                "eine",
                "einer",
                "eines",
                "und",
                "oder",
                "aber",
                "in",
                "an",
                "auf",
                "zu",
                "für",
                "von",
                "mit",
                "bei",
                "aus",
                "über",
                "durch",
                "um",
                "nach",
                "vor",
                "zwischen",
                "ist",
                "sind",
                "war",
                "waren",
                "sein",
                "haben",
                "hat",
                "hatte",
                "hatten",
                "werden",
                "wird",
                "wurde",
                "wurden",
                "kann",
                "könnte",
                "muss",
                "sollte",
                "mag",
                "darf",
                "dies",
                "diese",
                "dieser",
                "dieses",
                "jene",
                "welche",
                "wer",
                "was",
                "wo",
                "wann",
                "warum",
                "wie",
                "alle",
                "jede",
                "einige",
                "mehr",
                "meisten",
                "andere",
                "solche",
                "äh",
                "ähm",
                "hmm",
                "uh",
                "er",
                "also",
                "nicht",
                "noch",
                "schon",
                "nur",
                "ich",
                "du",
                "wir",
                "sie",
                "ihm",
                "ihr",
                "ihn",
                "mir",
                "dir",
            },
        }

        # Get appropriate stopwords for detected language
        stop_words = stopwords_by_lang.get(detected_lang, stopwords_by_lang["en"])

        # Split into words
        words = [w for w in re.split(r"\s+", t) if w]

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

            # Skip stopwords (case-insensitive, language-aware)
            if w_clean.lower() in stop_words:
                continue

            # Language-specific capitalization logic
            if detected_lang == "de":
                # German: All nouns are capitalized - strong signal for important words
                if w_clean[0].isupper():
                    filtered.append(w_clean)
                elif len(filtered) < 1:  # Allow first word even if lowercase
                    filtered.append(w_clean)
            else:
                # English/other: Prefer capitalized (proper nouns), but allow lowercase if few words
                if w_clean[0].isupper() or len(filtered) < 2:
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
        main_sections = self._find_semantic_boundaries(transcript, distance_percentile=25)
        if not main_sections:
            return []

        sections: list[dict[str, Any]] = []
        use_llm_titles = os.getenv("USE_LLM_TITLES", "true").lower() == "true"
        vectorstore_available = getattr(self, "vectorstore", None) is not None

        for idx, main_seg in enumerate(main_sections):
            main_start = float(main_seg.get("start", 0.0))
            context_window = self._aggregate_context(
                transcript, ts=main_start, window=90.0, max_chars=2000
            )

            main_title = self._clean_title_tokens(main_seg.get("text", ""), self.detected_language)
            if use_llm_titles and vectorstore_available and context_window:
                try:
                    main_title = self._refine_title_with_llm(context_window)
                    logger.info(
                        "[llama.cpp] Main title refined at %.1fs -> %s",
                        main_start,
                        main_title,
                    )
                except Exception:
                    logger.warning(
                        "Main title refinement failed at %.1fs; using heuristic", main_start
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

    def _find_subtopics_in_chunk(
        self,
        transcript_chunk: list[dict[str, Any]],
        max_subsections: int,
    ) -> list[dict[str, Any]]:
        """Use the local LLM to identify subtopics within a transcript slice."""
        if not transcript_chunk or max_subsections <= 0:
            return []

        chunk_text = self._create_transcript_text(transcript_chunk)
        if len(chunk_text) < 120:
            return []

        prompt = (
            "You are a video editor. Analyze the following transcript section and extract up to "
            f"{max_subsections} distinct subtopics. Use the timestamps from the [xx.xs] markers "
            "for each subtopic's start value. Respond strictly as JSON with schema: "
            '{"subtopics": [{"start": <seconds>, "title": "..."}]}.'
            "\n\nTranscript Section:\n"
            f"{chunk_text}\n\nJSON:"
        )

        prev_max_tokens = getattr(self.llm, "max_tokens", None)
        try:
            if prev_max_tokens and prev_max_tokens < 400:
                self.llm.max_tokens = 400
            raw_response = str(self.llm.invoke(prompt))  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Subtopic extraction failed: %s", exc)
            return []
        finally:
            if prev_max_tokens is not None:
                self.llm.max_tokens = prev_max_tokens

        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        json_payload = match.group(0) if match else raw_response.strip()
        try:
            payload = json.loads(json_payload)
        except json.JSONDecodeError:
            logger.warning("Subtopic JSON parse failed; raw response: %s", raw_response[:200])
            return []

        subtopics = payload.get("subtopics", []) if isinstance(payload, dict) else []
        results: list[dict[str, Any]] = []
        for topic in subtopics:
            if not isinstance(topic, dict):
                continue
            try:
                start_val_raw = topic.get("start")
                start_val = float(start_val_raw or 0.0)
            except (TypeError, ValueError):
                continue
            title_val = self._polish_title(str(topic.get("title", "")).strip())
            if not title_val:
                continue
            results.append({"title": title_val, "start": start_val, "level": 1})
            if len(results) >= max_subsections:
                break
        return results

    def _fallback_time_subsections(
        self,
        transcript: list[dict[str, Any]],
        window_start: float,
        window_end: float,
        count: int,
    ) -> list[dict[str, Any]]:
        """Fallback heuristic for subsection creation using evenly spaced timestamps."""
        if count <= 0 or window_end <= window_start:
            return []

        subsections: list[dict[str, Any]] = []
        for idx in range(count):
            frac = (idx + 1) / (count + 1)
            sub_ts = window_start + frac * (window_end - window_start)
            nearest = (
                min(transcript, key=lambda seg: abs(seg.get("start", 0.0) - sub_ts))
                if transcript
                else None
            )
            title = self._clean_title_tokens(
                nearest.get("text", "") if nearest else "Section", self.detected_language
            )
            if len(title.split()) < 2:
                title = self._clean_title_tokens("Section", self.detected_language)
            subsections.append({"title": title or "Section", "start": round(sub_ts, 1), "level": 1})
        return subsections
