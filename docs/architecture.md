# Architecture & Processing Pipeline

Overview

This document provides a concise, developer-focused overview of the system
architecture, the core processing pipeline, and configuration guidance.

Key components

- `src/core/` — Business logic: transcript extraction, section generation, RAG
- `src/utils/` — Utilities: file I/O, JSON helpers, logging configuration
- Web app: `src/web_app.py` (Flask-based)
- CLI: `src/main.py`

Processing pipeline (staged)

1. Extraction: Pull transcript segments from YouTube (auto-generated or manual)
2. Preprocessing: Language detection, normalization, optional basic cleaning
3. Translation (DE→EN): Optional, batched translations (DeepL or local LLM)
4. RAG indexing: Create vector store from transcript chunks (ChromaDB)
5. Section detection: Identify main anchors and subsections
6. Title generation: Multi-stage LLM refinement (keywords → raw title → polish)
7. Back-translation (EN→DE): If original language was DE, translate titles back
8. Validation & Formatting: Enforce length/format and produce final output

Performance & CPU-only considerations

- Use quantized local models (q4) to reduce RAM usage.
- Use `sentence-transformers/all-MiniLM-L6-v2` for CPU-optimized embeddings.
- Batch translations to reduce API calls (intelligent batching: group by char limit)

Internationalization & Fallbacks

- The pipeline is language-aware: prefer translation-first for non-English
  content (DE → EN → generate → EN → DE back-translate).
- If translation fails, the system must fall back gracefully. The fallback
  heuristic should be language-aware (not hard-coded English stopwords).

Observability

- Emit structured logs per stage (timings, sample outputs)
- Track per-video performance metrics and resource usage

See `docs/legacy/` for detailed legacy docs and design notes if you need to
explore original long-form documentation.

