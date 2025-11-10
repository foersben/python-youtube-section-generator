# Architecture & Processing Pipeline

Overview

This document explains what the system does and why it is designed this way.
It covers the architecture, key design choices, the staged processing pipeline,
fallback strategies, and performance considerations for CPU-only setups.

System goals and constraints

- Accurate sections and titles from noisy YouTube transcripts (often auto-gen)
- Multi-language support (German → English pipeline, back-translation)
- Runs on commodity CPUs (no GPU required); small local models
- Deterministic, observable behavior suitable for CI/CD

High-level architecture

- Web App (Flask): simple UI and REST endpoints for processing requests
- CLI: batch processing and automation
- Core Services:
  - Transcript extraction (YouTube Transcript API)
  - Translation (DeepL or local LLM fallback) — batched
  - Retrieval (ChromaDB) and embeddings (Sentence Transformers)
  - Title generation (multi-stage LLM refinement)
- LLM Providers:
  - Local (llama.cpp via GGUF — Phi-3-mini-4k-instruct-q4.gguf)
  - Cloud (Gemini), optional

Why staged processing?

- Decomposition improves small-model reliability: small 4b models are more
  consistent when the task is split into simple steps (keywords → title → polish)
- Translation-first improves quality: prompts and heuristics are optimized for
  English; back-translation preserves user language
- Deterministic tuning: low temperature and strict polishing rules reduce noise

Processing pipeline (staged)

1. Extraction — Get raw segments (start, duration, text)
2. Preprocessing — Normalize whitespace, detect language
3. Translation — Batch DE→EN to reduce API calls and preserve context
4. Indexing (RAG) — Chunk and embed transcript; persist Chroma collection
5. Section detection — Evenly distributed anchors + neighborhood retrieval
6. Title synthesis — 3-step LLM chain (keywords → noun phrase → polish)
7. Back-translation — EN titles → original language (if needed)
8. Validation & formatting — Enforce style, timestamp formats, uniqueness

Fallbacks and failure modes (with rationale)

- Translation unavailable (quota or offline): continue with original language
  but avoid English-only stopword heuristics; prefer language-aware filtering
- LLM call fails: fall back to heuristic title generation with language-aware
  stopwords; log ERROR with stack and proceed
- RAG unavailable: generate flat sections by time-based anchors only

Internationalization (why DE→EN→DE?)

- Prompts and heuristics were tuned in English; translating input to EN allows
  reuse and yields higher title quality. Back-translating final titles returns
  user-facing language.
- If translation fails, the fallback remains language-aware to avoid
  English-centric filtering on German text.

Observability & logging

- Log each stage with timings; log LLM prompts at DEBUG level for diagnosis
- Elevate silent failures to ERROR with `exc_info=True` to capture tracebacks
- Return non-blocking warnings to UI when LLM/translation is skipped

Performance & CPU-only choices

- Embeddings: `all-MiniLM-L6-v2` — optimized for CPU
- Quantized local LLM (q4) — fits into 5–8 GB RAM
- Batching translation reduces calls 50× and speeds up ~10× on long videos

Licensing & distribution

- Do not bundle model weights in source or wheels; document paths and provide
  download helpers. Respect upstream model licenses.

Design patterns (why they fit)

- Service layer — isolates orchestration and external calls for testability
- Factory — switches between local/cloud providers by configuration
- Strategy — enables alternative pipelines (staged vs direct, RAG vs none)
- Pipes & Filters — each stage transforms transcript/selections predictably

References

- Legacy deep dives: `docs/legacy/` (transcript refinement details,
  llm-configuration, and historical architecture).
