# Advanced Topics

This page consolidates deep technical details that used to be scattered across
multiple documents.

## Transcript Refinement Internals

- Batch context windows: include N segments before/after for context-aware
  correction
- Prompt design: keywords → noun phrase title → polish (remove artifacts,
  casing, punctuation)
- Failure handling: if refinement fails, return original text and log at ERROR

## RAG & Retrieval Details

- Chunking strategy: recursive character splitter (target ~1k, 20% overlap)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (CPU-friendly)
- Vector store: ChromaDB persisted per-video hash

## Fallback Title Heuristics (Language-Aware)

- Detect language for each snippet and load stopwords accordingly
- Prefer nouns and proper nouns; deprioritize fillers and short function words
- Hard limit to 3–5 tokens; title case post-processing

## Internationalization (DE⇄EN)

- Primary prompts are English; DE→EN yields higher quality intermediate titles
- Back-translation restores user language; if translation unavailable, keep
  titles in EN and notify user of fallback

## Observability & Debugging

- Log prompt input/output at DEBUG (ensure no sensitive data)
- Measure per-stage timings; include sample sizes in logs for context
- Expose health checks for model presence, API availability, and disk space

## Performance

- CPU-only guidance: 4-bit quantized model, thread count tuned to cores
- Batched translation reduces API calls dramatically; use markers to rebuild
  segment alignment reliably

## Security Notes

- Never bundle model weights; provide download scripts only
- Keep API keys in `.env` and out of logs

For original long-form explanations, see `docs/legacy/*`.

