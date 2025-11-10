# Prospects, Todos & Roadmap

This page groups short-term todos, the roadmap, and backlog research items.
Split detailed plans live in `docs/legacy/prospects/` for reference.

Short-term todos

- Interactive `poetry` setup and `.env` helper (high)
- Pipeline refactor (staged pipeline) — implement `src/core/pipeline.py` (high)
- Add CI skeleton (fast-tests + gated heavy-integration) (high)
- Add `filelock` and `@pytest.mark.llm` gating for heavy tests (high)

Roadmap (summary)

- Stabilize refinement pipeline and translation fallbacks (Q4 2025)
- Add self-hosted runner automation and heavy-test workflows (Q1 2026)
- Long-term: speaker diarization, multilingual fine-tuning, desktop bundle

Backlog & research

- Phonetic correction datasets for German/English
- Retrieval-augmented polishing (RAP) for title refinement
- Efficient CPU-embeddings alternatives

How to contribute

- Move an item from backlog into an actionable issue/PR and assign owners
- Use `scripts/tools/todos_to_issues.py` to create issues from `prospects/todos.md`


