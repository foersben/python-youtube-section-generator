# Prospects, Todos & Roadmap

Purpose

This page consolidates actionable next steps (todos), rationale for upcoming
changes, and a high-level roadmap. Detailed working notes remain in
`docs/legacy/prospects/`.

Why these priorities?

- Interactive setup reduces onboarding time and prevents misconfiguration
- Staged pipeline lowers prompt complexity and improves quality on small models
- CI split keeps PR feedback fast and isolates heavy tests to prepared runners
- LLM gating & locking prevent local resource exhaustion

Short-term todos (grouped)

Setup & DX

- Interactive `poetry` setup and `.env` helper (high)
- Pre-commit and CI check to prevent non-doc READMEs outside root (high)

Pipeline & Quality

- Implement `src/core/pipeline.py` orchestrator (high)
- Add language-aware heuristics to fallback title generator (high)
- Improve error logging visibility with `exc_info=True` (medium)

CI & Tests

- Add CI skeleton (fast-tests + gated heavy-integration) (done)
- Mark heavy tests with `@pytest.mark.llm` and gate on env (high)
- Add file-based serialization with `filelock` around LLM access (high)

Roadmap (summary)

- Q4 2025: Stabilize translation fallbacks; document staged pipeline; docs polish
- Q1 2026: Self-hosted runner automation; refine RAG retrieval & ranking
- Long-term: diarization, multilingual fine-tuning, portable desktop bundle

Backlog & research

- Phonetic correction datasets (DE/EN) for refinement evaluation
- Retrieval-augmented polishing (RAP) for titles
- More efficient CPU embeddings and chunking strategies

Ownership & tracking

- Use `scripts/tools/todos_to_issues.py` to open issues for top items
- Add labels `priority:high`, `area:pipeline`, `area:ci`, `doc:setup`
