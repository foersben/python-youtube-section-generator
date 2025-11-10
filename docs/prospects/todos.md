# Todos (short-term tasks)

This page is the compact master todo list. Detailed plans live in the grouped
pages linked below.

- [ ] Setup: interactive `poetry install` + `.env` helper — see `setup.md` (priority: high)
- [ ] Processing pipeline refactor (staged pipeline) — see `pipeline.md` (priority: high)
- [ ] Pipeline architecture & contracts — see `pipeline.md` (priority: medium)
- [ ] CI: Add fast-tests + heavy-integration skeleton and gating — see `ci-tests.md` (priority: high)
- [ ] Add `filelock` into test dev deps and ensure tests use it for LLM serialization — see `ci-tests.md` (priority: high)
- [ ] Release helper: interactive section selection in `scripts/release_changelog.py` (priority: medium)
- [ ] Add pre-commit/CI checks to prevent status files in `docs/` and validate changelog hygiene (priority: high)
- [ ] Create `tests/` markers and fixtures for llm-locked tests (priority: medium)


For the detailed plans and the rationale behind each item, see:

- `prospects/setup.md` — interactive installer and env config
- `prospects/pipeline.md` — processing pipeline architecture & staged pipeline
- `prospects/ci-tests.md` — CI skeleton, heavy-tests runner requirements
