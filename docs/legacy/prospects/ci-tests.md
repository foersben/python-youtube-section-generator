# CI & Heavy Tests (consolidated)

This file consolidates CI skeleton tasks and heavy-test guidance previously split
into smaller documents.

Summary

- Create `.github/workflows/ci.yml` with two jobs:
  - `fast-tests` (runs on `ubuntu-latest`) — installs deps and runs unit tests, lint
  - `heavy-integration` (runs on `self-hosted, self-hosted-llm`) — runs heavy LLM/RAG tests, gated by `RUN_HEAVY_INTEGRATION=true`

- Document heavy-runner requirements and how to prepare a self-hosted runner (see below)

Todos

- [ ] Add `.github/workflows/ci.yml` skeleton with fast-tests and gated heavy-integration (priority: high)
- [ ] Mark heavy LLM tests with `@pytest.mark.llm` and implement serialization via `filelock` (priority: high)
- [ ] Add documentation for runner setup in `prospects/heavy-tests.md` (already created)
- [ ] Provide a small `act` guide in docs for local CI testing (optional)

Heavy-runner summary

- RAM: 8-16+ GB depending on model quantization
- Set `LOCAL_MODEL_PATH` on the runner and ensure models live under `models/`
- Label the runner `self-hosted-llm` and restrict usage to trusted repos
- Set `RUN_HEAVY_INTEGRATION=true` to allow the heavy job to run

# Pipeline (processing architecture & staged pipeline)

Consolidated architecture and todos for the staged pipeline work.

Summary

- Implement a configurable orchestrator that sequences the following stages:
  - Preprocessing
  - Section detection (anchors)
  - Section refinement (time windows)
  - Title & description generation (flexible length)
  - Title polishing (length constraints)
  - Validation & formatting

Todos

- [ ] Implement `src/core/pipeline.py` orchestrator class and stage interface
- [ ] Implement `Section` dataclass and stage contracts in `src/core/models`
- [ ] Implement a `staged` pipeline variant and a flag `PROCESSING_PIPELINE=staged`
- [ ] Add AB/benchmark harness to compare `single` vs `staged` pipelines
- [ ] Add unit tests for each stage

Design notes

- Use a plugin/strategy loader so stages can be swapped by config.
- Keep LLM calls small and focused; stage outputs should be compact.
- Provide instrumentation per-stage (timing, sample outputs) to evaluate quality.


