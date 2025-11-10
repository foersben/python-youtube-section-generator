# Todos (short-term tasks)

This list contains small-to-medium tasks that are suitable for quick PRs.
Prioritize by impact/time and tag with an owner when possible.

- [ ] Add CI/pre-commit check to prevent status files in `docs/` (priority: high)
- [ ] Add CI job to validate changelog hygiene (per-release files) (priority: medium)
- [ ] Add a pre-commit config to normalize markdown filenames when committing (priority: low)
- [ ] Add `filelock` into test dev deps and ensure tests use it for LLM serialization (priority: high)
- [ ] Add a small README in `.github/status/` describing file conventions (priority: medium)
- [ ] Create a script to migrate legacy uppercase docs to kebab-case automatically (priority: low)
- [ ] Implement `scripts/release_changelog.py` interactive mode (priority: medium)
- [ ] Create `tests/` markers and fixtures for llm-locked tests (priority: medium)
- [ ] Add examples and usage to `scripts/release_changelog.py` docs (priority: low)

## New/Open Items (from user request)

- [ ] Interactive `poetry install` setup script that:
  - Checks for presence of local models and required files (e.g. `models/*.gguf`).
  - Validates environment variables (DEEPL_API_KEY, LOCAL_MODEL_PATH, GOOGLE_API_KEY).
  - Installs additional pip-only dependencies when needed (sentence-transformers, chromadb).
  - Offers a two-tier prompt: General (required) and Advanced (hidden by default) where the user can adjust model paths, batch sizes, GPU layers, etc.
  - Provide a `--yes` non-interactive mode for CI/automation.
  - See: `prospects/setup-interactive.md` (implementation plan).
  - Priority: high  • Suggested owner: @maintainer

- [ ] Processing pipeline refactor (staged pipeline):
  - Implement a multi-stage pipeline: (1) Section detection (no title constraints), (2) Title+description generation per section (flexible length), (3) Final title polishing with length constraints.
  - Make pipeline selectable via `.env`: `PROCESSING_PIPELINE=single|staged|rag`.
  - Add benchmarks/AB-tests comparing single vs staged pipeline quality and latency.
  - See: `prospects/processing-architecture.md` (architecture concept and implementation notes).
  - Priority: high  • Suggested owner: @core-team

- [ ] Complete architecture concept & patterns for processing pipeline:
  - Explore pipes-and-filters, chain-of-responsibility, or micro-step orchestration.
  - Define contract interfaces for each pipeline stage (inputs/outputs, error modes).
  - Provide config extensibility (JSON/YAML) and feature flags for new stages.
  - Document in `prospects/processing-architecture.md`.
  - Priority: medium • Suggested owner: @architect

- [ ] GitHub Actions workflow skeleton (fast tests + optional heavy self-hosted):
  - Add `.github/workflows/ci.yml` with `fast-tests` job (runs on GitHub hosted) and `heavy-integration` job (runs only on `self-hosted-llm` label and gated by `RUN_HEAVY_INTEGRATION=true`).
  - Include secrets/config guidance in doc `prospects/heavy-tests.md`.
  - Priority: high • Suggested owner: @ci

- [ ] Main README / docs entry describing heavy-test requirements:
  - Add a clear section in the project README and link to `prospects/heavy-tests.md` covering: local model placement, minimum RAM, how to label self-hosted runner, and gating environment variables.
  - Priority: high • Suggested owner: @maintainer

- [ ] Release helper enhancements: implement interactive selection of sections to move from `unreleased.md` into a release file (enhance `scripts/release_changelog.py`).
  - Priority: medium • Suggested owner: @maintainer

- [ ] Add labels and maintainers in `prospects/todos.md` and add a small script to convert todo entries into GitHub issues with labels (optional automation).
  - Priority: medium • Suggested owner: @maintainer
