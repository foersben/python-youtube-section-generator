# Heavy Tests & Self-Hosted Runner Requirements

This page documents requirements and steps to run heavy LLM/RAG tests that
cannot be executed on GitHub-hosted runners.

Minimum requirements (recommended)

- RAM: 8 GB (quantized models) — 16+ GB recommended for larger models
- Disk: 10 GB free
- Python: 3.11+
- Model files: place GGUF files under `models/` and set `LOCAL_MODEL_PATH` in the runner env

Runner setup

1. Install Python and Poetry
2. Install dependencies: `poetry install --with dev`
3. Download models to the runner's `models/` directory (see `scripts/download_model.sh`)
4. Add a self-hosted runner in GitHub and label it `self-hosted-llm`
5. Expose `LOCAL_MODEL_PATH` and `RUN_HEAVY_INTEGRATION=true` as runner-level env vars

CI gating

- The `heavy-integration` job should run only on runners with label `self-hosted-llm` and only when the env var `RUN_HEAVY_INTEGRATION=true` is set.
- Tests that require local LLM should be marked with `@pytest.mark.llm`
- Use a file-based lock (e.g., `filelock`) to serialize LLM loads across tests

Recommended workflow

- Keep `fast-tests` on GitHub-hosted runners for PR feedback
- Run `heavy-integration` on a scheduled cadence or manually when needed


