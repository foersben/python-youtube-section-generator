# Development & Testing

This page consolidates development guidance: testing policy, scripts,
serialization/locking for LLM tests, and CI notes.

Testing policy

- Fast tests: unit tests, linters and small integration tests — run on GitHub
  hosted runners. Command: `poetry run pytest -q`
- Heavy LLM/RAG tests: expensive CPU-bound tests that require local models —
  gated and run on self-hosted runners labeled `self-hosted-llm`.

LLM test gating and serialization

- Heavy tests are marked with `@pytest.mark.llm` and are skipped by default.
- Guarded execution via environment variable: set `RUN_HEAVY_INTEGRATION=true` to allow heavy tests.
- Serialization: tests using local LLMs must acquire a file lock to prevent
  concurrent model loads (use the `filelock` package). A common lock path is
  `models/.llm_load.lock`.

CI overview

- `fast-tests` job runs on `ubuntu-latest` and executes unit tests + lint
- `heavy-integration` job runs only on self-hosted runners labeled
  `self-hosted-llm` and requires `RUN_HEAVY_INTEGRATION=true`

Example local CI using `act`

- Install Docker and `act`.
- Run a single job locally:

```bash
act -j fast-tests --container-architecture linux/amd64
```

Scripts

- Scripts live under `scripts/` and are organized into `tools`, `verify`,
  `dev`, and `obsolete`.
- `scripts/setup_interactive.py` — interactive installer (planned)
- `scripts/download_model.sh` — model download helper
- `scripts/tools/todos_to_issues.py` — utility to create GitHub issues from `docs/prospects/todos.md` (dry-run by default)

Developer checklist

- Run `poetry install` and `poetry run pre-commit install` (if pre-commit configured)
- Run `poetry run pytest` locally before pushing
- Keep heavy tests gated and mark heavy LLM tests with `@pytest.mark.llm`

See `docs/legacy/development/` for original, more detailed dev notes.

