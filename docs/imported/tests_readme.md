# Tests — Imported LLM Serialization and Heavy Test Policy

This page was imported from `tests/README.md` and documents the test policy and locking behavior.

Policy & behavior

- Heavy tests that touch local LLMs are serialized to avoid concurrent model loads that can OOM the machine.
- The test runner uses a file lock at `models/.llm_load.lock` to serialize model initialization and usage across processes/workers.
- The project requires `filelock` (cross-platform) for locking; tests will raise a helpful error if it's not installed.

Locking policy (strict)

A test will be serialized (acquire the lock) only if one of the following applies:

- It is explicitly marked with the `@pytest.mark.llm` marker.
- The environment variable `RUN_HEAVY_INTEGRATION` is set to `1`, `true`, or `yes` (global opt-in).

This stricter policy prevents accidental serialization and makes heavy tests explicit and auditable.

Opt-in / opt-out

- To run fast unit tests only (default):

```bash
poetry run pytest -q
```

- To run heavy integration tests (explicit opt-in):

```bash
export LOCAL_MODEL_PATH=./models/Phi-3-mini-4k-instruct-q4.gguf
export RUN_HEAVY_INTEGRATION=true
poetry run pytest -q
```

- To mark a test explicitly as heavy:

```python
import pytest

@pytest.mark.llm
def test_my_heavy_llm():
    # heavy test that will acquire the LLM lock
    ...
```

Notes & troubleshooting

- Ensure `filelock` is installed (it's in `pyproject.toml`) and run `poetry install` if needed.

```bash
poetry install
```

- If a test seems to hang waiting for a lock, confirm no other process is holding `models/.llm_load.lock` and inspect which process via `lsof` or similar tools.

- The lock location can be overridden by setting `LOCAL_MODEL_PATH` to a different model path; the lock will live in the model's parent directory.

