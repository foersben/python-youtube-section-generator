# Tests — LLM Serialization and Heavy Test Policy

This repository contains unit tests and several heavier integration-style tests that depend on local LLM models (GGUF) and CPU inference.

Policy & behavior

- Heavy tests that touch local LLMs are serialized to avoid concurrent model loads that can OOM the machine.
- The test runner uses a file lock at `models/.llm_load.lock` to serialize model initialization and usage across processes/workers.
- The project prefers `filelock` (cross-platform) for locking; if `filelock` is not available it falls back to POSIX `fcntl`.

How tests are selected for locking

A test will be serialized (acquire the lock) if any of the following apply:

- It is explicitly marked with the `@pytest.mark.llm` marker.
- The environment variable `RUN_HEAVY_INTEGRATION` is set to `1`, `true`, or `yes`.
- The test file is located under `tests/scripts/`.
- A heuristic detects LLM-related tokens in the test source: `LocalLLMClient`, `LocalLLMProvider`, `LOCAL_MODEL_PATH`, or `USE_LOCAL_LLM`.

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

- Alternatively, mark tests explicitly:

```python
import pytest

@pytest.mark.llm
def test_my_heavy_llm():
    # heavy test that will acquire the LLM lock
    ...
```

Notes & troubleshooting

- If you're on Windows and the `filelock` package isn't installed, the fallback is an unsafe PID-file approach which is not cross-process safe. Install `filelock` (it is in `pyproject.toml`) and run `poetry install` to get proper cross-platform behavior.

```bash
poetry install
```

- If a test seems to hang waiting for a lock, confirm no other process is holding `models/.llm_load.lock` and inspect which process via `lsof` or similar tools.

- The lock location can be overridden by setting `LOCAL_MODEL_PATH` to a different model path; the lock will live in the model's parent directory.

