# Development & Testing

Purpose & Philosophy

Development practices aim for fast feedback (unit/lint) and controlled heavy
runs (LLM/RAG) to avoid wasting resources and causing local system strain.

Testing tiers (why separate?)

- Fast tier: ensures correctness of pure Python logic and lightweight services.
  These tests give near-instant feedback on PRs (<1 min typical) and catch most
  regressions early.
- Heavy tier: exercises integration of local model loading, RAG indexing,
  multi-stage refinement. Running them on-demand prevents timeouts and resource
  contention (LLM loads are expensive on CPU).

Heaviness indicators

- Large model load (GGUF via llama.cpp)
- Vector store creation (ChromaDB persistence)
- Extensive prompt chaining or refinement loops

Serialization rationale

Multiple concurrent heavy tests can overrun memory or degrade performance.
Using a file lock around LLM access guarantees only one model load/refinement
sequence at a time, avoiding OOM and preventing skewed timings.

Environment flags

- `RUN_HEAVY_INTEGRATION=true` — global opt-in to heavy tests
- `LOCAL_MODEL_PATH` — path to GGUF model (validated before heavy tests run)

LLM test marker

```python
import pytest

@pytest.mark.llm
def test_heavy_title_generation():
    # heavy test code
    ...
```

CI design (why two jobs?)

- `fast-tests`: runs on GitHub-hosted runners — universal reliability, no
  special requirements.
- `heavy-integration`: runs only where the model is present — prevents flaky
  failures due to missing model weights or insufficient RAM.

Local CI with `act` (benefits)

- Validate YAML logic and dependency install without consuming hosted minutes.
- Rapid iteration on workflow steps (e.g., adding caching or matrix changes).

Scripts grouping (why this structure?)

- `tools/` — direct setup & maintenance actions (download model, todos tool)
- `verify/` — diagnostics (GPU detection, config verification)
- `dev/` — developer productivity helpers (profiling, debug harnesses)
- `obsolete/` — archived one-off or replaced artifacts to keep root clean

Checklist before pushing

1. `poetry install` — dependencies in sync
2. `poetry run pytest -q` — fast tier green
3. Optional: `RUN_HEAVY_INTEGRATION=true poetry run pytest -m llm -q` when
   changing LLM/RAG code
4. Format & lint: (if configured) `black`, `ruff`, `mypy`
5. Update docs/changelog for user-facing changes

Failure diagnostics

- If tests hang: inspect lock file usage (`lsof models/.llm_load.lock`)
- If memory errors occur: ensure only one heavy test running; reduce context or
  use smaller model variant.

References

- Legacy development details: `docs/legacy/development/*.md`
