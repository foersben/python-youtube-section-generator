# Testing

Run fast tests locally:

```bash
poetry run pytest -q
```

Run heavy LLM/RAG tests (requires local model and resources):

```bash
export LOCAL_MODEL_PATH=./models/Phi-3-mini-4k-instruct-q4.gguf
export RUN_HEAVY_INTEGRATION=true
poetry run pytest -q -m llm
```

Guidelines:
- Heavy tests are marked with `@pytest.mark.llm` and are skipped by default.
- We serialize local model loads with a file lock (`models/.llm_load.lock`) to avoid concurrent loads.
