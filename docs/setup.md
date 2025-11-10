# Setup

This page explains how to set up the project for local development.

- Python 3.11+ recommended
- Use Poetry for dependency management

Steps:

1. Install poetry (https://python-poetry.org/)
2. Install dependencies:

```bash
poetry install
```

3. Optional: install CPU-only PyTorch and RAG dependencies (see scripts/setup_cpu_rag.sh)

4. Create a `.env` file with required environment variables (DEEP L keys, model paths):

```
LOCAL_MODEL_PATH=models/Phi-3-mini-4k-instruct-q4.gguf
USE_LOCAL_LLM=true
DEEPL_API_KEY=your_deepl_key
```

