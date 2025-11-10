# Setup & Interactive Installer

Overview

This page describes how to set up the project for local development, the
interactive setup helper, and optional steps for CPU-only RAG/LLM workflows.

Prerequisites

- Python 3.11+ (3.12 tested)
- Git
- Docker (optional, recommended for `act` local CI)
- Poetry for dependency management

Quick install

1. Clone the repo and enter the project directory:

```bash
git clone <repo-url>
cd PythonYoutubeTranscript
```

2. Install base dependencies with Poetry:

```bash
poetry install
```

3. (Optional) Install CPU-only PyTorch and RAG dependencies (recommended for
   local RAG work):

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
poetry run pip install sentence-transformers chromadb
```

Interactive installer

We provide an interactive installer script to streamline first-run setup,
manage `.env` entries, and optionally download local models.

- Script: `scripts/setup_interactive.py` (invokable via `poetry run setup_interactive` if added to `pyproject.toml`)
- Features:
  - Runs `poetry install` (base deps)
  - Checks for local model files and offers to download them
  - Prompts for API keys (e.g., `DEEPL_API_KEY`) and writes a secure `.env`
  - Optional advanced mode for power users (batch sizes, pipeline selection)
  - `--yes` for non-interactive CI-friendly installs

.env and recommended variables

Create a `.env` at project root (do NOT check into git). Minimum useful vars:

```
LOCAL_MODEL_PATH=models/Phi-3-mini-4k-instruct-q4.gguf
USE_LOCAL_LLM=true
USE_TRANSLATION=true
DEEPL_API_KEY=your_deepl_key
RUN_HEAVY_INTEGRATION=false
```

Security note: use `chmod 600 .env` to limit file access.

Model download

A helper script exists to download the common local model used for CPU
inference (GGUF): `scripts/download_model.sh`. The interactive installer can
invoke it for you if you choose to download the model.

Troubleshooting

- If a model fails to load, verify `LOCAL_MODEL_PATH` and file permissions.
- For memory errors, prefer a 4-bit quantized model (q4) and ensure adequate
  swap or RAM.


