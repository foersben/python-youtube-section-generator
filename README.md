# Python YouTube Transcript Section Generator

AI-powered YouTube transcript section generator. Extracts transcripts, optionally refines them with a local or cloud LLM, and generates timestamped sections and titles (supports translation + RAG-based refinement).

This README covers quick setup, the new interactive setup script, common configuration (.env), local-model notes, and usage commands.

---

## Quick start (recommended)

1. Clone the repo and enter the project directory

```bash
git clone <repo-url>
cd PythonYoutubeTranscript
```

2. Create a virtual environment and install dependencies with Poetry

```bash
poetry install
```

3. Run the interactive setup (prompts for API keys, local model checks, pip-only installs)

```bash
poetry run setup_interactive
```

4. Run tests to verify everything is working

```bash
poetry run pytest -q
```

5. Run the CLI for a quick demo

```bash
poetry run pythonyoutubetranscript --help
poetry run pythonyoutubetranscript <video_id_or_url>
```

6. Or start the web app

```bash
poetry run python src/web_app.py
# open http://127.0.0.1:5000/ in your browser
```

---

## What the interactive setup does

Run `poetry run setup_interactive` after `poetry install`. The setup will:

- Confirm presence of required local model files (downloads if you opt in)
- Ask for API keys (DeepL, Gemini) and write them to `.env` securely
- Offer to install pip-only dependencies (torch/cpu, sentence-transformers, chromadb)
- Offer advanced configuration (batch size, local model path, 4-bit quantization toggle)

If you prefer a non-interactive flow, set the environment variables in `.env` yourself (see below).

---

## Configuration (.env)

The interactive setup writes configuration to a `.env` file. Key variables used by the project:

- `REFINE_TRANSCRIPTS` (true/false) — enable LLM transcript refinement
- `REFINEMENT_BATCH_SIZE` (int) — segments per LLM batch (50 recommended for CPU)
- `USE_LOCAL_LLM` (true/false) — use local model instead of cloud
- `LOCAL_MODEL_PATH` — path to your local .gguf model (e.g. `models/Phi-3-mini-4k-instruct-q4.gguf`)
- `LOCAL_MODEL_4BIT` (true/false) — enable 4-bit quantized local model usage
- `DEEPL_API_KEY` — DeepL API key for translation (optional, recommended for non-English transcripts)
- `GEMINI_API_KEY` — Google Gemini API key (optional)

Example `.env` snippet:

```env
REFINE_TRANSCRIPTS=true
REFINEMENT_BATCH_SIZE=50
USE_LOCAL_LLM=true
LOCAL_MODEL_PATH=models/Phi-3-mini-4k-instruct-q4.gguf
LOCAL_MODEL_4BIT=true
DEEPL_API_KEY=your_deepl_key_here
GEMINI_API_KEY=your_gemini_key_here
```

---

## Local Models & CPU-only installation notes

Local LLM and heavy ML packages (torch, sentence-transformers, chromadb) are optionally installed via `pip` because Poetry may otherwise pull GPU-enabled builds from PyPI.

Recommended CPU-only install after `poetry install` (run inside the virtual environment):

```bash
# CPU-only PyTorch wheel (Linux example):
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Then install sentence-transformers and chromadb
poetry run pip install sentence-transformers chromadb
```

If you have a GPU and want CUDA-enabled torch, install the appropriate wheel from the official PyTorch instructions instead.

Model download tip: use the included script to download recommended models

```bash
# downloads into ./models/
./scripts/download_model.sh Phi-3-mini-4k-instruct-q4.gguf
```

---

## Advanced options (for experts)

The interactive setup offers advanced options. If you prefer manual configuration, set these in `.env` directly.

- `REFINEMENT_BATCH_SIZE`: Default 50 (increase for speed if memory allows)
- `LOCAL_MODEL_4BIT`: Use `true` for 4-bit quantized models to reduce memory usage
- `LLM_GPU_LAYERS`: `-1` to auto-use GPU layers (if using GPU)
- `USE_TRANSLATION`: `true`/`false` — enable DE→EN→DE pipeline for best title quality on non-English transcripts

---

## Troubleshooting

- If refinement seems skipped, ensure `REFINE_TRANSCRIPTS=true` in `.env` and restart the app (the app loads .env at startup).
- If section titles are poor for non-English content, enable `DEEPL_API_KEY` or set `USE_TRANSLATION=true` so the DE→EN→DE pipeline runs.
- If Poetry shows warnings about `[tool.poetry]` vs `[project]`, these are deprecation warnings and safe to ignore; your project is configured to work with current Poetry releases.

## Poetry warnings & lockfile (troubleshooting)

You may see *deprecation* warnings from `poetry check` like:

```
Warning: [tool.poetry.version] is set but 'version' is not in [project.dynamic].
Warning: [tool.poetry.description] is deprecated. Use [project.description] instead.
Warning: Defining console scripts in [tool.poetry.scripts] is deprecated. Use [project.scripts] instead.
```

These are informational deprecation messages from newer Poetry versions recommending PEP 621 ([project]) metadata. They are safe to ignore — your project still works — but if you want to remove them you can migrate metadata to the `[project]` table (optional).

If `poetry check` reports errors about missing dev extras or stale lockfile entries (for example: "Cannot find dependency X for extra dev"), regenerate the lockfile and reinstall:

```bash
# Regenerate lock file (resolves mismatches between pyproject and poetry.lock)
poetry lock

# Install dependencies
poetry install

# Re-run checks
poetry check
```

If you intentionally removed or moved the `dev` extras, `poetry.lock` may still reference them — `poetry lock` will refresh the lock to match `pyproject.toml`.

If you prefer a clean start (useful when switching dependency layout):

```bash
rm -f poetry.lock
poetry lock
poetry install
```

Notes:
- Keep a copy of `poetry.lock` in version control after you finish updates.
- The project uses some dev tools (pytest, black, ruff, mypy) and optional heavy ML packages (torch, sentence-transformers, chromadb) that are installed separately for CPU/GPU choices.

---

## Developer notes

- Run `poetry lock` if you update `pyproject.toml` to refresh the lockfile.
- Run `poetry run pytest -q` for the test suite.

---

## License

MIT
