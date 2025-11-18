# Python YouTube Transcript Section Generator

AI-powered YouTube transcript section generator. Extracts transcripts, optionally refines them with a local or cloud LLM, and generates timestamped sections and titles. Features: translation support, RAG-based refinement, local model support (GGUF), and a CLI + Flask web app.

This README focuses on a pragmatic, reproducible setup, CPU-only remediation, translation behavior (DeepL batching + cache + persisted cooldown), and useful commands for development and troubleshooting.

---

## Quick start (recommended)

1. Clone the repo and enter the project directory

```bash
git clone <repo-url>
cd python-youtube-section-generator
```

2. Create the virtual environment and install dependencies using Poetry

```bash
poetry install
```

3. Run the interactive setup (configures API keys, checks local model paths, offers CPU-only installs)

```bash
poetry run setup_interactive
```

4. Run tests to verify everything is working

```bash
poetry run pytest -q
```

5. Quick CLI demo

```bash
poetry run pythonyoutubetranscript --help
poetry run pythonyoutubetranscript <video_id_or_url>
```

6. Or start the web app (Flask)

```bash
poetry run python src/web_app.py
# open http://127.0.0.1:5000/ in your browser
```

---

## Interactive setup: what it does

Run `poetry run setup_interactive` after `poetry install`. The interactive setup:

- Checks for the configured local GGUF model(s) and can download them if you opt in
- Prompts for API keys (DEEPL_API_KEY, GEMINI_API_KEY) and writes them to a `.env` (you can manage .env manually instead)
- Offers to install pip-only dependencies (CPU-only torch wheel, sentence-transformers, chromadb)
- Provides an advanced mode for experts (local model path, batch sizes, optional flags)

Notes:
- The script will ask whether to install CPU-only ML wheels; if you choose yes, it will install with `--index-url https://download.pytorch.org/whl/cpu` to avoid GPU wheels.
- If you have already downloaded the model into `models/` the setup will detect that and skip downloading.

---

## Configuration (.env)

Key environment variables (set them in `.env` or export in your shell):

- REFINE_TRANSCRIPTS=true/false — enable LLM transcript refinement
- REFINEMENT_BATCH_SIZE=50 — segments per LLM batch (50 recommended for CPU)
- USE_LOCAL_LLM=true/false — use local GGUF model instead of cloud API
- LOCAL_MODEL_PATH=models/Phi-3-mini-4k-instruct-q4.gguf — relative to project root
- LOCAL_MODEL_4BIT=true/false — enable 4-bit quantized local model usage
- DEEPL_API_KEY=... — DeepL API key (recommended for non-English transcripts)
- GEMINI_API_KEY=... — Google Gemini / cloud LLM key (optional)

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

## CPU-only remediation & reproducible install

If the environment has GPU-contaminated wheels (CUDA-enabled torch, llama bindings), follow this protocol to achieve a verifiable CPU-only stack.

1. Aggressive uninstall (run inside the project's virtualenv):

```bash
pip uninstall -y torch torchvision torchaudio llama-cpp-python || true
pip cache purge || true
```

2. Install CPU-only PyTorch wheel (Linux example):

```bash
poetry run pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
```

3. Install CPU-only llama-cpp-python (prefer prebuilt CPU wheel if available) and other deps:

```bash
poetry run pip install llama-cpp-python==0.3.16 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --no-cache-dir
poetry run pip install sentence-transformers==5.1.2 chromadb --no-cache-dir
```

4. Reinstall project packages (will skip already-installed pinned packages):

```bash
poetry run pip install -e . --no-cache-dir
```

5. Verify CPU-only with our diagnostic script:

```bash
python scripts/check_environment.py > before_fix.log
# after remediation
python scripts/check_environment.py > after_fix.log
diff before_fix.log after_fix.log
```

This project includes `scripts/decontaminate_cpu.sh` and `scripts/check_environment.py` to automate much of the above.

---

## Translation behavior (important)

The extractor prefers the following order when dealing with non-English transcripts:

1. YouTube server-side features: attempt `YouTubeTranscriptApi.fetch(languages=[...])` and `Transcript.translate()` (server-side). This avoids external API calls when YouTube can produce the translation.
2. Batched DeepL fallback: if server-side translation is unavailable, the project will attempt a batched DeepL translation. Batching reduces request count and costs. Batching details:
   - Inputs are chunked into batches of up to 100 segments OR ~10,000 characters per batch, whichever is first.
   - Each batch is sent in a single request (joined with a separator) and split back into per-segment translations on return.
   - If DeepL responds with quota errors (HTTP 456 or quota messages), the adapter persists a cooldown and raises an internal `TranslationQuotaExceeded`. The extractor then falls back to local Llama translation if configured.
3. Local Llama fallback: as a last resort a configured local GGUF model (via `LlamaCppTranslator`) will be used to translate segments locally.

Caching & cooldown:
- Translated transcripts are cached to disk at `.cache/translations/{video_id}_{lang}.json` to avoid re-translating the same video repeatedly.
- DeepL quota cooldown is persisted to `.cache/translations/deepl_quota.json` and will prevent repeated DeepL attempts for a short period (defaults to 1 hour after a quota exceed event).

If you rely on DeepL for quality, consider upgrading your DeepL plan, or rely on the local LLM fallback and translation cache.

---

## Clearing caches and forcing fresh translations

```bash
# list caches
ls -la .cache/translations

# clear caches (force full re-translation next run)
rm -rf .cache/translations
```

---

## Troubleshooting

- "YouTube blocking requests" — YouTube may block requests from cloud IPs or rate-limited IPs. See `scripts/check_environment.py` logs. Workarounds: use local IP, proxy, or rely on manually uploaded transcripts.
- DeepL quota errors (HTTP 456) — the app now persists a cooldown to avoid repeated calls and will fall back to local model translation. To resume DeepL usage, wait for the cooldown or clear `.cache/translations/deepl_quota.json`.
- Local model not found — ensure `LOCAL_MODEL_PATH` points to an existing .gguf under `models/` and that `USE_LOCAL_LLM=true` if you intend to use it.
- If the web app fails to start due to import errors like "No module named 'src.core.models'", ensure you are running with project root as working directory (the repo root) and that `poetry run` is used or the PYTHONPATH includes project root.

---

## Developer notes

- Tests: `poetry run pytest -q`
- Linting & typechecks: `poetry run ruff .` and `poetry run mypy src` (if installed)
- When updating dependencies, run `poetry lock` then `poetry install` and commit `poetry.lock` when stable.

CI / workflows
- The repository contains workflows under `.github/workflows/` for CI, build, docs, and tests. Consider enabling caching for Poetry venvs and CPU wheel caches to speed up runs.

---

## Contributing

Contributions welcome — please open issues or PRs. Follow the project's code style (type hints, docstrings, tests) and place tests under `tests/`.

---

## License

MIT
