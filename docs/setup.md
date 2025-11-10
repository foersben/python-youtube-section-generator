# Setup & Interactive Installer

Overview

This page explains how to set up the project for local development, why the
choices were made (Poetry, local LLM, batching), and how the interactive setup
streamlines environment configuration.

Why Poetry?

- Reproducible dependency resolution (lock file)
- Built-in groups/extras for dev vs runtime
- Simple publishing workflow if the project becomes a distributable package

Why local models by default?

- Privacy: transcript text never leaves the machine
- Cost control: zero API token usage for title refinement
- Deterministic performance envelope on CPU hardware

When to prefer cloud (Gemini)?

- Faster per-request latency if network is reliable
- Lower RAM footprint for users without adequate memory
- Access to larger context windows or more capable models

Prerequisites

- Python 3.11+ (3.12 tested)
- Git
- Optional: Docker (for `act` GitHub Actions simulation)

Quick install

1. Clone the repo and enter the project directory:

```bash
git clone <repo-url>
cd PythonYoutubeTranscript
```

2. Install base dependencies:

```bash
poetry install
```

3. (Optional) Install CPU-only PyTorch & RAG dependencies:

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
poetry run pip install sentence-transformers chromadb
```

Interactive installer (design goals)

- Minimize friction for first-time users
- Detect missing local model and offer to download
- Collect API keys securely (no echo back) and write `.env`
- Offer advanced tuning only when explicitly requested

Planned flow (summary)

1. Base dependency install
2. Model presence check (`LOCAL_MODEL_PATH`)
3. Offer model download if missing (shell script or HF hub)
4. Prompt for DeepL / Gemini keys (masked input)
5. Advanced toggles (batch sizes, pipeline variant)
6. Write `.env` with only supplied keys and defaults
7. Post-setup sanity checks and next-step hints

.env essentials

```
LOCAL_MODEL_PATH=models/Phi-3-mini-4k-instruct-q4.gguf
USE_LOCAL_LLM=true
USE_TRANSLATION=true
USE_LLM_TITLES=true
RAG_HIERARCHICAL=true
```

Add keys only if you use services:

```
DEEPL_API_KEY=your_deepl_key
GOOGLE_API_KEY=your_gemini_key
```

Security considerations

- Do not commit `.env` (gitignore enforced)
- Use `chmod 600 .env` on shared machines
- Never embed API keys into scripts or docs examples

Model download rationale

A scripted download avoids manual mistakes (wrong quantization variant,
partial file). Integrity checks (size/hash) reduce runtime confusion.

Performance tuning knobs (exposed via env)

- `LLM_TEMPERATURE`: lower (0–0.2) for deterministic titles
- `RAG_HIERARCHICAL`: enable multi-level sectioning
- `USE_RAG`: `auto|always|never` for retrieval overhead control

Troubleshooting quick table

| Symptom | Likely Cause | Action |
|---------|--------------|--------|
| Model fails to load | Wrong path / missing file | Re-run download script |
| Titles look truncated | Translation skipped | Check DeepL quota / local translator fallback logs |
| High memory usage | Multiple heavy tests | Gate with `RUN_HEAVY_INTEGRATION` & ensure lock |
| Very slow refinement | Batch size too small | Increase refinement batch or disable titles temporarily |

References

- Legacy setup notes: `docs/legacy/setup.md`
- Interactive plan details: `docs/legacy/prospects/setup-interactive.md`
