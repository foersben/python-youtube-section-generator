# Python YouTube Transcript

[![CI](https://img.shields.io/github/actions/workflow/status/your-org/your-repo/ci.yml?branch=main)](../.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#)

Welcome to the documentation for the Python YouTube Transcript project.

This project extracts YouTube transcripts, optionally refines them with local or
cloud LLMs, translates when needed, and generates timestamped sections and
high-quality titles.

---
## Getting Started

```bash
git clone <repo-url>
cd PythonYoutubeTranscript
poetry install
poetry run python src/web_app.py  # start the Flask app
```

Optional CPU-only ML dependencies:

```bash
poetry run pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
poetry run pip install sentence-transformers chromadb
```

Interactive setup (planned):

```bash
poetry run setup_interactive  # guided model/key configuration
```

---
## Core Concepts

- Architecture & pipeline: see `architecture.md` (why staged, fallbacks, i18n)
- Development & testing: `development.md` (fast vs heavy tiers, locking rationale)
- Prospects & roadmap: `prospects.md` (priorities + strategic direction)
- Troubleshooting: `troubleshooting.md` (common failure patterns & fixes)

Advanced deep dives live in `advanced.md` (refinement, RAG details, heuristics).

---
## Quick Links

- Setup: `setup.md`
- Architecture: `architecture.md`
- Development: `development.md`
- Advanced: `advanced.md`
- Prospects: `prospects.md`
- Contributing: `contributing.md`
- Changelog: `changelog/index.md`

---
## Why Staged Processing?

Small local models produce better, more deterministic outputs when complex tasks
are decomposed. Translation-first enables English-centric prompt optimizations
while preserving original language via back-translation.

---
## Next Steps

- Run fast tests: `poetry run pytest -q`
- Enable heavy tests: `RUN_HEAVY_INTEGRATION=true poetry run pytest -m llm -q`
- Use `act` to iterate on CI locally.

---
## License

MIT
