# Project README (docs copy)

The content below mirrors the repository `README.md` for inclusion in the docs site.

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

## Configuration (.env)

Key env vars are documented in the root README; keep `LOCAL_MODEL_PATH` and keys out of Git.
