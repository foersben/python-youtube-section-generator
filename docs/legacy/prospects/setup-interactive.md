# Interactive setup via Poetry (implementation plan)

Goal

Provide an interactive setup script (invoked via `poetry run setup_interactive`) that:

- Installs missing dependencies (Poetry manages pyproject, pip installs optional heavy packages if requested)
- Validates presence of local model files and offers to download or configure paths
- Prompts for API keys (DEEPL_API_KEY, GOOGLE_API_KEY) and writes a `.env` template
- Offers advanced mode (hidden by default) for power users to adjust batch sizes, GPU layer counts, LLM parameters
- Supports a non-interactive `--yes` mode for automation/CI

UX flow

1. Welcome message and explanation of what will be configured
2. Check Python version and Poetry availability
3. Run `poetry install` to get base dependencies
4. Check for local model file: `models/Phi-3-mini-4k-instruct-q4.gguf`:
   - If missing, offer to run `scripts/download_model.sh` or skip
5. Check environment variables (`.env` or system env): list missing keys and offer to set them interactively
6. Ask about optional RAG dependencies (sentence-transformers, chromadb) and optionally `poetry run pip install` them (or instruct the user to run a follow-up command)
7. Advanced options (toggle): batch sizes, refinement batch size, LLM_GPU_LAYERS, REFINEMENT_BATCH_SIZE
8. Print summary and instructions to restart the web app or use the CLI

Implementation notes

- Script location: `scripts/setup_interactive.py` (executable via `poetry run setup_interactive` if added to `pyproject.toml` scripts)
- Use `questionary` or `inquirer` for interactive prompts (optional dependency)
- For non-interactive CI: accept `--yes` plus environment overrides
- Respect user privacy: do not store API keys in git; write `.env` and remind user to secure it

Security & safety

- Never echo API keys to stdout in logs unless explicitly asked
- Use file permissions when writing `.env` (e.g., `chmod 600 .env`)
