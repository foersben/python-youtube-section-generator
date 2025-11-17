# Setup (interactive installer)

This file contains the consolidated plan and todos for the interactive `poetry`
setup and environment validation.

Summary of tasks

- Implement `scripts/setup_interactive.py` (or `scripts/setup_interactive.sh`) to:
  - Run `poetry install` (base deps)
  - Validate or download local model(s)
  - Validate/transparently set `.env` entries
  - Offer advanced options (hidden) for power users
  - Provide a `--yes` mode for CI environments

Implementation notes

- Use `questionary` (or `inquirer`) for prompts if interactive; otherwise use
  command line args for non-interactive flows.
- Do not commit secrets: write `.env` and remind the user to secure it.
- For optional heavy packages (sentence-transformers, chromadb), present a
  separate checkbox prompt and `poetry run pip install` them if selected.

Todos

- [ ] Create `scripts/setup_interactive.py` with the flows above (priority: high)
- [ ] Add `poetry` script entry in `pyproject.toml` to expose `setup_interactive` (priority: medium)
- [ ] Add `--yes` and `--download-models` flags (priority: high)
- [ ] Add unit tests for the validation logic (priority: medium)
