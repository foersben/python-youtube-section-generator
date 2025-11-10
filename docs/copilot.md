# GitHub Copilot & Reusable Prompts

This page documents the project's GitHub Copilot prompt library, how to use the prompts,
and how Copilot is integrated into the repository and workflows.

## Overview

The repository includes a set of reusable prompts stored under `.github/prompts/` that
help standardize development tasks and speed up common edits while keeping them
aligned with project conventions (type hints, docstrings, logging, and error handling).

This documentation page was generated from the prompt README and expanded with
practical guidance for contributors.

## Prompt files

The `.github/prompts/` directory contains `.prompt.md` files. They are short templates
used by Copilot or developers as a guideline when authoring code. Typical prompt files:

- `flask-route.prompt.md` — prompts to create Flask route endpoints with validation and error handling
- `service-module.prompt.md` — templates for creating service modules (external API clients)
- `type-hints.prompt.md` — guidance to convert older code to Python 3.11+ style type hints
- `refactor.prompt.md` — instructions for refactor tasks and recommended code organization

### How to use a prompt

1. Open the relevant `.prompt.md` file in `.github/prompts/`.
2. Read and adapt the instructions to your task.
3. Use the prompt as context for Copilot or another assistant when generating code.
4. Follow project coding guidelines (type hints, Google-style docstrings, logging).

## Project-specific conventions used in prompts

- Python 3.11+ built-in generics (`list[str]`, `dict[str, Any]`), not `typing.List`.
- Google-style docstrings for modules, classes, and functions.
- Always use `logging` for errors and events; avoid bare `print()`.
- Use path-based file I/O with `pathlib.Path` and specify `encoding='utf-8'`.
- Tests live under `tests/` and heavy tests are marked `@pytest.mark.llm`.

## Copilot & workflow integration

We integrate the Copilot prompt files into developer workflows via documentation and
by providing explicit templates. This keeps generated code consistent.

Additionally, the repository contains GitHub Actions workflows that run tests and CI.
The relevant workflow files are:

- `.github/workflows/ci.yml` — runs fast tests on GitHub-hosted runners and heavy tests on `self-hosted-llm` runners.

### Recommended usage pattern

- Use Copilot with a prompt file open to get suggestions that match project patterns.
- Always lint and test generated code locally before committing.
- For heavy changes (LLM or RAG), adjust `.env` and run heavy tests on a proper self-hosted runner.

## Security & privacy

- Prompts may contain hints to reproduce local setup but do not store secrets.
- Never embed API keys or secrets within prompt files; use environment variables and `.env`.

## Example prompts

Each `.prompt.md` typically contains:
- A short task description
- Required inputs and outputs
- Example snippets matching project structure
- Test suggestions and validation checks

## Extending the prompt library

- Add new `.prompt.md` files to `.github/prompts/` following existing patterns.
- Keep prompts small and focused (one task per prompt).
- Include example inputs/outputs and testing instructions.

## Related documentation

- `docs/development/testing.md` — testing policies and heavy test guidance
- `docs/development/scripts.md` — scripts and helpers
- `.github/workflows/ci.yml` — CI configuration (fast and heavy tests)


