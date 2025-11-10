# GitHub Copilot & Reusable Prompts

This page documents the project's GitHub Copilot prompt library, how to use the prompts,
and how Copilot is integrated into the repository and workflows. It also explains
how the project's GitHub Actions workflows are structured (fast tests vs heavy
LLM tests) and how to run heavy tests on a self-hosted runner.

Quick checklist (what you'll learn on this page)

- Where prompts live and how to use them
- A reusable prompt template you can adopt
- Best practices for prompts and generated code
- How CI is set up: fast tests (GitHub-hosted) and heavy tests (self-hosted)
- How to provision a self-hosted runner for heavy LLM tests
- Security and privacy notes

---

## 1. Overview

The repository provides small, focused prompt files under `.github/prompts/`.
Prompts are intended as deterministic instructions for Copilot or a human
reviewer to produce code that follows project conventions (type hints,
Google-style docstrings, logging policy, testability).

Prompts are *not* replacements for code review. Always review generated code.

---

## 2. Where prompts live

- Prompt files: `.github/prompts/*.prompt.md`
- Documentation landing page: `docs/copilot.md` (this page)

If you add new prompts, create a single-file `.prompt.md` with a short title,
inputs, expected outputs, and a tiny example.

---

## 3. How to use a prompt (practical steps)

1. Open the appropriate `.prompt.md` file in your editor.
2. Read the instructions and the example.
3. Use Copilot (or your assistant) with that file visible so context is available.
4. Accept or edit suggestions, then run linters and tests locally.

Quick local verification commands

```bash
# Install project (if not done):
poetry install

# Run fast unit tests (skips heavy LLM tests):
poetry run pytest -q

# Build the docs locally to preview the prompt and docs pages:
poetry run mkdocs serve
```

---

## 4. Prompt template (recommended)

Use this small, repeatable template as the top of any `.prompt.md`:

```text
Title: <one-line task title>

Goal:
  - Describe the goal in 1-2 short sentences.

Inputs:
  - List of inputs and types (e.g. transcript: list[dict[str, Any]])

Output:
  - Describe expected output shape (e.g. list[dict[str, Any]] with keys title/start)

Constraints / Conventions:
  - Use Python 3.11 built-in generics (list[str], dict[str, Any])
  - Google-style docstrings
  - Type annotate all function signatures
  - Use logging (logger = logging.getLogger(__name__))

Example (minimal):
  # short example input -> output mapping

Tests:
  - Suggest 1-2 lightweight tests that validate the result
```

Keep prompts focused — one task per prompt.

---

## 5. Best practices for generated code

- Always run `ruff`/`black`/`mypy` and `pytest` locally before committing.
- Avoid hardcoding secrets; use environment variables and `.env`.
- Add a short unit test for any non-trivial logic you introduce.
- Prefer small helper functions with single responsibility over monoliths.
- Add Google-style docstrings for public functions and classes.

---

## 6. Example prompt (concrete)

Title: Generate a 2-4 word section title from a snippet

Goal:
  - Produce a concise, noun-based section title in English for a given transcript

Inputs:
  - snippet: str  # a text excerpt (100-300 chars)

Output:
  - title: str  # 2-4 words, nouns preferred, no punctuation

Constraints:
  - Use English when generating titles (we translate DE->EN first in the pipeline)
  - Keep titles short, 2-4 words

Example:
  Input: "In this section we discuss university policy and student councils..."
  Output: "Student Council Policy"

Tests:
  - Assert the output is 1-4 words and letters/numbers only (no newlines)

---

## 7. Integration with GitHub Actions (CI)

We use a two-tier CI model (see `.github/workflows/ci.yml`):

1. Fast tests (GitHub-hosted runners)
   - Runs on push and pull requests.
   - Installs dependencies and runs fast unit tests (`poetry run pytest -q`).
   - Skips heavy LLM/RAG tests by default (these are marked with `@pytest.mark.llm`).

2. Heavy tests (self-hosted runner with label `self-hosted-llm`)
   - Triggered manually (`workflow_dispatch`) or on schedule.
   - Runs only on a self-hosted runner that has the local GGUF model and sufficient resources.
   - Enables heavy tests by setting `RUN_HEAVY_INTEGRATION=true` and runs only tests marked `llm`:
     - `poetry run pytest -q -m llm`

Why this split?
- Fast feedback for typical PRs (unit tests + lint) without expensive model loads.
- Heavy LLM/RAG tests run only on machines that can handle the memory/CPU (local
  model weights are large and CPU inference is slow).

### Where to find the workflow
- File: `.github/workflows/ci.yml`
- Fast tests job: `fast-tests`
- Heavy tests job: `heavy-tests-self-hosted`

You can view the CI implementation in the repository to see exact steps and
environment variables used.

---

## 8. Self-hosted runner setup (quick guide)

If you want to run heavy LLM tests in CI, prepare a self-hosted runner:

1. Provision a machine or VM with enough RAM (recommended 16+ GB for non-quantized models; 5-8 GB with 4-bit quantization) and CPU/GPU as needed.
2. Install the GitHub Actions runner software and register it with your repository. See: https://docs.github.com/en/actions/hosting-your-own-runners
3. Add the runner labels `self-hosted` and `self-hosted-llm` (or use an existing label).
4. Ensure the machine has the local model weights placed at the expected `LOCAL_MODEL_PATH` (or set the env var in runner config).
5. Install Python, Poetry and run `poetry install` on the runner so the environment builds quickly.
6. Optionally, set `LOCAL_MODEL_PATH` and `USE_LOCAL_LLM=true` in the runner's environment variables (via the runner service or system env) so workflows find the model.

Notes:
- Keep runner access restricted to trusted CI jobs and trusted users — self-hosted runners execute arbitrary job steps.
- Regularly update the runner software and clean large caches to avoid disk saturation.

---

## 9. Security, privacy and secrets

- Prompts should never contain secrets. Use environment variables for API keys.
- Do not commit sensitive model artifacts to the repository. Keep weights outside Git.
- Use the repository's Settings > Secrets to store any secret tokens used in CI.

---

## 10. Contributing new prompts

1. Add a new file to `.github/prompts/` using the `.prompt.md` extension.
2. Follow the prompt template (see section 4).
3. Add a short example and suggested test(s).
4. Open a PR adding the prompt and a short note in `docs/copilot.md` or a new docs page if the prompt is large.

---

## 11. FAQ

Q: Where do I run the generated code before pushing?
A: Locally — use Poetry and the venv created by `poetry install`, run linters and the fast test suite.

Q: How do I enable heavy tests locally?
A: Set `RUN_HEAVY_INTEGRATION=true` and ensure `LOCAL_MODEL_PATH` points to a valid GGUF model.

---

## 12. Related pages

- `docs/development/testing.md` — testing policy and how the LLM tests are gated
- `.github/workflows/ci.yml` — CI pipeline
- `docs/imported/scripts_readme.md` — imported scripts README
