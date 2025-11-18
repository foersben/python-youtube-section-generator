# GitHub Copilot & Reusable Prompts

This page documents the project's GitHub Copilot prompt library, how to use prompts
consistently, and how Copilot fits into our CI / developer workflow. It also
explains the two-tier CI model used here (fast tests vs heavy LLM/RAG tests) and
how to run heavy tests on a self-hosted runner.

Table of contents

- [Overview](#1-overview)
- [Where prompts live](#2-where-prompts-live)
- [How to use a prompt (quick)](#3-how-to-use-a-prompt-quick)
- [Prompt template (recommended)](#4-prompt-template-recommended)
- [Best practices for generated code](#5-best-practices-for-generated-code)
- [Concrete example prompt](#6-concrete-example-prompt)
- [Integration with GitHub Actions (CI)](#7-integration-with-github-actions-ci)
- [Running workflows locally with act](#8-running-workflows-locally-with-act)
- [Self-hosted runner setup (quick)](#9-self-hosted-runner-setup-quick)
- [Security, licensing, and privacy notes](#10-security-licensing-and-privacy-notes)
- [Contributing new prompts](#11-contributing-new-prompts)
- [FAQ and quick commands](#12-faq-and-quick-commands)

---

## 1. Overview

Prompts are small, focused instruction files that help generate code snippets or
boilerplate that conform to the project's style and architecture. They live in
`.github/prompts/` and are intended to be used as context for Copilot or any
assistant. Prompts are guidance — not a substitute for careful review and tests.

Keep prompts concise, single-purpose, and example-driven.

---

## 2. Where prompts live

- Directory: `.github/prompts/`
- Filename convention: `*.prompt.md`
- This docs page: `docs/copilot.md`

When you add a prompt, include: title, short goal, inputs/outputs, constraints,
one compact example, and 1–2 suggested tests.

---

## 3. How to use a prompt (quick)

1. Open the `.prompt.md` file relevant to your task in the editor.
2. Keep the prompt visible to Copilot so generated content takes the prompt
   context into account.
3. Accept or iterate on suggestions and immediately run linters and tests.

Quick verification commands

```bash
# install deps (first time)
poetry install

# run fast tests (unit, lint — heavy LLM tests skipped by default)
poetry run pytest -q

# build or serve docs locally
poetry run mkdocs serve
```

---

## 4. Prompt template (recommended)

Copy this header into new `.prompt.md` files and fill the sections:

```text
Title: <one-line task title>

Goal:
  - Short description (1-2 lines)

Inputs:
  - name: type — brief description

Output:
  - name: type — brief description

Constraints / Conventions:
  - Python 3.11 built-ins: `list[str]`, `dict[str, Any]` (no `typing.List`)
  - Google-style docstrings
  - Type annotate public functions
  - Use `logging.getLogger(__name__)`

Example (minimal):
  # show input -> expected output mapping

Tests:
  - suggestion: pytest assertion(s) to validate behavior
```

Keep prompts narrow — one behavior per file.

---

## 5. Best practices for generated code

- Run `ruff`/`black`/`mypy` and `pytest` locally before committing.
- Avoid embedding secrets in prompts or generated code. Use environment variables.
- Add tests for non-trivial logic and a docstring describing inputs/outputs.
- Prefer small testable helper functions; keep functions short and focused.

---

## 6. Concrete example prompt

**Title**: Generate a concise section title from a snippet

**Goal**
- Produce a 2–4 word noun-based English title for a transcript snippet.

**Inputs**
- `snippet: str` — up to 300 characters of transcript or excerpt

**Output**
- `title: str` — 2–4 words, nouns preferred, no punctuation

**Constraints**
- Generate English titles (DE -> EN translation happens upstream).
- Avoid generic verbs or filler words.

**Example**
- Input: “We’ll discuss student governance, the council, and campus policy.”
- Output: `Student Council Policy`

**Tests**
- Ensure output has 1–4 words and contains only letters/numbers and spaces.

---

## 7. Integration with GitHub Actions (CI)

We use a two-tier CI model (see `.github/workflows/ci.yml`):

- `fast-tests` job (GitHub-hosted): installs deps and runs fast tests on push and PRs.
- `heavy-tests-self-hosted` job (self-hosted): runs LLM/RAG-heavy tests only on
  self-hosted runners with label `self-hosted-llm` and only for manual or
  scheduled triggers.

Example excerpt of the fast-tests job steps (illustrative):

```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install poetry
    poetry install --no-interaction
- name: Run fast tests
  run: poetry run pytest -q
```

Example excerpt of the heavy self-hosted job (illustrative):

```yaml
runs-on: [self-hosted, self-hosted-llm]
steps:
  - uses: actions/checkout@v4
  - name: Install deps
    run: poetry install --no-interaction
  - name: Run heavy tests
    env:
      RUN_HEAVY_INTEGRATION: 'true'
      LOCAL_MODEL_PATH: /path/to/model.gguf
    run: poetry run pytest -q -m llm
```

Rationale: keep PR feedback fast and inexpensive; run expensive checks only on
machines prepared to host the models.

---

## 8. Running workflows locally with `act`

`act` is a popular tool to run GitHub Actions locally inside Docker. It is
useful for iterating on CI logic (install/test steps) without pushing to GitHub.

Example `act` usage to run the `fast-tests` job (requires Docker):

```bash
# list available jobs
act --list

# run the fast-tests job
act -j fast-tests --container-architecture linux/amd64

# pass secrets from a file
act -j fast-tests --secret-file .secrets
```

Notes:
- `act` simulates GitHub runners but cannot simulate some hosted features
  (OIDC, some GitHub-provided services).
- Heavy LLM tests that require local GPU/large model files are better tested on
  a self-hosted runner; `act` can still be used to validate non-LLM parts of the job.

---

## 9. Self-hosted runner setup (quick)

To run heavy LLM tests in CI, prepare a self-hosted runner that satisfies the
requirements:

1. Provision a VM or physical machine (recommended: 16+ GB RAM for unquantized models; ~6–8 GB with 4-bit quantization).
2. Install the GitHub Actions runner and label it `self-hosted-llm`.
3. Place the local model(s) at `LOCAL_MODEL_PATH` (or set that env var on the runner).
4. Install Python and Poetry and run `poetry install` so the job setup is quick.
5. Restrict runner access and keep it patched/updated.

Security note: self-hosted runners run arbitrary code — keep them locked to
trusted repositories or dedicated runners.

---

## 10. Security, licensing and privacy notes

- Never store API keys or credentials in prompt files or checked-in markdown.
- For local models: check the model license before redistribution. Models like
  LLaMA-family and some Hugging Face weights have license requirements; do not
  commit model weights to the repository.
- Store CI secrets via GitHub repository Secrets (Settings → Secrets).

---

## 11. Contributing new prompts

1. Create a single `.prompt.md` file under `.github/prompts/`.
2. Follow the prompt template (Section 4).
3. Add an example and a suggested lightweight test.
4. Open a PR and reference `docs/copilot.md` explaining the new prompt.

---

## 12. FAQ and quick commands

Q: Where do I run generated code before pushing?
A: Locally, inside the Poetry venv: `poetry install` → `poetry run pytest -q`.

Q: How do I enable heavy tests locally?
A: Set env vars and run marked tests:

```bash
export LOCAL_MODEL_PATH=./models/Phi-3-mini-4k-instruct-q4.gguf
export RUN_HEAVY_INTEGRATION=true
poetry run pytest -q -m llm
```

Quick commands summary

```bash
# Fast tests (default)
poetry run pytest -q

# Heavy LLM tests (opt-in)
export RUN_HEAVY_INTEGRATION=true
poetry run pytest -q -m llm

# Serve docs locally
poetry run mkdocs serve
```

---

## 13. Related pages

- `docs/development/testing.md` — testing policy and how LLM tests are gated
- `.github/workflows/ci.yml` — CI pipeline implementation
- `docs/imported/scripts_readme.md` — imported scripts README

If you want, I can automatically add the remaining `docs/*.md` files into the
navigation and resolve the current mkdocs warnings (broken links and unused
pages). Which would you prefer: (A) add all remaining docs to the site nav, or
(B) add only a curated subset and fix links?
