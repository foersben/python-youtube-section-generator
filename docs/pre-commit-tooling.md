Pre-commit toolset and how we use it
====================================

This document explains the "pre-commit" toolset used in this repository, what each tool does, why we use it, how it is configured in this project, and practical instructions for running and troubleshooting the hooks locally.

Why we use pre-commit
---------------------
Pre-commit automates a collection of fast checks and code-style fixes when you make commits. It prevents low-level style/regression issues from reaching the repository and ensures a consistent baseline for formatting, linting, typing, and secret scanning. Our CI also runs the same hooks, so fixing locally prevents CI failures and broken merges.

Where the configuration lives
----------------------------
- The hooks are configured in `.pre-commit-config.yaml` at the repository root.
- A `detect-secrets` baseline file `.secrets.baseline` is tracked to allow safe exceptions for previously reviewed strings.

Quick install & common commands (zsh)
------------------------------------
Install the pre-commit runner (if not installed):

```bash
# install pre-commit in your virtualenv or a user env
pip install --upgrade pre-commit

# enable the git hooks for this repo (run once)
pre-commit install

# run all hooks against all files (use before committing large changes)
pre-commit run --all-files

# run a single hook against all files
pre-commit run black --all-files

# run a single hook against specific files
pre-commit run isort --files src/core/transcript/extractor.py
```

If a hook modifies files, `pre-commit` will exit non-zero so you can review the changes, stage them, and commit.

Top-level hooks used in this project
-----------------------------------
The repository uses the following repositories/hooks (configured in `.pre-commit-config.yaml`). The list below explains what each hook does and how we configure it.

1) pre-commit-hooks (Yelp)
   - trailing-whitespace
     - Removes trailing whitespace in lines. Saves lots of noisy diffs.
   - end-of-file-fixer
     - Ensures files end with a single newline and no extra blank lines.
   - check-merge-conflict
     - Detects Git conflict markers (e.g. <<<<<<<) and prevents committing them.
   - check-added-large-files
     - Prevents accidentally adding very large files to git. Useful to avoid committing model files or datasets.
   - check-yaml
     - Validates YAML syntax for repo files (workflows, configs).

2) black (code formatter)
   - Tool: `psf/black` (pinned to a specific version in the config)
   - Purpose: deterministic Python code formatting.
   - Project configuration: called with `--line-length=100` to match our `pyproject.toml` project style.
   - Behavior: black reformats files in-place. If pre-commit reports a black failure, run `pre-commit run black --all-files` or `python -m black .`, then `git add` and commit the resulting changes.

3) isort (import sorting)
   - Tool: `PyCQA/isort`.
   - Purpose: keep imports sorted and grouped (stdlib / third party / local) to reduce import-order churn.
   - Project configuration: run with `--profile black` so isort's wrapping matches Black's formatting. That prevents repeated toggling between isort and black.

4) ruff (linter + auto-fixer)
   - Tool: `astral-sh/ruff-pre-commit`.
   - Purpose: fast linter and auto-fixer for style, complexity, and a large portion of ruff/flake8/pyflakes rules. We apply `--fix` to automatically apply safe fixes.
   - Notes: Some ruff rules (like B028, B904) request code changes (stacklevel warnings or raising from original exception). We aim to fix those in code to keep the repo clean.

5) mypy (static typing)
   - Tool: `pre-commit/mirrors-mypy` (pinned). Configured with extra arguments:
     - `--ignore-missing-imports` (avoid failing on 3rd-party missing types),
     - `--allow-untyped-defs` and `--allow-untyped-calls` to enable incremental typing in the codebase.
   - Purpose: catch real type errors and maintain typing hygiene.
   - Project-specific: limited to files under `src/` (configured) and includes `types-requests` in additional dependencies so requests stubs are available.
   - Troubleshooting: mypy can be noisy when upgrading dependencies. If mypy fails in CI but not locally, ensure you run the same pre-commit environment via `pre-commit run mypy --all-files` because pre-commit uses the pinned mypy version.

6) detect-secrets (Yelp)
   - Tool: `Yelp/detect-secrets` for scanning committed content for likely secrets.
   - Purpose: prevent accidentally committing API keys, tokens, private keys, and other secrets.
   - Project configuration:
     - Uses `.secrets.baseline` (tracked) via the `--baseline .secrets.baseline` argument so previously-audited findings are allowed.
     - Excludes files under `^\.github/` (we don't scan `.github` prompts by default in the hook) — note: the config uses an exclude regex; the baseline can still contain findings.
   - Updating baseline: to accept a new true positive (a string that should be allowed), run the scanner, inspect, then update the baseline carefully:

```bash
# scan and interactively update baseline (careful: only accept true positives after review)
detect-secrets scan > .secrets.baseline
# or to update an existing baseline (use review)
detect-secrets audit .secrets.baseline
```

   - False positives: prefer to remove or redact the sensitive data from source, or mark false positives in the baseline after cross-team review. You can also add inline allowlist (rare) or adjust the baseline.

How we use the `detect-secrets` baseline in this repo
----------------------------------------------------
- The repo tracks `.secrets.baseline`. This file contains earlier flagged strings that were reviewed and allowed. The pre-commit hook uses this baseline with the `--baseline` argument.
- If detect-secrets raises a new finding, do NOT commit immediately. Investigate the finding, and if it is a false positive, add a reviewed exception to the baseline (via `detect-secrets audit`) and commit the updated baseline. If it is a real secret, rotate and remove it.

Practical checklist before committing
------------------------------------
- Run a local full check before pushing (this is cheap and avoids CI redlines):

```bash
pre-commit run --all-files
```

- If hooks modified files (Black, isort, ruff), stage and commit those changes:

```bash
git add -A
git commit -m "chore: apply automatic formatting/lint fixes"
```

- If `detect-secrets` fails, inspect the file and use `detect-secrets audit .secrets.baseline` if the finding is a verified false positive (team-reviewed).

Troubleshooting common failures
-------------------------------
- "Hook modified files" (Black / isort / ruff): this is normal. Re-run `pre-commit run --all-files`, stage the updated files and commit. Do not bypass with `--no-verify` unless you are certain.

- Black vs isort toggling repeatedly: ensure `isort` is configured with `profile = "black"` (we pass `--profile black` in the hook). Always run them in the order `isort` then `black` locally before committing to reduce churn.

- mypy errors in CI but not locally: pre-commit runs pinned versions. Run `pre-commit run mypy --all-files` to use the same environment and arguments. If a package lacks type stubs, add it to `additional_dependencies` in `.pre-commit-config.yaml` or install an appropriate `types-*` package.

- detect-secrets false-positive inside `.github` or prompt files: by default the hook excludes `^\.github/`, but the baseline may still include entries. If a prompt legitimately contains a placeholder that triggers the secret detector, (a) consider redacting it to a placeholder like `<<REDACTED>>` or (b) add an audited entry to `.secrets.baseline` with `detect-secrets audit` and commit the updated baseline.

CI integration notes
--------------------
- The repository's CI runs the pre-commit checks (the same set used locally). Passing locally makes CI runs fast and predictable.
- If you adjust `.pre-commit-config.yaml`, push and request reviewers to run `pre-commit autoupdate` locally or run `pre-commit run --all-files` to ensure compatibility.

Guidelines for contributors
---------------------------
- Always run `pre-commit run --all-files` before creating a PR.
- Do not commit secrets. Use environment variables and `.env` (gitignored) for local development. Use secrets management for CI.
- If you need to accept a detect-secrets finding, do it via `detect-secrets audit .secrets.baseline`, comment in the PR why it's safe, and get an explicit reviewer signoff.

Advanced: regenerating and auditing the secrets baseline
--------------------------------------------------------
Only maintainers should update the baseline after investigation. Typical flow:

```bash
# generate a baseline from current working tree
detect-secrets scan > .secrets.baseline

# open an interactive audit for team review
detect-secrets audit .secrets.baseline
```

After the audit, commit `.secrets.baseline` and explain the change in your PR description.

Appendix: mapping of configured hooks -> file locations
------------------------------------------------------
- `.pre-commit-config.yaml` (repo root) defines the installed hooks and pinned revisions.
- `.secrets.baseline` contains the detect-secrets baseline and must be committed after review.

If you want, I can:
- Add a short developer README snippet with a one-liner to run the hooks before a PR.
- Add a `scripts/` helper to run `pre-commit run --all-files` inside the repo virtualenv and emit a short summary.

Would you like me to add either of these helpers? If yes, I’ll implement the helper script and a small README entry and commit them to a branch for review.
