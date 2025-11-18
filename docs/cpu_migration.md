# Migrating from GPU-Contaminated Environment to CPU-Only Stack

This document provides a definitive protocol to purge GPU-enabled packages (PyTorch, llama-cpp-python), reinstall CPU-only binaries, and modernize dependencies safely.

## Executive summary

See the project's `.github/OPTIMIZATION_SUMMARY.md` and the in-repo docs for an extended audit. This page is a runnable companion containing a diagnostic script, an interactive decontamination script, and a recommended `pyproject.toml` snippet for stability.

## Quick checklist

- [ ] Run `python scripts/check_environment.py > before_fix.log` and save the output.
- [ ] Run `bash scripts/decontaminate_cpu.sh` (or `bash scripts/decontaminate_cpu.sh --yes`).
- [ ] Run `python scripts/check_environment.py > after_fix.log` and `diff before_fix.log after_fix.log`.

## Diagnostic script

`python scripts/check_environment.py` will:

- Report PyTorch version and CUDA availability
- Probe `llama-cpp-python` for compiled GPU offload symbols (best-effort)
- List versions for key packages (langchain, chromadb, sentence-transformers, google-genai)

## Decontamination script

`bash scripts/decontaminate_cpu.sh` performs the following (interactive):

1. Uninstall `torch`, `torchvision`, `torchaudio`, and `llama-cpp-python` (twice to be safe)
2. Purge `pip` cache (`pip cache purge`)
3. Install CPU-only PyTorch: `torch==2.9.1`, `torchvision==0.24.1`, `torchaudio==2.9.1` from the official CPU index
4. Install pre-built `llama-cpp-python==0.3.16` from a CPU wheel index
5. Install `sentence-transformers==5.1.2`
6. Pip-install the project editable (`pip install -e .`)

> Note: The scripts are idempotent and safe to rerun, but please ensure you run them inside the project's virtualenv.

## Verification

After running the decontamination script, execute:

```bash
python scripts/check_environment.py > after_fix.log
# compare
diff before_fix.log after_fix.log
```

Expectations:

- `CUDA Available` should move from `True` to `False`.
- `llama.cpp GPU Offload Supported (compiled)` should be `False`.
- PyTorch version should show CPU-only wheel (no `+cu` suffix) and match `2.9.1`.

## Recommended pyproject.toml (stable, non-breaking)

Below is the project's recommended `pyproject.toml` dependency block focused on compatibility and CPU-only installation. Install PyTorch and llama-cpp-python manually first (as above), then run `poetry install` or `pip install -e .`.

```toml
[tool.poetry.dependencies]
python = ">=3.10,<4.0"
# CPU-only packages installed separately (manually): torch, torchvision, torchaudio
llama-cpp-python = "0.3.16"
sentence-transformers = "5.1.2"
langchain = "0.1.20" # PIN to maintain imports
chromadb = "0.4.24"  # PIN to maintain API compatibility
google-genai = "1.50.1"
# ... other dependencies ...
```

## Modernization guidance

- DO NOT upgrade `langchain` or `chromadb` to their 1.x releases unless you are ready to perform a major refactor.
- `sentence-transformers` is safe to upgrade to 5.x.
- `google-generativeai` must be replaced with `google-genai` and code updated accordingly.

## Next steps (optional automation)

- Add `scripts/decontaminate_cpu.sh` to CI as a manual job (workflow_dispatch) for a reproducible remediation run.
- Add a small CI ``smoke`` test that runs `python scripts/check_environment.py` to detect accidental GPU dependency upgrades.

---

If you want, I can open a PR with these files and also update `pyproject.toml` with the recommended pinned versions (I did not modify `pyproject.toml` yet to avoid any immediate CI impacts).
