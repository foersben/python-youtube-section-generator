#!/usr/bin/env bash
set -euo pipefail

# scripts/decontaminate_cpu.sh
# Interactive script to remove GPU-enabled packages and install CPU-only equivalents.
# WARNING: This will modify the active Python environment. Run in the project's venv.

PROCEED=false
if [[ ${1:-""} == "--yes" ]]; then
  PROCEED=true
fi

echo "This script will:
  - Uninstall torch/torchvision/torchaudio/llama-cpp-python
  - Purge pip cache
  - Install CPU-only versions of PyTorch and llama-cpp-python
  - Install sentence-transformers (recommended update)
"

if ! $PROCEED; then
  read -rp "Proceed? Type 'yes' to continue: " confirm
  if [[ "$confirm" != "yes" ]]; then
    echo "Aborted by user."
    exit 1
  fi
fi

echo "== Uninstalling existing packages (may print warnings if not installed) =="
for pkg in torch torchvision torchaudio llama-cpp-python; do
  echo "Uninstalling: $pkg"
  pip uninstall -y "$pkg" || true
  # run twice to remove layered installs
  pip uninstall -y "$pkg" || true
done

echo "== Purging pip cache =="
pip cache purge || true

echo "== Installing CPU-only PyTorch stack =="
# Use the official PyTorch CPU index
pip install \
  torch==2.9.1 \
  torchvision==0.24.1 \
  torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cpu \
  --no-cache-dir

echo "== Installing pre-built CPU wheel for llama-cpp-python =="
# Use the community CPU wheel repo maintained for convenience
pip install \
  llama-cpp-python==0.3.16 \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
  --no-cache-dir

echo "== Installing sentence-transformers (recommended upgrade) =="
pip install sentence-transformers==5.1.2 --no-cache-dir

echo "== Reinstalling project (editable) dependencies =="
# Ensure project dependencies are installed; this will skip already present pinned packages
pip install -e . --no-cache-dir

echo "== Done. Please run: python scripts/check_environment.py > after_fix.log and compare with your before log. =="
