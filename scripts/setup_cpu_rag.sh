#!/bin/bash
# CPU-Only Setup Script - Install RAG dependencies without GPU packages
#
# This script ensures a pure CPU-only environment by:
# 1. Installing Poetry dependencies (excludes torch/sentence-transformers/chromadb)
# 2. Manually installing CPU-only torch from PyTorch index
# 3. Installing sentence-transformers and chromadb (will use CPU torch)

set -e  # Exit on error

echo "======================================================================"
echo "CPU-ONLY RAG DEPENDENCIES SETUP"
echo "======================================================================"
echo ""
echo "This will install RAG dependencies (sentence-transformers, chromadb)"
echo "with CPU-only torch (no NVIDIA GPU packages)."
echo ""

# Step 1: Install Poetry dependencies
echo "Step 1/3: Installing Poetry dependencies..."
poetry install
echo "✅ Poetry dependencies installed"
echo ""

# Step 2: Install CPU-only torch
echo "Step 2/3: Installing CPU-only torch (no GPU)..."
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo "✅ CPU-only torch installed"
echo ""

# Step 3: Install sentence-transformers and chromadb
echo "Step 3/3: Installing sentence-transformers and chromadb..."
poetry run pip install sentence-transformers chromadb
echo "✅ sentence-transformers and chromadb installed"
echo ""

# Verify no NVIDIA packages
echo "======================================================================"
echo "VERIFICATION"
echo "======================================================================"
echo ""
echo "Checking for NVIDIA GPU packages..."
NVIDIA_PKGS=$(poetry run pip list | grep nvidia || true)
if [ -z "$NVIDIA_PKGS" ]; then
    echo "✅ No NVIDIA GPU packages found - Pure CPU setup!"
else
    echo "⚠️  WARNING: Found NVIDIA packages:"
    echo "$NVIDIA_PKGS"
fi
echo ""

# Show torch version
echo "Torch version:"
poetry run python -c "import torch; print(f'  torch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
echo ""

echo "======================================================================"
echo "✅ SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Your environment is now configured for CPU-only RAG with:"
echo "  - llama-cpp-python (CPU LLM inference)"
echo "  - torch CPU-only"
echo "  - sentence-transformers (embeddings)"
echo "  - chromadb (vector database)"
echo ""
echo "Next steps:"
echo "  1. Test RAG: poetry run python scripts/test_rag.py"
echo "  2. Use web app: poetry run python src/web_app.py"
echo ""

