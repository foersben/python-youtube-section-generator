#!/usr/bin/env python3
"""Verify local LLM setup and test model loading."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

# Load environment
load_dotenv()

print("=" * 70)
print("LOCAL LLM SETUP VERIFICATION")
print("=" * 70)
print()

# Check environment configuration
print("📋 Configuration Check:")
print(f"  USE_LOCAL_LLM: {os.getenv('USE_LOCAL_LLM', 'false')}")
print(f"  LOCAL_MODEL_NAME: {os.getenv('LOCAL_MODEL_NAME', 'microsoft/Phi-3-mini-4k-instruct')}")
print(f"  LOCAL_MODEL_4BIT: {os.getenv('LOCAL_MODEL_4BIT', 'true')}")
print()

# Check CUDA availability
try:
    import torch

    cuda_available = torch.cuda.is_available()
    print("🖥️  Hardware:")
    print(f"  CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
except ImportError:
    print("❌ PyTorch not installed. Run: poetry install")
    sys.exit(1)

# Check HuggingFace token
print("🔑 HuggingFace Authentication:")
hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
if hf_token_path.exists():
    print("  ✅ HuggingFace token found")
else:
    print("  ⚠️  No HuggingFace token found")
    print("  To login: poetry run huggingface-cli login")
print()

# Check if model is already downloaded
model_name = os.getenv("LOCAL_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_dirs = list(cache_dir.glob("models--*")) if cache_dir.exists() else []

print("📦 Model Cache:")
if model_dirs:
    print(f"  Found {len(model_dirs)} cached models in {cache_dir}")
    for model_dir in model_dirs[:3]:  # Show first 3
        model_id = model_dir.name.replace("models--", "").replace("--", "/")
        print(f"    - {model_id}")
    if len(model_dirs) > 3:
        print(f"    ... and {len(model_dirs) - 3} more")
else:
    print("  No cached models found. Model will be downloaded on first use.")
print()

# Test import
print("🔧 Testing imports...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  ✅ transformers imported successfully")
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    print("  Run: poetry install")
    sys.exit(1)

try:
    from tests.local_llm_shim import LocalLLMClient

    print("  ✅ LocalLLMClient (test shim) imported successfully")
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("READY TO TEST")
print("=" * 70)
print()

# Ask user if they want to test model loading
print("Would you like to test loading the model now?")
print("⚠️  WARNING: This will download ~15GB if not already cached")
print()
response = input("Continue with model test? (y/N): ").strip().lower()

if response == "y":
    print()
    print("🚀 Loading model (this may take several minutes)...")
    print()

    try:
        # Simple test
        sample_transcript = [
            {
                "start": 0.0,
                "text": "Welcome to this video about Python programming.",
                "duration": 3.0,
            },
            {"start": 3.0, "text": "Today we'll cover the basics of functions.", "duration": 2.5},
            {"start": 5.5, "text": "Let's start with defining a simple function.", "duration": 2.8},
        ]

        print("Creating LocalLLMClient...")
        client = LocalLLMClient()

        print(f"✅ Model loaded successfully on {client.device}")
        print()
        print("Testing section generation...")

        sections = client.generate_sections(sample_transcript, num_sections=2, max_retries=1)

        print()
        print("✅ SUCCESS! Generated sections:")
        print("-" * 70)
        for i, section in enumerate(sections, 1):
            print(f"{i}. {section['start']:.1f}s - {section['title']}")
        print("-" * 70)
        print()
        print("🎉 Local LLM is working perfectly!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
else:
    print()
    print("Skipping model test. To test later, run:")
    print("  poetry run python scripts/verify_local_llm.py")
    print()
    print("Or use the model directly in your application by setting:")
    print("  USE_LOCAL_LLM=true")

print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
