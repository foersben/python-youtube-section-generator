#!/usr/bin/env python3
"""Minimal Phi-3 validation - just load the model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("PHI-3 MODEL VALIDATION")
print("=" * 70)
print()

print("Step 1: Testing imports...")
try:
    from src.core.adapters.local_llm_client import LocalLLMClient

    print("✅ LocalLLMClient imported from adapters")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print()
print("Step 2: Loading model...")
try:
    client = LocalLLMClient()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print()
print("Step 3: Checking device info...")
info = client.get_device_info()
print(f"✅ Device: {info['device']}")
print(f"✅ Model: {info['model']}")
print(f"✅ Quantization: {info.get('quantization', 'N/A')}")

print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print()
print("✅ Phi-3 model is properly configured and loaded!")
print()
print("Note: Text generation on CPU is very slow (~1-2 minutes).")
print("For production use, consider:")
print("  1. Using a GPU (10-20x faster)")
print("  2. Using Gemini API (USE_LOCAL_LLM=false)")
print("  3. Using a smaller model (Llama-3.2-1B)")
