#!/usr/bin/env python3
"""Quick Phi-3 test with simplified generation.

This script is a manual quick-check for local Phi-3 models (llama.cpp-compatible GGUF).
It is not a pytest test and is meant to be executed manually.

Behavior improvements:
- Loads `.env` if present.
- Resolves `LOCAL_MODEL_PATH` (defaults to `models/Phi-3-mini-4k-instruct-q4.gguf`).
- Exits with a helpful message if the model file is missing.
- Uses the current adapter API (`get_info()` and `generate_sections()`).
"""

import os
import sys
from pathlib import Path

# Ensure project src is on path for direct execution from repo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("PHI-3 QUICK TEST")
print("=" * 70)
print()

# Choose model path (can be overridden with env var LOCAL_MODEL_PATH)
DEFAULT_MODEL = "models/Phi-3-mini-4k-instruct-q4.gguf"
model_env = os.getenv("LOCAL_MODEL_PATH", DEFAULT_MODEL)
model_path = Path(model_env)

# Helpful guard: if model path is a directory or missing, instruct user how to proceed
if not model_path.exists() or model_path.is_dir():
    print(f"❌ Local model not found at: {model_path}")
    print()
    print("Please download a GGUF model and set LOCAL_MODEL_PATH, for example:")
    print("  ./scripts/tools/download_model.sh")
    print("or set environment to use cloud provider instead: export USE_LOCAL_LLM=false")
    sys.exit(1)

from src.core.adapters.local_llm_client import LocalLLMClient

print("Loading Phi-3 model...")
# Provide model_path explicitly to avoid None getting propagated
client = LocalLLMClient(model_path=str(model_path))

print()
print("Device info:")
info = client.get_info()
for key, value in info.items():
    print(f"  {key}: {value}")

print()
print("Testing generation with shorter input...")

# Very short test
short_transcript = [
    {"start": 0.0, "text": "Introduction to Python", "duration": 2.0},
    {"start": 2.0, "text": "Basic syntax", "duration": 2.0},
]

print("Generating 2 sections from 2 segments...")

try:
    sections = client.generate_sections(short_transcript, num_sections=2, max_retries=1)
    print()
    print("✅ SUCCESS!")
    for i, section in enumerate(sections, 1):
        # Section may be dict-like
        start = (
            section.get("start") if isinstance(section, dict) else getattr(section, "start", None)
        )
        title = section.get("title") if isinstance(section, dict) else getattr(section, "title", "")
        if start is not None:
            print(f"  {i}. {start:.1f}s - {title}")
        else:
            print(f"  {i}. - {title}")
    print()
    print("🎉 Phi-3 is working!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
