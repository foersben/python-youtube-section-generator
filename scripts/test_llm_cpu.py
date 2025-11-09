#!/usr/bin/env python3
"""Test script for CPU-only LLM with llama.cpp."""

import sys
from pathlib import Path

print("=" * 70)
print("CPU-ONLY LLM TEST (llama.cpp)")
print("=" * 70)
print()

# Check if model exists
model_path = Path("models/Phi-3-mini-4k-instruct-q4.gguf")

if not model_path.exists():
    print("❌ Model not found!")
    print()
    print(f"Expected: {model_path}")
    print()
    print("Download the model:")
    print("  wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf -P models/")
    print()
    print("Or download manually from:")
    print("  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf")
    print()
    sys.exit(1)

print(f"✅ Model found: {model_path.name}")
print(f"   Size: {model_path.stat().st_size / 1024**3:.2f} GB")
print()

# Test import
print("Testing imports...")
try:
    from src.core.adapters.local_llm_client import LocalLLMClient
    print("✅ LocalLLMClient imported from adapters")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print()

# Load model
print("Loading model (this may take 10-30 seconds)...")
try:
    client = LocalLLMClient(model_path=str(model_path))
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Show device info
print("=" * 70)
print("DEVICE INFORMATION")
print("=" * 70)
info = client.get_device_info()
for key, value in info.items():
    print(f"  {key}: {value}")
print()

# Test generation
print("=" * 70)
print("TESTING SECTION GENERATION")
print("=" * 70)
print()

sample_transcript = [
    {"start": 0.0, "text": "Welcome to this Python programming tutorial.", "duration": 3.0},
    {"start": 3.0, "text": "Today we'll learn about functions and classes.", "duration": 2.5},
    {"start": 5.5, "text": "Let's start with the basics of defining functions.", "duration": 3.0},
    {"start": 8.5, "text": "Functions allow us to reuse code efficiently.", "duration": 2.8},
    {"start": 11.3, "text": "Now let's move on to object-oriented programming.", "duration": 3.2},
    {"start": 14.5, "text": "Classes are the foundation of OOP in Python.", "duration": 2.7},
]

print("Generating sections (this may take 15-60 seconds on CPU)...")
print()

try:
    sections = client.generate_sections(sample_transcript, num_sections=3)
    
    print("✅ SUCCESS! Generated sections:")
    print("-" * 70)
    for i, section in enumerate(sections, 1):
        minutes = int(section["start"] // 60)
        seconds = int(section["start"] % 60)
        print(f"{i}. {minutes:02d}:{seconds:02d} - {section['title']}")
    print("-" * 70)
    print()
    print("🎉 CPU-only LLM is working perfectly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print()
print("✅ Your CPU-only LLM setup is ready to use!")
print()
