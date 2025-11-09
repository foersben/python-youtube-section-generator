#!/usr/bin/env python3
"""Test sections.py integration with llama-cpp-python."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("SECTIONS.PY INTEGRATION TEST")
print("=" * 70)
print()

# Set environment for local LLM
os.environ["USE_LOCAL_LLM"] = "true"
os.environ["LOCAL_MODEL_PATH"] = "models/Phi-3-mini-4k-instruct-q4.gguf"

print("Configuration:")
print(f"  USE_LOCAL_LLM: {os.getenv('USE_LOCAL_LLM')}")
print(f"  LOCAL_MODEL_PATH: {os.getenv('LOCAL_MODEL_PATH')}")
print()

# Check if model exists
model_path = Path(os.getenv("LOCAL_MODEL_PATH"))
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    print()
    print("Download the model first:")
    print("  ./scripts/download_model.sh")
    print()
    print("Or switch to Gemini API:")
    print("  export USE_LOCAL_LLM=false")
    sys.exit(1)

size_gb = model_path.stat().st_size / 1024**3
print(f"✅ Model found: {model_path.name} ({size_gb:.2f}GB)")
print()

# Test import
print("Testing imports...")
try:
    from src.core import sections
    print("✅ sections module imported")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Create sample transcript
print("Creating test transcript...")
sample_transcript = [
    {"start": 0.0, "text": "Welcome to this Python programming tutorial", "duration": 3.0},
    {"start": 3.0, "text": "Today we will learn about functions and classes", "duration": 2.5},
    {"start": 5.5, "text": "Let's start with defining our first function", "duration": 3.0},
    {"start": 8.5, "text": "Functions help us organize and reuse code", "duration": 2.8},
    {"start": 11.3, "text": "Now let's talk about object-oriented programming", "duration": 3.2},
    {"start": 14.5, "text": "Classes are the foundation of OOP in Python", "duration": 2.7},
    {"start": 17.2, "text": "We can create objects from classes", "duration": 2.5},
    {"start": 19.7, "text": "Objects have attributes and methods", "duration": 2.8},
]

print(f"✅ Created transcript with {len(sample_transcript)} segments")
print()

# Test section generation
print("=" * 70)
print("GENERATING SECTIONS")
print("=" * 70)
print()
print("This will test the full integration:")
print("  1. sections.py → _create_sections_with_local_llm()")
print("  2. LocalLLMClient initialization with model_path")
print("  3. llama.cpp GGUF model loading")
print("  4. Section generation")
print()
print("⏳ Generating sections (may take 15-60 seconds on CPU)...")
print()

try:
    result_sections = sections.create_section_timestamps(
        transcript=sample_transcript,
        section_count_range=(2, 4),
        title_length_range=(3, 6)
    )
    
    print("✅ SUCCESS! Generated sections:")
    print("-" * 70)
    for i, section in enumerate(result_sections, 1):
        minutes = int(section["start"] // 60)
        seconds = int(section["start"] % 60)
        print(f"{i}. {minutes:02d}:{seconds:02d} - {section['title']}")
    print("-" * 70)
    print()
    print("🎉 Web app integration is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print()
print("✅ sections.py correctly integrates with llama-cpp-python!")
print("✅ Your web app should now work!")
print()
print("Next step: Test web app")
print("  poetry run python src/web_app.py")
print()

