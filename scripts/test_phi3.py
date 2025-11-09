#!/usr/bin/env python3
"""Test Phi-3 model with multi-GPU support."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("PHI-3 MODEL TEST")
print("=" * 70)
print()

# Import and test
from src.core.adapters.local_llm_client import LocalLLMClient

# Create client
print("Loading Phi-3 model...")
print("(This will download ~8GB on first run)")
print()

client = LocalLLMClient()

# Show device info
print("=" * 70)
print("DEVICE INFORMATION")
print("=" * 70)
device_info = client.get_device_info()
print(f"Device: {device_info['device']}")
print(f"Model: {device_info['model']}")
print(f"Quantization: {device_info.get('quantization', False)}")

if "num_gpus" in device_info:
    print(f"GPUs: {device_info['num_gpus']}")
    for gpu in device_info.get("gpus", []):
        print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
print()

# Test with sample transcript
sample_transcript = [
    {"start": 0.0, "text": "Welcome to this Python programming tutorial.", "duration": 3.0},
    {"start": 3.0, "text": "Today we'll learn about functions and classes.", "duration": 2.5},
    {"start": 5.5, "text": "Let's start with the basics of defining functions.", "duration": 3.0},
    {"start": 8.5, "text": "Functions allow us to reuse code efficiently.", "duration": 2.8},
    {"start": 11.3, "text": "Now let's move on to object-oriented programming.", "duration": 3.2},
    {"start": 14.5, "text": "Classes are the foundation of OOP in Python.", "duration": 2.7},
    {"start": 17.2, "text": "We'll create our first class together.", "duration": 2.5},
    {"start": 19.7, "text": "Thanks for watching this tutorial!", "duration": 2.0},
]

print("=" * 70)
print("GENERATING SECTIONS")
print("=" * 70)
print()

try:
    sections = client.generate_sections(sample_transcript, num_sections=3, max_retries=2)
    
    print("✅ SUCCESS! Generated sections:")
    print("-" * 70)
    for i, section in enumerate(sections, 1):
        minutes = int(section["start"] // 60)
        seconds = int(section["start"] % 60)
        print(f"{i}. {minutes:02d}:{seconds:02d} - {section['title']}")
    print("-" * 70)
    print()
    print("🎉 Phi-3 model is working perfectly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
