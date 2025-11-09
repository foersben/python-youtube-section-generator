#!/usr/bin/env python3
"""Quick Phi-3 test with simplified generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("PHI-3 QUICK TEST")
print("=" * 70)
print()

from src.core.adapters.local_llm_client import LocalLLMClient

print("Loading Phi-3 model...")
client = LocalLLMClient()

print()
print("Device info:")
info = client.get_device_info()
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
        print(f"  {i}. {section['start']:.1f}s - {section['title']}")
    print()
    print("🎉 Phi-3 is working!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
