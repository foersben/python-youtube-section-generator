"""Debug script to see what the local LLM is actually generating."""

import sys
import logging
from pathlib import Path

# Setup logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from src.core.adapters.local_llm_client import LocalLLMClient

print("="*70)
print("DEBUGGING LOCAL LLM OUTPUT")
print("="*70)
print()

# Simple test transcript
sample_transcript = [
    {"start": 0.0, "text": "Welcome to this tutorial on Python.", "duration": 3.0},
    {"start": 3.0, "text": "We'll cover functions and classes.", "duration": 2.5},
    {"start": 5.5, "text": "Let's start with the basics.", "duration": 2.0},
]

print("Loading model...")
client = LocalLLMClient()

print("\nGenerating sections...")
print("-"*70)

try:
    sections = client.generate_sections(sample_transcript, num_sections=2, max_retries=1)
    print("\n✅ SUCCESS!")
    print("Generated sections:")
    for section in sections:
        print(f"  - {section['start']}s: {section['title']}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
