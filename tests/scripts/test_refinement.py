#!/usr/bin/env python3
"""Test script to verify transcript refinement is working."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment for testing
os.environ["REFINE_TRANSCRIPTS"] = "true"
os.environ["REFINEMENT_BATCH_SIZE"] = "10"  # Small batch for quick test

from src.core.services.transcript_refinement import TranscriptRefinementService

# Sample transcript segments with filler words
test_segments = [
    {"text": "um so wie I think that uh we should start", "start": 0.0, "duration": 3.5},
    {"text": "the the project today you know", "start": 3.5, "duration": 2.0},
    {"text": "and uh make sure äh everything is ready hmm", "start": 5.5, "duration": 3.0},
    {"text": "äh wir müssen das hmm heute noch machen", "start": 8.5, "duration": 2.5},
    {"text": "nicht imagine Claudia Wittig Claudia", "start": 11.0, "duration": 2.0},
]

print("=" * 60)
print("Testing Transcript Refinement Service")
print("=" * 60)
print()

print("Original segments:")
for i, seg in enumerate(test_segments, 1):
    print(f"{i}. [{seg['start']:.1f}s] {seg['text']}")
print()

try:
    service = TranscriptRefinementService()
    print(f"✅ Service initialized (batch_size={service.default_batch_size})")
    print()

    print("Running refinement...")
    refined_segments = service.refine_transcript_batch(test_segments, batch_size=5)
    print()

    print("Refined segments:")
    for i, seg in enumerate(refined_segments, 1):
        print(f"{i}. [{seg['start']:.1f}s] {seg['text']}")
    print()

    # Check for improvements
    original_text = " ".join(s["text"] for s in test_segments)
    refined_text = " ".join(s["text"] for s in refined_segments)

    improvements = []
    if "um" in original_text.lower() and "um" not in refined_text.lower():
        improvements.append("Removed 'um'")
    if "uh" in original_text.lower() and "uh" not in refined_text.lower():
        improvements.append("Removed 'uh'")
    if "äh" in original_text.lower() and "äh" not in refined_text.lower():
        improvements.append("Removed 'äh'")
    if "hmm" in original_text.lower() and "hmm" not in refined_text.lower():
        improvements.append("Removed 'hmm'")

    if improvements:
        print("✅ Refinement successful!")
        print(f"   Improvements: {', '.join(improvements)}")
    else:
        print("⚠️  No improvements detected")
        print("   (This might be expected if LLM chose to keep fillers for context)")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("Test complete!")
print("=" * 60)
