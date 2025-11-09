#!/usr/bin/env python3
"""Quick RAG smoke test using refactored architecture."""
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("USE_LOCAL_LLM", "true")
os.environ.setdefault("USE_RAG", "always")  # Force RAG for testing
os.environ.setdefault("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")

# Use refactored modules
from src.core import SectionGenerationService, SectionGenerationConfig
from src.core.models import TranscriptSegment

print("=" * 70)
print("RAG SYSTEM TEST - Refactored Architecture")
print("=" * 70)
print()

# Create transcript using data models
print("Creating test transcript...")
segments = [
    TranscriptSegment(
        start=0.0, 
        text="Introduction to the lecture about academic freedom.", 
        duration=30.0
    ),
    TranscriptSegment(
        start=30.0, 
        text="Discussion about student protests and political views.", 
        duration=45.0
    ),
    TranscriptSegment(
        start=75.0, 
        text="Panel discusses Gaza tribunal findings and international response.", 
        duration=60.0
    ),
    TranscriptSegment(
        start=135.0, 
        text="Closing remarks and further reading suggestions.", 
        duration=20.0
    ),
]
print(f"✅ Created {len(segments)} transcript segments")
print()

# Initialize service (Facade pattern)
print("Initializing Section Generation Service...")
service = SectionGenerationService()
print("✅ Service initialized")
print()

# Configure generation
config = SectionGenerationConfig(
    min_sections=2,
    max_sections=2,
    use_hierarchical=False,  # Flat for simple test
    temperature=0.2,
)
print(f"Configuration: {config.min_sections}-{config.max_sections} sections")
print()

# Generate sections
print("Generating sections...")
try:
    sections = service.generate_sections(
        transcript=segments,
        video_id="test_quick_refactored",
        generation_config=config,
    )
    
    print()
    print("=" * 70)
    print("GENERATED SECTIONS")
    print("=" * 70)
    for i, section in enumerate(sections, 1):
        print(f"{i}. {section.format_timestamp()} - {section.title}")
        if section.level > 0:
            print(f"   (Level {section.level} - subsection)")
    
    print()
    print("=" * 70)
    print(f"✅ SUCCESS! Generated {len(sections)} sections")
    print("=" * 70)
    
except Exception as e:
    print()
    print("=" * 70)
    print("❌ ERROR")
    print("=" * 70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Cleanup resources
    print()
    print("Cleaning up...")
    service.cleanup()
    print("✅ Cleanup complete")

print()
print("Test complete!")


