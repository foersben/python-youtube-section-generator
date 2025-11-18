#!/usr/bin/env python3
"""Verification script for refactored architecture.

Tests that all new modules can be imported and used correctly.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("REFACTORING VERIFICATION")
print("=" * 70)
print()

# Test 1: Import core modules
print("Test 1: Importing refactored modules...")
try:
    from src.core.config import AppConfig, config
    from src.core.embeddings import EmbeddingsFactory, EmbeddingsProvider
    from src.core.llm import LLMFactory, LLMProvider
    from src.core.models import (
        Section,
        SectionGenerationConfig,
        TranscriptSegment,
        VideoInfo,
    )
    from src.core.retrieval import RAGSystem
    from src.core.services import SectionGenerationService

    print("✅ All refactored modules imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

print()

# Test 2: Import from core package
print("Test 2: Importing from src.core...")
try:
    from src.core import (
        LLMFactory,
        RAGSystem,
        Section,
        SectionGenerationConfig,
        SectionGenerationService,
        TranscriptSegment,
        config,
    )

    print("✅ Successfully imported from src.core")
except ImportError as e:
    print(f"❌ Import from src.core failed: {e}")
    sys.exit(1)

print()

# Test 3: Legacy API still works
print("Test 3: Testing backward compatibility...")
try:
    from src.core import (
        create_section_timestamps,
        extract_transcript,
        extract_video_id,
        format_sections_for_youtube,
    )

    print("✅ Legacy API imports work")
except ImportError as e:
    print(f"❌ Legacy API import failed: {e}")
    sys.exit(1)

print()

# Test 4: Create data models
print("Test 4: Testing data models...")
try:
    segment = TranscriptSegment(start=0.0, text="Hello", duration=5.0)
    section = Section(title="Introduction", start=0.0, level=0)
    config_obj = SectionGenerationConfig(min_sections=10, max_sections=15, use_hierarchical=True)

    # Test model methods
    segment_dict = segment.to_dict()
    section_dict = section.to_dict()
    timestamp = section.format_timestamp()

    assert segment_dict["start"] == 0.0
    assert section_dict["title"] == "Introduction"
    assert timestamp == "00:00"

    print("✅ Data models work correctly")
except Exception as e:
    print(f"❌ Data model test failed: {e}")
    sys.exit(1)

print()

# Test 5: Configuration
print("Test 5: Testing configuration...")
try:
    assert config is not None
    assert isinstance(config, AppConfig)
    assert hasattr(config, "use_local_llm")
    assert hasattr(config, "use_rag")

    # Test config methods
    should_use_rag = config.should_use_rag(1800)  # 30 minutes
    config_dict = config.to_dict()

    print(f"  use_local_llm: {config.use_local_llm}")
    print(f"  use_rag: {config.use_rag}")
    print(f"  rag_hierarchical: {config.rag_hierarchical}")
    print("✅ Configuration system works")
except Exception as e:
    print(f"❌ Configuration test failed: {e}")
    sys.exit(1)

print()

# Test 6: LLM Factory
print("Test 6: Testing LLM Factory...")
try:
    # Don't actually create provider (might need model file)
    # Just test that factory exists and has methods
    assert hasattr(LLMFactory, "create_provider")
    assert hasattr(LLMFactory, "create_local_provider")
    print("✅ LLM Factory available")
except Exception as e:
    print(f"❌ LLM Factory test failed: {e}")
    sys.exit(1)

print()

# Test 7: Formatting with Section objects
print("Test 7: Testing formatting with Section objects...")
try:
    from src.core.formatting import format_sections_for_youtube

    # Test with Section objects
    sections = [
        Section(title="Introduction", start=0.0, level=0),
        Section(title="Setup", start=15.0, level=1),
        Section(title="Main Content", start=60.0, level=0),
    ]
    formatted = format_sections_for_youtube(sections)
    assert "00:00 Introduction" in formatted
    assert "  00:15 Setup" in formatted  # Indented subsection
    assert "01:00 Main Content" in formatted

    # Test with dicts (backward compatibility)
    sections_dict = [s.to_dict() for s in sections]
    formatted_dict = format_sections_for_youtube(sections_dict)

    print("✅ Formatting works with both Section objects and dicts")
except Exception as e:
    print(f"❌ Formatting test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print()

# Summary
print("=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print()
print("Refactored architecture is working correctly!")
print()
print("You can now:")
print("  - Run the web app: poetry run python src/web_app.py")
print("  - Run the CLI: poetry run python src/main.py")
print("  - Use the new API in your code")
print()
