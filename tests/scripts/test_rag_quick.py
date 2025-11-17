#!/usr/bin/env python3
"""Pytest-friendly quick RAG smoke test using refactored architecture.

Skips at collection when prerequisites (local model or services) are missing.
"""
import os
from pathlib import Path

import pytest

HEAVY = os.getenv("RUN_HEAVY_INTEGRATION", "false").lower() in ("1", "true", "yes")
if not HEAVY:
    pytest.skip(
        "Heavy integration tests disabled (set RUN_HEAVY_INTEGRATION=true to enable)",
        allow_module_level=True,
    )

MODEL_ENV = os.getenv("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
model_path = Path(MODEL_ENV)

if not model_path.exists():
    pytest.skip(
        f"Local model not present at {model_path}; skipping RAG quick test", allow_module_level=True
    )

try:
    from src.core.models.models import SectionGenerationConfig, TranscriptSegment
    from src.core.services.section_generation import SectionGenerationService
except Exception:
    pytest.skip(
        "Required service or models not available; skipping RAG quick test", allow_module_level=True
    )


def test_rag_quick_smoke():
    """Initialize service and run a small RAG generation to ensure end-to-end works."""
    os.environ.setdefault("USE_LOCAL_LLM", "true")
    os.environ.setdefault("USE_RAG", "always")
    os.environ.setdefault("LOCAL_MODEL_PATH", str(model_path))

    # Create simple segments
    segments = [
        TranscriptSegment(start=0.0, text="Intro to academic freedom", duration=30.0),
        TranscriptSegment(start=30.0, text="Student protests discussion", duration=45.0),
    ]

    service = SectionGenerationService()

    config = SectionGenerationConfig(
        min_sections=1,
        max_sections=2,
        use_hierarchical=False,
        temperature=0.2,
    )

    sections = service.generate_sections(
        transcript=segments, video_id="test_quick_refactored", generation_config=config
    )
    assert isinstance(sections, list)
    assert len(sections) >= 1
    assert all(hasattr(s, "to_dict") or isinstance(s, dict) for s in sections)
    service.cleanup()
