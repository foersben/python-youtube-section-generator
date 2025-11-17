"""Pytest-friendly smoke test for SectionGenerationService.

Skips at collection time if the local model or the SectionGenerationService is not available.
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

# Determine model path (can be overridden via env)
MODEL_ENV = os.getenv("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
model_path = Path(MODEL_ENV)

if not model_path.exists():
    pytest.skip(
        f"Local model not present at {model_path}; skipping heavy integration test",
        allow_module_level=True,
    )

try:
    from src.core.services.section_generation import SectionGenerationService
except Exception:
    pytest.skip(
        "SectionGenerationService not available; skipping integration test", allow_module_level=True
    )


def test_generate_sections_smoke() -> None:
    """Smoke test: run generation on a tiny transcript and assert a valid response structure."""
    service = SectionGenerationService()
    sample_transcript = [
        {"start": 0.0, "text": "Intro to testing", "duration": 3.0},
        {"start": 3.0, "text": "Deep dive", "duration": 4.0},
    ]

    sections = service.generate_sections(transcript=sample_transcript, video_id="test_smoke")
    assert isinstance(sections, list)
    assert len(sections) >= 1
    # Sections may be Section dataclass instances or dicts; normalize for test
    normalized = [s.to_dict() if hasattr(s, "to_dict") else s for s in sections]
    assert all("title" in s and "start" in s for s in normalized)
