#!/usr/bin/env python3
"""Pytest-friendly CPU LLM smoke tests.

Skip at collection when the local model is not present or required client cannot be imported.
"""

import os
from pathlib import Path

import pytest

MODEL_ENV = os.getenv("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
model_path = Path(MODEL_ENV)

if not model_path.exists():
    pytest.skip(
        f"Local model not present at {model_path}; skipping CPU LLM tests", allow_module_level=True
    )

try:
    from src.core.adapters.local_llm_client import LocalLLMClient
except Exception:
    pytest.skip("LocalLLMClient not available; skipping CPU LLM tests", allow_module_level=True)

HEAVY = os.getenv("RUN_HEAVY_INTEGRATION", "false").lower() in ("1", "true", "yes")
if not HEAVY:
    pytest.skip(
        "Heavy integration tests disabled (set RUN_HEAVY_INTEGRATION=true to enable)",
        allow_module_level=True,
    )


def test_local_llm_import_and_device():
    """Import LocalLLMClient and check it can be instantiated (device info reported)."""
    client = LocalLLMClient(model_path=str(model_path))
    # Adapter exposes get_info() which returns provider info
    info = client.get_info()
    assert isinstance(info, dict)
    assert "Context" in info or "device" in info


def test_generate_sections_smoke_with_client():
    """Run a lightweight generation to confirm the client can produce sections."""
    client = LocalLLMClient(model_path=str(model_path))
    sample_transcript = [
        {"start": 0.0, "text": "Intro to testing", "duration": 3.0},
        {"start": 3.0, "text": "Deep dive", "duration": 4.0},
    ]

    sections = client.generate_sections(sample_transcript, num_sections=2)
    assert isinstance(sections, list)
    assert len(sections) >= 1
    assert all("title" in s and "start" in s for s in sections)
