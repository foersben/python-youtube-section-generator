#!/usr/bin/env python3
"""Pytest-friendly smoke tests for LocalLLMClient (Phi-3).

Skip at collection when the local model is missing or client cannot be imported.
"""

import os
from pathlib import Path

import pytest

MODEL_ENV = os.getenv("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
model_path = Path(MODEL_ENV)

if not model_path.exists():
    pytest.skip(
        f"Local model not present at {model_path}; skipping Phi-3 tests", allow_module_level=True
    )

try:
    from src.core.adapters.local_llm_client import LocalLLMClient
except Exception:
    pytest.skip("LocalLLMClient not available; skipping Phi-3 tests", allow_module_level=True)

HEAVY = os.getenv("RUN_HEAVY_INTEGRATION", "false").lower() in ("1", "true", "yes")
if not HEAVY:
    pytest.skip(
        "Heavy integration tests disabled (set RUN_HEAVY_INTEGRATION=true to enable)",
        allow_module_level=True,
    )


def test_local_llm_client_instantiation():
    client = LocalLLMClient(model_path=str(model_path))
    # Adapter exposes get_info()
    info = client.get_info()
    assert isinstance(info, dict)


def test_local_llm_generate_smoke():
    client = LocalLLMClient(model_path=str(model_path))
    sample = [
        {"start": 0.0, "text": "Hello world", "duration": 2.0},
        {"start": 2.0, "text": "This is a test", "duration": 3.0},
    ]
    sections = client.generate_sections(sample, num_sections=1)
    assert isinstance(sections, list)
    assert len(sections) >= 1
    assert all("title" in s and "start" in s for s in sections)
