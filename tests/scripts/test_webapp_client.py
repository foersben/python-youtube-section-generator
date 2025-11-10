"""Pytest for /generate-sections endpoint that writes a JSON response.

This was originally a script that executed at import time. Converted to a
pytest test that uses tmp_path and is marked as heavy (`llm`).
"""

import os
import json
import pytest
from pathlib import Path

from src.web_app import app


@pytest.mark.llm
def test_webapp_generate_writes_response(tmp_path: Path) -> None:
    """POST to /generate-sections and write JSON response to tmp_path.

    The test is skipped unless RUN_HEAVY_INTEGRATION is enabled.
    """
    if os.getenv("RUN_HEAVY_INTEGRATION", "false").lower() not in ("1", "true", "yes"):
        pytest.skip("Heavy integration tests disabled. Set RUN_HEAVY_INTEGRATION=true to enable")

    data = {
        "video_id": "TESTVIDEO",
        "transcript_json": json.dumps([
            {"start": 0.0, "text": "Intro", "duration": 3.0},
            {"start": 3.0, "text": "Second part", "duration": 4.0},
        ]),
    }

    out = tmp_path / "webapp_response.json"
    with app.test_client() as client:
        resp = client.post("/generate-sections", data=data)
        assert resp is not None
        assert resp.status_code in (200, 201, 202)
        try:
            result = resp.get_json()
        except Exception:
            result = {"status_code": resp.status_code, "text": resp.get_data(as_text=True)}

    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    assert out.exists()
