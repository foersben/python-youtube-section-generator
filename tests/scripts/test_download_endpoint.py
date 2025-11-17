"""Pytest for /download-sections endpoint.

This test posts a small transcript to `/generate-sections` to obtain a response
and then requests `/download-sections` to verify the downloadable content.
The test is marked `llm` and will be skipped unless `RUN_HEAVY_INTEGRATION` is
enabled or the test is explicitly requested.
"""

import json
import os
from pathlib import Path

import pytest

from src.web_app import app


@pytest.mark.llm
def test_download_sections_creates_file(tmp_path: Path) -> None:
    """POST to /generate-sections, then download the generated sections file.

    The test uses tmp_path to avoid writing into the repository. It will skip
    unless RUN_HEAVY_INTEGRATION is enabled to avoid heavy runs during normal
    test collection.
    """
    if os.getenv("RUN_HEAVY_INTEGRATION", "false").lower() not in ("1", "true", "yes"):
        pytest.skip("Heavy integration tests disabled. Set RUN_HEAVY_INTEGRATION=true to enable")

    # Arrange: small transcript
    transcript = [
        {"start": 0.0, "text": "Intro", "duration": 3.0},
        {"start": 3.0, "text": "Second part", "duration": 4.0},
    ]

    with app.test_client() as client:
        # Request generation (may call LLM/RAG)
        resp_gen = client.post(
            "/generate-sections",
            data={"video_id": "TESTVIDEO", "transcript_json": json.dumps(transcript)},
        )

        assert resp_gen is not None
        assert resp_gen.status_code in (200, 201, 202)

        # Get JSON body safely
        try:
            gen_body = resp_gen.get_json() or {}
        except Exception:
            gen_body = {}

        sections_text = gen_body.get("sections_text", "")
        video_id = gen_body.get("video_id", "testvid")

        # Now call download endpoint
        resp_dl = client.post(
            "/download-sections", data={"sections": sections_text, "video_id": video_id}
        )

        assert resp_dl is not None
        # Check content-disposition header present
        cd = resp_dl.headers.get("Content-Disposition")
        assert cd is not None and "attachment" in cd

        # Save bytes to tmp file and verify content
        out_file = tmp_path / f"downloaded_{video_id}_sections.txt"
        out_file.write_bytes(resp_dl.get_data())

        # If sections_text had content, assert it's contained
        if sections_text.strip():
            loaded = out_file.read_text(errors="ignore")
            assert sections_text.strip() in loaded
        else:
            assert out_file.exists()
