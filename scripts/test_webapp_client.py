"""Test script for web_app generate-sections endpoint using Flask test_client.

Writes the JSON response to scripts/webapp_response.json for inspection.
"""
from pathlib import Path
import json
from src.web_app import app

out = Path(__file__).parent / "webapp_response.json"
with app.test_client() as client:
    data = {
        "video_id": "TESTVIDEO",
        "transcript_json": '[{"start":0.0,"text":"Intro","duration":3.0},{"start":3.0,"text":"Second part","duration":4.0}]',
    }
    resp = client.post('/generate-sections', data=data)
    result = {"status_code": resp.status_code, "json": resp.get_json()}
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote response to {out}")

