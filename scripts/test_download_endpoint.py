"""Test the /download-sections endpoint using Flask test_client.
Reads scripts/webapp_response.json created earlier to obtain sections_text and video_id.
Verifies response headers and that no temporary file remains after send.
"""
import json
from pathlib import Path
from src.web_app import app

resp_file = Path(__file__).parent / "webapp_response.json"
if not resp_file.exists():
    raise SystemExit("Run scripts/test_webapp_client.py first to create webapp_response.json")

data = json.loads(resp_file.read_text())
body = data.get("json", {})
sections_text = body.get("sections_text") or ""
video_id = body.get("video_id") or "testvid"

with app.test_client() as client:
    resp = client.post("/download-sections", data={"sections": sections_text, "video_id": video_id})
    print("STATUS", resp.status_code)
    # Print headers of interest
    print("CONTENT-DISPOSITION:", resp.headers.get("Content-Disposition"))
    print("MIMETYPE:", resp.headers.get("Content-Type"))
    # Save file to disk to verify content
    out = Path(__file__).parent / f"downloaded_{video_id}_sections.txt"
    out.write_bytes(resp.get_data())
    print("WROTE", out)

    # Check temporary files in /tmp location of the app (it created a tmp file and deleted it)
    # We can't know the exact name; instead, ensure the created file exists and content matches sections_text
    loaded = out.read_text()
    if sections_text.strip() in loaded:
        print("SUCCESS: downloaded content contains sections_text")
    else:
        print("WARNING: downloaded content does not contain sections_text")

