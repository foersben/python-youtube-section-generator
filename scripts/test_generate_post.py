"""Test helper: POST to /generate-sections via Flask test_client and save JSON response.
"""
import json
from src.web_app import app

with app.test_client() as client:
    resp = client.post('/generate-sections', data={'video_id': 'cZ9PHPta9v0'})
    try:
        data = resp.get_json()
    except Exception:
        data = {'status_code': resp.status_code, 'text': resp.get_data(as_text=True)}

    out_path = 'scripts/test_generate_response.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'status_code': resp.status_code, 'json': data}, f, ensure_ascii=False, indent=2)

print('Wrote', out_path)

