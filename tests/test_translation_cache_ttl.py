import json
import time
from pathlib import Path

from src.core.config import config
from src.core.transcript.extractor import extract_transcript


def test_translation_cache_ttl(monkeypatch, tmp_path):
    """Ensure that expired cache entries are ignored and fresh translation is used."""
    # Prepare a fake transcript (simulate original_transcript.fetch result)
    video_id = "testvideo123"
    fake_segments = [{"text": "Hallo Welt", "start": 0.0, "duration": 5.0}]

    # Create cache dir under project root .cache/translations
    cache_dir = Path(config.project_root) / ".cache" / "translations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{video_id}_en.json"

    # Write an expired cache (ts old)
    expired_ts = int(time.time()) - 999999
    cache_content = {"ts": expired_ts, "segments": fake_segments}
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache_content, f)

    # Monkeypatch YouTubeTranscriptApi behaviors to return a transcript-like object
    class DummyTrans:
        is_generated = False
        language_code = "de"
        is_translatable = True

        def fetch(self):
            return fake_segments

        def translate(self, target):
            return self

    class DummyList:
        def __iter__(self):
            yield DummyTrans()

        def find_transcript(self, langs):
            return DummyTrans()

    def fake_list(video):
        return DummyList()

    # Make fetch raise to force use of transcript_list.find_transcript and server-side translate path
    def make_api():
        class X:
            def list(self, v):
                return DummyList()

            def fetch(self, v, languages=None):
                raise Exception("forced fetch failure for test")

        return X()

    monkeypatch.setattr("src.core.transcript.extractor.YouTubeTranscriptApi", lambda: make_api())

    # Monkeypatch DeepLAdapter to avoid external calls and return English text
    class FakeDeepL:
        def __init__(self, key):
            pass

        def translate_batch(self, texts, target, source=None):
            return [t + " [EN]" for t in texts]

    monkeypatch.setattr("src.core.services.translation.DeepLAdapter", FakeDeepL)

    # Set TTL to 1 second so the existing cache is considered expired
    monkeypatch.setenv("DEEPL_CACHE_TTL_SECONDS", "1")
    # Disable LLM refinement for deterministic output
    monkeypatch.setenv("REFINE_TRANSCRIPTS", "false")
    # Ensure DeepL fallback is enabled (we mock DeepLAdapter)
    monkeypatch.setenv("DEEPL_API_KEY", "fake-key")

    # Run extraction - it should use DeepL (fake) and not the expired cache
    res = extract_transcript(video_id)
    assert isinstance(res, list)
    assert res[0]["text"].endswith("[EN]")

    # cleanup cache file
    try:
        cache_path.unlink()
    except Exception:
        pass
