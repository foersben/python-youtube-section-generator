import pytest

from src.core.adapters import deepl_client, gemini_client


def test_deepl_client_shim_raises():
    with pytest.raises(RuntimeError):
        deepl_client.DeepLClient()


def test_gemini_service_shim_raises():
    with pytest.raises(RuntimeError):
        gemini_client.GeminiService()
