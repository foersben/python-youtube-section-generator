import pytest

from src.core.adapters import local_llm_client


def test_local_llm_client_shim_raises():
    with pytest.raises(RuntimeError):
        local_llm_client.LocalLLMClient()
