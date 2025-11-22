"""Test-support LocalLLM shim moved to src for shared usage.

This module provides a lightweight LocalLLMClient shim used by tests and
developer scripts. It replaces the previous `src.core.local_llm_client` shim
so the core package surface can be simplified.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class AutoTokenizer:
    """Placeholder tokenizer used for tests (patched in unit tests)."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise ImportError("transformers not available; tests should patch this")


class AutoModelForCausalLM:
    """Placeholder model used for tests (patched in unit tests)."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise ImportError("transformers not available; tests should patch this")


class LocalLLMClient:
    """Lightweight compatibility client for tests and developer tools.

    Implemented to match the former test shim API used across tests.
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        temperature: float = 0.2,
    ) -> None:
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False

        self.device = "cuda" if cuda_ok else "cpu"
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception:
            self.tokenizer = None

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except Exception:
            self.model = None

        logger.info("Initialized test LocalLLMClient (device=%s)", self.device)

    def generate_sections(
        self, transcript: list[dict[str, Any]], num_sections: int = 5, max_retries: int = 3
    ) -> list[dict[str, Any]]:
        if not transcript:
            raise ValueError("Transcript cannot be empty")
        return [
            {"title": f"Section {i+1}", "start": float(transcript[0].get("start", 0))}
            for i in range(num_sections)
        ]

    def _validate_sections(
        self, sections: list[dict[str, Any]], transcript: list[dict[str, Any]]
    ) -> bool:
        if not sections:
            return False
        try:
            max_time = max(seg["start"] + seg.get("duration", 0) for seg in transcript)
        except Exception:
            return False
        for section in sections:
            if "title" not in section or "start" not in section:
                return False
            if not isinstance(section["start"], (int, float)):
                return False
            if section["start"] < 0 or section["start"] > max_time:
                return False
        return True

    def _extract_json(self, text: str) -> list[dict[str, Any]]:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        pattern = r'"title"\s*:\s*"([^"]+)"\s*,\s*"start"\s*:\s*(\d+\.?\d*)'
        matches = re.findall(pattern, text)
        if matches:
            return [{"title": m[0], "start": float(m[1])} for m in matches]
        raise ValueError("No JSON array found")

    def get_info(self) -> dict[str, Any]:
        return {"provider": "local-shim", "device": self.device, "model": self.model_name}


__all__ = ["LocalLLMClient", "AutoTokenizer", "AutoModelForCausalLM"]
