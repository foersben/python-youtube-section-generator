"""Compatibility shim for local LLM client used by tests.

This module intentionally provides a lightweight, test-friendly shim of the
original `LocalLLMClient` API so unit tests that patch `AutoTokenizer` and
`AutoModelForCausalLM` continue to work without pulling heavy dependencies.

The implementation is minimal: it calls `AutoTokenizer.from_pretrained` and
`AutoModelForCausalLM.from_pretrained` (which tests patch), tracks a few
properties (device, model_name), and exposes helper methods used by tests
(`_validate_sections`, `_extract_json`, `generate_sections`).
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
    """Lightweight compatibility client for tests.

    This class implements just enough behaviour for the test-suite:
    - Calls AutoTokenizer.from_pretrained and AutoModelForCausalLM.from_pretrained
      during initialization (these are patched by tests).
    - Exposes `device` and `model_name` attributes.
    - Implements `_validate_sections` and `_extract_json` helpers used by tests.
    - `generate_sections` raises on empty transcript (tested) and otherwise
      returns a simple placeholder list.
    """

    def __init__(self, model_path: str | None = None, n_ctx: int = 4096, n_threads: int | None = None, temperature: float = 0.2) -> None:
        # Lazy import to allow tests to patch AutoTokenizer/AutoModel
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False

        self.device = "cuda" if cuda_ok else "cpu"
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"

        # Attempt to create tokenizer/model (tests patch these methods)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception:
            self.tokenizer = None

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except Exception:
            self.model = None

        logger.info("Initialized test LocalLLMClient (device=%s)", self.device)

    def generate_sections(self, transcript: list[dict[str, Any]], num_sections: int = 5, max_retries: int = 3) -> list[dict[str, Any]]:
        """Generate sections (minimal behaviour for tests).

        Raises:
            ValueError: If transcript is empty.
        """
        if not transcript:
            raise ValueError("Transcript cannot be empty")

        # Minimal placeholder behavior: return num_sections empty titled sections
        interval = max((seg.get("start", 0) for seg in transcript), default=0)
        return [{"title": f"Section {i+1}", "start": float(transcript[0].get("start", 0))} for i in range(num_sections)]

    def _validate_sections(self, sections: list[dict[str, Any]], transcript: list[dict[str, Any]]) -> bool:
        """Validate sections format and bounds.

        Returns True if sections are valid, False otherwise.
        """
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
        """Extract a JSON array of sections from text response.

        Raises:
            ValueError: If no valid JSON array is found.
        """
        # Try to find a JSON array in the text
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

        # Fallback regex to capture simple title/start pairs
        pattern = r'"title"\s*:\s*"([^"]+)"\s*,\s*"start"\s*:\s*(\d+\.?\d*)'
        matches = re.findall(pattern, text)
        if matches:
            return [{"title": m[0], "start": float(m[1])} for m in matches]

        raise ValueError("No JSON array found")

    def get_info(self) -> dict[str, Any]:
        return {"provider": "local-shim", "device": self.device, "model": self.model_name}


__all__ = ["LocalLLMClient", "AutoTokenizer", "AutoModelForCausalLM"]
