"""LLM provider interface.

Defines the abstract LLMProvider interface used across the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers (Strategy pattern)."""

    @abstractmethod
    def generate_sections(
        self,
        transcript: list[dict[str, Any]],
        num_sections: int,
        max_retries: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate section timestamps from transcript."""
        raise NotImplementedError

    @abstractmethod
    def generate_text(
        self, prompt: str, max_tokens: int = 512, temperature: float | None = None
    ) -> str:
        """Generate text for a given prompt."""
        raise NotImplementedError

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return provider metadata (model, device, backend)."""
        raise NotImplementedError


__all__ = ["LLMProvider"]
