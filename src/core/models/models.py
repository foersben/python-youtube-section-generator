"""Data models for sections and transcript segments.

Provides Section, SectionGenerationConfig, and TranscriptSegment dataclasses used
across the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Section:
    """Represents a video section (timestamped).

    Attributes:
        title: Human readable title for the section.
        start: Start time in seconds.
        end: Optional end time in seconds.
        duration: Optional duration in seconds.
        level: Hierarchy level (0 = top-level).
        description: Optional longer description or bullets.
    """

    title: str | None = None
    start: float = 0.0
    end: float | None = None
    duration: float | None = None
    level: int = 0
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert Section to plain dict suitable for JSON/formatting.

        Returns:
            A dict with keys for title, start, end, duration, level, description.
        """
        return {
            "title": self.title or "",
            "start": float(self.start) if self.start is not None else 0.0,
            "end": float(self.end) if self.end is not None else None,
            "duration": float(self.duration) if self.duration is not None else None,
            "level": int(self.level),
            "description": self.description or "",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Section:
        """Create Section instance from dictionary-like data.

        Args:
            data: Mapping with keys like 'title' and 'start'.

        Returns:
            Section instance.
        """
        # Use safe coercions to avoid passing None to float()
        raw_start = data.get("start")
        raw_end = data.get("end")
        raw_duration = data.get("duration")

        return cls(
            title=data.get("title") or data.get("name"),
            start=float(raw_start or 0.0),
            end=(float(raw_end) if raw_end is not None else None),
            duration=(float(raw_duration) if raw_duration is not None else None),
            level=int(data.get("level", 0)),
            description=data.get("description"),
        )


@dataclass
class SectionGenerationConfig:
    """Configuration for section generation.

    Attributes:
        min_sections: Minimum number of sections to generate.
        max_sections: Maximum number of sections to generate.
        min_title_words: Minimum desired words in generated titles.
        max_title_words: Maximum desired words in generated titles.
        use_hierarchical: Whether to produce hierarchical sections.
        temperature: Sampling temperature for LLMs.
    """

    min_sections: int = 5
    max_sections: int = 15
    min_title_words: int = 3
    max_title_words: int = 6
    use_hierarchical: bool = False
    temperature: float = 0.2

    @property
    def title_length_range(self) -> tuple[int, int]:
        """Return the configured title length bounds as a tuple (min, max)."""
        return (self.min_title_words, self.max_title_words)


@dataclass
class TranscriptSegment:
    """Represents a single transcript segment.

    Attributes:
        start: Start time in seconds for the segment.
        text: Transcript text for the segment.
        duration: Optional segment duration in seconds.
        original_language_code: Optional original language code if known.
    """

    start: float = 0.0
    text: str = ""
    duration: float | None = None
    original_language_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert TranscriptSegment to dict."""
        result: dict[str, Any] = {"start": float(self.start), "text": self.text}
        if self.duration is not None:
            result["duration"] = float(self.duration)
        if self.original_language_code is not None:
            result["original_language_code"] = self.original_language_code
        return result
