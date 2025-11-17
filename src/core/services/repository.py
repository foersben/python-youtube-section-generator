"""Repository pattern for data persistence.

Provides abstraction layer for storing and retrieving transcripts and sections.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Repository(ABC):
    """Abstract base class for repositories (Repository pattern)."""

    @abstractmethod
    def save(self, data: Any, identifier: str) -> None:
        """Save data with identifier."""
        pass

    @abstractmethod
    def load(self, identifier: str) -> Any:
        """Load data by identifier."""
        pass

    @abstractmethod
    def exists(self, identifier: str) -> bool:
        """Check if data exists."""
        pass

    @abstractmethod
    def delete(self, identifier: str) -> None:
        """Delete data by identifier."""
        pass


class JSONFileRepository(Repository):
    """File-based repository using JSON format.

    Implements Repository pattern for persisting data to JSON files.
    """

    def __init__(self, base_dir: str | Path = "."):
        """Initialize repository.

        Args:
            base_dir: Base directory for storing files.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized JSONFileRepository at {self.base_dir}")

    def save(self, data: Any, identifier: str) -> None:
        """Save data to JSON file.

        Args:
            data: Data to save (must be JSON-serializable).
            identifier: Unique identifier (becomes filename).

        Raises:
            IOError: If save fails.
        """
        file_path = self._get_file_path(identifier)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved data to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save to {file_path}: {e}")
            raise IOError(f"Save failed: {e}") from e

    def load(self, identifier: str) -> Any:
        """Load data from JSON file.

        Args:
            identifier: Unique identifier (filename).

        Returns:
            Loaded data.

        Raises:
            FileNotFoundError: If file doesn't exist.
            IOError: If load fails.
        """
        file_path = self._get_file_path(identifier)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded data from {file_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load from {file_path}: {e}")
            raise IOError(f"Load failed: {e}") from e

    def exists(self, identifier: str) -> bool:
        """Check if file exists.

        Args:
            identifier: Unique identifier (filename).

        Returns:
            True if file exists, False otherwise.
        """
        file_path = self._get_file_path(identifier)
        return file_path.exists()

    def delete(self, identifier: str) -> None:
        """Delete file.

        Args:
            identifier: Unique identifier (filename).

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        file_path = self._get_file_path(identifier)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            file_path.unlink()
            logger.info(f"Deleted {file_path}")

        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            raise IOError(f"Delete failed: {e}") from e

    def _get_file_path(self, identifier: str) -> Path:
        """Get full file path for identifier.

        Args:
            identifier: File identifier (without extension).

        Returns:
            Full path to JSON file.
        """
        # Ensure .json extension
        if not identifier.endswith(".json"):
            identifier = f"{identifier}.json"

        return self.base_dir / identifier


class TranscriptRepository(JSONFileRepository):
    """Specialized repository for transcripts.

    Provides transcript-specific convenience methods.
    """

    def save_transcript(self, transcript: list[dict[str, Any]], video_id: str) -> None:
        """Save transcript for video.

        Args:
            transcript: Transcript segments.
            video_id: YouTube video ID.
        """
        self.save(transcript, f"transcript_{video_id}")

    def load_transcript(self, video_id: str) -> list[dict[str, Any]]:
        """Load transcript for video.

        Args:
            video_id: YouTube video ID.

        Returns:
            Transcript segments.
        """
        return self.load(f"transcript_{video_id}")

    def has_transcript(self, video_id: str) -> bool:
        """Check if transcript exists for video.

        Args:
            video_id: YouTube video ID.

        Returns:
            True if transcript cached, False otherwise.
        """
        return self.exists(f"transcript_{video_id}")


class SectionsRepository(JSONFileRepository):
    """Specialized repository for sections.

    Provides section-specific convenience methods.
    """

    def save_sections(self, sections: list[dict[str, Any]], video_id: str) -> None:
        """Save sections for video.

        Args:
            sections: Generated sections.
            video_id: YouTube video ID.
        """
        self.save(sections, f"sections_{video_id}")

    def load_sections(self, video_id: str) -> list[dict[str, Any]]:
        """Load sections for video.

        Args:
            video_id: YouTube video ID.

        Returns:
            Section data.
        """
        return self.load(f"sections_{video_id}")

    def has_sections(self, video_id: str) -> bool:
        """Check if sections exist for video.

        Args:
            video_id: YouTube video ID.

        Returns:
            True if sections cached, False otherwise.
        """
        return self.exists(f"sections_{video_id}")


__all__ = [
    "Repository",
    "JSONFileRepository",
    "TranscriptRepository",
    "SectionsRepository",
]
