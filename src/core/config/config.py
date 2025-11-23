"""Configuration management implementation.

Contains the AppConfig class implementation.
Separated from __init__.py to follow best practices.
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AppConfig:
    """Application configuration (Singleton pattern).

    Centralizes all configuration management with environment variable support.

    This class should only be instantiated once (Singleton pattern).
    Use the `config` instance from __init__.py instead of creating new instances.
    """

    _instance: "AppConfig | None" = None
    _initialized: bool

    def __new__(cls) -> "AppConfig":
        """Singleton implementation - only one instance allowed."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        if self._initialized:
            return

        self._load_from_env()
        self._initialized = True
        logger.info("Configuration initialized")

    def _load_from_env(self) -> None:
        """Internal method to load values from environment variables."""
        # Project paths
        self.project_root = self._get_project_root()
        self.models_dir = self.project_root / "models"
        self.chromadb_dir = self.project_root / ".chromadb"

        # API Keys
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.deepl_api_key = os.getenv("DEEPL_API_KEY", "")

        # LLM Configuration
        self.use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        self.local_model_path = self._resolve_model_path(
            os.getenv("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
        )
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.llm_context_size = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))

        # RAG Configuration
        self.use_rag = os.getenv("USE_RAG", "auto").lower()
        self.rag_hierarchical = os.getenv("RAG_HIERARCHICAL", "true").lower() == "true"
        self.rag_chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
        self.rag_chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        self.rag_retrieval_k = int(os.getenv("RAG_RETRIEVAL_K", "5"))
        self.rag_min_duration = float(os.getenv("RAG_MIN_DURATION", "1800"))

        # Embeddings Configuration
        self.embeddings_model = os.getenv(
            "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embeddings_device = os.getenv("EMBEDDINGS_DEVICE", "cpu")

        # Section Generation Defaults
        self.default_min_sections = int(os.getenv("DEFAULT_MIN_SECTIONS", "10"))
        self.default_max_sections = int(os.getenv("DEFAULT_MAX_SECTIONS", "20"))
        self.default_min_title_words = int(os.getenv("DEFAULT_MIN_TITLE_WORDS", "3"))
        self.default_max_title_words = int(os.getenv("DEFAULT_MAX_TITLE_WORDS", "7"))
        self.pipeline_strategy = os.getenv("PIPELINE_STRATEGY", "legacy").lower()
        self.pipeline_section_overshoot = float(os.getenv("PIPELINE_SECTION_OVERSHOOT", "0.1"))

    def reload(self) -> None:
        """Force reload configuration from environment variables."""
        logger.info("Reloading configuration from environment...")
        self._load_from_env()

    def _get_project_root(self) -> Path:
        """Get project root directory. Assumes config is in src/core/config/"""
        return Path(__file__).parent.parent.parent.parent

    def _resolve_model_path(self, path_str: str) -> Path:
        """Resolve model path to absolute path."""
        path = Path(path_str)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def should_use_rag(self, video_duration: float) -> bool:
        """Determine if RAG should be used based on video duration."""
        if self.use_rag == "always":
            return True
        if self.use_rag == "never":
            return False
        return video_duration > self.rag_min_duration

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []
        if self.use_local_llm:
            if not self.local_model_path.exists():
                issues.append(f"Local model not found: {self.local_model_path}")
        if not self.use_local_llm and not self.google_api_key:
            issues.append("GOOGLE_API_KEY not set and local LLM disabled")
        return issues

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "use_local_llm": self.use_local_llm,
            "use_rag": self.use_rag,
            "pipeline_strategy": self.pipeline_strategy,
            # ... add others if needed for debugging
        }


__all__ = ["AppConfig"]
