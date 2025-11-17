"""High-level services providing simplified interfaces (Facade pattern).
Services encapsulate complex subsystems and provide clean, simple APIs
for common use cases.
"""

import logging
from typing import Any

from src.core.config import config
from src.core.services.repository import (
    JSONFileRepository,
    Repository,
    SectionsRepository,
    TranscriptRepository,
)
from src.core.services.section_generation import SectionGenerationService
from src.core.services.translation import DeepLAdapter, TranslationProvider

logger = logging.getLogger(__name__)

__all__ = [
    "SectionGenerationService",
    "Repository",
    "JSONFileRepository",
    "TranscriptRepository",
    "SectionsRepository",
    "TranslationProvider",
    "DeepLAdapter",
]
