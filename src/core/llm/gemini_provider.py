"""Gemini API LLM provider using Adapter pattern.

Adapts Google Gemini API to the LLMProvider interface.
"""

import logging
import os
import time
from typing import Any

from .base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini API provider (Adapter pattern).

    Adapts the Google Gemini API to conform to our LLMProvider interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        """Initialize Gemini provider.

        Args:
            api_key: Google API key (None = use env var GOOGLE_API_KEY).
            model: Model name (None = use gemini-2.5-flash).
            temperature: Default sampling temperature.
        """
        try:
            from google import genai
            from google.genai import types

            self.genai = genai
            self.types = types
        except ImportError:
            raise RuntimeError(
                "google-genai not installed. " "Install with: poetry add google-genai"
            )

        # Get API key
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.default_temperature = temperature
        self.client = genai.Client(api_key=key)

        logger.info(f"Initialized Gemini provider: {self.model_name}")

    def generate_sections(
        self,
        transcript: list[dict[str, Any]],
        num_sections: int,
        max_retries: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate sections from transcript using Gemini."""
        if not transcript:
            raise ValueError("Transcript cannot be empty")

        # Build prompt
        prompt = self._build_section_prompt(transcript, num_sections)
        system_instruction = (
            "You are an expert at analyzing video transcripts and creating "
            "meaningful section timestamps. Generate clear, descriptive titles."
        )

        # Prepare cache contents
        cache_contents = [{"role": "user", "parts": [{"text": prompt}]}]

        for attempt in range(max_retries):
            try:
                logger.info(f"Generating sections (attempt {attempt + 1}/{max_retries})")

                # Generate with caching
                response = self._generate_with_cache(
                    prompt=prompt,
                    cache_contents=cache_contents,
                    system_instruction=system_instruction,
                )

                # Parse JSON response
                sections = self._extract_json(response)

                if self._validate_sections(sections, transcript):
                    logger.info(f"Successfully generated {len(sections)} sections")
                    return sections

                logger.warning(f"Validation failed on attempt {attempt + 1}")

            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError("Section generation failed after retries") from e

        raise RuntimeError("Section generation failed after retries")

    def generate_text(
        self, prompt: str, max_tokens: int = 512, temperature: float | None = None
    ) -> str:
        """Generate text from prompt using Gemini."""
        temp = temperature if temperature is not None else self.default_temperature

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text.strip()

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise

    def get_info(self) -> dict[str, Any]:
        """Get provider information."""
        return {
            "provider": "gemini",
            "model": self.model_name,
            "backend": "google-genai",
            "device": "cloud",
        }

    def _generate_with_cache(
        self,
        prompt: str,
        cache_contents: list[dict[str, Any]],
        system_instruction: str = "",
    ) -> str:
        """Generate with caching and retry logic."""
        cache = None
        try:
            # Create cache
            cache = self.client.caches.create(
                model=self.model_name,
                config=self.types.CreateCachedContentConfig(
                    system_instruction=system_instruction,
                    contents=cache_contents,
                ),
            )
            logger.debug(f"Created cache: {cache.name}")

            # Generate with exponential backoff
            delay = 5.0
            max_delay = 60.0

            while True:
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=self.types.GenerateContentConfig(
                            cached_content=cache.name,
                            temperature=self.default_temperature,
                        ),
                    )
                    return response.text.strip()

                except Exception as err:
                    msg = str(err).lower()
                    if "503" in msg and "unavailable" in msg:
                        logger.warning(f"Model overloaded (503). Retrying in {delay:.1f}s")
                        time.sleep(delay)
                        delay = min(delay * 2, max_delay)
                        continue
                    raise

        finally:
            # Cleanup cache
            if cache:
                try:
                    self.client.caches.delete(name=cache.name)
                    logger.debug(f"Deleted cache: {cache.name}")
                except Exception as e:
                    logger.warning(f"Cache cleanup failed: {e}")

    def _build_section_prompt(self, transcript: list[dict[str, Any]], num_sections: int) -> str:
        """Build prompt for section generation."""
        # Limit transcript for context window
        max_segments = 100
        transcript_text = "\n".join(
            f"[{seg['start']:.1f}s] {seg['text']}" for seg in transcript[:max_segments]
        )

        return f"""Analyze this video transcript and create {num_sections} section timestamps.

Transcript:
{transcript_text}

Generate a JSON array with sections. Each section should have:
- "title": Clear, descriptive title (4-7 words)
- "start": Start time in seconds (float)

Respond ONLY with the JSON array, no other text:
[
  {{"title": "Section Title", "start": 0.0}},
  ...
]

JSON:"""

    def _extract_json(self, text: str) -> list[dict[str, Any]]:
        """Extract JSON array from response."""
        import json
        import re

        # Find JSON array
        start = text.find("[")
        end = text.rfind("]") + 1

        if start != -1 and end > start:
            json_text = text[start:end]
            try:
                sections = json.loads(json_text)
                if isinstance(sections, list):
                    return sections
            except json.JSONDecodeError:
                pass

        # Regex fallback
        pattern = r'"title"\s*:\s*"([^"]+)"\s*,\s*"start"\s*:\s*(\d+\.?\d*)'
        matches = re.findall(pattern, text, re.IGNORECASE)

        sections = []
        for title, start in matches:
            sections.append({"title": title.strip(), "start": float(start)})

        if sections:
            return sections

        raise ValueError(f"Could not extract valid JSON from response")

    def _validate_sections(
        self, sections: list[dict[str, Any]], transcript: list[dict[str, Any]]
    ) -> bool:
        """Validate generated sections."""
        if not sections:
            return False

        max_time = max(seg["start"] for seg in transcript)

        for section in sections:
            if "title" not in section or "start" not in section:
                return False
            if not isinstance(section["start"], (int, float)):
                return False
            if section["start"] < 0 or section["start"] > max_time:
                return False

        return True


__all__ = ["GeminiProvider"]
