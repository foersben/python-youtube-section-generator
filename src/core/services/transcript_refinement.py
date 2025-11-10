"""Transcript refinement service using LLM for intelligent cleaning and correction.

This service uses AI to:
- Remove filler words contextually (only when they add no value)
- Correct phonetic errors based on surrounding context
- Fix grammar and sentence structure
- Preserve technical terms and important context
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.core.llm.factory import LLMProviderFactory

logger = logging.getLogger(__name__)


class TranscriptRefinementService:
    """Service for AI-powered transcript refinement and correction."""

    def __init__(self) -> None:
        """Initialize the transcript refinement service."""
        self.llm_provider = LLMProviderFactory.create()
        logger.info("TranscriptRefinementService initialized")

    def refine_transcript_segment(self, text: str, context: str = "") -> str:
        """Refine a single transcript segment using LLM.

        Args:
            text: The transcript text to refine.
            context: Optional surrounding context for better corrections.

        Returns:
            Refined and corrected text.

        Raises:
            Exception: If LLM refinement fails.
        """
        prompt = self._build_refinement_prompt(text, context)

        try:
            refined_text = self.llm_provider.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,  # Lower temperature for more deterministic corrections
            )

            # Extract just the refined text (remove any markdown or extra formatting)
            refined_text = refined_text.strip()
            if refined_text.startswith('"') and refined_text.endswith('"'):
                refined_text = refined_text[1:-1]

            logger.debug(f"Refined segment: '{text[:50]}...' -> '{refined_text[:50]}...'")
            return refined_text

        except Exception as e:
            logger.error(f"Failed to refine transcript segment: {e}")
            # Fallback to original text if refinement fails
            return text

    def refine_transcript_batch(
        self, segments: list[dict[str, Any]], batch_size: int = 5
    ) -> list[dict[str, Any]]:
        """Refine transcript segments in batches for efficiency.

        Args:
            segments: List of transcript segments with 'text', 'start', 'duration'.
            batch_size: Number of segments to process together for context.

        Returns:
            List of refined transcript segments.
        """
        refined_segments = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]

            # Build context from surrounding segments
            context_before = ""
            if i > 0:
                context_before = " ".join(s["text"] for s in segments[max(0, i - 2) : i])

            context_after = ""
            if i + batch_size < len(segments):
                context_after = " ".join(
                    s["text"] for s in segments[i + batch_size : min(len(segments), i + batch_size + 2)]
                )

            # Refine the batch together for better context
            refined_batch = self._refine_batch_with_context(batch, context_before, context_after)
            refined_segments.extend(refined_batch)

            logger.info(f"Refined batch {i // batch_size + 1}/{(len(segments) + batch_size - 1) // batch_size}")

        return refined_segments

    def _refine_batch_with_context(
        self, batch: list[dict[str, Any]], context_before: str, context_after: str
    ) -> list[dict[str, Any]]:
        """Refine a batch of segments with surrounding context.

        Args:
            batch: Batch of transcript segments.
            context_before: Text from previous segments.
            context_after: Text from following segments.

        Returns:
            Refined batch of segments.
        """
        # Combine batch texts
        batch_text = " ".join(s["text"] for s in batch)

        # Build full context
        full_context = f"{context_before} {batch_text} {context_after}".strip()

        prompt = self._build_batch_refinement_prompt(batch_text, context_before, context_after)

        try:
            refined_text = self.llm_provider.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,
            )

            # Parse the refined text back into segments
            refined_segments = self._split_refined_text(refined_text, batch)
            return refined_segments

        except Exception as e:
            logger.error(f"Failed to refine batch: {e}")
            # Return original batch on failure
            return batch

    def _build_refinement_prompt(self, text: str, context: str) -> str:
        """Build the LLM prompt for single segment refinement.

        Args:
            text: Text to refine.
            context: Surrounding context.

        Returns:
            LLM prompt string.
        """
        return f"""You are a transcript refinement expert. Your task is to clean and correct transcript text while preserving meaning and important details.

**Instructions:**
1. Remove filler words ONLY when they add no semantic value (uh, um, hmm, äh, ähm, etc.)
2. Correct phonetic errors and misspellings based on context
3. Fix grammar and sentence structure
4. Preserve technical terms, proper nouns, and domain-specific vocabulary
5. Maintain the original meaning and intent
6. Keep the text concise and natural

**Context (for reference):**
{context if context else "No additional context"}

**Text to refine:**
{text}

**Refined text (return ONLY the cleaned text, no explanations):**"""

    def _build_batch_refinement_prompt(
        self, batch_text: str, context_before: str, context_after: str
    ) -> str:
        """Build the LLM prompt for batch refinement with context.

        Args:
            batch_text: The batch text to refine.
            context_before: Previous context.
            context_after: Following context.

        Returns:
            LLM prompt string.
        """
        return f"""You are a transcript refinement expert. Clean and correct the following transcript segment while considering the surrounding context.

**Instructions:**
1. Remove filler words ONLY when contextually appropriate (uh, um, hmm, äh, ähm, like, you know, etc.)
2. Correct phonetic/spelling errors using the full context
3. Fix grammar while preserving the speaker's intent
4. Keep technical terms and proper nouns accurate
5. Maintain natural flow and readability
6. Return ONLY the refined text, preserving paragraph breaks

**Previous context:**
{context_before if context_before else "Start of transcript"}

**TEXT TO REFINE:**
{batch_text}

**Following context:**
{context_after if context_after else "End of transcript"}

**Refined text (return ONLY the cleaned text):**"""

    def _split_refined_text(
        self, refined_text: str, original_batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Split refined text back into segments matching original structure.

        Args:
            refined_text: The refined text from LLM.
            original_batch: Original segments with timestamps.

        Returns:
            List of segments with refined text and original timestamps.
        """
        # Simple approach: distribute refined text proportionally to original segments
        refined_text = refined_text.strip()

        # Remove any markdown formatting
        if refined_text.startswith('"') and refined_text.endswith('"'):
            refined_text = refined_text[1:-1]

        # Split by sentences or natural breaks
        import re

        sentences = re.split(r'(?<=[.!?])\s+', refined_text)

        # If we have the same number of sentences as segments, map 1:1
        if len(sentences) == len(original_batch):
            return [
                {
                    "text": sentences[i].strip(),
                    "start": seg["start"],
                    "duration": seg["duration"],
                }
                for i, seg in enumerate(original_batch)
            ]

        # Otherwise, distribute proportionally
        words = refined_text.split()
        total_words = len(words)
        original_lengths = [len(seg["text"].split()) for seg in original_batch]
        total_original = sum(original_lengths)

        refined_segments = []
        word_index = 0

        for i, seg in enumerate(original_batch):
            # Calculate proportional word count
            if total_original > 0:
                proportion = original_lengths[i] / total_original
                word_count = max(1, int(total_words * proportion))
            else:
                word_count = max(1, total_words // len(original_batch))

            # Extract words for this segment
            segment_words = words[word_index : word_index + word_count]
            word_index += word_count

            refined_segments.append(
                {
                    "text": " ".join(segment_words),
                    "start": seg["start"],
                    "duration": seg["duration"],
                }
            )

        return refined_segments


__all__ = ["TranscriptRefinementService"]

