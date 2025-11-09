"""Formatting utilities for transcript and section data.

Supports both legacy dict format and new Section objects.
"""

from typing import Any


def format_sections_for_youtube(sections: list[dict[str, Any] | Any]) -> str:
    """Formats sections into YouTube description format.

    Args:
        sections: List of section dictionaries or Section objects with:
            - 'start' (or .start): Start time in seconds (float)
            - 'title' (or .title): Section title (str)
            - 'level' (or .level): Optional hierarchy level (int)

    Returns:
        Formatted string ready for YouTube description with timestamps.
        Main sections use numbers, subsections use letters.
        Timestamps are in YouTube-clickable format (MM:SS or H:MM:SS).

        Example:
            1. 00:00 Introduction
               a. 00:15 Setup Instructions
               b. 00:45 First Steps
            2. 01:25 Main Content
               a. 01:40 Topic One
               b. 02:15 Topic Two

    Raises:
        KeyError: If required keys ('start', 'title') are missing
        AttributeError: If Section object missing required attributes
    """

    try:
        output = []
        main_section_counter = 0
        sub_section_counters = {}  # Track subsection counters per main section

        for section in sections:
            # Support both dict and Section objects
            if isinstance(section, dict):
                start = section["start"]
                title = section["title"]
                level = section.get("level", 0)
            else:
                # Assume it's a Section object
                start = section.start
                title = section.title
                level = getattr(section, "level", 0)

            # Format timestamp in YouTube-clickable format
            total_seconds = int(start)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60

            if hours > 0:
                timestamp = f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                timestamp = f"{minutes:02d}:{seconds:02d}"

            # Format based on hierarchy level
            if level == 0:
                # Main section
                main_section_counter += 1
                sub_section_counters[main_section_counter] = 0  # Reset subsection counter
                output.append(f"{main_section_counter}. {timestamp} {title}")
            else:
                # Subsection
                current_main = max(sub_section_counters.keys()) if sub_section_counters else 1
                sub_section_counters[current_main] += 1
                sub_letter = chr(ord('a') + (sub_section_counters[current_main] - 1))
                indent = "   "  # 3 spaces for alignment
                output.append(f"{indent}{sub_letter}. {timestamp} {title}")

        return "\n".join(output)

    except (KeyError, AttributeError) as e:
        raise ValueError(f"Invalid section data format: {str(e)}") from e


def format_transcript_for_display(transcript: list[dict[str, Any]]) -> str:
    """Formats transcript data for human-readable display.

    Converts the structured transcript data into a plain text format
    with each line showing the timestamp and corresponding text.

    Example Output:
        [0.0s] Hello and welcome to my video
        [2.5s] Today we'll be discussing AI
        [5.1s] First, let's look at the basics

    Args:
        transcript: List of transcript segment dictionaries. Each dictionary
          should contain:
          - 'start': Start time in seconds (float)
          - 'text': Transcript text content (str)

    Returns:
        Formatted transcript as a single string with line breaks
        between segments.

    Raises:
        KeyError: If required keys ('start', 'text') are missing
        TypeError: If 'start' is not a numeric value
    """

    return "\n".join(f"[{seg['start']:.1f}s] {seg['text']}" for seg in transcript)
