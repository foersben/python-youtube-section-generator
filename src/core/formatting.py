from typing import Any


def format_sections_for_youtube(sections: list[dict[str, Any]]) -> str:
    """Formats sections into YouTube description format.

    Args:
        sections: List of section dictionaries with:
            - 'start': Start time in seconds (float)
            - 'title': Section title (str)

    Returns:
        Formatted string ready for YouTube description with timestamps
        Example:
            1. 00:00 Introduction
            2. 01:25 Main Content

    Raises:
        KeyError: If required keys ('start', 'title') are missing
    """

    try:
        output = []
        for i, section in enumerate(sections, 1):
            minutes = int(section["start"] // 60)
            seconds = int(section["start"] % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"
            output.append(f"{i}. {timestamp} {section['title']}")
        return "\n".join(output)
    except KeyError as e:
        raise KeyError(f"Missing required key in section data: {str(e)}")


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
