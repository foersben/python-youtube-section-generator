from typing import Any
import re
from src.utils import file_io
from youtube_transcript_api import YouTubeTranscriptApi


def _convert_transcript_to_dict(transcript_data) -> list[dict[str, Any]]:
    """Converts transcript data to a serializable list of dictionaries.

    Args:
        transcript_data: Raw transcript data from YouTubeTranscriptApi

    Returns:
        List of dictionaries with:
        - 'text': Transcript text
        - 'start': Start time in seconds
        - 'duration': Duration in seconds
    """

    return [
        {"text": segment.text, "start": segment.start, "duration": segment.duration}
        for segment in transcript_data
    ]


def extract_transcript(
    video_id: str, output_file: str | None = None, translate_to: str | None = None
) -> list[dict[str, Any]]:
    """Extracts YouTube transcript and optionally translates it.

    Args:
        video_id: YouTube video ID (11-character string)
        output_file: Optional file path to save transcript
        translate_to: Optional language code for translation (e.g., 'en')

    Returns:
        List of transcript segments as dictionaries

    Raises:
        Exception: If transcript retrieval fails
    """

    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        transcripts = list(transcript_list)

        # Find the first manually created transcript, or fallback to generated
        transcript = next(
            (snippet for snippet in transcripts if not snippet.is_generated),
            transcripts[0],  # First transcript if no manually created found
        )
        # transcript = [line.fetch() for line in transcript_list][0]
        # transcript = next(
        #     (t for t in transcript_list if not t.is_generated), transcript_list[0]
        # )

        if translate_to:
            transcript_data = transcript.translate(translate_to).fetch()
        else:
            transcript_data = transcript.fetch()

        serializable_data = _convert_transcript_to_dict(transcript_data)

        print(f"\nTranscript Details:")
        print(f"- Video ID: {transcript.video_id}")
        print(f"- Language: {transcript.language} ({transcript.language_code})")
        print(f"- Generated: {'Yes' if transcript.is_generated else 'No'}")

        # transcript = transcript.to_raw_data()

        if output_file:
            file_io.write_to_file(serializable_data, output_file)
            print(f"Transcript saved to {output_file}")

        # return transcript
        return serializable_data

    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        raise


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL

    Args:
        url: ID or YouTube URL from which video ID has to be extracted

    Returns:
        Video ID

    Raises
        ValueError: in case an invalid ID or YouTube URL was passed.

    """

    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL or ID")
