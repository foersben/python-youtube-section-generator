from dotenv import load_dotenv
import os
import json
import re
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from typing import Any, Union

# Load environment variables once at startup
load_dotenv()


def write_to_file(content: Union[list, dict, str], filepath: str) -> None:
    """Writes content to a file in appropriate format.

    Args:
        content: Data to write (list, dict, or string)
        filepath: Output file path
    """

    with open(filepath, "w") as f:
        if isinstance(content, (list, dict)):
            json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            f.write(content)


def _extract_json(response_text: str) -> list[dict]:
    """Extracts JSON from model response, handling Markdown code blocks.

    Args:
        response_text: Raw text response from AI model

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If JSON cannot be extracted
    """

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Handle Markdown code block format
        match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find the first JSON array
        array_match = re.search(r"(\[.*\])", response_text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(1).strip())
            except json.JSONDecodeError:
                pass

    raise ValueError("Failed to extract JSON from response")


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
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        transcript = [line.fetch() for line in transcript_list][0]

        print(f"\nTranscript Details:")
        print(f"- Video ID: {transcript.video_id}")
        print(f"- Language: {transcript.language} ({transcript.language_code})")
        print(f"- Generated: {'Yes' if transcript.is_generated else 'No'}")

        transcript = transcript.to_raw_data()

        if output_file:
            write_to_file(transcript, output_file)
            print(f"Transcript saved to {output_file}")

        return transcript

    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        raise


def create_section_timestamps(
    transcript: list[dict[str, Any]],
    section_count_range: tuple[int, int] = (15, 20),
    title_length_range: tuple[int, int] = (3, 8),
    output_file: str | None = None,
) -> list[dict[str, Any]]:
    """Generates YouTube-style section timestamps using AI.

    Args:
        transcript: Transcript data from extract_transcript()
        section_count_range: Number of sections, [lower limit, upper limit]
        title_length_range: Words in the title, [lower limit, upper limit]
        output_file: Optional file path to save sections

    Returns:
        List of sections with start times and titles

    Raises:
        Exception: If API call fails or response is invalid
    """

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        formatted_transcript = "\n".join(
            f"[{segment['start']:.1f}s] {segment['text']}" for segment in transcript
        )

        client = genai.Client(api_key=api_key)

        prompt = (
            "Create YouTube-style chapter markers from this transcript. "
            "Return ONLY valid JSON with this structure: "
            "[{'start': seconds, 'title': string}, ...]\n\n"
            "Rules:\n"
            f"1. Create {section_count_range[0]}-{section_count_range[1]} sections\n"
            "2. Start time in seconds (float)\n"
            f"3. {title_length_range[0]}-{title_length_range[1]} word titles\n"
            "4. Capture key topics\n"
            "5. Output ONLY the JSON array\n"
            "6. In the language of the script\n\n"
            f"Transcript:\n{formatted_transcript}"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            # config=types.GenerateContentConfig(
            #     thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
            # ),
        )
        sections = _extract_json(response.text.strip())

        # Validate sections format
        if not all(("start" in sec and "title" in sec) for sec in sections):
            raise ValueError("Invalid section format in AI response")

        if output_file:
            write_to_file(sections, output_file)
            print(f"Section timestamps saved to {output_file}")

        return sections

    except Exception as e:
        print(f"Error generating sections: {e}")
        raise


def format_sections_for_youtube(sections: list[dict[str, Any]]) -> str:
    """Formats sections into YouTube description format.

    Args:
        sections: Section data from create_section_timestamps()

    Returns:
        Formatted string ready for YouTube description
    """

    output = []

    for i, section in enumerate(sections, 1):
        # Convert seconds to MM:SS format
        minutes = int(section["start"] // 60)
        seconds = int(section["start"] % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"

        output.append(f"{i}. {timestamp} {section['title']}")

    return "\n".join(output)


if __name__ == "__main__":
    VIDEO_ID = "kXhCEyix180"  # "G63yfp-owY8"  # Replace with your video ID
    TRANSCRIPT_FILE = "transcript.json"
    SECTIONS_FILE = "sections.json"

    try:
        print("Fetching transcript...")
        transcript = extract_transcript(
            video_id=VIDEO_ID,
            output_file=TRANSCRIPT_FILE,
            # translate_to="en",  # Remove for original language
        )

        print("\nGenerating section timestamps...")
        sections = create_section_timestamps(
            transcript=transcript, output_file=SECTIONS_FILE
        )

        print("\nYouTube-ready Section Timestamps:")
        youtube_format = format_sections_for_youtube(sections)
        print("\n" + youtube_format)

        with open("youtube_sections.txt", "w") as f:
            f.write(youtube_format)
        print("\nYouTube-formatted sections saved to youtube_sections.txt")

    except Exception as e:
        print(f"Processing failed: {e}")
