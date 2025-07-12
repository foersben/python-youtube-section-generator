import os
from google import genai
from typing import Any
from src.utils import file_io, json_utils


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
        sections = json_utils.extract_json(response.text.strip())

        # Validate sections format
        # if not all(("start" in sec and "title" in sec) for sec in sections):
        #     raise ValueError("Invalid section format in AI response")
        if not isinstance(sections, list) or not all(
            isinstance(sec, dict) and "start" in sec and "title" in sec
            for sec in sections
        ):
            raise ValueError("Invalid section format in AI response")

        if output_file:
            file_io.write_to_file(sections, output_file)
            print(f"Section timestamps saved to {output_file}")

        return sections

    except Exception as e:
        print(f"Error generating sections: {e}")
        raise
