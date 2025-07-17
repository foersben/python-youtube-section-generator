import os
from google import genai
from google.genai import types
from typing import Any
from src.utils import file_io, json_utils
import time

# MODEL="gemini-2.0-flash-lite-001" # best model that allows to cache context (for free)
# MODEL="gemini-1.5-flash-002" # best model that allows to cache context (for free)
MODEL="gemini-2.5-flash"

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

        models = client.models.list()
        for model in models:
            print(f"Model Name: {model.name}")
            print("Supported Methods:")
            for action in model.supported_actions:
                print(f" - {action}")
            print()  # Print a newline for better readability

        # Create a cached content object
        cache = client.caches.create(
            model=MODEL,
            config=types.CreateCachedContentConfig(
                system_instruction="Remember to each exact timestamp the text to summaries to section titles.",
                contents=[formatted_transcript],
            )
        )

        prompt = (
            "Create YouTube-style chapter markers from this transcript. "
            "Return ONLY valid JSON with this structure: "
            "[{'start': seconds, 'title': string}, ...]\n\n"
            "Rules:\n"
            f"1. Create {section_count_range[0]}-{section_count_range[1]} sections\n"
            "2. Start time in seconds (float)\n"
            "3. Always take the whole transcript and its timestamps into account\n"
            f"4. {title_length_range[0]}-{title_length_range[1]} word titles\n"
            "5. Capture key topics\n"
            "6. Output ONLY the JSON array\n"
            "7. In the language of the transcript.\n\n"
            # f"Transcript:\n{formatted_transcript}"
        )

        retry_delay = 5  # seconds – initial wait
        max_delay = 60   # cap so we don't wait forever between tries

        while True:
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cache.name,
                        # thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                        temperature=0.0,  # Adjust for creativity vs accuracy
                    )
                )
                break

            except Exception as err:
                # detect the "model is overloaded" case
                error_txt = str(err).lower()
                is_503 = ("503" in error_txt) and ("unavailable" in error_txt)

                if not is_503:
                    # Some other exception – re-raise immediately
                    raise

                # Otherwise, log and wait before the next attempt
                print(
                    f"Model overloaded (503). Retrying in {retry_delay}s …",
                    flush=True,
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)

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