import json
import re


def extract_json(response_text: str) -> list[dict]:
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
