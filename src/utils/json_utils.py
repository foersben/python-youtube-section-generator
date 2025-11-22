from __future__ import annotations

import json
import re
from typing import Any, cast


def _ensure_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array")
    for i, el in enumerate(value):
        if not isinstance(el, dict):
            raise ValueError(f"Array element at index {i} is not an object/dict")
    return cast(list[dict[str, Any]], value)


def extract_json(response_text: str) -> list[dict[str, Any]]:
    """Extracts JSON from model response, handling Markdown code blocks.

    Args:
        response_text: Raw text response from AI model

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If JSON cannot be extracted
    """

    try:
        parsed = json.loads(response_text)
        parsed_list: list[dict[str, Any]] = _ensure_list_of_dicts(parsed)
        return parsed_list
    except json.JSONDecodeError:
        # Handle Markdown code block format
        match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", response_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                parsed_from_codeblock: list[dict[str, Any]] = _ensure_list_of_dicts(parsed)
                return parsed_from_codeblock
            except json.JSONDecodeError:
                pass

        # Try to find the first JSON array
        array_match = re.search(r"(\[.*\])", response_text, re.DOTALL)
        if array_match:
            try:
                parsed = json.loads(array_match.group(1).strip())
                parsed_from_array: list[dict[str, Any]] = _ensure_list_of_dicts(parsed)
                return parsed_from_array
            except json.JSONDecodeError:
                pass

    raise ValueError("Failed to extract JSON from response")
