import json
from pathlib import Path
from typing import Any


def write_to_file(content: list[dict[str, Any]] | dict[str, Any] | str, filepath: str) -> None:
    """Writes content to a file in appropriate format.

    Args:
        content: Data to write (list of dicts, single dict, or string)
        filepath: Output file path

    Raises:
        TypeError: If content is not a supported type
        IOError: If file cannot be written
    """

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            if isinstance(content, (list, dict)):
                json.dump(content, f, indent=2, ensure_ascii=False)
            elif isinstance(content, str):
                f.write(content)
            else:
                raise TypeError(f"Unsupported content type: {type(content)}")
    except Exception as e:
        raise OSError(f"Failed to write to file {filepath}: {str(e)}") from e


def read_json_file(filepath: str) -> Any:
    """Read JSON content from a file and return parsed data.

    Args:
        filepath: Path to JSON file.

    Returns:
        Parsed JSON object (list/dict) or raises OSError on failure.
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise OSError(f"Failed to read JSON from {filepath}: {e}") from e
