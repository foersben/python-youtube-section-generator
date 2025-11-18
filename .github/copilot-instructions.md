# GitHub Copilot Workspace Instructions

## ⚠️ CRITICAL: File Location Rules

**NEVER create files outside the project directory structure:**
- ❌ NEVER use `/tmp/` for any files
- ❌ NEVER use `/` (root) or any system directories
- ❌ NEVER create files in the project root unless they are configuration files
- ✅ Source code: ONLY in `src/`
- ✅ Tests: ONLY in `tests/`
- ✅ Documentation: ONLY in `docs/`
- ✅ Scripts: ONLY in `scripts/` (if needed)

**File creation rules:**
- Test files → `tests/test_*.py`
- Documentation → `docs/*.md` (user-facing docs only)
- Operational status / run reports → `.github/status/*.md` (see policy below)
- Example scripts → `scripts/*.py`
- Source code → `src/**/*.py`
- Never create summary/report files in project root - use `.github/status/` for operational status or `docs/` for user-facing docs

## Project Overview
This is a YouTube transcript section generator built with Python 3.11+, featuring:
- YouTube transcript extraction using `youtube-transcript-api`
- AI-powered section timestamp generation using Google Gemini API or local LLM
- Support for local models (Meta-Llama-3-8B-Instruct) as alternative to cloud APIs
- Flask web application with REST API
- Command-line interface for batch processing
- Multi-language support with translation capabilities
- JSON-based data persistence

## Architecture Principles

### Design Patterns in Use
1. **Service Pattern**: `GeminiService` encapsulates AI client interactions
2. **Separation of Concerns**: Clear separation between core logic, web app, and utilities
3. **Error Handling**: Retry mechanisms with fallback strategies for AI failures
4. **Configuration as Code**: Environment-based configuration with `.env` support

### Package Structure
```
project_root/
├── src/             # Source code ONLY
│   ├── core/            # Core business logic
│   │   ├── transcript.py      # YouTube transcript extraction
│   │   ├── sections.py        # Section generation and validation
│   │   ├── formatting.py      # Output formatting utilities
│   │   ├── gemini_client.py   # Google Gemini AI client
│   │   ├── local_llm_client.py # Local LLM client (Llama-3-8B)
│   │   └── deepl_client.py    # DeepL translation client
│   ├── utils/           # Shared utilities
│   │   ├── file_io.py       # File read/write operations
│   │   └── json_utils.py    # JSON manipulation helpers
│   ├── templates/       # Flask HTML templates
│   ├── main.py          # CLI entry point
│   └── web_app.py       # Flask web application
├── tests/           # Tests ONLY (pytest)
│   ├── __init__.py
│   ├── conftest.py      # Shared fixtures
│   ├── test_transcript.py
│   ├── test_sections.py
│   └── test_*.py        # All test files here
├── docs/            # Documentation ONLY
│   ├── setup.md
│   ├── api.md
│   └── *.md         # All documentation here
├── scripts/         # Utility scripts (if needed)
│   └── *.py         # Helper scripts only
├── .github/         # GitHub config (CI, Copilot instructions)
├── pyproject.toml   # Poetry config
├── poetry.toml      # Poetry settings
├── .env             # Environment variables (git-ignored)
└── README.md        # Main project README
```

**Important**: Never create files in project root except configuration files!

## Code Generation Guidelines

### Type Hints
- **Always use built-in generics**: `list[dict[str, Any]]`, `dict[str, object]`, `tuple[float, float]`
- **Never use**: `List`, `Dict`, `Tuple` from typing module (use built-ins available in Python 3.11+)
- **Prefer precise types**: Use `Callable[[ArgType], ReturnType]` instead of bare `callable`
- Use `Any` from typing module for truly dynamic data (like API responses)
- Use union types: `str | None` instead of `Optional[str]`
- Always type annotate function parameters and return values
- Use `logging.Logger` for logger type hints

### Docstrings (Google Style - Required)
Every public module, class, and function must have:
```python
"""Module-level docstring.

Brief description of what this module provides.
"""

class MyService:
    """Brief class description.

    Longer explanation if needed.

    Attributes:
        attr_name: Description of attribute.
        another_attr: Description.
    """

    def process(self, param: str) -> dict[str, Any]:
        """Brief method description.

        Args:
            param: Description of param.

        Returns:
            Description of return value.

        Raises:
            ValueError: When and why.
        """
```

For Flask routes, document request/response format:
```python
@app.route("/api/endpoint", methods=["POST"])
def endpoint() -> jsonify:
    """Endpoint description.

    Request Form Parameters:
        param1: Description
        param2: Description (optional)

    Returns:
        JSON response with:
        - success: Boolean status
        - data: Result data (if successful)
        - error: Error message (if failed)

    HTTP Status Codes:
        200: Success
        400: Bad request
        500: Server error
    """
```

### Module Organization
- Each module should start with a module-level docstring
- Group imports: stdlib, third-party, local (separated by blank lines)
- Keep modules focused (single responsibility)
- Prefer multiple small modules over one large module

### Error Handling
- Always use logging for errors, warnings, and important info
- Guard against None values explicitly: `if value is None:`
- Prefer specific exceptions over generic ones
- Always include meaningful error messages with context
- Use try-except blocks for external API calls
- Implement retry mechanisms for transient failures
- Provide fallback strategies when AI/API calls fail
- Log exceptions with appropriate levels (error, warning, info)

Example:
```python
import logging

logger = logging.getLogger(__name__)

try:
    result = external_api_call()
    logger.info(f"Successfully processed: {result}")
except ValueError as e:
    logger.error(f"Validation failed: {e}")
    raise
except Exception as e:
    logger.exception("Unexpected error occurred")
    raise
```

### Flask Web Application
- Use Flask's `jsonify` for JSON responses
- Return appropriate HTTP status codes (200, 400, 404, 500)
- Use `request.form` for POST form data, `request.args` for query params
- Log all incoming requests with relevant details
- Handle frozen app detection for PyInstaller builds
- Use proper template and static folder configuration
- Add CORS headers when needed for cross-origin requests

### External API Integration
- Store API keys in environment variables, never in code
- Use `python-dotenv` for local development
- Implement retry logic with exponential backoff
- Handle rate limits gracefully
- Validate API responses before processing
- Provide meaningful error messages when APIs fail
- Cache expensive API calls when appropriate

### File I/O
- Use `pathlib.Path` for path manipulation
- Always specify encoding='utf-8' for text files
- Use context managers (`with` statements) for file operations
- Validate file paths before reading/writing
- Handle JSON serialization with `indent=2` and `ensure_ascii=False`
- Use try-except to catch IOError and provide context

### Testing
- Use pytest for testing (add `pytest` to dev dependencies)
- **Store ALL tests in `tests/` directory** - never in project root or src/
- Mirror `src/` structure in `tests/` directory
- Use descriptive test names: `test_<feature>_<scenario>_<expected_outcome>`
- Mock external API calls (YouTube API, Gemini API, Local LLM) in tests
- Use `@pytest.mark.parametrize` for data-driven tests
- Test both success and error cases
- Test edge cases (empty transcripts, invalid video IDs, etc.)
- Use fixtures for common test data

Example test structure:
```python
# tests/test_transcript.py  ← ALWAYS in tests/ directory!
import pytest
from unittest.mock import Mock, patch

def test_extract_transcript_success():
    """Test successful transcript extraction."""
    # Arrange
    video_id = "test_video_id"

    # Act & Assert
    with patch('youtube_transcript_api.YouTubeTranscriptApi') as mock_api:
        mock_api.return_value.list.return_value = [mock_transcript]
        result = extract_transcript(video_id)
        assert len(result) > 0
```

## Common Patterns in This Project

### Service Class Pattern
```python
import logging
import os

logger = logging.getLogger(__name__)

class MyService:
    """Service for handling external API interactions."""

    def __init__(self, api_key_env: str = 'API_KEY'):
        api_key = os.getenv(api_key_env)
        if not api_key:
            logger.error(f"Environment variable {api_key_env} not set")
            raise ValueError(f"{api_key_env} not found")

        self.client = SomeClient(api_key=api_key)
        logger.info("Initialized MyService")
```

### Retry Pattern with Fallback
```python
def process_with_retry(data: list[dict[str, Any]], max_attempts: int = 3) -> list[dict[str, Any]]:
    """Process data with retry logic and fallback."""
