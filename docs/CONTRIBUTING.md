# Contributing Guide

Welcome! This guide explains how to contribute to the YouTube Transcript Section Generator.

## 🚀 Quick Start for Contributors

### Development Setup

1. **Fork and clone:**
   ```bash
   git clone https://github.com/your-username/python-youtube-transcript.git
   cd python-youtube-transcript
   ```

2. **Set up development environment:**
   ```bash
   # Install Poetry (if not installed)
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install

   # Install development dependencies
   poetry install --with dev

   # CPU-only PyTorch for development
   poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   poetry run pip install sentence-transformers chromadb
   ```

3. **Set up pre-commit hooks:**
   ```bash
   poetry run pre-commit install
   ```

4. **Run tests:**
   ```bash
   poetry run pytest
   ```

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes with tests
# Run tests frequently
poetry run pytest

# Format code
poetry run black src/
poetry run isort src/

# Run linting
poetry run ruff src/

# Type checking
poetry run mypy src/

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

## 📋 Development Standards

### Code Style

#### Python Standards
- **PEP 8** compliant with Black formatting
- **Type hints** required for all functions
- **Google-style docstrings** for documentation
- **Maximum line length**: 88 characters (Black default)

#### Example Function
```python
from typing import Any, Optional

def process_video(
    video_id: str,
    options: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Process a YouTube video and extract sections.

    Args:
        video_id: YouTube video ID (11 characters)
        options: Optional processing parameters

    Returns:
        Dictionary containing video metadata and sections

    Raises:
        ValueError: If video_id is invalid
        RuntimeError: If processing fails

    Example:
        >>> result = process_video("dQw4w9WgXcQ")
        >>> print(result["title"])
        'Rick Astley - Never Gonna Give You Up'
    """
    if not is_valid_video_id(video_id):
        raise ValueError(f"Invalid video ID: {video_id}")

    # Implementation here
    return {"video_id": video_id, "sections": []}
```

### Package Structure

#### Core Package Organization
```
src/core/
├── __init__.py           # Export public API
├── models/               # Data models and DTOs
├── services/             # Business logic services
├── adapters/             # External service integrations
├── llm/                  # LLM provider abstractions
├── embeddings/           # Embedding provider abstractions
├── retrieval/            # RAG system components
└── config/               # Configuration management
```

#### Utils Package
```
src/utils/
├── __init__.py           # Utility exports
├── logging_config.py     # Centralized logging
├── file_io.py            # File operations
└── json_utils.py         # JSON helpers
```

### Design Patterns

#### Service Layer Pattern
Use services for business logic orchestration:

```python
class SectionGenerationService:
    """Facade for section generation logic."""

    def __init__(self):
        self.translator = TranslationService()
        self.llm = LLMFactory.create()
        self.rag = RAGSystem()

    def generate_sections(self, transcript: list, config: Config) -> list[Section]:
        # Orchestrate translation, RAG, and LLM
        translated = self.translator.translate(transcript)
        indexed = self.rag.index(translated)
        sections = self.llm.generate_sections(indexed, config)
        return sections
```

#### Adapter Pattern
Wrap external dependencies:

```python
class DeepLAdapter:
    """Adapter for DeepL translation API."""

    def __init__(self, api_key: str):
        self.client = deepl.Translator(api_key)

    def translate(self, text: str, target_lang: str) -> str:
        result = self.client.translate_text(text, target_lang=target_lang)
        return result.text
```

#### Factory Pattern
Create objects based on configuration:

```python
class LLMFactory:
    @staticmethod
    def create(provider: str = "auto") -> LLMProvider:
        if provider == "local" or (provider == "auto" and LOCAL_MODEL_PATH):
            return LocalLLMProvider()
        elif provider == "openai":
            return OpenAIProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")
```

### Error Handling

#### Custom Exceptions
```python
class YouTubeTranscriptError(Exception):
    """Base exception for transcript-related errors."""
    pass

class TranslationError(YouTubeTranscriptError):
    """Translation service failures."""
    pass

class LLMGenerationError(YouTubeTranscriptError):
    """LLM generation failures."""
    pass
```

#### Error Handling Pattern
```python
def process_with_fallback(operation: Callable, *args, **kwargs):
    """Execute operation with fallback strategies."""
    try:
        return operation(*args, **kwargs)
    except TranslationError:
        logger.warning("Translation failed, proceeding without")
        return args[0]  # Return original text
    except LLMGenerationError:
        logger.warning("LLM failed, using heuristic fallback")
        return generate_heuristic_title(args[0])
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### Testing

#### Test Structure
```
tests/
├── unit/                 # Unit tests
│   ├── test_models.py
│   ├── test_services.py
│   └── test_adapters.py
├── integration/          # Integration tests
│   ├── test_full_pipeline.py
│   └── test_web_interface.py
└── fixtures/             # Test data
    ├── sample_transcript.json
    └── sample_sections.json
```

#### Test Example
```python
import pytest
from src.core.models import Section

class TestSection:
    def test_format_timestamp(self):
        section = Section("Test Title", 3661.0, 0)
        assert section.format_timestamp() == "1:01:01"

    def test_to_dict(self):
        section = Section("Title", 100.0, 1)
        expected = {
            "title": "Title",
            "start": 100.0,
            "level": 1
        }
        assert section.to_dict() == expected

    @pytest.mark.parametrize("seconds,expected", [
        (61, "01:01"),
        (3661, "1:01:01"),
        (7323, "2:02:03"),
    ])
    def test_timestamp_formats(self, seconds, expected):
        section = Section("Test", float(seconds), 0)
        assert section.format_timestamp() == expected
```

#### Running Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test
poetry run pytest tests/unit/test_models.py::TestSection::test_format_timestamp

# Run integration tests
poetry run pytest tests/integration/
```

### Logging

#### Logging Guidelines
- Use the centralized logging system from `src.utils.logging_config`
- Log levels: DEBUG (development), INFO (normal operations), WARNING (issues), ERROR (failures)
- Include relevant context in log messages
- Don't log sensitive information (API keys, transcript content)

#### Logging Examples
```python
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def process_video(video_id: str) -> dict:
    logger.info(f"Starting processing for video {video_id}")

    try:
        transcript = extract_transcript(video_id)
        logger.debug(f"Extracted {len(transcript)} segments")

        sections = generate_sections(transcript)
        logger.info(f"Generated {len(sections)} sections")

        return {"video_id": video_id, "sections": sections}

    except Exception as e:
        logger.error(f"Failed to process video {video_id}: {e}")
        raise
```

### Configuration

#### Environment Variables
- Use `.env` files for local development
- Document all environment variables in README
- Provide sensible defaults
- Validate configuration on startup

#### Configuration Example
```python
# config.py
from typing import Any, Optional
import os

class Config:
    # Translation settings
    deepl_api_key: Optional[str] = os.getenv("DEEPL_API_KEY")
    use_translation: bool = os.getenv("USE_TRANSLATION", "true").lower() == "true"

    # LLM settings
    use_local_llm: bool = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    model_path: str = os.getenv("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")

    # Validation
    def validate(self) -> None:
        if self.use_translation and not self.deepl_api_key:
            raise ValueError("DEEPL_API_KEY required when USE_TRANSLATION=true")

        if self.use_local_llm and not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

# Global config instance
config = Config()
config.validate()
```

## 🔧 Development Tools

### Code Quality Tools

#### Black (Code Formatting)
```bash
# Format code
poetry run black src/

# Check formatting
poetry run black --check src/
```

#### Ruff (Linting)
```bash
# Lint code
poetry run ruff src/

# Fix auto-fixable issues
poetry run ruff --fix src/
```

#### MyPy (Type Checking)
```bash
# Type check
poetry run mypy src/

# Strict type checking
poetry run mypy --strict src/
```

#### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Performance Profiling

#### Memory Profiling
```python
import tracemalloc
from src.core.services import SectionGenerationService

def profile_memory_usage():
    tracemalloc.start()

    service = SectionGenerationService()
    # Run your code here

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

    tracemalloc.stop()
```

#### Performance Benchmarking
```python
import time
from src.core.services import SectionGenerationService

def benchmark_processing(video_id: str) -> dict[str, float]:
    service = SectionGenerationService()

    start_time = time.time()
    result = service.generate_sections(video_id)
    total_time = time.time() - start_time

    return {
        "total_time": total_time,
        "sections_generated": len(result.get("sections", [])),
        "time_per_section": total_time / len(result.get("sections", []))
    }
```

## 📝 Documentation

### README Updates
- Keep README focused on user-facing information
- Update examples and performance benchmarks
- Document new features and breaking changes

### Code Documentation
- All public functions need docstrings
- Include type hints for parameters and return values
- Provide usage examples in docstrings
- Document exceptions that can be raised

### Changelog
- Maintain chronological record of changes
- Group changes by type (Added, Changed, Fixed, Removed)
- Include breaking changes and migration notes

## 🚀 Release Process

### Version Numbering
Follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create git tag
- [ ] Build and test distribution
- [ ] Publish to PyPI

### Building Distribution
```bash
# Build distribution
poetry build

# Test installation
pip install dist/*.whl --force-reinstall

# Test basic functionality
python -c "from src.core.services import SectionGenerationService; print('OK')"
```

## 🤝 Code Review Guidelines

### PR Requirements
- [ ] Tests included for new features
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Type hints added
- [ ] No linting errors
- [ ] Backward compatibility maintained (unless breaking change)

### Review Checklist
- [ ] Code is readable and well-documented
- [ ] Tests cover edge cases
- [ ] Performance impact considered
- [ ] Security implications reviewed
- [ ] Error handling appropriate
- [ ] Configuration changes documented

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and help
- **Pull Request Comments**: Code review feedback

### Issue Reporting
When reporting bugs, please include:
1. **Python version**: `python --version`
2. **Dependencies**: `poetry show`
3. **Error traceback**: Full error message
4. **Steps to reproduce**: Minimal example
5. **Expected vs actual behavior**: Clear description

### Feature Requests
For new features, please provide:
1. **Use case**: Why is this feature needed?
2. **Implementation idea**: How should it work?
3. **Impact**: Does it affect existing functionality?

Thank you for contributing to the YouTube Transcript Section Generator! 🎉
