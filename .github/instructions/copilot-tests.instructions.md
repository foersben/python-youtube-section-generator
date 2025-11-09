---
applyTo:
  - "tests/**/*.py"
---

# Test File Instructions

## ⚠️ CRITICAL: Test File Locations

**ALL test files MUST be in the `tests/` directory:**
- ✅ `tests/test_*.py` - Correct location
- ❌ `test_*.py` in project root - WRONG
- ❌ `/tmp/test_*.py` - NEVER use /tmp/
- ❌ Any other location - WRONG

**Structure:**
```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_transcript.py       # Tests for src/core/transcript.py
├── test_sections.py         # Tests for src/core/sections.py
├── test_gemini_client.py    # Tests for src/core/gemini_client.py
├── test_local_llm.py        # Tests for src/core/local_llm_client.py
└── test_web_app.py          # Tests for src/web_app.py
```

## Testing Framework
- Use **pytest** exclusively (no unittest)
- Store all tests in `tests/` directory
- Name test files: `test_<feature>.py`
- Name test functions: `test_<feature>_<scenario>_<expected_outcome>`

## Test Structure

### Basic Test Pattern
```python
"""Tests for <feature> functionality.

This module tests <brief description of what's being tested>.
"""

import pytest
from src.module import Feature


def test_feature_with_valid_input_returns_expected_output():
    """Test that feature handles valid input correctly."""
    # Arrange
    feature = Feature()
    
    # Act
    result = feature.process("input")
    
    # Assert
    assert result == "expected"
```

### Fixtures (use conftest.py)
```python
@pytest.fixture
def network_config():
    """Provide a standard network configuration for tests."""
    return NetworkConfig(
        graph=build_test_graph(),
        paths={},
        mean_types=3,
        # ... other config
    )
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    (0, 0),
    (1, 1),
    (5, 25),
])
def test_feature_with_various_inputs(input, expected):
    """Test feature with multiple input values."""
    assert feature(input) == expected
```

## Test Categories

### Unit Tests (majority)
- Test single functions/methods in isolation
- Mock external dependencies
- Fast execution (< 100ms per test)
- No I/O operations

### Integration Tests
- Test multiple components together
- Can use real dependencies
- Place in `tests/integration/` if created
- Mark with `@pytest.mark.integration`

## Assertions

### Prefer Specific Assertions
```python
# Good
assert result == expected
assert value in collection
assert obj.attribute is not None

# Avoid
assert result  # Too vague
```

### Testing Exceptions
```python
def test_feature_with_invalid_input_raises_error():
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError, match="specific error message"):
        feature.process(None)
```

## Test Data

### Keep Test Data Small
- Use minimal data that proves the point
- Prefer explicit values over random generation
- Document why specific test values were chosen

### Use Fixtures for Complex Setup
```python
@pytest.fixture
def populated_graph():
    """Provide a graph with 5 nodes for testing."""
    graph = nx.Graph()
    # ... setup
    return graph
```

## Mock Usage

### When to Mock
- External API calls
- File system operations
- Time-dependent operations
- Expensive computations

### How to Mock
```python
from unittest.mock import Mock, patch

def test_feature_with_mocked_dependency():
    """Test feature with mocked external call."""
    mock_api = Mock(return_value="mocked")
    
    with patch('module.api_call', mock_api):
        result = feature()
    
    assert result == "expected"
    mock_api.assert_called_once()
```

## Coverage

### Aim For
- 80%+ line coverage for business logic
- 100% coverage for critical paths
- All error paths tested

### Don't Obsess Over
- 100% coverage everywhere (diminishing returns)
- Testing trivial property getters/setters
- Testing framework code

## Performance Tests

### Mark Slow Tests
```python
@pytest.mark.slow
def test_large_network_simulation():
    """Test simulation with 1000 nodes (slow)."""
    # ... test code
```

Run fast tests only: `pytest -m "not slow"`

## Common Patterns in This Project

### Testing Transcript Extraction
```python
from unittest.mock import Mock, patch

def test_extract_transcript_returns_valid_data():
    """Test transcript extraction with valid video ID."""
    mock_transcript = Mock()
    mock_transcript.fetch.return_value = [
        Mock(text="Hello", start=0.0, duration=2.0)
    ]
    
    with patch('youtube_transcript_api.YouTubeTranscriptApi.list') as mock_api:
        mock_api.return_value = [mock_transcript]
        result = extract_transcript("test_video_id")
    
    assert len(result) > 0
    assert "text" in result[0]
    assert "start" in result[0]
```

### Testing Section Generation
```python
def test_create_sections_generates_valid_timestamps():
    """Test section generation produces valid output."""
    transcript = [
        {"text": "Introduction", "start": 0.0, "duration": 5.0},
        {"text": "Main content", "start": 5.0, "duration": 10.0}
    ]
    
    sections = create_section_timestamps(
        transcript, 
        section_count_range=(2, 5)
    )
    
    assert len(sections) >= 2
    assert all("title" in s and "start" in s for s in sections)
    assert sections[0]["start"] < sections[1]["start"]
```

### Testing API Services
```python
def test_gemini_service_handles_retry_on_failure():
    """Test GeminiService retries on transient failures."""
    service = GeminiService()
    
    with patch.object(service, '_call_api') as mock_call:
        mock_call.side_effect = [Exception("Timeout"), {"result": "success"}]
        result = service.generate_with_retry(prompt="test")
    
    assert mock_call.call_count == 2
    assert result == {"result": "success"}
```

### Testing Flask Routes
```python
import pytest
from flask import Flask

@pytest.fixture
def client():
    """Provide Flask test client."""
    from src.web_app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_generate_sections_endpoint_success(client):
    """Test /generate-sections endpoint with valid input."""
    response = client.post('/generate-sections', data={
        'video_id': 'dQw4w9WgXcQ',
        'min_sections': '5',
        'max_sections': '10'
    })
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'sections' in data

def test_generate_sections_missing_video_id_returns_error(client):
    """Test endpoint returns error when video_id is missing."""
    response = client.post('/generate-sections', data={})
    
    assert response.status_code == 400
    data = response.get_json()
    assert data['success'] is False
    assert 'error' in data
```

### Testing Data Validation
```python
def test_validate_sections_detects_invalid_timestamps():
    """Test validation catches out-of-range timestamps."""
    sections = [
        {"title": "Intro", "start": -5.0},  # Invalid: negative
        {"title": "End", "start": 999.0}     # Invalid: beyond transcript
    ]
    transcript = [{"text": "test", "start": 0.0, "duration": 10.0}]
    
    valid, errors = _validate_sections(sections, transcript, (1, 3))
    
    assert not valid
    assert any("out of" in err.lower() for err in errors)
```

## What NOT to Do

- ❌ Don't test implementation details (test behavior)
- ❌ Don't create test dependencies (tests should be independent)
- ❌ Don't use sleep() for timing (use mocking or time control)
- ❌ Don't test third-party libraries (trust them)
- ❌ Don't write tests that depend on execution order
- ❌ Don't use unittest.TestCase (use plain functions + fixtures)

## Test Naming Convention

```python
# Pattern: test_<unit>_<scenario>_<expected_result>

# Good names
def test_event_manager_with_empty_queue_returns_immediately():
def test_network_state_after_node_depletion_marks_node_depleted():
def test_packet_handler_with_full_buffer_drops_packet():

# Poor names
def test_1():  # Not descriptive
def test_stuff():  # Too vague
def test_network():  # Missing scenario/expectation
```

## Docstrings for Tests

```python
def test_feature():
    """Brief one-line description of what this test verifies.
    
    Optional: Additional context about why this test exists,
    edge cases it covers, or important setup details.
    """
```

Keep test docstrings brief - the test name should be self-documenting.

