# Create New Service Module

## Task
Create a new service module for external API integration (e.g., AI services, translation APIs, etc.)

## Requirements

### 1. File Location
- Place in: `src/core/`
- Name: `<service>_client.py` (e.g., `openai_client.py`, `azure_client.py`)

### 2. Basic Structure

```python
"""<Service Name> API client.

This module provides integration with the <Service Name> API for <purpose>.
"""

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_API_ENV = 'API_KEY_ENV_PLACEHOLDER'
DEFAULT_ENDPOINT = '<API_ENDPOINT_URL>'


class <Service>Service:
    """Service for interacting with <Service Name> API.

    Attributes:
        client: The API client instance
        api_key: API key for authentication
    """

    def __init__(self, api_key_env: str = DEFAULT_API_ENV):
        """Initialize the service.

        Args:
            api_key_env: Environment variable name for API key

        Raises:
            ValueError: If API key is not found in environment
        """
        api_key = os.getenv(api_key_env)
        if not api_key:
            logger.error(f"Environment variable {api_key_env} not set")
            raise ValueError(f"{api_key_env} not found in environment variables")

        self.api_key = api_key
        self.client = self._initialize_client()
        logger.info(f"Initialized {self.__class__.__name__}")

    def _initialize_client(self) -> Any:
        """Initialize the API client."""
        # Implementation
        pass

    def process(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Process data using the API.

        Args:
            data: Input data to process
            **kwargs: Additional parameters for API call

        Returns:
            Processed result from API

        Raises:
            Exception: If API call fails
        """
        try:
            logger.info("Processing request...")
            result = self._call_api(data, **kwargs)
            logger.info("Request processed successfully")
            return result
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise


# Helper function for simple use cases
def process_with_<service>(data: dict[str, Any]) -> dict[str, Any]:
    """Convenience function for <service> processing.

    Args:
        data: Input data

    Returns:
        Processed result
    """
    service = <Service>Service()
    return service.process(data)
```

### 3. With Retry Logic

```python
import time
from typing import Any, Callable


def with_retry(
    func: Callable,
    max_attempts: int = 3,
    backoff_factor: int = 2
) -> Any:
    """Execute function with exponential backoff retry.

    Args:
        func: Function to execute
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff

    Returns:
        Result from successful function call

    Raises:
        Exception: If all retry attempts fail
    """
    for attempt in range(max_attempts):
        try:
            result = func()
            return result
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.error(f"All {max_attempts} attempts failed")
                raise

            wait_time = backoff_factor ** attempt
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)


class <Service>Service:
    # ... other methods ...

    def process_with_retry(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process with automatic retry on failure."""
        return with_retry(lambda: self.process(data))
```

### 4. Checklist
- [ ] Added module-level docstring
- [ ] API key retrieved from environment variable
- [ ] Proper error handling with logging
- [ ] Type hints for all functions
- [ ] Comprehensive docstrings
- [ ] Retry logic for transient failures
- [ ] Helper function for simple use cases
- [ ] Unit tests created

### 5. Testing

Create tests in `tests/test_<service>_client.py`:

```python
"""Tests for <service> client."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.<service>_client import <Service>Service


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for the service client."""
    # Template placeholder — not a real credential.
    # detect-secrets: allowlist
    monkeypatch.setenv("API_ENV_PLACEHOLDER", "placeholder")


def test_init_success(api_key_placeholder):
    """Test successful initialization."""
    service = <Service>Service()

    # This assertion intentionally checks the placeholder.
    # detect-secrets: allowlist
    assert service.api_key == api_key_placeholder


def test_init_no_api_key():
    """Test initialization when the API key is missing."""
    with pytest.raises(ValueError, match="not found"):
        <Service>Service()


@patch("src.core.<service>_client.SomeAPIClient")
def test_process_success(mock_client, mock_env):
    """Test successful API call."""
    mock_client.return_value.call.return_value = {"result": "success"}

    service = <Service>Service()
    result = service.process({"data": "test"})

    assert result == {"result": "success"}


def test_process_with_retry(mock_env):
    """Test retry logic."""
    service = <Service>Service()

    # Mock to fail twice then succeed
    service.process = Mock(side_effect=[Exception("Error"), Exception("Error"), {'result': 'success'}])

    result = service.process_with_retry({'data': 'test'})
    assert result == {'result': 'success'}
    assert service.process.call_count == 3
```

## Common Patterns

### HTTP Requests with httpx
```python
import httpx

response = httpx.post(
    url,
    json=data,
    headers={'Authorization': f'Bearer {self.api_key}'},
    timeout=30
)
response.raise_for_status()
return response.json()
```

### Rate Limiting
```python
import time
from datetime import datetime, timedelta

class RateLimitedService:
    def __init__(self):
        self.last_call = None
        self.min_interval = 1.0  # seconds

    def _wait_for_rate_limit(self):
        """Enforce rate limiting."""
        if self.last_call:
            elapsed = (datetime.now() - self.last_call).total_seconds()
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self.last_call = datetime.now()
```

### Response Validation
```python
def _validate_response(self, response: dict[str, Any]) -> bool:
    """Validate API response structure."""
    required_keys = ['status', 'data']
    return all(key in response for key in required_keys)
```
