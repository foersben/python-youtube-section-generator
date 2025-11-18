"""Pytest configuration and shared fixtures for tests."""

import logging
import os
from pathlib import Path

import pytest

# Prefer filelock for cross-platform locking; require it
try:
    from filelock import FileLock
except Exception as e:
    raise RuntimeError(
        "filelock is required for test serialization. Run `poetry install` to install project dependencies (filelock)."
    ) from e

logger = logging.getLogger(__name__)


def pytest_configure(config):
    """Register custom markers to avoid pytest warnings and document usage."""
    config.addinivalue_line(
        "markers",
        "llm: mark test as requiring local LLM resources; test will be serialized to avoid concurrent model loads",
    )


def _file_mentions_llm(path: Path) -> bool:
    """Quick heuristic: scan the test file for LLM-related tokens.

    This avoids requiring test authors to remember a marker; any test that
    references local LLM APIs will be detected and serialized.
    """
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return False
    tokens = ("LocalLLMClient", "LocalLLMProvider", "LOCAL_MODEL_PATH", "USE_LOCAL_LLM")
    for t in tokens:
        if t in text:
            return True
    return False


def _should_acquire_lock(item) -> bool:
    """Return True if this test should be serialized with the LLM lock.

    Rules (strict):
    - Explicit pytest marker `@pytest.mark.llm`
    - OR global opt-in: RUN_HEAVY_INTEGRATION is truthy
    """
    # 1) Explicit pytest marker
    try:
        if item.get_closest_marker("llm") is not None:
            return True
    except Exception:
        pass

    # 2) Heavy env var overrides
    heavy_env = os.getenv("RUN_HEAVY_INTEGRATION", "false").lower() in ("1", "true", "yes")
    if heavy_env:
        return True

    return False


def _lock_path() -> Path:
    model_env = os.getenv("LOCAL_MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
    try:
        lock_dir = Path(model_env).parent
        if not lock_dir.exists():
            lock_dir = Path.cwd() / "models"
    except Exception:
        lock_dir = Path.cwd() / "models"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / ".llm_load.lock"


def pytest_runtest_setup(item):
    if not _should_acquire_lock(item):
        return

    lockfile = _lock_path()
    item._llm_lockfile_path = lockfile

    # Always use FileLock (required)
    lock = FileLock(str(lockfile))
    logger.info(f"[pytest] acquiring LLM filelock for test {item.name}: {lockfile}")
    lock.acquire()
    item._llm_filelock = lock
    logger.info(f"[pytest] acquired LLM filelock for test {item.name}")


def pytest_runtest_teardown(item, nextitem):
    if not hasattr(item, "_llm_lockfile_path"):
        return

    filelock_obj = getattr(item, "_llm_filelock", None)

    try:
        # Release FileLock if present
        if filelock_obj is not None:
            try:
                filelock_obj.release()
                logger.info(f"[pytest] released LLM filelock for test {item.name}")
            except Exception:
                logger.exception(f"[pytest] failed to release LLM filelock for test {item.name}")
            finally:
                try:
                    delattr(item, "_llm_filelock")
                except Exception:
                    pass
            return
    finally:
        # Clean attributes
        try:
            delattr(item, "_llm_lockfile_path")
        except Exception:
            pass


@pytest.fixture
def sample_transcript():
    """Provide a sample transcript for testing."""
    return [
        {"start": 0.0, "text": "Hello and welcome to this tutorial.", "duration": 3.5},
        {"start": 3.5, "text": "Today we'll learn about Python.", "duration": 2.8},
        {"start": 6.3, "text": "First, let's cover the basics.", "duration": 2.5},
        {
            "start": 8.8,
            "text": "Python is a high-level programming language.",
            "duration": 3.2,
        },
        {"start": 12.0, "text": "It's known for its readability.", "duration": 2.5},
        {"start": 14.5, "text": "Now let's move to advanced topics.", "duration": 2.8},
        {"start": 17.3, "text": "We'll discuss object-oriented programming.", "duration": 3.5},
        {"start": 20.8, "text": "Classes and objects are fundamental.", "duration": 2.7},
        {"start": 23.5, "text": "Finally, let's talk about best practices.", "duration": 3.0},
        {"start": 26.5, "text": "Thanks for watching!", "duration": 1.5},
    ]


@pytest.fixture
def sample_sections():
    """Provide sample sections for testing."""
    return [
        {"title": "Introduction", "start": 0.0},
        {"title": "Python Basics", "start": 6.3},
        {"title": "Advanced Topics", "start": 14.5},
        {"title": "Conclusion", "start": 23.5},
    ]


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent
