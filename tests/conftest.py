"""Pytest configuration and shared fixtures for tests."""

import pytest
from pathlib import Path


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
