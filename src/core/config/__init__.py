"""Configuration package.

This package provides centralized configuration management.
Implementation is in config.py - this __init__.py only handles imports/exports.

Best Practice: Keep __init__.py minimal - just imports and singleton instance.
"""

from src.core.config.config import AppConfig

# Singleton instance - use this throughout the application
config = AppConfig()

__all__ = ["AppConfig", "config"]

