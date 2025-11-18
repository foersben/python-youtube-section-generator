"""Centralized logging configuration for the YouTube Transcript project.

This module provides consistent logging setup across all components with
proper formatting, levels, and output handling.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path


class ColorFormatter(logging.Formatter):
    """Colored console formatter for better readability."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        if record.levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            record.levelname = colored_levelname

        return super().format(record)


def setup_logging(
    level: str = "INFO", log_file: str | None = None, console: bool = True, colored: bool = True
) -> None:
    """Set up centralized logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging to file
        console: Whether to log to console
        colored: Whether to use colored console output
    """
    # Get configuration from environment
    level = os.getenv("LOG_LEVEL", level)
    log_to_file = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    log_file = os.getenv("LOG_FILE", log_file) if log_to_file else None

    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)

    # Common format
    base_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        if colored and sys.stdout.isatty():
            formatter: logging.Formatter = ColorFormatter(base_format, date_format)
        else:
            formatter = logging.Formatter(base_format, date_format)

        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(numeric_level)

        # File logs don't need colors
        file_formatter = logging.Formatter(base_format, date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Set specific loggers to reduce noise from dependencies
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("llama_cpp").setLevel(logging.WARNING)
    # deepl client is chatty at INFO; silence it to avoid log flooding when quota/errors occur
    logging.getLogger("deepl").setLevel(logging.WARNING)

    # But keep our app logs visible
    logging.getLogger("src").setLevel(numeric_level)
    logging.getLogger("web_app").setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Get a properly configured logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Initialize with default settings on import
setup_logging()
