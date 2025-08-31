"""Centralized logging configuration for Abzu."""

import logging
import sys
from pathlib import Path

from serf.config import config

def setup_logging() -> None:
    """Configure logging for the application."""
    # Get log directory from config
    log_dir = Path(config.get("logs.file.path", "logs"))

    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler(sys.stdout),  # Also log to console
        ],
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Parameters
    ----------
    name : str, optional
        Name for the logger. If None, returns the root logger.

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
