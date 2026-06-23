"""Centralized logging configuration using Loguru."""

import sys
from pathlib import Path
from loguru import logger
from backend.config import get_settings


def setup_logging():
    """Configure application logging with Loguru."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        serialize=False,
    )

    logger.info(f"Logging initialized | Level: {settings.log_level}")
    return logger


app_logger = setup_logging()
