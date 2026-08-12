"""Centralized structured logging configuration for Nina using loguru."""

import sys
import time
from collections.abc import Generator
from contextlib import contextmanager

from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """Configure centralized loguru logger sinks and formatting.

    Args:
        log_level: Logging severity threshold (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path to record log outputs.
    """
    logger.remove()  # Remove default sink handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Console stdout sink
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level.upper(),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Optional file sink
    if log_file:
        logger.add(
            log_file,
            format=log_format,
            level=log_level.upper(),
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

    logger.debug(f"Logging initialized at level: {log_level.upper()}")


@contextmanager
def log_execution_time(action_name: str) -> Generator[None, None, None]:
    """Context manager for profiling and logging execution duration of pipeline steps.

    Args:
        action_name: Human-readable identifier for the timed action block.
    """
    start_time = time.perf_counter()
    logger.debug(f"Starting action: '{action_name}'")
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"Action '{action_name}' completed in {elapsed_ms:.2f} ms")


__all__ = ["log_execution_time", "logger", "setup_logging"]
