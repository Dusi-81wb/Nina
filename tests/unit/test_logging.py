"""Unit tests for centralized logging initialization and timing decorator."""

import time

from nina.core.logging import log_execution_time, logger, setup_logging


def test_logging_setup() -> None:
    """Verify setup_logging configures loguru sinks without crashing."""
    setup_logging(log_level="DEBUG")
    logger.info("Test log message for unit test")


def test_log_execution_time() -> None:
    """Verify log_execution_time context manager runs and records timing."""
    with log_execution_time("UnitTestAction"):
        time.sleep(0.01)
