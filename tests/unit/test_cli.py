"""Unit tests for Nina CLI commands."""

from nina.cli import run_doctor


def test_cli_doctor() -> None:
    """Verify nina doctor executes diagnostic checks and returns exit code 0."""
    code = run_doctor()
    assert code == 0
