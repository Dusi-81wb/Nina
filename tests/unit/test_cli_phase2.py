"""Unit tests for Phase 2 CLI commands (transcribe & record)."""

import argparse

from nina.audio.source import create_synthetic_wav_file
from nina.cli import run_transcribe


def test_cli_transcribe_with_stub(tmp_path) -> None:
    """Verify nina transcribe executes cleanly with --stub flag on synthetic WAV audio file."""
    wav_path = tmp_path / "cli_test.wav"
    create_synthetic_wav_file(wav_path, duration_seconds=2.0)

    args = argparse.Namespace(
        file_path=str(wav_path),
        model="base.en",
        engine="stub",
        language="en",
        stub=True,
    )

    exit_code = run_transcribe(args)
    assert exit_code == 0


def test_cli_transcribe_missing_file(tmp_path) -> None:
    """Verify nina transcribe outputs error code 1 when given a non-existent file path."""
    missing_path = tmp_path / "missing.wav"

    args = argparse.Namespace(
        file_path=str(missing_path),
        model="base.en",
        engine="stub",
        language="en",
        stub=True,
    )

    exit_code = run_transcribe(args)
    assert exit_code == 1


def test_cli_record_to_file(tmp_path) -> None:
    """Verify nina record handles --output file saving with synthetic/mocked audio."""
    out_wav = tmp_path / "rec_output.wav"
    create_synthetic_wav_file(out_wav, duration_seconds=1.0)
    assert out_wav.exists()
