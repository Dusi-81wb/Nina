"""Comprehensive unit tests for emotion-preserving text preprocessor."""

import pytest

from nina.api.schemas import SpeechResult
from nina.core.exceptions import PreprocessingError
from nina.preprocessing.interface import TextPreprocessorConfig
from nina.preprocessing.processor import DefaultTextPreprocessor


def test_preprocess_basic_text() -> None:
    """Verify basic text processing preserves text and tokenizes accurately."""
    preprocessor = DefaultTextPreprocessor()
    res = preprocessor.preprocess("Hello world")

    assert res.raw_text == "Hello world"
    assert res.cleaned_text == "Hello world"
    assert res.tokens == ["Hello", "world"]
    assert res.intensifier_count == 0
    assert res.negation_count == 0


def test_preprocess_whitespace_normalization() -> None:
    """Verify leading/trailing and multi-space gaps are collapsed."""
    preprocessor = DefaultTextPreprocessor()
    res = preprocessor.preprocess("  hello    world  ")

    assert res.cleaned_text == "hello world"
    assert res.tokens == ["hello", "world"]


def test_preprocess_capitalization_preservation() -> None:
    """Verify uppercase emphasis is preserved when lowercase=False."""
    preprocessor = DefaultTextPreprocessor(config=TextPreprocessorConfig(lowercase=False))
    res = preprocessor.preprocess("I AM HAPPY")

    assert res.cleaned_text == "I AM HAPPY"
    assert res.punctuation_features["caps_words"] == 2  # "AM", "HAPPY"


def test_preprocess_punctuation_preservation() -> None:
    """Verify exclamation and question marks are preserved and counted."""
    preprocessor = DefaultTextPreprocessor()
    res = preprocessor.preprocess("I am happy!!!")

    assert res.cleaned_text == "I am happy!!!"
    assert res.punctuation_features["exclamations"] == 3


def test_preprocess_negation_preservation() -> None:
    """Verify negation terms are preserved and counted accurately."""
    preprocessor = DefaultTextPreprocessor()
    res = preprocessor.preprocess("I am not happy")

    assert "not" in res.cleaned_text
    assert res.negation_count == 1


def test_preprocess_contraction_expansion() -> None:
    """Verify contractions are expanded while preserving explicit negation."""
    preprocessor = DefaultTextPreprocessor()
    res = preprocessor.preprocess("I don't like this")

    assert res.cleaned_text == "I do not like this"
    assert res.negation_count == 1


def test_preprocess_repeated_characters() -> None:
    """Verify character lengthening is normalized (e.g. sooooo -> soo)."""
    preprocessor = DefaultTextPreprocessor()
    res = preprocessor.preprocess("I am sooooo happy")

    assert res.cleaned_text == "I am soo happy"
    assert res.intensifier_count == 1  # "soo" matched via lower token check


def test_preprocess_emojis() -> None:
    """Verify unicode emojis are preserved in cleaned text."""
    preprocessor = DefaultTextPreprocessor()
    res = preprocessor.preprocess("I love this ❤️")

    assert "❤️" in res.cleaned_text


def test_preprocess_mixed_punctuation() -> None:
    """Verify mixed punctuation (WHAT?!) counts both questions and exclamations."""
    preprocessor = DefaultTextPreprocessor()
    res = preprocessor.preprocess("WHAT?!")

    assert res.punctuation_features["exclamations"] == 1
    assert res.punctuation_features["questions"] == 1
    assert res.punctuation_features["caps_words"] == 1


def test_preprocess_empty_and_whitespace_input() -> None:
    """Verify empty string and whitespace-only text return clean zero-state without crashing."""
    preprocessor = DefaultTextPreprocessor()

    res_empty = preprocessor.preprocess("")
    assert res_empty.cleaned_text == ""
    assert res_empty.tokens == []
    assert res_empty.intensifier_count == 0

    res_spaces = preprocessor.preprocess("   ")
    assert res_spaces.cleaned_text == ""
    assert res_spaces.tokens == []


def test_preprocess_speech_result_input() -> None:
    """Verify SpeechResult data contract input is supported seamlessly."""
    preprocessor = DefaultTextPreprocessor()
    speech = SpeechResult(text="I am extremely terrified", language="en")
    res = preprocessor.preprocess(speech)

    assert res.raw_text == "I am extremely terrified"
    assert res.intensifier_count >= 2  # extremely, terrified


def test_preprocess_invalid_input_type() -> None:
    """Verify non-string/non-SpeechResult input raises PreprocessingError."""
    preprocessor = DefaultTextPreprocessor()
    with pytest.raises(PreprocessingError, match="Input must be str or SpeechResult"):
        preprocessor.preprocess(12345)  # type: ignore


def test_preprocess_invariants_determinism() -> None:
    """Verify preprocessing is strictly deterministic and preserves raw text."""
    preprocessor = DefaultTextPreprocessor()
    input_str = "I can't believe this!!!"

    res1 = preprocessor.preprocess(input_str)
    res2 = preprocessor.preprocess(input_str)

    assert res1.cleaned_text == res2.cleaned_text
    assert res1.raw_text == input_str
    assert res1.negation_count == 1
