"""Standard emotion-preserving text preprocessor implementation."""

import re
import time
import unicodedata
from typing import ClassVar

from nina.api.schemas import PreprocessedText, SpeechResult
from nina.core.exceptions import PreprocessingError
from nina.preprocessing.interface import TextPreprocessor, TextPreprocessorConfig


class DefaultTextPreprocessor(TextPreprocessor):
    """Emotion-preserving text preprocessor implementation."""

    DEFAULT_INTENSIFIERS: ClassVar[set[str]] = {
        "very",
        "extremely",
        "deeply",
        "wildly",
        "hugely",
        "terrified",
        "furious",
        "ecstatic",
        "screaming",
        "really",
        "super",
        "so",
        "soo",
        "completely",
        "totally",
        "absolutely",
    }

    DEFAULT_NEGATIONS: ClassVar[set[str]] = {
        "not",
        "no",
        "never",
        "cannot",
        "can't",
        "cant",
        "don't",
        "dont",
        "doesn't",
        "doesnt",
        "didn't",
        "didnt",
        "won't",
        "wont",
        "isn't",
        "isnt",
        "aren't",
        "arent",
        "wasn't",
        "wasnt",
        "weren't",
        "werent",
        "haven't",
        "havent",
        "hasn't",
        "hasnt",
        "hadn't",
        "hadnt",
        "wouldn't",
        "wouldnt",
        "couldn't",
        "couldnt",
        "shouldn't",
        "shouldnt",
    }

    CONTRACTION_MAP: ClassVar[dict[str, str]] = {
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bisn't\b": "is not",
        r"\baren't\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bhaven't\b": "have not",
        r"\bhasn't\b": "has not",
        r"\bhadn't\b": "had not",
        r"\bwouldn't\b": "would not",
        r"\bcouldn't\b": "could not",
        r"\bshouldn't\b": "should not",
        r"\bI'm\b": "I am",
        r"\bi'm\b": "i am",
        r"\byou're\b": "you are",
        r"\bit's\b": "it is",
        r"\bthey're\b": "they are",
        r"\bwe're\b": "we are",
    }

    def __init__(
        self,
        config: TextPreprocessorConfig | None = None,
        intensifiers: set[str] | None = None,
        negations: set[str] | None = None,
    ) -> None:
        self.config = config or TextPreprocessorConfig()
        self.intensifiers = intensifiers or self.DEFAULT_INTENSIFIERS
        self.negations = negations or self.DEFAULT_NEGATIONS

    def preprocess(self, input_text: str | SpeechResult) -> PreprocessedText:
        """Sanitize raw speech transcript, preserve sentiment modifiers, and tokenize.

        Args:
            input_text: Raw text string or SpeechResult data model.

        Returns:
            PreprocessedText: Standardized preprocessed container.
        """
        start_time = time.perf_counter()

        if isinstance(input_text, SpeechResult):
            raw_text = input_text.text
        elif isinstance(input_text, str):
            raw_text = input_text
        else:
            raise PreprocessingError(
                f"Input must be str or SpeechResult, got {type(input_text).__name__}"
            )

        if not raw_text or not raw_text.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PreprocessedText(
                raw_text=raw_text,
                cleaned_text="",
                tokens=[],
                intensifier_count=0,
                negation_count=0,
                punctuation_features={"exclamations": 0, "questions": 0, "caps_words": 0},
                processing_time_ms=round(elapsed_ms, 3),
                metadata={"config": self.config.model_dump()},
            )

        text = raw_text

        # 1. Unicode Normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 2. Whitespace Normalization
        if self.config.normalize_whitespace:
            text = re.sub(r"\s+", " ", text.strip())

        # 3. Contraction Expansion (preserving negation)
        if self.config.expand_contractions:
            for pattern, replacement in self.CONTRACTION_MAP.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 4. Repeated Character Normalization on alphabetic letters (e.g., sooooo -> soo)
        if self.config.handle_repeated_chars:
            text = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)

        # 5. Case Normalization (optional)
        if self.config.lowercase:
            text = text.lower()

        # Tokenization & Feature Extraction
        tokens = [t for t in re.findall(r"\w+|[^\w\s]", text) if t.strip()]
        lower_tokens = [t.lower() for t in tokens]

        # Count features
        intensifier_count = sum(1 for t in lower_tokens if t in self.intensifiers)
        negation_count = sum(1 for t in lower_tokens if t in self.negations)

        exclamations = text.count("!")
        questions = text.count("?")
        caps_words = sum(1 for t in tokens if len(t) >= 2 and t.isupper() and t.isalpha())

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return PreprocessedText(
            raw_text=raw_text,
            cleaned_text=text,
            tokens=tokens,
            intensifier_count=intensifier_count,
            negation_count=negation_count,
            punctuation_features={
                "exclamations": exclamations,
                "questions": questions,
                "caps_words": caps_words,
            },
            processing_time_ms=round(elapsed_ms, 3),
            metadata={
                "config": self.config.model_dump(),
                "token_count": len(tokens),
            },
        )
