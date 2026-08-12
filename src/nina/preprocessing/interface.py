"""Abstract interfaces and configuration policies for Text Preprocessing."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from nina.api.schemas import PreprocessedText, SpeechResult

# Alias for backward compatibility with Phase 1 contracts
CleanedText = PreprocessedText


class TextPreprocessorConfig(BaseModel):
    """Configurable policy settings for emotion-preserving text preprocessing."""

    normalize_unicode: bool = Field(default=True, description="Apply NFKC unicode normalization")
    normalize_whitespace: bool = Field(default=True, description="Collapse multi-space gaps")
    lowercase: bool = Field(default=False, description="Convert text to lowercase (False preserves emphasis caps)")
    preserve_punctuation: bool = Field(default=True, description="Preserve !, ?, and repeated marks")
    preserve_emojis: bool = Field(default=True, description="Preserve unicode emojis")
    expand_contractions: bool = Field(default=True, description="Expand contractions (e.g. don't -> do not)")
    handle_repeated_chars: bool = Field(default=True, description="Normalize character lengthening (e.g. happyyyy -> happyy)")
    remove_stopwords: bool = Field(default=False, description="Remove stop words (False preserves emotional modifiers)")
    apply_stemming: bool = Field(default=False, description="Apply word stemming (False preserves word inflections)")


class TextPreprocessor(ABC):
    """Abstract interface defining the contract for text cleaning components."""

    @abstractmethod
    def preprocess(self, input_text: str | SpeechResult) -> PreprocessedText:
        """Sanitize raw speech transcript, preserve sentiment modifiers, and tokenize.

        Args:
            input_text: Raw string or SpeechResult object from Speech-to-Text layer.

        Returns:
            PreprocessedText: Data contract containing normalized text and modifier metadata.

        Raises:
            PreprocessingError: If string normalization or tokenization fails.
        """
