"""Unit and integration tests for Phase 4B training, dataset verification, and model reloading."""

from pathlib import Path

from nina.api.schemas import SupportedEmotion
from nina.emotion.classical import ClassicalEmotionClassifier
from nina.emotion.trainer import verify_dataset_integrity
from nina.preprocessing.processor import DefaultTextPreprocessor


def test_verify_dataset_integrity_passes() -> None:
    """Verify verify_dataset_integrity loads valid DataFrames and passes zero leakage checks."""
    df_train, df_val, df_test = verify_dataset_integrity()

    assert len(df_train) == 69726
    assert len(df_val) == 8716
    assert len(df_test) == 8716

    canonical_set = {e.value for e in SupportedEmotion}
    assert set(df_train["emotion"].unique()).issubset(canonical_set)
    assert set(df_val["emotion"].unique()).issubset(canonical_set)
    assert set(df_test["emotion"].unique()).issubset(canonical_set)


def test_trained_classical_model_reload_and_predict() -> None:
    """Verify trained classical baseline joblib model loads from disk and returns consistent predictions."""
    model_path = Path("artifacts/models/classical_baseline.joblib")
    assert model_path.exists(), "Trained classical baseline joblib model must exist in artifacts/models/"

    classifier1 = ClassicalEmotionClassifier(model_path=model_path)
    assert classifier1._vectorizer is not None
    assert classifier1._classifier is not None

    preprocessor = DefaultTextPreprocessor()
    text = preprocessor.preprocess("I am so happy and ecstatic today!")

    res1 = classifier1.predict(text)
    assert res1.emotion == SupportedEmotion.HAPPY
    assert res1.confidence > 0.40

    # Model reload test
    classifier2 = ClassicalEmotionClassifier(model_path=model_path)
    res2 = classifier2.predict(text)

    assert res2.emotion == res1.emotion
    assert res2.confidence == res1.confidence
