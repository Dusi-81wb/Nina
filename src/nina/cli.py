"""CLI entrypoint for Nina diagnostic utilities, STT transcription, text preprocessing, and emotion classification."""

import argparse
import sys
import tempfile
from pathlib import Path

from nina.audio.recorder import MicrophoneRecorder
from nina.audio.source import FileAudioSource
from nina.core.config import get_settings
from nina.core.device import get_device_info
from nina.core.exceptions import NinaException
from nina.emotion.classical import ClassicalEmotionClassifier
from nina.emotion.evaluator import EmotionEvaluator
from nina.emotion.intensity import DefaultIntensityCalculator
from nina.emotion.stubs import StubEmotionClassifier
from nina.engine import NinaEmotionEngine
from nina.preprocessing.processor import DefaultTextPreprocessor
from nina.speech.engine import FasterWhisperSpeechToText
from nina.speech.stubs import StubSpeechToTextEngine


def run_doctor(args: argparse.Namespace | None = None) -> int:
    """Execute environment diagnostic check."""
    settings = get_settings()
    info = get_device_info(requested_device=settings.device)

    print("\n==================================================")
    print("             NINA SYSTEM DIAGNOSTICS              ")
    print("==================================================")
    print(f"Python Version:       {info.python_version}")
    print(f"OS Platform:          {info.os_info}")
    print(f"CPU Architecture:     {info.cpu_architecture}")
    print(f"PyTorch Version:      {info.pytorch_version}")
    print(f"CUDA Available:       {info.cuda_available}")
    print(f"Device Name:          {info.device_name}")
    print(f"Available VRAM:       {info.vram_gb:.2f} GB")
    print(f"Configured Device:    {settings.device}")
    print(f"Resolved Device:      {info.selected_device}")
    print(f"Default STT Model:    {settings.stt_model_size}")
    print(f"Default Emotion ML:   {settings.emotion_model_name}")
    print("==================================================\n")
    return 0


def run_transcribe(args: argparse.Namespace) -> int:
    """Execute speech-to-text transcription on an audio file or microphone input."""
    target_path_str = getattr(args, "audio_file", None) or getattr(args, "file_path", "")
    input_path = Path(target_path_str)

    print("\n==================================================")
    print("          NINA SPEECH-TO-TEXT ENGINE              ")
    print("==================================================\n")

    try:
        source = FileAudioSource(input_path)
        audio_info = source.get_audio_input()
        print(f"Loaded audio file: '{source.file_path.name}' ({audio_info.duration_seconds:.2f}s, {audio_info.sample_rate}Hz)")

        if getattr(args, "stub", False) or getattr(args, "engine", "") == "stub":
            engine = StubSpeechToTextEngine()
            print("Using StubSpeechToTextEngine (Dev Test)")
        else:
            engine = FasterWhisperSpeechToText(model_size=getattr(args, "model", "base.en"), device=getattr(args, "device", "auto"))
            print(f"Loaded FasterWhisper model '{args.model}' on device '{engine.resolved_device}'")

        print("Transcribing speech signal...")
        result = engine.transcribe(source)

        print("\n--- Transcription Result ---")
        print(f'Text:             "{result.text}"')
        print(f"Language:         {result.language}")
        print(f"ASR Confidence:   {result.confidence:.4f}")
        print(f"Audio Duration:   {audio_info.duration_seconds:.2f} s")
        print(f"Processing Time:  {result.processing_time_ms:.2f} ms")
        print(f"Real-Time Factor: {result.processing_time_ms / (audio_info.duration_seconds * 1000.0):.4f}x")
        print("--------------------------------------------------\n")
        return 0

    except NinaException as e:
        print(f"ERROR: {e.message}\n")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Transcription failed: {e!s}\n")
        return 1


def run_record(args: argparse.Namespace) -> int:
    """Record microphone audio to file or memory buffer."""
    output_path = Path(args.output) if getattr(args, "output", None) else None
    duration = getattr(args, "duration", 3.0)

    print("\n==================================================")
    print("         NINA MICROPHONE AUDIO RECORDER           ")
    print("==================================================\n")

    try:
        recorder = MicrophoneRecorder(sample_rate=getattr(args, "sample_rate", 16000))

        if output_path:
            print(f"Recording {duration}s microphone audio to '{output_path}'...")
            out_file = recorder.record_to_wav(output_path=output_path, duration_seconds=duration)
            print(f"Recording complete. Output saved to '{out_file.resolve()}'.")
        else:
            print(f"Recording {duration}s microphone audio into memory buffer...")
            audio_input = recorder.record(duration_seconds=duration)
            print(f"Recorded {audio_input.duration_seconds:.2f}s audio (RMS energy: {audio_input.signal_rms:.6f}).")

        return 0

    except NinaException as e:
        print(f"ERROR: {e.message}\n")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Microphone recording failed: {e!s}\n")
        return 1


def run_preprocess(args: argparse.Namespace) -> int:
    """Execute text preprocessing on a target string."""
    raw_text = args.text

    print("\n==================================================")
    print("         NINA TEXT PREPROCESSING DEBUGGER         ")
    print("==================================================\n")

    try:
        preprocessor = DefaultTextPreprocessor()
        result = preprocessor.preprocess(raw_text)

        print(f'Original Text:      "{result.raw_text}"')
        print(f'Cleaned Text:       "{result.cleaned_text}"')
        print(f"Tokens:             {result.tokens}")
        print(f"Intensifiers:       {result.intensifier_count}")
        print(f"Negations:          {result.negation_count}")
        print(f"Exclamations (!):   {result.punctuation_features.get('exclamations', 0)}")
        print(f"Questions (?):      {result.punctuation_features.get('questions', 0)}")
        print(f"Caps Words:         {result.punctuation_features.get('caps_words', 0)}")
        print(f"Processing Time:    {result.processing_time_ms:.3f} ms")
        print("--------------------------------------------------\n")
        return 0

    except NinaException as e:
        print(f"ERROR: {e.message}\n")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Preprocessing failed: {e!s}\n")
        return 1


def run_classify(args: argparse.Namespace) -> int:
    """Execute emotion classification and intensity estimation on a target string."""
    raw_text = args.text

    print("\n==================================================")
    print("     NINA EMOTION & INTENSITY CLASSIFICATION      ")
    print("==================================================\n")

    try:
        preprocessor = DefaultTextPreprocessor()
        preprocessed = preprocessor.preprocess(raw_text)

        engine_opt = getattr(args, "engine", "classical")
        stub_opt = getattr(args, "stub", False)

        # Select Classifier Engine
        if engine_opt == "classical":
            classifier = ClassicalEmotionClassifier()
            model_info_str = "Classical TF-IDF Lexicon Baseline"
        elif engine_opt == "stub" or stub_opt:
            classifier = StubEmotionClassifier()
            model_info_str = "StubEmotionClassifier (Dev Test)"
        else:
            try:
                from nina.emotion.transformer import TransformerEmotionClassifier
                classifier = TransformerEmotionClassifier(model_name=getattr(args, "model", None))
                model_info_str = f"Transformer ({args.model})"
            except Exception as e:  # noqa: BLE001
                print(f"Failed to load Transformer classifier: {e!s}. Falling back to Classical Baseline.")
                classifier = ClassicalEmotionClassifier()
                model_info_str = "Classical TF-IDF Lexicon Baseline (Fallback)"

        pred_res = classifier.predict(preprocessed)

        # Calculate Intensity
        intensity_calc = DefaultIntensityCalculator()
        intensity_res = intensity_calc.calculate_composite_intensity(pred_res, preprocessed)

        print(f'Text:               "{raw_text}"')
        print(f"Predicted Emotion:  {pred_res.emotion.value}")
        print(f"Model Confidence:   {pred_res.confidence:.4f}")
        print(f"Emotion Intensity:  {intensity_res.intensity:.1f} / 100 ({intensity_res.level.value.upper()})")

        print("\nClass Probabilities:")
        for emo, prob in pred_res.probabilities.items():
            print(f"  {emo.value:<12} {prob:.4f}")

        print("\nIntensity Sub-Score Breakdown:")
        print(f"  Text Score:        {intensity_res.components.text_score:.4f}")
        print(f"  Entropy:           {intensity_res.components.entropy:.4f}")
        print(f"  Margin Spread:     {intensity_res.components.margin:.4f}")
        print(f"  Intensifiers:      {intensity_res.components.intensifier_count}")
        print(f"  Punctuation:       {intensity_res.components.punctuation_score:.4f}")

        print(f"\nInference Duration: {pred_res.processing_time_ms:.2f} ms")
        print(f"Model Used:         {model_info_str}")
        print("--------------------------------------------------\n")
        return 0

    except NinaException as e:
        print(f"ERROR: {e.message}\n")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Classification failed: {e!s}\n")
        return 1


def run_evaluate(args: argparse.Namespace) -> int:
    """Evaluate emotion classifier on benchmark test set."""
    print("\n==================================================")
    print("         NINA MODEL EVALUATION SUITE              ")
    print("==================================================\n")

    try:
        evaluator = EmotionEvaluator()
        engine_opt = getattr(args, "engine", "classical")

        if engine_opt == "classical":
            classifier = ClassicalEmotionClassifier()
        else:
            try:
                from nina.emotion.transformer import TransformerEmotionClassifier
                classifier = TransformerEmotionClassifier()
            except Exception:  # noqa: BLE001
                classifier = ClassicalEmotionClassifier()

        print("Evaluating classifier engine against test dataset...")
        results = evaluator.evaluate(classifier)

        print(f"Evaluation Test Samples: {results.total_samples}")
        print(f"Overall Accuracy:        {results.accuracy * 100:.2f}%")
        print(f"Macro F1-Score:          {results.macro_f1:.4f}")
        print(f"Weighted F1-Score:       {results.weighted_f1:.4f}")
        print(f"Average Latency:         {results.average_latency_ms:.2f} ms / text")

        print("\nPer-Class Metrics:")
        for emo, m in results.per_class_metrics.items():
            print(f"  {emo:<10} Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1']:.4f} (Support: {m['support']})")

        print("--------------------------------------------------\n")
        return 0

    except NinaException as e:
        print(f"ERROR: {e.message}\n")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Evaluation failed: {e!s}\n")
        return 1


def run_listen(args: argparse.Namespace) -> int:
    """Diagnostic audio-to-emotion detection runner returning EmotionResult payload."""
    audio_file = getattr(args, "audio_file", None)
    duration = float(getattr(args, "duration", 4.0))
    interactive = getattr(args, "interactive", False)

    print("\n==================================================")
    print("      NINA VOICE EMOTION DETECTION RUNNER        ")
    print("==================================================\n")

    try:
        use_stub_stt = getattr(args, "stub", False)
        use_classical = getattr(args, "classical", False)

        stt = StubSpeechToTextEngine() if use_stub_stt else None
        emo = ClassicalEmotionClassifier() if use_classical else None

        print("Initializing Nina Emotion Engine...")
        engine = NinaEmotionEngine(stt_engine=stt, emotion_classifier=emo)
        init_ms = engine.preload_models()
        print(f"Model Preloading Complete: {init_ms:.2f} ms (Models now resident in memory)")

        if interactive:
            print("\n--------------------------------------------------")
            print("  INTERACTIVE SESSION (Models Reused Across Turns)")
            print("--------------------------------------------------")
            turn_idx = 1
            while True:
                user_in = input(f"\n[Turn {turn_idx}] Press ENTER to record {duration:.1f}s audio (or 'q' to quit): ").strip()
                if user_in.lower() == "q":
                    print("Exiting interactive session.")
                    break

                temp_wav = Path(tempfile.gettempdir()) / f"nina_live_turn_{turn_idx}.wav"
                print(f"Recording {duration:.1f}s LIVE microphone audio...")
                print(">>> SPEAK NOW into your microphone...")
                recorder = MicrophoneRecorder()
                recorded_path = recorder.record_to_wav(temp_wav, duration_seconds=duration)
                print("Processing speech & detecting emotion...")

                res = engine.process_file(recorded_path)

                print("\n--- EmotionResult Payload ---")
                print(f'Transcribed Text:   "{res.text}"')
                print(f"Detected Emotion:   {res.emotion.value.upper()}")
                print(f"Model Confidence:   {res.confidence:.4f}")
                if res.intensity is not None:
                    level_str = res.intensity_level.value.upper() if res.intensity_level else "N/A"
                    print(f"Emotion Intensity:  {res.intensity:.1f} / 100 ({level_str})")

                print("\nGranular Timing Breakdown:")
                print(f"  Model Init (One-time): {res.metadata.get('model_init_ms', 0.0):.2f} ms")
                print(f"  Audio Recording:       {res.metadata.get('recording_time_ms', 0.0):.2f} ms")
                print(f"  ASR Inference:         {res.metadata.get('asr_inference_ms', 0.0):.2f} ms")
                print(f"  Emotion Inference:     {res.metadata.get('emotion_inference_ms', 0.0):.2f} ms")
                print(f"  Total Turn Duration:   {res.metadata.get('total_turn_ms', 0.0):.2f} ms")
                print(f"  Models Reused:         {res.metadata.get('models_reused')}")
                print(f"  Actual Model Device:   {res.metadata.get('actual_model_device')}")
                turn_idx += 1
            return 0

        if audio_file:
            path = Path(audio_file)
            print(f"Processing target audio file: '{path.name}'...")
            res = engine.process_file(path)
        else:
            temp_wav = Path(tempfile.gettempdir()) / "nina_live_recording.wav"
            print(f"Recording {duration:.1f}s LIVE microphone audio...")
            print(">>> SPEAK NOW into your microphone...")
            recorder = MicrophoneRecorder()
            recorded_path = recorder.record_to_wav(temp_wav, duration_seconds=duration)
            print("Recording finished. Transcribing spoken speech & detecting emotion...")
            res = engine.process_file(recorded_path)

        print("\n==================================================")
        print("         STRUCTURED EmotionResult PAYLOAD         ")
        print("==================================================")
        print(f'Transcribed Text:   "{res.text}"')
        print(f"Detected Emotion:   {res.emotion.value.upper()}")
        print(f"Model Confidence:   {res.confidence:.4f}")

        if res.intensity is not None:
            level_str = res.intensity_level.value.upper() if res.intensity_level else "N/A"
            print(f"Emotion Intensity:  {res.intensity:.1f} / 100 ({level_str})")

        print("\n6-Class Probabilities:")
        for e, p in res.probabilities.items():
            print(f"  {e.value:<12} {p:.4f}")

        print("\nGranular Performance & Timing Breakdown:")
        print(f"  Model Initialization:  {init_ms:.2f} ms")
        print(f"  Audio Recording Time:  {res.metadata.get('recording_time_ms', 0.0):.2f} ms")
        print(f"  ASR Inference Latency: {res.metadata.get('asr_inference_ms', 0.0):.2f} ms")
        print(f"  Emotion Model Latency: {res.metadata.get('emotion_inference_ms', 0.0):.2f} ms")
        print(f"  Total Processing Time: {res.metadata.get('total_turn_ms', 0.0):.2f} ms")
        print(f"  Models Reused:         {res.metadata.get('models_reused')}")
        print(f"  Actual Model Device:   {res.metadata.get('actual_model_device')}")
        print("==================================================\n")
        return 0

    except NinaException as e:
        print(f"ERROR: {e.message}\n")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Audio emotion processing failed: {e!s}\n")
        return 1


def main() -> int:
    """CLI entrypoint router."""
    parser = argparse.ArgumentParser(
        prog="nina",
        description="Nina — Modular Voice-to-Text Emotion Detection Component",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # nina doctor
    subparsers.add_parser("doctor", help="Run system diagnostics and environment check")

    # nina transcribe <audio_file>
    stt_parser = subparsers.add_parser("transcribe", help="Transcribe audio file using speech-to-text engine")
    stt_parser.add_argument("audio_file", type=str, help="Path to input audio file")
    stt_parser.add_argument("--model", type=str, default="base.en", help="STT model size (tiny.en, base.en, small.en)")
    stt_parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    stt_parser.add_argument("--stub", action="store_true", help="Use stub ASR engine for testing")

    # nina record
    rec_parser = subparsers.add_parser("record", help="Record audio from microphone")
    rec_parser.add_argument("--duration", type=float, default=3.0, help="Recording duration in seconds")
    rec_parser.add_argument("--output", type=str, default=None, help="Optional WAV output filepath")
    rec_parser.add_argument("--sample-rate", type=int, default=16000, help="Recording sample rate in Hz")

    # nina preprocess <text>
    prep_parser = subparsers.add_parser("preprocess", help="Preprocess text transcript for NLP pipeline")
    prep_parser.add_argument("text", type=str, help="Target raw text to preprocess")

    # nina classify <text>
    cls_parser = subparsers.add_parser("classify", help="Predict emotion and intensity for a given text")
    cls_parser.add_argument("text", type=str, help="Target text to classify")
    cls_parser.add_argument("--engine", type=str, choices=["classical", "transformer", "stub"], default="classical", help="Classification engine")
    cls_parser.add_argument("--model", type=str, default=None, help="Transformer model checkpoint name")
    cls_parser.add_argument("--stub", action="store_true", help="Use stub emotion classifier for testing")

    # nina evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run benchmark evaluation suite")
    eval_parser.add_argument("--engine", type=str, choices=["classical", "transformer"], default="classical", help="Target engine")

    # nina listen
    lst_parser = subparsers.add_parser("listen", help="Record live microphone speech, transcribe, and detect emotion")
    lst_parser.add_argument("--audio-file", type=str, default=None, help="Path to input audio WAV file")
    lst_parser.add_argument("--duration", type=float, default=4.0, help="Live recording duration in seconds")
    lst_parser.add_argument("--interactive", action="store_true", help="Run persistent multi-turn interactive session without reloading models")
    lst_parser.add_argument("--stub", action="store_true", help="Use stub ASR mock for dev testing")
    lst_parser.add_argument("--classical", action="store_true", help="Use classical baseline emotion classifier")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "doctor":
        return run_doctor(args)
    elif args.command == "transcribe":
        return run_transcribe(args)
    elif args.command == "record":
        return run_record(args)
    elif args.command == "preprocess":
        return run_preprocess(args)
    elif args.command == "classify":
        return run_classify(args)
    elif args.command == "evaluate":
        return run_evaluate(args)
    elif args.command == "listen":
        return run_listen(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
