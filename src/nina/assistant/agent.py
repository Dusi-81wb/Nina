"""Main Assistant Loop tying together all subsystems."""
from pathlib import Path

from nina.audio.recorder import MicrophoneRecorder
from nina.core.config import NinaSettings, get_settings
from nina.core.logging import logger
from nina.engine import NinaEmotionEngine
from nina.llm.ollama import OllamaProvider
from nina.llm.provider import LLMProvider
from nina.memory.context import ConversationMemory
from nina.tools.builtin import get_current_time, get_weather
from nina.tools.manager import ToolManager
from nina.tts.engine import LocalTTSProvider, TTSProvider
from nina.wakeword.detector import WakeWordDetector


class NinaAssistant:
    """The central agent integrating all voice assistant components."""

    def __init__(
        self,
        voice_engine: NinaEmotionEngine | None = None,
        llm: LLMProvider | None = None,
        tts: TTSProvider | None = None,
        memory: ConversationMemory | None = None,
        tools: ToolManager | None = None,
        wake_word: WakeWordDetector | None = None,
        settings: NinaSettings | None = None
    ):
        self.settings = settings or get_settings()
        self.voice_engine = voice_engine or NinaEmotionEngine(auto_preload=True)

        # We will attempt to get ollama settings from settings, but provide defaults
        ollama_url = getattr(self.settings, "ollama_url", "http://localhost:11434")
        ollama_model = getattr(self.settings, "ollama_model", "llama3")

        self.llm = llm or OllamaProvider(base_url=ollama_url, model_name=ollama_model)
        self.tts = tts or LocalTTSProvider()
        self.memory = memory or ConversationMemory()

        if tools is None:
            self.tools = ToolManager()
            self.tools.register_tool("get_current_time", "Get the current time and date", get_current_time)
            self.tools.register_tool("get_weather", "Get the weather for a given location", get_weather)
        else:
            self.tools = tools

        self.wake_word = wake_word or WakeWordDetector()

    def process_turn(self, audio_file: Path) -> None:
        """Process a single conversation turn from an audio file."""
        logger.info(f"Processing turn from {audio_file}")

        # 1. Transcribe & Classify Emotion
        try:
            result = self.voice_engine.process_file(audio_file, include_intensity=False)
            text = result.text.strip()
            emotion = result.emotion.value
            logger.info(f"User (Emotion: {emotion}): {text}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to process audio: {e}")
            return

        if not text:
            logger.info("No speech detected.")
            return

        # 2. Add to Memory
        self.memory.add_message("user", text, metadata={"emotion": emotion})

        # 3. LLM Reasoning
        logger.info("Generating response via LLM...")
        try:
            context = self.memory.get_context_window()
            # Optional: System prompt that tells the LLM about the user's emotion
            # Not strictly required but useful context
            system_msg = {"role": "system", "content": f"The user is feeling {emotion}. Be helpful and concise."}
            messages = [system_msg] + context

            response = self.llm.generate_response(messages, tools=self.tools.get_tools_schema())
        except Exception as e:  # noqa: BLE001
            logger.error(f"LLM Generation failed: {e}")
            self.tts.speak("I'm sorry, I couldn't process that right now.")
            return

        # 4. Handle Tools
        tool_calls = response.get("tool_calls", [])
        if tool_calls:
            for call in tool_calls:
                func_name = call["function"]["name"]
                args = call["function"].get("arguments", {})
                logger.info(f"Executing tool: {func_name} with {args}")
                tool_result = self.tools.execute_tool(func_name, args)

                # Add tool result to memory and get another response
                self.memory.add_message("assistant", "", metadata={"tool_call": func_name})
                self.memory.add_message("tool", str(tool_result), metadata={"name": func_name})

                # Re-prompt LLM with tool result
                try:
                    messages = [system_msg] + self.memory.get_context_window()
                    response = self.llm.generate_response(messages)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"LLM Generation after tool failed: {e}")
                    self.tts.speak("I had trouble understanding the tool results.")
                    return

        # 5. TTS Output
        response_text = response.get("content", "")
        if response_text:
            logger.info(f"Assistant: {response_text}")
            self.memory.add_message("assistant", response_text)
            self.tts.speak(response_text)

    def start_listening(self) -> None:
        """Start a continuous listening loop (Simulated via prompt for now to avoid hardware lock)."""
        logger.info("Starting Nina Assistant Loop. Press Ctrl+C to exit.")
        import tempfile

        turn_idx = 1
        try:
            while True:
                user_in = input(f"\n[Turn {turn_idx}] Press ENTER to record 4s audio (or 'q' to quit): ").strip()
                if user_in.lower() == 'q':
                    break

                temp_wav = Path(tempfile.gettempdir()) / f"nina_agent_turn_{turn_idx}.wav"
                print(">>> SPEAK NOW into your microphone...")
                recorder = MicrophoneRecorder()
                try:
                    recorded_path = recorder.record_to_wav(temp_wav, duration_seconds=4.0)
                    self.process_turn(recorded_path)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Recording failed: {e}")
                turn_idx += 1
        except KeyboardInterrupt:
            logger.info("Exiting Assistant Loop.")
