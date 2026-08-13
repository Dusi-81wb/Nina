import os

os.makedirs("src/nina/memory", exist_ok=True)
with open("src/nina/memory/__init__.py", "w") as f:
    pass

with open("src/nina/memory/context.py", "w") as f:
    f.write('''"""Memory subsystem for conversation context tracking."""
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class ConversationMemory:
    """Tracks conversation history between user and assistant."""

    def __init__(self, max_history: int = 50) -> None:
        self.max_history = max_history
        self._messages: list[Message] = []

    def add_message(self, role: str, content: str, metadata: dict[str, Any] = None) -> None:
        """Add a message to the history."""
        if metadata is None:
            metadata = {}
        self._messages.append(Message(role=role, content=content, metadata=metadata))

        # Keep within max history
        if len(self._messages) > self.max_history:
            self._messages = self._messages[-self.max_history:]

    def get_history(self) -> list[Dict[str, str]]:
        """Get full history as a list of dictionaries suitable for LLMs."""
        return [{"role": msg.role, "content": msg.content} for msg in self._messages]

    def get_context_window(self, max_tokens: int = 2000, chars_per_token: int = 4) -> list[Dict[str, str]]:
        """
        Get a context window that roughly fits within max_tokens.
        Uses a simplistic heuristic of chars_per_token.
        """
        max_chars = max_tokens * chars_per_token
        current_chars = 0
        window_messages = []

        for msg in reversed(self._messages):
            msg_chars = len(msg.content)
            if current_chars + msg_chars <= max_chars:
                window_messages.insert(0, {"role": msg.role, "content": msg.content})
                current_chars += msg_chars
            else:
                break

        return window_messages

    def clear(self) -> None:
        """Clear the conversation history."""
        self._messages.clear()
''')

os.makedirs("src/nina/tools", exist_ok=True)
with open("src/nina/tools/__init__.py", "w") as f:
    pass

with open("src/nina/tools/manager.py", "w") as f:
    f.write('''"""Tool management system for the AI assistant."""
from typing import Dict, Any, List
from collections.abc import Callable
from pydantic import BaseModel
import inspect

class Tool(BaseModel):
    name: str
    description: str
    func: Callable

class ToolManager:
    """Manages tool registration and execution."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register_tool(self, name: str, description: str, func: Callable) -> None:
        """Register a new tool."""
        self._tools[name] = Tool(name=name, description=description, func=func)

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_tools_schema(self) -> list[Dict[str, Any]]:
        """Get schema of all tools for LLM consumption."""
        schemas = []
        for tool in self._tools.values():
            sig = inspect.signature(tool.func)
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                param_type = "string"
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == bool:
                    param_type = "boolean"

                parameters["properties"][param_name] = {
                    "type": param_type,
                    "description": f"The {param_name} parameter."
                }
                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)

            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters
                }
            })
        return schemas

    def execute_tool(self, name: str, kwargs: dict[str, Any]) -> Any:
        """Execute a tool by name with given arguments."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")

        try:
            return tool.func(**kwargs)
        except Exception as e:
            return f"Error executing tool {name}: {e!s}"
''')

with open("src/nina/tools/builtin.py", "w") as f:
    f.write('''"""Built-in tools for the assistant."""
from datetime import datetime

def get_current_time() -> str:
    """Get the current time and date."""
    now = datetime.now()
    return now.strftime("The current date and time is %Y-%m-%d %H:%M:%S")

def get_weather(location: str) -> str:
    """Get the weather for a given location (Stub)."""
    return f"The weather in {location} is currently sunny and 72 degrees Fahrenheit."
''')

os.makedirs("src/nina/llm", exist_ok=True)
with open("src/nina/llm/__init__.py", "w") as f:
    pass

with open("src/nina/llm/provider.py", "w") as f:
    f.write('''"""Base interfaces for LLM providers."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from collections.abc import Generator

class LLMProvider(ABC):
    """Abstract base class for LLM communication."""

    @abstractmethod
    def generate_response(self, messages: list[Dict[str, str]], tools: list[Dict[str, Any]] = None) -> dict[str, Any]:
        """
        Generate a response from the LLM given a conversation history.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional list of tool schemas available to the model

        Returns:
            Dictionary containing the response 'content' and optional 'tool_calls'
        """

    @abstractmethod
    def stream_response(self, messages: list[Dict[str, str]]) -> Generator[str, None, None]:
        """Stream a response back."""
''')

with open("src/nina/llm/ollama.py", "w") as f:
    f.write('''"""Ollama LLM Provider implementation."""
import httpx
from typing import List, Dict, Any, Generator
import json

from nina.llm.provider import LLMProvider
from nina.core.logging import logger

class OllamaProvider(LLMProvider):
    """Provider for local Ollama LLMs."""

    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a response using Ollama API."""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }

        if tools:
            payload["tools"] = tools

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                msg = data.get("message", {})
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])

                return {
                    "content": content,
                    "tool_calls": tool_calls
                }
        except httpx.RequestError as e:
            logger.error(f"Failed to communicate with Ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing Ollama response: {e}")
            raise

    def stream_response(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Stream a response using Ollama API."""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
        except Exception as e:
            logger.error(f"Error streaming from Ollama: {e}")
            raise
''')

os.makedirs("src/nina/wakeword", exist_ok=True)
with open("src/nina/wakeword/__init__.py", "w") as f:
    pass

with open("src/nina/wakeword/detector.py", "w") as f:
    f.write('''"""Wake word detection subsystem."""
import numpy as np

class WakeWordDetector:
    """Detects wake words in audio streams (Stub implementation)."""

    def __init__(self, wake_word: str = "nina", sensitivity: float = 0.5):
        self.wake_word = wake_word.lower()
        self.sensitivity = sensitivity

    def detect_wake_word(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Detect if the wake word is present in the audio buffer.

        Note: This is a stub implementation. In a production environment,
        this would use a specialized engine like Porcupine, PocketSphinx,
        or an energy-based acoustic model.
        For now, we simulate detection based on a simple energy threshold
        and random probability to allow testing the pipeline.
        """
        if len(audio_buffer) == 0:
            return False

        # Calculate RMS energy
        if np.issubdtype(audio_buffer.dtype, np.integer):
            max_val = float(np.iinfo(audio_buffer.dtype).max)
            samples = audio_buffer.astype(np.float32) / max_val
        else:
            samples = audio_buffer.astype(np.float32)

        energy = float(np.sqrt(np.mean(samples**2)))

        # Simple heuristic: If there is significant energy, we "detected" the wake word
        # In real life, this would actually decode the phonemes or use a wake word model
        if energy > 0.05:
            # Add a slight random element to simulate the "Wake Word" firing
            # only sometimes when there is noise, just for testing purposes.
            # Real implementation would be deterministic based on acoustic features.
            import random
            if random.random() < self.sensitivity:
                return True

        return False
''')

os.makedirs("src/nina/tts", exist_ok=True)
with open("src/nina/tts/__init__.py", "w") as f:
    pass

with open("src/nina/tts/engine.py", "w") as f:
    f.write('''"""Text-to-Speech (TTS) integration."""
from abc import ABC, abstractmethod
import pyttsx3

from nina.core.logging import logger

class TTSProvider(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Convert text to speech and play it."""

class LocalTTSProvider(TTSProvider):
    """Local TTS using pyttsx3."""

    def __init__(self, rate: int = 175, volume: float = 1.0):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            self._available = True
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            self._available = False

    def speak(self, text: str) -> None:
        if not self._available:
            logger.warning("TTS is unavailable. Skipping speech output.")
            return

        if not text or not text.strip():
            return

        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error during TTS playback: {e}")
''')

os.makedirs("src/nina/assistant", exist_ok=True)
with open("src/nina/assistant/__init__.py", "w") as f:
    pass

with open("src/nina/assistant/agent.py", "w") as f:
    f.write('''"""Main Assistant Loop tying together all subsystems."""
import time
from typing import Optional, Any
from pathlib import Path

from nina.core.logging import logger
from nina.core.config import get_settings, NinaSettings
from nina.engine import NinaEmotionEngine
from nina.memory.context import ConversationMemory
from nina.llm.provider import LLMProvider
from nina.llm.ollama import OllamaProvider
from nina.tts.engine import TTSProvider, LocalTTSProvider
from nina.tools.manager import ToolManager
from nina.tools.builtin import get_current_time, get_weather
from nina.audio.recorder import MicrophoneRecorder
from nina.wakeword.detector import WakeWordDetector

class NinaAssistant:
    """The central agent integrating all voice assistant components."""

    def __init__(
        self,
        voice_engine: Optional[NinaEmotionEngine] = None,
        llm: Optional[LLMProvider] = None,
        tts: Optional[TTSProvider] = None,
        memory: Optional[ConversationMemory] = None,
        tools: Optional[ToolManager] = None,
        wake_word: Optional[WakeWordDetector] = None,
        settings: Optional[NinaSettings] = None
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
        except Exception as e:
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
        except Exception as e:
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
                except Exception as e:
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
                except Exception as e:
                    logger.error(f"Recording failed: {e}")
                turn_idx += 1
        except KeyboardInterrupt:
            logger.info("Exiting Assistant Loop.")
''')

with open("tests/unit/test_llm.py", "w") as f:
    f.write('''"""Unit tests for the LLM Provider integration."""
import pytest
import respx
import httpx
from nina.llm.ollama import OllamaProvider

@respx.mock
def test_ollama_generate_response_success():
    provider = OllamaProvider(base_url="http://localhost:11434", model_name="llama3")

    # Mock Ollama API response
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={
            "model": "llama3",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help?"
            },
            "done": True
        })
    )

    messages = [{"role": "user", "content": "Hi there!"}]
    response = provider.generate_response(messages)

    assert response["content"] == "Hello! How can I help?"
    assert response["tool_calls"] == []

@respx.mock
def test_ollama_generate_response_with_tools():
    provider = OllamaProvider(base_url="http://localhost:11434", model_name="llama3")

    tool_call_payload = {
        "function": {
            "name": "get_weather",
            "arguments": {"location": "San Francisco"}
        }
    }

    # Mock Ollama API response with tool calls
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={
            "model": "llama3",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call_payload]
            },
            "done": True
        })
    )

    messages = [{"role": "user", "content": "What is the weather in SF?"}]
    tools_schema = [{"type": "function", "function": {"name": "get_weather"}}]

    response = provider.generate_response(messages, tools=tools_schema)

    assert response["content"] == ""
    assert len(response["tool_calls"]) == 1
    assert response["tool_calls"][0]["function"]["name"] == "get_weather"
''')

with open("tests/unit/test_memory.py", "w") as f:
    f.write('''"""Unit tests for Conversation Memory subsystem."""
from nina.memory.context import ConversationMemory

def test_memory_add_and_retrieve():
    memory = ConversationMemory()
    memory.add_message("user", "Hello Nina")
    memory.add_message("assistant", "Hello user")

    history = memory.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello Nina"
    assert history[1]["role"] == "assistant"

def test_memory_max_history():
    memory = ConversationMemory(max_history=5)
    for i in range(10):
        memory.add_message("user", f"Message {i}")

    history = memory.get_history()
    assert len(history) == 5
    assert history[0]["content"] == "Message 5"
    assert history[-1]["content"] == "Message 9"

def test_memory_context_window():
    memory = ConversationMemory()
    memory.add_message("user", "A" * 10) # 10 chars
    memory.add_message("user", "B" * 20) # 20 chars
    memory.add_message("user", "C" * 30) # 30 chars

    # 20 tokens * 4 = 80 chars max
    window = memory.get_context_window(max_tokens=20, chars_per_token=4)
    assert len(window) == 3 # 10+20+30 = 60 chars < 80 chars

    # 10 tokens * 4 = 40 chars max
    # Going backwards:
    # "C" * 30 = 30 chars -> fits
    # "B" * 20 = 20 chars -> 30+20 = 50 > 40 -> doesn't fit
    window2 = memory.get_context_window(max_tokens=10, chars_per_token=4)
    assert len(window2) == 1
    assert window2[0]["content"] == "C" * 30
''')

with open("tests/unit/test_tools.py", "w") as f:
    f.write('''"""Unit tests for the Tools subsystem."""
from nina.tools.manager import ToolManager

def dummy_tool(x: int, y: str) -> str:
    """A dummy tool for testing."""
    return f"{y}{x}"

def test_tool_manager_registration():
    manager = ToolManager()
    manager.register_tool("dummy", "A dummy tool", dummy_tool)

    tool = manager.get_tool("dummy")
    assert tool is not None
    assert tool.name == "dummy"
    assert tool.description == "A dummy tool"

def test_tool_manager_schema_generation():
    manager = ToolManager()
    manager.register_tool("dummy", "A dummy tool", dummy_tool)

    schemas = manager.get_tools_schema()
    assert len(schemas) == 1

    schema = schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "dummy"
    assert "x" in schema["function"]["parameters"]["properties"]
    assert "y" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["x"]["type"] == "integer"
    assert schema["function"]["parameters"]["properties"]["y"]["type"] == "string"

def test_tool_manager_execution():
    manager = ToolManager()
    manager.register_tool("dummy", "A dummy tool", dummy_tool)

    result = manager.execute_tool("dummy", {"x": 5, "y": "test"})
    assert result == "test5"

def test_tool_manager_execution_error():
    manager = ToolManager()
    # Missing tool
    try:
        manager.execute_tool("not_found", {})
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Bad arguments
    manager.register_tool("dummy", "A dummy tool", dummy_tool)
    result = manager.execute_tool("dummy", {"wrong": "args"})
    assert "Error executing tool" in result
''')
