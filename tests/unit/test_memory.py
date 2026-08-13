"""Unit tests for Conversation Memory subsystem."""
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
