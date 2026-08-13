"""Built-in tools for the assistant."""
from datetime import datetime

def get_current_time() -> str:
    """Get the current time and date."""
    now = datetime.now()
    return now.strftime("The current date and time is %Y-%m-%d %H:%M:%S")

def get_weather(location: str) -> str:
    """Get the weather for a given location (Stub)."""
    return f"The weather in {location} is currently sunny and 72 degrees Fahrenheit."
