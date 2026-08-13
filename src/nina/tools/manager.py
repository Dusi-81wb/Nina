"""Tool management system for the AI assistant."""
import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


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

    def get_tools_schema(self) -> list[dict[str, Any]]:
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
