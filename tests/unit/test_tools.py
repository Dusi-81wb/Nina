"""Unit tests for the Tools subsystem."""
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
