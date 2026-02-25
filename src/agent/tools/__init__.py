"""Tools package: imports all tools to register them, exports registry helpers."""

# Import tool modules so their register_tool() calls populate TOOL_REGISTRY.
import agent.tools.execute_command  # noqa: F401
import agent.tools.list_directory  # noqa: F401
import agent.tools.read_file  # noqa: F401
import agent.tools.search_files  # noqa: F401
import agent.tools.think  # noqa: F401
import agent.tools.write_file  # noqa: F401
from agent.tools.executor import ToolExecutor
from agent.tools.registry import (
    TOOL_REGISTRY,
    ToolEntry,
    get_safe_tool_names,
    get_tool_definitions,
)

__all__ = [
    "TOOL_REGISTRY",
    "ToolEntry",
    "ToolExecutor",
    "get_safe_tool_names",
    "get_tool_definitions",
]
