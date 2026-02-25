"""Tool registry: maps tool names to their definitions and implementations."""

from collections.abc import Callable, Coroutine
from typing import Any

from agent.models import ToolDefinition, ToolSafety

# Type alias for an async tool function.
ToolFn = Callable[..., Coroutine[Any, Any, str]]


class ToolEntry:
    """Combines a tool's schema definition with its implementation.

    Args:
        definition: The :class:`~agent.models.ToolDefinition` exposed to the model.
        fn: Async callable that executes the tool; receives **kwargs from arguments.
    """

    def __init__(self, definition: ToolDefinition, fn: ToolFn) -> None:
        self.definition = definition
        self.fn = fn


# Registry maps tool name → ToolEntry.
# Populated by each tool module via register_tool().
TOOL_REGISTRY: dict[str, ToolEntry] = {}


def register_tool(definition: ToolDefinition, fn: ToolFn) -> None:
    """Register a tool in the global registry.

    Args:
        definition: Tool schema definition.
        fn: Async function implementing the tool.
    """
    TOOL_REGISTRY[definition.name] = ToolEntry(definition=definition, fn=fn)


def get_tool_definitions() -> list[ToolDefinition]:
    """Return all registered tool definitions (for passing to the LLM).

    Returns:
        List of :class:`~agent.models.ToolDefinition` objects.
    """
    return [entry.definition for entry in TOOL_REGISTRY.values()]


def get_safe_tool_names() -> set[str]:
    """Return the names of all tools that are marked SAFE.

    Returns:
        Set of tool name strings.
    """
    return {
        name
        for name, entry in TOOL_REGISTRY.items()
        if entry.definition.safety == ToolSafety.SAFE
    }
