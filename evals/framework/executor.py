"""StubToolExecutor: records tool calls and returns canned output for evals."""

from __future__ import annotations

from agent.models import ToolCall, ToolResult
from agent.tools.executor import ToolExecutor
from agent.tools.registry import TOOL_REGISTRY


class StubToolExecutor(ToolExecutor):
    """Tool executor that records calls and returns canned responses.

    For tools listed in *real_tools*, the actual implementation is invoked
    directly (bypassing the approval gate). All other tools return the
    string from *tool_outputs* (or *default_output* if not listed there).

    Args:
        tool_outputs: Maps tool name → canned return string.
        default_output: Fallback string when a tool is not in *tool_outputs*.
        real_tools: Names of tools whose real implementations should run.
    """

    def __init__(
        self,
        tool_outputs: dict[str, str] | None = None,
        default_output: str = "ok",
        real_tools: set[str] | None = None,
    ) -> None:
        self._tool_outputs: dict[str, str] = tool_outputs or {}
        self._default_output: str = default_output
        self._real_tools: set[str] = real_tools or set()
        self.calls: list[ToolCall] = []

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call, recording it and bypassing the approval gate.

        Args:
            tool_call: The tool invocation requested by the assistant.

        Returns:
            A :class:`~agent.models.ToolResult` with either the canned output
            or the result of the real tool implementation.
        """
        self.calls.append(tool_call)

        if tool_call.name in self._real_tools:
            entry = TOOL_REGISTRY.get(tool_call.name)
            if entry is None:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    output=f"Error: unknown tool '{tool_call.name}'.",
                    is_error=True,
                )
            return await ToolExecutor._run_tool(tool_call, entry.fn, tool_call.arguments)

        output = self._tool_outputs.get(tool_call.name, self._default_output)
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            output=output,
        )
