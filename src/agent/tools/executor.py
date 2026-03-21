"""Tool executor: dispatches tool calls with an approval gate for unsafe tools."""

import asyncio
import logging
from typing import Any

from rich.console import Console

from agent.models import ToolCall, ToolResult, ToolSafety
from agent.tools.registry import TOOL_REGISTRY

logger = logging.getLogger(__name__)
_console = Console()

_TOOL_LABELS: dict[str, str] = {
    "read_file": "Reading file",
    "write_file": "Writing file",
    "search_files": "Searching files",
    "list_directory": "Listing directory",
    "execute_command": "Executing command",
    "think": "Thinking",
}


def _truncate_value(value: Any, max_len: int = 100) -> str:
    """Truncate a value for display, appending '...' if truncated."""
    s = repr(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


class ToolExecutor:
    """Dispatches tool calls from the assistant to their implementations.

    Safe tools run immediately; tools requiring approval prompt the user
    first via stdin (off the event loop via ``run_in_executor``).
    """

    async def execute(self, tool_call: ToolCall, *, iteration: int = 0) -> ToolResult:
        """Execute a tool call, prompting for approval if necessary.

        Args:
            tool_call: The tool invocation requested by the assistant.
            iteration: The loop iteration index (0-based); used to prefix the
                status line with the human-readable step number.

        Returns:
            A :class:`~agent.models.ToolResult` with the tool's output or an
            error description.
        """
        entry = TOOL_REGISTRY.get(tool_call.name)
        if entry is None:
            logger.warning("Unknown tool requested: %s", tool_call.name)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                output=f"Error: unknown tool '{tool_call.name}'.",
                is_error=True,
            )

        if entry.definition.safety == ToolSafety.REQUIRES_APPROVAL:
            approved = await self._request_approval(tool_call)
            if not approved:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    output="Tool execution denied by user.",
                    is_error=True,
                )

        first_val = next(iter(tool_call.arguments.values()), "") if tool_call.arguments else ""
        arg_str = f"  {str(first_val)[:60]}" if first_val else ""
        label = _TOOL_LABELS.get(tool_call.name, tool_call.name)
        step_prefix = f"step {iteration + 1}  "

        result = await self._run_tool(tool_call, entry.fn, tool_call.arguments)

        if result.is_error:
            _console.print(f"[red]  {step_prefix}✗  {label}{arg_str}[/red]")
        else:
            _console.print(f"[dim]  {step_prefix}✓  {label}{arg_str}[/dim]")

        return result

    async def _request_approval(self, tool_call: ToolCall) -> bool:
        """Print a proposal box and ask the user for confirmation.

        Args:
            tool_call: The tool call needing approval.

        Returns:
            True if the user approved, False otherwise.
        """
        args_display = "\n".join(
            f"  {k}: {_truncate_value(v)}" for k, v in tool_call.arguments.items()
        )
        _console.print(
            f"\n┌─ Tool request ──────────────────────────────\n"
            f"│ Tool : {tool_call.name}\n"
            f"│ Args :\n{args_display}\n"
            f"└─────────────────────────────────────────────"
        )
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, input, "Allow? [y/N] ")
        return answer.strip().lower() == "y"

    @staticmethod
    async def _run_tool(
        tool_call: ToolCall,
        fn: Any,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Invoke the tool function and capture the result.

        Args:
            tool_call: The originating tool call (for ID/name tracking).
            fn: The async callable to invoke.
            arguments: Keyword arguments to pass to *fn*.

        Returns:
            The :class:`~agent.models.ToolResult`.
        """
        try:
            output: str = await fn(**arguments)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                output=output,
            )
        except TypeError as exc:
            logger.warning("Tool %s called with bad arguments: %s", tool_call.name, exc)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                output=f"Error: invalid arguments for tool '{tool_call.name}': {exc}",
                is_error=True,
            )
        except Exception as exc:
            logger.exception("Unexpected error in tool %s", tool_call.name)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                output=f"Error: unexpected error in tool '{tool_call.name}': {exc}",
                is_error=True,
            )
