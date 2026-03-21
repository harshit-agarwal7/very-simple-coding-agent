"""Tool executor: dispatches tool calls with an approval gate for unsafe tools."""

import asyncio
import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.models import ToolCall, ToolResult, ToolSafety
from agent.tools.registry import TOOL_REGISTRY

logger = logging.getLogger(__name__)

_TOOL_LABELS: dict[str, str] = {
    "read_file": "Read file",
    "write_file": "Write file",
    "search_files": "Search files",
    "list_directory": "List directory",
    "execute_command": "Execute command",
    "think": "Think",
}

_LABEL_WIDTH = 16


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

    def __init__(self, console: Console | None = None) -> None:
        self._console = console if console is not None else Console()

    async def execute(self, tool_call: ToolCall, *, iteration: int = 0) -> ToolResult:
        """Execute a tool call, prompting for approval if necessary.

        Args:
            tool_call: The tool invocation requested by the assistant.
            iteration: The loop iteration index (0-based); unused but kept for
                API compatibility.

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
        label_padded = label.ljust(_LABEL_WIDTH)

        with self._console.status(f"  {label_padded}{arg_str}", spinner="dots"):
            result = await self._run_tool(tool_call, entry.fn, tool_call.arguments)

        if result.is_error:
            self._console.print(f"[red]  ✗  {label_padded}{arg_str}[/red]")
        else:
            self._console.print(f"[dim]  ✓  {label_padded}{arg_str}[/dim]")

        return result

    async def _request_approval(self, tool_call: ToolCall) -> bool:
        """Show a Rich Panel and ask the user for confirmation.

        Args:
            tool_call: The tool call needing approval.

        Returns:
            True if the user approved, False otherwise.
        """
        table = Table(show_header=False, box=None, padding=(0, 1))
        for k, v in tool_call.arguments.items():
            table.add_row(f"[dim]{k}[/dim]", _truncate_value(v))

        self._console.print(
            Panel(
                table,
                title="[bold]Tool Request[/bold]",
                subtitle=f"[bold]{tool_call.name}[/bold]",
            )
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
