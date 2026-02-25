"""Execute-command tool: runs a shell command (requires approval)."""

from __future__ import annotations

import asyncio
import logging

from agent.models import ToolDefinition, ToolSafety
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30  # seconds

_DEFINITION = ToolDefinition(
    name="execute_command",
    description=(
        "Execute a shell command and return its combined stdout + stderr output. "
        "Commands run in the current working directory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "number",
                "description": f"Timeout in seconds. Defaults to {_DEFAULT_TIMEOUT}.",
            },
        },
        "required": ["command"],
    },
    safety=ToolSafety.REQUIRES_APPROVAL,
)


async def execute_command(command: str, timeout: float = _DEFAULT_TIMEOUT) -> str:
    """Run a shell command and capture its output.

    Args:
        command: The shell command string to execute.
        timeout: Maximum seconds to wait for the command to finish.

    Returns:
        Combined stdout + stderr output, or an error message.
    """
    logger.info("execute_command: running: %s", command)
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.warning("execute_command: timed out after %ss: %s", timeout, command)
            return f"Error: command timed out after {timeout} seconds."

        output = stdout.decode(errors="replace")
        exit_code = proc.returncode
        logger.debug("execute_command: exit=%d, output_len=%d", exit_code, len(output))

        if exit_code != 0:
            return f"Exit code {exit_code}:\n{output}"
        return output if output else "(no output)"

    except OSError as exc:
        logger.warning("execute_command: OS error: %s", exc)
        return f"Error executing command: {exc}"


register_tool(_DEFINITION, execute_command)
