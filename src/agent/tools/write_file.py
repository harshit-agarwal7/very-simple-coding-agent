"""Write-file tool: writes content to a file (requires approval)."""

from __future__ import annotations

import logging
from pathlib import Path

from agent.models import ToolDefinition, ToolSafety
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

_DEFINITION = ToolDefinition(
    name="write_file",
    description=(
        "Write content to a file. Creates the file (and any parent directories) "
        "if it does not exist, or overwrites it if it does."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
    safety=ToolSafety.REQUIRES_APPROVAL,
)


async def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    Args:
        path: Destination file path.
        content: Text to write.

    Returns:
        Success message, or an error message if writing fails.
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.info("write_file: wrote %d bytes to %s", len(content), path)
        return f"Successfully wrote {len(content)} bytes to {path}"
    except PermissionError:
        logger.warning("write_file: permission denied: %s", path)
        return f"Error: permission denied: {path}"
    except OSError as exc:
        logger.warning("write_file: OS error writing %s: %s", path, exc)
        return f"Error writing file: {exc}"


register_tool(_DEFINITION, write_file)
