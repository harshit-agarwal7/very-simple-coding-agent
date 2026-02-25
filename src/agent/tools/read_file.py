"""Read-file tool: reads the content of a file from the filesystem."""

from __future__ import annotations

import logging
from pathlib import Path

from agent.models import ToolDefinition, ToolSafety
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

_DEFINITION = ToolDefinition(
    name="read_file",
    description=(
        "Read the contents of a file at the given path. "
        "Returns the file content as a string."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file to read.",
            }
        },
        "required": ["path"],
    },
    safety=ToolSafety.SAFE,
)


async def read_file(path: str) -> str:
    """Read a file and return its contents.

    Args:
        path: Path to the file.

    Returns:
        File contents as a string, or an error message if reading fails.
    """
    try:
        content = Path(path).read_text(encoding="utf-8")
        logger.debug("read_file: read %d bytes from %s", len(content), path)
        return content
    except FileNotFoundError:
        logger.warning("read_file: file not found: %s", path)
        return f"Error: file not found: {path}"
    except PermissionError:
        logger.warning("read_file: permission denied: %s", path)
        return f"Error: permission denied: {path}"
    except OSError as exc:
        logger.warning("read_file: OS error reading %s: %s", path, exc)
        return f"Error reading file: {exc}"


register_tool(_DEFINITION, read_file)
