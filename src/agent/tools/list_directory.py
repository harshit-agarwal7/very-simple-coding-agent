"""List-directory tool: lists files and directories at a given path."""

import logging
from pathlib import Path

from agent.models import ToolDefinition, ToolSafety
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

_DEFINITION = ToolDefinition(
    name="list_directory",
    description=(
        "List the contents of a directory. Returns the names of files and "
        "subdirectories, one per line, with a trailing '/' for directories."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the directory to list. Defaults to '.'.",
            }
        },
        "required": [],
    },
    safety=ToolSafety.SAFE,
)


async def list_directory(path: str = ".") -> str:
    """List contents of a directory.

    Args:
        path: Directory path to list. Defaults to current directory.

    Returns:
        Newline-separated listing, or an error message.
    """
    try:
        entries = sorted(Path(path).iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
        result = "\n".join(lines) if lines else "(empty directory)"
        logger.debug("list_directory: %d entries in %s", len(lines), path)
        return result
    except FileNotFoundError:
        logger.warning("list_directory: not found: %s", path)
        return f"Error: directory not found: {path}"
    except NotADirectoryError:
        logger.warning("list_directory: not a directory: %s", path)
        return f"Error: not a directory: {path}"
    except PermissionError:
        logger.warning("list_directory: permission denied: %s", path)
        return f"Error: permission denied: {path}"
    except OSError as exc:
        logger.warning("list_directory: OS error for %s: %s", path, exc)
        return f"Error listing directory: {exc}"


register_tool(_DEFINITION, list_directory)
