"""Search-files tool: glob file matching + text search within files."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from agent.models import ToolDefinition, ToolSafety
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

_DEFINITION = ToolDefinition(
    name="search_files",
    description=(
        "Search for files matching a glob pattern, optionally filtering by a regex "
        "pattern matched against file contents. "
        "Returns matching file paths (and, if a content pattern is given, the matching lines)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Root directory to search in. Defaults to '.'.",
            },
            "glob_pattern": {
                "type": "string",
                "description": "Glob pattern for filenames, e.g. '*.py' or '**/*.md'.",
            },
            "content_pattern": {
                "type": "string",
                "description": (
                    "Optional regex pattern to search for within matched files. "
                    "If omitted, only filenames are returned."
                ),
            },
        },
        "required": ["glob_pattern"],
    },
    safety=ToolSafety.SAFE,
)

_MAX_RESULTS = 200
_MAX_FILE_SIZE = 1024 * 1024  # 1 MB


async def search_files(
    glob_pattern: str,
    directory: str = ".",
    content_pattern: str | None = None,
) -> str:
    """Search for files and optionally grep their contents.

    Args:
        glob_pattern: Glob pattern to match filenames (e.g. ``*.py``).
        directory: Root directory for the search. Defaults to ``"."``.
        content_pattern: Optional regex to search within matched files.

    Returns:
        Formatted string with results, or an error message.
    """
    root = Path(directory)
    if not root.exists():
        return f"Error: directory not found: {directory}"
    if not root.is_dir():
        return f"Error: not a directory: {directory}"

    try:
        matched_files = list(root.rglob(glob_pattern))
    except OSError as exc:
        logger.warning("search_files: glob error: %s", exc)
        return f"Error during file search: {exc}"

    matched_files = [f for f in matched_files if f.is_file()]

    if not matched_files:
        return "No files found matching the pattern."

    if content_pattern is None:
        lines = [str(f) for f in sorted(matched_files)[:_MAX_RESULTS]]
        result = "\n".join(lines)
        if len(matched_files) > _MAX_RESULTS:
            result += f"\n... ({len(matched_files) - _MAX_RESULTS} more files not shown)"
        return result

    # Grep within files.
    try:
        regex = re.compile(content_pattern, re.MULTILINE)
    except re.error as exc:
        return f"Error: invalid regex pattern: {exc}"

    output_lines: list[str] = []
    total_matches = 0

    for filepath in sorted(matched_files):
        if total_matches >= _MAX_RESULTS:
            output_lines.append(f"... stopped after {_MAX_RESULTS} matches")
            break
        try:
            if filepath.stat().st_size > _MAX_FILE_SIZE:
                logger.debug("search_files: skipping large file %s", filepath)
                continue
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                output_lines.append(f"{filepath}:{lineno}: {line.rstrip()}")
                total_matches += 1
                if total_matches >= _MAX_RESULTS:
                    break

    if not output_lines:
        return "No matches found."
    return "\n".join(output_lines)


register_tool(_DEFINITION, search_files)
