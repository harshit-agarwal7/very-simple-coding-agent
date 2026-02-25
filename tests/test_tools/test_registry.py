"""Tests for tools/registry.py."""

from __future__ import annotations

import agent.tools  # noqa: F401 — trigger registration
from agent.models import ToolSafety
from agent.tools.registry import (
    TOOL_REGISTRY,
    get_safe_tool_names,
    get_tool_definitions,
)


class TestToolRegistry:
    def test_all_expected_tools_registered(self) -> None:
        expected = {
            "read_file",
            "write_file",
            "list_directory",
            "search_files",
            "execute_command",
            "think",
        }
        assert expected.issubset(TOOL_REGISTRY.keys())

    def test_get_tool_definitions_returns_all(self) -> None:
        defs = get_tool_definitions()
        names = {d.name for d in defs}
        assert "read_file" in names
        assert "think" in names

    def test_safe_tool_names(self) -> None:
        safe = get_safe_tool_names()
        assert "read_file" in safe
        assert "list_directory" in safe
        assert "search_files" in safe
        assert "think" in safe
        assert "write_file" not in safe
        assert "execute_command" not in safe

    def test_tool_safety_values(self) -> None:
        assert TOOL_REGISTRY["write_file"].definition.safety == ToolSafety.REQUIRES_APPROVAL
        assert TOOL_REGISTRY["execute_command"].definition.safety == ToolSafety.REQUIRES_APPROVAL
        assert TOOL_REGISTRY["read_file"].definition.safety == ToolSafety.SAFE
        assert TOOL_REGISTRY["think"].definition.safety == ToolSafety.SAFE
