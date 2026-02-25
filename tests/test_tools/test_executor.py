"""Tests for tools/executor.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import agent.tools  # noqa: F401 — ensure tools are registered
from agent.models import ToolCall
from agent.tools.executor import ToolExecutor
from agent.tools.registry import TOOL_REGISTRY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_call(name: str, arguments: dict | None = None) -> ToolCall:
    return ToolCall(id="tc-1", name=name, arguments=arguments or {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolExecutor:
    async def test_executes_safe_tool_without_prompt(self) -> None:
        executor = ToolExecutor()
        tc = _make_tool_call("think", {"thought": "testing"})
        with patch.object(executor, "_request_approval") as mock_approval:
            result = await executor.execute(tc)
        mock_approval.assert_not_called()
        assert result.output == "testing"
        assert result.is_error is False

    async def test_unknown_tool_returns_error(self) -> None:
        executor = ToolExecutor()
        tc = _make_tool_call("nonexistent_tool")
        result = await executor.execute(tc)
        assert result.is_error is True
        assert "unknown tool" in result.output.lower()

    async def test_approval_granted_executes_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        executor = ToolExecutor()
        tc = _make_tool_call("write_file", {"path": "/tmp/test.txt", "content": "x"})

        with (
            patch.object(executor, "_request_approval", new=AsyncMock(return_value=True)),
            patch("agent.tools.write_file.write_file", new=AsyncMock(return_value="ok")),
        ):
            # Directly patch fn in registry for this test.
            original_fn = TOOL_REGISTRY["write_file"].fn
            TOOL_REGISTRY["write_file"].fn = AsyncMock(return_value="written ok")
            try:
                result = await executor.execute(tc)
            finally:
                TOOL_REGISTRY["write_file"].fn = original_fn

        assert result.is_error is False
        assert result.output == "written ok"

    async def test_approval_denied_returns_error(self) -> None:
        executor = ToolExecutor()
        tc = _make_tool_call("write_file", {"path": "/tmp/x.txt", "content": "data"})

        with patch.object(executor, "_request_approval", new=AsyncMock(return_value=False)):
            result = await executor.execute(tc)

        assert result.is_error is True
        assert "denied" in result.output.lower()

    async def test_tool_bad_arguments_returns_error(self) -> None:
        """If a tool is called with wrong kwargs, TypeError → error result."""
        executor = ToolExecutor()
        # read_file requires 'path'; pass a wrong kwarg.
        tc = ToolCall(id="x", name="read_file", arguments={"wrong_arg": "value"})
        result = await executor.execute(tc)
        assert result.is_error is True

    async def test_result_has_correct_tool_call_id(self) -> None:
        executor = ToolExecutor()
        tc = ToolCall(id="my-id-123", name="think", arguments={"thought": "hi"})
        result = await executor.execute(tc)
        assert result.tool_call_id == "my-id-123"
