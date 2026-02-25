"""Tests for tools/execute_command.py."""

from __future__ import annotations

from agent.tools.execute_command import execute_command


class TestExecuteCommand:
    async def test_simple_command(self) -> None:
        result = await execute_command(command="echo hello")
        assert "hello" in result

    async def test_command_with_nonzero_exit(self) -> None:
        result = await execute_command(command="exit 1", timeout=5)
        assert "Exit code 1" in result or "1" in result

    async def test_command_timeout(self) -> None:
        result = await execute_command(command="sleep 10", timeout=0.2)
        assert "timed out" in result.lower() or "timeout" in result.lower()

    async def test_invalid_command(self) -> None:
        result = await execute_command(command="this_command_does_not_exist_xyz123")
        # Should return an error, not raise.
        assert result  # non-empty error message

    async def test_stderr_captured(self) -> None:
        result = await execute_command(command="echo error >&2 && exit 1", timeout=5)
        assert "error" in result or "Exit code" in result
