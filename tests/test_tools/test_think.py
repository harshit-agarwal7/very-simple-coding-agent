"""Tests for tools/think.py."""

from agent.tools.think import think


class TestThink:
    async def test_returns_thought_unchanged(self) -> None:
        thought = "I need to check the file first."
        result = await think(thought=thought)
        assert result == thought

    async def test_empty_thought(self) -> None:
        result = await think(thought="")
        assert result == ""
