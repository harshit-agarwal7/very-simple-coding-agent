"""Tests for src/agent/memory.py."""

from __future__ import annotations

from unittest.mock import AsyncMock

from agent.memory import History
from agent.models import Message, Role, Usage


class TestHistory:
    def test_initial_state(self) -> None:
        h = History()
        assert h.messages == []
        assert h.usage.total == 0
        assert h.is_over_limit is False

    def test_append(self) -> None:
        h = History()
        msg = Message(role=Role.USER, content="Hello")
        h.append(msg)
        assert len(h.messages) == 1
        assert h.messages[0].content == "Hello"

    def test_messages_returns_copy(self) -> None:
        h = History()
        h.append(Message(role=Role.USER, content="a"))
        snapshot = h.messages
        h.append(Message(role=Role.USER, content="b"))
        # snapshot should not be affected
        assert len(snapshot) == 1

    def test_record_usage_accumulates(self) -> None:
        h = History()
        h.record_usage(Usage(input_tokens=100, output_tokens=50))
        h.record_usage(Usage(input_tokens=200, output_tokens=75))
        assert h.usage.input_tokens == 300
        assert h.usage.output_tokens == 125
        assert h.usage.total == 425

    def test_is_over_limit_false(self) -> None:
        h = History(max_history_tokens=1000)
        h.record_usage(Usage(input_tokens=400, output_tokens=400))
        assert h.is_over_limit is False

    def test_is_over_limit_true(self) -> None:
        h = History(max_history_tokens=100)
        h.record_usage(Usage(input_tokens=60, output_tokens=60))
        assert h.is_over_limit is True

    def test_clear(self) -> None:
        h = History()
        h.append(Message(role=Role.USER, content="x"))
        h.record_usage(Usage(input_tokens=10, output_tokens=5))
        h.clear()
        assert h.messages == []
        assert h.usage.total == 0

    async def test_compact(self) -> None:
        h = History()
        h.append(Message(role=Role.USER, content="Hello"))
        h.append(Message(role=Role.ASSISTANT, content="Hi there"))
        h.record_usage(Usage(input_tokens=50, output_tokens=20))

        mock_provider = AsyncMock()
        mock_provider.summarize = AsyncMock(return_value="User greeted, assistant replied.")

        await h.compact(mock_provider, model="anthropic/claude-opus-4-6")

        # History should be a single summary message.
        assert len(h.messages) == 1
        assert h.messages[0].role == Role.ASSISTANT
        assert "User greeted" in h.messages[0].content
        # Usage should be reset.
        assert h.usage.total == 0

        mock_provider.summarize.assert_called_once()

    async def test_compact_empty_history(self) -> None:
        h = History()
        mock_provider = AsyncMock()
        # Should not raise, should not call summarize.
        await h.compact(mock_provider, model="any/model")
        mock_provider.summarize.assert_not_called()
        assert h.messages == []
