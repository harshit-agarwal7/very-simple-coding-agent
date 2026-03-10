"""Tests for providers/openrouter.py — all network calls are mocked."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from agent.models import Message, Role, ToolCall, ToolDefinition, ToolSafety
from agent.providers.openrouter import OpenRouterAdapter

# ---------------------------------------------------------------------------
# Helpers to build mock stream chunks
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
) -> SimpleNamespace:
    """Build a fake ChatCompletionChunk-like object."""
    usage_obj = None
    if usage:
        usage_obj = SimpleNamespace(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
        )

    tc_deltas = None
    if tool_calls:
        tc_deltas = [
            SimpleNamespace(
                index=i,
                id=tc.get("id", ""),
                function=SimpleNamespace(
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", ""),
                ),
            )
            for i, tc in enumerate(tool_calls)
        ]

    delta = SimpleNamespace(content=content, tool_calls=tc_deltas)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(usage=usage_obj, choices=[choice])


class _AsyncIterator:
    """Simple async iterator wrapping a list of chunks."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> _AsyncIterator:
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenRouterAdapter:
    def _make_adapter(self) -> OpenRouterAdapter:
        return OpenRouterAdapter(api_key="sk-test")

    # --- format_messages ---

    def test_format_messages_user(self) -> None:
        adapter = self._make_adapter()
        msgs = [Message(role=Role.USER, content="Hello")]
        result = adapter.format_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_format_messages_assistant(self) -> None:
        adapter = self._make_adapter()
        msgs = [Message(role=Role.ASSISTANT, content="Hi there")]
        result = adapter.format_messages(msgs)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_format_messages_tool_result(self) -> None:
        adapter = self._make_adapter()
        msgs = [Message(role=Role.TOOL, content="file data", tool_call_id="tc-1")]
        result = adapter.format_messages(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tc-1"
        assert result[0]["content"] == "file data"

    def test_format_messages_assistant_with_tool_calls(self) -> None:
        adapter = self._make_adapter()
        tc = ToolCall(id="tc-1", name="read_file", arguments={"path": "/tmp/x"})
        msgs = [Message(role=Role.ASSISTANT, content="", tool_calls=[tc])]
        result = adapter.format_messages(msgs)
        assert result[0]["role"] == "assistant"
        wire_tcs = result[0]["tool_calls"]
        assert isinstance(wire_tcs, list)
        assert wire_tcs[0]["id"] == "tc-1"  # type: ignore[index]
        assert wire_tcs[0]["function"]["name"] == "read_file"  # type: ignore[index]

    # --- format_tools ---

    def test_format_tools(self) -> None:
        adapter = self._make_adapter()
        td = ToolDefinition(
            name="think",
            description="Reason step-by-step",
            parameters={"type": "object", "properties": {}},
            safety=ToolSafety.SAFE,
        )
        result = adapter.format_tools([td])
        assert result[0]["type"] == "function"
        fn = result[0]["function"]
        assert fn["name"] == "think"  # type: ignore[index]
        assert fn["description"] == "Reason step-by-step"  # type: ignore[index]

    # --- stream_completion ---

    async def test_stream_completion_text_only(self, mocker: MockerFixture) -> None:
        adapter = self._make_adapter()
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world"),
            _make_chunk(usage={"prompt_tokens": 10, "completion_tokens": 5}),
        ]
        mock_stream = _AsyncIterator(chunks)

        mocker.patch.object(
            adapter._client.chat.completions,
            "create",
            new=mocker.AsyncMock(return_value=mock_stream),
        )
        message, usage = await adapter.stream_completion(
            messages=[Message(role=Role.USER, content="Hi")],
            tools=[],
            model="anthropic/claude-opus-4-6",
            max_tokens=256,
        )

        assert message.content == "Hello world"
        assert message.tool_calls == []
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5

    async def test_stream_completion_with_tool_call(self, mocker: MockerFixture) -> None:
        adapter = self._make_adapter()
        args_json = json.dumps({"path": "/tmp/foo.txt"})
        chunks = [
            _make_chunk(tool_calls=[{"id": "tc-1", "name": "read_file", "arguments": args_json}]),
            _make_chunk(usage={"prompt_tokens": 20, "completion_tokens": 10}),
        ]
        mock_stream = _AsyncIterator(chunks)

        mocker.patch.object(
            adapter._client.chat.completions,
            "create",
            new=mocker.AsyncMock(return_value=mock_stream),
        )
        message, usage = await adapter.stream_completion(
            messages=[Message(role=Role.USER, content="Read that file")],
            tools=[],
            model="anthropic/claude-opus-4-6",
            max_tokens=256,
        )

        assert len(message.tool_calls) == 1
        tc = message.tool_calls[0]
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/tmp/foo.txt"}
        assert usage.input_tokens == 20

    async def test_stream_completion_system_prompt_injected(self, mocker: MockerFixture) -> None:
        adapter = self._make_adapter()
        chunks = [_make_chunk(usage={"prompt_tokens": 5, "completion_tokens": 3})]
        mock_stream = _AsyncIterator(chunks)

        create_mock = mocker.AsyncMock(return_value=mock_stream)
        mocker.patch.object(adapter._client.chat.completions, "create", new=create_mock)
        await adapter.stream_completion(
            messages=[],
            tools=[],
            model="openai/gpt-4o",
            max_tokens=64,
            system_prompt="Be helpful.",
        )

        call_kwargs = create_mock.call_args.kwargs
        messages_sent = call_kwargs["messages"]
        assert messages_sent[0]["role"] == "system"
        assert "Be helpful" in str(messages_sent[0]["content"])

    async def test_stream_completion_tool_call_only_clears_spinner(
        self, mocker: MockerFixture
    ) -> None:
        adapter = self._make_adapter()
        args_json = json.dumps({"path": "/tmp/foo.txt"})
        chunks = [
            _make_chunk(tool_calls=[{"id": "tc-1", "name": "read_file", "arguments": args_json}]),
            _make_chunk(usage={"prompt_tokens": 20, "completion_tokens": 10}),
        ]
        mocker.patch.object(
            adapter._client.chat.completions,
            "create",
            new=mocker.AsyncMock(return_value=_AsyncIterator(chunks)),
        )

        mock_live = MagicMock()
        mock_live.__enter__ = MagicMock(return_value=mock_live)
        mock_live.__exit__ = MagicMock(return_value=False)
        mocker.patch("agent.providers.openrouter.Live", return_value=mock_live)

        await adapter.stream_completion(
            messages=[Message(role=Role.USER, content="Read that file")],
            tools=[],
            model="anthropic/claude-opus-4-6",
            max_tokens=256,
        )

        mock_live.update.assert_called_with("")

    # --- summarize ---

    async def test_summarize(self, mocker: MockerFixture) -> None:
        adapter = self._make_adapter()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "A brief summary."

        mocker.patch.object(
            adapter._client.chat.completions,
            "create",
            new=mocker.AsyncMock(return_value=mock_response),
        )
        result = await adapter.summarize(
            messages=[Message(role=Role.USER, content="Hello")],
            model="anthropic/claude-opus-4-6",
        )

        assert result == "A brief summary."


class TestGetProvider:
    def test_returns_openrouter_adapter(self) -> None:
        from agent.models import Config
        from agent.providers import get_provider

        cfg = Config(provider="openrouter", model="openai/gpt-4o", api_key="sk-test")
        adapter = get_provider(cfg)
        assert isinstance(adapter, OpenRouterAdapter)

    def test_unknown_provider_raises(self) -> None:
        from agent.models import Config
        from agent.providers import get_provider

        cfg = Config(provider="unknown", model="x/y", api_key="key")
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider(cfg)
