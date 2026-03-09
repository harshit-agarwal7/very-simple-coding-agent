"""Tests for providers/ollama.py — all network calls are mocked."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from agent.models import Message, Role, ToolCall, ToolDefinition, ToolSafety
from agent.providers.ollama import OllamaAdapter

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
    choice = SimpleNamespace(delta=delta, finish_reason=None)
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


class TestOllamaAdapter:
    def _make_adapter(self) -> OllamaAdapter:
        return OllamaAdapter(base_url="http://localhost:11434")

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
            model="llama3.2",
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
            _make_chunk(
                tool_calls=[{"id": "tc-1", "name": "read_file", "arguments": args_json}]
            ),
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
            model="llama3.2",
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
            model="llama3.2",
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
            _make_chunk(
                tool_calls=[{"id": "tc-1", "name": "read_file", "arguments": args_json}]
            ),
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
        mocker.patch("agent.providers.ollama.Live", return_value=mock_live)

        await adapter.stream_completion(
            messages=[Message(role=Role.USER, content="Read that file")],
            tools=[],
            model="llama3.2",
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
            model="llama3.2",
        )

        assert result == "A brief summary."


class TestParseTextToolCalls:
    """Unit tests for the _parse_text_tool_calls fallback helper."""

    def test_bare_json_single_call(self) -> None:
        from agent.providers.ollama import _parse_text_tool_calls

        text = '{"name": "list_directory", "arguments": {"path": "./"}}'
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "list_directory"
        assert calls[0].arguments == {"path": "./"}

    def test_tagged_single_call(self) -> None:
        from agent.providers.ollama import _parse_text_tool_calls

        text = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x.txt"}}</tool_call>'
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "/tmp/x.txt"}

    def test_tagged_multiple_calls(self) -> None:
        from agent.providers.ollama import _parse_text_tool_calls

        text = (
            '<tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>\n'
            '<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>'
        )
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].name == "list_directory"
        assert calls[1].name == "read_file"

    def test_non_tool_call_text_returns_empty(self) -> None:
        from agent.providers.ollama import _parse_text_tool_calls

        calls = _parse_text_tool_calls("Sure, here are the files in the directory.")
        assert calls == []

    def test_bare_json_with_string_arguments(self) -> None:
        from agent.providers.ollama import _parse_text_tool_calls

        args_str = json.dumps({"path": "/tmp"})
        text = json.dumps({"name": "read_file", "arguments": args_str})
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].arguments == {"path": "/tmp"}

    def test_empty_string_returns_empty(self) -> None:
        from agent.providers.ollama import _parse_text_tool_calls

        assert _parse_text_tool_calls("") == []

    def test_bare_json_multiple_calls(self) -> None:
        from agent.providers.ollama import _parse_text_tool_calls

        text = (
            '{"name": "think", "arguments": {"thought": "reasoning"}} '
            '{"name": "write_file", "arguments": {"path": "/tmp/x", "content": "hello"}}'
        )
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].name == "think"
        assert calls[1].name == "write_file"
        assert calls[1].arguments == {"path": "/tmp/x", "content": "hello"}


class TestOllamaFallbackToolCallPath:
    """Integration-level tests for the fallback path in stream_completion."""

    def _make_adapter(self) -> OllamaAdapter:
        return OllamaAdapter(base_url="http://localhost:11434")

    async def test_fallback_parses_bare_json_tool_call(self, mocker: MockerFixture) -> None:
        adapter = self._make_adapter()
        raw_content = '{"name": "list_directory", "arguments": {"path": "."}}'
        chunks = [
            _make_chunk(content=raw_content),
            _make_chunk(usage={"prompt_tokens": 10, "completion_tokens": 8}),
        ]
        mocker.patch.object(
            adapter._client.chat.completions,
            "create",
            new=mocker.AsyncMock(return_value=_AsyncIterator(chunks)),
        )
        message, _ = await adapter.stream_completion(
            messages=[Message(role=Role.USER, content="list files")],
            tools=[],
            model="qwen2.5-coder:7b",
            max_tokens=256,
        )
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].name == "list_directory"
        assert message.content == ""

    async def test_fallback_ignores_plain_text(self, mocker: MockerFixture) -> None:
        adapter = self._make_adapter()
        chunks = [
            _make_chunk(content="Here are the files."),
            _make_chunk(usage={"prompt_tokens": 5, "completion_tokens": 4}),
        ]
        mocker.patch.object(
            adapter._client.chat.completions,
            "create",
            new=mocker.AsyncMock(return_value=_AsyncIterator(chunks)),
        )
        message, _ = await adapter.stream_completion(
            messages=[Message(role=Role.USER, content="hello")],
            tools=[],
            model="qwen2.5-coder:7b",
            max_tokens=256,
        )
        assert message.tool_calls == []
        assert message.content == "Here are the files."


class TestGetProvider:
    def test_returns_ollama_adapter(self) -> None:
        from agent.models import Config
        from agent.providers import get_provider

        cfg = Config(provider="ollama", model="llama3.2", api_key="")
        adapter = get_provider(cfg)
        assert isinstance(adapter, OllamaAdapter)

    def test_ollama_default_base_url(self) -> None:
        from agent.models import Config
        from agent.providers import get_provider

        cfg = Config(provider="ollama", model="llama3.2", api_key="")
        adapter = get_provider(cfg)
        assert isinstance(adapter, OllamaAdapter)
        # The client base_url should include the default Ollama host
        assert "localhost:11434" in str(adapter._client.base_url)

    def test_ollama_custom_base_url(self) -> None:
        from agent.models import Config
        from agent.providers import get_provider

        cfg = Config(
            provider="ollama",
            model="llama3.2",
            api_key="",
            ollama_base_url="http://myhost:11434",
        )
        adapter = get_provider(cfg)
        assert isinstance(adapter, OllamaAdapter)
        assert "myhost:11434" in str(adapter._client.base_url)
