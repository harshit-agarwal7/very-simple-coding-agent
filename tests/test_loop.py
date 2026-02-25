"""Tests for src/agent/loop.py — fully mocked provider and tool executor."""

from __future__ import annotations

from unittest.mock import AsyncMock

from agent.loop import run_turn
from agent.memory import History
from agent.models import Config, Message, Role, ToolCall, ToolResult, Usage
from agent.tools.executor import ToolExecutor

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config() -> Config:
    return Config(
        provider="openrouter",
        model="anthropic/claude-opus-4-6",
        api_key="sk-test",
        max_tokens=256,
    )


def _plain_response(text: str = "Done.") -> tuple[Message, Usage]:
    """A response message with no tool calls."""
    return Message(role=Role.ASSISTANT, content=text), Usage(input_tokens=10, output_tokens=5)


def _tool_response(name: str, arguments: dict | None = None) -> tuple[Message, Usage]:
    """A response message with a single tool call."""
    tc = ToolCall(id="tc-1", name=name, arguments=arguments or {})
    return (
        Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
        Usage(input_tokens=20, output_tokens=10),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunTurn:
    async def test_single_text_response(self) -> None:
        """Agent responds with text only — no tools called."""
        history = History()
        provider = AsyncMock()
        provider.stream_completion = AsyncMock(return_value=_plain_response("Hello!"))
        executor = AsyncMock(spec=ToolExecutor)
        cfg = _make_config()

        await run_turn("Hi", history, provider, executor, cfg)

        # User + assistant message appended.
        msgs = history.messages
        assert msgs[0].role == Role.USER
        assert msgs[0].content == "Hi"
        assert msgs[1].role == Role.ASSISTANT
        assert msgs[1].content == "Hello!"

        # Usage recorded.
        assert history.usage.input_tokens == 10
        assert history.usage.output_tokens == 5

        executor.execute.assert_not_called()

    async def test_tool_call_then_text_response(self) -> None:
        """Agent calls a tool then responds with text."""
        history = History()
        provider = AsyncMock()
        provider.stream_completion = AsyncMock(
            side_effect=[
                _tool_response("think", {"thought": "Let me reason"}),
                _plain_response("All done."),
            ]
        )
        executor = AsyncMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(
                tool_call_id="tc-1",
                name="think",
                output="Let me reason",
            )
        )
        cfg = _make_config()

        await run_turn("Please help", history, provider, executor, cfg)

        msgs = history.messages
        roles = [m.role for m in msgs]
        # user → assistant (tool_call) → tool → assistant (final)
        assert roles == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]

        tool_msg = msgs[2]
        assert tool_msg.content == "Let me reason"
        assert tool_msg.tool_call_id == "tc-1"

        assert provider.stream_completion.call_count == 2
        executor.execute.assert_called_once()

    async def test_multiple_tool_calls_in_one_response(self) -> None:
        """Multiple tool calls in a single response are all executed."""
        history = History()

        tc1 = ToolCall(id="tc-1", name="think", arguments={"thought": "a"})
        tc2 = ToolCall(id="tc-2", name="think", arguments={"thought": "b"})
        multi_tool_response = (
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc1, tc2]),
            Usage(input_tokens=30, output_tokens=15),
        )

        provider = AsyncMock()
        provider.stream_completion = AsyncMock(
            side_effect=[
                multi_tool_response,
                _plain_response("Done with both."),
            ]
        )
        executor = AsyncMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            side_effect=[
                ToolResult(tool_call_id="tc-1", name="think", output="a"),
                ToolResult(tool_call_id="tc-2", name="think", output="b"),
            ]
        )
        cfg = _make_config()

        await run_turn("Do two things", history, provider, executor, cfg)

        assert executor.execute.call_count == 2

    async def test_max_iterations_guard(self) -> None:
        """Loop exits after MAX_ITERATIONS without infinite looping."""
        from agent.loop import _MAX_ITERATIONS

        history = History()
        provider = AsyncMock()
        # Always return a tool call — never a final text response.
        provider.stream_completion = AsyncMock(
            return_value=_tool_response("think", {"thought": "hmm"})
        )
        executor = AsyncMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(tool_call_id="tc-1", name="think", output="ok")
        )
        cfg = _make_config()

        await run_turn("loop forever", history, provider, executor, cfg)

        # Should have stopped at _MAX_ITERATIONS, not gone on forever.
        assert provider.stream_completion.call_count == _MAX_ITERATIONS

    async def test_usage_accumulates_across_iterations(self) -> None:
        history = History()
        provider = AsyncMock()
        provider.stream_completion = AsyncMock(
            side_effect=[
                _tool_response("think", {"thought": "x"}),
                _plain_response("ok"),
            ]
        )
        executor = AsyncMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(tool_call_id="tc-1", name="think", output="x")
        )
        cfg = _make_config()

        await run_turn("go", history, provider, executor, cfg)

        # First call: 20in+10out, second call: 10in+5out → 30in+15out.
        assert history.usage.input_tokens == 30
        assert history.usage.output_tokens == 15

    async def test_history_messages_passed_to_provider(self) -> None:
        """Provider receives full history including previous messages."""
        history = History()
        history.append(Message(role=Role.ASSISTANT, content="Previous response"))

        provider = AsyncMock()
        provider.stream_completion = AsyncMock(return_value=_plain_response("Hello"))
        executor = AsyncMock(spec=ToolExecutor)
        cfg = _make_config()

        await run_turn("New message", history, provider, executor, cfg)

        call_kwargs = provider.stream_completion.call_args.kwargs
        sent_messages = call_kwargs["messages"]
        # Should include previous response + new user message.
        assert any(m.content == "Previous response" for m in sent_messages)
        assert any(m.content == "New message" for m in sent_messages)
