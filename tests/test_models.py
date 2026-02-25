"""Tests for src/agent/models.py."""

from agent.models import (
    Config,
    Message,
    Role,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolSafety,
    Usage,
)


class TestRole:
    def test_values(self) -> None:
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"

    def test_is_str_subclass(self) -> None:
        assert isinstance(Role.USER, str)


class TestToolSafety:
    def test_values(self) -> None:
        assert ToolSafety.SAFE == "safe"
        assert ToolSafety.REQUIRES_APPROVAL == "requires_approval"


class TestToolCall:
    def test_creation(self) -> None:
        tc = ToolCall(id="tc1", name="read_file", arguments={"path": "/tmp/foo.txt"})
        assert tc.id == "tc1"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/tmp/foo.txt"}


class TestMessage:
    def test_user_message(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.tool_call_id is None
        assert msg.tool_calls == []

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="1", name="think", arguments={"thought": "..."})
        msg = Message(role=Role.ASSISTANT, content="", tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "think"

    def test_tool_message(self) -> None:
        msg = Message(role=Role.TOOL, content="file contents", tool_call_id="tc1")
        assert msg.tool_call_id == "tc1"

    def test_tool_calls_default_is_independent(self) -> None:
        # Each instance should have its own list.
        m1 = Message(role=Role.USER, content="a")
        m2 = Message(role=Role.USER, content="b")
        m1.tool_calls.append(ToolCall(id="x", name="y", arguments={}))
        assert m2.tool_calls == []


class TestToolResult:
    def test_defaults(self) -> None:
        r = ToolResult(tool_call_id="tc1", name="read_file", output="data")
        assert r.is_error is False

    def test_error(self) -> None:
        r = ToolResult(tool_call_id="tc1", name="read_file", output="err", is_error=True)
        assert r.is_error is True


class TestUsage:
    def test_total(self) -> None:
        u = Usage(input_tokens=100, output_tokens=50)
        assert u.total == 150

    def test_defaults(self) -> None:
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.total == 0


class TestToolDefinition:
    def test_creation(self) -> None:
        td = ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {}},
            safety=ToolSafety.SAFE,
        )
        assert td.name == "read_file"
        assert td.safety == ToolSafety.SAFE


class TestConfig:
    def test_defaults(self) -> None:
        cfg = Config(provider="openrouter", model="anthropic/claude-opus-4-6", api_key="sk-test")
        assert cfg.max_tokens == 4096
        assert cfg.max_history_tokens == 80_000
        assert cfg.system_prompt == ""

    def test_custom_values(self) -> None:
        cfg = Config(
            provider="openrouter",
            model="openai/gpt-4o",
            api_key="sk-test",
            max_tokens=2048,
            max_history_tokens=40_000,
            system_prompt="You are helpful.",
        )
        assert cfg.max_tokens == 2048
        assert cfg.system_prompt == "You are helpful."
