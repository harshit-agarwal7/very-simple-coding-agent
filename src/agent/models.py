"""Shared dataclasses and enums for the coding agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Message role in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolSafety(StrEnum):
    """Safety level of a tool — determines whether it requires user approval."""

    SAFE = "safe"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass
class ToolCall:
    """A tool invocation requested by the assistant.

    Args:
        id: Unique identifier for this tool call.
        name: Name of the tool to invoke.
        arguments: Parsed arguments dict for the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A single message in the conversation history.

    Args:
        role: Who produced this message.
        content: Text content of the message.
        tool_call_id: For role=TOOL, the ID of the tool call being responded to.
        tool_calls: For role=ASSISTANT, tool calls requested by the model.
    """

    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolResult:
    """The result of executing a tool.

    Args:
        tool_call_id: ID of the tool call this result corresponds to.
        name: Name of the tool that was executed.
        output: String output from the tool.
        is_error: Whether the tool encountered an error.
    """

    tool_call_id: str
    name: str
    output: str
    is_error: bool = False


@dataclass
class Usage:
    """Token usage for a single completion.

    Args:
        input_tokens: Number of tokens in the prompt.
        output_tokens: Number of tokens in the completion.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        """Total tokens consumed."""
        return self.input_tokens + self.output_tokens


@dataclass
class ToolDefinition:
    """Schema definition for a tool exposed to the model.

    Args:
        name: Tool name (used by the model to invoke it).
        description: Human/model-readable description of what the tool does.
        parameters: JSON Schema describing the tool's parameters.
        safety: Whether this tool auto-runs or requires user confirmation.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    safety: ToolSafety


@dataclass
class Config:
    """Runtime configuration for the agent.

    Args:
        provider: Provider identifier (e.g., 'openrouter').
        model: Model string (e.g., 'anthropic/claude-opus-4-6').
        api_key: API key for the provider.
        max_tokens: Maximum tokens per completion.
        max_history_tokens: Soft cap on history before compaction is suggested.
        system_prompt: System prompt prepended to every request.
    """

    provider: str
    model: str
    api_key: str
    max_tokens: int = 4096
    max_history_tokens: int = 80_000
    system_prompt: str = ""
