"""Data classes for eval cases and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agent.models import Message, Usage

if TYPE_CHECKING:
    from agent.memory import History


@dataclass
class ExpectedToolCall:
    """Describes a tool call the agent is expected to make.

    Args:
        name: Tool name that must be called.
        arguments: Expected arguments (or subset thereof).
        argument_match: ``"subset"`` checks only listed keys; ``"exact"``
            requires full equality.
    """

    name: str
    arguments: dict[str, Any] | None = None
    argument_match: Literal["exact", "subset"] = "subset"


@dataclass
class ScorerSpec:
    """Specification for a single output assertion.

    Args:
        kind: Type of check to perform.
        value: Value to check against (unused for ``"not_empty"``).
        case_sensitive: Whether string comparisons are case-sensitive.
    """

    kind: Literal["contains", "not_contains", "exact", "regex", "not_empty"]
    value: str = ""
    case_sensitive: bool = True


@dataclass
class EvalCase:
    """Complete specification of a single eval scenario.

    Args:
        id: Unique identifier for this eval.
        task: User prompt text sent to the agent.
        scripted_responses: Ordered list of ``(Message, Usage)`` tuples
            consumed one-per-provider-call by :class:`ScriptedProvider`.
        tool_outputs: Maps tool name → canned string returned for every call.
        expected_tool_calls: Tool calls the agent must make (in order or any order).
        tool_call_order: ``"ordered"`` (subsequence) or ``"unordered"`` (set membership).
        output_assertions: Scorer specs applied to the final assistant message.
        setup_files: Files to create under ``tmp_path`` before running (rel_path → content).
        tags: Arbitrary labels for filtering.
    """

    id: str
    task: str
    scripted_responses: list[tuple[Message, Usage]] = field(default_factory=list)
    tool_outputs: dict[str, str] = field(default_factory=dict)
    expected_tool_calls: list[ExpectedToolCall] = field(default_factory=list)
    tool_call_order: Literal["ordered", "unordered"] = "ordered"
    output_assertions: list[ScorerSpec] = field(default_factory=list)
    setup_files: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of running an :class:`EvalCase`.

    Args:
        case_id: ID of the eval case that was run.
        passed: True if all assertions passed.
        final_output: Content of the last non-tool-call assistant message.
        actual_tool_calls: List of ``(name, arguments)`` tuples in call order.
        failures: Human-readable failure descriptions (empty when passed).
        history: Full conversation history after the turn.
    """

    case_id: str
    passed: bool
    final_output: str
    actual_tool_calls: list[tuple[str, dict[str, Any]]]
    failures: list[str]
    history: History
