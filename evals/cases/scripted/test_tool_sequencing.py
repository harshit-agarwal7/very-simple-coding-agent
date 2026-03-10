"""Eval: agent follows a required multi-step tool sequence."""

from agent.models import Message, Role, ToolCall, Usage
from evals.framework import EvalCase, EvalResult, ExpectedToolCall, ScorerSpec, run_eval
from evals.framework.schema import EvalCase as EC

_USAGE = Usage(input_tokens=60, output_tokens=25)

# --- Case 1: think → read_file → final answer --------------------------------

_THINK_TURN = Message(
    role=Role.ASSISTANT,
    content="",
    tool_calls=[ToolCall(id="tc-1", name="think", arguments={"thought": "I should read first."})],
)
_READ_TURN = Message(
    role=Role.ASSISTANT,
    content="",
    tool_calls=[ToolCall(id="tc-2", name="read_file", arguments={"path": "notes.txt"})],
)
_FINAL_TURN = Message(
    role=Role.ASSISTANT,
    content="The notes say: important meeting tomorrow.",
    tool_calls=[],
)

CASE_ORDERED = EvalCase(
    id="think_then_read_ordered",
    task="Think about what to do, then read notes.txt and report its contents.",
    scripted_responses=[(_THINK_TURN, _USAGE), (_READ_TURN, _USAGE), (_FINAL_TURN, _USAGE)],
    tool_outputs={
        "think": "ok",
        "read_file": "important meeting tomorrow",
    },
    expected_tool_calls=[
        ExpectedToolCall(name="think"),
        ExpectedToolCall(name="read_file", arguments={"path": "notes.txt"}),
    ],
    tool_call_order="ordered",
    output_assertions=[
        ScorerSpec(kind="not_empty"),
        ScorerSpec(kind="contains", value="meeting"),
    ],
    tags=["sequencing", "ordered"],
)

# --- Case 2: unordered — read_file and think in any order -------------------

_READ_FIRST_TURN = Message(
    role=Role.ASSISTANT,
    content="",
    tool_calls=[ToolCall(id="tc-1", name="read_file", arguments={"path": "config.txt"})],
)
_THINK_SECOND_TURN = Message(
    role=Role.ASSISTANT,
    content="",
    tool_calls=[ToolCall(id="tc-2", name="think", arguments={"thought": "Parsing config."})],
)
_FINAL_TURN_2 = Message(
    role=Role.ASSISTANT,
    content="Config loaded and processed.",
    tool_calls=[],
)

CASE_UNORDERED = EvalCase(
    id="read_and_think_unordered",
    task="Read config.txt and think about its contents.",
    scripted_responses=[
        (_READ_FIRST_TURN, _USAGE),
        (_THINK_SECOND_TURN, _USAGE),
        (_FINAL_TURN_2, _USAGE),
    ],
    tool_outputs={
        "read_file": "debug=true\nport=8080",
        "think": "ok",
    },
    expected_tool_calls=[
        ExpectedToolCall(name="think"),
        ExpectedToolCall(name="read_file", arguments={"path": "config.txt"}),
    ],
    tool_call_order="unordered",
    output_assertions=[ScorerSpec(kind="not_empty")],
    tags=["sequencing", "unordered"],
)


class TestToolSequencing:
    async def test_ordered_sequence(self, tmp_path):  # type: ignore[no-untyped-def]
        result: EvalResult = await run_eval(CASE_ORDERED, tmp_path)
        assert result.passed, "\n".join(result.failures)

    async def test_unordered_sequence(self, tmp_path):  # type: ignore[no-untyped-def]
        result: EvalResult = await run_eval(CASE_UNORDERED, tmp_path)
        assert result.passed, "\n".join(result.failures)

    async def test_ordered_failure_detected(self, tmp_path):  # type: ignore[no-untyped-def]
        """Regression: if think and read_file are swapped, ordered check fails."""
        # Build a case that expects think BEFORE read_file, but scripts read BEFORE think
        swapped_case = EC(
            id="sequencing_regression",
            task="Read then think.",
            scripted_responses=[
                (_READ_FIRST_TURN, _USAGE),
                (_THINK_SECOND_TURN, _USAGE),
                (_FINAL_TURN_2, _USAGE),
            ],
            tool_outputs={"read_file": "data", "think": "ok"},
            expected_tool_calls=[
                ExpectedToolCall(name="think"),  # expects think first
                ExpectedToolCall(name="read_file"),  # then read_file
            ],
            tool_call_order="ordered",
            output_assertions=[],
        )
        result: EvalResult = await run_eval(swapped_case, tmp_path)
        # actual order is read_file then think, so "think → read_file" is NOT a subsequence
        assert not result.passed
        assert any("sequence" in f.lower() for f in result.failures)
