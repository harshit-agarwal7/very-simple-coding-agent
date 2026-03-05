"""Eval: agent reads a file and produces a summary."""

from agent.models import Message, Role, ToolCall, Usage
from evals.framework import EvalCase, EvalResult, ExpectedToolCall, ScorerSpec, run_eval

_READ_TURN = Message(
    role=Role.ASSISTANT,
    content="",
    tool_calls=[ToolCall(id="tc-1", name="read_file", arguments={"path": "README.md"})],
)
_FINAL_TURN = Message(
    role=Role.ASSISTANT,
    content="The README describes a ReAct-loop coding agent.",
    tool_calls=[],
)
_USAGE = Usage(input_tokens=50, output_tokens=20)

CASE = EvalCase(
    id="read_and_summarize_readme",
    task="Read README.md and give a one-sentence summary.",
    scripted_responses=[(_READ_TURN, _USAGE), (_FINAL_TURN, _USAGE)],
    tool_outputs={"read_file": "# Very Simple Coding Agent\nA ReAct-loop agent."},
    expected_tool_calls=[ExpectedToolCall(name="read_file", arguments={"path": "README.md"})],
    output_assertions=[
        ScorerSpec(kind="not_empty"),
        ScorerSpec(kind="contains", value="ReAct"),
    ],
    tags=["mvp", "read_file"],
)


class TestReadAndSummarize:
    async def test_read_and_summarize(self, tmp_path):  # type: ignore[no-untyped-def]
        result: EvalResult = await run_eval(CASE, tmp_path)
        assert result.passed, "\n".join(result.failures)

    async def test_tool_call_recorded(self, tmp_path):  # type: ignore[no-untyped-def]
        result: EvalResult = await run_eval(CASE, tmp_path)
        names = [name for name, _ in result.actual_tool_calls]
        assert "read_file" in names

    async def test_final_output_not_empty(self, tmp_path):  # type: ignore[no-untyped-def]
        result: EvalResult = await run_eval(CASE, tmp_path)
        assert result.final_output.strip() != ""
