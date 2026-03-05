"""Eval: agent writes a file with the correct arguments."""

from agent.models import Message, Role, ToolCall, Usage
from evals.framework import EvalCase, EvalResult, ExpectedToolCall, ScorerSpec, run_eval

_WRITE_TURN = Message(
    role=Role.ASSISTANT,
    content="",
    tool_calls=[
        ToolCall(
            id="tc-1",
            name="write_file",
            arguments={"path": "hello.txt", "content": "Hello, world!"},
        )
    ],
)
_FINAL_TURN = Message(
    role=Role.ASSISTANT,
    content="I have written 'Hello, world!' to hello.txt.",
    tool_calls=[],
)
_USAGE = Usage(input_tokens=40, output_tokens=15)

CASE = EvalCase(
    id="write_hello_file",
    task="Write 'Hello, world!' to a file named hello.txt.",
    scripted_responses=[(_WRITE_TURN, _USAGE), (_FINAL_TURN, _USAGE)],
    tool_outputs={"write_file": "File written successfully."},
    expected_tool_calls=[
        ExpectedToolCall(
            name="write_file",
            arguments={"path": "hello.txt", "content": "Hello, world!"},
            argument_match="exact",
        )
    ],
    output_assertions=[
        ScorerSpec(kind="not_empty"),
        ScorerSpec(kind="contains", value="hello.txt"),
    ],
    tags=["mvp", "write_file"],
)


class TestWriteFile:
    async def test_write_file(self, tmp_path):  # type: ignore[no-untyped-def]
        result: EvalResult = await run_eval(CASE, tmp_path)
        assert result.passed, "\n".join(result.failures)

    async def test_exact_args_checked(self, tmp_path):  # type: ignore[no-untyped-def]
        """Regression: wrong path would cause tool call mismatch."""
        from evals.framework.schema import EvalCase as EC

        wrong_case = EC(
            id="write_hello_file_wrong_path",
            task=CASE.task,
            scripted_responses=[(_WRITE_TURN, _USAGE), (_FINAL_TURN, _USAGE)],
            tool_outputs={"write_file": "File written successfully."},
            expected_tool_calls=[
                ExpectedToolCall(
                    name="write_file",
                    arguments={"path": "wrong.txt"},
                    argument_match="subset",
                )
            ],
            output_assertions=[],
        )
        result: EvalResult = await run_eval(wrong_case, tmp_path)
        assert not result.passed
        assert any("write_file" in f for f in result.failures)
