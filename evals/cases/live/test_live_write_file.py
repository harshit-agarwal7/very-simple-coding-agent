"""Live eval: agent creates a file with specified content."""

from __future__ import annotations

import pytest

from agent.providers.openrouter import OpenRouterAdapter
from evals.framework import EvalCase, EvalResult, ExpectedToolCall, ScorerSpec, run_eval


@pytest.mark.live
async def test_live_write_file(
    tmp_path: pytest.TempPathFactory, live_provider: OpenRouterAdapter
) -> None:
    result_path = tmp_path / "result.txt"

    case = EvalCase(
        id="live_write_file",
        task=(
            f"Create a file called result.txt in the directory {tmp_path}"
            " containing the text 'done'."
        ),
        expected_tool_calls=[ExpectedToolCall(name="write_file")],
        output_assertions=[ScorerSpec(kind="not_empty")],
        tool_outputs={"execute_command": "stubbed — not allowed in evals"},
        tags=["live", "write_file"],
    )

    result: EvalResult = await run_eval(
        case,
        tmp_path=tmp_path,
        provider=live_provider,
    )
    assert result.passed, "\n".join(result.failures)

    assert result_path.exists(), "Expected result.txt to exist on disk"
    assert "done" in result_path.read_text().lower(), "Expected 'done' in result.txt"
