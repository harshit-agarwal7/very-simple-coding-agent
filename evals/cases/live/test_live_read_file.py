"""Live eval: agent reads a file and produces a one-sentence summary."""

from __future__ import annotations

import pytest

from agent.providers.openrouter import OpenRouterAdapter
from evals.framework import EvalCase, EvalResult, ExpectedToolCall, ScorerSpec, run_eval


@pytest.mark.live
async def test_live_read_file(
    tmp_path: pytest.TempPathFactory, live_provider: OpenRouterAdapter
) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("The project meeting is on Friday.")

    case = EvalCase(
        id="live_read_file_summary",
        task=f"Read the file {notes} and give a one-sentence summary.",
        expected_tool_calls=[ExpectedToolCall(name="read_file")],
        output_assertions=[
            ScorerSpec(kind="not_empty"),
            ScorerSpec(kind="contains", value="Friday", case_sensitive=False),
        ],
        tool_outputs={"execute_command": "stubbed — not allowed in evals"},
        tags=["live", "read_file"],
    )

    result: EvalResult = await run_eval(
        case,
        tmp_path=tmp_path,
        provider=live_provider,
    )
    assert result.passed, "\n".join(result.failures)
