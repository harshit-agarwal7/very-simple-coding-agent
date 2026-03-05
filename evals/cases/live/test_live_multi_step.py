"""Live eval: agent reads a config file, thinks, then summarizes."""

from __future__ import annotations

import pytest

from agent.providers.openrouter import OpenRouterAdapter
from evals.framework import EvalCase, EvalResult, ExpectedToolCall, ScorerSpec, run_eval


@pytest.mark.live
async def test_live_multi_step(
    tmp_path: pytest.TempPathFactory, live_provider: OpenRouterAdapter
) -> None:
    config_file = tmp_path / "config.txt"
    config_file.write_text("port=8080\ndebug=false")

    case = EvalCase(
        id="live_multi_step_config",
        task=(
            f"Read {config_file}, think about what the port setting means, "
            "then summarize in one sentence."
        ),
        expected_tool_calls=[
            ExpectedToolCall(name="read_file"),
            ExpectedToolCall(name="think"),
        ],
        tool_call_order="ordered",
        output_assertions=[
            ScorerSpec(kind="not_empty"),
            ScorerSpec(kind="regex", value=r"808[0-9]|port", case_sensitive=False),
        ],
        tool_outputs={"execute_command": "stubbed — not allowed in evals"},
        tags=["live", "multi_step"],
    )

    result: EvalResult = await run_eval(
        case,
        tmp_path=tmp_path,
        provider=live_provider,
    )
    assert result.passed, "\n".join(result.failures)
