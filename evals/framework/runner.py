"""run_eval: wires ScriptedProvider + StubToolExecutor and executes an EvalCase."""

from __future__ import annotations

from pathlib import Path

from agent.loop import run_turn
from agent.memory import History
from agent.models import Config, Role
from agent.providers.base import ProviderAdapter
from evals.framework.executor import StubToolExecutor
from evals.framework.provider import ScriptedProvider
from evals.framework.schema import EvalCase, EvalResult
from evals.framework.scorers import score_output, score_tool_calls


async def run_eval(
    case: EvalCase,
    tmp_path: Path | None = None,
    provider: ProviderAdapter | None = None,
) -> EvalResult:
    """Execute an :class:`EvalCase` and return a scored :class:`EvalResult`.

    Steps:
    1. Write any ``setup_files`` into *tmp_path*.
    2. Wire a provider (supplied or :class:`ScriptedProvider`),
       :class:`StubToolExecutor`, and a minimal :class:`~agent.models.Config`.
    3. Call :func:`~agent.loop.run_turn` with the case task.
    4. Extract the final assistant message (last assistant turn with no tool calls).
    5. Score output and tool calls; return :class:`EvalResult`.

    Args:
        case: The eval scenario to run.
        tmp_path: Optional temporary directory for ``setup_files``.
        provider: Optional live provider. When ``None``, a
            :class:`ScriptedProvider` is built from ``case.scripted_responses``
            (which must be non-empty in that case).

    Returns:
        A fully populated :class:`EvalResult`.

    Raises:
        ValueError: If *provider* is ``None`` and ``case.scripted_responses``
            is empty.
    """
    if case.setup_files and tmp_path is not None:
        for rel_path, content in case.setup_files.items():
            file_path = tmp_path / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

    if provider is None:
        if not case.scripted_responses:
            raise ValueError(
                f"EvalCase '{case.id}': scripted_responses required when no provider is supplied"
            )
        provider = ScriptedProvider(case.scripted_responses)
    executor = StubToolExecutor(tool_outputs=case.tool_outputs)
    history = History()
    config = Config(provider="eval", model="eval", api_key="eval-no-key")

    await run_turn(case.task, history, provider, executor, config)

    # Find the last assistant message that has no tool calls (the final answer).
    final_output = ""
    for msg in reversed(history.messages):
        if msg.role == Role.ASSISTANT and not msg.tool_calls:
            final_output = msg.content
            break

    actual_tool_calls = [(tc.name, tc.arguments) for tc in executor.calls]

    failures: list[str] = []
    failures.extend(score_output(final_output, case.output_assertions))
    failures.extend(
        score_tool_calls(actual_tool_calls, case.expected_tool_calls, case.tool_call_order)
    )

    return EvalResult(
        case_id=case.id,
        passed=len(failures) == 0,
        final_output=final_output,
        actual_tool_calls=actual_tool_calls,
        failures=failures,
        history=history,
    )
