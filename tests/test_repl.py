"""Tests for src/agent/repl.py — plan mode and REPL commands."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from pytest_mock import MockerFixture

import agent.tools  # noqa: F401 — triggers tool registration  # isort: skip
from agent.memory import History
from agent.models import Config, Message, Role, ToolSafety
from agent.repl import _PLAN_CONFIRM_MSG, _do_clear, _do_plan_turn
from agent.tools.registry import get_safe_tool_names

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> Config:
    return Config(
        provider="openrouter",
        model="anthropic/claude-opus-4-6",
        api_key="sk-test",
        max_tokens=256,
        system_prompt="You are helpful.",
    )


def _history_with_assistant_plan(text: str) -> History:
    """Return a History that already has an assistant message (simulates planning phase)."""
    h = History()
    h.append(Message(role=Role.ASSISTANT, content=text))
    return h


# ---------------------------------------------------------------------------
# TestDoPlanTurn
# ---------------------------------------------------------------------------


class TestDoPlanTurn:
    async def test_confirmed_calls_run_turn_twice(self, mocker: MockerFixture) -> None:
        """When user confirms, run_turn is called for planning then execution."""
        history = History()
        provider = mocker.AsyncMock()
        executor = mocker.AsyncMock()
        config = _make_config()
        loop = asyncio.get_event_loop()

        mock_run_turn = mocker.patch("agent.repl.run_turn", new_callable=AsyncMock)
        # After the planning phase run_turn, inject a plan message into history.
        async def planning_side_effect(**kwargs: object) -> None:
            history.append(Message(role=Role.ASSISTANT, content="Here is the plan."))

        mock_run_turn.side_effect = [planning_side_effect, AsyncMock()]

        mocker.patch.object(loop, "run_in_executor", new=AsyncMock(return_value="y"))

        await _do_plan_turn("add a feature", history, provider, executor, config, loop)

        assert mock_run_turn.call_count == 2
        # Second call must use _PLAN_CONFIRM_MSG and no overrides.
        second_call = mock_run_turn.call_args_list[1]
        assert second_call.kwargs["user_input"] == _PLAN_CONFIRM_MSG
        assert second_call.kwargs.get("tools_override") is None
        assert second_call.kwargs.get("system_prompt_override") is None

    async def test_declined_calls_run_turn_once(self, mocker: MockerFixture) -> None:
        """When user declines, only the planning run_turn fires."""
        history = History()
        provider = mocker.AsyncMock()
        executor = mocker.AsyncMock()
        config = _make_config()
        loop = asyncio.get_event_loop()

        mock_run_turn = mocker.patch("agent.repl.run_turn", new_callable=AsyncMock)

        async def planning_side_effect(**kwargs: object) -> None:
            history.append(Message(role=Role.ASSISTANT, content="Plan text."))

        mock_run_turn.side_effect = planning_side_effect

        mocker.patch.object(loop, "run_in_executor", new=AsyncMock(return_value="n"))

        await _do_plan_turn("add a feature", history, provider, executor, config, loop)

        assert mock_run_turn.call_count == 1

    async def test_planning_phase_uses_safe_tools_only(self, mocker: MockerFixture) -> None:
        """tools_override passed to the first run_turn contains only SAFE tools."""
        history = History()
        provider = mocker.AsyncMock()
        executor = mocker.AsyncMock()
        config = _make_config()
        loop = asyncio.get_event_loop()

        captured: list[object] = []

        mock_run_turn = mocker.patch("agent.repl.run_turn", new_callable=AsyncMock)

        async def capture_side_effect(**kwargs: object) -> None:
            captured.append(kwargs.get("tools_override"))
            history.append(Message(role=Role.ASSISTANT, content="plan"))

        mock_run_turn.side_effect = capture_side_effect
        mocker.patch.object(loop, "run_in_executor", new=AsyncMock(return_value="n"))

        await _do_plan_turn("go", history, provider, executor, config, loop)

        tools_override = captured[0]
        assert tools_override is not None
        safe_names = get_safe_tool_names()
        for tool_def in tools_override:  # type: ignore[union-attr]
            assert tool_def.safety == ToolSafety.SAFE
            assert tool_def.name in safe_names

    async def test_planning_phase_uses_system_prompt_addendum(self, mocker: MockerFixture) -> None:
        """system_prompt_override includes the planning addendum."""
        history = History()
        provider = mocker.AsyncMock()
        executor = mocker.AsyncMock()
        config = _make_config()
        loop = asyncio.get_event_loop()

        captured: list[object] = []

        mock_run_turn = mocker.patch("agent.repl.run_turn", new_callable=AsyncMock)

        async def capture_side_effect(**kwargs: object) -> None:
            captured.append(kwargs.get("system_prompt_override"))
            history.append(Message(role=Role.ASSISTANT, content="plan"))

        mock_run_turn.side_effect = capture_side_effect
        mocker.patch.object(loop, "run_in_executor", new=AsyncMock(return_value="n"))

        await _do_plan_turn("go", history, provider, executor, config, loop)

        system_prompt_override = captured[0]
        assert isinstance(system_prompt_override, str)
        assert "[PLAN MODE]" in system_prompt_override
        assert config.system_prompt in system_prompt_override

    async def test_empty_plan_text_still_prompts_user(self, mocker: MockerFixture) -> None:
        """Even if no assistant text is produced, the confirm prompt still fires."""
        history = History()
        provider = mocker.AsyncMock()
        executor = mocker.AsyncMock()
        config = _make_config()
        loop = asyncio.get_event_loop()

        mock_run_turn = mocker.patch("agent.repl.run_turn", new_callable=AsyncMock)
        # Planning phase produces no assistant message.
        mock_run_turn.side_effect = AsyncMock(return_value=None)

        prompt_mock = mocker.patch.object(loop, "run_in_executor", new=AsyncMock(return_value="n"))

        await _do_plan_turn("go", history, provider, executor, config, loop)

        prompt_mock.assert_called_once()

    async def test_planning_modifies_main_history(self, mocker: MockerFixture) -> None:
        """Messages from the planning phase land in the passed history."""
        history = History()
        provider = mocker.AsyncMock()
        executor = mocker.AsyncMock()
        config = _make_config()
        loop = asyncio.get_event_loop()

        mock_run_turn = mocker.patch("agent.repl.run_turn", new_callable=AsyncMock)

        async def side_effect(**kwargs: object) -> None:
            # Simulate run_turn appending to the history it received.
            h = kwargs["history"]
            h.append(Message(role=Role.USER, content="go"))
            h.append(Message(role=Role.ASSISTANT, content="Here is my plan."))

        mock_run_turn.side_effect = side_effect
        mocker.patch.object(loop, "run_in_executor", new=AsyncMock(return_value="n"))

        await _do_plan_turn("go", history, provider, executor, config, loop)

        roles = [m.role for m in history.messages]
        assert Role.ASSISTANT in roles


# ---------------------------------------------------------------------------
# TestPlanModeCommand
# ---------------------------------------------------------------------------


class TestPlanModeCommand:
    async def test_plan_toggles_on_then_off(self, mocker: MockerFixture) -> None:
        """/plan twice toggles plan_mode ON then OFF."""
        config = _make_config()
        provider = mocker.AsyncMock()

        printed: list[str] = []

        def fake_print(msg: str = "", **kwargs: object) -> None:
            printed.append(str(msg))

        mocker.patch("agent.repl.console.print", side_effect=fake_print)
        mocker.patch("agent.repl.run_turn", new_callable=AsyncMock)
        mocker.patch("agent.repl._do_plan_turn", new_callable=AsyncMock)

        inputs = iter(["/plan", "/plan", "/quit"])
        mocker.patch("builtins.input", side_effect=inputs)

        from agent.repl import run_repl

        await run_repl(config, provider)

        on_msgs = [m for m in printed if "Plan mode: ON" in m]
        off_msgs = [m for m in printed if "Plan mode: OFF" in m]
        assert len(on_msgs) == 1
        assert len(off_msgs) == 1

    async def test_clear_does_not_call_run_turn(self, mocker: MockerFixture) -> None:
        """/clear then /quit never invokes run_turn."""
        config = _make_config()
        provider = mocker.AsyncMock()

        mocker.patch("agent.repl.console.print")
        mocker.patch("agent.repl.console.clear")
        mock_run_turn = mocker.patch("agent.repl.run_turn", new_callable=AsyncMock)
        mocker.patch("agent.repl._do_plan_turn", new_callable=AsyncMock)

        inputs = iter(["/clear", "/quit"])
        mocker.patch("builtins.input", side_effect=inputs)

        from agent.repl import run_repl

        await run_repl(config, provider)

        mock_run_turn.assert_not_called()

    async def test_empty_input_skipped(self, mocker: MockerFixture) -> None:
        """Empty input lines are ignored; run_turn is never called for them."""
        config = _make_config()
        provider = mocker.AsyncMock()

        mocker.patch("agent.repl.console.print")
        mock_run_turn = mocker.patch("agent.repl.run_turn", new_callable=AsyncMock)
        mocker.patch("agent.repl._do_plan_turn", new_callable=AsyncMock)

        inputs = iter(["", "/quit"])
        mocker.patch("builtins.input", side_effect=inputs)

        from agent.repl import run_repl

        await run_repl(config, provider)

        mock_run_turn.assert_not_called()


# ---------------------------------------------------------------------------
# TestClearCommand
# ---------------------------------------------------------------------------


class TestClearCommand:
    def test_clear_wipes_history_and_clears_screen(self, mocker: MockerFixture) -> None:
        """_do_clear empties history and calls console.clear."""
        history = History()
        history.append(Message(role=Role.USER, content="hello"))
        history.append(Message(role=Role.ASSISTANT, content="hi"))

        mock_clear = mocker.patch("agent.repl.console.clear")
        mock_print = mocker.patch("agent.repl.console.print")

        _do_clear(history)

        assert history.messages == []
        assert history.usage.total == 0
        mock_clear.assert_called_once()
        mock_print.assert_called_once()

    def test_clear_on_empty_history_is_safe(self, mocker: MockerFixture) -> None:
        """_do_clear on an already-empty history does not raise."""
        history = History()
        mocker.patch("agent.repl.console.clear")
        mocker.patch("agent.repl.console.print")

        _do_clear(history)  # should not raise

        assert history.messages == []

    async def test_clear_command_dispatches_in_loop(self, mocker: MockerFixture) -> None:
        """/clear in the REPL loop invokes _do_clear once."""
        config = _make_config()
        provider = mocker.AsyncMock()

        mocker.patch("agent.repl.console.print")
        mocker.patch("agent.repl.console.clear")
        mocker.patch("agent.repl.run_turn", new_callable=AsyncMock)
        mocker.patch("agent.repl._do_plan_turn", new_callable=AsyncMock)
        mock_do_clear = mocker.patch("agent.repl._do_clear")

        inputs = iter(["/clear", "/quit"])
        mocker.patch("builtins.input", side_effect=inputs)

        from agent.repl import run_repl

        await run_repl(config, provider)

        mock_do_clear.assert_called_once()

    async def test_clear_does_not_invoke_run_turn(self, mocker: MockerFixture) -> None:
        """/clear never triggers a model turn."""
        config = _make_config()
        provider = mocker.AsyncMock()

        mocker.patch("agent.repl.console.print")
        mocker.patch("agent.repl.console.clear")
        mock_run_turn = mocker.patch("agent.repl.run_turn", new_callable=AsyncMock)
        mocker.patch("agent.repl._do_plan_turn", new_callable=AsyncMock)

        inputs = iter(["/clear", "/quit"])
        mocker.patch("builtins.input", side_effect=inputs)

        from agent.repl import run_repl

        await run_repl(config, provider)

        mock_run_turn.assert_not_called()
