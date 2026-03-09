"""Async interactive REPL for the coding agent."""

import asyncio
import logging

from rich.console import Console
from rich.theme import Theme

from agent.loop import run_turn
from agent.memory import History
from agent.models import Config, Role
from agent.providers.base import ProviderAdapter
from agent.tools.executor import ToolExecutor
from agent.tools.registry import get_safe_tool_names, get_tool_definitions

logger = logging.getLogger(__name__)

_WARM_THEME = Theme({"markdown.code": "bold #e8a87c"})
console = Console(theme=_WARM_THEME)

_PLAN_CONFIRM_MSG = "Looks good, please go ahead and implement."

_WELCOME = """\
[bold green]Very Simple Coding Agent[/bold green]  [dim]powered by {model}[/dim]
Type your message and press Enter. Special commands:
  [bold]/compact[/bold]  — summarise and compress conversation history
  [bold]/clear[/bold]    — clear history and terminal display
  [bold]/plan[/bold]     — toggle plan mode (explore first, confirm before executing)
  [bold]/quit[/bold]     — exit the agent
"""

_PROMPT = "> "


async def run_repl(
    config: Config,
    provider: ProviderAdapter,
) -> None:
    """Run the interactive REPL loop until the user quits.

    Reads user input off the event loop (via ``run_in_executor``) to keep the
    async event loop free for streaming completions.

    Args:
        config: Agent runtime configuration.
        provider: LLM provider adapter.
    """
    history = History(max_history_tokens=config.max_history_tokens)
    executor = ToolExecutor()
    loop = asyncio.get_event_loop()
    plan_mode: bool = False

    console.print(_WELCOME.format(model=config.model))

    while True:
        # Read input off the event loop.
        try:
            user_input: str = await loop.run_in_executor(None, input, _PROMPT)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        user_input = user_input.strip()

        if not user_input:
            continue

        if user_input.lower() in {"/quit", "/exit", "quit", "exit"}:
            console.print("[dim]Bye![/dim]")
            break

        if user_input.lower() == "/compact":
            await _do_compact(history, provider, config)
            continue

        if user_input.lower() == "/clear":
            _do_clear(history)
            continue

        if user_input.lower() == "/plan":
            plan_mode = not plan_mode
            console.print(f"[cyan]Plan mode: {'ON' if plan_mode else 'OFF'}[/cyan]")
            continue

        if history.is_over_limit:
            console.print(
                "[yellow]History is getting long. Consider running /compact to "
                "summarise the conversation.[/yellow]"
            )

        try:
            if plan_mode:
                await _do_plan_turn(user_input, history, provider, executor, config, loop)
            else:
                await run_turn(
                    user_input=user_input,
                    history=history,
                    provider=provider,
                    tool_executor=executor,
                    config=config,
                )
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")
        except Exception as exc:
            logger.exception("Unhandled error during turn")
            console.print(f"[red]Error: {exc}[/red]")

        # Print usage summary after each turn.
        u = history.usage
        console.print(
            f"[dim]  tokens: {u.input_tokens} in / {u.output_tokens} out "
            f"(total {u.total})[/dim]"
        )


async def _do_plan_turn(
    user_input: str,
    history: History,
    provider: ProviderAdapter,
    executor: ToolExecutor,
    config: Config,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Run planning phase then, if confirmed, execution phase.

    Phase 1: agent explores with safe tools and produces a written plan (appended
    to main history). Phase 2: if user confirms, appends a confirmation user
    message and runs a normal turn with all tools.

    Args:
        user_input: Raw text from the user.
        history: Mutable conversation history shared across both phases.
        provider: LLM provider adapter.
        executor: Tool executor.
        config: Agent runtime configuration.
        loop: Running event loop for non-blocking input.
    """
    safe_tools = [d for d in get_tool_definitions() if d.name in get_safe_tool_names()]
    safe_names = ", ".join(sorted(get_safe_tool_names()))
    planning_prompt = (
        config.system_prompt
        + f"\n\n[PLAN MODE] Use only read-only tools ({safe_names}) to explore the task. "
        "Then write your complete implementation plan as your final response. "
        "Do NOT write files or execute commands."
    )

    # Phase 1: planning (modifies main history).
    await run_turn(
        user_input=user_input,
        history=history,
        provider=provider,
        tool_executor=executor,
        config=config,
        tools_override=safe_tools,
        system_prompt_override=planning_prompt,
    )

    # Extract plan text from last assistant message.
    plan_text = next(
        (m.content for m in reversed(history.messages) if m.role == Role.ASSISTANT and m.content),
        "",
    )

    console.print("\n[bold cyan]" + "─" * 60 + "[/bold cyan]")
    console.print("[bold cyan]PLAN[/bold cyan]")
    console.print("[bold cyan]" + "─" * 60 + "[/bold cyan]")
    console.print(plan_text or "[dim](no plan text produced)[/dim]")
    console.print("[bold cyan]" + "─" * 60 + "[/bold cyan]")

    answer: str = await loop.run_in_executor(None, input, "Proceed? [y/N] ")
    if answer.strip().lower() != "y":
        console.print("[dim]Plan cancelled.[/dim]")
        return

    # Phase 2: execution with plan context already in history.
    await run_turn(
        user_input=_PLAN_CONFIRM_MSG,
        history=history,
        provider=provider,
        tool_executor=executor,
        config=config,
    )


def _do_clear(history: History) -> None:
    """Clear conversation history and the terminal display.

    Args:
        history: The history to wipe.
    """
    history.clear()
    console.clear()
    console.print("[dim]History and display cleared.[/dim]")


async def _do_compact(
    history: History,
    provider: ProviderAdapter,
    config: Config,
) -> None:
    """Summarise and replace the current history.

    Args:
        history: The history to compact.
        provider: Provider to generate the summary.
        config: Runtime config (for the model string).
    """
    console.print("[dim]Compacting history...[/dim]")
    try:
        await history.compact(provider, config.model)
        console.print("[green]History compacted.[/green]")
    except Exception as exc:
        logger.exception("Failed to compact history")
        console.print(f"[red]Compaction failed: {exc}[/red]")
