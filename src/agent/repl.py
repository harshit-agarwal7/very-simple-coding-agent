"""Async interactive REPL for the coding agent."""

from __future__ import annotations

import asyncio
import logging

from rich.console import Console

from agent.loop import run_turn
from agent.memory import History
from agent.models import Config
from agent.providers.base import ProviderAdapter
from agent.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)

console = Console()

_WELCOME = """\
[bold green]Very Simple Coding Agent[/bold green]  [dim]powered by {model}[/dim]
Type your message and press Enter. Special commands:
  [bold]/compact[/bold]  — summarise and compress conversation history
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

        if history.is_over_limit:
            console.print(
                "[yellow]History is getting long. Consider running /compact to "
                "summarise the conversation.[/yellow]"
            )

        try:
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
