"""ReAct agent loop: Reason → Act → Observe → repeat."""

import logging

from agent.memory import History
from agent.models import Config, Message, Role
from agent.providers.base import ProviderAdapter
from agent.tools.executor import ToolExecutor
from agent.tools.registry import get_tool_definitions

logger = logging.getLogger(__name__)

# Maximum number of tool-call iterations per user turn (safety guard).
_MAX_ITERATIONS = 20


async def run_turn(
    user_input: str,
    history: History,
    provider: ProviderAdapter,
    tool_executor: ToolExecutor,
    config: Config,
) -> None:
    """Execute one full ReAct turn for a user message.

    Appends the user message to *history*, then loops:

    1. **Reason** — stream a completion; print tokens to stdout.
    2. **Act** — if the model requested tool calls, execute each one.
    3. **Observe** — append tool results to *history* so the model can reason again.

    The loop ends when the model returns a response with no tool calls.

    Args:
        user_input: Raw text from the user.
        history: Mutable conversation history (modified in place).
        provider: LLM provider adapter for streaming completions.
        tool_executor: Executor that dispatches tool calls (with approval gate).
        config: Agent runtime configuration.
    """
    history.append(Message(role=Role.USER, content=user_input))
    tools = get_tool_definitions()

    for iteration in range(_MAX_ITERATIONS):
        logger.debug("loop iteration %d", iteration)

        # REASON — stream the next completion.
        response_message, usage = await provider.stream_completion(
            messages=history.messages,
            tools=tools,
            model=config.model,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
        )
        history.record_usage(usage)
        history.append(response_message)

        # If no tool calls, the agent is done.
        if not response_message.tool_calls:
            logger.debug("No tool calls — turn complete after %d iteration(s)", iteration + 1)
            break

        # ACT + OBSERVE — execute each tool call and append results.
        for tool_call in response_message.tool_calls:
            logger.debug("Executing tool: %s", tool_call.name)
            result = await tool_executor.execute(tool_call)
            history.append(
                Message(
                    role=Role.TOOL,
                    content=result.output,
                    tool_call_id=result.tool_call_id,
                )
            )
    else:
        logger.warning("Reached max iterations (%d) — stopping turn", _MAX_ITERATIONS)
