"""ScriptedProvider: a deterministic ProviderAdapter for offline evals."""

from __future__ import annotations

from agent.models import Message, ToolDefinition, Usage
from agent.providers.base import ProviderAdapter


class ScriptedProvider(ProviderAdapter):
    """Returns pre-scripted responses in order instead of calling a real LLM.

    Args:
        responses: Ordered list of ``(Message, Usage)`` tuples. Each call to
            :meth:`stream_completion` consumes the next entry.

    Raises:
        IndexError: If :meth:`stream_completion` is called more times than
            there are scripted responses (indicates a test design error).
    """

    def __init__(self, responses: list[tuple[Message, Usage]]) -> None:
        self._responses: list[tuple[Message, Usage]] = list(responses)
        self.calls: list[list[Message]] = []

    def format_messages(self, messages: list[Message]) -> list[dict[str, object]]:
        """No-op: ScriptedProvider does not call a real API.

        Args:
            messages: Ignored.

        Returns:
            Empty list.
        """
        return []

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, object]]:
        """No-op: ScriptedProvider does not call a real API.

        Args:
            tools: Ignored.

        Returns:
            Empty list.
        """
        return []

    async def stream_completion(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        model: str,
        max_tokens: int,
        system_prompt: str = "",
    ) -> tuple[Message, Usage]:
        """Return the next scripted response without network I/O.

        Args:
            messages: Current conversation history (recorded for inspection).
            tools: Available tools (ignored).
            model: Model identifier (ignored).
            max_tokens: Max tokens (ignored).
            system_prompt: System prompt (ignored).

        Returns:
            The next ``(Message, Usage)`` pair from the scripted list.

        Raises:
            IndexError: When all scripted responses have been consumed.
        """
        if not self._responses:
            raise IndexError(
                "ScriptedProvider exhausted — add more scripted_responses to EvalCase"
            )
        self.calls.append(list(messages))
        return self._responses.pop(0)

    async def summarize(self, messages: list[Message], model: str) -> str:
        """Return a stub summary string.

        Args:
            messages: Messages to summarise (ignored).
            model: Model identifier (ignored).

        Returns:
            Placeholder summary string.
        """
        return "[eval-summary]"
