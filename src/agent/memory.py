"""Conversation history management with token tracking and compaction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent.models import Message, Role, Usage

if TYPE_CHECKING:
    from agent.providers.base import ProviderAdapter

logger = logging.getLogger(__name__)


class History:
    """Manages the conversation message history and cumulative token usage.

    Args:
        max_history_tokens: Soft cap; once exceeded, callers should invoke
            :meth:`compact` or warn the user.
    """

    def __init__(self, max_history_tokens: int = 80_000) -> None:
        self._messages: list[Message] = []
        self._usage: Usage = Usage()
        self.max_history_tokens = max_history_tokens

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def append(self, message: Message) -> None:
        """Append a message to the history.

        Args:
            message: The message to append.
        """
        self._messages.append(message)

    def record_usage(self, usage: Usage) -> None:
        """Accumulate token usage from a completion.

        Args:
            usage: The usage stats from one model call.
        """
        self._usage.input_tokens += usage.input_tokens
        self._usage.output_tokens += usage.output_tokens

    @property
    def messages(self) -> list[Message]:
        """Read-only view of the current message list."""
        return list(self._messages)

    @property
    def usage(self) -> Usage:
        """Cumulative token usage across all completions."""
        return Usage(
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
        )

    @property
    def is_over_limit(self) -> bool:
        """True if cumulative usage has exceeded *max_history_tokens*."""
        return self._usage.total > self.max_history_tokens

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def compact(self, provider: ProviderAdapter, model: str) -> None:
        """Replace verbose history with a compact summary.

        Calls ``provider.summarize`` to get a short prose summary of the
        conversation so far, then replaces the entire history with a single
        assistant message containing that summary.  Token counters are reset.

        Args:
            provider: The LLM provider to generate the summary.
            model: Model string to use for summarisation.
        """
        if not self._messages:
            logger.debug("compact() called on empty history — nothing to do")
            return

        logger.info("Compacting history (%d messages)", len(self._messages))
        summary = await provider.summarize(self._messages, model)

        self._messages = [
            Message(
                role=Role.ASSISTANT,
                content=f"[Conversation summary]\n{summary}",
            )
        ]
        self._usage = Usage()
        logger.info("History compacted to summary")

    def clear(self) -> None:
        """Wipe the full history and reset usage counters."""
        self._messages = []
        self._usage = Usage()
