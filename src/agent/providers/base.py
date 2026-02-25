"""Abstract base class for LLM provider adapters."""

from abc import ABC, abstractmethod

from agent.models import Message, ToolDefinition, Usage


class ProviderAdapter(ABC):
    """Interface that every LLM provider adapter must implement.

    Concrete subclasses wrap a specific API (e.g. OpenRouter) and handle
    message formatting, streaming, and summarisation.
    """

    @abstractmethod
    def format_messages(self, messages: list[Message]) -> list[dict[str, object]]:
        """Convert internal :class:`~agent.models.Message` objects to the API wire format.

        Args:
            messages: Conversation history in internal representation.

        Returns:
            List of dicts ready to send to the API.
        """

    @abstractmethod
    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, object]]:
        """Convert :class:`~agent.models.ToolDefinition` objects to the API tool schema.

        Args:
            tools: Tool definitions to convert.

        Returns:
            List of dicts in the provider's tool schema format.
        """

    @abstractmethod
    async def stream_completion(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        model: str,
        max_tokens: int,
        system_prompt: str = "",
    ) -> tuple[Message, Usage]:
        """Stream a chat completion, printing tokens to stdout as they arrive.

        Text chunks are written to ``sys.stdout`` in real-time.
        Tool-call JSON fragments are accumulated silently.

        Args:
            messages: Full conversation history.
            tools: Available tools exposed to the model.
            model: Model identifier string.
            max_tokens: Maximum output tokens.
            system_prompt: Optional system prompt prepended to the request.

        Returns:
            Tuple of ``(assembled_message, usage)`` once streaming completes.
        """

    @abstractmethod
    async def summarize(self, messages: list[Message], model: str) -> str:
        """Produce a concise prose summary of a conversation.

        Used by :meth:`~agent.memory.History.compact` to compress history.

        Args:
            messages: The messages to summarise.
            model: Model identifier to use for summarisation.

        Returns:
            Summary string.
        """
