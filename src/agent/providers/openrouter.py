"""OpenRouter provider adapter (OpenAI-compatible API with streaming)."""

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner

from agent.models import Message, Role, ToolCall, ToolDefinition, Usage
from agent.providers.base import ProviderAdapter

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_EXTRA_HEADERS = {
    "HTTP-Referer": "https://github.com/very-simple-coding-agent",
    "X-Title": "Very Simple Coding Agent",
}


class OpenRouterAdapter(ProviderAdapter):
    """Provider adapter for OpenRouter's OpenAI-compatible API.

    Args:
        api_key: OpenRouter API key (``OPENROUTER_API_KEY``).
    """

    def __init__(self, api_key: str, console: Console | None = None) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=_OPENROUTER_BASE_URL,
            default_headers=_EXTRA_HEADERS,
        )
        self._console = console if console is not None else Console()

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    def format_messages(self, messages: list[Message]) -> list[dict[str, object]]:
        """Convert internal messages to OpenAI-compatible wire format.

        Args:
            messages: Conversation history.

        Returns:
            List of message dicts for the OpenAI ``messages`` parameter.
        """
        result: list[dict[str, object]] = []
        for msg in messages:
            if msg.role == Role.TOOL:
                result.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id or "",
                    }
                )
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                wire_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                entry: dict[str, object] = {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": wire_tool_calls,
                }
                result.append(entry)
            else:
                result.append({"role": msg.role.value, "content": msg.content})
        return result

    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, object]]:
        """Convert tool definitions to OpenAI function-calling schema.

        Args:
            tools: Tool definitions to convert.

        Returns:
            List of tool dicts in ``{"type": "function", "function": {...}}`` format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------

    async def stream_completion(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        model: str,
        max_tokens: int,
        system_prompt: str = "",
    ) -> tuple[Message, Usage]:
        """Stream a completion from OpenRouter, rendering markdown live.

        Args:
            messages: Full conversation history.
            tools: Available tools for the model.
            model: OpenRouter model string (e.g. ``"anthropic/claude-opus-4-6"``).
            max_tokens: Maximum completion tokens.
            system_prompt: Optional system prompt.

        Returns:
            Tuple of ``(assembled_message, usage)``.
        """
        wire_messages: list[dict[str, object]] = []
        if system_prompt:
            wire_messages.append({"role": "system", "content": system_prompt})
        wire_messages.extend(self.format_messages(messages))

        wire_tools = self.format_tools(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": wire_messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if wire_tools:
            kwargs["tools"] = wire_tools

        # Accumulators for assembling the streamed response.
        text_parts: list[str] = []
        # tool_call_id → {id, name, args_fragments}
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage = Usage()

        from openai import AsyncStream

        accumulated_text = ""
        with Live(
            Spinner("dots", text=" Thinking…"),
            console=self._console,
            refresh_per_second=15,
            auto_refresh=True,
        ) as live:
            raw_stream = await self._client.chat.completions.create(**kwargs)
            stream: AsyncStream[ChatCompletionChunk] = raw_stream

            async for chunk in stream:
                # Capture usage from the final chunk (stream_options).
                if chunk.usage:
                    usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Accumulate and render text tokens as markdown.
                if delta.content:
                    accumulated_text += delta.content
                    text_parts.append(delta.content)
                    live.update(Markdown(accumulated_text, code_theme='github-dark'), refresh=True)

                # Accumulate tool call fragments.
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "args": ""}
                        if tc_delta.id:
                            tool_calls_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_calls_acc[idx]["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_calls_acc[idx]["args"] += tc_delta.function.arguments

        # Assemble tool calls.
        assembled_tool_calls: list[ToolCall] = []
        for idx in sorted(tool_calls_acc):
            raw = tool_calls_acc[idx]
            try:
                arguments = json.loads(raw["args"]) if raw["args"] else {}
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool call arguments: %s", raw["args"])
                arguments = {}
            assembled_tool_calls.append(
                ToolCall(id=raw["id"], name=raw["name"], arguments=arguments)
            )

        assembled = Message(
            role=Role.ASSISTANT,
            content="".join(text_parts),
            tool_calls=assembled_tool_calls,
        )
        return assembled, usage

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    async def summarize(self, messages: list[Message], model: str) -> str:
        """Summarise the conversation using a non-streaming completion.

        Args:
            messages: Messages to summarise.
            model: Model to use for summarisation.

        Returns:
            Prose summary string.
        """
        summary_prompt = (
            "Please provide a concise summary of the following conversation. "
            "Capture the main goals, decisions made, files examined, and any "
            "important findings or conclusions."
        )
        wire_messages: list[dict[str, object]] = [
            {"role": "system", "content": summary_prompt},
            *self.format_messages(messages),
        ]
        from openai.types.chat import ChatCompletion

        raw = await self._client.chat.completions.create(
            model=model,
            messages=wire_messages,  # type: ignore[arg-type]
            max_tokens=1024,
        )
        completion: ChatCompletion = raw
        content = completion.choices[0].message.content or ""
        logger.debug("summarize: %d chars", len(content))
        return content
