"""Ollama provider adapter (OpenAI-compatible local API with streaming)."""

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.theme import Theme

from agent.models import Message, Role, ToolCall, ToolDefinition, Usage
from agent.providers.base import ProviderAdapter

logger = logging.getLogger(__name__)

# Matches <tool_call>...</tool_call> blocks (Qwen-style native format).
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_text_tool_calls(text: str) -> list[ToolCall]:
    """Parse tool calls that Ollama leaked into content instead of tool_calls.

    Handles two formats emitted by models like ``qwen2.5-coder``:
    - Tagged:  ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``
    - Bare JSON: ``{"name": ..., "arguments": ...}``

    Args:
        text: Raw assistant content that may contain embedded tool calls.

    Returns:
        Parsed list of :class:`ToolCall` objects, or empty list if none found.
    """
    blobs: list[str] = _TOOL_CALL_TAG_RE.findall(text)
    if not blobs:
        # Try interpreting the whole content as a single bare JSON tool call.
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            blobs = [stripped]

    calls: list[ToolCall] = []
    for i, blob in enumerate(blobs):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            logger.warning("ollama fallback: failed to parse tool_call blob: %s", blob)
            continue
        name = obj.get("name") or obj.get("function")
        arguments = obj.get("arguments") or obj.get("parameters") or {}
        if not name:
            logger.warning("ollama fallback: tool_call blob has no name: %s", obj)
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append(ToolCall(id=f"call_{i}", name=name, arguments=arguments))
    return calls


class OllamaAdapter(ProviderAdapter):
    """Provider adapter for Ollama's OpenAI-compatible local API.

    Args:
        base_url: Base URL for the Ollama server (e.g. ``"http://localhost:11434"``).
        console: Optional Rich console to use for rendering output.
    """

    def __init__(self, base_url: str, console: Console | None = None) -> None:
        self._client = AsyncOpenAI(
            api_key="ollama",
            base_url=base_url + "/v1",
        )
        _warm_theme = Theme({"markdown.code": "bold #e8a87c"})
        self._console = console if console is not None else Console(theme=_warm_theme)

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
        """Stream a completion from Ollama, rendering markdown live.

        Args:
            messages: Full conversation history.
            tools: Available tools for the model.
            model: Ollama model name (e.g. ``"llama3.2"``).
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
            kwargs["tool_choice"] = "auto"

        # Accumulators for assembling the streamed response.
        text_parts: list[str] = []
        # tool_call_id → {id, name, args_fragments}
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage = Usage()

        accumulated_text = ""
        with Live(
            Spinner("dots", text=" Thinking…"),
            console=self._console,
            refresh_per_second=15,
            auto_refresh=True,
        ) as live:
            raw_stream: AsyncStream[ChatCompletionChunk] = (
                await self._client.chat.completions.create(**kwargs)
            )

            async for chunk in raw_stream:
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
                    live.update(Markdown(accumulated_text, code_theme="monokai"), refresh=True)

                # Accumulate tool call fragments.
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": f"call_{idx}", "name": "", "args": ""}
                        if tc_delta.id:
                            tool_calls_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_calls_acc[idx]["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_calls_acc[idx]["args"] += tc_delta.function.arguments

            # Clear the spinner if no text was generated (tool-call-only response).
            if not accumulated_text:
                live.update("")

        # Assemble tool calls.
        assembled_tool_calls: list[ToolCall] = []
        content_text = "".join(text_parts)

        if not tool_calls_acc and content_text:
            # Ollama's OpenAI-compat layer may not translate the model's native
            # <tool_call> format into structured tool_calls.  Detect and parse it.
            fallback = _parse_text_tool_calls(content_text)
            if fallback:
                assembled_tool_calls = fallback
                content_text = ""  # suppress raw JSON from the assistant message
        else:
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
            content=content_text,
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
