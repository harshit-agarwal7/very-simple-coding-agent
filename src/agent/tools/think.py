"""Think pseudo-tool: lets the agent reason without producing visible output."""

from __future__ import annotations

import logging

from agent.models import ToolDefinition, ToolSafety
from agent.tools.registry import register_tool

logger = logging.getLogger(__name__)

_DEFINITION = ToolDefinition(
    name="think",
    description=(
        "Use this tool to reason step-by-step before acting. The thought is recorded "
        "in the conversation but produces no external side effects. Useful for planning "
        "or reflecting on previous observations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Your internal reasoning or plan.",
            }
        },
        "required": ["thought"],
    },
    safety=ToolSafety.SAFE,
)


async def think(thought: str) -> str:
    """Record a reasoning step and return it unchanged.

    Args:
        thought: The agent's internal reasoning text.

    Returns:
        The same *thought* string (no side effects).
    """
    logger.debug("think: %s", thought[:80])
    return thought


register_tool(_DEFINITION, think)
