"""Shared fixtures and skip guard for live LLM evals."""

from __future__ import annotations

import os

import pytest

from agent.config import load_config
from agent.providers.openrouter import OpenRouterAdapter


def pytest_collection_modifyitems(items: list[pytest.Item], config: pytest.Config) -> None:
    """Skip all tests in the live/ directory when OPENROUTER_API_KEY is not set."""
    skip = pytest.mark.skip(reason="OPENROUTER_API_KEY not set — skipping live evals")
    for item in items:
        if "live" in item.path.parts and not os.environ.get("OPENROUTER_API_KEY"):
            item.add_marker(skip)


@pytest.fixture
def live_provider() -> OpenRouterAdapter:
    """Construct an OpenRouterAdapter from the environment."""
    return OpenRouterAdapter(api_key=os.environ["OPENROUTER_API_KEY"])


@pytest.fixture
def live_config():  # type: ignore[no-untyped-def]
    """Build a real Config using load_config()."""
    return load_config()
