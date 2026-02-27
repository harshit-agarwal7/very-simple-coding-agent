"""Providers package: factory function to get the configured provider adapter."""

from rich.console import Console

from agent.models import Config
from agent.providers.base import ProviderAdapter
from agent.providers.openrouter import OpenRouterAdapter


def get_provider(config: Config, console: Console | None = None) -> ProviderAdapter:
    """Return the appropriate provider adapter for the given config.

    Args:
        config: Agent runtime configuration.
        console: Optional Rich console to use for rendering output.

    Returns:
        A :class:`~agent.providers.base.ProviderAdapter` instance.

    Raises:
        ValueError: If the provider in *config* is not recognised.
    """
    if config.provider == "openrouter":
        return OpenRouterAdapter(api_key=config.api_key, console=console)
    raise ValueError(f"Unknown provider: {config.provider!r}")


__all__ = ["ProviderAdapter", "get_provider"]
