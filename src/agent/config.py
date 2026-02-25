"""Configuration loader: reads TOML defaults then applies env-var overrides."""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path

from agent.models import Config

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default.toml"


def load_config(config_path: Path | None = None) -> Config:
    """Load agent configuration from a TOML file with env-var overrides.

    Resolution order (later wins):
    1. Hard-coded defaults in ``config/default.toml``
    2. Values in *config_path* (if provided)
    3. Environment variables: ``OPENROUTER_API_KEY``, ``AGENT_MODEL``,
       ``AGENT_MAX_TOKENS``, ``AGENT_SYSTEM_PROMPT``

    Args:
        config_path: Optional path to an additional TOML config file.

    Returns:
        Populated :class:`~agent.models.Config` instance.

    Raises:
        FileNotFoundError: If *config_path* is given but does not exist.
        ValueError: If ``api_key`` cannot be resolved from any source.
    """
    data: dict[str, object] = {}

    # 1. Load built-in defaults.
    if _DEFAULT_CONFIG_PATH.exists():
        with _DEFAULT_CONFIG_PATH.open("rb") as fh:
            data.update(tomllib.load(fh))
        logger.debug("Loaded default config from %s", _DEFAULT_CONFIG_PATH)

    # 2. Overlay user-supplied config file.
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("rb") as fh:
            data.update(tomllib.load(fh))
        logger.debug("Overlaid config from %s", config_path)

    # 3. Environment variable overrides.
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        data["api_key"] = api_key

    if model_env := os.environ.get("AGENT_MODEL"):
        data["model"] = model_env

    if max_tokens_env := os.environ.get("AGENT_MAX_TOKENS"):
        data["max_tokens"] = int(max_tokens_env)

    if system_prompt_env := os.environ.get("AGENT_SYSTEM_PROMPT"):
        data["system_prompt"] = system_prompt_env

    # Validate required fields.
    if not data.get("api_key"):
        raise ValueError(
            "No API key found. Set the OPENROUTER_API_KEY environment variable."
        )
    if not data.get("model"):
        raise ValueError(
            "No model configured. Set AGENT_MODEL or provide a config file."
        )

    return Config(
        provider=str(data.get("provider", "openrouter")),
        model=str(data["model"]),
        api_key=str(data["api_key"]),
        max_tokens=int(str(data.get("max_tokens", 4096))),
        max_history_tokens=int(str(data.get("max_history_tokens", 80_000))),
        system_prompt=str(data.get("system_prompt", "")),
    )
