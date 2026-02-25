"""Entry point: parses CLI arguments, loads config, boots the REPL."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    """Configure root logger.

    Args:
        verbose: If True, set level to DEBUG; otherwise WARNING.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    """CLI entry point for the coding agent."""
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Very Simple Coding Agent — ReAct loop over your codebase.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to a TOML config file (overrides default.toml).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    # Lazy imports so startup is fast when --help is used.
    from agent.config import load_config
    from agent.providers import get_provider
    from agent.repl import run_repl

    config_path = Path(args.config) if args.config else None

    try:
        config = load_config(config_path=config_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    provider = get_provider(config)

    # Agent operates on the current working directory.
    cwd = os.getcwd()
    if not config.system_prompt:
        config.system_prompt = (
            f"You are a helpful coding assistant. "
            f"The current working directory is: {cwd}\n"
            "You have access to tools to read files, list directories, search code, "
            "write files, and execute commands. Always reason carefully before acting."
        )

    asyncio.run(run_repl(config=config, provider=provider))


if __name__ == "__main__":
    main()
