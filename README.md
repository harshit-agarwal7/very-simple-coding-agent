# Very Simple Coding Agent

A minimal, terminal-based coding agent that uses a ReAct (Reason → Act → Observe) loop to help you work with your codebase. It connects to LLMs via OpenRouter and gives the model access to filesystem and shell tools.

## Features

- **ReAct agent loop** — the model reasons, calls tools, observes results, and repeats until the task is done
- **Streaming responses** — tokens are rendered as markdown in the terminal via Rich
- **Built-in tools** — read/write files, list directories, search code, execute commands, and a scratchpad "think" tool
- **Safety gate** — dangerous tools (write, execute) require user approval before running
- **Conversation compaction** — `/compact` summarises history to stay within context limits
- **Configurable** — swap models, providers, and token limits via TOML config or environment variables

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- An [OpenRouter](https://openrouter.ai/) API key

## Quickstart

```bash
# Clone and install
git clone <repo-url>
cd very-simple-coding-agent
uv sync

# Set your API key
export OPENROUTER_API_KEY="sk-or-..."

# Run the agent
uv run agent
```

The agent operates on your current working directory, so `cd` into the project you want to work on before launching.

## Usage

```
agent [--config PATH] [--verbose]
```

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to a TOML config file (overrides `config/default.toml`) |
| `--verbose, -v` | Enable debug logging to stderr |

### REPL commands

| Command | Description |
|---------|-------------|
| `/compact` | Summarise and compress conversation history |
| `/quit` | Exit the agent |

## Configuration

Defaults live in `config/default.toml`:

```toml
provider = "openrouter"
model = "arcee-ai/trinity-large-preview:free"
max_tokens = 4096
max_history_tokens = 80000
system_prompt = ""
```

Override the model at runtime with the `AGENT_MODEL` environment variable, or point to a custom TOML file with `--config`.

## Tools

The agent exposes these tools to the LLM:

| Tool | Safety | Description |
|------|--------|-------------|
| `read_file` | safe | Read the contents of a file |
| `list_directory` | safe | List files and directories |
| `search_files` | safe | Search file contents with a pattern |
| `think` | safe | Scratchpad for the model to reason step-by-step |
| `write_file` | requires approval | Write or overwrite a file |
| `execute_command` | requires approval | Run a shell command |

Tools marked **requires approval** will prompt you for confirmation before executing.

## Project structure

```
src/agent/
  main.py            # CLI entry point
  repl.py            # Interactive REPL
  loop.py            # ReAct agent loop
  memory.py          # Conversation history and compaction
  models.py          # Shared dataclasses and enums
  config.py          # Config loading (TOML + env vars)
  providers/
    base.py          # Provider adapter interface
    openrouter.py    # OpenRouter implementation (OpenAI-compatible)
  tools/
    registry.py      # Tool registry
    executor.py      # Tool dispatcher with approval gate
    read_file.py
    write_file.py
    search_files.py
    list_directory.py
    execute_command.py
    think.py
config/default.toml  # Default configuration
tests/               # Mirrors src/ structure
```

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

## License

MIT
