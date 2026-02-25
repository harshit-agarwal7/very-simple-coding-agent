"""Tests for src/agent/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import load_config


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


class TestLoadConfig:
    def test_loads_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        monkeypatch.setenv("AGENT_MODEL", "anthropic/claude-opus-4-6")
        # Prevent default.toml from being found by providing a blank slate.
        cfg = load_config()
        assert cfg.api_key == "sk-test-key"
        assert cfg.model == "anthropic/claude-opus-4-6"
        assert cfg.provider == "openrouter"
        assert cfg.max_tokens == 4096

    def test_loads_from_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
        cfg_file = tmp_path / "my_config.toml"
        _write_toml(
            cfg_file,
            """
model = "openai/gpt-4o"
max_tokens = 2048
max_history_tokens = 40000
system_prompt = "Be concise."
""",
        )
        cfg = load_config(config_path=cfg_file)
        assert cfg.model == "openai/gpt-4o"
        assert cfg.max_tokens == 2048
        assert cfg.max_history_tokens == 40_000
        assert cfg.system_prompt == "Be concise."
        assert cfg.api_key == "sk-from-env"

    def test_env_overrides_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-key")
        monkeypatch.setenv("AGENT_MODEL", "google/gemini-2.0-flash")
        cfg_file = tmp_path / "cfg.toml"
        _write_toml(cfg_file, 'model = "openai/gpt-4o"\n')
        cfg = load_config(config_path=cfg_file)
        # Env var wins.
        assert cfg.model == "google/gemini-2.0-flash"
        assert cfg.api_key == "sk-env-key"

    def test_missing_config_file_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        with pytest.raises(FileNotFoundError):
            load_config(config_path=Path("/nonexistent/path/config.toml"))

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("AGENT_MODEL", "openai/gpt-4o")
        with pytest.raises(ValueError, match="API key"):
            load_config()

    def test_missing_model_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.delenv("AGENT_MODEL", raising=False)
        # Suppress default.toml so no model is pre-configured.
        monkeypatch.setattr("agent.config._DEFAULT_CONFIG_PATH", tmp_path / "noexist.toml")
        with pytest.raises(ValueError, match="model"):
            load_config()

    def test_agent_max_tokens_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key")
        monkeypatch.setenv("AGENT_MODEL", "openai/gpt-4o")
        monkeypatch.setenv("AGENT_MAX_TOKENS", "1024")
        cfg = load_config()
        assert cfg.max_tokens == 1024

    def test_agent_system_prompt_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-key")
        monkeypatch.setenv("AGENT_MODEL", "openai/gpt-4o")
        monkeypatch.setenv("AGENT_SYSTEM_PROMPT", "You are an expert.")
        cfg = load_config()
        assert cfg.system_prompt == "You are an expert."
