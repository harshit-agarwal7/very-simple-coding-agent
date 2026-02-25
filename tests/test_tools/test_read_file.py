"""Tests for tools/read_file.py."""

from __future__ import annotations

from pathlib import Path

from agent.tools.read_file import read_file


class TestReadFile:
    async def test_reads_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "hello.txt"
        target.write_text("Hello, world!", encoding="utf-8")
        result = await read_file(path=str(target))
        assert result == "Hello, world!"

    async def test_file_not_found(self) -> None:
        result = await read_file(path="/nonexistent/path/file.txt")
        assert "Error" in result
        assert "not found" in result

    async def test_empty_file(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.txt"
        target.write_text("", encoding="utf-8")
        result = await read_file(path=str(target))
        assert result == ""
