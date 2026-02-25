"""Tests for tools/write_file.py."""

from __future__ import annotations

from pathlib import Path

from agent.tools.write_file import write_file


class TestWriteFile:
    async def test_creates_new_file(self, tmp_path: Path) -> None:
        target = tmp_path / "output.txt"
        result = await write_file(path=str(target), content="Hello!")
        assert "Successfully wrote" in result
        assert target.read_text() == "Hello!"

    async def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c.txt"
        result = await write_file(path=str(target), content="nested")
        assert "Successfully wrote" in result
        assert target.read_text() == "nested"

    async def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "file.txt"
        target.write_text("old content")
        await write_file(path=str(target), content="new content")
        assert target.read_text() == "new content"

    async def test_reports_byte_count(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        content = "abc"
        result = await write_file(path=str(target), content=content)
        assert str(len(content)) in result
