"""Tests for tools/search_files.py."""

from __future__ import annotations

from pathlib import Path

from agent.tools.search_files import search_files


class TestSearchFiles:
    async def test_glob_finds_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.txt").write_text("hello")
        result = await search_files(glob_pattern="*.py", directory=str(tmp_path))
        assert "a.py" in result
        assert "b.txt" not in result

    async def test_no_matching_files(self, tmp_path: Path) -> None:
        result = await search_files(glob_pattern="*.rs", directory=str(tmp_path))
        assert "No files found" in result

    async def test_content_pattern_grep(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("def foo():\n    return 42\n")
        result = await search_files(
            glob_pattern="*.py",
            directory=str(tmp_path),
            content_pattern="def ",
        )
        assert "def foo" in result

    async def test_content_pattern_no_match(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("x = 1\n")
        result = await search_files(
            glob_pattern="*.py",
            directory=str(tmp_path),
            content_pattern="class ",
        )
        assert "No matches found" in result

    async def test_invalid_directory(self) -> None:
        result = await search_files(glob_pattern="*.py", directory="/nonexistent/dir")
        assert "Error" in result

    async def test_invalid_regex(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("pass\n")
        result = await search_files(
            glob_pattern="*.py",
            directory=str(tmp_path),
            content_pattern="[invalid",
        )
        assert "Error" in result
        assert "regex" in result.lower() or "invalid" in result.lower()

    async def test_recursive_glob(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("pass\n")
        result = await search_files(glob_pattern="**/*.py", directory=str(tmp_path))
        assert "nested.py" in result
