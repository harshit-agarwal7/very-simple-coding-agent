"""Tests for tools/list_directory.py."""

from __future__ import annotations

from pathlib import Path

from agent.tools.list_directory import list_directory


class TestListDirectory:
    async def test_lists_files_and_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("x")
        (tmp_path / "subdir").mkdir()
        result = await list_directory(path=str(tmp_path))
        assert "file.txt" in result
        assert "subdir/" in result

    async def test_empty_directory(self, tmp_path: Path) -> None:
        result = await list_directory(path=str(tmp_path))
        assert "empty" in result.lower()

    async def test_directory_not_found(self) -> None:
        result = await list_directory(path="/nonexistent/dir")
        assert "Error" in result
        assert "not found" in result

    async def test_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = await list_directory(path=str(f))
        assert "Error" in result
        assert "not a directory" in result

    async def test_dirs_sorted_before_files(self, tmp_path: Path) -> None:
        (tmp_path / "alpha.txt").write_text("x")
        (tmp_path / "zsubdir").mkdir()
        result = await list_directory(path=str(tmp_path))
        lines = result.splitlines()
        # Directories first (due to sort key), then files.
        dir_idx = next(i for i, line in enumerate(lines) if line.endswith("/"))
        file_idx = next(i for i, line in enumerate(lines) if not line.endswith("/"))
        assert dir_idx < file_idx
