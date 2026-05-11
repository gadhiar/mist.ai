"""Unit tests for backend.vault.conventions.ConventionsLoader."""

from pathlib import Path

from backend.vault.conventions import MAX_BYTES, ConventionsLoader


def test_load_vault_root_returns_none_if_no_file(tmp_path: Path) -> None:
    loader = ConventionsLoader(tmp_path)
    assert loader.load_vault_root() is None


def test_load_vault_root_prefers_mist_md_over_claude_md(tmp_path: Path) -> None:
    (tmp_path / "MIST.md").write_text("MIST content", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("CLAUDE content", encoding="utf-8")
    loader = ConventionsLoader(tmp_path)
    assert loader.load_vault_root() == "MIST content"


def test_load_vault_root_falls_back_to_claude_md(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("CLAUDE content", encoding="utf-8")
    loader = ConventionsLoader(tmp_path)
    assert loader.load_vault_root() == "CLAUDE content"


def test_load_vault_root_caches_by_mtime(tmp_path: Path) -> None:
    p = tmp_path / "MIST.md"
    p.write_text("v1", encoding="utf-8")
    loader = ConventionsLoader(tmp_path)
    assert loader.load_vault_root() == "v1"
    p.write_text("v2", encoding="utf-8")
    # Force mtime advance on filesystems with low mtime resolution
    import os

    new_mtime = p.stat().st_mtime + 1
    os.utime(p, (new_mtime, new_mtime))
    assert loader.load_vault_root() == "v2"


def test_load_vault_root_truncates_oversize_files(tmp_path: Path) -> None:
    big = "x" * (MAX_BYTES * 2)
    (tmp_path / "MIST.md").write_text(big, encoding="utf-8")
    loader = ConventionsLoader(tmp_path)
    out = loader.load_vault_root()
    assert out is not None
    assert len(out) == MAX_BYTES


def test_format_for_prompt_wraps_with_header(tmp_path: Path) -> None:
    (tmp_path / "MIST.md").write_text("body", encoding="utf-8")
    loader = ConventionsLoader(tmp_path)
    content = loader.load_vault_root()
    formatted = loader.format_for_prompt(content)
    assert formatted.startswith("=== VAULT CONVENTIONS")
    assert "body" in formatted
