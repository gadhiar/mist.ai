"""Phase 3 mist_admin extensions: vault-rebuild --scope/--retry-orphaned.

Tests cover the new async cmd_vault_rebuild signature added in Task 22:

- scope=<path>  -> calls regenerator.rebuild_from_path(path) exactly once
- scope="all"   -> iterates vault tree and calls rebuild_from_path for each
                   non-excluded .md file
- retry_orphaned=True -> calls regenerator.retry_orphaned()
- scope=None, retry_orphaned=False -> legacy sidecar-only rebuild path (no
                                       regenerator call)

All tests use fakes for the admin context and regenerator so no real Neo4j,
LLM, or sidecar dependency is required.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# scripts/ is not a package; insert repo root so mist_admin is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import mist_admin  # noqa: E402  -- after sys.path insertion

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRegenerator:
    """Minimal fake for GraphRegenerator used by cmd_vault_rebuild tests."""

    def __init__(self) -> None:
        self.rebuild_calls: list[Path] = []
        self.retry_orphaned_called: bool = False

    async def rebuild_from_path(self, path: Path) -> None:
        self.rebuild_calls.append(path)

    async def retry_orphaned(self) -> None:
        self.retry_orphaned_called = True


class FakeSidecar:
    """Minimal fake for the sidecar used by cmd_vault_rebuild legacy path."""

    def __init__(self) -> None:
        self.rebuild_all_called: bool = False

    async def rebuild_all(self) -> None:
        self.rebuild_all_called = True


class MockAdminContext:
    """Minimal fake AdminContext carrying regenerator, sidecar, and vault_root."""

    def __init__(self, vault_root: Path) -> None:
        self.regenerator = FakeRegenerator()
        self.sidecar = FakeSidecar()
        self.vault_root = vault_root


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_vault(tmp_path: Path) -> Path:
    """Create a minimal empty vault root."""
    root = tmp_path / "vault"
    root.mkdir()
    return root


@pytest.fixture()
def mock_admin_context(tmp_vault: Path) -> MockAdminContext:
    return MockAdminContext(vault_root=tmp_vault)


# ---------------------------------------------------------------------------
# Tests: scope=<path>
# ---------------------------------------------------------------------------


def test_vault_rebuild_scope_path_invokes_regenerator(
    mock_admin_context: MockAdminContext,
    tmp_vault: Path,
) -> None:
    """scope=<path> calls regenerator.rebuild_from_path with the given path exactly once."""
    (tmp_vault / "users").mkdir()
    p = tmp_vault / "users" / "raj.md"
    p.write_text("---\ntype: mist-user\nuser_id: raj\n---\nbody\n", encoding="utf-8")

    rc = asyncio.run(
        mist_admin.cmd_vault_rebuild(
            scope=str(p),
            retry_orphaned=False,
            ctx=mock_admin_context,
        )
    )

    assert rc == 0
    assert mock_admin_context.regenerator.rebuild_calls == [p]
    assert mock_admin_context.sidecar.rebuild_all_called is False


def test_vault_rebuild_scope_relative_path_resolves_against_vault_root(
    mock_admin_context: MockAdminContext,
    tmp_vault: Path,
) -> None:
    """A relative scope path is resolved against vault_root."""
    (tmp_vault / "sessions").mkdir()
    p = tmp_vault / "sessions" / "note.md"
    p.write_text("body", encoding="utf-8")

    # Pass relative path (relative to vault root)
    rc = asyncio.run(
        mist_admin.cmd_vault_rebuild(
            scope="sessions/note.md",
            retry_orphaned=False,
            ctx=mock_admin_context,
        )
    )

    assert rc == 0
    assert mock_admin_context.regenerator.rebuild_calls == [tmp_vault / "sessions" / "note.md"]


# ---------------------------------------------------------------------------
# Tests: scope="all"
# ---------------------------------------------------------------------------


def test_vault_rebuild_scope_all_iterates_all_files(
    mock_admin_context: MockAdminContext,
    tmp_vault: Path,
) -> None:
    """scope='all' calls rebuild_from_path for every .md file not excluded."""
    (tmp_vault / "identity").mkdir()
    (tmp_vault / "identity" / "mist.md").write_text("body", encoding="utf-8")
    (tmp_vault / "users").mkdir()
    (tmp_vault / "users" / "raj.md").write_text("body", encoding="utf-8")

    rc = asyncio.run(
        mist_admin.cmd_vault_rebuild(
            scope="all",
            retry_orphaned=False,
            ctx=mock_admin_context,
        )
    )

    assert rc == 0
    called_paths = set(mock_admin_context.regenerator.rebuild_calls)
    assert tmp_vault / "identity" / "mist.md" in called_paths
    assert tmp_vault / "users" / "raj.md" in called_paths
    assert len(called_paths) == 2


def test_vault_rebuild_scope_all_skips_excluded_filenames(
    mock_admin_context: MockAdminContext,
    tmp_vault: Path,
) -> None:
    """scope='all' skips MIST.md and CLAUDE.md (excluded conventions docs)."""
    p_normal = tmp_vault / "note.md"
    p_normal.write_text("body", encoding="utf-8")
    p_mist = tmp_vault / "MIST.md"
    p_mist.write_text("conventions", encoding="utf-8")
    p_claude = tmp_vault / "CLAUDE.md"
    p_claude.write_text("conventions", encoding="utf-8")

    rc = asyncio.run(
        mist_admin.cmd_vault_rebuild(
            scope="all",
            retry_orphaned=False,
            ctx=mock_admin_context,
        )
    )

    assert rc == 0
    called = mock_admin_context.regenerator.rebuild_calls
    assert p_normal in called
    assert p_mist not in called
    assert p_claude not in called


def test_vault_rebuild_scope_all_skips_meta_directory(
    mock_admin_context: MockAdminContext,
    tmp_vault: Path,
) -> None:
    """scope='all' skips files under meta/ directory."""
    (tmp_vault / "meta").mkdir()
    p_meta = tmp_vault / "meta" / "schema.md"
    p_meta.write_text("schema doc", encoding="utf-8")
    p_normal = tmp_vault / "sessions" / "note.md"
    p_normal.parent.mkdir()
    p_normal.write_text("body", encoding="utf-8")

    rc = asyncio.run(
        mist_admin.cmd_vault_rebuild(
            scope="all",
            retry_orphaned=False,
            ctx=mock_admin_context,
        )
    )

    assert rc == 0
    called = mock_admin_context.regenerator.rebuild_calls
    assert p_normal in called
    assert p_meta not in called


# ---------------------------------------------------------------------------
# Tests: retry_orphaned=True
# ---------------------------------------------------------------------------


def test_vault_rebuild_retry_orphaned_invokes_retry_path(
    mock_admin_context: MockAdminContext,
) -> None:
    """retry_orphaned=True calls regenerator.retry_orphaned() without sidecar rebuild."""
    rc = asyncio.run(
        mist_admin.cmd_vault_rebuild(
            scope=None,
            retry_orphaned=True,
            ctx=mock_admin_context,
        )
    )

    assert rc == 0
    assert mock_admin_context.regenerator.retry_orphaned_called is True
    assert mock_admin_context.regenerator.rebuild_calls == []
    assert mock_admin_context.sidecar.rebuild_all_called is False


def test_vault_rebuild_retry_orphaned_takes_priority_over_scope(
    mock_admin_context: MockAdminContext,
    tmp_vault: Path,
) -> None:
    """retry_orphaned=True takes priority when scope is also set."""
    p = tmp_vault / "note.md"
    p.write_text("body", encoding="utf-8")

    rc = asyncio.run(
        mist_admin.cmd_vault_rebuild(
            scope=str(p),
            retry_orphaned=True,
            ctx=mock_admin_context,
        )
    )

    assert rc == 0
    assert mock_admin_context.regenerator.retry_orphaned_called is True
    # rebuild_from_path should NOT have been called
    assert mock_admin_context.regenerator.rebuild_calls == []


# ---------------------------------------------------------------------------
# Tests: legacy mode (no flags)
# ---------------------------------------------------------------------------


def test_vault_rebuild_no_flags_calls_legacy_sidecar_rebuild(
    mock_admin_context: MockAdminContext,
) -> None:
    """scope=None, retry_orphaned=False calls legacy sidecar rebuild; no regenerator call."""
    rc = asyncio.run(
        mist_admin.cmd_vault_rebuild(
            scope=None,
            retry_orphaned=False,
            ctx=mock_admin_context,
        )
    )

    assert rc == 0
    assert mock_admin_context.sidecar.rebuild_all_called is True
    assert mock_admin_context.regenerator.rebuild_calls == []
    assert mock_admin_context.regenerator.retry_orphaned_called is False
