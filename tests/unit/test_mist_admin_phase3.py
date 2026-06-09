"""Phase 3 mist_admin extensions: vault-rebuild --scope/--retry-orphaned.

Tests cover the new async cmd_vault_rebuild signature added in Task 22:

- scope=<path>  -> calls regenerator.rebuild_from_path(path) exactly once
- scope="all"   -> iterates vault tree and calls rebuild_from_path for each
                   non-excluded .md file
- retry_orphaned=True -> calls regenerator.retry_orphaned()

The legacy sidecar-only rebuild (scope=None, retry_orphaned=False) is NOT
handled inside cmd_vault_rebuild. _dispatch_vault_rebuild routes that case to
_cmd_vault_rebuild_sidecar before this coroutine is reached, so cmd_vault_rebuild
is only entered with a truthy scope or retry_orphaned. The dispatch-level routing
test (TestVaultRebuildArgparseDispatch.test_no_flags_routes_to_legacy_sidecar)
covers the no-flags case.

All tests use fakes for the admin context and regenerator so no real Neo4j,
LLM, or sidecar dependency is required.
"""

from __future__ import annotations

import argparse
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
    """Inert sidecar sentinel carried on the admin context.

    cmd_vault_rebuild does NOT call any sidecar method (the legacy sidecar-only
    rebuild lives in _cmd_vault_rebuild_sidecar, reached via the dispatch before
    this coroutine). This fake deliberately defines no methods so that any
    attribute access from cmd_vault_rebuild is recorded in `touched_attrs` and
    surfaces as a test signal rather than silently conforming. The real
    VaultSidecarIndex has no `rebuild_all` method; pretending it does is what
    previously masked the dead `scope is None` branch.
    """

    def __init__(self) -> None:
        self.touched_attrs: list[str] = []

    def __getattr__(self, name: str) -> object:
        # __getattr__ only fires for attributes not found normally (i.e. not
        # touched_attrs, which __init__ sets via __dict__). Record and fail loud.
        self.__dict__["touched_attrs"].append(name)
        raise AttributeError(
            f"cmd_vault_rebuild touched sidecar.{name}; it must not call the "
            f"sidecar (legacy rebuild is handled by _cmd_vault_rebuild_sidecar)"
        )


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
    assert mock_admin_context.sidecar.touched_attrs == []


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
    assert mock_admin_context.sidecar.touched_attrs == []


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
# Tests: legacy no-flags case is handled upstream, not inside cmd_vault_rebuild
#
# The former `scope is None` branch of cmd_vault_rebuild called a nonexistent
# `sidecar.rebuild_all()` and was dead: _dispatch_vault_rebuild routes the
# no-flags case to _cmd_vault_rebuild_sidecar before cmd_vault_rebuild runs.
# These tests pin the corrected contract.
# ---------------------------------------------------------------------------


def test_cmd_vault_rebuild_never_touches_sidecar_on_retry_path(
    mock_admin_context: MockAdminContext,
) -> None:
    """cmd_vault_rebuild must not call any sidecar method (legacy path lives elsewhere).

    The inert FakeSidecar raises AttributeError on any access. retry_orphaned is
    the one combination still routed into cmd_vault_rebuild that previously sat
    adjacent to the dead branch, so it is the safest guard that the sidecar is
    never reached from this coroutine.
    """
    rc = asyncio.run(
        mist_admin.cmd_vault_rebuild(
            scope=None,
            retry_orphaned=True,
            ctx=mock_admin_context,
        )
    )

    assert rc == 0
    assert mock_admin_context.sidecar.touched_attrs == []


def test_cmd_vault_rebuild_source_has_no_rebuild_all_reference() -> None:
    """The removed dead branch must not regrow: no `rebuild_all` in cmd_vault_rebuild.

    `rebuild_all` exists nowhere on the real VaultSidecarIndex; a reference here
    is necessarily the dead branch returning. Inspect the coroutine source rather
    than executing the (now-unreachable) scope=None, retry_orphaned=False input,
    which the dispatch can never produce.
    """
    import inspect

    src = inspect.getsource(mist_admin.cmd_vault_rebuild)

    assert "rebuild_all" not in src


# ---------------------------------------------------------------------------
# Tests: argparse dispatch routing (Fix A -- P0 #4)
#
# These tests verify that the argparse subparser for vault-rebuild routes to
# the new async cmd_vault_rebuild when --scope or --retry-orphaned is set,
# and falls through to the legacy _cmd_vault_rebuild_sidecar otherwise.
# ---------------------------------------------------------------------------


class TestVaultRebuildArgparseDispatch:
    """Argparse dispatch from vault-rebuild subcommand routes correctly."""

    def _build_parser(self) -> argparse.ArgumentParser:
        """Reconstruct the mist_admin argument parser."""
        return mist_admin.build_parser()

    def _patch_all(self, monkeypatch: pytest.MonkeyPatch) -> tuple[list, list]:
        """Patch cmd_vault_rebuild, _build_vault_rebuild_ctx, and _cmd_vault_rebuild_sidecar.

        Returns (new_path_calls, legacy_calls) recording which handler was invoked.
        The ctx builder is patched to a no-op so heavy backend deps are never imported
        during routing tests.
        """
        new_path_calls: list[dict] = []
        legacy_calls: list[object] = []

        async def fake_cmd_vault_rebuild(
            scope: str | None,
            retry_orphaned: bool,
            ctx: object,
        ) -> int:
            new_path_calls.append({"scope": scope, "retry_orphaned": retry_orphaned})
            return 0

        def fake_sidecar(args: object) -> int:
            legacy_calls.append(args)
            return 0

        def fake_build_ctx() -> object:
            return object()  # opaque sentinel; cmd_vault_rebuild is patched anyway

        monkeypatch.setattr(mist_admin, "cmd_vault_rebuild", fake_cmd_vault_rebuild)
        monkeypatch.setattr(mist_admin, "_cmd_vault_rebuild_sidecar", fake_sidecar)
        monkeypatch.setattr(mist_admin, "_build_vault_rebuild_ctx", fake_build_ctx)
        return new_path_calls, legacy_calls

    def test_scope_flag_routes_to_cmd_vault_rebuild(
        self,
        tmp_vault: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """vault-rebuild --scope=foo.md invokes cmd_vault_rebuild, not the sidecar handler."""
        new_path_calls, legacy_calls = self._patch_all(monkeypatch)

        parser = self._build_parser()
        args = parser.parse_args(["vault-rebuild", "--scope", "foo.md"])

        # _dispatch_vault_rebuild is what set_defaults(func=...) points to
        args.func(args)

        assert len(new_path_calls) == 1
        assert new_path_calls[0]["scope"] == "foo.md"
        assert new_path_calls[0]["retry_orphaned"] is False
        assert legacy_calls == []

    def test_retry_orphaned_flag_routes_to_cmd_vault_rebuild(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """vault-rebuild --retry-orphaned invokes cmd_vault_rebuild, not the sidecar handler."""
        new_path_calls, legacy_calls = self._patch_all(monkeypatch)

        parser = self._build_parser()
        args = parser.parse_args(["vault-rebuild", "--retry-orphaned"])

        args.func(args)

        assert len(new_path_calls) == 1
        assert new_path_calls[0]["scope"] is None
        assert new_path_calls[0]["retry_orphaned"] is True
        assert legacy_calls == []

    def test_no_flags_routes_to_legacy_sidecar(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """vault-rebuild (no flags) invokes _cmd_vault_rebuild_sidecar, not cmd_vault_rebuild."""
        new_path_calls, legacy_calls = self._patch_all(monkeypatch)

        parser = self._build_parser()
        args = parser.parse_args(["vault-rebuild"])

        args.func(args)

        assert legacy_calls != []
        assert new_path_calls == []
