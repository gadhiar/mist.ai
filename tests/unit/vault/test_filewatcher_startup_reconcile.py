"""Startup reconcile tests for VaultFilewatcher._scan_vault_mtimes.

The sidecar accumulates orphaned chunks when a vault file is deleted while
the filewatcher is DOWN: the deletion is never observed, so on restart the
gone file is absent from the disk walk AND absent from the _known_mtimes
baseline (which was seeded only from disk). The audit's vanish-delete only
prunes paths that ARE in the baseline but gone from disk, so the orphan
lives forever.

The fix seeds _known_mtimes from the sidecar's distinct_paths() at startup
with an mtime sentinel of 0 for any path not already present from the disk
walk. A between-session deletion then lands in the baseline and the first
audit prunes it.

CRITICAL CORRECTNESS RISK exercised here: _run_audit compares _known_mtimes
keys against str(Path(dirpath) / fname) disk-walk keys. The sidecar paths
seeded into _known_mtimes MUST be in the same absolute, OS-native format the
disk walk produces. If they differ (e.g. relative vs absolute), a still-on-disk
file would be seeded under a second key that is absent from disk_paths and
falsely pruned. The tests below use realistic absolute path strings and
include an explicit guard test that a still-existing file is NOT scheduled
for deletion after reconcile + audit.

All async tests are decorated with @pytest.mark.asyncio (asyncio_mode=strict).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from backend.knowledge.config import FilewatcherConfig
from backend.vault.filewatcher import VaultFilewatcher

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeSidecarIndex:
    """Explicit test double for SidecarIndexProtocol.

    Mirrors the fake in test_filewatcher.py and adds distinct_paths() so the
    startup-reconcile path can be exercised without a bare MagicMock at the
    sidecar I/O boundary (per tests/CLAUDE.md). distinct_paths_return is the
    set of paths the sidecar reports as currently indexed.
    """

    def __init__(self, distinct_paths_return: list[str] | None = None) -> None:
        self.distinct_paths_return: list[str] = distinct_paths_return or []
        self.delete_path_calls: list[str] = []

    # -- SidecarIndexProtocol surface --

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def upsert_file(
        self,
        path: str,
        content: str,
        mtime: int,
        frontmatter: dict | None = None,
    ) -> int:
        return 1

    def delete_path(self, path: str) -> int:
        self.delete_path_calls.append(path)
        return 1

    def query_vector(self, embedding: list[float], k: int = 10) -> list[dict]:
        return []

    def query_fts(self, text: str, k: int = 10) -> list[dict]:
        return []

    def query_hybrid(
        self,
        embedding: list[float],
        text: str,
        k: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        return []

    def chunk_count(self) -> int:
        return 0

    def health_check(self) -> bool:
        return True

    def distinct_paths(self) -> list[str]:
        return list(self.distinct_paths_return)


class _RaisingSidecarIndex(FakeSidecarIndex):
    """Sidecar whose distinct_paths raises, to verify the seed is guarded."""

    def distinct_paths(self) -> list[str]:
        raise RuntimeError("sidecar query exploded")


class _StubWriter:
    async def mark_authored_by_user_edit(self, path: Path) -> None:  # noqa: ARG002
        pass


class _StubInvalidationBus:
    async def publish(self, event: object) -> None:  # noqa: ARG002
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> FilewatcherConfig:
    defaults = {
        "enabled": True,
        "observer_type": "polling",
        "debounce_ms": 100,
        "staleness_slo_seconds": 5,
        "audit_interval_seconds": 3600,  # disabled; tests drive _run_audit manually
    }
    defaults.update(kwargs)
    return FilewatcherConfig(**defaults)


def _make_filewatcher(
    config: FilewatcherConfig,
    vault_root: Path,
    sidecar: FakeSidecarIndex,
) -> VaultFilewatcher:
    """Construct VaultFilewatcher with the real ctor signature.

    The sidecar parameter is keyword `sidecar_index=`; Phase-3 deps are
    required positional/keyword args supplied here as minimal stubs.
    """
    return VaultFilewatcher(
        config=config,
        vault_root=vault_root,
        sidecar_index=sidecar,
        invalidation_bus=_StubInvalidationBus(),
        writer=_StubWriter(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def vault_root(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    (root / "sessions").mkdir(parents=True)
    return root


# ---------------------------------------------------------------------------
# TestStartupReconcile
# ---------------------------------------------------------------------------


class TestStartupReconcile:
    def test_seeds_vanished_sidecar_path_into_known_mtimes(self, vault_root: Path):
        # Arrange: disk has only keep.md; sidecar reports keep.md AND a gone.md
        # that was deleted while the watcher was down. Paths use the exact
        # absolute str(Path(...)) format the disk walk produces.
        keep = vault_root / "sessions" / "keep.md"
        keep.write_text("---\ntype: mist-session\n---\n\n# Keep\n", encoding="utf-8")
        keep_key = str(keep)
        gone_key = str(vault_root / "sessions" / "gone.md")

        sidecar = FakeSidecarIndex(distinct_paths_return=[keep_key, gone_key])
        fw = _make_filewatcher(_make_config(), vault_root, sidecar)

        # Act
        fw._scan_vault_mtimes()

        # Assert: both paths are baselined; gone.md carries the sentinel 0 so
        # the next audit treats it as known-but-gone and prunes it.
        assert gone_key in fw._known_mtimes
        assert fw._known_mtimes[gone_key] == 0
        assert keep_key in fw._known_mtimes

    def test_does_not_overwrite_existing_disk_mtime(self, vault_root: Path):
        # Arrange: keep.md exists on disk (real mtime > 0) and is ALSO reported
        # by the sidecar. setdefault must not clobber the real mtime with 0.
        keep = vault_root / "sessions" / "keep.md"
        keep.write_text("---\ntype: mist-session\n---\n\n# Keep\n", encoding="utf-8")
        keep_key = str(keep)

        sidecar = FakeSidecarIndex(distinct_paths_return=[keep_key])
        fw = _make_filewatcher(_make_config(), vault_root, sidecar)

        # Act
        fw._scan_vault_mtimes()

        # Assert: real mtime preserved (not reset to the 0 sentinel)
        assert fw._known_mtimes[keep_key] > 0

    def test_sidecar_failure_does_not_break_scan(self, vault_root: Path):
        # Arrange: disk has keep.md; sidecar.distinct_paths raises.
        keep = vault_root / "sessions" / "keep.md"
        keep.write_text("---\ntype: mist-session\n---\n\n# Keep\n", encoding="utf-8")
        keep_key = str(keep)

        fw = _make_filewatcher(_make_config(), vault_root, _RaisingSidecarIndex())

        # Act: must not raise -- the disk-walk baseline still populates.
        fw._scan_vault_mtimes()

        # Assert: disk file still baselined despite the sidecar error.
        assert keep_key in fw._known_mtimes

    @pytest.mark.asyncio
    async def test_audit_after_reconcile_prunes_vanished_path(self, vault_root: Path):
        # Arrange: end-to-end -- seed from sidecar, then run the real audit and
        # confirm the vanished path is scheduled for sidecar deletion while the
        # still-on-disk path is NOT (the false-prune correctness guard).
        keep = vault_root / "sessions" / "keep.md"
        keep.write_text("---\ntype: mist-session\n---\n\n# Keep\n", encoding="utf-8")
        keep_key = str(keep)
        gone_key = str(vault_root / "sessions" / "gone.md")

        sidecar = FakeSidecarIndex(distinct_paths_return=[keep_key, gone_key])
        fw = _make_filewatcher(_make_config(), vault_root, sidecar)
        fw.start(loop=asyncio.get_running_loop())
        await asyncio.sleep(0.05)  # let the start() scan settle

        # Act: the audit compares _known_mtimes against the live disk walk.
        await fw._run_audit()
        await asyncio.sleep(0.15)  # let the scheduled _do_delete task run

        fw.stop()

        # Assert: gone.md pruned, keep.md untouched.
        assert gone_key in sidecar.delete_path_calls
        assert keep_key not in sidecar.delete_path_calls
