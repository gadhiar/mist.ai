"""R1.3 vault-edit read-path propagation contract test.

Supersedes the ADR-010 invariant-5 contract test (2026-05-08, commit
0462802), which asserted a four-step user-edit coordination: authored_by
writeback, graph orphan-mark, graph re-derivation, then cache-invalidation
publish. R1.3 retired the two middle steps -- the vault is prose MIST reads,
not a fact source (Inv-A1) -- so what the filewatcher now guarantees on a
user edit is: sidecar reindex, authored_by writeback, then bus publish. No
graph store is constructed anywhere in this file; the filewatcher holds no
handle through which a vault edit could reach the graph.

The test calls VaultFilewatcher._do_reindex directly (bypassing the
watchdog observer thread) to exercise the in-process coordination contract
without thread / port dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.knowledge.config import FilewatcherConfig
from backend.vault.filewatcher import VaultFilewatcher
from backend.vault.invalidation_bus import InvalidationBus, VaultChangeEvent

# ---------------------------------------------------------------------------
# Inline fakes / stubs
# ---------------------------------------------------------------------------


class _FakeSidecarIndex:
    """Minimal SidecarIndexProtocol double; records upsert calls."""

    def __init__(self) -> None:
        self.upserts: list[str] = []

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def upsert_file(self, path: str, content: str, mtime: int, frontmatter=None) -> int:
        self.upserts.append(path)
        return 1

    def delete_path(self, path: str) -> int:
        return 0

    def query_vector(self, embedding, k=10):
        return []

    def query_fts(self, text, k=10):
        return []

    def query_hybrid(self, embedding, text, k=10, rrf_k=60):
        return []

    def chunk_count(self) -> int:
        return 0

    def health_check(self) -> bool:
        return True


class _FakeVaultWriter:
    """Minimal test double for the writer surface _do_reindex calls.

    Records the paths passed to mark_authored_by_user_edit so tests can
    assert the writeback fired, without driving VaultWriter's real
    queue-consumer lifecycle (mark_authored_by_user_edit is queue-backed on
    the real class and raises VaultWriteError unless the consumer is
    started).
    """

    def __init__(self) -> None:
        self.authored_by_calls: list[Path] = []

    async def mark_authored_by_user_edit(self, path: Path) -> None:
        self.authored_by_calls.append(path)


class _RecordingListener:
    """Subscribes to InvalidationBus and records published events."""

    def __init__(self) -> None:
        self.events: list[VaultChangeEvent] = []

    async def __call__(self, event: VaultChangeEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_filewatcher(
    vault_root: Path,
) -> tuple[VaultFilewatcher, _FakeSidecarIndex, _FakeVaultWriter, _RecordingListener]:
    """Wire a VaultFilewatcher against local fakes for every dependency it still has.

    R1.3 dropped the regenerator/graph-store handle entirely: _do_reindex's
    post-sidecar surface is exactly sidecar + writer + bus, so that is
    exactly what these fakes cover.
    """
    sidecar = _FakeSidecarIndex()
    writer = _FakeVaultWriter()
    bus = InvalidationBus()
    listener = _RecordingListener()
    bus.subscribe(listener)
    config = FilewatcherConfig(
        enabled=True,
        observer_type="polling",
        debounce_ms=500,
        staleness_slo_seconds=5,
        audit_interval_seconds=60,
    )
    watcher = VaultFilewatcher(
        config=config,
        vault_root=vault_root,
        sidecar_index=sidecar,
        invalidation_bus=bus,
        writer=writer,
    )
    return watcher, sidecar, writer, listener


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestVaultEditReadPathPropagation:
    """R1.3: a user edit reaches the read path and writes no graph facts.

    Supersedes the ADR-010 invariant-5 test. The old contract sequenced
    authored_by -> graph rebuild -> cache invalidation. The middle step retired
    with Inv-A1: the vault is prose MIST reads, not a fact source.
    """

    @pytest.mark.asyncio
    async def test_user_edit_triggers_two_step_coordination(self, tmp_path: Path) -> None:
        """Sidecar reindex, authored_by writeback, then bus publish."""
        watcher, sidecar, writer, listener = _build_filewatcher(tmp_path)
        target = tmp_path / "users" / "raj.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("---\nuser_id: raj\n---\n\nEdited by hand.\n", encoding="utf-8")

        await watcher._do_reindex(str(target), is_mist_write=False)

        assert sidecar.upserts == [str(target)], "step 0: sidecar reindexed"
        assert writer.authored_by_calls == [target], "step 1: authored_by flipped"
        assert [e.path for e in listener.events] == [target], "step 2: bus published"

    @pytest.mark.asyncio
    async def test_user_edit_writes_no_graph_facts(self, tmp_path: Path) -> None:
        """The core R1.3 contract, asserted at the wiring level.

        The filewatcher holds no graph handle at all now -- there is no seam
        through which a vault edit could reach the graph.
        """
        watcher, _sidecar, _writer, _listener = _build_filewatcher(tmp_path)
        target = tmp_path / "users" / "raj.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("---\nuser_id: raj\n---\n\nEdited.\n", encoding="utf-8")

        await watcher._do_reindex(str(target), is_mist_write=False)

        assert not hasattr(watcher, "_regenerator")
        assert not hasattr(watcher, "_graph_store")

    @pytest.mark.asyncio
    async def test_mist_write_skips_the_user_edit_steps(self, tmp_path: Path) -> None:
        """MIST's own writes reindex but never flip authored_by or publish."""
        watcher, sidecar, writer, listener = _build_filewatcher(tmp_path)
        target = tmp_path / "sessions" / "2026-07-30.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Session\n\nTurn 1.\n", encoding="utf-8")

        await watcher._do_reindex(str(target), is_mist_write=True)

        assert sidecar.upserts == [str(target)], "reindex still happens"
        assert writer.authored_by_calls == [], "no authored_by flip on a MIST write"
        assert listener.events == [], "no bus publish on a MIST write"

    @pytest.mark.asyncio
    async def test_bus_listener_receives_vault_change_event(self, tmp_path: Path) -> None:
        """The payload is the vault-owned event, carrying the edited path."""
        watcher, _sidecar, _writer, listener = _build_filewatcher(tmp_path)
        target = tmp_path / "users" / "raj.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("---\nuser_id: raj\n---\n\nEdited.\n", encoding="utf-8")

        await watcher._do_reindex(str(target), is_mist_write=False)

        assert len(listener.events) == 1
        event = listener.events[0]
        assert isinstance(event, VaultChangeEvent)
        assert event.path == target
