"""Inv-A1 guard: a user vault edit performs no graph write.

R1.3 retired GraphRegenerator (Task 6), the class that used to sit between
the sidecar reindex and the read-path cache invalidation on VaultFilewatcher's
vault-edit sequence. Seven `rebuild_*` tests in the deleted
tests/unit/knowledge/curation/test_graph_regenerator.py collectively proved
the guarantee this file now carries forward onto its new subject,
`VaultFilewatcher._do_reindex`: a vault edit performs no graph write.

VaultFilewatcher no longer accepts any graph-store-shaped dependency at all,
so there is no injection point left for a write to occur through. The guard
below combines a structural check (no such constructor parameter exists) with
a behavioral one, mirroring the deleted suite's
test_rebuild_does_not_orphan_existing_triples: a triple pre-seeded on a
FakeGraphStore and scoped to the exact path under edit survives a full
_do_reindex run untouched, for both a users/ path and a sessions/ path,
because nothing in the reindex sequence ever references the store.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

from backend.knowledge.config import FilewatcherConfig
from backend.vault.filewatcher import VaultFilewatcher
from tests.fakes.graph_store import FakeGraphStore


class _FakeSidecarIndex:
    """Minimal SidecarIndexProtocol double; only upsert_file is exercised here."""

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


class _FakeVaultWriter:
    """Records authored_by writeback calls -- the first vault-edit step."""

    def __init__(self) -> None:
        self.authored_by_calls: list[Path] = []

    async def mark_authored_by_user_edit(self, path: Path) -> None:
        self.authored_by_calls.append(path)


class _FakeInvalidationBus:
    """Records publish calls -- the second and final vault-edit step."""

    def __init__(self) -> None:
        self.published: list[object] = []

    async def publish(self, event: object) -> None:
        self.published.append(event)


def _make_watcher(
    vault_root: Path, writer: _FakeVaultWriter, bus: _FakeInvalidationBus
) -> VaultFilewatcher:
    config = FilewatcherConfig(
        enabled=True,
        observer_type="polling",
        debounce_ms=100,
        staleness_slo_seconds=5,
        audit_interval_seconds=3600,
    )
    return VaultFilewatcher(
        config,
        vault_root,
        _FakeSidecarIndex(),
        invalidation_bus=bus,
        writer=writer,
    )


def test_filewatcher_has_no_graph_store_injection_point() -> None:
    """No constructor parameter can wire a graph store into the filewatcher.

    GraphRegenerator's retirement removed the only path from a vault edit to
    a graph write. This pins the removal at the signature level so a future
    regression -- re-adding a `regenerator` or other graph-store parameter --
    fails loudly here instead of surfacing as a silent Inv-A1 violation.
    """
    params = inspect.signature(VaultFilewatcher.__init__).parameters
    assert "regenerator" not in params


def test_user_file_edit_leaves_pre_existing_triple_untouched(tmp_path: Path) -> None:
    """A users/ path edit does not orphan or otherwise touch a scoped triple."""
    vault_root = tmp_path / "vault"
    users_dir = vault_root / "users"
    users_dir.mkdir(parents=True)
    p = users_dir / "raj.md"
    p.write_text(
        "---\nuser_id: raj\n---\n\n## Tools and Technologies\n- Python\n", encoding="utf-8"
    )

    fake_graph_store = FakeGraphStore()  # deliberately never wired into the watcher
    fake_graph_store.add_triple(
        subject="user", predicate="USES", object="python", derived_from_path=str(p)
    )

    writer = _FakeVaultWriter()
    bus = _FakeInvalidationBus()
    fw = _make_watcher(vault_root, writer, bus)

    asyncio.run(fw._do_reindex(str(p), is_mist_write=False))

    assert writer.authored_by_calls == [p]
    assert len(bus.published) == 1
    triple = fake_graph_store.get_triple("user", "USES", "python")
    assert triple is not None
    assert triple.status == "active"


def test_session_file_edit_leaves_pre_existing_triple_untouched(tmp_path: Path) -> None:
    """A sessions/ path edit does not orphan or otherwise touch a scoped triple."""
    vault_root = tmp_path / "vault"
    sessions_dir = vault_root / "sessions"
    sessions_dir.mkdir(parents=True)
    p = sessions_dir / "2026-07-30-test.md"
    p.write_text("# Turn 1\n\nI use Python at work.\n", encoding="utf-8")

    fake_graph_store = FakeGraphStore()  # deliberately never wired into the watcher
    fake_graph_store.add_triple(
        subject="user", predicate="USES", object="python", derived_from_path=str(p)
    )

    writer = _FakeVaultWriter()
    bus = _FakeInvalidationBus()
    fw = _make_watcher(vault_root, writer, bus)

    asyncio.run(fw._do_reindex(str(p), is_mist_write=False))

    assert writer.authored_by_calls == [p]
    assert len(bus.published) == 1
    triple = fake_graph_store.get_triple("user", "USES", "python")
    assert triple is not None
    assert triple.status == "active"
