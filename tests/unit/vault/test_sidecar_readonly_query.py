"""`query_hybrid_readonly`: a turn's vault query never waits behind the file watcher's reindex.

The sidecar's single `vault-sidecar` worker serves both the filewatcher's `upsert_file`, which
embeds every chunk inside one write transaction, and the retriever's queries. The read path
queries instead on a second, read-only connection with its own `vault-sidecar-read` thread, and
the database runs in WAL so that connection reads the last committed snapshot while a write
transaction is open.
"""

import asyncio
import inspect
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from backend.errors import SidecarIndexError
from backend.interfaces import SidecarIndexProtocol
from backend.knowledge.config import SidecarIndexConfig
from backend.vault.sidecar_index import VaultSidecarIndex
from tests.mocks.embeddings import FakeEmbeddingGenerator

_EMBEDDING_DIMENSION = 384
_WAIT_S = 5.0
_QUERY = [1.0] + [0.0] * (_EMBEDDING_DIMENSION - 1)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_index(db_path: str) -> VaultSidecarIndex:
    config = SidecarIndexConfig(
        enabled=True,
        db_path=db_path,
        embedding_dimension=_EMBEDDING_DIMENSION,
        heading_context_weight=0.3,
        chunk_max_chars=6000,
        rebuild_on_startup=False,
    )
    idx = VaultSidecarIndex(config, FakeEmbeddingGenerator(dimension=_EMBEDDING_DIMENSION))
    idx.initialize()
    return idx


@pytest.fixture()
def sidecar(tmp_path):
    """A seeded VaultSidecarIndex over a tmp file database."""
    idx = _make_index(str(tmp_path / "sidecar.db"))
    idx.upsert_file("projects/kestrel.md", "Kestrel is the rust rewrite of the parser.", 1)
    idx.upsert_file("notes/garden.md", "Tomatoes and basil in the raised bed.", 2)
    yield idx
    idx.close()


def test_the_database_runs_in_wal(sidecar, tmp_path):
    probe = sqlite3.connect(str(tmp_path / "sidecar.db"))
    try:
        assert probe.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        probe.close()


@pytest.mark.asyncio
async def test_readonly_query_returns_what_query_hybrid_returns(sidecar):
    expected = sidecar.query_hybrid(_QUERY, "kestrel banana", k=5, match="any")

    rows = await sidecar.query_hybrid_readonly(_QUERY, "kestrel banana", k=5, match="any")

    assert rows == expected
    assert any(r["path"] == "projects/kestrel.md" and r["fts_rank"] for r in rows)


@pytest.mark.asyncio
async def test_readonly_query_returns_while_the_writer_thread_holds_a_write_transaction(
    sidecar,
):
    holding = threading.Event()
    release = threading.Event()

    def hold_a_write_transaction() -> None:
        conn = sidecar._conn
        # EXCLUSIVE keeps every rollback-journal reader out; under WAL it admits readers, so
        # this test fails unless the database is in WAL.
        conn.execute("BEGIN EXCLUSIVE")
        try:
            conn.execute("DELETE FROM vault_chunks_fts")
            conn.execute("DELETE FROM vault_chunks")
            holding.set()
            release.wait(_WAIT_S)
        finally:
            conn.execute("ROLLBACK")

    writer = asyncio.ensure_future(sidecar.run_async(hold_a_write_transaction))
    try:
        assert await asyncio.to_thread(holding.wait, _WAIT_S)
        queued_on_writer = asyncio.ensure_future(
            sidecar.run_async(sidecar.query_hybrid, _QUERY, "kestrel", 5)
        )

        rows = await asyncio.wait_for(
            sidecar.query_hybrid_readonly(_QUERY, "kestrel", k=5), timeout=_WAIT_S
        )

        assert not queued_on_writer.done()
        assert any(r["path"] == "projects/kestrel.md" and r["fts_rank"] == 1 for r in rows)
    finally:
        release.set()
        await writer
    assert await queued_on_writer == rows


@pytest.mark.asyncio
async def test_readonly_query_runs_on_the_vault_sidecar_read_thread(sidecar, monkeypatch):
    seen: list[str] = []
    hybrid = sidecar._hybrid

    def recording(conn, *args, **kwargs):
        seen.append(threading.current_thread().name)
        return hybrid(conn, *args, **kwargs)

    monkeypatch.setattr(sidecar, "_hybrid", recording)

    await sidecar.query_hybrid_readonly(_QUERY, "kestrel")

    assert len(seen) == 1
    assert seen[0].startswith("vault-sidecar-read")


@pytest.mark.asyncio
async def test_the_read_connection_refuses_writes(sidecar):
    def write_on_reader() -> None:
        sidecar._reader.execute("DELETE FROM vault_chunks")

    loop = asyncio.get_running_loop()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        await loop.run_in_executor(sidecar._read_worker, write_on_reader)


@pytest.mark.asyncio
async def test_readonly_query_refuses_an_unknown_match_mode(sidecar):
    with pytest.raises(ValueError, match="match"):
        await sidecar.query_hybrid_readonly(_QUERY, "kestrel", match="some")


@pytest.mark.asyncio
async def test_close_releases_the_read_connection_and_its_thread(sidecar):
    await sidecar.query_hybrid_readonly(_QUERY, "kestrel")

    sidecar.close()
    sidecar.close()

    assert sidecar._reader is None
    assert sidecar._read_worker is None
    with pytest.raises(SidecarIndexError):
        await sidecar.query_hybrid_readonly(_QUERY, "kestrel")


@pytest.mark.asyncio
async def test_readonly_query_before_initialize_raises(tmp_path):
    config = SidecarIndexConfig(db_path=str(tmp_path / "sidecar.db"))
    idx = VaultSidecarIndex(config, FakeEmbeddingGenerator(dimension=_EMBEDDING_DIMENSION))

    with pytest.raises(SidecarIndexError):
        await idx.query_hybrid_readonly(_QUERY, "kestrel")


@pytest.mark.asyncio
async def test_an_in_memory_index_serves_readonly_queries_on_its_writer():
    idx = _make_index(":memory:")
    try:
        idx.upsert_file("projects/kestrel.md", "Kestrel is the rust rewrite of the parser.", 1)

        rows = await idx.query_hybrid_readonly(_QUERY, "kestrel")

        assert idx._reader is None
        assert [r["path"] for r in rows if r["fts_rank"]] == ["projects/kestrel.md"]
    finally:
        idx.close()


# A second process standing in for the running backend: it opens the sidecar, writes to it and
# keeps it open, so its `-wal` and `-shm` stay on disk with live content until it is told to go.
_BACKEND = """
import sys
from backend.knowledge.config import SidecarIndexConfig
from backend.vault.sidecar_index import VaultSidecarIndex
from tests.mocks.embeddings import FakeEmbeddingGenerator

idx = VaultSidecarIndex(SidecarIndexConfig(db_path=sys.argv[1]), FakeEmbeddingGenerator(dimension=384))
idx.initialize()
for n in range(20):
    idx.upsert_file(f"old/note-{n}.md", f"Old note {n} about the kestrel parser.", n)
print("ready", flush=True)
sys.stdin.readline()
idx.close()
"""


def test_a_rebuild_that_unlinks_only_the_db_file_gets_an_intact_database(tmp_path):
    # The steps of `mist_admin vault-rebuild` (scripts/mist_admin.py) while the backend holds
    # the index open: only the `.db` is unlinked, so the old `-wal` and `-shm` stay beside the
    # new file, and replaying them into it would corrupt it.
    db_path = tmp_path / "sidecar.db"
    backend = subprocess.Popen(
        [sys.executable, "-c", _BACKEND, str(db_path)],
        cwd=_REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert backend.stdout.readline().strip() == "ready"
        assert (tmp_path / "sidecar.db-wal").exists()
        db_path.unlink()

        rebuilt = _make_index(str(db_path))
        try:
            rebuilt.upsert_file("notes/garden.md", "Tomatoes and basil in the raised bed.", 2)
            rebuilt.upsert_file("projects/kestrel.md", "Kestrel is the rust rewrite.", 3)
            check = rebuilt._conn.execute("PRAGMA integrity_check").fetchall()
            assert [row[0] for row in check] == ["ok"]
            assert sorted(rebuilt.distinct_paths()) == ["notes/garden.md", "projects/kestrel.md"]
        finally:
            rebuilt.close()
    finally:
        backend.communicate("stop\n", timeout=_WAIT_S)


def test_an_existing_database_keeps_its_wal(tmp_path):
    db_path = tmp_path / "sidecar.db"
    first = _make_index(str(db_path))
    second = None
    try:
        first.upsert_file("projects/kestrel.md", "Kestrel is the rust rewrite.", 1)
        wal = tmp_path / "sidecar.db-wal"
        assert wal.stat().st_size > 0

        second = _make_index(str(db_path))

        assert wal.exists()
        assert second.distinct_paths() == ["projects/kestrel.md"]
    finally:
        if second is not None:
            second.close()
        first.close()


def test_a_failing_snapshot_commit_raises_sidecar_index_error(sidecar):
    class CommitFails:
        """A connection whose COMMIT fails, as a raw sqlite3 error would surface."""

        def __init__(self, conn):
            self._conn = conn

        @property
        def in_transaction(self):
            return self._conn.in_transaction

        def execute(self, sql, *args):
            if sql == "COMMIT":
                self._conn.execute("ROLLBACK")
                raise sqlite3.OperationalError("disk I/O error")
            return self._conn.execute(sql, *args)

    def run() -> list[dict]:
        return sidecar._hybrid_in_snapshot(
            CommitFails(sidecar._reader), _QUERY, "kestrel", 5, 60, "all"
        )

    with pytest.raises(SidecarIndexError, match="disk I/O error"):
        run()


def test_the_protocol_declares_query_hybrid_readonly_with_match():
    method = SidecarIndexProtocol.query_hybrid_readonly
    assert inspect.iscoroutinefunction(method)
    assert inspect.signature(method).parameters["match"].default == "all"


# -- path_texts_readonly (mist-read-path T21): a note's whole text, for read-time suppression --

_SECTIONED = (
    "# Pottery\n\nIntro line.\n\n## Thursdays\n\nWheel course.\n\n## Studio\n\nKiln Street Pottery."
)


@pytest.mark.asyncio
async def test_path_texts_joins_every_chunk_of_each_path_in_chunk_order(sidecar):
    sidecar.upsert_file("hobbies/pottery.md", _SECTIONED, 3)
    stored = sidecar._conn.execute(
        "SELECT content FROM vault_chunks WHERE path = ? ORDER BY chunk_id",
        ("hobbies/pottery.md",),
    ).fetchall()
    assert len(stored) > 1  # the file-level chunk and the heading-block chunks

    texts = await sidecar.path_texts_readonly(["hobbies/pottery.md", "notes/garden.md"])

    assert texts == {
        "hobbies/pottery.md": "\n\n".join(row[0] for row in stored),
        "notes/garden.md": "Tomatoes and basil in the raised bed.",
    }
    assert "Kiln Street Pottery" in texts["hobbies/pottery.md"]


@pytest.mark.asyncio
async def test_path_texts_leaves_out_a_path_the_sidecar_does_not_hold(sidecar):
    assert await sidecar.path_texts_readonly(["notes/gone.md"]) == {}
    assert await sidecar.path_texts_readonly([]) == {}


@pytest.mark.asyncio
async def test_path_texts_runs_on_the_read_thread_and_returns_during_a_write(sidecar):
    holding = threading.Event()
    release = threading.Event()
    seen: list[str] = []
    rows = sidecar._path_texts

    def recording(conn, paths):
        seen.append(threading.current_thread().name)
        return rows(conn, paths)

    sidecar._path_texts = recording

    def hold_a_write_transaction() -> None:
        conn = sidecar._conn
        conn.execute("BEGIN EXCLUSIVE")
        try:
            conn.execute("DELETE FROM vault_chunks")
            holding.set()
            release.wait(_WAIT_S)
        finally:
            conn.execute("ROLLBACK")

    writer = asyncio.ensure_future(sidecar.run_async(hold_a_write_transaction))
    try:
        assert await asyncio.to_thread(holding.wait, _WAIT_S)
        texts = await asyncio.wait_for(
            sidecar.path_texts_readonly(["notes/garden.md"]), timeout=_WAIT_S / 2
        )
    finally:
        release.set()
        await writer

    assert texts == {"notes/garden.md": "Tomatoes and basil in the raised bed."}
    assert len(seen) == 1 and seen[0].startswith("vault-sidecar-read")


@pytest.mark.asyncio
async def test_path_texts_on_an_in_memory_index_reads_on_the_writer():
    idx = _make_index(":memory:")
    try:
        idx.upsert_file("notes/garden.md", "Tomatoes and basil in the raised bed.", 2)

        texts = await idx.path_texts_readonly(["notes/garden.md"])

        assert idx._reader is None
        assert texts == {"notes/garden.md": "Tomatoes and basil in the raised bed."}
    finally:
        idx.close()


@pytest.mark.asyncio
async def test_path_texts_before_initialize_raises(tmp_path):
    config = SidecarIndexConfig(db_path=str(tmp_path / "sidecar.db"))
    idx = VaultSidecarIndex(config, FakeEmbeddingGenerator(dimension=_EMBEDDING_DIMENSION))

    with pytest.raises(SidecarIndexError):
        await idx.path_texts_readonly(["notes/garden.md"])


def test_the_protocol_declares_path_texts_readonly():
    method = SidecarIndexProtocol.path_texts_readonly
    assert inspect.iscoroutinefunction(method)
    assert list(inspect.signature(method).parameters) == ["self", "paths"]
