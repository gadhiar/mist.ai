"""Tests for backend.memory.index_db: opening `memory_index.db` and checking what built it."""

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from backend.errors import MemoryIndexError
from backend.memory.index_db import (
    META_EMBEDDING_MODEL_ID,
    META_SCHEMA_VERSION,
    SCHEMA_DDL,
    SCHEMA_VERSION,
    get_meta,
    open_index,
    open_reader,
    set_meta,
    write_transaction,
)

_MODEL = "all-MiniLM-L6-v2"
_SCHEMA_OBJECTS = {"facts", "facts_fts", "facts_vec", "index_meta"}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path of the tmp `memory_index.db`."""
    return tmp_path / "memory_index.db"


@pytest.fixture
def opened() -> Iterator[Callable[[sqlite3.Connection], sqlite3.Connection]]:
    """Registers each connection a test opens and closes them all at teardown."""
    connections: list[sqlite3.Connection] = []

    def register(conn: sqlite3.Connection) -> sqlite3.Connection:
        connections.append(conn)
        return conn

    yield register
    for conn in connections:
        conn.close()


@pytest.fixture
def writer(
    db_path: Path, opened: Callable[[sqlite3.Connection], sqlite3.Connection]
) -> sqlite3.Connection:
    """The writer connection to the tmp index."""
    return opened(open_index(db_path, embedding_model_id=_MODEL))


def _schema_names(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}


def _insert_fact(conn: sqlite3.Connection, row_key: str) -> None:
    with write_transaction(conn):
        conn.execute(
            "INSERT INTO facts (row_key, fact_key, subject_id, subject_name, rel_type, "
            "object_id, object_name, text, confidence, content_hash) "
            "VALUES (?, 'fk', 'user', 'User', 'USES', 'kestrel', 'Kestrel', 'User uses Kestrel.', "
            "0.9, 'h')",
            (row_key,),
        )


# -- the writer --------------------------------------------------------------------------------


def test_opens_the_file_in_wal_mode(writer: sqlite3.Connection) -> None:
    assert writer.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_loads_sqlite_vec(writer: sqlite3.Connection) -> None:
    assert writer.execute("SELECT vec_version()").fetchone()[0]


def test_creates_the_fact_tables_and_meta(writer: sqlite3.Connection) -> None:
    assert _schema_names(writer) >= _SCHEMA_OBJECTS


def test_records_schema_version_and_embedding_model_id(writer: sqlite3.Connection) -> None:
    assert get_meta(writer, META_SCHEMA_VERSION) == str(SCHEMA_VERSION)
    assert get_meta(writer, META_EMBEDDING_MODEL_ID) == _MODEL


def test_creates_a_missing_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "memory_index.db"

    open_index(path, embedding_model_id=_MODEL).close()

    assert path.exists()


def test_reopening_keeps_rows_and_schema(db_path: Path, opened: Callable) -> None:
    first = open_index(db_path, embedding_model_id=_MODEL)
    _insert_fact(first, "rk-1")
    names_before = _schema_names(first)
    first.close()

    second = opened(open_index(db_path, embedding_model_id=_MODEL))

    assert second.execute("SELECT row_key FROM facts").fetchall() == [("rk-1",)]
    assert _schema_names(second) == names_before
    assert get_meta(second, META_SCHEMA_VERSION) == str(SCHEMA_VERSION)


def test_every_ddl_statement_is_idempotent() -> None:
    assert SCHEMA_DDL
    assert all("IF NOT EXISTS" in statement for statement in SCHEMA_DDL)


def test_memory_path_opens_without_a_file(opened: Callable) -> None:
    conn = opened(open_index(":memory:", embedding_model_id=_MODEL))

    assert _schema_names(conn) >= _SCHEMA_OBJECTS


def test_raises_when_the_stored_model_id_differs(db_path: Path) -> None:
    open_index(db_path, embedding_model_id=_MODEL).close()

    with pytest.raises(MemoryIndexError, match="other-model"):
        open_index(db_path, embedding_model_id="other-model")


def test_raises_when_the_stored_schema_version_differs(db_path: Path) -> None:
    conn = open_index(db_path, embedding_model_id=_MODEL)
    with write_transaction(conn):
        set_meta(conn, META_SCHEMA_VERSION, str(SCHEMA_VERSION + 1))
    conn.close()

    with pytest.raises(MemoryIndexError, match="schema"):
        open_index(db_path, embedding_model_id=_MODEL)


def _failing_loader(conn: sqlite3.Connection) -> None:
    raise sqlite3.OperationalError("vec0.so: cannot open shared object file")


def test_raises_when_sqlite_vec_cannot_load(db_path: Path) -> None:
    with pytest.raises(MemoryIndexError, match="sqlite_vec"):
        open_index(db_path, embedding_model_id=_MODEL, load_extension=_failing_loader)


def test_raises_on_a_file_that_is_not_a_database(db_path: Path) -> None:
    db_path.write_bytes(b"this is not an sqlite database, just bytes" * 100)

    with pytest.raises(MemoryIndexError):
        open_index(db_path, embedding_model_id=_MODEL)


def test_rejects_an_empty_model_id(db_path: Path) -> None:
    with pytest.raises(ValueError, match="embedding_model_id"):
        open_index(db_path, embedding_model_id="")


def test_meta_round_trips_a_watermark(writer: sqlite3.Connection) -> None:
    with write_transaction(writer):
        set_meta(writer, "facts_refreshed_at", "2026-10-22T18:00:00+00:00")

    assert get_meta(writer, "facts_refreshed_at") == "2026-10-22T18:00:00+00:00"
    assert get_meta(writer, "never_set") is None


def test_write_transaction_rolls_back_on_error(writer: sqlite3.Connection) -> None:
    with pytest.raises(RuntimeError), write_transaction(writer):
        set_meta(writer, "facts_refreshed_at", "lost")
        raise RuntimeError("abort")

    assert get_meta(writer, "facts_refreshed_at") is None


# -- the reader --------------------------------------------------------------------------------


def test_reader_sees_committed_rows_and_sqlite_vec(
    writer: sqlite3.Connection, db_path: Path, opened: Callable
) -> None:
    _insert_fact(writer, "rk-1")

    reader = opened(open_reader(db_path, embedding_model_id=_MODEL))

    assert reader.execute("SELECT row_key FROM facts").fetchall() == [("rk-1",)]
    assert reader.execute("SELECT vec_version()").fetchone()[0]


def test_reader_refuses_writes(writer: sqlite3.Connection, db_path: Path, opened: Callable) -> None:
    reader = opened(open_reader(db_path, embedding_model_id=_MODEL))

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.execute("DELETE FROM facts")


def test_reader_does_not_wait_for_an_open_write_transaction(
    writer: sqlite3.Connection, db_path: Path, opened: Callable
) -> None:
    _insert_fact(writer, "rk-1")
    reader = opened(open_reader(db_path, embedding_model_id=_MODEL))

    with write_transaction(writer):
        writer.execute("DELETE FROM facts")
        seen = reader.execute("SELECT row_key FROM facts").fetchall()

    assert seen == [("rk-1",)]


def test_reader_raises_when_the_file_is_missing(db_path: Path) -> None:
    with pytest.raises(MemoryIndexError):
        open_reader(db_path, embedding_model_id=_MODEL)

    assert not db_path.exists()


def test_reader_raises_when_the_stored_model_id_differs(
    writer: sqlite3.Connection, db_path: Path
) -> None:
    with pytest.raises(MemoryIndexError, match="other-model"):
        open_reader(db_path, embedding_model_id="other-model")


def test_reader_raises_when_sqlite_vec_cannot_load(
    writer: sqlite3.Connection, db_path: Path
) -> None:
    with pytest.raises(MemoryIndexError, match="sqlite_vec"):
        open_reader(db_path, embedding_model_id=_MODEL, load_extension=_failing_loader)


def test_reader_refuses_a_memory_path() -> None:
    with pytest.raises(ValueError, match="memory"):
        open_reader(":memory:", embedding_model_id=_MODEL)
