"""Tests for backend.memory.index_db: opening `memory_index.db`, checking what built it, and
rebuilding it in place when something else did.
"""

import logging
import sqlite3
import subprocess  # nosec B404 - runs this interpreter on a fixed script, to crash mid-open
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import sqlite_vec

from backend.errors import MemoryIndexError
from backend.memory.index_db import (
    META_EMBEDDING_MODEL_ID,
    META_INDEX_STATE,
    META_REBUILD_EMBEDDING_MODEL_ID,
    META_REBUILD_SCHEMA_VERSION,
    META_SCHEMA_VERSION,
    SCHEMA_DDL,
    SCHEMA_VERSION,
    STATE_REBUILDING,
    IndexRebuild,
    degraded_reason,
    get_meta,
    open_index,
    open_reader,
    set_meta,
    write_transaction,
)

_MODEL = "all-MiniLM-L6-v2"
_SCHEMA_OBJECTS = {
    "facts",
    "facts_fts",
    "facts_vec",
    "facets",
    "facets_fts",
    "facets_vec",
    "index_meta",
}


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
            "object_id, object_name, text, confidence, content_hash, view_text) "
            "VALUES (?, 'fk', 'user', 'User', 'USES', 'kestrel', 'Kestrel', 'User uses Kestrel.', "
            "0.9, 'h', 'You use Kestrel.')",
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


def test_schema_version_is_four() -> None:
    # 2 added the entity types and the FTS type terms; 3 added the embedded view text; 4 added
    # the facet tables.
    assert SCHEMA_VERSION == 4


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_facts_carry_both_entity_types(writer: sqlite3.Connection) -> None:
    columns = _columns(writer, "facts")

    assert {"subject_type", "object_type"} <= set(columns)


def test_facts_carry_the_view_text_as_their_last_column(writer: sqlite3.Connection) -> None:
    # Last, because `IndexedFact` is built from the row positionally and ends with it.
    assert _columns(writer, "facts")[-1] == "view_text"
    not_null = {row[1]: row[3] for row in writer.execute("PRAGMA table_info(facts)")}
    assert not_null["view_text"] == 1


def test_fts_mirror_has_a_type_terms_column(writer: sqlite3.Connection) -> None:
    assert _columns(writer, "facts_fts") == ["text", "subject_name", "object_name", "type_terms"]


def test_facets_carry_their_key_group_view_and_hash(writer: sqlite3.Connection) -> None:
    info = {row[1]: (row[2], row[3]) for row in writer.execute("PRAGMA table_info(facets)")}

    assert list(info) == [
        "id",
        "facet_key",
        "subject_id",
        "subject_name",
        "subject_type",
        "rel_type",
        "object_type",
        "view_text",
        "content_hash",
    ]
    assert info["view_text"] == ("TEXT", 1)
    assert info["content_hash"] == ("TEXT", 1)
    unique = [
        row
        for row in writer.execute("PRAGMA index_list(facets)")
        if row[2] == 1
        and [col[2] for col in writer.execute(f"PRAGMA index_info({row[1]})")] == ["facet_key"]
    ]
    assert unique


def test_the_facet_fts_mirror_holds_the_view_the_name_and_the_type_terms(
    writer: sqlite3.Connection,
) -> None:
    assert _columns(writer, "facets_fts") == ["view_text", "subject_name", "type_terms"]
    (sql,) = writer.execute("SELECT sql FROM sqlite_master WHERE name = 'facets_fts'").fetchone()
    assert "porter unicode61" in sql


def test_the_facet_vectors_are_384_d_by_cosine(writer: sqlite3.Connection) -> None:
    (sql,) = writer.execute("SELECT sql FROM sqlite_master WHERE name = 'facets_vec'").fetchone()

    assert "vec0" in sql and "float[384]" in sql and "distance_metric=cosine" in sql


def test_sibling_lookup_is_indexed(writer: sqlite3.Connection) -> None:
    plan = " ".join(
        str(row[-1])
        for row in writer.execute(
            "EXPLAIN QUERY PLAN SELECT row_key FROM facts "
            "WHERE subject_id = ? AND rel_type = ? AND source_utterance_id = ?",
            ("user", "USES", "utt-1"),
        )
    )

    assert "idx_facts_siblings" in plan


# The version 1 tables, as the schema 1 code created them.
_V1_DDL = (
    "CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE facts (id INTEGER PRIMARY KEY, row_key TEXT NOT NULL UNIQUE, "
    "fact_key TEXT NOT NULL, subject_id TEXT NOT NULL, subject_name TEXT NOT NULL, "
    "rel_type TEXT NOT NULL, object_id TEXT NOT NULL, object_name TEXT NOT NULL, "
    "text TEXT NOT NULL, valid_from TEXT, valid_to TEXT, recorded_at TEXT, "
    "confidence REAL NOT NULL, source_utterance_id TEXT, content_hash TEXT NOT NULL)",
    "CREATE VIRTUAL TABLE facts_fts USING fts5(text, subject_name, object_name, "
    "tokenize='porter unicode61')",
)


def _write_v1_file(path: Path) -> None:
    conn = sqlite3.connect(path)
    for statement in _V1_DDL:
        conn.execute(statement)
    conn.execute(
        "INSERT INTO facts (row_key, fact_key, subject_id, subject_name, rel_type, object_id, "
        "object_name, text, confidence, content_hash) "
        "VALUES ('rk-v1', 'fk', 'user', 'User', 'USES', 'vim', 'Vim', 'User uses Vim.', 0.9, 'h')"
    )
    conn.execute("INSERT INTO index_meta VALUES ('schema_version', '1')")
    conn.execute("INSERT INTO index_meta VALUES ('embedding_model_id', ?)", (_MODEL,))
    conn.commit()
    conn.close()


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == "backend.memory.index_db"
    ]


def _meta(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM index_meta").fetchall())


# -- the in-place rebuild ----------------------------------------------------------------------


def test_a_version_1_file_is_rebuilt_in_place_to_this_schema(
    db_path: Path, opened: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    _write_v1_file(db_path)

    with caplog.at_level(logging.WARNING, logger="backend.memory.index_db"):
        conn = opened(open_index(db_path, embedding_model_id=_MODEL))

    assert {"subject_type", "object_type"} <= set(_columns(conn, "facts"))
    assert _columns(conn, "facts_fts")[-1] == "type_terms"
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert _schema_names(conn) >= _SCHEMA_OBJECTS
    (warning,) = _warnings(caplog)
    assert "schema 1" in warning and f"schema {SCHEMA_VERSION}" in warning
    assert _MODEL in warning


def _write_v2_file(path: Path) -> None:
    """A schema 2 file: today's tables without `facts.view_text`, holding one row."""
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute("CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE facts (id INTEGER PRIMARY KEY, row_key TEXT NOT NULL UNIQUE, "
        "fact_key TEXT NOT NULL, subject_id TEXT NOT NULL, subject_name TEXT NOT NULL, "
        "subject_type TEXT, rel_type TEXT NOT NULL, object_id TEXT NOT NULL, "
        "object_name TEXT NOT NULL, object_type TEXT, text TEXT NOT NULL, valid_from TEXT, "
        "valid_to TEXT, recorded_at TEXT, confidence REAL NOT NULL, source_utterance_id TEXT, "
        "content_hash TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE facts_fts USING fts5(text, subject_name, object_name, type_terms, "
        "tokenize='porter unicode61')"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE facts_vec USING vec0(embedding float[384] distance_metric=cosine)"
    )
    conn.execute(
        "INSERT INTO facts (row_key, fact_key, subject_id, subject_name, rel_type, object_id, "
        "object_name, text, confidence, content_hash) "
        "VALUES ('rk-v2', 'fk', 'user', 'User', 'USES', 'vim', 'Vim', 'User uses Vim.', 0.9, 'h')"
    )
    conn.execute("INSERT INTO index_meta VALUES ('schema_version', '2')")
    conn.execute("INSERT INTO index_meta VALUES ('embedding_model_id', ?)", (_MODEL,))
    conn.commit()
    conn.close()


def test_a_version_2_file_is_rebuilt_in_place_with_the_view_text(
    db_path: Path, opened: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    _write_v2_file(db_path)
    inode = db_path.stat().st_ino

    with caplog.at_level(logging.WARNING, logger="backend.memory.index_db"):
        conn = opened(open_index(db_path, embedding_model_id=_MODEL))

    assert "view_text" in _columns(conn, "facts")
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert get_meta(conn, META_INDEX_STATE) == STATE_REBUILDING
    assert get_meta(conn, META_REBUILD_SCHEMA_VERSION) == "4"
    assert db_path.stat().st_ino == inode
    (warning,) = _warnings(caplog)
    assert "schema 2" in warning and "schema 4" in warning


def _write_v3_file(path: Path) -> None:
    """A schema 3 file: the fact tables with `view_text`, no facet tables, one row."""
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.execute("CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE facts (id INTEGER PRIMARY KEY, row_key TEXT NOT NULL UNIQUE, "
        "fact_key TEXT NOT NULL, subject_id TEXT NOT NULL, subject_name TEXT NOT NULL, "
        "subject_type TEXT, rel_type TEXT NOT NULL, object_id TEXT NOT NULL, "
        "object_name TEXT NOT NULL, object_type TEXT, text TEXT NOT NULL, valid_from TEXT, "
        "valid_to TEXT, recorded_at TEXT, confidence REAL NOT NULL, source_utterance_id TEXT, "
        "content_hash TEXT NOT NULL, view_text TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE facts_fts USING fts5(text, subject_name, object_name, type_terms, "
        "tokenize='porter unicode61')"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE facts_vec USING vec0(embedding float[384] distance_metric=cosine)"
    )
    conn.execute(
        "INSERT INTO facts (row_key, fact_key, subject_id, subject_name, rel_type, object_id, "
        "object_name, text, confidence, content_hash, view_text) "
        "VALUES ('rk-v3', 'fk', 'user', 'User', 'USES', 'vim', 'Vim', 'User uses Vim.', 0.9, 'h', "
        "'You use Vim.')"
    )
    conn.execute("INSERT INTO index_meta VALUES ('schema_version', '3')")
    conn.execute("INSERT INTO index_meta VALUES ('embedding_model_id', ?)", (_MODEL,))
    conn.commit()
    conn.close()


def test_a_version_3_file_is_rebuilt_in_place_with_the_facet_tables(
    db_path: Path, opened: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    _write_v3_file(db_path)
    inode = db_path.stat().st_ino

    with caplog.at_level(logging.WARNING, logger="backend.memory.index_db"):
        conn = opened(open_index(db_path, embedding_model_id=_MODEL))

    assert {"facets", "facets_fts", "facets_vec"} <= _schema_names(conn)
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM facets").fetchone()[0] == 0
    assert get_meta(conn, META_INDEX_STATE) == STATE_REBUILDING
    assert get_meta(conn, META_REBUILD_SCHEMA_VERSION) == "4"
    assert get_meta(conn, META_SCHEMA_VERSION) == "3"
    assert db_path.stat().st_ino == inode
    (warning,) = _warnings(caplog)
    assert "schema 3" in warning and "schema 4" in warning


def test_a_stale_model_rebuilds_and_stores_rebuilding_with_the_old_versions(
    db_path: Path, opened: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    first = open_index(db_path, embedding_model_id=_MODEL)
    _insert_fact(first, "rk-1")
    with write_transaction(first):
        set_meta(first, "facts_refreshed_at", "2026-10-22T18:00:00+00:00")
    first.close()

    with caplog.at_level(logging.WARNING, logger="backend.memory.index_db"):
        conn = opened(open_index(db_path, embedding_model_id="other-model"))

    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    # Only the rebuild's final commit stores the new versions; until then the old ones stand.
    assert _meta(conn) == {
        META_SCHEMA_VERSION: str(SCHEMA_VERSION),
        META_EMBEDDING_MODEL_ID: _MODEL,
        META_INDEX_STATE: STATE_REBUILDING,
        META_REBUILD_SCHEMA_VERSION: str(SCHEMA_VERSION),
        META_REBUILD_EMBEDDING_MODEL_ID: "other-model",
    }
    (warning,) = _warnings(caplog)
    assert _MODEL in warning and "other-model" in warning


def test_a_stale_schema_version_rebuilds_naming_both_versions(
    db_path: Path, opened: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    conn = open_index(db_path, embedding_model_id=_MODEL)
    _insert_fact(conn, "rk-1")
    with write_transaction(conn):
        set_meta(conn, META_SCHEMA_VERSION, str(SCHEMA_VERSION + 1))
    conn.close()

    with caplog.at_level(logging.WARNING, logger="backend.memory.index_db"):
        rebuilt = opened(open_index(db_path, embedding_model_id=_MODEL))

    assert rebuilt.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert get_meta(rebuilt, META_INDEX_STATE) == STATE_REBUILDING
    (warning,) = _warnings(caplog)
    assert f"schema {SCHEMA_VERSION + 1}" in warning and f"schema {SCHEMA_VERSION}" in warning


def test_the_rebuild_keeps_the_same_file(db_path: Path, opened: Callable) -> None:
    open_index(db_path, embedding_model_id=_MODEL).close()
    inode = db_path.stat().st_ino

    opened(open_index(db_path, embedding_model_id="other-model"))

    assert db_path.stat().st_ino == inode


def test_a_matching_index_is_opened_untouched(
    db_path: Path, opened: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    first = open_index(db_path, embedding_model_id=_MODEL)
    _insert_fact(first, "rk-1")
    meta_before = _meta(first)
    first.close()

    with caplog.at_level(logging.WARNING, logger="backend.memory.index_db"):
        second = opened(open_index(db_path, embedding_model_id=_MODEL))

    assert second.total_changes == 0
    assert _meta(second) == meta_before
    assert second.execute("SELECT row_key FROM facts").fetchall() == [("rk-1",)]
    assert _warnings(caplog) == []
    assert degraded_reason(second) is None


def test_reads_during_the_rebuild_report_index_rebuilding(db_path: Path, opened: Callable) -> None:
    open_index(db_path, embedding_model_id=_MODEL).close()
    opened(open_index(db_path, embedding_model_id="other-model"))

    reader = opened(open_reader(db_path, embedding_model_id="other-model"))

    assert degraded_reason(reader) == "index_rebuilding"


def test_a_reader_during_the_rebuild_needs_the_rebuild_target(
    db_path: Path, opened: Callable
) -> None:
    open_index(db_path, embedding_model_id=_MODEL).close()
    opened(open_index(db_path, embedding_model_id="other-model"))

    # The stored model is still the old one, but the tables are being refilled for the new.
    with pytest.raises(MemoryIndexError, match="other-model"):
        open_reader(db_path, embedding_model_id=_MODEL)


def test_a_reader_open_across_the_rebuild_keeps_working(db_path: Path, opened: Callable) -> None:
    first = open_index(db_path, embedding_model_id=_MODEL)
    _insert_fact(first, "rk-1")
    first.close()
    reader = opened(open_reader(db_path, embedding_model_id=_MODEL))
    reader.execute("BEGIN")
    assert reader.execute("SELECT row_key FROM facts").fetchall() == [("rk-1",)]

    opened(open_index(db_path, embedding_model_id="other-model"))

    # The read transaction keeps its snapshot; the next one sees the rebuilt, empty tables.
    assert reader.execute("SELECT row_key FROM facts").fetchall() == [("rk-1",)]
    reader.execute("COMMIT")
    assert reader.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert reader.execute("SELECT count(*) FROM facts_vec").fetchone()[0] == 0
    assert reader.execute("SELECT count(*) FROM facts_fts").fetchone()[0] == 0
    assert reader.execute("SELECT count(*) FROM facets").fetchone()[0] == 0
    assert degraded_reason(reader) == "index_rebuilding"


def test_completing_the_rebuild_stores_the_new_versions(db_path: Path, opened: Callable) -> None:
    open_index(db_path, embedding_model_id=_MODEL).close()
    writer = opened(open_index(db_path, embedding_model_id="other-model"))
    reader = opened(open_reader(db_path, embedding_model_id="other-model"))

    assert IndexRebuild(db_path, embedding_model_id="other-model").complete() is True

    assert _meta(writer) == {
        META_SCHEMA_VERSION: str(SCHEMA_VERSION),
        META_EMBEDDING_MODEL_ID: "other-model",
    }
    assert degraded_reason(reader) is None


def test_completing_when_no_rebuild_is_pending_changes_nothing(
    writer: sqlite3.Connection, db_path: Path
) -> None:
    meta_before = _meta(writer)

    assert IndexRebuild(db_path, embedding_model_id=_MODEL).complete() is False

    assert _meta(writer) == meta_before


def test_an_index_rebuilt_and_completed_reopens_untouched(
    db_path: Path, opened: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    open_index(db_path, embedding_model_id=_MODEL).close()
    open_index(db_path, embedding_model_id="other-model").close()
    IndexRebuild(db_path, embedding_model_id="other-model").complete()
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger="backend.memory.index_db"):
        conn = opened(open_index(db_path, embedding_model_id="other-model"))

    assert conn.total_changes == 0
    assert _warnings(caplog) == []


def test_completing_for_another_target_raises(db_path: Path, opened: Callable) -> None:
    open_index(db_path, embedding_model_id=_MODEL).close()
    opened(open_index(db_path, embedding_model_id="other-model"))

    with pytest.raises(MemoryIndexError, match="other-model"):
        IndexRebuild(db_path, embedding_model_id="third-model").complete()


def test_completing_a_missing_file_raises(db_path: Path) -> None:
    with pytest.raises(MemoryIndexError):
        IndexRebuild(db_path, embedding_model_id=_MODEL).complete()

    assert not db_path.exists()


def test_an_in_memory_index_has_no_rebuild() -> None:
    with pytest.raises(ValueError, match="memory"):
        IndexRebuild(":memory:", embedding_model_id=_MODEL)


# A process that dies inside the drop transaction: the derived tables are already dropped
# when the stand-in for the table creation kills the process without a commit.
_CRASH_IN_DROP = """
import os
import sys

from backend.memory import index_db


def _crash(conn):
    dropped = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'facts'").fetchone() is None
    os._exit(3 if dropped and conn.in_transaction else 4)


index_db._create_fact_tables = _crash
index_db.open_index(sys.argv[1], embedding_model_id=sys.argv[2])
os._exit(5)
"""

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_a_crash_inside_the_drop_transaction_leaves_the_old_version(
    db_path: Path, opened: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    first = open_index(db_path, embedding_model_id=_MODEL)
    _insert_fact(first, "rk-1")
    first.close()

    crashed = subprocess.run(
        [sys.executable, "-c", _CRASH_IN_DROP, str(db_path), "other-model"],
        cwd=_REPO_ROOT,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert crashed.returncode == 3, crashed.stderr.decode()
    plain = sqlite3.connect(db_path)
    try:
        assert _meta(plain) == {
            META_SCHEMA_VERSION: str(SCHEMA_VERSION),
            META_EMBEDDING_MODEL_ID: _MODEL,
        }
        assert plain.execute("SELECT row_key FROM facts").fetchall() == [("rk-1",)]
    finally:
        plain.close()
    # The next open still finds the old model and rebuilds again.
    with caplog.at_level(logging.WARNING, logger="backend.memory.index_db"):
        conn = opened(open_index(db_path, embedding_model_id="other-model"))
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert get_meta(conn, META_INDEX_STATE) == STATE_REBUILDING
    assert len(_warnings(caplog)) == 1


def test_a_crash_during_the_rescan_rebuilds_again_at_the_next_open(
    db_path: Path, opened: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    open_index(db_path, embedding_model_id=_MODEL).close()
    partial = open_index(db_path, embedding_model_id="other-model")
    _insert_fact(partial, "rk-partial")  # the rescan wrote part of the facts, then died
    partial.close()
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger="backend.memory.index_db"):
        conn = opened(open_index(db_path, embedding_model_id="other-model"))

    # A partial index never opens as valid: the tables are emptied again for a full rescan.
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert get_meta(conn, META_INDEX_STATE) == STATE_REBUILDING
    assert get_meta(conn, META_EMBEDDING_MODEL_ID) == _MODEL
    (warning,) = _warnings(caplog)
    assert _MODEL in warning and "other-model" in warning


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
