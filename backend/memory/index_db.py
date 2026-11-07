"""Open `memory_index.db`, the derived index the memory read path serves from.

This module is the contract home of the file: its DDL and `SCHEMA_VERSION` live here and
nowhere else. The file is derived from the graph (and, in Phase B, the event log), so it is
never migrated in place. A file built under another schema version or embedding model raises
MemoryIndexError at open, and the factory rebuilds it.

Tables:

- `facts`: one row per latest-belief fact version, keyed by `row_key` and grouped by
  `fact_key` when read, with the subject's and object's entity types. `id` is the rowid the
  two mirrors share; `idx_facts_siblings` serves `FactIndex.siblings`.
- `facts_fts`: FTS5 over the rendered text, the subject and object names, and the type terms
  of both entity types (`type_terms.type_terms_text`), each stored in `fts.fold` form.
- `facts_vec`: sqlite-vec `vec0` table of the rendered text's embedding, cosine distance.
- `index_meta`: `schema_version`, `embedding_model_id` and the refresh watermarks.

Two connections serve a file: `open_index` gives the writer, which creates and checks the
schema, and `open_reader` a read-only connection for searches, so under WAL a search reads the
last committed snapshot while the writer holds a transaction. Both are opened in autocommit
mode (`isolation_level=None`), so every transaction is explicit: `write_transaction` takes the
write lock up front with `BEGIN IMMEDIATE`, and `read_transaction` gives a multi-statement read
one snapshot.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from backend.errors import MemoryIndexError

logger = logging.getLogger(__name__)

# 2: `facts.subject_type`, `facts.object_type`, `facts_fts.type_terms`, `idx_facts_siblings`.
SCHEMA_VERSION = 2
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2, the embedder the vault sidecar also uses

META_SCHEMA_VERSION = "schema_version"
META_EMBEDDING_MODEL_ID = "embedding_model_id"

MEMORY_PATH = ":memory:"

_BUSY_TIMEOUT_MS = 5000

INDEX_META_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS index_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
)

FACTS_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS facts (
        id                  INTEGER PRIMARY KEY,
        row_key             TEXT NOT NULL UNIQUE,
        fact_key            TEXT NOT NULL,
        subject_id          TEXT NOT NULL,
        subject_name        TEXT NOT NULL,
        subject_type        TEXT,
        rel_type            TEXT NOT NULL,
        object_id           TEXT NOT NULL,
        object_name         TEXT NOT NULL,
        object_type         TEXT,
        text                TEXT NOT NULL,
        valid_from          TEXT,
        valid_to            TEXT,
        recorded_at         TEXT,
        confidence          REAL NOT NULL,
        source_utterance_id TEXT,
        content_hash        TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_facts_fact_key ON facts(fact_key)",
    "CREATE INDEX IF NOT EXISTS idx_facts_siblings "
    "ON facts(subject_id, rel_type, source_utterance_id)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
        text,
        subject_name,
        object_name,
        type_terms,
        tokenize='porter unicode61'
    )
    """,
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(
        embedding float[{EMBEDDING_DIMENSION}] distance_metric=cosine
    )
    """,
)

SCHEMA_DDL: tuple[str, ...] = INDEX_META_DDL + FACTS_DDL

ExtensionLoader = Callable[[sqlite3.Connection], None]


def open_index(
    path: str | os.PathLike[str],
    *,
    embedding_model_id: str,
    load_extension: ExtensionLoader = sqlite_vec.load,
) -> sqlite3.Connection:
    """Open or create `memory_index.db` and check it was built for this schema and model.

    The schema and the meta rows are written in one transaction, so a crash mid-open leaves
    either nothing or a complete schema. The connection allows use from other threads
    (`check_same_thread=False`) because it is opened on one thread and used from the
    `memory-index` executor's threads; FactIndex serialises access to it.

    Args:
        path: File path, or `:memory:` for a private in-memory index (WAL does not apply).
        embedding_model_id: Id of the model that embeds the rows. A file recorded under
            another id holds vectors from another space, so it is refused.
        load_extension: Loads sqlite-vec into a connection; tests inject a failing loader.

    Returns:
        An open connection in autocommit mode with sqlite-vec loaded and WAL enabled.

    Raises:
        ValueError: `embedding_model_id` is empty.
        MemoryIndexError: The file cannot be opened or is not a database, sqlite-vec does not
            load, or the stored schema version or embedding model id differs.
    """
    if not embedding_model_id:
        raise ValueError("embedding_model_id must name the embedding model")
    location = os.fspath(path)
    in_memory = location == MEMORY_PATH
    try:
        if not in_memory:
            Path(location).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(location, check_same_thread=False, isolation_level=None)
    except (sqlite3.Error, OSError) as exc:
        raise MemoryIndexError(f"cannot open memory index at {location}: {exc}") from exc
    try:
        _load_sqlite_vec(conn, load_extension)
        _configure(conn, in_memory=in_memory, location=location)
        _apply_schema(conn, embedding_model_id)
    except MemoryIndexError:
        conn.close()
        raise
    except sqlite3.Error as exc:
        conn.close()
        raise MemoryIndexError(f"cannot open memory index at {location}: {exc}") from exc
    logger.info("memory index open at %s (schema %d)", location, SCHEMA_VERSION)
    return conn


def open_reader(
    path: str | os.PathLike[str],
    *,
    embedding_model_id: str,
    load_extension: ExtensionLoader = sqlite_vec.load,
) -> sqlite3.Connection:
    """Open a read-only second connection to an index `open_index` has already created.

    The fetcher's searches run here so they never queue behind the refresher: under WAL a
    reader keeps reading the last committed snapshot while the writer holds a write
    transaction. The file is opened `mode=ro`, so a missing file is an error rather than a new
    empty database, and a stray write fails.

    Args:
        path: The file the writer opened. `:memory:` is refused: a private in-memory database
            has no second connection, so an in-memory FactIndex reads on its writer instead.
        embedding_model_id: The configured model id; the file must record the same one.
        load_extension: Loads sqlite-vec into a connection; tests inject a failing loader.

    Returns:
        An open read-only connection in autocommit mode with sqlite-vec loaded.

    Raises:
        ValueError: `path` is `:memory:` or `embedding_model_id` is empty.
        MemoryIndexError: The file is missing or not a database, sqlite-vec does not load, or
            the stored schema version or embedding model id differs or is absent.
    """
    if not embedding_model_id:
        raise ValueError("embedding_model_id must name the embedding model")
    location = os.fspath(path)
    if location == MEMORY_PATH:
        raise ValueError("an in-memory index has no second connection; read on the writer")
    uri = f"{Path(location).resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False, isolation_level=None)
    except sqlite3.Error as exc:
        raise MemoryIndexError(f"cannot open memory index reader at {location}: {exc}") from exc
    try:
        _load_sqlite_vec(conn, load_extension)
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        with read_transaction(conn):
            stored_version = get_meta(conn, META_SCHEMA_VERSION)
            stored_model = get_meta(conn, META_EMBEDDING_MODEL_ID)
        if stored_version is None or stored_model is None:
            raise MemoryIndexError(f"memory index at {location} records no schema or model")
        _check_meta(stored_version, stored_model, embedding_model_id)
    except MemoryIndexError:
        conn.close()
        raise
    except sqlite3.Error as exc:
        conn.close()
        raise MemoryIndexError(f"cannot open memory index reader at {location}: {exc}") from exc
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """The `index_meta` value stored under `key`, or None when unset."""
    row = conn.execute("SELECT value FROM index_meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row[0]


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Store `value` under `key` in `index_meta`; call it inside `write_transaction`."""
    conn.execute(
        "INSERT INTO index_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run the block in one write transaction, committed on success and rolled back otherwise.

    `BEGIN IMMEDIATE` takes the write lock before the first statement, so a second writer waits
    on the busy timeout at the start rather than failing to upgrade a read lock midway.
    """
    conn.execute("BEGIN IMMEDIATE")
    committed = False
    try:
        yield conn
        conn.execute("COMMIT")
        committed = True
    finally:
        # SQLite rolls some failures back itself; a second ROLLBACK would then raise and hide
        # the original error.
        if not committed and conn.in_transaction:
            conn.execute("ROLLBACK")


@contextmanager
def read_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run several reads against one snapshot, so a commit between them cannot split a result."""
    conn.execute("BEGIN")
    try:
        yield conn
    finally:
        if conn.in_transaction:
            conn.execute("COMMIT")


def _load_sqlite_vec(conn: sqlite3.Connection, load_extension: ExtensionLoader) -> None:
    """Load sqlite-vec the way the vault sidecar does (sidecar_index.py:319)."""
    try:
        conn.enable_load_extension(True)
        load_extension(conn)
        conn.enable_load_extension(False)
    # AttributeError: a Python built without extension loading has no `enable_load_extension`.
    except (sqlite3.Error, AttributeError, OSError) as exc:
        raise MemoryIndexError(f"sqlite_vec failed to load: {exc}") from exc


def _configure(conn: sqlite3.Connection, *, in_memory: bool, location: str) -> None:
    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if not in_memory and str(mode).lower() != "wal":
        raise MemoryIndexError(f"memory index at {location} cannot use WAL (journal mode {mode})")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")


def _check_meta(
    stored_version: str | None, stored_model: str | None, embedding_model_id: str
) -> None:
    """Refuse a file built under another schema version or embedding model; None is unset."""
    if stored_version is not None and stored_version != str(SCHEMA_VERSION):
        raise MemoryIndexError(
            f"memory index was built with schema {stored_version}, this code uses schema "
            f"{SCHEMA_VERSION}; rebuild the index"
        )
    if stored_model is not None and stored_model != embedding_model_id:
        raise MemoryIndexError(
            f"memory index was embedded with {stored_model!r}, the configured model is "
            f"{embedding_model_id!r}; rebuild the index"
        )


def _apply_schema(conn: sqlite3.Connection, embedding_model_id: str) -> None:
    """Create the schema, checking the stored version and model before the fact tables.

    The check comes first so this version's DDL never runs against a file of another version:
    `IF NOT EXISTS` leaves an older `facts` table as it is, so an index on a column that table
    lacks would fail with a message about the column instead of naming the version mismatch.
    """
    with write_transaction(conn):
        for statement in INDEX_META_DDL:
            conn.execute(statement)
        stored_version = get_meta(conn, META_SCHEMA_VERSION)
        stored_model = get_meta(conn, META_EMBEDDING_MODEL_ID)
        _check_meta(stored_version, stored_model, embedding_model_id)
        for statement in FACTS_DDL:
            conn.execute(statement)
        if stored_version is None:
            set_meta(conn, META_SCHEMA_VERSION, str(SCHEMA_VERSION))
        if stored_model is None:
            set_meta(conn, META_EMBEDDING_MODEL_ID, embedding_model_id)
