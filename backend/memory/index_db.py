"""Open `memory_index.db`, the derived index the memory read path serves from.

This module is the contract home of the file: its DDL and `SCHEMA_VERSION` live here and
nowhere else. The file is derived from the graph (and, in Phase B, the event log), and it is
not backed up (`LIVE_STORE_FILENAMES`, scripts/backup/stores.py, has no entry for it), so a
stale file is rebuilt rather than migrated, and rebuilding it loses nothing.

The rebuild runs in place, in two phases, so open readers never see a missing file:

1. `open_index` finds a stored schema version or embedding model id other than this code's,
   or a rebuild that never finished. One write transaction drops every derived table,
   recreates this schema's tables in the same file, and stores the state `rebuilding` with
   the versions it rebuilds for; the old versions stay stored. One WARNING names both.
2. The refresher's next full rescan refills the tables, and `IndexRebuild.complete` then
   stores the new versions and clears the state in one commit.

While the state is `rebuilding`, `degraded_reason` gives `index_rebuilding`, so the fetch
serves no block from a partial index. A crash inside phase 1 rolls back and leaves the old
versions; a crash in phase 2 leaves `rebuilding`. Either way the next open rebuilds again, so
a partial index never opens as valid.

Tables:

- `facts`: one row per latest-belief fact version, keyed by `row_key` and grouped by
  `fact_key` when read, with the subject's and object's entity types, the served text and the
  fact view (`view_text`, retrieval_view.py) the index embeds. `id` is the rowid the two
  mirrors share; `idx_facts_siblings` serves `FactIndex.siblings`.
- `facts_fts`: FTS5 over the served text, the subject and object names, and the type terms
  of both entity types (`type_terms.type_terms_text`), each stored in `fts.fold` form.
- `facts_vec`: sqlite-vec `vec0` table of the view text's embedding, cosine distance.
- `facets`: one row per facet (backend/memory/facets.py) of the stored fact versions: a
  subject, a relation and an object type held by two or more fact keys, keyed by `facet_key`,
  with the subject as last recorded and the facet's view (`render_facet_view`). It holds no
  members: they are the `facts` rows of its subject, relation and object type, read when the
  facet is fetched.
- `facets_fts`: FTS5 over the facet's view text, its subject name and the type terms of its
  subject and object types, folded as `facts_fts` is.
- `facets_vec`: sqlite-vec `vec0` table of the facet view's embedding, cosine distance.
- `index_meta`: `schema_version`, `embedding_model_id`, the rebuild state and its target
  versions while a rebuild runs, and the refresh watermarks.

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

from .models import DegradedReason

logger = logging.getLogger(__name__)

# 2: `facts.subject_type`, `facts.object_type`, `facts_fts.type_terms`, `idx_facts_siblings`.
# 3: `facts.view_text`, the fact view `facts_vec` holds the embedding of.
# 4: `facets`, `facets_fts` and `facets_vec`.
SCHEMA_VERSION = 4
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2, the embedder the vault sidecar also uses

META_SCHEMA_VERSION = "schema_version"
META_EMBEDDING_MODEL_ID = "embedding_model_id"
# Set only while a rebuild runs, with the versions it rebuilds for.
META_INDEX_STATE = "index_state"
META_REBUILD_SCHEMA_VERSION = "rebuild_schema_version"
META_REBUILD_EMBEDDING_MODEL_ID = "rebuild_embedding_model_id"

STATE_REBUILDING = "rebuilding"

_VERSION_KEYS = (META_SCHEMA_VERSION, META_EMBEDDING_MODEL_ID)

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
        content_hash        TEXT NOT NULL,
        view_text           TEXT NOT NULL
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
    """
    CREATE TABLE IF NOT EXISTS facets (
        id           INTEGER PRIMARY KEY,
        facet_key    TEXT NOT NULL UNIQUE,
        subject_id   TEXT NOT NULL,
        subject_name TEXT NOT NULL,
        subject_type TEXT,
        rel_type     TEXT NOT NULL,
        object_type  TEXT NOT NULL,
        view_text    TEXT NOT NULL,
        content_hash TEXT NOT NULL
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS facets_fts USING fts5(
        view_text,
        subject_name,
        type_terms,
        tokenize='porter unicode61'
    )
    """,
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS facets_vec USING vec0(
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
    """Open or create `memory_index.db`, rebuilding it in place when something else built it.

    The schema and the meta rows are written in one transaction, so a crash mid-open leaves
    either the file as it was or a complete schema. A file whose stored schema version or
    embedding model id differs, or whose last rebuild never finished, starts phase 1 of the
    rebuild (module docstring): its derived tables are dropped and recreated empty in the same
    transaction, the state `rebuilding` is stored, and one WARNING names both versions. A
    matching file is opened without a write.

    The connection allows use from other threads (`check_same_thread=False`) because it is
    opened on one thread and used from the `memory-index` executor's threads; FactIndex
    serialises access to it.

    Args:
        path: File path, or `:memory:` for a private in-memory index (WAL does not apply).
        embedding_model_id: Id of the model that embeds the rows. A file recorded under
            another id holds vectors from another space, so it is rebuilt.
        load_extension: Loads sqlite-vec into a connection; tests inject a failing loader.

    Returns:
        An open connection in autocommit mode with sqlite-vec loaded and WAL enabled.

    Raises:
        ValueError: `embedding_model_id` is empty.
        MemoryIndexError: The file cannot be opened or is not a database, sqlite-vec does not
            load, or the stale tables cannot be dropped.
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
        _apply_schema(conn, embedding_model_id, location=location)
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

    A file being rebuilt still stores its old versions, so it is checked against the versions
    the rebuild is for instead; its reads then report `index_rebuilding` (`degraded_reason`)
    until the rebuild completes.

    Args:
        path: The file the writer opened. `:memory:` is refused: a private in-memory database
            has no second connection, so an in-memory FactIndex reads on its writer instead.
        embedding_model_id: The configured model id; the file must record the same one, or be
            rebuilding for it.
        load_extension: Loads sqlite-vec into a connection; tests inject a failing loader.

    Returns:
        An open read-only connection in autocommit mode with sqlite-vec loaded.

    Raises:
        ValueError: `path` is `:memory:` or `embedding_model_id` is empty.
        MemoryIndexError: The file is missing or not a database, sqlite-vec does not load, or
            the schema version or embedding model id it records (or rebuilds for) differs or
            is absent.
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
            rebuilding = get_meta(conn, META_INDEX_STATE) == STATE_REBUILDING
            version_key, model_key = (
                (META_REBUILD_SCHEMA_VERSION, META_REBUILD_EMBEDDING_MODEL_ID)
                if rebuilding
                else _VERSION_KEYS
            )
            stored_version = get_meta(conn, version_key)
            stored_model = get_meta(conn, model_key)
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


def degraded_reason(conn: sqlite3.Connection) -> DegradedReason | None:
    """What a fetch on this index reports instead of a block, or None when it can serve one.

    `index_rebuilding` while a rebuild has not completed: the tables then hold part of the
    facts or none, and a block built from them would silently miss facts. One indexed lookup
    on `index_meta`, cheap enough to run before every fetch, on the reader or the writer.

    Raises:
        MemoryIndexError: The database fails.
    """
    try:
        rebuilding = get_meta(conn, META_INDEX_STATE) == STATE_REBUILDING
    except sqlite3.Error as exc:
        raise MemoryIndexError(f"cannot read the memory index state: {exc}") from exc
    return "index_rebuilding" if rebuilding else None


class IndexRebuild:
    """Completes an in-place rebuild of one index file: phase 2's final commit.

    The refresher calls `complete()` after the full rescan that refilled the tables has
    committed, so the stored versions change only once the index is whole. A crash between
    the two commits leaves `rebuilding`, and the next open rebuilds again.

    `complete()` opens its own short-lived connection to the file rather than writing on the
    FactIndex writer: that connection is serialised by FactIndex's private locks, which this
    commit cannot take, while `BEGIN IMMEDIATE` on a second connection is serialised with every
    writer by SQLite's own write lock. Only `index_meta` is touched, so sqlite-vec is not
    loaded.

    Args:
        path: The index file `open_index` opened. `:memory:` is refused: a private in-memory
            index is created fresh at every open, so it never rebuilds.
        embedding_model_id: The configured model id, the one `open_index` rebuilt for.

    Raises:
        ValueError: `path` is `:memory:` or `embedding_model_id` is empty.
    """

    def __init__(self, path: str | os.PathLike[str], *, embedding_model_id: str):
        if not embedding_model_id:
            raise ValueError("embedding_model_id must name the embedding model")
        location = os.fspath(path)
        if location == MEMORY_PATH:
            raise ValueError("an in-memory index is created fresh at each open; it never rebuilds")
        self._location = location
        self._embedding_model_id = embedding_model_id

    def complete(self) -> bool:
        """Store the new versions and clear the rebuild state, in one commit.

        Returns:
            True when a rebuild was pending and is now complete; False when none was pending,
            in which case nothing is written.

        Raises:
            MemoryIndexError: The file is missing or fails, or the pending rebuild is for
                another schema version or model (another process reopened the file under a
                different configuration since).
        """
        uri = f"{Path(self._location).resolve().as_uri()}?mode=rw"
        try:
            conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        except sqlite3.Error as exc:
            raise MemoryIndexError(
                f"cannot open memory index at {self._location} to complete its rebuild: {exc}"
            ) from exc
        try:
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            if get_meta(conn, META_INDEX_STATE) != STATE_REBUILDING:
                return False
            with write_transaction(conn):
                stored = _stored_meta(conn)
                target = (
                    stored.get(META_REBUILD_SCHEMA_VERSION),
                    stored.get(META_REBUILD_EMBEDDING_MODEL_ID),
                )
                if target != (str(SCHEMA_VERSION), self._embedding_model_id):
                    raise MemoryIndexError(
                        f"memory index at {self._location} is being rebuilt for schema "
                        f"{target[0]} and model {target[1]!r}, not schema {SCHEMA_VERSION} and "
                        f"model {self._embedding_model_id!r}; it rebuilds again at the next open"
                    )
                set_meta(conn, META_SCHEMA_VERSION, str(SCHEMA_VERSION))
                set_meta(conn, META_EMBEDDING_MODEL_ID, self._embedding_model_id)
                conn.execute(
                    "DELETE FROM index_meta WHERE key IN (?, ?, ?)",
                    (
                        META_INDEX_STATE,
                        META_REBUILD_SCHEMA_VERSION,
                        META_REBUILD_EMBEDDING_MODEL_ID,
                    ),
                )
        except sqlite3.Error as exc:
            raise MemoryIndexError(
                f"cannot complete the rebuild of memory index at {self._location}: {exc}"
            ) from exc
        finally:
            conn.close()
        logger.info(
            "memory index at %s rebuilt for schema %d and model %r",
            self._location,
            SCHEMA_VERSION,
            self._embedding_model_id,
        )
        return True


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


def _apply_schema(conn: sqlite3.Connection, embedding_model_id: str, *, location: str) -> None:
    """Create the schema, rebuilding a stale file's tables first, in one transaction.

    The stored versions are read before any fact DDL runs, so this version's DDL never runs
    against another version's tables: `IF NOT EXISTS` would leave an older `facts` table as it
    is, and an index on a column that table lacks would fail. A stale file's tables are
    dropped first and created anew here.
    """
    with write_transaction(conn):
        for statement in INDEX_META_DDL:
            conn.execute(statement)
        stored = _stored_meta(conn)
        stored_version = stored.get(META_SCHEMA_VERSION)
        stored_model = stored.get(META_EMBEDDING_MODEL_ID)
        rebuild = _needs_rebuild(stored, embedding_model_id)
        if rebuild:
            _drop_derived_tables(conn)
            conn.execute(
                "DELETE FROM index_meta WHERE key NOT IN (?, ?)",
                (META_SCHEMA_VERSION, META_EMBEDDING_MODEL_ID),
            )
            set_meta(conn, META_INDEX_STATE, STATE_REBUILDING)
            set_meta(conn, META_REBUILD_SCHEMA_VERSION, str(SCHEMA_VERSION))
            set_meta(conn, META_REBUILD_EMBEDDING_MODEL_ID, embedding_model_id)
        _create_fact_tables(conn)
        if not rebuild:
            if stored_version is None:
                set_meta(conn, META_SCHEMA_VERSION, str(SCHEMA_VERSION))
            if stored_model is None:
                set_meta(conn, META_EMBEDDING_MODEL_ID, embedding_model_id)
    if rebuild:
        unfinished = stored.get(META_INDEX_STATE) == STATE_REBUILDING
        logger.warning(
            "memory index at %s %s schema %s and model %r; rebuilding it in place for schema %d "
            "and model %r, and serving no facts from it until the next full rescan completes",
            location,
            "had an unfinished rebuild, still storing" if unfinished else "was built with",
            stored_version,
            stored_model,
            SCHEMA_VERSION,
            embedding_model_id,
        )


def _stored_meta(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM index_meta").fetchall())


def _needs_rebuild(stored: dict[str, str], embedding_model_id: str) -> bool:
    """Whether the stored meta names another build, or a rebuild that never finished.

    An absent version or model is unset, not stale: a new file stores both at its first open.
    """
    if stored.get(META_INDEX_STATE) == STATE_REBUILDING:
        return True
    stored_version = stored.get(META_SCHEMA_VERSION)
    stored_model = stored.get(META_EMBEDDING_MODEL_ID)
    return (stored_version is not None and stored_version != str(SCHEMA_VERSION)) or (
        stored_model is not None and stored_model != embedding_model_id
    )


def _drop_derived_tables(conn: sqlite3.Connection) -> None:
    """Drop every table but `index_meta`, whatever schema version created it.

    The tables are listed from `sqlite_master` rather than this version's DDL, so a table that
    only an older or newer schema had goes too. Virtual tables go first, each taking its shadow
    tables with it; dropping a shadow table on its own is refused, so the list is read again
    after every drop. Indexes go with their tables.
    """
    while True:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' "
            "AND name != 'index_meta' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        if not tables:
            return
        virtual = [name for name, sql in tables if _is_virtual(sql)]
        name = virtual[0] if virtual else tables[0][0]
        quoted = name.replace('"', '""')
        # The name is read from `sqlite_master` and quoted as an identifier; DDL takes no
        # bound parameters.
        conn.execute(f'DROP TABLE "{quoted}"')  # nosec B608


def _is_virtual(sql: str | None) -> bool:
    return sql is not None and sql.lstrip().upper().startswith("CREATE VIRTUAL TABLE")


def _create_fact_tables(conn: sqlite3.Connection) -> None:
    for statement in FACTS_DDL:
        conn.execute(statement)
