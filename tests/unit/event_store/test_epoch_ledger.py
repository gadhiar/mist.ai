"""Append-only epoch ledger on EventStore (F3).

`TestPackHashes`: `append_epoch` stores the two ontology pack hashes it is
given beside the stamp triple, and NULL when it is given none.

`TestPackHashColumnsMigrate` builds the shape a database had before the pack
hash columns by recreating `epoch_ledger` and `epoch_cutover` exactly as
`schema.sql` declared them at 307834c (`make_pre_pack_hash`).
"""

import sqlite3

import pytest

from backend.event_store.store import EventStore

PROMPT = "a" * 64
RAILS = "b" * 64
OTHER = "f" * 64
PACK_HASH_COLUMNS = ("pack_prompt_hash", "pack_rails_hash")
BOTH_FIELDS = pytest.mark.parametrize("field", PACK_HASH_COLUMNS)

# The two epoch tables as `schema.sql` declared them at 307834c, before the
# pack hash columns.
_PRE_PACK_HASH_DDL = """
CREATE TABLE epoch_ledger (
    epoch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_version TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    activated_at TEXT NOT NULL,
    prev_epoch_id INTEGER,
    provisional INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE epoch_cutover (
    cutover_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_version TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    bare_model_hash TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    state TEXT NOT NULL,
    source_epoch_id INTEGER,
    rebuild_job_id TEXT,
    rebuilt_through_event_id TEXT,
    check_report TEXT,
    promoted_epoch_id INTEGER,
    updated_at TEXT NOT NULL,
    serving_identity TEXT,
    serving_components TEXT,
    dispatch_identity TEXT,
    dispatch_components TEXT,
    identity_namespace TEXT,
    binding_source TEXT,
    binding_evidence TEXT,
    bound_at TEXT
);
"""
# `epoch_ledger` as it was declared before `provisional`, the oldest shape
# `initialize` migrates.
_PRE_PROVISIONAL_LEDGER_DDL = """
CREATE TABLE epoch_ledger (
    epoch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_version TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    activated_at TEXT NOT NULL,
    prev_epoch_id INTEGER
);
"""
_EPOCH_TABLES = ("epoch_ledger", "epoch_cutover")


def _store(tmp_path):
    store = EventStore(str(tmp_path / "es.db"))
    store.initialize()
    return store


def make_pre_pack_hash(path: str) -> None:
    """Give the database at `path` its two epoch tables as they were before the pack hashes.

    Over a freshly initialized database the two tables are replaced; over a
    path with no database, the file is created holding them alone. Either way
    both are built from the earlier DDL and given one row each: an active
    epoch, and an open, bound cutover from it.
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        "DROP TABLE IF EXISTS epoch_ledger; DROP TABLE IF EXISTS epoch_cutover;"
        + _PRE_PACK_HASH_DDL
    )
    conn.execute(
        "INSERT INTO epoch_ledger (ontology_version, extraction_version, model_hash, "
        "activated_at, prev_epoch_id, provisional) VALUES ('1.4.0', 'ev', 'm1', 't0', NULL, 1)"
    )
    conn.execute(
        "INSERT INTO epoch_cutover (ontology_version, extraction_version, model_hash, "
        "bare_model_hash, requested_at, state, source_epoch_id, check_report, updated_at, "
        "serving_identity, serving_components, dispatch_identity, dispatch_components, "
        "identity_namespace, binding_source, binding_evidence, bound_at) "
        "VALUES ('1.4.0', 'ev', 'm2|emb:e', 'm2', 't1', 'ready', 1, '{\"x\": 1}', 't2', "
        "'si1-a', '{}', 'di1-a', '{}', 'si1-a+di1-a', 'adopted', '{}', 't2')"
    )
    conn.commit()
    conn.close()


def _table_info(conn: sqlite3.Connection, table: str) -> list[tuple]:
    """Every column of `table` in declaration order: (cid, name, type, notnull, default, pk)."""
    return [tuple(row) for row in conn.execute(f"PRAGMA table_info({table})")]


def _rows(conn: sqlite3.Connection, table: str, columns: str = "*") -> list[tuple]:
    return [tuple(row) for row in conn.execute(f"SELECT {columns} FROM {table} ORDER BY rowid")]


def _pre_change_database(tmp_path) -> str:
    path = str(tmp_path / "old.db")
    fresh = EventStore(db_path=path)
    fresh.initialize()
    fresh.close()
    make_pre_pack_hash(path)
    return path


class TestEpochLedger:
    def test_append_then_get_current(self, tmp_path):
        store = _store(tmp_path)
        eid = store.append_epoch(
            "1.1.0", "2026-05-06-r1", "gemma-x", activated_at="2026-06-10T00:00:00Z"
        )
        current = store.get_current_epoch()
        assert current is not None
        assert current["epoch_id"] == eid
        assert current["ontology_version"] == "1.1.0"
        assert current["prev_epoch_id"] is None
        store.close()

    def test_append_same_triple_is_idempotent(self, tmp_path):
        store = _store(tmp_path)
        id1 = store.append_epoch("1.1.0", "v1", "m1", activated_at="t1")
        id2 = store.append_epoch("1.1.0", "v1", "m1", activated_at="t2")
        assert id2 == id1
        assert len(store.list_epochs()) == 1
        store.close()

    def test_changed_triple_appends_with_prev_pointer(self, tmp_path):
        store = _store(tmp_path)
        id1 = store.append_epoch("1.1.0", "v1", "m1", activated_at="t1")
        id3 = store.append_epoch("1.2.0", "v1", "m1", activated_at="t2")
        assert id3 != id1
        epochs = store.list_epochs()
        assert len(epochs) == 2
        current = store.get_current_epoch()
        assert current["ontology_version"] == "1.2.0"
        assert current["prev_epoch_id"] == id1
        store.close()

    def test_get_current_returns_none_when_empty(self, tmp_path):
        store = _store(tmp_path)
        assert store.get_current_epoch() is None
        store.close()


class TestPackHashes:
    """`append_epoch` stores what it is given and derives nothing."""

    def test_append_stores_the_hashes_it_is_given(self, tmp_path):
        store = _store(tmp_path)

        epoch_id = store.append_epoch(
            "1.1.0", "v1", "m1", activated_at="t1", pack_prompt_hash=PROMPT, pack_rails_hash=RAILS
        )

        row = store.get_epoch(epoch_id)
        assert (row["pack_prompt_hash"], row["pack_rails_hash"]) == (PROMPT, RAILS)
        assert (row["ontology_version"], row["extraction_version"], row["model_hash"]) == (
            "1.1.0",
            "v1",
            "m1",
        )
        store.close()

    def test_append_given_no_hash_stores_null(self, tmp_path):
        store = _store(tmp_path)

        store.append_epoch("1.1.0", "v1", "m1", activated_at="t1")

        row = store.get_current_epoch()
        assert row["pack_prompt_hash"] is None
        assert row["pack_rails_hash"] is None
        store.close()

    def test_the_hashes_are_keyword_only(self, tmp_path):
        store = _store(tmp_path)

        with pytest.raises(TypeError):
            store.append_epoch("1.1.0", "v1", "m1", "t1", PROMPT, RAILS)
        store.close()

    def test_the_same_five_values_return_the_latest_epoch_without_inserting(self, tmp_path):
        store = _store(tmp_path)
        first = store.append_epoch(
            "1.1.0", "v1", "m1", activated_at="t1", pack_prompt_hash=PROMPT, pack_rails_hash=RAILS
        )

        again = store.append_epoch(
            "1.1.0", "v1", "m1", activated_at="t2", pack_prompt_hash=PROMPT, pack_rails_hash=RAILS
        )

        assert again == first
        assert len(store.list_epochs()) == 1
        store.close()

    @BOTH_FIELDS
    def test_a_call_that_differs_only_in_one_pack_hash_appends(self, tmp_path, field):
        store = _store(tmp_path)
        stored = {"pack_prompt_hash": PROMPT, "pack_rails_hash": RAILS}
        first = store.append_epoch("1.1.0", "v1", "m1", activated_at="t1", **stored)

        second = store.append_epoch(
            "1.1.0", "v1", "m1", activated_at="t2", **{**stored, field: OTHER}
        )

        assert second != first
        current = store.get_current_epoch()
        assert current["epoch_id"] == second
        assert current[field] == OTHER
        assert current["prev_epoch_id"] == first
        assert len(store.list_epochs()) == 2
        store.close()

    def test_a_call_with_no_hash_over_an_epoch_that_holds_them_appends(self, tmp_path):
        """None is a value like any other here: five values, not a triple and a wildcard."""
        store = _store(tmp_path)
        first = store.append_epoch(
            "1.1.0", "v1", "m1", activated_at="t1", pack_prompt_hash=PROMPT, pack_rails_hash=RAILS
        )

        second = store.append_epoch("1.1.0", "v1", "m1", activated_at="t2")

        assert second != first
        current = store.get_current_epoch()
        assert (current["pack_prompt_hash"], current["pack_rails_hash"]) == (None, None)
        store.close()


class TestPackHashColumnsMigrate:
    """`initialize` adds the two columns to both epoch tables, last, and rewrites no row."""

    def test_a_pre_change_database_gains_the_four_columns(self, tmp_path):
        path = _pre_change_database(tmp_path)
        before = sqlite3.connect(path)
        for table in _EPOCH_TABLES:
            names = [column[1] for column in _table_info(before, table)]
            assert not set(PACK_HASH_COLUMNS) & set(names), table
        before.close()

        migrated = EventStore(db_path=path)
        migrated.initialize()

        for table in _EPOCH_TABLES:
            columns = _table_info(migrated._get_connection(), table)
            assert [column[1] for column in columns[-2:]] == list(PACK_HASH_COLUMNS), table
            # Nullable TEXT with no default: (type, notnull, default) per column.
            assert [column[2:5] for column in columns[-2:]] == [("TEXT", 0, None)] * 2, table
        migrated.close()

    def test_a_second_initialize_changes_nothing(self, tmp_path):
        path = _pre_change_database(tmp_path)
        migrated = EventStore(db_path=path)
        migrated.initialize()
        conn = migrated._get_connection()
        once = {table: (_table_info(conn, table), _rows(conn, table)) for table in _EPOCH_TABLES}

        migrated.initialize()

        twice = {table: (_table_info(conn, table), _rows(conn, table)) for table in _EPOCH_TABLES}
        assert twice == once
        migrated.close()

    def test_existing_rows_read_null_in_the_new_columns_and_keep_every_other_value(self, tmp_path):
        path = _pre_change_database(tmp_path)
        old = sqlite3.connect(path)
        before = {table: _rows(old, table) for table in _EPOCH_TABLES}
        old_columns = {
            table: ", ".join(column[1] for column in _table_info(old, table))
            for table in _EPOCH_TABLES
        }
        old.close()
        assert all(len(rows) == 1 for rows in before.values())

        migrated = EventStore(db_path=path)
        migrated.initialize()

        conn = migrated._get_connection()
        for table in _EPOCH_TABLES:
            assert _rows(conn, table, old_columns[table]) == before[table], table
            assert _rows(conn, table, ", ".join(PACK_HASH_COLUMNS)) == [(None, None)], table
        epoch = migrated.get_current_epoch()
        cutover = migrated.get_open_epoch_cutover()
        assert (epoch["pack_prompt_hash"], epoch["pack_rails_hash"]) == (None, None)
        assert (cutover["pack_prompt_hash"], cutover["pack_rails_hash"]) == (None, None)
        migrated.close()

    def test_a_fresh_and_a_migrated_database_have_the_same_columns_in_the_same_order(
        self, tmp_path
    ):
        """The CREATE TABLE and the guarded ALTER declare the columns independently."""
        fresh = EventStore(db_path=":memory:")
        fresh.initialize()
        migrated = EventStore(db_path=_pre_change_database(tmp_path))
        migrated.initialize()

        for table in _EPOCH_TABLES:
            assert _table_info(migrated._get_connection(), table) == _table_info(
                fresh._get_connection(), table
            ), table
        migrated.close()

    def test_the_oldest_ledger_shape_migrates_to_the_fresh_column_order(self, tmp_path):
        """A ledger from before `provisional` gains it first, then the two hashes."""
        path = str(tmp_path / "oldest.db")
        conn = sqlite3.connect(path)
        conn.executescript(_PRE_PROVISIONAL_LEDGER_DDL)
        conn.close()
        fresh = EventStore(db_path=":memory:")
        fresh.initialize()

        migrated = EventStore(db_path=path)
        migrated.initialize()

        assert _table_info(migrated._get_connection(), "epoch_ledger") == _table_info(
            fresh._get_connection(), "epoch_ledger"
        )
        migrated.close()
