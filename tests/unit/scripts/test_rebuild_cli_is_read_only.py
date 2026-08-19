"""`graph-rebuild-from-log`'s read-only contract, enforced BEHAVIOURALLY.

The command's entire advertised contract is "proof-first, dry-run only", and until
2026-08-05 it wrote to the LIVE SQLite event store on every run: `initialize()` on both
stores (`mkdir` + `executescript` + a conditional `ALTER TABLE`), then a
`rebuild-<epoch>-<uuid>` job row plus a checkpoint per turn, all doubled because
`_build_once` runs twice for the determinism gate.

Neither isolation guard could have caught it and neither ever will:
`assert_rebuild_target_not_live` and `assert_neo4j_isolated` both reason about bolt URIs,
and a SQLite path is invisible to them. `test_rebuild_journal_isolation.py` proves the
regenerator never writes to its replay source; these tests cover the other half, the CLI's
own wiring in `_build_log_regenerator`.

## Why these are behavioural and not AST checks

The first version of this file parsed `scripts/mist_admin.py` with `ast` and asserted three
properties of the `_build_log_regenerator` node. A whole-branch review defeated all three
with one mutant that fully restored the live write, and the two escapes are the ordinary
shape of everyday edits, not contrivances:

- **Code motion.** An AST rule scoped to one function node stops at the frame boundary.
  Moving the two `initialize()` calls into a module-level helper and calling THAT from the
  builder restores both live writes with the rule still green.
- **Import aliasing.** A rule asserting the call is spelled `NullRebuildJournal(...)` is
  satisfied by `import EventStoreRebuildJournal as NullRebuildJournal` -- the spelling is
  identical and the object is durable and bound to the live store. This is what a careless
  rename or a bad merge produces.

Both tests below assert on RUNTIME behaviour instead, so neither escape survives: patching
`initialize` catches the write wherever in the call graph it moved to, and asserting the
constructed journal's `type` sees through any alias.

The original file also justified the AST approach by claiming a behavioural test was out of
reach -- "reaching it in a test means a backend load, a real embedding model, and a live
Neo4j connection". **That was false, and it was never checked before being written.** The
collaborators are function-local imports, so patching their source modules reaches
`_build_log_regenerator` with no model load and no Neo4j. Recorded here because a false
justification for a weaker test is how the weaker test survives review.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from backend.event_store.store import EventStore
from backend.knowledge.extraction_cache import ExtractionCache
from backend.knowledge.regeneration.log_regenerator import LogRegenerator
from backend.knowledge.regeneration.rebuild_journal import NullRebuildJournal
from tests.mocks.config import build_test_config

# Function-local imports in `_build_log_regenerator`, so the SOURCE modules are the
# patch targets -- `mist_admin.EmbeddingGenerator` does not exist to patch.
EMBEDDINGS = "backend.knowledge.embeddings.embedding_generator.EmbeddingGenerator"
GRAPH_STORE = "backend.knowledge.storage.graph_store.GraphStore"
GRAPH_EXECUTOR = "backend.knowledge.storage.graph_executor.GraphExecutor"
BUILD_PIPELINE = "backend.factories.build_curation_pipeline"

ONTOLOGY = "1.4.0"
EXTRACTION = "2026-06-14-r5"
MODEL_HASH = "test-model-hash"
EPOCH_TS = "2026-07-01T08:00:00+00:00"


def _seed_replay_sources(root: Path) -> str:
    """Create an event store (with one epoch) + extraction cache under `root`.

    The TEST creates these, which is exactly the point: `_build_log_regenerator` must
    find them already present and must never bring them into being itself.
    """
    db_path = root / "event_store.db"
    store = EventStore(str(db_path))
    store.initialize()
    store.append_epoch(
        ontology_version=ONTOLOGY,
        extraction_version=EXTRACTION,
        model_hash=MODEL_HASH,
        activated_at=EPOCH_TS,
    )
    cache = ExtractionCache(str(root / "extraction_cache.db"))
    cache.initialize()
    # Closed rather than left to GC: an open writer keeps `-wal`/`-shm` sidecars
    # alive, and whether those two exist decides whether a read-only open of a WAL
    # database has to create them. The wal-index test below needs that deterministic.
    store.close()
    cache.close()
    return str(db_path)


def _backend_stub(db_path: str) -> SimpleNamespace:
    """Minimal stand-in for `_load_backend()`: the builder only calls `be.get_config()`."""
    config = build_test_config()
    config.event_store.db_path = db_path
    return SimpleNamespace(get_config=lambda: config)


def _build(db_path: str, extra_patches: dict[str, Any] | None = None):
    """Call the real `_build_log_regenerator` with its heavy collaborators patched out.

    Args:
        db_path: Event-store path the stub config points at; the extraction cache is
            derived from its parent, exactly as the builder does.
        extra_patches: Optional `{dotted.target: replacement}` applied for the call.
    """
    import scripts.mist_admin as mist_admin

    with ExitStack() as stack:
        for target in (EMBEDDINGS, GRAPH_STORE, GRAPH_EXECUTOR, BUILD_PIPELINE):
            stack.enter_context(patch(target))
        for target, replacement in (extra_patches or {}).items():
            stack.enter_context(patch(target, replacement))
        return mist_admin._build_log_regenerator(_backend_stub(db_path), object(), None)


class TestTheReplaySourcesAreNeverInitialized:
    """`initialize()` is a WRITE: mkdir + executescript + a conditional ALTER TABLE."""

    def test_the_builder_initializes_neither_replay_source(self, tmp_path):
        """Fails wherever in the call graph an `initialize()` is reintroduced.

        Patching the METHODS rather than inspecting one function's AST is what makes
        this immune to the code-motion escape: a helper called by the builder trips it
        exactly as the builder would.
        """
        # Arrange
        db_path = _seed_replay_sources(tmp_path)

        def _forbidden(*_args, **_kwargs):
            raise AssertionError(
                "_build_log_regenerator initialized a replay source. That is a write to "
                "live state (mkdir + executescript + ALTER TABLE) from a command whose "
                "contract is dry-run only. Use _assert_replay_source_exists instead."
            )

        # Act -- any initialize() on either store, from any frame, fails the test
        regen, _epoch = _build(
            db_path,
            {
                "backend.event_store.store.EventStore.initialize": _forbidden,
                "backend.knowledge.extraction_cache.ExtractionCache.initialize": _forbidden,
            },
        )

        # Assert -- and the builder still produced a working regenerator
        assert isinstance(regen, LogRegenerator)

    def test_a_missing_replay_source_is_refused_rather_than_created(self, tmp_path):
        """The guard must fire on absence instead of silently creating an empty store."""
        from backend.knowledge.regeneration.log_regenerator import ColdCacheError

        # Arrange -- an empty directory: neither store exists
        missing = tmp_path / "nothing" / "event_store.db"

        # Act / Assert
        with pytest.raises(ColdCacheError, match="event store"):
            _build(str(missing))

        # Structurally green today and RETAINED deliberately, so it is not deleted as
        # dead: `EventStore.__init__` (store.py:38-48) does no disk I/O and the guard
        # raises before any later statement touches the filesystem, so nothing in the
        # current path could create this file. It encodes the invariant, not the
        # current call graph -- an `__init__` that gained a `mkdir` would trip it.
        assert not missing.exists(), "the refused run created the store it refused to find"

    def test_a_missing_extraction_cache_is_refused_even_with_a_valid_event_store(self, tmp_path):
        """Covers the SECOND `_assert_replay_source_exists` call. Nothing else here does.

        Every other refusal test in this file points a broken path at the EVENT STORE,
        and `_build_log_regenerator` checks the event store first -- so in all of them
        the event-store call is the raiser and the extraction-cache call is never
        reached. Deleting the cache guard leaves every one of them green. This test
        seeds a valid, initialized event store so the first guard passes, then removes
        the cache alone, leaving the second guard as the only thing that can refuse.
        """
        from backend.knowledge.regeneration.log_regenerator import ColdCacheError

        # Arrange -- both replay sources seeded, then the cache alone removed
        db_path = _seed_replay_sources(tmp_path)
        cache_path = tmp_path / "extraction_cache.db"
        cache_path.unlink()

        # Act / Assert -- the event store is valid, so this can only be the cache guard
        with pytest.raises(ColdCacheError, match="extraction cache"):
            _build(db_path)

        assert not cache_path.exists(), "the refused run created the cache it refused to find"

    def test_an_event_store_missing_a_conversation_table_is_refused(self, tmp_path):
        """The gate must cover EVERY table the replay reads, not just `epoch_ledger`.

        `LogRegenerator` calls exactly two `EventStore` methods --
        `get_all_turns_for_reextraction` (`conversation_turn_events` LEFT JOIN
        `conversation_sessions`, store.py:406-407) and `get_turn_count`
        (`conversation_turn_events`, :454) -- while `_build_log_regenerator` itself
        reads `epoch_ledger`. A store carrying `epoch_ledger` alone passes an
        `epoch_ledger`-only gate and then raises a bare `OperationalError: no such
        table` on the first read, which is the outcome this guard exists to prevent.
        """
        import sqlite3

        from backend.knowledge.regeneration.log_regenerator import ColdCacheError

        # Arrange -- a fully valid store, then one conversation table removed
        db_path = _seed_replay_sources(tmp_path)
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE conversation_turn_events")
        conn.commit()
        conn.close()

        # Act / Assert -- `epoch_ledger` is still present, so only a widened gate refuses
        with pytest.raises(ColdCacheError, match="conversation_turn_events"):
            _build(db_path)

    def test_a_store_that_exists_but_was_never_initialized_is_refused(self, tmp_path):
        """Existence is not enough: a 0-byte file passes `Path.exists()`.

        `sqlite3.connect(path).close()` leaves exactly that -- a real, readable,
        schema-less database -- and so does `touch`. Under an existence-only guard the
        run proceeds and the first read raises a bare
        `sqlite3.OperationalError: no such table: epoch_ledger`, which
        `cmd_graph_rebuild_from_log` does not catch, so it escapes as a traceback
        rather than a refusal.

        This is NOT the state a pre-fix run of the CLI left behind, and an earlier
        version of this docstring claimed it was, calling it the common case.
        `EventStore.initialize()` executescripts `schema.sql`, which carries
        `CREATE TABLE IF NOT EXISTS epoch_ledger`, so a pre-fix machine's store HAS the
        table (empty) and passes this guard; it refuses later and for a different
        reason, `ColdCacheError("No epochs found in the event store")`.
        """
        # Arrange -- a real, readable, EMPTY SQLite database with no schema at all
        import sqlite3

        from backend.knowledge.regeneration.log_regenerator import ColdCacheError

        db_path = tmp_path / "event_store.db"
        sqlite3.connect(str(db_path)).close()
        assert db_path.exists(), "precondition: the file must exist for this to be a real test"

        # Act / Assert -- refused as a decision, not surfaced as OperationalError
        with pytest.raises(ColdCacheError, match="epoch_ledger"):
            _build(str(db_path))

    def test_the_guard_opens_the_store_on_a_connection_that_refuses_writes(self, tmp_path):
        """The `?mode=ro` the guard's docstring calls essential, actually enforced.

        Nothing else in this file can see it. Every other test's verdict is decided by
        the file's CONTENTS, so a mutant spelling the open `sqlite3.connect(db_path)`
        -- read-write, no URI -- passes all of them: the guard reads the same schema
        either way and never writes, so the mode is invisible to a black-box
        assertion.

        This probes the connection the guard actually opened rather than the text of
        the call. `sqlite3.connect` is wrapped, the real connection is handed back
        untouched, and a `CREATE TABLE` is attempted on it first and rolled back. On a
        `mode=ro` connection SQLite raises `OperationalError`; on a read-write one the
        statement succeeds. That is the property, not its spelling -- an equivalent
        read-only open written some other way still passes.

        The FIRST open is the event-store guard's. `EventStore.__init__` and
        `ExtractionCache.__init__` do no I/O, so nothing connects before it, and the
        read-write `get_current_epoch()` open that legitimately follows the guards is
        later in the list.
        """
        import sqlite3

        # Arrange
        db_path = _seed_replay_sources(tmp_path)
        opens: list[tuple[str, bool]] = []
        real_connect = sqlite3.connect

        def _spy(target, *args, **kwargs):
            conn = real_connect(target, *args, **kwargs)
            try:
                conn.execute("CREATE TABLE _writability_probe (x)")
                conn.rollback()
                opens.append((str(target), True))
            except sqlite3.OperationalError:
                opens.append((str(target), False))
            return conn

        # Act
        with patch("sqlite3.connect", _spy):
            regen, _epoch = _build(db_path)

        # Assert
        assert isinstance(regen, LogRegenerator)
        assert opens, "the guard opened no sqlite connection at all"
        target, writable = opens[0]
        assert not writable, (
            f"the replay-source guard opened {target} on a WRITABLE connection. The "
            "guard runs against the LIVE store under a command whose contract is "
            "dry-run only, so the open must be read-only."
        )

    def test_a_hash_in_the_store_path_neither_misreads_nor_creates_a_database(self, tmp_path):
        """`#` is URI syntax, and the guard interpolates the path into a URI.

        With the path f-string'd in unescaped, everything after `#` becomes the URI
        FRAGMENT. Two things follow, and the second is the serious one: SQLite opens a
        DIFFERENT path (everything before the `#`), and `?mode=ro` -- which sits inside
        the discarded fragment -- is never applied, so the open is read-write-CREATE.
        The run creates a database that did not exist, reads an empty `sqlite_master`,
        and refuses a healthy store for having no `epoch_ledger`.

        Which assertion does the work, stated so neither is read as more than it is:
        against the unescaped f-string it is the `_build` call itself that fails, on
        the refusal, before the stray-file assertion is reached. The stray-file
        assertion is retained for the variant the refusal cannot see -- a truncated
        path that happens to land on a database carrying these tables, where the run
        would proceed against the WRONG store having created or written it. That the
        creation is real, and not merely possible, was measured separately: opening
        `file:<dir>/release#2/event_store.db?mode=ro` created `<dir>/release` and then
        accepted a `CREATE TABLE` on it, because `?mode=ro` was inside the fragment.

        Path SHAPE, not a Windows drive letter: the suite runs in the Linux backend
        container, and `#` is a legal filename character on both platforms.
        """
        # Arrange -- a valid store beneath a directory whose name carries a `#`
        store_dir = tmp_path / "release#2"
        store_dir.mkdir()
        db_path = _seed_replay_sources(store_dir)

        # Act -- the guard must resolve to the real store, not to `<tmp_path>/release`
        regen, _epoch = _build(db_path)

        # Assert
        assert isinstance(regen, LogRegenerator)
        assert not (tmp_path / "release").exists(), (
            "the guard truncated the path at `#` and opened `<tmp_path>/release` "
            "read-write, creating a database file. `?mode=ro` landed in the URI "
            "fragment and was never applied."
        )

    def test_a_wal_store_that_cannot_build_its_index_is_not_reported_as_corrupt(self, tmp_path):
        """A healthy store must never be refused with a corruption diagnosis.

        A read-only open of a WAL database needs BOTH its `-wal` and its `-shm`
        wal-index on disk, and creates whichever is absent. Where it cannot --
        read-only media, a `:ro` bind mount, a directory this process cannot write --
        the read raises `sqlite3.OperationalError` on a store whose schema is
        entirely intact. Folding that into the same handler as `DatabaseError: file
        is not a database` reports a healthy store as corrupt and sends the operator
        to restore a backup they do not need.

        BOTH, not either alone, and this docstring previously said `-shm` only.
        MEASURED on the container (sqlite 3.37.2) varying the two files
        independently against a `chmod 0500` directory and a `:ro` bind mount: the
        read succeeds if and only if both are already present. A present `-shm` does
        not rescue a missing `-wal` and a present `-wal` does not rescue a missing
        `-shm`. This test seeds the both-absent state (`_drop_sidecars` runs after
        the positive control, which creates both), which is the row that raises
        "attempt to write a readonly database" under a `chmod 0500` directory.

        Refusing is still correct (the replay's own reads would fail identically);
        only the diagnosis has to be honest. The positive control below is what makes
        this a test of the DIAGNOSIS rather than of some error being raised: the same
        store passes the guard moments earlier, with only the directory's writability
        changed between the two calls.
        """
        import os
        import sqlite3
        import stat

        from backend.knowledge.regeneration.log_regenerator import ColdCacheError

        # Arrange -- a valid store, checkpointed so no sidecar carries unflushed data
        store_dir = tmp_path / "readonly"
        store_dir.mkdir()
        db_path = _seed_replay_sources(store_dir)

        def _drop_sidecars() -> None:
            for sidecar in list(store_dir.glob("*-wal")) + list(store_dir.glob("*-shm")):
                sidecar.unlink()

        conn = sqlite3.connect(db_path)
        assert (
            conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        ), "precondition: this test is about WAL databases; the store is not one"
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        _drop_sidecars()

        # Positive control -- the very same store passes while the directory is writable
        regen, _epoch = _build(db_path)
        assert isinstance(regen, LogRegenerator), "precondition: the store must be healthy"
        _drop_sidecars()

        os.chmod(store_dir, stat.S_IRUSR | stat.S_IXUSR)
        try:
            try:
                (store_dir / "canary").touch()
                pytest.skip("this process can write a read-only directory; precondition unmet")
            except OSError:
                pass

            # Act -- identical store, identical call, only the directory changed
            with pytest.raises(ColdCacheError) as refusal:
                _build(db_path)
        finally:
            os.chmod(store_dir, stat.S_IRWXU)

        # Assert -- refused, but NOT as a corrupt database
        message = str(refusal.value)
        assert "not a readable SQLite database" not in message, (
            "a healthy store whose wal-index could not be created was reported as "
            f"corrupt. Refusing is right; this diagnosis is not. Message: {message}"
        )
        assert (
            "could not be opened read-only" in message
        ), f"the refusal does not name the real cause. Message: {message}"

    def test_a_pre_migration_conversation_sessions_is_refused_not_tracebacked(self, tmp_path):
        """The gate is table-granular; the replay's dependency is column-granular.

        `LogRegenerator.rebuild` calls `get_all_turns_for_reextraction` with `origins`
        defaulting to `CANONICAL_ORIGINS` (`('real',)`), which is not `None`, so the
        query always carries `COALESCE(s.origin, 'real') IN (?)`.
        `conversation_sessions.origin` arrives via a conditional `ALTER TABLE` inside
        `EventStore.initialize()` -- and this command deliberately never calls it. So
        a store with all three tables but a pre-migration `conversation_sessions`
        passed the widened gate and then raised a bare
        `OperationalError: no such column: s.origin` out of `cmd_graph_rebuild_from_log`,
        which catches only `RebuildTargetError`, `ColdCacheError` and
        `RebuildDeterminismError`.

        The assertion that does the work is the ColdCacheError: without the column
        check `_build` returns a regenerator and this test fails on `DID NOT RAISE`.
        """
        import sqlite3

        from backend.knowledge.regeneration.log_regenerator import ColdCacheError

        # Arrange -- a fully valid store, then `origin` removed from the sessions table.
        # Rebuilt by hand rather than `ALTER TABLE ... DROP COLUMN`, which SQLite
        # refuses on this table ("incomplete input") because its DDL ends in a comment.
        db_path = _seed_replay_sources(tmp_path)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DROP TABLE conversation_sessions")
        conn.execute(
            "CREATE TABLE conversation_sessions ("
            "session_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, "
            "turn_count INTEGER DEFAULT 0, input_modality TEXT DEFAULT 'voice')"
        )
        conn.commit()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(conversation_sessions)")}
        conn.close()
        assert "origin" not in columns, "precondition: the column must actually be gone"

        # Act / Assert -- a decision, not a traceback, and it names the column
        with pytest.raises(ColdCacheError, match="conversation_sessions.origin"):
            _build(db_path)

    def test_an_empty_required_tables_is_rejected_instead_of_disabling_the_gate(self, tmp_path):
        """`required_tables=()` used to pass ANY openable file, silently.

        Not a relaxation of the gate but a removal of it: `", ".join("?" * 0)` is the
        empty string, `name IN ()` is valid SQLite returning no rows, so `present` is
        empty, `missing` is empty, and every file that opens passes. The negative
        control below is the point -- the same schema-less store this suite already
        proves is refused under a real gate sails through under an empty one.

        Reachable only because the parameter was widened from a single table name to
        a tuple; the scalar it replaced could not express "no tables".
        """
        import sqlite3

        import scripts.mist_admin as mist_admin
        from backend.knowledge.regeneration.log_regenerator import ColdCacheError

        # Arrange -- a real, readable, schema-less database: nothing to find
        db_path = tmp_path / "event_store.db"
        sqlite3.connect(str(db_path)).close()

        # Negative control -- a real gate refuses this store, so the file is not the
        # reason the empty-tuple call would have passed
        with pytest.raises(ColdCacheError, match="epoch_ledger"):
            mist_admin._assert_replay_source_exists(str(db_path), "event store", ("epoch_ledger",))

        # Act / Assert -- the empty tuple is a caller bug, not a verdict on the store
        with pytest.raises(ValueError, match="required_tables"):
            mist_admin._assert_replay_source_exists(str(db_path), "event store", ())

    def test_a_truncated_replay_source_is_refused(self, tmp_path):
        """A half-copied or interrupted-`cp` file also passes `Path.exists()`."""
        from backend.knowledge.regeneration.log_regenerator import ColdCacheError

        # Arrange -- bytes that are not a SQLite database
        db_path = tmp_path / "event_store.db"
        db_path.write_bytes(b"SQLite format 3\x00truncated-garbage")

        # Act / Assert
        with pytest.raises(ColdCacheError, match="not a readable SQLite database"):
            _build(str(db_path))


class TestTheJournalIsNonDurable:
    """A durable journal here sends the proof run's job rows to the live ledger."""

    def test_the_builder_wires_a_non_durable_journal(self, tmp_path):
        """Asserts the constructed object's TYPE, which sees through an import alias.

        The AST version asserted the call was spelled `NullRebuildJournal(...)`, which
        `import EventStoreRebuildJournal as NullRebuildJournal` satisfies while binding a
        durable journal to the live event store. A type check cannot be fooled that way.
        """
        # Arrange
        db_path = _seed_replay_sources(tmp_path)

        # Act
        regen, _epoch = _build(db_path)

        # Assert
        assert type(regen._journal) is NullRebuildJournal, (
            f"rebuild CLI wired a {type(regen._journal).__name__}. Anything durable writes "
            "job rows nothing reads into the LIVE event store, twice per invocation."
        )
        assert regen._journal.durable is False
