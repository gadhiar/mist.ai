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
    ExtractionCache(str(root / "extraction_cache.db")).initialize()
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
