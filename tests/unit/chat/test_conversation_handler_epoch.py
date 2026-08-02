"""Tests for the provisional-epoch write in `ConversationHandler.__init__`.

These exist because of a specific, repeated failure in this codebase, and the
shape of the tests matters more than their count.

`EventStore.ensure_initial_epoch` shipped in R1.4 Task 7 with five unit tests
of its own. Every one of them was correct. Not one of them asserted that
anything in production ever CALLED it -- and nothing did, for the whole of
R1.4, R1.4's whole-branch review, and the merge. The live `epoch_ledger` held
0 rows while the project record stated a provisional epoch had been written.

It was the sixth instance of that defect class on this branch, and a seventh
(`StalenessDetector.confirmation_list`, built and scheduled weekly, consumed
by nothing) was found in the same pass.

So the load-bearing test here is `TestProductionCallerExists`. It does not
check that `ensure_initial_epoch` works -- `tests/unit/event_store/test_store.py`
already does that thoroughly. It checks that CONSTRUCTING A HANDLER THE WAY
PRODUCTION CONSTRUCTS ONE leaves an epoch in the ledger. Delete the call site
and this fails; that is its entire job.
"""

from datetime import UTC, datetime

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.unit.conftest import make_test_conventions_loader

FIXED_CLOCK = datetime(2026, 8, 1, 12, 30, 0, tzinfo=UTC)


def _build_handler(config, *, now_fn=None) -> ConversationHandler:
    """Construct a handler the way production does, with fakes at the I/O edges.

    Deliberately goes through the real `ConversationHandler.__init__` rather
    than calling `ensure_initial_epoch` directly: the thing under test is the
    wiring, and a test that called the method itself would pass just as
    happily with the production call site deleted.
    """
    graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
    return ConversationHandler(
        config=config,
        graph_store=graph_store,
        extraction_pipeline=None,
        retriever=KnowledgeRetriever(config=config, graph_store=graph_store),
        llm_provider=FakeLLM(),
        conventions_loader=make_test_conventions_loader(),
        now_fn=now_fn,
    )


class TestProductionCallerExists:
    """The assertion the original five tests were missing."""

    def test_constructing_a_handler_leaves_an_epoch_in_the_ledger(self):
        # Arrange
        config = build_test_config(event_store_enabled=True, event_store_db_path=":memory:")

        # Act
        handler = _build_handler(config)

        # Assert
        epoch = handler.event_store.get_current_epoch()
        assert epoch is not None, (
            "constructing a ConversationHandler left the epoch ledger empty -- "
            "the production call to ensure_initial_epoch is gone. This is the "
            "exact regression this file exists to catch: the method's own unit "
            "tests all still pass without a caller."
        )

    def test_the_written_epoch_is_marked_provisional(self):
        config = build_test_config(event_store_enabled=True, event_store_db_path=":memory:")

        handler = _build_handler(config)

        epoch = handler.event_store.get_current_epoch()
        assert epoch["provisional"] == 1, (
            "the bootstrap epoch must be queryably provisional, not provisional "
            "by comment -- R1.6 is free to redefine epoch semantics and needs to "
            "be able to find this row"
        )

    def test_the_epoch_carries_the_stamp_triple_a_rebuild_reads(self):
        config = build_test_config(event_store_enabled=True, event_store_db_path=":memory:")

        handler = _build_handler(config)

        epoch = handler.event_store.get_current_epoch()
        # LogRegenerator.rebuild reads exactly these four off the epoch row.
        # A row present but unpopulated would satisfy the tests above and still
        # fail a rebuild at the cache-coverage gate.
        assert epoch["ontology_version"]
        assert epoch["extraction_version"]
        assert epoch["model_hash"]
        assert epoch["activated_at"]


class TestClockDiscipline:
    """`activated_at` is written from the injected clock, never read in the store."""

    def test_activated_at_comes_from_the_injected_clock(self):
        # Arrange
        config = build_test_config(event_store_enabled=True, event_store_db_path=":memory:")

        # Act
        handler = _build_handler(config, now_fn=lambda: FIXED_CLOCK)

        # Assert
        epoch = handler.event_store.get_current_epoch()
        assert epoch["activated_at"] == FIXED_CLOCK.isoformat(), (
            "activated_at did not come from the injected clock, so something in "
            "the path is reading wall-clock -- the failure mode that mis-dated "
            "the only session note MIST had ever written (R1.3.1)"
        )


class TestIdempotency:
    """A second handler on the same database must not append a second epoch."""

    def test_two_handlers_on_one_database_write_one_epoch(self, tmp_path):
        # Arrange -- a real file, because two `:memory:` stores share nothing
        # and the test would pass vacuously against separate databases.
        db_path = str(tmp_path / "event_store.db")
        config = build_test_config(event_store_enabled=True, event_store_db_path=db_path)

        # Act -- construct twice, as a process restart would
        first = _build_handler(config, now_fn=lambda: FIXED_CLOCK)
        second = _build_handler(config, now_fn=lambda: datetime(2027, 1, 1, tzinfo=UTC))

        # Assert
        assert len(second.event_store.list_epochs()) == 1
        assert second.event_store.get_current_epoch()["activated_at"] == FIXED_CLOCK.isoformat()
        first.event_store.close()
        second.event_store.close()


class TestDisabledEventStore:
    """The guard must prevent the write, not merely survive it."""

    def test_no_event_store_and_no_epoch_when_disabled(self, tmp_path):
        # Arrange
        db_path = str(tmp_path / "event_store.db")
        config = build_test_config(event_store_enabled=False, event_store_db_path=db_path)

        # Act
        handler = _build_handler(config)

        # Assert -- side-effect boundary: nothing was created at all
        assert handler.event_store is None
        assert not (tmp_path / "event_store.db").exists()
