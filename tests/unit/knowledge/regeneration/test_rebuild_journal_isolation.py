"""The replay SOURCE is read-only; the rebuild's own progress goes somewhere else.

## The gap these tests close

`LogRegenerator` took ONE event-store dependency and used it for two unrelated jobs: it
READ the log to replay (`get_all_turns_for_reextraction`, `get_turn_count`) and it WROTE
its own job/checkpoint rows back into that same store (`create_reextraction_job`,
`checkpoint_reextraction_job`, `finalize_reextraction_job`). The Neo4j leg of the very
same class had already separated those roles -- `source_conn` reads live, `staging_conn`
takes the writes -- and the SQLite leg had not.

The consequence was `mist_admin graph-rebuild-from-log --dry-run`, whose entire advertised
contract is "proof-first, dry-run only", inserting a `rebuild-<epoch>-<uuid>` row plus a
checkpoint per turn into the LIVE event store, twice per invocation (`_build_once` runs
twice for the determinism gate). Neither isolation guard could see it: both
`assert_rebuild_target_not_live` and `assert_neo4j_isolated` reason about bolt URIs and
have no notion of a SQLite file.

`RefusingEventStore` is the assertion that matters. It delegates every READ to a real
in-memory `EventStore` and RAISES on every write method, so "the rebuild wrote to its
source" fails here as a test error rather than as a row nobody looks at. A no-op journal
that quietly forwarded to the store would pass a call-count assertion and fail this one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.knowledge.curation.deduplication import DeduplicationResult
from backend.knowledge.curation.graph_writer import WriteResult
from backend.knowledge.curation.pipeline import CurationResult
from backend.knowledge.curation.reconciliation import ReconcileTurnResult
from backend.knowledge.extraction.validator import ValidationResult
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED, ExtractionCache
from backend.knowledge.regeneration.log_regenerator import LogRegenerator, RebuildError
from backend.knowledge.regeneration.rebuild_journal import (
    EventStoreRebuildJournal,
    NullRebuildJournal,
)

STAGING_URI = "bolt://mist-neo4j-staging:7687"
LIVE_URI = "bolt://mist-neo4j:7687"

ONTOLOGY = "1.4.0"
EXTRACTION = "2026-06-14-r5"
MODEL_HASH = "test-model-hash"
TURN_TS = "2026-07-01T09:00:00+00:00"
EPOCH_TS = "2026-07-01T08:00:00+00:00"

TURN_ONE = "t-1"
TURN_TWO = "t-2"


class SourceWriteAttemptedError(AssertionError):
    """Raised when the rebuild writes to the store it is replaying FROM."""


class RefusingEventStore:
    """Delegates reads to a real EventStore; raises on every write method.

    Explicit delegation rather than `__getattr__`, matching `RecordingEventStore` in
    test_rebuild_scoping.py: the delegated surface IS the read contract
    `LogRegenerator` depends on, so a method the regenerator starts calling has to be
    added here consciously rather than silently passing through.
    """

    def __init__(self, inner: EventStore) -> None:
        self.inner = inner

    # -- reads: allowed --------------------------------------------------
    def get_all_turns_for_reextraction(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.inner.get_all_turns_for_reextraction(**kwargs)

    def get_turn_count(self) -> int:
        return self.inner.get_turn_count()

    # -- writes: refused -------------------------------------------------
    def create_reextraction_job(self, **kwargs: Any) -> None:
        raise SourceWriteAttemptedError(
            "rebuild called create_reextraction_job on the replay SOURCE store"
        )

    def checkpoint_reextraction_job(self, *args: Any) -> None:
        raise SourceWriteAttemptedError(
            "rebuild called checkpoint_reextraction_job on the replay SOURCE store"
        )

    def finalize_reextraction_job(self, **kwargs: Any) -> None:
        raise SourceWriteAttemptedError(
            "rebuild called finalize_reextraction_job on the replay SOURCE store"
        )


@dataclass(slots=True)
class RecordingCurationPipeline:
    """Substitutes for the staging graph pipeline; records what the replay loop curates."""

    event_ids: list[str] = field(default_factory=list)

    async def curate_and_store(
        self,
        validation_result: ValidationResult,
        event_id: str,
        session_id: str,
        source_metadata: Any = None,
        recorded_at: str | None = None,
    ) -> CurationResult:
        self.event_ids.append(event_id)
        return CurationResult(
            write_result=WriteResult(),
            dedup_result=DeduplicationResult(entities=[], merge_actions=[], entities_merged=0),
            reconcile_result=ReconcileTurnResult(),
            curation_time_ms=0.0,
            validated_entities=validation_result.entities,
            validated_relationships=validation_result.relationships,
        )


def _epoch() -> dict[str, Any]:
    return {
        "epoch_id": 1,
        "ontology_version": ONTOLOGY,
        "extraction_version": EXTRACTION,
        "model_hash": MODEL_HASH,
        "activated_at": EPOCH_TS,
    }


def _seed_store() -> EventStore:
    store = EventStore(":memory:")
    store.initialize()
    store.start_session("s-real", input_modality="text", origin="real")
    for index, event_id in enumerate((TURN_ONE, TURN_TWO)):
        store.append_turn(
            ConversationTurnEvent(
                session_id="s-real",
                turn_index=index,
                timestamp=datetime.fromisoformat(TURN_TS),
                user_utterance="I use Rust.",
                system_response="Noted.",
                ontology_version=ONTOLOGY,
                event_id=event_id,
            )
        )
    return store


def _warm_cache() -> ExtractionCache:
    cache = ExtractionCache(":memory:")
    cache.initialize()
    for event_id in (TURN_ONE, TURN_TWO):
        cache.put(
            event_id,
            ONTOLOGY,
            EXTRACTION,
            MODEL_HASH,
            outcome=OUTCOME_EXTRACTED,
            entities=[{"id": "rust", "type": "Technology", "display_name": "Rust"}],
            relationships=[],
            created_at=TURN_TS,
        )
    return cache


def _regenerator(source: Any, journal: Any) -> tuple[LogRegenerator, RecordingCurationPipeline]:
    recorder = RecordingCurationPipeline()
    return (
        LogRegenerator(
            event_store=source,
            extraction_cache=_warm_cache(),
            staging_curation_pipeline=recorder,
            journal=journal,
        ),
        recorder,
    )


class TestTheSourceStoreIsNeverWritten:
    """The assertion whose absence let a 'dry-run' command write to the live ledger."""

    @pytest.mark.asyncio
    async def test_rebuild_with_a_null_journal_writes_nothing_to_the_source_store(self):
        # Arrange
        source = RefusingEventStore(_seed_store())
        regen, recorder = _regenerator(source, NullRebuildJournal())

        # Act -- RefusingEventStore turns any write into a test failure
        report = await regen.rebuild(staging_uri=STAGING_URI, live_uri=LIVE_URI, epoch=_epoch())

        # Assert -- the replay still did its real work
        assert report.turns_processed == 2
        assert recorder.event_ids == [TURN_ONE, TURN_TWO]

    @pytest.mark.asyncio
    async def test_journal_rows_land_in_the_journal_store_not_the_source(self):
        # Arrange -- source refuses writes; the journal has its own separate store
        source = RefusingEventStore(_seed_store())
        journal_store = EventStore(":memory:")
        journal_store.initialize()
        regen, _ = _regenerator(source, EventStoreRebuildJournal(journal_store))

        # Act
        report = await regen.rebuild(staging_uri=STAGING_URI, live_uri=LIVE_URI, epoch=_epoch())

        # Assert -- the row exists, and it exists in the journal's store
        job = journal_store.get_reextraction_job(report.job_id)
        assert job is not None
        assert job["status"] == "completed"
        assert job["processed"] == 2


class TestResumeRequiresADurableJournal:
    """A non-durable journal cannot support resume, and must say so rather than mis-resume."""

    @pytest.mark.asyncio
    async def test_resume_is_refused_when_the_journal_is_not_durable(self):
        # Arrange
        source = RefusingEventStore(_seed_store())
        regen, _ = _regenerator(source, NullRebuildJournal())

        # Act / Assert
        with pytest.raises(RebuildError, match="durable"):
            await regen.rebuild(
                staging_uri=STAGING_URI,
                live_uri=LIVE_URI,
                epoch=_epoch(),
                job_id="rebuild-1-abcdef",
                resume_from=TURN_ONE,
            )

    def test_null_journal_declares_itself_non_durable(self):
        assert NullRebuildJournal().durable is False

    def test_event_store_journal_declares_itself_durable(self):
        store = EventStore(":memory:")
        store.initialize()

        assert EventStoreRebuildJournal(store).durable is True
