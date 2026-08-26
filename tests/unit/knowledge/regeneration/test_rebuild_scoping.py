"""The epoch's scoping role in `graph = f(seed@version, log, epoch)`, made real.

## The gap these tests close

`EventStore.get_all_turns_for_reextraction` has carried an `ontology_version` filter and
(since R1.4) a session `origin` discriminator. Eight tests asserted the filter reads back
correctly and four asserted `origin` is written and surfaced. None asserted that a CALLER
passed either one -- and the sole caller, `LogRegenerator.rebuild`, passed neither. So the
`epoch` parameter of the determinism contract selected nothing: a rebuild took EVERY turn
ever logged, then demanded the current epoch's extraction cache cover all of them. Every
turn predating one of the five `EXTRACTION_VERSION` bumps or the ontology bumps was a
guaranteed miss -> `ColdCacheError` -> abort before a single node was written. It passed
only because the live log is empty. Separately, probe and eval traffic marked `origin='test'`
was replayed straight into the canonical graph.

Two shapes of assertion, deliberately:

- **The call.** `TestTheSelectionCall` fails if `rebuild()` stops passing a scoping argument,
  independent of whether any turn happens to be filtered. This is the assertion whose absence
  let a fully-tested filter sit unused.
- **The consequence.** `TestScopingConsequences` seeds a store with turns under TWO ontology
  versions and a `origin='test'` session, then proves the rebuild replays only the right
  subset. Dropping either argument changes an observable outcome here, not just a call record.

`RecordingEventStore` delegates to a REAL SQLite-backed `EventStore` (`:memory:`, the
convention for event-store tests) rather than stubbing the reads, so a signature change on
`get_all_turns_for_reextraction` fails here instead of being silently absorbed.
"""

from __future__ import annotations

import logging
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
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator, ValidationResult
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED, ExtractionCache
from backend.knowledge.regeneration.log_regenerator import ColdCacheError, LogRegenerator
from backend.knowledge.regeneration.rebuild_journal import EventStoreRebuildJournal

# The REAL staging endpoint, not a per-file synthetic name. Connections here are
# fakes and never dial it, but `assert_rebuild_target_not_live` is an allowlist
# of disposable endpoints (hardened 2026-08-04 from a denylist that let
# `bolt://localhost:7687` through to the live graph), so a made-up hostname is
# now correctly refused. Naming the instance the rebuild actually targets also
# stops these tests from asserting against a topology that does not exist.
STAGING_URI = "bolt://mist-neo4j-staging:7687"
LIVE_URI = "bolt://mist-neo4j:7687"

CURRENT_ONTOLOGY = "1.4.0"
SUPERSEDED_ONTOLOGY = "1.0.0"
EXTRACTION_VERSION = "2026-06-14-r5"
MODEL_HASH = "test-model-hash"
TURN_TS = "2026-07-01T09:00:00+00:00"

# Ids are literal so an assertion names the turn it is about.
REAL_ONE = "t-real-1"
REAL_TWO = "t-real-2"
SUPERSEDED = "t-superseded-ontology"
TEST_ORIGIN = "t-test-origin"
ORPHAN = "t-orphan-session"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class RecordingEventStore:
    """Records the selection call, delegating every method to a real EventStore.

    Explicit delegation rather than `__getattr__` passthrough: the delegated surface IS the
    contract `LogRegenerator` depends on, so a method the regenerator starts calling must be
    added here consciously.
    """

    def __init__(self, inner: EventStore) -> None:
        self.inner = inner
        self.selection_calls: list[dict[str, Any]] = []

    def get_all_turns_for_reextraction(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.selection_calls.append(dict(kwargs))
        return self.inner.get_all_turns_for_reextraction(**kwargs)

    def get_turn_count(self) -> int:
        return self.inner.get_turn_count()

    # No job/checkpoint methods, deliberately. They were here while `LogRegenerator`
    # wrote its progress back into the store it was replaying -- the coupling that sent
    # a dry-run's rows to the LIVE ledger. Progress now goes to an injected journal, so
    # the replay source's contract is reads only, and a regression that reaches for a
    # write here fails with AttributeError instead of passing silently.

    @property
    def only_selection_call(self) -> dict[str, Any]:
        assert len(self.selection_calls) == 1, self.selection_calls
        return self.selection_calls[0]


@dataclass(slots=True)
class RecordingCurationPipeline:
    """Records what the replay loop hands to curation. Substitutes for the graph pipeline."""

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


@dataclass(slots=True)
class ScopedWorld:
    """One seeded event store + cache + regenerator, plus the epoch row they share."""

    store: EventStore
    recording_store: RecordingEventStore
    cache: ExtractionCache
    epoch: dict[str, Any]
    recorder: RecordingCurationPipeline
    regenerator: LogRegenerator


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------


def _append_turn(
    store: EventStore, *, session_id: str, event_id: str, ontology_version: str, turn_index: int = 0
) -> None:
    store.append_turn(
        ConversationTurnEvent(
            session_id=session_id,
            turn_index=turn_index,
            timestamp=datetime.fromisoformat(TURN_TS),
            user_utterance="I use Rust.",
            system_response="Noted.",
            ontology_version=ontology_version,
            event_id=event_id,
        )
    )


def _append_turn_with_no_session_row(store: EventStore, *, event_id: str) -> None:
    """Seed the only state that makes a joined `origin` NULL.

    `initialize()` adds `origin` as `NOT NULL DEFAULT 'real'` and SQLite back-fills every
    pre-existing row with that default, so a session row can never itself hold NULL. The
    residual case is a turn whose session row is absent -- reachable in a legacy database
    written with foreign keys off, which is what this reproduces.
    """
    conn = store._get_connection()
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        _append_turn(
            store,
            session_id="s-row-was-never-written",
            event_id=event_id,
            ontology_version=CURRENT_ONTOLOGY,
        )
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


def _warm(cache: ExtractionCache, epoch: dict[str, Any], event_id: str) -> None:
    cache.put(
        event_id,
        epoch["ontology_version"],
        epoch["extraction_version"],
        epoch["model_hash"],
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": "rust", "type": "Technology", "display_name": "Rust"}],
        relationships=[],
        created_at=TURN_TS,
    )


def build_world(*, cache_superseded_turn: bool = True, with_orphan: bool = False) -> ScopedWorld:
    """Seed a log spanning two ontology epochs and two session origins.

    Layout (4 turns, or 5 with the orphan):
      `s-real`        origin='real'  -> REAL_ONE, REAL_TWO   @ CURRENT_ONTOLOGY
      `s-real-legacy` origin='real'  -> SUPERSEDED           @ SUPERSEDED_ONTOLOGY
      `s-test`        origin='test'  -> TEST_ORIGIN          @ CURRENT_ONTOLOGY
      (no session row)               -> ORPHAN               @ CURRENT_ONTOLOGY

    A correctly scoped rebuild replays REAL_ONE and REAL_TWO -- and ORPHAN when present.
    """
    store = EventStore(db_path=":memory:")
    store.initialize()
    store.append_epoch(
        ontology_version=CURRENT_ONTOLOGY,
        extraction_version=EXTRACTION_VERSION,
        model_hash=MODEL_HASH,
        activated_at=TURN_TS,
    )
    epoch = store.get_current_epoch()
    assert epoch is not None

    store.start_session("s-real", input_modality="text", origin="real")
    store.start_session("s-real-legacy", input_modality="text", origin="real")
    store.start_session("s-test", input_modality="text", origin="test")

    _append_turn(store, session_id="s-real", event_id=REAL_ONE, ontology_version=CURRENT_ONTOLOGY)
    _append_turn(
        store,
        session_id="s-real",
        event_id=REAL_TWO,
        ontology_version=CURRENT_ONTOLOGY,
        turn_index=1,
    )
    _append_turn(
        store,
        session_id="s-real-legacy",
        event_id=SUPERSEDED,
        ontology_version=SUPERSEDED_ONTOLOGY,
    )
    _append_turn(
        store, session_id="s-test", event_id=TEST_ORIGIN, ontology_version=CURRENT_ONTOLOGY
    )
    if with_orphan:
        _append_turn_with_no_session_row(store, event_id=ORPHAN)

    cache = ExtractionCache(":memory:")
    cache.initialize()
    warm = [REAL_ONE, REAL_TWO, TEST_ORIGIN]
    if with_orphan:
        warm.append(ORPHAN)
    if cache_superseded_turn:
        warm.append(SUPERSEDED)
    for event_id in warm:
        _warm(cache, epoch, event_id)

    recording_store = RecordingEventStore(store)
    recorder = RecordingCurationPipeline()
    return ScopedWorld(
        store=store,
        recording_store=recording_store,
        cache=cache,
        epoch=epoch,
        recorder=recorder,
        regenerator=LogRegenerator(
            event_store=recording_store,
            extraction_cache=cache,
            staging_curation_pipeline=recorder,
            # Durable, and pointed at the SAME underlying store the log came from --
            # which is the legitimate wiring for a rebuild of record, and what makes
            # `test_the_job_ledger_totals_only_the_scoped_turns` still meaningful.
            # It is a deliberate choice here, not the default it used to be.
            journal=EventStoreRebuildJournal(store),
            # This file's assertions are about SCOPING (which turns get selected),
            # not about Stages 3-6, so the real (pure, no external dependency)
            # production components are the simplest correct wiring -- Task 6
            # (extraction-cache-phase-1).
            confidence_scorer=ConfidenceScorer(),
            temporal_resolver=TemporalResolver(),
            normalizer=EntityNormalizer(embedding_generator=None, executor=None),
            validator=ExtractionValidator(),
        ),
    )


async def rebuild(world: ScopedWorld, **overrides: Any):
    return await world.regenerator.rebuild(
        staging_uri=STAGING_URI, live_uri=LIVE_URI, epoch=world.epoch, **overrides
    )


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


class TestTheSelectionCall:
    """Fails the moment `rebuild()` stops passing a scoping argument.

    Deliberately independent of what the filters happen to exclude: the P1/L2 bug was a
    caller that never passed the arguments, and no outcome assertion over an EMPTY log --
    which is what the live log is -- could have detected that.
    """

    @pytest.mark.asyncio
    async def test_rebuild_passes_the_ontology_version_to_the_store(self):
        world = build_world()

        await rebuild(world)

        assert world.recording_store.only_selection_call["ontology_version"] == CURRENT_ONTOLOGY

    @pytest.mark.asyncio
    async def test_rebuild_passes_the_origin_filter_to_the_store(self):
        world = build_world()

        await rebuild(world)

        assert world.recording_store.only_selection_call["origins"] == ("real",)

    @pytest.mark.asyncio
    async def test_the_scoping_version_is_the_epoch_rows_not_the_module_constant(self):
        """The anti-drift assertion.

        The cache keys are derived from `epoch["ontology_version"]`. If the selection were
        scoped by `version_stamps.ONTOLOGY_VERSION` instead, the two would agree today and
        diverge silently the moment an older epoch is rebuilt -- the same two-authorities
        drift the version-stamp collapse removed. Here the epoch row deliberately carries a
        version the module constant never holds, so only sourcing from the row passes.
        """
        from backend.knowledge.version_stamps import ONTOLOGY_VERSION

        epoch_only_version = "9.9.9-epoch-only"
        assert epoch_only_version != ONTOLOGY_VERSION

        world = build_world()
        world.epoch = {**world.epoch, "ontology_version": epoch_only_version}
        _append_turn(
            world.store,
            session_id="s-real",
            event_id="t-epoch-only",
            ontology_version=epoch_only_version,
            turn_index=2,
        )
        _warm(world.cache, world.epoch, "t-epoch-only")

        report = await rebuild(world)

        assert world.recording_store.only_selection_call["ontology_version"] == epoch_only_version
        assert world.recorder.event_ids == ["t-epoch-only"]
        assert report.turns_processed == 1

    @pytest.mark.asyncio
    async def test_the_resume_cursor_is_still_passed_alongside_the_new_scoping(self):
        """Adding scoping must not displace the argument that was already there."""
        world = build_world()

        await rebuild(world, job_id="job-resume", resume_from=REAL_ONE)

        call = world.recording_store.only_selection_call
        assert call["after_event_id"] == REAL_ONE
        assert call["ontology_version"] == CURRENT_ONTOLOGY
        assert call["origins"] == ("real",)


# ---------------------------------------------------------------------------
# The consequence
# ---------------------------------------------------------------------------


class TestScopingConsequences:
    """Two ontology versions and two session origins in one store, as the register asked."""

    @pytest.mark.asyncio
    async def test_only_the_current_epochs_real_turns_are_replayed(self):
        world = build_world()

        report = await rebuild(world)

        assert world.recorder.event_ids == [REAL_ONE, REAL_TWO]
        assert report.turns_processed == 2

    @pytest.mark.asyncio
    async def test_a_turn_from_a_superseded_ontology_epoch_is_not_replayed(self):
        # Arrange: the superseded turn IS cached under the current stamp pair, so its exclusion
        # can only be the ontology filter -- not an incidental ColdCacheError.
        world = build_world(cache_superseded_turn=True)
        assert (
            world.cache.get(
                SUPERSEDED,
                world.epoch["extraction_version"],
                world.epoch["model_hash"],
            )
            is not None
        )

        await rebuild(world)

        assert SUPERSEDED not in world.recorder.event_ids

    @pytest.mark.asyncio
    async def test_a_cold_turn_from_a_superseded_epoch_no_longer_aborts_the_rebuild(self):
        """P1's live failure mode, in miniature.

        Unscoped, the selection includes a turn logged under an older ontology, the current
        epoch's cache cannot cover it, and `_assert_cache_coverage` aborts the whole rebuild
        before writing anything. Scoped, that turn is not this epoch's business.
        """
        world = build_world(cache_superseded_turn=False)
        # The arrangement is real, not assumed: unscoped the turn is selectable, and it is
        # genuinely absent from the cache under this epoch.
        unscoped = world.store.get_all_turns_for_reextraction()
        assert SUPERSEDED in {t["event_id"] for t in unscoped}
        assert (
            world.cache.get(
                SUPERSEDED,
                world.epoch["extraction_version"],
                world.epoch["model_hash"],
            )
            is None
        )

        report = await rebuild(world)

        assert report.turns_processed == 2
        assert world.recorder.event_ids == [REAL_ONE, REAL_TWO]

    @pytest.mark.asyncio
    async def test_test_origin_traffic_is_not_replayed_into_the_canonical_graph(self):
        # Arrange: cached under the current stamp pair and tagged with the current ontology, so
        # only the origin guard can exclude it.
        world = build_world()

        await rebuild(world)

        assert TEST_ORIGIN not in world.recorder.event_ids

    @pytest.mark.asyncio
    async def test_test_origin_traffic_is_replayed_when_explicitly_requested(self):
        """The golden log's path: fixture traffic replays only when the caller declares it."""
        world = build_world()

        report = await rebuild(world, origins=("test",))

        assert world.recorder.event_ids == [TEST_ORIGIN]
        assert report.turns_processed == 1

    @pytest.mark.asyncio
    async def test_a_turn_whose_session_row_is_missing_is_counted_as_real(self):
        """The NULL-origin ruling: unmarked history is replayed, not silently dropped.

        Excluding it would lose history from a graph whose whole contract is that it is a
        total function of the log; including it matches both the `NOT NULL DEFAULT 'real'`
        back-fill and `start_session`'s own default.
        """
        world = build_world(with_orphan=True)

        report = await rebuild(world)

        assert ORPHAN in world.recorder.event_ids
        assert report.turns_processed == 3

    @pytest.mark.asyncio
    async def test_the_job_ledger_totals_only_the_scoped_turns(self):
        """`total_events` is the rebuild's own record of what it set out to do."""
        world = build_world()

        report = await rebuild(world)
        job = world.store.get_reextraction_job(report.job_id)

        assert job is not None
        assert job["total_events"] == 2
        assert world.store.get_turn_count() == 4


class TestScopingIsObservable:
    """Scoping can now select nothing from a populated log. That must not be silent."""

    @pytest.mark.asyncio
    async def test_a_rebuild_that_selects_nothing_from_a_populated_log_warns(self, caplog):
        world = build_world()
        world.epoch = {**world.epoch, "ontology_version": "0.0.0-matches-no-turn"}

        with caplog.at_level(logging.WARNING):
            report = await rebuild(world)

        assert report.turns_processed == 0
        assert "selected 0 of 4 logged turns" in caplog.text

    @pytest.mark.asyncio
    async def test_a_normal_rebuild_logs_the_scope_it_selected(self, caplog):
        world = build_world()

        with caplog.at_level(logging.INFO):
            await rebuild(world)

        assert "2 of 4 logged turns selected" in caplog.text


class TestEmptyOriginsIsRejected:
    @pytest.mark.asyncio
    async def test_an_empty_origins_tuple_raises_rather_than_replaying_nothing(self):
        """An empty tuple would select nothing and be indistinguishable from an empty log."""
        world = build_world()

        with pytest.raises(ValueError, match="non-empty"):
            await rebuild(world, origins=())


class TestColdCacheStillFires:
    """The scoping must not have turned the coverage gate off for turns it DOES select."""

    @pytest.mark.asyncio
    async def test_a_hole_in_the_current_epochs_cache_still_refuses(self):
        world = build_world()
        world.epoch = {**world.epoch, "extraction_version": "2026-01-01-r0"}

        with pytest.raises(ColdCacheError, match="uncached"):
            await rebuild(world)
