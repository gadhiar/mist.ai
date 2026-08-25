"""Shared fixtures for LogRegenerator replay tests (Task 6, fix round 1).

`regenerator_factory` builds a `LogRegenerator` wired to test doubles at the
documented I/O boundaries (`tests/CLAUDE.md`'s mocking table): a real
in-memory `EventStore` and a real in-memory `ExtractionCache` (both SQLite --
that table's "Filesystem / SQLite" row prescribes the real store over a
hand-rolled fake, precisely so a signature drift on `.get()`/`.put()` fails
here instead of being silently absorbed), `_ContentRecordingCurationPipeline`
(a local superset of `test_rebuild_scoping.RecordingCurationPipeline` -- fix
round 1 needs the actual relationships that reached curation, not just the
event_id, to prove a Stage-3 confidence adjustment survived replay) as the
curation-pipeline double, and `NullRebuildJournal` (a real, already-shipped
no-op production class -- not a fake) as the journal.

Stages 3-6 (confidence_scorer, temporal_resolver, normalizer, validator)
default to the REAL production components. This is not for realism alone:
none of `ConfidenceScorer()`, `TemporalResolver()`, or `ExtractionValidator()`
takes any argument that could reach a graph or a model, and `EntityNormalizer`
issues no graph queries at all (`normalize()` docstring). Only when a test
passes `stage_sink` are they swapped for recording fakes that prove replay's
CALL ORDER -- the real components have no externally observable order to
assert on.
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
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator, ValidationResult
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED, OUTCOME_SKIPPED, ExtractionCache
from backend.knowledge.regeneration.log_regenerator import LogRegenerator
from backend.knowledge.regeneration.rebuild_journal import NullRebuildJournal

# Must match REBUILD_ARGS["epoch"] in test_replay_runs_stages.py exactly: the turn
# seeded below is filtered by `epoch["ontology_version"]` and looked up in the
# cache by `epoch["extraction_version"]` + `epoch["model_hash"]`.
_ONTOLOGY_VERSION = "1.4.0"
_EXTRACTION_VERSION = "2026-06-14-r5"
_MODEL_HASH = "test-model-hash"
_TURN_EVENT_ID = "t-replay-1"
_TURN_SESSION_ID = "s-replay"
_TURN_TS = "2026-07-01T09:00:00+00:00"


# ---------------------------------------------------------------------------
# Stage 3-6 recording fakes -- only used when a test passes stage_sink.
# ---------------------------------------------------------------------------


class _RecordingConfidenceScorer:
    """Stands in for ConfidenceScorer when a test needs to observe call order."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def adjust_confidence(self, extraction: Any) -> Any:
        self._sink.append("confidence")
        return extraction


class _RecordingTemporalResolver:
    """Stands in for TemporalResolver when a test needs to observe call order."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def resolve(self, extraction: Any, reference_date: datetime) -> Any:
        self._sink.append("temporal")
        return extraction


class _RecordingNormalizer:
    """Stands in for EntityNormalizer when a test needs to observe call order.

    `normalize` is async on the real class (`grep -n "async def normalize"
    backend/knowledge/extraction/normalizer.py`) -- a sync recorder cannot stand in
    for it, so this is a distinct fake rather than a shared callable.
    """

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    async def normalize(self, extraction: Any) -> Any:
        self._sink.append("normalize")
        return extraction


class _RecordingValidator:
    """Stands in for ExtractionValidator when a test needs to observe call order."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def validate(self, extraction: Any) -> ValidationResult:
        self._sink.append("validate")
        return ValidationResult(
            valid=True,
            entities=extraction.entities,
            relationships=extraction.relationships,
        )


# ---------------------------------------------------------------------------
# Curation-pipeline double.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ContentRecordingCurationPipeline:
    """Records what the replay loop hands to curation -- event_ids AND content.

    A strict superset of `test_rebuild_scoping.RecordingCurationPipeline` (kept
    `event_ids` so `test_a_skipped_row_replays_as_a_no_op_not_a_failure`'s
    `regenerator._curation.event_ids == []` assertion is unaffected): fix round 1
    needs the actual relationships that reached curation to prove a Stage-3
    confidence adjustment survived replay, which an event-id-only recorder cannot
    show.
    """

    event_ids: list[str] = field(default_factory=list)
    relationships: list[list[dict]] = field(default_factory=list)

    async def curate_and_store(
        self,
        validation_result: ValidationResult,
        event_id: str,
        session_id: str,
        source_metadata: Any = None,
        recorded_at: str | None = None,
    ) -> CurationResult:
        self.event_ids.append(event_id)
        self.relationships.append(validation_result.relationships)
        return CurationResult(
            write_result=WriteResult(),
            dedup_result=DeduplicationResult(entities=[], merge_actions=[], entities_merged=0),
            reconcile_result=ReconcileTurnResult(),
            curation_time_ms=0.0,
            validated_entities=validation_result.entities,
            validated_relationships=validation_result.relationships,
        )


# ---------------------------------------------------------------------------
# I/O-boundary doubles: real SQLite stores, seeded with exactly one turn.
# ---------------------------------------------------------------------------


def _build_event_store(*, utterance: str = "I use Rust.") -> EventStore:
    """One turn, current ontology, origin='real' -- selected by rebuild()'s default scope."""
    store = EventStore(db_path=":memory:")
    store.initialize()
    store.start_session(_TURN_SESSION_ID, input_modality="text", origin="real")
    store.append_turn(
        ConversationTurnEvent(
            session_id=_TURN_SESSION_ID,
            turn_index=0,
            timestamp=datetime.fromisoformat(_TURN_TS),
            user_utterance=utterance,
            system_response="Noted.",
            ontology_version=_ONTOLOGY_VERSION,
            event_id=_TURN_EVENT_ID,
        )
    )
    return store


def _build_cache(
    *,
    outcome: str,
    skip_reason: str | None,
    entities: list[dict] | None = None,
    relationships: list[dict] | None = None,
) -> ExtractionCache:
    """One row for the seeded turn, keyed on the epoch's extraction stamp pair."""
    cache = ExtractionCache(":memory:")
    cache.initialize()
    if outcome == OUTCOME_SKIPPED:
        cache_entities, cache_relationships = [], []
    else:
        cache_entities = (
            entities if entities is not None else [{"id": "rust", "type": "Technology"}]
        )
        cache_relationships = relationships if relationships is not None else []
    cache.put(
        _TURN_EVENT_ID,
        _ONTOLOGY_VERSION,
        _EXTRACTION_VERSION,
        _MODEL_HASH,
        outcome=outcome,
        skip_reason=skip_reason,
        entities=cache_entities,
        relationships=cache_relationships,
        created_at=_TURN_TS,
    )
    return cache


@pytest.fixture
def regenerator_factory():
    """Return a factory building a LogRegenerator over a seeded one-turn world.

    All arguments to the returned callable are keyword-only:
        stage_sink: when given, Stages 3-6 are swapped for recording fakes that
            append their stage name to this list instead of running real logic.
        cached_outcome: OUTCOME_EXTRACTED (default) or OUTCOME_SKIPPED.
        skip_reason: required (and only valid) when cached_outcome is OUTCOME_SKIPPED
            -- ExtractionCache.put enforces this itself.
        llm: when given, wired into the (real) normalizer's `embedding_generator`
            slot -- the one constructor parameter Stages 3-6 collectively have for
            an external dependency, and the one `normalize()` never reads after
            assignment. Lets a test prove nothing touches it.
        utterance: the seeded turn's `user_utterance` (default "I use Rust.").
            Fix round 1: this is what `ConfidenceScorer.adjust_confidence` reads
            as `extraction.source_utterance` during replay.
        entities: overrides the cached (raw Stage-2) entities list. Defaults to
            a single Technology entity when cached_outcome is OUTCOME_EXTRACTED.
        relationships: overrides the cached (raw Stage-2) relationships list.
            Defaults to `[]`.
    """

    def _build(
        *,
        stage_sink: list[str] | None = None,
        cached_outcome: str = OUTCOME_EXTRACTED,
        skip_reason: str | None = None,
        llm: Any = None,
        utterance: str = "I use Rust.",
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
    ) -> LogRegenerator:
        if stage_sink is not None:
            confidence_scorer: Any = _RecordingConfidenceScorer(stage_sink)
            temporal_resolver: Any = _RecordingTemporalResolver(stage_sink)
            normalizer: Any = _RecordingNormalizer(stage_sink)
            validator: Any = _RecordingValidator(stage_sink)
        else:
            confidence_scorer = ConfidenceScorer()
            temporal_resolver = TemporalResolver()
            normalizer = EntityNormalizer(embedding_generator=llm, executor=None)
            validator = ExtractionValidator()

        return LogRegenerator(
            event_store=_build_event_store(utterance=utterance),
            extraction_cache=_build_cache(
                outcome=cached_outcome,
                skip_reason=skip_reason,
                entities=entities,
                relationships=relationships,
            ),
            staging_curation_pipeline=_ContentRecordingCurationPipeline(),
            journal=NullRebuildJournal(),
            confidence_scorer=confidence_scorer,
            temporal_resolver=temporal_resolver,
            normalizer=normalizer,
            validator=validator,
        )

    return _build
