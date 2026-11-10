"""A rebuild reads its epoch's identity namespace and no other.

The cache can hold, for one turn and one stamp pair, a 3-part row (the
namespace the first identity bound to the pair claims) and keyed rows (one per
`serving_identity+dispatch_identity` that went keyed). `LogRegenerator` takes
the namespace from `epoch["identity_namespace"]` (None, or absent in an epoch
dict built before bindings existed: the 3-part one) for both its coverage
check and its replay reads. Each test holds both kinds of row, with different
entities, so the entities that reach curation show which kind was read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.knowledge.curation.deduplication import DeduplicationResult
from backend.knowledge.curation.graph_writer import WriteResult
from backend.knowledge.curation.pipeline import CurationResult
from backend.knowledge.curation.reconciliation import ReconcileTurnResult
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED, ExtractionCache, keyed_namespace
from backend.knowledge.regeneration.log_regenerator import ColdCacheError, LogRegenerator
from backend.knowledge.regeneration.rebuild_journal import NullRebuildJournal
from tests.mocks.seeder import FakeStagingSeeder
from tests.unit.knowledge.regeneration.conftest import (
    _RecordingConfidenceScorer,
    _RecordingNormalizer,
    _RecordingTemporalResolver,
    _RecordingValidator,
)

ONTOLOGY_VERSION = "1.4.0"
EXTRACTION_VERSION = "2026-06-14-r5"
MODEL_HASH = "test-model-hash"
SESSION_ID = "s-namespace"
STAGING_URI = "bolt://localhost:7689"
LIVE_URI = "bolt://mist-neo4j:7687"
KEYED = keyed_namespace("si1-" + "a" * 64, "di1-" + "b" * 64)
EVENT_IDS = ("t-ns-0", "t-ns-1")


@dataclass
class _EntityRecordingCuration:
    """What the replay loop handed to curation: (event_id, entity ids), in order."""

    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    async def curate_and_store(
        self, validation_result, event_id, session_id, source_metadata=None, recorded_at=None
    ) -> CurationResult:
        self.calls.append((event_id, [e["id"] for e in validation_result.entities]))
        return CurationResult(
            write_result=WriteResult(),
            dedup_result=DeduplicationResult(entities=[], merge_actions=[], entities_merged=0),
            reconcile_result=ReconcileTurnResult(),
            curation_time_ms=0.0,
            validated_entities=list(validation_result.entities),
            validated_relationships=list(validation_result.relationships),
        )


def _event_store() -> EventStore:
    store = EventStore(db_path=":memory:")
    store.initialize()
    store.start_session(SESSION_ID, input_modality="text", origin="real")
    for index, event_id in enumerate(EVENT_IDS):
        store.append_turn(
            ConversationTurnEvent(
                session_id=SESSION_ID,
                turn_index=index,
                timestamp=datetime.fromisoformat(f"2026-07-01T09:0{index}:00+00:00"),
                user_utterance=f"I use tool{index}.",
                system_response="Noted.",
                ontology_version=ONTOLOGY_VERSION,
                event_id=event_id,
            )
        )
    return store


def _put(cache: ExtractionCache, event_id: str, namespace: str | None, marker: str) -> None:
    cache.put(
        event_id,
        ONTOLOGY_VERSION,
        EXTRACTION_VERSION,
        MODEL_HASH,
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": f"{marker}-{event_id}", "type": "Technology", "name": marker}],
        relationships=[],
        created_at="2026-07-01T09:00:00+00:00",
        identity_namespace=namespace,
    )


def _cache(*, three_part: bool = True, keyed: bool = True) -> ExtractionCache:
    cache = ExtractionCache(":memory:")
    cache.initialize()
    for event_id in EVENT_IDS:
        if three_part:
            _put(cache, event_id, None, "three-part")
        if keyed:
            _put(cache, event_id, KEYED, "keyed")
    return cache


def _regenerator(cache: ExtractionCache) -> tuple[LogRegenerator, _EntityRecordingCuration]:
    sink: list[str] = []
    curation = _EntityRecordingCuration()
    regenerator = LogRegenerator(
        event_store=_event_store(),
        extraction_cache=cache,
        staging_curation_pipeline=curation,
        journal=NullRebuildJournal(),
        staging_seeder=FakeStagingSeeder(),
        confidence_scorer=_RecordingConfidenceScorer(sink),
        temporal_resolver=_RecordingTemporalResolver(sink),
        normalizer=_RecordingNormalizer(sink),
        validator=_RecordingValidator(sink),
    )
    return regenerator, curation


def _epoch(**extra) -> dict:
    return {
        "epoch_id": 1,
        "ontology_version": ONTOLOGY_VERSION,
        "extraction_version": EXTRACTION_VERSION,
        "model_hash": MODEL_HASH,
        "activated_at": "2026-07-01T00:00:00+00:00",
        **extra,
    }


async def _rebuild(regenerator: LogRegenerator, epoch: dict):
    return await regenerator.rebuild(
        staging_uri=STAGING_URI, live_uri=LIVE_URI, epoch=epoch, min_seed_nodes=1
    )


class TestRebuildReadsTheEpochsNamespace:
    @pytest.mark.asyncio
    async def test_a_keyed_epoch_reads_its_keyed_rows_only(self):
        regenerator, curation = _regenerator(_cache())

        report = await _rebuild(regenerator, _epoch(identity_namespace=KEYED))

        assert report.turns_processed == 2
        assert curation.calls == [(e, [f"keyed-{e}"]) for e in EVENT_IDS]

    @pytest.mark.asyncio
    async def test_a_3_part_epoch_reads_its_3_part_rows_only(self):
        regenerator, curation = _regenerator(_cache())

        report = await _rebuild(regenerator, _epoch(identity_namespace=None))

        assert report.turns_processed == 2
        assert curation.calls == [(e, [f"three-part-{e}"]) for e in EVENT_IDS]

    @pytest.mark.asyncio
    async def test_an_epoch_dict_without_the_key_reads_the_3_part_rows(self):
        # An epoch dict built from a raw ledger row (no binding read) is the
        # 3-part namespace, as every epoch was before bindings existed.
        regenerator, curation = _regenerator(_cache())

        await _rebuild(regenerator, _epoch())

        assert curation.calls == [(e, [f"three-part-{e}"]) for e in EVENT_IDS]

    @pytest.mark.asyncio
    async def test_a_keyed_epoch_with_only_3_part_rows_is_cold(self):
        regenerator, curation = _regenerator(_cache(keyed=False))

        with pytest.raises(ColdCacheError, match="identity_namespace"):
            await _rebuild(regenerator, _epoch(identity_namespace=KEYED))

        assert curation.calls == []

    @pytest.mark.asyncio
    async def test_a_3_part_epoch_with_only_keyed_rows_is_cold(self):
        regenerator, curation = _regenerator(_cache(three_part=False))

        with pytest.raises(ColdCacheError):
            await _rebuild(regenerator, _epoch(identity_namespace=None))

        assert curation.calls == []
