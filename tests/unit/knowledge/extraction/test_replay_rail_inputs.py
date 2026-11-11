"""The two replay paths hand the cache row's rail inputs to the normalizer.

`apply_cached_turn` and the log regenerator read a turn back from the extraction cache.
The rails need more of that row than its payload: the batch record and the turn plan. A
row that carries neither is normalized as before.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
from backend.knowledge.extraction.pipeline import ExtractionPipeline, TurnToApply
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED, ExtractionCache
from backend.knowledge.regeneration.log_regenerator import LogRegenerator
from backend.knowledge.regeneration.rebuild_journal import NullRebuildJournal
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.mocks.seeder import FakeStagingSeeder
from tests.unit.extraction_backlog.fakes import FakeGraphCuration

EPOCH = {
    "epoch_id": 1,
    "ontology_version": "1.4.0",
    "extraction_version": "2026-06-14-r5",
    "model_hash": "test-model-hash",
    "activated_at": "2026-08-18T00:00:00+00:00",
}
_STAMP = "2026-07-01T09:00:00+00:00"
_BATCH_RECORD = [{"units": [], "context": [], "payload": None}]
_TURN_PLAN = {"entities": [], "templates": [], "back_references": []}


class _RecordingNormalizer:
    """Stands in for `EntityNormalizer` and keeps the keyword arguments of each call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def normalize(self, extraction, **row):
        self.calls.append(row)
        return extraction


class _CacheWithRailColumns(ExtractionCache):
    """A cache whose rows carry the batch record and the plan, as a later task stores them."""

    def get(self, *args, **kwargs):
        row = super().get(*args, **kwargs)
        if row is not None:
            row["batch_record"] = _BATCH_RECORD
            row["turn_plan"] = _TURN_PLAN
        return row


def _world(cache: ExtractionCache) -> EventStore:
    store = EventStore(db_path=":memory:")
    store.initialize()
    cache.initialize()
    store.start_session("s-a", input_modality="text", origin="real")
    event_id = store.append_turn(
        ConversationTurnEvent(
            session_id="s-a",
            turn_index=0,
            timestamp=datetime.fromisoformat(_STAMP),
            user_utterance="Ilse keeps a boat.",
            system_response="ok",
            ontology_version=EPOCH["ontology_version"],
            event_id="evt-s-a-0",
        )
    )
    cache.put(
        event_id,
        EPOCH["ontology_version"],
        EPOCH["extraction_version"],
        EPOCH["model_hash"],
        created_at=_STAMP,
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": "user", "type": "User", "name": "User"}],
        relationships=[],
    )
    return store


def _stages(normalizer) -> dict:
    return {
        "confidence_scorer": ConfidenceScorer(),
        "temporal_resolver": TemporalResolver(),
        "normalizer": normalizer,
        "validator": ExtractionValidator(),
    }


class _NoProgress:
    curated = False

    def mark_curated(self) -> None:
        pass

    def mark_applied(self) -> None:
        pass


async def _apply(cache: ExtractionCache) -> list[dict]:
    store = _world(cache)
    normalizer = _RecordingNormalizer()
    embeddings = FakeEmbeddingGenerator()
    pipeline = ExtractionPipeline(
        preprocessor=PreProcessor(),
        extractor=OntologyConstrainedExtractor(build_test_config(), llm=FakeLLM()),
        graph_store=GraphStore(FakeNeo4jConnection(), embeddings),
        curation_pipeline=FakeGraphCuration(),  # type: ignore[arg-type]
        embedding_provider=embeddings,
        extraction_cache=cache,
        rebuild_stamps=RebuildStamps(
            ontology_version=EPOCH["ontology_version"],
            extraction_version=EPOCH["extraction_version"],
            model_hash=EPOCH["model_hash"],
        ),
        **_stages(normalizer),
    )
    for turn in store.get_all_turns_for_reextraction():
        await pipeline.apply_cached_turn(
            TurnToApply(
                event_id=turn["event_id"],
                session_id=turn["session_id"],
                user_utterance=turn["user_utterance"],
                recorded_at=turn["timestamp"],
            ),
            cache.get(turn["event_id"], EPOCH["extraction_version"], EPOCH["model_hash"]),
            _NoProgress(),
        )
    return normalizer.calls


async def _rebuild(cache: ExtractionCache) -> list[dict]:
    store = _world(cache)
    normalizer = _RecordingNormalizer()
    regenerator = LogRegenerator(
        event_store=store,
        extraction_cache=cache,
        staging_curation_pipeline=FakeGraphCuration(),
        journal=NullRebuildJournal(),
        staging_seeder=FakeStagingSeeder(),
        **_stages(normalizer),
    )
    await regenerator.rebuild(
        staging_uri="bolt://mist-neo4j-staging:7687",
        live_uri="bolt://mist-neo4j:7687",
        epoch=EPOCH,
        min_seed_nodes=1,
    )
    return normalizer.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", [_apply, _rebuild], ids=["apply_cached_turn", "rebuild"])
async def test_a_replay_hands_the_rows_batch_record_and_plan_to_the_normalizer(replay):
    calls = await replay(_CacheWithRailColumns(":memory:"))

    assert calls == [{"batch_record": _BATCH_RECORD, "turn_plan": _TURN_PLAN}]


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", [_apply, _rebuild], ids=["apply_cached_turn", "rebuild"])
async def test_a_replay_of_a_row_without_them_calls_the_normalizer_as_before(replay):
    calls = await replay(ExtractionCache(":memory:"))

    assert calls == [{}]
