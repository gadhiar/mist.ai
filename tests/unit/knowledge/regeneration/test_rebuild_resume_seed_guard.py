"""A resumed rebuild is refused by the seed guard (MIS-177 D1), by design.

`LogRegenerator.rebuild()` applies the seed before its replay loop on every
run, the resume path included. A fresh run's caller empties staging first
(`cutover.py` `deps.wipe_staging()`, `mist_admin.py` `_build_once`), so the
seed guard sees an empty graph. A resume does not: staging already holds what
the first run replayed, and that is extraction-written data the seed would
adopt. The lead's ruling (MIS-177) is to keep that refusal rather than build a
wipe-then-resume path, so this pins it: the guard raises before any seed
write and before any turn is replayed.

The seeder here is the real `StagingSeeder` over a fake connection, not
`FakeStagingSeeder`: the refusal lives in `apply_seed_documents`, which the
fake seeder never calls.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from backend.errors import SeedTargetNotSeedOnlyError
from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED, ExtractionCache
from backend.knowledge.regeneration.log_regenerator import LogRegenerator
from backend.knowledge.regeneration.rebuild_gate import RebuildVacuityError
from backend.knowledge.regeneration.rebuild_journal import EventStoreRebuildJournal
from backend.knowledge.regeneration.staging_seeder import StagingSeeder
from backend.knowledge.seed.models import SeedDocument, SeedFact, SeedNode
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.unit.knowledge.regeneration.test_rebuild_journal_isolation import (
    RecordingCurationPipeline,
)
from tests.unit.knowledge.seed.seed_guard_rows import seed_guard_router

STAGING_URI = "bolt://mist-neo4j-staging:7687"
LIVE_URI = "bolt://mist-neo4j:7687"

ONTOLOGY = "1.4.0"
EXTRACTION = "2026-06-14-r5"
MODEL_HASH = "test-model-hash"
TURN_TS = "2026-07-01T09:00:00+00:00"
EPOCH_TS = "2026-07-01T08:00:00+00:00"
JOB_ID = "rebuild-1-resume"

TURN_ONE = "t-1"
TURN_TWO = "t-2"

# What staging holds after the first run replayed TURN_ONE: the `rust` node
# `CurationGraphWriter._upsert_entity` wrote (an `:__Entity__` node with no
# `seed_version`, stamped, `provenance='extraction'`). The guard's two
# statements count it in (a), (b) and (c) alike.
_PARTLY_REPLAYED_RESET_GUARD = [{"nodes": 1, "relationships": 0}]
_PARTLY_REPLAYED_SEED_GUARD = [
    {
        "stamped_nodes": 1,
        "extraction_nodes": 1,
        "stamped_relationships": 0,
        "extraction_relationships": 0,
    }
]


def _epoch() -> dict[str, Any]:
    return {
        "epoch_id": 1,
        "ontology_version": ONTOLOGY,
        "extraction_version": EXTRACTION,
        "model_hash": MODEL_HASH,
        "activated_at": EPOCH_TS,
    }


def _event_store() -> EventStore:
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


def _cache() -> ExtractionCache:
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


def _documents() -> list[SeedDocument]:
    return [
        SeedDocument(
            seed_version="resume-seed-1",
            nodes=[
                SeedNode(id="mist-identity", type="MistIdentity"),
                SeedNode(id="mist-trait-warm", type="MistTrait"),
            ],
            facts=[
                SeedFact(subject="mist-identity", predicate="HAS_TRAIT", object="mist-trait-warm")
            ],
            body="MIST is warm.",
            source_path=Path("mist.md"),
            partition="__SelfModel__",
        )
    ]


class _Embedder:
    def generate_embedding(self, text: str) -> list[float]:
        return [0.1] * 384


def _regenerator(
    staging: FakeNeo4jConnection,
) -> tuple[LogRegenerator, RecordingCurationPipeline]:
    journal_store = EventStore(":memory:")
    journal_store.initialize()
    recorder = RecordingCurationPipeline()
    regenerator = LogRegenerator(
        event_store=_event_store(),
        extraction_cache=_cache(),
        staging_curation_pipeline=recorder,
        # Durable, because `rebuild()` refuses a resume on a non-durable journal
        # before it reaches the seed (test_rebuild_journal_isolation.py).
        journal=EventStoreRebuildJournal(journal_store),
        staging_seeder=StagingSeeder(
            connection=staging,
            documents=_documents(),
            seed_version="resume-seed-1",
            embedding_generator=_Embedder(),
            expected_dimension=384,
        ),
        confidence_scorer=ConfidenceScorer(),
        temporal_resolver=TemporalResolver(),
        normalizer=EntityNormalizer(embedding_generator=None, executor=None),
        validator=ExtractionValidator(),
    )
    return regenerator, recorder


async def _resume(regenerator: LogRegenerator) -> Any:
    return await regenerator.rebuild(
        staging_uri=STAGING_URI,
        live_uri=LIVE_URI,
        epoch=_epoch(),
        min_seed_nodes=1,
        job_id=JOB_ID,
        resume_from=TURN_ONE,
    )


class TestAResumeIsRefusedByTheSeedGuard:
    @pytest.mark.asyncio
    async def test_a_partly_replayed_staging_graph_refuses_the_resume_before_any_write(self):
        staging = FakeNeo4jConnection(
            query_router=seed_guard_router(
                reset_guard_rows=_PARTLY_REPLAYED_RESET_GUARD,
                seed_guard_rows=_PARTLY_REPLAYED_SEED_GUARD,
            )
        )
        regenerator, recorder = _regenerator(staging)

        with pytest.raises(SeedTargetNotSeedOnlyError, match="Refusing applying seed documents"):
            await _resume(regenerator)

        assert staging.writes == [], (
            "the seed wrote to a partly replayed staging graph before refusing it; the "
            "guard must precede every seed write"
        )
        assert recorder.event_ids == [], "the resume replayed a turn after the seed refused"

    @pytest.mark.asyncio
    async def test_the_same_resume_over_a_clean_graph_reaches_the_seed_writes(self):
        """Non-vacuity: the refusal above comes from the graph, not the resume path.

        With clean guard rows the same resume gets past the guard and the seed
        writes. The fake answers no embedding read, so the embedding gate then
        stops the rebuild; that stop is not what this test is about.
        """
        staging = FakeNeo4jConnection(query_router=seed_guard_router())
        regenerator, _ = _regenerator(staging)

        with pytest.raises(RebuildVacuityError, match="embedding gate FAILED"):
            await _resume(regenerator)

        assert any("MERGE (n:__SelfModel__" in query for query, _ in staging.writes)
