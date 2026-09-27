"""`ExtractionPipeline.apply_cached_turn` must curate exactly what a rebuild curates.

The backlog's apply step and `LogRegenerator.rebuild`'s replay loop are two
callers of the same idea -- cached raw Stage-2 output -> Stages 3-6 ->
`curate_and_store` -- and `live == rebuilt` holds only if they hand curation
identical arguments. This feeds the SAME cached rows through both, with the
same real Stage 3-6 components and a recording curation double, and compares
every `curate_and_store` call field by field (dataclass equality, so a field
added to `CurationCall` later is compared too).

The rows cover the branches that differ between the in-process path and the
rebuild: a `skipped` row (no-op), an extracted row whose validation leaves no
entities (the rebuild still calls curation; the in-process path would not), a
hedged utterance (Stage 3 reads `source_utterance`), and a relative date
(Stage 4 anchors on the logged timestamp).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
from backend.knowledge.extraction.pipeline import ExtractionPipeline, TurnToApply
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_DUPLICATE,
    ExtractionCache,
)
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

# (session, turn_index, timestamp, utterance, cached-row kwargs)
TURNS = [
    (
        "s-a",
        0,
        "2026-07-01T09:00:00+00:00",
        "I think Alice uses Rust.",
        {
            "outcome": OUTCOME_EXTRACTED,
            "entities": [
                {"id": "user", "type": "User", "name": "User"},
                {"id": "rust", "type": "Technology", "name": "Rust"},
            ],
            "relationships": [
                {
                    "source": "user",
                    "target": "rust",
                    "type": "USES",
                    "properties": {"confidence": 0.9},
                }
            ],
        },
    ),
    (
        "s-b",
        0,
        "2026-07-01T09:05:00+00:00",
        "Hello again friend",
        {"outcome": OUTCOME_SKIPPED, "skip_reason": SKIP_DUPLICATE},
    ),
    (
        "s-a",
        1,
        "2026-07-02T10:00:00+00:00",
        "Nothing to see in this one",
        {"outcome": OUTCOME_EXTRACTED, "entities": [], "relationships": []},
    ),
    (
        "s-b",
        1,
        "2026-07-03T11:00:00+00:00",
        "Last year I started learning Go.",
        {
            "outcome": OUTCOME_EXTRACTED,
            "entities": [
                {"id": "user", "type": "User", "name": "User"},
                {"id": "go", "type": "Technology", "name": "Go"},
            ],
            "relationships": [
                {
                    "source": "user",
                    "target": "go",
                    "type": "LEARNING",
                    "properties": {"confidence": 0.8, "start_date": "last year"},
                }
            ],
        },
    ),
]


def _world() -> tuple[EventStore, ExtractionCache, list[str]]:
    store = EventStore(db_path=":memory:")
    store.initialize()
    cache = ExtractionCache(":memory:")
    cache.initialize()
    ids = []
    for session, index, stamp, utterance, row in TURNS:
        if store.get_session(session) is None:
            store.start_session(session, input_modality="text", origin="real")
        # Fixed ids: the two sides build separate worlds, and their calls are
        # compared by value.
        event_id = store.append_turn(
            ConversationTurnEvent(
                session_id=session,
                turn_index=index,
                timestamp=datetime.fromisoformat(stamp),
                user_utterance=utterance,
                system_response="ok",
                ontology_version=EPOCH["ontology_version"],
                event_id=f"evt-{session}-{index}",
            )
        )
        cache.put(
            event_id,
            EPOCH["ontology_version"],
            EPOCH["extraction_version"],
            EPOCH["model_hash"],
            created_at=stamp,
            **row,
        )
        ids.append(event_id)
    return store, cache, ids


def _stages() -> dict:
    return {
        "confidence_scorer": ConfidenceScorer(),
        "temporal_resolver": TemporalResolver(),
        "normalizer": EntityNormalizer(embedding_generator=None, executor=None),
        "validator": ExtractionValidator(),
    }


class _NoProgress:
    """ApplyProgress that never skips curation and records the marker sequence."""

    def __init__(self) -> None:
        self.marks: list[str] = []

    @property
    def curated(self) -> bool:
        return False

    def mark_curated(self) -> None:
        self.marks.append("curated")

    def mark_applied(self) -> None:
        self.marks.append("applied")


def _pipeline(cache: ExtractionCache, curation: FakeGraphCuration) -> ExtractionPipeline:
    embeddings = FakeEmbeddingGenerator()
    return ExtractionPipeline(
        preprocessor=PreProcessor(),
        extractor=OntologyConstrainedExtractor(build_test_config(), llm=FakeLLM()),
        graph_store=GraphStore(FakeNeo4jConnection(), embeddings),
        curation_pipeline=curation,  # type: ignore[arg-type]
        embedding_provider=embeddings,
        extraction_cache=cache,
        rebuild_stamps=RebuildStamps(
            ontology_version=EPOCH["ontology_version"],
            extraction_version=EPOCH["extraction_version"],
            model_hash=EPOCH["model_hash"],
        ),
        **_stages(),
    )


async def _rebuild_calls() -> list:
    store, cache, _ids = _world()
    curation = FakeGraphCuration()
    regenerator = LogRegenerator(
        event_store=store,
        extraction_cache=cache,
        staging_curation_pipeline=curation,
        journal=NullRebuildJournal(),
        staging_seeder=FakeStagingSeeder(),
        **_stages(),
    )
    await regenerator.rebuild(
        staging_uri="bolt://mist-neo4j-staging:7687",
        live_uri="bolt://mist-neo4j:7687",
        epoch=EPOCH,
        min_seed_nodes=1,
    )
    return curation.calls


async def _apply_calls() -> tuple[list, list[list[str]]]:
    store, cache, _ids = _world()
    curation = FakeGraphCuration()
    pipeline = _pipeline(cache, curation)
    marks = []
    for turn in store.get_all_turns_for_reextraction():
        cached = cache.get(turn["event_id"], EPOCH["extraction_version"], EPOCH["model_hash"])
        progress = _NoProgress()
        await pipeline.apply_cached_turn(
            TurnToApply(
                event_id=turn["event_id"],
                session_id=turn["session_id"],
                user_utterance=turn["user_utterance"],
                recorded_at=turn["timestamp"],
            ),
            cached,
            progress,
        )
        marks.append(progress.marks)
    return curation.calls, marks


@pytest.mark.asyncio
async def test_apply_cached_turn_makes_the_same_curation_calls_as_the_rebuild():
    rebuild_calls = await _rebuild_calls()
    apply_calls, _marks = await _apply_calls()

    assert apply_calls == rebuild_calls


@pytest.mark.asyncio
async def test_the_comparison_is_not_vacuous():
    """Three extracted rows -> three calls, one of them with no entities; the skip -> none."""
    rebuild_calls = await _rebuild_calls()

    assert len(rebuild_calls) == 3
    assert [len(c.entities) for c in rebuild_calls] == [2, 0, 2]
    hedged = rebuild_calls[0].relationships[0]["properties"]["confidence"]
    assert hedged < 0.9, "Stage 3's hedge penalty must have fired on the replayed utterance"


@pytest.mark.asyncio
async def test_a_skipped_row_marks_applied_without_curating():
    _calls, marks = await _apply_calls()

    assert marks == [
        ["curated", "applied"],
        ["applied"],
        ["curated", "applied"],
        ["curated", "applied"],
    ]
