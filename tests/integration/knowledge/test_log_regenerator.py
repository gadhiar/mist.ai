"""Integration coverage for LogRegenerator.rebuild (R1.2), staging-isolated."""

from __future__ import annotations

import socket
from datetime import datetime
from pathlib import Path

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.knowledge.config import Neo4jConfig
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.extraction_cache import OUTCOME_EXTRACTED, ExtractionCache
from backend.knowledge.regeneration.log_regenerator import ColdCacheError, LogRegenerator
from backend.knowledge.regeneration.rebuild_journal import EventStoreRebuildJournal
from backend.knowledge.storage.neo4j_connection import Neo4jConnection


def _stage_components() -> dict:
    """Real Stage 3-6 components (Task 6, extraction-cache-phase-1).

    None of `ConfidenceScorer()`/`TemporalResolver()`/`ExtractionValidator()` takes an
    argument that could reach a graph or a model, and `EntityNormalizer.normalize()`
    issues no graph queries at all -- so the real components are the simplest correct
    wiring here, same as `_build_log_regenerator` in scripts/mist_admin.py.
    """
    return {
        "confidence_scorer": ConfidenceScorer(),
        "temporal_resolver": TemporalResolver(),
        "normalizer": EntityNormalizer(embedding_generator=None, executor=None),
        "validator": ExtractionValidator(),
    }


# Staging endpoint (NEVER live). In-network service name, then host-published port.
_CANDIDATES = [("mist-neo4j-staging", 7687), ("localhost", 7689), ("127.0.0.1", 7689)]


def _staging_endpoint() -> tuple[str, int] | None:
    for host, port in _CANDIDATES:
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            return host, port
        except OSError:
            continue
    return None


_ENDPOINT = _staging_endpoint()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _ENDPOINT is None,
        reason=(
            "staging Neo4j not running (docker compose -f docker-compose.yml "
            "-f docker-compose.staging-neo4j.yml --profile staging up -d mist-neo4j-staging)"
        ),
    ),
]
_LIVE_URI = "bolt://mist-neo4j:7687"  # used only for the target!=live guard check


@pytest.fixture
def staging_conn():
    host, port = _ENDPOINT  # type: ignore[misc]
    conn = Neo4jConnection(
        Neo4jConfig(uri=f"bolt://{host}:{port}", username="neo4j", password="password")
    )
    conn.connect()
    conn.execute_write("MATCH (n) DETACH DELETE n", {})  # staging starts empty
    yield conn
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    conn.disconnect()


def _staging_uri() -> str:
    host, port = _ENDPOINT  # type: ignore[misc]
    return f"bolt://{host}:{port}"


# ---------------------------------------------------------------------------
# Test epoch and timestamp constants -- shared across both helpers
# ---------------------------------------------------------------------------

_TURN_TS = "2026-06-29T10:00:00+00:00"
_TEST_EPOCH = {
    "epoch_id": 1,
    "ontology_version": "1.4.0",
    "extraction_version": "r5-inttest",
    "model_hash": "test-model-hash-inttest",
    "activated_at": _TURN_TS,
    "prev_epoch_id": None,
}


def _make_event_store_with_turn(tmp_path: Path) -> tuple[EventStore, str, str]:
    """Create an EventStore with one seeded turn. Returns (store, session_id, event_id)."""
    event_store = EventStore(str(tmp_path / "events.db"))
    event_store.initialize()
    session_id = "log-regen-turn-session"
    event_store.start_session(session_id, input_modality="text")
    event = ConversationTurnEvent(
        session_id=session_id,
        turn_index=0,
        timestamp=datetime.fromisoformat(_TURN_TS),
        user_utterance="I love Python.",
        system_response="Python is great.",
    )
    event_id = event_store.append_turn(event)
    return event_store, session_id, event_id


def _build_staging_pipeline(staging_conn: Neo4jConnection):
    """Build a CurationPipeline wired to staging via the production factory.

    Uses build_curation_pipeline(config, executor) exactly as documented in
    factories.py build_curation_pipeline (~line 191). The staging GraphExecutor
    is constructed from the staging Neo4jConnection passed in. Schema is
    initialised (idempotent) before returning.
    """
    from backend.factories import build_curation_pipeline
    from backend.knowledge.storage.graph_executor import GraphExecutor
    from backend.knowledge.storage.graph_store import GraphStore
    from tests.mocks.config import build_test_config
    from tests.mocks.embeddings import FakeEmbeddingGenerator

    # Initialize staging schema (idempotent: safe to call on a pre-existing instance).
    staging_store = GraphStore(
        connection=staging_conn, embedding_generator=FakeEmbeddingGenerator()
    )
    staging_store.initialize_schema()

    executor = GraphExecutor(staging_conn)
    # build_test_config neo4j uri is irrelevant here: executor owns the connection.
    # ontology_version/extraction_version/model_hash come from KnowledgeConfig defaults
    # and are written as rebuild-stamp metadata on edges; they do not gate entity writes.
    config = build_test_config()
    return build_curation_pipeline(config, executor, embedding_provider=FakeEmbeddingGenerator())


def _build_regenerator_with_one_turn(
    tmp_path: Path, staging_conn: Neo4jConnection
) -> tuple[LogRegenerator, EventStore, ExtractionCache, dict]:
    """Build a LogRegenerator seeded with one turn that HAS a cache hit."""
    event_store, _session_id, event_id = _make_event_store_with_turn(tmp_path)

    cache = ExtractionCache(str(tmp_path / "cache.db"))
    cache.initialize()
    cache.put(
        event_id,
        _TEST_EPOCH["ontology_version"],
        _TEST_EPOCH["extraction_version"],
        _TEST_EPOCH["model_hash"],
        outcome=OUTCOME_EXTRACTED,
        entities=[{"id": "python", "type": "Technology", "display_name": "Python"}],
        relationships=[],
        created_at=_TURN_TS,
    )

    pipeline = _build_staging_pipeline(staging_conn)
    regen = LogRegenerator(
        event_store=event_store,
        extraction_cache=cache,
        staging_curation_pipeline=pipeline,
        journal=EventStoreRebuildJournal(event_store),
        **_stage_components(),
    )
    return regen, event_store, cache, _TEST_EPOCH


def _build_regenerator_with_uncached_turn(
    tmp_path: Path, staging_conn: Neo4jConnection
) -> tuple[LogRegenerator, EventStore, ExtractionCache, dict]:
    """Build a LogRegenerator seeded with one turn that has NO cache entry."""
    event_store, _session_id, _event_id = _make_event_store_with_turn(tmp_path)

    cache = ExtractionCache(str(tmp_path / "cache.db"))
    cache.initialize()
    # Intentionally no cache.put -- this turn is cold.

    pipeline = _build_staging_pipeline(staging_conn)
    regen = LogRegenerator(
        event_store=event_store,
        extraction_cache=cache,
        staging_curation_pipeline=pipeline,
        journal=EventStoreRebuildJournal(event_store),
        **_stage_components(),
    )
    return regen, event_store, cache, _TEST_EPOCH


class TestLogRegeneratorReplay:
    @pytest.mark.asyncio
    async def test_cache_driven_rebuild_builds_entity_graph(self, staging_conn, tmp_path):
        # Arrange: an event store with one turn + a cache hit for it.
        # (Build via the real EventStore + ExtractionCache against tmp sqlite files;
        #  the implementer wires these per the constructors -- see Step 3 notes.)
        regen, event_store, cache, epoch = _build_regenerator_with_one_turn(tmp_path, staging_conn)

        # Act
        report = await regen.rebuild(
            staging_uri=_staging_uri(),
            live_uri=_LIVE_URI,
            epoch=epoch,
        )

        # Assert: the turn's cached entity was written to staging.
        assert report.turns_processed == 1
        rows = staging_conn.execute_query(
            "MATCH (n:__Entity__ {id: $id}) RETURN count(n) AS n", {"id": "python"}
        )
        assert rows[0]["n"] == 1

    @pytest.mark.asyncio
    async def test_cold_cache_refuses(self, staging_conn, tmp_path):
        # Arrange: a turn with NO cache entry -> coverage check must refuse.
        regen, _, _, epoch = _build_regenerator_with_uncached_turn(tmp_path, staging_conn)
        with pytest.raises(ColdCacheError, match="uncached"):
            await regen.rebuild(staging_uri=_staging_uri(), live_uri=_LIVE_URI, epoch=epoch)

    @pytest.mark.asyncio
    async def test_rebuild_finalizes_job_completed(self, staging_conn, tmp_path):
        # Arrange: one cached turn.
        regen, event_store, _cache, epoch = _build_regenerator_with_one_turn(tmp_path, staging_conn)

        # Act
        report = await regen.rebuild(
            staging_uri=_staging_uri(),
            live_uri=_LIVE_URI,
            epoch=epoch,
        )

        # Assert report fields
        assert report.turns_processed == 1
        assert report.turns_failed == 0

        # Assert the job ledger row is in a terminal state
        job = event_store.get_reextraction_job(report.job_id)
        assert job is not None
        assert job["status"] == "completed"
        assert job["failed"] == 0
        assert job["errors"] is None

    @pytest.mark.asyncio
    async def test_rebuild_twice_same_epoch_no_collision(self, staging_conn, tmp_path):
        # Arrange: one cached turn. The fixture wipes + re-seeds staging between
        # each test, so both rebuilds share one event-store + cache db but write
        # to a clean staging graph. The PK collision risk is in the sqlite event
        # store job ledger, not in Neo4j.
        regen, event_store, _cache, epoch = _build_regenerator_with_one_turn(tmp_path, staging_conn)

        # Act: two independent rebuilds of the same epoch
        report1 = await regen.rebuild(
            staging_uri=_staging_uri(),
            live_uri=_LIVE_URI,
            epoch=epoch,
        )
        report2 = await regen.rebuild(
            staging_uri=_staging_uri(),
            live_uri=_LIVE_URI,
            epoch=epoch,
        )

        # Both runs must complete without IntegrityError and produce distinct ids
        assert report1.job_id != report2.job_id
        assert report1.job_id.startswith(f"rebuild-{epoch['epoch_id']}-")
        assert report2.job_id.startswith(f"rebuild-{epoch['epoch_id']}-")

        # Both job rows must exist and be finalized
        job1 = event_store.get_reextraction_job(report1.job_id)
        job2 = event_store.get_reextraction_job(report2.job_id)
        assert job1 is not None and job1["status"] == "completed"
        assert job2 is not None and job2["status"] == "completed"

    @pytest.mark.asyncio
    async def test_rebuild_twice_byte_identical(self, staging_conn, tmp_path):
        """Rebuild into a wiped staging twice; both canonical forms must be byte-identical.

        This is the headline DoD gate for the proof-first deliverable. Two
        independent rebuilds of the same epoch produce the same entity graph
        when the extraction cache is the sole fact source and the curation
        pipeline is deterministic.
        """
        from backend.knowledge.canonical_serialize import canonical_graph_form

        # Arrange: event store + cache with 2 turns.
        event_store = EventStore(str(tmp_path / "events.db"))
        event_store.initialize()
        session_id = "log-regen-rebuild-twice-session"
        event_store.start_session(session_id, input_modality="text")

        ts_a = "2026-06-29T10:00:00+00:00"
        ts_b = "2026-06-29T10:01:00+00:00"
        epoch = {
            "epoch_id": 1,
            "ontology_version": "1.4.0",
            "extraction_version": "r5-inttest",
            "model_hash": "test-model-hash-inttest",
            "activated_at": ts_a,
            "prev_epoch_id": None,
        }

        event_id_a = event_store.append_turn(
            ConversationTurnEvent(
                session_id=session_id,
                turn_index=0,
                timestamp=datetime.fromisoformat(ts_a),
                user_utterance="I love Python.",
                system_response="Python is great.",
            )
        )
        event_id_b = event_store.append_turn(
            ConversationTurnEvent(
                session_id=session_id,
                turn_index=1,
                timestamp=datetime.fromisoformat(ts_b),
                user_utterance="I also use FastAPI.",
                system_response="FastAPI is a good choice.",
            )
        )

        cache = ExtractionCache(str(tmp_path / "cache.db"))
        cache.initialize()
        cache.put(
            event_id_a,
            epoch["ontology_version"],
            epoch["extraction_version"],
            epoch["model_hash"],
            outcome=OUTCOME_EXTRACTED,
            entities=[{"id": "python", "type": "Technology", "display_name": "Python"}],
            relationships=[],
            created_at=ts_a,
        )
        cache.put(
            event_id_b,
            epoch["ontology_version"],
            epoch["extraction_version"],
            epoch["model_hash"],
            outcome=OUTCOME_EXTRACTED,
            entities=[{"id": "fastapi", "type": "Technology", "display_name": "FastAPI"}],
            relationships=[],
            created_at=ts_b,
        )

        pipeline = _build_staging_pipeline(staging_conn)
        regen = LogRegenerator(
            event_store=event_store,
            extraction_cache=cache,
            staging_curation_pipeline=pipeline,
            journal=EventStoreRebuildJournal(event_store),
            **_stage_components(),
        )

        # Act: first rebuild into staging.
        await regen.rebuild(
            staging_uri=_staging_uri(),
            live_uri=_LIVE_URI,
            epoch=epoch,
        )
        form_a = canonical_graph_form(staging_conn, include_provenance=False)

        # Wipe staging between runs (schema constraints/indexes survive DETACH DELETE).
        staging_conn.execute_write("MATCH (n) DETACH DELETE n", {})

        # Act: second rebuild from scratch.
        await regen.rebuild(
            staging_uri=_staging_uri(),
            live_uri=_LIVE_URI,
            epoch=epoch,
        )
        form_b = canonical_graph_form(staging_conn, include_provenance=False)

        # Assert: byte-identical proves deterministic log-driven rebuild.
        assert form_a == form_b
