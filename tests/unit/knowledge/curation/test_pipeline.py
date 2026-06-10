"""Tests for CurationPipeline orchestrator (C2 cutover)."""

import asyncio

import pytest

from backend.knowledge.curation.reconciliation import (
    ActionKind,
    ReconcileAction,
    ReconcileTurnResult,
)
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.unit.knowledge.curation.conftest import (
    make_entity_dict,
    make_relationship_dict,
    make_validation_result,
)

RECORDED_AT = "2026-06-10T12:00:00+00:00"


class _SpyEngine:
    """Records reconcile_turn calls; returns a configurable result."""

    def __init__(self, result=None, gate: asyncio.Event | None = None, raises: bool = False):
        self.calls = []
        self.returned = result if result is not None else ReconcileTurnResult()
        self._gate = gate
        self._raises = raises
        self.in_flight = 0
        self.max_in_flight = 0

    async def reconcile_turn(self, relationships, recorded_at, event_id, session_id):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.calls.append((relationships, recorded_at, event_id, session_id))
        try:
            if self._gate is not None:
                await self._gate.wait()
            if self._raises:
                raise RuntimeError("engine failed")
            return self.returned
        finally:
            self.in_flight -= 1


def _build_pipeline(*, connection=None, engine=None):
    """Build a CurationPipeline with fakes for testing."""
    from backend.knowledge.curation.confidence import ConfidenceManager
    from backend.knowledge.curation.deduplication import EntityDeduplicator
    from backend.knowledge.curation.graph_writer import CurationGraphWriter
    from backend.knowledge.curation.pipeline import CurationPipeline

    conn = connection or FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)
    embeddings = FakeEmbeddingGenerator()
    confidence = ConfidenceManager()
    eng = engine if engine is not None else _SpyEngine()

    return (
        CurationPipeline(
            deduplicator=EntityDeduplicator(executor, embeddings, confidence),
            reconciliation_engine=eng,
            graph_writer=CurationGraphWriter(executor, embeddings, confidence),
        ),
        conn,
        eng,
    )


class TestFullFlow:
    @pytest.mark.asyncio
    async def test_curate_and_store_full_pipeline(self):
        from backend.knowledge.curation.pipeline import CurationResult

        pipeline, conn, eng = _build_pipeline()

        validation = make_validation_result(
            entities=[make_entity_dict()],
            relationships=[make_relationship_dict()],
        )

        result = await pipeline.curate_and_store(
            validation, "evt-001", "sess-001", recorded_at=RECORDED_AT
        )

        assert isinstance(result, CurationResult)
        assert result.write_result.entities_created >= 1
        assert result.curation_time_ms > 0
        # Relationships route through the engine, with the threaded fact-time.
        assert eng.calls == [(validation.relationships, RECORDED_AT, "evt-001", "sess-001")]
        assert result.reconcile_result is eng.returned


class TestShortCircuit:
    @pytest.mark.asyncio
    async def test_short_circuits_on_empty_entities(self):
        from backend.knowledge.curation.pipeline import CurationResult

        pipeline, conn, eng = _build_pipeline()

        validation = make_validation_result(entities=[], relationships=[])
        result = await pipeline.curate_and_store(validation, "evt-001", "sess-001")

        assert isinstance(result, CurationResult)
        assert result.write_result.entities_created == 0
        assert result.reconcile_result.appended == 0
        assert eng.calls == []
        conn.assert_no_writes()


class TestEngineStage:
    @pytest.mark.asyncio
    async def test_engine_failure_logs_and_continues(self):
        pipeline, _conn, _eng = _build_pipeline(engine=_SpyEngine(raises=True))

        validation = make_validation_result(
            entities=[make_entity_dict()],
            relationships=[make_relationship_dict()],
        )
        result = await pipeline.curate_and_store(validation, "evt-001", "sess-001")

        assert result.reconcile_result.appended == 0
        assert any("Reconciliation failed" in e for e in result.stage_errors)

    @pytest.mark.asyncio
    async def test_writes_serialize_under_the_lock(self):
        # Two concurrent turns: the gated engine proves the second curate call
        # cannot enter the locked section until the first completes (Inv-A9).
        gate = asyncio.Event()
        eng = _SpyEngine(gate=gate)
        pipeline, _conn, _eng = _build_pipeline(engine=eng)

        validation = make_validation_result(
            entities=[make_entity_dict()],
            relationships=[make_relationship_dict()],
        )
        t1 = asyncio.create_task(pipeline.curate_and_store(validation, "evt-1", "s"))
        t2 = asyncio.create_task(pipeline.curate_and_store(validation, "evt-2", "s"))
        await asyncio.sleep(0.05)  # let both tasks reach the lock
        gate.set()
        await asyncio.gather(t1, t2)

        assert eng.max_in_flight == 1
        assert len(eng.calls) == 2

    @pytest.mark.asyncio
    async def test_close_bearing_actions_emit_learning_events(self):
        close = ReconcileAction(
            kind=ActionKind.CLOSE_TRANSACTION,
            predicate="DISLIKES",
            source="user",
            target="rust",
            reason="contradiction",
            edge_ref="ref-1",
        )
        eng = _SpyEngine(result=ReconcileTurnResult(actions=[close], closed=1))
        pipeline, conn, _eng = _build_pipeline(engine=eng)

        validation = make_validation_result(
            entities=[make_entity_dict()],
            relationships=[make_relationship_dict()],
        )
        result = await pipeline.curate_and_store(validation, "evt-001", "sess-001")

        # 2 = the fresh entity's new_fact event + the belief-change event.
        assert result.write_result.learning_events_created == 2
        belief_change_writes = [
            (q, p)
            for q, p in conn.writes
            if "LearningEvent" in q
            and p
            and p.get("learning_id") == "learning-evt-001-DISLIKES-rust"
        ]
        assert belief_change_writes, "the belief-change LearningEvent must be written"


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_dedup_failure_logs_and_continues(self):
        """Dedup failure produces partial result -- pipeline does not crash."""
        from backend.knowledge.curation.confidence import ConfidenceManager
        from backend.knowledge.curation.deduplication import EntityDeduplicator
        from backend.knowledge.curation.graph_writer import CurationGraphWriter
        from backend.knowledge.curation.pipeline import CurationPipeline

        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        embeddings = FakeEmbeddingGenerator()
        confidence = ConfidenceManager()

        class FailingDeduplicator(EntityDeduplicator):
            async def deduplicate(self, entities):
                raise RuntimeError("dedup failed")

        pipeline = CurationPipeline(
            deduplicator=FailingDeduplicator(executor, embeddings, confidence),
            reconciliation_engine=_SpyEngine(),
            graph_writer=CurationGraphWriter(executor, embeddings, confidence),
        )

        validation = make_validation_result(entities=[make_entity_dict()])

        # Pipeline catches the error and returns a partial result
        result = await pipeline.curate_and_store(validation, "evt-001", "sess-001")
        assert result.dedup_result.entities_merged == 0
        assert len(result.stage_errors) == 1
        assert "dedup" in result.stage_errors[0].lower()
