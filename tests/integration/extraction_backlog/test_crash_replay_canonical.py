"""Crash mid-apply, restart, and the REAL curation pipeline converges (T2a, AC3).

Runs the extraction-backlog dispatcher twice against the disposable eval Neo4j
with the production `CurationPipeline` (`build_curation_pipeline`):

- run A: two turns, uninterrupted;
- run B: the same two turns, with a fault injected after turn 1's graph write
  and before its `curated` marker, then a fresh dispatcher on the same stores.

Both runs must yield the same `canonical_graph_form`, and run B must not call
the extraction service again for the crashed turn (the cached result survives).

A second test checks what the canonical form cannot see: node `confidence`,
excluded from it (`canonical_serialize.NODE_ONLY_EXCLUDED_FIELDS`). Before the
MIS-171 replay guard, the re-applied turn's newly created entity took the
upsert's `ON MATCH` reinforce (`curation/graph_writer.py`, `_upsert_entity`) and
ended higher than after one apply. The guard skips the reinforce when this
event's EXTRACTED_FROM edge already exists, so the crashed run's confidences now
equal the uninterrupted run's, and turn 1 still reinforces the entity both
turns mention.

The EXTRACTED_FROM edge and a new entity's `new_fact` LearningEvent are written
by the entity's own upsert statement, so every entity write in both runs goes
through that one statement on real Neo4j, and the canonical form (built with
`include_provenance=True`) compares the edges and LearningEvents it writes. The
injected crash here lands after the whole curation write; kills between the
writer's statements are covered in the unit tier
(`tests/unit/knowledge/curation/test_graph_writer_replay_guard.py`,
`TestKillAtAnyStatementThenReplay`).

Only this test's own nodes are written and cleaned up (every id and session id
carries `_PREFIX`); anything else in the eval instance appears identically in
both canonical forms. Start the target first:

  docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml \
    --profile eval up -d mist-neo4j-eval
"""

from __future__ import annotations

import socket
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.extraction_backlog.dispatcher import ExtractionDispatcher
from backend.extraction_backlog.inference import RemoteExtractionInference
from backend.extraction_backlog.settings import DispatcherSettings
from backend.extraction_backlog.store import BacklogStore
from backend.knowledge.config import ExtractionConfig, Neo4jConfig
from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
from backend.knowledge.extraction.pipeline import ExtractionPipeline
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.extraction_cache import ExtractionCache
from backend.knowledge.storage.graph_executor import GraphExecutor
from backend.knowledge.storage.graph_store import GraphStore
from backend.knowledge.storage.neo4j_connection import Neo4jConnection
from backend.knowledge.version_stamps import compose_model_hash
from tests.integration.extraction_backlog.harness import PREFIX as _PREFIX
from tests.integration.extraction_backlog.harness import SESSION as _SESSION
from tests.integration.extraction_backlog.harness import TURNS as _TURNS
from tests.integration.extraction_backlog.harness import make_embeddings
from tests.integration.extraction_backlog.harness import payload as _payload
from tests.mocks.config import build_test_config
from tests.mocks.ollama import FakeLLM
from tests.unit.extraction_backlog.fakes import (
    SERVICE_URL,
    FakeServiceState,
    SimulatedCrashError,
    SwitchableTransport,
    build_fake_service_app,
)

# In-container service name first; host-published fallback ports. The live
# mist-neo4j:7687 and host localhost:7687 are NEVER in this list.
_CANDIDATES = [("mist-neo4j-eval", 7687), ("localhost", 7688), ("127.0.0.1", 7688)]


def _eval_endpoint() -> tuple[str, int] | None:
    for host, port in _CANDIDATES:
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            return host, port
        except OSError:
            continue
    return None


_ENDPOINT = _eval_endpoint()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _ENDPOINT is None,
        reason=(
            "disposable eval Neo4j not running (docker compose -f docker-compose.yml "
            "-f docker-compose.eval-neo4j.yml --profile eval up -d mist-neo4j-eval)"
        ),
    ),
]

_EMBEDDING_MODEL = "test-emb"
_ONTOLOGY_VERSION = "1.4.0"
# The turns, payload and embeddings live in `harness.py`: the embeddings must
# keep this test's entities apart under curation's cosine dedup on real Neo4j,
# which the unit tier checks there (see that module's docstring for the
# 2026-09-27 failure this prevents).


@pytest.fixture
def eval_conn():
    host, port = _ENDPOINT  # type: ignore[misc]
    conn = Neo4jConnection(
        Neo4jConfig(uri=f"bolt://{host}:{port}", username="neo4j", password="password")
    )
    conn.connect()
    GraphStore(connection=conn, embedding_generator=make_embeddings()).initialize_schema()
    _cleanup(conn)
    yield conn
    _cleanup(conn)
    conn.disconnect()


def _cleanup(conn: Neo4jConnection) -> None:
    conn.execute_write(
        "MATCH (n) WHERE n.id STARTS WITH $p OR n.conversation_id STARTS WITH $p "
        "OR n.id STARTS WITH $lp DETACH DELETE n",
        {"p": _PREFIX, "lp": f"learning-{_PREFIX}"},
    )


class _CrashAfterWrite:
    """Wraps the real CurationPipeline; raises once, AFTER a real graph write."""

    def __init__(self, inner, crash_for: str | None) -> None:
        self._inner = inner
        self._crash_for = crash_for

    async def curate_and_store(
        self, validation_result, event_id, session_id, source_metadata=None, recorded_at=None
    ):
        result = await self._inner.curate_and_store(
            validation_result,
            event_id=event_id,
            session_id=session_id,
            source_metadata=source_metadata,
            recorded_at=recorded_at,
        )
        if event_id == self._crash_for:
            self._crash_for = None
            raise SimulatedCrashError(f"crash after the real graph write for {event_id}")
        return result


class _Run:
    """One run's stores and fake service; dispatchers built over them share state."""

    def __init__(self, tmp_path: Path, conn: Neo4jConnection) -> None:
        from backend.factories import build_curation_pipeline

        self.service = FakeServiceState(payload_fn=_payload)
        self.events = EventStore(db_path=str(tmp_path / "events.db"))
        self.events.initialize()
        epoch = self.events.ensure_initial_epoch(
            now_iso="2026-09-01T00:00:00+00:00",
            ontology_version=_ONTOLOGY_VERSION,
            extraction_version=self.service.extraction_version,
            model_hash=compose_model_hash(_Identity(self.service.model_hash)),
        )
        self.cache = ExtractionCache(str(tmp_path / "cache.db"))
        self.cache.initialize()
        self.store = BacklogStore(self.events, self.cache)
        self.store.ensure_activation(self.store.active_epoch(), now_iso="2026-09-01T00:00:00+00:00")
        self.embeddings = make_embeddings()
        self.curation = build_curation_pipeline(
            build_test_config(embedding_model=_EMBEDDING_MODEL),
            GraphExecutor(conn),
            embedding_provider=self.embeddings,
        )
        self.stamps = RebuildStamps(
            ontology_version=epoch["ontology_version"],
            extraction_version=epoch["extraction_version"],
            model_hash=epoch["model_hash"],
        )
        self.conn = conn
        self.clients: list[httpx.AsyncClient] = []
        self.dispatchers: list[ExtractionDispatcher] = []

    def log_turns(self) -> None:
        self.events.start_session(_SESSION, input_modality="text", origin="test")
        for event_id, index, stamp, utterance in _TURNS:
            self.events.append_turn(
                ConversationTurnEvent(
                    session_id=_SESSION,
                    turn_index=index,
                    timestamp=datetime.fromisoformat(stamp),
                    user_utterance=utterance,
                    system_response="Noted.",
                    ontology_version=_ONTOLOGY_VERSION,
                    event_id=event_id,
                )
            )

    def dispatcher(self, *, crash_for: str | None) -> ExtractionDispatcher:
        pipeline = ExtractionPipeline(
            preprocessor=PreProcessor(),
            extractor=OntologyConstrainedExtractor(build_test_config(), llm=FakeLLM()),
            confidence_scorer=ConfidenceScorer(),
            temporal_resolver=TemporalResolver(),
            normalizer=EntityNormalizer(embedding_generator=None, executor=None),
            validator=ExtractionValidator(min_confidence=0.0),
            graph_store=GraphStore(connection=self.conn, embedding_generator=self.embeddings),
            curation_pipeline=_CrashAfterWrite(self.curation, crash_for),  # type: ignore[arg-type]
            embedding_provider=self.embeddings,
            extraction_config=ExtractionConfig(significance_threshold=0.0),
            extraction_cache=self.cache,
            rebuild_stamps=self.stamps,
        )
        client = httpx.AsyncClient(
            transport=SwitchableTransport(self.service, build_fake_service_app(self.service))
        )
        self.clients.append(client)
        dispatcher = ExtractionDispatcher(
            store=self.store,
            pipeline=pipeline,
            inference=RemoteExtractionInference(client, SERVICE_URL),
            settings=DispatcherSettings(
                service_url=SERVICE_URL,
                backoff_base_s=0.001,
                backoff_cap_s=0.005,
                idle_poll_s=0.05,
                stall_recheck_s=0.05,
            ),
            embedding_model_name=_EMBEDDING_MODEL,
            writer_stamps=self.stamps,
        )
        self.dispatchers.append(dispatcher)
        return dispatcher

    async def close(self) -> None:
        for dispatcher in self.dispatchers:
            await dispatcher.stop(timeout=5.0)
        for client in self.clients:
            await client.aclose()


class _Identity:
    """`compose_model_hash` input for a bare service hash and the test embedding."""

    def __init__(self, model_hash: str) -> None:
        self.model_hash = model_hash
        self.embedding = type("Emb", (), {"model_name": _EMBEDDING_MODEL})()


async def _wait_until_stopped(dispatcher: ExtractionDispatcher, timeout: float = 30.0) -> None:
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while dispatcher.running:
        if loop.time() > deadline:
            raise AssertionError("dispatcher did not stop at the injected crash")
        await asyncio.sleep(0.01)


async def _uninterrupted(run: _Run) -> None:
    run.log_turns()
    dispatcher = run.dispatcher(crash_for=None)
    await dispatcher.start()
    assert await dispatcher.drain(timeout=30.0)


async def _crashed_then_restarted(run: _Run) -> None:
    run.log_turns()
    first = run.dispatcher(crash_for=_TURNS[0][0])
    await first.start()
    await _wait_until_stopped(first)
    epoch = run.store.active_epoch()
    assert run.events.get_extraction_applied(epoch.epoch_id) == {}, "crash must precede markers"
    restarted = run.dispatcher(crash_for=None)
    await restarted.start()
    assert await restarted.drain(timeout=30.0)


def _node_confidence(conn: Neo4jConnection, node_id: str) -> float:
    rows = conn.execute_query(
        "MATCH (n:__Entity__ {id: $id}) RETURN n.confidence AS c", {"id": node_id}
    )
    assert rows, f"no node {node_id} in the graph (merged into another entity?)"
    return float(rows[0]["c"])


@pytest.mark.asyncio
async def test_crash_after_graph_write_then_restart_equals_one_uninterrupted_apply(
    eval_conn, tmp_path
):
    from backend.knowledge.canonical_serialize import canonical_graph_form

    # Arrange + Act: run A. Every run is closed in `finally`, so a failed step
    # cannot leave its dispatcher task pending past the test.
    run_a = _Run(tmp_path / "a", eval_conn)
    try:
        await _uninterrupted(run_a)
        form_a = canonical_graph_form(eval_conn, include_provenance=True)
    finally:
        await run_a.close()
    _cleanup(eval_conn)

    # Act: run B
    run_b = _Run(tmp_path / "b", eval_conn)
    try:
        await _crashed_then_restarted(run_b)
        form_b = canonical_graph_form(eval_conn, include_provenance=True)
    finally:
        await run_b.close()

    # Assert: run A wrote BOTH turns' technologies as separate nodes (a merge
    # of zig into rust would leave only rust; see harness.py).
    assert f'"{_PREFIX}rust"' in form_a, "run A wrote no rust node"
    assert f'"{_PREFIX}zig"' in form_a, "run A wrote no zig node (merged into rust?)"
    assert run_b.service.received_utterances == [t[3] for t in _TURNS]
    assert form_b == form_a


@pytest.mark.asyncio
async def test_reapplied_turn_leaves_node_confidence_as_one_apply(eval_conn, tmp_path):
    """Flipped from T2a's known-divergence pin once the replay guard landed.

    Re-running curation for turn 0, whose entities the crashed run already
    created, takes the `ON MATCH` branch of `_upsert_entity`. Turn 0's
    EXTRACTED_FROM edges already carry `source_utterance_id = evt-0`, so the
    guard leaves `confidence` at the ON CREATE value, as one apply does.

    `dev` is in both turns: turn 1 is a genuinely new event for it, so run A
    reinforces it there, and a guard that wrongly blocked that reinforce in
    run B (whose `dev` edge was just re-stamped by the replay) would show up as
    a lower `dev` in run B. `rust` is in turn 0 only. Every node is compared,
    which also checks that Neo4j evaluates the guard's `OPTIONAL MATCH` /
    `WITH count(...)` form; the unit tier's fake cannot.
    """
    ids = [f"{_PREFIX}dev", f"{_PREFIX}rust", f"{_PREFIX}zig"]
    run_a = _Run(tmp_path / "a", eval_conn)
    try:
        await _uninterrupted(run_a)
        single = {i: _node_confidence(eval_conn, i) for i in ids}
    finally:
        await run_a.close()
    _cleanup(eval_conn)

    run_b = _Run(tmp_path / "b", eval_conn)
    try:
        await _crashed_then_restarted(run_b)
        reapplied = {i: _node_confidence(eval_conn, i) for i in ids}
    finally:
        await run_b.close()

    assert reapplied == single
