"""Fixtures for the extraction-backlog suite.

`backlog_world` builds one hermetic world: a real in-memory `EventStore` and
`ExtractionCache` (real SQLite at that boundary; TESTING.md, Event Store
Testing), an epoch
whose stamps match the fake service, a real `ExtractionPipeline` whose LLM
stages are wired to a counting `FakeLLM` (the "main chat model" -- it must see
zero calls), a stateful `FakeGraphCuration`, and a dispatcher factory that can
be called more than once over the SAME stores (a restart).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.extraction_backlog.dispatcher import ExtractionDispatcher
from backend.extraction_backlog.inference import RemoteExtractionInference
from backend.extraction_backlog.settings import DispatcherSettings
from backend.extraction_backlog.store import BacklogStore
from backend.knowledge.config import ExtractionConfig
from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.extraction.confidence import ConfidenceScorer
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
from backend.knowledge.extraction.pipeline import ExtractionPipeline
from backend.knowledge.extraction.preprocessor import PreProcessor
from backend.knowledge.extraction.temporal import TemporalResolver
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.extraction_cache import ExtractionCache
from backend.knowledge.storage.graph_store import GraphStore
from backend.knowledge.version_stamps import compose_model_hash
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.unit.extraction_backlog.fakes import (
    SERVICE_URL,
    FakeGraphCuration,
    FakeInternalDeriver,
    FakeServiceState,
    SwitchableTransport,
    build_fake_service_app,
)

EMBEDDING_MODEL = "test-emb"
ONTOLOGY_VERSION = "1.4.0"


def composed(bare_model_hash: str) -> str:
    """The epoch-side model hash for a bare service hash, via the one authority."""
    return compose_model_hash(
        SimpleNamespace(
            model_hash=bare_model_hash, embedding=SimpleNamespace(model_name=EMBEDDING_MODEL)
        )
    )


def fast_settings(**overrides) -> DispatcherSettings:
    """Settings with millisecond backoffs so retry paths run instantly."""
    values = {
        "mode": "service",
        "service_url": SERVICE_URL,
        "max_attempts": 5,
        "backoff_base_s": 0.001,
        "backoff_cap_s": 0.005,
        "idle_poll_s": 0.05,
        "stall_recheck_s": 0.02,
        "history_messages": 10,
    }
    values.update(overrides)
    return DispatcherSettings(**values)


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    """Poll `predicate` on the event loop until it holds, or fail the test."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.005)


@dataclass
class BacklogWorld:
    event_store: EventStore
    cache: ExtractionCache
    service: FakeServiceState
    main_llm: FakeLLM
    curation: FakeGraphCuration
    deriver: FakeInternalDeriver | None
    epoch_id: int
    embeddings: FakeEmbeddingGenerator
    clients: list[httpx.AsyncClient] = field(default_factory=list)
    dispatchers: list[ExtractionDispatcher] = field(default_factory=list)

    @property
    def store(self) -> BacklogStore:
        return BacklogStore(self.event_store, self.cache)

    def log_turn(
        self,
        *,
        session_id: str,
        turn_index: int,
        timestamp: str,
        utterance: str,
        response: str = "Noted.",
        event_id: str | None = None,
    ) -> str:
        if self.event_store.get_session(session_id) is None:
            self.event_store.start_session(session_id, input_modality="text")
        event = ConversationTurnEvent(
            session_id=session_id,
            turn_index=turn_index,
            timestamp=datetime.fromisoformat(timestamp),
            user_utterance=utterance,
            system_response=response,
            ontology_version=ONTOLOGY_VERSION,
        )
        if event_id is not None:
            event.event_id = event_id
        return self.event_store.append_turn(event)

    def build_pipeline(self) -> ExtractionPipeline:
        config = build_test_config(embedding_model=EMBEDDING_MODEL)
        return ExtractionPipeline(
            preprocessor=PreProcessor(),
            extractor=OntologyConstrainedExtractor(config, llm=self.main_llm),
            confidence_scorer=ConfidenceScorer(),
            temporal_resolver=TemporalResolver(),
            normalizer=EntityNormalizer(embedding_generator=None, executor=None),
            validator=ExtractionValidator(min_confidence=0.0),
            graph_store=GraphStore(FakeNeo4jConnection(), self.embeddings),
            curation_pipeline=self.curation,  # type: ignore[arg-type]
            internal_deriver=self.deriver,  # type: ignore[arg-type]
            embedding_provider=self.embeddings,
            extraction_config=ExtractionConfig(
                significance_threshold=0.0,
                rate_limit_max_per_minute=0,  # Gate 1 would refuse everything
                dedup_similarity_threshold=0.99,
                dedup_cache_size=200,
                dedup_cache_ttl_seconds=300,
            ),
            extraction_cache=self.cache,
            rebuild_stamps=self.active_epoch_stamps(),
        )

    def active_epoch_stamps(self) -> RebuildStamps:
        """The active epoch's stamp triple, read now, as the writer's `RebuildStamps`."""
        epoch = self.event_store.get_current_epoch()
        return RebuildStamps(
            ontology_version=epoch["ontology_version"],
            extraction_version=epoch["extraction_version"],
            model_hash=epoch["model_hash"],
        )

    def build_dispatcher(
        self, *, writer_stamps: RebuildStamps | None = None, **setting_overrides
    ) -> ExtractionDispatcher:
        """A fresh dispatcher (and pipeline) over the SAME stores: a restart.

        `writer_stamps` None means the ACTIVE epoch's stamps, read at build
        time -- the operator recreating the backend with a matching
        `MIST_MODEL_HASH`. Pass other stamps to model a backend whose config
        disagrees with the epoch ledger.
        """
        client = httpx.AsyncClient(
            transport=SwitchableTransport(self.service, build_fake_service_app(self.service))
        )
        self.clients.append(client)
        dispatcher = ExtractionDispatcher(
            store=self.store,
            pipeline=self.build_pipeline(),
            inference=RemoteExtractionInference(client, SERVICE_URL),
            settings=fast_settings(**setting_overrides),
            embedding_model_name=EMBEDDING_MODEL,
            writer_stamps=(
                writer_stamps if writer_stamps is not None else self.active_epoch_stamps()
            ),
        )
        self.dispatchers.append(dispatcher)
        return dispatcher

    def attempts(self) -> list[dict]:
        return self.event_store.list_extraction_attempts(limit=1000)


def _build_world(*, with_deriver: bool, activated: bool = True) -> BacklogWorld:
    """Build one hermetic backlog world.

    `activated` records the first-activation floor over the still-empty log,
    modelling a backlog that has been live since before any turn the test
    logs -- otherwise every pre-start turn would be legacy.
    """
    event_store = EventStore(db_path=":memory:")
    event_store.initialize()
    service = FakeServiceState()
    epoch = event_store.ensure_initial_epoch(
        now_iso="2026-09-01T00:00:00+00:00",
        ontology_version=ONTOLOGY_VERSION,
        extraction_version=service.extraction_version,
        model_hash=composed(service.model_hash),
    )
    cache = ExtractionCache(":memory:")
    cache.initialize()
    if activated:
        store = BacklogStore(event_store, cache)
        store.ensure_activation(store.active_epoch(), now_iso="2026-09-01T00:00:00+00:00")
    return BacklogWorld(
        event_store=event_store,
        cache=cache,
        service=service,
        main_llm=FakeLLM(),
        curation=FakeGraphCuration(),
        deriver=FakeInternalDeriver() if with_deriver else None,
        epoch_id=int(epoch["epoch_id"]),
        embeddings=FakeEmbeddingGenerator(),
    )


@pytest_asyncio.fixture
async def backlog_world():
    world = _build_world(with_deriver=False)
    yield world
    for dispatcher in world.dispatchers:
        await dispatcher.stop(timeout=2.0)
    for client in world.clients:
        await client.aclose()


@pytest_asyncio.fixture
async def unactivated_backlog_world():
    """A world whose backlog has never run: turns logged now predate activation."""
    world = _build_world(with_deriver=False, activated=False)
    yield world
    for dispatcher in world.dispatchers:
        await dispatcher.stop(timeout=2.0)
    for client in world.clients:
        await client.aclose()


@pytest_asyncio.fixture
async def backlog_world_with_deriver():
    world = _build_world(with_deriver=True)
    yield world
    for dispatcher in world.dispatchers:
        await dispatcher.stop(timeout=2.0)
    for client in world.clients:
        await client.aclose()


@pytest.fixture
def ts() -> Callable[[int], str]:
    """Minute-granular UTC timestamps, e.g. minute 3 -> 2026-09-01T10:03:00+00:00."""

    def _ts(minute: int) -> str:
        return f"2026-09-01T10:{minute:02d}:00+00:00"

    return _ts
