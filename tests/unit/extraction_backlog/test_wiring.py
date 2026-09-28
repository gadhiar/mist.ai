"""Wiring: settings, the factory, the handler's wake/drain path, and the admin CLI."""

from __future__ import annotations

import io

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.extraction_backlog import admin
from backend.extraction_backlog.dispatcher import ExtractionDispatcher
from backend.extraction_backlog.settings import DEFAULT_SERVICE_URL, DispatcherSettings
from backend.knowledge.extraction.validator import ValidationResult
from backend.knowledge.extraction_cache import OUTCOME_SKIPPED, SKIP_EXTRACTION_FAILED
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.unit.conftest import make_test_conventions_loader


class TestSettings:
    def test_defaults_match_the_brief(self, monkeypatch):
        for name in (
            "MIST_EXTRACTION_INFERENCE",
            "MIST_EXTRACTION_SERVICE_URL",
            "MIST_EXTRACTION_MAX_ATTEMPTS",
            "MIST_EXTRACTION_BACKOFF_BASE_S",
            "MIST_EXTRACTION_BACKOFF_CAP_S",
        ):
            monkeypatch.delenv(name, raising=False)

        settings = DispatcherSettings.from_env()

        assert settings.mode == "service"
        assert settings.service_url == DEFAULT_SERVICE_URL == "http://mist-extraction:8090"
        assert (settings.max_attempts, settings.backoff_base_s, settings.backoff_cap_s) == (
            5,
            2.0,
            60.0,
        )

    def test_reads_mode_and_url_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("MIST_EXTRACTION_INFERENCE", "OFF")
        monkeypatch.setenv("MIST_EXTRACTION_SERVICE_URL", "http://box:9000")

        settings = DispatcherSettings.from_env()

        assert (settings.mode, settings.service_url) == ("off", "http://box:9000")

    def test_refuses_an_unknown_mode(self, monkeypatch):
        monkeypatch.setenv("MIST_EXTRACTION_INFERENCE", "inprocess")

        with pytest.raises(ValueError, match="MIST_EXTRACTION_INFERENCE"):
            DispatcherSettings.from_env()

    @pytest.mark.parametrize(
        "failures, expected",
        [
            pytest.param(1, 2.0, id="first"),
            pytest.param(3, 8.0, id="doubles"),
            pytest.param(10, 60.0, id="capped"),
        ],
    )
    def test_backoff_is_exponential_and_capped(self, failures, expected):
        assert DispatcherSettings().backoff_s(failures) == expected


class _RecordingDispatcher:
    """The handler-facing surface of ExtractionDispatcher, recorded."""

    state = "idle"

    def __init__(self) -> None:
        self.wakes = 0
        self.drains: list[float] = []
        self.listeners: list = []

    def wake(self) -> None:
        self.wakes += 1

    async def drain(self, timeout: float) -> bool:
        self.drains.append(timeout)
        return True

    def add_apply_listener(self, listener) -> None:
        self.listeners.append(listener)


class _RecordingPipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def extract_from_utterance(self, **kwargs):
        self.calls.append(kwargs)
        return ValidationResult(valid=True, entities=[], relationships=[])


def _handler(*, dispatcher=None) -> tuple[ConversationHandler, _RecordingPipeline, FakeLLM]:
    config = build_test_config(event_store_enabled=True, event_store_db_path=":memory:")
    gs = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
    pipeline = _RecordingPipeline()
    llm = FakeLLM(default_response="Hello there.")
    handler = ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=pipeline,  # type: ignore[arg-type]
        retriever=KnowledgeRetriever(config=config, graph_store=gs),
        llm_provider=llm,
        conventions_loader=make_test_conventions_loader(),
        extraction_dispatcher=dispatcher,  # type: ignore[arg-type]
    )
    return handler, pipeline, llm


class TestHandlerWakePath:
    @pytest.mark.asyncio
    async def test_a_recorded_turn_wakes_the_dispatcher_instead_of_extracting_in_process(self):
        dispatcher = _RecordingDispatcher()
        handler, pipeline, _llm = _handler(dispatcher=dispatcher)

        await handler.handle_message(user_message="I use rust every day", session_id="s1")
        await handler._drain_extraction_tasks()

        assert dispatcher.wakes == 1
        assert handler._extraction_tasks == {}
        assert pipeline.calls == []

    @pytest.mark.asyncio
    async def test_drain_also_drains_the_backlog(self):
        dispatcher = _RecordingDispatcher()
        handler, _pipeline, _llm = _handler(dispatcher=dispatcher)

        await handler.aclose()

        assert dispatcher.drains == [60.0]

    @pytest.mark.asyncio
    async def test_without_a_dispatcher_the_in_process_path_is_unchanged(self):
        handler, pipeline, _llm = _handler()

        await handler.handle_message(user_message="I use rust every day", session_id="s1")
        await handler._drain_extraction_tasks()

        assert len(pipeline.calls) == 1
        assert pipeline.calls[0]["utterance"] == "I use rust every day"

    def test_attaching_registers_the_user_note_refresh_listener(self):
        dispatcher = _RecordingDispatcher()
        handler, _pipeline, _llm = _handler(dispatcher=dispatcher)

        assert dispatcher.listeners == [handler._on_extraction_applied]


class TestFactory:
    def test_builds_an_unstarted_dispatcher_over_the_handlers_pipeline(self, tmp_path, monkeypatch):
        from backend.factories import build_conversation_handler
        from tests.unit.knowledge.conftest import FakeVectorStore

        monkeypatch.setenv("MIST_EXTRACTION_INFERENCE", "service")
        config = build_test_config(
            event_store_enabled=True, event_store_db_path=str(tmp_path / "events.db")
        )
        gs = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())

        handler = build_conversation_handler(
            config,
            llm_provider=FakeLLM(),
            graph_store=gs,
            vector_store=FakeVectorStore(),
            with_extraction_dispatcher=True,
        )

        dispatcher = handler._extraction_dispatcher
        assert isinstance(dispatcher, ExtractionDispatcher)
        assert dispatcher.running is False
        assert dispatcher._pipeline is handler._extraction_pipeline
        assert dispatcher._store.event_store is handler.event_store
        assert dispatcher._store.cache is handler._extraction_pipeline.extraction_cache

    def test_the_default_leaves_the_handler_on_the_in_process_path(self, tmp_path):
        from backend.factories import build_conversation_handler
        from tests.unit.knowledge.conftest import FakeVectorStore

        config = build_test_config(
            event_store_enabled=True, event_store_db_path=str(tmp_path / "events.db")
        )
        gs = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())

        handler = build_conversation_handler(
            config, llm_provider=FakeLLM(), graph_store=gs, vector_store=FakeVectorStore()
        )

        assert handler._extraction_dispatcher is None

    def test_off_mode_builds_no_inference_client(self, tmp_path):
        from backend.event_store.store import EventStore
        from backend.factories import build_extraction_dispatcher
        from tests.unit.extraction_backlog.conftest import _build_world

        world = _build_world(with_deriver=False)
        config = build_test_config()

        dispatcher = build_extraction_dispatcher(
            config,
            pipeline=world.build_pipeline(),
            event_store=world.event_store,
            settings=DispatcherSettings(mode="off"),
            extraction_cache=world.cache,
            http_client=None,
        )

        assert isinstance(world.event_store, EventStore)
        assert dispatcher._inference is None
        assert dispatcher.snapshot().state == "disabled"

    def test_writer_stamps_are_the_ones_the_curation_graph_writer_stamps_with(self):
        from backend.factories import build_curation_pipeline, build_extraction_dispatcher
        from backend.knowledge.curation.graph_writer import RebuildStamps
        from backend.knowledge.version_stamps import compose_model_hash
        from tests.mocks.neo4j import FakeGraphExecutor
        from tests.unit.extraction_backlog.conftest import _build_world

        world = _build_world(with_deriver=False)
        config = build_test_config(embedding_model="wiring-emb")
        config.model_hash = "wiring-model"
        config.extraction_version = "wiring-ev"
        config.ontology_version = "7.7.7"

        dispatcher = build_extraction_dispatcher(
            config,
            pipeline=world.build_pipeline(),
            event_store=world.event_store,
            settings=DispatcherSettings(mode="off"),
            extraction_cache=world.cache,
        )
        curation = build_curation_pipeline(
            config,
            FakeGraphExecutor(connection=FakeNeo4jConnection()),
            embedding_provider=FakeEmbeddingGenerator(),
        )

        writer = dispatcher._writer_stamps
        assert writer == RebuildStamps(
            ontology_version="7.7.7",
            extraction_version="wiring-ev",
            model_hash=compose_model_hash(config),
        )
        assert writer.model_hash == "wiring-model|emb:wiring-emb"
        assert writer == curation._graph_writer._rebuild_stamps
        assert writer == curation._engine._stamps


class TestAdminCli:
    def test_status_prints_the_backlog_counts(self, ts):
        from tests.unit.extraction_backlog.conftest import _build_world

        world = _build_world(with_deriver=False)
        world.log_turn(session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily")
        out = io.StringIO()

        code = admin.main(["status"], store=world.store, out=out)

        text = out.getvalue()
        assert code == 0
        assert "backlog_depth=1" in text
        assert "dead_lettered=0" in text
        assert "legacy_unextracted=0" in text

    def test_retry_refuses_a_turn_that_is_not_dead_lettered(self, ts):
        from tests.unit.extraction_backlog.conftest import _build_world

        world = _build_world(with_deriver=False)
        event_id = world.log_turn(
            session_id="s1", turn_index=0, timestamp=ts(0), utterance="I use rust daily"
        )
        out = io.StringIO()

        code = admin.main(
            ["retry-dead-letters", "--event-id", event_id], store=world.store, out=out
        )

        assert code == 2
        assert "not dead-lettered" in out.getvalue()
        assert "OUT OF LOG ORDER" not in out.getvalue()

    def test_retry_one_event_id_leaves_other_dead_letters_alone(self, ts):
        from tests.unit.extraction_backlog.conftest import _build_world

        world = _build_world(with_deriver=False)
        store = world.store
        epoch = store.active_epoch()
        ids = [
            world.log_turn(
                session_id="s1", turn_index=i, timestamp=ts(i), utterance=f"I use t{i} now"
            )
            for i in range(2)
        ]
        for event_id in ids:
            store.put_skip(event_id, epoch, skip_reason=SKIP_EXTRACTION_FAILED, created_at=ts(0))
        out = io.StringIO()

        code = admin.main(["retry-dead-letters", "--event-id", ids[0]], store=store, out=out)

        assert code == 0
        assert store.get_cached(ids[0], epoch) is None
        remaining = store.get_cached(ids[1], epoch)
        assert (remaining["outcome"], remaining["skip_reason"]) == (
            OUTCOME_SKIPPED,
            SKIP_EXTRACTION_FAILED,
        )
