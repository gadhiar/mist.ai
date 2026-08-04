"""Reachability tests for the curation scheduler's live collaborators.

These assert on the CONSTRUCTED OBJECT GRAPH, never on call arguments. That
distinction is the whole point of the file. A test of the form "assert the
factory was called with tracker=X" would have passed against the defective
2026-08-03 wiring the moment someone passed ANY tracker, because the defect
was not a missing argument -- it was the WRONG INSTANCE:
`build_conversation_handler` created one `ToolUsageTracker` and handed it to
`ConversationHandler`; `build_curation_scheduler` created a second and handed
it to `SkillDerivationJob`. Both objects existed, both were correctly typed,
`.record()` went to one and `detect_patterns()` read the other, and the
feature produced nothing for the life of every process.

The only assertion that separates those two worlds is identity -- `is`, not
`==`, and reached by walking from the scheduler down to the job rather than
by inspecting what a mock received.

`tests/unit/test_composition_root_completeness.py` covers the complementary
static property (the composition root MENTIONS every optional dependency).
Neither file subsumes the other: the guard cannot see runtime values, and
these tests cannot see a call site that was never written.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend import factories, server
from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.extraction.tool_usage_tracker import ToolCallRecord, ToolUsageTracker
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.unit.conftest import make_test_conventions_loader

FACTORIES = "backend.factories"


class FakeExtractionPipeline:
    """Test double standing in for the real extraction pipeline."""

    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


def _make_handler(*, event_store_enabled: bool = True) -> ConversationHandler:
    """Build a real ConversationHandler over fakes, with a real tracker.

    A real handler rather than a stand-in: the whole question these tests
    answer is which attribute of the PRODUCTION class the composition root
    reads, so substituting an object whose attribute names this file chose
    would beg it.
    """
    config = build_test_config(
        event_store_enabled=event_store_enabled,
        event_store_db_path=":memory:",
    )
    graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
    return ConversationHandler(
        config=config,
        graph_store=graph_store,
        extraction_pipeline=FakeExtractionPipeline(),
        retriever=KnowledgeRetriever(config=config, graph_store=graph_store),
        llm_provider=FakeLLM(),
        conventions_loader=make_test_conventions_loader(),
        tool_usage_tracker=ToolUsageTracker(config.skill_derivation),
    )


def _voice_processor_for(handler: ConversationHandler | None) -> SimpleNamespace:
    """Wrap a handler in the voice_processor -> models -> knowledge chain."""
    return SimpleNamespace(
        models=SimpleNamespace(knowledge=SimpleNamespace(conversation_handler=handler))
    )


def _job(scheduler, name: str):
    """Return the job instance registered under `name`, or fail loudly."""
    for job_config, job in scheduler._jobs:
        if job_config.name == name:
            return job
    registered = [c.name for c, _ in scheduler._jobs]
    raise AssertionError(f"No job named {name!r} on the scheduler. Registered: {registered}")


def _build_scheduler(config, **dependencies):
    """Build a real CurationScheduler with the graph layer faked out.

    Only `build_graph_store` / `build_graph_executor` are patched -- both
    open a live Neo4j connection. Everything downstream of them (the jobs,
    the deriver, the tracker wiring) is the real production construction,
    which is precisely what is under test.
    """
    connection = FakeNeo4jConnection()
    graph_store = GraphStore(connection, FakeEmbeddingGenerator())
    executor = FakeGraphExecutor(connection)

    with (
        patch(f"{FACTORIES}.build_graph_store", return_value=graph_store),
        patch(f"{FACTORIES}.build_graph_executor", return_value=executor),
    ):
        scheduler = factories.build_curation_scheduler(config, **dependencies)
    return scheduler, connection


def _tool_call(index: int) -> ToolCallRecord:
    """A tool call that clusters with its siblings under one context."""
    return ToolCallRecord(
        tool_name="graph_search",
        tool_type="search",
        context="searching the knowledge graph for entities",
        success=True,
        timestamp=datetime.now(UTC),
        session_id="session-wiring-001",
        event_id=f"event-wiring-{index:03d}",
    )


class TestResolveCurationDependencies:
    def test_returns_the_handlers_own_objects(self):
        """The resolver must hand back the live handler's collaborators, not
        equivalents of them.
        """
        handler = _make_handler()

        deps = server._resolve_curation_dependencies(_voice_processor_for(handler))

        assert deps.handler_found is True
        assert deps.tracker is handler._tool_usage_tracker
        assert deps.event_store is handler.event_store
        assert deps.llm_provider is handler._provider

    @pytest.mark.parametrize(
        "voice_processor",
        [
            pytest.param(None, id="no-voice-processor"),
            pytest.param(SimpleNamespace(models=None), id="no-models"),
            pytest.param(
                SimpleNamespace(models=SimpleNamespace(knowledge=None)),
                id="no-knowledge-integration",
            ),
            pytest.param(
                SimpleNamespace(models=SimpleNamespace(knowledge=SimpleNamespace())),
                id="no-conversation-handler-attribute",
            ),
        ],
    )
    def test_reports_handler_not_found_on_every_broken_link(self, voice_processor):
        """Each link in the chain is optional in production. A break at any
        of them must surface as `handler_found=False` -- the signal lifespan
        logs a warning on -- rather than as an AttributeError that the
        surrounding `except Exception` would swallow into "scheduler failed
        to start".
        """
        if isinstance(voice_processor, SimpleNamespace) and hasattr(voice_processor, "models"):
            knowledge = getattr(voice_processor.models, "knowledge", None)
            if knowledge is not None and not hasattr(knowledge, "conversation_handler"):
                knowledge.conversation_handler = None

        deps = server._resolve_curation_dependencies(voice_processor)

        assert deps.handler_found is False
        assert deps.event_store is None
        assert deps.tracker is None
        assert deps.llm_provider is None

    def test_distinguishes_a_disabled_event_store_from_a_missing_handler(self):
        """`config.event_store.enabled = False` leaves `handler.event_store`
        None legitimately. That must NOT read as broken wiring: the tracker
        and provider still arrive, and `handler_found` stays True so lifespan
        logs "disabled by configuration" rather than "inert".
        """
        handler = _make_handler(event_store_enabled=False)
        assert handler.event_store is None, "precondition: config disabled the store"

        deps = server._resolve_curation_dependencies(_voice_processor_for(handler))

        assert deps.handler_found is True
        assert deps.event_store is None
        assert deps.tracker is handler._tool_usage_tracker
        assert deps.llm_provider is handler._provider


class TestSchedulerObjectGraph:
    def test_skill_job_holds_the_same_tracker_object_the_handler_records_into(self):
        """The 2026-08-03 defect in one assertion. `==` would not catch it:
        two empty ToolUsageTrackers built from the same config are
        indistinguishable by value.
        """
        handler = _make_handler()
        deps = server._resolve_curation_dependencies(_voice_processor_for(handler))

        scheduler, _ = _build_scheduler(
            handler.config,
            event_store=deps.event_store,
            tracker=deps.tracker,
            llm_provider=deps.llm_provider,
        )

        assert _job(scheduler, "skill_derivation")._tracker is handler._tool_usage_tracker

    def test_reflection_job_holds_the_handlers_event_store_when_enabled(self):
        """With the store enabled, SelfReflectionJob must receive a non-None
        store -- and specifically the handler's, since that is the only one
        live traffic appends turns to.
        """
        handler = _make_handler(event_store_enabled=True)
        assert handler.event_store is not None, "precondition: config enabled the store"
        deps = server._resolve_curation_dependencies(_voice_processor_for(handler))

        scheduler, _ = _build_scheduler(
            handler.config,
            event_store=deps.event_store,
            tracker=deps.tracker,
            llm_provider=deps.llm_provider,
        )

        reflection_job = _job(scheduler, "self_reflection")
        assert reflection_job._event_store is not None
        assert reflection_job._event_store is handler.event_store

    def test_internal_deriver_uses_the_handlers_llm_provider(self):
        """Supplying the provider is what keeps the deriver's LLM calls on
        the same (possibly instrumented) provider the turn path uses, instead
        of a second one the factory builds with no debug logger.
        """
        handler = _make_handler()
        deps = server._resolve_curation_dependencies(_voice_processor_for(handler))

        scheduler, _ = _build_scheduler(
            handler.config,
            event_store=deps.event_store,
            tracker=deps.tracker,
            llm_provider=deps.llm_provider,
        )

        assert _job(scheduler, "self_reflection")._deriver._llm is handler._provider


class TestSkillDerivationIsReachable:
    @pytest.mark.asyncio
    async def test_tool_calls_recorded_on_the_handler_reach_the_schedulers_job(self):
        """End-to-end within the object graph: record through the handler's
        tracker, then run the scheduler's OWN job and require that it saw
        them. This is the property "the parameter was passed" cannot express.
        """
        handler = _make_handler()
        deps = server._resolve_curation_dependencies(_voice_processor_for(handler))
        scheduler, connection = _build_scheduler(
            handler.config,
            event_store=deps.event_store,
            tracker=deps.tracker,
            llm_provider=deps.llm_provider,
        )

        # skill_threshold is 3 -- three clustered calls make one pattern.
        for index in range(handler.config.skill_derivation.skill_threshold):
            handler._tool_usage_tracker.record(_tool_call(index))
        result = await _job(scheduler, "skill_derivation").run()

        assert result.patterns_detected == 1
        assert result.skills_created == 1
        connection.assert_write_executed("e.entity_type = 'Skill'")

    @pytest.mark.asyncio
    async def test_defaulted_tracker_reproduces_the_2026_08_03_defect(self):
        """Teeth for the assertions above. Built the way `server.py` built it
        until 2026-08-03 -- config only -- the scheduler's job reads a
        different tracker, and the same three recorded calls produce nothing.
        If this test ever starts failing, the identity assertions above have
        stopped proving anything and the factory defaults changed underneath
        them.
        """
        handler = _make_handler()
        scheduler, connection = _build_scheduler(handler.config)

        for index in range(handler.config.skill_derivation.skill_threshold):
            handler._tool_usage_tracker.record(_tool_call(index))
        skill_job = _job(scheduler, "skill_derivation")
        result = await skill_job.run()

        assert skill_job._tracker is not handler._tool_usage_tracker
        assert result.patterns_detected == 0
        assert result.skills_created == 0
        connection.assert_no_writes()

    @pytest.mark.asyncio
    async def test_defaulted_event_store_makes_reflection_inert(self):
        """The other half of the defect: with no event store the reflection
        job returns zeros on its first line, whatever is in the log.
        """
        handler = _make_handler()
        scheduler, _ = _build_scheduler(handler.config)

        result = await _job(scheduler, "self_reflection").run()

        assert result.events_processed == 0
        assert result.operations_applied == 0
        assert result.duration_ms == 0.0
