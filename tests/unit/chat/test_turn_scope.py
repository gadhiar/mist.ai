"""Tests for `TurnScope` and its use as the carrier of per-turn handler state.

Before T4 the FE event buffer, session id and turn index lived on the shared
`ConversationHandler`, so a second turn starting while the first was still in
flight reset the first turn's buffer and overwrote its graph seed. These tests
pin the fix: each turn owns a `TurnScope`, two interleaved turns on different
sessions see only their own tool events and seed, and the turn index stays one
counter per handler so a single session's seeds are what they were before.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest

from backend.chat.conversation_handler import ConversationHandler, _build_graph_subgraph_payload
from backend.chat.stream_events import Complete, WSEvent
from backend.chat.turn_scope import TurnScope
from backend.knowledge.models import RetrievalResult, RetrievedFact
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm.models import LLMRequest, LLMResponse, ToolCall
from backend.llm.provider import StreamingLLMProvider
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
from tests.unit.conftest import make_test_conventions_loader


class _NoopExtractionPipeline:
    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


def _retrieval_result(query: str, *, with_fact: bool) -> RetrievalResult:
    facts = (
        [
            RetrievedFact(
                subject="User",
                subject_type="Person",
                predicate="USES",
                object=query,
                object_type="Technology",
                properties={},
                similarity_score=0.9,
                graph_distance=0,
            )
        ]
        if with_fact
        else []
    )
    return RetrievalResult(
        query=query,
        user_id="User",
        facts=facts,
        entities_found=len(facts),
        total_facts=len(facts),
        formatted_context=f"context for {query}",
        retrieval_time_ms=0.0,
        vector_search_time_ms=0.0,
        graph_traversal_time_ms=0.0,
        config_used={},
    )


def _make_handler(provider: StreamingLLMProvider) -> ConversationHandler:
    gs = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
    config = build_test_config()
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=_NoopExtractionPipeline(),
        retriever=KnowledgeRetriever(config=config, graph_store=gs),
        llm_provider=provider,
        conventions_loader=make_test_conventions_loader(),
    )


async def _run_turn(
    handler: ConversationHandler, session_id: str, message: str
) -> tuple[list[dict], Complete]:
    payloads: list[dict] = []
    complete: Complete | None = None
    async for event in handler.handle_message_streaming(
        user_message=message, session_id=session_id
    ):
        if isinstance(event, WSEvent):
            payloads.append(event.payload)
        elif isinstance(event, Complete):
            complete = event
    assert complete is not None
    return payloads, complete


class TestTurnScope:
    def test_graph_seed_joins_session_and_turn_index(self):
        scope = TurnScope(session_id="sess-1", turn_index=7)

        assert scope.graph_seed == "sess-1:7"

    def test_drain_returns_emitted_events_in_order_and_empties_the_buffer(self):
        scope = TurnScope(session_id="s", turn_index=1)
        scope.emit({"type": "tool_call_started"})
        scope.emit({"type": "tool_call_completed"})

        drained = scope.drain_ws_events()

        assert [e["type"] for e in drained] == ["tool_call_started", "tool_call_completed"]
        assert scope.ws_events == []
        assert scope.drain_ws_events() == []

    def test_a_fresh_scope_has_no_trace_builder(self):
        scope = TurnScope(session_id="s", turn_index=1)

        assert (scope.trace, scope.trace_failed, scope.tool_call_index) == (None, False, 0)

    def test_scopes_do_not_share_an_event_buffer(self):
        a = TurnScope(session_id="a", turn_index=1)
        b = TurnScope(session_id="b", turn_index=2)

        a.emit({"type": "cards_dismiss"})

        assert b.ws_events == []


class TestScopeIsNotAToolArgument:
    @pytest.mark.asyncio
    async def test_model_supplied_scope_argument_fails_the_tool_call(self):
        """An LLM tool call that passes `scope` as an argument is a tool
        error, not a way to replace the turn's scope.
        """
        provider = FakeStreamingLLMProvider(
            [
                ScriptedPass(
                    tool_calls=[
                        ToolCall(
                            id="call-spoof",
                            name="frontend.dismiss_cards",
                            arguments={"scope": "other-session"},
                        )
                    ]
                ),
                ScriptedPass(chunks=["Done."]),
            ]
        )
        handler = _make_handler(provider)

        events, _ = await _run_turn(handler, "s1", "hi")

        assert [e["type"] for e in events] == ["tool_call_started", "tool_call_completed"]
        assert events[1]["error"].startswith("TypeError: ")


class _InterleavingProvider(StreamingLLMProvider):
    """Serves both sessions' turns from one provider, yielding to the loop
    between chunks so concurrent turns alternate at every chunk.

    The turn is identified by the last user message in the request: pass 1
    (a request carrying tools) asks for one `query_knowledge_graph` call whose
    query is that message; pass 2 answers in text.
    """

    model: str = "fake-interleaving-model"

    async def generate(
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncGenerator[LLMResponse, None]:
        tag = next(m["content"] for m in reversed(request.messages) if m["role"] == "user")
        if request.tools:
            yield LLMResponse(content="Checking.", partial=True)
            await asyncio.sleep(0)
            yield LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"call-{tag}", name="query_knowledge_graph", arguments={"query": tag}
                    )
                ],
                partial=False,
            )
            return
        for piece in ("Answer ", f"for {tag}."):
            await asyncio.sleep(0)
            yield LLMResponse(content=piece, partial=True)
        yield LLMResponse(content=None, partial=False)

    def generate_sync(
        self, request: LLMRequest, *, stream: bool = False
    ) -> Generator[LLMResponse, None, None]:
        raise NotImplementedError("async-only fake")


class TestInterleavedTurns:
    @pytest.mark.asyncio
    async def test_concurrent_turns_on_different_sessions_keep_their_own_events_and_seed(self):
        handler = _make_handler(_InterleavingProvider())
        # Both graph queries must be in flight at once: each waits here until
        # the other has also started its tool call.
        both_in_tool = asyncio.Barrier(2)

        async def retrieve(**kwargs):
            if kwargs.get("force_intent") == "historical":
                return _retrieval_result(kwargs["query"], with_fact=False)
            await both_in_tool.wait()
            return _retrieval_result(kwargs["query"], with_fact=True)

        handler.retriever.retrieve = retrieve

        (alpha_events, alpha_done), (beta_events, beta_done) = await asyncio.wait_for(
            asyncio.gather(
                _run_turn(handler, "sess-a", "alpha"),
                _run_turn(handler, "sess-b", "beta"),
            ),
            timeout=5,
        )

        expected_alpha_graph = _build_graph_subgraph_payload(
            _retrieval_result("alpha", with_fact=True), seed="sess-a:1"
        )
        expected_beta_graph = _build_graph_subgraph_payload(
            _retrieval_result("beta", with_fact=True), seed="sess-b:2"
        )
        assert [e["type"] for e in alpha_events] == [
            "tool_call_started",
            "graph_subgraph",
            "tool_call_completed",
        ]
        assert [e["type"] for e in beta_events] == [
            "tool_call_started",
            "graph_subgraph",
            "tool_call_completed",
        ]
        assert {e.get("tool_call_id") for e in alpha_events} - {None} == {"call-alpha"}
        assert {e.get("tool_call_id") for e in beta_events} - {None} == {"call-beta"}
        assert alpha_events[1] == expected_alpha_graph
        assert beta_events[1] == expected_beta_graph
        assert alpha_done.final_response.endswith("Answer for alpha.")
        assert beta_done.final_response.endswith("Answer for beta.")
        assert alpha_done.tool_calls_used == 1
        assert beta_done.tool_calls_used == 1


class TestTurnIndexIsProcessMonotonic:
    @staticmethod
    def _graph_turn_passes(count: int) -> list[ScriptedPass]:
        passes: list[ScriptedPass] = []
        for i in range(count):
            passes.append(
                ScriptedPass(
                    tool_calls=[
                        ToolCall(
                            id=f"call-{i}", name="query_knowledge_graph", arguments={"query": "x"}
                        )
                    ]
                )
            )
            passes.append(ScriptedPass(chunks=["Done."]))
        return passes

    @staticmethod
    def _stub_graph_retrieval(handler: ConversationHandler) -> None:
        async def retrieve(**kwargs):
            with_fact = kwargs.get("force_intent") != "historical"
            return _retrieval_result(kwargs["query"], with_fact=with_fact)

        handler.retriever.retrieve = retrieve

    @staticmethod
    def _graph_payload(events: list[dict]) -> dict:
        (payload,) = (e for e in events if e["type"] == "graph_subgraph")
        return payload

    @pytest.mark.asyncio
    async def test_successive_turns_in_one_session_seed_with_increasing_indices(self):
        handler = _make_handler(FakeStreamingLLMProvider(self._graph_turn_passes(2)))
        self._stub_graph_retrieval(handler)

        first, _ = await _run_turn(handler, "s1", "hi")
        second, _ = await _run_turn(handler, "s1", "hi")

        fact = _retrieval_result("x", with_fact=True)
        assert self._graph_payload(first) == _build_graph_subgraph_payload(fact, seed="s1:1")
        assert self._graph_payload(second) == _build_graph_subgraph_payload(fact, seed="s1:2")

    @pytest.mark.asyncio
    async def test_turn_index_is_one_counter_across_sessions(self):
        handler = _make_handler(FakeStreamingLLMProvider(self._graph_turn_passes(3)))
        self._stub_graph_retrieval(handler)

        first, _ = await _run_turn(handler, "s1", "hi")
        other, _ = await _run_turn(handler, "s2", "hi")
        third, _ = await _run_turn(handler, "s1", "hi")

        fact = _retrieval_result("x", with_fact=True)
        assert self._graph_payload(first) == _build_graph_subgraph_payload(fact, seed="s1:1")
        assert self._graph_payload(other) == _build_graph_subgraph_payload(fact, seed="s2:2")
        assert self._graph_payload(third) == _build_graph_subgraph_payload(fact, seed="s1:3")
