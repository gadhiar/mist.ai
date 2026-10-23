"""Cancelling a turn's stream records nothing; the scope keeps what happened.

A barge-in cancels the task consuming `stream_turn`. Wherever the cancel
lands (before the LLM call, mid pass 1, during a tool, mid pass 2), the
`CancelledError` must leave `stream_turn` with session history and the event
log untouched, and the scope must still say what was generated and which
tools ran, so the controller can commit the turn as `interrupted` with what
the user actually heard.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.chat.mist_context import MistContext
from backend.chat.stream_events import Token
from backend.chat.turn_scope import TurnDelivery, TurnRequest, TurnScope
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm.models import ToolCall as LLMToolCall
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
from tests.unit.conftest import make_test_conventions_loader

TOOL_CALL = LLMToolCall(id="call_1", name="query_knowledge_graph", arguments={"query": "rust"})


class _NoopExtractionPipeline:
    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


class _BlockingPersonaRetriever:
    """Retriever whose persona fetch blocks: the turn has not reached the LLM yet."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def retrieve_mist_context(self) -> MistContext:
        self.entered.set()
        await self.release.wait()
        return MistContext(
            display_name="MIST",
            pronouns="she/her",
            self_concept="",
            traits=[],
            capabilities=[],
            preferences=[],
        )


class _BlockingTool:
    """`_dispatch_tool` stand-in that runs until released."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[LLMToolCall] = []
        self.finished = 0

    async def __call__(self, scope: TurnScope, tool_call: LLMToolCall) -> str:
        self.calls.append(tool_call)
        self.entered.set()
        await self.release.wait()
        self.finished += 1
        return "tool result"


def _handler(passes: list[ScriptedPass]) -> tuple[ConversationHandler, FakeStreamingLLMProvider]:
    conn = FakeNeo4jConnection()
    graph_store = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config(event_store_enabled=True, event_store_db_path=":memory:")
    provider = FakeStreamingLLMProvider(passes)
    handler = ConversationHandler(
        config=config,
        graph_store=graph_store,
        extraction_pipeline=_NoopExtractionPipeline(),
        retriever=KnowledgeRetriever(config=config, graph_store=graph_store),
        llm_provider=provider,
        conventions_loader=make_test_conventions_loader(),
    )
    return handler, provider


def _request(origin: str = "voice", **kwargs) -> TurnRequest:
    channel = "tts" if origin == "voice" else "text"
    return TurnRequest(session_id="s1", user_text="hello", origin=origin, channel=channel, **kwargs)


def _consume(handler: ConversationHandler, scope: TurnScope, seen: list) -> asyncio.Task:
    async def run() -> None:
        async with contextlib.aclosing(handler.stream_turn(scope)) as events:
            async for event in events:
                seen.append(event)

    return asyncio.create_task(run())


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _assert_nothing_recorded(handler: ConversationHandler) -> None:
    session = handler.sessions.get("s1")
    assert session is None or session.messages == []
    assert handler.event_store.get_turns("s1") == []


class TestCancelRecordsNothing:
    @pytest.mark.asyncio
    async def test_cancel_before_the_llm_call(self):
        handler, provider = _handler([ScriptedPass(chunks=["Unused."])])
        retriever = _BlockingPersonaRetriever()
        handler.retriever = retriever
        scope = handler.open_turn(_request())
        task = _consume(handler, scope, [])
        await retriever.entered.wait()

        await _cancel(task)

        assert provider.calls == []
        _assert_nothing_recorded(handler)

    @pytest.mark.asyncio
    async def test_cancel_mid_pass_one(self):
        stall = asyncio.Event()
        handler, _ = _handler([ScriptedPass(chunks=["First sentence. ", stall, "Never."])])
        scope = handler.open_turn(_request())
        seen: list = []
        task = _consume(handler, scope, seen)
        while not seen:
            await asyncio.sleep(0)

        await _cancel(task)

        _assert_nothing_recorded(handler)
        assert scope.generated_text == "First sentence."

    @pytest.mark.asyncio
    async def test_cancel_during_a_tool(self):
        handler, _ = _handler(
            [ScriptedPass(chunks=[], tool_calls=[TOOL_CALL]), ScriptedPass(chunks=["Unused."])]
        )
        tool = _BlockingTool()
        handler._dispatch_tool = tool
        scope = handler.open_turn(_request())
        task = _consume(handler, scope, [])
        await tool.entered.wait()

        await _cancel(task)

        _assert_nothing_recorded(handler)
        assert scope.tool_dispatched is True
        assert scope.tool_calls == [{"name": "query_knowledge_graph", "args": {"query": "rust"}}]

    @pytest.mark.asyncio
    async def test_cancel_mid_pass_two(self):
        stall = asyncio.Event()
        handler, _ = _handler(
            [
                ScriptedPass(chunks=[], tool_calls=[TOOL_CALL]),
                ScriptedPass(chunks=["Pass two begins. ", stall, "Never."]),
            ]
        )

        async def quick_tool(scope, tc):
            return "tool result"

        handler._dispatch_tool = quick_tool
        scope = handler.open_turn(_request())
        seen: list = []
        task = _consume(handler, scope, seen)
        while not any(isinstance(e, Token) for e in seen):
            await asyncio.sleep(0)

        await _cancel(task)

        _assert_nothing_recorded(handler)
        assert scope.generated_text == "Pass two begins."
        assert scope.tool_results == [
            {"name": "query_knowledge_graph", "result": "tool result", "tool_call_id": "call_1"}
        ]

    @pytest.mark.asyncio
    async def test_aclose_at_a_yield_raises_no_runtime_error(self):
        stall = asyncio.Event()
        handler, _ = _handler([ScriptedPass(chunks=["First sentence. ", stall])])
        scope = handler.open_turn(_request())
        events = handler.stream_turn(scope)

        first = await events.__anext__()
        # An FE event waiting in the buffer, as a tool would leave one: a
        # drain that yielded from a `finally` or `except` block would hand it
        # out after `GeneratorExit`, and `aclose()` would raise RuntimeError.
        scope.emit({"type": "tool_call_started", "tool_call_id": "call_1"})
        await events.aclose()

        assert isinstance(first, Token)
        _assert_nothing_recorded(handler)


class TestCommitAfterCancel:
    @pytest.mark.asyncio
    async def test_interrupted_commit_records_heard_prefix_and_generated_text(self):
        stall = asyncio.Event()
        handler, _ = _handler(
            [ScriptedPass(chunks=["Heard this part. Not this part. ", stall, "Never."])]
        )
        scope = handler.open_turn(_request(origin="voice"))
        seen: list = []
        task = _consume(handler, scope, seen)
        while len(seen) < 2:
            await asyncio.sleep(0)
        await _cancel(task)

        handler.commit_turn(
            scope,
            TurnDelivery(
                outcome="interrupted",
                delivered_text="Heard this part.",
                stt_model="stt-fake",
                tts_model="tts-fake",
            ),
        )

        row = handler.event_store.get_turns("s1")[0]
        assert row["system_response"] == "Heard this part."
        assert row["generated_response"] == "Heard this part. Not this part."
        assert row["turn_outcome"] == "interrupted"
        assert row["turn_modality"] == "voice"
        assert (row["stt_model"], row["tts_model"]) == ("stt-fake", "tts-fake")
        history = [(m.role, m.content) for m in handler.sessions["s1"].messages]
        assert history == [("user", "hello"), ("assistant", "Heard this part.")]
