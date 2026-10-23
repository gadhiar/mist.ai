"""Turns split into `open_turn`, `stream_turn` and `commit_turn` (MIS-127).

`stream_turn` writes nothing durable; `commit_turn` writes session history
and the event-log row and starts extraction, recording what the user actually
received (`TurnDelivery.delivered_text`) and, when that differs from what the
model generated, the full generated text beside it. These tests pin the write
split, the recorded fields for each outcome, idempotency, the event-store
failure and disabled cases, and that `handle_message_streaming` still records
a completed turn exactly as before the split.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.chat.stream_events import Complete, StreamEvent, Token, WSEvent
from backend.chat.turn_scope import CommitResult, TurnDelivery, TurnRequest, TurnScope
from backend.errors import LLMConnectionError
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm.models import ToolCall as LLMToolCall
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
from tests.unit.conftest import make_test_conventions_loader


class _NoopExtractionPipeline:
    def __init__(self) -> None:
        self.utterances: list[str] = []

    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        self.utterances.append(kwargs.get("utterance", ""))
        return ValidationResult(valid=True, entities=[], relationships=[])


class _FakeDispatcher:
    """Stands in for `ExtractionDispatcher`: counts wakes, takes listeners."""

    def __init__(self) -> None:
        self.wakes = 0

    def add_apply_listener(self, listener: Callable) -> None:
        pass

    def wake(self) -> None:
        self.wakes += 1


class _FailingEventStore:
    """An event store whose turn write fails, as a locked SQLite file would."""

    def get_session(self, session_id: str):
        return None

    def start_session(self, session_id: str, input_modality: str = "voice", origin: str = "real"):
        pass

    def append_turn(self, event) -> str:
        raise sqlite3.OperationalError("database is locked")


def _handler(
    passes: list[ScriptedPass], *, event_store: bool = True
) -> tuple[ConversationHandler, FakeStreamingLLMProvider]:
    conn = FakeNeo4jConnection()
    graph_store = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config(event_store_enabled=event_store, event_store_db_path=":memory:")
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


def _request(
    session_id: str = "s1", text: str = "hello", origin: str = "text", **kwargs
) -> TurnRequest:
    channel = "tts" if origin == "voice" else "text"
    return TurnRequest(
        session_id=session_id, user_text=text, origin=origin, channel=channel, **kwargs
    )


def _delivery(outcome: str, delivered: str) -> TurnDelivery:
    return TurnDelivery(outcome=outcome, delivered_text=delivered, stt_model=None, tts_model=None)


async def _stream_all(handler: ConversationHandler, scope: TurnScope) -> list[StreamEvent]:
    return [event async for event in handler.stream_turn(scope)]


def _history(handler: ConversationHandler, session_id: str) -> list[tuple[str, str]]:
    session = handler.sessions.get(session_id)
    if session is None:
        return []
    return [(m.role, m.content) for m in session.messages]


def _rows(handler: ConversationHandler, session_id: str) -> list[dict]:
    return handler.event_store.get_turns(session_id)


class TestStreamWritesNothing:
    @pytest.mark.asyncio
    async def test_stream_turn_leaves_history_and_event_log_untouched(self):
        handler, _ = _handler([ScriptedPass(chunks=["Hello there."])])
        scope = handler.open_turn(_request())

        events = await _stream_all(handler, scope)

        assert isinstance(events[-1], Complete)
        assert _history(handler, "s1") == []
        assert _rows(handler, "s1") == []
        assert scope.generated_text == "Hello there."

    @pytest.mark.asyncio
    async def test_prompt_still_ends_with_the_uncommitted_user_message(self):
        handler, provider = _handler([ScriptedPass(chunks=["Hi."])])
        scope = handler.open_turn(_request(text="what is my name"))

        await _stream_all(handler, scope)

        assert provider.calls[0].messages[-1] == {"role": "user", "content": "what is my name"}


class TestCommitCompleted:
    @pytest.mark.asyncio
    async def test_commit_writes_history_and_one_completed_row(self):
        handler, _ = _handler([ScriptedPass(chunks=["Hello there."])])
        scope = handler.open_turn(_request())
        await _stream_all(handler, scope)

        result = handler.commit_turn(scope, _delivery("completed", scope.generated_text))

        assert _history(handler, "s1") == [("user", "hello"), ("assistant", "Hello there.")]
        rows = _rows(handler, "s1")
        assert len(rows) == 1
        assert rows[0]["event_id"] == result.event_id
        assert rows[0]["system_response"] == "Hello there."
        assert rows[0]["turn_outcome"] == "completed"
        assert rows[0]["generated_response"] is None
        assert rows[0]["turn_modality"] == "text"
        assert result == CommitResult(recorded=True, event_id=result.event_id, error=None)

    @pytest.mark.asyncio
    async def test_commit_without_dispatcher_creates_one_extraction_task(self):
        handler, _ = _handler([ScriptedPass(chunks=["Noted."])])
        scope = handler.open_turn(_request(text="I use Rust daily"))
        await _stream_all(handler, scope)

        handler.commit_turn(scope, _delivery("completed", scope.generated_text))

        assert list(handler._extraction_tasks.values()) == ["s1"]
        await handler._drain_extraction_tasks()
        assert handler._extraction_pipeline.utterances == ["I use Rust daily"]

    @pytest.mark.asyncio
    async def test_commit_with_dispatcher_wakes_it_and_creates_no_task(self):
        handler, _ = _handler([ScriptedPass(chunks=["Noted."])])
        dispatcher = _FakeDispatcher()
        handler.attach_extraction_dispatcher(dispatcher)
        scope = handler.open_turn(_request())
        await _stream_all(handler, scope)

        handler.commit_turn(scope, _delivery("completed", scope.generated_text))

        assert dispatcher.wakes == 1
        assert handler._extraction_tasks == {}


class TestHandleMessageStreamingRecordsAsBefore:
    @pytest.mark.asyncio
    async def test_tool_turn_events_and_every_recorded_field(self):
        tool_call = LLMToolCall(
            id="call_1", name="query_knowledge_graph", arguments={"query": "rust"}
        )
        handler, provider = _handler(
            [
                ScriptedPass(chunks=[], tool_calls=[tool_call]),
                ScriptedPass(chunks=["You know Rust."]),
            ]
        )

        async def stub_dispatch(scope, tc):
            return "stub tool result"

        handler._dispatch_tool = stub_dispatch

        events = [
            event
            async for event in handler.handle_message_streaming(
                user_message="what do I know", session_id="s-fields"
            )
        ]

        assert [type(e) for e in events] == [WSEvent, WSEvent, Token, Complete]
        assert [e.payload["type"] for e in events[:2]] == [
            "tool_call_started",
            "tool_call_completed",
        ]
        assert events[2].text == "You know Rust."
        assert events[2].pass_num == 2
        assert events[3].final_response == "You know Rust."
        assert events[3].tool_calls_used == 1
        rows = _rows(handler, "s-fields")
        assert len(rows) == 1
        row = rows[0]
        expected = {
            "session_id": "s-fields",
            "turn_index": 0,
            "user_utterance": "what do I know",
            "system_response": "You know Rust.",
            "context_window": provider.calls[1].messages,
            "retrieval_context": None,
            "tool_calls": [{"name": "query_knowledge_graph", "args": {"query": "rust"}}],
            "audio_hash": None,
            "audio_format": None,
            "audio_duration_ms": None,
            "audio_sample_rate": None,
            "stt_model": None,
            "tts_model": None,
            "llm_model": provider.model,
            "llm_parameters": {"temperature": 0.7},
            "ontology_version": handler.config.ontology_version,
            "turn_outcome": "completed",
            "generated_response": None,
            "turn_modality": "text",
        }
        assert {key: row[key] for key in expected} == expected
        session = handler.event_store.get_session("s-fields")
        assert session.input_modality == "text"
        assert session.turn_count == 1

    @pytest.mark.asyncio
    async def test_error_turn_records_error_outcome(self):
        handler, _ = _handler(
            [ScriptedPass(chunks=[], error=LLMConnectionError("llama-server unreachable"))]
        )

        events = [
            event
            async for event in handler.handle_message_streaming(
                user_message="hi", session_id="s-err"
            )
        ]

        complete = events[-1]
        assert complete.final_response == "I encountered an error: llama-server unreachable"
        rows = _rows(handler, "s-err")
        assert [r["turn_outcome"] for r in rows] == ["error"]
        assert rows[0]["system_response"] == complete.final_response
        assert _history(handler, "s-err") == [
            ("user", "hi"),
            ("assistant", complete.final_response),
        ]


class TestCommitInterrupted:
    @pytest.mark.asyncio
    async def test_interrupted_records_delivered_and_generated_text(self):
        handler, _ = _handler([ScriptedPass(chunks=["First part. Second part."])])
        scope = handler.open_turn(_request(origin="voice"))
        await _stream_all(handler, scope)

        handler.commit_turn(scope, _delivery("interrupted", "First part."))

        row = _rows(handler, "s1")[0]
        assert row["system_response"] == "First part."
        assert row["generated_response"] == "First part. Second part."
        assert row["turn_outcome"] == "interrupted"
        assert row["turn_modality"] == "voice"
        assert _history(handler, "s1") == [("user", "hello"), ("assistant", "First part.")]

    @pytest.mark.asyncio
    async def test_interrupted_with_nothing_delivered_records_utterance_and_empty_reply(self):
        handler, _ = _handler([ScriptedPass(chunks=["Never heard."])])
        scope = handler.open_turn(_request(text="are you there", origin="voice"))
        await _stream_all(handler, scope)

        handler.commit_turn(scope, _delivery("interrupted", ""))

        row = _rows(handler, "s1")[0]
        assert row["user_utterance"] == "are you there"
        assert row["system_response"] == ""
        assert row["generated_response"] == "Never heard."
        assert _history(handler, "s1") == [("user", "are you there"), ("assistant", "")]

    def test_scope_carries_the_request_with_its_interruption_note(self):
        handler, _ = _handler([])
        request = _request(interruption_note="mid_reply")

        scope = handler.open_turn(request)

        assert scope.request is request
        assert scope.request.interruption_note == "mid_reply"


class TestCommitResult:
    @pytest.mark.asyncio
    async def test_second_commit_returns_first_result_and_writes_nothing(self):
        handler, _ = _handler([ScriptedPass(chunks=["Once."])])
        scope = handler.open_turn(_request())
        await _stream_all(handler, scope)
        first = handler.commit_turn(scope, _delivery("completed", "Once."))

        second = handler.commit_turn(scope, _delivery("interrupted", ""))

        assert second == first
        assert len(_rows(handler, "s1")) == 1
        assert len(_history(handler, "s1")) == 2

    @pytest.mark.asyncio
    async def test_failed_event_store_write_returns_error_and_raises_nothing(self):
        handler, _ = _handler([ScriptedPass(chunks=["Hi."])])
        handler.event_store = _FailingEventStore()
        scope = handler.open_turn(_request())
        await _stream_all(handler, scope)

        result = handler.commit_turn(scope, _delivery("completed", "Hi."))

        assert result.recorded is False
        assert result.event_id is None
        assert result.error

    @pytest.mark.asyncio
    async def test_disabled_event_store_returns_unrecorded_result_without_error(self):
        handler, _ = _handler([ScriptedPass(chunks=["Hi."])], event_store=False)
        scope = handler.open_turn(_request())
        await _stream_all(handler, scope)

        result = handler.commit_turn(scope, _delivery("completed", "Hi."))

        assert handler.event_store is None
        assert result == CommitResult(False, None, None)


class TestModality:
    @pytest.mark.asyncio
    async def test_session_modality_is_first_turn_origin_and_each_turn_keeps_its_own(self):
        handler, _ = _handler([ScriptedPass(chunks=["One."]), ScriptedPass(chunks=["Two."])])
        voice = handler.open_turn(_request(origin="voice"))
        await _stream_all(handler, voice)
        handler.commit_turn(voice, _delivery("completed", "One."))

        typed = handler.open_turn(_request(origin="text"))
        await _stream_all(handler, typed)
        handler.commit_turn(typed, _delivery("completed", "Two."))

        assert handler.event_store.get_session("s1").input_modality == "voice"
        assert [r["turn_modality"] for r in _rows(handler, "s1")] == ["voice", "text"]

    def test_open_turn_refuses_system_origin(self):
        handler, _ = _handler([])

        with pytest.raises(ValueError, match="system"):
            handler.open_turn(_request(origin="system"))


class TestOpenTurnAndPrefetch:
    @pytest.mark.parametrize(
        ("held", "gate_set"),
        [pytest.param(False, True, id="not-held"), pytest.param(True, False, id="held")],
    )
    def test_open_turn_scope_exposes_the_design_fields(self, held: bool, gate_set: bool):
        handler, _ = _handler([])

        scope = handler.open_turn(_request(held=held))

        assert scope.dispatch_gate.is_set() is gate_set
        assert scope.generated_text == ""
        assert scope.tool_dispatched is False
        assert scope.tool_calls == []
        assert scope.tool_results == []
        assert scope.prompt_stats is None

    def test_prefetch_returns_none_and_changes_nothing(self):
        handler, _ = _handler([])
        before = handler.open_turn(_request()).turn_index
        sessions_before = dict(handler.sessions)

        result = handler.prefetch("s1", "partial words")

        assert result is None
        assert handler.sessions == sessions_before
        assert handler.open_turn(_request()).turn_index == before + 1
