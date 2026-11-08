"""Capability commands answered before the model, through an attached `CapabilityTurnPort`.

`ConversationHandler.attach_capability_host` binds the port. Each turn then asks `pre_llm`
first, inside the turn's `try`, so a ringing item, a proposal decision or a fast-path intent is
answered with no LLM call, and an unexpected error from capability code reaches the turn's
existing error backstop. These tests drive whole turns through `handle_message_streaming` with
`FakeStreamingLLMProvider` and a fake port that follows the real port's signatures
(`backend/capabilities/contracts.py`, `CapabilityTurnPort`).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from backend.capabilities.contracts import ActionRecord, PreTurnResolution, ToolOutcome
from backend.chat.conversation_handler import KNOWLEDGE_TOOL_SCHEMAS, ConversationHandler
from backend.chat.retrieval_trace import TurnTraceBuilder
from backend.chat.stream_events import Complete, StreamEvent, Token, WSEvent
from backend.knowledge.models import RetrievalResult
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
from tests.unit.conftest import make_test_conventions_loader

_SECRET = "tok-quince-7741"
_SENTENCE = "Actions run only through the action system."

# --- Fakes --------------------------------------------------------------------


class _FakePort:
    """A `CapabilityTurnPort` whose `pre_llm` answers from a script, one item per call.

    A script item is a resolution, None, or an exception to raise. `drain_script` holds one
    list of payloads per `drain_events` call; a call past its end drains nothing.
    """

    def __init__(
        self,
        script: Sequence[PreTurnResolution | BaseException | None] = (),
        *,
        drain_script: Sequence[list[dict[str, object]]] = (),
        decide_result: PreTurnResolution | None = None,
    ) -> None:
        self._script = list(script)
        self._drain_script = list(drain_script)
        self._decide_result = decide_result
        self.pre_llm_calls: list[dict[str, Any]] = []
        self.decide_calls: list[tuple[str, str, str]] = []
        self.drain_calls = 0
        self.before_raise: list[Any] = []

    async def pre_llm(
        self, session_id: str, turn_id: str, utterance: str, *, received_at: float | None = None
    ) -> PreTurnResolution | None:
        self.pre_llm_calls.append(
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "utterance": utterance,
                "received_at": received_at,
            }
        )
        item = self._script.pop(0) if self._script else None
        if isinstance(item, BaseException):
            for hook in self.before_raise:
                hook()
            raise item
        return item

    def plan_turn(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        raise AssertionError("routing is not part of this turn")

    async def run_actions(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        raise AssertionError("the action decode is not part of this turn")

    async def decide(
        self, session_id: str, proposal_id: str, decision: Literal["approve", "reject"]
    ) -> PreTurnResolution:
        self.decide_calls.append((session_id, proposal_id, decision))
        assert self._decide_result is not None
        return self._decide_result

    def static_sentence(self) -> str:
        return _SENTENCE

    def redact(self, text: str) -> str:
        return text.replace(_SECRET, "***")

    def record_turn(self, session_id: str, used: Any) -> None:
        return None

    def drain_events(self, session_id: str) -> list[dict[str, object]]:
        self.drain_calls += 1
        return self._drain_script.pop(0) if self._drain_script else []


class _FakeEventStore:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self._sessions: dict[str, SimpleNamespace] = {}

    def get_session(self, session_id: str):  # noqa: ANN201
        return self._sessions.get(session_id)

    def start_session(self, session_id: str, **_kwargs: Any) -> None:
        self._sessions[session_id] = SimpleNamespace(turn_count=0)

    def append_turn(self, event: Any) -> str:
        self.events.append(event)
        self._sessions[event.session_id].turn_count += 1
        return f"evt-{len(self.events)}"


class _CountingDispatcher:
    def __init__(self) -> None:
        self.wakes = 0

    def wake(self) -> None:
        self.wakes += 1


class _RecordingRetriever(KnowledgeRetriever):
    def __init__(self, config: Any, graph_store: GraphStore) -> None:
        super().__init__(config=config, graph_store=graph_store)
        self.queries: list[str] = []

    async def retrieve(self, query: str, user_id: str = "User", **_kwargs: Any) -> RetrievalResult:
        self.queries.append(query)
        return RetrievalResult(
            query=query,
            user_id=user_id,
            facts=[],
            entities_found=0,
            total_facts=0,
            formatted_context="",
            retrieval_time_ms=0.0,
            vector_search_time_ms=0.0,
            graph_traversal_time_ms=0.0,
            config_used={},
        )


class _FakeExtractionPipeline:
    async def extract_from_utterance(self, **_kwargs: Any):  # noqa: ANN201
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


def _make_handler(
    passes: Sequence[ScriptedPass] = (),
    *,
    port: _FakePort | None = None,
    dispatcher: _CountingDispatcher | None = None,
) -> tuple[ConversationHandler, FakeStreamingLLMProvider, _FakeEventStore]:
    config = build_test_config()
    graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
    provider = FakeStreamingLLMProvider(list(passes))
    handler = ConversationHandler(
        config=config,
        graph_store=graph_store,
        extraction_pipeline=_FakeExtractionPipeline(),
        retriever=_RecordingRetriever(config, graph_store),
        llm_provider=provider,
        conventions_loader=make_test_conventions_loader(),
    )
    store = _FakeEventStore()
    handler.event_store = store  # type: ignore[assignment]
    if dispatcher is not None:
        handler._extraction_dispatcher = dispatcher  # type: ignore[assignment]
    if port is not None:
        handler.attach_capability_host(port)
    return handler, provider, store


def _record(
    *,
    tool: str = "timers.start",
    capability: str = "timers",
    status: Literal[
        "ok", "invalid_args", "pending_confirmation", "unavailable", "error", "denied"
    ] = "ok",
    summary: str = "Timer set for 10 minutes.",
    reason: str | None = None,
    data: dict[str, object] | None = None,
    tool_call_id: str = "call-1",
) -> ActionRecord:
    return ActionRecord(
        tool_call_id=tool_call_id,
        tool=tool,
        capability=capability,
        risk="reversible",
        args_summary="minutes=10",
        outcome=ToolOutcome(status=status, summary=summary, data=data, reason=reason),
        duration_ms=12,
        power_level="full",
        ack="Setting a timer.",
        view="timers",
        source="fastpath",
    )


def _handled(
    reply: str,
    *records: ActionRecord,
    remainder: str | None = None,
) -> PreTurnResolution:
    return PreTurnResolution(
        handled=True, reply=reply, remainder=remainder, records=records, events=()
    )


async def _collect(
    handler: ConversationHandler, message: str, session_id: str = "s1"
) -> list[StreamEvent]:
    return [
        event
        async for event in handler.handle_message_streaming(
            user_message=message, session_id=session_id
        )
    ]


def _payloads(events: list[StreamEvent]) -> list[dict[str, Any]]:
    return [event.payload for event in events if isinstance(event, WSEvent)]


def _tokens(events: list[StreamEvent]) -> list[Token]:
    return [event for event in events if isinstance(event, Token)]


def _complete(events: list[StreamEvent]) -> Complete:
    assert isinstance(events[-1], Complete)
    return events[-1]


def _history(handler: ConversationHandler, session_id: str = "s1") -> list[tuple[str, str]]:
    return [(m.role, m.content) for m in handler.sessions[session_id].messages]


def _static_message(request: Any) -> str:
    """The static system template, found by the section heading it always carries."""
    templates = [
        m["content"]
        for m in request.messages
        if m["role"] == "system" and "=== GUIDELINES ===" in m["content"]
    ]
    assert len(templates) == 1
    return templates[0]


# --- No host attached ---------------------------------------------------------


class TestWithoutHost:
    @pytest.mark.asyncio
    async def test_turn_keeps_todays_tools_prompt_and_stream(self):
        handler, provider, _store = _make_handler([ScriptedPass(chunks=["Hello there."])])

        events = await _collect(handler, "hi")

        assert provider.calls[0].tools == KNOWLEDGE_TOOL_SCHEMAS
        assert all(_SENTENCE not in str(m["content"]) for m in provider.calls[0].messages)
        assert [type(e) for e in events] == [Token, Complete]
        assert _complete(events).final_response == "Hello there."

    @pytest.mark.asyncio
    async def test_build_messages_error_gives_the_recorded_error_turn(self, monkeypatch):
        handler, provider, store = _make_handler()

        def _raise(*_args: Any, **_kwargs: Any) -> list[dict[str, str]]:
            raise RuntimeError(f"prompt assembly failed near {_SECRET}")

        monkeypatch.setattr(handler, "_build_messages", _raise)

        events = await _collect(handler, "hi")

        expected = f"I encountered an error: prompt assembly failed near {_SECRET}"
        assert [t.text for t in _tokens(events)] == [expected]
        assert _complete(events).final_response == expected
        assert _history(handler) == [("user", "hi"), ("assistant", expected)]
        assert [e.system_response for e in store.events] == [expected]
        assert provider.calls == []


# --- Host attached, turn not handled ------------------------------------------


class TestStaticSentence:
    @pytest.mark.asyncio
    async def test_static_template_ends_with_the_sentence_and_is_identical_across_turns(self):
        port = _FakePort([None, None])
        handler, provider, _store = _make_handler(
            [ScriptedPass(chunks=["One."]), ScriptedPass(chunks=["Two."])], port=port
        )

        await _collect(handler, "hi")
        await _collect(handler, "hello again")

        first, second = (_static_message(request) for request in provider.calls)
        assert first.endswith(_SENTENCE)
        assert first == second


class TestUnhandledTurn:
    @pytest.mark.asyncio
    async def test_none_from_pre_llm_continues_as_todays_turn(self):
        port = _FakePort([None])
        handler, provider, _store = _make_handler(
            [ScriptedPass(chunks=["Model reply."])], port=port
        )

        events = await _collect(handler, "tell me a story")

        assert len(provider.calls) == 1
        assert provider.calls[0].tools == KNOWLEDGE_TOOL_SCHEMAS
        assert _complete(events).final_response == "Model reply."
        assert [c["utterance"] for c in port.pre_llm_calls] == ["tell me a story"]

    @pytest.mark.asyncio
    async def test_pre_llm_gets_the_monotonic_time_the_message_was_received(self):
        port = _FakePort([None])
        handler, _provider, _store = _make_handler([ScriptedPass(chunks=["Ok."])], port=port)

        before = time.monotonic()
        await _collect(handler, "hi")
        after = time.monotonic()

        received_at = port.pre_llm_calls[0]["received_at"]
        assert received_at is not None
        assert before <= received_at <= after

    @pytest.mark.asyncio
    async def test_payloads_drained_after_an_unhandled_pre_llm_are_yielded(self):
        superseded = {"type": "action_resolved", "proposal_id": "p1", "resolution": "superseded"}
        port = _FakePort([None], drain_script=[[superseded]])
        handler, _provider, _store = _make_handler([ScriptedPass(chunks=["Ok."])], port=port)

        events = await _collect(handler, "something else")

        assert _payloads(events) == [superseded]


# --- Host attached, turn handled ----------------------------------------------


class TestHandledTurn:
    @pytest.mark.asyncio
    async def test_fast_path_reply_streams_with_no_llm_call(self):
        port = _FakePort([_handled("Timer set for 10 minutes.", _record())])
        handler, provider, _store = _make_handler(port=port)

        events = await _collect(handler, "set a timer for 10 minutes")

        assert provider.calls == []
        assert [t.text for t in _tokens(events)] == ["Timer set for 10 minutes."]
        assert _complete(events).final_response == "Timer set for 10 minutes."

    @pytest.mark.asyncio
    async def test_reply_is_streamed_as_one_token_per_sentence(self):
        port = _FakePort([_handled("Alarm stopped. Have a good morning.", _record())])
        handler, _provider, _store = _make_handler(port=port)

        events = await _collect(handler, "stop")

        tokens = _tokens(events)
        assert [(t.text, t.pass_num) for t in tokens] == [
            ("Alarm stopped.", 1),
            (" Have a good morning.", 1),
        ]
        assert "".join(t.text for t in tokens) == _complete(events).final_response

    @pytest.mark.asyncio
    async def test_retrieval_and_prompt_assembly_are_skipped(self, monkeypatch):
        port = _FakePort([_handled("Done.", _record())])
        handler, _provider, _store = _make_handler(port=port)
        built: list[Any] = []
        monkeypatch.setattr(handler, "_build_messages", lambda *a, **k: built.append(a))

        await _collect(handler, "set a timer for 10 minutes")

        assert handler.retriever.queries == []  # type: ignore[attr-defined]
        assert built == []

    @pytest.mark.asyncio
    async def test_fast_path_timer_events_carry_the_capability_fields(self):
        record = _record(data={"timer_id": "t-1", "seconds": 600})
        port = _FakePort([_handled("Timer set for 10 minutes.", record)])
        handler, _provider, _store = _make_handler(port=port)

        events = await _collect(handler, "set a timer for 10 minutes")

        started, completed = _payloads(events)
        assert started == {
            "type": "tool_call_started",
            "tool_call_id": "call-1",
            "name": "timers.start",
            "args_summary": "minutes=10",
            "capability": "timers",
            "risk": "reversible",
            "ack": "Setting a timer.",
            "view": "timers",
        }
        assert completed == {
            "type": "tool_call_completed",
            "tool_call_id": "call-1",
            "name": "timers.start",
            "duration_ms": 12,
            "result_summary": "Timer set for 10 minutes.",
            "error": None,
            "capability": "timers",
            "status": "ok",
            "source": "fastpath",
            "power_level": "full",
            "view": "timers",
            "data": {"timer_id": "t-1", "seconds": 600},
        }

    @pytest.mark.parametrize(
        ("status", "reason", "error"),
        [
            pytest.param("pending_confirmation", None, None, id="pending-is-not-an-error"),
            pytest.param("unavailable", "unreachable", "unavailable: unreachable", id="reason"),
            pytest.param("invalid_args", None, "invalid_args: Unknown light.", id="summary"),
        ],
    )
    @pytest.mark.asyncio
    async def test_completed_event_sets_error_for_statuses_other_than_ok_and_pending(
        self, status, reason, error
    ):
        record = _record(status=status, summary="Unknown light.", reason=reason)
        port = _FakePort([_handled("Unknown light.", record)])
        handler, _provider, _store = _make_handler(port=port)

        events = await _collect(handler, "turn on the moon")

        completed = _payloads(events)[1]
        assert completed["status"] == status
        assert completed["error"] == error

    @pytest.mark.asyncio
    async def test_complete_counts_one_tool_call_per_record(self):
        records = (_record(tool_call_id="a"), _record(tool_call_id="b"))
        port = _FakePort([_handled("Both done.", *records)])
        handler, _provider, _store = _make_handler(port=port)

        events = await _collect(handler, "yes")

        assert _complete(events).tool_calls_used == 2
        types = [p["type"] for p in _payloads(events)]
        assert types == ["tool_call_started", "tool_call_completed"] * 2

    @pytest.mark.asyncio
    async def test_turn_is_recorded_once_with_no_extraction(self):
        dispatcher = _CountingDispatcher()
        port = _FakePort([_handled("Timer set.", _record())])
        handler, _provider, store = _make_handler(port=port, dispatcher=dispatcher)

        await _collect(handler, "set a timer for 10 minutes")

        assert _history(handler) == [
            ("user", "set a timer for 10 minutes"),
            ("assistant", "Timer set."),
        ]
        assert [(e.user_utterance, e.system_response) for e in store.events] == [
            ("set a timer for 10 minutes", "Timer set.")
        ]
        assert dispatcher.wakes == 0
        assert handler._extraction_tasks == {}

    @pytest.mark.asyncio
    async def test_no_in_process_extraction_without_a_dispatcher(self):
        port = _FakePort([_handled("Timer set.", _record())])
        handler, _provider, store = _make_handler(port=port)

        await _collect(handler, "set a timer for 10 minutes")

        assert len(store.events) == 1
        assert handler._extraction_tasks == {}

    @pytest.mark.asyncio
    async def test_trace_is_built_and_reset_with_nothing_added(self):
        port = _FakePort([_handled("Timer set.", _record())])
        handler, _provider, store = _make_handler(port=port)

        await _collect(handler, "set a timer for 10 minutes")

        assert store.events[0].retrieval_context == {"trace": TurnTraceBuilder().build()}
        assert handler._turn_trace is None

    @pytest.mark.asyncio
    async def test_drained_payloads_are_yielded_on_each_pass(self):
        first = {"type": "action_resolved", "proposal_id": "p1", "resolution": "approved"}
        second = {"type": "action_proposed", "proposal_id": "p2"}
        port = _FakePort(
            [_handled("Approved.", remainder="dim them"), _handled("Should I dim them?")],
            drain_script=[[first], [second]],
        )
        handler, _provider, _store = _make_handler(port=port)

        events = await _collect(handler, "yes, and dim them")

        assert _payloads(events) == [first, second]
        first_at = events.index(WSEvent(payload=first))
        second_at = events.index(WSEvent(payload=second))
        reply_at = [i for i, e in enumerate(events) if isinstance(e, Token)]
        assert first_at < reply_at[0] < second_at < reply_at[1]


class TestRemainder:
    @pytest.mark.asyncio
    async def test_handled_remainder_yields_its_reply_after_the_first(self):
        port = _FakePort(
            [_handled("Approved.", remainder="turn off the alarm"), _handled("Alarm stopped.")]
        )
        handler, provider, _store = _make_handler(port=port)

        events = await _collect(handler, "yes, and turn off the alarm")

        assert [c["utterance"] for c in port.pre_llm_calls] == [
            "yes, and turn off the alarm",
            "turn off the alarm",
        ]
        assert [t.text for t in _tokens(events)] == ["Approved.", " Alarm stopped."]
        assert provider.calls == []
        assert _history(handler) == [
            ("user", "yes, and turn off the alarm"),
            ("assistant", "Approved. Alarm stopped."),
        ]

    @pytest.mark.asyncio
    async def test_both_passes_share_the_turn_id_and_received_at(self):
        port = _FakePort([_handled("Approved.", remainder="dim them"), _handled("Dimmed.")])
        handler, _provider, _store = _make_handler(port=port)

        await _collect(handler, "yes, and dim them")

        first, second = port.pre_llm_calls
        assert first["turn_id"] == second["turn_id"]
        assert first["received_at"] == second["received_at"]

    @pytest.mark.asyncio
    async def test_remainder_on_the_second_resolution_is_not_re_entered(self):
        port = _FakePort(
            [
                _handled("Approved.", remainder="dim them"),
                _handled("Dimmed.", remainder="and more"),
                _handled("Never asked."),
            ]
        )
        handler, provider, _store = _make_handler(port=port)

        events = await _collect(handler, "yes, and dim them")

        assert len(port.pre_llm_calls) == 2
        assert provider.calls == []
        assert _complete(events).final_response == "Approved. Dimmed."

    @pytest.mark.asyncio
    async def test_unhandled_remainder_continues_to_the_model(self):
        port = _FakePort([_handled("Approved.", _record(), remainder="tell me a joke"), None])
        handler, provider, _store = _make_handler(
            [ScriptedPass(chunks=["Here is one."])], port=port
        )

        events = await _collect(handler, "yes, and tell me a joke")

        assert len(provider.calls) == 1
        assert [t.text for t in _tokens(events)] == ["Approved.", " Here is one."]
        assert _history(handler) == [
            ("user", "yes, and tell me a joke"),
            ("assistant", "Approved. Here is one."),
        ]
        assert _complete(events).tool_calls_used == 1


# --- Who answered the recorded turn -------------------------------------------


class TestRecordedAnsweredBy:
    """`answered_by` on the recorded turn: `"capability"` only when no model reply ran."""

    @pytest.mark.asyncio
    async def test_a_handled_turn_records_capability(self):
        port = _FakePort([_handled("Timer set.", _record())])
        handler, _provider, store = _make_handler(port=port)

        await _collect(handler, "set a timer for 10 minutes")

        assert [e.answered_by for e in store.events] == ["capability"]

    @pytest.mark.asyncio
    async def test_a_handled_remainder_turn_records_capability(self):
        port = _FakePort([_handled("Approved.", remainder="dim them"), _handled("Dimmed.")])
        handler, provider, store = _make_handler(port=port)

        await _collect(handler, "yes, and dim them")

        assert provider.calls == []
        assert [e.answered_by for e in store.events] == ["capability"]

    @pytest.mark.asyncio
    async def test_a_model_turn_records_model(self):
        handler, _provider, store = _make_handler([ScriptedPass(chunks=["Hello there."])])

        await _collect(handler, "hi")

        assert [e.answered_by for e in store.events] == ["model"]

    @pytest.mark.asyncio
    async def test_an_unhandled_turn_with_a_host_records_model(self):
        port = _FakePort([None])
        handler, _provider, store = _make_handler([ScriptedPass(chunks=["Hi."])], port=port)

        await _collect(handler, "hi")

        assert [e.answered_by for e in store.events] == ["model"]

    @pytest.mark.asyncio
    async def test_a_handled_proposal_whose_remainder_runs_the_model_records_model(self):
        port = _FakePort([_handled("Approved.", _record(), remainder="tell me a joke"), None])
        handler, provider, store = _make_handler([ScriptedPass(chunks=["Here is one."])], port=port)

        await _collect(handler, "yes, and tell me a joke")

        assert len(provider.calls) == 1
        assert [e.answered_by for e in store.events] == ["model"]

    @pytest.mark.asyncio
    async def test_the_error_turn_records_model(self):
        port = _FakePort([RuntimeError("hook failed")])
        handler, _provider, store = _make_handler(port=port)

        await _collect(handler, "stop")

        assert [e.system_response for e in store.events] == ["I encountered an error: hook failed"]
        assert [e.answered_by for e in store.events] == ["model"]

    def test_a_direct_call_without_the_keyword_records_model(self):
        handler, _provider, store = _make_handler()

        event_id, _recorded_at = handler._record_turn_event(
            session_id="s1", user_message="hello", assistant_message="hi"
        )

        assert event_id == "evt-1"
        assert [e.answered_by for e in store.events] == ["model"]


# --- Error backstop -----------------------------------------------------------


class TestErrorBackstop:
    @pytest.mark.asyncio
    async def test_pre_llm_error_gives_a_redacted_error_turn(self):
        port = _FakePort([RuntimeError(f"hook failed with {_SECRET}")])
        handler, provider, store = _make_handler(port=port)
        orphan = {"type": "tool_call_started", "tool_call_id": "orphan", "name": "x"}
        port.before_raise.append(lambda: handler._turn_ws_events.append(orphan))

        events = await _collect(handler, "stop")

        expected = "I encountered an error: hook failed with ***"
        assert [t.text for t in _tokens(events)] == [expected]
        assert _payloads(events) == [orphan]
        assert _complete(events).final_response == expected
        assert _history(handler) == [("user", "stop"), ("assistant", expected)]
        assert [e.system_response for e in store.events] == [expected]
        assert provider.calls == []

    @pytest.mark.asyncio
    async def test_provider_error_text_is_redacted_with_a_host(self):
        from backend.errors import LLMConnectionError

        port = _FakePort([None])
        handler, _provider, _store = _make_handler(
            [ScriptedPass(error=LLMConnectionError(f"bad key {_SECRET}"))], port=port
        )

        events = await _collect(handler, "hi")

        assert _complete(events).final_response == "I encountered an error: bad key ***"


# --- decide_proposal ----------------------------------------------------------


def _decision(reply: str = "Bedroom lights off.") -> PreTurnResolution:
    record = _record(
        tool="lights.power",
        capability="lights",
        summary=reply,
        tool_call_id="call-p",
    )
    resolved = {"type": "action_resolved", "proposal_id": "p1", "resolution": "approved"}
    return PreTurnResolution(
        handled=True, reply=reply, remainder=None, records=(record,), events=(resolved,)
    )


class TestDecideProposal:
    @pytest.mark.asyncio
    async def test_returns_the_tool_events_and_the_resolution_payloads(self):
        port = _FakePort(decide_result=_decision())
        handler, _provider, _store = _make_handler(port=port)

        payloads = await handler.decide_proposal("s1", "p1", "approve")

        assert port.decide_calls == [("s1", "p1", "approve")]
        assert [p["type"] for p in payloads] == [
            "tool_call_started",
            "tool_call_completed",
            "action_resolved",
        ]
        assert payloads[1]["capability"] == "lights"

    @pytest.mark.asyncio
    async def test_decision_lands_in_history_before_the_next_user_message(self):
        port = _FakePort([None, None], decide_result=_decision())
        handler, provider, _store = _make_handler(
            [ScriptedPass(chunks=["Should I turn them off?"]), ScriptedPass(chunks=["Sure."])],
            port=port,
        )
        await _collect(handler, "turn off the bedroom lights")

        await handler.decide_proposal("s1", "p1", "approve")
        await _collect(handler, "thanks")

        assert _history(handler) == [
            ("user", "turn off the bedroom lights"),
            ("assistant", "Should I turn them off?"),
            ("user", "(Approved on screen.)"),
            ("assistant", "Bedroom lights off."),
            ("user", "thanks"),
            ("assistant", "Sure."),
        ]
        sent = [(m["role"], m["content"]) for m in provider.calls[1].messages][-3:]
        assert sent == [
            ("user", "(Approved on screen.)"),
            ("assistant", "Bedroom lights off."),
            ("user", "thanks"),
        ]

    @pytest.mark.asyncio
    async def test_decision_during_a_paused_turn_does_not_interleave_its_history(self):
        gate = asyncio.Event()
        port = _FakePort([None, None], decide_result=_decision())
        handler, _provider, _store = _make_handler(
            [
                ScriptedPass(chunks=["First sentence here. ", gate, "Second sentence here."]),
                ScriptedPass(chunks=["Ok."]),
            ],
            port=port,
        )
        stream = handler.handle_message_streaming(
            user_message="tell me two things", session_id="s1"
        )
        event = await anext(stream)
        while not isinstance(event, Token):
            event = await anext(stream)

        await handler.decide_proposal("s1", "p1", "approve")
        gate.set()
        rest = [e async for e in stream]

        assert isinstance(rest[-1], Complete)
        assert _history(handler) == [
            ("user", "tell me two things"),
            ("assistant", "First sentence here. Second sentence here."),
        ]
        await _collect(handler, "and then")
        assert _history(handler)[2:5] == [
            ("user", "(Approved on screen.)"),
            ("assistant", "Bedroom lights off."),
            ("user", "and then"),
        ]

    @pytest.mark.asyncio
    async def test_unresolved_decision_sends_nothing_and_queues_nothing(self):
        unresolved = PreTurnResolution(
            handled=False, reply="Nothing is waiting.", remainder=None, records=(), events=()
        )
        port = _FakePort([None], decide_result=unresolved)
        handler, _provider, _store = _make_handler([ScriptedPass(chunks=["Hi."])], port=port)

        payloads = await handler.decide_proposal("s1", "gone", "reject")
        await _collect(handler, "hello")

        assert payloads == []
        assert _history(handler) == [("user", "hello"), ("assistant", "Hi.")]
