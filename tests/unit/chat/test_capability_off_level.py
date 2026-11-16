"""Capability commands answered while gaming mode holds the GPU (the off level).

`ConversationHandler.handle_capability_only` runs only the host's pre-LLM step (a ringing item,
a proposal decision, the fast path). A message it answers is recorded as a turn answered by a
capability, with no extraction. A message nothing answers comes back unhandled, records nothing
and leaves the session untouched, so the server can refuse it as before; it still carries the
payloads the pre-LLM step drained, such as a superseded proposal's `action_resolved`. None
means only that no host is attached. No path here may reach the provider, so every handler is
built over a provider that fails when called.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from backend.capabilities.action_log import ActionLog
from backend.capabilities.contracts import (
    ActionRecord,
    DecodeReply,
    FastIntent,
    PreTurnResolution,
    ToolOutcome,
)
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import ActionDecoder
from backend.capabilities.host import CapabilityHost, HostSettings
from backend.capabilities.intents import normalize_utterance
from backend.capabilities.policy import PolicyTable
from backend.capabilities.power import FixedTiers
from backend.capabilities.router import RouterConfig, ToolRouter
from backend.capabilities.secrets import SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.chat.conversation_handler import CapabilityOnlyTurn, ConversationHandler
from backend.chat.stream_events import Complete, StreamEvent, Token, WSEvent
from backend.knowledge.models import RetrievalResult
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm.models import LLMRequest
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider
from tests.unit.capabilities.fakes import (
    FakeCapability,
    FakeClock,
    RecordingEventSink,
    ScriptedDecode,
    make_tool_spec,
)
from tests.unit.conftest import make_test_conventions_loader

_TIMER = "set a timer for 10 minutes"
_STORY = "tell me a story"
_TIMER_SUMMARY = "Timer set for 10 minutes."

# --- Fakes --------------------------------------------------------------------


class _FailingProvider(FakeStreamingLLMProvider):
    """A provider that fails the test the moment anything asks it for a reply."""

    def __init__(self) -> None:
        super().__init__([])

    def generate(self, request: LLMRequest, *, stream: bool = False) -> Any:  # type: ignore[override]
        self.calls.append(request)
        raise AssertionError("the provider was called at the off level")


class _FailingRetriever(KnowledgeRetriever):
    """Retrieval belongs to the model path, so the off level must never reach it."""

    async def retrieve(self, query: str, user_id: str = "User", **_kwargs: Any) -> RetrievalResult:
        raise AssertionError("retrieval ran at the off level")


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
    """The T2a dispatcher: a recorded turn it should extract wakes it."""

    def __init__(self) -> None:
        self.wakes = 0

    def wake(self) -> None:
        self.wakes += 1


class _FakeExtractionPipeline:
    async def extract_from_utterance(self, **_kwargs: Any):  # noqa: ANN201
        raise AssertionError("extraction ran for a capability-only turn")


class _ScriptedPort:
    """A `CapabilityTurnPort` whose `pre_llm` answers from a script, None past its end.

    Only the pre-LLM step and the per-turn bookkeeping may run at the off level, so routing,
    the action decode and the screen decision fail the test when called.
    """

    def __init__(
        self,
        script: Sequence[PreTurnResolution | None] = (),
        *,
        drain_script: Sequence[list[dict[str, object]]] = (),
    ) -> None:
        self._script = list(script)
        self._drain_script = list(drain_script)
        self.pre_llm_calls: list[str] = []
        self.record_turn_calls: list[tuple[str, list[str]]] = []

    async def pre_llm(
        self, session_id: str, turn_id: str, utterance: str, *, received_at: float | None = None
    ) -> PreTurnResolution | None:
        self.pre_llm_calls.append(utterance)
        return self._script.pop(0) if self._script else None

    def plan_turn(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        raise AssertionError("routing ran at the off level")

    async def run_actions(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        raise AssertionError("the action decode ran at the off level")

    async def decide(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        raise AssertionError("no screen decision is part of this turn")

    def static_sentence(self) -> str:
        return "Actions run only through the action system."

    def redact(self, text: str) -> str:
        return text

    def record_turn(self, session_id: str, used: Any) -> None:
        self.record_turn_calls.append((session_id, list(used)))

    def drain_events(self, session_id: str) -> list[dict[str, object]]:
        return self._drain_script.pop(0) if self._drain_script else []


def _timers() -> FakeCapability:
    """A timers capability whose fast path knows `_TIMER`."""
    timers = FakeCapability(
        "timers",
        tools=(make_tool_spec("timers.start"),),
        keywords=("timer",),
        intents={
            normalize_utterance(_TIMER, HostSettings().wake_words): FastIntent(
                tool="timers.start", args={}
            )
        },
    )
    timers.outcomes.append(ToolOutcome(status="ok", summary=_TIMER_SUMMARY))
    return timers


def _real_host(capability: FakeCapability) -> CapabilityHost:
    counter = ApproximateTokenCounter()
    router = ToolRouter(
        [capability], RouterConfig(token_budget=1200, reserved_tokens=100), counter, None
    )
    return CapabilityHost(
        [capability],
        router=router,
        counter=counter,
        policy=PolicyTable.from_specs(capability.tools(), None),
        decoder=ActionDecoder(),
        secrets=SecretRegistry(),
        action_log=ActionLog(CapabilitiesDb(":memory:")),
        event_sink=RecordingEventSink(),
        clock=FakeClock(),
        settings=HostSettings(),
    )


def _make_handler(
    port: Any | None,
    *,
    dispatcher: _CountingDispatcher | None = None,
) -> tuple[ConversationHandler, _FailingProvider, _FakeEventStore]:
    """Build a handler over a failing provider, with tiers served by that provider too."""
    config = build_test_config()
    graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
    provider = _FailingProvider()
    handler = ConversationHandler(
        config=config,
        graph_store=graph_store,
        extraction_pipeline=_FakeExtractionPipeline(),
        retriever=_FailingRetriever(config=config, graph_store=graph_store),
        llm_provider=provider,
        conventions_loader=make_test_conventions_loader(),
    )
    store = _FakeEventStore()
    handler.event_store = store  # type: ignore[assignment]
    if dispatcher is not None:
        handler._extraction_dispatcher = dispatcher  # type: ignore[assignment]
    if port is not None:
        handler.attach_capability_host(port, tiers=FixedTiers({"interaction": provider}))
    return handler, provider, store


def _record(*, summary: str = "Turned the lights on.") -> ActionRecord:
    return ActionRecord(
        tool_call_id="call-1",
        tool="lights.set",
        capability="lights",
        risk="reversible",
        args_summary="on",
        outcome=ToolOutcome(status="ok", summary=summary),
        duration_ms=5,
        power_level="off",
        source="fastpath",
    )


def _handled(reply: str, *records: ActionRecord, remainder: str | None = None) -> PreTurnResolution:
    return PreTurnResolution(
        handled=True, reply=reply, remainder=remainder, records=records, events=()
    )


def _payloads(events: Sequence[StreamEvent]) -> list[dict[str, Any]]:
    return [event.payload for event in events if isinstance(event, WSEvent)]


def _text(events: Sequence[StreamEvent]) -> str:
    return "".join(event.text for event in events if isinstance(event, Token))


def _complete(events: Sequence[StreamEvent]) -> Complete:
    assert isinstance(events[-1], Complete)
    assert sum(isinstance(event, Complete) for event in events) == 1
    return events[-1]


def _history(handler: ConversationHandler, session_id: str = "s1") -> list[tuple[str, str]]:
    return [(m.role, m.content) for m in handler.sessions[session_id].messages]


def _answered(turn: CapabilityOnlyTurn | None) -> tuple[StreamEvent, ...]:
    """The events of a handled turn; fails the test when the message was not handled."""
    assert turn is not None
    assert turn.handled is True
    return turn.events


# --- The fast path ------------------------------------------------------------


class TestFastPathAtOff:
    @pytest.mark.asyncio
    async def test_a_timer_is_answered_with_no_llm_call(self):
        timers = _timers()
        host = _real_host(timers)
        handler, provider, _store = _make_handler(host)

        try:
            events = _answered(await handler.handle_capability_only(_TIMER, "s1"))
        finally:
            await host.stop()

        assert provider.calls == []
        assert [tool for tool, _args, _ctx in timers.invocations] == ["timers.start"]
        assert [(p["type"], p["name"]) for p in _payloads(events)] == [
            ("tool_call_started", "timers.start"),
            ("tool_call_completed", "timers.start"),
        ]
        reply = _text(events)
        assert _TIMER_SUMMARY in reply
        complete = _complete(events)
        assert complete.final_response == reply
        assert complete.tool_calls_used == 1
        assert _history(handler) == [("user", _TIMER), ("assistant", reply)]

    @pytest.mark.asyncio
    async def test_a_story_is_not_handled_and_leaves_no_trace(self):
        host = _real_host(_timers())
        handler, provider, store = _make_handler(host)

        try:
            turn = await handler.handle_capability_only(_STORY, "s1")
        finally:
            await host.stop()

        assert turn is not None
        assert turn.handled is False
        assert turn.events == ()
        assert provider.calls == []
        assert store.events == []
        assert "s1" not in handler.sessions

    @pytest.mark.asyncio
    async def test_with_no_host_nothing_is_handled(self):
        handler, provider, store = _make_handler(None)

        assert await handler.handle_capability_only(_TIMER, "s1") is None
        assert provider.calls == []
        assert store.events == []


# --- Ringing items --------------------------------------------------------------


class TestRingingAtOff:
    @pytest.mark.asyncio
    async def test_stop_answers_a_ringing_alarm(self):
        alarms = FakeCapability("alarms", tools=(make_tool_spec("alarms.set"),))
        alarms.ringing = True
        alarms.active_replies[normalize_utterance("stop", HostSettings().wake_words)] = ToolOutcome(
            status="ok", summary="Alarm stopped."
        )
        host = _real_host(alarms)
        handler, provider, store = _make_handler(host)

        try:
            events = _answered(await handler.handle_capability_only("stop", "s1"))
        finally:
            await host.stop()

        assert provider.calls == []
        assert "Alarm stopped." in _text(events)
        assert [e.answered_by for e in store.events] == ["capability"]


# --- Recording ----------------------------------------------------------------


class TestRecordingAtOff:
    @pytest.mark.asyncio
    async def test_the_turn_is_recorded_as_answered_by_a_capability_and_sends_no_job(self):
        dispatcher = _CountingDispatcher()
        host = _real_host(_timers())
        handler, _provider, store = _make_handler(host, dispatcher=dispatcher)

        try:
            events = _answered(await handler.handle_capability_only(_TIMER, "s1"))
        finally:
            await host.stop()

        assert [(e.user_utterance, e.system_response, e.answered_by) for e in store.events] == [
            (_TIMER, _complete(events).final_response, "capability")
        ]
        assert dispatcher.wakes == 0
        assert handler._extraction_tasks == {}

    @pytest.mark.asyncio
    async def test_the_turn_is_recorded_as_the_pre_llm_branch_records_it(self):
        # The same resolution answered on the normal path and at off gives the same row.
        reply = "Turned the lights on."
        normal_port = _ScriptedPort([_handled(reply, _record())])
        off_port = _ScriptedPort([_handled(reply, _record())])
        normal, _p1, normal_store = _make_handler(normal_port)
        off, _p2, off_store = _make_handler(off_port)

        async for _event in normal.handle_message_streaming(
            user_message="turn the lights on", session_id="s1"
        ):
            pass
        await off.handle_capability_only("turn the lights on", "s1")

        def _row(event: Any) -> tuple[Any, ...]:
            return (
                event.session_id,
                event.turn_index,
                event.user_utterance,
                event.system_response,
                event.context_window,
                event.retrieval_context,
                event.tool_calls,
                event.answered_by,
            )

        assert [_row(e) for e in off_store.events] == [_row(e) for e in normal_store.events]
        assert off_port.record_turn_calls == normal_port.record_turn_calls == [("s1", ["lights"])]

    @pytest.mark.asyncio
    async def test_an_unhandled_message_does_not_count_as_a_turn(self):
        port = _ScriptedPort([None])
        handler, _provider, store = _make_handler(port)

        turn = await handler.handle_capability_only(_STORY, "s1")

        assert turn is not None
        assert turn.handled is False
        assert port.record_turn_calls == []
        assert store.events == []


# --- Proposal decisions ---------------------------------------------------------


class TestDecisionsAtOff:
    @pytest.mark.asyncio
    async def test_a_yes_to_a_live_proposal_is_answered(self):
        port = _ScriptedPort([_handled("Turned the lights on.", _record())])
        handler, provider, store = _make_handler(port)

        events = _answered(await handler.handle_capability_only("yes", "s1"))

        assert provider.calls == []
        assert _text(events) == "Turned the lights on."
        assert [e.answered_by for e in store.events] == ["capability"]

    @pytest.mark.asyncio
    async def test_a_yes_with_a_remainder_nothing_handles_answers_the_yes_alone(self):
        # The yes ran its proposal, so the turn is handled; the remainder would need the
        # model, which the off level never calls.
        port = _ScriptedPort([_handled("Turned the lights on.", _record(), remainder=_STORY), None])
        handler, provider, store = _make_handler(port)

        events = _answered(await handler.handle_capability_only(f"yes, and {_STORY}", "s1"))

        assert provider.calls == []
        assert port.pre_llm_calls == [f"yes, and {_STORY}", _STORY]
        assert _complete(events).final_response == "Turned the lights on."
        assert [(e.user_utterance, e.answered_by) for e in store.events] == [
            (f"yes, and {_STORY}", "capability")
        ]

    @pytest.mark.asyncio
    async def test_screen_decisions_enter_history_before_the_message(self):
        port = _ScriptedPort([_handled("Turned the lights on.", _record())])
        handler, _provider, _store = _make_handler(port)
        handler._capability_history["s1"] = [
            ("user", "(Rejected on screen.)"),
            ("assistant", "Left the heating alone."),
        ]

        await handler.handle_capability_only("yes", "s1")

        assert _history(handler) == [
            ("user", "(Rejected on screen.)"),
            ("assistant", "Left the heating alone."),
            ("user", "yes"),
            ("assistant", "Turned the lights on."),
        ]

    @pytest.mark.asyncio
    async def test_payloads_drained_after_the_resolution_reach_the_turn(self):
        resolved = {"type": "action_resolved", "proposal_id": "p1", "status": "approved"}
        port = _ScriptedPort(
            [_handled("Turned the lights on.", _record())], drain_script=[[resolved]]
        )
        handler, _provider, _store = _make_handler(port)

        events = _answered(await handler.handle_capability_only("yes", "s1"))

        assert resolved in _payloads(events)


# --- A live proposal superseded at off ------------------------------------------------


def _lock() -> FakeCapability:
    """A capability whose one tool always waits for a yes."""
    return FakeCapability(
        "lock", tools=(make_tool_spec("lock.open", default_policy="confirm"),), keywords=("lock",)
    )


async def _propose(host: CapabilityHost, session_id: str = "s1") -> str:
    """Make a proposal live through the model path, as the host tests do, and drain its event."""
    plan = host.plan_turn(session_id, "open the lock", [])
    reply = DecodeReply(
        content=json.dumps({"calls": [{"tool": "lock.open", "args": {}}], "also_reply": False}),
        finish_reason="stop",
    )
    result = await host.run_actions(
        session_id, "turn-0", "open the lock", plan, ScriptedDecode(reply)
    )
    host.drain_events(session_id)
    (proposal_id,) = {r.outcome.proposal_id for r in result.records if r.outcome.proposal_id}
    return proposal_id


class TestSupersededAtOff:
    @pytest.mark.asyncio
    async def test_an_unhandled_message_returns_the_superseded_resolution(self):
        lock = _lock()
        host = _real_host(lock)
        handler, provider, store = _make_handler(host)
        recorded: list[tuple[str, list[str]]] = []
        host.record_turn = lambda session_id, used: recorded.append(  # type: ignore[method-assign]
            (session_id, list(used))
        )

        try:
            proposal_id = await _propose(host)
            turn = await handler.handle_capability_only(_STORY, "s1")
        finally:
            await host.stop()

        assert turn is not None
        assert turn.handled is False
        assert len(turn.events) == 1
        (event,) = turn.events
        assert isinstance(event, WSEvent)
        assert (event.payload["type"], event.payload["proposal_id"]) == (
            "action_resolved",
            proposal_id,
        )
        assert event.payload["decision"] == "superseded"
        assert not any(isinstance(e, Token | Complete) for e in turn.events)
        assert lock.invocations == []
        assert provider.calls == []
        assert store.events == []
        assert "s1" not in handler.sessions
        assert recorded == []

    @pytest.mark.asyncio
    async def test_the_result_is_a_frozen_value(self):
        turn = CapabilityOnlyTurn(handled=False, events=())

        with pytest.raises(AttributeError):
            turn.handled = True  # type: ignore[misc]
