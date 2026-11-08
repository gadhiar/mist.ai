"""Capability actions chosen on the model path, through an attached `CapabilityTurnPort`.

When `pre_llm` does not answer a turn, the handler asks the host to route it (`plan_turn`),
then hands `run_actions` a decode callback that runs the constrained decode on the
`interaction` tier. The result either answers the turn from a template, with no main pass, or
gives today's passes an actions block and appends its notices. These tests drive whole turns
through `handle_message_streaming` with `FakeStreamingLLMProvider`, scripting the decode as one
of its passes, and a fake port that follows the real port's signatures
(`backend/capabilities/contracts.py`, `CapabilityTurnPort`).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from backend.capabilities.contracts import (
    ActionDecode,
    ActionRecord,
    ActionResult,
    DecodeFn,
    DecodeReply,
    PreTurnResolution,
    ToolOutcome,
    ToolSpec,
    TurnPlan,
)
from backend.capabilities.decoder import ActionDecoder
from backend.capabilities.power import FixedTiers, ModelTier
from backend.capabilities.router import RouteDecision
from backend.capabilities.schema import to_args_schema
from backend.chat.context_budget import ContextBudgetPlanner
from backend.chat.conversation_handler import KNOWLEDGE_TOOL_SCHEMAS, ConversationHandler
from backend.chat.stream_events import Complete, StreamEvent, Token, WSEvent
from backend.errors import ActionDecodeError, CapabilityUnavailableError
from backend.knowledge.models import RetrievalResult, RetrievedFact
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm.instrumented_provider import get_llm_call_context
from backend.llm.models import LLMRequest, LLMResponse, ThinkingConfig
from backend.llm.models import ToolCall as LLMToolCall
from backend.llm.provider import StreamingLLMProvider
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
from tests.unit.conftest import make_test_conventions_loader

_SECRET = "tok-quince-7741"
_SENTENCE = "Actions run only through the action system."
_NOTICE = "(I could not work out an action, so none ran.)"
_PROFILE = "=== USER PROFILE ===\nRaj takes his tea without sugar."
_SENTINEL = "SENTINEL-plum-4471 Raj keeps his spare key under the blue pot."

_DECODE = ActionDecode(
    messages=(
        {"role": "system", "content": "Choose actions."},
        {"role": "user", "content": "set a timer for ten minutes"},
    ),
    response_schema={
        "type": "object",
        "properties": {"calls": {"type": "array"}, "also_reply": {"type": "boolean"}},
        "required": ["calls", "also_reply"],
        "additionalProperties": False,
    },
    max_tokens=256,
    temperature=0.0,
)
_DECODE_JSON = '{"calls":[{"tool":"timers.start","args":{"minutes":10}}],"also_reply":false}'


def _decode_pass(content: str = _DECODE_JSON) -> ScriptedPass:
    """The decode's scripted pass, streamed in two pieces so the callback must join them."""
    half = len(content) // 2
    return ScriptedPass(chunks=[content[:half], content[half:]], finish_reason="stop")


# --- Fakes --------------------------------------------------------------------


def _route() -> RouteDecision:
    return RouteDecision(capabilities=("timers",), reasons={"timers": "keyword:timer"}, dropped={})


def _plan(decode: ActionDecode | None = _DECODE) -> TurnPlan:
    offered = frozenset({"timers.start"}) if decode is not None else frozenset()
    return TurnPlan(route=_route(), offered=offered, decode=decode)


def _result(
    *records: ActionRecord,
    also_reply: bool = True,
    reply: str | None = None,
    block: str = "",
    notices: Sequence[str] = (),
    events: Sequence[dict[str, object]] = (),
) -> ActionResult:
    return ActionResult(
        records=records,
        also_reply=also_reply,
        reply=reply,
        block=block,
        notices=tuple(notices),
        events=tuple(events),
    )


class _Port:
    """A `CapabilityTurnPort` for the model path.

    `pre_llm` answers from `pre_script` (None past its end). `plan_turn` returns `plan`, or
    calls it when it is callable, or raises it when it is an exception. `run_actions` calls the
    handler's decode callback when the plan has a decode, as the host does, and treats the
    failures the host turns into its decode-failure notice the same way; then it returns
    `result` or raises it.
    """

    def __init__(
        self,
        *,
        plan: TurnPlan | BaseException | Callable[..., TurnPlan] | None = None,
        result: ActionResult | BaseException | None = None,
        pre_script: Sequence[PreTurnResolution | None] = (),
        drain_script: Sequence[list[dict[str, object]]] = (),
        decode_wrapper: Callable[[Any], Any] | None = None,
    ) -> None:
        self._plan = plan
        self._result = result if result is not None else _result()
        self._pre_script = list(pre_script)
        self._drain_script = list(drain_script)
        self._decode_wrapper = decode_wrapper
        self.pre_llm_calls: list[dict[str, Any]] = []
        self.plan_calls: list[dict[str, Any]] = []
        self.run_calls: list[dict[str, Any]] = []
        self.decode_replies: list[DecodeReply] = []
        self.decode_errors: list[BaseException] = []
        self.record_turn_calls: list[tuple[str, list[str]]] = []
        self.before_raise: list[Callable[[], None]] = []

    async def pre_llm(
        self, session_id: str, turn_id: str, utterance: str, *, received_at: float | None = None
    ) -> PreTurnResolution | None:
        self.pre_llm_calls.append(
            {"turn_id": turn_id, "utterance": utterance, "received_at": received_at}
        )
        return self._pre_script.pop(0) if self._pre_script else None

    def plan_turn(
        self,
        session_id: str,
        utterance: str,
        recent: Sequence[tuple[str, str]],
        *,
        received_at: float | None = None,
    ) -> TurnPlan:
        self.plan_calls.append(
            {
                "session_id": session_id,
                "utterance": utterance,
                "recent": list(recent),
                "received_at": received_at,
            }
        )
        if self._plan is None:
            raise AssertionError("routing is not part of this turn")
        if isinstance(self._plan, BaseException):
            raise self._plan
        if callable(self._plan):
            return self._plan(session_id, utterance, recent)
        return self._plan

    async def run_actions(
        self, session_id: str, turn_id: str, utterance: str, plan: TurnPlan, decode: DecodeFn
    ) -> ActionResult:
        self.run_calls.append({"turn_id": turn_id, "utterance": utterance, "plan": plan})
        if plan.decode is not None:
            call = decode(plan.decode)
            if self._decode_wrapper is not None:
                call = self._decode_wrapper(call)
            try:
                self.decode_replies.append(await call)
            except (ActionDecodeError, TimeoutError, CapabilityUnavailableError) as exc:
                self.decode_errors.append(exc)
                return _result(notices=[_NOTICE])
        if isinstance(self._result, BaseException):
            for hook in self.before_raise:
                hook()
            raise self._result
        return self._result

    async def decide(self, *args: Any, **kwargs: Any):  # noqa: ANN201
        raise AssertionError("no decision is part of this turn")

    def static_sentence(self) -> str:
        return _SENTENCE

    def redact(self, text: str) -> str:
        return text.replace(_SECRET, "***")

    def record_turn(self, session_id: str, used: Any) -> None:
        self.record_turn_calls.append((session_id, list(used)))

    def drain_events(self, session_id: str) -> list[dict[str, object]]:
        return self._drain_script.pop(0) if self._drain_script else []


class _RecordingTiers:
    """Tiers that log each lookup, so a test sees when the provider is asked for."""

    def __init__(self, provider: StreamingLLMProvider | None) -> None:
        self._provider = provider
        self.asked: list[ModelTier] = []

    def provider(self, tier: ModelTier) -> StreamingLLMProvider | None:
        self.asked.append(tier)
        return self._provider


class _BlockingStream:
    """A provider stream that never yields, and records whether it was closed."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    def __aiter__(self) -> _BlockingStream:
        return self

    async def __anext__(self) -> LLMResponse:
        self.started.set()
        await asyncio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _BlockingDecodeProvider(FakeStreamingLLMProvider):
    """Blocks the decode on a `_BlockingStream`; every other pass plays from the script."""

    def __init__(self, passes: list[ScriptedPass]) -> None:
        super().__init__(passes)
        self.stream = _BlockingStream()

    def generate(self, request: LLMRequest, *, stream: bool = False) -> Any:  # type: ignore[override]
        if request.response_schema is not None:
            self.calls.append(request)
            return self.stream
        return super().generate(request, stream=stream)


class _ContextProvider(FakeStreamingLLMProvider):
    """Records the LLM call context each request was made under."""

    def __init__(self, passes: list[ScriptedPass]) -> None:
        super().__init__(passes)
        self.contexts: list[dict[str, Any]] = []

    async def generate(  # type: ignore[override]
        self, request: LLMRequest, *, stream: bool = False
    ) -> AsyncIterator[LLMResponse]:
        self.contexts.append(dict(get_llm_call_context()))
        async for chunk in super().generate(request, stream=stream):
            yield chunk


class _SpyPlanner(ContextBudgetPlanner):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.extra_fixed_texts: list[list[str] | None] = []

    def plan(self, **kwargs: Any):  # noqa: ANN201
        self.extra_fixed_texts.append(kwargs.get("extra_fixed_texts"))
        return super().plan(**kwargs)


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


def _vault_fact(text: str) -> RetrievedFact:
    return RetrievedFact(
        subject="VaultNote",
        subject_type="VaultSession",
        predicate="MENTIONS",
        object="Keys",
        object_type="VaultChunk",
        properties={"path": "sessions/keys.md", "text": text, "content": text},
        similarity_score=0.9,
        graph_distance=999,
    )


class _Retriever(KnowledgeRetriever):
    """Returns one vault fact carrying `text` when set, else nothing."""

    def __init__(self, config: Any, graph_store: GraphStore, text: str | None = None) -> None:
        super().__init__(config=config, graph_store=graph_store)
        self.text = text

    async def retrieve(self, query: str, user_id: str = "User", **_kwargs: Any) -> RetrievalResult:
        facts = [_vault_fact(self.text)] if self.text else []
        return RetrievalResult(
            query=query,
            user_id=user_id,
            facts=facts,
            entities_found=len(facts),
            total_facts=len(facts),
            formatted_context=f"Relevant prose:\n{self.text}" if self.text else "",
            retrieval_time_ms=0.0,
            vector_search_time_ms=0.0,
            graph_traversal_time_ms=0.0,
            config_used={},
        )


class _FakeExtractionPipeline:
    async def extract_from_utterance(self, **_kwargs: Any):  # noqa: ANN201
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


class _Dispatch:
    """Stands in for `_dispatch_tool` and records what reached it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, tool_call: LLMToolCall) -> str:
        self.calls.append(tool_call.name)
        return "stub tool result"


def _make_handler(
    passes: Sequence[ScriptedPass] = (),
    *,
    port: _Port | None = None,
    tiers: Any = "same",
    provider: FakeStreamingLLMProvider | None = None,
    retrieval_text: str | None = None,
    planner: bool = False,
    dispatcher: _CountingDispatcher | None = None,
) -> tuple[ConversationHandler, FakeStreamingLLMProvider, _FakeEventStore]:
    """Build a handler; `tiers="same"` serves the interaction tier from the handler's provider."""
    config = build_test_config()
    graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
    provider = provider if provider is not None else FakeStreamingLLMProvider(list(passes))
    handler = ConversationHandler(
        config=config,
        graph_store=graph_store,
        extraction_pipeline=_FakeExtractionPipeline(),
        retriever=_Retriever(config, graph_store, retrieval_text),
        llm_provider=provider,
        conventions_loader=make_test_conventions_loader(),
        budget_planner=_SpyPlanner(config.context_budget) if planner else None,
    )
    store = _FakeEventStore()
    handler.event_store = store  # type: ignore[assignment]
    handler._dispatch_tool = _Dispatch()  # type: ignore[method-assign]
    if dispatcher is not None:
        handler._extraction_dispatcher = dispatcher  # type: ignore[assignment]
    if port is not None:
        if tiers == "same":
            tiers = FixedTiers({"interaction": provider})
        handler.attach_capability_host(port, tiers=tiers)
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
    tool_call_id: str = "call-1",
) -> ActionRecord:
    return ActionRecord(
        tool_call_id=tool_call_id,
        tool=tool,
        capability=capability,
        risk="reversible",
        args_summary="minutes=10",
        outcome=ToolOutcome(status=status, summary=summary, reason=reason),
        duration_ms=7,
        power_level="full",
        ack="Setting a timer.",
        view="timers",
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


def _without_sentence(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The messages with the static sentence a host adds taken out again."""
    return [{**m, "content": str(m["content"]).replace(f"\n{_SENTENCE}", "")} for m in messages]


def _shape(events: list[StreamEvent]) -> list[Any]:
    """The stream with `Complete.duration_ms`, the one field that varies by run, left out."""
    return [(e.final_response, e.tool_calls_used) if isinstance(e, Complete) else e for e in events]


def _main_calls(provider: FakeStreamingLLMProvider) -> list[LLMRequest]:
    return [request for request in provider.calls if request.response_schema is None]


# --- Today's turn -------------------------------------------------------------


class TestTodaysTurn:
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param("plan-without-decode", id="plan-without-decode"),
            pytest.param("no-tiers", id="no-tiers"),
        ],
    )
    @pytest.mark.asyncio
    async def test_turn_matches_the_turn_with_no_host(self, case):
        reply = ScriptedPass(chunks=["Here is a story. It is short."])
        baseline, base_provider, _ = _make_handler([reply])
        if case == "plan-without-decode":
            port = _Port(plan=_plan(decode=None))
            handler, provider, _ = _make_handler(
                [ScriptedPass(chunks=["Here is a story. It is short."])], port=port
            )
        else:
            port = _Port()
            handler, provider, _ = _make_handler(
                [ScriptedPass(chunks=["Here is a story. It is short."])], port=port, tiers=None
            )

        expected = await _collect(baseline, "tell me a story")
        events = await _collect(handler, "tell me a story")

        assert len(provider.calls) == len(base_provider.calls) == 1
        assert provider.calls[0].tools == base_provider.calls[0].tools == KNOWLEDGE_TOOL_SCHEMAS
        assert _without_sentence(provider.calls[0].messages) == base_provider.calls[0].messages
        assert _shape(events) == _shape(expected)
        assert port.run_calls == []
        assert len(port.plan_calls) == (1 if case == "plan-without-decode" else 0)

    @pytest.mark.asyncio
    async def test_no_records_and_no_notices_runs_todays_turn(self):
        baseline, base_provider, _ = _make_handler([ScriptedPass(chunks=["Sure thing."])])
        port = _Port(plan=_plan(), result=_result())
        handler, provider, _ = _make_handler(
            [_decode_pass(), ScriptedPass(chunks=["Sure thing."])], port=port
        )

        expected = await _collect(baseline, "set a timer maybe")
        events = await _collect(handler, "set a timer maybe")

        main = _main_calls(provider)
        assert len(provider.calls) == 2 and len(main) == 1
        assert main[0].tools == KNOWLEDGE_TOOL_SCHEMAS
        assert _without_sentence(main[0].messages) == base_provider.calls[0].messages
        assert _shape(events) == _shape(expected)

    @pytest.mark.asyncio
    async def test_a_decode_failure_notice_follows_todays_turn(self):
        port = _Port(plan=_plan())
        handler, provider, _ = _make_handler(
            [_decode_pass("not json"), ScriptedPass(chunks=["Sure thing."])], port=port
        )
        port._result = _result(notices=[_NOTICE])

        events = await _collect(handler, "set a timer maybe")

        assert len(_main_calls(provider)) == 1
        assert [t.text for t in _tokens(events)] == ["Sure thing.", f" {_NOTICE}"]
        assert _complete(events).final_response == f"Sure thing. {_NOTICE}"
        assert _history(handler)[-1] == ("assistant", f"Sure thing. {_NOTICE}")


# --- The decode callback ------------------------------------------------------


class TestDecodeCallback:
    @pytest.mark.asyncio
    async def test_the_decode_request_carries_every_field(self):
        port = _Port(plan=_plan(), result=_result())
        handler, provider, _ = _make_handler(
            [_decode_pass(), ScriptedPass(chunks=["Ok."])], port=port
        )

        await _collect(handler, "set a timer for ten minutes")

        assert provider.calls[0] == LLMRequest(
            messages=[dict(m) for m in _DECODE.messages],
            tools=None,
            response_schema=dict(_DECODE.response_schema),
            thinking=ThinkingConfig(budget_tokens=0),
            temperature=_DECODE.temperature,
            max_tokens=_DECODE.max_tokens,
        )
        assert port.decode_replies == [DecodeReply(content=_DECODE_JSON, finish_reason="stop")]

    @pytest.mark.asyncio
    async def test_the_decode_is_built_and_run_as_the_capability_actions_call_site(
        self, monkeypatch
    ):
        port = _Port(plan=_plan(), result=_result())
        provider = _ContextProvider([_decode_pass(), ScriptedPass(chunks=["Ok."])])
        handler, _provider, _ = _make_handler(port=port, provider=provider)
        sites: list[str] = []
        build = handler._build_request

        def _spy(*, call_site: str, **kwargs: Any) -> LLMRequest:
            sites.append(call_site)
            return build(call_site=call_site, **kwargs)

        monkeypatch.setattr(handler, "_build_request", _spy)

        await _collect(handler, "set a timer for ten minutes")

        assert sites == ["chat.capability_actions", "chat.initial"]
        assert provider.contexts[0]["call_site"] == "chat.capability_actions"
        assert provider.contexts[0]["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_the_interaction_tier_is_asked_for_its_provider_at_call_time(self):
        decode_provider = FakeStreamingLLMProvider([_decode_pass()])
        tiers = _RecordingTiers(decode_provider)
        port = _Port(plan=_plan(), result=_result())
        handler, provider, _ = _make_handler([ScriptedPass(chunks=["Ok."])], port=port, tiers=tiers)

        assert tiers.asked == []
        await _collect(handler, "set a timer for ten minutes")

        assert tiers.asked == ["interaction"]
        assert len(decode_provider.calls) == 1
        assert decode_provider.calls[0].response_schema is not None
        assert [r.response_schema for r in provider.calls] == [None]

    @pytest.mark.asyncio
    async def test_no_interaction_provider_raises_capability_unavailable(self):
        port = _Port(plan=_plan())
        handler, provider, _ = _make_handler(
            [ScriptedPass(chunks=["Sure thing."])], port=port, tiers=FixedTiers({})
        )

        events = await _collect(handler, "set a timer for ten minutes")

        assert [type(e) for e in port.decode_errors] == [CapabilityUnavailableError]
        assert len(provider.calls) == 1
        assert _complete(events).final_response == f"Sure thing. {_NOTICE}"

    @pytest.mark.asyncio
    async def test_a_cancelled_decode_closes_the_provider_stream(self):
        provider = _BlockingDecodeProvider([ScriptedPass(chunks=["Sure thing."])])
        port = _Port(plan=_plan(), decode_wrapper=lambda call: asyncio.wait_for(call, timeout=0.05))
        handler, _provider, _ = _make_handler(port=port, provider=provider)

        events = await _collect(handler, "set a timer for ten minutes")

        assert provider.stream.started.is_set()
        assert provider.stream.closed
        assert [type(e) for e in port.decode_errors] == [TimeoutError]
        assert _complete(events).final_response == f"Sure thing. {_NOTICE}"


# --- Routing input ------------------------------------------------------------


class TestRoutingInput:
    @pytest.mark.asyncio
    async def test_plan_turn_reads_history_before_this_message_and_the_received_time(self):
        port = _Port(plan=_plan(decode=None))
        handler, _provider, _ = _make_handler(
            [ScriptedPass(chunks=["Hi."]), ScriptedPass(chunks=["Ok."])], port=port
        )

        await _collect(handler, "hello")
        await _collect(handler, "set a timer")

        assert port.plan_calls[1]["recent"] == [("user", "hello"), ("assistant", "Hi.")]
        assert port.plan_calls[1]["utterance"] == "set a timer"
        assert port.plan_calls[1]["received_at"] == port.pre_llm_calls[1]["received_at"]
        assert port.plan_calls[1]["received_at"] is not None

    @pytest.mark.asyncio
    async def test_the_remainder_after_an_approval_is_routed_and_decoded(self):
        approved = PreTurnResolution(
            handled=True, reply="Approved.", remainder="dim them", records=(), events=()
        )
        port = _Port(plan=_plan(), result=_result(), pre_script=[approved, None])
        handler, _provider, _ = _make_handler(
            [_decode_pass(), ScriptedPass(chunks=["Done."])], port=port
        )

        await _collect(handler, "yes, and dim them")

        assert [c["utterance"] for c in port.plan_calls] == ["dim them"]
        assert [c["utterance"] for c in port.run_calls] == ["dim them"]
        assert port.run_calls[0]["turn_id"] == port.pre_llm_calls[0]["turn_id"]
        assert _history(handler)[0] == ("user", "yes, and dim them")


# --- Error backstop -----------------------------------------------------------


class TestErrorBackstop:
    @pytest.mark.parametrize("step", ["plan_turn", "run_actions"])
    @pytest.mark.asyncio
    async def test_a_host_error_gives_the_redacted_error_turn(self, step):
        error = RuntimeError(f"routing failed with {_SECRET}")
        if step == "plan_turn":
            port = _Port(plan=error)
            passes: list[ScriptedPass] = []
        else:
            port = _Port(plan=_plan(), result=error)
            passes = [_decode_pass()]
        handler, provider, store = _make_handler(passes, port=port)

        events = await _collect(handler, "set a timer")

        expected = "I encountered an error: routing failed with ***"
        assert [t.text for t in _tokens(events)] == [expected]
        assert _complete(events).final_response == expected
        assert _history(handler) == [("user", "set a timer"), ("assistant", expected)]
        assert [e.system_response for e in store.events] == [expected]
        assert _main_calls(provider) == []

    @pytest.mark.asyncio
    async def test_the_finally_drain_still_pairs_every_started_event(self):
        port = _Port(plan=_plan(), result=RuntimeError("boom"))
        handler, _provider, _ = _make_handler([_decode_pass()], port=port)
        orphan = {"type": "tool_call_started", "tool_call_id": "orphan", "name": "x"}
        port.before_raise.append(lambda: handler._turn_ws_events.append(orphan))

        events = await _collect(handler, "set a timer")

        assert _payloads(events) == [orphan]

    @pytest.mark.asyncio
    async def test_each_record_pairs_its_events_when_a_main_pass_fails(self):
        from backend.errors import LLMConnectionError

        records = (_record(tool_call_id="a"), _record(tool_call_id="b"))
        port = _Port(plan=_plan(), result=_result(*records, block="Actions this turn."))
        handler, _provider, _ = _make_handler(
            [_decode_pass(), ScriptedPass(error=LLMConnectionError("down"))], port=port
        )

        events = await _collect(handler, "set two timers")

        started = [p["tool_call_id"] for p in _payloads(events) if p["type"] == "tool_call_started"]
        completed = [
            p["tool_call_id"] for p in _payloads(events) if p["type"] == "tool_call_completed"
        ]
        assert started == completed == ["a", "b"]
        assert _complete(events).final_response == "I encountered an error: down"


# --- Calls answered by the template -------------------------------------------


class TestAnsweredByActions:
    @pytest.mark.asyncio
    async def test_the_reply_streams_as_tokens_and_no_main_pass_runs(self):
        dispatcher = _CountingDispatcher()
        port = _Port(
            plan=_plan(),
            result=_result(_record(), also_reply=False, reply="Timer set. It ends at noon."),
        )
        handler, provider, store = _make_handler([_decode_pass()], port=port, dispatcher=dispatcher)

        events = await _collect(handler, "set a timer for ten minutes")

        assert len(provider.calls) == 1
        assert [(t.text, t.pass_num) for t in _tokens(events)] == [
            ("Timer set.", 1),
            (" It ends at noon.", 1),
        ]
        assert _complete(events).final_response == "Timer set. It ends at noon."
        assert _history(handler) == [
            ("user", "set a timer for ten minutes"),
            ("assistant", "Timer set. It ends at noon."),
        ]
        assert [e.system_response for e in store.events] == ["Timer set. It ends at noon."]
        assert dispatcher.wakes == 0
        assert handler._extraction_tasks == {}

    @pytest.mark.asyncio
    async def test_records_give_tool_events_with_the_model_source(self):
        proposed = {"type": "action_proposed", "proposal_id": "p1"}
        later = {"type": "action_resolved", "proposal_id": "p0", "resolution": "superseded"}
        records = (
            _record(),
            _record(
                tool="lights.power",
                capability="lights",
                status="unavailable",
                summary="Lights are unavailable.",
                reason=f"unreachable {_SECRET}",
                tool_call_id="call-2",
            ),
        )
        port = _Port(
            plan=_plan(),
            result=_result(*records, also_reply=False, reply="Timer set.", events=[proposed]),
            drain_script=[[], [later]],
        )
        handler, _provider, _ = _make_handler([_decode_pass()], port=port)

        events = await _collect(handler, "set a timer and turn on the lights")

        payloads = _payloads(events)
        assert [p["type"] for p in payloads] == [
            "tool_call_started",
            "tool_call_completed",
            "tool_call_started",
            "tool_call_completed",
            "action_proposed",
            "action_resolved",
        ]
        assert payloads[0] == {
            "type": "tool_call_started",
            "tool_call_id": "call-1",
            "name": "timers.start",
            "args_summary": "minutes=10",
            "capability": "timers",
            "risk": "reversible",
            "ack": "Setting a timer.",
            "view": "timers",
        }
        assert payloads[1] == {
            "type": "tool_call_completed",
            "tool_call_id": "call-1",
            "name": "timers.start",
            "duration_ms": 7,
            "result_summary": "Timer set for 10 minutes.",
            "error": None,
            "capability": "timers",
            "status": "ok",
            "source": "model",
            "power_level": "full",
            "view": "timers",
        }
        assert payloads[3]["status"] == "unavailable"
        assert payloads[3]["error"] == "unavailable: unreachable ***"
        assert payloads[3]["source"] == "model"
        last_payload = max(i for i, e in enumerate(events) if isinstance(e, WSEvent))
        first_token = min(i for i, e in enumerate(events) if isinstance(e, Token))
        assert last_payload < first_token
        assert _complete(events).tool_calls_used == 2
        assert handler._dispatch_tool.calls == []  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_record_turn_gets_the_capabilities_the_calls_used(self):
        records = (
            _record(),
            _record(tool="lights.power", capability="lights", tool_call_id="call-2"),
            _record(tool_call_id="call-3"),
        )
        port = _Port(plan=_plan(), result=_result(*records, also_reply=False, reply="Done."))
        handler, _provider, _ = _make_handler([_decode_pass()], port=port)

        await _collect(handler, "set a timer and turn on the lights")

        assert port.record_turn_calls == [("s1", ["timers", "lights"])]

    @pytest.mark.asyncio
    async def test_record_turn_is_called_once_on_a_turn_with_no_call(self):
        port = _Port(plan=_plan(decode=None))
        handler, _provider, _ = _make_handler([ScriptedPass(chunks=["Hi."])], port=port)

        await _collect(handler, "hello there")

        assert port.record_turn_calls == [("s1", [])]


# --- Calls with also_reply ----------------------------------------------------


_BLOCK = 'Actions this turn, already decided.\n- timers.start (minutes=10): done. {"status":"ok"}'


class TestAlsoReply:
    @pytest.mark.asyncio
    async def test_todays_passes_run_with_the_core_tools_in_their_order(self):
        tool_call = LLMToolCall(id="kg-1", name="query_knowledge_graph", arguments={"query": "x"})
        port = _Port(
            plan=_plan(),
            result=_result(_record(), block=_BLOCK, notices=["(Waiting for your yes or no.)"]),
        )
        handler, provider, _ = _make_handler(
            [
                _decode_pass(),
                ScriptedPass(chunks=["Let me look."], tool_calls=[tool_call]),
                ScriptedPass(chunks=["Timer is set and you like tea."]),
            ],
            port=port,
        )

        events = await _collect(handler, "set a timer and tell me what I drink")

        decode, first, final = provider.calls
        assert decode.response_schema is not None
        assert first.tools == KNOWLEDGE_TOOL_SCHEMAS
        assert final.tools is None
        assert handler._dispatch_tool.calls == ["query_knowledge_graph"]  # type: ignore[attr-defined]
        assert [(t.text, t.pass_num) for t in _tokens(events)] == [
            ("Let me look.", 1),
            (" Timer is set and you like tea.", 2),
            (" (Waiting for your yes or no.)", 2),
        ]
        final_text = "Let me look. Timer is set and you like tea. (Waiting for your yes or no.)"
        assert _complete(events).final_response == final_text
        assert _history(handler)[-1] == ("assistant", final_text)
        types = [p["type"] for p in _payloads(events)]
        assert types == ["tool_call_started", "tool_call_completed"] * 2
        assert _complete(events).tool_calls_used == 2

    @pytest.mark.asyncio
    async def test_the_block_sits_after_the_profile_and_before_retrieval(self, monkeypatch):
        port = _Port(plan=_plan(), result=_result(_record(), block=_BLOCK))
        handler, provider, _ = _make_handler(
            [_decode_pass(), ScriptedPass(chunks=["Done."])],
            port=port,
            retrieval_text=_SENTINEL,
        )
        monkeypatch.setattr(handler, "_load_user_profile_block", lambda: (None, _PROFILE))

        await _collect(handler, "set a timer for ten minutes")

        contents = [str(m["content"]) for m in _main_calls(provider)[0].messages]
        profile_at = contents.index(_PROFILE)
        block_at = contents.index(_BLOCK)
        retrieval_at = next(i for i, c in enumerate(contents) if _SENTINEL in c)
        assert profile_at < block_at < retrieval_at
        assert _main_calls(provider)[0].messages[block_at]["role"] == "system"

    @pytest.mark.asyncio
    async def test_the_block_is_charged_to_the_budget_planner(self, monkeypatch):
        port = _Port(plan=_plan(), result=_result(_record(), block=_BLOCK))
        handler, _provider, _ = _make_handler(
            [_decode_pass(), ScriptedPass(chunks=["Done."])], port=port, planner=True
        )
        monkeypatch.setattr(handler, "_load_user_profile_block", lambda: (None, _PROFILE))

        await _collect(handler, "set a timer for ten minutes")

        planner = handler._budget_planner
        assert isinstance(planner, _SpyPlanner)
        fixed = planner.extra_fixed_texts[0] or []
        assert _BLOCK in fixed and _PROFILE in fixed

    def test_build_messages_default_keeps_todays_messages(self):
        handler, _provider, _ = _make_handler()
        session = handler.get_or_create_session("s1", "User")
        session.add_message("user", "hello")

        plain = handler._build_messages(session, 10)
        defaulted = handler._build_messages(session, 10, capability_block=None)
        with_block = handler._build_messages(session, 10, capability_block=_BLOCK)

        assert defaulted == plain
        assert [m for m in with_block if m["content"] != _BLOCK] == plain


# --- The decode prompt holds no turn context ----------------------------------


class _TimerArgs(BaseModel):
    minutes: int


_TIMER_SPEC = ToolSpec(
    name="timers.start",
    description="Start a timer.",
    args_model=_TimerArgs,
    risk="reversible",
    default_policy="allow",
    summarize=lambda args: f"a {args.minutes} minute timer",  # type: ignore[attr-defined]
)


class TestDecodePromptHoldsNoContext:
    @pytest.mark.asyncio
    async def test_the_retrieval_context_never_reaches_the_decode(self, monkeypatch):
        decoder = ActionDecoder()

        def _real_plan(session_id: str, utterance: str, recent: Sequence[tuple[str, str]]):
            offered = [(_TIMER_SPEC, to_args_schema(_TIMER_SPEC))]
            return TurnPlan(
                route=_route(),
                offered=frozenset({"timers.start"}),
                decode=decoder.build(offered, recent, utterance),
            )

        port = _Port(plan=_real_plan, result=_result(_record(), block=_BLOCK))
        handler, provider, _ = _make_handler(
            [
                ScriptedPass(chunks=["Hello Raj."]),
                _decode_pass(),
                ScriptedPass(chunks=["Timer set."]),
            ],
            port=port,
        )
        # The first turn has no decode, so it only gives the second its history.
        port._plan = _plan(decode=None)
        await _collect(handler, "hello")
        port._plan = _real_plan
        handler.retriever.text = _SENTINEL  # type: ignore[attr-defined]
        monkeypatch.setattr(handler, "_load_user_profile_block", lambda: (None, _PROFILE))

        await _collect(handler, "set a timer for ten minutes")

        decode = next(r for r in provider.calls if r.response_schema is not None)
        expected = decoder.build(
            [(_TIMER_SPEC, to_args_schema(_TIMER_SPEC))],
            [("user", "hello"), ("assistant", "Hello Raj.")],
            "set a timer for ten minutes",
        )
        assert decode.messages == [dict(m) for m in expected.messages]
        assert [m["role"] for m in decode.messages] == ["system", "user", "assistant", "user"]
        assert _SENTINEL not in json.dumps(decode.messages)
        assert _PROFILE not in json.dumps(decode.messages)
        main = _main_calls(provider)[-1]
        assert _SENTINEL in json.dumps(main.messages)
