"""Tests for `CapabilityHost`: the invoke pipeline, pre-LLM order, the action flow and stats.

Classes follow the groups of T4's acceptance: the pipeline stages, redaction, lifecycle and
statuses, `pre_llm`, `plan_turn`, `run_actions`, stats, and concurrency on one loop.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, replace
from typing import TypeVar, cast

import pytest
import pytest_asyncio
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.capabilities.action_log import ActionLog
from backend.capabilities.contracts import (
    CapabilityStatus,
    CapabilityTurnPort,
    DecodeReply,
    FastIntent,
    InvocationContext,
    ToolOutcome,
)
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import ActionDecoder
from backend.capabilities.events import CapabilitiesStatusEvent
from backend.capabilities.host import (
    DECODE_FAILURE_NOTICE,
    NO_LIVE_PROPOSAL_REPLY,
    REJECTED_REPLY,
    STATIC_SENTENCE,
    CapabilityHost,
    HostSettings,
)
from backend.capabilities.policy import PolicyTable
from backend.capabilities.power import PowerState, SingleModelTiers, TierProviders
from backend.capabilities.replies import (
    NoAddress,
    ReplyRegister,
    actions_block,
    notices_for,
    reply_for,
)
from backend.capabilities.router import RouterConfig, ToolRouter
from backend.capabilities.schema import to_args_schema
from backend.capabilities.secrets import Secret, SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.errors import (
    CapabilityInvocationError,
    CapabilityUnavailableError,
    McpTransportError,
    TimeParseError,
)
from backend.llm.models import ToolCall
from backend.llm.provider import StreamingLLMProvider
from tests.unit.capabilities.fakes import (
    DEFAULT_START,
    EmptyArgs,
    FakeCapability,
    FakeClock,
    FakePower,
    FakeTiers,
    RecordingEventSink,
    ScriptedDecode,
    make_tool_spec,
)

T = TypeVar("T")
SESSION = "s1"
OTHER = "s2"
TOKEN = "tok-5f1e2d9c"


# -- Argument models and tools ---------------------------------------------------------------


class TargetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str | None = None


class LevelArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: int = Field(ge=0, le=100)


class AdjustArgs(BaseModel):
    """A cross-field rule the grammar cannot express, so only validation catches it."""

    model_config = ConfigDict(extra="forbid")
    brightness: int | None = None
    color: str | None = None

    @model_validator(mode="after")
    def _needs_one(self) -> AdjustArgs:
        if self.brightness is None and self.color is None:
            raise ValueError("give brightness or color")
        return self


def _power_summary(args: BaseModel) -> str:
    assert isinstance(args, TargetArgs)
    return f"turn off the {args.target or 'default'} lights"


POWER = make_tool_spec(
    "lights.power",
    TargetArgs,
    risk="physical",
    default_policy="confirm_inferred",
    summarize=_power_summary,
)
ADJUST = make_tool_spec("lights.adjust", AdjustArgs, summarize=lambda args: "adjust the lights")
PLAY = make_tool_spec(
    "music.play",
    EmptyArgs,
    risk="physical",
    default_policy="confirm_inferred",
    summarize=lambda args: "play jazz",
)
CONTROL = make_tool_spec("music.control", EmptyArgs, summarize=lambda args: "pause the music")


def _lights(**kwargs: object) -> FakeCapability:
    return FakeCapability("lights", tools=(POWER, ADJUST), **kwargs)  # type: ignore[arg-type]


def _music(**kwargs: object) -> FakeCapability:
    return FakeCapability("music", tools=(PLAY, CONTROL), **kwargs)  # type: ignore[arg-type]


def _alarms() -> FakeCapability:
    return FakeCapability("alarms", tools=(make_tool_spec("alarms.list"),))


# -- The rig ---------------------------------------------------------------------------------


class RecordingTimeouts:
    """A timeout runner that records each bound and delegates to `asyncio.wait_for`."""

    def __init__(self) -> None:
        self.bounds: list[float] = []

    async def __call__(self, awaitable: Awaitable[T], timeout: float) -> T:
        self.bounds.append(timeout)
        return await asyncio.wait_for(awaitable, timeout)


async def _always_times_out(awaitable: Awaitable[T], timeout: float) -> T:
    if inspect.iscoroutine(awaitable):
        awaitable.close()
    raise TimeoutError


@dataclass
class Rig:
    host: CapabilityHost
    caps: dict[str, FakeCapability]
    sink: RecordingEventSink
    clock: FakeClock
    secrets: SecretRegistry
    db: CapabilitiesDb


_RIGS: list[Rig] = []


@pytest_asyncio.fixture(autouse=True)
async def _stop_hosts():
    """Stop every host a test built, so no proposal timer outlives the test's loop."""
    yield
    while _RIGS:
        await _RIGS.pop().host.stop()


def _rig(
    *capabilities: FakeCapability,
    settings: HostSettings | None = None,
    timeout_runner: object | None = None,
    policy: PolicyTable | None = None,
    power: PowerState | None = None,
    tiers: TierProviders | None = None,
    register: ReplyRegister | None = None,
) -> Rig:
    counter = ApproximateTokenCounter()
    router = ToolRouter(
        list(capabilities),
        RouterConfig(token_budget=1200, reserved_tokens=100),
        counter,
        None,
    )
    specs = [spec for capability in capabilities for spec in capability.tools()]
    sink = RecordingEventSink()
    clock = FakeClock()
    secrets = SecretRegistry()
    db = CapabilitiesDb(":memory:")
    extra: dict[str, object] = {} if timeout_runner is None else {"timeout_runner": timeout_runner}
    if power is not None:
        extra["power"] = power
    if tiers is not None:
        extra["tiers"] = tiers
    if register is not None:
        extra["register"] = register
    host = CapabilityHost(
        list(capabilities),
        router=router,
        counter=counter,
        policy=policy or PolicyTable.from_specs(specs, None),
        decoder=ActionDecoder(),
        secrets=secrets,
        action_log=ActionLog(db),
        event_sink=sink,
        clock=clock,
        settings=settings or HostSettings(),
        **extra,  # type: ignore[arg-type]
    )
    rig = Rig(host, {c.name: c for c in capabilities}, sink, clock, secrets, db)
    _RIGS.append(rig)
    return rig


def _ctx(source: str = "model", call_id: str = "call-1", session: str = SESSION):
    return InvocationContext(
        session_id=session,
        turn_id="turn-1",
        utterance="do it",
        tool_call_id=call_id,
        source=source,  # type: ignore[arg-type]
        now=DEFAULT_START,
    )


def _call(name: str, call_id: str = "call-1", **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))


def _decoded(*calls: tuple[str, dict[str, object]], also_reply: bool = False) -> DecodeReply:
    content = json.dumps(
        {"calls": [{"tool": tool, "args": args} for tool, args in calls], "also_reply": also_reply}
    )
    return DecodeReply(content=content, finish_reason="stop")


async def _propose(rig: Rig, *calls: tuple[str, dict[str, object]], session: str = SESSION):
    """Create a live proposal through the model path, and drain its event."""
    plan = rig.host.plan_turn(session, "lights and music", [])
    result = await rig.host.run_actions(
        session, "turn-0", "lights and music", plan, ScriptedDecode(_decoded(*calls))
    )
    rig.host.drain_events(session)
    (proposal_id,) = {r.outcome.proposal_id for r in result.records if r.outcome.proposal_id}
    return proposal_id


# -- CapabilityTurnPort ----------------------------------------------------------------------


class TestPort:
    def test_host_implements_the_turn_port(self):
        assert CapabilityTurnPort in CapabilityHost.__mro__
        port_methods = [
            name
            for name, value in vars(CapabilityTurnPort).items()
            if callable(value) and not name.startswith("_")
        ]
        assert set(port_methods) >= {"pre_llm", "plan_turn", "run_actions", "decide"}
        for name in port_methods:
            port_signature = inspect.signature(getattr(CapabilityTurnPort, name))
            host_signature = inspect.signature(getattr(CapabilityHost, name))
            assert list(host_signature.parameters) == list(port_signature.parameters), name
            assert inspect.iscoroutinefunction(
                getattr(CapabilityHost, name)
            ) == inspect.iscoroutinefunction(getattr(CapabilityTurnPort, name)), name

    def test_static_sentence_is_the_fixed_honesty_sentence(self):
        rig = _rig(_lights())

        assert rig.host.static_sentence() == STATIC_SENTENCE
        assert rig.host.static_sentence() == rig.host.static_sentence()
        assert "only when the actions block says so" in STATIC_SENTENCE

    def test_record_turn_feeds_router_stickiness(self):
        rig = _rig(_lights(), _music())

        rig.host.record_turn(SESSION, ["music"])

        assert rig.host.plan_turn(SESSION, "and another", []).route.capabilities == ("music",)

    def test_settings_come_from_the_capabilities_config(self):
        from backend.capabilities.config import load_capabilities_config

        config = load_capabilities_config(
            {
                "MIST_CAPABILITY_TOOL_TIMEOUT_S": "3",
                "MIST_CAPABILITY_PROPOSAL_TTL_S": "30",
                "MIST_CAPABILITY_DECODE_TIMEOUT_S": "4",
                "MIST_CAPABILITY_FAST_PATH": "false",
                "MIST_CAPABILITY_WAKE_WORDS": "hey mist",
            }
        )

        assert HostSettings.from_config(config) == HostSettings(
            tool_timeout_s=3.0,
            proposal_ttl_s=30.0,
            decode_timeout_s=4.0,
            fast_path=False,
            wake_words=("hey mist",),
        )


# -- The invoke pipeline ---------------------------------------------------------------------


class TestPipeline:
    @pytest.mark.asyncio
    async def test_unknown_tool_is_invalid_args(self):
        rig = _rig(_lights())

        record = await rig.host.invoke(_call("lights.explode"), _ctx(), None)

        assert record.outcome.status == "invalid_args"
        assert "lights.explode" in record.outcome.summary
        assert rig.caps["lights"].invocations == []

    @pytest.mark.asyncio
    async def test_known_tool_not_offered_this_turn_is_invalid_args(self):
        rig = _rig(_lights())

        record = await rig.host.invoke(
            _call("lights.adjust", brightness=10), _ctx(), frozenset({"lights.power"})
        )

        assert record.outcome.status == "invalid_args"
        assert "not offered" in record.outcome.summary
        assert rig.caps["lights"].invocations == []

    @pytest.mark.asyncio
    async def test_arguments_failing_validation_are_invalid_args(self):
        rig = _rig(FakeCapability("dim", tools=(make_tool_spec("dim.set", LevelArgs),)))

        record = await rig.host.invoke(_call("dim.set", level=200), _ctx("fastpath"), None)

        assert record.outcome.status == "invalid_args"
        assert "level" in record.outcome.summary
        assert rig.caps["dim"].invocations == []

    @pytest.mark.asyncio
    async def test_unavailable_capability_stops_at_the_status_gate(self):
        rig = _rig(_lights())
        rig.caps["lights"].set_status("unavailable", "hub down")

        record = await rig.host.invoke(_call("lights.adjust", color="red"), _ctx(), None)

        assert record.outcome.status == "unavailable"
        assert record.outcome.reason == "hub down"
        assert rig.caps["lights"].invocations == []

    @pytest.mark.asyncio
    async def test_degraded_capability_passes_the_status_gate(self):
        rig = _rig(_lights())
        rig.caps["lights"].set_status("degraded", "no HassLightSet")

        record = await rig.host.invoke(_call("lights.adjust", color="red"), _ctx(), None)

        assert record.outcome.status == "ok"

    @pytest.mark.asyncio
    async def test_deny_refuses_the_call(self):
        spec = make_tool_spec("vault.open", default_policy="deny")
        rig = _rig(FakeCapability("vault", tools=(spec,)))

        record = await rig.host.invoke(_call("vault.open"), _ctx("fastpath"), None)

        assert record.outcome.status == "denied"
        assert rig.caps["vault"].invocations == []

    @pytest.mark.asyncio
    async def test_confirm_inferred_runs_a_fastpath_call(self):
        rig = _rig(_lights())

        record = await rig.host.invoke(
            _call("lights.power", target="bedroom"), _ctx("fastpath"), None
        )

        assert record.outcome.status == "ok"
        assert len(rig.caps["lights"].invocations) == 1

    @pytest.mark.asyncio
    async def test_confirm_inferred_proposes_a_model_call(self):
        rig = _rig(_lights())

        record = await rig.host.invoke(_call("lights.power", target="bedroom"), _ctx("model"), None)

        assert record.outcome.status == "pending_confirmation"
        assert record.outcome.proposal_id
        assert record.args_summary == "turn off the bedroom lights"
        assert rig.caps["lights"].invocations == []
        (event,) = rig.host.drain_events(SESSION)
        assert event["type"] == "action_proposed"
        assert event["proposal_id"] == record.outcome.proposal_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize("source", ["fastpath", "model"])
    async def test_confirm_proposes_from_both_sources(self, source):
        spec = make_tool_spec("lock.open", default_policy="confirm")
        rig = _rig(FakeCapability("lock", tools=(spec,)))

        record = await rig.host.invoke(_call("lock.open"), _ctx(source), None)

        assert record.outcome.status == "pending_confirmation"
        assert rig.caps["lock"].invocations == []

    @pytest.mark.asyncio
    async def test_proposal_source_runs_under_confirm(self):
        spec = make_tool_spec("lock.open", default_policy="confirm")
        rig = _rig(FakeCapability("lock", tools=(spec,)))

        record = await rig.host.invoke(_call("lock.open"), _ctx("proposal"), None)

        assert record.outcome.status == "ok"
        assert rig.caps["lock"].invocations[0][2].source == "proposal"

    @pytest.mark.asyncio
    async def test_proposal_source_still_stops_at_the_status_gate(self):
        spec = make_tool_spec("lock.open", default_policy="confirm")
        rig = _rig(FakeCapability("lock", tools=(spec,)))
        rig.caps["lock"].set_status("unavailable", "offline")

        record = await rig.host.invoke(_call("lock.open"), _ctx("proposal"), None)

        assert record.outcome.status == "unavailable"
        assert rig.caps["lock"].invocations == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("source", ["fastpath", "model"])
    async def test_escalated_call_proposes_and_its_approval_runs_once(self, source):
        spec = make_tool_spec("alarms.cancel", summarize=lambda args: "cancel every alarm")
        capability = FakeCapability("alarms", tools=(spec,))
        capability.escalate_fn = lambda tool, args: True
        rig = _rig(capability)

        pending = await rig.host.invoke(_call("alarms.cancel"), _ctx(source), None)
        rig.host.drain_events(SESSION)
        resolution = await rig.host.decide(SESSION, pending.outcome.proposal_id, "approve")

        assert pending.outcome.status == "pending_confirmation"
        assert [r.outcome.status for r in resolution.records] == ["ok"]
        assert len(capability.invocations) == 1
        assert capability.invocations[0][2].source == "proposal"
        assert [e["type"] for e in resolution.events] == ["action_resolved"]
        assert rig.host.drain_events(SESSION) == []
        assert rig.host.stats()["proposals"]["approved"] == 1
        again = await rig.host.decide(SESSION, pending.outcome.proposal_id, "approve")
        assert again.records == ()

    @pytest.mark.asyncio
    async def test_timeout_is_an_error(self):
        rig = _rig(_lights(), timeout_runner=_always_times_out)

        record = await rig.host.invoke(_call("lights.adjust", color="red"), _ctx(), None)

        assert record.outcome.status == "error"
        assert "timed out" in (record.outcome.reason or "")

    @pytest.mark.asyncio
    async def test_invoke_is_bounded_by_the_tool_timeout(self):
        timeouts = RecordingTimeouts()
        rig = _rig(_lights(), settings=HostSettings(tool_timeout_s=2.5), timeout_runner=timeouts)

        await rig.host.invoke(_call("lights.adjust", color="red"), _ctx(), None)

        assert timeouts.bounds == [2.5]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error", "status"),
        [
            (CapabilityInvocationError("provider said no"), "error"),
            (CapabilityUnavailableError("provider unreachable"), "unavailable"),
            (McpTransportError("connection reset"), "unavailable"),
            (TimeParseError("cannot parse 'whenever'"), "invalid_args"),
        ],
    )
    async def test_capability_error_is_normalised(self, error, status):
        rig = _rig(_lights())
        rig.caps["lights"].outcomes.append(error)

        record = await rig.host.invoke(_call("lights.adjust", color="red"), _ctx(), None)

        assert record.outcome.status == status
        assert record.outcome.reason == str(error)

    @pytest.mark.asyncio
    async def test_ok_call_returns_its_record(self):
        rig = _rig(_lights())
        rig.caps["lights"].outcomes.append(ToolOutcome(status="ok", summary="Lights red."))

        record = await rig.host.invoke(
            _call("lights.adjust", "call-9", color="red"), _ctx("fastpath", "call-9"), None
        )

        assert record.outcome == ToolOutcome(status="ok", summary="Lights red.")
        assert (record.tool_call_id, record.tool, record.capability, record.risk) == (
            "call-9",
            "lights.adjust",
            "lights",
            "reversible",
        )
        assert record.args_summary == "adjust the lights"
        ((tool, args, ctx),) = rig.caps["lights"].invocations
        assert tool == "lights.adjust"
        assert args == AdjustArgs(color="red")
        assert ctx.source == "fastpath"


# -- Redaction -------------------------------------------------------------------------------


class TestRedaction:
    @pytest.mark.asyncio
    async def test_secret_in_summary_is_redacted(self):
        rig = _rig(_lights())
        rig.secrets.register(Secret(TOKEN))
        rig.caps["lights"].outcomes.append(
            ToolOutcome(status="error", summary=f"failed with {TOKEN}", reason=f"bad {TOKEN}")
        )

        record = await rig.host.invoke(_call("lights.adjust", color="red"), _ctx(), None)

        assert TOKEN not in record.outcome.summary
        assert TOKEN not in (record.outcome.reason or "")
        assert "***" in record.outcome.summary
        assert "***" in (record.outcome.reason or "")

    @pytest.mark.asyncio
    async def test_secret_in_capability_error_reason_is_redacted(self):
        rig = _rig(_lights())
        rig.secrets.register(Secret(TOKEN))
        rig.caps["lights"].outcomes.append(CapabilityInvocationError(f"401 for {TOKEN}"))

        record = await rig.host.invoke(_call("lights.adjust", color="red"), _ctx(), None)

        assert record.outcome.reason == "401 for ***"

    def test_redact_is_exposed_for_the_handler(self):
        rig = _rig(_lights())
        rig.secrets.register(Secret(TOKEN))

        assert rig.host.redact(f"error: {TOKEN}") == "error: ***"


# -- Lifecycle and statuses ------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_returns_while_a_start_is_pending(self):
        slow = FakeCapability("slow")
        slow.start_gate = asyncio.Event()
        broken = FakeCapability("broken")
        broken.start_error = CapabilityUnavailableError("database locked")
        rig = _rig(slow, broken)

        await asyncio.wait_for(rig.host.start(), 1.0)
        waiter = asyncio.create_task(rig.host.wait_started())
        for _ in range(5):
            await asyncio.sleep(0)

        assert not slow.started
        assert not waiter.done()
        statuses = {s.name: s for s in rig.host.statuses()}
        assert statuses["broken"].state == "unavailable"
        assert statuses["broken"].reason == "database locked"

        slow.start_gate.set()
        await asyncio.wait_for(waiter, 1.0)

        assert slow.started
        assert {s.name: s.state for s in rig.host.statuses()} == {
            "slow": "available",
            "broken": "unavailable",
        }

    @pytest.mark.asyncio
    async def test_wait_started_returns_at_once_with_no_capabilities(self):
        rig = _rig()

        await asyncio.wait_for(rig.host.wait_started(), 0.5)
        await rig.host.start()
        await asyncio.wait_for(rig.host.wait_started(), 0.5)

        assert rig.host.statuses() == []

    @pytest.mark.asyncio
    async def test_failed_start_makes_calls_unavailable(self):
        broken = FakeCapability("broken")
        broken.start_error = CapabilityUnavailableError("no token")
        rig = _rig(broken)
        await rig.host.start()
        await rig.host.wait_started()

        record = await rig.host.invoke(_call("broken.do"), _ctx("fastpath"), None)

        assert record.outcome.status == "unavailable"
        assert record.outcome.reason == "no token"

    @pytest.mark.asyncio
    async def test_unexpected_start_exception_is_logged_and_unavailable(self, caplog):
        broken = FakeCapability("broken")
        broken.start_error = RuntimeError("bug in start")
        rig = _rig(broken)

        with caplog.at_level(logging.ERROR):
            await rig.host.start()
            await rig.host.wait_started()

        (status,) = rig.host.statuses()
        assert (status.state, status.reason) == ("unavailable", "RuntimeError")
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "broken" in errors[0].getMessage()

    def test_statuses_pass_task_dead_through(self):
        rig = _rig(_alarms())
        rig.caps["alarms"].set_status("unavailable", "scheduler died", task_dead=True)

        assert rig.host.statuses() == [
            CapabilityStatus(
                name="alarms",
                state="unavailable",
                reason="scheduler died",
                checked_at=DEFAULT_START,
                task_dead=True,
            )
        ]

    @pytest.mark.asyncio
    async def test_stop_stops_every_capability(self):
        rig = _rig(_lights(), _music())
        await rig.host.start()
        await rig.host.wait_started()

        await rig.host.stop()

        assert rig.caps["lights"].stopped
        assert rig.caps["music"].stopped


def _unfinished_timers() -> list[asyncio.Task[object]]:
    return [
        task
        for task in asyncio.all_tasks()
        if not task.done() and task.get_coro().__qualname__ == "ProposalStore._expire_after_ttl"
    ]


class _FailingStop(FakeCapability):
    """A capability whose first `stop` raises `RuntimeError`, as a bug outside the contract."""

    async def stop(self) -> None:
        if not self.stopped:
            self.stopped = True
            raise RuntimeError("bug in stop")


class TestStopClosesProposals:
    @pytest.mark.asyncio
    async def test_stop_leaves_no_timer_and_announces_no_expiry(self):
        rig = _rig(_lights())
        await _propose(rig, ("lights.power", {"target": "bedroom"}))
        await rig.clock.wait_for_sleepers(1)
        assert len(_unfinished_timers()) == 1

        await rig.host.stop()

        assert _unfinished_timers() == []
        assert rig.host.stats()["proposals"]["expired"] == 0
        await rig.clock.advance(120)
        assert rig.host.stats()["proposals"]["expired"] == 0
        assert rig.sink.of_type("action_resolved") == []

    @pytest.mark.asyncio
    async def test_stop_raising_outside_the_contract_still_closes_the_store(self):
        failing = _FailingStop("failing", tools=(make_tool_spec("failing.do"),))
        rig = _rig(_lights(), failing)
        await _propose(rig, ("lights.power", {"target": "bedroom"}))

        with pytest.raises(RuntimeError, match="bug in stop"):
            await rig.host.stop()

        assert rig.caps["lights"].stopped
        assert _unfinished_timers() == []

    @pytest.mark.asyncio
    async def test_run_actions_after_stop_records_the_held_call_as_an_error(self):
        rig = _rig(_lights())
        await rig.host.stop()
        plan = rig.host.plan_turn(SESSION, "lights", [])

        result = await rig.host.run_actions(
            SESSION,
            "turn-1",
            "lights",
            plan,
            ScriptedDecode(_decoded(("lights.power", {"target": "bedroom"}))),
        )

        (record,) = result.records
        assert record.outcome.status == "error"
        assert record.outcome.reason == "proposal store is closed"
        assert record.outcome.proposal_id is None
        assert rig.caps["lights"].invocations == []
        assert [e["type"] for e in rig.host.drain_events(SESSION)] == []
        assert await _log_rows(rig.db) == [
            ("lights.power", "lights", "model", "error", "proposal store is closed")
        ]
        assert _unfinished_timers() == []

    @pytest.mark.asyncio
    async def test_decide_after_stop_finds_no_live_proposal(self):
        rig = _rig(_lights())
        proposal_id = await _propose(rig, ("lights.power", {"target": "bedroom"}))
        await rig.host.stop()

        resolution = await rig.host.decide(SESSION, proposal_id, "approve")

        assert resolution.handled is False
        assert resolution.reply == NO_LIVE_PROPOSAL_REPLY
        assert rig.caps["lights"].invocations == []


# -- pre_llm ---------------------------------------------------------------------------------


class TestPreLlm:
    @pytest.mark.asyncio
    async def test_ringing_item_handles_stop_and_the_proposal_stays(self):
        alarms = _alarms()
        alarms.ringing = True
        alarms.active_replies["stop"] = ToolOutcome(status="ok", summary="Alarm stopped.")
        rig = _rig(_lights(), alarms)
        proposal_id = await _propose(rig, ("lights.power", {"target": "bedroom"}))

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "Stop.")

        assert resolution is not None and resolution.handled
        assert resolution.reply == "Alarm stopped."
        assert alarms.active_utterances == [("stop", SESSION)]
        decided = await rig.host.decide(SESSION, proposal_id, "approve")
        assert [r.tool for r in decided.records] == ["lights.power"]

    @pytest.mark.asyncio
    async def test_stop_phrase_while_ringing_wins_over_a_fast_path_match(self):
        alarms = _alarms()
        alarms.ringing = True
        alarms.active_replies["turn off the alarm"] = ToolOutcome(
            status="ok", summary="Alarm stopped."
        )
        lights = _lights(
            intents={"turn off the alarm": FastIntent("lights.power", {"target": "alarm"})}
        )
        rig = _rig(lights, alarms)

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "Turn off the alarm")

        assert resolution is not None and resolution.reply == "Alarm stopped."
        assert lights.invocations == []
        assert lights.intent_queries == []

    @pytest.mark.asyncio
    async def test_approval_runs_every_bundled_call_once_in_order(self):
        rig = _rig(_lights(), _music())
        rig.caps["lights"].outcomes.append(ToolOutcome(status="ok", summary="Lights off."))
        rig.caps["music"].outcomes.append(ToolOutcome(status="ok", summary="Playing jazz."))
        await _propose(rig, ("lights.power", {"target": "bedroom"}), ("music.play", {}))

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "yes please")
        again = await rig.host.pre_llm(SESSION, "turn-2", "yes")

        assert resolution is not None and resolution.handled
        assert [r.tool for r in resolution.records] == ["lights.power", "music.play"]
        assert {r.outcome.status for r in resolution.records} == {"ok"}
        assert len(rig.caps["lights"].invocations) == 1
        assert len(rig.caps["music"].invocations) == 1
        assert rig.caps["lights"].invocations[0][2].source == "proposal"
        assert resolution.reply == "Lights off. Playing jazz."
        assert again is None

    @pytest.mark.asyncio
    async def test_reject_phrase_rejects_the_whole_bundle(self):
        rig = _rig(_lights(), _music())
        await _propose(rig, ("lights.power", {"target": "bedroom"}), ("music.play", {}))

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "No thanks.")
        later = await rig.host.pre_llm(SESSION, "turn-2", "yes")

        assert resolution is not None and resolution.handled
        assert resolution.reply == REJECTED_REPLY
        assert resolution.records == ()
        assert later is None
        assert rig.caps["lights"].invocations == []
        assert rig.caps["music"].invocations == []

    @pytest.mark.asyncio
    async def test_yes_and_approves_and_returns_the_remainder(self):
        rig = _rig(_lights())
        await _propose(rig, ("lights.power", {"target": "bedroom"}))

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "Yes, and dim them")

        assert resolution is not None and resolution.handled
        assert resolution.remainder == "dim them"
        assert len(rig.caps["lights"].invocations) == 1

    @pytest.mark.asyncio
    async def test_remainder_pass_runs_step_one_while_an_item_rings(self):
        alarms = _alarms()
        alarms.active_replies["turn off the alarm"] = ToolOutcome(
            status="ok", summary="Alarm stopped."
        )
        lights = _lights(
            intents={"turn off the alarm": FastIntent("lights.power", {"target": "alarm"})}
        )
        rig = _rig(lights, alarms)
        await _propose(rig, ("lights.power", {"target": "bedroom"}))
        first = await rig.host.pre_llm(SESSION, "turn-1", "yes, and turn off the alarm")
        alarms.ringing = True

        second = await rig.host.pre_llm(SESSION, "turn-1", first.remainder)

        assert first.remainder == "turn off the alarm"
        assert second is not None and second.reply == "Alarm stopped."
        assert [inv[1] for inv in lights.invocations] == [TargetArgs(target="bedroom")]

    @pytest.mark.asyncio
    async def test_remainder_pass_without_a_live_proposal_reaches_the_fast_path(self):
        lights = _lights(intents={"dim them": FastIntent("lights.adjust", {"brightness": 30})})
        rig = _rig(lights)
        await _propose(rig, ("lights.power", {"target": "bedroom"}))
        first = await rig.host.pre_llm(SESSION, "turn-1", "yes and dim them")

        second = await rig.host.pre_llm(SESSION, "turn-1", first.remainder)

        assert second is not None and second.handled
        assert [r.tool for r in second.records] == ["lights.adjust"]
        assert lights.invocations[-1][2].source == "fastpath"
        assert rig.host.drain_events(SESSION)[-1]["decision"] == "approved"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("utterance", ["yeah but only the lamp", "what's the weather"])
    async def test_other_utterance_supersedes_then_reaches_the_fast_path(self, utterance):
        rig = _rig(_lights())
        proposal_id = await _propose(rig, ("lights.power", {"target": "bedroom"}))

        resolution = await rig.host.pre_llm(SESSION, "turn-1", utterance)
        later = await rig.host.pre_llm(SESSION, "turn-2", "yes")

        assert resolution is None
        assert later is None
        assert rig.caps["lights"].invocations == []
        assert len(rig.caps["lights"].intent_queries) == 2
        (event,) = rig.host.drain_events(SESSION)
        assert (event["type"], event["proposal_id"], event["decision"]) == (
            "action_resolved",
            proposal_id,
            "superseded",
        )

    @pytest.mark.asyncio
    async def test_fast_path_runs_a_single_match(self):
        lights = _lights(
            intents={"dim the lights": FastIntent("lights.adjust", {"brightness": 30})}
        )
        rig = _rig(lights, _music())
        lights.outcomes.append(ToolOutcome(status="ok", summary="Lights dimmed."))

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "Hey MIST, dim the lights")

        assert resolution is not None and resolution.handled
        assert resolution.reply == "Lights dimmed."
        assert resolution.remainder is None
        (record,) = resolution.records
        assert record.outcome.status == "ok"
        ((_, args, ctx),) = lights.invocations
        assert args == AdjustArgs(brightness=30)
        assert (ctx.source, ctx.turn_id, ctx.tool_call_id) == (
            "fastpath",
            "turn-1",
            record.tool_call_id,
        )

    @pytest.mark.asyncio
    async def test_fast_path_with_two_matches_returns_none(self):
        intent_text = "turn off the music"
        lights = _lights(intents={intent_text: FastIntent("lights.power", {"target": "music"})})
        music = _music(intents={intent_text: FastIntent("music.control", {})})
        rig = _rig(lights, music)

        assert await rig.host.pre_llm(SESSION, "turn-1", intent_text) is None
        assert lights.invocations == []
        assert music.invocations == []

    @pytest.mark.asyncio
    async def test_fast_path_switched_off_asks_no_capability(self):
        lights = _lights(
            intents={"dim the lights": FastIntent("lights.adjust", {"brightness": 30})}
        )
        rig = _rig(lights, settings=HostSettings(fast_path=False))

        assert await rig.host.pre_llm(SESSION, "turn-1", "dim the lights") is None
        assert lights.intent_queries == []

    @pytest.mark.asyncio
    async def test_fast_path_to_an_unavailable_capability_gives_its_reply(self):
        lights = _lights(
            intents={"dim the lights": FastIntent("lights.adjust", {"brightness": 30})}
        )
        lights.set_status("unavailable", "Home Assistant unreachable")
        rig = _rig(lights)

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "dim the lights")

        assert resolution is not None and resolution.handled
        assert resolution.reply == "Lights is unavailable: Home Assistant unreachable."
        assert lights.invocations == []

    @pytest.mark.asyncio
    async def test_fast_path_with_a_rejected_target_gives_its_reply(self):
        text = "turn off the garage lights"
        lights = _lights(intents={text: FastIntent("lights.power", {"target": "garage"})})
        lights.outcomes.append(
            ToolOutcome(status="invalid_args", summary="I don't know garage. Known: bedroom.")
        )
        rig = _rig(lights)

        resolution = await rig.host.pre_llm(SESSION, "turn-1", text)

        assert resolution is not None
        assert resolution.reply == "I don't know garage. Known: bedroom."

    @pytest.mark.asyncio
    async def test_nothing_to_resolve_returns_none(self):
        rig = _rig(_lights(), _music())

        assert await rig.host.pre_llm(SESSION, "turn-1", "tell me a story") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("step", ["active", "approval", "fast path"])
    async def test_handled_resolution_carries_no_events(self, step):
        alarms = _alarms()
        alarms.active_replies["stop"] = ToolOutcome(status="ok", summary="Alarm stopped.")
        lights = _lights(intents={"stop": FastIntent("lights.adjust", {"brightness": 30})})
        rig = _rig(lights, alarms)
        if step == "active":
            alarms.ringing = True
        utterance = "stop"
        if step == "approval":
            await _propose(rig, ("lights.power", {"target": "bedroom"}))
            utterance = "yes"

        resolution = await rig.host.pre_llm(SESSION, "turn-1", utterance)

        assert resolution is not None and resolution.handled
        assert resolution.records
        assert resolution.events == ()


class _RaisingHook(FakeCapability):
    """A ringing capability whose `on_active_utterance` raises `error`."""

    def __init__(self, error: BaseException) -> None:
        super().__init__("alarms", tools=(make_tool_spec("alarms.list"),))
        self.ringing = True
        self.error = error

    async def on_active_utterance(self, utterance: str, session_id: str) -> ToolOutcome | None:
        self.active_utterances.append((utterance, session_id))
        raise self.error


async def _log_rows(db: CapabilitiesDb) -> list[tuple[object, ...]]:
    """The action log's rows; none when no write has created its table yet."""

    def read(conn) -> list[tuple[object, ...]]:
        table = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'action_log'"
        if conn.execute(table).fetchone() is None:
            return []
        return conn.execute(
            "SELECT tool, capability, source, status, reason FROM action_log ORDER BY id"
        ).fetchall()

    return await db.run(read)


class TestActiveHookContainment:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error", "status", "reply"),
        [
            (
                CapabilityInvocationError(f"store locked {TOKEN}"),
                "error",
                "That didn't work: store locked ***.",
            ),
            (
                CapabilityUnavailableError(f"database gone {TOKEN}"),
                "unavailable",
                "Alarms is unavailable: database gone ***.",
            ),
        ],
    )
    async def test_capability_error_is_normalised_redacted_recorded_and_answered(
        self, error, status, reply
    ):
        alarms = _RaisingHook(error)
        lights = _lights(intents={"stop": FastIntent("lights.adjust", {"brightness": 30})})
        rig = _rig(lights, alarms)
        rig.secrets.register(Secret(TOKEN))

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "Stop.")

        assert resolution is not None and resolution.handled
        (record,) = resolution.records
        assert record.tool == "alarms.on_active_utterance"
        assert record.outcome.status == status
        assert TOKEN not in record.outcome.summary
        assert record.outcome.reason == str(error).replace(TOKEN, "***")
        assert resolution.reply == reply
        assert resolution.events == ()
        assert alarms.active_utterances == [("stop", SESSION)]
        assert lights.intent_queries == []
        assert rig.host.stats()["invocations"][status] == 1
        assert rig.host.stats()["by_source"]["fastpath"] == 1
        assert await _log_rows(rig.db) == [
            ("alarms.on_active_utterance", "alarms", "fastpath", status, record.outcome.reason)
        ]

    @pytest.mark.asyncio
    async def test_hook_overrunning_the_call_timeout_gives_the_timeout_outcome(self):
        alarms = _alarms()
        alarms.ringing = True
        alarms.active_replies["stop"] = ToolOutcome(status="ok", summary="Alarm stopped.")
        rig = _rig(
            alarms, settings=HostSettings(tool_timeout_s=2.5), timeout_runner=_always_times_out
        )

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "stop")

        assert resolution is not None and resolution.handled
        (record,) = resolution.records
        assert record.outcome.status == "error"
        assert record.outcome.reason == "timed out after 2.5 s"
        assert record.outcome.summary == "alarms.on_active_utterance took too long."
        assert resolution.reply == "That didn't work: timed out after 2.5 s."
        assert rig.host.stats()["invocations"]["error"] == 1

    @pytest.mark.asyncio
    async def test_hook_is_bounded_by_the_tool_timeout(self):
        timeouts = RecordingTimeouts()
        alarms = _alarms()
        alarms.ringing = True
        alarms.active_replies["stop"] = ToolOutcome(status="ok", summary="Alarm stopped.")
        rig = _rig(alarms, settings=HostSettings(tool_timeout_s=2.5), timeout_runner=timeouts)

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "stop")

        assert resolution is not None and resolution.reply == "Alarm stopped."
        assert timeouts.bounds == [2.5]

    @pytest.mark.asyncio
    async def test_other_exception_propagates_and_records_nothing(self):
        alarms = _RaisingHook(RuntimeError("bug in the hook"))
        rig = _rig(_lights(), alarms)

        with pytest.raises(RuntimeError, match="bug in the hook"):
            await rig.host.pre_llm(SESSION, "turn-1", "stop")

        stats = rig.host.stats()
        assert set(stats["invocations"].values()) == {0}
        assert set(stats["by_source"].values()) == {0}
        assert await _log_rows(rig.db) == []
        assert rig.host.drain_events(SESSION) == []


# -- plan_turn -------------------------------------------------------------------------------


def _catalog_capability(names: Sequence[str], allowance: int) -> FakeCapability:
    capability = FakeCapability("lights", tools=(POWER,))
    capability.refine_allowance_tokens = allowance
    capability.catalog = list(names)  # type: ignore[attr-defined]

    def refine(tool: str, schema: dict[str, object]) -> dict[str, object]:
        properties = dict(schema["properties"])  # type: ignore[arg-type]
        properties["target"] = {"type": "string", "enum": list(capability.catalog)}  # type: ignore[attr-defined]
        return dict(schema, properties=properties)

    capability.refine_fn = refine
    return capability


class TestPlanTurn:
    def test_nothing_routed_gives_no_decode(self):
        rig = _rig(_lights(), _music())

        plan = rig.host.plan_turn(SESSION, "tell me a story", [])

        assert plan.decode is None
        assert plan.offered == frozenset()
        assert plan.route.capabilities == ()

    def test_routed_turn_offers_routed_tools_and_builds_the_decode(self):
        rig = _rig(_lights(), _music(), _alarms())
        recent = [("user", "hi"), ("assistant", "hello")]

        plan = rig.host.plan_turn(SESSION, "dim the lamp", recent)

        assert plan.route.capabilities == ("lights",)
        assert plan.offered == frozenset({"lights.power", "lights.adjust"})
        expected = ActionDecoder().build(
            [(POWER, to_args_schema(POWER)), (ADJUST, to_args_schema(ADJUST))],
            recent,
            "dim the lamp",
        )
        assert plan.decode == expected

    def test_refinement_within_its_allowance_is_used(self):
        capability = _catalog_capability(["bedroom", "kitchen"], allowance=40)
        rig = _rig(capability)

        plan = rig.host.plan_turn(SESSION, "lights off", [])

        expected = ActionDecoder().build(
            [(POWER, to_args_schema(POWER, capability.refine_schema))], [], "lights off"
        )
        assert plan.decode == expected
        assert "bedroom" in json.dumps(plan.decode.response_schema)

    def test_refinement_over_its_allowance_falls_back_with_one_warning_per_change(self, caplog):
        capability = _catalog_capability([f"room number {i}" for i in range(20)], allowance=10)
        rig = _rig(capability)
        unrefined = ActionDecoder().build([(POWER, to_args_schema(POWER))], [], "lights off")

        with caplog.at_level(logging.WARNING, logger="backend.capabilities.host"):
            first = rig.host.plan_turn(SESSION, "lights off", [])
            second = rig.host.plan_turn(SESSION, "lights off", [])
            warnings_before_change = len(caplog.records)
            capability.catalog.append("one more room")  # type: ignore[attr-defined]
            third = rig.host.plan_turn(SESSION, "lights off", [])

        assert first.decode == unrefined
        assert second.decode == unrefined
        assert third.decode == unrefined
        assert warnings_before_change == 1
        assert len(caplog.records) == 2
        assert "lights" in caplog.records[0].getMessage()

    def test_refinement_the_grammar_cannot_express_falls_back_with_one_warning(self, caplog):
        capability = FakeCapability("lights", tools=(POWER,))

        def refine(tool: str, schema: dict[str, object]) -> dict[str, object]:
            properties = dict(schema["properties"])  # type: ignore[arg-type]
            properties["target"] = {"type": "string", "pattern": "^[a-z]+$"}
            return dict(schema, properties=properties)

        capability.refine_fn = refine
        rig = _rig(capability)

        with caplog.at_level(logging.WARNING, logger="backend.capabilities.host"):
            first = rig.host.plan_turn(SESSION, "lights off", [])
            rig.host.plan_turn(SESSION, "lights off", [])

        assert first.decode == ActionDecoder().build(
            [(POWER, to_args_schema(POWER))], [], "lights off"
        )
        assert len(caplog.records) == 1


# -- run_actions -----------------------------------------------------------------------------


class TestRunActions:
    @pytest.mark.asyncio
    async def test_decode_is_bounded_and_calls_run_in_order_as_model_calls(self):
        timeouts = RecordingTimeouts()
        settings = HostSettings(decode_timeout_s=7.5, tool_timeout_s=3.0)
        rig = _rig(_lights(), _music(), settings=settings, timeout_runner=timeouts)
        plan = rig.host.plan_turn(SESSION, "lights and music", [])
        decode = ScriptedDecode(
            _decoded(("music.control", {}), ("lights.adjust", {"color": "red"}))
        )

        result = await rig.host.run_actions(SESSION, "turn-1", "lights and music", plan, decode)

        assert decode.requests == [plan.decode]
        assert timeouts.bounds == [7.5, 3.0, 3.0]
        assert [r.tool for r in result.records] == ["music.control", "lights.adjust"]
        assert rig.caps["music"].invocations[0][2].source == "model"
        assert rig.caps["lights"].invocations[0][2].source == "model"
        assert rig.caps["lights"].invocations[0][2].turn_id == "turn-1"

    @pytest.mark.asyncio
    async def test_invalid_call_is_repaired_once_and_runs(self):
        rig = _rig(_lights())
        plan = rig.host.plan_turn(SESSION, "lights warmer", [])
        decode = ScriptedDecode(
            _decoded(("lights.adjust", {})),
            DecodeReply(content='{"brightness": 40}', finish_reason="stop"),
        )

        result = await rig.host.run_actions(SESSION, "turn-1", "lights warmer", plan, decode)

        assert [r.outcome.status for r in result.records] == ["ok"]
        assert rig.caps["lights"].invocations[0][1] == AdjustArgs(brightness=40)
        assert len(decode.requests) == 2
        repair = decode.requests[1]
        assert repair.response_schema == to_args_schema(ADJUST)
        assert "lights warmer" in str(repair.messages)
        assert "give brightness or color" in str(repair.messages)

    @pytest.mark.asyncio
    async def test_repair_that_still_fails_is_invalid_args(self):
        rig = _rig(_lights())
        plan = rig.host.plan_turn(SESSION, "lights warmer", [])
        decode = ScriptedDecode(
            _decoded(("lights.adjust", {})),
            DecodeReply(content="{}", finish_reason="stop"),
        )

        result = await rig.host.run_actions(SESSION, "turn-1", "lights warmer", plan, decode)

        assert [r.outcome.status for r in result.records] == ["invalid_args"]
        assert len(decode.requests) == 2
        assert rig.caps["lights"].invocations == []

    @pytest.mark.asyncio
    async def test_confirm_calls_are_bundled_while_allowed_calls_run(self):
        rig = _rig(_lights(), _music())
        plan = rig.host.plan_turn(SESSION, "lights and music", [])
        decode = ScriptedDecode(
            _decoded(
                ("lights.power", {"target": "bedroom"}),
                ("music.control", {}),
                ("music.play", {}),
            )
        )

        result = await rig.host.run_actions(SESSION, "turn-1", "lights and music", plan, decode)

        statuses = [r.outcome.status for r in result.records]
        assert statuses == ["pending_confirmation", "ok", "pending_confirmation"]
        assert [inv[0] for inv in rig.caps["music"].invocations] == ["music.control"]
        assert rig.caps["lights"].invocations == []
        proposal_ids = {r.outcome.proposal_id for r in result.records if r.outcome.proposal_id}
        assert len(proposal_ids) == 1
        (event,) = rig.host.drain_events(SESSION)
        assert event["type"] == "action_proposed"
        assert event["summary"] == "turn off the bedroom lights, then play jazz"
        assert [c["tool"] for c in event["calls"]] == ["lights.power", "music.play"]
        assert result.reply == (
            "music.control done."
            " Should I turn off the bedroom lights, then play jazz? Say yes or no."
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reply",
        [
            DecodeReply(content="not json", finish_reason="stop"),
            DecodeReply(content='{"calls": [], "also_reply"', finish_reason="length"),
        ],
    )
    async def test_decode_error_gives_no_records_and_the_notice(self, reply):
        rig = _rig(_lights())
        plan = rig.host.plan_turn(SESSION, "lights", [])

        result = await rig.host.run_actions(
            SESSION, "turn-1", "lights", plan, ScriptedDecode(reply)
        )

        assert result.records == ()
        assert result.also_reply is True
        assert result.notices == (DECODE_FAILURE_NOTICE,)
        assert result.reply is None
        assert result.block == ""

    @pytest.mark.asyncio
    async def test_decode_timeout_gives_no_records_and_the_notice(self):
        rig = _rig(_lights(), timeout_runner=_always_times_out)
        plan = rig.host.plan_turn(SESSION, "lights", [])

        result = await rig.host.run_actions(
            SESSION, "turn-1", "lights", plan, ScriptedDecode(_decoded())
        )

        assert result.records == ()
        assert result.also_reply is True
        assert result.notices == (DECODE_FAILURE_NOTICE,)

    def test_decode_failure_notice_text(self):
        # It says the meant action did not run, and not that nothing ran: a "yes, and ..."
        # turn may already have run approved calls.
        assert DECODE_FAILURE_NOTICE == (
            "(I could not work out which action you meant, so I did not run it.)"
        )

    @pytest.mark.asyncio
    async def test_yes_and_turn_with_a_failed_decode_ends_with_the_notice(self):
        rig = _rig(_lights())
        rig.caps["lights"].outcomes.append(ToolOutcome(status="ok", summary="Lights off."))
        await _propose(rig, ("lights.power", {"target": "bedroom"}))

        approved = await rig.host.pre_llm(SESSION, "turn-1", "yes, and dim the lights")
        assert approved is not None and approved.remainder == "dim the lights"
        second = await rig.host.pre_llm(SESSION, "turn-1", approved.remainder)
        plan = rig.host.plan_turn(SESSION, approved.remainder, [])
        failed = await rig.host.run_actions(
            SESSION,
            "turn-1",
            approved.remainder,
            plan,
            ScriptedDecode(DecodeReply(content="not json", finish_reason="stop")),
        )

        assert second is None
        assert plan.decode is not None
        assert [(r.tool, r.outcome.status) for r in approved.records] == [("lights.power", "ok")]
        assert failed.records == ()
        # The turn's text, in the order the handler sends it: the approval's reply, then the
        # remainder's notices.
        turn = [approved.reply, *failed.notices]
        assert turn == ["Lights off.", DECODE_FAILURE_NOTICE]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("also_reply", [True, False])
    async def test_also_reply_passes_through_and_texts_come_from_replies(self, also_reply):
        rig = _rig(_lights())
        rig.caps["lights"].set_status("degraded")
        rig.caps["lights"].outcomes.append(CapabilityInvocationError("bulb missing"))
        plan = rig.host.plan_turn(SESSION, "lights", [])
        decode = ScriptedDecode(
            _decoded(
                ("lights.adjust", {"color": "red"}),
                ("lights.power", {"target": "bedroom"}),
                also_reply=also_reply,
            )
        )

        result = await rig.host.run_actions(SESSION, "turn-1", "lights", plan, decode)

        assert result.also_reply is also_reply
        assert result.reply == reply_for(result.records)
        assert result.block == actions_block(result.records)
        assert result.notices == notices_for(result.records)
        assert len(result.notices) == 2
        assert result.events == ()

    @pytest.mark.asyncio
    async def test_no_calls_is_an_empty_result(self):
        rig = _rig(_lights())
        plan = rig.host.plan_turn(SESSION, "lights are nice", [])

        result = await rig.host.run_actions(
            SESSION, "turn-1", "lights are nice", plan, ScriptedDecode(_decoded(also_reply=True))
        )

        assert (result.records, result.also_reply, result.block, result.notices) == (
            (),
            True,
            "",
            (),
        )


# -- stats -----------------------------------------------------------------------------------


class TestStats:
    def test_initial_shape(self):
        rig = _rig(_lights())

        stats = rig.host.stats()

        assert stats == {
            "since": DEFAULT_START.isoformat(),
            "invocations": {
                "ok": 0,
                "invalid_args": 0,
                "pending_confirmation": 0,
                "unavailable": 0,
                "error": 0,
                "denied": 0,
            },
            "by_source": {"fastpath": 0, "model": 0, "proposal": 0},
            "decodes": 0,
            "decode_failures": 0,
            "repairs": 0,
            "repairs_ok": 0,
            "routed_no_call": 0,
            "proposals": {"approved": 0, "rejected": 0, "expired": 0, "superseded": 0},
            "status_watcher": "stopped",
        }

    @pytest.mark.asyncio
    async def test_counts_invocations_by_status(self):
        rig = _rig(_lights())
        rig.caps["lights"].outcomes.append(CapabilityInvocationError("no"))

        await rig.host.invoke(_call("lights.adjust", color="red"), _ctx(), None)
        await rig.host.invoke(_call("lights.adjust", color="red"), _ctx(), None)
        await rig.host.invoke(_call("lights.nope"), _ctx(), None)

        invocations = rig.host.stats()["invocations"]
        assert (invocations["error"], invocations["ok"], invocations["invalid_args"]) == (1, 1, 1)

    @pytest.mark.asyncio
    async def test_counts_invocations_by_source(self):
        rig = _rig(_lights())

        await rig.host.invoke(_call("lights.adjust", color="red"), _ctx("fastpath"), None)
        await rig.host.invoke(_call("lights.adjust", color="red"), _ctx("model"), None)
        await rig.host.invoke(_call("lights.adjust", color="red"), _ctx("proposal"), None)
        await rig.host.invoke(_call("lights.adjust", color="red"), _ctx("proposal"), None)

        assert rig.host.stats()["by_source"] == {"fastpath": 1, "model": 1, "proposal": 2}

    @pytest.mark.asyncio
    async def test_counts_decodes(self):
        rig = _rig(_lights())
        plan = rig.host.plan_turn(SESSION, "lights", [])

        await rig.host.run_actions(SESSION, "t1", "lights", plan, ScriptedDecode(_decoded()))
        await rig.host.run_actions(SESSION, "t2", "lights", plan, ScriptedDecode(_decoded()))

        assert rig.host.stats()["decodes"] == 2

    @pytest.mark.asyncio
    async def test_counts_decode_failures(self):
        rig = _rig(_lights())
        plan = rig.host.plan_turn(SESSION, "lights", [])
        bad = DecodeReply(content="{", finish_reason="stop")

        await rig.host.run_actions(SESSION, "t1", "lights", plan, ScriptedDecode(bad))

        assert rig.host.stats()["decode_failures"] == 1

    @pytest.mark.asyncio
    async def test_counts_repairs(self):
        rig = _rig(_lights())
        plan = rig.host.plan_turn(SESSION, "lights", [])
        decode = ScriptedDecode(
            _decoded(("lights.adjust", {})), DecodeReply(content="{}", finish_reason="stop")
        )

        await rig.host.run_actions(SESSION, "t1", "lights", plan, decode)

        assert (rig.host.stats()["repairs"], rig.host.stats()["repairs_ok"]) == (1, 0)

    @pytest.mark.asyncio
    async def test_counts_repairs_that_validated(self):
        rig = _rig(_lights())
        plan = rig.host.plan_turn(SESSION, "lights", [])
        decode = ScriptedDecode(
            _decoded(("lights.adjust", {})),
            DecodeReply(content='{"color": "red"}', finish_reason="stop"),
        )

        await rig.host.run_actions(SESSION, "t1", "lights", plan, decode)

        assert (rig.host.stats()["repairs"], rig.host.stats()["repairs_ok"]) == (1, 1)

    @pytest.mark.asyncio
    async def test_counts_routed_turns_with_no_call(self):
        rig = _rig(_lights())
        plan = rig.host.plan_turn(SESSION, "lights", [])

        await rig.host.run_actions(SESSION, "t1", "lights", plan, ScriptedDecode(_decoded()))

        assert rig.host.stats()["routed_no_call"] == 1

    @pytest.mark.asyncio
    async def test_counts_approved_proposals(self):
        rig = _rig(_lights())
        await _propose(rig, ("lights.power", {"target": "bedroom"}))

        await rig.host.pre_llm(SESSION, "t1", "yes")

        assert rig.host.stats()["proposals"]["approved"] == 1

    @pytest.mark.asyncio
    async def test_counts_rejected_proposals(self):
        rig = _rig(_lights())
        proposal_id = await _propose(rig, ("lights.power", {"target": "bedroom"}))

        resolution = await rig.host.decide(SESSION, proposal_id, "reject")

        assert rig.host.stats()["proposals"]["rejected"] == 1
        assert [e["decision"] for e in resolution.events] == ["rejected"]

    @pytest.mark.asyncio
    async def test_counts_expired_proposals_and_emits_to_the_sink(self):
        rig = _rig(_lights(), settings=HostSettings(proposal_ttl_s=60.0))
        proposal_id = await _propose(rig, ("lights.power", {"target": "bedroom"}))

        await rig.clock.advance(60)

        assert rig.host.stats()["proposals"]["expired"] == 1
        (event,) = rig.sink.of_type("action_resolved")
        assert (event["proposal_id"], event["decision"], event["outcomes"]) == (
            proposal_id,
            "expired",
            [],
        )
        assert rig.host.drain_events(SESSION) == []

    @pytest.mark.asyncio
    async def test_counts_superseded_proposals(self):
        rig = _rig(_lights())
        await _propose(rig, ("lights.power", {"target": "bedroom"}))

        await rig.host.pre_llm(SESSION, "t1", "tell me a story")

        assert rig.host.stats()["proposals"]["superseded"] == 1

    @pytest.mark.asyncio
    async def test_new_proposal_supersedes_a_live_one(self):
        rig = _rig(_lights())
        first = await _propose(rig, ("lights.power", {"target": "bedroom"}))

        await rig.host.invoke(_call("lights.power", target="kitchen"), _ctx("model"), None)

        assert rig.host.stats()["proposals"]["superseded"] == 1
        events = rig.host.drain_events(SESSION)
        assert [(e["type"], e.get("decision")) for e in events] == [
            ("action_resolved", "superseded"),
            ("action_proposed", None),
        ]
        assert events[0]["proposal_id"] == first


# -- Concurrency on one loop -----------------------------------------------------------------


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_pre_llm_and_decide_race_executes_once_when_pre_llm_pops_first(self):
        rig = _rig(_lights())
        proposal_id = await _propose(rig, ("lights.power", {"target": "bedroom"}))
        rig.caps["lights"].invoke_gate = asyncio.Event()

        spoken = asyncio.create_task(rig.host.pre_llm(SESSION, "t1", "yes"))
        tapped = asyncio.create_task(rig.host.decide(SESSION, proposal_id, "approve"))
        for _ in range(5):
            await asyncio.sleep(0)
        rig.caps["lights"].invoke_gate.set()
        spoken_result, tapped_result = await asyncio.gather(spoken, tapped)

        assert len(rig.caps["lights"].invocations) == 1
        assert spoken_result is not None and len(spoken_result.records) == 1
        assert tapped_result.handled is False
        assert tapped_result.reply == NO_LIVE_PROPOSAL_REPLY
        assert tapped_result.records == ()

    @pytest.mark.asyncio
    async def test_pre_llm_and_decide_race_executes_once_when_decide_pops_first(self):
        rig = _rig(_lights())
        proposal_id = await _propose(rig, ("lights.power", {"target": "bedroom"}))
        rig.caps["lights"].invoke_gate = asyncio.Event()

        tapped = asyncio.create_task(rig.host.decide(SESSION, proposal_id, "approve"))
        spoken = asyncio.create_task(rig.host.pre_llm(SESSION, "t1", "yes"))
        for _ in range(5):
            await asyncio.sleep(0)
        rig.caps["lights"].invoke_gate.set()
        tapped_result, spoken_result = await asyncio.gather(tapped, spoken)

        assert len(rig.caps["lights"].invocations) == 1
        assert len(tapped_result.records) == 1
        assert spoken_result is None

    @pytest.mark.asyncio
    async def test_decide_with_another_sessions_id_is_refused(self):
        rig = _rig(_lights())
        proposal_id = await _propose(rig, ("lights.power", {"target": "bedroom"}))

        refused = await rig.host.decide(OTHER, proposal_id, "approve")
        owned = await rig.host.decide(SESSION, proposal_id, "approve")

        assert refused.handled is False
        assert refused.reply == NO_LIVE_PROPOSAL_REPLY
        assert refused.records == ()
        assert len(owned.records) == 1
        assert len(rig.caps["lights"].invocations) == 1

    @pytest.mark.asyncio
    async def test_decide_on_an_expired_proposal_is_refused_with_its_reason(self):
        rig = _rig(_lights())
        proposal_id = await _propose(rig, ("lights.power", {"target": "bedroom"}))
        await rig.clock.advance(60)

        resolution = await rig.host.decide(SESSION, proposal_id, "approve")

        assert resolution.handled is False
        assert resolution.reply is not None and "expired" in resolution.reply
        assert rig.caps["lights"].invocations == []

    @pytest.mark.asyncio
    async def test_events_from_decide_are_returned_not_buffered(self):
        rig = _rig(_lights())
        proposal_id = await _propose(rig, ("lights.power", {"target": "bedroom"}))

        resolution = await rig.host.decide(SESSION, proposal_id, "approve")

        (event,) = resolution.events
        assert event == {
            "type": "action_resolved",
            "proposal_id": proposal_id,
            "session_id": SESSION,
            "decision": "approved",
            "outcomes": [{"tool": "lights.power", "status": "ok", "summary": "lights.power done."}],
        }
        assert rig.host.drain_events(SESSION) == []
        assert rig.host.drain_events(OTHER) == []
        assert rig.sink.payloads == []

    @pytest.mark.asyncio
    async def test_drain_events_returns_only_that_sessions_events_in_order(self):
        rig = _rig(_lights())
        first = await rig.host.invoke(_call("lights.power", target="a"), _ctx("model"), None)
        other = await rig.host.invoke(
            _call("lights.power", target="b"), _ctx("model", session=OTHER), None
        )
        await rig.host.pre_llm(SESSION, "t1", "yes")

        mine = rig.host.drain_events(SESSION)
        theirs = rig.host.drain_events(OTHER)

        assert [(e["type"], e["proposal_id"]) for e in mine] == [
            ("action_proposed", first.outcome.proposal_id),
            ("action_resolved", first.outcome.proposal_id),
        ]
        assert mine[1]["decision"] == "approved"
        assert [(e["type"], e["proposal_id"]) for e in theirs] == [
            ("action_proposed", other.outcome.proposal_id)
        ]
        assert {e["session_id"] for e in mine} == {SESSION}
        assert rig.host.drain_events(SESSION) == []


# -- Power levels and tiers (T24) ------------------------------------------------------------

PROVIDER = cast(StreamingLLMProvider, object())
TIMER_PHRASE = "set a timer for five minutes"
TIMER_START = make_tool_spec("timers.start", summarize=lambda args: "start a five minute timer")


def _timers() -> FakeCapability:
    # The grammar reads the normalised text, in which number words are digits.
    return FakeCapability(
        "timers",
        tools=(TIMER_START,),
        intents={"set a timer for 5 minutes": FastIntent("timers.start", {})},
    )


async def _log_columns(db: CapabilitiesDb) -> list[tuple[object, ...]]:
    return await db.run(
        lambda conn: conn.execute(
            "SELECT tool, power_level, turn_ms FROM action_log ORDER BY id"
        ).fetchall()
    )


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


class TestPowerPort:
    def test_received_at_is_a_keyword_defaulting_to_the_call_time(self):
        for owner in (CapabilityTurnPort, CapabilityHost):
            for name in ("pre_llm", "plan_turn"):
                parameter = inspect.signature(getattr(owner, name)).parameters["received_at"]
                assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (owner, name)
                assert parameter.default is None, (owner, name)

    def test_power_and_tiers_are_keywords_whose_defaults_keep_full_and_the_decode(self):
        parameters = inspect.signature(CapabilityHost).parameters
        assert parameters["power"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["tiers"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["tiers"].default is None

        rig = _rig(_lights())

        assert rig.host.decode_available() is True
        assert rig.host.plan_turn(SESSION, "dim the lamp", []).decode is not None

    def test_decode_available_at_low_with_the_interaction_tier_present(self):
        tiers = FakeTiers({"interaction": PROVIDER})
        rig = _rig(_lights(), power=FakePower("low"), tiers=tiers)

        assert rig.host.decode_available() is True
        assert tiers.asked == ["interaction"]

    def test_decode_unavailable_at_full_when_the_interaction_tier_is_none(self):
        tiers = FakeTiers({"interaction": None, "smart": PROVIDER})
        rig = _rig(_lights(), power=FakePower("full"), tiers=tiers)

        assert rig.host.decode_available() is False

    @pytest.mark.asyncio
    async def test_plan_turn_builds_no_decode_while_it_is_unavailable(self):
        tiers = FakeTiers({})
        rig = _rig(_lights(), power=FakePower("off"), tiers=tiers)
        decode = ScriptedDecode()

        plan = rig.host.plan_turn(SESSION, "dim the lamp", [])
        result = await rig.host.run_actions(SESSION, "turn-1", "dim the lamp", plan, decode)

        assert plan.decode is None
        assert plan.offered == frozenset()
        assert decode.requests == []
        assert result.records == ()
        tiers.providers["interaction"] = PROVIDER
        assert rig.host.plan_turn(SESSION, "dim the lamp", []).decode is not None


class TestModelFreePathsAtOff:
    @pytest.mark.asyncio
    async def test_fast_path_timer_at_off_is_handled_with_no_decode(self):
        tiers = FakeTiers({})
        timers = _timers()
        timers.outcomes.append(ToolOutcome(status="ok", summary="Timer set for five minutes."))
        rig = _rig(timers, power=FakePower("off"), tiers=tiers)

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "Set a timer for five minutes")

        assert resolution is not None and resolution.handled
        assert resolution.reply == "Timer set for five minutes."
        (record,) = resolution.records
        assert (record.tool, record.outcome.status, record.power_level) == (
            "timers.start",
            "ok",
            "off",
        )
        assert tiers.asked == []
        assert rig.host.decode_available() is False

    @pytest.mark.asyncio
    async def test_ringing_stop_at_off_is_handled(self):
        alarms = _alarms()
        alarms.ringing = True
        alarms.active_replies["stop"] = ToolOutcome(status="ok", summary="Alarm stopped.")
        rig = _rig(alarms, power=FakePower("off"), tiers=FakeTiers({}))

        resolution = await rig.host.pre_llm(SESSION, "turn-1", "Stop.")

        assert resolution is not None and resolution.reply == "Alarm stopped."
        assert [r.power_level for r in resolution.records] == ["off"]

    @pytest.mark.asyncio
    async def test_proposal_yes_and_no_at_off_are_handled(self):
        power = FakePower("full")
        rig = _rig(
            _lights(), _music(), power=power, tiers=SingleModelTiers(lambda: PROVIDER, power)
        )
        await _propose(rig, ("lights.power", {"target": "bedroom"}))
        power.current = "off"

        approved = await rig.host.pre_llm(SESSION, "turn-1", "yes")
        power.current = "full"
        await _propose(rig, ("music.play", {}))
        power.current = "off"
        rejected = await rig.host.pre_llm(SESSION, "turn-2", "no")

        assert approved is not None and approved.handled
        assert [(r.tool, r.power_level) for r in approved.records] == [("lights.power", "off")]
        assert rejected is not None and rejected.reply == REJECTED_REPLY
        assert rig.caps["music"].invocations == []


class TestLevelAndLatencyRecords:
    @pytest.mark.asyncio
    async def test_every_record_and_row_carries_the_level_read_at_its_call(self):
        power = FakePower("low")
        rig = _rig(_lights(), power=power, tiers=FakeTiers({"interaction": PROVIDER}))

        first = await rig.host.invoke(_call("lights.adjust", brightness=30), _ctx("model"), None)
        power.current = "full"
        second = await rig.host.invoke(_call("nope.do", call_id="call-2"), _ctx("model"), None)

        assert (first.power_level, second.power_level) == ("low", "full")
        rows = await _log_columns(rig.db)
        assert [row[:2] for row in rows] == [("lights.adjust", "low"), ("nope.do", "full")]

    @pytest.mark.asyncio
    async def test_turn_ms_runs_from_received_at_to_the_fast_path_outcome(self):
        rig = _rig(_timers())

        await rig.host.pre_llm(SESSION, "turn-1", TIMER_PHRASE, received_at=time.monotonic() - 0.4)
        await rig.host.pre_llm(SESSION, "turn-2", TIMER_PHRASE)

        (late, prompt) = await _log_columns(rig.db)
        assert late[0] == "timers.start" and late[2] >= 400
        assert 0 <= prompt[2] < 400

    @pytest.mark.asyncio
    async def test_turn_ms_runs_from_plan_turns_received_at_to_the_model_outcome(self):
        rig = _rig(_lights())
        received_at = time.monotonic() - 0.4

        plan = rig.host.plan_turn(SESSION, "dim the lamp", [], received_at=received_at)
        await rig.host.run_actions(
            SESSION,
            "turn-1",
            "dim the lamp",
            plan,
            ScriptedDecode(_decoded(("lights.adjust", {"brightness": 30}))),
        )

        ((tool, level, turn_ms),) = await _log_columns(rig.db)
        assert (tool, level) == ("lights.adjust", "full")
        assert turn_ms >= 400


class TestLevelInStatus:
    @pytest.mark.asyncio
    async def test_status_payload_carries_the_level_and_decode_availability(self):
        rig = _rig(_lights(), power=FakePower("low"), tiers=FakeTiers({"interaction": PROVIDER}))

        (status, *_) = await rig.host.snapshot_events()

        assert CapabilitiesStatusEvent.model_validate(status).to_payload() == status
        assert (status["power_level"], status["decode_available"]) == ("low", True)

    @pytest.mark.asyncio
    async def test_watcher_broadcasts_a_switch_from_full_to_off(self):
        power = FakePower("full")
        rig = _rig(_lights(), power=power, tiers=SingleModelTiers(lambda: PROVIDER, power))
        await rig.host.start()
        await rig.host.wait_started()
        await _settle()
        before = rig.sink.of_type("capabilities_status")
        assert (before[-1]["power_level"], before[-1]["decode_available"]) == ("full", True)

        power.current = "off"
        await rig.clock.advance(1.0)

        after = rig.sink.of_type("capabilities_status")
        assert len(after) == len(before) + 1
        assert (after[-1]["power_level"], after[-1]["decode_available"]) == ("off", False)
        assert after[-1]["rev"] == before[-1]["rev"] + 1
        assert rig.clock.stalled == []


# -- Voice-ready records and the address form (T25) ------------------------------------------


VOICED_POWER = replace(POWER, ack="Turning them off.", view="lights")
VOICED_ADJUST = replace(ADJUST, ack="Adjusting them.", view="lights")
VOICED_TIMER = replace(TIMER_START, ack="Starting it.", view="timers")


def _voiced_lights() -> FakeCapability:
    return FakeCapability("lights", tools=(VOICED_POWER, VOICED_ADJUST))


def _voiced_timers() -> FakeCapability:
    return FakeCapability(
        "timers",
        tools=(VOICED_TIMER,),
        intents={"set a timer for 5 minutes": FastIntent("timers.start", {})},
    )


class TestVoiceRecords:
    @pytest.mark.asyncio
    async def test_a_fast_path_record_carries_the_specs_ack_and_view_and_source_fastpath(self):
        rig = _rig(_voiced_timers())

        resolution = await rig.host.pre_llm(SESSION, "turn-1", TIMER_PHRASE)

        assert resolution is not None
        (record,) = resolution.records
        assert (record.ack, record.view, record.source) == ("Starting it.", "timers", "fastpath")
        assert record.power_level == "full"

    @pytest.mark.asyncio
    async def test_a_decoded_call_record_carries_ack_view_and_source_model(self):
        rig = _rig(_voiced_lights())
        plan = rig.host.plan_turn(SESSION, "dim the lamp", [])

        result = await rig.host.run_actions(
            SESSION,
            "turn-1",
            "dim the lamp",
            plan,
            ScriptedDecode(_decoded(("lights.adjust", {"brightness": 30}))),
        )

        (record,) = result.records
        assert record.outcome.status == "ok"
        assert (record.ack, record.view, record.source) == ("Adjusting them.", "lights", "model")

    @pytest.mark.asyncio
    async def test_a_proposed_call_is_model_and_its_approved_run_is_proposal(self):
        rig = _rig(_voiced_lights(), _music())
        plan = rig.host.plan_turn(SESSION, "lights and music", [])
        proposed = await rig.host.run_actions(
            SESSION,
            "turn-0",
            "lights and music",
            plan,
            ScriptedDecode(_decoded(("lights.power", {"target": "bedroom"}))),
        )

        approved = await rig.host.pre_llm(SESSION, "turn-1", "yes")

        (pending,) = proposed.records
        assert pending.outcome.status == "pending_confirmation"
        assert (pending.ack, pending.view, pending.source) == (
            "Turning them off.",
            "lights",
            "model",
        )
        assert approved is not None
        (ran,) = approved.records
        assert ran.outcome.status == "ok"
        assert (ran.ack, ran.view, ran.source) == ("Turning them off.", "lights", "proposal")

    @pytest.mark.asyncio
    async def test_a_client_approval_runs_with_source_proposal(self):
        rig = _rig(_voiced_lights(), _music())
        proposal_id = await _propose(rig, ("lights.power", {"target": "bedroom"}))

        decided = await rig.host.decide(SESSION, proposal_id, "approve")

        (ran,) = decided.records
        assert (ran.ack, ran.view, ran.source) == ("Turning them off.", "lights", "proposal")

    @pytest.mark.asyncio
    async def test_an_invalid_call_keeps_its_specs_ack_and_view(self):
        rig = _rig(_voiced_lights())

        record = await rig.host.invoke(_call("lights.adjust"), _ctx("model"), None)

        assert record.outcome.status == "invalid_args"
        assert (record.ack, record.view, record.source) == ("Adjusting them.", "lights", "model")

    @pytest.mark.asyncio
    async def test_a_record_with_no_spec_has_no_ack_or_view(self):
        alarms = _alarms()
        alarms.ringing = True
        alarms.active_replies["stop"] = ToolOutcome(status="ok", summary="Alarm stopped.")
        rig = _rig(_voiced_lights(), alarms)

        unknown = await rig.host.invoke(_call("lights.explode"), _ctx("fastpath"), None)
        stopped = await rig.host.pre_llm(SESSION, "turn-1", "Stop.")

        assert (unknown.ack, unknown.view, unknown.source) == (None, None, "fastpath")
        assert stopped is not None
        (hook,) = stopped.records
        assert (hook.ack, hook.view, hook.source) == (None, None, "fastpath")


class _Boss:
    def __init__(self) -> None:
        self.asked = 0

    def address_form(self) -> str | None:
        self.asked += 1
        return "boss"


TIMER_SET = "Timer set for five minutes."


def _timer_rig(register: ReplyRegister | None = None, turns: int = 3) -> Rig:
    timers = _timers()
    timers.outcomes.extend(ToolOutcome(status="ok", summary=TIMER_SET) for _ in range(turns))
    return _rig(timers, _lights(), register=register)


class TestAddressForm:
    def test_register_is_a_keyword_defaulting_to_no_address(self):
        parameter = inspect.signature(CapabilityHost).parameters["register"]

        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert isinstance(parameter.default, NoAddress)

    @pytest.mark.asyncio
    async def test_the_form_ends_the_first_templated_reply_of_a_session_only(self):
        rig = _timer_rig(_Boss())

        first = await rig.host.pre_llm(SESSION, "turn-1", TIMER_PHRASE)
        second = await rig.host.pre_llm(SESSION, "turn-2", TIMER_PHRASE)
        other = await rig.host.pre_llm(OTHER, "turn-1", TIMER_PHRASE)

        assert first is not None and second is not None and other is not None
        assert first.reply == "Timer set for five minutes, boss."
        assert second.reply == TIMER_SET
        assert other.reply == "Timer set for five minutes, boss."
        assert first.records[0].outcome.summary == TIMER_SET

    @pytest.mark.asyncio
    async def test_a_decode_reply_and_a_decision_after_it_are_addressed_once(self):
        rig = _rig(_lights(), _music(), register=_Boss())
        plan = rig.host.plan_turn(SESSION, "lights and music", [])

        proposed = await rig.host.run_actions(
            SESSION,
            "turn-1",
            "lights and music",
            plan,
            ScriptedDecode(_decoded(("lights.power", {"target": "bedroom"}))),
        )
        (pending,) = proposed.records
        decided = await rig.host.decide(SESSION, pending.outcome.proposal_id or "", "reject")

        assert proposed.reply == "Should I turn off the bedroom lights? Say yes or no, boss."
        assert decided.reply == REJECTED_REPLY

    @pytest.mark.asyncio
    async def test_a_client_decision_can_be_the_first_reply(self):
        rig = _rig(_lights(), register=_Boss())
        # `invoke` gives a record and no reply, so the decision is the session's first reply.
        pending = await rig.host.invoke(_call("lights.power", target="bedroom"), _ctx(), None)

        decided = await rig.host.decide(SESSION, pending.outcome.proposal_id or "", "reject")

        assert decided.handled is True
        assert decided.reply == "Okay, I won't do that, boss."

    @pytest.mark.asyncio
    async def test_a_reply_the_main_pass_replaces_is_not_addressed_and_waits(self):
        rig = _timer_rig(_Boss())
        plan = rig.host.plan_turn(SESSION, "dim the lamp", [])

        result = await rig.host.run_actions(
            SESSION,
            "turn-1",
            "dim the lamp",
            plan,
            ScriptedDecode(_decoded(("lights.adjust", {"brightness": 30}), also_reply=True)),
        )
        fast = await rig.host.pre_llm(SESSION, "turn-2", TIMER_PHRASE)

        assert result.also_reply is True
        assert result.reply == reply_for(result.records)
        assert fast is not None and fast.reply == "Timer set for five minutes, boss."

    @pytest.mark.asyncio
    async def test_an_unhandled_decision_reply_is_not_addressed_and_does_not_count(self):
        rig = _timer_rig(_Boss())

        missing = await rig.host.decide(SESSION, "p-unknown", "approve")
        fast = await rig.host.pre_llm(SESSION, "turn-1", TIMER_PHRASE)

        assert missing.handled is False
        assert missing.reply == "proposal p-unknown is unknown"
        assert fast is not None and fast.reply == "Timer set for five minutes, boss."

    @pytest.mark.asyncio
    async def test_no_address_replies_are_byte_identical_to_the_default_hosts(self):
        async def replies(rig: Rig) -> list[str | None]:
            fast = await rig.host.pre_llm(SESSION, "turn-1", TIMER_PHRASE)
            plan = rig.host.plan_turn(SESSION, "dim the lamp", [])
            decoded = await rig.host.run_actions(
                SESSION,
                "turn-2",
                "dim the lamp",
                plan,
                ScriptedDecode(_decoded(("lights.power", {"target": "bedroom"}))),
            )
            approved = await rig.host.pre_llm(SESSION, "turn-3", "yes")
            assert fast is not None and approved is not None
            return [fast.reply, decoded.reply, approved.reply]

        default = await replies(_timer_rig())
        explicit = await replies(_timer_rig(NoAddress()))

        assert (
            explicit
            == default
            == [
                TIMER_SET,
                "Should I turn off the bedroom lights? Say yes or no.",
                "lights.power done.",
            ]
        )
