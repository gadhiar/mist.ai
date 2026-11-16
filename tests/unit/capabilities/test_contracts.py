"""Tests for the capability contracts, the error classes, the clock and the shared fakes."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel

from backend import errors
from backend.capabilities.clock import SystemClock
from backend.capabilities.contracts import (
    SCREEN_VIEWS,
    ActionDecode,
    ActionRecord,
    ActionResult,
    Capability,
    CapabilityStatus,
    DecodeReply,
    FastIntent,
    InvocationContext,
    PreTurnResolution,
    RoutingHints,
    ToolOutcome,
    ToolSpec,
)
from backend.capabilities.db import CapabilitiesDb
from tests.unit.capabilities.fakes import (
    DEFAULT_START,
    EmptyArgs,
    FakeCapability,
    FakeClock,
    RecordingEventSink,
    ScriptedDecode,
    make_tool_spec,
)


def _context(source: str = "model") -> InvocationContext:
    return InvocationContext(
        session_id="s1",
        turn_id="t1",
        utterance="turn off the lights",
        tool_call_id="call-1",
        source=source,  # type: ignore[arg-type]
        now=DEFAULT_START,
    )


class _BareCapability(Capability):
    """Implements only the abstract members, so the optional hooks keep their defaults."""

    name = "bare"
    description = "a capability with no optional hooks"

    def tools(self) -> Sequence[ToolSpec]:
        return ()

    def routing_hints(self) -> RoutingHints:
        return RoutingHints(keywords=("bare",))

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def status(self) -> CapabilityStatus:
        return CapabilityStatus(
            name="bare", state="available", reason=None, checked_at=DEFAULT_START
        )

    async def invoke(self, tool: str, args: BaseModel, ctx: InvocationContext) -> ToolOutcome:
        return ToolOutcome(status="ok", summary="done")


class TestDataclasses:
    def test_capability_status_task_dead_defaults_to_false(self):
        status = CapabilityStatus(
            name="lights", state="available", reason=None, checked_at=DEFAULT_START
        )

        assert status.task_dead is False

    @pytest.mark.parametrize(
        "instance",
        [
            pytest.param(ToolOutcome(status="ok", summary="done"), id="tool-outcome"),
            pytest.param(
                CapabilityStatus(name="x", state="degraded", reason="r", checked_at=DEFAULT_START),
                id="capability-status",
            ),
            pytest.param(RoutingHints(keywords=("a",)), id="routing-hints"),
            pytest.param(_context(), id="invocation-context"),
            pytest.param(FastIntent(tool="x.do", args={}), id="fast-intent"),
            pytest.param(
                ActionDecode(messages=(), response_schema={}, max_tokens=256, temperature=0.0),
                id="action-decode",
            ),
            pytest.param(DecodeReply(content="{}", finish_reason="stop"), id="decode-reply"),
            pytest.param(make_tool_spec("x.do"), id="tool-spec"),
        ],
    )
    def test_internal_data_is_frozen_and_slotted(self, instance):
        field_name = dataclasses.fields(instance)[0].name

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, None)
        assert not hasattr(instance, "__dict__")

    def test_routing_hints_exemplars_default_to_empty(self):
        assert RoutingHints(keywords=("lamp",)).exemplars == ()

    def test_action_result_and_resolution_carry_records(self):
        record = ActionRecord(
            tool_call_id="c1",
            tool="x.do",
            capability="x",
            risk="read",
            args_summary="x",
            outcome=ToolOutcome(status="ok", summary="done"),
            duration_ms=3,
        )

        result = ActionResult(
            records=(record,), also_reply=False, reply="done", block="", notices=(), events=()
        )
        resolution = PreTurnResolution(
            handled=True, reply="done", remainder=None, records=(record,), events=()
        )

        assert result.records == resolution.records == (record,)


class TestToolOutcomeJson:
    def test_ok_outcome_is_compact_json_without_unset_fields(self):
        outcome = ToolOutcome(status="ok", summary="Bedroom lights off.")

        text = outcome.to_llm_json()

        assert json.loads(text) == {"status": "ok", "summary": "Bedroom lights off."}
        assert ", " not in text and ": " not in text

    def test_pending_outcome_says_the_action_has_not_run(self):
        outcome = ToolOutcome(
            status="pending_confirmation", summary="turn off the lights", proposal_id="p1"
        )

        payload = json.loads(outcome.to_llm_json())

        assert payload["proposal_id"] == "p1"
        assert "not run" in payload["note"]

    def test_reason_and_data_are_included_when_set(self):
        outcome = ToolOutcome(
            status="unavailable",
            summary="Lights are unavailable.",
            reason="unreachable",
            data={"n": 2},
        )

        payload = json.loads(outcome.to_llm_json())

        assert payload["reason"] == "unreachable"
        assert payload["data"] == {"n": 2}


class TestCapabilityBase:
    def test_capability_without_abstract_members_cannot_be_built(self):
        with pytest.raises(TypeError):
            Capability()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_optional_hooks_have_inert_defaults(self):
        capability = _BareCapability()
        schema: dict[str, object] = {"type": "object"}

        assert capability.dynamic_keywords() == ()
        assert capability.refine_allowance_tokens == 0
        assert capability.refine_schema("bare.do", schema) is schema
        assert capability.match_intent("turn off the lights") is None
        assert capability.escalate("bare.do", EmptyArgs()) is False
        assert capability.active() is False
        assert await capability.on_active_utterance("stop", "s1") is None
        assert capability.provider_tools() == ()


class TestErrors:
    @pytest.mark.parametrize(
        ("name", "parent"),
        [
            pytest.param("CapabilityError", "MistError", id="capability"),
            pytest.param("CapabilityConfigError", "CapabilityError", id="config"),
            pytest.param("CapabilityRegistrationError", "CapabilityError", id="registration"),
            pytest.param("CapabilityUnavailableError", "CapabilityError", id="unavailable"),
            pytest.param("CapabilityInvocationError", "CapabilityError", id="invocation"),
            pytest.param("CapabilityStoreError", "CapabilityError", id="store"),
            pytest.param("TimeParseError", "CapabilityError", id="time-parse"),
            pytest.param("ActionDecodeError", "CapabilityError", id="action-decode"),
            pytest.param("SecretMissingError", "CapabilityError", id="secret-missing"),
            pytest.param("McpError", "CapabilityError", id="mcp"),
            pytest.param("McpTransportError", "McpError", id="mcp-transport"),
            pytest.param("McpProtocolError", "McpError", id="mcp-protocol"),
            pytest.param("SpotifyError", "CapabilityError", id="spotify"),
            pytest.param("SpotifyAuthError", "SpotifyError", id="spotify-auth"),
            pytest.param("SpotifyApiError", "SpotifyError", id="spotify-api"),
        ],
    )
    def test_error_class_sits_under_its_parent(self, name, parent):
        cls = getattr(errors, name)

        assert issubclass(cls, getattr(errors, parent))
        assert issubclass(cls, errors.MistError)

    def test_spotify_api_error_carries_status_and_reason(self):
        error = errors.SpotifyApiError(403, "premium_required")

        assert error.status == 403
        assert error.reason == "premium_required"
        assert "403" in str(error) and "premium_required" in str(error)


class TestSystemClock:
    def test_now_is_aware_utc(self):
        now = SystemClock().now()

        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)

    @pytest.mark.asyncio
    async def test_sleep_zero_returns(self):
        await SystemClock().sleep(0)


class TestFakeClock:
    def test_rejects_a_naive_start(self):
        with pytest.raises(ValueError):
            FakeClock(datetime(2026, 1, 1))

    @pytest.mark.asyncio
    async def test_advance_wakes_sleepers_in_deadline_order_at_their_deadline(self):
        clock = FakeClock()
        woken: list[tuple[float, datetime]] = []

        async def sleeper(seconds: float) -> None:
            await clock.sleep(seconds)
            woken.append((seconds, clock.now()))

        tasks = [asyncio.create_task(sleeper(s)) for s in (30, 10, 20)]
        await asyncio.sleep(0)
        await clock.advance(60)
        await asyncio.gather(*tasks)

        assert woken == [
            (10, DEFAULT_START + timedelta(seconds=10)),
            (20, DEFAULT_START + timedelta(seconds=20)),
            (30, DEFAULT_START + timedelta(seconds=30)),
        ]
        assert clock.now() == DEFAULT_START + timedelta(seconds=60)

    @pytest.mark.asyncio
    async def test_advance_short_of_a_deadline_leaves_the_sleeper_waiting(self):
        clock = FakeClock()
        task = asyncio.create_task(clock.sleep(60))
        await asyncio.sleep(0)

        await clock.advance(59)

        assert not task.done()
        assert clock.pending == 1
        await clock.advance(1)
        assert task.done()

    @pytest.mark.asyncio
    async def test_a_sleeper_that_sleeps_again_is_woken_within_the_same_advance(self):
        clock = FakeClock()
        ticks: list[datetime] = []

        async def loop() -> None:
            for _ in range(3):
                await clock.sleep(60)
                ticks.append(clock.now())

        task = asyncio.create_task(loop())
        await asyncio.sleep(0)
        await clock.advance(180)
        await task

        assert ticks == [DEFAULT_START + timedelta(seconds=60 * n) for n in (1, 2, 3)]
        assert clock.sleeps == [60, 60, 60]

    @pytest.mark.asyncio
    async def test_every_tick_lands_when_the_sleeper_awaits_a_worker_thread(self, tmp_path):
        # The alarm scheduler awaits `CapabilitiesDb.run` (a thread hop) between sleeps. A fixed
        # number of loop yields cannot cover that, so this repeats the case to expose a race.
        for trial in range(30):
            clock = FakeClock()
            db = CapabilitiesDb(str(tmp_path / f"trial_{trial}.db"))
            ticks: list[datetime] = []

            async def loop(clock: FakeClock, db: CapabilitiesDb, ticks: list[datetime]) -> None:
                for _ in range(3):
                    await clock.sleep(60)
                    await db.run(lambda conn: conn.execute("SELECT 1").fetchone())
                    ticks.append(clock.now())

            task = asyncio.create_task(loop(clock, db, ticks))
            await asyncio.sleep(0)
            await clock.advance(180)

            assert len(ticks) == 3, f"trial {trial}: {len(ticks)} of 3 ticks"
            assert task.done()
            assert clock.stalled == []
            db.close()

    @pytest.mark.asyncio
    async def test_a_woken_task_that_never_sleeps_again_stalls_at_the_bound(self):
        clock = FakeClock(settle_timeout_s=0.05)
        never = asyncio.Event()

        async def waits_elsewhere() -> None:
            await clock.sleep(10)
            await never.wait()

        task = asyncio.create_task(waits_elsewhere())
        await asyncio.sleep(0)
        await clock.advance(30)

        assert clock.stalled == [DEFAULT_START + timedelta(seconds=10)]
        assert clock.now() == DEFAULT_START + timedelta(seconds=30)
        task.cancel()

    @pytest.mark.asyncio
    async def test_wait_for_sleepers_returns_once_enough_tasks_sleep(self):
        clock = FakeClock()

        async def sleeps_after_a_thread_hop() -> None:
            await asyncio.to_thread(lambda: None)
            await clock.sleep(5)

        task = asyncio.create_task(sleeps_after_a_thread_hop())
        await clock.wait_for_sleepers(1)

        assert clock.pending == 1
        task.cancel()

    @pytest.mark.asyncio
    async def test_wait_for_sleepers_fails_loudly_at_the_bound(self):
        clock = FakeClock(settle_timeout_s=0.05)

        with pytest.raises(AssertionError, match="1 sleeper"):
            await clock.wait_for_sleepers(1)


class TestRecordingEventSink:
    def test_records_payloads_from_other_threads(self):
        sink = RecordingEventSink()
        thread = threading.Thread(target=sink.emit, args=({"type": "alarm_fired"},))

        thread.start()
        thread.join()
        sink.emit({"type": "alarm_state"})

        assert sink.of_type("alarm_fired") == [{"type": "alarm_fired"}]
        assert len(sink.payloads) == 2


class TestScriptedDecode:
    @pytest.mark.asyncio
    async def test_returns_queued_replies_and_records_requests(self):
        first = DecodeReply(content='{"calls":[],"also_reply":true}', finish_reason="stop")
        decode = ScriptedDecode(first, errors.ActionDecodeError("bad json"))
        request = ActionDecode(messages=(), response_schema={}, max_tokens=256, temperature=0.0)

        reply = await decode(request)
        with pytest.raises(errors.ActionDecodeError):
            await decode(request)

        assert reply is first
        assert decode.requests == [request, request]


class TestFakeCapability:
    def test_match_intent_answers_from_its_script(self):
        intent = FastIntent(tool="fake.do", args={})
        capability = FakeCapability(intents={"do it now": intent})

        assert capability.match_intent("do it now") is intent
        assert capability.match_intent("something else") is None
        assert capability.intent_queries == ["do it now", "something else"]

    @pytest.mark.asyncio
    async def test_invoke_records_the_call_and_returns_scripted_outcomes(self):
        capability = FakeCapability()
        scripted = ToolOutcome(status="error", summary="failed", reason="boom")
        capability.outcomes.append(scripted)

        first = await capability.invoke("fake.do", EmptyArgs(), _context())
        second = await capability.invoke("fake.do", EmptyArgs(), _context())

        assert first is scripted
        assert second.status == "ok"
        assert [call[0] for call in capability.invocations] == ["fake.do", "fake.do"]

    def test_set_status_reports_task_dead(self):
        capability = FakeCapability()

        capability.set_status("unavailable", "scheduler died", task_dead=True)

        assert capability.status().task_dead is True
        assert capability.status().checked_at.tzinfo is UTC


class TestVoiceContract:
    """T25: the spoken `ack` and screen `view` on a tool, and what a record carries of them."""

    def test_tool_spec_ack_and_view_default_to_none(self):
        spec = make_tool_spec("x.do")

        assert (spec.ack, spec.view) == (None, None)

    def test_tool_spec_takes_an_ack_of_at_most_five_words_and_a_view(self):
        spec = dataclasses.replace(
            make_tool_spec("lights.power"), ack="Turning the bedroom lights off.", view="lights"
        )

        assert (spec.ack, spec.view) == ("Turning the bedroom lights off.", "lights")

    def test_an_ack_over_five_words_raises(self):
        with pytest.raises(ValueError, match="at most 5 words"):
            dataclasses.replace(make_tool_spec("x.do"), ack="Turning all of the lights off.")

    def test_an_unknown_view_raises(self):
        with pytest.raises(ValueError, match="view"):
            dataclasses.replace(make_tool_spec("x.do"), view="weather")  # type: ignore[arg-type]

    def test_screen_views_are_the_five_the_screen_draws(self):
        assert frozenset({"alarms", "timers", "clock", "lights", "music"}) == SCREEN_VIEWS

    def test_action_record_ack_view_and_source_have_defaults(self):
        record = ActionRecord(
            tool_call_id="c1",
            tool="x.do",
            capability="x",
            risk="read",
            args_summary="x",
            outcome=ToolOutcome(status="ok", summary="done"),
            duration_ms=3,
        )

        assert (record.ack, record.view, record.source, record.power_level) == (
            None,
            None,
            "model",
            None,
        )

    def test_action_record_carries_ack_view_and_source(self):
        record = ActionRecord(
            tool_call_id="c1",
            tool="timers.start",
            capability="timers",
            risk="reversible",
            args_summary="start a timer",
            outcome=ToolOutcome(status="ok", summary="Timer set for 10 minutes."),
            duration_ms=3,
            power_level="low",
            ack="Starting it.",
            view="timers",
            source="fastpath",
        )

        assert (record.ack, record.view, record.source) == ("Starting it.", "timers", "fastpath")
