"""T13: `CapabilitySink`, capabilities' `EventSink` over the session registry (D7, D8)."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
import pytest_asyncio

from backend.chat.conversation_handler import CapabilityOnlyTurn
from backend.chat.stream_events import Complete, Token
from backend.realtime.budget import BargeInConfig
from backend.realtime.capability_sink import CapabilitySink
from backend.realtime.turn import PlaybackProgress

from .turn_fakes import Pause, close_live_controllers, make_harness

pytestmark = pytest.mark.asyncio

AT = "2026-01-05T07:00:00Z"


def alarm_state(*ringing: str) -> dict[str, object]:
    return {
        "type": "alarm_state",
        "rev": 1,
        "alarms": [],
        "ringing": [{"alarm_id": alarm_id, "late_by_s": 0.0} for alarm_id in ringing],
        "missed": [],
        "fallback_after_s": 30.0,
        "missed_grace_s": 60.0,
        "ring_timeout_s": 300.0,
        "snooze_minutes": 9,
        "tz": "UTC",
        "at": AT,
    }


ALARM_FIRED = {"type": "alarm_fired", "alarm_id": "a1", "record_id": "r1", "fallback": False}
STATUS = {"type": "capabilities_status", "rev": 3, "capabilities": [], "at": AT}


class RecordingRegistry:
    def __init__(self) -> None:
        self.broadcasts: list[tuple[object, str | None]] = []
        self.threads: list[int] = []

    def broadcast(self, message: object, *, endpoint: str | None = None) -> None:
        self.broadcasts.append((message, endpoint))
        self.threads.append(threading.get_ident())


@pytest_asyncio.fixture(autouse=True)
async def _close_controllers():
    yield
    await close_live_controllers()


def attached() -> tuple[CapabilitySink, RecordingRegistry]:
    registry = RecordingRegistry()
    sink = CapabilitySink()
    sink.attach(registry, asyncio.get_running_loop())
    return sink, registry


async def test_alarm_fired_goes_to_desktop_sessions_only():
    sink, registry = attached()

    sink.emit(ALARM_FIRED)

    ((message, endpoint),) = registry.broadcasts
    assert endpoint == "desktop"
    assert json.loads(message) == ALARM_FIRED  # type: ignore[arg-type]


@pytest.mark.parametrize("payload", [STATUS, alarm_state("a1"), {"type": "action_resolved"}])
async def test_every_other_capability_payload_goes_to_every_session(payload):
    sink, registry = attached()

    sink.emit(payload)

    ((message, endpoint),) = registry.broadcasts
    assert endpoint is None
    assert json.loads(message) == payload  # type: ignore[arg-type]


async def test_ringing_follows_the_last_alarm_state_relayed():
    sink, _registry = attached()
    assert sink.ringing() is False  # before any `alarm_state`

    sink.emit(alarm_state("a1"))
    assert sink.ringing() is True
    sink.emit(STATUS)  # another payload changes nothing
    assert sink.ringing() is True
    sink.emit(alarm_state())
    assert sink.ringing() is False


async def test_a_payload_before_the_registry_is_attached_reaches_nobody_but_sets_ringing():
    sink = CapabilitySink()

    sink.emit(alarm_state("a1"))

    assert sink.ringing() is True


async def test_emit_from_another_thread_is_routed_on_the_loop_as_one_json_string():
    """The invariant `QueueEventSink`'s threaded test pinned, on the registry instead."""
    sink, registry = attached()
    loop_thread = threading.get_ident()

    thread = threading.Thread(target=sink.emit, args=(STATUS,))
    thread.start()
    thread.join()
    assert registry.broadcasts == []  # handed to the loop, not routed on the thread
    for _ in range(5):
        await asyncio.sleep(0)

    ((message, endpoint),) = registry.broadcasts
    assert isinstance(message, str)
    assert json.loads(message) == STATUS
    assert endpoint is None
    assert registry.threads == [loop_thread]


async def test_stop_during_a_reply_while_the_sink_relayed_a_ringing_alarm_reaches_capability_only():
    """D8 wired to the sink: "stop" cuts MIST at once and goes to `capability_only`."""
    sink, _registry = attached()
    sink.emit(alarm_state("a1"))
    h = make_harness(
        [Token("One two three. Four five. "), Pause()],
        barge_in=BargeInConfig(),
        alarm_ringing=sink.ringing,
        aec="headset",
    )
    h.start_voice()
    await h.speak("hello")
    turn_id = h.last_turn()
    await h.clock.advance(300)
    h.post(PlaybackProgress(turn_id=turn_id, played_ms=300))
    await h.settle()
    h.responder.capability_results.append(
        CapabilityOnlyTurn(
            handled=True,
            events=(Token("The alarm is off."), Complete(final_response="The alarm is off.")),
        )
    )
    other = h.ids()

    h.post(h.ingress.started(other), h.ingress.partial(other, "Stop", heard_at_ms=300))
    await h.settle()
    h.post(h.ingress.ended(other, "Stop."))
    await h.settle()

    assert h.responder.capability_calls == [("Stop.", "session-1")]
    assert [c.outcome for c in h.responder.commits] == ["interrupted"]
