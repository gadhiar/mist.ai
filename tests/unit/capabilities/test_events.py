"""Tests for the WS protocol 1.4.0 capability payload schemas and the queue event sink."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from backend.capabilities.contracts import ToolOutcome
from backend.capabilities.events import (
    ActionDecisionMessage,
    ActionProposedEvent,
    ActionResolvedEvent,
    AlarmActionMessage,
    AlarmActionResultEvent,
    AlarmFiredEvent,
    AlarmStateEvent,
    CapabilitiesStatusEvent,
    QueueEventSink,
    alarm_action_result_payload,
)

AT = "2026-01-05T07:00:00Z"
LATER = "2026-01-05T07:01:00Z"

PAYLOADS: list[tuple[type[BaseModel], dict[str, object]]] = [
    (
        CapabilitiesStatusEvent,
        {
            "type": "capabilities_status",
            "rev": 3,
            "capabilities": [
                {
                    "name": "lights",
                    "state": "unavailable",
                    "reason": "unreachable",
                    "tools": ["lights.power", "lights.adjust"],
                },
                {"name": "clock", "state": "available", "reason": None, "tools": ["clock.now"]},
            ],
            "at": AT,
        },
    ),
    (
        ActionProposedEvent,
        {
            "type": "action_proposed",
            "proposal_id": "p1",
            "session_id": "s1",
            "summary": "turn off the bedroom lights, then play jazz",
            "calls": [
                {
                    "capability": "lights",
                    "tool": "lights.power",
                    "summary": "turn off the bedroom lights",
                    "risk": "physical",
                },
                {
                    "capability": "music",
                    "tool": "music.play",
                    "summary": "play jazz",
                    "risk": "physical",
                },
            ],
            "expires_at": LATER,
        },
    ),
    (
        ActionResolvedEvent,
        {
            "type": "action_resolved",
            "proposal_id": "p1",
            "session_id": "s1",
            "decision": "approved",
            "outcomes": [
                {"tool": "lights.power", "status": "ok", "summary": "Bedroom lights off."},
                {"tool": "music.play", "status": "unavailable", "summary": "Spotify is down."},
            ],
        },
    ),
    (
        ActionResolvedEvent,
        {
            "type": "action_resolved",
            "proposal_id": "p2",
            "session_id": "s1",
            "decision": "expired",
            "outcomes": [],
        },
    ),
    (
        AlarmFiredEvent,
        {
            "type": "alarm_fired",
            "alarm_id": "a1",
            "kind": "alarm",
            "label": None,
            "scheduled_for": AT,
            "fired_at": LATER,
            "late_by_s": 60.0,
            "ring_timeout_s": 600.0,
        },
    ),
    (
        AlarmStateEvent,
        {
            "type": "alarm_state",
            "rev": 7,
            "alarms": [
                {
                    "alarm_id": "a1",
                    "kind": "alarm",
                    "label": "work",
                    "state": "ringing",
                    "next_fire_at": None,
                    "repeat": "weekdays",
                },
                {
                    "alarm_id": "t1",
                    "kind": "timer",
                    "label": None,
                    "state": "scheduled",
                    "next_fire_at": LATER,
                    "repeat": "none",
                },
            ],
            "ringing": [{"alarm_id": "a1", "late_by_s": 0.0}],
            "missed": [{"alarm_id": "a0", "label": "gym", "scheduled_for": AT}],
            "at": AT,
        },
    ),
    (
        ActionDecisionMessage,
        {"type": "action_decision", "proposal_id": "p1", "decision": "approve"},
    ),
    (
        AlarmActionMessage,
        {"type": "alarm_action", "alarm_id": "a1", "action": "snooze", "minutes": 5},
    ),
]


IDS = [f"{model.__name__}-{index}" for index, (model, _) in enumerate(PAYLOADS)]


@pytest.mark.parametrize(("model", "payload"), PAYLOADS, ids=IDS)
def test_payload_round_trips(model, payload):
    parsed = model.model_validate(payload)

    assert parsed.to_payload() == payload
    assert model.model_validate(json.loads(json.dumps(parsed.to_payload()))) == parsed


@pytest.mark.parametrize(("model", "payload"), PAYLOADS, ids=IDS)
def test_payload_missing_a_required_field_is_rejected(model, payload):
    # Every field of these payloads is required except `minutes` on `alarm_action`.
    for field in payload:
        if model is AlarmActionMessage and field == "minutes":
            continue
        broken = {key: value for key, value in payload.items() if key != field}
        with pytest.raises(ValidationError):
            model.model_validate(broken)


def test_alarm_action_minutes_is_optional():
    parsed = AlarmActionMessage.model_validate(
        {"type": "alarm_action", "alarm_id": "a1", "action": "stop"}
    )

    assert parsed.minutes is None


def test_wrong_type_tag_is_rejected():
    with pytest.raises(ValidationError):
        ActionDecisionMessage.model_validate(
            {"type": "alarm_action", "proposal_id": "p1", "decision": "approve"}
        )


def test_outbound_payload_with_an_undocumented_field_is_rejected():
    payload = dict(PAYLOADS[2][1], extra_field=1)

    with pytest.raises(ValidationError):
        ActionResolvedEvent.model_validate(payload)


def test_datetimes_serialise_as_iso_strings():
    event = ActionProposedEvent(
        type="action_proposed",
        proposal_id="p1",
        session_id="s1",
        summary="x",
        calls=[{"capability": "c", "tool": "c.do", "summary": "x", "risk": "read"}],
        expires_at=datetime(2026, 1, 5, 7, 1, tzinfo=UTC),
    )

    assert event.to_payload()["expires_at"] == LATER


@pytest.mark.asyncio
async def test_queue_event_sink_emit_from_another_thread_lands_one_json_string():
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | bytes] = asyncio.Queue()
    sink = QueueEventSink(loop, queue)
    payload = PAYLOADS[3][1]

    thread = threading.Thread(target=sink.emit, args=(payload,))
    thread.start()
    thread.join()
    message = await asyncio.wait_for(queue.get(), 2.0)

    assert isinstance(message, str)
    assert json.loads(message) == payload
    assert queue.empty()


class _RecordingLoop:
    """Wraps a loop and records each `call_soon_threadsafe`, so a test sees how `emit` queues."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.threadsafe_calls: list[tuple[object, tuple[object, ...]]] = []

    def call_soon_threadsafe(self, callback, *args):
        self.threadsafe_calls.append((callback, args))
        return self._loop.call_soon_threadsafe(callback, *args)

    def __getattr__(self, name: str) -> object:
        return getattr(self._loop, name)


@pytest.mark.asyncio
async def test_queue_event_sink_hands_its_put_to_call_soon_threadsafe():
    loop = _RecordingLoop(asyncio.get_running_loop())
    queue: asyncio.Queue[str | bytes] = asyncio.Queue()
    sink = QueueEventSink(loop, queue)  # type: ignore[arg-type]
    payload = PAYLOADS[3][1]

    sink.emit(payload)

    # Nothing is put until the loop runs the handed-over callback.
    assert queue.empty()
    ((callback, args),) = loop.threadsafe_calls
    assert callback == queue.put_nowait
    assert [json.loads(arg) for arg in args] == [payload]
    message = await asyncio.wait_for(queue.get(), 2.0)
    assert json.loads(message) == payload


class TestAlarmActionResult:
    AT_TIME = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)

    def _payload(self, outcome: ToolOutcome, request_id: str | None = "r1") -> dict[str, object]:
        return alarm_action_result_payload("a1", "stop", request_id, outcome, self.AT_TIME)

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("ok", "ok"),
            ("invalid_args", "invalid_args"),
            ("unavailable", "unavailable"),
            ("error", "failed"),
        ],
    )
    def test_each_outcome_maps_to_its_wire_name(self, status, expected):
        payload = self._payload(ToolOutcome(status=status, summary="s", reason="why"))

        assert payload == {
            "type": "alarm_action_result",
            "alarm_id": "a1",
            "action": "stop",
            "request_id": "r1",
            "outcome": expected,
            "reason": None if expected == "ok" else "why",
            "at": AT,
        }
        assert AlarmActionResultEvent.model_validate(payload).to_payload() == payload

    def test_ok_reason_is_null_even_when_the_outcome_has_one(self):
        payload = self._payload(ToolOutcome(status="ok", summary="Stopped.", reason="extra"))

        assert payload["reason"] is None

    @pytest.mark.parametrize("status", ["pending_confirmation", "denied"])
    def test_outcomes_alarm_actions_never_take_raise(self, status):
        with pytest.raises(ValueError, match=status):
            self._payload(ToolOutcome(status=status, summary="s"))

    def test_request_id_is_absent_when_none(self):
        payload = self._payload(ToolOutcome(status="ok", summary="s"), request_id=None)

        assert "request_id" not in payload
        assert AlarmActionResultEvent.model_validate(payload).to_payload() == payload

    def test_reason_is_capped_at_200_characters(self):
        payload = self._payload(ToolOutcome(status="error", summary="s", reason="x" * 500))

        assert payload["reason"] == "x" * 200

    @pytest.mark.parametrize("request_id", ["", "r" * 65])
    def test_request_id_of_0_or_65_characters_fails_validation(self, request_id):
        with pytest.raises(ValidationError):
            AlarmActionMessage.model_validate(
                {
                    "type": "alarm_action",
                    "alarm_id": "a1",
                    "action": "stop",
                    "request_id": request_id,
                }
            )
        with pytest.raises(ValidationError):
            self._payload(ToolOutcome(status="ok", summary="s"), request_id=request_id)

    def test_request_id_of_1_and_64_characters_validates_and_round_trips(self):
        for request_id in ("r", "r" * 64):
            message = {
                "type": "alarm_action",
                "alarm_id": "a1",
                "action": "stop",
                "request_id": request_id,
            }
            parsed = AlarmActionMessage.model_validate(message)
            assert parsed.request_id == request_id
            assert parsed.to_payload() == dict(message, minutes=None)

    def test_message_without_request_id_leaves_it_out(self):
        parsed = AlarmActionMessage.model_validate(
            {"type": "alarm_action", "alarm_id": "a1", "action": "stop"}
        )

        assert parsed.request_id is None
        assert "request_id" not in parsed.to_payload()

    def test_dismiss_validates_in_the_message_and_the_result(self):
        message = AlarmActionMessage.model_validate(
            {"type": "alarm_action", "alarm_id": "a1", "action": "dismiss"}
        )
        payload = alarm_action_result_payload(
            "a1", "dismiss", None, ToolOutcome(status="ok", summary="s"), self.AT_TIME
        )

        assert message.action == "dismiss"
        assert payload["action"] == "dismiss"
        assert AlarmActionResultEvent.model_validate(payload).action == "dismiss"

    def test_unknown_action_fails_validation(self):
        with pytest.raises(ValidationError):
            AlarmActionMessage.model_validate(
                {"type": "alarm_action", "alarm_id": "a1", "action": "pause"}
            )
        with pytest.raises(ValidationError):
            alarm_action_result_payload(
                "a1",
                "pause",  # type: ignore[arg-type]
                None,
                ToolOutcome(status="ok", summary="s"),
                self.AT_TIME,
            )

    def test_result_rejects_an_unknown_outcome_name(self):
        payload = self._payload(ToolOutcome(status="ok", summary="s"))

        with pytest.raises(ValidationError):
            AlarmActionResultEvent.model_validate(dict(payload, outcome="error"))
