"""Tests for the WS protocol 1.4.0 capability payload schemas and the queue event sink."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from backend.capabilities.events import (
    ActionDecisionMessage,
    ActionProposedEvent,
    ActionResolvedEvent,
    AlarmActionMessage,
    AlarmFiredEvent,
    AlarmStateEvent,
    CapabilitiesStatusEvent,
    QueueEventSink,
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
