"""Tests for the WS protocol 1.4.0 capability payload schemas and the queue event sink."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta

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
    AlarmReportMessage,
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
            "power_level": "low",
            "decode_available": True,
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
            "record_id": "r1",
            "kind": "alarm",
            "category": "one_off",
            "label": None,
            "scheduled_for": AT,
            "fired_at": LATER,
            "late_by_s": 60.0,
            "ring_timeout_s": 600.0,
            "ring_on": "all",
            "fallback": False,
        },
    ),
    (
        AlarmFiredEvent,
        {
            "type": "alarm_fired",
            "alarm_id": "a2",
            "record_id": "r2",
            "kind": "timer",
            "category": "tagged",
            "label": "pasta",
            "scheduled_for": AT,
            "fired_at": AT,
            "late_by_s": 0.0,
            "ring_timeout_s": 600.0,
            "ring_on": ["bedroom", "desktop"],
            "fallback": True,
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
                    "record_id": "r1",
                    "kind": "alarm",
                    "category": "recurring",
                    "label": None,
                    "state": "ringing",
                    "next_fire_at": AT,
                    "wall_time": "07:00",
                    "repeat": "weekdays",
                    "days": [0, 1, 2, 3, 4],
                    "duration_s": None,
                    "ring_on": "all",
                },
                {
                    "alarm_id": "a3",
                    "record_id": "r3",
                    "kind": "alarm",
                    "category": "tagged",
                    "label": "gym",
                    "state": "snoozed",
                    "next_fire_at": LATER,
                    "wall_time": "06:30",
                    "repeat": "custom",
                    "days": [0, 3],
                    "duration_s": None,
                    "ring_on": ["bedroom"],
                },
                {
                    "alarm_id": "t1",
                    "record_id": "r2",
                    "kind": "timer",
                    "category": "one_off",
                    "label": None,
                    "state": "scheduled",
                    "next_fire_at": LATER,
                    "wall_time": None,
                    "repeat": "none",
                    "days": None,
                    "duration_s": 600,
                    "ring_on": "all",
                },
            ],
            "ringing": [{"alarm_id": "a1", "late_by_s": 0.0}],
            "missed": [
                {
                    "alarm_id": "a0",
                    "record_id": "r0",
                    "label": "gym",
                    "scheduled_for": AT,
                    "reason": "late",
                },
                {
                    "alarm_id": "a4",
                    "record_id": "r4",
                    "label": None,
                    "scheduled_for": AT,
                    "reason": "unheard",
                },
            ],
            "fallback_after_s": 30.0,
            "missed_grace_s": 900.0,
            "ring_timeout_s": 600.0,
            "snooze_minutes": 9,
            "tz": "America/New_York",
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


@pytest.mark.parametrize(
    ("field", "value"), [("power_level", "medium"), ("decode_available", "sometimes")]
)
def test_status_with_an_unknown_level_or_decode_flag_is_rejected(field, value):
    payload = dict(PAYLOADS[0][1], **{field: value})

    with pytest.raises(ValidationError):
        CapabilitiesStatusEvent.model_validate(payload)


ALARM_STATE = next(payload for model, payload in PAYLOADS if model is AlarmStateEvent)


def _alarm_state_with(entry_changes: dict[str, object], index: int = 0) -> dict[str, object]:
    alarms = [dict(entry) for entry in ALARM_STATE["alarms"]]  # type: ignore[attr-defined]
    alarms[index].update(entry_changes)
    return dict(ALARM_STATE, alarms=alarms)


@pytest.mark.parametrize("field", ["record_id", "category", "wall_time", "days", "ring_on"])
def test_an_alarm_entry_missing_a_1_4_0_field_is_rejected(field):
    alarms = [dict(entry) for entry in ALARM_STATE["alarms"]]  # type: ignore[attr-defined]
    del alarms[0][field]

    with pytest.raises(ValidationError):
        AlarmStateEvent.model_validate(dict(ALARM_STATE, alarms=alarms))


@pytest.mark.parametrize("field", ["record_id", "reason"])
def test_a_missed_entry_missing_a_1_4_0_field_is_rejected(field):
    missed = [dict(entry) for entry in ALARM_STATE["missed"]]  # type: ignore[attr-defined]
    del missed[0][field]

    with pytest.raises(ValidationError):
        AlarmStateEvent.model_validate(dict(ALARM_STATE, missed=missed))


@pytest.mark.parametrize(
    "changes",
    [
        {"category": "repeating"},
        {"repeat": "hourly"},
        {"days": [7]},
        {"days": [-1]},
        {"days": []},
        {"wall_time": "7:00"},
        {"wall_time": "24:00"},
        {"duration_s": 0},
        {"duration_s": 86_401},
        {"ring_on": "bedroom"},
        {"ring_on": []},
    ],
)
def test_an_alarm_entry_outside_the_protocol_is_rejected(changes):
    with pytest.raises(ValidationError):
        AlarmStateEvent.model_validate(_alarm_state_with(changes))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallback_after_s", 0.0),
        ("missed_grace_s", -1.0),
        ("ring_timeout_s", 0.0),
        ("snooze_minutes", 0),
        ("tz", ""),
    ],
)
def test_alarm_state_settings_outside_the_protocol_are_rejected(field, value):
    with pytest.raises(ValidationError):
        AlarmStateEvent.model_validate(dict(ALARM_STATE, **{field: value}))


def test_a_missed_entry_with_an_unknown_reason_is_rejected():
    missed = [dict(ALARM_STATE["missed"][0], reason="ignored")]  # type: ignore[index]

    with pytest.raises(ValidationError):
        AlarmStateEvent.model_validate(dict(ALARM_STATE, missed=missed))


@pytest.mark.parametrize(
    ("field", "value"), [("ring_on", "everywhere"), ("fallback", "maybe"), ("category", "x")]
)
def test_alarm_fired_outside_the_protocol_is_rejected(field, value):
    fired = next(payload for model, payload in PAYLOADS if model is AlarmFiredEvent)

    with pytest.raises(ValidationError):
        AlarmFiredEvent.model_validate(dict(fired, **{field: value}))


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

    @pytest.mark.parametrize(
        ("status", "result", "wire"),
        [
            ("ok", "applied", "ok"),
            ("ok", "late", "ok"),
            ("ok", "already_ended", "ok"),
            ("ok", "duplicate", "ok"),
            ("invalid_args", "not_due", "invalid_args"),
            ("invalid_args", "unknown", "invalid_args"),
        ],
    )
    def test_a_report_result_sets_detail(self, status, result, wire):
        outcome = ToolOutcome(status=status, summary="s", data={"result": result}, reason="why")

        payload = alarm_action_result_payload("a1", "report", "q1", outcome, self.AT_TIME)

        assert payload == {
            "type": "alarm_action_result",
            "alarm_id": "a1",
            "action": "report",
            "request_id": "q1",
            "outcome": wire,
            "reason": None if wire == "ok" else "why",
            "detail": result,
            "at": AT,
        }
        assert AlarmActionResultEvent.model_validate(payload).to_payload() == payload

    def test_a_report_refused_before_it_ran_has_no_detail(self):
        outcome = ToolOutcome(status="invalid_args", summary="s", reason="at is too old")

        payload = alarm_action_result_payload("a1", "report", None, outcome, self.AT_TIME)

        assert payload["outcome"] == "invalid_args"
        assert "detail" not in payload

    def test_detail_is_refused_on_an_action_that_is_not_a_report(self):
        payload = self._payload(ToolOutcome(status="ok", summary="s"))

        with pytest.raises(ValidationError):
            AlarmActionResultEvent.model_validate(dict(payload, detail="applied"))

    def test_an_unknown_report_result_is_refused(self):
        outcome = ToolOutcome(status="ok", summary="s", data={"result": "maybe"})

        with pytest.raises(ValidationError):
            alarm_action_result_payload("a1", "report", None, outcome, self.AT_TIME)


class TestAlarmReportMessage:
    NOW = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)

    def _parse(self, **changes: object) -> AlarmReportMessage:
        message: dict[str, object] = {
            "type": "alarm_report",
            "alarm_id": "a1",
            "outcome": "rang",
            "at": "2026-01-05T06:59:00Z",
            "rang_at": "2026-01-05T06:59:00Z",
        }
        message.update(changes)
        message = {key: value for key, value in message.items() if value is not None}
        return AlarmReportMessage.model_validate(message, context={"now": self.NOW})

    def test_a_rang_report_validates_and_round_trips(self):
        parsed = self._parse(request_id="q1")

        assert parsed.outcome == "rang"
        assert parsed.at == datetime(2026, 1, 5, 6, 59, tzinfo=UTC)
        assert parsed.to_payload() == {
            "type": "alarm_report",
            "alarm_id": "a1",
            "outcome": "rang",
            "at": "2026-01-05T06:59:00Z",
            "rang_at": "2026-01-05T06:59:00Z",
            "request_id": "q1",
        }

    def test_a_stopped_report_may_come_after_its_ring_started(self):
        parsed = self._parse(outcome="stopped", rang_at="2026-01-05T06:30:00Z")

        assert parsed.rang_at < parsed.at

    def test_a_snoozed_report_carries_snoozed_until(self):
        parsed = self._parse(
            outcome="snoozed",
            rang_at="2026-01-05T06:30:00Z",
            snoozed_until="2026-01-05T07:08:00Z",
        )

        assert parsed.snoozed_until == datetime(2026, 1, 5, 7, 8, tzinfo=UTC)
        assert parsed.to_payload()["snoozed_until"] == "2026-01-05T07:08:00Z"

    @pytest.mark.parametrize("field", ["type", "alarm_id", "outcome", "at", "rang_at"])
    def test_a_missing_required_field_is_refused(self, field):
        with pytest.raises(ValidationError):
            self._parse(**{field: None})

    @pytest.mark.parametrize("field", ["endpoint", "minutes", "extra"])
    def test_an_unknown_field_is_refused_and_no_payload_names_its_endpoint(self, field):
        with pytest.raises(ValidationError):
            self._parse(**{field: "desktop"})

    def test_it_has_no_endpoint_field(self):
        assert "endpoint" not in AlarmReportMessage.model_fields

    @pytest.mark.parametrize(
        "changes",
        [
            pytest.param({"at": "2026-01-05T06:59:00"}, id="naive-at"),
            pytest.param({"rang_at": "2026-01-05T06:59:00"}, id="naive-rang-at"),
            pytest.param(
                {"at": "2026-01-04T06:59:59Z", "rang_at": "2026-01-04T06:59:59Z"},
                id="over-24h-old",
            ),
            pytest.param(
                {"at": "2026-01-05T07:01:01Z", "rang_at": "2026-01-05T07:01:01Z"},
                id="over-60s-ahead",
            ),
            pytest.param(
                {"outcome": "stopped", "rang_at": "2026-01-04T06:00:00Z"},
                id="rang-at-over-24h-old",
            ),
            pytest.param(
                {"outcome": "stopped", "rang_at": "2026-01-05T06:59:01Z"},
                id="rang-at-after-at",
            ),
            pytest.param({"rang_at": "2026-01-05T06:58:00Z"}, id="rang-with-rang-at-not-at"),
            pytest.param({"outcome": "snoozed"}, id="snoozed-without-until"),
            pytest.param(
                {"outcome": "stopped", "snoozed_until": "2026-01-05T07:08:00Z"},
                id="until-with-stopped",
            ),
            pytest.param({"snoozed_until": "2026-01-05T07:08:00Z"}, id="until-with-rang"),
            pytest.param(
                {"outcome": "snoozed", "snoozed_until": "2026-01-05T07:08:00"},
                id="naive-until",
            ),
            pytest.param({"outcome": "dismiss"}, id="unknown-outcome"),
            pytest.param({"type": "alarm_action"}, id="wrong-type"),
            pytest.param({"request_id": ""}, id="request-id-empty"),
            pytest.param({"request_id": "q" * 65}, id="request-id-65"),
        ],
    )
    def test_a_report_outside_the_protocol_is_refused(self, changes):
        with pytest.raises(ValidationError):
            self._parse(**changes)

    @pytest.mark.parametrize(
        ("at", "loads"),
        [
            ("2026-01-04T07:00:00Z", True),
            ("2026-01-05T07:01:00Z", True),
            ("2026-01-04T06:59:59Z", False),
            ("2026-01-05T07:01:01Z", False),
        ],
        ids=["24h-old", "60s-ahead", "past-24h", "past-60s"],
    )
    def test_the_window_bounds_are_inclusive(self, at, loads):
        if loads:
            assert self._parse(at=at, rang_at=at).at == datetime.fromisoformat(at)
        else:
            with pytest.raises(ValidationError):
                self._parse(at=at, rang_at=at)

    def test_window_error_names_what_is_wrong_against_a_given_now(self):
        parsed = self._parse()

        assert parsed.window_error(self.NOW) is None
        error = parsed.window_error(self.NOW + timedelta(days=2))
        assert error is not None
        assert "at" in error

    def test_without_a_context_the_window_is_the_real_clock(self):
        now = datetime.now(UTC)

        def report(at: datetime) -> dict[str, object]:
            stamp = at.isoformat()
            return {
                "type": "alarm_report",
                "alarm_id": "a1",
                "outcome": "rang",
                "at": stamp,
                "rang_at": stamp,
            }

        assert AlarmReportMessage.model_validate(report(now)).at == now
        with pytest.raises(ValidationError):
            AlarmReportMessage.model_validate(report(now - timedelta(days=2)))
