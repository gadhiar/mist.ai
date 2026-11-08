"""WebSocket protocol 1.4.0 payloads for capabilities, and the sink that queues them.

They are Pydantic models because they are WebSocket schemas (CLAUDE.md, Data types). Building a
server-to-client payload through its model makes a payload that drifts from the protocol fail
where it is built rather than in a client, and outbound models forbid unknown fields for the
same reason. Inbound models ignore unknown fields, because the server reads only the fields it
needs.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import CapabilityState, OutcomeStatus, RiskClass, ToolOutcome
from .power import PowerLevel

AlarmKind = Literal["alarm", "timer"]
AlarmStateName = Literal["scheduled", "ringing", "snoozed", "done", "cancelled", "missed"]
AlarmRepeat = Literal["none", "daily", "weekdays", "weekends", "custom"]
ResolutionDecision = Literal["approved", "rejected", "expired", "superseded"]


class _Outbound(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready dict sent on the wire; datetimes become ISO 8601 strings."""
        return self.model_dump(mode="json")


class _Inbound(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    def to_payload(self) -> dict[str, object]:
        """Return the JSON-ready dict of the fields the server reads."""
        return self.model_dump(mode="json")


class CapabilityStatusEntry(_Outbound):
    """One capability in `capabilities_status`."""

    name: str
    state: CapabilityState
    reason: str | None
    tools: list[str]


class CapabilitiesStatusEvent(_Outbound):
    """`capabilities_status`: sent on connect and broadcast on every state change.

    `rev` increases monotonically, so a client can order the on-connect snapshot against later
    broadcasts. `power_level` and `decode_available` are part of the state, so a change of
    either is broadcast like a capability's: at off the client can show that only model-free
    paths answer.
    """

    type: Literal["capabilities_status"]
    rev: int = Field(ge=0)
    capabilities: list[CapabilityStatusEntry]
    power_level: PowerLevel
    decode_available: bool
    at: datetime


class ProposedCallEntry(_Outbound):
    """One call of a bundled proposal."""

    capability: str
    tool: str
    summary: str
    risk: RiskClass


class ActionProposedEvent(_Outbound):
    """`action_proposed`: one proposal carries every call of its turn that needs confirmation."""

    type: Literal["action_proposed"]
    proposal_id: str
    session_id: str
    summary: str
    calls: list[ProposedCallEntry] = Field(min_length=1)
    expires_at: datetime


class OutcomeEntry(_Outbound):
    """One call's outcome in `action_resolved`."""

    tool: str
    status: OutcomeStatus
    summary: str


class ActionResolvedEvent(_Outbound):
    """`action_resolved`: `outcomes` is empty unless the decision is `approved`."""

    type: Literal["action_resolved"]
    proposal_id: str
    session_id: str
    decision: ResolutionDecision
    outcomes: list[OutcomeEntry]


class AlarmFiredEvent(_Outbound):
    """`alarm_fired`: an occurrence of an alarm or timer started ringing, `late_by_s` late.

    `alarm_id` is the occurrence and `record_id` the alarm it belongs to, so a client can tell
    two rings of one daily alarm apart and still group them. `ring_on` is the occurrence's own
    target, `"all"` or endpoint names; `fallback` says none of the named endpoints was present
    and the ring went to every endpoint that was.
    """

    type: Literal["alarm_fired"]
    alarm_id: str
    record_id: str
    kind: AlarmKind
    category: Literal["one_off", "recurring", "tagged"]
    label: str | None
    scheduled_for: datetime
    fired_at: datetime
    late_by_s: float = Field(ge=0)
    ring_timeout_s: float = Field(gt=0)
    ring_on: Literal["all"] | list[str] = Field(min_length=1)
    fallback: bool


class AlarmEntry(_Outbound):
    """One live occurrence in `alarm_state`, with the values of the alarm it belongs to.

    `wall_time` (`"HH:MM"`) is set for an alarm and `duration_s` for a timer. `days` is the
    record's weekday set, Monday 0, or null for one that does not repeat; `repeat` names the
    set, `"custom"` for one that is not a preset.
    """

    alarm_id: str
    record_id: str
    kind: AlarmKind
    category: Literal["one_off", "recurring", "tagged"]
    label: str | None
    state: AlarmStateName
    next_fire_at: datetime | None
    wall_time: str | None = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    repeat: AlarmRepeat
    days: list[int] | None = Field(min_length=1, max_length=7)
    duration_s: int | None = Field(ge=1, le=86_400)
    ring_on: Literal["all"] | list[str] = Field(min_length=1)

    @field_validator("days")
    @classmethod
    def _days_are_weekdays(cls, days: list[int] | None) -> list[int] | None:
        if days is not None and any(day < 0 or day > 6 for day in days):
            raise ValueError(f"days must be weekdays from 0 to 6: {days}")
        return days


class RingingEntry(_Outbound):
    """A ringing item in `alarm_state`, with how late it rang."""

    alarm_id: str
    late_by_s: float = Field(ge=0)


class MissedEntry(_Outbound):
    """A missed occurrence in `alarm_state`, until it is dismissed or expires.

    `reason` is `"late"` when it was found past the grace window and `"unheard"` when it rang
    to its ring timeout and no endpoint acknowledged playing it.
    """

    alarm_id: str
    record_id: str
    label: str | None
    scheduled_for: datetime
    reason: Literal["late", "unheard"]


class AlarmStateEvent(_Outbound):
    """`alarm_state`: sent on connect and broadcast on every change, ordered by `rev`.

    It carries the settings an endpoint rings by, so an endpoint that loses the backend can
    still ring and time out alone: `missed_grace_s` is the effective window, never under 60 s,
    after `next_fire_at` beyond which an endpoint never rings alone, and `tz` is the user's
    time zone (`MIST_TIMEZONE`).
    """

    type: Literal["alarm_state"]
    rev: int = Field(ge=0)
    alarms: list[AlarmEntry]
    ringing: list[RingingEntry]
    missed: list[MissedEntry]
    fallback_after_s: float = Field(gt=0)
    missed_grace_s: float = Field(ge=60)
    ring_timeout_s: float = Field(gt=0)
    snooze_minutes: int = Field(ge=1, le=60)
    tz: str = Field(min_length=1)
    at: datetime


class ActionDecisionMessage(_Inbound):
    """`action_decision` from the client; resolved for the connection's own session only."""

    type: Literal["action_decision"]
    proposal_id: str
    decision: Literal["approve", "reject"]


class AlarmActionMessage(_Inbound):
    """`alarm_action` from the client; `minutes` applies to a snooze only.

    `stop` and `snooze` act on a ringing alarm and `dismiss` on a missed one. `request_id` is
    an opaque client string that `alarm_action_result` echoes, so a client can pair the answer
    with its request.
    """

    type: Literal["alarm_action"]
    alarm_id: str
    action: Literal["stop", "snooze", "dismiss"]
    minutes: int | None = Field(default=None, ge=1, le=60)
    request_id: str | None = Field(default=None, min_length=1, max_length=64)

    def to_payload(self) -> dict[str, object]:
        """Return the fields the server reads; `request_id` is left out when none was sent."""
        exclude = {"request_id"} if self.request_id is None else None
        return self.model_dump(mode="json", exclude=exclude)


class AlarmActionResultEvent(_Outbound):
    """`alarm_action_result`: the one answer to each valid `alarm_action`, on its socket only.

    `reason` is null for `ok`. `request_id` echoes the client's and is absent from the payload
    when the client sent none.
    """

    type: Literal["alarm_action_result"]
    alarm_id: str
    action: Literal["stop", "snooze", "dismiss"]
    request_id: str | None = Field(default=None, min_length=1, max_length=64)
    outcome: Literal["ok", "invalid_args", "unavailable", "failed"]
    reason: str | None = Field(max_length=200)
    at: datetime

    def to_payload(self) -> dict[str, object]:
        """Return the wire payload, without `request_id` when it is None."""
        exclude = {"request_id"} if self.request_id is None else None
        return self.model_dump(mode="json", exclude=exclude)


def alarm_action_result_payload(
    alarm_id: str,
    action: Literal["stop", "snooze", "dismiss"],
    request_id: str | None,
    outcome: ToolOutcome,
    at: datetime,
) -> dict[str, object]:
    """Build the `alarm_action_result` payload for an outcome `CapabilityHost.alarm_action` gave.

    `ok`, `invalid_args` and `unavailable` keep their names and `error` becomes `failed`. The
    caller passes an outcome that is already redacted; its reason is capped at 200 characters.

    Raises:
        ValueError: The outcome is `pending_confirmation` or `denied`; an alarm action takes
            neither path, so building one is a bug in the caller.
        pydantic.ValidationError: An argument breaks the payload's schema, such as an unknown
            action or a `request_id` outside 1 to 64 characters.
    """
    wire_names: dict[str, Literal["ok", "invalid_args", "unavailable", "failed"]] = {
        "ok": "ok",
        "invalid_args": "invalid_args",
        "unavailable": "unavailable",
        "error": "failed",
    }
    wire = wire_names.get(outcome.status)
    if wire is None:
        raise ValueError(f"an alarm action never has the outcome {outcome.status}")
    reason = None if wire == "ok" or outcome.reason is None else outcome.reason[:200]
    return AlarmActionResultEvent(
        type="alarm_action_result",
        alarm_id=alarm_id,
        action=action,
        request_id=request_id,
        outcome=wire,
        reason=reason,
        at=at,
    ).to_payload()


class EventSink(Protocol):
    """Receives capability payloads that are broadcast rather than delivered with a turn.

    A proposal's expiry is one: the store reports it from its timer task outside any turn, or
    from inside a `pop` made by a turn or by `decide`.
    """

    def emit(self, payload: dict[str, object]) -> None:
        """Send one payload; safe to call from any thread."""
        ...


class QueueEventSink:
    """Puts each payload, as one JSON string, on the server's broadcast queue.

    `emit` hands the put to the queue's loop with `call_soon_threadsafe`, because an
    `asyncio.Queue` may only be touched from its own loop's thread.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[str | bytes]) -> None:
        self._loop = loop
        self._queue = queue

    def emit(self, payload: dict[str, object]) -> None:
        """Serialise `payload` now and queue it on the loop's thread."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, json.dumps(payload))
