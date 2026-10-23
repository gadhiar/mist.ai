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

from pydantic import BaseModel, ConfigDict, Field

from .contracts import CapabilityState, OutcomeStatus, RiskClass

AlarmKind = Literal["alarm", "timer"]
AlarmStateName = Literal["scheduled", "ringing", "snoozed", "done", "cancelled", "missed"]
AlarmRepeat = Literal["none", "daily", "weekdays", "weekends"]
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
    broadcasts.
    """

    type: Literal["capabilities_status"]
    rev: int = Field(ge=0)
    capabilities: list[CapabilityStatusEntry]
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
    """`alarm_fired`: an alarm or timer started ringing, `late_by_s` after its time."""

    type: Literal["alarm_fired"]
    alarm_id: str
    kind: AlarmKind
    label: str | None
    scheduled_for: datetime
    fired_at: datetime
    late_by_s: float = Field(ge=0)
    ring_timeout_s: float = Field(gt=0)


class AlarmEntry(_Outbound):
    """One alarm or timer in `alarm_state`."""

    alarm_id: str
    kind: AlarmKind
    label: str | None
    state: AlarmStateName
    next_fire_at: datetime | None
    repeat: AlarmRepeat


class RingingEntry(_Outbound):
    """A ringing item in `alarm_state`, with how late it rang."""

    alarm_id: str
    late_by_s: float = Field(ge=0)


class MissedEntry(_Outbound):
    """An item that fired beyond the grace window, in `alarm_state`."""

    alarm_id: str
    label: str | None
    scheduled_for: datetime


class AlarmStateEvent(_Outbound):
    """`alarm_state`: sent on connect and broadcast on every change, ordered by `rev`."""

    type: Literal["alarm_state"]
    rev: int = Field(ge=0)
    alarms: list[AlarmEntry]
    ringing: list[RingingEntry]
    missed: list[MissedEntry]
    at: datetime


class ActionDecisionMessage(_Inbound):
    """`action_decision` from the client; resolved for the connection's own session only."""

    type: Literal["action_decision"]
    proposal_id: str
    decision: Literal["approve", "reject"]


class AlarmActionMessage(_Inbound):
    """`alarm_action` from the client; `minutes` applies to a snooze only."""

    type: Literal["alarm_action"]
    alarm_id: str
    action: Literal["stop", "snooze"]
    minutes: int | None = Field(default=None, ge=1, le=60)


class EventSink(Protocol):
    """Receives capability payloads produced outside any turn, such as a proposal's expiry."""

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
