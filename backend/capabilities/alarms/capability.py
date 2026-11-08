"""`AlarmsCapability`, `TimersCapability`, `ClockCapability` and `SchedulerAlarmControl`.

Alarms and timers share one `AlarmRecords` store and one `AlarmScheduler`, the only writer of
occurrence transitions, so each capability here reads the store and asks the scheduler to change
it. The model passes the user's words for a time ("7am", "tomorrow"); `parse_alarm_time` reads
them in the user's zone (`MIST_TIMEZONE`) at the call, as the fast path does, so both paths arm
the same instant. Without a zone no alarm can be placed in time: alarms, timers and the clock are
then `unavailable` with the reason, and no scheduler is built.

`active()` is synchronous, while the store is not, so `RingingTracker` wraps the scheduler's
notifier and keeps the ids of the occurrences ringing now in memory. The factory builds the
scheduler with the tracker as its notifier, around the `RingRouter`, and passes the same tracker
here.

`SchedulerAlarmControl` is the host's `AlarmControl`: the connect snapshot and a client's stop,
snooze, dismiss and report reach the scheduler through it, never through a tool call.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta, tzinfo
from typing import Literal, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.errors import CapabilityStoreError, CapabilityUnavailableError, TimeParseError

from ..clock import Clock
from ..config import AlarmSettings
from ..contracts import (
    Capability,
    CapabilityStatus,
    FastIntent,
    InvocationContext,
    RoutingHints,
    ToolOutcome,
    ToolSpec,
)
from ..replies import is_brief_sentence, ok_summary, render_read
from .intents import (
    SNOOZE_MINUTES_MAX,
    match_alarms,
    match_clock_intent,
    match_ringing_phrase,
    match_timers,
)
from .notifier import AlarmNotifier, alarm_state_payload
from .records import (
    MAX_DURATION_S,
    AlarmKind,
    AlarmRecord,
    AlarmRecords,
    ArmRequest,
    ArmResult,
    EventSource,
    Occurrence,
    ReportOutcome,
    ReportResult,
    normalise_label,
)
from .ringing import DESKTOP, RingDelivery
from .scheduler import AlarmScheduler, SchedulerHealth
from .timeparse import RepeatRule, next_occurrence, parse_alarm_time

logger = logging.getLogger(__name__)

ALARMS_SET = "alarms.set"
ALARMS_LIST = "alarms.list"
ALARMS_CANCEL = "alarms.cancel"
ALARMS_RINGING = "alarms.ringing"
TIMERS_START = "timers.start"
TIMERS_LIST = "timers.list"
TIMERS_CANCEL = "timers.cancel"
TIMERS_STOP = "timers.stop"
CLOCK_NOW = "clock.now"

TZ_UNSET_REASON = "MIST_TIMEZONE unset"
"""The status reason of alarms, timers and the clock when no zone is configured."""

MAX_RING_ON = 8
"""The most endpoint names one `ring_on` may list."""

_WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_PRESET_DAYS: dict[frozenset[int], str] = {
    frozenset(range(7)): "every day",
    frozenset(range(5)): "every weekday",
    frozenset({5, 6}): "on weekends",
}
# A bare hour from 1 to 12 with no meridiem names its morning or its evening, so a cancel by
# such a time matches an alarm at either.
_BARE_TWELVE_HOUR = re.compile(r"(1[0-2]|[1-9])(?:[: ]([0-5]\d))?")
_NO_ALARMS = "You have no alarms set."

Weekday = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
Repeat = Literal["none", "daily", "weekdays", "weekends", "custom"]

_Args = TypeVar("_Args", bound=BaseModel)


# -- Zone --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UserZone:
    """The user's time zone as configured, or why there is none.

    Attributes:
        name: `MIST_TIMEZONE` as set, stripped; None when unset.
        tz: The zone; None when unset or not found.
        reason: The status reason when `tz` is None; None otherwise.
    """

    name: str | None
    tz: tzinfo | None
    reason: str | None


def resolve_zone(name: str | None) -> UserZone:
    """Resolve `MIST_TIMEZONE` to a zone, or to the reason alarms cannot run without one.

    A missing tzdata package makes every IANA name unknown on a host with no system zone files,
    so the reason names that cause too.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        return UserZone(name=None, tz=None, reason=TZ_UNSET_REASON)
    try:
        tz = ZoneInfo(cleaned)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return UserZone(
            name=cleaned, tz=None, reason=f"timezone {cleaned} not found (tzdata missing?)"
        )
    return UserZone(name=cleaned, tz=tz, reason=None)


def scheduler_reason(health: SchedulerHealth) -> str:
    """Return the reason alarms are unavailable while the scheduler is not running.

    The capabilities' status and `SchedulerAlarmControl` both use it, so the status surface and
    a refused client action name the same cause.
    """
    return health.reason or f"alarm scheduler {health.state}"


# -- Ringing tracker ---------------------------------------------------------------------------


class RingingTracker:
    """The scheduler's notifier, wrapping another, that remembers which occurrences ring.

    `Capability.active()` is synchronous and is asked on every turn, so it cannot read the
    store. Each `state_changed` replaces the ringing set from the occurrences it carries, unless
    its `rev` is older than one already seen; `ring` and `stopped` adjust it between states.
    The wrapped notifier hears every call unchanged.
    """

    def __init__(self, inner: AlarmNotifier) -> None:
        """Wrap `inner`, normally the `RingRouter`."""
        self._inner = inner
        self._ringing: dict[str, AlarmKind] = {}
        self._rev = -1

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float) -> RingDelivery:
        """Record `occurrence` as ringing, then ring it through the wrapped notifier."""
        self._ringing[occurrence.occurrence_id] = occurrence.record.kind
        return await self._inner.ring(occurrence, ring_timeout_s)

    async def stopped(self, occurrence_id: str) -> None:
        """Forget `occurrence_id`, then pass the stop on."""
        self._ringing.pop(occurrence_id, None)
        await self._inner.stopped(occurrence_id)

    async def state_changed(
        self, occurrences: Sequence[Occurrence], rev: int, at: datetime
    ) -> None:
        """Take the ringing set from `occurrences`, then pass the state on."""
        self._take(occurrences, rev)
        await self._inner.state_changed(occurrences, rev, at)

    async def load(self, records: AlarmRecords) -> None:
        """Take the ringing set from the store, for an occurrence still ringing at a restart.

        The scheduler publishes a state only when one changes, so without this an alarm that
        was ringing when the backend stopped would not make its capability active.

        Raises:
            CapabilityStoreError: The read failed.
        """
        rev, live = await records.snapshot()
        self._take(live, rev)

    def ringing(self, kind: AlarmKind) -> tuple[str, ...]:
        """Return the sorted ids of the occurrences of `kind` ringing now."""
        return tuple(sorted(i for i, k in self._ringing.items() if k == kind))

    def _take(self, occurrences: Sequence[Occurrence], rev: int) -> None:
        if rev < self._rev:
            return
        self._rev = rev
        self._ringing = {
            o.occurrence_id: o.record.kind for o in occurrences if o.state == "ringing"
        }


# -- Arguments ---------------------------------------------------------------------------------


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.split()) or None


class SetArgs(BaseModel):
    """Arguments of `alarms.set`: the user's words for the time and day, or a saved label."""

    model_config = ConfigDict(extra="forbid")

    time: str | None = None
    day: str | None = None
    label: str | None = None
    repeat: Repeat = "none"
    days: list[Weekday] | None = Field(default=None, max_length=7)
    ring_on: list[str] | None = Field(default=None, max_length=MAX_RING_ON)

    @field_validator("time", "day", "label")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        """Collapse spacing; a blank value is no value."""
        return _clean(value)

    @model_validator(mode="after")
    def check_shape(self) -> SetArgs:
        """Refuse argument sets that name no alarm or contradict themselves."""
        if self.time is None:
            if self.label is None:
                raise ValueError("give a time, or the label of a saved alarm")
            if self.day is not None or self.days is not None or self.repeat != "none":
                raise ValueError("a saved alarm re-armed by label takes no day or repeat")
        if self.days is not None and (not self.days or len(set(self.days)) != len(self.days)):
            raise ValueError("days must list each weekday once, at least one")
        if (self.repeat == "custom") != (self.days is not None):
            raise ValueError("days go with repeat custom, and custom needs days")
        if self.day is not None and self.repeat != "none":
            raise ValueError("a repeating alarm takes no day")
        _check_ring_on(self.ring_on)
        return self


class AlarmListArgs(BaseModel):
    """Arguments of `alarms.list`: the live alarms, or every saved one by use."""

    model_config = ConfigDict(extra="forbid")

    scope: Literal["live", "saved"] = "live"


class AlarmCancelArgs(BaseModel):
    """Arguments of `alarms.cancel`: at most one way of naming the alarm."""

    model_config = ConfigDict(extra="forbid")

    alarm_id: str | None = None
    label: str | None = None
    time: str | None = None
    all: bool = False

    @field_validator("alarm_id", "label", "time")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        """Collapse spacing; a blank value is no value."""
        return _clean(value)

    @model_validator(mode="after")
    def one_selector(self) -> AlarmCancelArgs:
        """Refuse two ways of naming the alarm at once."""
        named = [self.alarm_id, self.label, self.time, True if self.all else None]
        if sum(value is not None for value in named) > 1:
            raise ValueError("name the alarm one way: alarm_id, label, time or all")
        return self


class RingingArgs(BaseModel):
    """Arguments of `alarms.ringing`."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["stop", "snooze"]
    minutes: int | None = Field(default=None, ge=1, le=SNOOZE_MINUTES_MAX)

    @model_validator(mode="after")
    def minutes_for_snooze(self) -> RingingArgs:
        """Refuse minutes on a stop."""
        if self.minutes is not None and self.action != "snooze":
            raise ValueError("minutes go with snooze only")
        return self


class TimerStartArgs(BaseModel):
    """Arguments of `timers.start`: a duration, a saved label, or both."""

    model_config = ConfigDict(extra="forbid")

    hours: int | None = Field(default=None, ge=0, le=24)
    minutes: int | None = Field(default=None, ge=0, le=MAX_DURATION_S // 60)
    seconds: int | None = Field(default=None, ge=0, le=MAX_DURATION_S)
    label: str | None = None
    ring_on: list[str] | None = Field(default=None, max_length=MAX_RING_ON)

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str | None) -> str | None:
        """Collapse spacing; a blank label is no label."""
        return _clean(value)

    @model_validator(mode="after")
    def check_duration(self) -> TimerStartArgs:
        """Refuse no duration without a label, and a duration outside 1 s to 24 h."""
        total = self.total_s()
        if total is None:
            if self.label is None:
                raise ValueError("give a duration, or the label of a saved timer")
        elif not 1 <= total <= MAX_DURATION_S:
            raise ValueError(f"a timer runs 1 second to 24 hours, not {total} seconds")
        _check_ring_on(self.ring_on)
        return self

    def total_s(self) -> int | None:
        """Return the duration in seconds, or None when no unit was given."""
        parts = (self.hours, self.minutes, self.seconds)
        if all(part is None for part in parts):
            return None
        hours, minutes, seconds = (part or 0 for part in parts)
        return hours * 3600 + minutes * 60 + seconds


class TimerCancelArgs(BaseModel):
    """Arguments of `timers.cancel`: at most one way of naming the timer."""

    model_config = ConfigDict(extra="forbid")

    timer_id: str | None = None
    label: str | None = None
    all: bool = False

    @field_validator("timer_id", "label")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        """Collapse spacing; a blank value is no value."""
        return _clean(value)

    @model_validator(mode="after")
    def one_selector(self) -> TimerCancelArgs:
        """Refuse two ways of naming the timer at once."""
        named = [self.timer_id, self.label, True if self.all else None]
        if sum(value is not None for value in named) > 1:
            raise ValueError("name the timer one way: timer_id, label or all")
        return self


class NoArgs(BaseModel):
    """Arguments of a tool that takes none."""

    model_config = ConfigDict(extra="forbid")


def _check_ring_on(ring_on: list[str] | None) -> None:
    # The schema grammar has no `minItems` or `minLength`, so the model may send either empty.
    if ring_on is not None and (not ring_on or any(not name.strip() for name in ring_on)):
        raise ValueError("ring_on lists at least one endpoint name")


# -- Alarms and timers -------------------------------------------------------------------------


class _ScheduledCapability(Capability):
    """What alarms and timers share: the store, the scheduler, the zone and the ringing hook."""

    _kind: AlarmKind
    _noun: str

    def __init__(
        self,
        records: AlarmRecords,
        scheduler: AlarmScheduler | None,
        *,
        clock: Clock,
        zone: UserZone,
        ringing: RingingTracker | None,
        endpoints: Callable[[], Sequence[str]],
        snooze_minutes: int,
        turn_endpoint: str = DESKTOP,
    ) -> None:
        """Keep the collaborators; nothing is read until `start`.

        Args:
            records: The alarm store, shared with the scheduler.
            scheduler: The one scheduler, shared by alarms and timers; None exactly when the
                zone did not resolve, since repeats and wall times need it.
            clock: The host's clock.
            zone: `resolve_zone(MIST_TIMEZONE)`.
            ringing: The tracker the scheduler notifies; required with a scheduler.
            endpoints: The ring endpoints by name (`RingRouter.endpoints`), read at each call
                so one added later is known.
            snooze_minutes: The configured snooze, for a snooze phrase that names none.
            turn_endpoint: The endpoint a spoken stop or snooze is recorded from. The
                `on_active_utterance` hook carries no context, so the host's turn endpoint is
                passed here.

        Raises:
            ValueError: A scheduler is passed without a zone, a zone without a scheduler, or a
                scheduler without a tracker.
        """
        if (scheduler is None) != (zone.tz is None):
            raise ValueError(
                f"{self.name}: pass a scheduler exactly when the zone resolves "
                f"(scheduler {'set' if scheduler else 'None'}, zone reason {zone.reason!r})"
            )
        if scheduler is not None and ringing is None:
            raise ValueError(f"{self.name}: a scheduler needs the ringing tracker it notifies")
        self._records = records
        self._scheduler = scheduler
        self._clock = clock
        self._zone = zone
        self._ringing = ringing
        self._endpoints = endpoints
        self._snooze_minutes = snooze_minutes
        self._turn_endpoint = turn_endpoint

    def routing_hints(self) -> RoutingHints:
        """Return no keywords beyond the router's floor for this capability."""
        return RoutingHints(keywords=())

    async def start(self) -> None:
        """Migrate the store, load what rings now, and start the shared scheduler.

        Alarms and timers both call it; each step is idempotent, and the scheduler starts once.

        Raises:
            CapabilityStoreError: The store cannot be opened or migrated; the message names
                the alarm records, and for a failed open the database path
                (`CapabilitiesDb` puts it in its error).
        """
        if self._scheduler is None or self._ringing is None:
            return
        try:
            await asyncio.to_thread(self._records.migrate)
        except (sqlite3.Error, OSError) as exc:
            raise CapabilityStoreError(f"alarm records cannot migrate: {exc}") from exc
        await self._ringing.load(self._records)
        await self._scheduler.start()

    async def stop(self) -> None:
        """Stop the shared scheduler; a second stop does nothing."""
        if self._scheduler is not None:
            await self._scheduler.stop()

    def status(self) -> CapabilityStatus:
        """Follow the scheduler's health, then the store's quarantined rows."""
        now = self._clock.now()
        if self._scheduler is None:
            return CapabilityStatus(self.name, "unavailable", self._zone.reason, now)
        health = self._scheduler.health()
        if health.state != "running":
            return CapabilityStatus(
                self.name,
                "unavailable",
                scheduler_reason(health),
                now,
                task_dead=self._scheduler.task_dead(),
            )
        held = len(self._records.quarantined())
        if held:
            noun = "alarm" if held == 1 else "alarms"
            return CapabilityStatus(
                self.name, "degraded", f"{held} stored {noun} quarantined; see the log", now
            )
        return CapabilityStatus(self.name, "available", None, now)

    def active(self) -> bool:
        """Return True while an occurrence of this kind rings."""
        return self._ringing is not None and bool(self._ringing.ringing(self._kind))

    async def on_active_utterance(self, utterance: str, session_id: str) -> ToolOutcome | None:
        """Stop or snooze what rings on a stop or snooze phrase; None for anything else.

        A phrase that names the other kind ("stop the timer" to alarms) is left to that
        capability while one of its items rings, so each stops its own.

        Raises:
            CapabilityStoreError: The store cannot be read or written.
        """
        if self._ringing is None or not self.active():
            return None
        phrase = match_ringing_phrase(utterance)
        if phrase is None:
            return None
        if (
            phrase.kind is not None
            and phrase.kind != self._kind
            and self._ringing.ringing(phrase.kind)
        ):
            return None
        return await self._ring_action(
            phrase.action, phrase.minutes, source="fastpath", endpoint=self._turn_endpoint
        )

    async def invoke(self, tool: str, args: BaseModel, ctx: InvocationContext) -> ToolOutcome:
        """Run a call, or answer `unavailable` with the zone reason when there is no zone.

        Raises:
            CapabilityStoreError: The store cannot be read or written.
            ValueError: `tool` is not one of this capability's tools.
        """
        if self._scheduler is None or self._zone.tz is None:
            return ToolOutcome(
                status="unavailable",
                summary=f"{self._noun.capitalize()}s need a time zone.",
                reason=self._zone.reason,
            )
        return await self._dispatch(tool, args, ctx, self._scheduler, self._zone.tz)

    async def _dispatch(
        self,
        tool: str,
        args: BaseModel,
        ctx: InvocationContext,
        scheduler: AlarmScheduler,
        tz: tzinfo,
    ) -> ToolOutcome:
        raise NotImplementedError

    # -- Shared steps ----------------------------------------------------------------------

    async def _live(self) -> list[Occurrence]:
        """Return this kind's live occurrences, in firing order."""
        _, live = await self._records.snapshot()
        return [o for o in live if o.record.kind == self._kind]

    async def _ring_action(
        self,
        action: Literal["stop", "snooze"],
        minutes: int | None,
        *,
        source: EventSource,
        endpoint: str | None,
    ) -> ToolOutcome:
        """Stop or snooze every ringing occurrence of this kind."""
        scheduler = self._scheduler
        if scheduler is None:
            return ToolOutcome(
                status="unavailable",
                summary=f"{self._noun.capitalize()}s need a time zone.",
                reason=self._zone.reason,
            )
        ringing = [o for o in await self._live() if o.state == "ringing"]
        snooze_for = self._snooze_minutes if minutes is None else minutes
        done = 0
        for occurrence in ringing:
            if action == "stop":
                changed = await scheduler.stop_ringing(
                    occurrence.occurrence_id, source=source, endpoint=endpoint
                )
            else:
                changed = await scheduler.snooze(
                    occurrence.occurrence_id, snooze_for, source=source, endpoint=endpoint
                )
            done += int(changed)
        if not done:
            return ToolOutcome(status="invalid_args", summary=f"No {self._noun} is ringing.")
        if action == "snooze":
            summary = ok_summary("alarms.snooze", minutes=snooze_for)
        elif done == 1:
            summary = ok_summary(f"{self._noun}s.stop")
        else:
            summary = f"{done} {self._noun}s stopped."
        return ToolOutcome(status="ok", summary=summary, data={"count": done})

    def _ring_on(self, names: list[str] | None) -> tuple[str, ...] | ToolOutcome | None:
        """Return `names` as a ring target, or the outcome naming an endpoint that is unknown."""
        if names is None:
            return None
        known = tuple(self._endpoints())
        unknown = [name for name in names if name not in known]
        if unknown:
            return ToolOutcome(
                status="invalid_args",
                summary=(
                    f"There is no endpoint called {unknown[0]}; known ones are "
                    f"{_join(list(known)) or 'none'}."
                ),
                data={"known_endpoints": list(known)},
            )
        return tuple(sorted(set(names)))

    async def _cancel_ids(self, ids: Sequence[str], source: EventSource) -> int:
        scheduler = self._scheduler
        if scheduler is None:
            return 0
        return await scheduler.cancel(ids, source=source)


class AlarmsCapability(_ScheduledCapability):
    """Sets, lists, cancels and silences alarms in the user's zone."""

    name = "alarms"
    description = "Set, list, cancel and silence alarms"
    _kind: AlarmKind = "alarm"
    _noun = "alarm"

    def __init__(
        self,
        records: AlarmRecords,
        scheduler: AlarmScheduler | None,
        *,
        clock: Clock,
        zone: UserZone,
        ringing: RingingTracker | None,
        endpoints: Callable[[], Sequence[str]],
        snooze_minutes: int,
        turn_endpoint: str = DESKTOP,
    ) -> None:
        """Build the capability (`_ScheduledCapability.__init__` documents the arguments)."""
        super().__init__(
            records,
            scheduler,
            clock=clock,
            zone=zone,
            ringing=ringing,
            endpoints=endpoints,
            snooze_minutes=snooze_minutes,
            turn_endpoint=turn_endpoint,
        )
        self._tools = (
            ToolSpec(
                name=ALARMS_SET,
                description="set an alarm from the user's words, or re-arm a saved label",
                args_model=SetArgs,
                risk="reversible",
                default_policy="allow",
                summarize=_summarize_set,
                ack="Setting the alarm.",
                view="alarms",
            ),
            ToolSpec(
                name=ALARMS_LIST,
                description="list live alarms, or saved ones by use",
                args_model=AlarmListArgs,
                risk="read",
                default_policy="allow",
                summarize=_summarize_alarm_list,
                ack="Checking your alarms.",
                view="alarms",
            ),
            ToolSpec(
                name=ALARMS_CANCEL,
                description="cancel an alarm by id, label or time, or all",
                args_model=AlarmCancelArgs,
                risk="reversible",
                default_policy="allow",
                summarize=_summarize_alarm_cancel,
                ack="Cancelling the alarm.",
                view="alarms",
            ),
            ToolSpec(
                name=ALARMS_RINGING,
                description="stop or snooze the ringing alarm",
                args_model=RingingArgs,
                risk="reversible",
                default_policy="allow",
                summarize=_summarize_ringing,
                ack="Silencing the alarm.",
                view="alarms",
            ),
        )

    def tools(self) -> Sequence[ToolSpec]:
        """Return `alarms.set`, `alarms.list`, `alarms.cancel` and `alarms.ringing`."""
        return self._tools

    def match_intent(self, text: str) -> FastIntent | None:
        """Match the alarm rows, the time checked in the user's zone at the clock's now.

        The saved-tag rows read `AlarmRecords.tag_labels`, which the store holds in memory, so
        this stays synchronous.
        """
        if self._zone.tz is None:
            return None
        return match_alarms(
            text, self._clock.now(), self._zone.tz, self._records.tag_labels(self._kind)
        )

    def escalate(self, tool: str, args: BaseModel) -> bool:
        """Confirm cancelling every alarm, whatever the policy."""
        return tool == ALARMS_CANCEL and isinstance(args, AlarmCancelArgs) and args.all

    async def _dispatch(
        self,
        tool: str,
        args: BaseModel,
        ctx: InvocationContext,
        scheduler: AlarmScheduler,
        tz: tzinfo,
    ) -> ToolOutcome:
        if isinstance(args, SetArgs) and tool == ALARMS_SET:
            return await self._set(args, ctx, scheduler, tz)
        if isinstance(args, AlarmListArgs) and tool == ALARMS_LIST:
            return await self._list(args, ctx.now, tz)
        if isinstance(args, AlarmCancelArgs) and tool == ALARMS_CANCEL:
            return await self._cancel(args, ctx, tz)
        if isinstance(args, RingingArgs) and tool == ALARMS_RINGING:
            return await self._ring_action(
                args.action, args.minutes, source=ctx.source, endpoint=ctx.endpoint
            )
        raise ValueError(f"alarms has no tool {tool!r} taking {type(args).__name__}")

    async def _set(
        self, args: SetArgs, ctx: InvocationContext, scheduler: AlarmScheduler, tz: tzinfo
    ) -> ToolOutcome:
        ring_on = self._ring_on(args.ring_on)
        if isinstance(ring_on, ToolOutcome):
            return ring_on
        request = _alarm_request(args, ctx.now, tz, ring_on)
        if isinstance(request, ToolOutcome):
            return request
        try:
            result = await scheduler.schedule(request, source=ctx.source)
        except LookupError:
            return ToolOutcome(
                status="invalid_args", summary=f"There is no saved alarm called {args.label}."
            )
        occurrence = result.occurrence
        key = (
            "alarms.set.already"
            if result.already_set
            else "alarms.set.moved" if result.moved else "alarms.set"
        )
        return ToolOutcome(
            status="ok",
            summary=_brief(key, occurrence.record, occurrence.next_fire_at, ctx.now, tz),
            data=_set_data(result, tz),
        )

    async def _list(self, args: AlarmListArgs, now: datetime, tz: tzinfo) -> ToolOutcome:
        if args.scope == "saved":
            saved = [r for r in await self._records.saved() if r.kind == "alarm"]
            return ToolOutcome(
                status="ok",
                summary=render_read(ALARMS_LIST, [_saved_phrase(r) for r in saved]),
                data={"scope": "saved", "alarms": [_saved_entry(r) for r in saved]},
            )
        live = await self._live()
        return ToolOutcome(
            status="ok",
            summary=render_read(ALARMS_LIST, [_alarm_phrase(o, now, tz) for o in live]),
            data={"scope": "live", "alarms": [_alarm_entry(o, tz) for o in live]},
        )

    async def _cancel(
        self, args: AlarmCancelArgs, ctx: InvocationContext, tz: tzinfo
    ) -> ToolOutcome:
        live = await self._live()
        if args.all:
            if not live:
                return ToolOutcome(status="ok", summary=_NO_ALARMS)
            count = await self._cancel_ids([o.occurrence_id for o in live], ctx.source)
            noun = "alarm" if count == 1 else "alarms"
            return ToolOutcome(
                status="ok", summary=f"{count} {noun} cancelled.", data={"cancelled": count}
            )
        if args.alarm_id is not None:
            return await self._cancel_by_id(args.alarm_id, live, ctx, tz)
        if args.label is not None:
            wanted = normalise_label(args.label)
            matches = [
                o
                for o in live
                if o.record.label is not None and normalise_label(o.record.label) == wanted
            ]
            if not matches:
                return ToolOutcome(
                    status="invalid_args", summary=f"No alarm is called {args.label}."
                )
            return await self._cancel_next(matches, ctx, tz)
        if args.time is not None:
            try:
                wall_times = _wall_times(args.time, ctx.now, tz)
            except TimeParseError as exc:
                return ToolOutcome(status="invalid_args", summary=f"I can't cancel that: {exc}.")
            matches = [
                o
                for o in live
                if o.record.wall_time is not None
                and (o.record.wall_time.hour, o.record.wall_time.minute) in wall_times
            ]
            if not matches:
                return ToolOutcome(
                    status="invalid_args", summary=f"You have no alarm at {args.time}."
                )
            return await self._cancel_next(matches, ctx, tz)
        if not live:
            return ToolOutcome(status="ok", summary=_NO_ALARMS)
        if len(live) == 1:
            return await self._cancel_next(live, ctx, tz)
        listed = render_read(ALARMS_LIST, [_alarm_phrase(o, ctx.now, tz) for o in live])
        return ToolOutcome(
            status="invalid_args",
            summary=f"Which alarm should I cancel? {listed}",
            data={"alarms": [_alarm_entry(o, tz) for o in live]},
        )

    async def _cancel_by_id(
        self, alarm_id: str, live: list[Occurrence], ctx: InvocationContext, tz: tzinfo
    ) -> ToolOutcome:
        target = next((o for o in live if o.occurrence_id == alarm_id), None)
        if target is not None:
            return await self._cancel_next([target], ctx, tz)
        # A quarantined occurrence reads as absent, yet cancelling it is how the user clears
        # a row the store cannot decode.
        if alarm_id in self._records.quarantined():
            count = await self._cancel_ids([alarm_id], ctx.source)
            if count:
                return ToolOutcome(
                    status="ok", summary="Alarm cancelled.", data={"cancelled": count}
                )
        return ToolOutcome(status="invalid_args", summary=f"There is no alarm {alarm_id}.")

    async def _cancel_next(
        self, matches: list[Occurrence], ctx: InvocationContext, tz: tzinfo
    ) -> ToolOutcome:
        """Cancel the first of `matches` to fire, saying how many of the rest stay set."""
        target = matches[0]
        repeating = target.record.days is not None
        others = [
            o
            for o in matches[1:]
            if not (repeating and o.record.record_id == target.record.record_id)
        ]
        count = await self._cancel_ids([target.occurrence_id], ctx.source)
        summary = _brief("alarms.cancel", target.record, target.next_fire_at, ctx.now, tz)
        if others:
            verb = "stays" if len(others) == 1 else "stay"
            summary = f"{summary[:-1]}; {len(others)} more {verb} set."
        return ToolOutcome(
            status="ok",
            summary=summary,
            data={"cancelled": count, "alarm_id": target.occurrence_id},
        )


class TimersCapability(_ScheduledCapability):
    """Starts, lists, cancels and stops timers."""

    name = "timers"
    description = "Start, list, cancel and stop timers"
    _kind: AlarmKind = "timer"
    _noun = "timer"

    def __init__(
        self,
        records: AlarmRecords,
        scheduler: AlarmScheduler | None,
        *,
        clock: Clock,
        zone: UserZone,
        ringing: RingingTracker | None,
        endpoints: Callable[[], Sequence[str]],
        snooze_minutes: int,
        turn_endpoint: str = DESKTOP,
    ) -> None:
        """Build the capability (`_ScheduledCapability.__init__` documents the arguments)."""
        super().__init__(
            records,
            scheduler,
            clock=clock,
            zone=zone,
            ringing=ringing,
            endpoints=endpoints,
            snooze_minutes=snooze_minutes,
            turn_endpoint=turn_endpoint,
        )
        self._tools = (
            ToolSpec(
                name=TIMERS_START,
                description="start a timer, or re-arm a saved label",
                args_model=TimerStartArgs,
                risk="reversible",
                default_policy="allow",
                summarize=_summarize_timer_start,
                ack="Starting the timer.",
                view="timers",
            ),
            ToolSpec(
                name=TIMERS_LIST,
                description="list running timers",
                args_model=NoArgs,
                risk="read",
                default_policy="allow",
                summarize=lambda args: "list the timers",
                ack="Checking your timers.",
                view="timers",
            ),
            ToolSpec(
                name=TIMERS_CANCEL,
                description="cancel a timer by id or label, or all",
                args_model=TimerCancelArgs,
                risk="reversible",
                default_policy="allow",
                summarize=_summarize_timer_cancel,
                ack="Cancelling the timer.",
                view="timers",
            ),
            ToolSpec(
                name=TIMERS_STOP,
                description="stop the ringing timer",
                args_model=NoArgs,
                risk="reversible",
                default_policy="allow",
                summarize=lambda args: "stop the ringing timer",
                ack="Stopping the timer.",
                view="timers",
            ),
        )

    def tools(self) -> Sequence[ToolSpec]:
        """Return `timers.start`, `timers.list`, `timers.cancel` and `timers.stop`."""
        return self._tools

    def match_intent(self, text: str) -> FastIntent | None:
        """Match the timer rows; none while there is no zone, as no timer can run.

        The saved-tag rows read `AlarmRecords.tag_labels`, held in memory, as alarms do.
        """
        if self._scheduler is None:
            return None
        return match_timers(text, self._records.tag_labels(self._kind))

    def escalate(self, tool: str, args: BaseModel) -> bool:
        """Confirm cancelling every timer, whatever the policy."""
        return tool == TIMERS_CANCEL and isinstance(args, TimerCancelArgs) and args.all

    async def _dispatch(
        self,
        tool: str,
        args: BaseModel,
        ctx: InvocationContext,
        scheduler: AlarmScheduler,
        tz: tzinfo,
    ) -> ToolOutcome:
        if isinstance(args, TimerStartArgs) and tool == TIMERS_START:
            return await self._start_timer(args, ctx, scheduler)
        if isinstance(args, NoArgs) and tool == TIMERS_LIST:
            live = await self._live()
            return ToolOutcome(
                status="ok",
                summary=render_read(TIMERS_LIST, [_timer_phrase(o, ctx.now) for o in live]),
                data={"timers": [_timer_entry(o, ctx.now) for o in live]},
            )
        if isinstance(args, TimerCancelArgs) and tool == TIMERS_CANCEL:
            return await self._cancel(args, ctx)
        if isinstance(args, NoArgs) and tool == TIMERS_STOP:
            return await self._ring_action("stop", None, source=ctx.source, endpoint=ctx.endpoint)
        raise ValueError(f"timers has no tool {tool!r} taking {type(args).__name__}")

    async def _start_timer(
        self, args: TimerStartArgs, ctx: InvocationContext, scheduler: AlarmScheduler
    ) -> ToolOutcome:
        ring_on = self._ring_on(args.ring_on)
        if isinstance(ring_on, ToolOutcome):
            return ring_on
        total = args.total_s()
        request = ArmRequest(
            kind="timer",
            fire_at=None if total is None else ctx.now + timedelta(seconds=total),
            wall_time=None,
            duration_s=total,
            label=args.label,
            days=None,
            ring_on=ring_on,
        )
        try:
            result = await scheduler.schedule(request, source=ctx.source)
        except LookupError:
            return ToolOutcome(
                status="invalid_args", summary=f"There is no saved timer called {args.label}."
            )
        occurrence = result.occurrence
        duration = occurrence.record.duration_s or 0
        if result.already_set:
            summary = "That timer is already running."
        else:
            summary = ok_summary(TIMERS_START, duration=_duration_phrase(duration))
        return ToolOutcome(
            status="ok",
            summary=summary,
            data={
                "timer_id": occurrence.occurrence_id,
                "record_id": occurrence.record.record_id,
                "duration_s": duration,
                "already_set": result.already_set,
            },
        )

    async def _cancel(self, args: TimerCancelArgs, ctx: InvocationContext) -> ToolOutcome:
        live = await self._live()
        if args.all:
            if not live:
                return ToolOutcome(status="ok", summary=ok_summary("timers.cancel.none"))
            count = await self._cancel_ids([o.occurrence_id for o in live], ctx.source)
            noun = "timer" if count == 1 else "timers"
            return ToolOutcome(
                status="ok", summary=f"{count} {noun} cancelled.", data={"cancelled": count}
            )
        if args.timer_id is not None:
            if any(o.occurrence_id == args.timer_id for o in live) or (
                args.timer_id in self._records.quarantined()
            ):
                count = await self._cancel_ids([args.timer_id], ctx.source)
                if count:
                    return _timer_cancelled(count)
            return ToolOutcome(status="invalid_args", summary=f"There is no timer {args.timer_id}.")
        if args.label is not None:
            wanted = normalise_label(args.label)
            matches = [
                o
                for o in live
                if o.record.label is not None and normalise_label(o.record.label) == wanted
            ]
            if not matches:
                return ToolOutcome(
                    status="invalid_args", summary=f"No timer is called {args.label}."
                )
            return _timer_cancelled(await self._cancel_ids([matches[0].occurrence_id], ctx.source))
        if not live:
            return ToolOutcome(status="ok", summary=ok_summary("timers.cancel.none"))
        if len(live) == 1:
            return _timer_cancelled(await self._cancel_ids([live[0].occurrence_id], ctx.source))
        listed = render_read(TIMERS_LIST, [_timer_phrase(o, ctx.now) for o in live])
        return ToolOutcome(
            status="invalid_args",
            summary=f"Which timer should I cancel? {listed}",
            data={"timers": [_timer_entry(o, ctx.now) for o in live]},
        )


def _timer_cancelled(count: int) -> ToolOutcome:
    return ToolOutcome(status="ok", summary=ok_summary("timers.cancel"), data={"cancelled": count})


# -- Clock -------------------------------------------------------------------------------------


class ClockCapability(Capability):
    """Tells the local time, date and weekday in the user's zone.

    It needs no store or scheduler, so it stays available whatever the scheduler's health.
    """

    name = "clock"
    description = "Tell the local time and date"

    def __init__(self, *, clock: Clock, zone: UserZone) -> None:
        """Keep the clock and the user's zone."""
        self._clock = clock
        self._zone = zone
        self._tools = (
            ToolSpec(
                name=CLOCK_NOW,
                description="read the local time, date and weekday",
                args_model=NoArgs,
                risk="read",
                default_policy="allow",
                summarize=lambda args: "read the clock",
                ack="Checking the clock.",
                view="clock",
            ),
        )

    def tools(self) -> Sequence[ToolSpec]:
        """Return `clock.now`."""
        return self._tools

    def routing_hints(self) -> RoutingHints:
        """Return no keywords beyond the router's clock floor."""
        return RoutingHints(keywords=())

    async def start(self) -> None:
        """Nothing to start."""

    async def stop(self) -> None:
        """Nothing to release."""

    def status(self) -> CapabilityStatus:
        """Return `unavailable` with the zone reason while there is no zone."""
        now = self._clock.now()
        if self._zone.tz is None:
            return CapabilityStatus(self.name, "unavailable", self._zone.reason, now)
        return CapabilityStatus(self.name, "available", None, now)

    def match_intent(self, text: str) -> FastIntent | None:
        """Match the clock rows; none while there is no zone."""
        if self._zone.tz is None:
            return None
        return match_clock_intent(text)

    async def invoke(self, tool: str, args: BaseModel, ctx: InvocationContext) -> ToolOutcome:
        """Read `ctx.now` in the user's zone.

        Raises:
            ValueError: `tool` is not `clock.now`.
        """
        if tool != CLOCK_NOW or not isinstance(args, NoArgs):
            raise ValueError(f"clock has no tool {tool!r} taking {type(args).__name__}")
        if self._zone.tz is None:
            return ToolOutcome(
                status="unavailable",
                summary="The clock needs a time zone.",
                reason=self._zone.reason,
            )
        local = ctx.now.astimezone(self._zone.tz)
        weekday = _WEEKDAY_NAMES[local.weekday()].capitalize()
        return ToolOutcome(
            status="ok",
            summary=render_read(
                CLOCK_NOW,
                [],
                time=_spoken_clock(local.hour, local.minute),
                date=f"{weekday} {_MONTH_NAMES[local.month - 1]} {local.day}",
            ),
            data={
                "local_time": local.isoformat(timespec="seconds"),
                "time": f"{local.hour:02d}:{local.minute:02d}",
                "date": local.date().isoformat(),
                "weekday": weekday,
                "tz": self._zone.name,
            },
        )


# -- Alarm control -----------------------------------------------------------------------------


class SchedulerAlarmControl:
    """The host's `AlarmControl` over the shared scheduler and store.

    Every method refuses while the scheduler is not running, with its health reason, so a
    client's action never writes beside a dead or failing scheduler. The settings and zone are
    what `alarm_state` carries.
    """

    def __init__(
        self,
        records: AlarmRecords,
        scheduler: AlarmScheduler,
        clock: Clock,
        settings: AlarmSettings,
        tz: tzinfo,
    ) -> None:
        """Keep the collaborators; the scheduler is the one alarms and timers use."""
        self._records = records
        self._scheduler = scheduler
        self._clock = clock
        self._settings = settings
        self._tz = tz

    async def snapshot(self) -> dict[str, object]:
        """Return the `alarm_state` payload from one store read of `rev` and the alarms.

        Raises:
            CapabilityUnavailableError: The scheduler is not running.
            CapabilityStoreError: The store cannot be read.
        """
        self._require_running()
        rev, live = await self._records.snapshot()
        return alarm_state_payload(
            live, rev, self._clock.now(), settings=self._settings, tz=self._tz
        )

    async def act(
        self,
        alarm_id: str,
        action: Literal["stop", "snooze", "dismiss"],
        minutes: int | None,
        *,
        endpoint: str,
    ) -> ToolOutcome:
        """Stop or snooze a ringing alarm or timer, or dismiss a missed one, from `endpoint`.

        A stop also ends a snoozed occurrence, as `AlarmScheduler.stop_ringing` does.

        Raises:
            CapabilityUnavailableError: The scheduler is not running.
            CapabilityStoreError: The store cannot be read or written.
            ValueError: `action` is unknown.
        """
        self._require_running()
        if action == "stop":
            if await self._scheduler.stop_ringing(
                alarm_id, source="alarm_action", endpoint=endpoint
            ):
                return ToolOutcome(status="ok", summary=ok_summary("alarms.stop"))
            return ToolOutcome(status="invalid_args", summary=f"Alarm {alarm_id} is not ringing.")
        if action == "snooze":
            snooze_for = self._settings.snooze_minutes if minutes is None else minutes
            if not 1 <= snooze_for <= SNOOZE_MINUTES_MAX:
                return ToolOutcome(
                    status="invalid_args",
                    summary=f"A snooze is 1 to {SNOOZE_MINUTES_MAX} minutes, not {snooze_for}.",
                )
            if await self._scheduler.snooze(
                alarm_id, snooze_for, source="alarm_action", endpoint=endpoint
            ):
                return ToolOutcome(
                    status="ok", summary=ok_summary("alarms.snooze", minutes=snooze_for)
                )
            return ToolOutcome(status="invalid_args", summary=f"Alarm {alarm_id} is not ringing.")
        if action == "dismiss":
            if await self._scheduler.dismiss(alarm_id, endpoint=endpoint):
                return ToolOutcome(status="ok", summary=ok_summary("alarms.dismiss"))
            return ToolOutcome(status="invalid_args", summary=f"Alarm {alarm_id} is not missed.")
        raise ValueError(f"unknown alarm action {action!r}")

    async def report(
        self,
        alarm_id: str,
        *,
        endpoint: str,
        outcome: ReportOutcome,
        at: datetime,
        rang_at: datetime,
        snoozed_until: datetime | None,
    ) -> ReportResult:
        """Apply an endpoint's report through `AlarmScheduler.report`.

        Raises:
            CapabilityUnavailableError: The scheduler is not running.
            CapabilityStoreError: The store cannot be read or written.
        """
        self._require_running()
        return await self._scheduler.report(
            alarm_id,
            endpoint=endpoint,
            outcome=outcome,
            at=at,
            rang_at=rang_at,
            snoozed_until=snoozed_until,
        )

    def _require_running(self) -> None:
        health = self._scheduler.health()
        if health.state != "running":
            raise CapabilityUnavailableError(scheduler_reason(health))


# -- Requests ----------------------------------------------------------------------------------


def _alarm_request(
    args: SetArgs, now: datetime, tz: tzinfo, ring_on: tuple[str, ...] | None
) -> ArmRequest | ToolOutcome:
    """Read the user's words into an `ArmRequest`, or the outcome saying why they don't read."""
    if args.time is None:
        return ArmRequest(
            kind="alarm",
            fire_at=None,
            wall_time=None,
            duration_s=None,
            label=args.label,
            days=None,
            ring_on=ring_on,
        )
    try:
        if args.repeat == "none":
            fire_at = parse_alarm_time(args.time, args.day, now, tz)
            local = fire_at.astimezone(tz)
            wall = time(local.hour, local.minute)
            days = None
        else:
            # Elapsed time names one instant, not a wall time a repeat could keep.
            if args.time.casefold().startswith("in "):
                return ToolOutcome(
                    status="invalid_args",
                    summary=f"A repeating alarm needs a clock time, not {args.time}.",
                )
            first = parse_alarm_time(args.time, None, now, tz).astimezone(tz)
            custom = None
            if args.repeat == "custom" and args.days is not None:
                custom = frozenset(_WEEKDAY_NAMES.index(day) for day in args.days)
            rule = RepeatRule(args.repeat, first.hour, first.minute, custom)
            fire_at = next_occurrence(rule, now, tz)
            wall = time(rule.hour, rule.minute)
            days = rule.days_set()
    except TimeParseError as exc:
        return ToolOutcome(status="invalid_args", summary=f"I can't set that alarm: {exc}.")
    return ArmRequest(
        kind="alarm",
        fire_at=fire_at,
        wall_time=wall,
        duration_s=None,
        label=args.label,
        days=days,
        ring_on=ring_on,
    )


def _wall_times(time_text: str, now: datetime, tz: tzinfo) -> frozenset[tuple[int, int]]:
    """Return the wall times `time_text` may name: both halves of the day for a bare hour."""
    local = parse_alarm_time(time_text, None, now, tz).astimezone(tz)
    found = {(local.hour, local.minute)}
    if _BARE_TWELVE_HOUR.fullmatch(time_text.strip()):
        found.add(((local.hour + 12) % 24, local.minute))
    return frozenset(found)


# -- Spoken forms ------------------------------------------------------------------------------


def _spoken_clock(hour: int, minute: int) -> str:
    return f"{hour % 12 or 12}:{minute:02d} {'AM' if hour < 12 else 'PM'}"


def _join(words: list[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"


def _repeat_phrase(days: frozenset[int]) -> str:
    preset = _PRESET_DAYS.get(days)
    if preset is not None:
        return preset
    names = [_WEEKDAY_NAMES[day].capitalize() for day in sorted(days)]
    if len(days) >= 5:
        missing = [_WEEKDAY_NAMES[day].capitalize() for day in range(7) if day not in days]
        return f"every day except {_join(missing)}"
    return f"every {_join(names)}"


def _day_phrase(at: datetime, now: datetime) -> str:
    """Name the local day of `at` as seen from `now`, both in the user's zone."""
    ahead = (at.date() - now.date()).days
    if ahead == 0:
        return "today"
    if ahead == 1:
        return "tomorrow"
    weekday = _WEEKDAY_NAMES[at.weekday()].capitalize()
    if 1 < ahead < 7:
        return weekday
    year = f" {at.year}" if at.year != now.year else ""
    return f"{weekday} {_MONTH_NAMES[at.month - 1]} {at.day}{year}"


def _when(record: AlarmRecord, at: datetime, now: datetime, tz: tzinfo) -> tuple[str, str]:
    """Return the full spoken time of an alarm firing at `at`, and the clock time alone."""
    local = at.astimezone(tz)
    clock = _spoken_clock(local.hour, local.minute)
    if record.days is not None:
        return f"{clock} {_repeat_phrase(record.days)}", clock
    return f"{clock} {_day_phrase(local, now.astimezone(tz))}", clock


def _brief(key: str, record: AlarmRecord, at: datetime, now: datetime, tz: tzinfo) -> str:
    """Render ok template `key` with the full time, or the clock alone when that runs long."""
    full, clock = _when(record, at, now, tz)
    summary = ok_summary(key, when=full)
    return summary if is_brief_sentence(summary) else ok_summary(key, when=clock)


def _duration_phrase(seconds: int) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [
        f"{value} {unit}{'' if value == 1 else 's'}"
        for value, unit in ((hours, "hour"), (minutes, "minute"), (secs, "second"))
        if value
    ]
    return _join(parts) or "0 seconds"


def _labelled(label: str | None, phrase: str) -> str:
    return f"{label}, {phrase}" if label else phrase


def _alarm_phrase(occurrence: Occurrence, now: datetime, tz: tzinfo) -> str:
    full, _ = _when(occurrence.record, occurrence.next_fire_at, now, tz)
    if occurrence.state != "scheduled":
        full = f"{full}, {occurrence.state}"
    return _labelled(occurrence.record.label, full)


def _saved_phrase(record: AlarmRecord) -> str:
    clock = (
        ""
        if record.wall_time is None
        else _spoken_clock(record.wall_time.hour, record.wall_time.minute)
    )
    repeat = "once" if record.days is None else _repeat_phrase(record.days)
    times = "time" if record.use_count == 1 else "times"
    state = "set" if record.active else "not set"
    return _labelled(record.label, f"{clock} {repeat}, used {record.use_count} {times}, {state}")


def _remaining_s(occurrence: Occurrence, now: datetime) -> int:
    return max(0, math.ceil((occurrence.next_fire_at - now).total_seconds()))


def _timer_phrase(occurrence: Occurrence, now: datetime) -> str:
    if occurrence.state == "scheduled":
        phrase = f"{_duration_phrase(_remaining_s(occurrence, now))} left"
    else:
        phrase = occurrence.state
    return _labelled(occurrence.record.label, phrase)


# -- Data for the model ------------------------------------------------------------------------


def _set_data(result: ArmResult, tz: tzinfo) -> dict[str, object]:
    occurrence = result.occurrence
    return {
        "alarm_id": occurrence.occurrence_id,
        "record_id": occurrence.record.record_id,
        "fires_at": occurrence.next_fire_at.astimezone(tz).isoformat(timespec="seconds"),
        "already_set": result.already_set,
        "moved": bool(result.moved),
    }


def _wall_text(record: AlarmRecord) -> str | None:
    return None if record.wall_time is None else record.wall_time.strftime("%H:%M")


def _alarm_entry(occurrence: Occurrence, tz: tzinfo) -> dict[str, object]:
    record = occurrence.record
    return {
        "alarm_id": occurrence.occurrence_id,
        "record_id": record.record_id,
        "label": record.label,
        "state": occurrence.state,
        "fires_at": occurrence.next_fire_at.astimezone(tz).isoformat(timespec="seconds"),
        "wall_time": _wall_text(record),
        "days": None if record.days is None else sorted(record.days),
    }


def _saved_entry(record: AlarmRecord) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "label": record.label,
        "wall_time": _wall_text(record),
        "days": None if record.days is None else sorted(record.days),
        "use_count": record.use_count,
        "active": record.active,
    }


def _timer_entry(occurrence: Occurrence, now: datetime) -> dict[str, object]:
    return {
        "timer_id": occurrence.occurrence_id,
        "record_id": occurrence.record.record_id,
        "label": occurrence.record.label,
        "state": occurrence.state,
        "duration_s": occurrence.record.duration_s,
        "remaining_s": _remaining_s(occurrence, now),
    }


# -- Summaries for the action log and proposals --------------------------------------------------


def _expect(args: BaseModel, model: type[_Args]) -> _Args:
    # Each spec pairs its summarize hook with its own args model, so a mismatch is a bug.
    if not isinstance(args, model):
        raise TypeError(f"expected {model.__name__}, got {type(args).__name__}")
    return args


def _summarize_set(args: BaseModel) -> str:
    set_args = _expect(args, SetArgs)
    if set_args.time is None:
        return f"set the {set_args.label} alarm"
    words = [set_args.time]
    if set_args.day is not None:
        words.append(set_args.day)
    if set_args.repeat == "custom" and set_args.days is not None:
        words.append("every " + _join([d.capitalize() for d in set_args.days]))
    elif set_args.repeat != "none":
        words.append(set_args.repeat)
    return "set an alarm for " + " ".join(words)


def _summarize_alarm_list(args: BaseModel) -> str:
    return (
        "list the saved alarms"
        if _expect(args, AlarmListArgs).scope == "saved"
        else ("list the alarms")
    )


def _summarize_alarm_cancel(args: BaseModel) -> str:
    cancel = _expect(args, AlarmCancelArgs)
    if cancel.all:
        return "cancel every alarm"
    if cancel.label is not None:
        return f"cancel the {cancel.label} alarm"
    if cancel.time is not None:
        return f"cancel the {cancel.time} alarm"
    return "cancel the alarm"


def _summarize_ringing(args: BaseModel) -> str:
    ringing = _expect(args, RingingArgs)
    if ringing.action == "stop":
        return "stop the ringing alarm"
    if ringing.minutes is None:
        return "snooze the ringing alarm"
    return f"snooze the ringing alarm for {ringing.minutes} minutes"


def _summarize_timer_start(args: BaseModel) -> str:
    start = _expect(args, TimerStartArgs)
    total = start.total_s()
    if total is None:
        return f"start the {start.label} timer"
    return f"start a {_duration_phrase(total)} timer"


def _summarize_timer_cancel(args: BaseModel) -> str:
    cancel = _expect(args, TimerCancelArgs)
    if cancel.all:
        return "cancel every timer"
    if cancel.label is not None:
        return f"cancel the {cancel.label} timer"
    return "cancel the timer"
