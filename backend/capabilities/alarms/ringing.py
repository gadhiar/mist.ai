"""Endpoint-neutral ringing: which endpoints an occurrence rings on, and where it went.

An endpoint is a named place MIST can ring, reached through one `RingSink`. `desktop` is every
connection of the daily WebSocket (`DesktopSink`), a sink fixed when the router is built; each
registered room device is another (`backend.devices.ring_sink.DeviceRingSink`), given by a sink
source the router reads at each call, so a device registered or revoked while the backend runs
becomes, or stops being, an endpoint without a restart. `RingRouter` is the scheduler's
`AlarmNotifier` over those sinks:

- The fixed sinks come first, then the source's. A sourced sink whose name cannot be stored
  beside the others (empty, holding a comma, `all`, or a name an earlier sink has) is skipped
  and logged at ERROR once per name, so one unusable sink cannot stop the rest ringing.
- An occurrence's `ring_on` is `all` or endpoint names. It resolves to the named sinks that are
  present. When none of them is, the ring goes to every present sink instead and a `fallback`
  event is recorded, so a ring meant for an absent room unit still reaches someone.
- Each sink that emitted the ring to at least one live connection goes into the occurrence's
  `sent_to` through `AlarmRecords.sent`. A WebSocket send is fire-and-forget, so `sent_to` is
  where a ring went, not proof it was heard: an endpoint acknowledges playing a ring with an
  `alarm_report` of `rang`, and the scheduler's ring timeout ends an occurrence nobody
  acknowledged as missed, reason `unheard`, with `sent_to` in the event's detail.
- A sink failing with a contained error is logged at ERROR, naming its class, and the other
  sinks are still called, so a broken room unit cannot silence the desktop.

`alarm_state` goes to every sink, present or not, built once from the settings the endpoints
ring by; a sink with no live connection drops it, and a connecting client gets the snapshot.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Protocol

from backend.errors import MistError

from ..clock import Clock
from ..config import AlarmSettings
from ..events import EventSink
from .notifier import alarm_fired_payload, alarm_state_payload
from .records import AlarmRecords, Occurrence, RingTarget

logger = logging.getLogger(__name__)

FALLBACK_AFTER_S_DEFAULT = 30.0
"""The default `MIST_RING_FALLBACK_AFTER_S`: how long an endpoint that cannot reach the owner
waits past an alarm's time before it rings on its own."""

DESKTOP = "desktop"
"""The endpoint of the daily WebSocket's connections."""

# Failures one sink contains rather than passing on: the MIST error tree, storage and OS-level
# I/O (a closed socket is an `OSError`). Anything else is a bug and propagates. No broader catch
# is added.
_CONTAINED_ERRORS = (MistError, sqlite3.Error, OSError)


@dataclass(frozen=True, slots=True)
class RingDelivery:
    """Where one ring went.

    Attributes:
        sent_to: The sinks that emitted it to at least one live connection, in sink order.
        fallback: `ring_on` named endpoints, none was present, and it rang everywhere instead.
    """

    sent_to: tuple[str, ...]
    fallback: bool


class RingSink(Protocol):
    """One endpoint MIST can ring. Every method is called on the main loop."""

    name: str
    """The endpoint's name, as `ring_on` names it: `desktop`, or a room device's own name."""

    def present(self) -> bool:
        """Return whether the endpoint has a live connection now."""
        ...

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float, *, fallback: bool) -> bool:
        """Ring `occurrence` on this endpoint; return whether it reached a live connection."""
        ...

    async def stopped(self, occurrence_id: str) -> None:
        """Tell the endpoint the occurrence stopped ringing."""
        ...

    async def state_changed(self, payload: Mapping[str, object]) -> None:
        """Pass the endpoint the `alarm_state` payload."""
        ...


class RingRouter:
    """The scheduler's `AlarmNotifier`: rings each occurrence on the endpoints it names.

    The records receive each ring's `sent_to`; the clock stamps those writes; the settings and
    time zone are what `alarm_state` carries. Each public method reads the sink source once, so
    one call sees one set of endpoints from start to end.
    """

    def __init__(
        self,
        sinks: Sequence[RingSink],
        records: AlarmRecords,
        clock: Clock,
        settings: AlarmSettings,
        tz: tzinfo,
        *,
        sink_source: Callable[[], Sequence[RingSink]] | None = None,
    ) -> None:
        """Keep the fixed sinks in the order they are rung, and the source of the others.

        Args:
            sinks: The endpoints known when the router is built, rung first.
            records: Receive each ring's `sent_to`.
            clock: Stamps those writes.
            settings: What `alarm_state` carries.
            tz: The zone `alarm_state` carries.
            sink_source: Gives the other endpoints, read at each call and rung after `sinks`;
                None rings `sinks` only.

        Raises:
            ValueError: A fixed sink's name is empty, holds a comma, is `all`, or repeats
                another's; `AlarmRecords` stores endpoint names as a comma list beside the word
                `all`. A sourced sink with such a name is skipped instead, since it can appear
                long after the build.
        """
        names = [sink.name for sink in sinks]
        for name in names:
            if not _storable(name):
                raise ValueError(f"a sink's name must be an endpoint name: {name!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"each sink needs a name of its own: {names}")
        self._fixed = tuple(sinks)
        self._source = sink_source
        self._reported: set[str] = set()
        self._records = records
        self._clock = clock
        self._settings = settings
        self._tz = tz

    def endpoints(self) -> tuple[str, ...]:
        """Return every endpoint's name, present or not, in sink order."""
        return tuple(sink.name for sink in self._sinks())

    def resolve(self, ring_on: RingTarget) -> tuple[tuple[RingSink, ...], bool]:
        """Return the sinks `ring_on` rings now, and whether that is the fallback.

        `all` gives every present sink. Names give the present sinks among them; when none is
        present, every present sink, with the fallback flag. With no sink present it is no sink
        and no fallback, since there is nothing to fall back to.
        """
        return _resolve(self._sinks(), ring_on)

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float) -> RingDelivery:
        """Ring the occurrence on its resolved sinks and store where it went.

        A failure to store the delivery is logged and the delivery still returned: the ring has
        already gone out, and the stored ringing state still times out.
        """
        sinks = self._sinks()
        targets, fallback = _resolve(sinks, occurrence.ring_on)
        sent: list[str] = []
        for sink in targets:
            try:
                if await sink.ring(occurrence, ring_timeout_s, fallback=fallback):
                    sent.append(sink.name)
            except _CONTAINED_ERRORS as exc:
                _log_failure(sink, "ring", exc)
        delivery = RingDelivery(sent_to=tuple(sent), fallback=fallback)
        if sent or fallback:
            try:
                await self._records.sent(
                    occurrence.occurrence_id, sent, fallback=fallback, at=self._clock.now()
                )
            except _CONTAINED_ERRORS as exc:
                logger.error(
                    "Where alarm %s rang (%s) was not stored: %s",
                    occurrence.occurrence_id,
                    ", ".join(sent) or "nowhere",
                    exc,
                    exc_info=exc,
                )
        if not sent:
            logger.warning(
                "Alarm %s rang on no endpoint: none of %s is connected",
                occurrence.occurrence_id,
                ", ".join(sink.name for sink in sinks) or "no sink",
            )
        return delivery

    async def stopped(self, occurrence_id: str) -> None:
        """Tell every sink the occurrence stopped ringing."""
        for sink in self._sinks():
            try:
                await sink.stopped(occurrence_id)
            except _CONTAINED_ERRORS as exc:
                _log_failure(sink, "stopped", exc)

    async def state_changed(
        self, occurrences: Sequence[Occurrence], rev: int, at: datetime
    ) -> None:
        """Build one `alarm_state` payload and pass it to every sink.

        Raises:
            pydantic.ValidationError: A value falls outside the protocol.
        """
        payload = alarm_state_payload(occurrences, rev, at, settings=self._settings, tz=self._tz)
        for sink in self._sinks():
            try:
                await sink.state_changed(payload)
            except _CONTAINED_ERRORS as exc:
                _log_failure(sink, "state_changed", exc)

    def _sinks(self) -> tuple[RingSink, ...]:
        """Return the fixed sinks, then the usable ones the source gives now.

        The source is read once per call. A sourced sink whose name could not be stored, or
        that an earlier sink already has, is left out and logged once per name, and the other
        endpoints still ring: raising here would fail the ring itself.
        """
        if self._source is None:
            return self._fixed
        sinks = list(self._fixed)
        taken = {sink.name for sink in sinks}
        for sink in self._source():
            name = sink.name
            if _storable(name) and name not in taken:
                sinks.append(sink)
                taken.add(name)
                continue
            if name not in self._reported:
                self._reported.add(name)
                reason = "repeats an earlier endpoint's" if _storable(name) else "is not storable"
                logger.error(
                    "Ring sink %s (%r) is skipped: its name %s; ring_on cannot name it",
                    type(sink).__name__,
                    name,
                    reason,
                )
        return tuple(sinks)


def _storable(name: str) -> bool:
    """Whether `AlarmRecords` can store `name` in its comma list beside the word `all`."""
    return bool(name) and "," not in name and name != "all"


def _resolve(sinks: Sequence[RingSink], ring_on: RingTarget) -> tuple[tuple[RingSink, ...], bool]:
    """`RingRouter.resolve` over sinks already read."""
    present = tuple(sink for sink in sinks if sink.present())
    if ring_on == "all":
        return present, False
    named = tuple(sink for sink in present if sink.name in ring_on)
    if named:
        return named, False
    return present, bool(present)


class DesktopSink:
    """The `desktop` endpoint: every connection of the daily WebSocket, through the event sink.

    The event sink broadcasts to every connection, so this sink is present while at least one
    is open, which the callable it is given reports.
    """

    name = DESKTOP

    def __init__(self, event_sink: EventSink, present: Callable[[], bool]) -> None:
        """Keep the broadcast sink and the callable that says whether a client is connected."""
        self._sink = event_sink
        self._present = present

    def present(self) -> bool:
        """Return what the callable says now."""
        return self._present()

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float, *, fallback: bool) -> bool:
        """Emit `alarm_fired` when a client is connected; return whether one was."""
        if not self._present():
            return False
        self._sink.emit(alarm_fired_payload(occurrence, ring_timeout_s, fallback=fallback))
        return True

    async def stopped(self, occurrence_id: str) -> None:
        """Emit nothing: protocol 1.4.0 has no stop event, and the `alarm_state` that follows
        every stop carries the occurrence's new state.
        """

    async def state_changed(self, payload: Mapping[str, object]) -> None:
        """Emit the `alarm_state` payload."""
        self._sink.emit(dict(payload))


def _log_failure(sink: RingSink, method: str, exc: BaseException) -> None:
    logger.error(
        "Ring sink %s (%s) failed in %s: %s",
        type(sink).__name__,
        sink.name,
        method,
        exc,
        exc_info=exc,
    )
