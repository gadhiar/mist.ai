"""Tests for endpoint-neutral ringing: `RingRouter`'s routing and delivery record, `DesktopSink`.

The router runs over real `AlarmRecords`, so what it stores (`sent_to` and the `fallback` event)
is read back from the database rather than from a fake.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Collection, Mapping
from datetime import UTC, time, timedelta

import pytest

from backend.capabilities.alarms.notifier import alarm_state_payload
from backend.capabilities.alarms.records import AlarmRecords, ArmRequest, Occurrence, RingTarget
from backend.capabilities.alarms.ringing import (
    FALLBACK_AFTER_S_DEFAULT,
    DesktopSink,
    RingDelivery,
    RingRouter,
)
from backend.capabilities.config import AlarmSettings, load_capabilities_config
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.events import AlarmFiredEvent
from backend.errors import CapabilityStoreError
from tests.unit.capabilities.fakes import DEFAULT_START, FakeClock, RecordingEventSink

T0 = DEFAULT_START
RINGING_LOGGER = "backend.capabilities.alarms.ringing"
SETTINGS = AlarmSettings(
    ring_timeout_s=600.0, snooze_minutes=9, missed_grace_s=900.0, fallback_after_s=30.0
)


class FakeSink:
    """A `RingSink` whose presence is an attribute, recording every call it gets.

    `delivers` is what `ring` answers: whether it reached a live connection.
    """

    def __init__(self, name: str, *, present: bool = True, delivers: bool = True) -> None:
        self.name = name
        self.is_present = present
        self.delivers = delivers
        self.rings: list[tuple[str, float, bool]] = []
        self.stops: list[str] = []
        self.states: list[Mapping[str, object]] = []

    def present(self) -> bool:
        return self.is_present

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float, *, fallback: bool) -> bool:
        self.rings.append((occurrence.occurrence_id, ring_timeout_s, fallback))
        return self.delivers

    async def stopped(self, occurrence_id: str) -> None:
        self.stops.append(occurrence_id)

    async def state_changed(self, payload: Mapping[str, object]) -> None:
        self.states.append(payload)


class BrokenSink(FakeSink):
    """A `FakeSink` that records each call, then raises `error`."""

    def __init__(self, name: str, error: BaseException) -> None:
        super().__init__(name)
        self.error = error

    async def ring(self, occurrence: Occurrence, ring_timeout_s: float, *, fallback: bool) -> bool:
        await super().ring(occurrence, ring_timeout_s, fallback=fallback)
        raise self.error

    async def stopped(self, occurrence_id: str) -> None:
        await super().stopped(occurrence_id)
        raise self.error

    async def state_changed(self, payload: Mapping[str, object]) -> None:
        await super().state_changed(payload)
        raise self.error


class UnwritableRecords(AlarmRecords):
    """Real `AlarmRecords` whose `sent` fails, as a locked database would."""

    async def sent(  # type: ignore[override]
        self, occurrence_id: str, endpoints: Collection[str], **kwargs: object
    ) -> bool:
        raise CapabilityStoreError("database is locked")


@pytest.fixture()
def db(tmp_path):
    database = CapabilitiesDb(str(tmp_path / "capabilities.db"))
    yield database
    database.close()


@pytest.fixture()
def records(db):
    store = AlarmRecords(db)
    store.migrate()
    return store


def _router(records: AlarmRecords, *sinks: FakeSink) -> RingRouter:
    return RingRouter(list(sinks), records, FakeClock(), SETTINGS, UTC)


async def _ringing(records: AlarmRecords, ring_on: RingTarget = "all") -> Occurrence:
    """Arm a one-off alarm at T0 for `ring_on` and fire it, as the scheduler's tick does."""
    request = ArmRequest(
        kind="alarm",
        fire_at=T0,
        wall_time=time(7, 0),
        duration_s=None,
        label=None,
        days=None,
        ring_on=ring_on,
    )
    armed = await records.arm(request, at=T0 - timedelta(hours=1), tz=UTC, source="fastpath")
    occurrence_id = armed.occurrence.occurrence_id
    assert await records.fire(
        occurrence_id, outcome="ringing", at=T0, late_by_s=0.0, next_fire_at=None
    )
    return await _live(records, occurrence_id)


async def _live(records: AlarmRecords, occurrence_id: str) -> Occurrence:
    _, live = await records.snapshot()
    return next(o for o in live if o.occurrence_id == occurrence_id)


async def _events(db: CapabilitiesDb) -> list[tuple[str, object]]:
    def read(conn: sqlite3.Connection) -> list[tuple[str, object]]:
        rows = conn.execute("SELECT kind, detail FROM alarm_events ORDER BY seq").fetchall()
        return [(kind, None if detail is None else json.loads(detail)) for kind, detail in rows]

    return await db.run(read)


class TestRouting:
    @pytest.mark.asyncio
    async def test_all_rings_every_present_sink(self, db, records):
        desktop, room = FakeSink("desktop"), FakeSink("room")
        occurrence = await _ringing(records)

        delivery = await _router(records, desktop, room).ring(occurrence, 600.0)

        assert delivery == RingDelivery(sent_to=("desktop", "room"), fallback=False)
        assert desktop.rings == [(occurrence.occurrence_id, 600.0, False)]
        assert room.rings == [(occurrence.occurrence_id, 600.0, False)]
        stored = await _live(records, occurrence.occurrence_id)
        assert stored.sent_to == frozenset({"desktop", "room"})
        assert "fallback" not in [kind for kind, _ in await _events(db)]

    @pytest.mark.asyncio
    async def test_all_leaves_out_a_sink_that_is_not_present(self, records):
        desktop, room = FakeSink("desktop"), FakeSink("room", present=False)
        occurrence = await _ringing(records)

        delivery = await _router(records, desktop, room).ring(occurrence, 600.0)

        assert delivery == RingDelivery(sent_to=("desktop",), fallback=False)
        assert room.rings == []

    @pytest.mark.asyncio
    async def test_a_named_sink_rings_alone_while_it_is_present(self, db, records):
        desktop, room = FakeSink("desktop"), FakeSink("room")
        occurrence = await _ringing(records, ("room",))

        delivery = await _router(records, desktop, room).ring(occurrence, 600.0)

        assert delivery == RingDelivery(sent_to=("room",), fallback=False)
        assert desktop.rings == []
        assert room.rings == [(occurrence.occurrence_id, 600.0, False)]
        assert (await _live(records, occurrence.occurrence_id)).sent_to == frozenset({"room"})
        assert "fallback" not in [kind for kind, _ in await _events(db)]

    @pytest.mark.asyncio
    async def test_an_absent_named_sink_falls_back_to_desktop_and_records_it(self, db, records):
        desktop, room = FakeSink("desktop"), FakeSink("room", present=False)
        occurrence = await _ringing(records, ("room",))

        delivery = await _router(records, desktop, room).ring(occurrence, 600.0)

        assert delivery == RingDelivery(sent_to=("desktop",), fallback=True)
        assert desktop.rings == [(occurrence.occurrence_id, 600.0, True)]
        assert room.rings == []
        assert (await _live(records, occurrence.occurrence_id)).sent_to == frozenset({"desktop"})
        assert (await _events(db))[-1] == ("fallback", {"sent_to": ["desktop"]})

    @pytest.mark.parametrize("ring_on", ["all", ("room",)], ids=["all", "room"])
    @pytest.mark.asyncio
    async def test_with_no_sink_present_nothing_rings(self, db, records, ring_on):
        desktop = FakeSink("desktop", present=False)
        room = FakeSink("room", present=False)
        occurrence = await _ringing(records, ring_on)

        delivery = await _router(records, desktop, room).ring(occurrence, 600.0)

        assert delivery == RingDelivery(sent_to=(), fallback=False)
        assert desktop.rings == room.rings == []
        assert (await _live(records, occurrence.occurrence_id)).sent_to == frozenset()
        assert [kind for kind, _ in await _events(db)] == ["set", "fired"]

    @pytest.mark.asyncio
    async def test_a_sink_that_reaches_no_connection_is_not_in_sent_to(self, records):
        desktop, room = FakeSink("desktop", delivers=False), FakeSink("room")
        occurrence = await _ringing(records)

        delivery = await _router(records, desktop, room).ring(occurrence, 600.0)

        assert delivery.sent_to == ("room",)
        assert (await _live(records, occurrence.occurrence_id)).sent_to == frozenset({"room"})

    def test_resolve_and_endpoints(self, records):
        desktop, room = FakeSink("desktop"), FakeSink("room", present=False)
        router = _router(records, desktop, room)

        assert router.endpoints() == ("desktop", "room")
        assert router.resolve("all") == ((desktop,), False)
        assert router.resolve(("desktop",)) == ((desktop,), False)
        assert router.resolve(("room",)) == ((desktop,), True)
        desktop.is_present = False
        assert router.resolve(("room",)) == ((), False)

    @pytest.mark.parametrize("names", [("desktop", "desktop"), ("",), ("a,b",), ("all",)])
    def test_sink_names_must_be_unique_endpoint_names(self, records, names):
        with pytest.raises(ValueError, match="sink"):
            _router(records, *(FakeSink(name) for name in names))


class TestContainment:
    @pytest.mark.parametrize(
        "error", [OSError("socket gone"), CapabilityStoreError("room unit store down")]
    )
    @pytest.mark.asyncio
    async def test_a_failing_sink_is_logged_by_class_and_the_others_still_ring(
        self, records, caplog, error
    ):
        broken, room = BrokenSink("desktop", error), FakeSink("room")
        router = _router(records, broken, room)
        occurrence = await _ringing(records)

        with caplog.at_level(logging.ERROR, logger=RINGING_LOGGER):
            delivery = await router.ring(occurrence, 600.0)
            await router.stopped(occurrence.occurrence_id)
            await router.state_changed([occurrence], 3, T0)

        assert delivery == RingDelivery(sent_to=("room",), fallback=False)
        assert len(room.rings) == len(room.stops) == len(room.states) == 1
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 3
        assert all("BrokenSink" in r.getMessage() for r in errors)

    @pytest.mark.asyncio
    async def test_an_error_outside_the_contained_set_propagates(self, records):
        router = _router(records, BrokenSink("desktop", RuntimeError("bug")))

        with pytest.raises(RuntimeError, match="bug"):
            await router.stopped("o1")

    @pytest.mark.asyncio
    async def test_a_failed_delivery_write_is_logged_and_the_delivery_returned(self, db, caplog):
        store = UnwritableRecords(db)
        store.migrate()
        occurrence = await _ringing(store)
        router = RingRouter([FakeSink("desktop")], store, FakeClock(), SETTINGS, UTC)

        with caplog.at_level(logging.ERROR, logger=RINGING_LOGGER):
            delivery = await router.ring(occurrence, 600.0)

        assert delivery == RingDelivery(sent_to=("desktop",), fallback=False)
        [error] = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert occurrence.occurrence_id in error.getMessage()


class TestBroadcasts:
    @pytest.mark.asyncio
    async def test_state_changed_passes_one_alarm_state_payload_to_every_sink(self, records):
        desktop, room = FakeSink("desktop"), FakeSink("room", present=False)
        occurrence = await _ringing(records)

        await _router(records, desktop, room).state_changed([occurrence], 7, T0)

        expected = alarm_state_payload([occurrence], 7, T0, settings=SETTINGS, tz=UTC)
        assert desktop.states == room.states == [expected]
        assert expected["fallback_after_s"] == 30.0

    @pytest.mark.asyncio
    async def test_stopped_reaches_every_sink(self, records):
        desktop, room = FakeSink("desktop"), FakeSink("room", present=False)

        await _router(records, desktop, room).stopped("o1")

        assert desktop.stops == room.stops == ["o1"]


class TestDesktopSink:
    @pytest.mark.asyncio
    async def test_present_reads_the_callable_at_each_call(self):
        connected = [False]
        sink = DesktopSink(RecordingEventSink(), lambda: connected[0])

        assert sink.name == "desktop"
        assert sink.present() is False
        connected[0] = True
        assert sink.present() is True

    @pytest.mark.parametrize("fallback", [False, True])
    @pytest.mark.asyncio
    async def test_ring_emits_alarm_fired_through_the_event_sink(self, records, fallback):
        events = RecordingEventSink()
        occurrence = await _ringing(records)

        delivered = await DesktopSink(events, lambda: True).ring(
            occurrence, 600.0, fallback=fallback
        )

        assert delivered is True
        [payload] = events.payloads
        assert payload["type"] == "alarm_fired"
        assert payload["alarm_id"] == occurrence.occurrence_id
        assert payload["fallback"] is fallback
        assert AlarmFiredEvent.model_validate(payload).to_payload() == payload

    @pytest.mark.asyncio
    async def test_ring_with_no_connection_emits_nothing(self, records):
        events = RecordingEventSink()
        occurrence = await _ringing(records)

        delivered = await DesktopSink(events, lambda: False).ring(occurrence, 600.0, fallback=False)

        assert delivered is False
        assert events.payloads == []

    @pytest.mark.asyncio
    async def test_state_changed_emits_the_payload_and_stopped_emits_nothing(self):
        events = RecordingEventSink()
        sink = DesktopSink(events, lambda: True)
        payload = alarm_state_payload([], 2, T0, settings=SETTINGS, tz=UTC)

        await sink.state_changed(payload)
        await sink.stopped("o1")

        assert events.payloads == [payload]

    @pytest.mark.asyncio
    async def test_a_router_over_the_desktop_sink_rings_and_stores_the_delivery(self, records):
        events = RecordingEventSink()
        router = RingRouter(
            [DesktopSink(events, lambda: True)], records, FakeClock(), SETTINGS, UTC
        )
        occurrence = await _ringing(records)

        delivery = await router.ring(occurrence, 600.0)

        assert delivery == RingDelivery(sent_to=("desktop",), fallback=False)
        assert [p["type"] for p in events.payloads] == ["alarm_fired"]
        assert (await _live(records, occurrence.occurrence_id)).sent_to == frozenset({"desktop"})


def test_the_fallback_default_is_the_configuration_default():
    assert FALLBACK_AFTER_S_DEFAULT == 30.0
    assert load_capabilities_config({}).alarms.fallback_after_s == FALLBACK_AFTER_S_DEFAULT
