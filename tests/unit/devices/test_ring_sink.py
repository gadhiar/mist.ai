"""Tests for `DeviceRingSink`: a room device's own ring endpoint.

The sink is checked alone, under `RingRouter` beside the desktop sink over a fake session
registry, and end to end: a real capability host built by `build_capability_host` with the
device's sink rings a timer on a fake device served by the real device app, and the device's
`alarm_action` stop ends it, recorded from the device's endpoint.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from datetime import UTC, time, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from backend import factories
from backend.capabilities.alarms.notifier import alarm_fired_payload
from backend.capabilities.alarms.records import AlarmRecords, ArmRequest, Occurrence, RingTarget
from backend.capabilities.alarms.ringing import DesktopSink, RingRouter, RingSink
from backend.capabilities.config import AlarmSettings, load_capabilities_config
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.replies import NoAddress
from backend.capabilities.secrets import EnvSecretSource, SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.devices.limiter import AuthLimiter
from backend.devices.listener import build_device_app
from backend.devices.registry import DeviceRegistry
from backend.devices.ring_sink import DeviceRingSink, device_ring_sinks
from backend.factories import build_capability_host
from tests.unit.capabilities.fakes import (
    DEFAULT_START,
    FakeClock,
    FakePower,
    FakeTiers,
    RecordingEventSink,
)
from tests.unit.capabilities.test_factory import ZONE, ZONE_NAME
from tests.unit.devices.test_device_turn import DeviceClient
from tests.unit.devices.test_listener import ROOM, entry, secret_for, write_registry
from tests.unit.devices.test_listener import FakeClock as ListenerClock
from tests.unit.realtime.turn_fakes import settle
from tests.unit.test_session_id_propagation import (  # noqa: F401 -- autouse fixtures
    RecordingConversationHandler,
    install_runtime,
    no_runtime,
    reset_session_context,
)

DEVICE = "example-room"
OTHER = "other-room"
RING_TIMEOUT_S = 600.0
SETTINGS = AlarmSettings(
    ring_timeout_s=RING_TIMEOUT_S, snooze_minutes=9, missed_grace_s=900.0, fallback_after_s=30.0
)


class FakeSessions:
    """The session registry's side the sinks use: which endpoints are connected, what was sent."""

    def __init__(self, *connected: str) -> None:
        self.connected = set(connected)
        self.sent: list[tuple[str, dict[str, object]]] = []

    def present(self, endpoint: str) -> bool:
        return endpoint in self.connected

    def send(self, endpoint: str, payload: dict[str, object]) -> None:
        self.sent.append((endpoint, payload))


@pytest.fixture()
def db(tmp_path):
    database = CapabilitiesDb(str(tmp_path / "ring.db"))
    yield database
    database.close()


@pytest.fixture()
def records(db):
    store = AlarmRecords(db)
    store.migrate()
    return store


async def _ringing(records: AlarmRecords, ring_on: RingTarget = "all") -> Occurrence:
    """Arm a one-off alarm for `ring_on` and fire it, as the scheduler's tick does."""
    request = ArmRequest(
        kind="alarm",
        fire_at=DEFAULT_START,
        wall_time=time(7, 0),
        duration_s=None,
        label=None,
        days=None,
        ring_on=ring_on,
    )
    armed = await records.arm(
        request, at=DEFAULT_START - timedelta(hours=1), tz=UTC, source="fastpath"
    )
    occurrence_id = armed.occurrence.occurrence_id
    assert await records.fire(
        occurrence_id, outcome="ringing", at=DEFAULT_START, late_by_s=0.0, next_fire_at=None
    )
    _, live = await records.snapshot()
    return next(o for o in live if o.occurrence_id == occurrence_id)


# -- The sink alone ----------------------------------------------------------------------------


class TestDeviceRingSink:
    def test_it_is_named_as_the_device_and_present_while_its_endpoint_is(self):
        sessions = FakeSessions()
        sink: RingSink = DeviceRingSink(DEVICE, sessions.send, sessions.present)

        assert sink.name == DEVICE
        assert sink.present() is False
        sessions.connected.add(DEVICE)
        assert sink.present() is True

    @pytest.mark.asyncio
    async def test_ring_sends_alarm_fired_to_its_own_endpoint_only(self, records):
        sessions = FakeSessions(DEVICE, OTHER)
        occurrence = await _ringing(records)

        reached = await DeviceRingSink(DEVICE, sessions.send, sessions.present).ring(
            occurrence, RING_TIMEOUT_S, fallback=True
        )

        assert reached is True
        assert sessions.sent == [
            (DEVICE, alarm_fired_payload(occurrence, RING_TIMEOUT_S, fallback=True))
        ]

    @pytest.mark.asyncio
    async def test_ring_with_no_connection_sends_nothing_and_says_so(self, records):
        sessions = FakeSessions(OTHER)
        occurrence = await _ringing(records)

        reached = await DeviceRingSink(DEVICE, sessions.send, sessions.present).ring(
            occurrence, RING_TIMEOUT_S, fallback=False
        )

        assert reached is False
        assert sessions.sent == []

    @pytest.mark.asyncio
    async def test_stopped_and_state_changed_send_nothing(self):
        sessions = FakeSessions(DEVICE)
        sink = DeviceRingSink(DEVICE, sessions.send, sessions.present)

        await sink.stopped("occ-1")
        await sink.state_changed({"type": "alarm_state", "alarms": []})

        assert sessions.sent == []

    def test_one_sink_per_registered_device_that_is_not_revoked(self, tmp_path):
        path = tmp_path / "registry.json"
        write_registry(
            path,
            [
                entry(DEVICE, secret_for(1)),
                entry("old-room", secret_for(2), revoked=True),
                entry(OTHER, secret_for(3)),
            ],
        )
        sessions = FakeSessions(DEVICE)

        sinks = device_ring_sinks(
            DeviceRegistry.load(path).devices(), sessions.send, sessions.present
        )

        assert [sink.name for sink in sinks] == [DEVICE, OTHER]
        assert [sink.present() for sink in sinks] == [True, False]


# -- Under the router, beside the desktop ------------------------------------------------------


class TestRouting:
    def _router(
        self, records: AlarmRecords, sessions: FakeSessions, desktop: RecordingEventSink
    ) -> RingRouter:
        sinks: list[RingSink] = [
            DesktopSink(desktop, lambda: True),
            DeviceRingSink(DEVICE, sessions.send, sessions.present),
            DeviceRingSink(OTHER, sessions.send, sessions.present),
        ]
        return RingRouter(sinks, records, FakeClock(), SETTINGS, UTC)

    @pytest.mark.asyncio
    async def test_a_ring_naming_a_present_device_rings_that_device_only(self, records):
        sessions, desktop = FakeSessions(DEVICE, OTHER), RecordingEventSink()
        router = self._router(records, sessions, desktop)
        occurrence = await _ringing(records, ring_on=frozenset({DEVICE}))

        delivery = await router.ring(occurrence, RING_TIMEOUT_S)

        assert router.endpoints() == ("desktop", DEVICE, OTHER)
        assert (delivery.sent_to, delivery.fallback) == ((DEVICE,), False)
        assert [endpoint for endpoint, _ in sessions.sent] == [DEVICE]
        assert desktop.of_type("alarm_fired") == []

    @pytest.mark.asyncio
    async def test_a_ring_for_an_absent_device_falls_back_to_every_present_sink(self, records):
        sessions, desktop = FakeSessions(OTHER), RecordingEventSink()
        router = self._router(records, sessions, desktop)
        occurrence = await _ringing(records, ring_on=frozenset({DEVICE}))

        delivery = await router.ring(occurrence, RING_TIMEOUT_S)

        assert (delivery.sent_to, delivery.fallback) == (("desktop", OTHER), True)
        (fired,) = desktop.of_type("alarm_fired")
        assert fired["fallback"] is True
        assert sessions.sent == [
            (OTHER, alarm_fired_payload(occurrence, RING_TIMEOUT_S, fallback=True))
        ]


# -- End to end: rung on the device, stopped from the device -----------------------------------


def _events(db_path: Path, occurrence_id: str) -> list[tuple[str, str | None, str | None]]:
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT kind, source, endpoint FROM alarm_events WHERE occurrence_id = ? ORDER BY seq",
            (occurrence_id,),
        ).fetchall()
    return list(rows)


def _send_to_endpoint(endpoint: str, payload: Mapping[str, object]) -> None:
    """What `lifespan` gives each sink: the registry read when called, the payload as text."""
    from backend import server

    registry = server.session_registry
    if registry is not None:
        registry.broadcast(json.dumps(payload), endpoint=endpoint)


def _endpoint_present(endpoint: str) -> bool:
    from backend import server

    registry = server.session_registry
    return registry is not None and registry.endpoint_present(endpoint)


@pytest_asyncio.fixture
async def device_host(monkeypatch, tmp_path):
    """A real host with the device's ring sink, installed as `lifespan` leaves it."""
    from backend import server

    monkeypatch.setenv("MIST_EVAL_ISOLATION", "0")
    monkeypatch.delenv("MIST_HYDRATION_ISOLATION", raising=False)
    monkeypatch.delenv("MIST_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setattr(factories, "resolve_user_timezone", lambda environ: ZONE)
    db_path = tmp_path / "capabilities.db"
    config = load_capabilities_config(
        {
            "MIST_CAPABILITIES_ENABLED": "true",
            "MIST_CAPABILITIES_DB_PATH": str(db_path),
            "MIST_CAPABILITIES": "alarms,timers,clock",
            "MIST_TIMEZONE": ZONE_NAME,
        }
    )
    clock = FakeClock()
    build = build_capability_host(
        config,
        event_sink=RecordingEventSink(),
        secret_source=EnvSecretSource({}, lambda path: ""),
        secret_registry=SecretRegistry(),
        clock=clock,
        embedder=None,
        token_counter=ApproximateTokenCounter(),
        transport_factory=None,
        power=FakePower(),
        tiers=FakeTiers(),
        register=NoAddress(),
        desktop_present=lambda: False,
        ring_sinks=(DeviceRingSink(DEVICE, _send_to_endpoint, _endpoint_present),),
    )
    assert build.host is not None
    monkeypatch.setattr(server, "_capability_build", build)
    monkeypatch.setattr(server, "message_queue", asyncio.Queue())
    await build.host.start()
    await build.host.wait_started()
    yield build.host, clock, db_path
    await build.host.stop()
    await build.close()


@pytest_asyncio.fixture
async def device(monkeypatch, tmp_path, device_host):
    """The fake device on the real device app, served as `lifespan` serves it."""
    from backend import server

    install_runtime(monkeypatch, RecordingConversationHandler())
    path = tmp_path / "registry.json"
    write_registry(path, [entry(DEVICE, secret_for(1))])
    listener_clock = ListenerClock()
    app = build_device_app(
        DeviceRegistry.load(path),
        server._serve_session,
        AuthLimiter(listener_clock),
        listener_clock,
    )
    client = DeviceClient(app, f"Bearer {ROOM}")
    await settle()
    yield client
    if not client.task.done():
        client.leave()
        await client.ended()


async def _until(predicate) -> None:
    for _ in range(200):
        await settle()
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached in time")


async def _ring_a_timer_on_the_device(host, clock: FakeClock, device: DeviceClient) -> str:
    """Set a one-minute timer, let it fire, and return the id the device was rung with."""
    handled = await host.pre_llm("s1", "t1", "set a timer for 1 minute")
    assert handled is not None
    await clock.advance(61.0)
    await _until(lambda: bool(device.of_type("alarm_fired")))
    (fired,) = device.of_type("alarm_fired")
    assert fired["fallback"] is False
    return str(fired["alarm_id"])


@pytest.mark.asyncio
async def test_a_timer_rings_on_the_device_and_its_stop_ends_it_from_that_endpoint(
    device_host, device
):
    host, clock, db_path = device_host
    assert device.accepted
    occurrence_id = await _ring_a_timer_on_the_device(host, clock, device)

    device.send(
        {"type": "alarm_action", "alarm_id": occurrence_id, "action": "stop", "request_id": "s-1"}
    )
    await _until(lambda: bool(device.of_type("alarm_action_result")))

    (result,) = device.of_type("alarm_action_result")
    assert (result["action"], result["outcome"]) == ("stop", "ok")
    stops = [e for e in _events(db_path, occurrence_id) if e[0] == "stopped"]
    assert stops == [("stopped", "alarm_action", DEVICE)]
    # The ring is over: a spoken stop now finds nothing ringing and is not taken.
    assert await host.pre_llm("s1", "t2", "stop the timer") is None


@pytest.mark.asyncio
async def test_the_spoken_stop_ends_a_ring_on_the_device_as_the_device_stop_does(
    device_host, device
):
    host, clock, db_path = device_host
    occurrence_id = await _ring_a_timer_on_the_device(host, clock, device)

    spoken = await host.pre_llm("s1", "t2", "stop the timer")

    assert spoken is not None and spoken.handled
    assert [e[0] for e in _events(db_path, occurrence_id)].count("stopped") == 1
    device.send(
        {"type": "alarm_action", "alarm_id": occurrence_id, "action": "stop", "request_id": "s-2"}
    )
    await _until(lambda: bool(device.of_type("alarm_action_result")))
    (result,) = device.of_type("alarm_action_result")
    assert result["outcome"] == "invalid_args"
