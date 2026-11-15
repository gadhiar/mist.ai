"""Offline degradation of the real capability host, assembled by `build_capability_host`.

Home Assistant refuses every connection and every Spotify request fails to connect, so lights
and music are offline from the start. The host must still start without waiting on either,
answer their commands with a templated `unavailable` reply that names the reason, report both
reasons in `capabilities_status`, and keep alarms and timers working, because those need no
network and no model.

The Home Assistant fake goes in through the factory's `transport_factory`. The factory builds
music's `httpx.AsyncClient` itself, so the Spotify `MockTransport` goes in by patching
`factories.httpx.AsyncClient` for the test, as `test_factory.py` does for its disabled build.
`tests/unit/conftest.py` sets `MIST_EVAL_ISOLATION`, which disables the host, so the `offline`
fixture switches it off. The worker image may lack a tz database, so `MIST_TIMEZONE` resolves
to a fixed-offset `ZoneInfo` built from TZif bytes.
"""

from __future__ import annotations

import asyncio
import io
import json
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import httpx
import pytest
import pytest_asyncio

from backend import factories
from backend.capabilities.config import load_capabilities_config
from backend.capabilities.contracts import DecodeReply
from backend.capabilities.host import CapabilityHost
from backend.capabilities.music.token_store import SpotifyToken, SpotifyTokenStore
from backend.capabilities.power import PowerLevel
from backend.capabilities.replies import NoAddress
from backend.capabilities.secrets import EnvSecretSource, Secret, SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.factories import CapabilityBuild, build_capability_host
from backend.llm.provider import StreamingLLMProvider
from tests.unit.capabilities.fakes import (
    DEFAULT_START,
    FakeClock,
    FakePower,
    FakeTiers,
    RecordingEventSink,
    ScriptedDecode,
)
from tests.unit.capabilities.mcp.fake_home_assistant import FakeHomeAssistant, connect_refused

COUNTER = ApproximateTokenCounter()
HA_URL = "http://homeassistant.invalid:8123/api/mcp"
HA_TOKEN = "ha-token-for-offline-tests"  # noqa: S105 - a fake token
ZONE_NAME = "Test/Zone"
ALLOWED_ORIGIN = "http://tauri.localhost"
PROVIDER = cast(StreamingLLMProvider, object())
"""Stands in for the interaction tier: the decode itself is a `ScriptedDecode`."""

ALL_CAPABILITIES = "alarms,timers,clock,lights,music"
RING_TIMEOUT_S = 60.0
ALARM_AT = DEFAULT_START + timedelta(minutes=5)
"""07:05 on the fakes' fixed Monday; the zone has no offset, so this is also the wall time."""

LIGHTS_REASON = "connection refused"
MUSIC_REASON = "unreachable"
ROUTED_LIGHTS_REQUEST = "it is too dark in the bedroom"
"""Routed to lights by the keyword "dark", and matched by no fast-path row."""

SCHEDULER_SLEEP_TASK = "alarm-scheduler-sleep"
"""The name of the task in which `AlarmScheduler._sleep` calls `Clock.sleep`."""


def _fixed_zone(key: str = ZONE_NAME, offset_s: int = 0) -> ZoneInfo:
    """A `ZoneInfo` with one fixed offset, built from TZif v1 bytes, no tz database needed."""
    header = b"TZif" + b"\0" + b"\0" * 15 + struct.pack(">6l", 0, 0, 0, 0, 1, 4)
    body = struct.pack(">lBB", offset_s, 0, 0) + b"TST\0"
    return ZoneInfo.from_file(io.BytesIO(header + body), key=key)


ZONE = _fixed_zone()


class _TaskClock(FakeClock):
    """A `FakeClock` that also records the deadline of each pending sleep by its task's name.

    The host has several tasks sleeping on the clock at once (the status watcher, the lights
    catalog refresh, the Home Assistant reconnect backoff), so a count of pending sleepers can
    be met by the wrong one. The name tells the alarm scheduler's sleep apart from the rest.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pending: dict[asyncio.Task[object], datetime] = {}

    async def sleep(self, seconds: float) -> None:
        task = asyncio.current_task()
        if seconds <= 0 or task is None:
            await super().sleep(seconds)
            return
        # Recorded in the same step as `FakeClock.sleep` registers the sleeper, so a deadline
        # seen here is one `advance` will wake.
        self._pending[task] = self.now() + timedelta(seconds=seconds)
        try:
            await super().sleep(seconds)
        finally:
            self._pending.pop(task, None)

    def deadline_of(self, task_name: str) -> datetime | None:
        """The deadline of the pending sleep of the task named `task_name`, if it has one."""
        return next((d for t, d in self._pending.items() if t.get_name() == task_name), None)


@dataclass
class Offline:
    """One offline host and what it was given."""

    build: CapabilityBuild
    host: CapabilityHost
    clock: _TaskClock
    power: FakePower
    tiers: FakeTiers
    sink: RecordingEventSink
    home_assistant: FakeHomeAssistant
    spotify_requests: list[httpx.Request]


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Patch in the offline Spotify transport and the zone; return a builder for the host.

    The builder's first call may seed the database with `seed_alarm`, through a host built
    with alarms only, which stands for the backend before the outage.
    """
    monkeypatch.setenv("MIST_EVAL_ISOLATION", "0")
    monkeypatch.delenv("MIST_HYDRATION_ISOLATION", raising=False)
    monkeypatch.setenv("MIST_ALLOWED_ORIGINS", ALLOWED_ORIGIN)
    monkeypatch.setattr(factories, "resolve_user_timezone", lambda environ: ZONE)

    spotify_requests: list[httpx.Request] = []

    def unreachable(request: httpx.Request) -> httpx.Response:
        spotify_requests.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    real_client = httpx.AsyncClient

    def offline_client(**kwargs: object) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(unreachable), **kwargs)

    monkeypatch.setattr(factories.httpx, "AsyncClient", offline_client)

    token_path = tmp_path / "secrets" / "spotify_token.json"
    SpotifyTokenStore(token_path).save(
        SpotifyToken(
            access_token=Secret("spotify-access-for-offline-tests"),
            refresh_token=Secret("spotify-refresh-for-offline-tests"),
            expires_at=DEFAULT_START + timedelta(hours=1),
            scope="user-modify-playback-state",
        )
    )
    env = {
        "MIST_CAPABILITIES_ENABLED": "true",
        "MIST_CAPABILITIES_DB_PATH": str(tmp_path / "capabilities.db"),
        "MIST_TIMEZONE": ZONE_NAME,
        "MIST_ALARM_RING_TIMEOUT_S": str(RING_TIMEOUT_S),
        "MIST_HA_MCP_URL": HA_URL,
        "MIST_SPOTIFY_CLIENT_ID": "spotify-client-for-offline-tests",
        "MIST_SPOTIFY_TOKEN_PATH": str(token_path),
    }
    return _Builder(env, spotify_requests)


class _Builder:
    def __init__(self, env: Mapping[str, str], spotify_requests: list[httpx.Request]) -> None:
        self._env = env
        self._spotify_requests = spotify_requests
        self.builds: list[CapabilityBuild] = []

    def __call__(
        self,
        *,
        capabilities: str = ALL_CAPABILITIES,
        level: PowerLevel = "full",
        tiers: FakeTiers | None = None,
    ) -> Offline:
        home_assistant = FakeHomeAssistant()

        def refusing(url: str, token: object) -> object:
            home_assistant.open_failures.append(connect_refused())
            return home_assistant.transport()

        clock = _TaskClock()
        power = FakePower(level)
        tiers = tiers if tiers is not None else FakeTiers({"interaction": PROVIDER})
        sink = RecordingEventSink()
        build = build_capability_host(
            load_capabilities_config({**self._env, "MIST_CAPABILITIES": capabilities}),
            event_sink=sink,
            secret_source=EnvSecretSource({"MIST_HA_TOKEN": HA_TOKEN}, _read_file),
            secret_registry=SecretRegistry(),
            clock=clock,
            embedder=None,
            token_counter=COUNTER,
            transport_factory=refusing,
            power=power,
            tiers=tiers,
            register=NoAddress(),
            desktop_present=lambda: True,
        )
        self.builds.append(build)
        assert build.host is not None, build.disabled_reason
        return Offline(
            build, build.host, clock, power, tiers, sink, home_assistant, self._spotify_requests
        )

    async def close(self) -> None:
        for build in self.builds:
            if build.host is not None:
                await build.host.stop()
            await build.close()


@pytest_asyncio.fixture
async def builder(offline):
    """The offline builder; stops and closes every host it built at teardown."""
    yield offline
    await offline.close()


def _read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _decoded(*calls: tuple[str, dict[str, object]], also_reply: bool) -> DecodeReply:
    content = json.dumps(
        {"calls": [{"tool": tool, "args": args} for tool, args in calls], "also_reply": also_reply}
    )
    return DecodeReply(content=content, finish_reason="stop")


def _statuses(host: CapabilityHost) -> dict[str, tuple[str, str | None]]:
    return {s.name: (s.state, s.reason) for s in host.statuses()}


async def _until(predicate: Callable[[], bool], *, within_s: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    give_up_at = loop.time() + within_s
    while not predicate():
        assert loop.time() < give_up_at, "condition not reached in time"
        await asyncio.sleep(0.001)


async def _started_offline(rig: Offline) -> None:
    """Start the host and wait until lights report the refused connection."""
    await rig.host.start()
    await rig.host.wait_started()
    await _until(lambda: _statuses(rig.host)["lights"] == ("unavailable", LIGHTS_REASON))


async def _seed_alarm(builder: _Builder) -> None:
    """Set the 07:05 alarm on a host built before the outage, then stop it."""
    before = builder(capabilities="alarms")
    await before.host.start()
    await before.host.wait_started()
    resolution = await before.host.pre_llm("seed", "seed-1", "set an alarm for 7:05")
    assert resolution is not None and resolution.handled
    assert [r.outcome.status for r in resolution.records] == ["ok"]
    await before.host.stop()
    await before.build.close()
    builder.builds.remove(before.build)


async def _advance_with_scheduler(rig: Offline, to: datetime) -> None:
    """Move the clock to `to` one alarm scheduler sleep at a time.

    The scheduler reads the store on a worker thread, then starts its sleep in a task of its
    own, so its sleep is registered some loop iterations after it computed the duration. Time
    that moves in that gap pushes the deadline past the one it computed, and past `to`. Each
    step therefore waits until the scheduler's own sleep is registered and advances exactly to
    its deadline, and the clock stays there while the scheduler ticks and sleeps again.
    """
    while rig.clock.now() < to:
        await _until(lambda: rig.clock.deadline_of(SCHEDULER_SLEEP_TASK) is not None)
        deadline = rig.clock.deadline_of(SCHEDULER_SLEEP_TASK)
        assert deadline is not None and deadline <= to, f"scheduler sleeps until {deadline}"
        await rig.clock.advance((deadline - rig.clock.now()).total_seconds())


async def _advance_to_ring(rig: Offline) -> None:
    """Move the clock to 07:05, then wait for the ring the scheduler sends from its own task.

    The scheduler's tick at 07:05 runs after the `advance` that woke it returns; the clock
    stays at 07:05 while the test waits.
    """
    await _advance_with_scheduler(rig, ALARM_AT)
    await _until(lambda: bool(_fired(rig.sink)))


def _fired(sink: RecordingEventSink) -> list[dict[str, object]]:
    return sink.of_type("alarm_fired")


def _missed(sink: RecordingEventSink) -> list[tuple[object, object]]:
    """The missed occurrences of the last `alarm_state`, as (alarm id, reason) pairs."""
    states = sink.of_type("alarm_state")
    if not states:
        return []
    return [(m["alarm_id"], m["reason"]) for m in states[-1]["missed"]]


def _parse_utc(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


class TestOfflineDegradation:
    @pytest.mark.asyncio
    async def test_offline_lights_and_music_answer_unavailable_while_an_alarm_fires_on_time(
        self, builder
    ):
        await _seed_alarm(builder)
        rig = builder()

        start = asyncio.create_task(rig.host.start())
        await asyncio.sleep(0)
        start_done_in_one_turn = start.done()
        await start
        await rig.host.wait_started()
        await _until(lambda: _statuses(rig.host)["lights"] == ("unavailable", LIGHTS_REASON))
        lights = await rig.host.pre_llm("s1", "t1", "turn off the bedroom lights")
        music = await rig.host.pre_llm("s1", "t2", "pause the music")
        assert await rig.host.pre_llm("s1", "t3", ROUTED_LIGHTS_REQUEST) is None
        decode = ScriptedDecode(
            _decoded(("lights.power", {"target": "bedroom", "on": True}), also_reply=True)
        )
        plan = rig.host.plan_turn("s1", ROUTED_LIGHTS_REQUEST, [])
        routed = await rig.host.run_actions("s1", "t3", ROUTED_LIGHTS_REQUEST, plan, decode)
        status = (await rig.host.snapshot_events())[0]
        await _advance_to_ring(rig)

        assert start_done_in_one_turn
        assert lights is not None and lights.handled
        assert [(r.tool, r.source, r.outcome.status) for r in lights.records] == [
            ("lights.power", "fastpath", "unavailable")
        ]
        assert lights.reply == f"Lights is unavailable: {LIGHTS_REASON}."
        assert music is not None and music.handled
        assert [(r.tool, r.source, r.outcome.status) for r in music.records] == [
            ("music.control", "fastpath", "unavailable")
        ]
        assert music.reply == f"Music is unavailable: {MUSIC_REASON}."
        assert rig.spotify_requests, "the pause reached the Spotify transport"
        assert rig.home_assistant.calls == []
        assert plan.route.capabilities == ("lights",)
        assert len(decode.requests) == 1
        assert [(r.tool, r.source, r.outcome.status) for r in routed.records] == [
            ("lights.power", "model", "unavailable")
        ]
        assert routed.notices == (f"(Lights is unavailable: {LIGHTS_REASON}.)",)
        reasons = {
            entry["name"]: (entry["state"], entry["reason"]) for entry in status["capabilities"]
        }
        assert reasons["lights"] == ("unavailable", LIGHTS_REASON)
        assert reasons["music"] == ("unavailable", MUSIC_REASON)
        (fired,) = _fired(rig.sink)
        assert _parse_utc(fired["scheduled_for"]) == ALARM_AT
        assert _parse_utc(fired["fired_at"]) == ALARM_AT
        assert fired["late_by_s"] == 0.0
        assert rig.clock.stalled == []

    @pytest.mark.asyncio
    async def test_at_the_off_level_the_fast_path_runs_and_no_decode_is_requested(self, builder):
        rig = builder(level="off", tiers=FakeTiers({}))
        await _started_offline(rig)
        decode = ScriptedDecode()

        timer = await rig.host.pre_llm("s1", "t1", "set a timer for 10 minutes")
        assert await rig.host.pre_llm("s1", "t2", ROUTED_LIGHTS_REQUEST) is None
        plan = rig.host.plan_turn("s1", ROUTED_LIGHTS_REQUEST, [])
        routed = await rig.host.run_actions("s1", "t2", ROUTED_LIGHTS_REQUEST, plan, decode)

        assert timer is not None and timer.handled
        assert [(r.tool, r.source, r.outcome.status, r.power_level) for r in timer.records] == [
            ("timers.start", "fastpath", "ok", "off")
        ]
        assert rig.host.decode_available() is False
        assert plan.route.capabilities == ("lights",)
        assert plan.decode is None
        assert routed.records == ()
        assert decode.requests == []

    @pytest.mark.asyncio
    async def test_a_ring_no_endpoint_acknowledges_ends_missed_unheard(self, builder):
        await _seed_alarm(builder)
        rig = builder()
        await _started_offline(rig)

        await _advance_to_ring(rig)
        (fired,) = _fired(rig.sink)
        await _advance_with_scheduler(rig, ALARM_AT + timedelta(seconds=RING_TIMEOUT_S))
        await _until(lambda: bool(_missed(rig.sink)))

        assert _missed(rig.sink) == [(fired["alarm_id"], "unheard")]
        assert rig.sink.of_type("alarm_state")[-1]["ringing"] == []
        assert rig.clock.stalled == []
