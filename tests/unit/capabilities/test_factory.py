"""Tests for `backend.factories.build_capability_host`, the capability composition root.

Every build runs against a temp-file database, a `FakeClock` and fakes for the event sink,
power, tiers and the Home Assistant transport, so nothing here touches the network or a model.
`tests/unit/conftest.py` sets `MIST_EVAL_ISOLATION` for every unit test, so the `enabled`
fixture switches it off for the cases that build a host.

The worker image may lack a tz database. A valid zone is therefore a `ZoneInfo` built from TZif
bytes and served by a patched resolver, and the tzdata-missing case patches the resolver's
answer to `timezone.utc`, as the real resolver gives it.
"""

from __future__ import annotations

import asyncio
import io
import struct
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
import pytest_asyncio

from backend import factories
from backend.capabilities.alarms.capability import RingingTracker
from backend.capabilities.alarms.ringing import DesktopSink, RingRouter
from backend.capabilities.config import CapabilitiesConfig, load_capabilities_config
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import DECODE_INSTRUCTIONS
from backend.capabilities.host import CapabilityHost, HostSettings
from backend.capabilities.replies import NoAddress
from backend.capabilities.secrets import EnvSecretSource, SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.factories import CapabilityBuild, build_capability_host
from tests.unit.capabilities.fakes import FakeClock, FakePower, FakeTiers, RecordingEventSink
from tests.unit.capabilities.mcp.fake_home_assistant import FakeHomeAssistant

COUNTER = ApproximateTokenCounter()
HA_URL = "http://homeassistant.local:8123/api/mcp"
HA_TOKEN = "ha-token-value-0123456789"  # noqa: S105 - a fake token for the redaction check
ZONE_NAME = "Test/Zone"


def _fixed_zone(key: str = ZONE_NAME, offset_s: int = 0) -> ZoneInfo:
    """A `ZoneInfo` with one fixed offset, built from TZif v1 bytes, no tz database needed."""
    header = b"TZif" + b"\0" + b"\0" * 15 + struct.pack(">6l", 0, 0, 0, 0, 1, 4)
    body = struct.pack(">lBB", offset_s, 0, 0) + b"TST\0"
    return ZoneInfo.from_file(io.BytesIO(header + body), key=key)


ZONE = _fixed_zone()


@dataclass
class Rig:
    """What one build was given, so a test can read what the host did with it."""

    sink: RecordingEventSink
    registry: SecretRegistry
    clock: FakeClock
    power: FakePower
    tiers: FakeTiers
    server: FakeHomeAssistant


@pytest.fixture
def enabled(monkeypatch):
    """Clear every flag that disables the host, so the build under test decides."""
    monkeypatch.setenv("MIST_EVAL_ISOLATION", "0")
    monkeypatch.delenv("MIST_HYDRATION_ISOLATION", raising=False)
    monkeypatch.delenv("MIST_ALLOWED_ORIGINS", raising=False)


@pytest.fixture
def zone(monkeypatch):
    """Serve `ZONE_NAME` as a loadable zone, and record each name the factory resolves."""
    resolved: list[str] = []

    def resolver(environ: Mapping[str, str]):
        resolved.append(environ["MIST_TIMEZONE"])
        return ZONE

    monkeypatch.setattr(factories, "resolve_user_timezone", resolver)
    return resolved


@pytest_asyncio.fixture
async def build(tmp_path):
    """Build hosts for a test, and stop and close every build at teardown."""
    builds: list[CapabilityBuild] = []

    def _build(
        env: Mapping[str, str] | None = None,
        *,
        secrets: Mapping[str, str] | None = None,
        transport_factory: Callable[..., object] | None = None,
        desktop_present: Callable[[], bool] = lambda: False,
    ) -> tuple[CapabilityBuild, Rig]:
        rig = Rig(
            sink=RecordingEventSink(),
            registry=SecretRegistry(),
            clock=FakeClock(),
            power=FakePower(),
            tiers=FakeTiers(),
            server=FakeHomeAssistant(),
        )
        config = _config(tmp_path, env)
        result = build_capability_host(
            config,
            event_sink=rig.sink,
            secret_source=EnvSecretSource(dict(secrets or {}), _read_file),
            secret_registry=rig.registry,
            clock=rig.clock,
            embedder=None,
            token_counter=COUNTER,
            transport_factory=transport_factory,
            power=rig.power,
            tiers=rig.tiers,
            register=NoAddress(),
            desktop_present=desktop_present,
        )
        builds.append(result)
        return result, rig

    yield _build
    for result in builds:
        if result.host is not None:
            await result.host.stop()
        await result.close()


def _config(tmp_path: Path, env: Mapping[str, str] | None) -> CapabilitiesConfig:
    base = {
        "MIST_CAPABILITIES_ENABLED": "true",
        "MIST_CAPABILITIES_DB_PATH": str(tmp_path / "capabilities.db"),
    }
    return load_capabilities_config({**base, **(env or {})})


def _read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _statuses(host: CapabilityHost) -> dict[str, tuple[str, str | None]]:
    return {s.name: (s.state, s.reason) for s in host.statuses()}


async def _until(predicate: Callable[[], bool]) -> None:
    loop = asyncio.get_running_loop()
    give_up_at = loop.time() + 2.0
    while not predicate():
        assert loop.time() < give_up_at, "condition not reached in time"
        await asyncio.sleep(0.001)


def _recording(monkeypatch, name: str) -> list[tuple[tuple, dict, object]]:
    """Wrap `factories.<name>` so each construction is recorded with its arguments."""
    real = getattr(factories, name)
    calls: list[tuple[tuple, dict, object]] = []

    def construct(*args, **kwargs):
        built = real(*args, **kwargs)
        calls.append((args, kwargs, built))
        return built

    monkeypatch.setattr(factories, name, construct)
    return calls


# -- Disabled builds ---------------------------------------------------------------------------


class TestDisabled:
    def test_capabilities_switched_off_builds_no_host(self, enabled, build):
        result, _ = build({"MIST_CAPABILITIES_ENABLED": "false"})

        assert result == CapabilityBuild(host=None, disabled_reason="capabilities disabled")

    @pytest.mark.parametrize("origins", ["*", "http://tauri.localhost, *"])
    def test_an_allowlist_that_allows_every_origin_builds_no_host(
        self, enabled, build, monkeypatch, origins
    ):
        monkeypatch.setenv("MIST_ALLOWED_ORIGINS", origins)

        result, _ = build()

        assert result.host is None
        assert result.disabled_reason == "MIST_ALLOWED_ORIGINS allows every origin"

    @pytest.mark.parametrize(
        ("variable", "reason"),
        [
            ("MIST_EVAL_ISOLATION", "eval isolation active (MIST_EVAL_ISOLATION)"),
            ("MIST_HYDRATION_ISOLATION", "hydration isolation active (MIST_HYDRATION_ISOLATION)"),
        ],
    )
    def test_active_isolation_builds_no_host(self, enabled, build, monkeypatch, variable, reason):
        monkeypatch.setenv(variable, "1")

        result, _ = build()

        assert result == CapabilityBuild(host=None, disabled_reason=reason)

    @pytest.mark.parametrize("variable", ["MIST_EVAL_ISOLATION", "MIST_HYDRATION_ISOLATION"])
    def test_an_unreadable_isolation_flag_fails_closed(self, enabled, build, monkeypatch, variable):
        monkeypatch.setenv(variable, "maybe")

        result, _ = build()

        assert result.host is None
        assert result.disabled_reason is not None
        assert result.disabled_reason.startswith("isolation flag invalid: ")
        assert variable in result.disabled_reason

    def test_the_unit_tier_isolation_alone_builds_no_host(self, build):
        result, _ = build()

        assert result.host is None
        assert result.disabled_reason == "eval isolation active (MIST_EVAL_ISOLATION)"

    def test_an_enabled_config_builds_a_host_and_no_reason(self, enabled, build):
        result, _ = build({"MIST_CAPABILITIES": "clock"})

        assert isinstance(result.host, CapabilityHost)
        assert result.disabled_reason is None


# -- Time zone ---------------------------------------------------------------------------------


class TestZone:
    @pytest.mark.asyncio
    async def test_unset_zone_skips_the_resolver_and_builds_no_scheduler(
        self, enabled, build, monkeypatch
    ):
        def resolver(environ):
            raise AssertionError("the resolver is not called for an unset MIST_TIMEZONE")

        monkeypatch.setattr(factories, "resolve_user_timezone", resolver)
        schedulers = _recording(monkeypatch, "AlarmScheduler")
        controls = _recording(monkeypatch, "SchedulerAlarmControl")

        result, _ = build({"MIST_CAPABILITIES": "alarms,timers,clock"})
        host = result.host
        await host.start()
        await host.wait_started()

        assert schedulers == []
        assert controls == []
        assert _statuses(host) == {
            "alarms": ("unavailable", "MIST_TIMEZONE unset"),
            "timers": ("unavailable", "MIST_TIMEZONE unset"),
            "clock": ("unavailable", "MIST_TIMEZONE unset"),
        }

    @pytest.mark.asyncio
    async def test_an_alarm_action_without_a_zone_answers_with_the_zone_reason(
        self, enabled, build
    ):
        result, _ = build({"MIST_CAPABILITIES": "alarms,timers,clock"})
        host = result.host
        await host.start()
        await host.wait_started()

        outcome = await host.alarm_action("any-id", "stop", None, endpoint="desktop")

        alarms_reason = _statuses(host)["alarms"][1]
        assert outcome.status == "unavailable"
        assert outcome.reason == alarms_reason == "MIST_TIMEZONE unset"
        assert "MIST_TIMEZONE unset" in outcome.summary
        assert "not enabled" not in outcome.summary
        assert "not enabled" not in (outcome.reason or "")
        assert [e["type"] for e in await host.snapshot_events()] == ["capabilities_status"]

    @pytest.mark.asyncio
    async def test_a_valid_zone_builds_the_scheduler_in_it(self, enabled, build, zone, monkeypatch):
        schedulers = _recording(monkeypatch, "AlarmScheduler")
        controls = _recording(monkeypatch, "SchedulerAlarmControl")

        result, _ = build({"MIST_CAPABILITIES": "alarms,timers,clock", "MIST_TIMEZONE": ZONE_NAME})
        host = result.host
        await host.start()
        await host.wait_started()

        assert zone == [ZONE_NAME]
        assert len(schedulers) == 1
        args, _, scheduler = schedulers[0]
        assert args[4] is ZONE
        assert len(controls) == 1
        assert controls[0][0][1] is scheduler
        assert controls[0][0][4] is ZONE
        assert {name: state for name, (state, _) in _statuses(host).items()} == {
            "alarms": "available",
            "timers": "available",
            "clock": "available",
        }

    @pytest.mark.parametrize("answer", ["raises", "utc"])
    @pytest.mark.asyncio
    async def test_a_zone_that_does_not_load_gives_the_not_found_reason(
        self, enabled, build, monkeypatch, answer
    ):
        def resolver(environ):
            if answer == "raises":
                raise ValueError("MIST_TIMEZONE='Nowhere/City' is not a known time zone")
            return UTC

        monkeypatch.setattr(factories, "resolve_user_timezone", resolver)
        schedulers = _recording(monkeypatch, "AlarmScheduler")

        result, _ = build(
            {"MIST_CAPABILITIES": "alarms,timers,clock", "MIST_TIMEZONE": "Nowhere/City"}
        )
        host = result.host
        await host.start()
        await host.wait_started()

        reason = "timezone Nowhere/City not found (tzdata missing?)"
        assert schedulers == []
        assert _statuses(host) == {
            "alarms": ("unavailable", reason),
            "timers": ("unavailable", reason),
            "clock": ("unavailable", reason),
        }
        outcome = await host.alarm_action("any-id", "stop", None, endpoint="desktop")
        assert (outcome.status, outcome.reason) == ("unavailable", reason)


# -- Alarms wiring -----------------------------------------------------------------------------


class TestAlarmsWiring:
    @pytest.mark.asyncio
    async def test_alarms_and_timers_share_one_tracker_one_scheduler_and_the_control(
        self, enabled, build, zone, monkeypatch
    ):
        schedulers = _recording(monkeypatch, "AlarmScheduler")
        routers = _recording(monkeypatch, "RingRouter")
        sinks = _recording(monkeypatch, "DesktopSink")
        alarms = _recording(monkeypatch, "AlarmsCapability")
        timers = _recording(monkeypatch, "TimersCapability")
        controls = _recording(monkeypatch, "SchedulerAlarmControl")
        present = [False]

        result, rig = build(
            {"MIST_CAPABILITIES": "alarms,timers,clock", "MIST_TIMEZONE": ZONE_NAME},
            desktop_present=lambda: present[0],
        )

        (_, _, sink), (router_args, _, router), (scheduler_args, _, scheduler) = (
            sinks[0],
            routers[0],
            schedulers[0],
        )
        assert isinstance(sink, DesktopSink)
        assert sink.present() is False
        present[0] = True
        assert sink.present() is True
        assert list(router_args[0]) == [sink]
        assert router_args[2] is rig.clock
        assert router_args[4] is ZONE
        tracker = scheduler_args[2]
        assert isinstance(tracker, RingingTracker)
        assert isinstance(router, RingRouter)
        records = scheduler_args[0]
        assert router_args[1] is records
        for calls in (alarms, timers):
            args, kwargs, _ = calls[0]
            assert args[0] is records
            assert args[1] is scheduler
            assert kwargs["ringing"] is tracker
            assert kwargs["endpoints"]() == ("desktop",)
        control_args = controls[0][0]
        assert control_args[0] is records
        assert control_args[1] is scheduler

    @pytest.mark.asyncio
    async def test_the_alarm_control_reads_the_store_the_capabilities_write(
        self, enabled, build, zone
    ):
        result, _ = build({"MIST_CAPABILITIES": "alarms,timers,clock", "MIST_TIMEZONE": ZONE_NAME})
        host = result.host
        await host.start()
        await host.wait_started()

        handled = await host.pre_llm("s1", "t1", "set a timer for 5 minutes")
        events = await host.snapshot_events()

        assert handled is not None
        assert [e["type"] for e in events] == ["capabilities_status", "alarm_state"]
        (timer,) = events[1]["alarms"]
        assert timer["kind"] == "timer"
        outcome = await host.alarm_action(timer["alarm_id"], "stop", None, endpoint="desktop")
        assert outcome.status == "invalid_args"
        assert "not ringing" in outcome.summary

    @pytest.mark.asyncio
    async def test_an_unwritable_database_never_fails_the_build(
        self, enabled, build, zone, tmp_path
    ):
        blocker = tmp_path / "blocker"
        blocker.write_text("a regular file where a directory should be")
        db_path = blocker / "capabilities.db"

        result, _ = build(
            {
                "MIST_CAPABILITIES": "alarms,timers,clock",
                "MIST_TIMEZONE": ZONE_NAME,
                "MIST_CAPABILITIES_DB_PATH": str(db_path),
            }
        )
        host = result.host
        assert host is not None
        await host.start()
        await host.wait_started()

        statuses = _statuses(host)
        for name in ("alarms", "timers"):
            state, reason = statuses[name]
            assert state == "unavailable"
            assert str(db_path) in (reason or "")
        assert statuses["clock"] == ("available", None)
        assert [e["type"] for e in await host.snapshot_events()] == ["capabilities_status"]


# -- Lights ------------------------------------------------------------------------------------


class TestLights:
    @pytest.mark.asyncio
    async def test_lights_calls_carry_the_configured_tool_timeout(self, enabled, build):
        server = FakeHomeAssistant()
        urls: list[tuple[str, str]] = []

        def transport(url, token):
            urls.append((url, token.reveal()))
            return server.transport()

        result, rig = build(
            {
                "MIST_CAPABILITIES": "lights",
                "MIST_HA_MCP_URL": HA_URL,
                "MIST_CAPABILITY_TOOL_TIMEOUT_S": "4.5",
            },
            secrets={"MIST_HA_TOKEN": HA_TOKEN},
            transport_factory=transport,
        )
        host = result.host
        await host.start()
        await _until(lambda: _statuses(host)["lights"][0] == "available")
        await rig.clock.advance(1.0)
        await _until(lambda: bool(server.calls))
        await host.pre_llm("s1", "t1", "turn on the bedroom lamp")

        assert urls[0] == (HA_URL, HA_TOKEN)
        assert server.calls
        assert {call.timeout_s for call in server.calls} == {4.5}
        assert "HassTurnOn" in {call.name for call in server.calls}

    @pytest.mark.asyncio
    async def test_a_connection_change_is_broadcast_without_the_poll(self, enabled, build):
        server = FakeHomeAssistant()

        result, rig = build(
            {"MIST_CAPABILITIES": "lights", "MIST_HA_MCP_URL": HA_URL},
            secrets={"MIST_HA_TOKEN": HA_TOKEN},
            transport_factory=lambda url, token: server.transport(),
        )
        host = result.host
        await host.start()
        await _until(lambda: _statuses(host)["lights"][0] == "available")

        def lights_available() -> bool:
            for payload in rig.sink.of_type("capabilities_status"):
                for entry in payload["capabilities"]:
                    if entry["name"] == "lights" and entry["state"] == "available":
                        return True
            return False

        # The clock never advances, so the watcher's one-second poll has not run.
        await _until(lights_available)
        assert rig.clock.pending >= 1

    @pytest.mark.asyncio
    async def test_the_home_assistant_token_is_registered_for_redaction(self, enabled, build):
        result, rig = build(
            {"MIST_CAPABILITIES": "lights", "MIST_HA_MCP_URL": HA_URL},
            secrets={"MIST_HA_TOKEN": HA_TOKEN},
            transport_factory=lambda url, token: FakeHomeAssistant().transport(),
        )

        assert result.host.redact(f"Bearer {HA_TOKEN}") == "Bearer ***"

    @pytest.mark.asyncio
    async def test_a_missing_mcp_package_makes_lights_unavailable(
        self, enabled, build, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "backend.capabilities.mcp.sdk_transport", None)

        result, _ = build(
            {"MIST_CAPABILITIES": "lights", "MIST_HA_MCP_URL": HA_URL},
            secrets={"MIST_HA_TOKEN": HA_TOKEN},
        )
        host = result.host
        await host.start()
        await host.wait_started()

        assert _statuses(host) == {"lights": ("unavailable", "mcp package not installed")}
        assert host.statuses()[0].task_dead is False

    @pytest.mark.asyncio
    async def test_a_missing_token_makes_lights_unavailable(self, enabled, build):
        result, _ = build(
            {"MIST_CAPABILITIES": "lights", "MIST_HA_MCP_URL": HA_URL},
            transport_factory=lambda url, token: FakeHomeAssistant().transport(),
        )
        host = result.host
        await host.start()
        await host.wait_started()

        assert _statuses(host) == {"lights": ("unavailable", "MIST_HA_TOKEN unset")}

    def test_lights_need_the_home_assistant_url(self, enabled, build):
        result, _ = build(
            {"MIST_CAPABILITIES": "clock,lights"}, secrets={"MIST_HA_TOKEN": HA_TOKEN}
        )

        assert [s.name for s in result.host.statuses()] == ["clock"]


# -- Music, the capability list and the host settings ------------------------------------------


class TestAssembly:
    def test_the_named_capabilities_are_built_and_no_others(self, enabled, build):
        result, _ = build(
            {
                "MIST_CAPABILITIES": "clock,lights,music",
                "MIST_HA_MCP_URL": HA_URL,
                "MIST_SPOTIFY_CLIENT_ID": "spotify-client-id",
            },
            secrets={"MIST_HA_TOKEN": HA_TOKEN},
            transport_factory=lambda url, token: FakeHomeAssistant().transport(),
        )

        assert [s.name for s in result.host.statuses()] == ["clock", "lights", "music"]

    def test_music_needs_the_spotify_client_id(self, enabled, build):
        result, _ = build({"MIST_CAPABILITIES": "clock,music"})

        assert [s.name for s in result.host.statuses()] == ["clock"]

    def test_music_gets_the_device_the_clock_and_the_token_store(
        self, enabled, build, monkeypatch, tmp_path
    ):
        music = _recording(monkeypatch, "MusicCapability")
        token_path = tmp_path / "secrets" / "spotify_token.json"

        result, rig = build(
            {
                "MIST_CAPABILITIES": "music",
                "MIST_SPOTIFY_CLIENT_ID": "spotify-client-id",
                "MIST_SPOTIFY_DEVICE": "Kitchen Speaker",
                "MIST_SPOTIFY_TOKEN_PATH": str(token_path),
            }
        )

        (_, kwargs, _) = music[0]
        assert kwargs["device_name"] == "Kitchen Speaker"
        assert kwargs["clock"] is rig.clock
        assert kwargs["token_store"].path == token_path

    def test_the_router_reserves_the_decode_instructions(self, enabled, build, monkeypatch):
        routers = _recording(monkeypatch, "ToolRouter")

        build({"MIST_CAPABILITIES": "clock", "MIST_CAPABILITY_TOOL_TOKEN_BUDGET": "900"})

        (args, _, _) = routers[0]
        router_config = args[1]
        assert router_config.reserved_tokens == COUNTER.count(DECODE_INSTRUCTIONS)
        assert router_config.token_budget == 900

    def test_the_host_gets_the_fast_path_decode_timeout_wake_words_and_ports(
        self, enabled, build, monkeypatch, tmp_path
    ):
        hosts = _recording(monkeypatch, "CapabilityHost")
        env = {
            "MIST_CAPABILITIES": "clock",
            "MIST_CAPABILITY_FAST_PATH": "false",
            "MIST_CAPABILITY_DECODE_TIMEOUT_S": "3.5",
            "MIST_CAPABILITY_WAKE_WORDS": "computer,hey computer",
        }

        result, rig = build(env)

        (_, kwargs, built) = hosts[0]
        assert built is result.host
        settings = kwargs["settings"]
        assert settings == HostSettings.from_config(_config(tmp_path, env))
        assert (settings.fast_path, settings.decode_timeout_s, settings.wake_words) == (
            False,
            3.5,
            ("computer", "hey computer"),
        )
        assert kwargs["power"] is rig.power
        assert kwargs["tiers"] is rig.tiers
        assert isinstance(kwargs["register"], NoAddress)
        assert kwargs["event_sink"] is rig.sink
        assert kwargs["secrets"] is rig.registry
        assert kwargs["alarm_control"] is None

    def test_the_build_lists_the_hosts_capabilities_in_order(self, enabled, build, monkeypatch):
        hosts = _recording(monkeypatch, "CapabilityHost")

        result, _ = build({"MIST_CAPABILITIES": "music,clock", "MIST_SPOTIFY_CLIENT_ID": "id"})

        (args, _, _) = hosts[0]
        assert [c.name for c in result.capabilities] == ["music", "clock"]
        assert list(result.capabilities) == list(args[0])


# -- Close -------------------------------------------------------------------------------------


class ClosingDb(CapabilitiesDb):
    """A real database that counts its closes."""

    instances: list[ClosingDb] = []

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.closes = 0
        ClosingDb.instances.append(self)

    def close(self) -> None:
        self.closes += 1
        super().close()


class TestClose:
    @pytest.mark.asyncio
    async def test_close_releases_the_database_and_the_music_client(
        self, enabled, build, zone, monkeypatch
    ):
        ClosingDb.instances = []
        monkeypatch.setattr(factories, "CapabilitiesDb", ClosingDb)
        apis = _recording(monkeypatch, "SpotifyApi")
        result, _ = build(
            {
                "MIST_CAPABILITIES": "timers,clock,music",
                "MIST_TIMEZONE": ZONE_NAME,
                "MIST_SPOTIFY_CLIENT_ID": "spotify-client-id",
            }
        )
        host = result.host
        await host.start()
        await host.wait_started()
        assert await host.pre_llm("s1", "t1", "set a timer for 5 minutes") is not None
        (db,) = ClosingDb.instances
        http = apis[0][0][0]
        assert isinstance(http, httpx.AsyncClient)
        assert not http.is_closed

        await host.stop()
        await result.close()

        assert db.closes == 1
        assert http.is_closed

    @pytest.mark.asyncio
    async def test_a_disabled_build_opens_nothing_and_closes_nothing(
        self, enabled, build, monkeypatch
    ):
        def refuse(*args, **kwargs):
            raise AssertionError("a disabled build constructs no database or client")

        monkeypatch.setattr(factories, "CapabilitiesDb", refuse)
        monkeypatch.setattr(factories.httpx, "AsyncClient", refuse)

        result, _ = build({"MIST_CAPABILITIES_ENABLED": "false"})
        await result.close()

        assert result.host is None
        assert result.capabilities == ()
