"""Fast-path arbitration in the real capability host, assembled by `build_capability_host`.

The fast path runs a command only when exactly one capability's grammar matches it. A light
named "Music" makes "turn off the music" match both the lights row for a known target and the
music pause row, so neither may run: the utterance goes on to routing and the decode, and no
device is touched. A command only lights match still runs through the fast path.

Home Assistant is the in-memory fake, connected, serving a catalog with the Bedroom area and
the Music light; it goes in through the factory's `transport_factory`. The factory builds
music's `httpx.AsyncClient` itself, so a recording Spotify `MockTransport` goes in by patching
`factories.httpx.AsyncClient` for the test. `tests/unit/conftest.py` sets
`MIST_EVAL_ISOLATION`, which disables the host, so the `connected` fixture switches it off.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from backend import factories
from backend.capabilities.config import load_capabilities_config
from backend.capabilities.contracts import Capability
from backend.capabilities.host import CapabilityHost
from backend.capabilities.intents import normalize_utterance
from backend.capabilities.music.token_store import SpotifyToken, SpotifyTokenStore
from backend.capabilities.replies import NoAddress
from backend.capabilities.secrets import EnvSecretSource, Secret, SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.factories import CapabilityBuild, build_capability_host
from tests.unit.capabilities.fakes import (
    DEFAULT_START,
    FakeClock,
    FakePower,
    FakeTiers,
    RecordingEventSink,
)
from tests.unit.capabilities.mcp.fake_home_assistant import (
    LIVE_CONTEXT,
    TURN_OFF,
    FakeHomeAssistant,
)

COUNTER = ApproximateTokenCounter()
HA_URL = "http://homeassistant.invalid:8123/api/mcp"
HA_TOKEN = "ha-token-for-arbitration-tests"  # noqa: S105 - a fake token
ALLOWED_ORIGIN = "http://tauri.localhost"

CATALOG = (
    "Live Context: An overview of the areas and the devices in this smart home:\n"
    "- names: Bedroom Lamp\n"
    "  domain: light\n"
    "  state: 'on'\n"
    "  areas: Bedroom\n"
    "- names: Music\n"
    "  domain: light\n"
    "  state: 'on'\n"
    "  areas: Hallway\n"
)
"""The fake's `GetLiveContext` text: the Bedroom area, and a light named "Music"."""

AMBIGUOUS = "turn off the music"
LIGHTS_ONLY = "turn off the bedroom lights"


@dataclass
class Connected:
    """One host with Home Assistant connected, and what it was given."""

    build: CapabilityBuild
    host: CapabilityHost
    clock: FakeClock
    home_assistant: FakeHomeAssistant
    spotify_requests: list[httpx.Request]

    def capability(self, name: str) -> Capability:
        (found,) = (c for c in self.build.capabilities if c.name == name)
        return found


@pytest_asyncio.fixture
async def connected(monkeypatch, tmp_path):
    """Build lights and music over a connected fake Home Assistant and a recording Spotify."""
    monkeypatch.setenv("MIST_EVAL_ISOLATION", "0")
    monkeypatch.delenv("MIST_HYDRATION_ISOLATION", raising=False)
    monkeypatch.setenv("MIST_ALLOWED_ORIGINS", ALLOWED_ORIGIN)

    spotify_requests: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        spotify_requests.append(request)
        return httpx.Response(204)

    real_client = httpx.AsyncClient

    def recording_client(**kwargs: object) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(recording), **kwargs)

    monkeypatch.setattr(factories.httpx, "AsyncClient", recording_client)

    token_path = tmp_path / "secrets" / "spotify_token.json"
    SpotifyTokenStore(token_path).save(
        SpotifyToken(
            access_token=Secret("spotify-access-for-arbitration-tests"),
            refresh_token=Secret("spotify-refresh-for-arbitration-tests"),
            expires_at=DEFAULT_START + timedelta(hours=1),
            scope="user-modify-playback-state",
        )
    )
    home_assistant = FakeHomeAssistant()
    home_assistant.live_context = CATALOG
    clock = FakeClock()
    config = load_capabilities_config(
        {
            "MIST_CAPABILITIES_ENABLED": "true",
            "MIST_CAPABILITIES": "lights,music",
            "MIST_CAPABILITIES_DB_PATH": str(tmp_path / "capabilities.db"),
            "MIST_HA_MCP_URL": HA_URL,
            "MIST_SPOTIFY_CLIENT_ID": "spotify-client-for-arbitration-tests",
            "MIST_SPOTIFY_TOKEN_PATH": str(token_path),
        }
    )
    build = build_capability_host(
        config,
        event_sink=RecordingEventSink(),
        secret_source=EnvSecretSource({"MIST_HA_TOKEN": HA_TOKEN}, _read_file),
        secret_registry=SecretRegistry(),
        clock=clock,
        embedder=None,
        token_counter=COUNTER,
        transport_factory=lambda url, token: home_assistant.transport(),
        power=FakePower(),
        tiers=FakeTiers(),
        register=NoAddress(),
        desktop_present=lambda: False,
    )
    assert build.host is not None, build.disabled_reason
    rig = Connected(build, build.host, clock, home_assistant, spotify_requests)
    yield rig
    await rig.host.stop()
    await build.close()


def _read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _statuses(host: CapabilityHost) -> dict[str, str]:
    return {s.name: s.state for s in host.statuses()}


async def _until(predicate: Callable[[], bool], *, within_s: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    give_up_at = loop.time() + within_s
    while not predicate():
        assert loop.time() < give_up_at, "condition not reached in time"
        await asyncio.sleep(0.001)


async def _started_with_catalog(rig: Connected) -> None:
    """Start the host, wait for lights to connect, then for the catalog to name the Music light.

    `McpConnection.start()` returns before the first connect completes, so lights become
    `available` later. The catalog refresh polls on the fake clock, so one poll interval is
    advanced to let it read `GetLiveContext`.
    """
    await rig.host.start()
    await rig.host.wait_started()
    await _until(lambda: _statuses(rig.host)["lights"] == "available")
    await rig.clock.advance(1.0)
    lights = rig.capability("lights")
    await _until(lambda: "Music" in lights.dynamic_keywords())


class TestFastPathArbitration:
    @pytest.mark.asyncio
    async def test_a_command_both_lights_and_music_match_runs_neither(self, connected):
        await _started_with_catalog(connected)
        text = normalize_utterance(AMBIGUOUS, ())
        calls_before = len(connected.home_assistant.calls)

        resolution = await connected.host.pre_llm("s1", "t1", AMBIGUOUS)

        assert connected.capability("lights").match_intent(text) is not None
        assert connected.capability("music").match_intent(text) is not None
        assert resolution is None
        assert connected.home_assistant.calls[calls_before:] == []
        assert connected.spotify_requests == []

    @pytest.mark.asyncio
    async def test_a_command_only_lights_match_runs_through_the_fast_path(self, connected):
        await _started_with_catalog(connected)
        calls_before = len(connected.home_assistant.calls)

        resolution = await connected.host.pre_llm("s1", "t1", LIGHTS_ONLY)

        assert resolution is not None and resolution.handled
        assert [(r.tool, r.source, r.outcome.status) for r in resolution.records] == [
            ("lights.power", "fastpath", "ok")
        ]
        new_calls = [
            (call.name, call.arguments)
            for call in connected.home_assistant.calls[calls_before:]
            if call.name != LIVE_CONTEXT
        ]
        assert new_calls == [(TURN_OFF, {"area": "Bedroom", "domain": ["light"]})]
        assert connected.spotify_requests == []
