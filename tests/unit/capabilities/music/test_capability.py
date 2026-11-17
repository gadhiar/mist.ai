"""Tests for `MusicCapability` over the Spotify Web API client.

Every request goes to an `httpx.MockTransport` that plays a small Spotify: a device list, an
optional active device, search results and the player. Tokens and ids are synthetic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from backend.capabilities.config import load_capabilities_config
from backend.capabilities.contracts import FastIntent, InvocationContext, ToolOutcome
from backend.capabilities.decoder import DECODE_INSTRUCTIONS
from backend.capabilities.music.capability import (
    CONTROL,
    NOW_PLAYING,
    PLAY,
    RECHECK_AFTER_S,
    VOLUME,
    VOLUME_STEP_PCT,
    MusicCapability,
)
from backend.capabilities.music.spotify_api import (
    API_BASE,
    SpotifyAccounts,
    SpotifyApi,
    SpotifyTokenManager,
)
from backend.capabilities.music.token_store import (
    AUTHORIZE_COMMAND,
    SpotifyToken,
    SpotifyTokenStore,
)
from backend.capabilities.replies import is_brief_sentence
from backend.capabilities.router import RouterConfig, ToolRouter, count_catalog_tokens
from backend.capabilities.schema import to_args_schema, to_catalog_line
from backend.capabilities.secrets import Secret, SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from tests.unit.capabilities.fakes import DEFAULT_START, FakeClock

COUNTER = ApproximateTokenCounter()
CLIENT_ID = "fake-client-id-0001"
ACCESS = "fake-access-token-0001"
REFRESH = "fake-refresh-token-0001"

KITCHEN = {
    "id": "dev-kitchen",
    "is_active": False,
    "name": "Kitchen Speaker",
    "type": "Speaker",
    "volume_percent": 40,
}
DESK = {"id": "dev-desk", "is_active": False, "name": "Desk", "type": "Computer"}

SO_WHAT = {
    "uri": "spotify:track:fake-so-what",
    "name": "So What",
    "artists": [{"name": "Miles Davis"}],
    "album": {"name": "Kind of Blue"},
}
KIND_OF_BLUE = {
    "uri": "spotify:album:fake-kind-of-blue",
    "name": "Kind of Blue",
    "artists": [{"name": "Miles Davis"}],
}
JAZZ_PLAYLIST = {"uri": "spotify:playlist:fake-jazz", "name": "Jazz Classics"}
MILES = {"uri": "spotify:artist:fake-miles", "name": "Miles Davis"}


def _error(status: int, reason: str) -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"status": status, "message": "synthetic", "reason": reason}}
    )


NO_ACTIVE_DEVICE = _error(404, "NO_ACTIVE_DEVICE")


@dataclass
class FakeSpotify:
    """A MockTransport handler playing the Web API endpoints the capability uses.

    A player command without `device_id` answers `NO_ACTIVE_DEVICE` while no device is active;
    a transfer makes its device active. `forced` answers the next request to a path with a
    scripted response or exception instead.
    """

    devices: list[dict[str, object]] = field(default_factory=lambda: [dict(KITCHEN), dict(DESK)])
    active: str | None = "dev-kitchen"
    is_playing: bool = True
    item: dict[str, object] | None = field(default_factory=lambda: dict(SO_WHAT))
    search: dict[str, list[object]] = field(default_factory=dict)
    forced: dict[str, list[httpx.Response | BaseException]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def force(self, path: str, *steps: httpx.Response | BaseException) -> None:
        self.forced.setdefault(path, []).extend(steps)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/v1")
        steps = self.forced.get(path)
        if steps:
            step = steps.pop(0)
            if isinstance(step, BaseException):
                raise step
            return step
        params = dict(request.url.params)
        if path == "/me/player/devices":
            return httpx.Response(200, json={"devices": self._device_list()})
        if path == "/search":
            kinds = params["type"].split(",")
            body = {f"{kind}s": {"items": self.search.get(kind, [])} for kind in kinds}
            return httpx.Response(200, json=body)
        if path == "/me/player" and request.method == "GET":
            if self.active is None:
                return httpx.Response(204)
            device = next(d for d in self._device_list() if d["id"] == self.active)
            return httpx.Response(
                200,
                json={
                    "is_playing": self.is_playing,
                    "progress_ms": 1000,
                    "device": device,
                    "item": self.item,
                },
            )
        if path == "/me/player" and request.method == "PUT":
            body = json.loads(request.content)
            self.active = body["device_ids"][0]
            return httpx.Response(204)
        if self.active is None and "device_id" not in params:
            return NO_ACTIVE_DEVICE
        if path == "/me/player/volume":
            for device in self.devices:
                if device["id"] == (params.get("device_id") or self.active):
                    device["volume_percent"] = int(params["volume_percent"])
        return httpx.Response(204)

    def _device_list(self) -> list[dict[str, object]]:
        return [{**device, "is_active": device["id"] == self.active} for device in self.devices]

    def player_calls(self) -> list[tuple[str, str, dict[str, str], object]]:
        """Return (method, path, params, json body) for every request but searches."""
        calls = []
        for request in self.requests:
            path = request.url.path.removeprefix("/v1")
            if path == "/search":
                continue
            body = json.loads(request.content) if request.content else None
            calls.append((request.method, path, dict(request.url.params), body))
        return calls


@dataclass
class Rig:
    capability: MusicCapability
    spotify: FakeSpotify
    clock: FakeClock
    store: SpotifyTokenStore


def _rig(
    tmp_path: Path,
    spotify: FakeSpotify | None = None,
    *,
    device_name: str | None = "kitchen speaker",
    token: bool = True,
) -> Rig:
    spotify = spotify or FakeSpotify()
    clock = FakeClock()
    store = SpotifyTokenStore(tmp_path / "secrets" / "spotify_token.json")
    if token:
        store.save(
            SpotifyToken(
                access_token=Secret(ACCESS),
                refresh_token=Secret(REFRESH),
                expires_at=clock.now() + timedelta(hours=1),
                scope="user-read-playback-state",
            )
        )
    registry = SecretRegistry()
    http = httpx.AsyncClient(transport=httpx.MockTransport(spotify))
    accounts = SpotifyAccounts(http, client_id=CLIENT_ID, registry=registry, clock=clock)
    tokens = SpotifyTokenManager(store=store, accounts=accounts, registry=registry, clock=clock)
    capability = MusicCapability(
        SpotifyApi(http, tokens), device_name=device_name, clock=clock, token_store=store
    )
    return Rig(capability, spotify, clock, store)


def _ctx() -> InvocationContext:
    return InvocationContext(
        session_id="s1",
        turn_id="t1",
        utterance="",
        tool_call_id="c1",
        source="fastpath",
        now=DEFAULT_START,
    )


def _spec(capability: MusicCapability, name: str):
    return next(spec for spec in capability.tools() if spec.name == name)


async def _invoke(capability: MusicCapability, tool: str, **args: object) -> ToolOutcome:
    validated = _spec(capability, tool).args_model.model_validate(args)
    return await capability.invoke(tool, validated, _ctx())


def _lines(capability: MusicCapability) -> list[str]:
    return [to_catalog_line(spec, to_args_schema(spec)) for spec in capability.tools()]


def _search_params(spotify: FakeSpotify) -> dict[str, str]:
    (request,) = (r for r in spotify.requests if r.url.path.endswith("/search"))
    return dict(request.url.params)


# -- tools -------------------------------------------------------------------------------------


class TestTools:
    def test_four_tools_with_the_contract_risk_and_policy(self, tmp_path: Path) -> None:
        specs = {spec.name: spec for spec in _rig(tmp_path).capability.tools()}
        assert list(specs) == [PLAY, CONTROL, VOLUME, NOW_PLAYING]
        assert (specs[PLAY].risk, specs[PLAY].default_policy) == ("physical", "confirm_inferred")
        assert (specs[CONTROL].risk, specs[CONTROL].default_policy) == ("reversible", "allow")
        assert (specs[VOLUME].risk, specs[VOLUME].default_policy) == ("reversible", "allow")
        assert (specs[NOW_PLAYING].risk, specs[NOW_PLAYING].default_policy) == ("read", "allow")

    def test_every_tool_sets_a_short_ack_and_the_music_view(self, tmp_path: Path) -> None:
        for spec in _rig(tmp_path).capability.tools():
            assert spec.ack is not None and 1 <= len(spec.ack.split()) <= 5, spec.name
            assert spec.view == "music", spec.name

    def test_arguments_schemas_follow_the_contract(self, tmp_path: Path) -> None:
        capability = _rig(tmp_path).capability
        play = to_args_schema(_spec(capability, PLAY))
        assert play["properties"] == {
            "query": {"type": "string"},
            "kind": {"type": "string", "enum": ["album", "artist", "playlist", "track"]},
            "device": {"type": "string"},
        }
        assert "required" not in play or play["required"] == []
        control = to_args_schema(_spec(capability, CONTROL))
        assert control["properties"] == {
            "action": {"type": "string", "enum": ["next", "pause", "previous", "resume"]}
        }
        assert control["required"] == ["action"]
        volume = to_args_schema(_spec(capability, VOLUME))
        assert volume["properties"] == {
            "level_pct": {"type": "integer", "minimum": 0, "maximum": 100},
            "change": {"type": "string", "enum": ["down", "up"]},
        }
        assert to_args_schema(_spec(capability, NOW_PLAYING))["properties"] == {}

    @pytest.mark.parametrize(
        "args",
        [{}, {"level_pct": 40, "change": "up"}, {"level_pct": 101}, {"level_pct": -1}],
    )
    def test_volume_takes_exactly_one_of_level_and_change(
        self, tmp_path: Path, args: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError):
            _spec(_rig(tmp_path).capability, VOLUME).args_model.model_validate(args)

    def test_unknown_arguments_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            _spec(_rig(tmp_path).capability, CONTROL).args_model.model_validate(
                {"action": "pause", "loud": True}
            )

    def test_proposal_summaries_name_the_action(self, tmp_path: Path) -> None:
        capability = _rig(tmp_path).capability

        def summary(tool: str, **args: object) -> str:
            spec = _spec(capability, tool)
            return spec.summarize(spec.args_model.model_validate(args))

        assert summary(PLAY, query="Kind of Blue", kind="album") == "play the album Kind of Blue"
        assert summary(PLAY, query="jazz") == "play jazz"
        assert summary(PLAY, query="jazz", device="Desk") == "play jazz on Desk"
        assert summary(PLAY) == "resume the music"
        assert summary(CONTROL, action="pause") == "pause the music"
        assert summary(CONTROL, action="next") == "skip to the next track"
        assert summary(VOLUME, level_pct=40) == "set the volume to 40 percent"
        assert summary(VOLUME, change="up") == "turn the volume up"
        assert summary(NOW_PLAYING) == "check what is playing"

    def test_no_is_explicit_hook_exists(self) -> None:
        assert not hasattr(MusicCapability, "is_explicit")

    def test_match_intent_uses_the_music_grammar(self, tmp_path: Path) -> None:
        capability = _rig(tmp_path).capability
        assert capability.match_intent("pause") == FastIntent(
            tool=CONTROL, args={"action": "pause"}
        )
        assert capability.match_intent("play some jazz and tell me about coltrane") is None

    def test_catalog_fits_the_per_capability_budget_at_default_config(self, tmp_path: Path) -> None:
        capability = _rig(tmp_path).capability
        config = load_capabilities_config({})
        router_config = RouterConfig(
            token_budget=config.tool_token_budget,
            reserved_tokens=COUNTER.count(DECODE_INSTRUCTIONS),
        )
        # Registration refuses a capability whose worst case exceeds the budget.
        router = ToolRouter([capability], router_config, COUNTER, None)
        worst = count_catalog_tokens(_lines(capability), COUNTER)
        worst += capability.refine_allowance_tokens
        assert worst <= router.per_capability_budget()

    def test_arguments_schemas_are_byte_stable(self, tmp_path: Path) -> None:
        first = [json.dumps(to_args_schema(s)) for s in _rig(tmp_path).capability.tools()]
        second = [json.dumps(to_args_schema(s)) for s in _rig(tmp_path).capability.tools()]
        assert first == second


# -- music.control -----------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "method", "path", "summary"),
    [
        ("pause", "PUT", "/me/player/pause", "Paused."),
        ("resume", "PUT", "/me/player/play", "Resumed."),
        ("next", "POST", "/me/player/next", "Skipping to the next track."),
        ("previous", "POST", "/me/player/previous", "Back to the previous track."),
    ],
)
async def test_control_maps_each_action_onto_the_client(
    tmp_path: Path, action: str, method: str, path: str, summary: str
) -> None:
    rig = _rig(tmp_path)
    outcome = await _invoke(rig.capability, CONTROL, action=action)
    assert outcome == ToolOutcome(status="ok", summary=summary, data={"action": action})
    assert rig.spotify.player_calls() == [(method, path, {}, None)]
    assert is_brief_sentence(outcome.summary)


# -- music.play --------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_play_with_a_query_searches_every_kind_and_plays_the_first_track(
    tmp_path: Path,
) -> None:
    spotify = FakeSpotify(search={"track": [SO_WHAT], "album": [KIND_OF_BLUE], "artist": [MILES]})
    rig = _rig(tmp_path, spotify)
    outcome = await _invoke(rig.capability, PLAY, query="so what")
    assert outcome.status == "ok"
    assert outcome.summary == "Playing So What by Miles Davis."
    assert outcome.data == {
        "kind": "track",
        "uri": SO_WHAT["uri"],
        "name": "So What",
        "artists": ["Miles Davis"],
    }
    params = _search_params(spotify)
    assert params["q"] == "so what"
    assert params["type"] == "track,album,playlist,artist"
    assert spotify.player_calls() == [("PUT", "/me/player/play", {}, {"uris": [SO_WHAT["uri"]]})]


@pytest.mark.asyncio
async def test_play_with_a_kind_searches_that_kind_and_plays_a_context(tmp_path: Path) -> None:
    spotify = FakeSpotify(search={"album": [KIND_OF_BLUE]})
    rig = _rig(tmp_path, spotify)
    outcome = await _invoke(rig.capability, PLAY, query="kind of blue", kind="album")
    assert outcome.summary == "Playing Kind of Blue by Miles Davis."
    assert is_brief_sentence(outcome.summary)
    assert _search_params(spotify)["type"] == "album"
    assert spotify.player_calls() == [
        ("PUT", "/me/player/play", {}, {"context_uri": KIND_OF_BLUE["uri"]})
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("search", "uri", "summary"),
    [
        (
            {"album": [KIND_OF_BLUE], "playlist": [JAZZ_PLAYLIST], "artist": [MILES]},
            KIND_OF_BLUE["uri"],
            "Playing Kind of Blue by Miles Davis.",
        ),
        (
            {"playlist": [None, JAZZ_PLAYLIST], "artist": [MILES]},
            JAZZ_PLAYLIST["uri"],
            "Playing the playlist Jazz Classics.",
        ),
        ({"artist": [MILES]}, MILES["uri"], "Playing Miles Davis."),
    ],
)
async def test_play_without_a_kind_falls_back_in_the_order_track_album_playlist_artist(
    tmp_path: Path, search: dict[str, list[object]], uri: str, summary: str
) -> None:
    spotify = FakeSpotify(search=search)
    rig = _rig(tmp_path, spotify)
    outcome = await _invoke(rig.capability, PLAY, query="jazz")
    assert outcome.summary == summary
    assert is_brief_sentence(outcome.summary)
    assert spotify.player_calls() == [("PUT", "/me/player/play", {}, {"context_uri": uri})]


@pytest.mark.asyncio
async def test_play_with_no_match_is_invalid_args_and_plays_nothing(tmp_path: Path) -> None:
    spotify = FakeSpotify(search={})
    rig = _rig(tmp_path, spotify)
    outcome = await _invoke(rig.capability, PLAY, query="zzzz", kind="playlist")
    assert outcome.status == "invalid_args"
    assert outcome.reason == "no_results"
    assert outcome.summary == "Nothing on Spotify matches zzzz."
    assert spotify.player_calls() == []


@pytest.mark.asyncio
async def test_play_without_a_query_resumes(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    outcome = await _invoke(rig.capability, PLAY)
    assert outcome == ToolOutcome(status="ok", summary="Resumed.", data={"action": "resume"})
    assert rig.spotify.player_calls() == [("PUT", "/me/player/play", {}, None)]
    assert not [r for r in rig.spotify.requests if r.url.path.endswith("/search")]


@pytest.mark.asyncio
async def test_play_on_a_named_device_passes_its_id(tmp_path: Path) -> None:
    spotify = FakeSpotify(search={"track": [SO_WHAT]})
    rig = _rig(tmp_path, spotify)
    outcome = await _invoke(rig.capability, PLAY, query="so what", device="DESK")
    assert outcome.status == "ok"
    assert spotify.player_calls() == [
        ("GET", "/me/player/devices", {}, None),
        ("PUT", "/me/player/play", {"device_id": "dev-desk"}, {"uris": [SO_WHAT["uri"]]}),
    ]


@pytest.mark.asyncio
async def test_play_on_an_unknown_device_names_the_known_ones(tmp_path: Path) -> None:
    rig = _rig(tmp_path, FakeSpotify(search={"track": [SO_WHAT]}))
    outcome = await _invoke(rig.capability, PLAY, query="so what", device="garage")
    assert outcome.status == "invalid_args"
    assert outcome.reason == "unknown device"
    assert (
        outcome.summary
        == "There is no Spotify device called garage; known ones are Kitchen Speaker, Desk."
    )
    assert outcome.data == {"known_devices": ["Kitchen Speaker", "Desk"]}


@pytest.mark.asyncio
async def test_no_active_device_transfers_to_the_configured_device_once_and_retries(
    tmp_path: Path,
) -> None:
    spotify = FakeSpotify(active=None, search={"track": [SO_WHAT]})
    rig = _rig(tmp_path, spotify, device_name="Kitchen speaker")
    outcome = await _invoke(rig.capability, PLAY, query="so what")
    assert outcome.status == "ok"
    assert outcome.summary == "Playing So What by Miles Davis."
    body = {"uris": [SO_WHAT["uri"]]}
    assert spotify.player_calls() == [
        ("PUT", "/me/player/play", {}, body),
        ("GET", "/me/player/devices", {}, None),
        ("PUT", "/me/player", {}, {"device_ids": ["dev-kitchen"], "play": False}),
        ("PUT", "/me/player/play", {"device_id": "dev-kitchen"}, body),
    ]


@pytest.mark.asyncio
async def test_no_active_device_after_the_transfer_is_an_error_naming_the_reason(
    tmp_path: Path,
) -> None:
    spotify = FakeSpotify(active=None)
    spotify.force("/me/player/play", NO_ACTIVE_DEVICE, NO_ACTIVE_DEVICE, NO_ACTIVE_DEVICE)
    rig = _rig(tmp_path, spotify)
    outcome = await _invoke(rig.capability, PLAY)
    assert outcome.status == "error"
    assert outcome.reason == "no_active_device"
    transfers = [c for c in spotify.player_calls() if c[:2] == ("PUT", "/me/player")]
    assert len(transfers) == 1
    assert [c[1] for c in spotify.player_calls()].count("/me/player/play") == 2


@pytest.mark.asyncio
async def test_no_active_device_without_a_configured_device_is_an_error(tmp_path: Path) -> None:
    spotify = FakeSpotify(active=None)
    rig = _rig(tmp_path, spotify, device_name=None)
    outcome = await _invoke(rig.capability, PLAY)
    assert (outcome.status, outcome.reason) == ("error", "no_active_device")
    assert spotify.player_calls() == [("PUT", "/me/player/play", {}, None)]


@pytest.mark.asyncio
async def test_no_active_device_when_the_configured_device_is_missing_is_an_error(
    tmp_path: Path,
) -> None:
    spotify = FakeSpotify(active=None)
    rig = _rig(tmp_path, spotify, device_name="Living Room")
    outcome = await _invoke(rig.capability, CONTROL, action="pause")
    assert (outcome.status, outcome.reason) == ("error", "no_active_device")
    assert "Living Room" in outcome.summary
    assert [c[:2] for c in spotify.player_calls()] == [
        ("PUT", "/me/player/pause"),
        ("GET", "/me/player/devices"),
    ]


@pytest.mark.asyncio
async def test_a_long_title_still_gives_one_brief_sentence(tmp_path: Path) -> None:
    long_track = {
        **SO_WHAT,
        "name": "A Very Long Song Title That Goes On And On Past Any Spoken Limit",
        "artists": [{"name": "The Extremely Long Named Orchestra"}],
    }
    rig = _rig(tmp_path, FakeSpotify(search={"track": [long_track]}))
    outcome = await _invoke(rig.capability, PLAY, query="long")
    assert outcome.status == "ok"
    assert is_brief_sentence(outcome.summary)
    assert outcome.summary.startswith("Playing A Very Long Song")
    assert outcome.data is not None and outcome.data["name"] == long_track["name"]


@pytest.mark.asyncio
async def test_a_title_with_inner_full_stops_still_gives_one_sentence(tmp_path: Path) -> None:
    track = {**SO_WHAT, "name": "Mr. Brightside", "artists": [{"name": "The Killers"}]}
    rig = _rig(tmp_path, FakeSpotify(search={"track": [track]}))
    outcome = await _invoke(rig.capability, PLAY, query="mr brightside")
    assert outcome.summary == "Playing Mr Brightside by The Killers."
    assert is_brief_sentence(outcome.summary)


# -- music.volume ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_volume_to_a_level_sets_it(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    outcome = await _invoke(rig.capability, VOLUME, level_pct=40)
    assert outcome == ToolOutcome(status="ok", summary="Volume 40 percent.", data={"level_pct": 40})
    assert rig.spotify.player_calls() == [
        ("PUT", "/me/player/volume", {"volume_percent": "40"}, None)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start", "change", "expected"),
    [
        (40, "up", 40 + VOLUME_STEP_PCT),
        (40, "down", 40 - VOLUME_STEP_PCT),
        (95, "up", 100),
        (5, "down", 0),
    ],
)
async def test_volume_up_or_down_steps_from_the_current_level(
    tmp_path: Path, start: int, change: str, expected: int
) -> None:
    spotify = FakeSpotify()
    spotify.devices[0]["volume_percent"] = start
    rig = _rig(tmp_path, spotify)
    outcome = await _invoke(rig.capability, VOLUME, change=change)
    assert outcome.summary == f"Volume {expected} percent."
    assert spotify.player_calls()[-1] == (
        "PUT",
        "/me/player/volume",
        {"volume_percent": str(expected)},
        None,
    )


@pytest.mark.asyncio
async def test_volume_change_with_no_active_device_transfers_first(tmp_path: Path) -> None:
    spotify = FakeSpotify(active=None)
    rig = _rig(tmp_path, spotify)
    outcome = await _invoke(rig.capability, VOLUME, change="up")
    assert outcome.summary == "Volume 50 percent."
    assert spotify.player_calls()[-1] == (
        "PUT",
        "/me/player/volume",
        {"volume_percent": "50", "device_id": "dev-kitchen"},
        None,
    )


# -- music.now_playing -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_now_playing_reads_the_current_track(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    outcome = await _invoke(rig.capability, NOW_PLAYING)
    assert outcome.status == "ok"
    assert outcome.summary == "Playing So What by Miles Davis."
    assert is_brief_sentence(outcome.summary)
    assert outcome.data == {
        "is_playing": True,
        "name": "So What",
        "artists": ["Miles Davis"],
        "album": "Kind of Blue",
        "uri": SO_WHAT["uri"],
        "device": "Kitchen Speaker",
        "progress_ms": 1000,
    }


@pytest.mark.asyncio
async def test_now_playing_while_paused_says_so(tmp_path: Path) -> None:
    rig = _rig(tmp_path, FakeSpotify(is_playing=False))
    outcome = await _invoke(rig.capability, NOW_PLAYING)
    assert outcome.summary == "Paused on So What by Miles Davis."


@pytest.mark.asyncio
@pytest.mark.parametrize("spotify", [FakeSpotify(active=None), FakeSpotify(item=None)])
async def test_now_playing_with_nothing_playing(tmp_path: Path, spotify: FakeSpotify) -> None:
    outcome = await _invoke(_rig(tmp_path, spotify).capability, NOW_PLAYING)
    assert outcome == ToolOutcome(
        status="ok", summary="Nothing is playing.", data={"is_playing": False}
    )


# -- status ------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_is_available_with_a_token_file(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    await rig.capability.start()
    status = rig.capability.status()
    assert (status.name, status.state, status.reason) == ("music", "available", None)
    assert status.checked_at == rig.clock.now()


@pytest.mark.asyncio
async def test_status_is_unavailable_when_no_token_file_exists_at_start(tmp_path: Path) -> None:
    rig = _rig(tmp_path, token=False)
    await rig.capability.start()
    status = rig.capability.status()
    assert status.state == "unavailable"
    assert status.reason is not None and status.reason.startswith("not_authorized")
    assert AUTHORIZE_COMMAND in status.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("step", "reason"),
    [
        (_error(403, "PREMIUM_REQUIRED"), "premium_required"),
        (httpx.ConnectError("synthetic"), "unreachable"),
    ],
)
async def test_premium_required_and_unreachable_are_unavailable_with_the_reason(
    tmp_path: Path, step: httpx.Response | BaseException, reason: str
) -> None:
    spotify = FakeSpotify()
    spotify.force("/me/player/pause", step)
    rig = _rig(tmp_path, spotify)
    outcome = await _invoke(rig.capability, CONTROL, action="pause")
    assert (outcome.status, outcome.reason) == ("unavailable", reason)
    status = rig.capability.status()
    assert (status.state, status.reason) == ("unavailable", reason)


@pytest.mark.asyncio
async def test_not_authorized_is_unavailable_naming_the_authorize_cli(tmp_path: Path) -> None:
    rig = _rig(tmp_path, token=False)
    outcome = await _invoke(rig.capability, NOW_PLAYING)
    assert outcome.status == "unavailable"
    assert outcome.reason is not None and outcome.reason.startswith("not_authorized")
    assert AUTHORIZE_COMMAND in outcome.reason
    status = rig.capability.status()
    assert (status.state, status.reason) == ("unavailable", outcome.reason)
    assert rig.spotify.requests == []


@pytest.mark.asyncio
async def test_an_unavailable_status_rechecks_after_the_window_and_clears_on_success(
    tmp_path: Path,
) -> None:
    # The host refuses calls while a capability is unavailable, so the state must lapse for
    # the next call to find out whether Spotify is back.
    spotify = FakeSpotify()
    spotify.force("/me/player/pause", httpx.ConnectError("synthetic"))
    rig = _rig(tmp_path, spotify)
    await _invoke(rig.capability, CONTROL, action="pause")
    await rig.clock.advance(RECHECK_AFTER_S - 1)
    assert rig.capability.status().state == "unavailable"
    await rig.clock.advance(1)
    status = rig.capability.status()
    assert (status.state, status.reason) == ("degraded", "unreachable")
    outcome = await _invoke(rig.capability, CONTROL, action="pause")
    assert outcome.status == "ok"
    assert (rig.capability.status().state, rig.capability.status().reason) == ("available", None)


@pytest.mark.asyncio
async def test_rate_limited_and_other_errors_are_errors_and_leave_the_status(
    tmp_path: Path,
) -> None:
    spotify = FakeSpotify()
    spotify.force(
        "/me/player/next",
        httpx.Response(429, headers={"Retry-After": "30"}),
        _error(500, "UNKNOWN"),
    )
    rig = _rig(tmp_path, spotify)
    limited = await _invoke(rig.capability, CONTROL, action="next")
    assert (limited.status, limited.reason) == ("error", "rate_limited")
    assert limited.data == {"retry_after_s": 30.0}
    failed = await _invoke(rig.capability, CONTROL, action="next")
    assert (failed.status, failed.reason) == ("error", "server_error")
    assert rig.capability.status().state == "available"


@pytest.mark.asyncio
async def test_every_ok_summary_is_one_brief_sentence(tmp_path: Path) -> None:
    rig = _rig(tmp_path, FakeSpotify(search={"track": [SO_WHAT], "album": [KIND_OF_BLUE]}))
    calls: list[tuple[str, dict[str, object]]] = [
        (PLAY, {"query": "so what"}),
        (PLAY, {"query": "kind of blue", "kind": "album"}),
        (PLAY, {}),
        (CONTROL, {"action": "pause"}),
        (CONTROL, {"action": "resume"}),
        (CONTROL, {"action": "next"}),
        (CONTROL, {"action": "previous"}),
        (VOLUME, {"level_pct": 40}),
        (VOLUME, {"change": "down"}),
        (NOW_PLAYING, {}),
    ]
    for tool, args in calls:
        outcome = await _invoke(rig.capability, tool, **args)
        assert outcome.status == "ok", (tool, args, outcome)
        assert is_brief_sentence(outcome.summary), (tool, args, outcome.summary)


def test_the_fake_serves_the_clients_base_url() -> None:
    # The fake strips this prefix; a different base would route nothing.
    assert API_BASE.endswith("/v1")
