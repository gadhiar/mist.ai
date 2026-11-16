"""Tests for the Spotify Web API client, the accounts token endpoint and the token manager.

Every request goes to an `httpx.MockTransport`; nothing leaves the process. Tokens and client
ids are synthetic.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from backend.capabilities.music.spotify_api import (
    API_BASE,
    TOKEN_URL,
    PlayableItem,
    SpotifyAccounts,
    SpotifyApi,
    SpotifyDevice,
    SpotifyRateLimitedError,
    SpotifyTokenManager,
)
from backend.capabilities.music.token_store import (
    AUTHORIZE_COMMAND,
    SpotifyToken,
    SpotifyTokenStore,
)
from backend.capabilities.secrets import (
    Secret,
    SecretRegistry,
    install_log_redaction,
    redact,
)
from backend.errors import SpotifyApiError, SpotifyAuthError, SpotifyError
from tests.unit.capabilities.fakes import FakeClock

CLIENT_ID = "fake-client-id-0001"
REDIRECT = "http://127.0.0.1:8888/callback"
OLD_ACCESS = "fake-access-token-old-0001"
OLD_REFRESH = "fake-refresh-token-old-0001"
NEW_ACCESS = "fake-access-token-new-0002"
NEW_REFRESH = "fake-refresh-token-new-0002"

Step = httpx.Response | BaseException | Callable[[httpx.Request], httpx.Response]


class Script:
    """A MockTransport handler that answers each request with the next scripted step."""

    def __init__(self, *steps: Step) -> None:
        self.steps: list[Step] = list(steps)
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        # Yield, so concurrent callers interleave the way they would over a network.
        await asyncio.sleep(0)
        if not self.steps:
            raise AssertionError(f"unscripted request {request.method} {request.url}")
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(request)
        return step

    def token_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if str(r.url) == TOKEN_URL]

    def api_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if str(r.url).startswith(API_BASE)]


def token_response(
    access: str = NEW_ACCESS, refresh: str | None = NEW_REFRESH, expires_in: int = 3600
) -> httpx.Response:
    body: dict[str, object] = {
        "access_token": access,
        "token_type": "Bearer",
        "scope": "user-read-playback-state",
        "expires_in": expires_in,
    }
    if refresh is not None:
        body["refresh_token"] = refresh
    return httpx.Response(200, json=body)


def api_error(status: int, reason: str | None, headers: dict[str, str] | None = None):
    error: dict[str, object] = {"status": status, "message": "synthetic"}
    if reason is not None:
        error["reason"] = reason
    return httpx.Response(status, json={"error": error}, headers=headers)


@dataclass
class Rig:
    api: SpotifyApi
    tokens: SpotifyTokenManager
    accounts: SpotifyAccounts
    store: SpotifyTokenStore
    registry: SecretRegistry
    clock: FakeClock
    script: Script
    http: httpx.AsyncClient


def build(tmp_path: Path, script: Script, *, expires_in_s: float = 3600) -> Rig:
    clock = FakeClock()
    store = SpotifyTokenStore(tmp_path / "secrets" / "spotify_token.json")
    store.save(
        SpotifyToken(
            access_token=Secret(OLD_ACCESS),
            refresh_token=Secret(OLD_REFRESH),
            expires_at=clock.now() + timedelta(seconds=expires_in_s),
            scope="user-read-playback-state",
        )
    )
    registry = SecretRegistry()
    http = httpx.AsyncClient(transport=httpx.MockTransport(script))
    accounts = SpotifyAccounts(http, client_id=CLIENT_ID, registry=registry, clock=clock)
    tokens = SpotifyTokenManager(store=store, accounts=accounts, registry=registry, clock=clock)
    api = SpotifyApi(http, tokens)
    return Rig(api, tokens, accounts, store, registry, clock, script, http)


def bearer(request: httpx.Request) -> str:
    return request.headers["Authorization"]


def form(request: httpx.Request) -> dict[str, list[str]]:
    return parse_qs(request.content.decode("ascii"), strict_parsing=True)


PLAYER_STATE = {
    "is_playing": True,
    "progress_ms": 1200,
    "device": {
        "id": "dev-1",
        "is_active": True,
        "name": "Kitchen speaker",
        "type": "Speaker",
        "volume_percent": 40,
    },
    "item": {
        "uri": "spotify:track:fake1",
        "name": "So What",
        "artists": [{"name": "Miles Davis"}],
        "album": {"name": "Kind of Blue"},
    },
}


# -- endpoints ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_playback_state_parses_device_and_item(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(httpx.Response(200, json=PLAYER_STATE)))
    state = await rig.api.playback_state()
    assert state is not None
    assert state.is_playing is True
    assert state.progress_ms == 1200
    assert state.device == SpotifyDevice(
        id="dev-1", name="Kitchen speaker", type="Speaker", is_active=True, volume_percent=40
    )
    assert state.item == PlayableItem(
        uri="spotify:track:fake1", name="So What", artists=("Miles Davis",), album="Kind of Blue"
    )
    (request,) = rig.script.requests
    assert request.method == "GET"
    assert str(request.url) == f"{API_BASE}/me/player"
    assert bearer(request) == f"Bearer {OLD_ACCESS}"


@pytest.mark.asyncio
async def test_playback_state_is_none_on_204(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(httpx.Response(204)))
    assert await rig.api.playback_state() is None


@pytest.mark.asyncio
async def test_devices_lists_every_device(tmp_path: Path) -> None:
    body = {
        "devices": [
            {
                "id": "dev-1",
                "is_active": False,
                "name": "Desk",
                "type": "Computer",
                "volume_percent": None,
            },
            {"id": None, "is_active": True, "name": "Car", "type": "Automobile"},
        ]
    }
    rig = build(tmp_path, Script(httpx.Response(200, json=body)))
    devices = await rig.api.devices()
    assert devices == [
        SpotifyDevice(
            id="dev-1", name="Desk", type="Computer", is_active=False, volume_percent=None
        ),
        SpotifyDevice(id=None, name="Car", type="Automobile", is_active=True, volume_percent=None),
    ]
    assert str(rig.script.requests[0].url) == f"{API_BASE}/me/player/devices"


@pytest.mark.asyncio
async def test_play_context_on_a_device(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(httpx.Response(204)))
    await rig.api.play(context_uri="spotify:album:fake1", device_id="dev-1")
    (request,) = rig.script.requests
    assert request.method == "PUT"
    assert request.url.path == "/v1/me/player/play"
    assert request.url.params["device_id"] == "dev-1"
    assert json.loads(request.content) == {"context_uri": "spotify:album:fake1"}


@pytest.mark.asyncio
async def test_play_uris(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(httpx.Response(202)))
    await rig.api.play(uris=["spotify:track:fake1"])
    (request,) = rig.script.requests
    assert "device_id" not in request.url.params
    assert json.loads(request.content) == {"uris": ["spotify:track:fake1"]}


@pytest.mark.asyncio
async def test_play_without_context_or_uris_resumes(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(httpx.Response(204)))
    await rig.api.play()
    (request,) = rig.script.requests
    assert request.method == "PUT"
    assert request.url.path == "/v1/me/player/play"
    assert request.content == b""


@pytest.mark.asyncio
async def test_play_refuses_both_context_and_uris(tmp_path: Path) -> None:
    rig = build(tmp_path, Script())
    with pytest.raises(ValueError, match="context_uri or uris"):
        await rig.api.play(context_uri="spotify:album:fake1", uris=["spotify:track:fake1"])
    assert rig.script.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "method", "path"),
    [
        ("pause", "PUT", "/v1/me/player/pause"),
        ("next", "POST", "/v1/me/player/next"),
        ("previous", "POST", "/v1/me/player/previous"),
    ],
)
async def test_transport_controls(tmp_path: Path, call: str, method: str, path: str) -> None:
    rig = build(tmp_path, Script(httpx.Response(204), httpx.Response(204)))
    await getattr(rig.api, call)()
    await getattr(rig.api, call)(device_id="dev-1")
    first, second = rig.script.requests
    assert (first.method, first.url.path) == (method, path)
    assert "device_id" not in first.url.params
    assert second.url.params["device_id"] == "dev-1"


@pytest.mark.asyncio
async def test_set_volume(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(httpx.Response(204)))
    await rig.api.set_volume(40, device_id="dev-1")
    (request,) = rig.script.requests
    assert (request.method, request.url.path) == ("PUT", "/v1/me/player/volume")
    assert request.url.params["volume_percent"] == "40"
    assert request.url.params["device_id"] == "dev-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("level", [-1, 101])
async def test_set_volume_refuses_out_of_range(tmp_path: Path, level: int) -> None:
    rig = build(tmp_path, Script())
    with pytest.raises(ValueError, match="0 to 100"):
        await rig.api.set_volume(level)
    assert rig.script.requests == []


@pytest.mark.asyncio
async def test_transfer(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(httpx.Response(204)))
    await rig.api.transfer("dev-2", play=True)
    (request,) = rig.script.requests
    assert (request.method, request.url.path) == ("PUT", "/v1/me/player")
    assert json.loads(request.content) == {"device_ids": ["dev-2"], "play": True}


@pytest.mark.asyncio
async def test_search_parses_each_kind_and_skips_null_items(tmp_path: Path) -> None:
    body = {
        "tracks": {
            "items": [
                {
                    "uri": "spotify:track:t1",
                    "name": "So What",
                    "artists": [{"name": "Miles Davis"}],
                    "album": {"name": "Kind of Blue"},
                }
            ]
        },
        "albums": {
            "items": [
                {
                    "uri": "spotify:album:a1",
                    "name": "Kind of Blue",
                    "artists": [{"name": "Miles Davis"}],
                }
            ]
        },
        "playlists": {"items": [None, {"uri": "spotify:playlist:p1", "name": "Jazz Classics"}]},
        "artists": {"items": [{"uri": "spotify:artist:r1", "name": "Miles Davis"}]},
    }
    rig = build(tmp_path, Script(httpx.Response(200, json=body)))
    results = await rig.api.search("kind of blue", limit=3)
    (request,) = rig.script.requests
    assert (request.method, request.url.path) == ("GET", "/v1/search")
    assert request.url.params["q"] == "kind of blue"
    assert request.url.params["type"] == "track,album,playlist,artist"
    assert request.url.params["limit"] == "3"
    assert [(i.kind, i.uri, i.name, i.artists) for i in results.tracks] == [
        ("track", "spotify:track:t1", "So What", ("Miles Davis",))
    ]
    assert [(i.kind, i.uri, i.artists) for i in results.albums] == [
        ("album", "spotify:album:a1", ("Miles Davis",))
    ]
    assert [(i.kind, i.name) for i in results.playlists] == [("playlist", "Jazz Classics")]
    assert [(i.kind, i.name, i.artists) for i in results.artists] == [("artist", "Miles Davis", ())]


@pytest.mark.asyncio
async def test_search_with_selected_kinds_reports_the_others_empty(tmp_path: Path) -> None:
    body = {"tracks": {"items": []}}
    rig = build(tmp_path, Script(httpx.Response(200, json=body)))
    results = await rig.api.search("jazz", kinds=("track",))
    assert rig.script.requests[0].url.params["type"] == "track"
    assert results.tracks == results.albums == results.playlists == results.artists == ()


@pytest.mark.asyncio
async def test_search_refuses_empty_query_and_bad_kind(tmp_path: Path) -> None:
    rig = build(tmp_path, Script())
    with pytest.raises(ValueError):
        await rig.api.search("   ")
    with pytest.raises(ValueError):
        await rig.api.search("jazz", kinds=("episode",))  # type: ignore[arg-type]
    assert rig.script.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=[1, 2]),
        httpx.Response(200, json={"is_playing": "yes"}),
    ],
    ids=["not-json", "list", "wrong-type"],
)
async def test_malformed_success_body_is_an_api_error(
    tmp_path: Path, response: httpx.Response
) -> None:
    rig = build(tmp_path, Script(response))
    with pytest.raises(SpotifyApiError) as excinfo:
        await rig.api.playback_state()
    assert (excinfo.value.status, excinfo.value.reason) == (200, "malformed_response")


# -- status mapping ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_401_refreshes_once_and_retries(tmp_path: Path) -> None:
    script = Script(api_error(401, None), token_response(), httpx.Response(204))
    rig = build(tmp_path, script)
    await rig.api.pause()
    first, refresh, retry = script.requests
    assert bearer(first) == f"Bearer {OLD_ACCESS}"
    assert str(refresh.url) == TOKEN_URL
    assert bearer(retry) == f"Bearer {NEW_ACCESS}"
    assert retry.url.path == "/v1/me/player/pause"


@pytest.mark.asyncio
async def test_401_after_the_retry_is_not_authorized(tmp_path: Path) -> None:
    script = Script(api_error(401, None), token_response(), api_error(401, None))
    rig = build(tmp_path, script)
    with pytest.raises(SpotifyAuthError) as excinfo:
        await rig.api.pause()
    assert str(excinfo.value).startswith("not_authorized")
    assert len(script.token_requests()) == 1
    assert len(script.api_requests()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "spotify_reason", "reason"),
    [
        (403, "PREMIUM_REQUIRED", "premium_required"),
        (403, "UNKNOWN", "forbidden"),
        (404, "NO_ACTIVE_DEVICE", "no_active_device"),
        (404, None, "not_found"),
        (400, None, "bad_request"),
        (500, None, "server_error"),
        (503, None, "server_error"),
        (418, None, "unexpected_status"),
    ],
)
async def test_error_status_mapping(
    tmp_path: Path, status: int, spotify_reason: str | None, reason: str
) -> None:
    rig = build(tmp_path, Script(api_error(status, spotify_reason)))
    with pytest.raises(SpotifyApiError) as excinfo:
        await rig.api.play()
    assert (excinfo.value.status, excinfo.value.reason) == (status, reason)
    assert not isinstance(excinfo.value, SpotifyRateLimitedError)
    assert len(rig.script.requests) == 1


@pytest.mark.asyncio
async def test_error_status_with_a_non_json_body_still_maps(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(httpx.Response(404, content=b"<html>nope</html>")))
    with pytest.raises(SpotifyApiError) as excinfo:
        await rig.api.next()
    assert (excinfo.value.status, excinfo.value.reason) == (404, "not_found")


@pytest.mark.asyncio
async def test_429_is_rate_limited_with_retry_after(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(api_error(429, None, headers={"Retry-After": "7"})))
    with pytest.raises(SpotifyRateLimitedError) as excinfo:
        await rig.api.devices()
    error = excinfo.value
    assert isinstance(error, SpotifyApiError)
    assert (error.status, error.reason, error.retry_after_s) == (429, "rate_limited", 7.0)
    assert len(rig.script.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [None, {"Retry-After": "soon"}, {"Retry-After": "-3"}])
async def test_429_without_a_usable_retry_after(
    tmp_path: Path, headers: dict[str, str] | None
) -> None:
    rig = build(tmp_path, Script(api_error(429, None, headers=headers)))
    with pytest.raises(SpotifyRateLimitedError) as excinfo:
        await rig.api.devices()
    assert excinfo.value.retry_after_s is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("refused"), httpx.ConnectTimeout("slow")],
    ids=["connect-error", "connect-timeout"],
)
async def test_connect_failure_is_unreachable(tmp_path: Path, error: Exception) -> None:
    rig = build(tmp_path, Script(error))
    with pytest.raises(SpotifyApiError) as excinfo:
        await rig.api.pause()
    assert (excinfo.value.status, excinfo.value.reason) == (0, "unreachable")


@pytest.mark.asyncio
async def test_other_transport_failure_is_transport_error(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(httpx.ReadTimeout("slow body")))
    with pytest.raises(SpotifyApiError) as excinfo:
        await rig.api.pause()
    assert (excinfo.value.status, excinfo.value.reason) == (0, "transport_error")


@pytest.mark.asyncio
async def test_token_endpoint_connect_error_is_unreachable(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(httpx.ConnectError("refused")), expires_in_s=10)
    with pytest.raises(SpotifyApiError) as excinfo:
        await rig.api.pause()
    assert (excinfo.value.status, excinfo.value.reason) == (0, "unreachable")


# -- token manager -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_with_more_than_60s_left_is_used_without_refresh(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(), expires_in_s=61)
    assert (await rig.tokens.access_token()).reveal() == OLD_ACCESS
    assert rig.script.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_in_s", [60, 59, 0, -600])
async def test_token_within_60s_of_expiry_is_refreshed_first(
    tmp_path: Path, expires_in_s: float
) -> None:
    script = Script(token_response(), httpx.Response(204))
    rig = build(tmp_path, script, expires_in_s=expires_in_s)
    await rig.api.pause()
    refresh, call = script.requests
    assert str(refresh.url) == TOKEN_URL
    assert bearer(call) == f"Bearer {NEW_ACCESS}"


@pytest.mark.asyncio
async def test_token_ages_into_the_refresh_window(tmp_path: Path) -> None:
    script = Script(token_response())
    rig = build(tmp_path, script, expires_in_s=120)
    assert (await rig.tokens.access_token()).reveal() == OLD_ACCESS
    await rig.clock.advance(61)
    assert (await rig.tokens.access_token()).reveal() == NEW_ACCESS
    assert len(script.token_requests()) == 1


@pytest.mark.asyncio
async def test_refresh_request_is_a_pkce_refresh_without_a_client_secret(tmp_path: Path) -> None:
    script = Script(token_response())
    rig = build(tmp_path, script, expires_in_s=0)
    await rig.tokens.access_token()
    (request,) = script.requests
    assert request.method == "POST"
    assert request.headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert "Authorization" not in request.headers
    assert form(request) == {
        "grant_type": ["refresh_token"],
        "refresh_token": [OLD_REFRESH],
        "client_id": [CLIENT_ID],
    }


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_refresh(tmp_path: Path) -> None:
    script = Script(token_response())
    rig = build(tmp_path, script, expires_in_s=0)
    first, second, third = await asyncio.gather(
        rig.tokens.access_token(), rig.tokens.access_token(), rig.tokens.access_token()
    )
    assert {first.reveal(), second.reveal(), third.reveal()} == {NEW_ACCESS}
    assert len(script.token_requests()) == 1


@pytest.mark.asyncio
async def test_concurrent_401s_share_one_refresh(tmp_path: Path) -> None:
    script = Script(
        api_error(401, None),
        api_error(401, None),
        token_response(),
        httpx.Response(204),
        httpx.Response(204),
    )
    rig = build(tmp_path, script)
    await asyncio.gather(rig.api.pause(), rig.api.next())
    assert len(script.token_requests()) == 1
    retries = script.api_requests()[2:]
    assert [bearer(r) for r in retries] == [f"Bearer {NEW_ACCESS}"] * 2


@pytest.mark.asyncio
async def test_rotated_refresh_token_is_saved(tmp_path: Path) -> None:
    script = Script(token_response(refresh=NEW_REFRESH, expires_in=1800))
    rig = build(tmp_path, script, expires_in_s=0)
    await rig.tokens.access_token()
    saved = rig.store.load()
    assert saved.access_token.reveal() == NEW_ACCESS
    assert saved.refresh_token.reveal() == NEW_REFRESH
    assert saved.expires_at == rig.clock.now() + timedelta(seconds=1800)


@pytest.mark.asyncio
async def test_refresh_without_rotation_keeps_the_old_refresh_token(tmp_path: Path) -> None:
    script = Script(token_response(refresh=None), token_response(access="fake-access-0003"))
    rig = build(tmp_path, script, expires_in_s=0)
    await rig.tokens.access_token()
    saved = rig.store.load()
    assert saved.access_token.reveal() == NEW_ACCESS
    assert saved.refresh_token.reveal() == OLD_REFRESH
    await rig.clock.advance(3600)
    await rig.tokens.access_token()
    assert form(script.requests[1])["refresh_token"] == [OLD_REFRESH]


@pytest.mark.asyncio
async def test_loaded_and_refreshed_tokens_are_all_registered(tmp_path: Path) -> None:
    script = Script(token_response())
    rig = build(tmp_path, script, expires_in_s=0)
    for value in (OLD_ACCESS, OLD_REFRESH, NEW_ACCESS, NEW_REFRESH):
        assert redact(value, rig.registry) == value
    await rig.tokens.access_token()
    for value in (OLD_ACCESS, OLD_REFRESH, NEW_ACCESS, NEW_REFRESH):
        assert redact(value, rig.registry) == "***"


@pytest.mark.asyncio
async def test_exchanged_code_tokens_are_registered(tmp_path: Path) -> None:
    script = Script(token_response(access="fake-exchanged-access", refresh="fake-exchanged-ref"))
    rig = build(tmp_path, script)
    token = await rig.accounts.exchange_code(
        code="fake-code", verifier="v" * 43, redirect_uri=REDIRECT
    )
    assert token.access_token.reveal() == "fake-exchanged-access"
    assert token.refresh_token.reveal() == "fake-exchanged-ref"
    assert token.expires_at == rig.clock.now() + timedelta(seconds=3600)
    assert redact("fake-exchanged-access fake-exchanged-ref", rig.registry) == "*** ***"
    (request,) = script.requests
    assert form(request) == {
        "grant_type": ["authorization_code"],
        "code": ["fake-code"],
        "redirect_uri": [REDIRECT],
        "client_id": [CLIENT_ID],
        "code_verifier": ["v" * 43],
    }


@pytest.mark.asyncio
async def test_exchange_without_a_refresh_token_is_malformed(tmp_path: Path) -> None:
    rig = build(tmp_path, Script(token_response(refresh=None)))
    with pytest.raises(SpotifyApiError) as excinfo:
        await rig.accounts.exchange_code(code="c", verifier="v" * 43, redirect_uri=REDIRECT)
    assert excinfo.value.reason == "malformed_response"


@pytest.mark.asyncio
async def test_invalid_grant_is_not_authorized(tmp_path: Path) -> None:
    script = Script(httpx.Response(400, json={"error": "invalid_grant"}))
    rig = build(tmp_path, script, expires_in_s=0)
    with pytest.raises(SpotifyAuthError) as excinfo:
        await rig.api.pause()
    message = str(excinfo.value)
    assert message.startswith("not_authorized")
    assert "invalid_grant" in message
    assert AUTHORIZE_COMMAND in message
    assert script.api_requests() == []


@pytest.mark.asyncio
async def test_after_invalid_grant_a_reauthorized_file_is_picked_up(tmp_path: Path) -> None:
    script = Script(httpx.Response(400, json={"error": "invalid_grant"}), httpx.Response(204))
    rig = build(tmp_path, script, expires_in_s=0)
    with pytest.raises(SpotifyAuthError):
        await rig.tokens.access_token()
    rig.store.save(
        SpotifyToken(
            access_token=Secret("fake-reauthorized-access"),
            refresh_token=Secret("fake-reauthorized-refresh"),
            expires_at=rig.clock.now() + timedelta(hours=1),
            scope="",
        )
    )
    await rig.api.pause()
    assert bearer(script.requests[-1]) == "Bearer fake-reauthorized-access"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error_type", "reason"),
    [
        (httpx.Response(400, json={"error": "invalid_client"}), SpotifyAuthError, None),
        (httpx.Response(401, json={"error": "invalid_client"}), SpotifyAuthError, None),
        (httpx.Response(500, content=b"down"), SpotifyApiError, "server_error"),
        (
            httpx.Response(429, headers={"Retry-After": "2"}),
            SpotifyRateLimitedError,
            "rate_limited",
        ),
        (httpx.Response(200, json={"token_type": "Bearer"}), SpotifyApiError, "malformed_response"),
        (httpx.Response(200, content=b"not json"), SpotifyApiError, "malformed_response"),
    ],
    ids=["invalid-client-400", "invalid-client-401", "500", "429", "no-access", "not-json"],
)
async def test_token_endpoint_failures(
    tmp_path: Path, response: httpx.Response, error_type: type[SpotifyError], reason: str | None
) -> None:
    rig = build(tmp_path, Script(response), expires_in_s=0)
    with pytest.raises(error_type) as excinfo:
        await rig.tokens.access_token()
    if reason is None:
        assert str(excinfo.value).startswith("not_authorized")
    else:
        assert isinstance(excinfo.value, SpotifyApiError)
        assert excinfo.value.reason == reason


@pytest.mark.asyncio
async def test_missing_token_file_is_not_authorized_without_a_request(tmp_path: Path) -> None:
    script = Script(httpx.Response(204))
    rig = build(tmp_path, script)
    rig.store.path.unlink()
    with pytest.raises(SpotifyAuthError) as excinfo:
        await rig.api.pause()
    assert AUTHORIZE_COMMAND in str(excinfo.value)
    assert script.requests == []


@pytest.mark.asyncio
async def test_failed_save_after_refresh_is_logged_and_the_token_still_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    script = Script(token_response(), httpx.Response(204))
    rig = build(tmp_path, script, expires_in_s=0)

    def failing_save(token: SpotifyToken) -> None:
        raise SpotifyError("token file cannot be written: OSError")

    monkeypatch.setattr(rig.store, "save", failing_save)
    with caplog.at_level(logging.ERROR, logger="backend.capabilities.music.spotify_api"):
        await rig.api.pause()
    assert bearer(script.requests[-1]) == f"Bearer {NEW_ACCESS}"
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "cannot be written" in errors[0].getMessage()


# -- logs --------------------------------------------------------------------------------


@pytest.fixture()
def restore_record_factory() -> Iterator[None]:
    original = logging.getLogRecordFactory()
    yield
    logging.setLogRecordFactory(original)


@pytest.mark.asyncio
@pytest.mark.usefixtures("restore_record_factory")
async def test_no_token_value_reaches_a_log_record(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    script = Script(
        token_response(),
        api_error(401, None),
        token_response(access="fake-access-token-0003", refresh="fake-refresh-token-0003"),
        httpx.Response(200, json=PLAYER_STATE),
        httpx.Response(400, json={"error": "invalid_grant"}),
    )
    rig = build(tmp_path, script, expires_in_s=0)
    install_log_redaction(rig.registry)
    values = (
        OLD_ACCESS,
        OLD_REFRESH,
        NEW_ACCESS,
        NEW_REFRESH,
        "fake-access-token-0003",
        "fake-refresh-token-0003",
    )
    with caplog.at_level(logging.DEBUG):
        await rig.api.playback_state()
        await rig.clock.advance(7200)
        with pytest.raises(SpotifyAuthError):
            await rig.tokens.access_token()
        # A record that carries a refreshed token, as a careless caller might log it.
        logging.getLogger("tests.spotify").warning("token was %s", "fake-access-token-0003")

    module_records = [r for r in caplog.records if r.name.startswith("backend.capabilities.music")]
    assert module_records, "the client logs its requests and refreshes"
    for record in caplog.records:
        message = record.getMessage()
        for value in values:
            assert value not in message
    for value in values:
        assert value not in caplog.text
    assert "token was ***" in caplog.text
