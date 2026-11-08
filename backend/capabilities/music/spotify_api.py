"""The Spotify Web API client, the accounts token endpoint, and the token manager between them.

Every response is checked: a 2xx is parsed, and every other status becomes a `SpotifyError`
whose `reason` is a short word the music capability reports (`premium_required`,
`no_active_device`, `rate_limited`, `unreachable`, ...). A transport failure carries status 0,
because no response arrived.

Token values cross this module only as `Secret`s. Every value that enters the process, from the
token file or from the token endpoint, is registered with the `SecretRegistry` at the point it
enters, so the log redaction knows it before any record could carry it.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal, TypeVar

import httpx

from backend.capabilities.clock import Clock
from backend.capabilities.secrets import Secret, SecretRegistry
from backend.errors import SpotifyApiError, SpotifyAuthError, SpotifyError

from .token_store import SpotifyToken, SpotifyTokenStore, not_authorized

logger = logging.getLogger(__name__)

API_BASE = "https://api.spotify.com/v1"
# bandit B105 reads "token" in the name as a password; this is the endpoint's URL.
TOKEN_URL = "https://accounts.spotify.com/api/token"  # nosec B105

# Refresh this long before expiry, so a token never expires between the check and the request.
REFRESH_MARGIN_S = 60.0

SearchKind = Literal["track", "album", "playlist", "artist"]
SEARCH_KINDS: tuple[SearchKind, ...] = ("track", "album", "playlist", "artist")
_MAX_SEARCH_LIMIT = 50

T = TypeVar("T")


class SpotifyRateLimitedError(SpotifyApiError):
    """A 429; `retry_after_s` is Spotify's Retry-After in seconds, or None when absent."""

    def __init__(self, retry_after_s: float | None) -> None:
        super().__init__(429, "rate_limited")
        self.retry_after_s = retry_after_s


@dataclass(frozen=True, slots=True)
class SpotifyDevice:
    """A Spotify Connect device; `id` is None for a device Spotify will not let MIST target."""

    id: str | None
    name: str
    type: str
    is_active: bool
    volume_percent: int | None


@dataclass(frozen=True, slots=True)
class PlayableItem:
    """The current track or episode; for an episode, `artists` holds the show's name."""

    uri: str
    name: str
    artists: tuple[str, ...]
    album: str | None


@dataclass(frozen=True, slots=True)
class PlaybackState:
    """The player as `GET /me/player` reports it."""

    is_playing: bool
    device: SpotifyDevice | None
    item: PlayableItem | None
    progress_ms: int | None


@dataclass(frozen=True, slots=True)
class SearchItem:
    """One search hit; `artists` is empty for a playlist or an artist."""

    kind: SearchKind
    uri: str
    name: str
    artists: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SearchResults:
    """Search hits by kind, each in Spotify's order; a kind not searched for is empty."""

    tracks: tuple[SearchItem, ...]
    albums: tuple[SearchItem, ...]
    playlists: tuple[SearchItem, ...]
    artists: tuple[SearchItem, ...]


class SpotifyAccounts:
    """The accounts service's token endpoint, for a public PKCE client (no client secret).

    Args:
        http: the client every request goes through; its timeouts bound each call.
        client_id: the Spotify app's client id, which is not a secret.
        registry: receives every access and refresh token the endpoint returns.
        clock: turns `expires_in` into an absolute expiry.
        token_url: the endpoint; tests and the factory pass the default.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        client_id: str,
        registry: SecretRegistry,
        clock: Clock,
        token_url: str = TOKEN_URL,
    ) -> None:
        self._http = http
        self._client_id = client_id
        self._registry = registry
        self._clock = clock
        self._token_url = token_url

    async def exchange_code(self, *, code: str, verifier: str, redirect_uri: str) -> SpotifyToken:
        """Exchange an authorization code for a token (the authorize CLI's last step).

        Raises:
            SpotifyAuthError: the endpoint refused the code or the client id.
            SpotifyApiError: unreachable, rate limited, a server error, or a malformed body,
                including one without a refresh token.
        """
        return await self._request(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": self._client_id,
                "code_verifier": verifier,
            },
            previous_refresh=None,
        )

    async def refresh(self, token: SpotifyToken) -> SpotifyToken:
        """Return a renewed token; it keeps `token`'s refresh token unless Spotify rotated it.

        Raises:
            SpotifyAuthError: `invalid_grant` (the refresh token was revoked or has expired) or
                another refusal; the message starts with `not_authorized`.
            SpotifyApiError: unreachable, rate limited, a server error, or a malformed body.
        """
        return await self._request(
            {
                "grant_type": "refresh_token",
                "refresh_token": token.refresh_token.reveal(),
                "client_id": self._client_id,
            },
            previous_refresh=token.refresh_token,
        )

    async def _request(
        self, form: Mapping[str, str], *, previous_refresh: Secret | None
    ) -> SpotifyToken:
        grant = form["grant_type"]
        response = await _send(self._http, "POST", self._token_url, data=form)
        status = response.status_code
        logger.debug("Spotify token endpoint (%s) -> %d", grant, status)
        if status == 200:
            return self._parse_token(response, previous_refresh)
        if status == 429:
            raise SpotifyRateLimitedError(_retry_after(response))
        if status in (400, 401):
            error = _oauth_error(response)
            logger.warning("Spotify token endpoint refused the %s request: %s", grant, error)
            raise not_authorized(f"Spotify refused the {grant} request ({error})")
        raise SpotifyApiError(status, "server_error" if status >= 500 else "unexpected_status")

    def _parse_token(
        self, response: httpx.Response, previous_refresh: Secret | None
    ) -> SpotifyToken:
        data = _json_body(response)
        if not isinstance(data, dict):
            raise _malformed(response, "the token response is not a JSON object")
        access = data.get("access_token")
        refresh = data.get("refresh_token")
        # Register before validating, so even a token in a rejected response is redacted.
        for value in (access, refresh):
            if isinstance(value, str):
                self._registry.register(Secret(value))
        if not isinstance(access, str) or not access:
            raise _malformed(response, "the token response has no access_token")
        if refresh is None and previous_refresh is not None:
            refresh_secret = previous_refresh
        elif isinstance(refresh, str) and refresh:
            refresh_secret = Secret(refresh)
        else:
            raise _malformed(response, "the token response has no refresh_token")
        expires_in = data.get("expires_in")
        if (
            isinstance(expires_in, bool)
            or not isinstance(expires_in, int | float)
            or not math.isfinite(expires_in)
            or expires_in <= 0
        ):
            raise _malformed(response, "the token response has no positive expires_in")
        scope = data.get("scope", "")
        token_type = data.get("token_type", "Bearer")
        if not isinstance(scope, str) or not isinstance(token_type, str):
            raise _malformed(response, "the token response has a non-string scope or token_type")
        return SpotifyToken(
            access_token=Secret(access),
            refresh_token=refresh_secret,
            expires_at=self._clock.now() + timedelta(seconds=expires_in),
            scope=scope,
            token_type=token_type,
        )


class SpotifyTokenManager:
    """Hands out a current access token, refreshing it under a single-flight lock.

    The token is loaded from the store on first use, not at construction, so a missing file
    fails the call that needs it (naming the authorize CLI) instead of startup, and a file the
    CLI writes later is picked up without a restart.

    Args:
        store: the token file; read on first use and after a refused refresh, written after
            every refresh.
        accounts: the token endpoint.
        registry: receives every token loaded from the file.
        clock: decides when a token is due for refresh.
        refresh_margin_s: how long before expiry a token is refreshed.
    """

    def __init__(
        self,
        *,
        store: SpotifyTokenStore,
        accounts: SpotifyAccounts,
        registry: SecretRegistry,
        clock: Clock,
        refresh_margin_s: float = REFRESH_MARGIN_S,
    ) -> None:
        self._store = store
        self._accounts = accounts
        self._registry = registry
        self._clock = clock
        self._margin = timedelta(seconds=refresh_margin_s)
        self._token: SpotifyToken | None = None
        self._lock = asyncio.Lock()

    async def access_token(self) -> Secret:
        """Return an access token with more than the margin left, refreshing first if needed.

        Concurrent callers that find the token due wait on one refresh rather than each
        sending their own; Spotify may rotate the refresh token, and a second refresh with the
        old one could then be refused.

        Raises:
            SpotifyAuthError: no usable token (missing file or a refused refresh).
            SpotifyApiError: the token endpoint failed in another way.
        """
        token = self._token
        if token is not None and not self._due(token):
            return token.access_token
        async with self._lock:
            token = await self._current()
            if self._due(token):
                token = await self._refresh(token)
            return token.access_token

    async def refresh_rejected(self, rejected: Secret) -> Secret:
        """Return a token to retry with after Spotify answered 401 to `rejected`.

        When another caller has already replaced `rejected`, its replacement is returned
        without a second refresh.

        Raises:
            SpotifyAuthError, SpotifyApiError: as `access_token`.
        """
        async with self._lock:
            token = await self._current()
            if token.access_token.reveal() != rejected.reveal():
                return token.access_token
            token = await self._refresh(token)
            return token.access_token

    def _due(self, token: SpotifyToken) -> bool:
        return self._clock.now() >= token.expires_at - self._margin

    async def _current(self) -> SpotifyToken:
        if self._token is None:
            token = await asyncio.to_thread(self._store.load)
            self._register(token)
            self._token = token
        return self._token

    async def _refresh(self, token: SpotifyToken) -> SpotifyToken:
        try:
            renewed = await self._accounts.refresh(token)
        except SpotifyAuthError:
            # Forget the dead token, so the next call rereads the file the authorize CLI writes.
            self._token = None
            raise
        self._register(renewed)
        self._token = renewed
        rotated = renewed.refresh_token.reveal() != token.refresh_token.reveal()
        try:
            await asyncio.to_thread(self._store.save, renewed)
        except SpotifyError as exc:
            # The renewed token still works in memory; a restart before the next successful
            # save falls back to the file's token, and to the authorize CLI if Spotify rotated
            # the refresh token.
            logger.error(
                "Spotify token refreshed but not saved (refresh token rotated: %s): %s",
                rotated,
                exc,
            )
        else:
            logger.info(
                "Spotify access token refreshed (refresh token rotated: %s); expires %s",
                rotated,
                renewed.expires_at.isoformat(),
            )
        return renewed

    def _register(self, token: SpotifyToken) -> None:
        self._registry.register(token.access_token)
        self._registry.register(token.refresh_token)


class SpotifyApi:
    """The Web API endpoints the music capability uses: player, devices and search.

    Args:
        http: the client every request goes through; its timeouts bound each call.
        tokens: supplies the bearer token and the one refresh after a 401.
        base_url: the API root; tests and the factory pass the default.
    """

    def __init__(
        self, http: httpx.AsyncClient, tokens: SpotifyTokenManager, *, base_url: str = API_BASE
    ) -> None:
        self._http = http
        self._tokens = tokens
        self._base = base_url.rstrip("/")

    async def playback_state(self) -> PlaybackState | None:
        """Return the player's state, or None when Spotify reports no player (204)."""
        response = await self._call("GET", "/me/player")
        if response.status_code == 204:
            return None
        return _parse(response, _player_state)

    async def devices(self) -> list[SpotifyDevice]:
        """Return the account's Spotify Connect devices."""
        response = await self._call("GET", "/me/player/devices")
        return _parse(response, _device_list)

    async def play(
        self,
        *,
        context_uri: str | None = None,
        uris: Sequence[str] | None = None,
        device_id: str | None = None,
    ) -> None:
        """Play a context (album, playlist, artist) or tracks; with neither, resume.

        Raises:
            ValueError: both `context_uri` and `uris` were given, or `uris` is empty.
        """
        if context_uri is not None and uris is not None:
            raise ValueError("play takes context_uri or uris, not both")
        body: dict[str, object] | None = None
        if context_uri is not None:
            body = {"context_uri": context_uri}
        elif uris is not None:
            if not uris:
                raise ValueError("play needs at least one uri")
            body = {"uris": list(uris)}
        await self._call("PUT", "/me/player/play", params={"device_id": device_id}, body=body)

    async def pause(self, *, device_id: str | None = None) -> None:
        """Pause playback."""
        await self._call("PUT", "/me/player/pause", params={"device_id": device_id})

    async def next(self, *, device_id: str | None = None) -> None:
        """Skip to the next item."""
        await self._call("POST", "/me/player/next", params={"device_id": device_id})

    async def previous(self, *, device_id: str | None = None) -> None:
        """Go back to the previous item."""
        await self._call("POST", "/me/player/previous", params={"device_id": device_id})

    async def set_volume(self, level_pct: int, *, device_id: str | None = None) -> None:
        """Set the volume, 0 to 100 percent.

        Raises:
            ValueError: `level_pct` is not an integer from 0 to 100.
        """
        if isinstance(level_pct, bool) or not isinstance(level_pct, int):
            raise ValueError("volume is an integer from 0 to 100")
        if not 0 <= level_pct <= 100:
            raise ValueError(f"volume is 0 to 100 percent, got {level_pct}")
        await self._call(
            "PUT",
            "/me/player/volume",
            params={"volume_percent": str(level_pct), "device_id": device_id},
        )

    async def transfer(self, device_id: str, *, play: bool = True) -> None:
        """Move playback to `device_id`; with `play`, start playing there."""
        await self._call("PUT", "/me/player", body={"device_ids": [device_id], "play": play})

    async def search(
        self, query: str, *, kinds: Sequence[SearchKind] = SEARCH_KINDS, limit: int = 5
    ) -> SearchResults:
        """Search the catalog for `query`, returning up to `limit` hits per kind.

        Raises:
            ValueError: an empty query, no kinds or an unknown kind, or `limit` outside 1 to 50.
        """
        if not query.strip():
            raise ValueError("search needs a query")
        if not kinds or any(kind not in SEARCH_KINDS for kind in kinds):
            raise ValueError(f"search kinds are {', '.join(SEARCH_KINDS)}")
        if not 1 <= limit <= _MAX_SEARCH_LIMIT:
            raise ValueError(f"search limit is 1 to {_MAX_SEARCH_LIMIT}, got {limit}")
        response = await self._call(
            "GET",
            "/search",
            params={"q": query, "type": ",".join(kinds), "limit": str(limit)},
        )
        return _parse(response, _search_results)

    async def _call(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | None] | None = None,
        body: Mapping[str, object] | None = None,
    ) -> httpx.Response:
        """Send one request, refreshing once and retrying on a 401; return only a 2xx."""
        query = {key: value for key, value in (params or {}).items() if value is not None}
        url = f"{self._base}{path}"
        token = await self._tokens.access_token()
        response = await self._send(method, url, path, token, query, body)
        if response.status_code == 401:
            logger.info("Spotify rejected the access token on %s %s; refreshing once", method, path)
            token = await self._tokens.refresh_rejected(token)
            response = await self._send(method, url, path, token, query, body)
            if response.status_code == 401:
                raise not_authorized("Spotify rejected a freshly refreshed access token")
        if 200 <= response.status_code < 300:
            return response
        error = _api_error(response)
        logger.warning("Spotify %s %s -> %d (%s)", method, path, error.status, error.reason)
        raise error

    async def _send(
        self,
        method: str,
        url: str,
        path: str,
        token: Secret,
        params: Mapping[str, str],
        body: Mapping[str, object] | None,
    ) -> httpx.Response:
        response = await _send(
            self._http,
            method,
            url,
            headers={"Authorization": f"Bearer {token.reveal()}"},
            params=params,
            json=body,
        )
        # The path only: a search query is the user's words and stays out of the log.
        logger.debug("Spotify %s %s -> %d", method, path, response.status_code)
        return response


async def _send(
    http: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    json: Mapping[str, object] | None = None,
    data: Mapping[str, str] | None = None,
) -> httpx.Response:
    """Send one request, turning a transport failure into a `SpotifyApiError` with status 0.

    A connect failure means the request never reached Spotify (`unreachable`); any other
    transport failure may have happened after Spotify acted on it (`transport_error`).
    """
    try:
        return await http.request(method, url, headers=headers, params=params, json=json, data=data)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        logger.warning("Spotify unreachable: %s", type(exc).__name__)
        raise SpotifyApiError(0, "unreachable") from exc
    except httpx.TransportError as exc:
        logger.warning("Spotify request failed in transport: %s", type(exc).__name__)
        raise SpotifyApiError(0, "transport_error") from exc


def _api_error(response: httpx.Response) -> SpotifyApiError:
    status = response.status_code
    if status == 429:
        return SpotifyRateLimitedError(_retry_after(response))
    spotify_reason = _api_error_reason(response)
    if status == 403:
        reason = "premium_required" if spotify_reason == "PREMIUM_REQUIRED" else "forbidden"
    elif status == 404:
        reason = "no_active_device" if spotify_reason == "NO_ACTIVE_DEVICE" else "not_found"
    elif status == 400:
        reason = "bad_request"
    elif status >= 500:
        reason = "server_error"
    else:
        reason = "unexpected_status"
    return SpotifyApiError(status, reason)


def _api_error_reason(response: httpx.Response) -> str | None:
    """Return the Web API error object's `reason`, or None when the body has none."""
    try:
        data = response.json()
    except ValueError:
        return None
    error = data.get("error") if isinstance(data, dict) else None
    reason = error.get("reason") if isinstance(error, dict) else None
    return reason if isinstance(reason, str) else None


def _oauth_error(response: httpx.Response) -> str:
    """Return the OAuth `error` code (`invalid_grant`, ...), or the status when there is none.

    Only a short lowercase word is returned, so a body that echoes request content never
    reaches a message.
    """
    try:
        data = response.json()
    except ValueError:
        data = None
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, str) and 0 < len(error) <= 40 and error.replace("_", "").isalpha():
        return error.lower()
    return f"status {response.status_code}"


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


# -- response parsing ----------------------------------------------------------------------


class _MalformedBodyError(ValueError):
    """A 2xx body that does not have the shape Spotify documents."""


def _json_body(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        raise _malformed(response, "the body is not JSON") from None


def _malformed(response: httpx.Response, detail: str) -> SpotifyApiError:
    logger.warning("Spotify %s returned an unexpected body: %s", response.request.url.path, detail)
    return SpotifyApiError(response.status_code, "malformed_response")


def _parse(response: httpx.Response, parser: Callable[[object], T]) -> T:
    data = _json_body(response)
    try:
        return parser(data)
    except _MalformedBodyError as exc:
        raise _malformed(response, str(exc)) from None


def _object(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _MalformedBodyError(f"{where} is not an object")
    return value


def _string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise _MalformedBodyError(f"{key} is not a string")
    return value


def _optional_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise _MalformedBodyError(f"{key} is not a string")
    return value


def _optional_int(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _MalformedBodyError(f"{key} is not an integer")
    return value


def _boolean(data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise _MalformedBodyError(f"{key} is not a boolean")
    return value


def _list(data: Mapping[str, object], key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise _MalformedBodyError(f"{key} is not a list")
    return value


def _names(data: Mapping[str, object], key: str) -> tuple[str, ...]:
    if data.get(key) is None:
        return ()
    return tuple(_string(_object(entry, key), "name") for entry in _list(data, key))


def _device(value: object) -> SpotifyDevice:
    data = _object(value, "device")
    return SpotifyDevice(
        id=_optional_string(data, "id"),
        name=_string(data, "name"),
        type=_string(data, "type"),
        is_active=_boolean(data, "is_active"),
        volume_percent=_optional_int(data, "volume_percent"),
    )


def _device_list(value: object) -> list[SpotifyDevice]:
    return [_device(entry) for entry in _list(_object(value, "body"), "devices")]


def _playable(value: object) -> PlayableItem:
    data = _object(value, "item")
    artists = _names(data, "artists")
    if not artists and data.get("show") is not None:
        artists = (_string(_object(data["show"], "show"), "name"),)
    album = data.get("album")
    return PlayableItem(
        uri=_string(data, "uri"),
        name=_string(data, "name"),
        artists=artists,
        album=None if album is None else _string(_object(album, "album"), "name"),
    )


def _player_state(value: object) -> PlaybackState:
    data = _object(value, "body")
    device = data.get("device")
    item = data.get("item")
    return PlaybackState(
        is_playing=_boolean(data, "is_playing"),
        device=None if device is None else _device(device),
        item=None if item is None else _playable(item),
        progress_ms=_optional_int(data, "progress_ms"),
    )


def _search_kind(data: Mapping[str, object], kind: SearchKind) -> tuple[SearchItem, ...]:
    page = data.get(f"{kind}s")
    if page is None:
        return ()
    items: list[SearchItem] = []
    # Spotify returns null entries in some result pages (playlists especially); skip them.
    for entry in _list(_object(page, f"{kind}s"), "items"):
        if entry is None:
            continue
        hit = _object(entry, kind)
        artists = _names(hit, "artists") if kind in ("track", "album") else ()
        items.append(
            SearchItem(
                kind=kind, uri=_string(hit, "uri"), name=_string(hit, "name"), artists=artists
            )
        )
    return tuple(items)


def _search_results(value: object) -> SearchResults:
    data = _object(value, "body")
    return SearchResults(
        tracks=_search_kind(data, "track"),
        albums=_search_kind(data, "album"),
        playlists=_search_kind(data, "playlist"),
        artists=_search_kind(data, "artist"),
    )
