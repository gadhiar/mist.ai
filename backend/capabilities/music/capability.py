"""`MusicCapability`: MIST's music contract over the Spotify Web API client.

The tools (`music.play`, `music.control`, `music.volume`, `music.now_playing`), their argument
rules, the fast path and the spoken summaries belong to MIST; `SpotifyApi` only carries the
calls. Two Spotify behaviours shape the code here:

- A player command with no active device fails with `no_active_device`, which is the normal
  state of an account whose apps are all idle. The capability then moves playback to
  `MIST_SPOTIFY_DEVICE` once and retries, so "play some jazz" works from a cold start.
- The host refuses every call while a capability's status is `unavailable`, and this capability
  has no supervisor that could notice Spotify coming back. So `not_authorized`,
  `premium_required` and `unreachable` read `unavailable` only for `RECHECK_AFTER_S` after the
  failure, then `degraded` until a call succeeds: the next call is let through to find out.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.errors import SpotifyApiError, SpotifyAuthError, SpotifyError

from ..clock import Clock
from ..contracts import (
    Capability,
    CapabilityStatus,
    FastIntent,
    InvocationContext,
    RoutingHints,
    ToolOutcome,
    ToolSpec,
)
from ..replies import SPOKEN_WORDS_MAX, ok_summary, render_read
from .intents import match_music_intent
from .spotify_api import (
    SEARCH_KINDS,
    PlaybackState,
    SearchItem,
    SearchKind,
    SearchResults,
    SpotifyApi,
    SpotifyDevice,
    SpotifyRateLimitedError,
)
from .token_store import SpotifyTokenStore, not_authorized

PLAY = "music.play"
CONTROL = "music.control"
VOLUME = "music.volume"
NOW_PLAYING = "music.now_playing"

RECHECK_AFTER_S = 60.0
"""How long a Spotify outage keeps the status `unavailable` before a call may try again."""

VOLUME_STEP_PCT = 10
"""How far "volume up" or "volume down" moves the level, in percentage points."""

KNOWN_DEVICES_NAMED = 10
"""How many device names an unknown-device answer lists."""

SEARCH_LIMIT = 5
"""Hits asked for per kind; more than one, because Spotify returns null playlist entries."""

NO_ACTIVE_DEVICE = "no_active_device"
NO_RESULTS = "no_results"
UNKNOWN_DEVICE = "unknown device"

# The reasons that mean no call can work until something outside MIST changes.
_OUTAGE_REASONS = frozenset({"premium_required", "unreachable"})
_OUTAGE_SUMMARIES = {
    "premium_required": "Spotify needs a Premium account for that.",
    "unreachable": "Spotify can't be reached right now.",
}
_NOT_AUTHORIZED_SUMMARY = "Spotify is not authorized yet."

_Action = Literal["pause", "resume", "next", "previous"]
_Change = Literal["up", "down"]
_Args = TypeVar("_Args", bound=BaseModel)
_T = TypeVar("_T")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.split()) or None


class PlayArgs(BaseModel):
    """Arguments of `music.play`; no query resumes."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = None
    kind: SearchKind | None = None
    device: str | None = None

    @field_validator("query", "device")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        """Collapse spacing; a blank value is no value."""
        return _clean(value)


class ControlArgs(BaseModel):
    """Arguments of `music.control`."""

    model_config = ConfigDict(extra="forbid")

    action: _Action


class VolumeArgs(BaseModel):
    """Arguments of `music.volume`: exactly one of a level and a change."""

    model_config = ConfigDict(extra="forbid")

    level_pct: int | None = Field(default=None, ge=0, le=100)
    change: _Change | None = None

    @model_validator(mode="after")
    def require_exactly_one(self) -> VolumeArgs:
        """Refuse both or neither; the grammar cannot express this rule."""
        if (self.level_pct is None) == (self.change is None):
            raise ValueError("give exactly one of level_pct and change")
        return self


class NowPlayingArgs(BaseModel):
    """`music.now_playing` takes no arguments."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class _Failure:
    reason: str
    at: datetime


class _NoFallbackDeviceError(SpotifyApiError):
    """No device was active and `MIST_SPOTIFY_DEVICE` is unset or not among the devices."""

    def __init__(self, device_name: str | None) -> None:
        super().__init__(404, NO_ACTIVE_DEVICE)
        self.device_name = device_name


class MusicCapability(Capability):
    """Plays, controls and reports Spotify playback through `SpotifyApi`.

    No `is_explicit` hook exists: under the default policy a model-chosen `music.play` is
    confirmed, and only the anchored fast-path grammar runs one directly.
    """

    name = "music"
    description = "Play and control music on Spotify"

    def __init__(
        self,
        api: SpotifyApi,
        *,
        device_name: str | None,
        clock: Clock,
        token_store: SpotifyTokenStore | None = None,
    ) -> None:
        """Build the capability; nothing is sent until a call.

        Args:
            api: The Web API client.
            device_name: `MIST_SPOTIFY_DEVICE`, the device playback moves to when none is
                active, matched ignoring case; None leaves such a call an error.
            clock: Times status changes and the recheck window.
            token_store: When given, `start` reads whether its file exists, so a missing
                authorization shows in the status before the first call.
        """
        self._api = api
        self._device_name = _clean(device_name)
        self._clock = clock
        self._token_store = token_store
        self._failure: _Failure | None = None
        self._tools = (
            ToolSpec(
                name=PLAY,
                description="play a song, album, artist or playlist; no query resumes",
                args_model=PlayArgs,
                risk="physical",
                default_policy="confirm_inferred",
                summarize=_summarize_play,
                ack="Finding that on Spotify.",
                view="music",
            ),
            ToolSpec(
                name=CONTROL,
                description="pause, resume, skip to next or go back to previous",
                args_model=ControlArgs,
                risk="reversible",
                default_policy="allow",
                summarize=_summarize_control,
                ack="One moment.",
                view="music",
            ),
            ToolSpec(
                name=VOLUME,
                description="set the volume to a level or turn it up or down",
                args_model=VolumeArgs,
                risk="reversible",
                default_policy="allow",
                summarize=_summarize_volume,
                ack="Changing the volume.",
                view="music",
            ),
            ToolSpec(
                name=NOW_PLAYING,
                description="say what is playing",
                args_model=NowPlayingArgs,
                risk="read",
                default_policy="allow",
                summarize=_summarize_now_playing,
                ack="Checking Spotify.",
                view="music",
            ),
        )

    def tools(self) -> Sequence[ToolSpec]:
        """Return `music.play`, `music.control`, `music.volume` and `music.now_playing`."""
        return self._tools

    def routing_hints(self) -> RoutingHints:
        """Return no keywords beyond the router's music floor, and a few exemplars."""
        return RoutingHints(
            keywords=(),
            exemplars=(
                "play some jazz",
                "pause the music",
                "turn the volume up",
                "what song is this",
            ),
        )

    async def start(self) -> None:
        """Note a missing token file, so the status says so before any call."""
        store = self._token_store
        if store is not None and not store.path.exists():
            detail = str(not_authorized(f"no Spotify token file at {store.path}"))
            self._failure = _Failure(detail, self._clock.now())

    async def stop(self) -> None:
        """Hold nothing to release; the HTTP client belongs to the caller."""

    def status(self) -> CapabilityStatus:
        """Return the status from the last call's outcome, with no network call."""
        now = self._clock.now()
        failure = self._failure
        if failure is None:
            return CapabilityStatus(self.name, "available", None, now)
        if now - failure.at < timedelta(seconds=RECHECK_AFTER_S):
            return CapabilityStatus(self.name, "unavailable", failure.reason, now)
        return CapabilityStatus(self.name, "degraded", failure.reason, now)

    def match_intent(self, text: str) -> FastIntent | None:
        """Match the music fast-path rows."""
        return match_music_intent(text)

    async def invoke(self, tool: str, args: BaseModel, ctx: InvocationContext) -> ToolOutcome:
        """Run the call against Spotify and render the outcome.

        Raises:
            ValueError: `tool` is not one of this capability's tools.
        """
        try:
            if isinstance(args, PlayArgs) and tool == PLAY:
                outcome = await self._play(args)
            elif isinstance(args, ControlArgs) and tool == CONTROL:
                outcome = await self._control(args.action, None)
            elif isinstance(args, VolumeArgs) and tool == VOLUME:
                outcome = await self._volume(args)
            elif isinstance(args, NowPlayingArgs) and tool == NOW_PLAYING:
                outcome = await self._now_playing()
            else:
                raise ValueError(f"music has no tool {tool!r} taking {type(args).__name__}")
        except SpotifyAuthError as exc:
            return self._outage(str(exc), _NOT_AUTHORIZED_SUMMARY)
        except SpotifyApiError as exc:
            if exc.reason in _OUTAGE_REASONS:
                return self._outage(exc.reason, _OUTAGE_SUMMARIES[exc.reason])
            self._failure = None
            return _error_outcome(exc)
        except SpotifyError as exc:
            self._failure = None
            return ToolOutcome(
                status="error",
                summary="Spotify did not accept that.",
                reason=str(exc) or type(exc).__name__,
            )
        self._failure = None
        return outcome

    # -- Tools -----------------------------------------------------------------------------

    async def _play(self, args: PlayArgs) -> ToolOutcome:
        device_id: str | None = None
        if args.device is not None:
            resolved = await self._resolve_device(args.device)
            if isinstance(resolved, ToolOutcome):
                return resolved
            device_id = resolved
        if args.query is None:
            return await self._control("resume", device_id)
        kinds: tuple[SearchKind, ...] = SEARCH_KINDS if args.kind is None else (args.kind,)
        results = await self._api.search(args.query, kinds=kinds, limit=SEARCH_LIMIT)
        item = _first_hit(results, kinds)
        if item is None:
            return ToolOutcome(
                status="invalid_args",
                summary=f"Nothing on Spotify matches {args.query}.",
                reason=NO_RESULTS,
            )

        async def play(target: str | None) -> None:
            if item.kind == "track":
                await self._api.play(uris=[item.uri], device_id=target)
            else:
                await self._api.play(context_uri=item.uri, device_id=target)

        await self._on_device(play, device_id)
        return ToolOutcome(
            status="ok",
            summary=_play_summary(item),
            data={
                "kind": item.kind,
                "uri": item.uri,
                "name": item.name,
                "artists": list(item.artists),
            },
        )

    async def _control(self, action: _Action, device_id: str | None) -> ToolOutcome:
        calls: dict[str, Callable[[str | None], Awaitable[None]]] = {
            "pause": lambda target: self._api.pause(device_id=target),
            "resume": lambda target: self._api.play(device_id=target),
            "next": lambda target: self._api.next(device_id=target),
            "previous": lambda target: self._api.previous(device_id=target),
        }
        await self._on_device(calls[action], device_id)
        summaries = {
            "pause": ok_summary("music.pause"),
            "resume": ok_summary("music.resume"),
            "next": "Skipping to the next track.",
            "previous": "Back to the previous track.",
        }
        return ToolOutcome(status="ok", summary=summaries[action], data={"action": action})

    async def _volume(self, args: VolumeArgs) -> ToolOutcome:
        change = args.change

        async def set_volume(target: str | None) -> int | None:
            if args.level_pct is not None:
                level = args.level_pct
            else:
                current = _current_volume(await self._api.playback_state())
                if current is None:
                    return None
                step = VOLUME_STEP_PCT if change == "up" else -VOLUME_STEP_PCT
                level = min(100, max(0, current + step))
            await self._api.set_volume(level, device_id=target)
            return level

        level = await self._on_device(set_volume, None)
        if level is None:
            return ToolOutcome(
                status="error",
                summary="Spotify does not report this device's volume.",
                reason="volume_unknown",
            )
        return ToolOutcome(
            status="ok",
            summary=ok_summary("music.volume", percent=level),
            data={"level_pct": level},
        )

    async def _now_playing(self) -> ToolOutcome:
        state = await self._api.playback_state()
        if state is None or state.item is None:
            return ToolOutcome(
                status="ok", summary=ok_summary("music.idle"), data={"is_playing": False}
            )
        item = state.item
        artist = item.artists[0] if item.artists else None
        if state.is_playing:
            title, fitted_artist = _fit(item.name, artist, prefix_words=1)
            summary = (
                render_read(NOW_PLAYING, [], title=title, artist=fitted_artist)
                if fitted_artist
                else f"Playing {title}."
            )
        else:
            title, fitted_artist = _fit(item.name, artist, prefix_words=2)
            by = f" by {fitted_artist}" if fitted_artist else ""
            summary = f"Paused on {title}{by}."
        return ToolOutcome(
            status="ok",
            summary=summary,
            data={
                "is_playing": state.is_playing,
                "name": item.name,
                "artists": list(item.artists),
                "album": item.album,
                "uri": item.uri,
                "device": state.device.name if state.device else None,
                "progress_ms": state.progress_ms,
            },
        )

    # -- Devices ---------------------------------------------------------------------------

    async def _on_device(
        self, call: Callable[[str | None], Awaitable[_T]], device_id: str | None
    ) -> _T:
        """Run `call`; on `no_active_device`, move playback to the fallback device once.

        A call that named its device is not moved, because the user chose that device.

        Raises:
            SpotifyApiError: The call failed in another way, the fallback device is unset or
                missing (`_NoFallbackDeviceError`), or the retry failed too.
        """
        try:
            return await call(device_id)
        except SpotifyApiError as exc:
            if exc.reason != NO_ACTIVE_DEVICE or device_id is not None:
                raise
        fallback = await self._fallback_device()
        await self._api.transfer(fallback, play=False)
        return await call(fallback)

    async def _fallback_device(self) -> str:
        if self._device_name is None:
            raise _NoFallbackDeviceError(None)
        device = _named(await self._api.devices(), self._device_name)
        if device is None or device.id is None:
            raise _NoFallbackDeviceError(self._device_name)
        return device.id

    async def _resolve_device(self, name: str) -> str | ToolOutcome:
        devices = await self._api.devices()
        device = _named(devices, name)
        if device is not None and device.id is not None:
            return device.id
        known = [d.name for d in devices if d.id is not None][:KNOWN_DEVICES_NAMED]
        listed = ", ".join(known) if known else "none"
        return ToolOutcome(
            status="invalid_args",
            summary=f"There is no Spotify device called {name}; known ones are {listed}.",
            reason=UNKNOWN_DEVICE,
            data={"known_devices": known},
        )

    # -- Failures --------------------------------------------------------------------------

    def _outage(self, reason: str, summary: str) -> ToolOutcome:
        self._failure = _Failure(reason, self._clock.now())
        return ToolOutcome(status="unavailable", summary=summary, reason=reason)


def _error_outcome(exc: SpotifyApiError) -> ToolOutcome:
    if isinstance(exc, _NoFallbackDeviceError):
        if exc.device_name is None:
            summary = "No Spotify device is active."
        else:
            summary = f"No Spotify device is active, and {exc.device_name} was not found."
        return ToolOutcome(status="error", summary=summary, reason=NO_ACTIVE_DEVICE)
    if exc.reason == NO_ACTIVE_DEVICE:
        return ToolOutcome(
            status="error", summary="No Spotify device is active.", reason=NO_ACTIVE_DEVICE
        )
    if isinstance(exc, SpotifyRateLimitedError):
        return ToolOutcome(
            status="error",
            summary="Spotify asked for a pause between requests.",
            reason=exc.reason,
            data=None if exc.retry_after_s is None else {"retry_after_s": exc.retry_after_s},
        )
    return ToolOutcome(status="error", summary="Spotify did not accept that.", reason=exc.reason)


def _first_hit(results: SearchResults, kinds: Sequence[SearchKind]) -> SearchItem | None:
    by_kind = {
        "track": results.tracks,
        "album": results.albums,
        "playlist": results.playlists,
        "artist": results.artists,
    }
    for kind in kinds:
        if by_kind[kind]:
            return by_kind[kind][0]
    return None


def _named(devices: Sequence[SpotifyDevice], name: str) -> SpotifyDevice | None:
    wanted = name.casefold()
    return next((d for d in devices if d.name.casefold() == wanted), None)


def _current_volume(state: PlaybackState | None) -> int | None:
    """Return the active device's volume.

    Raises:
        SpotifyApiError: No device is active, so `_on_device` can move playback and retry.
    """
    if state is None or state.device is None:
        raise SpotifyApiError(404, NO_ACTIVE_DEVICE)
    return state.device.volume_percent


# -- Summaries ---------------------------------------------------------------------------------


def _speakable(text: str) -> str:
    # A full stop, question mark or exclamation mark inside a title would end the spoken
    # sentence early ("Mr. Brightside"), so titles lose them.
    kept = "".join(" " if char in ".?!" else char for char in text)
    return " ".join(kept.split())


def _fit(title: str, artist: str | None, *, prefix_words: int) -> tuple[str, str | None]:
    """Return the title and artist that fit one spoken sentence after `prefix_words`.

    The artist goes first when the whole would pass `SPOKEN_WORDS_MAX`; a title alone that
    still would is cut, since the full name is in the outcome's data for the screen.
    """
    words = (_speakable(title) or "that").split()
    budget = SPOKEN_WORDS_MAX - prefix_words
    spoken_artist = _speakable(artist) if artist else ""
    if spoken_artist and len(words) + 1 + len(spoken_artist.split()) <= budget:
        return " ".join(words), spoken_artist
    return " ".join(words[:budget]), None


def _play_summary(item: SearchItem) -> str:
    if item.kind == "playlist":
        title, _ = _fit(item.name, None, prefix_words=3)
        return f"Playing the playlist {title}."
    artist = item.artists[0] if item.kind in ("track", "album") and item.artists else None
    title, fitted_artist = _fit(item.name, artist, prefix_words=1)
    if fitted_artist:
        return ok_summary("music.play", title=title, artist=fitted_artist)
    return f"Playing {title}."


def _expect(args: BaseModel, model: type[_Args]) -> _Args:
    # Each spec pairs its summarize hook with its own args model, so a mismatch is a bug.
    if not isinstance(args, model):
        raise TypeError(f"expected {model.__name__}, got {type(args).__name__}")
    return args


def _summarize_play(args: BaseModel) -> str:
    play = _expect(args, PlayArgs)
    on = f" on {play.device}" if play.device else ""
    if play.query is None:
        return f"resume the music{on}"
    kind = f"the {play.kind} " if play.kind else ""
    return f"play {kind}{play.query}{on}"


def _summarize_control(args: BaseModel) -> str:
    return {
        "pause": "pause the music",
        "resume": "resume the music",
        "next": "skip to the next track",
        "previous": "go back a track",
    }[_expect(args, ControlArgs).action]


def _summarize_volume(args: BaseModel) -> str:
    volume = _expect(args, VolumeArgs)
    if volume.level_pct is not None:
        return f"set the volume to {volume.level_pct} percent"
    return f"turn the volume {volume.change}"


def _summarize_now_playing(args: BaseModel) -> str:
    _expect(args, NowPlayingArgs)
    return "check what is playing"
