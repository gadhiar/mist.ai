"""Capability settings, parsed once from the environment into frozen dataclasses.

Parsing takes the environment as a mapping, never `os.environ` directly, so the caller decides
what is read and tests pass a plain dict. A blank value counts as unset, because compose
interpolation of an unset variable yields an empty string. Secret values are not read here:
`EnvSecretSource` reads them, and this module records only whether one is configured.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from backend.errors import CapabilityConfigError

KNOWN_CAPABILITIES = ("clock", "alarms", "timers", "lights", "music")

_TRUE_WORDS = frozenset({"1", "true", "yes", "on"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True, slots=True)
class RouterSettings:
    """Router stickiness and the embedding stage's threshold (None keeps the stage off)."""

    sticky_turns: int
    embedding_threshold: float | None


@dataclass(frozen=True, slots=True)
class AlarmSettings:
    """Ring timeout, default snooze and the late-ring grace window."""

    ring_timeout_s: float
    snooze_minutes: int
    missed_grace_s: float


@dataclass(frozen=True, slots=True)
class LightsSettings:
    """Home Assistant MCP endpoint and the default light target.

    `ha_token_configured` says whether `MIST_HA_TOKEN` or `MIST_HA_TOKEN_FILE` is set, so a
    caller can name the missing variable without this object ever holding the token.
    """

    ha_mcp_url: str | None
    ha_token_configured: bool
    default_target: str | None


@dataclass(frozen=True, slots=True)
class MusicSettings:
    """Spotify PKCE client, redirect URI, token file and fallback device."""

    spotify_client_id: str | None
    spotify_redirect_uri: str
    spotify_token_path: str
    spotify_device: str | None


@dataclass(frozen=True, slots=True)
class CapabilitiesConfig:
    """Every capability environment variable, parsed and validated."""

    enabled: bool
    capabilities: tuple[str, ...]
    ws_allowed_origins: tuple[str, ...]
    backend_bind_addr: str
    timezone: str | None
    db_path: str
    policy_file: str | None
    tool_timeout_s: float
    proposal_ttl_s: float
    tool_token_budget: int
    fast_path: bool
    decode_timeout_s: float
    wake_words: tuple[str, ...]
    router: RouterSettings
    alarms: AlarmSettings
    lights: LightsSettings
    music: MusicSettings


def load_capabilities_config(env: Mapping[str, str]) -> CapabilitiesConfig:
    """Parse the capability variables from `env`.

    Args:
        env: The environment to read, usually `os.environ` passed in by the caller.

    Returns:
        The parsed configuration; unset or blank variables take their documented defaults.

    Raises:
        CapabilityConfigError: A value does not parse or is out of range, or
            `MIST_CAPABILITIES` names an unknown capability. The message names the variable.
    """
    return CapabilitiesConfig(
        enabled=_bool(env, "MIST_CAPABILITIES_ENABLED", default=False),
        capabilities=_capabilities(env),
        ws_allowed_origins=_list(env, "MIST_WS_ALLOWED_ORIGINS", default=()),
        # Compose binds port 8001 on this address; the value is only parsed here.
        backend_bind_addr=_text(env, "MIST_BACKEND_BIND_ADDR") or "0.0.0.0",  # nosec B104
        timezone=_text(env, "MIST_TIMEZONE"),
        db_path=_text(env, "MIST_CAPABILITIES_DB_PATH") or "data/capabilities.db",
        policy_file=_text(env, "MIST_CAPABILITY_POLICY_FILE"),
        tool_timeout_s=_seconds(env, "MIST_CAPABILITY_TOOL_TIMEOUT_S", default=8.0),
        proposal_ttl_s=_seconds(env, "MIST_CAPABILITY_PROPOSAL_TTL_S", default=60.0),
        tool_token_budget=_int(env, "MIST_CAPABILITY_TOOL_TOKEN_BUDGET", default=1200, low=1),
        fast_path=_bool(env, "MIST_CAPABILITY_FAST_PATH", default=True),
        decode_timeout_s=_seconds(env, "MIST_CAPABILITY_DECODE_TIMEOUT_S", default=10.0),
        wake_words=_list(
            env, "MIST_CAPABILITY_WAKE_WORDS", default=("mist", "hey mist", "ok mist", "okay mist")
        ),
        router=RouterSettings(
            sticky_turns=_int(env, "MIST_CAPABILITY_ROUTER_STICKY_TURNS", default=2, low=0),
            embedding_threshold=_threshold(env, "MIST_CAPABILITY_ROUTER_EMBEDDING_THRESHOLD"),
        ),
        alarms=AlarmSettings(
            ring_timeout_s=_seconds(env, "MIST_ALARM_RING_TIMEOUT_S", default=600.0),
            snooze_minutes=_int(env, "MIST_ALARM_SNOOZE_MINUTES", default=9, low=1, high=60),
            missed_grace_s=_seconds(
                env, "MIST_ALARM_MISSED_GRACE_S", default=900.0, allow_zero=True
            ),
        ),
        lights=LightsSettings(
            ha_mcp_url=_text(env, "MIST_HA_MCP_URL"),
            ha_token_configured=bool(
                _text(env, "MIST_HA_TOKEN") or _text(env, "MIST_HA_TOKEN_FILE")
            ),
            default_target=_text(env, "MIST_LIGHTS_DEFAULT_TARGET"),
        ),
        music=MusicSettings(
            spotify_client_id=_text(env, "MIST_SPOTIFY_CLIENT_ID"),
            spotify_redirect_uri=_text(env, "MIST_SPOTIFY_REDIRECT_URI")
            or "http://127.0.0.1:8888/callback",
            spotify_token_path=_text(env, "MIST_SPOTIFY_TOKEN_PATH")
            or "data/secrets/spotify_token.json",
            spotify_device=_text(env, "MIST_SPOTIFY_DEVICE"),
        ),
    )


def _text(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name, "").strip()
    return value or None


def _list(env: Mapping[str, str], name: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _text(env, name)
    if raw is None:
        return default
    return tuple(entry.strip() for entry in raw.split(",") if entry.strip())


def _capabilities(env: Mapping[str, str]) -> tuple[str, ...]:
    names = _list(env, "MIST_CAPABILITIES", default=KNOWN_CAPABILITIES)
    unknown = sorted(set(names) - set(KNOWN_CAPABILITIES))
    if unknown:
        raise CapabilityConfigError(
            f"MIST_CAPABILITIES names unknown capabilities {unknown}; "
            f"known: {', '.join(KNOWN_CAPABILITIES)}"
        )
    return tuple(dict.fromkeys(names))


def _bool(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = _text(env, name)
    if raw is None:
        return default
    word = raw.casefold()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise CapabilityConfigError(
        f"{name}={raw!r} is not a boolean (true/false, 1/0, yes/no, on/off)"
    )


def _int(
    env: Mapping[str, str], name: str, *, default: int, low: int, high: int | None = None
) -> int:
    raw = _text(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise CapabilityConfigError(f"{name}={raw!r} is not an integer") from exc
    if value < low or (high is not None and value > high):
        bound = f"{low}..{high}" if high is not None else f">= {low}"
        raise CapabilityConfigError(f"{name}={value} is out of range ({bound})")
    return value


def _seconds(
    env: Mapping[str, str], name: str, *, default: float, allow_zero: bool = False
) -> float:
    raw = _text(env, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise CapabilityConfigError(f"{name}={raw!r} is not a number of seconds") from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise CapabilityConfigError(f"{name}={raw!r} must be a finite number {bound}")
    return value


def _threshold(env: Mapping[str, str], name: str) -> float | None:
    raw = _text(env, name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise CapabilityConfigError(f"{name}={raw!r} is not a number") from exc
    if not -1.0 <= value <= 1.0:
        raise CapabilityConfigError(f"{name}={raw!r} is not a cosine similarity (-1..1)")
    return value
