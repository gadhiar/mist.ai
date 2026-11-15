"""Configuration for the memory read path, nested as `KnowledgeConfig.memory`.

Every field maps to one variable documented in `.env.example`: a `MIST_MEMORY_*` one, or for
the zone `MIST_TIMEZONE`, which usage counting shares through `backend.user_timezone`.
Validation runs in `__post_init__` and raises MemoryConfigError naming the field, so a bad value
stops startup instead of degrading recall quietly.
"""

from __future__ import annotations

import ipaddress
import math
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, tzinfo
from typing import Literal

import httpx

from backend.errors import MemoryConfigError
from backend.user_timezone import USER_TIMEZONE_ENV, ZoneLoader, resolve_user_timezone

from .models import ServingProfile
from .retrieval_view import FACT_VIEWS

FactViewId = Literal["F0", "F1", "F2", "F3"]
RerankBackend = Literal["local", "remote", "off"]
RerankDevice = Literal["cpu", "cuda"]
RemoteScoreKind = Literal["logit", "probability"]
TzResolver = ZoneLoader

_RERANK_BACKENDS: tuple[str, ...] = ("local", "remote", "off")
_RERANK_DEVICES: tuple[str, ...] = ("cpu", "cuda")
_REMOTE_SCORE_KINDS: tuple[str, ...] = ("logit", "probability")
_WEIGHT_TOLERANCE = 1e-6
_BLOCK_TOKENS_LOW_RANGE = (0, 300)
_LIST_MAX_ITEMS_RANGE = (2, 25)
_MIN_RESCAN_INTERVAL_S = 30

# Where `rerank_url` may point. The remote reranker is sent the query and the memory texts, so
# its host must be on the local network or the tailnet: loopback, RFC 1918, and CGNAT
# 100.64.0.0/10, which Tailscale assigns its nodes from.
_PRIVATE_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = tuple(
    ipaddress.ip_network(network)
    for network in (
        "127.0.0.0/8",
        "::1/128",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
    )
)
_PRIVATE_SUFFIXES: tuple[str, ...] = (".ts.net", ".lan", ".local", ".internal")
_RERANK_URL_SCHEMES: tuple[str, ...] = ("http", "https")
_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})

# Retired variables and what replaced each. One that is still set fails startup rather than
# being ignored, so a deployment that relied on it learns where its setting went.
RETIRED_MEMORY_ENV_VARS: Mapping[str, str] = {
    "MIST_MEMORY_BLOCK_ROLE": "MIST_CTX_BUDGET_DELTA_PLACEMENT",
    "MIST_MEMORY_TZ": USER_TIMEZONE_ENV,
}

# Phase A variables, in `.env.example` order. Tests clear these to isolate a run, and the
# `.env.example` check walks this tuple, so a new variable added here must be documented there.
# `MIST_TIMEZONE` is shared with usage counting and is documented once, in its own section.
MEMORY_ENV_VARS: tuple[str, ...] = (
    "MIST_MEMORY_ENABLED",
    "MIST_MEMORY_BLOCK_TOKENS",
    "MIST_MEMORY_CONTINUITY_SHARE",
    "MIST_MEMORY_MIN_SCORE",
    "MIST_MEMORY_DEADLINE_MS",
    "MIST_MEMORY_RERANK_BACKEND",
    "MIST_MEMORY_RERANK_MODEL",
    "MIST_MEMORY_RERANK_TOP_N",
    "MIST_MEMORY_RERANK_DEVICE",
    "MIST_MEMORY_RERANK_URL",
    "MIST_MEMORY_RERANK_REMOTE_MODEL",
    "MIST_MEMORY_RERANK_TIMEOUT_MS",
    "MIST_MEMORY_RERANK_REMOTE_SCORES",
    "MIST_MEMORY_WEIGHTS",
    "MIST_MEMORY_INDEX_PATH",
    "MIST_MEMORY_REFRESH_DEBOUNCE_S",
    "MIST_MEMORY_BLOCK_TOKENS_LOW",
    "MIST_MEMORY_LIST_MAX_ITEMS",
    "MIST_MEMORY_RESCAN_INTERVAL_S",
    "MIST_MEMORY_FACT_VIEW",
    USER_TIMEZONE_ENV,
)


def _env(name: str) -> str | None:
    """The stripped value of `name`, or None when unset or blank.

    Compose forwards `${VAR:-}` as an empty string, and an empty value must mean the default
    rather than fail a parse.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise MemoryConfigError(f"{name}={raw!r} is not a boolean (use true or false)")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise MemoryConfigError(f"{name}={raw!r} is not an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise MemoryConfigError(f"{name}={raw!r} is not a number") from exc


def _env_weights(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = _env(name)
    if raw is None:
        return default
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise MemoryConfigError(
            f"{name}={raw!r} needs three comma-separated weights (rerank, recency, confidence)"
        )
    try:
        rerank, recency, confidence = (float(part) for part in parts)
    except ValueError as exc:
        raise MemoryConfigError(f"{name}={raw!r} holds a weight that is not a number") from exc
    return rerank, recency, confidence


def _reject_retired() -> None:
    """Fail on a retired variable that is still set, naming the variable that replaced it.

    Blank counts as unset, as everywhere here: Compose forwards `${VAR:-}` as an empty string.
    """
    for name, replacement in RETIRED_MEMORY_ENV_VARS.items():
        if _env(name) is not None:
            raise MemoryConfigError(f"{name} is retired; set {replacement} instead")


def _resolve_tz(load_zone: TzResolver | None) -> tuple[tzinfo, str | None]:
    """Resolve `MIST_TIMEZONE` once, at config time, through the shared user-zone resolver.

    Unset gives UTC with the resolver's one warning per process: startup builds
    `KnowledgeConfig` several times, and the Neo4j and event-store health probes rebuild it on
    every 10 s tick while the graph connection or the event store was never built (server.py
    `_knowledge_configured` and `_event_store_configured`, called from health.py's probes when
    their source is None). A name the resolver rejects fails startup: answering "yesterday" in
    the wrong zone is worse than refusing to start.

    Returns:
        The zone, and the name it came from, or None when the zone is UTC because the variable
        is unset or no zone database is installed.
    """
    name = _env(USER_TIMEZONE_ENV)
    try:
        zone = resolve_user_timezone(os.environ, load_zone=load_zone)
    except ValueError as exc:
        raise MemoryConfigError(
            f"{USER_TIMEZONE_ENV}={name!r} does not resolve to a time zone: {exc}"
        ) from exc
    return zone, (None if zone is UTC else name)


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Phase A settings of the memory read path.

    Attributes:
        enabled: False restores the legacy vault-only auto-inject.
        block_tokens: Token budget of the memory block; a request can lower it per turn.
        continuity_share: Share of the budget reserved for episodes and exchanges when the
            turn has a temporal window or starts a conversation.
        min_score: Items below this final score are dropped. 0.35 is a placeholder until the
            fixture calibration.
        deadline_ms: Critical-path budget of `fetch`, counted from the call and covering any
            wait on stage 1.
        rerank_backend: `local` cross-encoder, `remote` HTTP endpoint, or `off`.
        rerank_model: Local cross-encoder path; used by `local` only.
        rerank_top_n: Ceiling on reranked pairs; the pairs scored are those that fit the
            deadline.
        rerank_device: Device for the local cross-encoder.
        rerank_url: Base URL of the rerank server; required for `remote`. Its scheme must be
            http or https and its host private (`_require_private_rerank_url`), since the
            server is sent the query and the memory texts.
        rerank_remote_model: Model name sent to the remote server, if it needs one.
        rerank_timeout_ms: Remote call timeout, capped at the deadline left when it starts.
        rerank_remote_scores: Whether the remote server returns logits (a sigmoid is applied)
            or probabilities.
        rerank_weight: Weight of the rerank score in the final score.
        recency_weight: Weight of recency (or the window-match term) in the final score.
        confidence_weight: Weight of confidence in the final score.
        index_path: Path of `memory_index.db`; None means beside the event store, resolved by
            the factory, which also propagates `:memory:`.
        refresh_debounce_s: Minimum seconds between two index refreshes.
        block_tokens_low: Block budget at the low serving level, 0 to 300; 0 sends no block.
            `block_tokens` is the budget at full, and off gets none.
        list_max_items: Most siblings one list item can bring into the block, 2 to 25.
        rescan_interval_s: Seconds between two rescans of the graph for changed facts, at
            least 30.
        fact_view: The fact view the index embeds, F0 to F3 (`retrieval_view.FACT_VIEWS`).
            The index's content hash covers the view text, so a FactIndex built with a changed
            setting rewrites and re-embeds, at its next diff, every row whose view text the
            change alters. The default equals `retrieval_view.DEFAULT_FACT_VIEW`.
        tz: Zone used to read temporal phrases, from `MIST_TIMEZONE`. Resolved once in
            `from_env`.
        tz_name: The IANA name `tz` came from, or None when `tz` is UTC because
            `MIST_TIMEZONE` is unset or no zone database is installed; the trace records which.
    """

    enabled: bool = True
    block_tokens: int = 500
    continuity_share: float = 0.35
    min_score: float = 0.35
    deadline_ms: int = 50
    rerank_backend: RerankBackend = "local"
    rerank_model: str = "/opt/models/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 20
    rerank_device: RerankDevice = "cpu"
    rerank_url: str | None = None
    rerank_remote_model: str | None = None
    rerank_timeout_ms: int = 40
    rerank_remote_scores: RemoteScoreKind = "logit"
    rerank_weight: float = 0.8
    recency_weight: float = 0.1
    confidence_weight: float = 0.1
    index_path: str | None = None
    refresh_debounce_s: int = 30
    block_tokens_low: int = 0
    list_max_items: int = 10
    rescan_interval_s: int = 300
    fact_view: FactViewId = "F3"
    tz: tzinfo = UTC
    tz_name: str | None = None

    def __post_init__(self) -> None:
        """Reject values the read path cannot run with, naming the field."""
        if self.block_tokens <= 0:
            raise MemoryConfigError(f"block_tokens must be positive, got {self.block_tokens}")
        if not 0.0 <= self.continuity_share <= 1.0:
            raise MemoryConfigError(
                f"continuity_share must be in [0, 1], got {self.continuity_share}"
            )
        if not 0.0 <= self.min_score <= 1.0:
            raise MemoryConfigError(f"min_score must be in [0, 1], got {self.min_score}")
        if self.deadline_ms <= 0:
            raise MemoryConfigError(f"deadline_ms must be positive, got {self.deadline_ms}")
        if self.rerank_top_n <= 0:
            raise MemoryConfigError(f"rerank_top_n must be positive, got {self.rerank_top_n}")
        if self.refresh_debounce_s < 0:
            raise MemoryConfigError(
                f"refresh_debounce_s must not be negative, got {self.refresh_debounce_s}"
            )
        _require_range("block_tokens_low", self.block_tokens_low, _BLOCK_TOKENS_LOW_RANGE)
        _require_range("list_max_items", self.list_max_items, _LIST_MAX_ITEMS_RANGE)
        if self.rescan_interval_s < _MIN_RESCAN_INTERVAL_S:
            raise MemoryConfigError(
                f"rescan_interval_s must be at least {_MIN_RESCAN_INTERVAL_S}, "
                f"got {self.rescan_interval_s}"
            )
        _require_choice("rerank_backend", self.rerank_backend, _RERANK_BACKENDS)
        _require_choice("rerank_device", self.rerank_device, _RERANK_DEVICES)
        _require_choice("rerank_remote_scores", self.rerank_remote_scores, _REMOTE_SCORE_KINDS)
        _require_choice("fact_view", self.fact_view, tuple(FACT_VIEWS))
        weights = (self.rerank_weight, self.recency_weight, self.confidence_weight)
        if any(weight < 0 for weight in weights):
            raise MemoryConfigError(f"weights must not be negative, got {weights}")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=_WEIGHT_TOLERANCE):
            raise MemoryConfigError(f"weights must sum to 1, got {weights} (sum {sum(weights)})")
        if self.rerank_backend == "remote":
            if not self.rerank_url:
                raise MemoryConfigError("rerank_url is required when rerank_backend is remote")
            if self.rerank_timeout_ms <= 0:
                raise MemoryConfigError(
                    f"rerank_timeout_ms must be positive for remote, got {self.rerank_timeout_ms}"
                )
        # Checked whatever the backend, so a public URL fails when it is set rather than on
        # the day the backend switches to remote.
        if self.rerank_url:
            _require_private_rerank_url(self.rerank_url)
        if not isinstance(self.tz, tzinfo):
            raise MemoryConfigError(f"tz must be a tzinfo, got {type(self.tz).__name__}")

    @classmethod
    def from_env(cls, *, tz_resolver: TzResolver | None = None) -> MemoryConfig:
        """Load the Phase A settings from `MIST_MEMORY_*` variables and `MIST_TIMEZONE`.

        Args:
            tz_resolver: Replaces the zone lookup inside `resolve_user_timezone`, which turns
                an IANA name into a tzinfo; None keeps the real lookup. Tests pass a fake.

        Raises:
            MemoryConfigError: A retired variable is set, a variable does not parse, a value
                fails validation, or `MIST_TIMEZONE` names a zone the resolver rejects.
        """
        _reject_retired()
        rerank_weight, recency_weight, confidence_weight = _env_weights(
            "MIST_MEMORY_WEIGHTS", (0.8, 0.1, 0.1)
        )
        tz, tz_name = _resolve_tz(tz_resolver)
        return cls(
            enabled=_env_bool("MIST_MEMORY_ENABLED", True),
            block_tokens=_env_int("MIST_MEMORY_BLOCK_TOKENS", 500),
            continuity_share=_env_float("MIST_MEMORY_CONTINUITY_SHARE", 0.35),
            min_score=_env_float("MIST_MEMORY_MIN_SCORE", 0.35),
            deadline_ms=_env_int("MIST_MEMORY_DEADLINE_MS", 50),
            rerank_backend=_env("MIST_MEMORY_RERANK_BACKEND") or "local",  # type: ignore[arg-type]
            rerank_model=_env("MIST_MEMORY_RERANK_MODEL") or "/opt/models/ms-marco-MiniLM-L-6-v2",
            rerank_top_n=_env_int("MIST_MEMORY_RERANK_TOP_N", 20),
            rerank_device=_env("MIST_MEMORY_RERANK_DEVICE") or "cpu",  # type: ignore[arg-type]
            rerank_url=_env("MIST_MEMORY_RERANK_URL"),
            rerank_remote_model=_env("MIST_MEMORY_RERANK_REMOTE_MODEL"),
            rerank_timeout_ms=_env_int("MIST_MEMORY_RERANK_TIMEOUT_MS", 40),
            rerank_remote_scores=_env("MIST_MEMORY_RERANK_REMOTE_SCORES")  # type: ignore[arg-type]
            or "logit",
            rerank_weight=rerank_weight,
            recency_weight=recency_weight,
            confidence_weight=confidence_weight,
            index_path=_env("MIST_MEMORY_INDEX_PATH"),
            refresh_debounce_s=_env_int("MIST_MEMORY_REFRESH_DEBOUNCE_S", 30),
            block_tokens_low=_env_int("MIST_MEMORY_BLOCK_TOKENS_LOW", 0),
            list_max_items=_env_int("MIST_MEMORY_LIST_MAX_ITEMS", 10),
            rescan_interval_s=_env_int("MIST_MEMORY_RESCAN_INTERVAL_S", 300),
            fact_view=_env("MIST_MEMORY_FACT_VIEW") or "F3",  # type: ignore[arg-type]
            tz=tz,
            tz_name=tz_name,
        )

    def block_tokens_for(self, profile: ServingProfile) -> int:
        """The block budget at `profile`'s level: full, low, or 0 at off.

        The tier does not enter, because one tier can run at more than one level and so at
        more than one prefill rate.
        """
        if profile.level == "full":
            return self.block_tokens
        if profile.level == "low":
            return self.block_tokens_low
        return 0


def _require_range(field_name: str, value: int, bounds: tuple[int, int]) -> None:
    low, high = bounds
    if not low <= value <= high:
        raise MemoryConfigError(f"{field_name} must be in [{low}, {high}], got {value}")


def _require_choice(field_name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise MemoryConfigError(f"{field_name} must be one of {choices}, got {value!r}")


def _require_private_rerank_url(url: str) -> None:
    """Refuse a rerank URL whose scheme is not http or https or whose host is not private.

    The URL is parsed by httpx, the client that will send the request, so the host checked is
    the host connected to; a second parser could read an ambiguous URL differently. Messages
    name the host and never the whole URL, which can carry credentials.

    Raises:
        MemoryConfigError: The URL does not parse, its scheme is not http or https, it names no
            host, or its host is not one `_is_private_host` accepts.
    """
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL as exc:
        raise MemoryConfigError(f"rerank_url is not a valid URL: {exc}") from exc
    host = parsed.host
    if parsed.scheme not in _RERANK_URL_SCHEMES:
        raise MemoryConfigError(
            f"rerank_url scheme {parsed.scheme!r} for host {host!r} must be http or https"
        )
    if not host:
        raise MemoryConfigError("rerank_url names no host")
    if not _is_private_host(host):
        raise MemoryConfigError(
            f"rerank_url host {host!r} is not private: use a loopback, RFC 1918 or CGNAT "
            "(100.64.0.0/10) address, localhost, a single-label name, or a name ending .ts.net, "
            ".lan, .local or .internal"
        )


def _is_private_host(host: str) -> bool:
    """Whether `host`, lowercased as httpx gives it, is on the local network or the tailnet.

    An address must fall in `_PRIVATE_NETWORKS`. A name passes when it is `localhost`, a single
    label (resolved through the local search domains), or ends in one of `_PRIVATE_SUFFIXES`.
    A trailing dot marks an absolute name: it is dropped before the suffix check, but a single
    label written with one is a top-level domain, resolved publicly, and fails.
    """
    address = _host_address(host)
    if address is not None:
        return any(address in network for network in _PRIVATE_NETWORKS)
    name = host[:-1] if host.endswith(".") else host
    if name == "localhost":
        return True
    if "." not in name:
        return not host.endswith(".")
    return name.endswith(_PRIVATE_SUFFIXES)


def _host_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The IP address `host` spells, or None when it is a name.

    Beyond the standard forms this reads what the system resolver also takes as IPv4: a single
    number or hex (`134744072`, `0x08080808`), which would otherwise pass as a single-label
    name. An IPv4-mapped IPv6 address is read as the IPv4 address it carries.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if not host.isascii():
            return None
        try:
            return ipaddress.IPv4Address(socket.inet_aton(host))
        except OSError:
            return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address
