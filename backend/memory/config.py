"""Configuration for the memory read path, nested as `KnowledgeConfig.memory`.

Every field maps to one `MIST_MEMORY_*` variable documented in `.env.example`. Validation runs
in `__post_init__` and raises MemoryConfigError naming the field, so a bad value stops startup
instead of degrading recall quietly.
"""

from __future__ import annotations

import logging
import math
import os
import zoneinfo
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, tzinfo
from typing import Literal

from backend.errors import MemoryConfigError

logger = logging.getLogger(__name__)

BlockRole = Literal["user_prefix", "system_tail"]
RerankBackend = Literal["local", "remote", "off"]
RerankDevice = Literal["cpu", "cuda"]
RemoteScoreKind = Literal["logit", "probability"]
TzResolver = Callable[[str], tzinfo]

_BLOCK_ROLES: tuple[str, ...] = ("user_prefix", "system_tail")
_RERANK_BACKENDS: tuple[str, ...] = ("local", "remote", "off")
_RERANK_DEVICES: tuple[str, ...] = ("cpu", "cuda")
_REMOTE_SCORE_KINDS: tuple[str, ...] = ("logit", "probability")
_WEIGHT_TOLERANCE = 1e-6
_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})

# Set once the unset-zone warning has been logged; see `_resolve_tz`. Tests reset it with
# monkeypatch so each test sees a fresh process.
_tz_unset_warned = False

# Phase A variables, in `.env.example` order. Tests clear these to isolate a run, and the
# `.env.example` check walks this tuple, so a new variable added here must be documented there.
MEMORY_ENV_VARS: tuple[str, ...] = (
    "MIST_MEMORY_ENABLED",
    "MIST_MEMORY_BLOCK_TOKENS",
    "MIST_MEMORY_BLOCK_ROLE",
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
    "MIST_MEMORY_TZ",
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


def _resolve_tz(resolver: TzResolver) -> tuple[tzinfo, str | None]:
    """Resolve `MIST_MEMORY_TZ` once, at config time.

    An unset zone maps to `datetime.timezone.utc` without calling the resolver, because
    `zoneinfo.ZoneInfo("UTC")` itself needs a tz database, and the current image has none
    (Design point 5). A name the resolver rejects fails startup: answering "yesterday" in the
    wrong zone is worse than refusing to start.

    The unset warning goes out once per process. Startup builds `KnowledgeConfig` several times,
    and the Neo4j and event-store health probes rebuild it on every 10 s tick while the graph
    connection or the event store was never built, whether it failed at boot or is switched off
    (server.py `_knowledge_configured` and `_event_store_configured`, called from health.py's
    probes when their source is None). Warning on every call would repeat the same line without
    adding anything.
    """
    global _tz_unset_warned
    name = _env("MIST_MEMORY_TZ")
    if name is None:
        if not _tz_unset_warned:
            _tz_unset_warned = True
            logger.warning(
                "MIST_MEMORY_TZ is unset; memory read path resolves dates such as 'yesterday' "
                "in UTC"
            )
        return UTC, None
    try:
        return resolver(name), name
    except (KeyError, ValueError) as exc:
        # ZoneInfoNotFoundError subclasses KeyError; a malformed key raises ValueError.
        raise MemoryConfigError(
            f"MIST_MEMORY_TZ={name!r} does not resolve to a time zone: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Phase A settings of the memory read path.

    Attributes:
        enabled: False restores the legacy vault-only auto-inject.
        block_tokens: Token budget of the memory block; a request can lower it per turn.
        block_role: Where the block goes: prepended to the last user message
            (`user_prefix`) or as a system message directly before it (`system_tail`).
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
        rerank_url: Base URL of the rerank server; required for `remote`.
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
        tz: Zone used to read temporal phrases. Resolved once in `from_env`.
        tz_name: The IANA name `tz` came from, or None when `MIST_MEMORY_TZ` is unset and `tz`
            is UTC; the trace records which.
    """

    enabled: bool = True
    block_tokens: int = 500
    block_role: BlockRole = "user_prefix"
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
        _require_choice("block_role", self.block_role, _BLOCK_ROLES)
        _require_choice("rerank_backend", self.rerank_backend, _RERANK_BACKENDS)
        _require_choice("rerank_device", self.rerank_device, _RERANK_DEVICES)
        _require_choice("rerank_remote_scores", self.rerank_remote_scores, _REMOTE_SCORE_KINDS)
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
        if not isinstance(self.tz, tzinfo):
            raise MemoryConfigError(f"tz must be a tzinfo, got {type(self.tz).__name__}")

    @classmethod
    def from_env(cls, *, tz_resolver: TzResolver = zoneinfo.ZoneInfo) -> MemoryConfig:
        """Load the Phase A settings from `MIST_MEMORY_*` variables.

        Args:
            tz_resolver: Turns an IANA name into a tzinfo. Production uses
                `zoneinfo.ZoneInfo`; tests pass a fake so they need no tz database.

        Raises:
            MemoryConfigError: A variable does not parse, a value fails validation, or
                `MIST_MEMORY_TZ` names a zone the resolver rejects.
        """
        rerank_weight, recency_weight, confidence_weight = _env_weights(
            "MIST_MEMORY_WEIGHTS", (0.8, 0.1, 0.1)
        )
        tz, tz_name = _resolve_tz(tz_resolver)
        return cls(
            enabled=_env_bool("MIST_MEMORY_ENABLED", True),
            block_tokens=_env_int("MIST_MEMORY_BLOCK_TOKENS", 500),
            block_role=_env("MIST_MEMORY_BLOCK_ROLE") or "user_prefix",  # type: ignore[arg-type]
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
            tz=tz,
            tz_name=tz_name,
        )


def _require_choice(field_name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise MemoryConfigError(f"{field_name} must be one of {choices}, got {value!r}")
