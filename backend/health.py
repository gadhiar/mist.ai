"""Dependency health probes and the cache the `/health` endpoint renders.

Why a background loop instead of probing in the request
-------------------------------------------------------
`GraphDatabase.driver(uri, auth=...)` is built with no timeouts
(`backend/knowledge/storage/neo4j_connection.py:44-46`; `KNOWN_ISSUES.md:331`
records the missing `max_connection_pool_size`, timeouts and retry settings), so
a driver call can block for the default 60s acquisition timeout. An endpoint
that probed in-request would inherit that. This module owns a loop that probes
on its own cadence and writes a cache; the endpoint renders the cache and does
no I/O.

Single-flight, because a timeout does not stop the work
-------------------------------------------------------
`asyncio.wait_for` abandons what it was waiting on rather than stopping it: a
`run_in_executor` call whose thread is wedged keeps that thread. A hung
dependency probed every tick would leak one worker per tick into the default
executor and eventually starve the voice pipeline's own `run_in_executor` calls
(`backend/voice_processor.py:178`). So the sync graph probe runs in an injected
`ThreadPoolExecutor(max_workers=1)`, and the registry tracks in-flight probes:
a probe whose previous invocation has not returned is SKIPPED this tick. Its
cached result ages and goes stale, which is the honest report.

Two statuses, not one
---------------------
`status` answers "can MIST serve?"; `restart_recommended` answers "would
restarting help?". Conflating them is how healthchecks thrash. Note that
`restart_recommended` has no consumer today: Docker restart policies react to
container exit, not to healthcheck state, and there is no autoheal sidecar. It
is advisory output for a human or a future supervisor.

Null means NOT MEASURED
-----------------------
A dependency that was not probed reports `status: null` with a reason, never
`up`. `age_seconds` and `stale` are rendered at read time from the injected
clock, not stored, so a snapshot taken long after the last probe reports stale
without anything having run.

The numbers, and why each is what it is
---------------------------------------
- `interval_seconds = 10.0`. Docker reads `/health` every 30s (the
  `mist-backend` healthcheck in `docker-compose.yml`, `interval: 30s`), so a
  10s loop keeps the cache at most 10s old at read time and gives roughly three
  independent samples per `retries: 3` window. A 30s loop would align with
  Docker's sampling and let one unlucky probe drive a whole retry cycle. Load
  is about 0.2 queries/sec against Neo4j, and the neo4j container already
  self-probes at `interval: 10s` (the `mist-neo4j` healthcheck in the same
  file). Cited by service name rather than by line: this module's own branch
  inserted eight lines above both blocks and invalidated the line numbers that
  used to stand here.
- Probe timeouts 2.0s (neo4j, llm), 2.5s (extraction_service) and 0.5s
  (sqlite- and file-backed). extraction_service's own HTTP client is built
  with `httpx.AsyncClient(timeout=2.0)`, which httpx applies to each phase
  (connect, read, write, pool acquisition) separately, not to the whole call.
  A refused connection or a host that never accepts it fails inside 2.0s and
  reads `unreachable`; a call whose phases each finish under 2.0s but
  together pass 2.5s reaches the probe's own bound first and reads `timeout`.
  Probes run concurrently, so every probe timing out at once still leaves the
  cycle finishing inside its own 10s interval. These bound the loop, never the
  request.
- `stale_after_seconds = 30.0`, three intervals. Survives two missed ticks
  without flapping.
- `dead_after_seconds = 90.0`, nine intervals. Every check going stale at once
  means the loop is gone, not a dependency: `probe_loop` renders `dead` and
  `restart_recommended` becomes true. This is how the loop's own death
  self-reports, rather than depending on a done-callback having fired.
- `settled_failures = 5`, about 50s. Shorter than Docker's 90s detection window
  (30s interval times 3 retries), so the flag settles before Docker has sampled
  three times, and a sub-50s transient never surfaces as settled.

How the top-level status is derived, so a reader can reproduce it
-----------------------------------------------------------------
    effective(c) = "down"    if c["status"] in ("down", "timeout")
                 = "unknown" if c["status"] is None or c["stale"]
                 = "yielded" if c["status"] == "yielded"
                 = "up"      otherwise

    status = "unhealthy" if any(effective(c) == "down" and c["severity"] == "fatal")
             "yielded"   elif any(effective(c) == "yielded")
             "degraded"  elif any(effective(c) != "up")
             "healthy"   else

    restart_recommended = probe_loop == "dead"
                          or any(c["restart_repairs"] and effective(c) == "down"
                                 and c["consecutive_failures"] >= settled_failures)

An empty `checks` mapping is `unhealthy` with `restart_recommended` false:
all three `any(...)` terms are vacuously false over nothing, and nothing
verified is not healthy.

`yielded` is a check status a probe signals with `ProbeYielded`: the
dependency was stopped on purpose (gaming mode stops the LLM to free the GPU).
It is evaluated after `down` and after null-or-stale, so a stale yielded check
reads unknown like any stale check, and a dead probe loop never reports
`yielded`. It outranks `degraded` because `status` answers "can MIST hold a
conversation?", and while yielded the answer is "no, by choice"; every
degrading check is still listed with its reason. A yielded check records
`consecutive_failures` 0 and is never `down`, so it never makes
`restart_recommended` true: restarting would not bring back a model stopped
on purpose.

`derive_status` implements exactly that, takes no clock and performs no I/O.

Nothing dynamic reaches the response
------------------------------------
`reason` is drawn from a closed vocabulary (`PROBE_REASONS`) and coerced to
`query_failed` if a probe supplies anything else. No exception text, URI,
hostname, port, database name, filesystem path or version string is rendered.
`Neo4jConnection.health_check` returns `uri`, `database` and raw `str(e)`
(`neo4j_connection.py:140-155`), a bolt URI can carry credentials, and `/health`
is unauthenticated behind a CORS wildcard (`KNOWN_ISSUES.md:154`). Detail goes
to the log; the payload gets a vocabulary word.
"""

import asyncio
import functools
import json
import logging
import math
import os
from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from backend.errors import MistError
from backend.extraction_backlog.errors import (
    InferenceResponseInvalidError,
    InferenceServiceError,
    InferenceUnreachableError,
)
from backend.extraction_contract.models import ExtractionStatus, HealthResponse

logger = logging.getLogger(__name__)

# Closed vocabulary. Membership is enforced structurally in `_coerce_reason`, so
# the no-leak property holds by construction rather than by discipline.
#
# Every term here is either raised by a probe today or is the designated answer
# for a rule this module commits to. `no_probe_implemented` is the one currently
# unused term, kept deliberately: the rule is that a dependency we choose not to
# probe is NAMED in the response rather than omitted, and this is the reason it
# would carry. `auth_failed` was removed rather than left unused, because no
# probe can ever raise it -- `Neo4jConnection.health_check` collapses every
# failure into `{"status": "unhealthy"}` (`neo4j_connection.py:149-155`), so
# publishing it would advertise a distinction `/health` never makes.
#
# The yielded-LLM terms, read from the backend's yield state:
# - `gpu_yielded`: the LLM was stopped on purpose to free the GPU; the check
#   renders `status: "yielded"`, not `down`.
# - `resume_failed`: the yield ended but the models did not reload; that is a
#   fault, so the check renders `down`.
#
# The extraction and backup terms, one per row of their checks:
# - `epoch_mismatch`: the dispatcher refuses to dispatch because the service's
#   epoch stamps differ from the backend's (dispatcher state `epoch_mismatch`).
# - `service_degraded`: the extraction service answered `GET /v1/health`, but
#   its `status` was not `ok` or it could not reach its own LLM.
# - `stalled`: the dispatcher state is `stalled`; nothing is being extracted.
# - `backlog_overdue`: the oldest pending turn is older than the backlog max age.
# - `dead_lettered`: at least one turn is dead-lettered and awaits a retry.
# - `backup_never_run`: backups are configured but no status file exists yet.
# - `backup_failed`: the last attempt did not succeed and no recent success
#   covers it.
# - `backup_suspect`: as `backup_failed`, but the last dump was set aside as
#   empty or shrunken, which wants a review rather than a retry.
# - `backup_overdue`: the last success is older than the backup max age.
PROBE_REASONS: frozenset[str] = frozenset(
    {
        "ok",
        "not_probed_yet",
        "not_started",
        "disabled_by_config",
        "unavailable_at_boot",
        "no_probe_implemented",
        "unreachable",
        "query_failed",
        "closed",
        "task_dead",
        "gpu_yielded",
        "resume_failed",
        "epoch_mismatch",
        "service_degraded",
        "stalled",
        "backlog_overdue",
        "dead_lettered",
        "backup_never_run",
        "backup_failed",
        "backup_suspect",
        "backup_overdue",
    }
)

DEFAULT_INTERVAL_SECONDS = 10.0
DEFAULT_STALE_AFTER_SECONDS = 30.0
DEFAULT_DEAD_AFTER_SECONDS = 90.0
DEFAULT_SETTLED_FAILURES = 5

DEFAULT_BACKLOG_MAX_AGE_S = 21600.0
DEFAULT_BACKUP_MAX_AGE_HOURS = 26.0

Severity = Literal["fatal", "degrades"]


# N818 asks for an `Error` suffix. These three are control-flow signals a probe
# raises to report an outcome, not I/O faults, and the names are part of the
# probe protocol that `backend/server.py` wiring will import.
class ProbeUnavailable(MistError):  # noqa: N818
    """Raised by a probe when nothing was measured, and that is expected.

    Renders as `status: null` with the supplied reason -- the not-applicable
    case, distinct from a measured failure.
    """


class ProbeFailed(MistError):  # noqa: N818
    """Raised by a probe when the dependency was reached and found unhealthy.

    Renders as `status: "down"` with the supplied reason.
    """


class ProbeYielded(MistError):  # noqa: N818
    """Raised by a probe when the dependency was stopped on purpose.

    Renders as `status: "yielded"` with the supplied reason and
    `consecutive_failures` 0: a choice, not a fault, so it never counts toward
    `restart_recommended`.
    """


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One dependency's last recorded probe outcome.

    `age_seconds` and `stale` are deliberately absent: they are functions of the
    read-time clock, so storing them would freeze a lie into the cache.

    Attributes:
        status: `up`, `down`, `timeout`, `yielded`, or None meaning NOT MEASURED.
        reason: A member of `PROBE_REASONS`, or None for a timeout.
        measured_at: Monotonic timestamp of the record. None iff never probed.
        latency_ms: Wall duration of the probe, None when nothing was measured.
        consecutive_failures: Count of consecutive `down`/`timeout` records;
            any other record, `yielded` included, resets it to 0.
        severity: Whether a failure here stops MIST serving.
        restart_repairs: Whether restarting the process plausibly fixes it.
    """

    status: str | None
    reason: str | None
    measured_at: float | None
    latency_ms: float | None
    consecutive_failures: int
    severity: Severity
    restart_repairs: bool


class Probe(Protocol):
    """A dependency probe the registry can run.

    `check` signals by control flow: returning normally means up, raising
    `ProbeUnavailable` means not measured, raising `ProbeFailed` means down,
    raising `ProbeYielded` means stopped on purpose, and anything else is
    treated as a `query_failed` down by the registry.
    """

    name: str
    severity: Severity
    restart_repairs: bool
    timeout_seconds: float

    async def check(self) -> None:
        """Probe the dependency, raising to signal anything but success."""
        ...


class GraphHealthSource(Protocol):
    """The one method `Neo4jProbe` needs from a graph connection."""

    def health_check(self) -> dict:
        """Return a dict whose `status` key is `healthy` on success."""
        ...


class LLMHealthSource(Protocol):
    """The one method `LLMProbe` needs from an LLM provider."""

    async def health_check(self) -> bool:
        """Return True when the inference server answers."""
        ...


class EventStoreHealthSource(Protocol):
    """The one method `EventStoreProbe` needs from the event store."""

    # Return type left unannotated (it is a `sqlite3.Connection`) so this module
    # does not import sqlite3 for a structural type it only calls one method on.
    def _get_connection(self):
        """Return the store's sqlite connection, opening it if needed."""
        ...


class SidecarHealthSource(Protocol):
    """The one method `VaultSidecarProbe` needs from the sidecar index."""

    def health_check(self) -> bool:
        """Return True when the index is open and its schema is intact."""
        ...


class ExtractionServiceHealthSource(Protocol):
    """The one method `ExtractionServiceProbe` needs from `RemoteExtractionInference`."""

    async def health(self) -> HealthResponse:
        """Call `GET /v1/health`, raising a typed inference error on failure."""
        ...


class ExtractionDispatcherHealthSource(Protocol):
    """What the extraction probes read from the dispatcher itself."""

    @property
    def mode(self) -> str:
        """`service` or `off`."""
        ...

    @property
    def state(self) -> str:
        """The dispatcher state, e.g. `idle`, `stalled` or `epoch_mismatch`."""
        ...


@dataclass(frozen=True, slots=True)
class HealthProbeSettings:
    """The thresholds and paths the extraction and backup probes read.

    Attributes:
        backlog_max_age_s: Age of the oldest pending turn past which
            `extraction_backlog` reads `backlog_overdue`
            (`MIST_HEALTH_BACKLOG_MAX_AGE_S`). Six hours: an extraction host
            that is off for an evening is reported by `extraction_service` at
            once, and the backlog only counts as overdue once it has outlasted
            that.
        backup_status_path: The backup scheduler's status file
            (`MIST_BACKUP_STATUS_PATH`), or None when backups are not
            configured here, which renders `backup` null.
        backup_max_age_hours: Age of the last success past which `backup`
            reads down (`MIST_BACKUP_MAX_AGE_HOURS`). 26 hours is the daily
            interval plus two hours of slack for a slow or retried dump.
    """

    backlog_max_age_s: float = DEFAULT_BACKLOG_MAX_AGE_S
    backup_status_path: Path | None = None
    backup_max_age_hours: float = DEFAULT_BACKUP_MAX_AGE_HOURS

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "HealthProbeSettings":
        """Read the settings from the environment, keeping a default for a bad value.

        A malformed or non-positive age is logged and replaced by its default
        rather than raised: this runs in `lifespan`, and a typo in a health
        threshold should not stop MIST serving, nor switch a check off.

        Args:
            environ: The variables to read; `os.environ` when None.

        Returns:
            The settings.
        """
        env = os.environ if environ is None else environ
        raw_path = (env.get("MIST_BACKUP_STATUS_PATH") or "").strip()
        return cls(
            backlog_max_age_s=_positive_float(
                env, "MIST_HEALTH_BACKLOG_MAX_AGE_S", DEFAULT_BACKLOG_MAX_AGE_S
            ),
            backup_status_path=Path(raw_path) if raw_path else None,
            backup_max_age_hours=_positive_float(
                env,
                "MIST_BACKUP_MAX_AGE_HOURS",
                DEFAULT_BACKUP_MAX_AGE_HOURS,
                limit=_MAX_TIMEDELTA_HOURS,
            ),
        )


# The largest hour count `timedelta(hours=...)` holds, in whole days so float
# rounding cannot push it past `timedelta.max`. `BackupProbe` builds that
# `timedelta` in `lifespan`, where an `OverflowError` would stop the boot.
_MAX_TIMEDELTA_HOURS = float(timedelta.max.days * 24)


def _positive_float(
    env: Mapping[str, str], name: str, default: float, *, limit: float = math.inf
) -> float:
    """Parse one finite positive number up to `limit`, logging and defaulting a bad value.

    `inf` and `nan` parse as floats, so they are refused explicitly: neither is
    an age anyone means, and an infinite one overflows `timedelta`.
    """
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if math.isfinite(value) and 0 < value <= limit:
        return value
    logger.error(
        "%s=%r is not a finite positive number up to %s; using %s", name, raw, limit, default
    )
    return default


def _coerce_reason(reason: str | None, probe_name: str) -> str | None:
    """Force a reason into the closed vocabulary before it can reach a response.

    Args:
        reason: The reason a probe supplied, or None.
        probe_name: Probe name, for the log line only.

    Returns:
        The reason if it is in `PROBE_REASONS` or None, else `query_failed`.
    """
    if reason is None or reason in PROBE_REASONS:
        return reason
    logger.error(
        "health probe %s supplied an off-vocabulary reason %r; coercing to query_failed",
        probe_name,
        reason,
    )
    return "query_failed"


class Neo4jProbe:
    """Probes the knowledge graph through `Neo4jConnection.health_check`.

    Never calls `Neo4jConnection.connect`, for two reasons. First, calling it on
    a dead driver is reconnect logic, which this goal explicitly excludes.
    Second, `connect` calls `assert_neo4j_isolated`
    (`neo4j_connection.py:35,40`), which raises `EvalIsolationError` -- a
    `RuntimeError`, not a `MistError` (`backend/knowledge/eval_isolation.py:169`)
    -- and `Neo4jConnection.health_check` catches only `(Neo4jError, MistError)`
    (`neo4j_connection.py:149`), so it would not catch it and `/health` would
    return 500 across the entire unit tier, which runs under
    `MIST_EVAL_ISOLATION=1` via the autouse fixture in `tests/unit/conftest.py`.
    That path is currently unreachable from `health_check`, because
    `is_connected` returns False when `_driver is None`
    (`neo4j_connection.py:65-67`) and short-circuits before any driver call --
    this is a hazard to preserve against, not a live bug.

    What this probe cannot tell you: an authentication rejection is
    indistinguishable from an unreachable server here, because
    `Neo4jConnection.health_check` collapses every failure into
    `{"status": "unhealthy"}` (`neo4j_connection.py:149-155`). Both therefore
    report `unreachable`. The vocabulary lists no auth reason for exactly that
    reason -- the contract names only outcomes that can actually occur.

    `severity="degrades"` and `restart_repairs=False`: the system's own boot path
    treats Neo4j as non-fatal (`ModelManager.__init__` logs `Knowledge
    integration disabled (Neo4j unavailable)` and continues --
    `backend/voice_models/model_manager.py:97-98`), and restarting the backend
    costs about 120s of model
    reload during which MIST serves nothing, against a memory-only fault where
    conversation still works. The backend also cannot distinguish "our driver is
    dead" (restart would fix it) from "Neo4j is down" (restart is destructive);
    the two present identically here.
    """

    def __init__(
        self,
        get_connection: Callable[[], GraphHealthSource | None],
        knowledge_configured: Callable[[], bool],
        executor: Executor,
    ) -> None:
        self.name = "neo4j"
        self.severity: Severity = "degrades"
        self.restart_repairs = False
        self.timeout_seconds = 2.0
        self._get_connection = get_connection
        self._knowledge_configured = knowledge_configured
        self._executor = executor

    async def check(self) -> None:
        """Run the graph health check in the injected single-worker executor."""
        connection = self._get_connection()
        if connection is None:
            # The distinction is load-bearing. `ModelManager.__init__` sets
            # `self.knowledge = None` when Neo4j is unavailable at boot
            # (`backend/voice_models/model_manager.py:96-103`, both the
            # `is_enabled()` False branch and the `except`), so
            # rendering null there would satisfy the letter of the null rule
            # while hiding exactly the fault it exists to expose.
            if self._knowledge_configured():
                raise ProbeFailed("unavailable_at_boot")
            raise ProbeUnavailable("disabled_by_config")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, connection.health_check)
        if not isinstance(result, dict) or result.get("status") != "healthy":
            # The dict also carries `uri`, `database` and raw error text. None of
            # it is read here; the vocabulary word is all that leaves.
            raise ProbeFailed("unreachable")


class LLMProbe:
    """Probes llama-server through the provider's own async health check.

    `severity="fatal"`: no LLM means MIST cannot answer at all.
    `restart_repairs=False`: the provider holds a stateless httpx client
    (`backend/llm/llama_server_provider.py:115-122`) and self-heals when
    llama-server returns, and llama-server has its own healthcheck and restart
    policy (the `mist-llm` service in `docker-compose.yml`).

    `yield_state` reads the backend's GPU yield state: `normal`, `yielded` or
    `resume_failed`. While `yielded` the LLM was stopped on purpose, so the
    check renders `yielded` (`gpu_yielded`) instead of a `down` that would
    call the choice a crash; `resume_failed` is a failed reload and renders
    `down`. Both are decided before the provider is called, since a provider
    stopped on purpose can only answer "unreachable". None, the default, or
    `normal` probes as before.
    """

    def __init__(
        self,
        get_provider: Callable[[], LLMHealthSource | None],
        *,
        yield_state: Callable[[], str] | None = None,
    ) -> None:
        self.name = "llm"
        self.severity: Severity = "fatal"
        self.restart_repairs = False
        self.timeout_seconds = 2.0
        self._get_provider = get_provider
        self._yield_state = yield_state

    async def check(self) -> None:
        """Report a yield, else await the provider health check and fail on False."""
        state = self._yield_state() if self._yield_state is not None else "normal"
        if state == "yielded":
            raise ProbeYielded("gpu_yielded")
        if state == "resume_failed":
            raise ProbeFailed("resume_failed")

        provider = self._get_provider()
        if provider is None:
            raise ProbeFailed("unavailable_at_boot")

        if not await provider.health_check():
            raise ProbeFailed("unreachable")


class EventStoreProbe:
    """Probes the append-only event store with a `SELECT 1`.

    Two accessors, not one, because a None store is genuinely ambiguous. Both
    paths in `backend/chat/conversation_handler.py` leave the attribute None:
    :845 initialises it to None, :847 only builds it `if es_config.enabled`
    (config-off), and :896-898 nulls it again in the `except` (initialize or
    epoch write raised). The store accessor cannot tell those two apart.

    The flag behind them is independently readable, so the ambiguity is
    resolvable and is resolved rather than collapsed: `EventStoreConfig.enabled`
    defaults to True from `EVENT_STORE_ENABLED`, itself defaulting to "true"
    (`backend/knowledge/config.py:203,218`). Configured plus a None store is
    `down` / `unavailable_at_boot`; not configured is `null` /
    `disabled_by_config`. Reporting `down` for a subsystem an operator switched
    off on purpose would claim a failure that did not occur -- the mirror image
    of the defect this module exists to remove -- and would fire in exactly the
    configuration where the operator already knows the thing is off.

    This makes the probe symmetric with `Neo4jProbe`, which takes a `configured`
    predicate for the same reason. Two probes facing the same ambiguity resolve
    it the same way, so a reader who has understood one has understood both.

    `severity="degrades"`, `restart_repairs=False`: conversation continues
    without Layer 1 recording, and a restart re-runs the same initialize against
    the same file.
    """

    def __init__(
        self,
        get_store: Callable[[], EventStoreHealthSource | None],
        event_store_configured: Callable[[], bool],
    ) -> None:
        self.name = "event_store"
        self.severity: Severity = "degrades"
        self.restart_repairs = False
        self.timeout_seconds = 0.5
        self._get_store = get_store
        self._event_store_configured = event_store_configured

    async def check(self) -> None:
        """Run a `SELECT 1` in a thread, since sqlite3 is blocking."""
        store = self._get_store()
        if store is None:
            if self._event_store_configured():
                raise ProbeFailed("unavailable_at_boot")
            raise ProbeUnavailable("disabled_by_config")

        await asyncio.to_thread(self._select_one, store)

    @staticmethod
    def _select_one(store: EventStoreHealthSource) -> None:
        """Force a real read; a cursor that is never fetched proves nothing."""
        store._get_connection().execute("SELECT 1").fetchone()


class VaultSidecarProbe:
    """Probes the sqlite-vec sidecar index (`backend/vault/sidecar_index.py:759`).

    A None sidecar renders null: the vault layer is genuinely optional and
    `backend/server.py`'s `lifespan` logs `Vault layer initialization failed
    (continuing without vault)` and carries on without it.

    `severity="degrades"`, `restart_repairs=False`: vault retrieval stops, the
    conversation does not, and the index is reopened by its own lifecycle rather
    than by a new process.
    """

    def __init__(self, get_sidecar: Callable[[], SidecarHealthSource | None]) -> None:
        self.name = "vault_sidecar"
        self.severity: Severity = "degrades"
        self.restart_repairs = False
        self.timeout_seconds = 0.5
        self._get_sidecar = get_sidecar

    async def check(self) -> None:
        """Run the sidecar health check in a thread, since sqlite3 is blocking."""
        sidecar = self._get_sidecar()
        if sidecar is None:
            raise ProbeUnavailable("disabled_by_config")

        if not await asyncio.to_thread(sidecar.health_check):
            # False covers both a closed connection and a missing table
            # (`sidecar_index.py:765-766` and the schema check below it).
            raise ProbeFailed("closed")


class BroadcasterState:
    """Single record of whether the broadcast task is still running.

    One object owns this fact so that every consumer -- the health probe, the
    log line, and any future escalation -- reads and writes it in one place
    rather than each keeping its own copy. Deliberately mutable, unlike the
    frozen dataclasses in this module: it is a live state holder, not a value.

    Constructed by the caller and injected (`BroadcasterProbe` takes the bound
    `state` method, not this object), so there is no module-level singleton here
    and the probe stays testable against a plain callable.
    """

    def __init__(self) -> None:
        self._state: str | None = None
        self._last_exception: BaseException | None = None

    def mark_alive(self) -> None:
        """Record that the broadcast task is running."""
        self._state = "alive"

    def mark_dead(self, exc: BaseException | None) -> None:
        """Record that the broadcast task has ended, keeping detail for the log.

        The exception is held for logging and for any future escalation. It
        never reaches a response: `state` returns only the three-valued state,
        and the probe renders the vocabulary word `task_dead`.
        """
        self._state = "dead"
        self._last_exception = exc
        logger.error("broadcast task is dead: %s", exc, exc_info=exc)

    def state(self) -> str | None:
        """Return `alive`, `dead`, or None when nothing has been recorded yet."""
        return self._state

    def last_exception(self) -> BaseException | None:
        """Return the exception that ended the task, for logs and escalation."""
        return self._last_exception


class BroadcasterProbe:
    """Reports whether the WebSocket broadcaster task is still alive.

    Reads memory only, so `timeout_seconds` never binds here; it exists because
    the protocol requires it.

    The accessor returns `alive`, `dead`, or None when nothing has published the
    state yet. `restart_repairs=True`, uniquely in this system: every outbound
    event -- transcription, LLM tokens, TTS audio frames, `state_cycle`,
    `vad_status`, heartbeats, `system_status`, `health_status`, log streaming --
    funnels through the broadcaster (18 `message_queue.put` sites across
    `backend/server.py` and `backend/voice_processor.py`), and a dead
    `asyncio.Task` is revived by nothing short of a new process.
    """

    def __init__(self, get_state: Callable[[], str | None]) -> None:
        self.name = "broadcaster"
        self.severity: Severity = "fatal"
        self.restart_repairs = True
        self.timeout_seconds = 0.5
        self._get_state = get_state

    async def check(self) -> None:
        """Translate the broadcaster state into a probe signal."""
        state = self._get_state()
        if state is None:
            # The probe ran and found that the task has not reported in yet.
            # Not `no_probe_implemented` (which would claim we chose not to
            # look) and not `not_probed_yet` (which means the loop has not
            # completed a cycle, when in fact this cycle ran).
            raise ProbeUnavailable("not_started")
        if state == "dead":
            raise ProbeFailed("task_dead")


class ExtractionServiceProbe:
    """Calls the extraction service's `GET /v1/health` on every probe cycle.

    The dispatcher only learns the service is gone when it has a job to send,
    so an idle dispatcher beside a switched-off extraction host reads `idle`.
    This probe makes its own call through its own `RemoteExtractionInference`
    (built by the server over a dedicated `httpx.AsyncClient` with a 2-second
    timeout per phase), so
    the host's absence shows within one interval whether or not work is queued.

    The dispatcher is read for two things only: whether there is one at all
    (none, or `off` mode, is null `disabled_by_config`) and whether its state
    is `epoch_mismatch`, which a healthy reply from the service cannot show.
    `start_failed` is the server's record that the dispatcher raised at boot,
    which a None dispatcher alone cannot tell apart from one never configured.

    `severity="degrades"`: conversation goes on, and turns are logged for later
    extraction. `restart_repairs=False`: the host is a separate machine.
    """

    def __init__(
        self,
        inference: ExtractionServiceHealthSource,
        get_dispatcher: Callable[[], ExtractionDispatcherHealthSource | None],
        start_failed: Callable[[], bool],
    ) -> None:
        self.name = "extraction_service"
        self.severity: Severity = "degrades"
        self.restart_repairs = False
        self.timeout_seconds = 2.5
        self._inference = inference
        self._get_dispatcher = get_dispatcher
        self._start_failed = start_failed

    async def check(self) -> None:
        """Evaluate item 3's `extraction_service` rows, top-down."""
        if self._start_failed():
            raise ProbeFailed("unavailable_at_boot")
        dispatcher = self._get_dispatcher()
        if dispatcher is None or dispatcher.mode == "off":
            raise ProbeUnavailable("disabled_by_config")

        try:
            reply = await self._inference.health()
        except InferenceUnreachableError as exc:
            logger.debug("extraction service unreachable: %s", exc)
            raise ProbeFailed("unreachable") from exc
        except (InferenceServiceError, InferenceResponseInvalidError) as exc:
            logger.warning("extraction service health call failed: %s", exc)
            raise ProbeFailed("query_failed") from exc

        if reply.status != "ok" or not reply.llm_reachable:
            raise ProbeFailed("service_degraded")
        if dispatcher.state == "epoch_mismatch":
            raise ProbeFailed("epoch_mismatch")


class ExtractionBacklogProbe:
    """Reports whether MIST is still learning from what it hears.

    Reads the producer behind the `extraction` block, through
    `checked_extraction_status` (injected as `read_status`), which returns
    None where the block falls back to zero counts. Zero counts from a backlog
    that could not be read look like an empty one, so None reads
    `query_failed` here instead of `ok`.

    `stalled` and `backlog_overdue` are evaluated before `dead_lettered`, so a
    lasting dead letter never hides a stalled or overdue backlog. The count is
    current, not cumulative: a retried dead letter leaves it.

    Reads SQLite through `asyncio.to_thread`, as `EventStoreProbe` does, and
    takes no executor: behind the Neo4j probe's single worker a wedged graph
    call would read as an extraction timeout.

    `severity="degrades"`, `restart_repairs=False`: turns keep being logged,
    and a restart replays the same backlog.
    """

    def __init__(
        self,
        get_dispatcher: Callable[[], ExtractionDispatcherHealthSource | None],
        read_status: Callable[..., ExtractionStatus | None],
        start_failed: Callable[[], bool],
        settings: HealthProbeSettings,
    ) -> None:
        self.name = "extraction_backlog"
        self.severity: Severity = "degrades"
        self.restart_repairs = False
        self.timeout_seconds = 0.5
        self._get_dispatcher = get_dispatcher
        self._read_status = read_status
        self._start_failed = start_failed
        self._max_age_ms = settings.backlog_max_age_s * 1000.0

    async def check(self) -> None:
        """Evaluate item 3's `extraction_backlog` rows, top-down."""
        if self._start_failed():
            raise ProbeFailed("unavailable_at_boot")
        dispatcher = self._get_dispatcher()
        if dispatcher is None or dispatcher.mode == "off":
            raise ProbeUnavailable("disabled_by_config")

        status = await asyncio.to_thread(self._read_status, dispatcher)
        if status is None:
            raise ProbeFailed("query_failed")
        if status.state == "stalled":
            raise ProbeFailed("stalled")
        oldest = status.oldest_pending_age_ms
        if oldest is not None and oldest > self._max_age_ms:
            raise ProbeFailed("backlog_overdue")
        if status.dead_lettered > 0:
            raise ProbeFailed("dead_lettered")


# Every key of the backup scheduler's status file. A file missing any of them
# was not written by the scheduler this build pairs with, so it is not trusted.
_BACKUP_STATUS_KEYS: frozenset[str] = frozenset(
    {
        "v",
        "last_attempt_at",
        "last_result",
        "last_exit_code",
        "last_success_at",
        "last_label",
        "last_duration_s",
        "retained",
        "last_success_counts",
    }
)


@dataclass(frozen=True, slots=True)
class _BackupRecord:
    """The two status-file fields the backup rows decide on."""

    last_result: str | None
    last_success_at: datetime | None


class BackupProbe:
    """Reports whether the last good backup is recent enough.

    Reads the status file the backup scheduler writes after every attempt.
    With no status path configured (the workstation stack), the check is
    null: nothing here backs up, and saying `down` would claim a failure.

    `backup_suspect` comes before `backup_failed` because a dump set aside as
    empty or shrunken needs a person to look at it, while a failed one is
    retried on its own. Both apply only once the last good success is older
    than the max age: a failed attempt beside a recent success is still
    backed up.

    Reads the file through `asyncio.to_thread` and takes no executor, for the
    same reason as `ExtractionBacklogProbe`. The wall clock is injected
    because the file's timestamps are wall time.

    `severity="degrades"`, `restart_repairs=False`: backups run in their own
    container, and a backend restart does nothing for them.
    """

    def __init__(
        self,
        settings: HealthProbeSettings,
        wall_clock: Callable[[], datetime],
    ) -> None:
        self.name = "backup"
        self.severity: Severity = "degrades"
        self.restart_repairs = False
        self.timeout_seconds = 0.5
        self._path = settings.backup_status_path
        self._max_age = timedelta(hours=settings.backup_max_age_hours)
        self._wall_clock = wall_clock

    async def check(self) -> None:
        """Evaluate item 3's `backup` rows, top-down."""
        if self._path is None:
            raise ProbeUnavailable("disabled_by_config")
        try:
            text = await asyncio.to_thread(self._read_text, self._path)
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("backup status file unreadable: %s", exc)
            raise ProbeFailed("query_failed") from exc
        if text is None:
            raise ProbeFailed("backup_never_run")

        record = _parse_backup_status(text)
        success_at = record.last_success_at
        # A null success is stale even beside `last_result: "ok"`: no success
        # within the max age means not backed up, so that reads `backup_overdue`.
        stale = success_at is None or self._wall_clock() - success_at > self._max_age
        if record.last_result == "suspect" and stale:
            raise ProbeFailed("backup_suspect")
        if record.last_result != "ok" and stale:
            raise ProbeFailed("backup_failed")
        if stale:
            raise ProbeFailed("backup_overdue")

    @staticmethod
    def _read_text(path: Path) -> str | None:
        """Return the file's text, or None when it does not exist."""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None


def _parse_backup_status(text: str) -> _BackupRecord:
    """Validate a status file body, raising `ProbeFailed("query_failed")` if unusable.

    Args:
        text: The file's contents.

    Returns:
        The two fields the backup rows read.
    """
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("backup status file is not JSON: %s", exc)
        raise ProbeFailed("query_failed") from exc
    if not isinstance(body, dict):
        logger.warning("backup status file is not a JSON object")
        raise ProbeFailed("query_failed")
    missing = _BACKUP_STATUS_KEYS - body.keys()
    if missing:
        logger.warning("backup status file lacks keys: %s", sorted(missing))
        raise ProbeFailed("query_failed")

    last_result = body["last_result"]
    if last_result is not None and not isinstance(last_result, str):
        logger.warning("backup status file has a non-string last_result: %r", last_result)
        raise ProbeFailed("query_failed")
    return _BackupRecord(
        last_result=last_result,
        last_success_at=_parse_utc(body["last_success_at"]),
    )


def _parse_utc(value: object) -> datetime | None:
    """Parse an ISO-8601 timestamp, reading one without an offset as UTC."""
    if value is None:
        return None
    if not isinstance(value, str):
        logger.warning("backup status timestamp is not a string: %r", value)
        raise ProbeFailed("query_failed")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        logger.warning("backup status timestamp is not ISO-8601: %r", value)
        raise ProbeFailed("query_failed") from exc
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def derive_status(
    rendered_checks: dict[str, dict],
    probe_loop: str,
    settled_failures: int = DEFAULT_SETTLED_FAILURES,
) -> tuple[str, bool]:
    """Derive the top-level status and restart advice from rendered checks.

    Pure: no I/O, no clock, no registry state. A reader holding only the JSON
    payload can reproduce this exactly, which is the contract in place of an ADR.

    A check whose status is `yielded` maps to `yielded` only once it is
    neither down nor null-or-stale, and the top-level `yielded` sits between
    `unhealthy` and `degraded`; the module docstring says why.

    An empty mapping is `unhealthy`, not `healthy`: the `any(...)` calls below
    are False over an empty iterable, so without this guard a registry that
    probed nothing would report the system healthy -- the same vacuity this
    module rejects per check, at the level of the set rather than the member.
    Nothing verified is not healthy. `restart_recommended` stays False, because
    a restart does not add probes.

    Args:
        rendered_checks: The `checks` mapping from a snapshot.
        probe_loop: `alive` or `dead`.
        settled_failures: Consecutive failures before restart advice settles.

    Returns:
        A `(status, restart_recommended)` pair.
    """
    if not rendered_checks:
        return "unhealthy", False

    effective: dict[str, str] = {}
    for name, check in rendered_checks.items():
        if check["status"] in ("down", "timeout"):
            effective[name] = "down"
        elif check["status"] is None or check["stale"]:
            effective[name] = "unknown"
        elif check["status"] == "yielded":
            effective[name] = "yielded"
        else:
            effective[name] = "up"

    if any(
        effective[name] == "down" and check["severity"] == "fatal"
        for name, check in rendered_checks.items()
    ):
        status = "unhealthy"
    elif any(state == "yielded" for state in effective.values()):
        status = "yielded"
    elif any(state != "up" for state in effective.values()):
        status = "degraded"
    else:
        status = "healthy"

    restart_recommended = probe_loop == "dead" or any(
        check["restart_repairs"]
        and effective[name] == "down"
        and check["consecutive_failures"] >= settled_failures
        for name, check in rendered_checks.items()
    )
    return status, restart_recommended


class HealthRegistry:
    """Runs probes on a cadence and renders their cached results.

    The two clocks are injected separately and neither is derived from the
    other: monotonic drives ages and staleness (unaffected by a wall-clock step),
    wall time only labels `checked_at` for a human reader.
    """

    def __init__(
        self,
        probes: list[Probe],
        monotonic: Callable[[], float],
        wall_clock: Callable[[], datetime],
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
        dead_after_seconds: float = DEFAULT_DEAD_AFTER_SECONDS,
        settled_failures: int = DEFAULT_SETTLED_FAILURES,
    ) -> None:
        self._probes = list(probes)
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._interval_seconds = interval_seconds
        self._stale_after_seconds = stale_after_seconds
        self._dead_after_seconds = dead_after_seconds
        self._settled_failures = settled_failures

        self._created_at = monotonic()
        self._last_cycle_completed: float | None = None
        self._last_cycle_wall: datetime | None = None
        self._in_flight: dict[str, asyncio.Task] = {}
        self._results: dict[str, CheckResult] = {
            probe.name: CheckResult(
                status=None,
                reason="not_probed_yet",
                measured_at=None,
                latency_ms=None,
                consecutive_failures=0,
                severity=probe.severity,
                restart_repairs=probe.restart_repairs,
            )
            for probe in self._probes
        }

    async def run_once(self) -> None:
        """Probe every idle dependency concurrently and update the cache.

        A probe still in flight from an earlier tick is skipped rather than
        started again; its cached result ages instead.
        """
        pending = []
        for probe in self._probes:
            running = self._in_flight.get(probe.name)
            if running is not None and not running.done():
                logger.warning(
                    "health probe %s still in flight from an earlier cycle; skipping", probe.name
                )
                continue
            pending.append(self._run_probe(probe))

        if pending:
            await asyncio.gather(*pending)

        self._last_cycle_completed = self._monotonic()
        self._last_cycle_wall = self._wall_clock()

    async def run_forever(self) -> None:
        """Probe on the configured interval until cancelled."""
        while True:
            await self.run_once()
            await asyncio.sleep(self._interval_seconds)

    async def _run_probe(self, probe: Probe) -> None:
        """Run one probe under its own timeout and record a structured result."""
        started = self._monotonic()
        task = asyncio.create_task(probe.check())
        self._in_flight[probe.name] = task
        timed_out = False
        try:
            # Shielded so the timeout abandons the probe rather than cancelling
            # it: an abandoned probe stays in `_in_flight` and is skipped next
            # tick, which is what keeps a wedged dependency from stacking.
            await asyncio.wait_for(asyncio.shield(task), timeout=probe.timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            # Attached here and nowhere else. Only a probe abandoned by this
            # branch can still be running, so only this branch can produce a
            # completion that is genuinely late.
            task.add_done_callback(functools.partial(self._drain_late_failure, probe))
            self._record(probe, started, status="timeout", reason=None, measured=True)
        except ProbeUnavailable as exc:
            self._record(probe, started, status=None, reason=str(exc), measured=False)
        except ProbeFailed as exc:
            self._record(probe, started, status="down", reason=str(exc), measured=True)
        except ProbeYielded as exc:
            self._record(probe, started, status="yielded", reason=str(exc), measured=True)
        except Exception as exc:  # noqa: BLE001
            # The one justified bare catch in this module, mirroring the one in
            # `system_status_loop` in `backend/server.py`. Cited by symbol: that
            # line number has moved twice already.
            # The registry's contract is that ANY probe
            # failure becomes a structured result rather than propagating into
            # the loop and killing it -- a health system that dies on an
            # unexpected exception is the failure it exists to report. The
            # detail goes to the log with a traceback; the response gets a
            # vocabulary word.
            logger.error(
                "health probe %s raised an unexpected error: %s", probe.name, exc, exc_info=True
            )
            self._record(probe, started, status="down", reason="query_failed", measured=True)
        else:
            self._record(probe, started, status="up", reason="ok", measured=True)
        finally:
            if not timed_out:
                self._in_flight.pop(probe.name, None)

    def _drain_late_failure(self, probe: Probe, task: asyncio.Task) -> None:
        """Consume the exception of a probe abandoned at its own timeout.

        Attached only from the `TimeoutError` branch of `_run_probe`, and that
        placement is what makes the message below true. This callback used to be
        attached to every probe task at creation, so a `ProbeFailed` or
        `ProbeUnavailable` raised and fully handled INSIDE the timeout still
        reached here and logged `health probe finished after its timeout with an
        error: unreachable`. At the 10s production interval an absent vault
        sidecar or a down Neo4j therefore produced a false timeout claim every
        ten seconds for the life of the process.

        A probe that finishes inside its timeout has its exception retrieved by
        the `await` in `_run_probe` -- `asyncio.shield` calls `.exception()` on
        the inner task to propagate it -- so it needs no drain and no longer
        reaches this callback at all.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "health probe %s was abandoned at its %ss timeout and then failed: %s",
                probe.name,
                probe.timeout_seconds,
                exc,
            )

    def _record(
        self,
        probe: Probe,
        started: float,
        status: str | None,
        reason: str | None,
        measured: bool,
    ) -> None:
        """Write one probe outcome into the cache."""
        now = self._monotonic()
        previous = self._results[probe.name]
        failures = previous.consecutive_failures + 1 if status in ("down", "timeout") else 0
        self._results[probe.name] = CheckResult(
            status=status,
            reason=_coerce_reason(reason, probe.name),
            measured_at=now,
            latency_ms=round((now - started) * 1000.0, 3) if measured else None,
            consecutive_failures=failures,
            severity=probe.severity,
            restart_repairs=probe.restart_repairs,
        )

    def snapshot(self) -> dict:
        """Render the cache. Pure read: no I/O, safe before the first cycle.

        Returns:
            The health payload, with ages and staleness computed against the
            injected monotonic clock at call time.
        """
        now = self._monotonic()
        checks: dict[str, dict] = {}
        for name, result in self._results.items():
            age = None if result.measured_at is None else max(0.0, now - result.measured_at)
            checks[name] = {
                "status": result.status,
                "reason": result.reason,
                "age_seconds": None if age is None else round(age, 3),
                # Never measured counts as stale: there is no fresh observation
                # behind it, and `status` is null for the same reason.
                "stale": True if age is None else age > self._stale_after_seconds,
                "latency_ms": result.latency_ms,
                "consecutive_failures": result.consecutive_failures,
                "severity": result.severity,
                "restart_repairs": result.restart_repairs,
            }

        since_cycle = now - (
            self._last_cycle_completed
            if self._last_cycle_completed is not None
            else self._created_at
        )
        probe_loop = "dead" if since_cycle > self._dead_after_seconds else "alive"
        status, restart_recommended = derive_status(checks, probe_loop, self._settled_failures)

        return {
            "status": status,
            "restart_recommended": restart_recommended,
            "probe_loop": probe_loop,
            "probe_interval_seconds": self._interval_seconds,
            # Wall time of the last COMPLETED cycle, null before the first one --
            # a render timestamp would imply a measurement that never happened.
            "checked_at": _format_wall(self._last_cycle_wall),
            "checks": checks,
        }


def _format_wall(moment: datetime | None) -> str | None:
    """Render a wall-clock instant as ISO 8601 with a Z suffix, or None."""
    if moment is None:
        return None
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")
