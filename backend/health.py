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
- `interval_seconds = 10.0`. Docker reads `/health` every 30s
  (`docker-compose.yml:92`), so a 10s loop keeps the cache at most 10s old at
  read time and gives roughly three independent samples per `retries: 3`
  window. A 30s loop would align with Docker's sampling and let one unlucky
  probe drive a whole retry cycle. Load is about 0.2 queries/sec against Neo4j,
  and the neo4j container already self-probes at `interval: 10s`
  (`docker-compose.yml:105`).
- Probe timeouts 2.0s (neo4j, llm) and 0.5s (sqlite-backed). Chosen so that
  every probe timing out at once still leaves the cycle finishing inside its own
  10s interval. These bound the loop, never the request.
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
                 = "up"      otherwise

    status = "unhealthy" if any(effective(c) == "down" and c["severity"] == "fatal")
             "degraded"  elif any(effective(c) != "up")
             "healthy"   else

    restart_recommended = probe_loop == "dead"
                          or any(c["restart_repairs"] and effective(c) == "down"
                                 and c["consecutive_failures"] >= settled_failures)

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
import logging
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from backend.errors import MistError

logger = logging.getLogger(__name__)

# Closed vocabulary. Membership is enforced structurally in `_coerce_reason`, so
# the no-leak property holds by construction rather than by discipline.
PROBE_REASONS: frozenset[str] = frozenset(
    {
        "ok",
        "not_probed_yet",
        "disabled_by_config",
        "unavailable_at_boot",
        "no_probe_implemented",
        "unreachable",
        "auth_failed",
        "query_failed",
        "closed",
        "task_dead",
    }
)

DEFAULT_INTERVAL_SECONDS = 10.0
DEFAULT_STALE_AFTER_SECONDS = 30.0
DEFAULT_DEAD_AFTER_SECONDS = 90.0
DEFAULT_SETTLED_FAILURES = 5

Severity = Literal["fatal", "degrades"]


# N818 asks for an `Error` suffix. These two are control-flow signals a probe
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


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One dependency's last recorded probe outcome.

    `age_seconds` and `stale` are deliberately absent: they are functions of the
    read-time clock, so storing them would freeze a lie into the cache.

    Attributes:
        status: `up`, `down`, `timeout`, or None meaning NOT MEASURED.
        reason: A member of `PROBE_REASONS`, or None for a timeout.
        measured_at: Monotonic timestamp of the record. None iff never probed.
        latency_ms: Wall duration of the probe, None when nothing was measured.
        consecutive_failures: Count of consecutive `down`/`timeout` records.
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
    `ProbeUnavailable` means not measured, raising `ProbeFailed` means down, and
    anything else is treated as a `query_failed` down by the registry.
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

    `severity="degrades"` and `restart_repairs=False`: the system's own boot path
    treats Neo4j as non-fatal (`backend/voice_models/model_manager.py:92-95`
    warns and continues), and restarting the backend costs about 120s of model
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
            # The distinction is load-bearing. `model_manager.py:85-98` sets
            # `self.knowledge = None` when Neo4j is unavailable at boot, so
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
    policy at `docker-compose.yml:171-177`.
    """

    def __init__(self, get_provider: Callable[[], LLMHealthSource | None]) -> None:
        self.name = "llm"
        self.severity: Severity = "fatal"
        self.restart_repairs = False
        self.timeout_seconds = 2.0
        self._get_provider = get_provider

    async def check(self) -> None:
        """Await the provider health check and fail on a False answer."""
        provider = self._get_provider()
        if provider is None:
            raise ProbeFailed("unavailable_at_boot")

        if not await provider.health_check():
            raise ProbeFailed("unreachable")


class EventStoreProbe:
    """Probes the append-only event store with a `SELECT 1`.

    A None store is reported as a failure, not as config-off. Both paths in
    `backend/chat/conversation_handler.py:845-898` leave the attribute None: the
    `if es_config.enabled` guard at :847 (config-off) and the `except` at
    :896-898 (initialize or epoch write raised). The accessor cannot tell them
    apart, and `EventStoreConfig.enabled` defaults to True with
    `EVENT_STORE_ENABLED` defaulting to "true"
    (`backend/knowledge/config.py:203,218`), so in any default deployment None
    means the failure path. Reporting null here would hide a boot failure behind
    an "optional" label.

    `severity="degrades"`, `restart_repairs=False`: conversation continues
    without Layer 1 recording, and a restart re-runs the same initialize against
    the same file.
    """

    def __init__(self, get_store: Callable[[], EventStoreHealthSource | None]) -> None:
        self.name = "event_store"
        self.severity: Severity = "degrades"
        self.restart_repairs = False
        self.timeout_seconds = 0.5
        self._get_store = get_store

    async def check(self) -> None:
        """Run a `SELECT 1` in a thread, since sqlite3 is blocking."""
        store = self._get_store()
        if store is None:
            raise ProbeFailed("unavailable_at_boot")

        await asyncio.to_thread(self._select_one, store)

    @staticmethod
    def _select_one(store: EventStoreHealthSource) -> None:
        """Force a real read; a cursor that is never fetched proves nothing."""
        store._get_connection().execute("SELECT 1").fetchone()


class VaultSidecarProbe:
    """Probes the sqlite-vec sidecar index (`backend/vault/sidecar_index.py:759`).

    A None sidecar renders null: the vault layer is genuinely optional and
    `backend/server.py:492` warns and continues without it.

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
            # Nothing publishes the state yet; null rather than a guess.
            raise ProbeUnavailable("no_probe_implemented")
        if state == "dead":
            raise ProbeFailed("task_dead")


def derive_status(
    rendered_checks: dict[str, dict],
    probe_loop: str,
    settled_failures: int = DEFAULT_SETTLED_FAILURES,
) -> tuple[str, bool]:
    """Derive the top-level status and restart advice from rendered checks.

    Pure: no I/O, no clock, no registry state. A reader holding only the JSON
    payload can reproduce this exactly, which is the contract in place of an ADR.

    Args:
        rendered_checks: The `checks` mapping from a snapshot.
        probe_loop: `alive` or `dead`.
        settled_failures: Consecutive failures before restart advice settles.

    Returns:
        A `(status, restart_recommended)` pair.
    """
    effective: dict[str, str] = {}
    for name, check in rendered_checks.items():
        if check["status"] in ("down", "timeout"):
            effective[name] = "down"
        elif check["status"] is None or check["stale"]:
            effective[name] = "unknown"
        else:
            effective[name] = "up"

    if any(
        effective[name] == "down" and check["severity"] == "fatal"
        for name, check in rendered_checks.items()
    ):
        status = "unhealthy"
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
        task.add_done_callback(self._drain_late_failure)
        timed_out = False
        try:
            # Shielded so the timeout abandons the probe rather than cancelling
            # it: an abandoned probe stays in `_in_flight` and is skipped next
            # tick, which is what keeps a wedged dependency from stacking.
            await asyncio.wait_for(asyncio.shield(task), timeout=probe.timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            self._record(probe, started, status="timeout", reason=None, measured=True)
        except ProbeUnavailable as exc:
            self._record(probe, started, status=None, reason=str(exc), measured=False)
        except ProbeFailed as exc:
            self._record(probe, started, status="down", reason=str(exc), measured=True)
        except Exception as exc:  # noqa: BLE001
            # The one justified bare catch in this module, mirroring
            # `backend/server.py:191`. The registry's contract is that ANY probe
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

    def _drain_late_failure(self, task: asyncio.Task) -> None:
        """Consume the exception of an abandoned probe so asyncio stays quiet."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("health probe finished after its timeout with an error: %s", exc)

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
