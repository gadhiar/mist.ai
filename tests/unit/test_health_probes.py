"""Tests for the health probe registry.

These tests are written against the failure modes a health endpoint actually
has: a probe that reports up without touching anything, a cached result served
as fresh, a timeout collapsed into a failure, a never-probed dependency rendered
as healthy, and infrastructure detail leaking into an unauthenticated response.
The clock is injected everywhere, so no test sleeps to make time pass.
"""

import asyncio
import inspect
import json
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from backend.extraction_backlog.inference import RemoteExtractionInference
from backend.extraction_backlog.status import checked_extraction_status
from backend.extraction_contract.models import (
    ErrorCode,
    ExtractionStatus,
    ServiceStatus,
    error_envelope,
)
from backend.health import (
    PROBE_REASONS,
    BackupProbe,
    BroadcasterProbe,
    BroadcasterState,
    EventStoreProbe,
    ExtractionBacklogProbe,
    ExtractionServiceProbe,
    HealthProbeSettings,
    HealthRegistry,
    LLMProbe,
    Neo4jProbe,
    ProbeFailed,
    ProbeUnavailable,
    VaultSidecarProbe,
    derive_status,
)
from backend.knowledge.eval_isolation import EvalIsolationError

SECRET_URI = "bolt://neo4j:hunter2@mist-neo4j:7687 refused"
SECRET_PATH = "/app/dev-state/event_store.db"
LEAKY_MESSAGE = f"{SECRET_URI} while opening {SECRET_PATH}"


class FakeClock:
    """A monotonic source and a wall-clock source that only move when told."""

    def __init__(self, start: float = 1000.0) -> None:
        self._monotonic = start
        self._wall = datetime(2026, 9, 22, 23, 41, 7, 412000, tzinfo=UTC)

    def monotonic(self) -> float:
        return self._monotonic

    def wall(self) -> datetime:
        return self._wall

    def advance(self, seconds: float) -> None:
        self._monotonic += seconds


class FakeNeo4jConnection:
    """Records every call, so a probe that touches nothing is visible."""

    def __init__(self, result: dict | None = None, raises: Exception | None = None) -> None:
        self.result = result if result is not None else {"status": "healthy", "connected": True}
        self.raises = raises
        self.health_check_calls = 0
        self.connect_calls = 0

    def health_check(self) -> dict:
        self.health_check_calls += 1
        if self.raises is not None:
            raise self.raises
        return self.result

    def connect(self) -> None:
        self.connect_calls += 1
        raise EvalIsolationError("eval isolation refuses the live graph")


class FakeLLMProvider:
    def __init__(self, healthy: bool = True, raises: Exception | None = None) -> None:
        self.healthy = healthy
        self.raises = raises
        self.calls = 0

    async def health_check(self) -> bool:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.healthy


class FakeEventStore:
    def __init__(self, raises: Exception | None = None) -> None:
        self.raises = raises
        self.calls = 0

    def _get_connection(self):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return _FakeSqliteConnection()


class _FakeSqliteConnection:
    def execute(self, query: str):
        assert query == "SELECT 1"
        return self

    def fetchone(self):
        return (1,)


class FakeSidecar:
    def __init__(self, healthy: bool = True, raises: Exception | None = None) -> None:
        self.healthy = healthy
        self.raises = raises
        self.calls = 0

    def health_check(self) -> bool:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.healthy


class StubProbe:
    """A probe whose behaviour is a coroutine supplied by the test."""

    def __init__(
        self,
        name: str,
        behaviour,
        severity: str = "degrades",
        restart_repairs: bool = False,
        timeout_seconds: float = 0.05,
    ) -> None:
        self.name = name
        self.severity = severity
        self.restart_repairs = restart_repairs
        self.timeout_seconds = timeout_seconds
        self.calls = 0
        self._behaviour = behaviour

    async def check(self) -> None:
        self.calls += 1
        await self._behaviour()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def executor():
    pool = ThreadPoolExecutor(max_workers=1)
    yield pool
    pool.shutdown(wait=False)


def build_registry(clock: FakeClock, probes: list) -> HealthRegistry:
    return HealthRegistry(probes, monotonic=clock.monotonic, wall_clock=clock.wall)


def neo4j_registry(clock: FakeClock, executor, connection, configured: bool = True):
    probe = Neo4jProbe(lambda: connection, lambda: configured, executor)
    return build_registry(clock, [probe])


def cancel_stragglers(registry: HealthRegistry) -> None:
    """Cancel probes deliberately left hanging, so the loop closes quietly."""
    for task in registry._in_flight.values():
        task.cancel()


# 1
@pytest.mark.asyncio
async def test_probe_actually_calls_the_dependency(clock, executor):
    connection = FakeNeo4jConnection()
    registry = neo4j_registry(clock, executor, connection)

    await registry.run_once()

    assert connection.health_check_calls == 1
    assert registry.snapshot()["checks"]["neo4j"]["status"] == "up"


# 2
@pytest.mark.asyncio
async def test_a_failing_neo4j_is_detected(clock, executor):
    connection = FakeNeo4jConnection(result={"status": "unhealthy", "connected": False})
    registry = neo4j_registry(clock, executor, connection)

    await registry.run_once()
    snapshot = registry.snapshot()

    assert snapshot["checks"]["neo4j"]["status"] == "down"
    assert snapshot["checks"]["neo4j"]["reason"] == "unreachable"
    assert snapshot["status"] == "degraded"


# 3
@pytest.mark.asyncio
async def test_the_value_tracks_the_dependency(clock, executor):
    """Non-vacuity: flip the dependency inside one test and see both answers.

    A hardcoded `up` passes the healthy test, and a hardcoded `down` passes the
    failing one if the two happen to run in that order. Neither survives this.
    """
    connection = FakeNeo4jConnection()
    registry = neo4j_registry(clock, executor, connection)

    await registry.run_once()
    assert registry.snapshot()["checks"]["neo4j"]["status"] == "up"

    connection.result = {"status": "unhealthy"}
    await registry.run_once()
    assert registry.snapshot()["checks"]["neo4j"]["status"] == "down"


# 4
@pytest.mark.asyncio
async def test_a_cached_result_is_not_reused_as_fresh(clock, executor):
    connection = FakeNeo4jConnection()
    registry = neo4j_registry(clock, executor, connection)

    await registry.run_once()
    assert registry.snapshot()["checks"]["neo4j"]["status"] == "up"

    connection.raises = RuntimeError("driver gone")
    clock.advance(10.0)
    await registry.run_once()

    check = registry.snapshot()["checks"]["neo4j"]
    assert check["status"] == "down"
    assert check["reason"] == "query_failed"
    assert check["age_seconds"] == 0.0


# 5
@pytest.mark.asyncio
async def test_timeout_is_distinguishable_from_failure(clock):
    async def hang():
        await asyncio.sleep(3600)

    async def fail():
        raise ProbeFailed("unreachable")

    hanging = StubProbe("hanging", hang, timeout_seconds=0.02)
    failing = StubProbe("failing", fail)
    registry = build_registry(clock, [hanging, failing])

    await registry.run_once()
    checks = registry.snapshot()["checks"]

    assert checks["hanging"]["status"] == "timeout"
    assert checks["failing"]["status"] == "down"
    assert checks["hanging"]["status"] != checks["failing"]["status"]
    assert checks["hanging"]["reason"] is None
    cancel_stragglers(registry)


@pytest.mark.asyncio
async def test_handled_failures_do_not_claim_a_late_timeout(clock, caplog):
    """A failure handled inside the timeout must not be reported as a late one.

    `_drain_late_failure` was attached to every probe task at creation, so any
    probe that raised -- including a `ProbeFailed` or `ProbeUnavailable` caught
    and recorded well inside its own timeout -- logged `health probe finished
    after its timeout with an error: ...`. With the loop at 10s in production,
    an absent vault sidecar or a down Neo4j meant that false claim every ten
    seconds, forever. The drain now belongs to the timeout branch alone.

    The hanging probe is here so the test cannot pass by the drain having been
    deleted outright: a genuinely abandoned probe that later fails must still
    be logged, and must name itself when it is.
    """

    async def fail():
        raise ProbeFailed("unreachable")

    async def absent():
        raise ProbeUnavailable("disabled_by_config")

    async def hang_then_fail():
        await asyncio.sleep(0.05)
        raise ProbeFailed("unreachable")

    failing = StubProbe("failing", fail)
    unavailable = StubProbe("unavailable", absent)
    late = StubProbe("late", hang_then_fail, timeout_seconds=0.01)
    registry = build_registry(clock, [failing, unavailable, late])

    with caplog.at_level(logging.WARNING, logger="backend.health"):
        await registry.run_once()
        drained = registry._in_flight["late"]
        with pytest.raises(ProbeFailed):
            await drained
        await asyncio.sleep(0)

    messages = [record.getMessage() for record in caplog.records]
    in_timeout = [m for m in messages if "failing" in m or "unavailable" in m]
    assert in_timeout == [], f"a handled in-timeout failure was logged as late: {in_timeout}"

    late_lines = [m for m in messages if "abandoned at its" in m]
    assert len(late_lines) == 1, messages
    assert "late" in late_lines[0]
    assert "unreachable" in late_lines[0]


# 6
@pytest.mark.asyncio
async def test_unprobed_is_null_not_healthy(clock, executor):
    registry = neo4j_registry(clock, executor, FakeNeo4jConnection())

    snapshot = registry.snapshot()
    check = snapshot["checks"]["neo4j"]

    assert check["status"] is None
    assert check["reason"] == "not_probed_yet"
    assert check["age_seconds"] is None
    assert check["latency_ms"] is None
    assert snapshot["status"] != "healthy"
    assert snapshot["checked_at"] is None


# 7
@pytest.mark.asyncio
async def test_neo4j_unavailable_at_boot_is_down_not_null(clock, executor):
    registry = neo4j_registry(clock, executor, connection=None, configured=True)
    await registry.run_once()
    check = registry.snapshot()["checks"]["neo4j"]

    assert check["status"] == "down"
    assert check["reason"] == "unavailable_at_boot"

    # The mirror case: knowledge genuinely switched off IS a null.
    off = neo4j_registry(clock, executor, connection=None, configured=False)
    await off.run_once()
    off_check = off.snapshot()["checks"]["neo4j"]
    assert off_check["status"] is None
    assert off_check["reason"] == "disabled_by_config"


# 8
@pytest.mark.asyncio
async def test_stale_is_not_counted_healthy(clock, executor):
    registry = neo4j_registry(clock, executor, FakeNeo4jConnection())
    await registry.run_once()
    assert registry.snapshot()["status"] == "healthy"

    clock.advance(31.0)
    snapshot = registry.snapshot()

    assert snapshot["checks"]["neo4j"]["stale"] is True
    assert snapshot["checks"]["neo4j"]["age_seconds"] == 31.0
    assert snapshot["status"] == "degraded"


# 9
@pytest.mark.asyncio
async def test_probe_loop_death_makes_everything_stale(clock, executor):
    probes = [
        Neo4jProbe(lambda: FakeNeo4jConnection(), lambda: True, executor),
        LLMProbe(lambda: FakeLLMProvider()),
    ]
    registry = build_registry(clock, probes)
    await registry.run_once()
    assert registry.snapshot()["probe_loop"] == "alive"

    clock.advance(91.0)
    snapshot = registry.snapshot()

    assert all(check["stale"] is True for check in snapshot["checks"].values())
    assert snapshot["probe_loop"] == "dead"
    assert snapshot["restart_recommended"] is True
    # Stale is "unknown", not "down": we stopped measuring, we did not measure a
    # failure. So the serving verdict is degraded even though the loop is dead.
    assert snapshot["status"] == "degraded"


# 10
def _leaky_neo4j(executor):
    connection = FakeNeo4jConnection(
        result={"status": "unhealthy", "uri": SECRET_URI, "error": LEAKY_MESSAGE}
    )
    return Neo4jProbe(lambda: connection, lambda: True, executor)


def _leaky_llm(executor):
    provider = FakeLLMProvider(raises=RuntimeError(LEAKY_MESSAGE))
    return LLMProbe(lambda: provider)


def _leaky_event_store(executor):
    store = FakeEventStore(raises=RuntimeError(LEAKY_MESSAGE))
    return EventStoreProbe(lambda: store, lambda: True)


def _leaky_sidecar(executor):
    sidecar = FakeSidecar(raises=RuntimeError(LEAKY_MESSAGE))
    return VaultSidecarProbe(lambda: sidecar)


def _leaky_broadcaster(executor):
    def get_state():
        raise RuntimeError(LEAKY_MESSAGE)

    return BroadcasterProbe(get_state)


def _leaky_broadcaster_state(executor):
    state = BroadcasterState()
    state.mark_dead(RuntimeError(LEAKY_MESSAGE))
    return BroadcasterProbe(state.state)


def _off_vocabulary_probe(executor):
    async def fail_with_detail():
        raise ProbeFailed(LEAKY_MESSAGE)

    return StubProbe("off_vocabulary", fail_with_detail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "build_probe",
    [
        _leaky_neo4j,
        _leaky_llm,
        _leaky_event_store,
        _leaky_sidecar,
        _leaky_broadcaster,
        _leaky_broadcaster_state,
        _off_vocabulary_probe,
    ],
)
async def test_no_infrastructure_detail_leaks(clock, executor, build_probe):
    probe = build_probe(executor)
    registry = build_registry(clock, [probe])

    await registry.run_once()
    snapshot = registry.snapshot()
    payload = json.dumps(snapshot)

    assert "hunter2" not in payload
    assert SECRET_URI not in payload
    assert SECRET_PATH not in payload
    assert "mist-neo4j" not in payload
    assert "7687" not in payload

    check = snapshot["checks"][probe.name]
    assert check["status"] == "down"
    assert check["reason"] in {"unreachable", "query_failed", "task_dead"}
    if probe.name == "off_vocabulary":
        # A probe that supplies its own detailed reason is coerced structurally.
        assert check["reason"] == "query_failed"


# 11
def _check(
    status,
    *,
    stale=False,
    severity="degrades",
    restart_repairs=False,
    failures=0,
):
    return {
        "status": status,
        "reason": None,
        "age_seconds": 1.0,
        "stale": stale,
        "latency_ms": 1.0,
        "consecutive_failures": failures,
        "severity": severity,
        "restart_repairs": restart_repairs,
    }


def _recompute(checks: dict[str, dict], probe_loop: str, settled: int = 5):
    """An independently written derivation, set-based rather than branch-based."""
    down = {n for n, c in checks.items() if c["status"] in {"down", "timeout"}}
    unknown = {
        n
        for n, c in checks.items()
        if n not in down and (c["status"] is None or c["stale"] is True)
    }
    fatal_down = {n for n in down if checks[n]["severity"] == "fatal"}

    if fatal_down:
        status = "unhealthy"
    elif down | unknown:
        status = "degraded"
    else:
        status = "healthy"

    settled_restarts = {
        n
        for n in down
        if checks[n]["restart_repairs"] and checks[n]["consecutive_failures"] >= settled
    }
    return status, probe_loop == "dead" or bool(settled_restarts)


MATRIX = [
    ({"a": _check("up"), "b": _check("up", severity="fatal")}, "alive"),
    ({"a": _check("down"), "b": _check("up", severity="fatal")}, "alive"),
    ({"a": _check("up"), "b": _check("down", severity="fatal")}, "alive"),
    ({"a": _check("up"), "b": _check("timeout", severity="fatal")}, "alive"),
    ({"a": _check(None), "b": _check("up", severity="fatal")}, "alive"),
    ({"a": _check("up", stale=True), "b": _check("up", severity="fatal")}, "alive"),
    ({"a": _check("timeout"), "b": _check(None, severity="fatal")}, "alive"),
    ({"a": _check("up"), "b": _check("up", severity="fatal")}, "dead"),
    (
        {"a": _check("down", severity="fatal", restart_repairs=True, failures=5)},
        "alive",
    ),
    (
        {"a": _check("down", severity="fatal", restart_repairs=True, failures=4)},
        "alive",
    ),
    (
        {"a": _check("down", restart_repairs=False, failures=9)},
        "alive",
    ),
]


@pytest.mark.parametrize(("checks", "probe_loop"), MATRIX)
def test_top_level_status_is_derivable_from_the_parts(checks, probe_loop):
    assert derive_status(checks, probe_loop) == _recompute(checks, probe_loop)


def test_zero_checks_is_not_healthy():
    """The degenerate row of the matrix above, which `any()` gets wrong.

    Both `any(...)` terms are vacuously false over an empty mapping, so an
    unguarded derivation calls a registry that probed nothing healthy.
    """
    assert derive_status({}, "alive") == ("unhealthy", False)


# 12
@pytest.mark.asyncio
async def test_the_probe_never_calls_connect(clock, executor):
    """Reconnect is out of scope, and the isolation guard would 500 the endpoint.

    `Neo4jConnection.connect` calls `assert_neo4j_isolated`, which raises
    `EvalIsolationError` -- a `RuntimeError`, not a `MistError` -- so
    `Neo4jConnection.health_check` would not catch it. The whole unit tier runs
    under `MIST_EVAL_ISOLATION=1`.
    """
    connection = FakeNeo4jConnection()
    registry = neo4j_registry(clock, executor, connection)

    await registry.run_once()

    assert registry.snapshot()["checks"]["neo4j"]["status"] == "up"
    assert connection.connect_calls == 0
    assert connection.health_check_calls == 1


# 13
@pytest.mark.asyncio
async def test_a_hung_probe_does_not_stack(clock, executor):
    """Single-flight: a timeout abandons the work, so it must not be restarted.

    `asyncio.wait_for` does not stop a `run_in_executor` thread. Re-probing a
    wedged dependency every tick would leak one worker per tick.
    """

    async def hang():
        await asyncio.sleep(3600)

    async def fine():
        return None

    hung = StubProbe("hung", hang, timeout_seconds=0.02)
    healthy = StubProbe("healthy", fine)
    registry = build_registry(clock, [hung, healthy])

    await registry.run_once()
    clock.advance(15.0)
    await registry.run_once()
    clock.advance(16.0)
    await registry.run_once()

    checks = registry.snapshot()["checks"]

    assert hung.calls == 1
    assert healthy.calls == 3
    assert checks["hung"]["status"] == "timeout"
    assert checks["hung"]["age_seconds"] == 31.0
    assert checks["hung"]["stale"] is True
    assert checks["healthy"]["status"] == "up"
    assert checks["healthy"]["age_seconds"] == 0.0
    assert checks["healthy"]["stale"] is False
    cancel_stragglers(registry)


# 14
@pytest.mark.asyncio
async def test_a_fresh_broadcaster_state_is_null_not_alive(clock):
    state = BroadcasterState()
    assert state.state() is None

    registry = build_registry(clock, [BroadcasterProbe(state.state)])
    await registry.run_once()
    snapshot = registry.snapshot()

    assert snapshot["checks"]["broadcaster"]["status"] is None
    # Not `no_probe_implemented`: the probe is implemented and it ran. What it
    # found is that the broadcast task has not reported in yet.
    assert snapshot["checks"]["broadcaster"]["reason"] == "not_started"
    assert snapshot["status"] != "healthy"

    state.mark_alive()
    await registry.run_once()
    assert registry.snapshot()["checks"]["broadcaster"]["status"] == "up"


# 15
@pytest.mark.asyncio
async def test_broadcaster_death_settles_into_restart_advice(clock):
    state = BroadcasterState()
    state.mark_dead(RuntimeError(LEAKY_MESSAGE))
    assert state.state() == "dead"

    registry = build_registry(clock, [BroadcasterProbe(state.state)])
    for _ in range(4):
        await registry.run_once()

    snapshot = registry.snapshot()
    assert snapshot["checks"]["broadcaster"]["status"] == "down"
    assert snapshot["checks"]["broadcaster"]["reason"] == "task_dead"
    assert snapshot["status"] == "unhealthy"
    # Four consecutive failures is still inside the transient window.
    assert snapshot["checks"]["broadcaster"]["consecutive_failures"] == 4
    assert snapshot["restart_recommended"] is False

    await registry.run_once()
    settled = registry.snapshot()
    assert settled["checks"]["broadcaster"]["consecutive_failures"] == 5
    assert settled["restart_recommended"] is True


# 16
@pytest.mark.asyncio
async def test_event_store_none_is_down_when_configured(clock):
    """Configured plus absent is a failure that happened, so it is `down`.

    `conversation_handler.py:847` only builds the store when
    `EventStoreConfig.enabled`, and :898 nulls it again when initialize raised.
    With the flag on, the second is the only explanation left.
    """
    registry = build_registry(clock, [EventStoreProbe(lambda: None, lambda: True)])

    await registry.run_once()
    check = registry.snapshot()["checks"]["event_store"]

    assert check["status"] == "down"
    assert check["reason"] == "unavailable_at_boot"
    assert registry.snapshot()["status"] == "degraded"


# 17
@pytest.mark.asyncio
async def test_event_store_none_is_null_when_not_configured(clock):
    """Switched off on purpose is not a fault, and must not be reported as one.

    The mirror of `test_neo4j_unavailable_at_boot_is_down_not_null`: the same
    ambiguity, resolved the same way. Calling a deliberately disabled subsystem
    `down` claims a failure that did not occur, and does it in exactly the
    configuration where an operator already knows the thing is off.
    """
    registry = build_registry(clock, [EventStoreProbe(lambda: None, lambda: False)])

    await registry.run_once()
    check = registry.snapshot()["checks"]["event_store"]

    assert check["status"] is None
    assert check["reason"] == "disabled_by_config"
    assert check["latency_ms"] is None


# 18
@pytest.mark.asyncio
async def test_the_event_store_verdict_tracks_the_flag(clock):
    """Non-vacuity: one store accessor, both answers, inside one test.

    Neither branch above survives a probe that hardcodes its own verdict once
    the flag is flipped under a single registry.
    """
    configured = True
    registry = build_registry(clock, [EventStoreProbe(lambda: None, lambda: configured)])

    await registry.run_once()
    assert registry.snapshot()["checks"]["event_store"]["status"] == "down"

    configured = False
    await registry.run_once()
    assert registry.snapshot()["checks"]["event_store"]["status"] is None

    # And a store that IS there is `up` regardless of the flag, because the
    # predicate is consulted only when the accessor returns None.
    store = FakeEventStore()
    present = build_registry(clock, [EventStoreProbe(lambda: store, lambda: False)])
    await present.run_once()
    assert present.snapshot()["checks"]["event_store"]["status"] == "up"
    assert store.calls == 1


# ---------------------------------------------------------------------------
# Extraction service, extraction backlog and backup probes
# ---------------------------------------------------------------------------

SERVICE_URL = "http://extraction.test:8090"
DEFAULT_SETTINGS = HealthProbeSettings()
# `HealthRegistry`'s default `settled_failures`: past it, a restart-repairable
# down check would recommend a restart.
SETTLED_FAILURES = 5
_HOUR_MS = 3600 * 1000


def _status(
    *, state: str = "idle", oldest_pending_age_ms: int | None = None, dead_lettered: int = 0
) -> ExtractionStatus:
    """An `ExtractionStatus` with only the fields the backlog rows read set."""
    return ExtractionStatus(
        state=state,  # type: ignore[arg-type]
        backlog_depth=3,
        apply_pending=0,
        dead_lettered=dead_lettered,
        oldest_pending_age_ms=oldest_pending_age_ms,
        service=ServiceStatus(
            reachable=True,
            location_label=None,
            model_id=None,
            extraction_version=None,
            contract_version=None,
            last_health_ms=None,
        ),
        last_job=None,
    )


class FakeDispatcher:
    """What the probes and `checked_extraction_status` read from a dispatcher."""

    def __init__(
        self,
        *,
        mode: str = "service",
        state: str = "idle",
        status: ExtractionStatus | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.mode = mode
        self.state = state
        self.status = status if status is not None else _status(state=state)
        self.raises = raises
        self.snapshot_calls = 0

    def snapshot(self) -> ExtractionStatus:
        self.snapshot_calls += 1
        if self.raises is not None:
            raise self.raises
        return self.status


class FakeExtractionService:
    """An `httpx.MockTransport` handler standing in for `GET /v1/health`."""

    def __init__(self, respond) -> None:
        self._respond = respond
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        assert request.url.path == "/v1/health"
        return self._respond(request)


def _answering(status: str = "ok", llm_reachable: bool = True):
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"status": status, "llm_reachable": llm_reachable, "uptime_s": 42.0}
        )

    return respond


def _refusing(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("[Errno 111] Connection refused", request=request)


def _enveloped_error(request: httpx.Request) -> httpx.Response:
    envelope = error_envelope(ErrorCode.MODEL_LOADING, "model still loading")
    return httpx.Response(503, json=envelope.model_dump(mode="json"))


def _invalid_body(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"not": "a health response"})


@pytest_asyncio.fixture
async def service_probe():
    """Build an `ExtractionServiceProbe` over a real `RemoteExtractionInference`.

    The inference talks to an in-process `httpx.MockTransport`, so the probe's
    reaction to each typed error is exercised through the real client mapping.
    """
    clients: list[httpx.AsyncClient] = []

    def build(service: FakeExtractionService, dispatcher, *, start_failed: bool = False):
        client = httpx.AsyncClient(transport=httpx.MockTransport(service))
        clients.append(client)
        inference = RemoteExtractionInference(client, SERVICE_URL)
        return ExtractionServiceProbe(inference, lambda: dispatcher, lambda: start_failed)

    yield build
    for client in clients:
        await client.aclose()


def backlog_probe(dispatcher, *, start_failed: bool = False, settings=DEFAULT_SETTINGS):
    return ExtractionBacklogProbe(
        lambda: dispatcher, checked_extraction_status, lambda: start_failed, settings
    )


async def run_single(clock: FakeClock, probe) -> dict:
    """Run one probe through a registry and return its rendered check."""
    registry = build_registry(clock, [probe])
    await registry.run_once()
    return registry.snapshot()["checks"][probe.name]


class TestExtractionServiceProbe:
    def test_declares_degrades_no_restart_and_a_timeout_above_its_client(self):
        probe = ExtractionServiceProbe(object(), lambda: None, lambda: False)

        assert probe.name == "extraction_service"
        assert probe.severity == "degrades"
        assert probe.restart_repairs is False
        assert probe.timeout_seconds == 2.5

    @pytest.mark.asyncio
    async def test_start_failed_is_down_unavailable_at_boot_without_a_call(
        self, clock, service_probe
    ):
        service = FakeExtractionService(_answering())
        probe = service_probe(service, FakeDispatcher(), start_failed=True)

        check = await run_single(clock, probe)

        assert (check["status"], check["reason"]) == ("down", "unavailable_at_boot")
        assert service.calls == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "dispatcher",
        [
            pytest.param(None, id="no-dispatcher"),
            pytest.param(FakeDispatcher(mode="off", state="disabled"), id="mode-off"),
        ],
    )
    async def test_no_dispatcher_or_mode_off_is_null_disabled_by_config(
        self, clock, service_probe, dispatcher
    ):
        service = FakeExtractionService(_answering())
        probe = service_probe(service, dispatcher)

        check = await run_single(clock, probe)

        assert (check["status"], check["reason"]) == (None, "disabled_by_config")
        assert service.calls == 0

    @pytest.mark.asyncio
    async def test_a_refused_connection_is_down_unreachable_and_degraded_without_restart(
        self, clock, service_probe
    ):
        """The extraction host off, with nothing queued: the dispatcher alone is idle."""
        service = FakeExtractionService(_refusing)
        probe = service_probe(service, FakeDispatcher(state="idle"))
        registry = build_registry(clock, [probe])

        for _ in range(SETTLED_FAILURES + 1):
            await registry.run_once()
        snapshot = registry.snapshot()

        check = snapshot["checks"]["extraction_service"]
        assert (check["status"], check["reason"]) == ("down", "unreachable")
        assert service.calls == SETTLED_FAILURES + 1
        assert derive_status(snapshot["checks"], snapshot["probe_loop"]) == ("degraded", False)
        assert snapshot["restart_recommended"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "respond",
        [
            pytest.param(_enveloped_error, id="service-error-envelope"),
            pytest.param(_invalid_body, id="response-invalid"),
        ],
    )
    async def test_a_service_error_or_invalid_reply_is_down_query_failed(
        self, clock, service_probe, respond
    ):
        probe = service_probe(FakeExtractionService(respond), FakeDispatcher())

        check = await run_single(clock, probe)

        assert (check["status"], check["reason"]) == ("down", "query_failed")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "respond",
        [
            pytest.param(_answering(status="loading"), id="status-loading"),
            pytest.param(_answering(status="degraded"), id="status-degraded"),
            pytest.param(_answering(llm_reachable=False), id="llm-unreachable"),
        ],
    )
    async def test_a_reply_that_is_not_ok_is_down_service_degraded(
        self, clock, service_probe, respond
    ):
        probe = service_probe(FakeExtractionService(respond), FakeDispatcher())

        check = await run_single(clock, probe)

        assert (check["status"], check["reason"]) == ("down", "service_degraded")

    @pytest.mark.asyncio
    async def test_a_dispatcher_in_epoch_mismatch_is_down_epoch_mismatch(
        self, clock, service_probe
    ):
        probe = service_probe(
            FakeExtractionService(_answering()), FakeDispatcher(state="epoch_mismatch")
        )

        check = await run_single(clock, probe)

        assert (check["status"], check["reason"]) == ("down", "epoch_mismatch")

    @pytest.mark.asyncio
    async def test_an_answering_service_with_an_idle_dispatcher_is_up(self, clock, service_probe):
        """The case the dispatcher alone cannot report: idle says nothing about the host."""
        service = FakeExtractionService(_answering())
        probe = service_probe(service, FakeDispatcher(state="idle"))

        check = await run_single(clock, probe)

        assert (check["status"], check["reason"]) == ("up", "ok")
        assert service.calls == 1


class TestExtractionBacklogProbe:
    def test_declares_degrades_no_restart_and_a_half_second_timeout(self):
        probe = backlog_probe(None)

        assert probe.name == "extraction_backlog"
        assert probe.severity == "degrades"
        assert probe.restart_repairs is False
        assert probe.timeout_seconds == 0.5

    @pytest.mark.asyncio
    async def test_start_failed_is_down_unavailable_at_boot(self, clock):
        dispatcher = FakeDispatcher()

        check = await run_single(clock, backlog_probe(dispatcher, start_failed=True))

        assert (check["status"], check["reason"]) == ("down", "unavailable_at_boot")
        assert dispatcher.snapshot_calls == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "dispatcher",
        [
            pytest.param(None, id="no-dispatcher"),
            pytest.param(FakeDispatcher(mode="off", state="disabled"), id="mode-off"),
        ],
    )
    async def test_no_dispatcher_or_mode_off_is_null_disabled_by_config(self, clock, dispatcher):
        check = await run_single(clock, backlog_probe(dispatcher))

        assert (check["status"], check["reason"]) == (None, "disabled_by_config")

    @pytest.mark.asyncio
    async def test_an_unreadable_backlog_is_down_query_failed_not_ok(self, clock):
        """`extraction_status()` reports zero counts here; the probe must not read that as ok."""
        dispatcher = FakeDispatcher(raises=sqlite3.OperationalError("database is locked"))

        check = await run_single(clock, backlog_probe(dispatcher))

        assert (check["status"], check["reason"]) == ("down", "query_failed")
        assert dispatcher.snapshot_calls == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "reason"),
        [
            pytest.param(_status(state="stalled"), "stalled", id="stalled"),
            pytest.param(
                _status(state="stalled", dead_lettered=2), "stalled", id="stalled-beats-dead"
            ),
            pytest.param(
                _status(oldest_pending_age_ms=6 * _HOUR_MS + 1), "backlog_overdue", id="overdue"
            ),
            pytest.param(
                _status(oldest_pending_age_ms=7 * _HOUR_MS, dead_lettered=1),
                "backlog_overdue",
                id="overdue-beats-dead",
            ),
            pytest.param(_status(dead_lettered=1), "dead_lettered", id="dead-lettered"),
        ],
    )
    async def test_rows_are_evaluated_in_order(self, clock, status, reason):
        dispatcher = FakeDispatcher(state=status.state, status=status)

        check = await run_single(clock, backlog_probe(dispatcher))

        assert (check["status"], check["reason"]) == ("down", reason)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [
            pytest.param(_status(), id="empty"),
            pytest.param(_status(oldest_pending_age_ms=6 * _HOUR_MS), id="exactly-at-max-age"),
            pytest.param(_status(state="working", oldest_pending_age_ms=1000), id="working"),
        ],
    )
    async def test_otherwise_up_ok(self, clock, status):
        dispatcher = FakeDispatcher(state=status.state, status=status)

        check = await run_single(clock, backlog_probe(dispatcher))

        assert (check["status"], check["reason"]) == ("up", "ok")

    @pytest.mark.asyncio
    async def test_the_max_age_comes_from_the_injected_settings(self, clock):
        dispatcher = FakeDispatcher(status=_status(oldest_pending_age_ms=61_000))
        settings = HealthProbeSettings(backlog_max_age_s=60.0)

        check = await run_single(clock, backlog_probe(dispatcher, settings=settings))

        assert check["reason"] == "backlog_overdue"


def _backup_status(**overrides) -> dict:
    """A status file body with every key item 6 names."""
    body = {
        "v": 1,
        "last_attempt_at": "2026-09-22T22:00:00Z",
        "last_result": "ok",
        "last_exit_code": 0,
        "last_success_at": "2026-09-22T22:00:00Z",
        "last_label": "2026-09-22T22-00-00Z",
        "last_duration_s": 12.5,
        "retained": 3,
        "last_success_counts": {"graph_nodes": 10, "vault_corpus_files": 4, "stores": []},
    }
    body.update(overrides)
    return body


def _hours_before(clock: FakeClock, hours: float) -> str:
    return (clock.wall() - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def backup_probe(clock: FakeClock, path: Path | None, **settings) -> BackupProbe:
    return BackupProbe(HealthProbeSettings(backup_status_path=path, **settings), clock.wall)


def write_status(tmp_path: Path, body) -> Path:
    path = tmp_path / "backup-status.json"
    path.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    return path


class TestBackupProbe:
    def test_declares_degrades_no_restart_and_a_half_second_timeout(self, clock):
        probe = backup_probe(clock, None)

        assert probe.name == "backup"
        assert probe.severity == "degrades"
        assert probe.restart_repairs is False
        assert probe.timeout_seconds == 0.5

    @pytest.mark.asyncio
    async def test_no_status_path_is_null_disabled_by_config(self, clock):
        check = await run_single(clock, backup_probe(clock, None))

        assert (check["status"], check["reason"]) == (None, "disabled_by_config")

    @pytest.mark.asyncio
    async def test_an_absent_file_is_down_backup_never_run(self, clock, tmp_path):
        check = await run_single(clock, backup_probe(clock, tmp_path / "absent.json"))

        assert (check["status"], check["reason"]) == ("down", "backup_never_run")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body",
        [
            pytest.param("{not json", id="unparsable"),
            pytest.param("[1, 2]", id="not-an-object"),
            pytest.param(
                {k: v for k, v in _backup_status().items() if k != "last_success_counts"},
                id="missing-key",
            ),
            pytest.param(_backup_status(last_success_at="yesterday"), id="bad-timestamp"),
            pytest.param(_backup_status(last_result=7), id="non-string-result"),
        ],
    )
    async def test_an_unreadable_status_is_down_query_failed(self, clock, tmp_path, body):
        path = write_status(tmp_path, body)

        check = await run_single(clock, backup_probe(clock, path))

        assert (check["status"], check["reason"]) == ("down", "query_failed")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("result", "success_hours", "reason"),
        [
            pytest.param("suspect", 30, "backup_suspect", id="suspect-30h"),
            pytest.param("suspect", None, "backup_suspect", id="suspect-never-succeeded"),
            pytest.param("failed", 30, "backup_failed", id="failed-30h"),
            pytest.param("refused", None, "backup_failed", id="refused-never-succeeded"),
            pytest.param("ok", 30, "backup_overdue", id="ok-but-overdue"),
        ],
    )
    async def test_stale_rows_are_evaluated_in_order(
        self, clock, tmp_path, result, success_hours, reason
    ):
        success_at = None if success_hours is None else _hours_before(clock, success_hours)
        path = write_status(
            tmp_path, _backup_status(last_result=result, last_success_at=success_at)
        )

        check = await run_single(clock, backup_probe(clock, path))

        assert (check["status"], check["reason"]) == ("down", reason)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "result",
        [
            pytest.param("ok", id="ok"),
            # The last attempt failed, but the last success is still inside the
            # max age: the backup is current, and the next attempt is the
            # scheduler's business.
            pytest.param("failed", id="failed-but-recent-success"),
            pytest.param("suspect", id="suspect-but-recent-success"),
        ],
    )
    async def test_a_recent_success_is_up_ok(self, clock, tmp_path, result):
        path = write_status(
            tmp_path, _backup_status(last_result=result, last_success_at=_hours_before(clock, 1))
        )

        check = await run_single(clock, backup_probe(clock, path))

        assert (check["status"], check["reason"]) == ("up", "ok")

    @pytest.mark.asyncio
    async def test_the_max_age_comes_from_the_injected_settings(self, clock, tmp_path):
        path = write_status(tmp_path, _backup_status(last_success_at=_hours_before(clock, 3)))

        check = await run_single(clock, backup_probe(clock, path, backup_max_age_hours=2.0))

        assert check["reason"] == "backup_overdue"

    @pytest.mark.asyncio
    async def test_ok_with_no_success_on_record_is_overdue(self, clock, tmp_path):
        """No success within the max age means not backed up, whatever `last_result` says."""
        path = write_status(tmp_path, _backup_status(last_result="ok", last_success_at=None))

        check = await run_single(clock, backup_probe(clock, path))

        assert (check["status"], check["reason"]) == ("down", "backup_overdue")


@pytest.mark.asyncio
async def test_file_backed_probes_answer_while_a_single_worker_executor_is_blocked(clock, tmp_path):
    """They read through `asyncio.to_thread`, never the Neo4j probe's executor.

    The graph probe's pool has one worker so a wedged driver cannot leak
    threads. A probe that queued behind it would read a wedged graph as an
    extraction or backup timeout.
    """
    assert "executor" not in inspect.signature(ExtractionBacklogProbe).parameters
    assert "executor" not in inspect.signature(BackupProbe).parameters
    blocked = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    blocked.submit(release.wait)
    path = write_status(tmp_path, _backup_status(last_success_at=_hours_before(clock, 1)))

    async def graph_call_behind_the_wedge() -> None:
        # Stands in for `Neo4jProbe`, whose 2 s timeout would only slow this down.
        await asyncio.get_running_loop().run_in_executor(blocked, lambda: None)

    probes = [
        StubProbe("graph", graph_call_behind_the_wedge, timeout_seconds=0.05),
        backlog_probe(FakeDispatcher()),
        backup_probe(clock, path),
    ]
    registry = build_registry(clock, probes)

    try:
        await registry.run_once()
        checks = registry.snapshot()["checks"]
    finally:
        cancel_stragglers(registry)
        release.set()
        blocked.shutdown(wait=True)

    assert checks["graph"]["status"] == "timeout"
    assert checks["extraction_backlog"]["status"] == "up"
    assert checks["backup"]["status"] == "up"


class TestHealthProbeSettings:
    def test_defaults_match_the_contract(self):
        settings = HealthProbeSettings.from_env({})

        assert settings == HealthProbeSettings(
            backlog_max_age_s=21600.0, backup_status_path=None, backup_max_age_hours=26.0
        )

    def test_reads_each_variable(self):
        settings = HealthProbeSettings.from_env(
            {
                "MIST_HEALTH_BACKLOG_MAX_AGE_S": "60",
                "MIST_BACKUP_STATUS_PATH": "/app/ops/backup-status.json",
                "MIST_BACKUP_MAX_AGE_HOURS": "48",
            }
        )

        assert settings.backlog_max_age_s == 60.0
        assert settings.backup_status_path == Path("/app/ops/backup-status.json")
        assert settings.backup_max_age_hours == 48.0

    @pytest.mark.parametrize(
        "raw",
        [pytest.param("soon", id="not-a-number"), pytest.param("0", id="zero")],
    )
    def test_a_malformed_age_keeps_its_default_and_is_logged(self, raw, caplog):
        with caplog.at_level(logging.ERROR, logger="backend.health"):
            settings = HealthProbeSettings.from_env({"MIST_BACKUP_MAX_AGE_HOURS": raw})

        assert settings.backup_max_age_hours == 26.0
        assert any("MIST_BACKUP_MAX_AGE_HOURS" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        ("name", "raw", "field", "default"),
        [
            pytest.param(
                "MIST_BACKUP_MAX_AGE_HOURS", "inf", "backup_max_age_hours", 26.0, id="hours-inf"
            ),
            pytest.param(
                "MIST_BACKUP_MAX_AGE_HOURS", "nan", "backup_max_age_hours", 26.0, id="hours-nan"
            ),
            pytest.param(
                "MIST_BACKUP_MAX_AGE_HOURS",
                "1e20",
                "backup_max_age_hours",
                26.0,
                id="hours-beyond-timedelta",
            ),
            pytest.param(
                "MIST_HEALTH_BACKLOG_MAX_AGE_S", "inf", "backlog_max_age_s", 21600.0, id="s-inf"
            ),
            pytest.param(
                "MIST_HEALTH_BACKLOG_MAX_AGE_S", "nan", "backlog_max_age_s", 21600.0, id="s-nan"
            ),
        ],
    )
    def test_a_non_finite_or_unrepresentable_age_keeps_its_default(
        self, clock, name, raw, field, default, caplog
    ):
        """`timedelta(hours=inf)` raises `OverflowError` in `lifespan`, so the boot would fail."""
        with caplog.at_level(logging.ERROR, logger="backend.health"):
            settings = HealthProbeSettings.from_env({name: raw})

        assert getattr(settings, field) == default
        assert any(name in r.getMessage() for r in caplog.records)
        BackupProbe(settings, clock.wall)

    def test_the_largest_accepted_backup_age_still_builds_a_probe(self, clock):
        largest = str(timedelta.max.days * 24)

        settings = HealthProbeSettings.from_env({"MIST_BACKUP_MAX_AGE_HOURS": largest})

        assert settings.backup_max_age_hours == float(largest)
        BackupProbe(settings, clock.wall)


def test_new_reasons_are_in_the_closed_vocabulary():
    for reason in (
        "epoch_mismatch",
        "service_degraded",
        "stalled",
        "dead_lettered",
        "backlog_overdue",
        "backup_never_run",
        "backup_failed",
        "backup_suspect",
        "backup_overdue",
    ):
        assert reason in PROBE_REASONS, reason
