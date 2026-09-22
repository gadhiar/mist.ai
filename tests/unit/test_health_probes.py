"""Tests for the health probe registry.

These tests are written against the failure modes a health endpoint actually
has: a probe that reports up without touching anything, a cached result served
as fresh, a timeout collapsed into a failure, a never-probed dependency rendered
as healthy, and infrastructure detail leaking into an unauthenticated response.
The clock is injected everywhere, so no test sleeps to make time pass.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from backend.health import (
    BroadcasterProbe,
    BroadcasterState,
    EventStoreProbe,
    HealthRegistry,
    LLMProbe,
    Neo4jProbe,
    ProbeFailed,
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
    return EventStoreProbe(lambda: store)


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
