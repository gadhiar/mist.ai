"""Tests for the wiring between `backend.server` and the health registry.

`tests/unit/test_health_probes.py` covers the registry in isolation. What is
left is everything that can only be wrong at the seam: a readiness flag set
before the thing it describes, a done-callback that mistakes a clean shutdown
for a crash, an accessor that captured a value instead of reading one, and an
endpoint that answers with a status code the hydrator cannot distinguish from a
dead socket.

No test here boots the ASGI lifespan. The pieces `lifespan` calls are module
level for exactly that reason, so the sequence and the wiring are assertable
without faking the model stack.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from backend import server
from backend.health import (
    BroadcasterProbe,
    BroadcasterState,
    HealthRegistry,
    ProbeFailed,
    ProbeUnavailable,
)

_GLOBALS = (
    "_models_ready",
    "_broadcaster_state",
    "_health_registry",
    "voice_processor",
    "vault_sidecar",
)


@pytest.fixture(autouse=True)
def restore_server_globals():
    """Save and restore every module global these tests rebind.

    `backend.server` is imported once per session, so a test that leaves
    `_health_registry` set would change what `/health` returns for every later
    test in the tier, including `tests/unit/hydration/`.
    """
    saved = {name: getattr(server, name) for name in _GLOBALS}
    server._models_ready = False
    server._broadcaster_state = None
    server._health_registry = None
    yield
    for name, value in saved.items():
        setattr(server, name, value)


@pytest.fixture
def executor():
    pool = ThreadPoolExecutor(max_workers=1)
    yield pool
    pool.shutdown(wait=False)


def build_registry(probes: list) -> HealthRegistry:
    """A registry on the real clocks. Nothing here waits for time to pass."""
    return HealthRegistry(probes, monotonic=time.monotonic, wall_clock=lambda: datetime.now(UTC))


class _StaticProbe:
    """A probe whose outcome is fixed by the test."""

    def __init__(self, name: str, behaviour, severity: str = "degrades") -> None:
        self.name = name
        self.severity = severity
        self.restart_repairs = False
        self.timeout_seconds = 0.5
        self._behaviour = behaviour

    async def check(self) -> None:
        await self._behaviour()


class FakeSidecar:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy
        self.calls = 0

    def health_check(self) -> bool:
        self.calls += 1
        return self.healthy


class FakeLLMProvider:
    def __init__(self, healthy: bool = True) -> None:
        self.healthy = healthy
        self.calls = 0

    async def health_check(self) -> bool:
        self.calls += 1
        return self.healthy


class FakeVoiceProcessorHoldingLLM:
    """Only what `_health_llm_provider` reads. No models, no VAD, no torch."""

    def __init__(self, provider) -> None:
        self._llm_provider = provider


async def settle() -> None:
    """Let a done-callback run: `add_done_callback` fires via `call_soon`."""
    await asyncio.sleep(0)


# 1
@pytest.mark.asyncio
async def test_models_loaded_is_false_during_initialize():
    """The shipped defect: `models_loaded` was true throughout the model load.

    `lifespan` binds `voice_processor` and THEN awaits `initialize()`, which
    takes roughly 120s of model loading. The old field was
    `voice_processor is not None`, so it reported loaded for that entire window
    -- true of the assignment, false of the system. The flag must not flip until
    `initialize()` has returned.
    """
    initializing = asyncio.Event()
    release = asyncio.Event()

    class FakeVoiceProcessor:
        async def initialize(self) -> None:
            initializing.set()
            await release.wait()

    processor = FakeVoiceProcessor()
    # Bound exactly as `lifespan` binds it: before initialize is awaited.
    server.voice_processor = processor
    task = asyncio.create_task(server._initialize_voice_processor(processor))

    await initializing.wait()
    mid = await server.health()
    assert mid["models_loaded"] is False
    assert server.voice_processor is not None

    release.set()
    await task
    after = await server.health()
    assert after["models_loaded"] is True
    # A plain bool, not a string or an enum: `scripts/hydration/target.py` and
    # the hydration tests read this field.
    assert isinstance(after["models_loaded"], bool)


# 2
@pytest.mark.asyncio
async def test_broadcaster_death_is_logged_and_surfaced(caplog):
    """A raising broadcaster is logged with a traceback and shows up in /health."""
    state = BroadcasterState()
    state.mark_alive()
    server._broadcaster_state = state

    async def explode() -> None:
        raise RuntimeError("broadcast exploded")

    task = asyncio.create_task(explode())
    task.add_done_callback(server._on_broadcaster_done)
    with caplog.at_level(logging.ERROR, logger="backend.server"):
        with pytest.raises(RuntimeError):
            await task
        await settle()

    server_records = [
        record
        for record in caplog.records
        if record.name == "backend.server" and record.levelno == logging.ERROR
    ]
    assert server_records, "the death of the broadcaster must not be silent"
    # The traceback is the point: without exc_info the log names a failure it
    # cannot help anyone locate.
    assert any(record.exc_info is not None for record in server_records)
    assert state.state() == "dead"

    registry = build_registry([BroadcasterProbe(state.state)])
    for _ in range(5):
        await registry.run_once()
    server._health_registry = registry

    payload = await server.health()
    assert payload["checks"]["broadcaster"]["status"] == "down"
    assert payload["checks"]["broadcaster"]["reason"] == "task_dead"
    assert payload["status"] == "unhealthy"
    assert payload["restart_recommended"] is True


# 3
@pytest.mark.asyncio
async def test_clean_shutdown_does_not_report_death():
    """The anti-false-alarm case, and the reason the callback guards `cancelled`.

    `lifespan` calls `broadcaster_task.cancel()` on every orderly shutdown. A
    cancelled task fires its done-callback like any other, and
    `task.exception()` RAISES `CancelledError` on one rather than returning it.
    An unguarded callback would therefore mark the broadcaster dead every time
    the server stopped correctly, and `/health` would report `unhealthy` with
    `restart_recommended: true` as a consequence of shutting down properly.
    """
    state = BroadcasterState()
    state.mark_alive()
    server._broadcaster_state = state

    async def forever() -> None:
        await asyncio.sleep(3600)

    task = asyncio.create_task(forever())
    task.add_done_callback(server._on_broadcaster_done)
    await settle()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await settle()

    assert state.state() == "alive"
    assert state.last_exception() is None

    registry = build_registry([BroadcasterProbe(state.state)])
    for _ in range(5):
        await registry.run_once()
    server._health_registry = registry

    payload = await server.health()
    assert payload["checks"]["broadcaster"]["status"] == "up"
    assert payload["restart_recommended"] is False


# 4
@pytest.mark.asyncio
async def test_broadcaster_returning_normally_is_death():
    """`broadcast_messages` is a `while True`; returning is a fault, not a stop.

    Nothing drains `message_queue` after it returns, exactly as if it had
    raised, so the two must report the same way. Treating a clean return as
    health would leave the outage invisible in the one case where there is no
    exception to log.
    """
    state = BroadcasterState()
    state.mark_alive()
    server._broadcaster_state = state

    async def returns_immediately() -> None:
        return None

    task = asyncio.create_task(returns_immediately())
    task.add_done_callback(server._on_broadcaster_done)
    await task
    await settle()

    assert state.state() == "dead"
    assert state.last_exception() is None


def _healthy_registry() -> HealthRegistry:
    async def fine() -> None:
        return None

    return build_registry([_StaticProbe("fine", fine)])


def _degraded_registry() -> HealthRegistry:
    async def absent() -> None:
        raise ProbeUnavailable("disabled_by_config")

    async def fine() -> None:
        return None

    return build_registry([_StaticProbe("absent", absent), _StaticProbe("fine", fine)])


def _unhealthy_registry() -> HealthRegistry:
    async def down() -> None:
        raise ProbeFailed("unreachable")

    return build_registry([_StaticProbe("down", down, severity="fatal")])


# 5
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("build", "expected_status"),
    [
        (_healthy_registry, "healthy"),
        (_degraded_registry, "degraded"),
        (_unhealthy_registry, "unhealthy"),
        # No registry at all: `lifespan` has not run. Nothing measured is not
        # healthy, and it is still an answer rather than an error.
        (None, "unhealthy"),
    ],
)
async def test_health_is_always_200(build, expected_status):
    """Never a non-2xx, in any state, because the body carries the verdict.

    `scripts/hydration/target.py:91-100` fetches this under
    `except (urllib.error.URLError, OSError)`, and `HTTPError` is a subclass of
    `URLError`. A 503 for a degraded backend would therefore reach the operator
    as `HydrationTargetError("could not be reached")` -- a false claim about the
    socket, from a backend that answered. Returning a dict is what makes
    FastAPI serialize it with a 200; raising `HTTPException` is what would not.
    """
    if build is not None:
        registry = build()
        await registry.run_once()
        server._health_registry = registry

    payload = await server.health()

    assert isinstance(payload, dict)
    assert payload["status"] == expected_status
    # Present in every state: the hydrator's handshake must not depend on the
    # backend being well.
    assert "hydration_isolation" in payload
    assert isinstance(payload["hydration_isolation"], bool)
    assert "restart_recommended" in payload
    assert "checks" in payload


# 6
@pytest.mark.asyncio
async def test_accessors_are_lazy(executor):
    """Probes must read module state when they run, not when they were built.

    `lifespan` rebinds `voice_processor` and `vault_sidecar`, and the objects
    behind them are built at different points in startup. An accessor that
    captured a value at construction would freeze whatever was there -- and a
    captured None freezes the probe at "absent" for the life of the process,
    reporting an outage that has ended.
    """
    server.voice_processor = None
    server.vault_sidecar = None

    registry = server._build_health_registry(executor, BroadcasterState())

    provider = FakeLLMProvider()
    sidecar = FakeSidecar()
    server.voice_processor = FakeVoiceProcessorHoldingLLM(provider)
    server.vault_sidecar = sidecar

    await registry.run_once()
    checks = registry.snapshot()["checks"]

    assert checks["llm"]["status"] == "up"
    assert provider.calls == 1
    assert checks["vault_sidecar"]["status"] == "up"
    assert sidecar.calls == 1

    # And the reverse direction, which a lazily-read accessor also has to get
    # right: clearing the global must take the probe back to absent.
    server.vault_sidecar = None
    await registry.run_once()
    assert registry.snapshot()["checks"]["vault_sidecar"]["status"] is None


# 7
@pytest.mark.asyncio
async def test_existing_fields_preserved(monkeypatch):
    """The three pre-registry fields survive, with a registry and without.

    `scripts/hydration/target.py` reads `hydration_isolation`, and
    `tests/unit/hydration/` pins all three. The registry is an addition on top
    of them, not a replacement for the payload.
    """
    monkeypatch.delenv("MIST_HYDRATION_ISOLATION", raising=False)

    without_registry = await server.health()
    for field in ("models_loaded", "active_connections", "hydration_isolation"):
        assert field in without_registry, field

    registry = _healthy_registry()
    await registry.run_once()
    server._health_registry = registry

    with_registry = await server.health()
    for field in ("models_loaded", "active_connections", "hydration_isolation"):
        assert field in with_registry, field
    assert with_registry["models_loaded"] is False
    assert with_registry["active_connections"] == len(server.active_connections)
    assert with_registry["hydration_isolation"] is False
    assert with_registry["status"] == "healthy"
