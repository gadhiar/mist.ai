"""Tests for the wiring between `backend.server` and the health registry.

`tests/unit/test_health_probes.py` covers the registry in isolation. What is
left is everything that can only be wrong at the seam: a readiness flag set
before the thing it describes, a done-callback that mistakes a clean shutdown
for a crash, an accessor that captured a value instead of reading one, and an
endpoint that answers with a status code the hydrator cannot distinguish from a
dead socket.

Most tests here do not boot the ASGI lifespan. The pieces `lifespan` calls are
module level for exactly that reason, so the sequence and the wiring are
assertable without faking the model stack. The gaming-mode tests at the end
are the exception: what they pin is which object `lifespan` hands to the runtime
build, the session registry, the probes and the curation scheduler, so they run
`lifespan` with every builder that would reach the model stack, Neo4j or the
vault replaced (`build_mist_runtime` by a `RuntimeFactory`).

`test_health_is_always_200` uses `fastapi.testclient.TestClient`, and that is
the first use of it in `tests/` -- `grep -rln TestClient tests/` found nothing
before this module. It is deliberate, not an import that wandered in: the
property under test is an HTTP status code, and a test that only calls
`server.health()` cannot observe one. `TestClient` is constructed WITHOUT the
`with` block on purpose, because entering it would run the real ASGI lifespan
and load the model stack.
"""

import asyncio
import json
import logging
import socket
import subprocess  # nosec B404 - runs a fixed openssl argument list, never a shell
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import server
from backend.capabilities.power import FixedPower, GpuYieldPower
from backend.extraction_contract.models import HealthResponse
from backend.extraction_hold import EXTRACTION_PAUSED_MARKER
from backend.gpu_yield import GpuYield
from backend.health import (
    BroadcasterProbe,
    BroadcasterState,
    HealthProbeSettings,
    HealthRegistry,
    ProbeFailed,
    ProbeUnavailable,
)

_GLOBALS = (
    "_models_ready",
    "_broadcaster_state",
    "_health_registry",
    "runtime",
    "session_registry",
    "_capability_sink",
    "vault_sidecar",
    "extraction_dispatcher",
    "_extraction_dispatcher_start_failed",
    "_extraction_health_client",
    "_gpu_yield",
    "_device_listener",
    "_device_app",
    "_device_listener_state",
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
    server.extraction_dispatcher = None
    server._extraction_dispatcher_start_failed = False
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


class FakeRuntimeHoldingLLM:
    """Only what `_health_llm_provider` reads: the runtime's base provider. No models."""

    def __init__(self, provider) -> None:
        self.llm_provider = provider


class FakeGpu:
    """The runtime's GPU control (`GpuEngines`), recording what the server asks of it."""

    def __init__(self) -> None:
        self.attached: tuple[object, object] | None = None
        self.levels: list[str] = []
        self.calls: list[str] = []
        self.reload_error: BaseException | None = None

    def attach(self, loop, host) -> None:
        self.attached = (loop, host)

    def level_changed(self, level) -> None:
        self.levels.append(level)

    def release(self) -> int:
        self.calls.append("release")
        return 0

    def reload(self) -> None:
        self.calls.append("reload")
        if self.reload_error is not None:
            raise self.reload_error

    def stop(self) -> None:
        self.calls.append("stop")


def fake_mist_runtime(
    *,
    handler=None,
    provider=None,
    gpu: FakeGpu | None = None,
    power_gate=None,
    alarm_ringing=None,
    on_capability_message=None,
    closed: list[str] | None = None,
):
    """A real `MistRuntime` over fakes: no engine loaded, inline lanes, a recording GPU."""
    from backend.realtime.budget import VoiceSettings
    from backend.realtime.runtime import MistRuntime
    from backend.realtime.session import InferenceLanes, SessionServices, SpeechEngines
    from tests.unit.realtime.turn_fakes import FakeResponder, InlineLane, floor_placement

    class ClosingLane(InlineLane):
        def shutdown(self) -> None:
            if closed is not None:
                closed.append("lane")

    engines = SpeechEngines()
    lanes = InferenceLanes(
        vad=ClosingLane(),
        eot=ClosingLane(),
        stt=ClosingLane(),
        tts=ClosingLane(),
        kws=ClosingLane(),
        verify=ClosingLane(),
    )
    settings = VoiceSettings()
    responder = handler if handler is not None else FakeResponder()
    extra: dict[str, object] = {}
    if alarm_ringing is not None:
        extra["alarm_ringing"] = alarm_ringing
    if on_capability_message is not None:
        extra["on_capability_message"] = on_capability_message
    services = SessionServices(
        responder=responder,  # type: ignore[arg-type]
        engines=engines,
        lanes=lanes,
        settings=settings,
        placement=floor_placement,
        power_gate=power_gate,
        **extra,  # type: ignore[arg-type]
    )
    return MistRuntime(
        llm_provider=provider if provider is not None else FakeLLMProvider(),
        conversation_handler=handler,
        responder=responder,  # type: ignore[arg-type]
        engines=engines,
        lanes=lanes,
        settings=settings,
        services=services,
        agent_available=handler is not None,
        gpu=gpu if gpu is not None else FakeGpu(),  # type: ignore[arg-type]
    )


class RuntimeFactory:
    """`build_mist_runtime` in place: records its arguments and what boot looked like then.

    The build runs on a worker thread (`asyncio.to_thread`), so what it reads from the loop it
    reads through `run_coroutine_threadsafe`.
    """

    def __init__(self, handler=None, provider=None) -> None:
        self.handler = handler
        self.provider = provider
        self.calls: list[dict[str, object]] = []
        self.level_at_build: str | None = None
        self.gpu_yield_tasks_during_build: list[str] | None = None
        self.models_ready_during_build: bool | None = None
        self.runtime = None
        self.closed: list[str] = []
        self.loop = None

    def __call__(self, config, **kwargs):
        self.calls.append(kwargs)
        gate = kwargs["power_gate"]
        self.level_at_build = gate.level()
        self.models_ready_during_build = server._models_ready
        if self.loop is not None:

            async def yield_tasks() -> list[str]:
                return [t.get_name() for t in _gpu_yield_tasks()]

            future = asyncio.run_coroutine_threadsafe(yield_tasks(), self.loop)
            self.gpu_yield_tasks_during_build = future.result(timeout=5)
        self.runtime = fake_mist_runtime(
            handler=self.handler,
            provider=kwargs["llm_provider"],
            power_gate=gate,
            alarm_ringing=kwargs["alarm_ringing"],
            on_capability_message=kwargs["on_capability_message"],
            closed=self.closed,
        )
        return self.runtime


class FakeExtractionInference:
    """The one method `ExtractionServiceProbe` calls, answering ok."""

    def __init__(self) -> None:
        self.calls = 0

    async def health(self) -> HealthResponse:
        self.calls += 1
        return HealthResponse(status="ok", llm_reachable=True, uptime_s=5.0)


class FakeIdleDispatcher:
    """A service-mode dispatcher with nothing queued and a readable backlog."""

    mode = "service"
    state = "idle"

    def snapshot(self):
        from backend.extraction_backlog.status import extraction_status

        return extraction_status(None).model_copy(update={"state": "idle"})


def build_server_registry(
    executor,
    settings: HealthProbeSettings | None = None,
    yield_state=None,
    ops_dir=None,
    inference=None,
):
    """`_build_health_registry` with fakes for the extraction service and settings."""
    return server._build_health_registry(
        executor,
        BroadcasterState(),
        inference if inference is not None else FakeExtractionInference(),
        settings if settings is not None else HealthProbeSettings(),
        yield_state=yield_state,
        ops_dir=ops_dir,
    )


async def settle() -> None:
    """Let a done-callback run: `add_done_callback` fires via `call_soon`."""
    await asyncio.sleep(0)


# 1
@pytest.mark.asyncio
async def test_models_loaded_is_false_during_the_runtime_build(monkeypatch):
    """The shipped defect: `models_loaded` was true throughout the model load.

    The old field was "the voice processor global is set", bound before the load, so it
    reported loaded for that entire window -- true of the assignment, false of
    the system. The flag must not flip until the runtime build has returned, and
    the build runs off the loop, so `/health` answers while it loads.
    """
    building = threading.Event()
    release = threading.Event()
    provider = FakeLLMProvider()
    built = fake_mist_runtime()

    def blocking_factory(config, **kwargs):
        building.set()
        release.wait(timeout=10)
        return built

    monkeypatch.setattr(server, "build_llm_provider", lambda config, **kwargs: provider)
    task = asyncio.create_task(
        server._build_runtime(
            object(),  # type: ignore[arg-type]
            voice_settings=object(),  # type: ignore[arg-type]
            power_gate=object(),  # type: ignore[arg-type]
            alarm_ringing=lambda: False,
            register_prompt=None,
            factory=blocking_factory,
        )
    )

    assert await asyncio.to_thread(building.wait, 10)
    mid = await server.health()
    assert mid["models_loaded"] is False
    assert server.runtime is None

    release.set()
    assert await task is built
    after = await server.health()
    assert after["models_loaded"] is True
    assert server.runtime is built
    # A plain bool, not a string or an enum: `scripts/hydration/target.py` and
    # the hydration tests read this field.
    assert isinstance(after["models_loaded"], bool)


@pytest.mark.asyncio
async def test_the_runtime_is_built_with_the_base_provider_and_the_lifespan_wiring(monkeypatch):
    """`build_mist_runtime` gets the base provider (T39), the gate, D8's query and the persona."""
    provider = FakeLLMProvider()
    factory = RuntimeFactory()
    monkeypatch.setattr(server, "build_llm_provider", lambda config, **kwargs: provider)
    gate = server._build_power_gate(
        FixedPower("full"), server._capability_tiers(FixedPower("full")), None
    )
    ringing = lambda: False  # noqa: E731

    await server._build_runtime(
        object(),  # type: ignore[arg-type]
        voice_settings=object(),  # type: ignore[arg-type]
        power_gate=gate,
        alarm_ringing=ringing,
        register_prompt="Speak plainly.",
        factory=factory,
    )

    (kwargs,) = factory.calls
    assert kwargs["llm_provider"] is provider
    assert kwargs["power_gate"] is gate
    assert kwargs["alarm_ringing"] is ringing
    assert kwargs["register_prompt"] == "Speak plainly."
    assert kwargs["on_capability_message"] is server._on_capability_message
    assert factory.models_ready_during_build is False
    assert server._health_llm_provider() is provider


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

    What the guard is worth was measured, not reasoned about. Deleting the
    `if task.cancelled(): return` block in a scratch copy and driving a real
    cancelled task through the callback leaves the state `alive` and `/health`
    `healthy` with `restart_recommended: false`: the raise lands above
    `mark_dead`, and `asyncio.events.Handle._run` catches it and hands it to the
    loop's exception handler. The single observable consequence is one
    `Exception in callback _on_broadcaster_done(...)` per shutdown, so that is
    what this test pins -- the loop exception handler must never fire. The
    earlier version of this test asserted only `state()` and the payload, which
    the unguarded callback satisfies just as well; it passed against that
    mutant and pinned nothing.
    """
    state = BroadcasterState()
    state.mark_alive()
    server._broadcaster_state = state

    async def forever() -> None:
        await asyncio.sleep(3600)

    loop = asyncio.get_running_loop()
    routed_to_loop: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: routed_to_loop.append(context))
    try:
        task = asyncio.create_task(forever())
        task.add_done_callback(server._on_broadcaster_done)
        await settle()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await settle()
        await settle()
    finally:
        loop.set_exception_handler(previous_handler)

    # The mutation-sensitive assertion. Without the guard this list holds one
    # context whose `exception` is a `CancelledError` and whose `message` names
    # `_on_broadcaster_done`.
    assert routed_to_loop == [], (
        "a clean shutdown routed "
        f"{[(c.get('message'), type(c.get('exception')).__name__) for c in routed_to_loop]} "
        "to the event loop exception handler"
    )

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
def test_health_is_always_200(build, expected_status):
    """Never a non-2xx, in any state, because the body carries the verdict.

    `scripts/hydration/target.py:91-100` fetches this under
    `except (urllib.error.URLError, OSError)`, and `HTTPError` is a subclass of
    `URLError`. A 503 for a degraded backend would therefore reach the operator
    as `HydrationTargetError("could not be reached")` -- a false claim about the
    socket, from a backend that answered. Returning a dict is what makes
    FastAPI serialize it with a 200; raising `HTTPException` is what would not.

    Driven over real HTTP rather than by awaiting `server.health()`: the claim
    in the name is about a status code, and a direct await never produces one.
    This test is sync so that the blocking `TestClient` request does not run
    inside an already-running event loop.
    """
    if build is not None:
        registry = build()
        asyncio.run(registry.run_once())
        server._health_registry = registry

    response = TestClient(server.app).get("/health")

    assert response.status_code == 200
    payload = response.json()

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

    `lifespan` rebinds `runtime` and `vault_sidecar`, and the objects
    behind them are built at different points in startup. An accessor that
    captured a value at construction would freeze whatever was there -- and a
    captured None freezes the probe at "absent" for the life of the process,
    reporting an outage that has ended. The LLM probe reads the runtime's base
    provider, the one curation reads (T39).
    """
    server.runtime = None
    server.vault_sidecar = None

    registry = build_server_registry(executor)

    provider = FakeLLMProvider()
    sidecar = FakeSidecar()
    server.runtime = FakeRuntimeHoldingLLM(provider)
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
    assert with_registry["active_connections"] == server._session_count() == 0
    assert with_registry["hydration_isolation"] is False
    assert with_registry["status"] == "healthy"


@pytest.mark.asyncio
async def test_active_connections_counts_the_registrys_sessions():
    server.session_registry = SimpleNamespace(count=lambda: 3)

    payload = await server.health()

    assert payload["active_connections"] == 3
    assert server._session_count() == 3


@pytest.mark.asyncio
async def test_the_voice_block_reports_the_runtimes_engines_and_power_level():
    server.runtime = None
    before = await server.health()

    power = FixedPower("off")
    gate = server._build_power_gate(power, server._capability_tiers(power), None)
    server.runtime = fake_mist_runtime(power_gate=gate)
    after = await server.health()

    assert before["voice"] is None
    voice = after["voice"]
    assert set(voice) == {
        "engines",
        "tts_voice",
        "vram_budget_mib",
        "vram_over_budget",
        "power_level",
    }
    assert set(voice["engines"]) == {"vad", "eot", "stt", "tts", "kws"}
    assert voice["engines"]["stt"]["location"] == "off"
    assert voice["tts_voice"]["engine"] == "off"
    assert voice["vram_budget_mib"] == server.runtime.settings.placement.vram_budget_mib
    assert voice["vram_over_budget"] is False
    assert voice["power_level"] == "off"


# 8
@pytest.mark.asyncio
async def test_registry_carries_the_extraction_and_backup_checks(executor):
    """Three degrading checks join the five, none of which a restart repairs."""
    registry = build_server_registry(executor)

    checks = registry.snapshot()["checks"]

    assert set(checks) == {
        "neo4j",
        "llm",
        "event_store",
        "vault_sidecar",
        "broadcaster",
        "extraction_service",
        "extraction_backlog",
        "backup",
    }
    for name in ("extraction_service", "extraction_backlog", "backup"):
        assert checks[name]["severity"] == "degrades", name
        assert checks[name]["restart_repairs"] is False, name


# 9
@pytest.mark.asyncio
async def test_extraction_accessors_read_the_dispatcher_and_start_flag_lazily(executor):
    """The registry is built before `lifespan` starts the dispatcher.

    So both the dispatcher and the start-failed flag must be read when the
    probe runs: a captured None would report `disabled_by_config` for the life
    of the process, and a captured False would hide a failed start.
    """
    registry = build_server_registry(executor)

    server._extraction_dispatcher_start_failed = True
    await registry.run_once()
    failed = registry.snapshot()["checks"]
    server._extraction_dispatcher_start_failed = False
    server.extraction_dispatcher = FakeIdleDispatcher()
    await registry.run_once()
    running = registry.snapshot()["checks"]

    for name in ("extraction_service", "extraction_backlog"):
        assert (failed[name]["status"], failed[name]["reason"]) == (
            "down",
            "unavailable_at_boot",
        ), name
        assert (running[name]["status"], running[name]["reason"]) == ("up", "ok"), name


# 10
@pytest.mark.asyncio
async def test_backup_reads_the_injected_settings(executor, tmp_path):
    """Unset on the workstation stack, so `backup` is null there, not down."""
    unset = build_server_registry(executor)
    await unset.run_once()
    absent = build_server_registry(
        executor, HealthProbeSettings(backup_status_path=tmp_path / "absent.json")
    )
    await absent.run_once()

    assert unset.snapshot()["checks"]["backup"]["reason"] == "disabled_by_config"
    assert absent.snapshot()["checks"]["backup"]["reason"] == "backup_never_run"


# 11
@pytest.mark.asyncio
async def test_health_top_level_keys_and_extraction_block_are_unchanged(executor):
    """The new checks live under `checks`; `voice` (T13) is the one top-level field added."""
    registry = build_server_registry(executor)
    await registry.run_once()
    server._health_registry = registry

    payload = await server.health()

    assert set(payload) == {
        "status",
        "restart_recommended",
        "probe_loop",
        "probe_interval_seconds",
        "checked_at",
        "checks",
        "models_loaded",
        "active_connections",
        "hydration_isolation",
        "extraction",
        "voice",
        "devices",
    }
    assert payload["devices"] == {"listener": "off", "connected": []}
    extraction = payload["extraction"]
    assert set(extraction) == {
        "state",
        "backlog_depth",
        "dead_lettered",
        "unrecorded_turns",
        "service",
    }
    assert extraction["service"] == {"reachable": False}
    assert extraction["state"] == "disabled"


# 12
@pytest.mark.asyncio
async def test_a_dispatcher_start_that_raises_reads_unavailable_at_boot(executor):
    """The path `lifespan` takes when `_start_extraction_dispatcher` raises.

    Without the recorded flag a None dispatcher reads `disabled_by_config`,
    which would report a failed start as a deliberate switch-off.
    """

    async def failing_start():
        raise ValueError("MIST_EXTRACTION_MAX_ATTEMPTS must be >= 1, got 0")

    registry = build_server_registry(executor)

    dispatcher = await server._start_extraction_dispatcher_recording_failure(failing_start)
    server.extraction_dispatcher = dispatcher
    await registry.run_once()
    checks = registry.snapshot()["checks"]

    assert dispatcher is None
    assert server._extraction_dispatcher_start_failed is True
    for name in ("extraction_service", "extraction_backlog"):
        assert (checks[name]["status"], checks[name]["reason"]) == (
            "down",
            "unavailable_at_boot",
        ), name


# 13
@pytest.mark.asyncio
async def test_a_dispatcher_start_that_succeeds_clears_the_flag():
    started = FakeIdleDispatcher()

    async def start():
        return started

    server._extraction_dispatcher_start_failed = True

    dispatcher = await server._start_extraction_dispatcher_recording_failure(start)

    assert dispatcher is started
    assert server._extraction_dispatcher_start_failed is False


# 14
@pytest.mark.asyncio
async def test_the_probe_client_has_a_two_second_timeout_and_is_closed_at_shutdown():
    """Its own bound, independent of the dispatcher's 300 s extract client."""
    client = server._new_extraction_health_client()
    server._extraction_health_client = client

    assert client.timeout == httpx.Timeout(2.0)

    await server._close_extraction_health_client()

    assert client.is_closed
    assert server._extraction_health_client is None
    # A second shutdown call, or one with no client built, is a no-op.
    await server._close_extraction_health_client()


def test_extraction_service_url_follows_the_dispatcher_setting(monkeypatch):
    """The probe checks the same service the dispatcher sends jobs to."""
    monkeypatch.setenv("MIST_EXTRACTION_SERVICE_URL", "http://extraction-host:8090")

    assert server._extraction_service_url() == "http://extraction-host:8090"


def test_extraction_service_url_survives_a_malformed_dispatcher_setting(monkeypatch):
    """A bad `MIST_EXTRACTION_*` value fails the dispatcher's start, not the boot.

    The probe then reports `unavailable_at_boot` before it calls anything, so
    the URL it holds only has to be well formed.
    """
    monkeypatch.setenv("MIST_EXTRACTION_MAX_ATTEMPTS", "many")
    monkeypatch.setenv("MIST_EXTRACTION_SERVICE_URL", "http://extraction-host:8090")

    assert server._extraction_service_url() == "http://extraction-host:8090"


# -- Gaming mode ---------------------------------------------------------------


def _noop() -> None:
    return None


def _failing_resume() -> None:
    raise RuntimeError("no memory for Whisper")


def _marked(tmp_path) -> Path:
    marker = tmp_path / "gpu-yielded"
    marker.write_text('{"v": 1, "since": "2026-10-23T00:00:00Z", "by": "manual"}')
    return marker


@pytest.mark.asyncio
async def test_llm_check_reads_yielded_from_the_yield_state(executor, tmp_path):
    gpu = GpuYield(_marked(tmp_path), _noop, _noop, poll_s=2.0)
    registry = build_server_registry(executor, yield_state=gpu.state)
    server._health_registry = registry

    await registry.run_once()
    payload = await server.health()

    llm = payload["checks"]["llm"]
    assert (llm["status"], llm["reason"]) == ("yielded", "gpu_yielded")
    assert payload["status"] == "yielded"
    assert payload["restart_recommended"] is False


@pytest.mark.asyncio
async def test_llm_check_reads_down_when_the_reload_failed(executor, tmp_path):
    marker = _marked(tmp_path)
    gpu = GpuYield(marker, _noop, _failing_resume, poll_s=2.0)
    marker.unlink()
    await gpu.poll_once()
    registry = build_server_registry(executor, yield_state=gpu.state)

    await registry.run_once()

    llm = registry.snapshot()["checks"]["llm"]
    assert (llm["status"], llm["reason"]) == ("down", "resume_failed")


# -- Night extraction paused ---------------------------------------------------


def _paused_marker(ops_dir: Path, until: str | None = None) -> Path:
    marker = ops_dir / EXTRACTION_PAUSED_MARKER
    marker.write_text(
        json.dumps({"v": 1, "since": "2026-11-15T10:00:00Z", "reason": "window", "until": until}),
        encoding="utf-8",
    )
    return marker


@pytest.mark.asyncio
async def test_extraction_service_reads_yielded_from_the_yield_state(executor, tmp_path):
    """The same yield state reaches both GPU-side checks."""
    gpu = GpuYield(_marked(tmp_path), _noop, _noop, poll_s=2.0)
    inference = FakeExtractionInference()
    server.extraction_dispatcher = FakeIdleDispatcher()
    registry = build_server_registry(executor, yield_state=gpu.state, inference=inference)

    await registry.run_once()

    check = registry.snapshot()["checks"]["extraction_service"]
    assert (check["status"], check["reason"]) == ("yielded", "gpu_yielded")
    assert inference.calls == 0


@pytest.mark.asyncio
async def test_extraction_service_reads_paused_from_the_ops_dir_marker(executor, tmp_path):
    marker = _paused_marker(tmp_path)
    inference = FakeExtractionInference()
    server.extraction_dispatcher = FakeIdleDispatcher()
    registry = build_server_registry(executor, ops_dir=str(tmp_path), inference=inference)
    server._health_registry = registry

    await registry.run_once()
    payload = await server.health()
    # Read on every cycle, not captured at build time.
    marker.unlink()
    await registry.run_once()
    resumed = registry.snapshot()["checks"]["extraction_service"]

    check = payload["checks"]["extraction_service"]
    assert (check["status"], check["reason"]) == ("paused", "extraction_paused")
    assert check["consecutive_failures"] == 0
    assert (resumed["status"], resumed["reason"]) == ("up", "ok")
    assert inference.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ops_dir", [None, ""])
async def test_no_ops_dir_reads_no_marker(executor, tmp_path, monkeypatch, ops_dir):
    """The dev stack: a marker in the working directory is never looked for."""
    _paused_marker(tmp_path)
    monkeypatch.chdir(tmp_path)
    inference = FakeExtractionInference()
    server.extraction_dispatcher = FakeIdleDispatcher()
    registry = build_server_registry(executor, ops_dir=ops_dir, inference=inference)

    await registry.run_once()

    check = registry.snapshot()["checks"]["extraction_service"]
    assert (check["status"], check["reason"]) == ("up", "ok")
    assert inference.calls == 1


class FakeHandler:
    """The handler attributes `lifespan` reads, and a recording `attach_capability_host`."""

    def __init__(self) -> None:
        self.event_store = None
        self._tool_usage_tracker = None
        self._provider = None
        self.attached: tuple[object, object] | None = None
        self.closed = False

    def attach_capability_host(self, host, *, tiers=None) -> None:
        self.attached = (host, tiers)

    async def aclose(self) -> None:
        self.closed = True


class FakeScheduler:
    async def start(self) -> bool:
        return False

    async def stop(self) -> None:
        return None


class RecordingPersona:
    """`PersonaRegister` in place: records where it was loaded from."""

    loads: list[tuple[Path, str]] = []

    def __init__(self, root: Path, profile: str) -> None:
        self.root = root
        self.profile = profile

    @classmethod
    def load(cls, root: Path, profile: str) -> "RecordingPersona":
        cls.loads.append((root, profile))
        return cls(root, profile)

    def prompt_line(self) -> str:
        return f"register of {self.profile}"

    def address_form(self) -> str:
        return "you"


@pytest.fixture
def lifespan_harness(monkeypatch, tmp_path):
    """Run the real `lifespan` with nothing behind it.

    The builders that would open the vault, Neo4j or the model stack return
    None or fakes (`build_mist_runtime` is a `RuntimeFactory`), the file log and
    the system-status task are switched off, and every global `lifespan` rebinds
    is restored afterwards. What is left is the wiring under test: the yield
    state, the power gate, the runtime build, the session registry, the probe
    registry and the curation scheduler call.
    """
    for name in (
        "runtime",
        "session_registry",
        "curation_scheduler",
        "log_handler",
        "extraction_dispatcher",
        "vault_writer",
        "vault_sidecar",
        "vault_filewatcher",
        "vault_invalidation_bus",
        "_broadcaster_state",
        "_health_registry",
        "_health_probe_executor",
        "_extraction_health_client",
        "_extraction_dispatcher_start_failed",
        "_gpu_yield",
        "_models_ready",
        "_capability_build",
        "_capability_config",
        "_capability_power",
        "_capability_sink",
        "_device_listener",
        "_device_app",
        "_device_listener_state",
    ):
        monkeypatch.setattr(server, name, getattr(server, name))
    monkeypatch.setattr(server, "message_queue", asyncio.Queue())
    monkeypatch.setattr(server, "configure_file_logging", lambda log_dir: None)
    # The record factory is process-global; installing it here would outlive the test.
    monkeypatch.setattr(server, "install_log_redaction", lambda registry: None)
    monkeypatch.setattr(server, "build_vault_writer", lambda *a, **k: None)
    monkeypatch.setattr(server, "build_sidecar_index", lambda *a, **k: None)
    monkeypatch.setattr(server, "build_phase3_components", lambda *a, **k: None)
    provider = FakeLLMProvider()
    monkeypatch.setattr(server, "build_llm_provider", lambda config, **kwargs: provider)
    handler = FakeHandler()
    factory = RuntimeFactory(handler=handler)
    monkeypatch.setattr(server, "build_mist_runtime", factory)
    RecordingPersona.loads = []
    monkeypatch.setattr(server, "PersonaRegister", RecordingPersona)
    host_kwargs: dict = {}
    real_build_host = server.build_capability_host

    def recording_build_host(config, **kwargs):
        host_kwargs.update(kwargs)
        return real_build_host(config, **kwargs)

    monkeypatch.setattr(server, "build_capability_host", recording_build_host)
    scheduler_kwargs: dict = {}

    def fake_build_curation_scheduler(config, **kwargs):
        scheduler_kwargs.update(kwargs)
        return FakeScheduler()

    monkeypatch.setattr(server, "build_curation_scheduler", fake_build_curation_scheduler)
    for name in (
        "MIST_OPS_DIR",
        "MIST_BACKUP_STATUS_PATH",
        "MIST_GPU_YIELD_POLL_S",
        "MIST_VOICE_PROFILE",
        "MIST_CAPABILITIES_ENABLED",
        "MIST_DEVICE_LISTENER",
        "MIST_DEVICE_PORT",
        "MIST_DEVICE_TLS_CERT",
        "MIST_DEVICE_TLS_KEY",
        "MIST_DEVICE_REGISTRY",
    ):
        monkeypatch.delenv(name, raising=False)

    def configure(*, tts_enabled: bool) -> None:
        system_status = server.config.system_status.model_copy(update={"enabled": False})
        monkeypatch.setattr(
            server,
            "config",
            server.config.model_copy(
                update={"tts_enabled": tts_enabled, "system_status": system_status}
            ),
        )

    async def start() -> None:
        factory.loop = asyncio.get_running_loop()

    return SimpleNamespace(
        configure=configure,
        start=start,
        scheduler_kwargs=scheduler_kwargs,
        host_kwargs=host_kwargs,
        factory=factory,
        handler=handler,
        provider=provider,
    )


def _gpu_yield_tasks() -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() if t.get_name() == "gpu-yield"]


def _named_tasks(name: str) -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() if t.get_name() == name]


@pytest.mark.asyncio
async def test_lifespan_without_ops_dir_builds_no_gpu_yield(lifespan_harness):
    lifespan_harness.configure(tts_enabled=False)
    await lifespan_harness.start()

    async with server.lifespan(server.app):
        await server._health_registry.run_once()
        llm = server._health_registry.snapshot()["checks"]["llm"]
        tasks = _gpu_yield_tasks()
        power = server._capability_power

    assert server._gpu_yield is None
    assert tasks == []
    assert isinstance(power, FixedPower)
    assert power.level() == "full"
    assert lifespan_harness.factory.level_at_build == "full"
    assert lifespan_harness.scheduler_kwargs["defer_llm_jobs"] is None
    # No yield state reaches the probe, so the llm check asks the runtime's base provider.
    assert (llm["status"], llm["reason"]) == ("up", "ok")
    assert lifespan_harness.provider.calls >= 1


@pytest.mark.asyncio
async def test_lifespan_without_ops_dir_reads_no_paused_marker(
    lifespan_harness, monkeypatch, tmp_path
):
    """The dev stack: a marker in the working directory is never looked for."""
    lifespan_harness.configure(tts_enabled=False)
    _paused_marker(tmp_path)
    monkeypatch.chdir(tmp_path)

    async with server.lifespan(server.app):
        await server._health_registry.run_once()
        check = server._health_registry.snapshot()["checks"]["extraction_service"]

    assert check["status"] != "paused"


@pytest.mark.asyncio
@pytest.mark.parametrize("tts_enabled", [False, True], ids=["gaming-mode", "tts-on"])
async def test_lifespan_with_ops_dir_wires_the_paused_marker(
    lifespan_harness, monkeypatch, tmp_path, tts_enabled
):
    """Read from `MIST_OPS_DIR` whether or not gaming mode could be built."""
    lifespan_harness.configure(tts_enabled=tts_enabled)
    _paused_marker(tmp_path)
    monkeypatch.setenv("MIST_OPS_DIR", str(tmp_path))

    async with server.lifespan(server.app):
        await server._health_registry.run_once()
        payload = await server.health()

    check = payload["checks"]["extraction_service"]
    assert (check["status"], check["reason"]) == ("paused", "extraction_paused")
    assert payload["checks"]["llm"]["status"] != "yielded"


@pytest.mark.asyncio
async def test_lifespan_with_a_marker_boots_yielded_and_wires_every_reader(
    lifespan_harness, monkeypatch, tmp_path
):
    lifespan_harness.configure(tts_enabled=False)
    await lifespan_harness.start()
    _marked(tmp_path)
    monkeypatch.setenv("MIST_OPS_DIR", str(tmp_path))

    async with server.lifespan(server.app):
        gpu = server._gpu_yield
        power = server._capability_power
        await server._health_registry.run_once()
        payload = await server.health()
        status = server._health_status_payload()
        tasks = _gpu_yield_tasks()

    assert isinstance(gpu, GpuYield)
    assert gpu.state() == "yielded"
    assert isinstance(power, GpuYieldPower)
    # The build read the level at boot: `off`, so no CUDA engine and no warm-up.
    assert lifespan_harness.factory.level_at_build == "off"
    assert lifespan_harness.scheduler_kwargs["defer_llm_jobs"] == gpu.is_yielded
    assert (payload["checks"]["llm"]["status"], payload["checks"]["llm"]["reason"]) == (
        "yielded",
        "gpu_yielded",
    )
    assert payload["status"] == "yielded"
    assert payload["voice"]["power_level"] == "off"
    assert status["llm"] is False
    assert len(tasks) == 1
    assert tasks[0].done()
    # The poll starts only after the runtime exists: a poll running during the
    # build could release or reload engines that are still loading.
    assert lifespan_harness.factory.gpu_yield_tasks_during_build == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tts_enabled", [True, False])
async def test_lifespan_builds_the_gpu_yield_whenever_the_ops_dir_is_set_voice_on_or_off(
    lifespan_harness, monkeypatch, tmp_path, tts_enabled
):
    """The `tts_enabled` rule is gone: the runtime releases every CUDA engine, TTS included."""
    lifespan_harness.configure(tts_enabled=tts_enabled)
    await lifespan_harness.start()
    monkeypatch.setenv("MIST_OPS_DIR", str(tmp_path))

    async with server.lifespan(server.app):
        gpu = server._gpu_yield
        tasks = _gpu_yield_tasks()

    assert isinstance(gpu, GpuYield)
    assert gpu.state() == "normal"
    assert len(tasks) == 1
    assert lifespan_harness.factory.level_at_build == "full"


@pytest.mark.asyncio
async def test_a_dead_gpu_yield_poll_logs_one_error_naming_the_exception_class(
    lifespan_harness, monkeypatch, tmp_path, caplog
):
    lifespan_harness.configure(tts_enabled=False)
    await lifespan_harness.start()
    marker = _marked(tmp_path)
    monkeypatch.setenv("MIST_OPS_DIR", str(tmp_path))
    monkeypatch.setenv("MIST_GPU_YIELD_POLL_S", "0.01")

    with caplog.at_level(logging.ERROR, logger="backend.server"):
        async with server.lifespan(server.app):
            # A reload error `GpuYield` does not catch.
            server.runtime.gpu.reload_error = ValueError("unexpected reload failure")
            (task,) = _gpu_yield_tasks()
            marker.unlink()
            await asyncio.wait([task], timeout=5.0)
            state = server._gpu_yield.state()

    assert task.done() and not task.cancelled()
    assert isinstance(task.exception(), ValueError)
    died = [r for r in caplog.records if r.levelno >= logging.ERROR and "poll" in r.getMessage()]
    assert len(died) == 1
    assert "ValueError" in died[0].getMessage()
    assert state == "yielded"


@pytest.mark.asyncio
async def test_lifespan_wires_the_registry_the_sink_the_gate_and_the_gpu(lifespan_harness):
    """Each object `lifespan` builds reaches the one that reads it, and shutdown stops each."""
    from backend.realtime.capability_sink import CapabilitySink
    from backend.realtime.power import PowerGate
    from backend.realtime.registry import SessionRegistry

    lifespan_harness.configure(tts_enabled=False)
    await lifespan_harness.start()
    host_kwargs = lifespan_harness.host_kwargs

    async with server.lifespan(server.app):
        loop = asyncio.get_running_loop()
        built = server.runtime
        registry = server.session_registry
        sink = server._capability_sink
        watchers = _named_tasks("power-watcher")
        log_handler = server.log_handler

    (kwargs,) = lifespan_harness.factory.calls
    assert built is lifespan_harness.factory.runtime
    assert isinstance(registry, SessionRegistry)
    # The registry is built on the runtime's services, gated by the log handler.
    assert registry._services is built.services
    assert registry._log_gate is log_handler
    # `SessionServices.power_gate` and D8's `alarm_ringing`.
    assert isinstance(kwargs["power_gate"], PowerGate)
    assert built.services.power_gate is kwargs["power_gate"]
    assert isinstance(sink, CapabilitySink)
    assert kwargs["alarm_ringing"] == sink.ringing
    # The capability host's sink is the registry's, and the desktop is the registry's.
    assert host_kwargs["event_sink"] is sink
    assert sink._registry is registry
    assert host_kwargs["desktop_present"] is server._desktop_present
    # The persona: the host's register, and its prompt line the handler's register_prompt.
    ((root, profile),) = RecordingPersona.loads
    assert root == server.project_root / "voice_profiles"
    assert host_kwargs["register"].profile == profile
    assert kwargs["register_prompt"] == f"register of {profile}"
    # The runtime's GPU control is attached to the loop and the registry, then stopped.
    assert built.gpu.attached == (loop, registry)
    assert "stop" in built.gpu.calls
    # The power watcher runs while serving and is stopped at shutdown.
    assert len(watchers) == 1
    assert watchers[0].done()
    # The handler is drained, then every lane closed last.
    assert lifespan_harness.handler.closed
    assert lifespan_harness.factory.closed == ["lane"] * 6


@pytest.mark.asyncio
async def test_the_gpu_yield_reaches_the_runtime_late_and_the_watcher_reaches_its_gpu(tmp_path):
    """`on_yield` and `on_resume` read the module's runtime when called, not when built."""
    marker = _marked(tmp_path)
    gpu_yield = server._build_gpu_yield(str(tmp_path), poll_s=2.0)
    assert gpu_yield is not None
    gpu = FakeGpu()
    server.runtime = fake_mist_runtime(gpu=gpu)

    marker.unlink()
    await gpu_yield.poll_once()  # resumed
    _marked(tmp_path)
    await gpu_yield.poll_once()  # yielded
    server._level_changed("off")

    assert gpu.calls == ["reload", "release"]
    assert gpu.levels == ["off"]


def test_the_power_gate_refuses_with_the_yield_message_only_while_yielded(tmp_path):
    from backend.gpu_yield import YIELDED_MESSAGE
    from backend.realtime.power import NO_MODEL_MESSAGE

    _marked(tmp_path)
    yielded = server._build_gpu_yield(str(tmp_path), poll_s=2.0)
    power = server._build_power(yielded)
    gate = server._build_power_gate(power, server._capability_tiers(power), yielded)
    no_yield = server._build_power_gate(
        FixedPower("full"), server._capability_tiers(FixedPower("full")), None
    )
    server.runtime = None

    assert gate.level() == "off"
    assert gate.refusal_message() == YIELDED_MESSAGE
    assert no_yield.level() == "full"
    assert no_yield.model_available() is False  # no runtime, so no provider yet
    assert no_yield.refusal_message() == NO_MODEL_MESSAGE
    assert server._build_power(None).level() == "full"


# -- The room device listener (voice PE T10) ----------------------------------------------


def _device_registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"version": 1, "devices": []}), encoding="utf-8")
    return path


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _tls_pair(directory: Path) -> tuple[Path, Path]:
    """A throwaway certificate and key, made by `openssl` (no key is committed)."""
    key, cert = directory / "server.key", directory / "server.pem"
    subprocess.run(  # nosec B603 B607 - fixed argument list, no shell
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=device-listener-test",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


class _RecordingListener:
    """Stands in for `DeviceListener` to show that none is built."""

    built: list[object] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        _RecordingListener.built.append(args)


@pytest.mark.asyncio
async def test_with_no_device_variable_no_listener_is_built(lifespan_harness, monkeypatch):
    lifespan_harness.configure(tts_enabled=False)
    await lifespan_harness.start()
    _RecordingListener.built = []
    monkeypatch.setattr(server, "DeviceListener", _RecordingListener)

    async with server.lifespan(server.app):
        payload = await server.health()

    assert _RecordingListener.built == []
    assert server._device_listener is None
    assert payload["devices"] == {"listener": "off", "connected": []}


@pytest.mark.asyncio
async def test_a_listener_on_a_taken_port_fails_and_the_backend_serves_on(
    lifespan_harness, monkeypatch, tmp_path, caplog
):
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("0.0.0.0", 0))  # nosec B104 - a test holding the port the listener wants
    holder.listen()
    monkeypatch.setenv("MIST_DEVICE_LISTENER", "on")
    monkeypatch.setenv("MIST_DEVICE_PORT", str(holder.getsockname()[1]))
    monkeypatch.setenv("MIST_DEVICE_REGISTRY", str(_device_registry(tmp_path)))
    lifespan_harness.configure(tts_enabled=False)
    await lifespan_harness.start()

    try:
        with caplog.at_level(logging.ERROR):
            async with server.lifespan(server.app):
                payload = await server.health()
                registry_built = server.session_registry is not None
    finally:
        holder.close()

    assert payload["devices"] == {"listener": "failed", "connected": []}
    assert registry_built
    assert any("device_listener_failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_listener_with_a_missing_certificate_fails_and_the_backend_serves_on(
    lifespan_harness, monkeypatch, tmp_path
):
    monkeypatch.setenv("MIST_DEVICE_LISTENER", "on")
    monkeypatch.setenv("MIST_DEVICE_PORT", str(_free_port()))
    monkeypatch.setenv("MIST_DEVICE_TLS_CERT", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("MIST_DEVICE_TLS_KEY", str(tmp_path / "missing.key"))
    monkeypatch.setenv("MIST_DEVICE_REGISTRY", str(_device_registry(tmp_path)))
    lifespan_harness.configure(tts_enabled=False)
    await lifespan_harness.start()

    async with server.lifespan(server.app):
        payload = await server.health()

    assert payload["devices"]["listener"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env",
    [
        pytest.param({"MIST_DEVICE_LISTENER": "maybe"}, id="bad-switch"),
        pytest.param({"MIST_DEVICE_LISTENER": "on"}, id="missing-registry"),
    ],
)
async def test_an_unreadable_configuration_or_registry_reads_failed(
    lifespan_harness, monkeypatch, tmp_path, env
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MIST_DEVICE_REGISTRY", str(tmp_path / "absent.json"))
    lifespan_harness.configure(tts_enabled=False)
    await lifespan_harness.start()

    async with server.lifespan(server.app):
        payload = await server.health()

    assert payload["devices"] == {"listener": "failed", "connected": []}
    assert server._device_listener is None


@pytest.mark.asyncio
async def test_an_enabled_listener_serves_and_is_stopped_at_shutdown(
    lifespan_harness, monkeypatch, tmp_path
):
    cert, key = _tls_pair(tmp_path)
    monkeypatch.setenv("MIST_DEVICE_LISTENER", "on")
    monkeypatch.setenv("MIST_DEVICE_PORT", str(_free_port()))
    monkeypatch.setenv("MIST_DEVICE_TLS_CERT", str(cert))
    monkeypatch.setenv("MIST_DEVICE_TLS_KEY", str(key))
    monkeypatch.setenv("MIST_DEVICE_REGISTRY", str(_device_registry(tmp_path)))
    lifespan_harness.configure(tts_enabled=False)
    await lifespan_harness.start()

    async with server.lifespan(server.app):
        payload = await server.health()
        listener = server._device_listener

    assert payload["devices"] == {"listener": "on", "connected": []}
    assert listener is not None and listener.state() == "off"
    assert (await server.health())["devices"]["listener"] == "off"
