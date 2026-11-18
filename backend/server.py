"""Voice AI WebSocket Server.

Based on CSM demo architecture - production-ready for web frontend
"""

import asyncio
import contextlib
import functools
import json
import logging
import logging.handlers
import os
import sqlite3
import sys
import time
import uuid
from collections.abc import Coroutine
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # Reconfigure stdout/stderr to use UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import httpx
import uvicorn
from fastapi import FastAPI, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

# `python backend/server.py` makes CPython prepend the SCRIPT's directory --
# backend/ -- to sys.path, but not the repository root. The root has to be added
# here or the `backend.`-qualified imports below cannot resolve. Adding backend/
# a second time was redundant with sys.path[0] and is gone.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Every first-party import below is `backend.`-qualified deliberately. backend/
# is a PEP 420 namespace package that is ALSO on sys.path, so `config` and
# `backend.config` resolve to two DISTINCT module objects built from one file.
# Module-level state belongs to the module object, so the factories wiring,
# ContextVars and caches would each exist twice, and a write through one name
# would be invisible through the other. The bare spelling was additionally
# ambiguous with dependencies/csm/config.py, which an import-order workaround
# once papered over. The qualified name removes both hazards outright.
from backend.capabilities.clock import SystemClock  # isort:skip
from backend.capabilities.config import (  # isort:skip
    CapabilitiesConfig,
    load_capabilities_config,
)
from backend.capabilities.contracts import ToolOutcome  # isort:skip
from backend.capabilities.events import (  # isort:skip
    ActionDecisionMessage,
    AlarmActionMessage,
    AlarmReportMessage,
    alarm_action_result_payload,
)
from backend.capabilities.power import (  # isort:skip
    FixedPower,
    GpuYieldPower,
    PowerState,
    SingleModelTiers,
)
from backend.capabilities.secrets import (  # isort:skip
    EnvSecretSource,
    SecretRegistry,
    install_log_redaction,
    redact,
)
from backend.chat.context_budget import ApproximateTokenCounter  # isort:skip
from backend.chat.persona_register import PersonaRegister  # isort:skip
from backend.config import DEFAULT_CONFIG  # isort:skip
from backend.devices.config import (  # isort:skip
    DeviceListenerConfig,
    DeviceListenerConfigError,
)
from backend.devices.limiter import AuthLimiter  # isort:skip
from backend.devices.listener import (  # isort:skip
    DeviceApp,
    DeviceSocket,
    SystemClock as DeviceClock,
    build_device_app,
)
from backend.devices.registry import DeviceRegistry, DeviceRegistryError  # isort:skip
from backend.devices.ring_sink import DeviceRingSink, registry_ring_sinks  # isort:skip
from backend.devices.server import DeviceListener, ListenerState  # isort:skip
from backend.errors import CapabilityConfigError, MistError  # isort:skip
from backend.event_store.status_intervals import (  # isort:skip
    degraded_beneath,
    record_status_tick,
)
from backend.factories import (  # isort:skip
    CapabilityBuild,
    build_capability_host,
    build_curation_scheduler,
    build_extraction_dispatcher,
    build_llm_provider,
    build_mist_runtime,
    build_phase3_components,
    build_sidecar_index,
    build_vault_writer,
)
from backend.gpu_yield import (  # isort:skip
    MARKER_NAME,
    GpuYield,
    gpu_yield_poll_s,
)
from backend.health import (  # isort:skip
    BackupProbe,
    BroadcasterProbe,
    BroadcasterState,
    CapabilitiesProbe,
    EventStoreProbe,
    ExtractionBacklogProbe,
    ExtractionServiceHealthSource,
    ExtractionServiceProbe,
    HealthProbeSettings,
    HealthRegistry,
    LLMProbe,
    Neo4jProbe,
    VaultSidecarProbe,
    derive_status,
)
from backend.knowledge.config import KnowledgeConfig  # isort:skip
from backend.knowledge.eval_isolation import (  # isort:skip
    EvalIsolationError,
    is_hydration_isolation_active,
)
from backend.log_handler import LogLine, WebSocketLogHandler  # isort:skip
from backend.origins import allowed_origins, origin_allowed  # isort:skip
from backend.realtime.budget import load_voice_settings  # isort:skip
from backend.realtime.capability_sink import CapabilitySink  # isort:skip
from backend.realtime.power import (  # isort:skip
    PowerGate,
    PowerWatcher,
    gpu_yield_refusal,
)
from backend.realtime.protocol import (  # isort:skip
    PROTOCOL_VERSION,
    SessionCapabilities,
    SessionStartedMessage,
)
from backend.realtime.registry import SessionRegistry  # isort:skip
from backend.realtime.session import CapabilityMessage  # isort:skip

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from backend.capabilities.host import CapabilityHost
    from backend.chat.conversation_handler import ConversationHandler
    from backend.extraction_backlog.dispatcher import ExtractionDispatcher
    from backend.interfaces import EventStoreProvider
    from backend.knowledge.extraction.tool_usage_tracker import ToolUsageTracker
    from backend.llm import StreamingLLMProvider
    from backend.realtime.budget import VoiceSettings
    from backend.realtime.protocol import PowerLevel
    from backend.realtime.runtime import MistRuntime

# Setup logging -- console + persistent file
_log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=_log_format)

# Persistent file log (survives container removal via bind mount). Attaching
# the FileHandler is deferred to configure_file_logging(), called from
# lifespan() rather than here at import time -- see that function's
# docstring for why.


def _resolve_log_dir() -> Path:
    """Resolve the directory for the persistent backend log file.

    Read at call time, not import time, so the resolution reflects whatever
    `MIST_LOG_DIR` is set to when `lifespan()` actually runs.

    Returns:
        The path named by the `MIST_LOG_DIR` environment variable if set,
        otherwise the repository-relative `logs/` directory.
    """
    env_value = os.environ.get("MIST_LOG_DIR")
    if env_value:
        return Path(env_value)
    return Path(__file__).resolve().parents[1] / "logs"


DEFAULT_LOG_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 5


def _log_rotation_setting(name: str, default: int) -> int:
    """Read one positive integer rotation setting, keeping the default for a bad value.

    Zero is refused as well as negatives and non-numbers: `RotatingFileHandler`
    never rolls over when either `maxBytes` or `backupCount` is 0, which would
    silently restore the unbounded log this exists to prevent. Logged rather
    than raised, because this runs first in `lifespan` and a typo in a log
    setting should not stop the backend from starting.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value >= 1:
        return value
    logging.getLogger(__name__).error(
        "%s=%r is not a positive integer; using %d", name, raw, default
    )
    return default


def configure_file_logging(log_dir: Path) -> logging.Handler:
    """Attach a persistent, rotating file handler for the backend log to the root logger.

    Creates `log_dir` if it does not already exist and installs a
    `RotatingFileHandler` for `log_dir / "mist-backend.log"` at DEBUG level,
    using the same formatter as the console handler `logging.basicConfig`
    installs at import time. Idempotent: if a `FileHandler` (rotating or not)
    for that same file is already on the root logger, that handler is
    returned instead of adding a duplicate.

    The file rotates at `MIST_LOG_MAX_BYTES` (default 50 MiB) and keeps
    `MIST_LOG_BACKUP_COUNT` old files (default 5, as `mist-backend.log.1` to
    `.5`), both read at call time. A daily instance runs for weeks and logs
    at DEBUG to a bind mount, so an unrotated file grows without bound.

    Only records emitted after this function runs reach the file. Records
    logged during module import, before `lifespan()` calls this, reach the
    console handler only.

    Args:
        log_dir: Directory to hold `mist-backend.log`. Created if absent.

    Returns:
        The `FileHandler` attached to the root logger, new or pre-existing.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / "mist-backend.log").resolve()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file:
            return handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=_log_rotation_setting("MIST_LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES),
        backupCount=_log_rotation_setting("MIST_LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT),
    )
    file_handler.setFormatter(logging.Formatter(_log_format))
    file_handler.setLevel(logging.DEBUG)  # Capture everything to disk
    root_logger.addHandler(file_handler)
    return file_handler


logger = logging.getLogger(__name__)

# The WebSocket protocol version `session_started` announces, imported from the
# realtime protocol: 2.0.0 is main's 1.4.0 control messages plus protocol 2's
# own. Pinned by tests/unit/test_server_protocol_version.py.

# Global state
# Process-wide producers (heartbeat, system and health status, extraction
# status, log lines, decided proposals) put here; `broadcast_messages` drains
# it into the session registry.
message_queue: asyncio.Queue[str | LogLine] = asyncio.Queue()
# The realtime runtime and the registry that holds one `RealtimeSession` per
# `/ws` connection; both built in `lifespan`, None before it.
runtime: "MistRuntime | None" = None
session_registry: SessionRegistry | None = None
# The room devices' listener (`backend/devices/`), built in `lifespan` only when
# `MIST_DEVICE_LISTENER=on`; `_device_listener_state` is what `/health` shows
# when there is no listener: `off`, or `failed` when its configuration or
# registry could not be read.
_device_listener: DeviceListener | None = None
_device_app: DeviceApp | None = None
_device_listener_state: ListenerState = "off"
curation_scheduler = None
# T2a extraction backlog dispatcher (started/stopped in lifespan).
extraction_dispatcher = None
log_handler: WebSocketLogHandler | None = None
config = DEFAULT_CONFIG

# Cluster 8 Phase 5: vault layer subsystems (initialized in lifespan)
vault_writer = None
vault_sidecar = None
vault_filewatcher = None
# Phase 5.5: shared InvalidationBus wired from filewatcher to ConversationHandler
vault_invalidation_bus = None

# `/health` state. This flag is set only once the runtime's build has returned,
# so `models_loaded` never claims a state the process has not reached. Kept a
# plain bool because
# `tests/unit/hydration/test_health_isolation_flag.py` asserts the field is
# still present and unchanged in kind. `scripts/hydration/target.py` consumes
# this response but reads only `hydration_isolation` from it
# (`target.py:118,126`), so it is that field, not this one, whose type is
# load-bearing outside the repo's own tests.
_models_ready: bool = False
_broadcaster_state: BroadcasterState | None = None
_health_registry: HealthRegistry | None = None
_health_probe_executor: Executor | None = None
# Set when `_start_extraction_dispatcher` raised at boot. Without it a None
# `extraction_dispatcher` reads the same whether the dispatcher failed to start
# or was never configured, and the extraction checks would report a failed
# start as `disabled_by_config`.
_extraction_dispatcher_start_failed: bool = False
# The extraction-service health check's own HTTP client. The dispatcher's client
# carries `/v1/extract`'s read timeout (300 s by default) and a 5 s connect
# timeout (`backend.factories.build_extraction_dispatcher`), and exists only
# while a service-mode dispatcher runs; the probe needs a 2 s timeout on every
# phase (httpx applies it to connect, read, write and pool separately) regardless.
_extraction_health_client: httpx.AsyncClient | None = None
EXTRACTION_HEALTH_CLIENT_TIMEOUT_S = 2.0
# Gaming mode (`backend/gpu_yield.py`). None when `MIST_OPS_DIR` is unset: the
# backend is then never yielded.
_gpu_yield: GpuYield | None = None

# Capabilities (`backend/capabilities/`). `lifespan` builds them before the
# runtime. `_capability_build` holds the host, or the reason there is
# none, which `GET /capabilities` reports; None before `lifespan` runs.
# `_capability_config` is None when the capability variables do not parse.
# The secret registry exists from import, so the log redaction `lifespan`
# installs first covers every secret the host registers later.
_capability_build: CapabilityBuild | None = None
_capability_config: CapabilitiesConfig | None = None
_capability_power: PowerState | None = None
_capability_secrets = SecretRegistry()
# Relays the host's broadcasts to the sessions and answers D8's `alarm_ringing`.
_capability_sink: CapabilitySink | None = None
# Each session's capability calls in flight, cancelled when it disconnects so no
# call outlives its connection.
_capability_tasks: dict[str, set[asyncio.Task]] = {}
# A `reason` in `GET /capabilities`, and the text of a capability failure or a
# validation error answered on the WebSocket, is cut to this many characters.
CAPABILITY_REASON_MAX_CHARS = 200
# The endpoint every `/ws` connection is bound to (D7), recorded as the actor
# of each alarm action and report that arrives on it. A payload never names its
# own endpoint.
WS_ENDPOINT = "desktop"
# Where `PersonaRegister` reads `<profile>/personality.yaml`; `/app/voice_profiles`
# in the image.
VOICE_PROFILES_ROOT = project_root / "voice_profiles"
_T = TypeVar("_T")
_M = TypeVar("_M", bound=BaseModel)

# `probe_loop` value for "the registry was never constructed", distinct from
# `health.py`'s `alive` / `dead`, which both describe a loop that ran. It is
# reachable only before or without `lifespan` -- a direct `server.health()` call
# in a test, or a lifespan that raised before building the registry. It never
# reaches `derive_status`'s `probe_loop == "dead"` branch, because the checks
# mapping is empty in that case and the empty-mapping guard returns first.
PROBE_LOOP_NOT_STARTED = "not_started"

# The uptime ledger's tick loop (`uptime_record_loop`). A row continues across
# a silence of up to 2.5 periods, so one late or skipped tick does not split
# it, while a backend that stopped for longer leaves its gap unrecorded.
UPTIME_RECORD_ENV = "MIST_UPTIME_RECORD_S"
DEFAULT_UPTIME_RECORD_S = 60.0
UPTIME_GAP_PERIODS = 2.5
# The first tick waits for every health check to report once, so a boot is
# not recorded as the `degraded` that unmeasured checks derive to; a probe
# that never reports does not hold the ledger back for longer than this.
UPTIME_FIRST_TICK_WAIT_S = 60.0
UPTIME_FIRST_TICK_POLL_S = 1.0


def _route_broadcast(message: "str | LogLine") -> None:
    """Hand one queued message to the registry: a log line to its subscribers, else to all.

    Each session queues it on its own `bulk` lane and its own sender writes it, so a slow
    client delays only itself. With no registry (before `lifespan` built one) there is no
    session to send to, and the message is dropped.
    """
    registry = session_registry
    if registry is None:
        return
    if isinstance(message, LogLine):
        registry.publish_log(message.text, message.level)
    else:
        registry.broadcast(message)


async def broadcast_messages():
    """Drain `message_queue` into the session registry, one message at a time."""
    while True:
        message = await message_queue.get()
        _route_broadcast(message)
        message_queue.task_done()


async def heartbeat_loop(interval_seconds: float = 5.0) -> None:
    """Emit heartbeat events to connected clients every ``interval_seconds``.

    Per ADR-017 the FE uses heartbeats for ConnectionStatus liveness
    (~10s without heartbeat -> 'disconnected', attempt reconnect).
    Heartbeats are queued unconditionally; the broadcaster drains the
    queue and noops when no clients are connected.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            payload = json.dumps({"type": "heartbeat", "timestamp": int(time.time() * 1000)})
            await message_queue.put(payload)
        except Exception as e:  # noqa: BLE001
            logger.error("Heartbeat enqueue failed: %s", e)


async def system_status_loop(interval_seconds: float = 5.0) -> None:
    """Emit ADR-017 ``system_status`` events every ``interval_seconds``.

    Snapshots CPU / RAM / GPU via :mod:`backend.system_metrics` and pushes the
    payload onto :data:`message_queue` for the broadcaster to fan out to
    connected clients. Cadence defaults to 5s (BE prompt + config default);
    overridable via ``config.system_status.interval_seconds``.

    Failures during a single tick (psutil hiccup, NVML driver crash,
    serialization error) are logged and swallowed so the periodic emit
    survives the next interval. The collector itself emits placeholder GPU
    payloads on NVML failure, so most paths return cleanly.

    ``uptime_seconds`` / ``uptime_source`` (ADR-017 v1.2.0) are read
    straight off the snapshot -- this loop does no psutil call of its own,
    it only serializes what ``collect_metrics()`` already measured.

    Caveat this loop does NOT protect against, and what now catches it: the
    try/except here means a bad tick is logged and skipped, so THIS loop keeps
    emitting correctly on the next interval. But the payload only reaches
    clients via ``broadcast_messages()``, and if that downstream task dies this
    loop keeps emitting into ``message_queue`` while nothing drains it -- the
    payload, including ``uptime_seconds``, stops reaching clients.

    That death is no longer silent. ``broadcast_messages()`` is started with
    ``_on_broadcaster_done`` attached, which records the death in
    ``BroadcasterState`` (and logs it with a traceback when it was a raise), and
    ``BroadcasterProbe`` renders that record in ``/health`` as
    ``checks.broadcaster.status: "down"`` with ``reason: "task_dead"``. The
    broadcaster is the one dependency whose severity is fatal AND whose
    ``restart_repairs`` is true, so a settled failure drives ``status:
    "unhealthy"`` and ``restart_recommended: true``. What is still NOT fixed:
    nothing restarts the task, and nothing acts on that flag -- it is advisory,
    and respawn is out of scope. So this loop's output still stops; it now stops
    visibly.

    Cited by symbol, not line: ``KNOWN_ISSUES.md:150-152`` records this defect
    against ``server.py:87`` and that number is long stale, and an earlier draft
    of THIS sentence said 510, which the very hunk that wrote it pushed out of
    date.
    """
    from backend import system_metrics

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            snapshot = system_metrics.collect_metrics()
            payload = json.dumps(
                {
                    "type": "system_status",
                    "timestamp": snapshot.timestamp,
                    "cpu": {
                        "percent": snapshot.cpu.percent,
                        "cores": snapshot.cpu.cores,
                    },
                    "ram": {
                        "used_gb": snapshot.ram.used_gb,
                        "total_gb": snapshot.ram.total_gb,
                        "percent": snapshot.ram.percent,
                    },
                    "gpu": {
                        "name": snapshot.gpu.name,
                        "utilization_percent": snapshot.gpu.utilization_percent,
                        "vram_used_gb": snapshot.gpu.vram_used_gb,
                        "vram_total_gb": snapshot.gpu.vram_total_gb,
                        "temperature_c": snapshot.gpu.temperature_c,
                    },
                    "uptime_seconds": snapshot.uptime_seconds,
                    "uptime_source": system_metrics.UPTIME_SOURCE,
                }
            )
            await message_queue.put(payload)
        except Exception as e:  # noqa: BLE001
            logger.error("system_status emit failed: %s", e)


async def health_status_loop(interval_seconds: float = 30.0) -> None:
    """Emit ADR-017 ``health_status`` events every ``interval_seconds``.

    The payload is `_health_status_payload`'s. Transition-event emission
    (v1.1+) would supplement this periodic push; v1.0 is periodic only.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await message_queue.put(json.dumps(_health_status_payload()))
        except Exception as e:  # noqa: BLE001
            logger.error("Health status emit failed: %s", e)


def _health_status_payload() -> dict[str, object]:
    """Three booleans for the FE's persistent health indicator, read from the runtime now.

    - `llm`: the runtime is built and gaming mode does not hold the GPU. While
      yielded, or while the reload after gaming mode is failing, no model serves
      a turn, so it is False.
    - `agent`: the runtime answers turns through the conversation handler;
      False means degraded MIST, answering without memory (`DirectResponder`).
    - `local`: True always (if this loop is running, the BE is up).
    """
    built = runtime
    return {
        "type": "health_status",
        "llm": built is not None and not _gpu_is_yielded(),
        "agent": built is not None and built.agent_available,
        "local": True,
    }


def _gpu_is_yielded() -> bool:
    """Whether gaming mode holds the GPU (`yielded` or `resume_failed`)."""
    return _gpu_yield is not None and _gpu_yield.is_yielded()


def _release_gpu_engines() -> None:
    """`GpuYield.on_yield`: drop the runtime's CUDA speech engines.

    Reads `runtime` when called, because the runtime is built after the
    `GpuYield`: its boot reads whether the marker is present.
    """
    if runtime is not None:
        runtime.release_gpu_engines()


def _reload_gpu_engines() -> None:
    """`GpuYield.on_resume`: bring the runtime's speech engines back; never raises."""
    if runtime is not None:
        runtime.reload_gpu_engines()


def _build_gpu_yield(ops_dir: str | None, *, poll_s: float) -> GpuYield | None:
    """Build the gaming-mode watcher on `<ops_dir>/gpu-yielded`, or None.

    None when `ops_dir` (`MIST_OPS_DIR`) is unset or blank: no gaming mode,
    never yielded. Built otherwise, voice on or off: the runtime releases and
    reloads every CUDA speech engine, TTS included (D2).

    Args:
        ops_dir: The `MIST_OPS_DIR` value.
        poll_s: Marker poll interval (`MIST_GPU_YIELD_POLL_S`).
    """
    if ops_dir is None or not ops_dir.strip():
        return None
    return GpuYield(
        Path(ops_dir.strip()) / MARKER_NAME,
        _release_gpu_engines,
        _reload_gpu_engines,
        poll_s=poll_s,
    )


def _build_power(gpu_yield: GpuYield | None) -> PowerState:
    """The process's power level: gaming mode's state, or always `full` without it."""
    return FixedPower("full") if gpu_yield is None else GpuYieldPower(gpu_yield)


def _build_power_gate(
    power: PowerState, tiers: SingleModelTiers, gpu_yield: GpuYield | None
) -> PowerGate:
    """The gate every session applies (D2).

    The refusal names the yield while `GpuYield` is `yielded`, and the missing
    model otherwise or with no `GpuYield`.
    """
    state = gpu_yield.state if gpu_yield is not None else (lambda: "normal")
    return PowerGate(power, tiers, gpu_yield_refusal(state))


def _level_changed(level: "PowerLevel") -> None:
    """The power watcher's report, to the runtime's GPU control: `voice_state`s and the rise."""
    built = runtime
    if built is not None and built.gpu is not None:
        built.gpu.level_changed(level)


def _on_power_watcher_done(task: "asyncio.Task") -> None:
    """Done-callback for the power watcher; logs a death, ignores a cancel."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Power watcher died (%s); level changes no longer reach the sessions",
            type(exc).__name__,
            exc_info=exc,
        )
    else:
        logger.error("Power watcher returned; level changes no longer reach the sessions")


def _on_gpu_yield_loop_done(task: "asyncio.Task") -> None:
    """Done-callback for the gaming-mode poll task; logs a death, ignores a cancel.

    The same cancellation guard as `_on_broadcaster_done`: `lifespan` cancels
    this task at every shutdown. `GpuYield` catches only `RuntimeError` and
    `OSError` from its callbacks, so any other error ends the poll and freezes
    the yield state; this one ERROR line, naming the class, keeps that from
    being silent.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Gaming-mode poll loop died (%s); the yield state is now frozen",
            type(exc).__name__,
            exc_info=exc,
        )
    else:
        logger.error("Gaming-mode poll loop returned; the yield state is now frozen")


def _extraction_status_payload(dispatcher) -> str:
    """The `extraction_status` message as JSON: flat `ExtractionStatus.model_dump`.

    Built by `backend.extraction_backlog.status.extraction_status`, which never
    raises and reports `disabled` when `dispatcher` is None.
    """
    from backend.extraction_backlog.status import extraction_status

    return json.dumps(extraction_status(dispatcher).model_dump(mode="json"))


def push_extraction_status(dispatcher) -> None:
    """Enqueue one `extraction_status` message for `dispatcher` now (a transition push).

    Registered as the dispatcher's state listener in
    `_start_extraction_dispatcher`, so it runs synchronously on the event loop
    inside `ExtractionDispatcher._set_state`; `put_nowait` on the unbounded
    `message_queue` does not block. Takes the dispatcher explicitly rather than
    reading the module global, because the first transitions happen inside
    `start()`, before `lifespan` has assigned `extraction_dispatcher`.
    """
    message_queue.put_nowait(_extraction_status_payload(dispatcher))


async def extraction_status_loop(interval_seconds: float = 5.0) -> None:
    """Emit ADR-017 `extraction_status` events every `interval_seconds`.

    The periodic half of the extraction status push; the other half is
    `push_extraction_status` on every dispatcher state transition. Runs with
    or without a dispatcher: with none, the message reports `disabled` and the
    process's `unrecorded_turns`. Cadence defaults to 5 s, overridable via
    `MIST_EXTRACTION_STATUS_INTERVAL_S`
    (`backend.extraction_backlog.settings.status_interval_s`).

    Same failure policy as `system_status_loop`: a failed tick is logged and
    the loop survives to the next one.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await message_queue.put(_extraction_status_payload(extraction_dispatcher))
        except Exception as e:  # noqa: BLE001 -- the periodic push must outlive one bad tick
            logger.error("extraction_status emit failed: %s", e)


def _log_catchup_task_exception(task: "asyncio.Task") -> None:
    """Done-callback for the session-note catch-up background task.

    `run_forever` already swallows exceptions from individual `run()`
    passes internally (logged, loop continues) -- anything that reaches
    here is a bug in that internal handling itself, not a modeled failure
    mode. Without this callback such an exception would surface only as
    "Task exception was never retrieved" when the task object is garbage
    collected, which is easy to miss in production logs.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Session-note catch-up task failed: %s", exc, exc_info=exc)


def _session_count() -> int:
    """How many `/ws` sessions are connected; 0 before the registry exists."""
    registry = session_registry
    return 0 if registry is None else registry.count()


def _build_session_note_catchup(runtime, vault_writer):
    """Construct the catch-up instance from live server state, or None if ineligible.

    Extracted from `lifespan()` (final-review fix, C1) so the WIRING --
    which collaborator each injected callable actually points at -- is
    testable without booting the full ASGI lifespan, the same testability
    reason `_schedule_session_note_catchup` and
    `_shutdown_session_note_catchup` were extracted in fix round 2.

    `is_conversation_active` is wired to the session registry's count
    (`_session_count`), NOT `ConversationHandler.sessions`. `sessions` is
    populated on every turn and, in production, is removed from in exactly one
    place -- `clear_session` -- which no server path calls. The WebSocket
    disconnect path calls `end_session`, which pops `_vault_paths` but never
    touches `self.sessions`. Wiring the gate to `sessions` therefore means
    "this process has ever had a conversation," not "a conversation is active
    right now" -- true forever after the first turn, which silently disables
    `run_forever`'s retry (R1.3.1 fix round 1, I5) for the rest of the
    process's life. The registry forgets a session once it has committed its
    turn on disconnect, so its count reflects live connections.

    Returns:
        A `SessionNoteCatchup` ready to schedule, or None when the vault
        layer is disabled, no `ConversationHandler` is reachable yet, or
        the event store is unavailable.
    """
    catchup_handler = getattr(runtime, "conversation_handler", None)
    catchup_event_store = catchup_handler.event_store if catchup_handler is not None else None
    if vault_writer is None or catchup_handler is None or catchup_event_store is None:
        return None

    # Scoped to the production event store only: `catchup_event_store` is
    # whatever ConversationHandler.__init__ built from EVENT_STORE_DB_PATH
    # (the same store live traffic writes to), never a separate path such
    # as data/eval-run/event_store.db. If this wiring is ever parameterized
    # to accept an alternate event store, that path must NOT be routed
    # through catch-up.
    from backend.vault.session_catchup import SessionNoteCatchup

    return SessionNoteCatchup(
        event_store=catchup_event_store,
        synthesizer=catchup_handler.session_synthesizer,
        vault_writer=vault_writer,
        sessions_with_graph_state=catchup_handler.graph_store.sessions_with_graph_state,
        session_path_for=catchup_handler.derive_session_note_path,
        is_conversation_active=lambda: _session_count() > 0,
        is_llm_ready=catchup_handler.session_synthesizer.is_ready,
    )


def _schedule_session_note_catchup(catchup) -> "asyncio.Task":
    """Schedule catch-up as a periodic background task with exception visibility.

    Extracted from `lifespan()` (R1.3.1 fix round 2, N1) so the exact
    coroutine that gets scheduled -- `run_forever`, not a one-shot `run` --
    and the done-callback attachment are both independently testable
    without booting the full ASGI lifespan, which would otherwise require
    faking the runtime's build and the entire model stack. See
    `tests/unit/test_server_catchup_lifecycle.py`.
    """
    task = asyncio.create_task(catchup.run_forever(), name="session-note-catchup")
    # The task is never awaited on the happy path, so an exception escaping
    # run_forever's own internal try/except (a genuine bug, not a modeled
    # failure mode) would otherwise surface only as "Task exception was
    # never retrieved" at garbage collection.
    task.add_done_callback(_log_catchup_task_exception)
    return task


async def _shutdown_session_note_catchup(catchup_task: "asyncio.Task | None") -> None:
    """Cancel and await the catch-up background task.

    Extracted from `lifespan()` (R1.3.1 fix round 2, N1) for the same
    testability reason as `_schedule_session_note_catchup`. Must run before
    anything else in shutdown that awaits (`curation_scheduler.stop()`,
    `ConversationHandler.aclose()`, `vault_writer.stop()`) -- those awaits
    yield the event loop, and a still-resuming catch-up pass could
    otherwise reach a write against an already-stopped `vault_writer`.
    `lifespan()` calls this as its first shutdown action for that reason;
    a no-op when no task was scheduled (vault layer disabled).
    """
    if catchup_task is None:
        return
    catchup_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await catchup_task


@dataclass(frozen=True, slots=True)
class CurationDependencies:
    """The live collaborators `build_curation_scheduler` must be given.

    `handler_found` exists to separate the two ways a field can be None,
    which are indistinguishable at the call site: the ConversationHandler
    was not reachable at all (a wiring failure -- two jobs are inert), versus
    the handler was reached and its event store is legitimately absent
    because `config.event_store.enabled` is False (a configured choice).
    """

    event_store: "EventStoreProvider | None"
    tracker: "ToolUsageTracker | None"
    llm_provider: "StreamingLLMProvider | None"
    handler_found: bool


def _resolve_curation_dependencies(runtime) -> CurationDependencies:
    """Pull the curation scheduler's live collaborators off the runtime and its handler.

    Until 2026-08-03 `lifespan` called `build_curation_scheduler(knowledge_config)`
    and nothing else, leaving `event_store`, `tracker` and `llm_provider` at
    their None defaults. That one line disabled two of the scheduler's nine
    jobs with no error, no warning, and a fully green test suite:

      - `SelfReflectionJob.run` returns `ReflectionResult(0, 0, 0.0)` on its
        first line when `event_store` is None, so no internal knowledge was
        ever derived from conversation history.
      - the factory built a FRESH `ToolUsageTracker` for `SkillDerivationJob`,
        while every `.record()` call went to the DIFFERENT instance
        `build_conversation_handler` wired into `ConversationHandler` as
        `self._tool_usage_tracker`. Skill derivation therefore called
        `detect_patterns()` on a tracker that had never received a record --
        zero patterns, always, so no Skill / KNOWS / MistCapability /
        HAS_CAPABILITY was ever derived.

    `llm_provider` is the base provider, `runtime.llm_provider`: the object
    `lifespan` builds and `build_mist_runtime` injects into
    `build_conversation_handler`, which the extraction pipeline holds and
    `/health` probes (`_health_llm_provider`). It is not the handler's `_provider`, because with
    the conversation role set (`MIST_LLM_CONVERSATION_URL`) the handler holds
    the role's provider, and curation stays on `LLM_SERVER_URL` and `MODEL`.
    With the role unset the two are one object. It is supplied rather than
    left to the factory's own `build_llm_provider(config)`: the factory's call
    passes no `debug_logger`, so the deriver's LLM calls would be structurally
    invisible to the `phase: "llm_call"` JSONL stream even with the gate on,
    and a second `LlamaServerProvider` means a duplicate pair of OpenAI
    clients (and connection pools) that nothing closes.

    Extracted from `lifespan()` for the same testability reason as
    `_build_session_note_catchup`: WHICH object each dependency points at is
    the entire content of the fix, and identity cannot be asserted without
    booting the full model stack unless the resolution is its own function.
    See `tests/unit/test_curation_scheduler_wiring.py`.

    Returns:
        A `CurationDependencies` holding the live handler's own event store
        and tracker and the runtime's base provider, or all-None with
        `handler_found=False` when no handler is reachable.
    """
    handler = getattr(runtime, "conversation_handler", None)
    if handler is None:
        return CurationDependencies(
            event_store=None,
            tracker=None,
            llm_provider=None,
            handler_found=False,
        )
    return CurationDependencies(
        event_store=handler.event_store,
        tracker=handler._tool_usage_tracker,
        llm_provider=getattr(runtime, "llm_provider", None),
        handler_found=True,
    )


async def _start_extraction_dispatcher(runtime, knowledge_config: KnowledgeConfig):
    """Build, attach and start the T2a extraction-backlog dispatcher.

    Reaches the live ConversationHandler on the runtime, as the curation
    wiring does, builds the dispatcher over that handler's OWN
    pipeline and event store (so the backlog reads the log live traffic writes),
    starts it, and attaches it, so from here on a recorded turn wakes the
    backlog instead of spawning in-process extraction on the chat model.

    `MIST_EXTRACTION_INFERENCE=off` still attaches and starts it: `start()` in
    `off` mode only records the first-activation floor and sets state
    `disabled`, and runs no loop. Attaching is what keeps extraction off the
    chat model in that mode; recording the floor is what keeps turns logged
    while it is off eligible once it is turned back on.

    Returns:
        The started dispatcher, or None when no handler or event store is
        reachable (extraction then stays on the in-process path, as before T2a).

    Raises:
        ValueError: A `MIST_EXTRACTION_*` variable is malformed
            (`DispatcherSettings.from_env`).
    """
    from backend.extraction_backlog.settings import DispatcherSettings

    handler = getattr(runtime, "conversation_handler", None)
    if handler is None or handler.event_store is None:
        logger.warning(
            "Extraction dispatcher not started: no ConversationHandler or event store "
            "reachable -- extraction stays on the in-process path for this process"
        )
        return None
    dispatcher = build_extraction_dispatcher(
        knowledge_config,
        pipeline=handler._extraction_pipeline,
        event_store=handler.event_store,
        settings=DispatcherSettings.from_env(),
        extraction_cache=handler._extraction_pipeline.extraction_cache,
        http_client=None,
    )
    # Registered before `start()` so the transitions start() itself makes
    # (e.g. -> disabled in `off` mode) are pushed too.
    dispatcher.add_state_listener(lambda _previous, _current: push_extraction_status(dispatcher))
    # Started BEFORE attaching: if `start()` raises (e.g. the activation write
    # fails), the handler is left on the in-process path rather than attached
    # to a dispatcher that is not running, which would silently stop
    # extraction. No turn can arrive in between -- this runs before the server
    # accepts connections.
    await dispatcher.start()
    handler.attach_extraction_dispatcher(dispatcher)
    return dispatcher


def _on_broadcaster_done(task: "asyncio.Task") -> None:
    """Done-callback for the WebSocket broadcast task.

    Module-level and named, mirroring `_log_catchup_task_exception`, for the
    same reason that one was extracted: the behaviour is testable without
    booting the ASGI lifespan. It is also the single place a future supervisor
    would hook, so escalating broadcaster death becomes one added call here
    rather than a search through the lifespan.

    Three outcomes, and they are not interchangeable:

    - Cancelled. Returns early and marks nothing. `lifespan` cancels this task
      on every orderly shutdown, a cancelled task fires its done-callback like
      any other, and `task.exception()` RAISES `CancelledError` rather than
      returning one. What the guard actually buys was measured, by deleting it
      in a scratch copy and driving a real cancelled task through the callback:
      `mark_dead` is never reached, because the raise happens above it, so the
      state stays `alive` and `/health` still answers `healthy` with
      `restart_recommended: false`. The cost of dropping the guard is one
      `Exception in callback _on_broadcaster_done(...)` with a `CancelledError`,
      routed to the event loop's exception handler by
      `asyncio.events.Handle._run`, on every clean stop -- a server that shut
      down correctly logging a callback fault. This docstring previously
      claimed the missing guard would drive `/health` to `unhealthy` with
      `restart_recommended: true`; that was never executed and is false.
    - Raised. Logged at ERROR with the traceback, then recorded as dead.
    - Returned normally. Also recorded as dead. `broadcast_messages` is a
      `while True` with no `break` and no `return`, so a clean return is itself
      a bug; treating it as health would hide the same outage as a crash.

    Every outbound event funnels through this task, so its death is silent by
    construction: producers keep enqueueing onto `message_queue` and nothing
    drains it. `BroadcasterProbe` renders the recorded state in `/health`.
    """
    if task.cancelled():
        return

    exc = task.exception()
    if exc is not None:
        logger.error("Broadcast task died: %s", exc, exc_info=exc)
    else:
        logger.error(
            "Broadcast task returned without an exception; broadcast_messages is a "
            "`while True` that must never return, so this is a fault, not a clean stop"
        )

    if _broadcaster_state is not None:
        _broadcaster_state.mark_dead(exc)


def _on_health_probe_loop_done(task: "asyncio.Task") -> None:
    """Done-callback for the health registry's own probe loop.

    Same cancellation trap as `_on_broadcaster_done`, and the same reasoning:
    `lifespan` cancels this task on shutdown, so an unguarded `task.exception()`
    would raise `CancelledError` out of a callback on every clean stop.

    Nothing is marked here, and that is the design rather than an omission.
    `HealthRegistry.snapshot` already self-reports a stopped loop -- it computes
    `probe_loop` from how long ago the last cycle COMPLETED, going `dead` once
    that exceeds `dead_after_seconds`, reading no state this callback writes --
    so the loop's death is visible in the payload whether or not this callback
    ever fires. That independence is the point: detection must not depend on the
    mechanism that may itself have failed, and a done-callback is part of the
    task machinery whose failure it would be reporting. Adding a marker here
    would couple the detection path to this callback firing, which is exactly
    what the elapsed-time test is designed to avoid, so do not add one. What
    this callback contributes is the log line and the traceback, which an
    elapsed-time test cannot supply.
    """
    if task.cancelled():
        return

    exc = task.exception()
    if exc is not None:
        logger.error("Health probe loop died: %s", exc, exc_info=exc)
    else:
        logger.error(
            "Health probe loop returned without an exception; `run_forever` is a "
            "`while True`, so /health is now serving a cache nothing refreshes"
        )


def _health_conversation_handler():
    """Reach the live `ConversationHandler`, or None.

    None before `lifespan` has built the runtime, and when the runtime answers
    without memory: knowledge integration off, or its build raised a
    `MistError` such as Neo4j unreachable (`build_mist_runtime`).
    """
    return getattr(runtime, "conversation_handler", None)


def _health_graph_connection():
    """Reach the Neo4j connection `Neo4jProbe` calls `health_check` on.

    `conversation_handler.graph_store` (`chat/conversation_handler.py:785`) ->
    `graph_store.connection` (`knowledge/storage/graph_store.py:126`).
    """
    graph_store = getattr(_health_conversation_handler(), "graph_store", None)
    return getattr(graph_store, "connection", None)


def _health_event_store():
    """Reach the event store (`chat/conversation_handler.py:845-849`)."""
    return getattr(_health_conversation_handler(), "event_store", None)


def _health_llm_provider():
    """Reach the base LLM provider, `runtime.llm_provider`, or None before the runtime.

    The base, not the handler's: it is the object the extraction pipeline and
    curation hold, so the probe checks the server those use (T39's split).
    """
    return getattr(runtime, "llm_provider", None)


def _knowledge_configured() -> bool:
    """Whether knowledge integration is switched on, read at call time."""
    return KnowledgeConfig.from_env().enable_knowledge_integration


def _event_store_configured() -> bool:
    """Whether the event store is switched on, read at call time."""
    return KnowledgeConfig.from_env().event_store.enabled


def _capability_host() -> "CapabilityHost | None":
    """Reach the capability host, or None when `lifespan` built none.

    Read at each call, as the other health accessors are, so the probe and
    `GET /capabilities` follow whatever build is current.
    """
    build = _capability_build
    return build.host if build is not None else None


def _desktop_present() -> bool:
    """Whether a session bound `desktop` is connected, so the desktop endpoint can ring."""
    registry = session_registry
    return registry is not None and registry.endpoint_present(WS_ENDPOINT)


def _capability_tiers(power: PowerState) -> SingleModelTiers:
    """The action decode's and the sessions' tiers over the runtime's one provider.

    The provider is read through `_health_llm_provider` at each call, because
    it exists only once the runtime is built: before that the decode is
    unavailable, after it available. The same object is passed to the host,
    the conversation handler and the power gate, so they cannot disagree.
    """
    return SingleModelTiers(_health_llm_provider, power)


def _read_secret_file(path: str) -> str:
    """Read a `*_FILE` secret, such as a Docker secret holding the Home Assistant token."""
    return Path(path).read_text(encoding="utf-8")


def _attach_capability_host(tiers: SingleModelTiers) -> None:
    """Attach the built host to the live conversation handler, if both exist.

    Called once the runtime is built, since the handler exists only
    then. With no handler (knowledge integration off, or its build failed)
    the host still serves alarm actions and the status surface, but no turn
    reaches it, which is logged.
    """
    host = _capability_host()
    if host is None:
        return
    handler = _health_conversation_handler()
    if handler is None:
        logger.warning("Capability host built but no conversation handler exists; turns skip it")
        return
    handler.attach_capability_host(host, tiers=tiers)
    logger.info("Capability host attached to the conversation handler")


async def _stop_capability_host() -> None:
    """Stop the host, then close what the build owns (database, Spotify client).

    A no-op when no host was built. A failure is logged and shutdown goes on.
    """
    build = _capability_build
    if build is None or build.host is None:
        return
    try:
        await build.host.stop()
    except (MistError, sqlite3.Error, OSError) as e:
        logger.warning("Capability host stop error (non-fatal): %s", e)
    try:
        await build.close()
    except (MistError, sqlite3.Error, OSError, httpx.HTTPError) as e:
        logger.warning("Capability build close error (non-fatal): %s", e)


async def _build_runtime(
    knowledge_config: KnowledgeConfig,
    *,
    voice_settings: "VoiceSettings",
    power_gate: PowerGate,
    alarm_ringing: "Callable[[], bool]",
    register_prompt: str | None,
    factory: "Callable[..., MistRuntime] | None" = None,
) -> "MistRuntime":
    """Build the realtime runtime off the loop, publish it, and only then publish readiness.

    The ordering is the whole point, so it is a function rather than lines in
    `lifespan`: `_models_ready` must flip AFTER the build returns, and that is
    not assertable without faking the model stack unless the sequence is
    independently callable. See
    `tests/unit/test_server_health_wiring.py::test_models_loaded_is_false_during_the_runtime_build`.

    The build loads the speech engines and warms the LLM, so it runs on a worker
    thread: the capability host is already running on the loop, and an alarm due
    at boot rings without waiting for the models. The gate's level is read at
    boot, so with gaming mode's marker present no CUDA engine is built and the
    LLM warm-up is skipped.

    Args:
        knowledge_config: The knowledge configuration.
        voice_settings: From `load_voice_settings`.
        power_gate: Every session's gate; its level is the process's.
        alarm_ringing: D8's ringing query, `CapabilitySink.ringing`.
        register_prompt: The persona's register sentence for the handler's prompt.
        factory: Builds the runtime; `lifespan` passes the module's `build_mist_runtime`,
            looked up at its call. The module's `build_mist_runtime` too when None.
    """
    global runtime, _models_ready

    from backend.debug_jsonl_logger import DebugJSONLLogger

    provider = build_llm_provider(knowledge_config, debug_logger=DebugJSONLLogger.from_env())
    built = await asyncio.to_thread(
        functools.partial(
            factory or build_mist_runtime,
            knowledge_config,
            voice_settings=voice_settings,
            llm_provider=provider,
            tts_enabled=bool(config.tts_enabled),
            vault_writer=vault_writer,
            vault_sidecar=vault_sidecar,
            invalidation_bus=vault_invalidation_bus,
            power_gate=power_gate,
            alarm_ringing=alarm_ringing,
            on_capability_message=_on_capability_message,
            register_prompt=register_prompt,
        )
    )
    runtime = built
    _models_ready = True
    return built


async def _start_extraction_dispatcher_recording_failure(
    start: "Callable[[], Awaitable[ExtractionDispatcher | None]]",
) -> "ExtractionDispatcher | None":
    """Start the dispatcher, recording a failed start for the health checks.

    A module function rather than lines in `lifespan` so the failure path is
    testable without booting the model stack. The flag is what lets the two
    extraction checks report `unavailable_at_boot` until a restart succeeds;
    without it a None dispatcher reads `disabled_by_config`, as if extraction
    had been switched off on purpose.

    Args:
        start: Awaitable factory that builds, starts and attaches the
            dispatcher (`_start_extraction_dispatcher` bound to its arguments).

    Returns:
        The started dispatcher, or None when there is none to start or the
        start raised one of the errors `_start_extraction_dispatcher` can.
    """
    global _extraction_dispatcher_start_failed

    _extraction_dispatcher_start_failed = False
    try:
        return await start()
    except (MistError, sqlite3.Error, OSError, ValueError) as e:
        logger.error("Extraction dispatcher failed to start: %s", e, exc_info=True)
        _extraction_dispatcher_start_failed = True
        return None


def _new_extraction_health_client() -> httpx.AsyncClient:
    """The `extraction_service` check's own HTTP client, closed at shutdown.

    httpx applies the 2.0 s timeout to each phase (connect, read, write, pool)
    separately; the probe's own 2.5 s timeout bounds the whole call.
    """
    return httpx.AsyncClient(timeout=EXTRACTION_HEALTH_CLIENT_TIMEOUT_S)


async def _close_extraction_health_client() -> None:
    """Close and forget the probe client; a no-op when none was built."""
    global _extraction_health_client

    client = _extraction_health_client
    _extraction_health_client = None
    if client is not None:
        await client.aclose()


def _extraction_service_url() -> str:
    """The extraction service root the dispatcher would send jobs to.

    Read through `DispatcherSettings.from_env`, so the probe checks the same
    host the dispatcher uses. A malformed `MIST_EXTRACTION_*` variable makes
    that raise; it also makes the dispatcher's start fail, after which the
    probe reports `unavailable_at_boot` without calling anything, so the URL is
    then read on its own rather than failing the boot here.
    """
    from backend.extraction_backlog.settings import DEFAULT_SERVICE_URL, DispatcherSettings

    try:
        return DispatcherSettings.from_env().service_url
    except ValueError:
        return (os.getenv("MIST_EXTRACTION_SERVICE_URL") or DEFAULT_SERVICE_URL).strip()


def _build_health_registry(
    executor: Executor,
    broadcaster_state: BroadcasterState,
    extraction_inference: ExtractionServiceHealthSource,
    probe_settings: HealthProbeSettings,
    yield_state: "Callable[[], str] | None" = None,
    ops_dir: str | None = None,
    capability_host: "Callable[[], CapabilityHost | None] | None" = None,
) -> HealthRegistry:
    """Construct the dependency probes and the registry that runs them.

    Every accessor is a callable that reads module state WHEN THE PROBE RUNS,
    never a value captured here. `runtime`, `vault_sidecar` and the
    objects reachable through them are rebound during `lifespan`, and this
    registry is built after that -- but a captured value would still be wrong
    the moment any of them is replaced, and a captured None would freeze the
    probe at "absent" for the life of the process, reporting an outage that
    ended. The laziness is what `tests/unit/test_server_health_wiring.py`
    pins in `test_accessors_are_lazy`.

    Extracted from `lifespan` for the same testability reason as
    `_build_session_note_catchup`: WHICH object each probe ends up reading is
    the entire content of this wiring, and it cannot be asserted without booting
    the full model stack unless the construction is its own function.

    The registry is built before `lifespan` starts the extraction dispatcher,
    so `extraction_dispatcher` and `_extraction_dispatcher_start_failed` in
    particular must be read when the probe runs.

    Args:
        executor: Single-worker pool for the blocking Neo4j driver call. Owned
            by the caller, which is also responsible for shutting it down.
            Only `Neo4jProbe` uses it; the extraction-backlog and backup probes
            read through `asyncio.to_thread`.
        broadcaster_state: The live record the broadcast task publishes into.
        extraction_inference: The extraction service client the
            `extraction_service` check calls, over its own HTTP client. Owned
            by the caller, which closes that client at shutdown.
        probe_settings: Thresholds and the backup status path.
        yield_state: `GpuYield.state`, so the llm check reads `yielded` while
            gaming mode holds the GPU and `down` (`resume_failed`) while the
            reload fails, and the extraction_service check reads `yielded`
            too, since gaming mode stops extraction. None without gaming mode.
        ops_dir: `MIST_OPS_DIR`, where the host CLI writes the paused marker
            the extraction_service check reads, so it reports `paused` while
            night extraction is held off. None or blank on the dev stack,
            which builds no reader and leaves the check as it was.
        capability_host: The capability host's accessor, read at each check.
            None when no host was built: the `capabilities` check is then
            absent rather than unmeasured, which would derive `degraded`.

    Returns:
        A registry whose `run_forever` the caller schedules.
    """
    from backend.extraction_backlog.status import checked_extraction_status
    from backend.extraction_hold import build_extraction_hold_reader

    hold_reader = build_extraction_hold_reader(ops_dir)

    def dispatcher():
        return extraction_dispatcher

    def start_failed() -> bool:
        return _extraction_dispatcher_start_failed

    probes = [
        Neo4jProbe(_health_graph_connection, _knowledge_configured, executor),
        LLMProbe(_health_llm_provider, yield_state=yield_state),
        EventStoreProbe(_health_event_store, _event_store_configured),
        VaultSidecarProbe(lambda: vault_sidecar),
        BroadcasterProbe(broadcaster_state.state),
        ExtractionServiceProbe(
            extraction_inference,
            dispatcher,
            start_failed,
            yield_state=yield_state,
            hold_state=hold_reader.current if hold_reader is not None else None,
        ),
        ExtractionBacklogProbe(dispatcher, checked_extraction_status, start_failed, probe_settings),
        BackupProbe(probe_settings, lambda: datetime.now(UTC)),
    ]
    if capability_host is not None:
        probes.append(CapabilitiesProbe(capability_host))
    return HealthRegistry(
        probes,
        monotonic=time.monotonic,
        wall_clock=lambda: datetime.now(UTC),
    )


def uptime_record_s() -> float:
    """The uptime ledger's tick period in seconds, from `MIST_UPTIME_RECORD_S`.

    A value that is not a positive number is logged and replaced by the
    default rather than raised, as `gpu_yield_poll_s` does: it is read while
    the backend starts, and a typo here should not stop it.
    """
    raw = (os.environ.get(UPTIME_RECORD_ENV) or "").strip()
    if not raw:
        return DEFAULT_UPTIME_RECORD_S
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    if value > 0:
        return value
    logger.error(
        "%s=%r is not a positive number; using %s",
        UPTIME_RECORD_ENV,
        raw,
        DEFAULT_UPTIME_RECORD_S,
    )
    return DEFAULT_UPTIME_RECORD_S


def _every_check_measured(snapshot: dict) -> bool:
    """Whether every check in a registry snapshot has reported at least once."""
    checks = snapshot.get("checks") or {}
    return bool(checks) and all(check.get("age_seconds") is not None for check in checks.values())


async def uptime_record_loop(
    db_path: str,
    snapshot: "Callable[[], dict]",
    *,
    period_s: float,
    tick: "Callable[..., None]" = record_status_tick,
    sleep: "Callable[[float], Awaitable[object]]" = asyncio.sleep,
    clock: "Callable[[], datetime]" = lambda: datetime.now(UTC),
) -> None:
    """Record the derived `/health` status in the uptime ledger until cancelled.

    The first tick waits until every check in the snapshot has been measured,
    checking each `UPTIME_FIRST_TICK_POLL_S`, for at most
    `UPTIME_FIRST_TICK_WAIT_S`, and then ticks anyway with `first_tick=True`
    so a restart starts a new row. A first tick that fails leaves the next
    one first as well: otherwise a quick restart would extend the previous
    process's row across the time this one was down.

    Each tick runs `tick` in a worker thread through `asyncio.to_thread`: it
    may wait up to 5 s on another writer's lock, which must not stall the
    event loop that carries the WebSocket. A `sqlite3.Error` or `MistError`
    is logged and that tick skipped; the loop goes on.

    Args:
        db_path: The event store's database file.
        snapshot: Renders the health registry's current snapshot.
        period_s: Seconds between ticks; a row continues across a silence of
            up to `UPTIME_GAP_PERIODS` periods.
        tick: `record_status_tick`; injected in tests.
        sleep: `asyncio.sleep`; injected in tests.
        clock: The wall clock, aware; injected in tests.
    """
    waited = 0.0
    while not _every_check_measured(snapshot()) and waited < UPTIME_FIRST_TICK_WAIT_S:
        await sleep(UPTIME_FIRST_TICK_POLL_S)
        waited += UPTIME_FIRST_TICK_POLL_S

    first_tick = True
    max_gap_s = UPTIME_GAP_PERIODS * period_s
    while True:
        current = snapshot()
        try:
            await asyncio.to_thread(
                tick,
                db_path,
                current["status"],
                degraded_beneath(current),
                clock(),
                max_gap_s=max_gap_s,
                first_tick=first_tick,
            )
        except (sqlite3.Error, MistError) as exc:
            logger.warning("Uptime ledger tick skipped: %s", exc)
        else:
            first_tick = False
        await sleep(period_s)


def _build_uptime_recorder(
    event_store: "EventStoreProvider | None",
    snapshot: "Callable[[], dict]",
    period_s: float,
) -> "Callable[[], Awaitable[None]] | None":
    """Bind the uptime tick loop to the event store's database file, or None.

    None when there is no event store (switched off, or the handler that owns
    it was not built): the ledger lives in that store, so nothing is recorded.
    Also None for an in-memory store, which a second connection cannot reach.

    Args:
        event_store: The live handler's event store.
        snapshot: Renders the health registry's current snapshot.
        period_s: `uptime_record_s()`.

    Returns:
        A no-argument coroutine function the caller schedules as a task.
    """
    if event_store is None:
        logger.info("Uptime ledger: no event store, so no uptime is recorded")
        return None
    db_path = getattr(event_store, "db_path", None)
    if db_path is None or str(db_path) == ":memory:":
        logger.info("Uptime ledger: the event store has no database file; not recording")
        return None
    return functools.partial(uptime_record_loop, str(db_path), snapshot, period_s=period_s)


def _on_uptime_loop_done(task: "asyncio.Task") -> None:
    """Done-callback for the uptime tick loop; logs a death, ignores a cancel.

    The same cancellation guard as `_on_broadcaster_done`. The loop survives
    `sqlite3.Error` and `MistError`, so anything that ends it is a fault, and
    every day after it reads as unrecorded; this line says why.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Uptime ledger loop died: %s", exc, exc_info=exc)
    else:
        logger.error("Uptime ledger loop returned; no further uptime is recorded")


async def _stop_uptime_recorder(task: "asyncio.Task | None") -> None:
    """Cancel the uptime tick loop and wait for the cancel to land."""
    if task is None:
        return
    task.cancel()
    # `return_exceptions` keeps a loop that already died (logged by its
    # done-callback) from raising here.
    await asyncio.gather(task, return_exceptions=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # First, before any handler is attached or any secret loaded: the record
    # factory redacts every record created from here on, whichever handler
    # (the DEBUG file log below, the WebSocket log handler later) emits it.
    install_log_redaction(_capability_secrets)
    configure_file_logging(_resolve_log_dir())

    global curation_scheduler, log_handler, extraction_dispatcher, session_registry
    global vault_writer, vault_sidecar, vault_filewatcher, vault_invalidation_bus
    global _broadcaster_state, _health_registry, _health_probe_executor
    global _extraction_health_client, _gpu_yield
    global _capability_build, _capability_config, _capability_power, _capability_sink
    global _device_listener, _device_app, _device_listener_state

    # Startup
    logger.info("=" * 60)
    logger.info("STARTING MIST.AI VOICE SERVER")
    logger.info("=" * 60)

    # Acquire event loop early -- needed by both filewatcher.start and VAD.
    loop = asyncio.get_running_loop()

    # Single config load shared between vault and curation subsystems.
    knowledge_config = KnowledgeConfig.from_env()

    # Cluster 8 Phase 5: vault layer subsystems are built FIRST so the
    # single server-owned VaultWriter can be plumbed into the runtime's
    # ConversationHandler (`build_mist_runtime`). Each
    # subsystem is optional via config and degrades cleanly when disabled.
    vault_writer = None
    vault_sidecar = None
    vault_filewatcher = None

    try:
        # Phase 12 vault observability shares the existing JSONL sink. Built
        # once here so the vault_writer can emit phase: "vault" records when
        # MIST_DEBUG_VAULT_JSONL=1. Cheap construction; safe when the env
        # var is unset (logger is then a no-op).
        from backend.debug_jsonl_logger import DebugJSONLLogger

        vault_debug_logger = DebugJSONLLogger.from_env()
        vault_writer = build_vault_writer(knowledge_config, debug_logger=vault_debug_logger)
        if vault_writer is not None:
            await vault_writer.start()
            logger.info("Vault writer started at %s", knowledge_config.vault.root)

        vault_sidecar = build_sidecar_index(knowledge_config)
        if vault_sidecar is not None:
            logger.info(
                "Vault sidecar index initialized at %s",
                knowledge_config.sidecar_index.db_path,
            )
            # Warm the sidecar's embedding model off-loop: otherwise the
            # first vault event after startup lazy-loads a SentenceTransformer
            # (seconds) on whichever thread fires it, under live traffic.
            try:
                await loop.run_in_executor(None, vault_sidecar.warmup)
            except Exception as e:  # noqa: BLE001 -- warmup is best-effort
                logger.warning("Sidecar embedder warmup failed (non-fatal): %s", e)

        # Phase 5.5: migrate from build_filewatcher (throwaway bus) to
        # build_phase3_components so the shared InvalidationBus instance can
        # be forwarded through `build_mist_runtime` to
        # build_conversation_handler, wiring ConversationHandler._on_vault_rebuild
        # for ADR-010 read-path cache invalidation on vault edits.
        phase3 = build_phase3_components(
            config=knowledge_config,
            sidecar_index=vault_sidecar,
            writer=vault_writer,
        )
        if phase3 is not None:
            vault_filewatcher = phase3.filewatcher
            vault_invalidation_bus = phase3.invalidation_bus
            vault_filewatcher.start(loop)
            logger.info(
                "Vault filewatcher started (observer=%s, debounce=%dms)",
                knowledge_config.filewatcher.observer_type,
                knowledge_config.filewatcher.debounce_ms,
            )
    except Exception as e:
        logger.warning("Vault layer initialization failed (continuing without vault): %s", e)

    # Gaming mode, built before the runtime so a backend that boots while the
    # marker exists builds no CUDA speech engine and skips the LLM warmup, and
    # before the capability host, whose power level reads it. Its callbacks
    # reach the runtime late, through the module global. Its poll starts only
    # after the runtime is built, so a marker that came or went during the
    # load is acted on by the first poll.
    _gpu_yield = _build_gpu_yield(os.environ.get("MIST_OPS_DIR"), poll_s=gpu_yield_poll_s())

    # Capabilities: built and started before the runtime and before the model
    # load, so an alarm due at boot rings without waiting for the models. The
    # tiers read the provider lazily (see `_capability_tiers`), so the host is
    # ready before the provider is. The sink reaches the sessions once the
    # registry is attached below.
    _capability_power = _build_power(_gpu_yield)
    capability_tiers = _capability_tiers(_capability_power)
    power_gate = _build_power_gate(_capability_power, capability_tiers, _gpu_yield)
    voice_settings = load_voice_settings(os.environ)
    persona = PersonaRegister.load(VOICE_PROFILES_ROOT, voice_settings.profile)
    _capability_sink = CapabilitySink()

    # Room devices' configuration and registry, read before the capability
    # host so each registered, unrevoked device rings through its own sink.
    # Either one unreadable logs ERROR and shows `failed` in `/health`: no
    # device rings and no listener starts, and the desktop path runs on. A
    # device registered later rings after a restart.
    _device_listener = None
    _device_app = None
    _device_listener_state = "off"
    device_registry: DeviceRegistry | None = None
    try:
        device_config: DeviceListenerConfig | None = DeviceListenerConfig.from_env(os.environ)
    except DeviceListenerConfigError as e:
        logger.error("device_listener_failed error=%s; the desktop path is unaffected", e)
        device_config = None
        _device_listener_state = "failed"
    if device_config is not None and device_config.enabled:
        try:
            device_registry = DeviceRegistry.load(device_config.registry)
        except DeviceRegistryError as e:
            logger.error("device_listener_failed error=%s; the desktop path is unaffected", e)
            _device_listener_state = "failed"

    # The sinks read the session registry when they are called, as
    # `_desktop_present` does, since the registry is built after the host.
    ring_sinks: tuple[DeviceRingSink, ...] = (
        registry_ring_sinks(device_registry.devices(), lambda: session_registry)
        if device_registry is not None
        else ()
    )

    try:
        _capability_config = load_capabilities_config(os.environ)
    except CapabilityConfigError as e:
        logger.error("Capability configuration invalid; no capability host: %s", e)
        _capability_config = None
        _capability_build = CapabilityBuild(
            host=None, disabled_reason=f"capability config invalid: {e}"
        )
    else:
        _capability_build = build_capability_host(
            _capability_config,
            event_sink=_capability_sink,
            secret_source=EnvSecretSource(os.environ, _read_secret_file),
            secret_registry=_capability_secrets,
            clock=SystemClock(),
            # No embedder exists before the model load; the router's
            # embedding stage stays off until one is wired.
            embedder=None,
            token_counter=ApproximateTokenCounter(),
            # None: the factory uses the MCP SDK transport for lights.
            transport_factory=None,
            power=_capability_power,
            tiers=capability_tiers,
            register=persona,
            desktop_present=_desktop_present,
            ring_sinks=ring_sinks,
        )
    if _capability_build.host is not None:
        await _capability_build.host.start()

    # The realtime runtime, with the server-owned vault_writer, sidecar and
    # invalidation_bus, so the handler shares a single started writer, the
    # retriever routes to the same initialized sidecar (Phase 9), and
    # _on_vault_rebuild is subscribed to the shared bus for ADR-010 read-path
    # cache invalidation (Phase 5.5). Sets `_models_ready` only once the build
    # has returned, so `/health.models_loaded` reports a state the process has
    # actually reached.
    yield_state = _gpu_yield.state if _gpu_yield is not None else None
    defer_llm_jobs = _gpu_yield.is_yielded if _gpu_yield is not None else None
    built = await _build_runtime(
        knowledge_config,
        voice_settings=voice_settings,
        power_gate=power_gate,
        alarm_ringing=_capability_sink.ringing,
        register_prompt=persona.prompt_line(),
        factory=build_mist_runtime,
    )
    # The log handler streams through the registry: `subscribe_logs` gates it
    # on every session's subscription (`SessionRegistry.set_log_subscription`).
    log_handler = WebSocketLogHandler(event_loop=loop, message_queue=message_queue)
    logging.getLogger().addHandler(log_handler)
    logger.info("WebSocket log handler attached")
    session_registry = SessionRegistry(built.services, log_gate=log_handler)
    _capability_sink.attach(session_registry, loop)
    if built.gpu is not None:
        built.gpu.attach(loop, session_registry)
    _attach_capability_host(capability_tiers)
    power_watcher = PowerWatcher(power_gate, yield_state or (lambda: "normal"), _level_changed)
    power_watcher_task = asyncio.create_task(power_watcher.run(), name="power-watcher")
    power_watcher_task.add_done_callback(_on_power_watcher_done)
    gpu_yield_task: asyncio.Task | None = None
    if _gpu_yield is not None:
        gpu_yield_task = asyncio.create_task(_gpu_yield.run_forever(), name="gpu-yield")
        gpu_yield_task.add_done_callback(_on_gpu_yield_loop_done)
        logger.info("Gaming mode enabled (initial state: %s)", _gpu_yield.state())

    # Start message broadcaster. The state record is created and marked alive
    # here, beside the task it describes: `BroadcasterProbe` renders
    # `not_started` until something publishes, so a registry wired to a state
    # nobody marks would report the broadcaster missing forever.
    _broadcaster_state = BroadcasterState()
    broadcaster_task = asyncio.create_task(broadcast_messages(), name="broadcast-messages")
    _broadcaster_state.mark_alive()
    broadcaster_task.add_done_callback(_on_broadcaster_done)
    # Start heartbeat task (5s interval per ADR-017 Heartbeat semantics)
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    # Start health-status task (30s interval per ADR-017 health_status)
    health_status_task = asyncio.create_task(health_status_loop())

    # Start the dependency probe loop behind `/health`. The executor is
    # single-worker on purpose: the Neo4j driver is built with no timeouts, and
    # `asyncio.wait_for` abandons a wedged `run_in_executor` call rather than
    # stopping it, so an unbounded pool would leak one thread per tick into a
    # hung dependency. See `backend/health.py`'s module docstring.
    _health_probe_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="health-probe")
    from backend.extraction_backlog.inference import RemoteExtractionInference

    _extraction_health_client = _new_extraction_health_client()
    _health_registry = _build_health_registry(
        _health_probe_executor,
        _broadcaster_state,
        RemoteExtractionInference(_extraction_health_client, _extraction_service_url()),
        HealthProbeSettings.from_env(),
        yield_state=yield_state,
        # The paused marker's directory, passed whether or not gaming mode was
        # built: night extraction is scheduled even with TTS on.
        ops_dir=os.environ.get("MIST_OPS_DIR"),
        capability_host=_capability_host if _capability_host() is not None else None,
    )
    health_probe_task = asyncio.create_task(_health_registry.run_forever(), name="health-probes")
    health_probe_task.add_done_callback(_on_health_probe_loop_done)

    # The uptime ledger samples the registry above into the event store, so it
    # starts after both exist; the runtime is built, so the handler that owns
    # the event store is the live one.
    uptime_task: asyncio.Task | None = None
    uptime_recorder = _build_uptime_recorder(
        _health_event_store(), _health_snapshot, uptime_record_s()
    )
    if uptime_recorder is not None:
        uptime_task = asyncio.create_task(uptime_recorder(), name="uptime-ledger")
        uptime_task.add_done_callback(_on_uptime_loop_done)

    # Start system-status task (5s interval per ADR-017 system_status).
    # GPU init is best-effort; failure produces placeholder GPU blocks
    # (gpu.name='none') rather than crashing the lifespan.
    system_status_task = None
    if config.system_status.enabled:
        from backend import system_metrics

        if config.system_status.gpu_enabled:
            system_metrics.init_gpu()
        system_status_task = asyncio.create_task(
            system_status_loop(interval_seconds=float(config.system_status.interval_seconds))
        )
        logger.info(
            "System-status emit task started (interval=%ds, gpu=%s)",
            config.system_status.interval_seconds,
            config.system_status.gpu_enabled,
        )

    # Start curation scheduler for periodic graph maintenance.
    #
    # Every falsy-default parameter is supplied explicitly -- see
    # `_resolve_curation_dependencies` for what leaving them defaulted cost
    # (two of the nine jobs structurally incapable of producing output).
    # The runtime is built above, so the ConversationHandler these come off
    # is the live one.
    curation_deps = _resolve_curation_dependencies(built)
    if not curation_deps.handler_found:
        logger.warning(
            "Curation scheduler: no ConversationHandler reachable -- self_reflection and "
            "skill_derivation will be inert for this process"
        )
    elif curation_deps.event_store is None:
        logger.info(
            "Curation scheduler: event store disabled (config.event_store.enabled=False) "
            "-- self_reflection reports zero counts by configuration, not by omission"
        )
    try:
        curation_scheduler = build_curation_scheduler(
            knowledge_config,
            event_store=curation_deps.event_store,
            tracker=curation_deps.tracker,
            llm_provider=curation_deps.llm_provider,
            # `self_reflection` waits while gaming mode has stopped mist-llm.
            defer_llm_jobs=defer_llm_jobs,
        )
        if await curation_scheduler.start():
            logger.info("Curation scheduler started")
        else:
            # start() now returns False when it declined (hydration isolation,
            # or the disable knob) rather than raising. Logging "started"
            # unconditionally was misleading in exactly the situation an
            # operator would be reading this log to diagnose.
            logger.info("Curation scheduler not started (see reason above)")
    except Exception as e:
        logger.warning("Curation scheduler failed to start: %s", e)
        curation_scheduler = None

    # T2a: extraction backlog. After the runtime's build so the handler it
    # attaches to is the live one.
    extraction_dispatcher = None
    extraction_dispatcher = await _start_extraction_dispatcher_recording_failure(
        lambda: _start_extraction_dispatcher(built, knowledge_config)
    )

    # ADR-017 additive `extraction_status`: periodic push, with or without a
    # dispatcher (none -> `disabled` plus `unrecorded_turns`). Transitions are
    # pushed by the state listener `_start_extraction_dispatcher` registered.
    from backend.extraction_backlog.settings import status_interval_s

    try:
        extraction_status_interval = status_interval_s()
    except ValueError as e:
        logger.error("MIST_EXTRACTION_STATUS_INTERVAL_S invalid, using 5s: %s", e)
        extraction_status_interval = 5.0
    extraction_status_task = asyncio.create_task(
        extraction_status_loop(interval_seconds=extraction_status_interval)
    )

    # R1.3.1: synthesize vault notes for sessions that crashed before session
    # end (process kill, container restart) and so never reached
    # ConversationHandler.end_session. Background and deferential -- boot is
    # when the user is most likely to start talking, and a synthesis call
    # would hold the GPU and delay their first token. Reached through the
    # runtime's handler, as the shutdown path below reaches it.
    # Runs as `run_forever` (periodic, not one-shot): a single pass can come
    # back empty because the LLM backend is still cold at boot or a
    # conversation is live throughout, and a one-shot task would strand the
    # backlog for the rest of the process's life in either case.
    catchup_task: asyncio.Task | None = None
    try:
        catchup = _build_session_note_catchup(built, vault_writer)
        if catchup is not None:
            catchup_task = _schedule_session_note_catchup(catchup)
            logger.info("Session-note catch-up scheduled as a periodic background task")
    except Exception as e:  # noqa: BLE001 -- catch-up must never block startup
        logger.warning("Session-note catch-up scheduling failed (non-fatal): %s", e)

    # Room devices: a second, TLS listener for `/v2/device`, off unless
    # `MIST_DEVICE_LISTENER=on`. Its configuration and registry were read
    # before the capability host, which rings each device; the listener
    # starts once the session registry exists, since each device is served as
    # a session. A listener that cannot start (the port taken, a bad
    # certificate) logs ERROR and shows `failed` in `/health`; the desktop
    # path runs on.
    if device_config is not None and device_registry is not None:
        device_clock = DeviceClock()
        _device_app = build_device_app(
            device_registry, _serve_session, AuthLimiter(device_clock), device_clock
        )
        _device_listener = DeviceListener(device_config, _device_app)
        await _device_listener.start()

    logger.info(f"Server ready on ws://{config.host}:{config.port}/ws")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Server shutting down...")

    # Cancel the catch-up background task FIRST, as shutdown's first awaiting
    # step, before anything else below touches vault_writer or the
    # ConversationHandler it holds a reference to. Every later awaiting step
    # (the device listener's stop, curation_scheduler.stop(), ch.aclose()) can
    # yield the event loop -- a still-running catch-up pass could resume in that
    # window and reach a write against an already-stopped vault_writer if
    # cancellation happened later (or not at all).
    await _shutdown_session_note_catchup(catchup_task)

    # The device listener next, right after the catch-up and before anything
    # its sessions use stops: its graceful shutdown is capped, and its task
    # cancelled after 6 s.
    if _device_listener is not None:
        await _device_listener.stop()

    if curation_scheduler is not None:
        await curation_scheduler.stop()

    # Cluster 8 Phase 5: vault layer shutdown
    if vault_filewatcher is not None:
        try:
            vault_filewatcher.stop()
        except Exception as e:
            logger.warning("Vault filewatcher stop error: %s", e)

    # Drain in-flight conversation extraction tasks BEFORE the writer stops:
    # loop teardown would otherwise cancel them mid commit-protocol (belief
    # retired, successor never written) and drop their vault appends.
    try:
        ch = _health_conversation_handler()
        if ch is not None and hasattr(ch, "aclose"):
            await ch.aclose()
    except Exception as e:  # noqa: BLE001
        logger.warning("ConversationHandler aclose error (non-fatal): %s", e)

    # After the handler drain, so no turn is still calling the host; then the
    # build closes the database and the Spotify client the host used.
    await _stop_capability_host()

    # T2a: stop the extraction dispatcher AFTER the handler drain above (which
    # waits on the backlog). A job still waiting on the service is abandoned
    # and re-dispatched next start; an apply in progress finishes first.
    if extraction_dispatcher is not None:
        try:
            await extraction_dispatcher.stop()
        except (MistError, sqlite3.Error, OSError) as e:
            logger.warning("Extraction dispatcher stop error (non-fatal): %s", e)

    if vault_writer is not None:
        try:
            await vault_writer.stop()
        except Exception as e:
            logger.warning("Vault writer stop error: %s", e)
    if vault_sidecar is not None:
        try:
            vault_sidecar.close()
        except Exception as e:
            logger.warning("Vault sidecar close error: %s", e)

    logging.getLogger().removeHandler(log_handler)
    if system_status_task is not None:
        system_status_task.cancel()
        from backend import system_metrics

        system_metrics.shutdown_gpu()
    extraction_status_task.cancel()
    health_status_task.cancel()
    heartbeat_task.cancel()
    health_probe_task.cancel()
    # Awaited, as the gpu-yield task is below: a probe can be suspended in
    # `asyncio.to_thread` (the paused-marker read), and a loop closed before the
    # cancel lands destroys the task while it is still pending.
    await asyncio.gather(health_probe_task, return_exceptions=True)
    await _stop_uptime_recorder(uptime_task)
    broadcaster_task.cancel()
    power_watcher_task.cancel()
    await asyncio.gather(power_watcher_task, return_exceptions=True)
    if gpu_yield_task is not None:
        gpu_yield_task.cancel()
        # Waits for the cancel to land; `return_exceptions` keeps a loop that
        # already died (logged by its done-callback) from raising here.
        await asyncio.gather(gpu_yield_task, return_exceptions=True)
    if built.gpu is not None:
        built.gpu.stop()
    # `wait=False` deliberately: a probe thread wedged on the timeout-less Neo4j
    # driver would otherwise hold shutdown for the driver's own 60s acquisition
    # timeout. The pool has one worker and the process is ending.
    if _health_probe_executor is not None:
        _health_probe_executor.shutdown(wait=False)
    # After the probe loop is cancelled, so no new health call starts on it.
    await _close_extraction_health_client()
    # Last: each lane's worker thread stops once nothing can submit to it.
    built.close()


# FastAPI app with lifespan
app = FastAPI(title="Mist.AI Voice Server", lifespan=lifespan)


def configure_cors(app: FastAPI) -> None:
    """Add `CORSMiddleware` allowing only the browser origins in `allowed_origins()`.

    A function rather than inline so the tests can configure a fresh app under a
    changed `MIST_ALLOWED_ORIGINS` without reloading this module, because the
    middleware reads its options once, here.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins()),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


configure_cors(app)


@app.get("/")
async def root():
    """Health check."""
    return {"status": "online", "service": "Mist.AI Voice Server"}


@app.get("/health")
async def health() -> dict:
    """Report measured dependency health, never a literal.

    Until this wiring landed the endpoint returned `"status": "healthy"` as a
    constant: it was true the moment the process could answer an HTTP request,
    including with Neo4j down, llama-server unreachable and the broadcaster
    dead. The body now renders `HealthRegistry.snapshot()`, which is a cache of
    probes that actually called their dependencies, plus the three fields that
    predate it.

    There is deliberately no ADR for this shape -- ADR-017 governs the WebSocket
    contract, not this HTTP endpoint -- so this docstring and
    `tests/unit/test_health_probes.py` are the contract. Each claim below is
    separately checkable against `backend/health.py`.

    Always HTTP 200, in every state, including `unhealthy`
    ------------------------------------------------------
    The body is the machine-readable answer; the status code is not. The
    hydrator fetches this with `urllib` under
    `except (urllib.error.URLError, OSError)`
    (`scripts/hydration/target.py:91-100`), and `HTTPError` is a subclass of
    `URLError`, so any non-2xx would be caught there and reported as
    `HydrationTargetError("could not be reached")` -- converting a clear
    refusal into a false "unreachable", against a dev backend that plausibly
    runs degraded on purpose. The compose healthcheck does the derivation
    instead, exiting non-zero on `restart_recommended`.

    Fields from the snapshot
    ------------------------
    - `status`: `healthy` / `degraded` / `yielded` / `unhealthy`, derived from
      `checks` by `health.derive_status`, which is pure and reproducible from
      this payload. `yielded` means a check reports `status: "yielded"`: the
      LLM was stopped on purpose to free the GPU, so MIST cannot hold a
      conversation, by choice rather than by fault. It outranks `degraded`
      and yields to `unhealthy` (another fatal check down), and a yielded
      check that goes stale reads unknown, so a dead probe loop reads
      `degraded`, never `yielded`. A check that reports `status: "paused"`
      counts as up, so extraction waiting for its night window leaves the
      status `healthy`; a stale paused check reads unknown like any other.
    - `restart_recommended`: whether restarting the PROCESS would plausibly
      help, which is a different question from `status`. Advisory only -- Docker
      restart policies react to container exit, not to healthcheck state, and
      there is no autoheal sidecar, so nothing acts on this today. A yielded
      or paused check never sets it.
    - `probe_loop`: `alive`, `dead` (no cycle completed within
      `dead_after_seconds`, so the cache is unrefreshed), or
      `PROBE_LOOP_NOT_STARTED` when no registry exists -- reachable only
      outside a completed `lifespan`, and reported as `unhealthy` with an empty
      `checks` mapping, because nothing verified is not healthy.
    - `checks`: one entry per dependency. `status: null` means NOT MEASURED and
      carries a `reason` from a closed vocabulary; it never means `up`.
      `status: "yielded"` (reason `gpu_yielded`) is set by the `llm` and
      `extraction_service` checks when given a yield state that reads
      `yielded`, and records `consecutive_failures` 0; a yield state of
      `resume_failed` reads `down` on the `llm` check. `status: "paused"`
      (reason `extraction_paused`) is set only by `extraction_service`, and
      also records `consecutive_failures` 0.
      The checks are `neo4j`, `llm`, `event_store`, `vault_sidecar`,
      `broadcaster` and three that say whether MIST is still learning and
      backed up, each `severity: degrades` and `restart_repairs: false`:
      - `extraction_service`: `yielded` while gaming mode holds the GPU, else
        `paused` while the host CLI's current paused marker
        (`<MIST_OPS_DIR>/extraction-paused`) says the service was stopped on
        purpose, without calling it in either case. Otherwise its own
        `GET /v1/health` call to the extraction service every cycle, so a
        switched-off host reads `down` / `unreachable` even while the
        dispatcher is idle. Also `down` for `query_failed`,
        `service_degraded`, `epoch_mismatch`, or `unavailable_at_boot` when
        the dispatcher failed to start; null `disabled_by_config` with no
        dispatcher or in `off` mode.
      - `extraction_backlog`: the producer behind the `extraction` block,
        `down` for `stalled`, `backlog_overdue` (oldest pending turn past
        `MIST_HEALTH_BACKLOG_MAX_AGE_S`) and `dead_lettered`, in that order, or
        `query_failed` when the backlog could not be read -- where the
        `extraction` block shows zero counts.
      - `backup`: the backup scheduler's status file at
        `MIST_BACKUP_STATUS_PATH`; null `disabled_by_config` when that is
        unset, otherwise `down` for `backup_never_run`, `query_failed`,
        `backup_suspect`, `backup_failed` or `backup_overdue` (last success
        older than `MIST_BACKUP_MAX_AGE_HOURS`).
    - `checked_at`: wall time of the last COMPLETED cycle, null before the
      first.

    Fields that predate the registry, preserved for existing consumers
    -----------------------------------------------------------------
    - `models_loaded`: `_models_ready`, set after the runtime's build returns.
      A plain bool. Two consumers outside this module, not one:
      `tests/unit/hydration/test_health_isolation_flag.py`, which asserts it is
      present, and a documented operator wait-loop at
      `scripts/eval_harness/websocket_gauntlets_runbook.md:96-97`
      (`until curl ... | grep -q "models_loaded.:true"; do sleep 3; done`).
      That loop is the point of the fix rather than an incidental reader: the
      field once read an object bound before its models loaded, so the loop
      exited almost at once and handed the operator a ready signal roughly
      120s of model loading early. It now blocks until the models are up.
    - `active_connections`: live WebSocket count, `SessionRegistry.count()`.
    - `voice`: the runtime's engine report (`engines`, `tts_voice`,
      `vram_budget_mib`, `vram_over_budget`) and `power_level`; null before
      the runtime is built.
    - `hydration_isolation` (F4): the positive handshake the hydrator requires
      before sending its first turn. The live backend never sets
      `MIST_HYDRATION_ISOLATION`, so this field is how a hydrator pointed at the
      wrong port finds out before it writes anything. It must stay a plain bool:
      `scripts/hydration/target.py:126` tests it with `is not True`, so a
      truthy string or an int would be refused as a non-isolated target.
    - `extraction`: a subset of `GET /extraction/status` (state, backlog depth,
      dead letters, unrecorded turns, service reachability), from the same
      producer, so the two cannot disagree.

    Added after the registry
    ------------------------
    - `devices`: the room device listener (`backend/devices/`): `listener` is
      `off` (not enabled, or stopped), `on` (serving) or `failed` (it could not
      start; the desktop path is unaffected), and `connected` the sorted names
      of the devices with a live session.

    Returns:
        The snapshot mapping with those fields added. Returned as a dict
        so FastAPI serializes it with a 200.
    """
    from backend.extraction_backlog.status import extraction_status, health_block

    payload = _health_snapshot()
    payload["models_loaded"] = _models_ready
    payload["active_connections"] = _session_count()
    payload["hydration_isolation"] = _hydration_isolation_for_health()
    payload["extraction"] = health_block(extraction_status(extraction_dispatcher))
    payload["voice"] = _voice_health()
    listener = _device_listener
    device_app = _device_app
    payload["devices"] = {
        "listener": listener.state() if listener is not None else _device_listener_state,
        "connected": device_app.connected() if device_app is not None else [],
    }
    return payload


def _voice_health() -> dict[str, object] | None:
    """The runtime's engine report for `/health`, or None before the runtime is built.

    `engines` is each slot's status as `voice_state` reports it, `tts_voice` the
    synthesizer's identity (engine `off` with TTS off), and `power_level` the
    level every session shows, read now.
    """
    from backend.realtime.session import tts_voice_of

    built = runtime
    if built is None:
        return None
    engines = built.engines
    services = built.services
    gate = services.power_gate
    return {
        "engines": engines.statuses.model_dump(mode="json"),
        "tts_voice": tts_voice_of(engines, built.settings).model_dump(mode="json"),
        "vram_budget_mib": built.settings.placement.vram_budget_mib,
        "vram_over_budget": engines.vram_over_budget,
        "power_level": gate.level() if gate is not None else services.power_level(),
    }


def _health_snapshot() -> dict:
    """Render the probe cache, or the honest answer when there is no registry.

    `snapshot()` builds a fresh mapping per call and does no I/O, so the caller
    may add fields to what comes back.

    A missing registry is not an error and not a healthy answer: it means
    `lifespan` has not built one, so nothing has been measured. That renders as
    an empty `checks` mapping, which `derive_status` reports `unhealthy` with
    `restart_recommended` false -- a restart does not add probes.

    Returns:
        A mapping with the snapshot's keys, always.
    """
    registry = _health_registry
    if registry is None:
        status, restart_recommended = derive_status({}, PROBE_LOOP_NOT_STARTED)
        return {
            "status": status,
            "restart_recommended": restart_recommended,
            "probe_loop": PROBE_LOOP_NOT_STARTED,
            "probe_interval_seconds": None,
            "checked_at": None,
            "checks": {},
        }
    return registry.snapshot()


@app.get("/extraction/status")
async def get_extraction_status():
    """The extraction backlog's `ExtractionStatus`, the same object the WS pushes.

    Never errors: with no dispatcher the state is `disabled` with zero counts
    and the process's `unrecorded_turns`.
    """
    from backend.extraction_backlog.status import extraction_status

    return extraction_status(extraction_dispatcher).model_dump(mode="json")


def _capability_reason(text: str | None) -> str | None:
    """Redact a reason and cut it to `CAPABILITY_REASON_MAX_CHARS`."""
    if text is None:
        return None
    return redact(text, _capability_secrets)[:CAPABILITY_REASON_MAX_CHARS]


def _capability_secret_presence(config: CapabilitiesConfig | None) -> list[dict]:
    """Whether each configured capability's secret is present; never a value.

    The Home Assistant token counts as present when `MIST_HA_TOKEN` or
    `MIST_HA_TOKEN_FILE` is set, and the Spotify token when its file exists,
    which the authorize CLI writes at any time, so it is checked per request.
    """
    if config is None:
        return []
    secrets: list[dict] = []
    if "lights" in config.capabilities and config.lights.ha_mcp_url is not None:
        secrets.append({"name": "MIST_HA_TOKEN", "present": config.lights.ha_token_configured})
    if "music" in config.capabilities and config.music.spotify_client_id is not None:
        token_file = Path(config.music.spotify_token_path)
        secrets.append({"name": "MIST_SPOTIFY_TOKEN_PATH", "present": token_file.is_file()})
    return secrets


def _capabilities_payload() -> dict:
    """The `GET /capabilities` body: the build, each capability, secrets and stats.

    Every reason passes through `redact` and is capped, and no secret value is
    read here, so the body is safe for any allowed page to read.
    """
    build = _capability_build
    host = build.host if build is not None else None
    power = _capability_power
    capabilities: list[dict] = []
    if build is not None and host is not None:
        statuses = {status.name: status for status in host.statuses()}
        for capability in build.capabilities:
            status = statuses[capability.name]
            capabilities.append(
                {
                    "name": capability.name,
                    "state": status.state,
                    "reason": _capability_reason(status.reason),
                    "tools": [spec.name for spec in capability.tools()],
                    "provider_tools": list(capability.provider_tools()),
                }
            )
    return {
        "enabled": host is not None,
        "reason": _capability_reason(
            build.disabled_reason if build is not None else "capabilities not started"
        ),
        "power_level": power.level() if power is not None else None,
        "decode_available": host.decode_available() if host is not None else False,
        "capabilities": capabilities,
        "secrets": _capability_secret_presence(_capability_config),
        "stats": host.stats() if host is not None else {},
    }


@app.get("/capabilities", response_model=None)
async def get_capabilities(request: Request) -> dict | JSONResponse:
    """The capability host's state, for the status view and the live checks.

    CORS allows the configured origins only, but a page a browser loads from
    any site can still send this request, so the same Origin rule as `/ws`
    applies: a request whose `Origin` the allowlist refuses gets 403, and one
    with no `Origin` (a local program) or an allowed one gets the body.
    """
    if not origin_allowed(request.headers.get("origin"), allowed_origins()):
        return JSONResponse(status_code=403, content={"detail": "origin not allowed"})
    return _capabilities_payload()


@app.get("/capabilities/usage", response_model=None)
async def get_capability_usage(
    request: Request, days: int = Query(7, ge=1, le=90)
) -> dict | JSONResponse:
    """Capability use and alarm events per local day: the milestone metric's source.

    The body is `CapabilityHost.usage`'s: `tz`, `since`, `alarms_since`, and `days` and
    `alarms`, newest first with one entry per day. It holds names, dates and numbers, never an
    argument, summary or secret. The Origin rule is `GET /capabilities`'s: a refused `Origin`
    gets 403, none or an allowed one gets the body.

    - `days` (1-90, default 7); out of range is a 422 from the query validation.
    - With no host the body is `{"enabled": false, "reason"}`, as `GET /capabilities` reports.
    - A capabilities database that cannot be read is a 503 with
      `{"error": "store_unavailable"}`, as the `/usage` views answer.
    """
    if not origin_allowed(request.headers.get("origin"), allowed_origins()):
        return JSONResponse(status_code=403, content={"detail": "origin not allowed"})
    host = _capability_host()
    if host is None:
        build = _capability_build
        return {
            "enabled": False,
            "reason": _capability_reason(
                build.disabled_reason if build is not None else "capabilities not started"
            ),
        }
    try:
        return await host.usage(days)
    except MistError as exc:
        logger.warning("GET /capabilities/usage: %s", _capability_reason(str(exc)))
        return JSONResponse(status_code=503, content={"error": "store_unavailable"})


@app.get("/usage/turns-per-day")
async def usage_turns_per_day(days: int = Query(7, ge=1, le=90)):
    """Real-origin turns per local day, counts only: ADR-042's measure of real use.

    This endpoint is how the lead reads real use without opening the event
    store: it answers with dates and integers and nothing else, so reading
    the counts never means reading a conversation. The body is exactly
    `{"timezone", "origin": "real", "days": [{"date", "real_turns"}]}`,
    with no derived field, and holds no session id, utterance, response or
    time finer than a date.

    - `days` (1-90, default 7) is dense, oldest first, ends today in
      `timezone`, and covers exactly the days asked for; out of range is a
      422 from the query validation.
    - `timezone` is `MIST_TIMEZONE`, default UTC; an unknown name falls
      back to UTC, is logged, and is reported here as `UTC`.
    - A turn counts when its session's origin is `real` or its session row
      is missing, the event store's own rule.
    - A store that is missing, locked or unreadable is a 503 with
      `{"error": "store_unavailable"}`. The store is opened read-only through
      a `mode=ro` URI, never through `EventStore`, so a missing file is not
      created.

    The read runs in a worker thread: SQLite waits up to
    `usage.BUSY_TIMEOUT_SECONDS` on a lock, which must not stall the event
    loop that carries the WebSocket.
    """
    from fastapi.responses import JSONResponse

    from backend.event_store.usage import (
        UsageStoreUnavailableError,
        build_turns_per_day,
        resolve_store_path,
    )
    from backend.knowledge.config import EventStoreConfig

    # `EventStoreConfig` alone, not `KnowledgeConfig.from_env()`: the counts
    # need one path, and an unrelated malformed setting should not turn them
    # into a 500.
    db_path = resolve_store_path(EventStoreConfig.from_env().db_path)
    try:
        result = await asyncio.to_thread(build_turns_per_day, db_path, days=days)
    except UsageStoreUnavailableError as exc:
        logger.warning("GET /usage/turns-per-day: %s", exc)
        return JSONResponse(status_code=503, content={"error": "store_unavailable"})
    return result.to_payload()


@app.get("/usage/uptime")
async def usage_uptime(days: int = Query(7, ge=1, le=90)):
    """Seconds per local day that MIST was healthy, degraded, yielded or unhealthy.

    Counts only: this is how the lead reads MIST's uptime without opening the
    event store or a conversation. The body is `{"timezone", "current",
    "days": [{"date", "healthy_s", "degraded_s", "yielded_s", "unhealthy_s",
    "unrecorded_s", "yielded_degraded_s"}]}`, a zone name, a status word,
    dates and integers, with no session id, utterance or time finer than a
    date.

    - The seconds come from the uptime ledger the server's own tick loop
      writes every `MIST_UPTIME_RECORD_S` seconds. Time with no tick behind
      it, because MIST was not running or had not yet ticked, is
      unrecorded, never up: an interval ends at its last tick.
    - Each day's first five values sum to that local day's real length
      (86400 s, or 82800 s or 90000 s on a DST change), and today's to the
      seconds elapsed so far. `yielded_s` is time with the LLM stopped on
      purpose to free the GPU (gaming mode); `yielded_degraded_s` is the part
      of it during which another check was also not up, and is not one of
      the five.
    - `current` is the live `/health` status at the time of the request,
      not a stored row.
    - `days` (1-90, default 7) is dense, oldest first, and ends today in
      `timezone`; out of range is a 422 from the query validation.
    - `timezone` is `MIST_TIMEZONE`, default UTC; an unknown name falls
      back to UTC, is logged, and is reported here as `UTC`.
    - A store that is missing, locked or unreadable is a 503 with
      `{"error": "store_unavailable"}`. The store is opened read-only through
      a `mode=ro` URI, never through `EventStore`, so a missing file is not
      created.

    The read runs in a worker thread, as `/usage/turns-per-day`'s does.
    """
    from fastapi.responses import JSONResponse

    from backend.event_store.status_intervals import (
        UptimeStoreUnavailableError,
        build_uptime_per_day,
    )
    from backend.event_store.usage import resolve_store_path
    from backend.knowledge.config import EventStoreConfig

    db_path = resolve_store_path(EventStoreConfig.from_env().db_path)
    current = _health_snapshot()["status"]
    try:
        result = await asyncio.to_thread(build_uptime_per_day, db_path, current=current, days=days)
    except UptimeStoreUnavailableError as exc:
        logger.warning("GET /usage/uptime: %s", exc)
        return JSONResponse(status_code=503, content={"error": "store_unavailable"})
    return result.to_payload()


def _hydration_isolation_for_health() -> bool:
    """Report the F4 flag, degrading a config typo to False rather than a 500.

    `is_hydration_isolation_active` raises on an unrecognized value, which is
    correct for a guard invoked from a CLI that can print the refusal. A health
    endpoint is different: raising here would hand the hydrator a connection
    error instead of an answer, and an operator a stack trace instead of a
    cause.

    False is the fail-closed direction -- the hydrator refuses unless True --
    so a typo blocks the run either way. The warning names the raw value so the
    block is diagnosable from the target container's log.
    """
    try:
        return is_hydration_isolation_active()
    except EvalIsolationError as exc:
        logger.warning("hydration_isolation unparsable, reporting False: %s", exc)
        return False


async def _run_capability_call(
    call: Coroutine[object, object, _T], what: str
) -> tuple[_T | None, BaseException | None]:
    """Run one capability call as its own task, and return its result or its failure.

    No capability exception may reach the session or the loop's exception
    handler. `asyncio.wait` returns when the task ends instead of re-raising,
    and the outcome is read from `task.cancelled()` and then `task.exception()`,
    so any exception is contained without a broad `except` clause (CLAUDE.md,
    Errors). The task only computes; the caller queues what it returns on the
    session's lanes.

    If the caller is cancelled while it waits (its session disconnected), the
    `finally` cancels the task, so no capability call outlives its connection.

    Returns:
        `(result, None)` when the call returned; `(None, exception)` when it
        raised, logged here at ERROR with its traceback; `(None, None)` when it
        was cancelled, which the caller answers with nothing.
    """
    task = asyncio.create_task(call, name=f"capability-{what}")
    try:
        await asyncio.wait({task})
    finally:
        if not task.done():
            task.cancel()
    if task.cancelled():
        logger.info("Capability call %s was cancelled; nothing is sent", what)
        return None, None
    error = task.exception()
    if error is not None:
        logger.error("Capability call %s failed", what, exc_info=error)
        return None, error
    return task.result(), None


def _capability_failure_text(host: "CapabilityHost", error: BaseException) -> str:
    """The client's text for an unexpected capability failure: class and message, redacted."""
    text = host.redact(f"{type(error).__name__}: {error}")
    return text[:CAPABILITY_REASON_MAX_CHARS]


def _ws_error(kind: str, message: str) -> dict[str, object]:
    """The ADR-017 `error` event, never retriable, for a capability message."""
    return {
        "type": "error",
        "kind": kind,
        "message": message,
        "retriable": False,
        "context": None,
    }


def _validate_capability_message(model: type[_M], data: dict[str, object]) -> _M | str:
    """Return the validated message, or the text of the `validation` error that answers it.

    The text names each failing field with pydantic's message and never the
    input value, so a client's payload is not echoed back.
    """
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'message'}: {error['msg']}"
            for error in exc.errors()
        )
        return f"invalid {data.get('type')}: {problems}"[:CAPABILITY_REASON_MAX_CHARS]


async def _capability_snapshot() -> list[dict[str, object]]:
    """The host's connect snapshot for one connecting client, or nothing with no host.

    The payloads go to the connecting client only, after `session_started`. A
    failure building them is answered with a `server` error and the connection
    goes on.
    """
    host = _capability_host()
    if host is None:
        return []
    events, error = await _run_capability_call(host.snapshot_events(), "connect snapshot")
    if error is not None:
        return [_ws_error("server", _capability_failure_text(host, error))]
    return list(events or ())


async def _decided_payloads(
    handler: "ConversationHandler", session_id: str, message: ActionDecisionMessage
) -> list[str]:
    """Decide a proposal and serialise what to broadcast, inside the contained task."""
    payloads = await handler.decide_proposal(session_id, message.proposal_id, message.decision)
    return [json.dumps(payload) for payload in payloads]


async def _alarm_action_result(
    host: "CapabilityHost", message: AlarmActionMessage, endpoint: str
) -> dict[str, object]:
    """Run a client's alarm action from `endpoint` and build its `alarm_action_result`."""
    outcome = await host.alarm_action(
        message.alarm_id, message.action, message.minutes, endpoint=endpoint
    )
    return alarm_action_result_payload(
        message.alarm_id, message.action, message.request_id, outcome, datetime.now(UTC)
    )


async def _alarm_report_result(
    host: "CapabilityHost", message: AlarmReportMessage, endpoint: str
) -> dict[str, object]:
    """Apply `endpoint`'s alarm report and build its `alarm_action_result`."""
    outcome = await host.alarm_report(message, endpoint=endpoint)
    return alarm_action_result_payload(
        message.alarm_id, "report", message.request_id, outcome, datetime.now(UTC)
    )


def _failed_alarm_result(
    host: "CapabilityHost",
    alarm_id: str,
    action: Literal["stop", "snooze", "dismiss", "report"],
    request_id: str | None,
    error: BaseException,
) -> dict[str, object]:
    """The `alarm_action_result` for an unexpected exception: `failed`, in place of `error`."""
    reason = _capability_failure_text(host, error)
    outcome = ToolOutcome(status="error", summary=f"Alarm {action} failed.", reason=reason)
    return alarm_action_result_payload(alarm_id, action, request_id, outcome, datetime.now(UTC))


def _on_capability_message(session_id: str, message: CapabilityMessage) -> None:
    """`SessionServices.on_capability_message`: answer it on its own task, off the receive loop.

    The session hands the message over synchronously, so the capability call runs as a task the
    server tracks per session; the session's disconnect cancels what is still running.
    """
    registry = session_registry
    if registry is None:
        return

    def send(payload: dict[str, object]) -> None:
        registry.send_to(session_id, payload)

    task = asyncio.get_running_loop().create_task(
        _handle_capability_message(message.model_dump(mode="json"), session_id, send),
        name=f"capability-message-{message.type}",
    )
    tasks = _capability_tasks.setdefault(session_id, set())
    tasks.add(task)
    task.add_done_callback(tasks.discard)


async def _cancel_capability_tasks(session_id: str) -> None:
    """Cancel the session's capability calls still in flight, and wait for the cancels."""
    tasks = _capability_tasks.pop(session_id, set())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _handle_capability_message(
    data: dict[str, object], session_id: str, send: "Callable[[dict[str, object]], None]"
) -> None:
    """Answer `action_decision`, `alarm_action` or `alarm_report` from one session.

    The session already validated the message against the wire contract; it is
    validated again with capabilities' own model, as main's server did, because
    that model is what the host takes. Each capability call runs through
    `_run_capability_call`. `action_decision`'s payloads are broadcast (every
    capability payload is, until turns are routed per connection); an alarm
    action or report gets exactly one `alarm_action_result`, on the session's
    `control` lane through `send`.

    An alarm action or report is attributed to the endpoint its session is
    bound to (`SessionRegistry.endpoint_of`): `desktop` for `/ws`, the device's
    name for a room device. A session the registry no longer knows is answered
    with a `server` error and one WARNING, and never reaches the host under
    another endpoint's name.
    """
    host = _capability_host()
    if host is None:
        send(_ws_error("validation", "capabilities disabled"))
        return
    msg_type = data.get("type")
    if msg_type == "action_decision":
        await _handle_action_decision(host, data, session_id, send)
        return
    registry = session_registry
    endpoint = registry.endpoint_of(session_id) if registry is not None else None
    if endpoint is None:
        logger.warning(
            "Capability message %s from session %s, which is no longer connected; not applied",
            msg_type,
            session_id,
        )
        send(_ws_error("server", f"session {session_id} is not connected"))
        return
    if msg_type == "alarm_action":
        action = _validate_capability_message(AlarmActionMessage, data)
        if isinstance(action, str):
            send(_ws_error("validation", action))
            return
        result, error = await _run_capability_call(
            _alarm_action_result(host, action, endpoint), "alarm_action"
        )
        if error is not None:
            result = _failed_alarm_result(
                host, action.alarm_id, action.action, action.request_id, error
            )
    else:
        report = _validate_capability_message(AlarmReportMessage, data)
        if isinstance(report, str):
            send(_ws_error("validation", report))
            return
        result, error = await _run_capability_call(
            _alarm_report_result(host, report, endpoint), "alarm_report"
        )
        if error is not None:
            result = _failed_alarm_result(host, report.alarm_id, "report", report.request_id, error)
    if result is not None:
        send(result)


async def _handle_action_decision(
    host: "CapabilityHost",
    data: dict[str, object],
    session_id: str,
    send: "Callable[[dict[str, object]], None]",
) -> None:
    """Resolve a proposal for this connection's own session, and broadcast what it produced.

    `decide_proposal` belongs to the conversation handler, which merges the
    decision into the session's history. With no handler no turn reached the
    host, so there is no proposal to decide, and the client is told so.
    """
    decision = _validate_capability_message(ActionDecisionMessage, data)
    if isinstance(decision, str):
        send(_ws_error("validation", decision))
        return
    handler = _health_conversation_handler()
    if handler is None:
        send(_ws_error("server", "no conversation handler, so no proposal can be decided"))
        return
    payloads, error = await _run_capability_call(
        _decided_payloads(handler, session_id, decision), "action_decision"
    )
    if error is not None:
        send(_ws_error("server", _capability_failure_text(host, error)))
        return
    for payload in payloads or ():
        await message_queue.put(payload)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Serve one `RealtimeSession` per connection, through the session registry.

    The authoritative message contract is the realtime protocol, 2.0.0
    (`backend/realtime/protocol.py`): main's 1.4.0 control messages plus
    protocol 2's own. `session_started` carries `protocol_version` and
    `capabilities.voice`; the session then owns the socket. Its receive loop
    sends binary frames to uplink handling and text to control handling and
    never awaits a turn, and its sender drains the `control`, `turn` and `bulk`
    lanes, so an `interrupt` is handled while a turn is still generating.

    Every connection is bound `desktop` (D7). With a capability host, the
    connect snapshot (`capabilities_status`, then `alarm_state` when the alarm
    state can be read) follows `session_started` to this client only.
    `action_decision`, `alarm_action` and `alarm_report` reach the host through
    `_on_capability_message`; with no host they get a `validation` error,
    "capabilities disabled". An alarm action or report is answered with one
    `alarm_action_result` to this session only. Process-wide messages arrive
    through the registry: `broadcast_messages` (`action_proposed`,
    `action_resolved`) and `CapabilitySink` (`alarm_fired` to `desktop`
    sessions only).

    On disconnect the session commits its turn before `run` returns; only then
    is the handler told the session ended, so the session note sees the last
    turn. A browser origin outside `allowed_origins()` is closed with 1008
    before `accept`, because browsers apply no CORS to WebSockets
    (`backend/origins.py`).
    """
    origin = websocket.headers.get("origin")
    if not origin_allowed(origin, allowed_origins()):
        logger.warning("WebSocket refused origin: %s", origin)
        await websocket.close(code=1008)
        return

    await websocket.accept(headers=None)
    await _serve_session(websocket, WS_ENDPOINT, WS_ENDPOINT)


async def _serve_session(
    websocket: "WebSocket | DeviceSocket",
    endpoint: str,
    endpoint_kind: Literal["desktop", "room"],
) -> None:
    """Serve one accepted connection as a `RealtimeSession` bound to `endpoint`.

    Both routes call it: `/ws` with `desktop`, and the device listener
    (`backend/devices/listener.py`) with the device's registered name and kind
    `room`, after it has authenticated the device. Before the runtime and the
    registry exist the client gets a retriable `server` error and a close 1013.
    """
    registry = session_registry
    built = runtime
    if registry is None or built is None:
        await websocket.send_json(
            {
                "type": "error",
                "kind": "server",
                "message": "Server not ready",
                "retriable": True,
                "context": None,
            }
        )
        await websocket.close(code=1013)
        return

    # A fresh id per connection, threaded into every turn the session opens, so
    # the event store session, the context cache key and the session note all
    # key off it rather than a shared placeholder.
    session_id = str(uuid.uuid4())
    session = registry.create_session(
        session_id=session_id,
        websocket=websocket,  # type: ignore[arg-type]
        endpoint=endpoint,
        endpoint_kind=endpoint_kind,
    )
    logger.info("Client connected: %s (total: %d)", endpoint, registry.count())
    try:
        session.send_control(
            SessionStartedMessage(
                session_id=session_id,
                protocol_version=PROTOCOL_VERSION,
                mist_state="idle",
                capabilities=SessionCapabilities(
                    tts_enabled=bool(config.tts_enabled),
                    vad_enabled=True,
                    voice=built.voice_capabilities(),
                ),
            )
        )
        for payload in await _capability_snapshot():
            session.send_control(payload)
        await session.run()
    finally:
        # A no-op once `run` has torn the session down; it covers a cancel
        # before `run` started.
        registry.unregister(session)
        await _cancel_capability_tasks(session_id)
        logger.info("Client disconnected (remaining: %d)", registry.count())
        await _end_handler_session(session_id)


async def _end_handler_session(session_id: str) -> None:
    """Tell the handler this connection's session ended (Gap #1a, ADR-011 bucket 2).

    Scoped to THIS connection's session: `end_session(session_id=None)` ends
    every session the handler tracks, which would write the vault notes of
    every other open connection early and evict their paths. Failures are
    logged and swallowed, per Invariant 6.
    """
    handler = _health_conversation_handler()
    if handler is None or not hasattr(handler, "end_session"):
        return
    try:
        await handler.end_session(session_id)
    except (MistError, sqlite3.Error, OSError) as exc:
        logger.warning("Session-end signal handler failed (non-fatal): %s", exc, exc_info=False)


if __name__ == "__main__":
    # Pass the app OBJECT, never the "server:app" import STRING.
    #
    # The image runs `python backend/server.py`, which binds this file to
    # `__main__`. Uvicorn resolves a string target with
    # `importlib.import_module("server")`, which finds this same file again on
    # sys.path and executes the ENTIRE module body a second time as a second
    # module object. `app`, `message_queue`, the runtime, the session registry
    # and `log_handler` all existed twice,
    # the root logger collected two FileHandlers on the same log file (every
    # line was written twice), and the copy uvicorn actually served was the
    # `server` one -- so state written by the `__main__` copy was invisible to
    # every request. `import_from_string` returns a non-string target
    # unchanged, so passing the object cannot re-import.
    #
    # Trade-off, stated explicitly: an app object disables uvicorn's `reload`
    # and multi-`workers` modes, which both require an import string to respawn
    # the app in a child process. `reload` is False here and MIST runs a single
    # worker holding GPU models and in-process state, so neither is usable
    # anyway -- but switching either one back on now requires moving the entry
    # point out of this file rather than just flipping the flag.
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )
