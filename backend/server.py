"""Voice AI WebSocket Server.

Based on CSM demo architecture - production-ready for web frontend
"""

import asyncio
import contextlib
import functools
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # Reconfigure stdout/stderr to use UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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
# ambiguous with dependencies/csm/config.py -- which is what the old
# "import config BEFORE voice_processor" ordering existed to work around. The
# qualified name removes both hazards outright.
from backend.config import DEFAULT_CONFIG  # isort:skip
from backend.voice_processor import VoiceProcessor  # isort:skip
from backend.factories import (  # isort:skip
    build_curation_scheduler,
    build_phase3_components,
    build_sidecar_index,
    build_vault_writer,
)
from backend.knowledge.config import KnowledgeConfig  # isort:skip
from backend.log_handler import WebSocketLogHandler  # isort:skip

if TYPE_CHECKING:
    from backend.interfaces import EventStoreProvider
    from backend.knowledge.extraction.tool_usage_tracker import ToolUsageTracker
    from backend.llm import StreamingLLMProvider

# Setup logging -- console + persistent file
_log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=_log_format)

# Persistent file log (survives container removal via bind mount)
_log_dir = Path("/app/logs")
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(_log_dir / "mist-backend.log")
_file_handler.setFormatter(logging.Formatter(_log_format))
_file_handler.setLevel(logging.DEBUG)  # Capture everything to disk
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger(__name__)

# WebSocket protocol version sent on session_started per ADR-017. Bump
# minor on additive event/field additions, major on breaking changes.
# v1.1.0 (2026-05-25): documents vault_results + system_status (shipped 2026-05-11).
PROTOCOL_VERSION = "1.1.0"

# Global state
active_connections: set[WebSocket] = set()
active_connections_lock = asyncio.Lock()
# Connections that opted into log streaming (ADR-017: per-connection
# ephemeral). The global handler gate derives from "any subscriber present".
log_subscribers: set[WebSocket] = set()
message_queue: asyncio.Queue[str | bytes] = asyncio.Queue()
voice_processor: VoiceProcessor | None = None
curation_scheduler = None
log_handler: WebSocketLogHandler | None = None
config = DEFAULT_CONFIG

# Cluster 8 Phase 5: vault layer subsystems (initialized in lifespan)
vault_writer = None
vault_sidecar = None
vault_filewatcher = None
# Phase 5.5: shared InvalidationBus wired from filewatcher to ConversationHandler
vault_invalidation_bus = None


async def broadcast_messages():
    """Background task to broadcast messages to all connected clients."""
    while True:
        message = await message_queue.get()

        # Send to all connected clients
        async with active_connections_lock:
            stale = []
            for websocket in active_connections:
                try:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    elif isinstance(message, str):
                        await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    stale.append(websocket)
            for ws in stale:
                active_connections.discard(ws)

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
                }
            )
            await message_queue.put(payload)
        except Exception as e:  # noqa: BLE001
            logger.error("system_status emit failed: %s", e)


async def health_status_loop(interval_seconds: float = 30.0) -> None:
    """Emit ADR-017 ``health_status`` events every ``interval_seconds``.

    Three booleans for the FE's persistent bottom-left health indicator
    (per prototype ``edges-v10`` ``Offline persistent indicator``):

    - ``llm``: True iff ``voice_processor.models`` is initialized (LLM
      backend loaded)
    - ``agent``: True iff ``voice_processor.models.knowledge.enabled``
      (knowledge subsystem operational; False means degraded MIST)
    - ``local``: True always (if this loop is running, the BE is up)

    Transition-event emission (v1.1+) would supplement this periodic
    push; v1.0 is periodic only.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            llm = bool(voice_processor and voice_processor.models)
            agent = bool(
                voice_processor
                and voice_processor.models
                and voice_processor.models.knowledge
                and voice_processor.models.knowledge.enabled
            )
            local = True
            payload = json.dumps(
                {
                    "type": "health_status",
                    "llm": llm,
                    "agent": agent,
                    "local": local,
                }
            )
            await message_queue.put(payload)
        except Exception as e:  # noqa: BLE001
            logger.error("Health status emit failed: %s", e)


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


def _build_session_note_catchup(voice_processor, vault_writer):
    """Construct the catch-up instance from live server state, or None if ineligible.

    Extracted from `lifespan()` (final-review fix, C1) so the WIRING --
    which collaborator each injected callable actually points at -- is
    testable without booting the full ASGI lifespan, the same testability
    reason `_schedule_session_note_catchup` and
    `_shutdown_session_note_catchup` were extracted in fix round 2.

    `is_conversation_active` is wired to the module-level `active_connections`
    set, NOT `ConversationHandler.sessions`. `sessions` is populated on
    every turn and, in production, is removed from in exactly one place --
    `clear_session` -- whose only caller (`KnowledgeIntegration.clear_session`)
    has zero production callers itself. The WebSocket disconnect path calls
    `end_session`, which pops `_vault_paths` but never touches `self.sessions`.
    Wiring the gate to `sessions` therefore means "this process has ever had
    a conversation," not "a conversation is active right now" -- true
    forever after the first turn, which silently disables `run_forever`'s
    retry (R1.3.1 fix round 1, I5) for the rest of the process's life.
    `active_connections` IS discarded on disconnect (see the `finally`
    block in `websocket_endpoint`), so it reflects live connections
    correctly.

    Returns:
        A `SessionNoteCatchup` ready to schedule, or None when the vault
        layer is disabled, no `ConversationHandler` is reachable yet, or
        the event store is unavailable.
    """
    catchup_handler = (
        voice_processor.models.knowledge.conversation_handler
        if voice_processor and voice_processor.models and voice_processor.models.knowledge
        else None
    )
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
        is_conversation_active=lambda: bool(active_connections),
        is_llm_ready=catchup_handler.session_synthesizer.is_ready,
    )


def _schedule_session_note_catchup(catchup) -> "asyncio.Task":
    """Schedule catch-up as a periodic background task with exception visibility.

    Extracted from `lifespan()` (R1.3.1 fix round 2, N1) so the exact
    coroutine that gets scheduled -- `run_forever`, not a one-shot `run` --
    and the done-callback attachment are both independently testable
    without booting the full ASGI lifespan, which would otherwise require
    faking `VoiceProcessor.initialize` and the entire model stack. See
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


def _resolve_curation_dependencies(voice_processor) -> CurationDependencies:
    """Pull the curation scheduler's live collaborators off the ConversationHandler.

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

    `llm_provider` is taken from the handler rather than left to the factory's
    own `build_llm_provider(config)`: the factory's call passes no
    `debug_logger`, so the deriver's LLM calls would be structurally invisible
    to the `phase: "llm_call"` JSONL stream even with the gate on, and a
    second `LlamaServerProvider` means a duplicate pair of OpenAI clients
    (and connection pools) that nothing closes.

    Extracted from `lifespan()` for the same testability reason as
    `_build_session_note_catchup`: WHICH object each dependency points at is
    the entire content of the fix, and identity cannot be asserted without
    booting the full model stack unless the resolution is its own function.
    See `tests/unit/test_curation_scheduler_wiring.py`.

    Reaches the handler through the same defensive chain the end-session
    broadcast and the shutdown drain already use.

    Returns:
        A `CurationDependencies` holding the live handler's own objects, or
        all-None with `handler_found=False` when no handler is reachable.
    """
    handler = (
        voice_processor.models.knowledge.conversation_handler
        if voice_processor and voice_processor.models and voice_processor.models.knowledge
        else None
    )
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
        llm_provider=handler._provider,
        handler_found=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global voice_processor, curation_scheduler, log_handler
    global vault_writer, vault_sidecar, vault_filewatcher, vault_invalidation_bus

    # Startup
    logger.info("=" * 60)
    logger.info("STARTING MIST.AI VOICE SERVER")
    logger.info("=" * 60)

    # Acquire event loop early -- needed by both filewatcher.start and VAD.
    loop = asyncio.get_running_loop()

    # Single config load shared between vault and curation subsystems.
    knowledge_config = KnowledgeConfig.from_env()

    # Cluster 8 Phase 5: vault layer subsystems are built FIRST so the
    # single server-owned VaultWriter can be plumbed into VoiceProcessor
    # -> ModelManager -> KnowledgeIntegration -> ConversationHandler. Each
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
        # be forwarded to VoiceProcessor -> ModelManager -> KnowledgeIntegration
        # -> build_conversation_handler, wiring ConversationHandler._on_vault_rebuild
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

    # Initialize voice processor with the server-owned vault_writer, sidecar,
    # and invalidation_bus so that the voice-path ConversationHandler shares a
    # single started writer, the retriever routes to the same initialized sidecar
    # (Phase 9), and _on_vault_rebuild is subscribed to the shared bus for
    # ADR-010 read-path cache invalidation (Phase 5.5).
    voice_processor = VoiceProcessor(
        config,
        message_queue,
        vault_writer=vault_writer,
        vault_sidecar=vault_sidecar,
        invalidation_bus=vault_invalidation_bus,
    )
    await voice_processor.initialize()

    # Start message broadcaster
    broadcaster_task = asyncio.create_task(broadcast_messages())
    # Start heartbeat task (5s interval per ADR-017 Heartbeat semantics)
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    # Start health-status task (30s interval per ADR-017 health_status)
    health_status_task = asyncio.create_task(health_status_loop())

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

    # Attach WebSocket log handler to root logger
    log_handler = WebSocketLogHandler(event_loop=loop, message_queue=message_queue)
    logging.getLogger().addHandler(log_handler)
    logger.info("WebSocket log handler attached")

    # Start curation scheduler for periodic graph maintenance.
    #
    # Every falsy-default parameter is supplied explicitly -- see
    # `_resolve_curation_dependencies` for what leaving them defaulted cost
    # (two of the nine jobs structurally incapable of producing output).
    # `voice_processor.initialize()` has already completed above, so the
    # ConversationHandler these come off is the live one.
    curation_deps = _resolve_curation_dependencies(voice_processor)
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
        )
        await curation_scheduler.start()
        logger.info("Curation scheduler started")
    except Exception as e:
        logger.warning("Curation scheduler failed to start: %s", e)
        curation_scheduler = None

    # R1.3.1: synthesize vault notes for sessions that crashed before session
    # end (process kill, container restart) and so never reached
    # ConversationHandler.end_session. Background and deferential -- boot is
    # when the user is most likely to start talking, and a synthesis call
    # would hold the GPU and delay their first token. Reached through the
    # same voice_processor -> models -> knowledge -> conversation_handler
    # chain the shutdown path below already uses to reach the handler.
    # Runs as `run_forever` (periodic, not one-shot): a single pass can come
    # back empty because the LLM backend is still cold at boot or a
    # conversation is live throughout, and a one-shot task would strand the
    # backlog for the rest of the process's life in either case.
    catchup_task: asyncio.Task | None = None
    try:
        catchup = _build_session_note_catchup(voice_processor, vault_writer)
        if catchup is not None:
            catchup_task = _schedule_session_note_catchup(catchup)
            logger.info("Session-note catch-up scheduled as a periodic background task")
    except Exception as e:  # noqa: BLE001 -- catch-up must never block startup
        logger.warning("Session-note catch-up scheduling failed (non-fatal): %s", e)

    logger.info(f"Server ready on ws://{config.host}:{config.port}/ws")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Server shutting down...")

    # Cancel the catch-up background task FIRST, before anything else below
    # touches vault_writer or the ConversationHandler it holds a reference
    # to. curation_scheduler.stop() and ch.aclose() both await, which yields
    # the event loop -- a still-running catch-up pass could resume in that
    # window and reach a write against an already-stopped vault_writer if
    # cancellation happened later (or not at all).
    await _shutdown_session_note_catchup(catchup_task)

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
        ch = (
            voice_processor.models.knowledge.conversation_handler
            if voice_processor and voice_processor.models and voice_processor.models.knowledge
            else None
        )
        if ch is not None and hasattr(ch, "aclose"):
            await ch.aclose()
    except Exception as e:  # noqa: BLE001
        logger.warning("ConversationHandler aclose error (non-fatal): %s", e)

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
    health_status_task.cancel()
    heartbeat_task.cancel()
    broadcaster_task.cancel()
    if voice_processor and voice_processor.models:
        voice_processor.models.shutdown()


# FastAPI app with lifespan
app = FastAPI(title="Mist.AI Voice Server", lifespan=lifespan)

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check."""
    return {"status": "online", "service": "Mist.AI Voice Server"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "models_loaded": voice_processor is not None,
        "active_connections": len(active_connections),
    }


def _reset_voice_state_if_last_connection() -> bool:
    """Clear singleton connection state once no client remains.

    `voice_processor` is shared by every entry in `active_connections`, so its
    connection-scoped state (a parked utterance) outlives the connection that
    produced it and becomes the next session's starting state.

    Guarded on the set being EMPTY rather than firing on every disconnect: with
    a concurrent connection still open, the parked utterance may be that
    connection's, and clearing it would drop a live client's queued message.

    Returns:
        True if the reset ran, False if a connection remains or the voice
        processor has not been constructed yet.
    """
    if active_connections:
        return False
    if voice_processor is None:
        return False
    voice_processor.reset_connection_state()
    return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice + text conversation.

    The authoritative message contract is ADR-017 (WebSocket Message
    Contract, knowledge-vault/Decisions/) -- this docstring only enumerates
    the surface; field-level schemas live in the ADR.

    Inbound (client -> server):
        audio, text, interrupt, reset_vad, log_config, subscribe_logs

    Outbound (server -> client):
        session_started, heartbeat, health_status, system_status,
        vad_status, transcription, state_cycle,
        stream_start / stream_token / stream_complete / stream_cancelled,
        tool_call_started / tool_call_completed, cards_summon /
        cards_dismiss, graph_subgraph, vault_results, form_switch,
        discriminated error, status, log_config_ack, log,
        plus binary audio frames (MIST protocol: MSG_AUDIO_CHUNK /
        MSG_AUDIO_COMPLETE).
    """
    await websocket.accept(headers=None)
    async with active_connections_lock:
        active_connections.add(websocket)

    logger.info(f"Client connected (total: {len(active_connections)})")

    # Ensure voice processor is initialized
    if voice_processor is None:
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
        async with active_connections_lock:
            active_connections.discard(websocket)
        return

    # Send session_started handshake per ADR-017. The session_id is a fresh
    # UUID per WebSocket connection and is threaded into every turn this
    # connection dispatches (see the audio / text branches below), so the
    # EventStore session, the mist-context cache key, and vault session-note
    # frontmatter all key off this value rather than a shared placeholder.
    session_id = str(uuid.uuid4())
    await websocket.send_json(
        {
            "type": "session_started",
            "session_id": session_id,
            "protocol_version": PROTOCOL_VERSION,
            "mist_state": "idle",
            "capabilities": {
                "tts_enabled": bool(config.tts_enabled),
                "vad_enabled": True,
            },
        }
    )

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            msg_type = data.get("type")
            if msg_type is None:
                await websocket.send_json(
                    {
                        "type": "error",
                        "kind": "validation",
                        "message": "Missing 'type' field",
                        "retriable": False,
                        "context": None,
                    }
                )
                continue

            # Handle different message types
            if msg_type == "audio":
                # Complete audio from client (no VAD, just transcribe and process)
                audio_payload = data.get("audio")
                if audio_payload is None:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "kind": "validation",
                            "message": "Missing 'audio' field",
                            "retriable": False,
                            "context": None,
                        }
                    )
                    continue
                audio_data = np.asarray(audio_payload, dtype=np.float32)
                sample_rate = data.get("sample_rate", 16000)

                logger.info(f"Received complete audio: {len(audio_data)} samples @ {sample_rate}Hz")

                # Process complete audio directly (transcribe -> LLM -> TTS).
                # functools.partial because run_in_executor forwards positional
                # args only, and the session id must arrive by keyword to stay
                # readable at the boundary. The executor thread does not inherit
                # this coroutine's context, so the callee publishes the session
                # id into `current_session_id` itself.
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    functools.partial(
                        voice_processor.process_complete_audio,
                        audio_data,
                        sample_rate,
                        session_id=session_id,
                    ),
                )

            elif msg_type == "text":
                # Text message (manual input)
                user_text = data.get("text", "")
                if not user_text:
                    continue
                logger.info(f"Text message from client: '{user_text}'")

                # No transcription broadcast here -- frontend already added
                # the user message optimistically in sendTextMessage().
                # Voice path sends its own transcription from _process_user_speech().

                # Process (will spawn thread internally)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    functools.partial(
                        voice_processor._process_conversation_turn,
                        user_text,
                        session_id=session_id,
                    ),
                )

            elif msg_type == "interrupt":
                # Manual interrupt request
                logger.info("Manual interrupt requested")
                voice_processor.interrupt_flag.set()

                await websocket.send_json({"type": "status", "message": "Interrupt acknowledged"})

            elif msg_type == "reset_vad":
                # Reset VAD state
                voice_processor.reset_vad()
                await websocket.send_json({"type": "status", "message": "VAD reset"})

            elif msg_type == "log_config":
                # Runtime log level control
                action = data.get("action")
                if action != "set_level":
                    await websocket.send_json(
                        {
                            "type": "log_config_error",
                            "message": (f"Invalid action: '{action}'. " "Must be 'set_level'."),
                        }
                    )
                    continue

                level = data.get("level", "")
                if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
                    await websocket.send_json(
                        {
                            "type": "log_config_error",
                            "message": (
                                f"Invalid level: '{level}'. "
                                "Must be one of DEBUG, INFO, WARNING, ERROR."
                            ),
                        }
                    )
                    continue

                target_logger = data.get("logger")
                if not target_logger:
                    await websocket.send_json(
                        {
                            "type": "log_config_error",
                            "message": "Missing 'logger' field.",
                        }
                    )
                    continue

                if log_handler is not None:
                    log_handler.set_logger_level(target_logger, level)

                await websocket.send_json(
                    {
                        "type": "log_config_ack",
                        "logger": target_logger,
                        "level": level,
                    }
                )

            elif msg_type == "subscribe_logs":
                # ADR-017 subscribe_logs: opt-in WebSocket log streaming.
                # Streaming is off by default; FE toggles per session.
                enabled = bool(data.get("enabled", False))
                levels = data.get("levels")
                if levels is not None and not isinstance(levels, list):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "kind": "validation",
                            "message": "subscribe_logs 'levels' must be a list of strings or null",
                            "retriable": False,
                            "context": None,
                        }
                    )
                    continue
                # ADR-017: subscription state is per-connection ephemeral.
                # Track subscribers so a disconnect cannot leave the global
                # handler gate stuck on (every later client would receive the
                # full log stream without opting in).
                if enabled:
                    log_subscribers.add(websocket)
                else:
                    log_subscribers.discard(websocket)
                if log_handler is not None:
                    if log_subscribers:
                        log_handler.set_streaming(True, levels)
                    else:
                        log_handler.set_streaming(False, None)
                await websocket.send_json(
                    {
                        "type": "status",
                        "message": f"Log streaming: {'enabled' if enabled else 'disabled'}",
                    }
                )

            else:
                logger.warning(f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected (remaining: {len(active_connections) - 1})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        async with active_connections_lock:
            active_connections.discard(websocket)
        # Ordering matters: this must run AFTER the discard (so the emptiness
        # check sees this connection gone) and BEFORE `end_session` below (so a
        # parked utterance cannot be drained into a turn while the session it
        # belongs to is being closed).
        _reset_voice_state_if_last_connection()
        # ADR-017: log subscription is per-connection ephemeral -- drop this
        # connection's subscription and close the global gate when no
        # subscriber remains.
        log_subscribers.discard(websocket)
        if not log_subscribers and log_handler is not None:
            log_handler.set_streaming(False, None)
        # Gap #1a / ADR-011 bucket 2: flip status of any active sessions
        # this connection touched. Fire-and-forget; failures swallowed
        # internally per Invariant 6.
        #
        # Scoped to THIS connection's session. `end_session(session_id=None)`
        # ends every session the handler tracks, which was harmless while all
        # connections shared the single "default" session -- but the P6 fix gave
        # each connection its own id, which turned that no-op into a live bug: a
        # disconnect on one connection would end the sessions of every other
        # connection still open, writing their vault notes early and evicting
        # their paths.
        #
        # The parameter was already implemented and documented ("Specific
        # session to end. When None, ends every tracked session_id"); only the
        # call site omitted it. Passing the minted id is the whole fix.
        try:
            handler = (
                voice_processor.models.knowledge.conversation_handler
                if voice_processor and voice_processor.models and voice_processor.models.knowledge
                else None
            )
            if handler is not None and hasattr(handler, "end_session"):
                await handler.end_session(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Session-end signal handler failed (non-fatal): %s", exc, exc_info=False)


if __name__ == "__main__":
    # Pass the app OBJECT, never the "server:app" import STRING.
    #
    # The image runs `python backend/server.py`, which binds this file to
    # `__main__`. Uvicorn resolves a string target with
    # `importlib.import_module("server")`, which finds this same file again on
    # sys.path and executes the ENTIRE module body a second time as a second
    # module object. `app`, `active_connections`, `active_connections_lock`,
    # `message_queue`, `voice_processor` and `log_handler` all existed twice,
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
