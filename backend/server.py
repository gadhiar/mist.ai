"""Voice AI WebSocket Server.

Based on CSM demo architecture - production-ready for web frontend
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

# Import config BEFORE voice_processor to avoid CSM config conflict
from config import DEFAULT_CONFIG  # isort:skip
from voice_processor import VoiceProcessor  # isort:skip
from factories import (  # isort:skip
    build_curation_scheduler,
    build_filewatcher,
    build_sidecar_index,
    build_vault_writer,
)
from knowledge.config import KnowledgeConfig  # isort:skip
from log_handler import WebSocketLogHandler  # isort:skip

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

# Global state
active_connections: set[WebSocket] = set()
active_connections_lock = asyncio.Lock()
message_queue: asyncio.Queue[str | bytes] = asyncio.Queue()
voice_processor: VoiceProcessor | None = None
curation_scheduler = None
log_handler: WebSocketLogHandler | None = None
config = DEFAULT_CONFIG

# Cluster 8 Phase 5: vault layer subsystems (initialized in lifespan)
vault_writer = None
vault_sidecar = None
vault_filewatcher = None


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global voice_processor, curation_scheduler, log_handler
    global vault_writer, vault_sidecar, vault_filewatcher

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
        vault_writer = build_vault_writer(knowledge_config)
        if vault_writer is not None:
            await vault_writer.start()
            logger.info("Vault writer started at %s", knowledge_config.vault.root)

        vault_sidecar = build_sidecar_index(knowledge_config)
        if vault_sidecar is not None:
            logger.info(
                "Vault sidecar index initialized at %s",
                knowledge_config.sidecar_index.db_path,
            )

        vault_filewatcher = build_filewatcher(knowledge_config, vault_sidecar)
        if vault_filewatcher is not None:
            vault_filewatcher.start(loop)
            logger.info(
                "Vault filewatcher started (observer=%s, debounce=%dms)",
                knowledge_config.filewatcher.observer_type,
                knowledge_config.filewatcher.debounce_ms,
            )
    except Exception as e:
        logger.warning("Vault layer initialization failed (continuing without vault): %s", e)

    # Initialize voice processor with the server-owned vault_writer so that
    # the voice-path ConversationHandler shares a single started writer.
    voice_processor = VoiceProcessor(config, message_queue, vault_writer=vault_writer)
    await voice_processor.initialize()

    # Start message broadcaster
    broadcaster_task = asyncio.create_task(broadcast_messages())

    # Attach WebSocket log handler to root logger
    log_handler = WebSocketLogHandler(event_loop=loop, message_queue=message_queue)
    logging.getLogger().addHandler(log_handler)
    logger.info("WebSocket log handler attached")

    # Start curation scheduler for periodic graph maintenance
    try:
        curation_scheduler = build_curation_scheduler(knowledge_config)
        await curation_scheduler.start()
        logger.info("Curation scheduler started")
    except Exception as e:
        logger.warning("Curation scheduler failed to start: %s", e)
        curation_scheduler = None

    logger.info(f"Server ready on ws://{config.host}:{config.port}/ws")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Server shutting down...")
    if curation_scheduler is not None:
        await curation_scheduler.stop()

    # Cluster 8 Phase 5: vault layer shutdown
    if vault_filewatcher is not None:
        try:
            vault_filewatcher.stop()
        except Exception as e:
            logger.warning("Vault filewatcher stop error: %s", e)
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for voice conversation.

    Message types:
    - From client:
        {"type": "audio", "audio": [...], "sample_rate": 16000}
        {"type": "text", "text": "user message"}
        {"type": "interrupt"}
        {"type": "reset_vad"}

    - To client:
        {"type": "vad_status", "status": "speech_started"}
        {"type": "transcription", "text": "..."}
        {"type": "llm_token", "token": "..."}
        {"type": "llm_response", "text": "full response"}
        {"type": "audio_chunk", "audio": [...], "sample_rate": 24000, "chunk_num": 1}
        {"type": "audio_complete"}
        {"type": "error", "message": "..."}
    """
    await websocket.accept(headers=None)
    async with active_connections_lock:
        active_connections.add(websocket)

    logger.info(f"Client connected (total: {len(active_connections)})")

    # Ensure voice processor is initialized
    if voice_processor is None:
        await websocket.send_json({"type": "error", "message": "Server not ready"})
        await websocket.close(code=1013)
        async with active_connections_lock:
            active_connections.discard(websocket)
        return

    # Send welcome message
    await websocket.send_json({"type": "status", "message": "Connected to Mist.AI Voice Server"})

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            msg_type = data.get("type")
            if msg_type is None:
                await websocket.send_json({"type": "error", "message": "Missing 'type' field"})
                continue

            # Handle different message types
            if msg_type == "audio":
                # Complete audio from client (no VAD, just transcribe and process)
                audio_payload = data.get("audio")
                if audio_payload is None:
                    await websocket.send_json({"type": "error", "message": "Missing 'audio' field"})
                    continue
                audio_data = np.asarray(audio_payload, dtype=np.float32)
                sample_rate = data.get("sample_rate", 16000)

                logger.info(f"Received complete audio: {len(audio_data)} samples @ {sample_rate}Hz")

                # Process complete audio directly (transcribe -> LLM -> TTS)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, voice_processor.process_complete_audio, audio_data, sample_rate
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
                    None, voice_processor._process_conversation_turn, user_text
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

            elif msg_type == "config":
                # Update configuration (future feature)
                logger.info("Config update requested (not implemented)")
                await websocket.send_json(
                    {"type": "status", "message": "Config updates not yet supported"}
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


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        log_level="info",
        reload=False,  # Set to True for development
    )
