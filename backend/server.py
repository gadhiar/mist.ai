"""
Voice AI WebSocket Server

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

from voice_processor import VoiceProcessor

from config import DEFAULT_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
active_connections = []
message_queue = asyncio.Queue()
voice_processor = None
config = DEFAULT_CONFIG


async def broadcast_messages():
    """Background task to broadcast messages to all connected clients"""
    while True:
        message = await message_queue.get()

        # Send to all connected clients
        for websocket in active_connections[:]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                if websocket in active_connections:
                    active_connections.remove(websocket)

        message_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global voice_processor

    # Startup
    logger.info("=" * 60)
    logger.info("STARTING MIST.AI VOICE SERVER")
    logger.info("=" * 60)

    # Initialize voice processor
    voice_processor = VoiceProcessor(config, message_queue)
    await voice_processor.initialize()

    # Start message broadcaster
    broadcaster_task = asyncio.create_task(broadcast_messages())

    logger.info(f"Server ready on ws://{config.host}:{config.port}/ws")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Server shutting down...")
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
    """Health check"""
    return {"status": "online", "service": "Mist.AI Voice Server"}


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": voice_processor is not None,
        "active_connections": len(active_connections),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for voice conversation

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
    active_connections.append(websocket)

    logger.info(f"Client connected (total: {len(active_connections)})")

    # Send welcome message
    await websocket.send_json({"type": "status", "message": "Connected to Mist.AI Voice Server"})

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            # Handle different message types
            if data["type"] == "audio":
                # Complete audio from client (no VAD, just transcribe and process)
                audio_data = np.asarray(data["audio"], dtype=np.float32)
                sample_rate = data.get("sample_rate", 16000)

                logger.info(
                    f" Received complete audio: {len(audio_data)} samples @ {sample_rate}Hz"
                )

                # Process complete audio directly (transcribe -> LLM -> TTS)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, voice_processor.process_complete_audio, audio_data, sample_rate
                )

            elif data["type"] == "text":
                # Text message (manual input)
                user_text = data["text"]
                logger.info(f" Text message from client: '{user_text}'")

                # Send to message queue
                await message_queue.put({"type": "transcription", "text": user_text})

                # Process (will spawn thread internally)
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, voice_processor._process_conversation_turn, user_text
                )

            elif data["type"] == "interrupt":
                # Manual interrupt request
                logger.info("Manual interrupt requested")
                voice_processor.interrupt_flag.set()

                await websocket.send_json({"type": "status", "message": "Interrupt acknowledged"})

            elif data["type"] == "reset_vad":
                # Reset VAD state
                voice_processor.reset_vad()
                await websocket.send_json({"type": "status", "message": "VAD reset"})

            elif data["type"] == "config":
                # Update configuration (future feature)
                logger.info("Config update requested (not implemented)")
                await websocket.send_json(
                    {"type": "status", "message": "Config updates not yet supported"}
                )

            else:
                logger.warning(f"Unknown message type: {data.get('type')}")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected (remaining: {len(active_connections) - 1})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        log_level="info",
        reload=False,  # Set to True for development
    )
