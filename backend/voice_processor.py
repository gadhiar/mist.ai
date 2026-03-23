"""Voice Processor - Handles voice conversation logic."""

import asyncio
import logging
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import VAD
sys.path.insert(0, str(project_root / "dependencies" / "csm"))
from vad import AudioStreamProcessor

# Import from backend.voice_models explicitly
sys.path.insert(0, str(project_root / "backend"))
from voice_models.model_manager import ModelManager

logger = logging.getLogger(__name__)


def log_timestamp(msg: str):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    logger.info(f"[{timestamp}] {msg}")


class VoiceProcessor:
    """Handles voice conversation processing."""

    def __init__(self, config, message_queue):
        """Initialize voice processor.

        Args:
            config: VoiceConfig object
            message_queue: asyncio.Queue for sending messages to clients
        """
        self.config = config
        self.message_queue = message_queue
        self.models = None  # Will be initialized in initialize()

        # State
        self.is_speaking = False
        self.interrupt_flag = threading.Event()
        self.generation_lock = threading.Lock()
        self.latest_user_input = None
        self.input_lock = threading.Lock()

        # VAD
        self.vad_processor = None

        # Audio generation
        self.audio_queue = queue.Queue()

        # Event loop reference (for VAD callbacks in executor threads)
        self.loop = None

    async def initialize(self):
        """Initialize models and VAD."""
        log_timestamp("Initializing voice processor...")

        # Save event loop reference for VAD callbacks
        self.loop = asyncio.get_running_loop()

        # Initialize model manager with event loop
        self.models = ModelManager(self.config, event_loop=self.loop)

        # Load models in thread pool to not block event loop
        await self.loop.run_in_executor(None, self.models.load_all_models)

        # Initialize VAD
        log_timestamp("Loading VAD (Voice Activity Detection)...")
        vad_model, vad_utils = torch.hub.load(
            "snakers4/silero-vad", model="silero_vad", force_reload=False
        )

        self.vad_processor = AudioStreamProcessor(
            model=vad_model,
            utils=vad_utils,
            sample_rate=self.config.vad_sample_rate,
            vad_threshold=self.config.vad_threshold,
            callbacks={
                "on_speech_start": self._on_speech_start,
                "on_speech_end": self._on_speech_end,
            },
        )

        log_timestamp("Voice processor initialized")

    def _on_speech_start(self):
        """Called by VAD when user starts speaking."""
        # Send message to clients (using saved event loop reference)
        asyncio.run_coroutine_threadsafe(
            self.message_queue.put({"type": "vad_status", "status": "speech_started"}),
            self.loop,
        )

        # Check if we should interrupt
        if self.is_speaking:
            log_timestamp("User started speaking - interrupting AI")
            self.interrupt_flag.set()

    def _on_speech_end(self, audio_data, sample_rate):
        """Called by VAD when user stops speaking - SPAWN NEW THREAD."""
        log_timestamp("Speech ended, spawning processing thread...")

        # Process in separate thread (CSM pattern!)
        threading.Thread(
            target=self._process_user_speech,
            args=(audio_data, sample_rate),
            daemon=True,
        ).start()

    def _process_user_speech(self, audio_data, sample_rate):
        """Process user speech (runs in separate thread)."""
        try:
            # Transcribe
            log_timestamp(f"Transcribing audio ({len(audio_data)} samples @ {sample_rate}Hz)...")
            t_start = time.time()
            user_text = self.models.transcribe_audio(audio_data, sample_rate)
            t_elapsed = time.time() - t_start
            log_timestamp(f"Transcription complete ({t_elapsed:.2f}s)")

            # Send transcription to clients
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put({"type": "transcription", "text": user_text}),
                self.loop,
            )

            # If AI is speaking, queue it for later (CSM pattern: only keep latest!)
            if self.is_speaking:
                with self.input_lock:
                    self.latest_user_input = user_text
                    log_timestamp(f"AI currently speaking, input queued: '{user_text[:50]}...'")
                return

            # Otherwise, process immediately (don't set latest_user_input)
            self._process_conversation_turn(user_text)

        except Exception as e:
            logger.error(f"Error processing user speech: {e}", exc_info=True)
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(
                    {"type": "error", "message": f"Speech processing error: {e}"}
                ),
                self.loop,
            )

    def _process_conversation_turn(self, user_text):
        """Process one conversation turn (LLM + TTS)."""
        # Try to acquire generation lock (non-blocking like CSM!)
        if not self.generation_lock.acquire(blocking=False):
            log_timestamp("Generation already in progress, skipping")
            return

        try:
            log_timestamp(f"Starting conversation turn for: '{user_text}'")

            # Clear interrupt flag
            self.interrupt_flag.clear()
            self.is_speaking = True

            # Generate LLM response
            log_timestamp("LLM: Generating response...")
            llm_start = time.time()
            full_response = ""

            for token in self.models.generate_llm_response(user_text):
                full_response += token

                # Send token to clients
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put({"type": "llm_token", "token": token}),
                    self.loop,
                )

            llm_time = time.time() - llm_start

            # Trim to last complete sentence (CSM pattern)
            # This ensures audio never cuts off mid-sentence
            original_length = len(full_response)
            full_response = self.models.trim_to_last_sentence(full_response)
            trimmed_length = len(full_response)

            if trimmed_length < original_length:
                log_timestamp(
                    f"LLM complete ({llm_time:.2f}s, {original_length} chars, "
                    + f"trimmed to {trimmed_length} chars at sentence boundary)"
                )
            else:
                log_timestamp(f"LLM complete ({llm_time:.2f}s, {len(full_response)} chars)")

            # Send full response
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put({"type": "llm_response", "text": full_response}),
                self.loop,
            )

            # Generate TTS audio (if enabled)
            if self.config.tts_enabled:
                log_timestamp("TTS: Starting audio generation...")
                tts_start = time.time()
                chunk_count = 0
                first_chunk_time = None

                for audio_chunk in self.models.generate_tts_audio(full_response):
                    # Check for interruption
                    if self.interrupt_flag.is_set():
                        log_timestamp("TTS generation interrupted by user")
                        break

                    chunk_count += 1

                    if first_chunk_time is None:
                        first_chunk_time = time.time() - tts_start
                        log_timestamp(f"TTS: First chunk generated ({first_chunk_time:.2f}s)")

                    # Convert to numpy and send to clients
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                    else:
                        audio_np = audio_chunk.astype(np.float32)

                    asyncio.run_coroutine_threadsafe(
                        self.message_queue.put(
                            {
                                "type": "audio_chunk",
                                "audio": audio_np.tolist(),
                                "sample_rate": 24000,
                                "chunk_num": chunk_count,
                            }
                        ),
                        self.loop,
                    )

                tts_total = time.time() - tts_start
                log_timestamp(f"TTS generation complete ({tts_total:.2f}s, {chunk_count} chunks)")
            else:
                log_timestamp("TTS: Disabled (text-only mode)")

            # Send completion signal
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put({"type": "audio_complete"}),
                self.loop,
            )

        except Exception as e:
            logger.error(f"Error in conversation turn: {e}", exc_info=True)
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put({"type": "error", "message": f"Generation error: {e}"}),
                self.loop,
            )

        finally:
            self.is_speaking = False
            self.generation_lock.release()

            # Check if there's a pending input (CSM pattern!)
            with self.input_lock:
                if self.latest_user_input and not self.interrupt_flag.is_set():
                    # Process the latest input
                    pending_input = self.latest_user_input
                    self.latest_user_input = None
                    log_timestamp(f"Processing pending input: '{pending_input}'")
                    # Process in new thread to avoid blocking
                    threading.Thread(
                        target=self._process_conversation_turn,
                        args=(pending_input,),
                        daemon=True,
                    ).start()

    def process_complete_audio(self, audio_data, sample_rate):
        """Process complete audio from client (no VAD needed - Flutter controls recording)."""
        log_timestamp(f"Processing complete audio: {len(audio_data)} samples @ {sample_rate}Hz")

        # Transcribe and process immediately in a new thread
        threading.Thread(
            target=self._process_user_speech,
            args=(audio_data, sample_rate),
            daemon=True,
        ).start()

    def process_audio_chunk(self, audio_data, sample_rate):
        """Process incoming audio chunk from client (VAD mode - deprecated)."""
        if self.vad_processor and self.config.vad_enabled:
            # Resample if needed
            if sample_rate != self.config.vad_sample_rate:
                import scipy.signal

                num_samples = int(len(audio_data) * self.config.vad_sample_rate / sample_rate)
                audio_data = scipy.signal.resample(audio_data, num_samples)

            # Feed to VAD
            self.vad_processor.process_audio(audio_data)

    def reset_vad(self):
        """Reset VAD state."""
        if self.vad_processor:
            self.vad_processor.reset()
            log_timestamp("VAD reset")
