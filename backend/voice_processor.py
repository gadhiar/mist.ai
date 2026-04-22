"""Voice Processor - Handles voice conversation logic."""

import asyncio
import json
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
from audio_protocol import (
    MSG_AUDIO_CHUNK,
    MSG_AUDIO_COMPLETE,
    MSG_INTERRUPT_FADE,
    build_audio_frame,
    float32_to_pcm16,
    generate_fade_out,
    rms_normalize,
)
from request_context import new_request_id, spawn_with_context
from voice_models.model_manager import ModelManager

logger = logging.getLogger(__name__)


def log_timestamp(msg: str):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    logger.info(f"[{timestamp}] {msg}")


class VoiceProcessor:
    """Handles voice conversation processing."""

    def __init__(self, config, message_queue, vault_writer=None):
        """Initialize voice processor.

        Args:
            config: VoiceConfig object
            message_queue: asyncio.Queue for sending messages to clients
            vault_writer: Optional pre-started VaultWriter (Cluster 8 Phase 5).
                Threaded through ModelManager -> KnowledgeIntegration ->
                ConversationHandler so the vault layer shares a single
                writer across the voice path. Server lifespan owns the
                lifecycle.
        """
        self.config = config
        self.message_queue = message_queue
        self._vault_writer = vault_writer
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

        # Build LLM provider
        try:
            from backend.factories import build_llm_provider
            from backend.knowledge.config import KnowledgeConfig

            knowledge_config = KnowledgeConfig.from_env()
            self._llm_provider = build_llm_provider(knowledge_config)
        except Exception as e:
            logger.warning("LLM provider build failed, ModelManager will use None: %s", e)
            self._llm_provider = None

        # Initialize model manager with event loop
        self.models = ModelManager(
            self.config,
            event_loop=self.loop,
            llm_provider=self._llm_provider,
            vault_writer=self._vault_writer,
        )

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
            self.message_queue.put(json.dumps({"type": "vad_status", "status": "speech_started"})),
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
        spawn_with_context(self._process_user_speech, audio_data, sample_rate)

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
                self.message_queue.put(json.dumps({"type": "transcription", "text": user_text})),
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
                    json.dumps({"type": "error", "message": f"Speech processing error: {e}"})
                ),
                self.loop,
            )

    def _tts_consumer(self, sentence_queue: queue.Queue, tts_start_time: float) -> None:
        """Consume sentences from queue, generate TTS, send audio to client.

        Runs in a dedicated thread. Processes sentences as they arrive
        from the LLM producer, generating and sending audio for each.

        Args:
            sentence_queue: Queue of sentences to synthesize. None = stop signal.
            tts_start_time: Timestamp when TTS phase started (for logging).
        """
        chunk_count = 0
        chunk_seq = 0
        first_chunk_time = None
        first_sentence_time = None
        first_sentence = True

        # Minimum chars for quality TTS output. Short inputs (<40 chars)
        # cause Chatterbox to glitch on the first utterance because the
        # model lacks enough text context to match the reference voice.
        min_tts_chars = 40

        while True:
            try:
                sentence = sentence_queue.get(timeout=1.0)
            except queue.Empty:
                if self.interrupt_flag.is_set():
                    break
                continue
            if sentence is None:
                break
            if self.interrupt_flag.is_set():
                break

            # Coalesce short sentences with the next to avoid TTS glitches.
            # Peek at queue and merge until we have enough text or the queue
            # is empty / signals end. Skip coalescing for the first sentence
            # to minimize time-to-first-audio.
            if not first_sentence:
                while len(sentence) < min_tts_chars:
                    try:
                        next_item = sentence_queue.get(timeout=2.0)
                    except queue.Empty:
                        break
                    if next_item is None:
                        # End-of-stream -- generate what we have
                        break
                    sentence = sentence + " " + next_item
            first_sentence = False

            if first_sentence_time is None:
                first_sentence_time = time.time()

            log_timestamp(f"TTS: Generating sentence ({len(sentence)} chars)")

            for audio_chunk in self.models.generate_tts_audio(sentence):
                if self.interrupt_flag.is_set():
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                    else:
                        audio_np = audio_chunk.astype(np.float32)
                    fade_audio = generate_fade_out(audio_np, sample_rate=24000)
                    chunk_seq += 1
                    fade_frame = build_audio_frame(
                        MSG_INTERRUPT_FADE,
                        0,
                        chunk_seq,
                        24000,
                        float32_to_pcm16(rms_normalize(fade_audio)),
                    )
                    asyncio.run_coroutine_threadsafe(
                        self.message_queue.put(fade_frame),
                        self.loop,
                    )
                    break

                chunk_count += 1
                if first_chunk_time is None:
                    elapsed_from_sentence = time.time() - first_sentence_time
                    elapsed_from_turn = time.time() - tts_start_time
                    log_timestamp(
                        f"TTS: First audio chunk "
                        f"({elapsed_from_sentence:.2f}s from first sentence, "
                        f"{elapsed_from_turn:.2f}s from turn start)"
                    )
                    first_chunk_time = elapsed_from_turn

                if isinstance(audio_chunk, torch.Tensor):
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = audio_chunk.astype(np.float32)

                pcm16_bytes = float32_to_pcm16(rms_normalize(audio_np))
                chunk_seq += 1
                frame = build_audio_frame(
                    MSG_AUDIO_CHUNK,
                    0,
                    chunk_seq,
                    24000,
                    pcm16_bytes,
                )
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(frame),
                    self.loop,
                )

        # Send completion frame
        chunk_seq += 1
        complete_frame = build_audio_frame(
            MSG_AUDIO_COMPLETE,
            0,
            chunk_seq,
            24000,
            b"",
        )
        asyncio.run_coroutine_threadsafe(
            self.message_queue.put(complete_frame),
            self.loop,
        )

        tts_total = time.time() - tts_start_time
        log_timestamp(f"TTS consumer done ({tts_total:.2f}s, {chunk_count} chunks)")

    def _process_conversation_turn(self, user_text):
        """Process one conversation turn with LLM-TTS pipeline parallelism."""
        if not self.generation_lock.acquire(blocking=False):
            log_timestamp("Generation already in progress, skipping")
            return

        try:
            from backend.sentence_detector import SentenceBoundaryDetector

            new_request_id()
            log_timestamp(f"Starting conversation turn for: '{user_text}'")

            self.interrupt_flag.clear()
            self.is_speaking = True

            # === LLM + TTS Pipeline ===
            sentence_detector = SentenceBoundaryDetector()
            sentence_queue = queue.Queue()
            tts_start_time = time.time()

            # Start TTS consumer thread (reads sentences, generates audio)
            if self.config.tts_enabled:
                tts_thread = spawn_with_context(self._tts_consumer, sentence_queue, tts_start_time)

            # LLM producer: stream tokens, detect sentences, feed TTS
            log_timestamp("LLM: Generating response (streaming)...")
            llm_start = time.time()
            full_response = ""

            for token in self.models.generate_llm_response(user_text):
                if self.interrupt_flag.is_set():
                    log_timestamp("LLM generation interrupted")
                    break
                full_response += token

                # Send token to client for real-time text display
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(json.dumps({"type": "llm_token", "token": token})),
                    self.loop,
                )

                # Detect sentence boundaries and feed TTS
                if self.config.tts_enabled:
                    sentences = sentence_detector.feed(token)
                    for sentence in sentences:
                        sentence_queue.put(sentence)

            llm_time = time.time() - llm_start

            # Trim to last complete sentence
            full_response = self.models.trim_to_last_sentence(full_response)
            log_timestamp(f"LLM complete ({llm_time:.2f}s, {len(full_response)} chars)")

            # Send full response
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(json.dumps({"type": "llm_response", "text": full_response})),
                self.loop,
            )

            # Flush remaining text to TTS
            if self.config.tts_enabled:
                for sentence in sentence_detector.flush():
                    sentence_queue.put(sentence)
                sentence_queue.put(None)  # Signal end
                tts_thread.join(timeout=120)
                if tts_thread.is_alive():
                    logger.warning("TTS consumer thread did not finish within 120s")
            else:
                log_timestamp("TTS: Disabled (text-only mode)")

            # Send completion signal (text-only mode only; TTS consumer
            # sends the binary MSG_AUDIO_COMPLETE frame when TTS is enabled)
            if not self.config.tts_enabled:
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(json.dumps({"type": "audio_complete"})),
                    self.loop,
                )

        except Exception as e:
            logger.error("Error in conversation turn: %s", e, exc_info=True)
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(
                    json.dumps({"type": "error", "message": f"Generation error: {e}"})
                ),
                self.loop,
            )

        finally:
            self.is_speaking = False
            self.generation_lock.release()

            # Check for pending input
            with self.input_lock:
                if self.latest_user_input and not self.interrupt_flag.is_set():
                    pending_input = self.latest_user_input
                    self.latest_user_input = None
                    log_timestamp(f"Processing pending input: '{pending_input}'")
                    spawn_with_context(self._process_conversation_turn, pending_input)

    def process_complete_audio(self, audio_data, sample_rate):
        """Process complete audio from client (no VAD needed - Flutter controls recording)."""
        log_timestamp(f"Processing complete audio: {len(audio_data)} samples @ {sample_rate}Hz")

        # Transcribe and process immediately in a new thread
        spawn_with_context(self._process_user_speech, audio_data, sample_rate)

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
