"""Voice Processor - Handles voice conversation logic."""

import asyncio
import functools
import json
import logging
import queue
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import VAD. `isort:skip` pins this to the sys.path insert directly above it;
# without the marker it would be sorted in with the backend block below and
# would then run before the path that makes it resolvable.
sys.path.insert(0, str(project_root / "dependencies" / "csm"))
from vad import AudioStreamProcessor  # isort:skip

# Package-qualified on purpose, all three of them. `backend/` is a PEP 420
# namespace package that is ALSO on sys.path, so a bare `request_context` and
# `backend.request_context` resolve to two DISTINCT module objects built from
# one file, holding two distinct sets of module-level state -- a write through
# one name is invisible through the other. request_context holds the session-id
# ContextVars, so the bare spelling made session propagation a silent no-op;
# `audio_protocol` and `voice_models.model_manager` were in fact being loaded
# twice this way until the imports were unified on 2026-08-03. Every importer
# must use the `backend.`-qualified name -- see also log_handler.py and
# chat/knowledge_integration.py.
from backend.audio_protocol import (
    MSG_AUDIO_CHUNK,
    MSG_AUDIO_COMPLETE,
    MSG_INTERRUPT_FADE,
    build_audio_frame,
    float32_to_pcm16,
    generate_fade_out,
    rms_normalize,
)
from backend.request_context import (
    current_request_id,
    current_session_id,
    current_turn_complete,
    current_turn_error,
    new_request_id,
    spawn_with_context,
)
from backend.voice_models.model_manager import ModelManager

logger = logging.getLogger(__name__)


def log_timestamp(msg: str):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    logger.info(f"[{timestamp}] {msg}")


class VoiceProcessor:
    """Handles voice conversation processing."""

    def __init__(
        self, config, message_queue, vault_writer=None, vault_sidecar=None, invalidation_bus=None
    ):
        """Initialize voice processor.

        Args:
            config: VoiceConfig object
            message_queue: asyncio.Queue for sending messages to clients
            vault_writer: Optional pre-started VaultWriter (Cluster 8 Phase 5).
                Threaded through ModelManager -> KnowledgeIntegration ->
                ConversationHandler so the vault layer shares a single
                writer across the voice path. Server lifespan owns the
                lifecycle.
            vault_sidecar: Optional initialized VaultSidecarIndex (Cluster 8
                Phase 9). Threaded through ModelManager -> KnowledgeIntegration
                -> ConversationHandler so the retriever's historical and
                three-way hybrid RRF paths route to the sidecar. Server
                lifespan owns the lifecycle.
            invalidation_bus: Optional InvalidationBus shared with the filewatcher
                (Phase 5.5). Threaded through ModelManager -> KnowledgeIntegration
                -> ConversationHandler so _on_vault_rebuild is subscribed for
                ADR-010 read-path cache invalidation. Must be the same
                instance returned by build_phase3_components. Server lifespan
                owns the lifecycle. None preserves pre-Phase-5.5 behavior.
        """
        self.config = config
        self.message_queue = message_queue
        self._vault_writer = vault_writer
        self._vault_sidecar = vault_sidecar
        self._invalidation_bus = invalidation_bus
        self.models = None  # Will be initialized in initialize()

        # State
        self.is_speaking = False
        self.interrupt_flag = threading.Event()
        # `generation_lock` protects LLM token generation only. Released as
        # soon as LLM completes so the next turn's LLM can start while this
        # turn's TTS is still rendering audio (gap #5 / 2026-05-06 unification).
        self.generation_lock = threading.Lock()
        # `tts_render_lock` serializes TTS rendering across turns so two
        # concurrent _tts_consumer threads do not interleave audio frames on
        # the WebSocket. Acquired at the start of _tts_consumer's loop;
        # held for the entire turn's TTS duration; released when the turn's
        # sentence queue drains. Independent of generation_lock -- LLM for
        # turn N+1 can run while TTS for turn N still holds this lock.
        self.tts_render_lock = threading.Lock()
        # Latest-wins pending input as `(user_text, session_id)`. The session
        # travels WITH the text because the drain in `_process_conversation_turn`
        # respawns the queued turn from the FINISHING turn's thread, and
        # `spawn_with_context` snapshots that thread's context -- so an
        # id-less slot silently hands the queued turn the previous session's
        # `current_session_id`. That id selects the conversation history, the
        # EventStore session, the vault note and the graph provenance, so the
        # crossing misfiles one connection's utterance into another's memory.
        # A single slot, not a map: latest-wins overwrites in place, so this
        # cannot accumulate per-connection entries.
        self.latest_user_input: tuple[str, str | None] | None = None
        self.input_lock = threading.Lock()
        # Monotonic turn counter (incremented under generation_lock). A
        # previous turn's still-draining TTS consumer compares its captured
        # epoch against this to skip a LATE state_cycle 'idle' that would
        # clobber the next turn's think/speak state on the FE.
        self._turn_epoch = 0

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

        # Build LLM provider. Pass the debug logger so the provider gets
        # wrapped with InstrumentedStreamingLLMProvider when MIST_DEBUG_LLM_JSONL=1.
        # Without this, the WebSocket text path's extraction calls bypass
        # llm_call instrumentation (the conversation_handler factory's own
        # build_llm_provider call short-circuits because llm_provider is
        # passed in pre-built from here).
        try:
            from backend.debug_jsonl_logger import DebugJSONLLogger
            from backend.factories import build_llm_provider
            from backend.knowledge.config import KnowledgeConfig

            knowledge_config = KnowledgeConfig.from_env()
            debug_logger = DebugJSONLLogger.from_env()
            self._llm_provider = build_llm_provider(knowledge_config, debug_logger=debug_logger)
        except Exception as e:
            logger.warning("LLM provider build failed, ModelManager will use None: %s", e)
            self._llm_provider = None

        # Initialize model manager with event loop
        self.models = ModelManager(
            self.config,
            event_loop=self.loop,
            llm_provider=self._llm_provider,
            vault_writer=self._vault_writer,
            vault_sidecar=self._vault_sidecar,
            invalidation_bus=self._invalidation_bus,
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

    def _emit_state_cycle(self, state: str) -> None:
        """Enqueue a ``state_cycle`` WebSocket event per ADR-017.

        Called at the explicit transition points of the voice/text turn
        pipeline (listen / think / speak / idle). Sleep tiers are FE-local
        per ADR-016 and never emitted from BE. Caller is on a worker
        thread; emission goes through the same threadsafe queue path as
        every other JSON event.

        Args:
            state: One of 'idle', 'listen', 'think', 'speak'.
        """
        asyncio.run_coroutine_threadsafe(
            self.message_queue.put(json.dumps({"type": "state_cycle", "state": state})),
            self.loop,
        )

    def _on_speech_start(self):
        """Called by VAD when user starts speaking."""
        # Send message to clients (using saved event loop reference)
        asyncio.run_coroutine_threadsafe(
            self.message_queue.put(json.dumps({"type": "vad_status", "status": "speech_started"})),
            self.loop,
        )
        # Voice path: user has begun speaking -> MIST is now listening.
        self._emit_state_cycle("listen")

        # Check if we should interrupt
        if self.is_speaking:
            log_timestamp("User started speaking - interrupting AI")
            self.interrupt_flag.set()

    def _on_speech_end(self, audio_data, sample_rate):
        """Called by VAD when user stops speaking - SPAWN NEW THREAD."""
        log_timestamp("Speech ended, spawning processing thread...")

        # Extend ADR-017 vad_status with speech_ended (paired with the
        # earlier speech_started emit in _on_speech_start). FE uses the
        # pair to bracket the user's utterance; state_cycle('think')
        # fires later at the start of _process_conversation_turn.
        asyncio.run_coroutine_threadsafe(
            self.message_queue.put(json.dumps({"type": "vad_status", "status": "speech_ended"})),
            self.loop,
        )

        # Process in separate thread (CSM pattern!)
        spawn_with_context(self._process_user_speech, audio_data, sample_rate)

    def _process_user_speech(self, audio_data, sample_rate):
        """Process user speech (runs in separate thread)."""
        try:
            # Rotate the request id at VOICE INGRESS so the STT phase (Whisper
            # logs + the transcription emit) groups under THIS turn's id, not
            # the previous turn's contextvars snapshot
            # (deep review febe-observability-10).
            request_id = new_request_id()

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
                    # Capture the session from THIS thread's context: the
                    # drain that later replays this text runs on a different
                    # thread carrying a different session.
                    self.latest_user_input = (user_text, current_session_id.get())
                    log_timestamp(f"AI currently speaking, input queued: '{user_text[:50]}...'")
                return

            # Otherwise, process immediately (don't set latest_user_input)
            self._process_conversation_turn(user_text, request_id=request_id)

        except Exception as e:
            logger.error(f"Error processing user speech: {e}", exc_info=True)
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(
                    json.dumps(
                        {
                            "type": "error",
                            "kind": "model",
                            "message": f"Speech processing error: {e}",
                            "retriable": True,
                            "context": None,
                        }
                    )
                ),
                self.loop,
            )

    def _tts_consumer(
        self,
        sentence_queue: queue.Queue,
        tts_start_time: float,
        turn_epoch: int | None = None,
    ) -> None:
        """Consume sentences from queue, generate TTS, send audio to client.

        Runs in a dedicated thread. Processes sentences as they arrive
        from the LLM producer, generating and sending audio for each.

        TTS rendering is serialized across turns via `self.tts_render_lock`
        (acquired here, released when this turn's queue drains). LLM for
        the next turn may already be running by the time we acquire -- in
        that case we wait for the previous turn's TTS to finish before
        emitting audio for this turn. Without this serialization, two
        concurrent _tts_consumer threads would interleave audio frames on
        the WebSocket and produce garbled playback.

        Args:
            sentence_queue: Queue of sentences to synthesize. None = stop signal.
            tts_start_time: Timestamp when TTS phase started (for logging).
            turn_epoch: This turn's epoch; the final idle emit is skipped when
                a newer turn has started (None = always emit, legacy callers).
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

        with self.tts_render_lock:
            self._tts_consumer_loop(
                sentence_queue,
                tts_start_time,
                chunk_count,
                chunk_seq,
                first_chunk_time,
                first_sentence_time,
                first_sentence,
                min_tts_chars,
                turn_epoch,
            )

    def _tts_consumer_loop(
        self,
        sentence_queue: queue.Queue,
        tts_start_time: float,
        chunk_count: int,
        chunk_seq: int,
        first_chunk_time,
        first_sentence_time,
        first_sentence: bool,
        min_tts_chars: int,
        turn_epoch: int | None = None,
    ) -> None:
        """Inner loop body of _tts_consumer. Extracted so the outer method can
        wrap the entire loop in `tts_render_lock` without restructuring the
        existing logic.
        """
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
                        # End-of-stream -- generate what we have. Put the
                        # sentinel back on the queue so the OUTER loop also
                        # sees it on its next iteration and breaks cleanly
                        # (otherwise the consumer hangs on get(timeout=1.0)
                        # forever, state_cycle('idle') never emits, and
                        # the FE stays stuck in SPEAK -- the canonical
                        # tool-using-turn hang surfaced by the 2026-05-25
                        # smoke walk).
                        sentence_queue.put(None)
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
        # TTS-on mode: turn complete after final audio frame, MIST returns to
        # idle -- UNLESS a newer turn already started (early generation_lock
        # release lets turn N+1 emit think/speak while this consumer drains;
        # a late idle would clobber the live turn's FE state, since
        # state_cycle carries no turn_id). MSG_AUDIO_COMPLETE above stays
        # unconditional: the FE audio pipeline needs the terminal frame.
        if turn_epoch is None or turn_epoch == self._turn_epoch:
            self._emit_state_cycle("idle")
        else:
            log_timestamp("TTS consumer: newer turn in flight; skipping stale idle emit")

        tts_total = time.time() - tts_start_time
        log_timestamp(f"TTS consumer done ({tts_total:.2f}s, {chunk_count} chunks)")

    def _process_conversation_turn(
        self, user_text, request_id: str | None = None, session_id: str | None = None
    ):
        """Process one conversation turn with LLM-TTS pipeline parallelism.

        Args:
            user_text: The user's utterance for this turn.
            request_id: Adopt this id (voice path mints it at speech ingress
                so STT logs and the turn share one id); None mints fresh
                (text path, pending-input respawns).
            session_id: The originating connection's session id, published to
                `current_session_id` for the rest of this call chain. Set here
                rather than by the caller because `loop.run_in_executor` does
                not copy the caller's context into the worker thread. None
                leaves the ambient value untouched, which is what the
                pending-input respawn needs -- `spawn_with_context` already
                copied the in-flight turn's session into that thread.
        """
        if not self.generation_lock.acquire(blocking=False):
            # Latest-wins queueing (mirrors the voice path): the in-flight
            # turn's finally-block drain picks this up. Silently dropping the
            # input left the FE turn promise waiting forever
            # (deep review febe-observability-11).
            with self.input_lock:
                # `session_id` is the caller's argument rather than the context
                # var: this early return happens BEFORE the publish below, so
                # the ambient value here is still whatever the executor thread
                # was last used for. Fall back to the context var for callers
                # that legitimately pass None (the voice path and the respawn,
                # both of which arrive with their session already published).
                self.latest_user_input = (user_text, session_id or current_session_id.get())
            log_timestamp("Generation already in progress; queued latest input")
            return

        # Hoisted above try: the finally path must reach the queue (and know
        # whether a consumer exists) even when the failure happens before the
        # LLM loop (deep review febe-observability-2).
        sentence_queue = queue.Queue()
        tts_consumer_spawned = False
        self._turn_epoch += 1  # under generation_lock
        turn_epoch = self._turn_epoch

        try:
            from backend.sentence_detector import SentenceBoundaryDetector

            if request_id is None:
                new_request_id()
            else:
                current_request_id.set(request_id)
            if session_id is not None:
                current_session_id.set(session_id)
            log_timestamp(f"Starting conversation turn for: '{user_text}'")

            self.interrupt_flag.clear()
            self.is_speaking = True
            # Both voice (post-transcription) and text (composer submit) paths
            # enter here -- MIST has the user message and is preparing reply.
            self._emit_state_cycle("think")

            # Open a streaming turn per ADR-017. turn_id correlates stream_token
            # events through stream_complete / stream_cancelled, and lets FE
            # match tool_call_started/completed events to the right turn.
            turn_id = str(uuid.uuid4())
            channel = "tts" if self.config.tts_enabled else "text"
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(
                    json.dumps({"type": "stream_start", "turn_id": turn_id, "channel": channel})
                ),
                self.loop,
            )

            # === LLM + TTS Pipeline ===
            sentence_detector = SentenceBoundaryDetector()
            tts_start_time = time.time()

            # Start TTS consumer thread (reads sentences, generates audio).
            # Gap #5 (2026-05-06): we don't keep the thread reference. The
            # thread runs in background, acquires `tts_render_lock` for the
            # entire turn's TTS duration to serialize audio frames across
            # turns, and exits naturally when the sentence_queue drains.
            # generation_lock release no longer waits for TTS completion.
            if self.config.tts_enabled:
                spawn_with_context(self._tts_consumer, sentence_queue, tts_start_time, turn_epoch)
                tts_consumer_spawned = True

            # LLM producer: stream tokens, detect sentences, feed TTS
            log_timestamp("LLM: Generating response (streaming)...")
            llm_start = time.time()
            full_response = ""

            first_token_seen = False
            interrupted = False
            for item in self.models.generate_llm_response(user_text):
                if self.interrupt_flag.is_set():
                    log_timestamp("LLM generation interrupted")
                    interrupted = True
                    break

                # ADR-017 Wave 2: dict items are pre-formed FE-bound event
                # payloads (tool_call_started/completed, cards_summon/dismiss,
                # graph_subgraph) yielded from the conversation pipeline. They
                # are forwarded onto the message_queue as-is.
                if isinstance(item, dict):
                    asyncio.run_coroutine_threadsafe(
                        self.message_queue.put(json.dumps(item)),
                        self.loop,
                    )
                    continue

                # String items are response tokens.
                token = item
                full_response += token
                if not first_token_seen:
                    # First token of the turn -> MIST has begun replying.
                    self._emit_state_cycle("speak")
                    first_token_seen = True

                # Send token to client for real-time text display per ADR-017.
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(
                        json.dumps({"type": "stream_token", "turn_id": turn_id, "token": token})
                    ),
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

            # Emit one of error / stream_cancelled / stream_complete per
            # ADR-017. Priority: error (set on the bridge side-channel by
            # KnowledgeIntegration on bridge timeout or streaming exception)
            # > cancellation (user interrupt) > clean completion.
            last_error = current_turn_error.get()
            if last_error is not None:
                error_kind, error_message = last_error
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(
                        json.dumps(
                            {
                                "type": "error",
                                "kind": error_kind,
                                "message": error_message,
                                "retriable": True,
                                "context": {"turn_id": turn_id},
                            }
                        )
                    ),
                    self.loop,
                )
            elif interrupted:
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(
                        json.dumps(
                            {
                                "type": "stream_cancelled",
                                "turn_id": turn_id,
                                "partial_text": full_response,
                            }
                        )
                    ),
                    self.loop,
                )
            else:
                # tool_calls_used / duration_ms come from the per-turn
                # ContextVar the producer sets. None when knowledge is disabled
                # (the producer never runs) or the stream ended before Complete
                # was seen; default to 0 in that case.
                last_complete = current_turn_complete.get()
                tool_calls_used = last_complete.tool_calls_used if last_complete else 0
                asyncio.run_coroutine_threadsafe(
                    self.message_queue.put(
                        json.dumps(
                            {
                                "type": "stream_complete",
                                "turn_id": turn_id,
                                "full_text": full_response,
                                "duration_ms": int(llm_time * 1000),
                                "tool_calls_used": tool_calls_used,
                            }
                        )
                    ),
                    self.loop,
                )

            # Flush remaining text to TTS
            if self.config.tts_enabled:
                for sentence in sentence_detector.flush():
                    sentence_queue.put(sentence)
                sentence_queue.put(None)  # Signal end
                # Gap #5 (2026-05-06): do NOT join the TTS thread here.
                # generation_lock release (in `finally`) must happen as soon
                # as LLM generation completes so the next turn's LLM can run
                # while this turn's audio is still rendering. The TTS thread
                # continues in the background and exits naturally when its
                # sentence_queue drains. Audio ordering across turns is
                # preserved by `tts_render_lock` (held by _tts_consumer for
                # the entire turn's TTS duration).
            else:
                log_timestamp("TTS: Disabled (text-only mode)")

            # Text-only mode: stream_complete already signaled end-of-turn per
            # ADR-017 (the JSON audio_complete variant is subsumed). The binary
            # MSG_AUDIO_COMPLETE frame is still emitted by _tts_consumer when
            # TTS is enabled.
            if not self.config.tts_enabled:
                # Text-only mode: turn complete, MIST returns to idle.
                self._emit_state_cycle("idle")

        except Exception as e:
            logger.error("Error in conversation turn: %s", e, exc_info=True)
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(
                    json.dumps(
                        {
                            "type": "error",
                            "kind": "model",
                            "message": f"Generation error: {e}",
                            "retriable": True,
                            "context": None,
                        }
                    )
                ),
                self.loop,
            )
            # Error aborts the turn; signal idle so FE animation returns to
            # rest -- unless a newer turn already started (stale epoch).
            if turn_epoch == self._turn_epoch:
                self._emit_state_cycle("idle")

        finally:
            if tts_consumer_spawned:
                # Guarantee the consumer's stop sentinel on EVERY exit path:
                # without it an LLM exception leaves the consumer spinning on
                # an empty queue holding tts_render_lock forever -- voice
                # output dead until restart (febe-observability-2). A second
                # None on the clean path is harmless (consumer breaks on the
                # first).
                sentence_queue.put(None)
            self.is_speaking = False
            self.generation_lock.release()

            # Check for pending input
            with self.input_lock:
                if self.latest_user_input and not self.interrupt_flag.is_set():
                    pending_input, pending_session_id = self.latest_user_input
                    self.latest_user_input = None
                    log_timestamp(f"Processing pending input: '{pending_input}'")
                    # `session_id` is passed EXPLICITLY, not left to context
                    # inheritance: `spawn_with_context` copies THIS thread's
                    # context, whose `current_session_id` is the finishing
                    # turn's session, not the queued turn's. `functools.partial`
                    # because `spawn_with_context`'s **kwargs go to
                    # `threading.Thread`, not to the target.
                    spawn_with_context(
                        functools.partial(
                            self._process_conversation_turn,
                            pending_input,
                            session_id=pending_session_id,
                        )
                    )

    def process_complete_audio(self, audio_data, sample_rate, session_id: str | None = None):
        """Process complete audio from client (no VAD needed; the client controls recording window).

        Args:
            audio_data: Complete utterance samples from the client.
            sample_rate: Sample rate of `audio_data`.
            session_id: The originating connection's session id. Published to
                `current_session_id` here, on the executor side, because
                `loop.run_in_executor` does not copy the caller's context.
                The set must precede `spawn_with_context`, which snapshots
                the current context for the transcription thread.
        """
        log_timestamp(f"Processing complete audio: {len(audio_data)} samples @ {sample_rate}Hz")

        if session_id is not None:
            current_session_id.set(session_id)

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
