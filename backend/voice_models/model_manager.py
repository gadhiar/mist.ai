"""Model Manager - Handles loading and lifecycle of all ML models."""

import logging
import queue
import sys
import threading
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "dependencies/csm")

import ollama
import torch
from generator import Segment

from src.multimodal.stt import WhisperSTT
from src.multimodal.text_preprocessing import preprocess_text_for_tts
from src.multimodal.tts import SesameTTS
from src.multimodal.voice_profile import VoiceProfileRegistry

logger = logging.getLogger(__name__)

# Knowledge graph integration
try:
    from backend.chat.knowledge_integration import KnowledgeIntegration
    from backend.knowledge.config import KnowledgeConfig as _KnowledgeConfig

    KNOWLEDGE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Knowledge integration not available: {e}")
    KNOWLEDGE_AVAILABLE = False


class ModelManager:
    """Manages all ML models for the voice AI system."""

    def __init__(self, config, event_loop=None):
        """Initialize model manager.

        Args:
            config: VoiceConfig object
            event_loop: Optional event loop for async operations
        """
        self.config = config
        self.stt = None
        self.tts = None
        self.llm_model = config.llm_model
        self.event_loop = event_loop

        # Knowledge graph integration
        self.knowledge = None
        if KNOWLEDGE_AVAILABLE:
            try:
                knowledge_config = _KnowledgeConfig.from_env()
                if knowledge_config.enable_knowledge_integration:
                    self.knowledge = KnowledgeIntegration(config=knowledge_config)
                    if self.knowledge.is_enabled():
                        logger.info("Knowledge graph integration ENABLED")
                    else:
                        logger.warning("Knowledge integration disabled (Neo4j unavailable)")
                        self.knowledge = None
                else:
                    logger.info("Knowledge graph integration disabled in config")
            except Exception as e:
                logger.warning(f"Knowledge integration disabled: {e}")
                self.knowledge = None

        # TTS model worker thread (CSM pattern)
        self.tts_request_queue = queue.Queue()
        self.tts_result_queue = queue.Queue()
        self.tts_worker_thread = None
        self.tts_worker_running = threading.Event()
        self.tts_generation_id = 0  # Track which generation is active
        self.tts_lock = threading.Lock()

    def load_all_models(self):
        """Load all models (STT, TTS, LLM)."""
        logger.info("Loading all models...")
        start = time.time()

        # Load STT
        logger.info("1/3 Loading Whisper STT...")
        self.stt = WhisperSTT(model_size=self.config.whisper_model)

        # Start TTS worker thread (CSM pattern) - only if TTS is enabled
        if self.config.tts_enabled:
            logger.info("2/3 Starting TTS model worker thread...")
            self._start_tts_worker()

            # Wait for TTS warmup to complete
            logger.info("   Waiting for TTS warmup...")
            self.wait_for_tts_warmup()
        else:
            logger.info("2/3 TTS disabled - skipping TTS model loading")

        # Warmup LLM
        logger.info("3/3 Warming up LLM...")
        ollama.chat(model=self.llm_model, messages=[{"role": "user", "content": "hi"}])

        elapsed = time.time() - start
        logger.info(f"All models loaded in {elapsed:.1f}s")

    def _start_tts_worker(self):
        """Start dedicated TTS model worker thread (CSM pattern)."""
        if self.tts_worker_thread is not None and self.tts_worker_thread.is_alive():
            return

        self.tts_worker_running.set()
        self.tts_worker_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="tts_worker"
        )
        self.tts_worker_thread.start()
        logger.info("Started dedicated TTS model worker thread")

    def _tts_worker(self):
        """TTS model worker thread - loads and runs model (CSM pattern)."""
        logger.info("TTS worker thread started")

        # Disable CUDA graphs to avoid threading issues (CSM pattern)
        torch._inductor.config.triton.cudagraphs = False
        torch._inductor.config.fx_graph_cache = False

        logger.info("Loading Sesame TTS in worker thread...")
        project_root = Path(__file__).parent.parent.parent
        registry = VoiceProfileRegistry(project_root / "data" / "voice_profiles")
        profile = registry.get_active()
        logger.info(f"Active voice profile: {profile.name} ({profile.description})")
        self.tts = SesameTTS(
            profile=profile,
            device=self.config.tts_device,
            use_context=self.config.use_voice_context,
        )

        # Warmup TTS -- exercise BOTH code paths to trigger torch.compile
        # speak() uses generate(), but real requests use generate_stream()
        logger.info("Warming up TTS (non-streaming path)...")
        dummy_audio = self.tts.speak("Initialization complete.", play=False)
        logger.info(f"TTS non-streaming warmed up (audio shape: {dummy_audio.shape})")

        logger.info("Warming up TTS (streaming path)...")
        stream_chunks = []
        for chunk in self.tts.generator.generate_stream(
            text="System ready.",
            speaker=self.tts.speaker_id,
            context=self.tts.context,
            max_audio_length_ms=5000,
        ):
            stream_chunks.append(chunk)
        logger.info(f"TTS streaming warmed up ({len(stream_chunks)} chunks)")

        # Reset context to reference clips only -- discard warmup dummy audio
        # Warmup generates low-quality audio that degrades voice anchoring
        ref_count = len(self.tts.profile.reference_clips)
        self.tts.context = self.tts.context[:ref_count]
        logger.info(
            f"Context reset to {len(self.tts.context)} reference clips (warmup audio discarded)"
        )

        # Signal warmup complete
        self.tts_result_queue.put(("warmup_complete", None))

        # Process requests
        while self.tts_worker_running.is_set():
            try:
                request = self.tts_request_queue.get(timeout=0.1)
                if request is None:
                    break

                gen_id, text, speaker_id, context, max_ms, temperature, topk = request
                logger.info(f"TTS worker: Processing generation ID {gen_id}")
                logger.info(
                    f"TTS worker: Context has {len(context)} segments, max_audio={max_ms}ms"
                )

                # Synchronize CUDA before TTS to prevent conflicts with Whisper
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Collect all chunks for context update
                audio_chunks = []
                chunk_count = 0

                try:
                    logger.info(
                        f"TTS worker: Starting generate_stream "
                        f"(text={len(text)} chars, max_audio={max_ms}ms, "
                        f"temp={temperature}, topk={topk})"
                    )
                    for chunk in self.tts.generator.generate_stream(
                        text=text,
                        speaker=speaker_id,
                        context=context,
                        max_audio_length_ms=max_ms,
                        temperature=temperature,
                        topk=topk,
                    ):
                        audio_chunks.append(chunk)
                        chunk_count += 1
                        self.tts_result_queue.put(("chunk", gen_id, chunk))

                        if not self.tts_worker_running.is_set():
                            break

                    logger.info(f"TTS worker: Generated {chunk_count} audio chunks")

                except Exception as gen_error:
                    logger.error(f"TTS generation error: {gen_error}")
                    import traceback

                    logger.error(traceback.format_exc())
                    raise

                # Synchronize CUDA after TTS to ensure completion
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Update context with generated audio (CRITICAL for consistency)
                if len(audio_chunks) > 0 and self.tts and self.tts.use_context:
                    # Concatenate all chunks with crossfade for smooth boundaries
                    complete_audio = self._crossfade_chunks(audio_chunks)

                    # Move to CPU to prevent CUDA memory fragmentation
                    # Context audio should be on CPU for stable long-term storage
                    complete_audio = complete_audio.cpu()

                    # Create segment and update context
                    segment = Segment(speaker=speaker_id, text=text, audio=complete_audio)
                    self.tts.context.append(segment)

                    # Keep reference clips + only 1 recent utterance
                    # 4 refs (~26s) + 1 recent (~10-15s) = ~40s context max
                    ref_count = len(self.tts.profile.reference_clips)
                    max_segments = ref_count + 1
                    if len(self.tts.context) > max_segments:
                        self.tts.context = self.tts.context[:ref_count] + self.tts.context[-1:]

                    logger.info(f"Updated context: now {len(self.tts.context)} segments")

                self.tts_result_queue.put(("complete", gen_id, None))  # EOS marker
                logger.info(f"TTS worker: Completed generation ID {gen_id}")

            except queue.Empty:
                continue
            except Exception as e:
                import traceback

                logger.error(f"Error in TTS worker: {e}\n{traceback.format_exc()}")

                # Clean up CUDA memory on error to prevent cascading failures
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache after error")

                # Put error with generation ID if available
                error_gen_id = gen_id if "gen_id" in locals() else 0
                self.tts_result_queue.put(("error", error_gen_id, str(e)))

        logger.info("TTS worker thread exiting")

    def wait_for_tts_warmup(self, timeout: float = 120.0):
        """Wait for TTS worker to finish warming up.

        Args:
            timeout: Maximum seconds to wait for warmup (default 120s).

        Raises:
            RuntimeError: If warmup times out or reports failure.
        """
        try:
            msg_type, _ = self.tts_result_queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                f"TTS warmup timed out after {timeout}s - model may have failed to load"
            ) from None
        if msg_type != "warmup_complete":
            raise RuntimeError("TTS warmup failed")

    def transcribe_audio(self, audio_data, sample_rate=16000):
        """Transcribe audio using Whisper."""
        return self.stt.transcribe_audio(audio_data, sample_rate)

    def trim_to_last_sentence(self, text: str) -> str:
        """Trim text to last complete sentence boundary (CSM pattern).

        Returns text truncated at the final full sentence boundary.
        A boundary is considered to be any '.', '!' or '?' followed by
        optional quotes/brackets, optional whitespace, and then end-of-string.

        This ensures audio generation never cuts off mid-sentence.
        """
        import re

        # Regex: (.*?[.!?]["')\]]?) matches text ending with sentence terminator
        # \s*$ ensures we're at end of string (with optional trailing whitespace)
        m = re.match(r"^(.*?[.!?][\"')\]]?)\s*$", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Fall back to manual search if regex doesn't match
        # (handles cases with additional text after last punctuation)
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ".!?":
                return text[: i + 1].strip()

        # If no sentence boundary found, return as-is
        return text.strip()

    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences at natural boundaries.

        Returns list of sentences, preserving sentence terminators.
        """
        import re

        # Split on sentence boundaries: .!? followed by space or end
        # Keep the punctuation with the sentence
        sentences = re.split(r'([.!?]+["\')]?\s+)', text)

        # Recombine sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append((sentences[i] + sentences[i + 1]).strip())
            else:
                result.append(sentences[i].strip())

        # Handle last sentence if it doesn't have trailing punctuation
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())

        return [s for s in result if s]  # Filter empty strings

    def _chunk_text_by_tokens(self, text: str, max_text_tokens: int) -> list:
        """Split text into chunks that fit within token limits.

        Groups sentences together until approaching token limit, then starts new chunk.
        Ensures each chunk ends at a sentence boundary.

        Args:
            text: Text to chunk
            max_text_tokens: Maximum tokens per chunk (for text only, not including audio)

        Returns:
            List of text chunks, each ending at sentence boundary
        """
        sentences = self._split_into_sentences(text)

        if not sentences:
            return [text]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            # If single sentence exceeds limit, include it anyway (unavoidable)
            if sentence_tokens > max_text_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                chunks.append(sentence)
                continue

            # If adding this sentence would exceed limit, start new chunk
            if current_tokens + sentence_tokens > max_text_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def generate_llm_response(self, user_text):
        """Generate LLM response with optional knowledge graph integration.

        If knowledge integration is enabled and Neo4j is available, uses
        knowledge-augmented conversation with autonomous tool use.
        Otherwise falls back to standard LLM generation.
        """
        # Use knowledge-augmented generation if available
        if self.knowledge and self.knowledge.is_enabled():
            logger.info("Using knowledge-augmented LLM response")
            yield from self.knowledge.generate_response_streaming(
                user_text, event_loop=self.event_loop
            )
        else:
            # Fallback to standard LLM (original implementation)
            logger.debug("Using standard LLM response (no knowledge integration)")

            system_prompt = """You are M.I.S.T, a helpful voice assistant and friend to your creator, Raj Gadhia.

Response Guidelines:
- For simple questions or greetings: 1-3 sentences (brief and direct)
- For requests asking for multiple items, stories, examples, or detailed explanations: provide complete, thorough responses
- When user says "give me N things/stories/examples", provide all N items fully
- When user asks "explain" or "tell me about", give comprehensive answers
- Use a warm, friendly tone suitable for spoken conversation
- Always prioritize correctness, accuracy, and thoroughness
- Don't artificially truncate content the user explicitly requested

Match your response depth to what the user is asking for - be concise when appropriate, thorough when needed."""

            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                stream=True,
                options={
                    "num_predict": 400,  # Allow detailed responses when needed, chunking handles overflow
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            )

            for chunk in response:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

    def _estimate_tokens(self, text, audio_tensor=None):
        """Estimate token count for text and audio following CSM's approach.

        Returns: estimated token count
        """
        # Text token estimation: ~1.3 tokens per word (conservative)
        words = text.split()
        punctuation = sum(1 for char in text if char in ".,!?;:\"'()[]{}")
        text_tokens = len(words) + punctuation

        # Audio token estimation: frames / conversion ratio
        # CSM uses: audio_frames = segment.audio.size(0) // 6094
        audio_tokens = 0
        if audio_tensor is not None:
            audio_frames = audio_tensor.size(0) // 6094
            audio_tokens = audio_frames

        return text_tokens + audio_tokens

    def _calculate_context_tokens(self, context):
        """Calculate total tokens in context following CSM's approach.

        Returns: total token count for all segments in context
        """
        total_tokens = 0

        for segment in context:
            # Use tokenizer if available, otherwise estimate
            if hasattr(self.tts, "generator") and hasattr(self.tts.generator, "_text_tokenizer"):
                try:
                    tokens = self.tts.generator._text_tokenizer.encode(
                        f"[{segment.speaker}]{segment.text}"
                    )
                    segment_text_tokens = len(tokens)
                except:
                    # Fallback to estimation
                    segment_text_tokens = self._estimate_tokens(segment.text)
            else:
                segment_text_tokens = self._estimate_tokens(segment.text)

            # Add audio tokens
            if segment.audio is not None:
                audio_frames = segment.audio.size(0) // 6094
                segment_audio_tokens = audio_frames
            else:
                segment_audio_tokens = 0

            total_tokens += segment_text_tokens + segment_audio_tokens

        return total_tokens

    def generate_tts_audio(self, text):
        """Generate TTS audio (streaming) with intelligent chunking for long responses.

        Strategy:
        1. Preprocess and analyze text token requirements
        2. If text fits in token budget: generate directly
        3. If text exceeds budget: chunk at sentence boundaries and generate sequentially
        4. Preserve context between chunks for voice consistency
        """
        # If TTS is disabled, return empty generator
        if not self.config.tts_enabled or self.tts is None:
            logger.debug("TTS disabled - skipping audio generation")
            return

        # Preprocess text (prosody-preserving substitutions)
        preprocessed_text = preprocess_text_for_tts(text)

        # Get context if enabled
        context = self.tts.context if self.tts and self.tts.use_context else []
        speaker_id = self.tts.speaker_id if self.tts else 0
        ref_count = len(self.tts.profile.reference_clips) if self.tts else 3

        # CSM model limits
        MAX_CONTEXT_TOKENS = 800  # Safe context limit before trimming

        # Calculate tokens in context
        context_tokens = self._calculate_context_tokens(context)

        # Safety check: If context is too large, trim it before proceeding
        # This prevents crashes from previous long responses
        if context_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(
                f"Context too large ({context_tokens} tokens) - trimming to references only"
            )
            context = context[:ref_count] if len(context) >= ref_count else context
            context_tokens = self._calculate_context_tokens(context)
            logger.info(f"Trimmed context to {context_tokens} tokens")

        # Additional safety: If even references are huge, reduce them
        if context_tokens > 400:
            logger.warning(f"Even reference context is large ({context_tokens} tokens)")
            logger.warning("Using only 2 reference clips to prevent CUDA errors")
            context = context[:2] if len(context) >= 2 else context
            context_tokens = self._calculate_context_tokens(context)
            logger.info(f"Reduced to {context_tokens} tokens")

        # Calculate tokens needed for input text
        text_tokens = self._estimate_tokens(preprocessed_text)

        # CRITICAL: Check if combined input (context + text) is safe
        # CSM has max_seq_len=2048 but we need to be conservative
        # Leave room for audio output generation (~same size as text input)
        total_input_tokens = context_tokens + text_tokens
        MAX_SAFE_INPUT = 900  # Conservative but allows reasonable responses

        if total_input_tokens > MAX_SAFE_INPUT:
            logger.warning(
                f"Total input ({total_input_tokens} tokens) exceeds safe limit ({MAX_SAFE_INPUT})"
            )
            logger.warning("Aggressively trimming context to prevent CUDA index errors")

            # Keep only references, drop all conversation history
            context = context[:ref_count] if len(context) >= ref_count else context
            context_tokens = self._calculate_context_tokens(context)
            total_input_tokens = context_tokens + text_tokens
            logger.info(
                f"After trimming: context={context_tokens}, total_input={total_input_tokens}"
            )

            # If STILL too large after trimming, we must chunk the text
            if total_input_tokens > MAX_SAFE_INPUT:
                logger.error(
                    f"Even with minimal context, input is too large ({total_input_tokens} tokens)"
                )
                logger.error("Text is too long for single generation - forcing chunking")
                # Set text_tokens high to force chunking path below
                text_tokens = 999  # Will trigger EXTREME_LENGTH_THRESHOLD

        logger.info(
            f"Token analysis: context={context_tokens}, text={text_tokens}, total_input={total_input_tokens}"
        )

        # Decision: Chunk or single generation?
        # Real-world testing shows CSM quality degrades on generations >150 tokens
        # Symptoms: stuttering, repetition, random utterances, losing coherence
        # Solution: Chunk at sentence boundaries for stable quality
        CHUNK_THRESHOLD = 150  # Tested sweet spot - balances quality vs latency

        if text_tokens <= CHUNK_THRESHOLD:
            logger.info(f"Short response ({text_tokens} tokens) - single generation")
            yield from self._generate_single_chunk(preprocessed_text, context, speaker_id)
        else:
            # Long response - use sentence-boundary chunking
            logger.info(f"Long response ({text_tokens} tokens) - sentence-boundary chunking")

            # Use richer context: 3 references + most recent conversation turn (if available)
            # This maintains voice quality better than references-only
            ref_count = len(self.tts.profile.reference_clips)
            reference_context = list(context[:ref_count] if len(context) >= ref_count else [])
            recent_context = context[-1:] if len(context) > ref_count else []
            rich_context = list(reference_context + recent_context)

            rich_context_tokens = self._calculate_context_tokens(rich_context)
            logger.info(
                f"Using rich context: {len(reference_context)} references + "
                f"{len(recent_context)} recent = {rich_context_tokens} tokens"
            )

            # Target: ~100-120 tokens per chunk for stable generation
            TARGET_CHUNK_SIZE = 120

            # Chunk text at sentence boundaries
            text_chunks = self._chunk_text_by_tokens(preprocessed_text, TARGET_CHUNK_SIZE)
            logger.info(f"Split into {len(text_chunks)} chunks (~{TARGET_CHUNK_SIZE} tokens each)")

            # Generate chunks with context carryover: after generating chunk N,
            # add its audio as context for chunk N+1 so prosody stays continuous.
            # Reference clips are always kept; only one carryover segment at a time
            # to stay within the ~800 token context budget.
            for i, chunk_text in enumerate(text_chunks):
                logger.info(f"[CHUNK {i+1}/{len(text_chunks)}] Generating: '{chunk_text[:50]}...'")

                # Generate this chunk, collecting audio parts for context carryover
                chunk_audio_parts = []
                for audio_part in self._generate_single_chunk(chunk_text, rich_context, speaker_id):
                    chunk_audio_parts.append(audio_part)
                    yield audio_part

                # Critical: Clean up between chunks
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                # Reset KV cache for clean state
                if self.tts and hasattr(self.tts.generator, "_model"):
                    try:
                        self.tts.generator._model.reset_caches()
                        logger.info(f"[CHUNK {i+1}] Reset KV cache")
                    except Exception as e:
                        logger.warning(f"[CHUNK {i+1}] KV reset failed: {e}")

                # Add this chunk as context for the next chunk so it "hears"
                # the previous output and maintains prosody continuity.
                # Keep reference clips, replace any previous carryover segment.
                if i < len(text_chunks) - 1 and chunk_audio_parts:
                    chunk_audio = self._crossfade_chunks(chunk_audio_parts)
                    chunk_audio_cpu = chunk_audio.cpu()

                    chunk_segment = Segment(
                        speaker=speaker_id,
                        text=chunk_text,
                        audio=chunk_audio_cpu,
                    )

                    # Budget check: references + carryover must stay under limit
                    candidate_context = list(reference_context) + [chunk_segment]
                    candidate_tokens = self._calculate_context_tokens(candidate_context)

                    if candidate_tokens <= MAX_CONTEXT_TOKENS:
                        rich_context = candidate_context
                        logger.info(
                            f"[CHUNK {i+1}] Added carryover context " f"({candidate_tokens} tokens)"
                        )
                    else:
                        # Carryover would exceed budget; keep references only
                        rich_context = list(reference_context)
                        logger.warning(
                            f"[CHUNK {i+1}] Carryover too large "
                            f"({candidate_tokens} tokens), using references only"
                        )

                logger.info(f"[CHUNK {i+1}/{len(text_chunks)}] Complete")

            # After all chunks complete, aggressively trim the main TTS context
            # The TTS worker has been adding each chunk to self.tts.context
            # For multi-chunk responses, we need to ensure context doesn't exceed token budget
            if self.tts and self.tts.use_context:
                logger.info(
                    f"Chunking complete. Context before cleanup: {len(self.tts.context)} segments"
                )

                # Trim: keep reference clips + 1 most recent
                rc = len(self.tts.profile.reference_clips)
                max_seg = rc + 1
                if len(self.tts.context) > max_seg:
                    self.tts.context = self.tts.context[:rc] + self.tts.context[-1:]
                    logger.info(f"Trimmed to {len(self.tts.context)} segments")

                # Then, token-aware trimming: ensure we're under safe limit
                context_tokens = self._calculate_context_tokens(self.tts.context)
                MAX_CONTEXT_TOKENS = 800  # Leave plenty of room for next generation

                # If still too large, drop the oldest non-reference segment
                while context_tokens > MAX_CONTEXT_TOKENS and len(self.tts.context) > 3:
                    # Remove the oldest non-reference segment (index 3)
                    self.tts.context.pop(3)
                    logger.info(
                        f"Removed segment to fit token budget (was {context_tokens} tokens)"
                    )
                    context_tokens = self._calculate_context_tokens(self.tts.context)

                logger.info(
                    f"Final context: {len(self.tts.context)} segments, {context_tokens} tokens"
                )

    def _crossfade_chunks(self, chunks: list, crossfade_ms: int = 50) -> torch.Tensor:
        """Concatenate audio chunks with raised-cosine crossfade.

        Applies a smooth fade-out/fade-in at chunk boundaries to prevent
        amplitude discontinuities (clicks/pops) from hard concatenation.

        Args:
            chunks: List of 1-D audio tensors (24kHz sample rate assumed)
            crossfade_ms: Crossfade duration in milliseconds

        Returns:
            Single concatenated tensor with crossfaded boundaries
        """
        if not chunks:
            return torch.tensor([])
        if len(chunks) == 1:
            return chunks[0]

        sample_rate = 24000
        crossfade_samples = int(crossfade_ms / 1000 * sample_rate)

        result = chunks[0]
        for chunk in chunks[1:]:
            if len(result) < crossfade_samples or len(chunk) < crossfade_samples:
                # Too short to crossfade, just concatenate
                result = torch.cat([result, chunk], dim=0)
                continue

            # Raised cosine fade
            fade = torch.linspace(0, 1, crossfade_samples, device=result.device)

            # Fade out end of result, fade in start of chunk
            result_end = result[-crossfade_samples:]
            chunk_start = chunk[:crossfade_samples]

            crossfaded = result_end * (1 - fade) + chunk_start * fade

            result = torch.cat(
                [
                    result[:-crossfade_samples],
                    crossfaded,
                    chunk[crossfade_samples:],
                ],
                dim=0,
            )

        return result

    def _generate_single_chunk(self, preprocessed_text, context, speaker_id):
        """Generate TTS audio for a single chunk of text.

        This is the core generation logic extracted for reuse by chunking system.
        """
        # Increment generation ID and clear stale results
        with self.tts_lock:
            self.tts_generation_id += 1
            current_gen_id = self.tts_generation_id

            # Clear any stale chunks from previous generations
            cleared = 0
            while not self.tts_result_queue.empty():
                try:
                    self.tts_result_queue.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break

            if cleared > 0:
                logger.info(f"Cleared {cleared} stale audio chunks from result queue")

        logger.info(f"TTS generation ID {current_gen_id}: '{preprocessed_text[:50]}...'")

        # Calculate token requirements for this chunk
        context_tokens = self._calculate_context_tokens(context)
        text_tokens = self._estimate_tokens(preprocessed_text)
        input_tokens = context_tokens + text_tokens

        MAX_SEQ_LEN = 2048
        SAFETY_MARGIN = 50
        available_output_tokens = MAX_SEQ_LEN - input_tokens - SAFETY_MARGIN

        logger.info(
            f"Chunk tokens: context={context_tokens}, text={text_tokens}, "
            + f"available_output={available_output_tokens}"
        )

        # Convert available tokens to audio frames
        # Each frame = 80ms of audio (CSM spec)
        max_output_frames = max(1, available_output_tokens)
        max_audio_length_from_tokens = max_output_frames * 80

        # Estimate audio duration based on text length
        words = len(preprocessed_text.split())
        # ~450ms/word estimate (133 words/min, moderate pace)
        # Add 50% buffer for pauses and natural speech variation
        estimated_ms = int(words * 450 * 1.5)

        # Use text-based estimate as primary, token-based as upper bound
        # Text estimate is grounded in actual content; token budget is just a ceiling
        max_audio_length = estimated_ms
        max_audio_length = min(max_audio_length, max_audio_length_from_tokens)
        max_audio_length = min(max_audio_length, 90_000)  # Hard cap at 90s per chunk
        max_audio_length = max(max_audio_length, 5000)  # Min 5s

        logger.info(
            f"Max audio: {max_audio_length}ms (token_limit={max_audio_length_from_tokens}ms, "
            + f"text_estimate={estimated_ms}ms, words={words})"
        )

        # Guard against uninitialized TTS before accessing profile
        if self.tts is None:
            raise RuntimeError("TTS not initialized - worker thread may have failed to start")

        # Submit generation request to worker thread
        self.tts_request_queue.put(
            (
                current_gen_id,
                preprocessed_text,
                speaker_id,
                context,
                max_audio_length,
                self.tts.profile.temperature,
                self.tts.profile.topk,
            )
        )

        # Yield audio chunks from result queue
        RESULT_QUEUE_TIMEOUT = 120.0  # First generation includes torch.compile (~30-60s)
        while True:
            try:
                msg_type, gen_id, data = self.tts_result_queue.get(timeout=RESULT_QUEUE_TIMEOUT)
            except queue.Empty:
                logger.error(
                    f"TTS generation timed out after {RESULT_QUEUE_TIMEOUT}s "
                    "- worker may be stuck"
                )
                raise RuntimeError(
                    f"TTS generation timeout after {RESULT_QUEUE_TIMEOUT}s"
                ) from None

            # Skip chunks from old generations
            if gen_id != current_gen_id:
                logger.warning(
                    f"Skipping chunk from old generation {gen_id} (current: {current_gen_id})"
                )
                continue

            if msg_type == "chunk":
                yield data
            elif msg_type == "complete":
                logger.info(f"TTS generation ID {current_gen_id} complete")
                break
            elif msg_type == "error":
                raise RuntimeError(f"TTS generation failed: {data}")

    def shutdown(self):
        """Shutdown model workers."""
        if self.tts_worker_running.is_set():
            logger.info("Shutting down TTS worker...")
            self.tts_worker_running.clear()
            self.tts_request_queue.put(None)
            if self.tts_worker_thread:
                self.tts_worker_thread.join(timeout=2.0)
