"""
Interruptible Voice Interface

Allows user to interrupt AI mid-response by speaking.
Uses continuous VAD-based microphone listening + OutputStream interruption.
"""
import ollama
import queue
import threading
import re
import sounddevice as sd
import torch
import time
import numpy as np
from datetime import datetime
from src.multimodal.stt import WhisperSTT
from src.multimodal.tts import SesameTTS
from src.utils.cleanup import register_cleanup

# Import VAD from csm-streaming
import sys
sys.path.insert(0, 'dependencies/csm')
from vad import AudioStreamProcessor


def log(msg: str):
    """Print message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}")


class InterruptibleVoiceInterface:
    """Voice interface with interrupt capability - user can speak at any time"""

    def __init__(self, llm_model: str = "qwen2.5:7b-instruct", vad_threshold: float = 0.5):
        """
        Initialize interruptible voice interface

        Args:
            llm_model: Ollama model name (must support streaming)
            vad_threshold: Voice activity detection threshold (0.0-1.0, default 0.5)
                          Lower = more sensitive, Higher = less sensitive
        """
        print("Initializing interruptible voice interface...")
        print("This will take 1-2 minutes on first run (loading all models)...\n")

        # Initialize components
        print("1/4 Loading Whisper STT...")
        self.stt = WhisperSTT(model_size="base")

        print("2/4 Loading Sesame CSM TTS...")
        self.tts = SesameTTS(use_context=True)

        # Warmup TTS
        print("   Warming up TTS...")
        self.tts.speak("Initialization complete.", play=False)

        self.llm_model = llm_model

        # Register cleanup
        register_cleanup()

        # Warmup LLM
        print("3/4 Warming up LLM...")
        ollama.chat(model=self.llm_model, messages=[{'role': 'user', 'content': 'hi'}])

        # Initialize VAD
        print("4/4 Loading VAD (Voice Activity Detection)...")
        vad_model, vad_utils = torch.hub.load(
            'snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            verbose=False
        )

        self.vad_processor = AudioStreamProcessor(
            model=vad_model,
            utils=vad_utils,
            sample_rate=16000,
            vad_threshold=vad_threshold,
            callbacks={
                'on_speech_start': self._on_speech_start,
                'on_speech_end': self._on_speech_end
            }
        )

        print("\nInterruptible voice interface ready!")
        print(f"VAD threshold: {vad_threshold} (you can interrupt at any time)\n")

        # Audio playback management
        self.audio_queue = queue.Queue()
        self.playback_thread = None
        self.stop_playback = threading.Event()
        self.interrupt_playback = threading.Event()  # NEW: Interrupt signal
        self.first_playback_time = None
        self.playback_lock = threading.Lock()

        # Microphone management
        self.mic_thread = None
        self.stop_mic = threading.Event()
        self.pending_transcription = queue.Queue()  # NEW: Queue for interruptions

        # State management
        self.ai_speaking = False  # NEW: Track if AI is currently speaking
        self.state_lock = threading.Lock()

    def _on_speech_start(self):
        """Called by VAD when user starts speaking"""
        with self.state_lock:
            if self.ai_speaking:
                log("[INTERRUPT] User started speaking, stopping AI...")
                self.interrupt_playback.set()

    def _on_speech_end(self, audio_data, sample_rate):
        """Called by VAD when user stops speaking"""
        log(f"Speech detected, transcribing... (audio: {len(audio_data)} samples)")

        # Transcribe
        t_start = time.time()
        user_text = self.stt.transcribe_audio(audio_data, sample_rate)
        t_elapsed = time.time() - t_start
        log(f"Transcription complete ({t_elapsed:.2f}s)")
        print(f"You: {user_text}")

        # Add to pending queue
        self.pending_transcription.put(user_text)

    def _continuous_microphone_worker(self):
        """
        Background thread that continuously listens to microphone
        Uses VAD to detect speech and handle interruptions
        """
        print("Microphone listening started (VAD-based)...")

        def audio_callback(indata, frames, time_info, status):
            """Called by sounddevice for each audio chunk"""
            if status:
                print(f"Mic status: {status}")

            # Process audio with VAD
            audio_chunk = indata[:, 0].copy()  # Get mono channel
            self.vad_processor.process_audio(audio_chunk)

        # Start continuous input stream
        log("Starting continuous microphone stream (16kHz, 1024 blocksize)...")
        with sd.InputStream(
            samplerate=16000,
            channels=1,
            blocksize=1024,  # Increased from 512 to reduce overflow
            callback=audio_callback,
            dtype=np.float32
        ):
            log("Microphone stream active")
            # Keep running until stop signal
            while not self.stop_mic.is_set():
                time.sleep(0.1)
        log("Microphone stream stopped")

    def _audio_playback_worker(self):
        """
        Background thread using OutputStream for gap-free continuous playback
        Now supports interruption when user speaks
        """
        import numpy as np

        # Internal buffer for the OutputStream callback
        self._playback_buffer = queue.Queue(maxsize=50)
        self._current_chunk = None
        self._chunk_position = 0
        self._stream_active = False

        def audio_callback(outdata, frames, time_info, status):
            """Called by sounddevice when it needs more audio data"""
            if status:
                print(f"\nStream status: {status}")

            # Check for interruption
            if self.interrupt_playback.is_set():
                # Stop playback immediately
                outdata[:, 0] = 0
                raise sd.CallbackStop()

            samples_needed = frames
            samples_written = 0

            while samples_written < samples_needed:
                # Get current chunk or fetch new one
                if self._current_chunk is None or self._chunk_position >= len(self._current_chunk):
                    try:
                        self._current_chunk = self._playback_buffer.get_nowait()
                        self._chunk_position = 0

                        # Check for stop signal
                        if self._current_chunk is None:
                            outdata[samples_written:, 0] = 0
                            raise sd.CallbackStop()

                    except queue.Empty:
                        # No audio available - fill with silence
                        outdata[samples_written:, 0] = 0
                        break

                # Copy audio from current chunk to output
                remaining_in_chunk = len(self._current_chunk) - self._chunk_position
                samples_to_copy = min(remaining_in_chunk, samples_needed - samples_written)

                outdata[samples_written:samples_written + samples_to_copy, 0] = \
                    self._current_chunk[self._chunk_position:self._chunk_position + samples_to_copy]

                self._chunk_position += samples_to_copy
                samples_written += samples_to_copy

        # Start the continuous output stream
        stream = sd.OutputStream(
            samplerate=24000,
            channels=1,
            blocksize=2048,
            callback=audio_callback,
            dtype=np.float32
        )

        try:
            # PRE-BUFFER: Wait for first 3 chunks before starting stream
            initial_chunks = []
            log("🔊 Playback: Waiting for initial 3 chunks to pre-buffer...")
            prebuffer_start = time.time()
            for i in range(3):
                try:
                    chunk = self.audio_queue.get(timeout=5.0)
                    if chunk is None:
                        log("🔊 Playback: Received stop signal during pre-buffer")
                        break

                    # Skip empty/invalid chunks
                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.cpu().numpy()
                    if len(chunk) < 100:
                        log(f"🔊 Playback: Skipping tiny chunk ({len(chunk)} samples)")
                        continue

                    initial_chunks.append(chunk)
                    log(f"🔊 Playback: Got chunk {i+1}/3 ({len(chunk)} samples)")
                except queue.Empty:
                    break

            # Feed initial chunks to playback buffer
            prebuffer_time = time.time() - prebuffer_start
            log(f"🔊 Playback: Pre-buffer complete ({prebuffer_time:.2f}s, {len(initial_chunks)} chunks)")
            for chunk in initial_chunks:
                self._playback_buffer.put(chunk)

            if not initial_chunks:
                log("⚠ No audio chunks received, skipping playback")
                return

            # NOW start stream - buffer is ready
            log("🔊 Playback: Starting OutputStream (24kHz, blocksize 2048)...")
            stream_start = time.time()
            try:
                stream.start()
                self._stream_active = True
                stream_open_time = time.time() - stream_start
                log(f"Audio playback started (stream opened in {stream_open_time:.3f}s)")
                if stream_open_time > 0.1:
                    log(f"WARNING: Stream opening took >{stream_open_time:.3f}s - this may cause initial stutter")
            except Exception as e:
                log(f"ERROR: Failed to start OutputStream: {e}")
                raise

            # Record timing
            if self.first_playback_time is None and initial_chunks:
                with self.playback_lock:
                    if self.first_playback_time is None:
                        self.first_playback_time = time.time()

            # Mark AI as speaking
            with self.state_lock:
                self.ai_speaking = True
            log("Playback: AI marked as speaking (interruption enabled)")

            # Process incoming chunks and feed to stream
            while not self.stop_playback.is_set() and not self.interrupt_playback.is_set():
                try:
                    # Get next audio chunk from generation
                    audio_chunk = self.audio_queue.get(timeout=0.5)

                    if audio_chunk is None:  # Stop signal
                        self._playback_buffer.put(None)
                        break

                    # Skip empty chunks
                    if isinstance(audio_chunk, torch.Tensor) and audio_chunk.numel() == 0:
                        continue

                    # Convert to numpy
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_np = audio_chunk.cpu().numpy()
                    else:
                        audio_np = audio_chunk

                    # Skip very short chunks
                    if len(audio_np) < 100:
                        continue

                    # Feed to stream
                    self._playback_buffer.put(audio_np, timeout=5.0)

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"\nWarning: Audio playback error: {e}")
                    continue

            # Wait for stream to finish if not interrupted
            if not self.interrupt_playback.is_set():
                log("Playback: Waiting for buffer to drain...")
                while not self._playback_buffer.empty():
                    time.sleep(0.1)
                time.sleep(0.2)
                log("Playback complete (natural end)")
            else:
                log("Playback: Interrupted, clearing buffer...")
                # Clear playback buffer on interruption
                while not self._playback_buffer.empty():
                    try:
                        self._playback_buffer.get_nowait()
                    except queue.Empty:
                        break
                log("Playback stopped (interrupted)")

        finally:
            stream.stop()
            stream.close()
            self._stream_active = False
            log("Playback: OutputStream closed")

            # Mark AI as not speaking
            with self.state_lock:
                self.ai_speaking = False
            log("🔊 Playback: AI no longer speaking (interruption disabled)")

    def start_listening(self):
        """Start continuous microphone listening"""
        if self.mic_thread is None or not self.mic_thread.is_alive():
            self.stop_mic.clear()
            self.mic_thread = threading.Thread(target=self._continuous_microphone_worker)
            self.mic_thread.daemon = True
            self.mic_thread.start()

    def stop_listening(self):
        """Stop continuous microphone listening"""
        self.stop_mic.set()
        if self.mic_thread and self.mic_thread.is_alive():
            self.mic_thread.join(timeout=2.0)

    def _preprocess_text_for_tts(self, text: str) -> str:
        """Preprocess text for TTS"""
        pattern = r'[^\w\s.,!?\']'
        cleaned_text = re.sub(pattern, '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'([.,!?])(\S)', r'\1 \2', cleaned_text)
        return cleaned_text.strip()

    def continuous_conversation(self, debug: bool = False):
        """
        Continuous conversation with interruption support

        User can speak at any time to interrupt AI response
        """
        print("\n=== Interruptible Voice Conversation ===")
        print("Speak at ANY time - even while AI is responding")
        print("Say 'goodbye' or 'exit' to end\n")

        # Start continuous microphone listening
        self.start_listening()

        try:
            while True:
                # Check for pending user input (from VAD)
                try:
                    user_text = self.pending_transcription.get(timeout=0.1)
                except queue.Empty:
                    continue

                log(f"New user input received: '{user_text[:50]}...'")

                # INTERRUPT if AI is currently speaking
                if hasattr(self, 'playback_thread') and self.playback_thread and self.playback_thread.is_alive():
                    log("INTERRUPT: Stopping current AI playback...")
                    self.interrupt_playback.set()
                    self.stop_playback.set()

                    # Wait briefly for thread to stop
                    self.playback_thread.join(timeout=0.5)
                    if self.playback_thread.is_alive():
                        log("WARNING: Playback thread still alive after interrupt signal")
                    else:
                        log("Playback stopped successfully")

                # Clear audio queue from interrupted/previous generation
                queue_cleared = 0
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                        queue_cleared += 1
                    except queue.Empty:
                        break
                if queue_cleared > 0:
                    log(f"Cleared {queue_cleared} stale audio chunks from queue")

                # Check for exit
                if any(word in user_text.lower() for word in ['goodbye', 'exit', 'quit', 'stop']):
                    print("\nEnding conversation. Goodbye!")
                    break

                # Empty input check
                if not user_text.strip():
                    log("WARNING: Empty input, skipping")
                    print("(No speech detected)")
                    continue

                log(f"Processing: '{user_text}'")
                print(f"You: {user_text}")

                # Generate AI response
                log(f"LLM: Generating response for: '{user_text[:50]}...'")
                llm_start = time.time()
                print("Mist: ", end='', flush=True)

                # Stream LLM response and collect full text
                full_response = ""
                for token in self._stream_llm(user_text):
                    full_response += token
                    print(token, end='', flush=True)
                print()
                llm_time = time.time() - llm_start
                log(f"LLM complete ({llm_time:.2f}s, {len(full_response)} chars)")

                if not full_response.strip():
                    continue

                # Preprocess text
                preprocessed_text = self._preprocess_text_for_tts(full_response.lower())
                log(f"TTS: Preprocessed text ({len(preprocessed_text)} chars)")

                # Start audio playback thread
                log("Starting playback worker thread...")
                self.stop_playback.clear()
                self.interrupt_playback.clear()
                self.playback_thread = threading.Thread(target=self._audio_playback_worker)
                self.playback_thread.daemon = True
                self.playback_thread.start()

                # Stream TTS audio
                log("TTS: Starting audio generation...")
                tts_start = time.time()
                chunk_count = 0
                first_chunk_time = None
                for audio_chunk in self.tts.generator.generate_stream(
                    text=preprocessed_text,
                    speaker=self.tts.speaker_id,
                    context=self.tts.context if self.tts.use_context else [],
                    max_audio_length_ms=self.tts._estimate_audio_length(preprocessed_text),
                    temperature=0.8,
                    topk=50
                ):
                    # Check for interruption
                    if self.interrupt_playback.is_set():
                        log("TTS: Generation interrupted by user speech")
                        break

                    chunk_count += 1
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - tts_start
                        log(f"TTS: First chunk generated ({first_chunk_time:.2f}s)")

                    if chunk_count <= 5 or chunk_count % 10 == 0:
                        log(f"TTS: Chunk #{chunk_count} ({len(audio_chunk) if hasattr(audio_chunk, '__len__') else 'N/A'} samples)")

                    self.audio_queue.put(audio_chunk)

                tts_total = time.time() - tts_start
                log(f"TTS generation complete ({tts_total:.2f}s, {chunk_count} chunks)")

                # Signal end of audio generation
                self.audio_queue.put(None)
                log("Sent playback stop signal")

                # DON'T wait for playback to finish - allow immediate interruption
                # The next iteration will handle any interruption if user speaks
                log("Audio generation complete, playback continues in background")
                log("─" * 60)  # Separator for next turn

        except KeyboardInterrupt:
            print("\n\n👋 Conversation interrupted. Goodbye!")

        finally:
            # Stop microphone listening
            self.stop_listening()

    def _stream_llm(self, prompt: str):
        """Stream tokens from LLM"""
        response = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )

        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
