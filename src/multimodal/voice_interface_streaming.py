"""
Streaming voice interface: Real-time STT -> LLM streaming -> TTS streaming
Lowest latency voice interaction
"""

import queue
import re
import threading
import time

import ollama
import sounddevice as sd
import torch

from src.multimodal.stt import WhisperSTT
from src.multimodal.tts import SesameTTS
from src.utils.cleanup import register_cleanup


class StreamingVoiceInterface:
    """Voice interface with aggressive streaming for minimum latency"""

    def __init__(self, llm_model: str = "qwen2.5:7b-instruct"):
        """
        Initialize streaming voice interface

        Args:
            llm_model: Ollama model name (must support streaming)
        """
        print("Initializing streaming voice interface...")
        print("This will take 1-2 minutes on first run (loading all models)...\n")

        # Initialize components
        print("1/3 Loading Whisper STT...")
        self.stt = WhisperSTT(model_size="base")

        print("2/3 Loading Sesame CSM TTS...")
        self.tts = SesameTTS(use_context=True)

        # Warmup TTS (loads models to GPU)
        print("   Warming up TTS...")
        self.tts.speak("Initialization complete.", play=False)

        self.llm_model = llm_model

        # Register cleanup
        register_cleanup()

        # Warmup LLM
        print("3/3 Warming up LLM...")
        ollama.chat(model=self.llm_model, messages=[{"role": "user", "content": "hi"}])

        print("\n Streaming voice interface ready!\n")

        # Audio playback management
        self.audio_queue = queue.Queue()
        self.playback_thread = None
        self.stop_playback = threading.Event()
        self.first_playback_time = None
        self.playback_lock = threading.Lock()

    def _audio_playback_worker(self):
        """
        Background thread using OutputStream for gap-free continuous playback

        Uses sounddevice.OutputStream which keeps audio device open continuously,
        eliminating gaps caused by sd.play() reopening the device each time.
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
                            # Fill remaining with silence
                            outdata[samples_written:, 0] = 0
                            raise sd.CallbackStop()

                    except queue.Empty:
                        # No audio available - fill with silence (prevents glitches)
                        outdata[samples_written:, 0] = 0
                        break

                # Copy audio from current chunk to output
                remaining_in_chunk = len(self._current_chunk) - self._chunk_position
                samples_to_copy = min(remaining_in_chunk, samples_needed - samples_written)

                outdata[samples_written : samples_written + samples_to_copy, 0] = (
                    self._current_chunk[
                        self._chunk_position : self._chunk_position + samples_to_copy
                    ]
                )

                self._chunk_position += samples_to_copy
                samples_written += samples_to_copy

        # Start the continuous output stream
        stream = sd.OutputStream(
            samplerate=24000,
            channels=1,
            blocksize=2048,  # ~85ms latency, good balance for conversational AI
            callback=audio_callback,
            dtype=np.float32,
        )

        try:
            # PRE-BUFFER: Wait for first 2-3 chunks before starting stream
            # This prevents initial stutter when callback requests audio before queue is ready
            initial_chunks = []
            for _ in range(3):  # Buffer 3 chunks
                try:
                    chunk = self.audio_queue.get(timeout=5.0)
                    if chunk is None:
                        break

                    # Skip empty/invalid chunks
                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.cpu().numpy()
                    if len(chunk) < 100:
                        continue

                    initial_chunks.append(chunk)
                except queue.Empty:
                    break

            # Feed initial chunks to playback buffer
            for chunk in initial_chunks:
                self._playback_buffer.put(chunk)

            # NOW start stream - buffer is ready
            stream.start()
            self._stream_active = True

            # Record timing when stream actually starts
            if self.first_playback_time is None and initial_chunks:
                with self.playback_lock:
                    if self.first_playback_time is None:
                        self.first_playback_time = time.time()

            # Process incoming chunks and feed to stream
            while not self.stop_playback.is_set():
                try:
                    # Get next audio chunk from generation
                    audio_chunk = self.audio_queue.get(timeout=0.5)

                    if audio_chunk is None:  # Stop signal
                        # Send stop signal to callback
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

                    # Feed to stream (blocks if buffer full - prevents memory overflow)
                    self._playback_buffer.put(audio_np, timeout=5.0)

                except queue.Empty:
                    # Queue empty but not stopping - keep waiting
                    continue
                except Exception as e:
                    print(f"\nWarning: Audio playback error: {e}")
                    continue

            # Wait for stream to finish playing buffered audio
            while not self._playback_buffer.empty():
                time.sleep(0.1)

            time.sleep(0.2)  # Small buffer for last audio

        finally:
            stream.stop()
            stream.close()
            self._stream_active = False

    def _stream_llm(self, prompt: str):
        """
        Stream tokens from LLM

        Args:
            prompt: User input

        Yields:
            Text chunks as they're generated
        """
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,  # Enable streaming mode
        )

        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    def _split_into_speech_chunks(self, text: str) -> list:
        """
        Split text into natural speech chunks

        Splits on sentence boundaries and punctuation for natural pauses

        Args:
            text: Full text to split

        Returns:
            List of text chunks
        """
        # Split on sentence endings or long clauses
        chunks = re.split(r"([.!?;:,]\s+)", text)

        # Recombine chunks with their punctuation
        result = []
        for i in range(0, len(chunks) - 1, 2):
            if i + 1 < len(chunks):
                result.append(chunks[i] + chunks[i + 1])
            else:
                result.append(chunks[i])

        # Filter out empty chunks
        return [chunk.strip() for chunk in result if chunk.strip()]

    def _is_chunk_boundary(self, text: str) -> bool:
        """
        Check if text contains a good boundary for speech synthesis

        Args:
            text: Text chunk to check

        Returns:
            True if this is a good place to pause and generate audio
        """
        # Check for sentence endings or natural pauses
        return any(punct in text for punct in [".", "!", "?", ";\n", ":\n"])

    def _preprocess_text_for_tts(self, text: str) -> str:
        """
        Preprocess text for TTS following csm-streaming best practices

        Removes all punctuation except periods, commas, exclamation points,
        and question marks to create cleaner speech output while preserving intonation.

        Args:
            text: Input text with potential punctuation

        Returns:
            Cleaned text with only allowed punctuation
        """
        # Remove all punctuation except . , ! ? '
        pattern = r"[^\w\s.,!?\']"
        cleaned_text = re.sub(pattern, "", text)

        # Normalize multiple spaces to single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        # Ensure there's a space after punctuation for better speech pacing
        cleaned_text = re.sub(r"([.,!?])(\S)", r"\1 \2", cleaned_text)

        return cleaned_text.strip()

    def converse_streaming(self, duration: int = 5, debug: bool = False) -> tuple[str, str]:
        """
        Streaming conversation: Start speaking as soon as possible

        Following csm-streaming best practices:
        1. Buffer complete LLM response
        2. Preprocess entire text for TTS
        3. Stream at AUDIO level (not text level) using generate_stream()

        Timeline:
        0-Xs:   User speaks (X = duration)
        X-X+2s: STT transcription
        X+2s:   LLM generates response
        X+Ys:   First audio chunk arrives from TTS
        X+Ys:   USER HEARS FIRST WORDS (vs X+18s with blocking!)

        Args:
            duration: Recording duration in seconds
            debug: Enable timing debug output

        Returns:
            (user_text, ai_response)
        """
        t_start = time.time()

        # Reset playback timing
        self.first_playback_time = None

        # 1. Listen (blocking, no way around this)
        print(" Listening...")
        user_text = self.stt.listen(duration=duration)
        t_stt = time.time()
        if debug:
            print(f"[DEBUG] STT: {t_stt - t_start:.2f}s")
        print(f"You: {user_text}\n")

        if not user_text.strip():
            print("(No speech detected)")
            return user_text, ""

        # 2. Stream LLM response and collect full text
        print("Mist: ", end="", flush=True)

        full_response = ""
        t_first_token = None

        try:
            # Collect full LLM response
            for token in self._stream_llm(user_text):
                if t_first_token is None:
                    t_first_token = time.time()
                    if debug:
                        print(f"\n[DEBUG] First token: {t_first_token - t_stt:.2f}s", flush=True)
                        print("Mist: ", end="", flush=True)

                full_response += token
                print(token, end="", flush=True)

            print()  # Newline after response
            t_llm_done = time.time()

            if debug:
                print(f"[DEBUG] LLM response complete: {t_llm_done - t_stt:.2f}s")

            if not full_response.strip():
                print("(No response generated)")
                return user_text, ""

            # 3. Preprocess text for TTS (csm-streaming pattern)
            preprocessed_text = self._preprocess_text_for_tts(full_response.lower())

            if debug:
                print(f"[DEBUG] Preprocessed text: '{preprocessed_text[:100]}...'")

            # 4. Start audio playback thread
            self.stop_playback.clear()
            self.playback_thread = threading.Thread(target=self._audio_playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()

            # 5. Stream TTS at AUDIO level (not text level)
            # Pass ENTIRE text to generate_stream - it handles chunking internally
            print(" Generating speech...")
            t_tts_start = time.time()
            audio_chunk_count = 0
            t_first_audio_chunk = None

            # Use the streaming TTS API directly
            # Using production csm-streaming parameters for best quality
            for audio_chunk in self.tts.generator.generate_stream(
                text=preprocessed_text,
                speaker=self.tts.speaker_id,
                context=self.tts.context if self.tts.use_context else [],
                max_audio_length_ms=self.tts._estimate_audio_length(preprocessed_text),
                temperature=0.8,  # Production setting (was 0.7)
                topk=50,  # Production setting (was 30)
            ):
                if t_first_audio_chunk is None:
                    t_first_audio_chunk = time.time()
                    if debug:
                        print(
                            f"[DEBUG] First audio chunk: {t_first_audio_chunk - t_stt:.2f}s from end of speech"
                        )

                audio_chunk_count += 1

                # Queue audio chunk for playback
                self.audio_queue.put(audio_chunk)

            t_tts_done = time.time()

            if debug:
                print(
                    f"[DEBUG] TTS generation complete: {t_tts_done - t_tts_start:.2f}s ({audio_chunk_count} chunks)"
                )

            # Update TTS context for voice consistency
            if self.tts.use_context and audio_chunk_count > 0:
                # Concatenate all chunks for context
                # (audio already sent to queue, we just need to track context)

                # We'll update context after playback completes
                pass

            # Signal end of audio
            self.audio_queue.put(None)

            # Wait for playback to finish
            if self.playback_thread.is_alive():
                self.playback_thread.join()

            t_end = time.time()

            if debug:
                print("\n[DEBUG] === TIMING BREAKDOWN ===")
                print(f"[DEBUG] Total time: {t_end - t_start:.2f}s")
                print(f"[DEBUG] STT: {t_stt - t_start:.2f}s")
                if t_first_token:
                    print(f"[DEBUG] First LLM token: {t_first_token - t_stt:.2f}s")
                if t_llm_done:
                    print(f"[DEBUG] LLM complete: {t_llm_done - t_stt:.2f}s")
                if t_first_audio_chunk:
                    print(f"[DEBUG] First audio chunk: {t_first_audio_chunk - t_stt:.2f}s")
                if self.first_playback_time:
                    print(
                        f"[DEBUG] First audio PLAYING: {self.first_playback_time - t_start:.2f}s -> USER HEARS"
                    )
                print(
                    f"[DEBUG] TTS generation: {t_tts_done - t_tts_start:.2f}s ({audio_chunk_count} chunks)"
                )
                print(f"[DEBUG] Total playback: {t_end - t_tts_done:.2f}s")
                print("[DEBUG] ========================\n")

        except Exception as e:
            print(f"\nError during streaming: {e}")
            import traceback

            traceback.print_exc()
            self.stop_playback.set()
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join()
            return user_text, full_response

        return user_text, full_response

    def converse(self, duration: int = 5, debug: bool = False) -> tuple[str, str]:
        """
        Single conversation turn - alias for converse_streaming

        Args:
            duration: Recording duration in seconds
            debug: Enable timing debug output

        Returns:
            (user_text, ai_response)
        """
        return self.converse_streaming(duration, debug=debug)

    def continuous_conversation(self, debug: bool = False):
        """
        Continuous conversation loop with low-latency streaming
        Say 'goodbye' or 'exit' to end

        Args:
            debug: Enable timing debug output
        """
        print("\n=== Streaming Voice Conversation ===")
        print("Say 'goodbye' or 'exit' to end\n")

        while True:
            try:
                user_text, ai_text = self.converse_streaming(duration=5, debug=debug)

                # Check for exit
                if any(word in user_text.lower() for word in ["goodbye", "exit", "quit", "stop"]):
                    print("\n Ending conversation. Goodbye!")
                    break

            except KeyboardInterrupt:
                print("\n\n Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n Error: {e}")
                break


# Convenience function for quick testing
def test_streaming_voice():
    """Quick test of streaming voice interface"""
    print("=== Testing Streaming Voice Interface ===\n")

    vi = StreamingVoiceInterface()

    print("\n--- Test 1: Single interaction ---")
    vi.converse_streaming(duration=5)

    print("\n--- Test complete ---")


if __name__ == "__main__":
    test_streaming_voice()
