"""
Model Manager - Handles loading and lifecycle of all ML models
"""
import sys
import time
import logging
import queue
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.multimodal.stt import WhisperSTT
from src.multimodal.tts import SesameTTS
import ollama
import torch

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages all ML models for the voice AI system"""

    def __init__(self, config):
        """
        Initialize model manager

        Args:
            config: VoiceConfig object
        """
        self.config = config
        self.stt = None
        self.tts = None
        self.llm_model = config.llm_model

        # TTS model worker thread (CSM pattern)
        self.tts_request_queue = queue.Queue()
        self.tts_result_queue = queue.Queue()
        self.tts_worker_thread = None
        self.tts_worker_running = threading.Event()

    def load_all_models(self):
        """Load all models (STT, TTS, LLM)"""
        logger.info("Loading all models...")
        start = time.time()

        # Load STT
        logger.info("1/3 Loading Whisper STT...")
        self.stt = WhisperSTT(model_size=self.config.whisper_model)

        # Start TTS worker thread (CSM pattern)
        logger.info("2/3 Starting TTS model worker thread...")
        self._start_tts_worker()

        # Wait for TTS warmup to complete
        logger.info("   Waiting for TTS warmup...")
        self.wait_for_tts_warmup()

        # Warmup LLM
        logger.info("3/3 Warming up LLM...")
        ollama.chat(
            model=self.llm_model, messages=[{"role": "user", "content": "hi"}]
        )

        elapsed = time.time() - start
        logger.info(f"All models loaded in {elapsed:.1f}s")

    def _start_tts_worker(self):
        """Start dedicated TTS model worker thread (CSM pattern)"""
        if self.tts_worker_thread is not None and self.tts_worker_thread.is_alive():
            return

        self.tts_worker_running.set()
        self.tts_worker_thread = threading.Thread(
            target=self._tts_worker,
            daemon=True,
            name="tts_worker"
        )
        self.tts_worker_thread.start()
        logger.info("Started dedicated TTS model worker thread")

    def _tts_worker(self):
        """TTS model worker thread - loads and runs model (CSM pattern)"""
        logger.info("TTS worker thread started")

        # Disable CUDA graphs to avoid threading issues (CSM pattern)
        torch._inductor.config.triton.cudagraphs = False
        torch._inductor.config.fx_graph_cache = False

        logger.info("Loading Sesame TTS in worker thread...")
        self.tts = SesameTTS(
            device=self.config.tts_device,
            use_context=self.config.use_voice_context
        )

        # Warmup TTS
        logger.info("Warming up TTS with dummy text...")
        dummy_audio = self.tts.speak("Initialization complete.", play=False)
        logger.info(f"TTS warmed up (audio shape: {dummy_audio.shape})")

        # Signal warmup complete
        self.tts_result_queue.put(("warmup_complete", None))

        # Process requests
        while self.tts_worker_running.is_set():
            try:
                request = self.tts_request_queue.get(timeout=0.1)
                if request is None:
                    break

                text, speaker_id, context, max_ms, temperature, topk = request

                for chunk in self.tts.generator.generate_stream(
                    text=text,
                    speaker=speaker_id,
                    context=context,
                    max_audio_length_ms=max_ms,
                    temperature=temperature,
                    topk=topk
                ):
                    self.tts_result_queue.put(("chunk", chunk))

                    if not self.tts_worker_running.is_set():
                        break

                self.tts_result_queue.put(("complete", None))  # EOS marker

            except queue.Empty:
                continue
            except Exception as e:
                import traceback
                logger.error(f"Error in TTS worker: {e}\n{traceback.format_exc()}")
                self.tts_result_queue.put(("error", e))

        logger.info("TTS worker thread exiting")

    def wait_for_tts_warmup(self):
        """Wait for TTS worker to finish warming up"""
        msg_type, _ = self.tts_result_queue.get()
        if msg_type != "warmup_complete":
            raise RuntimeError("TTS warmup failed")

    def transcribe_audio(self, audio_data, sample_rate=16000):
        """Transcribe audio using Whisper"""
        return self.stt.transcribe_audio(audio_data, sample_rate)

    def generate_llm_response(self, user_text):
        """Generate LLM response (streaming)"""
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": user_text}],
            stream=True,
        )

        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]

    def generate_tts_audio(self, text):
        """Generate TTS audio (streaming) via worker thread"""
        import re

        # Preprocess text
        pattern = r"[^\w\s.,!?\']"
        cleaned_text = re.sub(pattern, "", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        cleaned_text = re.sub(r"([.,!?])(\S)", r"\1 \2", cleaned_text)
        preprocessed_text = cleaned_text.strip().lower()

        # Get context if enabled
        context = self.tts.context if self.tts and self.tts.use_context else []
        speaker_id = self.tts.speaker_id if self.tts else 0

        # Send request to worker thread (CSM pattern)
        self.tts_request_queue.put((
            preprocessed_text,
            speaker_id,
            context,
            30000,  # 30 seconds max
            self.config.tts_temperature,
            self.config.tts_topk,
        ))

        # Yield chunks from result queue
        while True:
            msg_type, data = self.tts_result_queue.get()

            if msg_type == "chunk":
                yield data
            elif msg_type == "complete":
                break
            elif msg_type == "error":
                raise RuntimeError(f"TTS generation failed: {data}")

    def shutdown(self):
        """Shutdown model workers"""
        if self.tts_worker_running.is_set():
            logger.info("Shutting down TTS worker...")
            self.tts_worker_running.clear()
            self.tts_request_queue.put(None)
            if self.tts_worker_thread:
                self.tts_worker_thread.join(timeout=2.0)
