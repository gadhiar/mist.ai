"""
Speech-to-Text using Whisper
"""
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import wave


class WhisperSTT:
    """Simple wrapper for Whisper speech-to-text"""

    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper STT

        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large'
                       base = good balance of speed/accuracy
        """
        print(f"Loading Whisper ({model_size})...")
        self.model = whisper.load_model(model_size)
        self.sample_rate = 16000  # Whisper expects 16kHz

        # Initial prompt to bias Whisper towards "MIST" instead of "missed"
        self.initial_prompt = (
            "MIST is an AI assistant with knowledge graph capabilities. "
            "The user is asking about MIST, the AI system, not missed or mist."
        )

        print("Whisper loaded successfully!")

    def listen(self, duration: int = 5) -> str:
        """
        Record audio from microphone and transcribe

        Args:
            duration: Recording duration in seconds

        Returns:
            Transcribed text
        """
        print(f"Recording for {duration} seconds...")
        print("Speak now!")

        # Record audio
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        print("Recording complete. Transcribing...")

        # Transcribe with initial prompt to bias towards "MIST"
        result = self.model.transcribe(
            audio.flatten(),
            fp16=False,
            initial_prompt=self.initial_prompt
        )
        text = result["text"].strip()

        return text

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data directly (for VAD-based systems)

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio (default: 16000)

        Returns:
            Transcribed text
        """
        import torch

        # Ensure audio is float32 and 1D
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()

        # Resample if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            # Simple resampling by decimation/interpolation
            # For production, consider using librosa or torchaudio for better quality
            from scipy import signal
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)

        # Synchronize CUDA before transcription to prevent conflicts with TTS CUDA graphs
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Transcribe with initial prompt to bias towards "MIST"
        result = self.model.transcribe(
            audio_data,
            fp16=False,
            initial_prompt=self.initial_prompt
        )

        # Synchronize CUDA after transcription to ensure completion before TTS starts
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return result["text"].strip()

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        result = self.model.transcribe(
            audio_path,
            fp16=False,
            initial_prompt=self.initial_prompt
        )
        return result["text"].strip()
