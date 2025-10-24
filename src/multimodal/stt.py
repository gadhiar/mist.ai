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

        # Transcribe
        result = self.model.transcribe(audio.flatten(), fp16=False)
        text = result["text"].strip()

        return text

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        result = self.model.transcribe(audio_path, fp16=False)
        return result["text"].strip()
