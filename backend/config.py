"""Configuration for Voice AI Backend."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class VoiceConfig(BaseModel):
    """Configuration for voice AI system."""

    # Model paths
    whisper_model: str = "base"
    llm_model: str = "qwen2.5:7b-instruct"
    tts_device: str = "cuda"

    # VAD settings
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_sample_rate: int = 16000

    # TTS settings
    tts_enabled: bool = True  # Set to False to disable TTS (text-only mode)
    voice_profile: str = "cortana"  # Selected via VOICE_PROFILE env var
    use_voice_context: bool = True

    # Server settings
    host: str = "0.0.0.0"  # nosec B104 - intentional bind to all interfaces for dev server
    port: int = 8001  # Changed from 8000 to avoid conflict with HYTE Nexus
    max_connections: int = 10

    # Debug
    debug: bool = False


# Load configuration from environment
def load_config() -> VoiceConfig:
    """Load configuration from environment variables."""
    return VoiceConfig(
        tts_enabled=os.getenv("TTS_ENABLED", "true").lower() == "true",
        voice_profile=os.getenv("VOICE_PROFILE", "cortana"),
    )


# Default configuration
DEFAULT_CONFIG = load_config()
