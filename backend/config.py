"""
Configuration for Voice AI Backend
"""
from pydantic import BaseModel
from typing import Optional


class VoiceConfig(BaseModel):
    """Configuration for voice AI system"""

    # Model paths
    whisper_model: str = "base"
    llm_model: str = "qwen2.5:7b-instruct"
    tts_device: str = "cuda"

    # VAD settings
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_sample_rate: int = 16000

    # TTS settings
    tts_temperature: float = 0.8
    tts_topk: int = 50
    use_voice_context: bool = True

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8001  # Changed from 8000 to avoid conflict with HYTE Nexus
    max_connections: int = 10

    # Debug
    debug: bool = False


# Default configuration
DEFAULT_CONFIG = VoiceConfig()
