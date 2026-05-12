"""Configuration for Voice AI Backend."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class SystemStatusConfig(BaseModel):
    """Configuration for the ADR-017 system_status periodic emit.

    Drives backend/server.py::system_status_loop which samples CPU /
    RAM / GPU via backend/system_metrics.py and broadcasts the payload
    over the WS message queue on a fixed cadence.

    - enabled: master switch for the periodic task. Defaults True so
      the FE MetricsPanel populates out of the box.
    - interval_seconds: cadence in seconds between emits. Defaults 5
      per the BE prompt; lower values increase WS traffic, higher
      values make the FE bars look stale.
    - gpu_enabled: whether to attempt NVML init. Defaults True;
      production hardware has an NVIDIA GPU. Setting False forces the
      placeholder GPU block (name='none') without touching NVML.
    """

    enabled: bool = True
    interval_seconds: int = 5
    gpu_enabled: bool = True


class VoiceConfig(BaseModel):
    """Configuration for voice AI system."""

    # Model paths
    whisper_model: str = "base"
    # 2026-05-11: default aligned with docker-compose.yml MODEL env var
    # (gemma-4-e4b). Qwen 2.5 7B was the pre-2026-04-22 default; replaced by
    # Gemma 4 E4B Q5_K_M per ADR-008 model-backend-selection.
    llm_model: str = "gemma-4-e4b"
    tts_device: str = "cuda"

    # VAD settings
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_sample_rate: int = 16000

    # TTS settings
    tts_enabled: bool = True  # Set to False to disable TTS (text-only mode)
    tts_engine: str = "chatterbox"  # "chatterbox" (default) or "csm" (legacy)
    voice_profile: str = "cortana"  # Selected via VOICE_PROFILE env var
    use_voice_context: bool = True

    # Server settings
    host: str = "0.0.0.0"  # nosec B104 - intentional bind to all interfaces for dev server
    port: int = 8001  # Changed from 8000 to avoid conflict with HYTE Nexus
    max_connections: int = 10

    # Debug
    debug: bool = False

    # System telemetry (ADR-017 system_status emit)
    system_status: SystemStatusConfig = SystemStatusConfig()


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var; supports 'true' / 'false' (case-insensitive)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


# Load configuration from environment
def load_config() -> VoiceConfig:
    """Load configuration from environment variables."""
    return VoiceConfig(
        tts_enabled=os.getenv("TTS_ENABLED", "true").lower() == "true",
        tts_engine=os.getenv("TTS_ENGINE", "chatterbox"),
        voice_profile=os.getenv("VOICE_PROFILE", "jarvis"),
        system_status=SystemStatusConfig(
            enabled=_bool_env("MIST_SYSTEM_STATUS_ENABLED", True),
            interval_seconds=int(os.getenv("MIST_SYSTEM_STATUS_INTERVAL_SECONDS", "5")),
            gpu_enabled=_bool_env("MIST_SYSTEM_STATUS_GPU_ENABLED", True),
        ),
    )


# Default configuration
DEFAULT_CONFIG = load_config()
