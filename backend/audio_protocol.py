"""Binary audio frame protocol for WebSocket transport.

Defines the MIST binary audio frame format:
  Bytes 0-3:   Magic (0x4D495354 = "MIST")
  Bytes 4-5:   Message type (uint16)
  Bytes 6-7:   Session ID (uint16)
  Bytes 8-11:  Chunk sequence number (uint32)
  Bytes 12-15: Sample rate (uint32)
  Bytes 16+:   Raw PCM16 little-endian payload

Message types:
  0x01 = audio_chunk
  0x02 = audio_complete (empty payload)
  0x03 = interrupt_fade
"""

import struct

import numpy as np

MAGIC = 0x4D495354  # "MIST"
HEADER_FORMAT = "<IHHII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 16 bytes

MSG_AUDIO_CHUNK = 0x01
MSG_AUDIO_COMPLETE = 0x02
MSG_INTERRUPT_FADE = 0x03

MAX_GAIN = 10.0


def build_audio_frame(
    message_type: int,
    session_id: int,
    chunk_seq: int,
    sample_rate: int,
    pcm16_payload: bytes,
) -> bytes:
    """Construct a binary audio frame with 16-byte header + PCM16 payload."""
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        message_type,
        session_id,
        chunk_seq,
        sample_rate,
    )
    return header + pcm16_payload


def rms_normalize(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Normalize audio chunk to target RMS level in dBFS."""
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-8:
        return audio
    target_rms = 10 ** (target_dbfs / 20.0)
    gain = min(target_rms / rms, MAX_GAIN)
    normalized = audio * gain
    return np.clip(normalized, -1.0, 1.0)


def generate_fade_out(audio: np.ndarray, fade_ms: int = 20, sample_rate: int = 24000) -> np.ndarray:
    """Apply a linear fade-out to the end of an audio chunk."""
    fade_samples = int(fade_ms / 1000 * sample_rate)
    fade_samples = min(fade_samples, len(audio))
    result = audio.copy()
    fade_ramp = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    result[-fade_samples:] *= fade_ramp
    return result


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 audio [-1.0, 1.0] to PCM16 little-endian bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    return pcm16.tobytes()
