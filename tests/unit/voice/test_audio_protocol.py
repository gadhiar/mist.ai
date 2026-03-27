"""Tests for the binary audio frame protocol."""

import struct

import numpy as np


class TestBuildAudioFrame:
    """Tests for binary frame construction."""

    def test_header_magic_bytes(self):
        from backend.audio_protocol import build_audio_frame

        frame = build_audio_frame(
            message_type=0x01,
            session_id=0,
            chunk_seq=1,
            sample_rate=24000,
            pcm16_payload=b"\x00\x00",
        )
        magic = struct.unpack_from("<I", frame, 0)[0]
        assert magic == 0x4D495354

    def test_header_fields(self):
        from backend.audio_protocol import build_audio_frame

        frame = build_audio_frame(
            message_type=0x02, session_id=5, chunk_seq=42, sample_rate=24000, pcm16_payload=b""
        )
        magic, msg_type, session_id, chunk_seq, sample_rate = struct.unpack_from("<IHHII", frame, 0)
        assert magic == 0x4D495354
        assert msg_type == 0x02
        assert session_id == 5
        assert chunk_seq == 42
        assert sample_rate == 24000

    def test_payload_follows_header(self):
        from backend.audio_protocol import build_audio_frame

        payload = b"\x01\x02\x03\x04"
        frame = build_audio_frame(
            message_type=0x01, session_id=0, chunk_seq=0, sample_rate=24000, pcm16_payload=payload
        )
        assert frame[16:] == payload

    def test_header_size_is_16_bytes(self):
        from backend.audio_protocol import build_audio_frame

        frame = build_audio_frame(
            message_type=0x01, session_id=0, chunk_seq=0, sample_rate=24000, pcm16_payload=b""
        )
        assert len(frame) == 16

    def test_audio_complete_has_empty_payload(self):
        from backend.audio_protocol import build_audio_frame

        frame = build_audio_frame(
            message_type=0x02, session_id=0, chunk_seq=10, sample_rate=24000, pcm16_payload=b""
        )
        assert len(frame) == 16


class TestRmsNormalize:
    """Tests for RMS volume normalization."""

    def test_normalizes_loud_chunk(self):
        from backend.audio_protocol import rms_normalize

        loud = np.full(2400, 0.5, dtype=np.float32)
        result = rms_normalize(loud, target_dbfs=-20.0)
        result_rms = np.sqrt(np.mean(result**2))
        target_rms = 10 ** (-20.0 / 20.0)
        assert abs(result_rms - target_rms) < 0.01

    def test_normalizes_quiet_chunk(self):
        from backend.audio_protocol import rms_normalize

        quiet = np.full(2400, 0.01, dtype=np.float32)
        result = rms_normalize(quiet, target_dbfs=-20.0)
        result_rms = np.sqrt(np.mean(result**2))
        target_rms = 10 ** (-20.0 / 20.0)
        assert abs(result_rms - target_rms) < 0.01

    def test_clamps_gain_on_near_silence(self):
        from backend.audio_protocol import rms_normalize

        silence = np.full(2400, 0.0001, dtype=np.float32)
        result = rms_normalize(silence, target_dbfs=-20.0)
        assert np.max(np.abs(result)) < 0.01

    def test_clips_output_to_valid_range(self):
        from backend.audio_protocol import rms_normalize

        loud = np.array([0.9, -0.95, 0.85, -0.88] * 600, dtype=np.float32)
        result = rms_normalize(loud, target_dbfs=-6.0)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_returns_zeros_for_silent_chunk(self):
        from backend.audio_protocol import rms_normalize

        silent = np.zeros(2400, dtype=np.float32)
        result = rms_normalize(silent, target_dbfs=-20.0)
        assert np.all(result == 0.0)


class TestGenerateFadeOut:
    """Tests for interrupt fade-out generation."""

    def test_output_length_matches_duration(self):
        from backend.audio_protocol import generate_fade_out

        audio = np.ones(4800, dtype=np.float32)
        result = generate_fade_out(audio, fade_ms=20, sample_rate=24000)
        assert len(result) == 4800

    def test_first_sample_unchanged(self):
        from backend.audio_protocol import generate_fade_out

        audio = np.ones(4800, dtype=np.float32)
        result = generate_fade_out(audio, fade_ms=20, sample_rate=24000)
        assert result[0] == 1.0

    def test_last_sample_is_zero(self):
        from backend.audio_protocol import generate_fade_out

        audio = np.ones(4800, dtype=np.float32)
        result = generate_fade_out(audio, fade_ms=20, sample_rate=24000)
        assert abs(result[-1]) < 0.01

    def test_fade_region_is_monotonically_decreasing(self):
        from backend.audio_protocol import generate_fade_out

        audio = np.ones(4800, dtype=np.float32)
        result = generate_fade_out(audio, fade_ms=20, sample_rate=24000)
        fade_samples = int(20 / 1000 * 24000)
        fade_region = result[-fade_samples:]
        for i in range(1, len(fade_region)):
            assert fade_region[i] <= fade_region[i - 1]

    def test_short_audio_fades_entirely(self):
        from backend.audio_protocol import generate_fade_out

        audio = np.ones(100, dtype=np.float32)
        result = generate_fade_out(audio, fade_ms=20, sample_rate=24000)
        assert len(result) == 100
        assert abs(result[-1]) < 0.02


class TestFloat32ToPcm16:
    """Tests for float32-to-PCM16 byte conversion."""

    def test_converts_zero(self):
        from backend.audio_protocol import float32_to_pcm16

        audio = np.array([0.0], dtype=np.float32)
        result = float32_to_pcm16(audio)
        assert result == b"\x00\x00"

    def test_converts_positive_max(self):
        from backend.audio_protocol import float32_to_pcm16

        audio = np.array([1.0], dtype=np.float32)
        result = float32_to_pcm16(audio)
        value = struct.unpack("<h", result)[0]
        assert value == 32767

    def test_converts_negative_max(self):
        from backend.audio_protocol import float32_to_pcm16

        audio = np.array([-1.0], dtype=np.float32)
        result = float32_to_pcm16(audio)
        value = struct.unpack("<h", result)[0]
        assert value == -32767

    def test_clips_out_of_range(self):
        from backend.audio_protocol import float32_to_pcm16

        audio = np.array([1.5, -1.5], dtype=np.float32)
        result = float32_to_pcm16(audio)
        values = struct.unpack("<hh", result)
        assert values == (32767, -32767)

    def test_output_length(self):
        from backend.audio_protocol import float32_to_pcm16

        audio = np.ones(100, dtype=np.float32)
        result = float32_to_pcm16(audio)
        assert len(result) == 200
