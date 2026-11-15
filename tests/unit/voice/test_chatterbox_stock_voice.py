"""Tests that `ChatterboxTTS` selects the engine's built-in voice for a profile with no clip.

Chatterbox only loads a reference clip when `audio_prompt_path` is truthy; otherwise it uses
the built-in conditionals shipped with the pretrained weights. A profile without a clip must
therefore pass `None`, not the string "None", which the engine would try to open as a file.
"""

import sys
import types

import pytest
import torch


class _RecordingModel:
    """Stands in for the pretrained Chatterbox model and records each `generate` call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return torch.zeros(1, 8)


@pytest.fixture
def fake_chatterbox(monkeypatch):
    """Install a fake `chatterbox.tts` module whose `from_pretrained` returns a recorder."""
    model = _RecordingModel()

    class _FakeChatterboxTTS:
        @classmethod
        def from_pretrained(cls, device):
            return model

    package = types.ModuleType("chatterbox")
    module = types.ModuleType("chatterbox.tts")
    module.ChatterboxTTS = _FakeChatterboxTTS
    package.tts = module
    monkeypatch.setitem(sys.modules, "chatterbox", package)
    monkeypatch.setitem(sys.modules, "chatterbox.tts", module)
    return model


def _profile(reference_audio_path):
    from src.multimodal.voice_profile import VoiceProfile

    return VoiceProfile(
        name="sample",
        description="",
        speaker_id=0,
        weights_path=None,
        reference_clips=(),
        tts_engine="chatterbox",
        reference_audio_path=reference_audio_path,
    )


def test_profile_without_clip_passes_none(fake_chatterbox):
    from src.multimodal.tts import ChatterboxTTS

    tts = ChatterboxTTS(_profile(None), device="cpu")
    tts.generate("Hello.")

    assert tts.reference_audio_path is None
    assert fake_chatterbox.calls[-1]["audio_prompt_path"] is None


def test_profile_with_clip_passes_path_string(fake_chatterbox, tmp_path):
    from src.multimodal.tts import ChatterboxTTS

    clip = tmp_path / "clip.wav"
    tts = ChatterboxTTS(_profile(clip), device="cpu")
    tts.generate("Hello.")

    assert fake_chatterbox.calls[-1]["audio_prompt_path"] == str(clip)
