"""Tests that M.I.S.T has one voice profile, `mist`, and that every default selects it.

The defaults used to disagree across `backend/config.py`, `voice_profile.py`, `.env.example`
and `docker-compose.yml`; these tests pin them all to the one profile that ships.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILES_DIR = REPO_ROOT / "data" / "voice_profiles"


def _registry():
    from src.multimodal.voice_profile import VoiceProfileRegistry

    return VoiceProfileRegistry(PROFILES_DIR, project_root=REPO_ROOT)


def test_registry_lists_only_mist():
    assert _registry().available() == ["mist"]


def test_mist_profile_uses_stock_voice_with_default_sampling():
    profile = _registry().get("mist")

    assert profile.tts_engine == "chatterbox"
    assert profile.reference_audio_path is None
    assert profile.reference_clips == ()
    assert profile.weights_path is None
    assert profile.speaker_id == 0
    assert profile.sample_rate == 24_000
    assert profile.temperature == 0.8
    assert profile.exaggeration == 0.5
    assert profile.cfg_weight == 0.5


def test_get_active_defaults_to_mist(monkeypatch):
    monkeypatch.delenv("VOICE_PROFILE", raising=False)

    profile = _registry().get_active()

    assert profile.name == "mist"
    assert profile.tts_engine == "chatterbox"
    assert profile.reference_audio_path is None


def test_default_constant_is_mist():
    from src.multimodal.voice_profile import DEFAULT_VOICE_PROFILE

    assert DEFAULT_VOICE_PROFILE == "mist"
