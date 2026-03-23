"""Tests for VoiceProfile dataclass and VoiceProfileRegistry."""

import wave
from pathlib import Path

import pytest
import yaml


class TestReferenceClip:
    """Tests for the ReferenceClip frozen dataclass."""

    def test_creation(self, tmp_path):
        from src.multimodal.voice_profile import ReferenceClip

        clip = ReferenceClip(
            audio_path=tmp_path / "clip.wav",
            transcription="Hello world",
        )
        assert clip.audio_path == tmp_path / "clip.wav"
        assert clip.transcription == "Hello world"

    def test_frozen_immutability(self, tmp_path):
        from src.multimodal.voice_profile import ReferenceClip

        clip = ReferenceClip(
            audio_path=tmp_path / "clip.wav",
            transcription="Hello world",
        )
        with pytest.raises(AttributeError):
            clip.transcription = "Changed"


class TestVoiceProfile:
    """Tests for the VoiceProfile frozen dataclass."""

    def test_creation_with_defaults(self, tmp_path):
        from src.multimodal.voice_profile import ReferenceClip, VoiceProfile

        profile = VoiceProfile(
            name="test",
            description="A test profile",
            speaker_id=0,
            weights_path=tmp_path / "weights.pt",
            reference_clips=(ReferenceClip(audio_path=tmp_path / "c.wav", transcription="hi"),),
        )
        assert profile.sample_rate == 24_000
        assert profile.temperature == 0.8
        assert profile.topk == 40

    def test_frozen_immutability(self, tmp_path):
        from src.multimodal.voice_profile import VoiceProfile

        profile = VoiceProfile(
            name="test",
            description="desc",
            speaker_id=0,
            weights_path=tmp_path / "w.pt",
            reference_clips=(),
        )
        with pytest.raises(AttributeError):
            profile.name = "changed"

    def test_custom_sampling_params(self, tmp_path):
        from src.multimodal.voice_profile import VoiceProfile

        profile = VoiceProfile(
            name="test",
            description="desc",
            speaker_id=1,
            weights_path=tmp_path / "w.pt",
            reference_clips=(),
            sample_rate=16_000,
            temperature=0.5,
            topk=20,
        )
        assert profile.sample_rate == 16_000
        assert profile.temperature == 0.5
        assert profile.topk == 20
        assert profile.speaker_id == 1


def _write_wav(path: Path, *, num_frames: int = 100, sample_rate: int = 24000) -> None:
    """Write a minimal valid WAV file (mono, 16-bit PCM)."""
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write silence frames
        wf.writeframes(b"\x00\x00" * num_frames)


class TestVoiceProfileRegistry:
    """Tests for VoiceProfileRegistry discovery and loading."""

    @pytest.fixture
    def fake_profile_dir(self, tmp_path):
        """Create a minimal valid profile directory structure.

        Structure:
            tmp_path/
              profiles/
                testvoice/
                  profile.yaml
                  clip_01.wav
              weights/
                testvoice.pt
        """
        profiles_dir = tmp_path / "profiles"
        profile_sub = profiles_dir / "testvoice"
        profile_sub.mkdir(parents=True)

        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        weights_file = weights_dir / "testvoice.pt"
        weights_file.write_bytes(b"fake-weights-data")

        # Write a real (tiny) WAV file
        clip_path = profile_sub / "clip_01.wav"
        _write_wav(clip_path)

        # Write profile.yaml
        profile_data = {
            "name": "testvoice",
            "description": "A test voice profile",
            "speaker_id": 0,
            "weights_path": str(weights_file),
            "reference_clips": [
                {
                    "audio": "clip_01.wav",
                    "transcription": "Hello, this is a test.",
                },
            ],
            "sample_rate": 24000,
            "temperature": 0.8,
            "topk": 40,
        }
        yaml_path = profile_sub / "profile.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(profile_data, f)

        return profiles_dir

    def test_discover_profiles(self, fake_profile_dir):
        from src.multimodal.voice_profile import VoiceProfileRegistry

        registry = VoiceProfileRegistry(
            profiles_dir=fake_profile_dir,
            project_root=fake_profile_dir.parent,
        )
        assert "testvoice" in registry.available()

    def test_get_profile(self, fake_profile_dir):
        from src.multimodal.voice_profile import VoiceProfileRegistry

        registry = VoiceProfileRegistry(
            profiles_dir=fake_profile_dir,
            project_root=fake_profile_dir.parent,
        )
        profile = registry.get("testvoice")
        assert profile.name == "testvoice"
        assert profile.description == "A test voice profile"
        assert profile.speaker_id == 0

    def test_get_missing_profile_raises(self, fake_profile_dir):
        from src.multimodal.voice_profile import VoiceProfileRegistry

        registry = VoiceProfileRegistry(
            profiles_dir=fake_profile_dir,
            project_root=fake_profile_dir.parent,
        )
        with pytest.raises(KeyError, match="nonexistent"):
            registry.get("nonexistent")

    def test_get_active_from_env(self, fake_profile_dir, monkeypatch):
        from src.multimodal.voice_profile import VoiceProfileRegistry

        monkeypatch.setenv("VOICE_PROFILE", "testvoice")
        registry = VoiceProfileRegistry(
            profiles_dir=fake_profile_dir,
            project_root=fake_profile_dir.parent,
        )
        profile = registry.get_active()
        assert profile.name == "testvoice"

    def test_get_active_default(self, fake_profile_dir, monkeypatch):
        from src.multimodal.voice_profile import VoiceProfileRegistry

        monkeypatch.delenv("VOICE_PROFILE", raising=False)
        registry = VoiceProfileRegistry(
            profiles_dir=fake_profile_dir,
            project_root=fake_profile_dir.parent,
        )
        # Default is 'cortana', but only 'testvoice' exists -> KeyError
        with pytest.raises(KeyError, match="cortana"):
            registry.get_active()

    def test_skip_null_weights(self, tmp_path):
        """Profile with weights_path: null should be skipped (not in available())."""
        from src.multimodal.voice_profile import VoiceProfileRegistry

        profiles_dir = tmp_path / "profiles"
        profile_sub = profiles_dir / "nullvoice"
        profile_sub.mkdir(parents=True)

        profile_data = {
            "name": "nullvoice",
            "description": "Placeholder profile",
            "weights_path": None,
        }
        with open(profile_sub / "profile.yaml", "w") as f:
            yaml.dump(profile_data, f)

        registry = VoiceProfileRegistry(
            profiles_dir=profiles_dir,
            project_root=tmp_path,
        )
        assert "nullvoice" not in registry.available()
        assert registry.available() == []

    def test_missing_weights_file_raises(self, tmp_path):
        from src.multimodal.voice_profile import VoiceProfileRegistry

        profiles_dir = tmp_path / "profiles"
        profile_sub = profiles_dir / "badvoice"
        profile_sub.mkdir(parents=True)

        profile_data = {
            "name": "badvoice",
            "weights_path": str(tmp_path / "nonexistent_weights.pt"),
            "reference_clips": [],
        }
        with open(profile_sub / "profile.yaml", "w") as f:
            yaml.dump(profile_data, f)

        with pytest.raises(FileNotFoundError, match="weights file not found"):
            VoiceProfileRegistry(
                profiles_dir=profiles_dir,
                project_root=tmp_path,
            )

    def test_missing_reference_clip_raises(self, tmp_path):
        from src.multimodal.voice_profile import VoiceProfileRegistry

        profiles_dir = tmp_path / "profiles"
        profile_sub = profiles_dir / "badclip"
        profile_sub.mkdir(parents=True)

        weights_file = tmp_path / "weights.pt"
        weights_file.write_bytes(b"fake")

        profile_data = {
            "name": "badclip",
            "weights_path": str(weights_file),
            "reference_clips": [
                {"audio": "missing.wav", "transcription": "Hello"},
            ],
        }
        with open(profile_sub / "profile.yaml", "w") as f:
            yaml.dump(profile_data, f)

        with pytest.raises(FileNotFoundError, match="reference clip not found"):
            VoiceProfileRegistry(
                profiles_dir=profiles_dir,
                project_root=tmp_path,
            )

    def test_unknown_yaml_fields_ignored(self, fake_profile_dir):
        """Unknown YAML fields should be silently ignored for forward compat."""
        from src.multimodal.voice_profile import VoiceProfileRegistry

        # Add an unknown field to the existing profile.yaml
        yaml_path = fake_profile_dir / "testvoice" / "profile.yaml"
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        data["future_field"] = "some_value"
        data["another_unknown"] = 42
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        # Should load without errors
        registry = VoiceProfileRegistry(
            profiles_dir=fake_profile_dir,
            project_root=fake_profile_dir.parent,
        )
        profile = registry.get("testvoice")
        assert profile.name == "testvoice"

    def test_relative_weights_path_resolved(self, tmp_path):
        """Relative weights_path should be resolved against project_root."""
        from src.multimodal.voice_profile import VoiceProfileRegistry

        profiles_dir = tmp_path / "profiles"
        profile_sub = profiles_dir / "relvoice"
        profile_sub.mkdir(parents=True)

        # Create weights at project_root / relative_path
        weights_dir = tmp_path / "models"
        weights_dir.mkdir()
        weights_file = weights_dir / "voice.pt"
        weights_file.write_bytes(b"fake-weights")

        clip_path = profile_sub / "clip.wav"
        _write_wav(clip_path)

        profile_data = {
            "name": "relvoice",
            "description": "Relative weights test",
            "speaker_id": 0,
            "weights_path": "models/voice.pt",  # Relative path
            "reference_clips": [
                {"audio": "clip.wav", "transcription": "Test"},
            ],
        }
        with open(profile_sub / "profile.yaml", "w") as f:
            yaml.dump(profile_data, f)

        registry = VoiceProfileRegistry(
            profiles_dir=profiles_dir,
            project_root=tmp_path,
        )
        profile = registry.get("relvoice")
        assert profile.weights_path.is_absolute()
        assert profile.weights_path == weights_file.resolve()

    def test_reference_clip_paths_absolute(self, fake_profile_dir):
        from src.multimodal.voice_profile import VoiceProfileRegistry

        registry = VoiceProfileRegistry(
            profiles_dir=fake_profile_dir,
            project_root=fake_profile_dir.parent,
        )
        profile = registry.get("testvoice")
        for clip in profile.reference_clips:
            assert clip.audio_path.is_absolute()

    def test_reference_clips_is_tuple(self, fake_profile_dir):
        """reference_clips should be a tuple (immutable), not a list."""
        from src.multimodal.voice_profile import VoiceProfileRegistry

        registry = VoiceProfileRegistry(
            profiles_dir=fake_profile_dir,
            project_root=fake_profile_dir.parent,
        )
        profile = registry.get("testvoice")
        assert isinstance(profile.reference_clips, tuple)

    def test_corrupted_wav_raises(self, tmp_path):
        """A corrupted WAV file should raise RuntimeError at discovery time."""
        from src.multimodal.voice_profile import VoiceProfileRegistry

        profiles_dir = tmp_path / "profiles"
        profile_sub = profiles_dir / "corrupt"
        profile_sub.mkdir(parents=True)

        weights_file = tmp_path / "weights.pt"
        weights_file.write_bytes(b"fake-weights")

        # Write garbage data that is not a valid WAV
        bad_wav = profile_sub / "bad.wav"
        bad_wav.write_bytes(b"NOT A WAV FILE AT ALL")

        profile_data = {
            "name": "corrupt",
            "weights_path": str(weights_file),
            "reference_clips": [
                {"audio": "bad.wav", "transcription": "Hello"},
            ],
        }
        with open(profile_sub / "profile.yaml", "w") as f:
            yaml.dump(profile_data, f)

        with pytest.raises(RuntimeError, match="reference clip unreadable"):
            VoiceProfileRegistry(
                profiles_dir=profiles_dir,
                project_root=tmp_path,
            )
