"""VoiceProfile abstraction for MIST TTS voice identity management."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ReferenceClip:
    """A reference audio clip used for CSM voice anchoring at inference time."""

    audio_path: Path
    transcription: str


@dataclass(frozen=True)
class ProfileLineage:
    """Records which parent profile contributed to a derived/blended profile."""

    profile: str
    weight: float


@dataclass(frozen=True)
class VoiceProfile:
    """Immutable voice identity: weights, reference clips, and sampling config.

    Supports two TTS engines:
    - CSM: requires fine-tuned weights_path + reference_clips for context
    - Chatterbox: requires reference_audio_path for zero-shot voice cloning
    """

    name: str
    description: str
    speaker_id: int
    weights_path: Path | None
    reference_clips: tuple[ReferenceClip, ...]
    sample_rate: int = 24_000
    temperature: float = 0.8
    topk: int = 40
    derived_from: tuple[ProfileLineage, ...] | None = None
    # Chatterbox-specific fields
    tts_engine: str = "csm"  # "csm" or "chatterbox"
    reference_audio_path: Path | None = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5


class VoiceProfileRegistry:
    """Discover and load VoiceProfile instances from profile directories.

    Scans a profiles directory for subdirectories containing profile.yaml files,
    validates them, and makes them available by name.

    Args:
        profiles_dir: Path to directory containing profile subdirectories.
        project_root: Project root for resolving relative weights_path.
            Defaults to 3 levels up from this file.
    """

    def __init__(self, profiles_dir: Path, project_root: Path | None = None):
        self._profiles_dir = profiles_dir
        self._project_root = project_root or Path(__file__).parent.parent.parent
        self._profiles: dict[str, VoiceProfile] = {}
        self._discover()

    def _discover(self) -> None:
        if not self._profiles_dir.is_dir():
            return
        for profile_dir in sorted(self._profiles_dir.iterdir()):
            yaml_path = profile_dir / "profile.yaml"
            if profile_dir.is_dir() and yaml_path.exists():
                profile = self._load(yaml_path, profile_dir)
                if profile is not None:
                    self._profiles[profile.name] = profile

    def _load(self, yaml_path: Path, profile_dir: Path) -> VoiceProfile | None:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        tts_engine = data.get("tts_engine", "csm")

        # Resolve weights_path (required for CSM, optional for Chatterbox)
        raw_weights = data.get("weights_path")
        weights_path = None
        if raw_weights is not None:
            weights_path = Path(raw_weights)
            if not weights_path.is_absolute():
                weights_path = self._project_root / weights_path
            weights_path = weights_path.resolve()
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Voice profile '{data.get('name', profile_dir.name)}': "
                    f"weights file not found: {weights_path}"
                )
        elif tts_engine == "csm":
            return None  # CSM requires weights

        # Resolve reference_audio_path (for Chatterbox zero-shot cloning)
        reference_audio_path = None
        raw_ref_audio = data.get("reference_audio_path")
        if raw_ref_audio is not None:
            reference_audio_path = profile_dir / raw_ref_audio
            reference_audio_path = reference_audio_path.resolve()
            if not reference_audio_path.exists():
                raise FileNotFoundError(
                    f"Voice profile '{data.get('name', profile_dir.name)}': "
                    f"reference audio not found: {reference_audio_path}"
                )

        clips = []
        for clip_data in data.get("reference_clips", []):
            audio_path = profile_dir / clip_data["audio"]
            audio_path = audio_path.resolve()
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Voice profile '{data.get('name', profile_dir.name)}': "
                    f"reference clip not found: {audio_path}"
                )
            # Validate WAV is readable (fail-fast on corrupted files)
            try:
                import torchaudio

                torchaudio.load(str(audio_path))
            except Exception as e:
                raise RuntimeError(
                    f"Voice profile '{data.get('name', profile_dir.name)}': "
                    f"reference clip unreadable: {audio_path}: {e}"
                ) from e
            clips.append(
                ReferenceClip(
                    audio_path=audio_path,
                    transcription=clip_data["transcription"],
                )
            )

        lineage = None
        raw_lineage = data.get("derived_from")
        if raw_lineage:
            lineage = tuple(
                ProfileLineage(profile=entry["profile"], weight=entry["weight"])
                for entry in raw_lineage
            )

        return VoiceProfile(
            name=data["name"],
            description=data.get("description", ""),
            speaker_id=data.get("speaker_id", 0),
            weights_path=weights_path,
            reference_clips=tuple(clips),
            sample_rate=data.get("sample_rate", 24_000),
            temperature=data.get("temperature", 0.8),
            topk=data.get("topk", 40),
            derived_from=lineage,
            tts_engine=tts_engine,
            reference_audio_path=reference_audio_path,
            exaggeration=data.get("exaggeration", 0.5),
            cfg_weight=data.get("cfg_weight", 0.5),
        )

    def get(self, name: str) -> VoiceProfile:
        """Get a voice profile by name.

        Args:
            name: Profile name as declared in profile.yaml.

        Returns:
            The matching VoiceProfile.

        Raises:
            KeyError: If no profile with the given name exists.
        """
        if name not in self._profiles:
            available = ", ".join(sorted(self._profiles.keys())) or "(none)"
            raise KeyError(f"Voice profile '{name}' not found. Available profiles: {available}")
        return self._profiles[name]

    def get_active(self) -> VoiceProfile:
        """Get the currently active voice profile.

        Reads VOICE_PROFILE env var, defaults to 'cortana'.

        Returns:
            The active VoiceProfile.

        Raises:
            KeyError: If the active profile name is not found.
        """
        name = os.environ.get("VOICE_PROFILE", "jarvis")
        return self.get(name)

    def available(self) -> list[str]:
        """Return sorted list of available profile names."""
        return sorted(self._profiles.keys())
