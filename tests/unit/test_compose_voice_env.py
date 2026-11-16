"""Voice settings and mounts of the mist-backend service in docker-compose.yml.

Hermetic: parses the committed compose file only. `TTS_EXAGGERATION`, `TTS_TEMPERATURE` and
`TTS_CFG_WEIGHT` were passed to the container although no Python code reads them
(`grep -rnE 'TTS_(EXAGGERATION|TEMPERATURE|CFG_WEIGHT)' --include='*.py' .` matches only this
file), so they looked like working knobs and were not. The production service mounts no voice
profile directory either, and this keeps it that way: no reference clip enters the container from
the checkout.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"

UNREAD_TTS_KEYS = ("TTS_EXAGGERATION", "TTS_TEMPERATURE", "TTS_CFG_WEIGHT")


def _backend() -> dict:
    with COMPOSE_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["services"]["mist-backend"]


def _env_keys() -> set[str]:
    return {str(entry).partition("=")[0] for entry in _backend().get("environment", [])}


def test_backend_passes_no_unread_tts_settings() -> None:
    passed = set(UNREAD_TTS_KEYS) & _env_keys()
    assert not passed, f"mist-backend passes unread TTS settings: {sorted(passed)}"


def test_backend_mounts_no_voice_profile_directory() -> None:
    volumes = [str(volume) for volume in _backend().get("volumes", [])]
    mounted = [volume for volume in volumes if "voice_profile" in volume]
    assert not mounted, f"mist-backend mounts a voice profile directory: {mounted}"
