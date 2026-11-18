"""Voice settings and mounts of the mist-backend service in docker-compose.yml.

Hermetic: parses the committed compose file only. `TTS_EXAGGERATION`, `TTS_TEMPERATURE` and
`TTS_CFG_WEIGHT` were passed to the container although no Python code reads them
(`grep -rnE 'TTS_(EXAGGERATION|TEMPERATURE|CFG_WEIGHT)' --include='*.py' .` matches only this
file), so they looked like working knobs and were not. `TTS_ENGINE` is the same kind of knob:
`backend/config.py` reads it into `VoiceConfig.tts_engine`, and nothing reads that field
(`git grep -nw tts_engine -- backend` matches only its declaration and that read), because the
realtime chains choose the engines. `TTS_ENABLED` is still the switch for spoken replies
(`backend/config.py`), so it stays, defaulting on.

The production service mounts no voice profile directory from the checkout. Her voice
identities come from a directory outside it, `VOICE_IDENTITY_DIR`, so no reference clip can be
committed by accident; the engine assets built from them go to a named cache volume.
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"

UNREAD_TTS_KEYS = ("TTS_EXAGGERATION", "TTS_TEMPERATURE", "TTS_CFG_WEIGHT", "TTS_ENGINE")
VOICES_MOUNT = "${VOICE_IDENTITY_DIR:-../mist-voice/voices}:/voices:ro"
VOICE_CACHE_VOLUME = "mist-voice-cache"
# A short-syntax volume: source (a `${...}` expression may hold colons), target, optional mode.
SHORT_VOLUME = re.compile(
    r"(?P<source>(?:\$\{[^}]*\}|[^:$])+):(?P<target>[^:]+)(?::(?P<mode>\w+))?"
)
DEFAULT_INTERPOLATION = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-([^}]*)\}")


def _compose() -> dict:
    with COMPOSE_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _backend() -> dict:
    return _compose()["services"]["mist-backend"]


def _environment() -> dict[str, str]:
    pairs = (str(entry).partition("=") for entry in _backend().get("environment", []))
    return {key: value for key, _sep, value in pairs}


def _mounts(service: dict) -> list[tuple[str, str, str]]:
    """(source, target, mode) for each short-syntax volume; mode is "rw" when none is given."""
    mounts = []
    for spec in service.get("volumes", []):
        match = SHORT_VOLUME.fullmatch(str(spec))
        assert match, f"not a short-syntax volume: {spec!r}"
        mounts.append((match["source"], match["target"], match["mode"] or "rw"))
    return mounts


def _resolved(source: str) -> str:
    """The source with every `${NAME:-default}` replaced by its default."""
    return DEFAULT_INTERPOLATION.sub(lambda match: match.group(1), source)


def test_backend_passes_no_unread_tts_settings() -> None:
    passed = set(UNREAD_TTS_KEYS) & set(_environment())
    assert not passed, f"mist-backend passes unread TTS settings: {sorted(passed)}"


def test_spoken_replies_default_on() -> None:
    assert _environment()["TTS_ENABLED"] == "${TTS_ENABLED:-true}"


def test_backend_mounts_no_voice_profile_directory() -> None:
    volumes = [str(volume) for volume in _backend().get("volumes", [])]
    mounted = [volume for volume in volumes if "voice_profile" in volume]
    assert not mounted, f"mist-backend mounts a voice profile directory: {mounted}"


def test_backend_mounts_the_models_as_mist_llm_does() -> None:
    services = _compose()["services"]
    llm_models = [spec for spec in services["mist-llm"]["volumes"] if spec.endswith(":/models:ro")]
    backend_models = [spec for spec in _backend()["volumes"] if ":/models" in spec]

    assert len(llm_models) == 1
    assert backend_models == llm_models


def test_backend_mounts_the_voice_identities_read_only() -> None:
    assert VOICES_MOUNT in _backend()["volumes"]


def test_voices_default_resolves_outside_the_checkout() -> None:
    (source,) = (source for source, target, _mode in _mounts(_backend()) if target == "/voices")
    default = _resolved(source)

    assert default.startswith("../")
    assert not (REPO_ROOT / default).resolve().is_relative_to(REPO_ROOT)


def test_no_voices_source_lies_in_the_repository() -> None:
    offenders = []
    for name, service in _compose()["services"].items():
        for source, target, _mode in _mounts(service):
            if target != "/voices":
                continue
            default = _resolved(source)
            inside = (REPO_ROOT / default).resolve().is_relative_to(REPO_ROOT)
            if default.startswith("./") or inside:
                offenders.append((name, source))
    assert not offenders


def test_backend_mounts_the_voice_cache_volume() -> None:
    compose = _compose()
    mounts = _mounts(compose["services"]["mist-backend"])

    assert (VOICE_CACHE_VOLUME, "/voice-cache", "rw") in mounts
    assert compose["volumes"][VOICE_CACHE_VOLUME] == {"name": VOICE_CACHE_VOLUME}
