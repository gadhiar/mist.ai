"""Each speech engine adapter's defaults equal the bench arm it was measured as.

The bench (`scripts/model_bench/speech/speech_arms.json`) records, per arm, the repository,
revision, device and threads it ran an engine with, and the package versions in its image. An
adapter that drifted from its arm would run an engine nobody measured, so each adapter is pinned
here to its arm's fields, and each package the arm records to an exact pin in
`requirements-voice.txt` or `requirements.txt`. torch is the exception: the arm records the
index's `+cu124` build, which the Dockerfile installs from the PyTorch index, not from a
requirements file. The arm's `verified` flag is not required: the engine was picked by ruling,
and the host measures it (A42).
"""

import json
import re
from pathlib import Path

import pytest

from backend.realtime.engines import pocket_tts

REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS = json.loads(
    (REPO_ROOT / "scripts" / "model_bench" / "speech" / "speech_arms.json").read_text(
        encoding="utf-8"
    )
)
REQUIREMENT_FILES = (REPO_ROOT / "requirements-voice.txt", REPO_ROOT / "requirements.txt")
# Packages the arms record that no requirements file pins, each with its reason above.
UNPINNED = frozenset({"torch"})
PIN = re.compile(r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9._,-]+\])?)==(?P<version>\S+)")

# (section, arm) -> the adapter's module, which carries its pinned defaults.
ADAPTERS = {("tts", "pocket-clone"): pocket_tts}


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins() -> dict[str, str]:
    """Normalised `name[extra]` to version, for every exact pin in the requirements files."""
    pins = {}
    for path in REQUIREMENT_FILES:
        for raw in path.read_text(encoding="utf-8").splitlines():
            match = PIN.fullmatch(raw.split("#", 1)[0].strip())
            if match:
                pins[_normalise(match.group("name"))] = match.group("version")
    return pins


def _arm(section: str, name: str) -> dict:
    return ARMS[section][name]


@pytest.mark.parametrize(("section", "name"), list(ADAPTERS))
class TestAdapterDefaults:
    def test_engine_id_is_the_arm(self, section, name):
        assert name == ADAPTERS[(section, name)].ENGINE_ID

    def test_repository_and_revision(self, section, name):
        model = _arm(section, name)["model"]
        module = ADAPTERS[(section, name)]

        assert model["repository"] == module.REPOSITORY
        assert model["revision"] == module.REVISION
        assert f"{model['repository']}@{model['revision']}" == module.MODEL_PIN

    def test_device_and_threads(self, section, name):
        arm = _arm(section, name)
        module = ADAPTERS[(section, name)]

        assert arm["device"] == module.DEVICE
        assert arm["threads"] == module.THREADS

    def test_every_package_is_pinned_at_the_arms_version(self, section, name):
        pins = _pins()
        packages = {
            _normalise(package): version
            for package, version in _arm(section, name)["packages"].items()
            if _normalise(package) not in UNPINNED
        }

        assert packages
        assert {package: pins.get(package) for package in packages} == packages


def test_pocket_clone_classes_carry_the_arms_pin():
    model = _arm("tts", "pocket-clone")["model"]
    pin = f"{model['repository']}@{model['revision']}"

    assert pocket_tts.PocketCloneSynthesizer.engine_id == "pocket-clone"
    assert pocket_tts.PocketCloneSynthesizer.model_id == pin
    assert pocket_tts.PocketVoiceBuilder.engine_id == "pocket-clone"
    assert pocket_tts.PocketVoiceBuilder.model_pin == pin
