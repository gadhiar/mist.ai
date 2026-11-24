"""Each speech engine adapter's defaults equal the bench arm it was measured as.

The bench (`scripts/model_bench/speech/speech_arms.json`) records, per arm, the model it ran an
engine with (a repository and revision, or an onnx-asr id), the device, threads or streaming
chunk size, and the package versions in its image. An
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

from backend.realtime.engines import nemotron_stt, parakeet_stt, pocket_tts
from scripts.model_bench.speech.probes import stt_engines

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
ADAPTERS = {
    ("tts", "pocket-clone"): pocket_tts,
    ("stt", "nemotron-160"): nemotron_stt,
    ("stt", "parakeet-cpu"): parakeet_stt,
}
# The arms that record a revision; `nemotron-160` and `parakeet-cpu` record none.
REVISIONED = [key for key in ADAPTERS if "revision" in ARMS[key[0]][key[1]]["model"]]
# `nemotron-160`'s arm records no revision and step 0's result none either, so host step H13
# recorded the `.nemo` file's size and sha256, which the adapter pins.
NEMOTRON_FILE_BYTES = 2473041920
NEMOTRON_FILE_SHA256 = "283638054c44f6794e74fe9af9048d78a6d9d6c058c12131856c7859a62ac9cd"


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

    def test_device(self, section, name):
        assert _arm(section, name)["device"] == ADAPTERS[(section, name)].DEVICE

    def test_threads_where_the_arm_records_them(self, section, name):
        arm = _arm(section, name)
        module = ADAPTERS[(section, name)]

        if "threads" in arm:
            assert arm["threads"] == module.THREADS
        else:
            assert not hasattr(module, "THREADS")

    def test_every_package_is_pinned_at_the_arms_version(self, section, name):
        pins = _pins()
        packages = {
            _normalise(package): version
            for package, version in _arm(section, name)["packages"].items()
            if _normalise(package) not in UNPINNED
        }

        assert packages
        assert {package: pins.get(package) for package in packages} == packages


@pytest.mark.parametrize(("section", "name"), REVISIONED)
def test_repository_and_revision(section, name):
    model = _arm(section, name)["model"]
    module = ADAPTERS[(section, name)]

    assert model["repository"] == module.REPOSITORY
    assert model["revision"] == module.REVISION
    assert f"{model['repository']}@{model['revision']}" == module.MODEL_PIN


def test_the_package_check_covers_every_adapter():
    assert set(ADAPTERS) == {
        ("tts", "pocket-clone"),
        ("stt", "nemotron-160"),
        ("stt", "parakeet-cpu"),
    }
    assert [("tts", "pocket-clone")] == REVISIONED


def test_nemotron_160_pins_the_arms_repository_and_chunk_and_the_recorded_file():
    arm = _arm("stt", "nemotron-160")
    pin = f"{arm['model']['repository']}@sha256:{NEMOTRON_FILE_SHA256}"

    assert arm["model"]["repository"] == nemotron_stt.REPOSITORY
    assert arm["streaming"] is True
    assert arm["chunk_ms"] == nemotron_stt.CHUNK_MS
    assert (nemotron_stt.MODEL_BYTES, nemotron_stt.MODEL_SHA256) == (
        NEMOTRON_FILE_BYTES,
        NEMOTRON_FILE_SHA256,
    )
    assert pin == nemotron_stt.MODEL_ID
    assert nemotron_stt.NemotronStreamingTranscriber.engine_id == "nemotron-160"
    assert nemotron_stt.NemotronStreamingTranscriber.model_id == pin


def test_parakeet_cpu_pins_the_arms_onnx_asr_id():
    arm = _arm("stt", "parakeet-cpu")

    assert arm["model"]["onnx_asr_id"] == parakeet_stt.ONNX_ASR_ID
    assert arm["streaming"] is False
    assert parakeet_stt.ParakeetTranscriber.engine_id == "parakeet-cpu"
    assert parakeet_stt.ParakeetTranscriber.model_id == arm["model"]["onnx_asr_id"]


@pytest.mark.parametrize("module", [nemotron_stt, parakeet_stt], ids=lambda m: m.ENGINE_ID)
def test_stt_weights_are_where_the_bench_probe_reads_them(module):
    # The probe reads `<weights_dir>/<path>`, and the bench mounts `<models_dir>/speech/` there.
    assert stt_engines.WEIGHT_PATHS[module.ENGINE_ID] == module.WEIGHT_PATH


def test_pocket_clone_classes_carry_the_arms_pin():
    model = _arm("tts", "pocket-clone")["model"]
    pin = f"{model['repository']}@{model['revision']}"

    assert pocket_tts.PocketCloneSynthesizer.engine_id == "pocket-clone"
    assert pin == pocket_tts.MODEL_PIN
    assert pocket_tts.PocketVoiceBuilder.engine_id == "pocket-clone"
    assert pin == pocket_tts.MODEL_PIN
