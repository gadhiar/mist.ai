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

The bench's `pocket-clone` adapter builds the model config the runtime adapter builds, for a clip
and for an identity's trained weights and tokenizer, so a benched identity loads the files the
runtime would load.
"""

import copy
import hashlib
import json
import re
import wave
from pathlib import Path

import numpy as np
import pytest

from backend.realtime import voice_assets
from backend.realtime.engines import nemotron_stt, parakeet_stt, pocket_tts
from scripts.model_bench.speech.probes import stt_engines, tts_engines

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


# ---------------------------------------------------------------------------
# The bench's `pocket-clone` config for an identity is the runtime adapter's (plan v5, M17)
# ---------------------------------------------------------------------------

VOICE_ID = "parity-voice"
STOCK_TOKENIZER = b"stock tokenizer"
# pocket-tts 4.0.1's installed `english.yaml`, reduced to the keys `local_config` reads.
INSTALLED = {
    "weights_path": pocket_tts.CLONE_MODEL_URI,
    "weights_path_without_voice_cloning": pocket_tts.MODEL_URI,
    "flow_lm": {"lookup_table": {"tokenizer_path": pocket_tts.TOKENIZER_URI}},
    "mimi": {"sample_rate": 24000},
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stage_models(root: Path) -> Path:
    """`<root>/speech/<org>/<repo>/...`, the runtime's layout; the bench mounts `<root>/speech`."""
    for uri in pocket_tts.CONFIG_URIS:
        org, repo, *rest = uri.removeprefix("hf://").rsplit("@", 1)[0].split("/")
        path = root.joinpath("speech", org, repo, *rest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(STOCK_TOKENIZER if uri == pocket_tts.TOKENIZER_URI else b"stock weights")
    return root


def _identity(voice_root: Path, kind: str) -> Path:
    """An identity as `student export` writes it: stock, trained, or trained with a tokenizer."""
    identity = voice_root / VOICE_ID
    identity.mkdir(parents=True)
    rate = 24000
    frames = (np.sin(2 * np.pi * 220.0 * np.arange(4 * rate) / rate) * 8000).astype("<i2")
    with wave.open(str(identity / "neutral.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(frames.tobytes())
    (identity / "neutral.txt").write_text("a parity line", encoding="utf-8")
    clip = {
        "state": "neutral",
        "audio": "neutral.wav",
        "transcript": "neutral.txt",
        "audio_sha256": _sha((identity / "neutral.wav").read_bytes()),
        "transcript_sha256": _sha(b"a parity line"),
    }
    manifest: dict = {"schema": 1, "voice_id": VOICE_ID, "status": "frozen", "clips": [clip]}
    if kind != "stock":
        (identity / "model.safetensors").write_bytes(b"trained weights")
        model = {
            "engine": pocket_tts.ENGINE_ID,
            "base": pocket_tts.MODEL_PIN,
            "weights": "model.safetensors",
            "sha256": _sha(b"trained weights"),
            "tokenizer_sha256": _sha(STOCK_TOKENIZER),
        }
        if kind == "trained-with-tokenizer":
            (identity / "tokenizer.json").write_bytes(b"trained tokenizer")
            model.update(tokenizer="tokenizer.json", tokenizer_sha256=_sha(b"trained tokenizer"))
        manifest.update(schema=2, model=model)
    (identity / "voice.json").write_text(json.dumps(manifest), encoding="utf-8")
    return identity


@pytest.mark.parametrize("kind", ["stock", "trained", "trained-with-tokenizer"])
def test_the_bench_config_for_an_identity_is_the_runtime_adapter_s(tmp_path, kind):
    models_root = _stage_models(tmp_path / "models")
    voice_root = tmp_path / "voices"
    _identity(voice_root, kind)
    trained = voice_assets.load_identity(voice_root, VOICE_ID).model
    engine = tts_engines.PocketCloneEngine(
        "pocket-clone",
        _arm("tts", "pocket-clone"),
        models_dir=models_root / "speech",
        voice_root=voice_root,
        voice_id=VOICE_ID,
    )

    bench = engine._local_config(copy.deepcopy(INSTALLED))

    runtime = pocket_tts.local_config(
        copy.deepcopy(INSTALLED),
        models_root,
        None if trained is None else trained.weights_path,
        None if trained is None else trained.tokenizer_path,
    )
    assert bench == runtime
    if trained is not None:
        assert bench["weights_path"] == str(trained.weights_path)


def test_the_bench_config_for_a_clip_is_the_runtime_adapter_s_stock_config(tmp_path):
    models_root = _stage_models(tmp_path / "models")
    engine = tts_engines.PocketCloneEngine(
        "pocket-clone", _arm("tts", "pocket-clone"), models_dir=models_root / "speech"
    )

    assert engine._local_config(copy.deepcopy(INSTALLED)) == pocket_tts.local_config(
        copy.deepcopy(INSTALLED), models_root
    )
