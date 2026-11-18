"""Tests for the microWakeWord name spotter over a fake LiteRT interpreter and a fake frontend.

The unit tier has no ai-edge-litert, no pymicro-features and no wake model, so every test writes a
manifest and a placeholder model file under a temporary `models_dir` and hands the spotter a fake
library: an interpreter that records each input it is set and answers a scripted uint8 output,
and a frontend that answers scripted features per 160-sample call. That pins what the adapter
owns: the manifest it accepts and refuses, the model checks at load, the device's integer input
formula, the output scale, the zero-filled sliding window, the hit rule and its cooldown, warm
and reset, and the offline scores. Parity with microWakeWord on the real model is T36's.
"""

import hashlib
import json
import math
import pathlib
import random
import sys
import types
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import PlacementConfig
from backend.realtime.engines import kws
from backend.realtime.engines.kws import MicroLibrary, MicroNameSpotter, hit_steps
from backend.realtime.errors import SpeechEngineError
from backend.realtime.speech import NameHit, NameSpotter, NameSpotterStream, SpotterCalibration

MODEL_ID = "mist-mww-test"
MODEL_FILE = "device/mist.tflite"
MODEL_BYTES = b"placeholder streaming model"
SCALE = 0.0390625
CHUNK = 160
FRAME_BYTES = 640
"""A 20 ms uplink frame: 320 samples, two frontend calls."""

Respond = Callable[[int, np.ndarray], int]
Features = Callable[[int, bytes], Sequence[float]]


# -- Fakes ------------------------------------------------------------------------------------


def details(index: int, shape: Sequence[int], dtype: type, zero_point: int = 0) -> dict:
    """Tensor details in LiteRT's form: the shape an int32 array, the dtype a numpy type."""
    return {
        "index": index,
        "shape": np.array(shape, dtype=np.int32),
        "dtype": dtype,
        "quantization": (0.1, zero_point),
    }


@dataclass
class FakeInterpreter:
    """Records each input it is set and answers `respond(call, input)` on each invoke."""

    model: bytes
    threads: int
    input_details: list[dict]
    output_details: list[dict]
    respond: Respond
    inputs: list[np.ndarray] = field(default_factory=list)
    allocated: int = 0
    invokes: int = 0
    _output: int = 0

    def allocate_tensors(self) -> None:
        self.allocated += 1

    def get_input_details(self) -> list[dict]:
        return self.input_details

    def get_output_details(self) -> list[dict]:
        return self.output_details

    def set_tensor(self, tensor_index: int, value: np.ndarray) -> None:
        assert self.allocated
        assert tensor_index == self.input_details[0]["index"]
        self.inputs.append(np.array(value, copy=True))

    def invoke(self) -> None:
        self.invokes += 1
        self._output = self.respond(self.invokes, self.inputs[-1])

    def get_tensor(self, tensor_index: int) -> np.ndarray:
        assert tensor_index == self.output_details[0]["index"]
        return np.array([[self._output]], dtype=np.uint8)


@dataclass
class FakeFrontend:
    """Answers `script(call, pcm)` for each 160-sample call and records the calls."""

    script: Features
    chunks: list[bytes] = field(default_factory=list)

    def features(self, pcm: bytes) -> Sequence[float]:
        assert len(pcm) == 2 * CHUNK
        self.chunks.append(pcm)
        return self.script(len(self.chunks), pcm)


@dataclass
class FakeLibrary:
    """Builds the fakes and keeps every one, so a test can read what reached them."""

    respond: Respond
    features: Features
    input_details: list[dict]
    output_details: list[dict]
    interpreters: list[FakeInterpreter] = field(default_factory=list)
    frontends: list[FakeFrontend] = field(default_factory=list)
    fail: BaseException | None = None
    loads: int = 0

    def interpreter(self, model: bytes, threads: int) -> FakeInterpreter:
        if self.fail is not None:
            raise self.fail
        built = FakeInterpreter(
            model, threads, self.input_details, self.output_details, self.respond
        )
        self.interpreters.append(built)
        return built

    def frontend(self) -> FakeFrontend:
        built = FakeFrontend(self.features)
        self.frontends.append(built)
        return built

    def __call__(self) -> MicroLibrary:
        self.loads += 1
        return MicroLibrary(interpreter=self.interpreter, frontend=self.frontend)


def scripted(*outputs: int) -> Respond:
    """The n-th invoke of an interpreter answers `outputs[n - 1]`, then 0."""

    def respond(call: int, value: np.ndarray) -> int:
        return outputs[call - 1] if call <= len(outputs) else 0

    return respond


def constant(output: int) -> Respond:
    return lambda call, value: output


def by_content(call: int, value: np.ndarray) -> int:
    """An output that depends only on the input, so two contexts fed the same audio agree."""
    return (int(value.astype(np.int64).sum()) * 31 + 7) % 256


def flat(call: int, pcm: bytes) -> list[float]:
    return [0.0] * 40


def by_level(call: int, pcm: bytes) -> list[float]:
    """Every feature is the chunk's first sample v, as the frontend gives it: `v * 0.0390625`."""
    level = int(np.frombuffer(pcm[:2], dtype="<i2")[0])
    return [level * SCALE] * 40


# -- Manifest and rig helpers -----------------------------------------------------------------


def manifest_path(models_dir: Path) -> Path:
    return models_dir / "speech" / "wake" / MODEL_ID / "manifest.json"


def base_manifest(digest: str) -> dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "names": ["mist"],
        "family": "microwakeword",
        "sample_rate": 16000,
        "feature_step_ms": 10,
        "window_ms": 1000,
        "sliding_window": 5,
        "warmup_ms": 1000,
        "model": {"file": MODEL_FILE, "sha256": digest},
        "hit_lag_ms": 120,
        "calibration": {
            "server": {
                "false_accepts": [[0.5, 2.0], [0.7, 0.5], [0.9, 0.5]],
                "positive_quantile": [[0.5, 0.1], [0.7, 0.1], [0.9, 0.8]],
            }
        },
        "training": "ab" * 32,
    }


def write_model(
    models_dir: Path,
    mutate: Callable[[dict], None] | None = None,
    *,
    content: bytes = MODEL_BYTES,
    **changes: object,
) -> Path:
    """Stage a placeholder model file and its manifest; `changes` and `mutate` alter the manifest."""
    path = manifest_path(models_dir)
    model = path.parent / MODEL_FILE
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(content)
    manifest = base_manifest(hashlib.sha256(content).hexdigest())
    manifest.update(changes)
    if mutate is not None:
        mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def placement(models_dir: Path, **changes: object) -> PlacementConfig:
    return PlacementConfig(models_dir=str(models_dir), kws_model=MODEL_ID, **changes)


def library(
    *,
    stride: int = 2,
    respond: Respond = constant(0),
    features: Features = flat,
    zero_point: int = 0,
    inputs: list[dict] | None = None,
    outputs: list[dict] | None = None,
) -> FakeLibrary:
    return FakeLibrary(
        respond=respond,
        features=features,
        input_details=inputs or [details(0, (1, stride, 40), np.int8, zero_point=-128)],
        output_details=outputs or [details(7, (1, 1), np.uint8, zero_point=zero_point)],
    )


@dataclass
class Rig:
    spotter: MicroNameSpotter
    library: FakeLibrary
    manifest: Path

    def stream(self, threshold: float = 0.5) -> tuple[kws.MicroNameSpotterStream, FakeInterpreter]:
        """A new stream, and the interpreter it built."""
        stream = self.spotter.open_stream(threshold)
        return stream, self.library.interpreters[-1]


def build(
    models_dir: Path,
    *,
    stride: int = 2,
    respond: Respond = constant(0),
    features: Features = flat,
    zero_point: int = 0,
    threads: int = 1,
    window_ms: int = 60,
    sliding_window: int = 1,
    warmup_ms: int = 0,
    **changes: object,
) -> Rig:
    """A loaded spotter; by default a 20 ms frame is one step, the cooldown is 3 steps, the
    window one probability and there is no warm-up, so a scripted output is a frame's score.
    """
    path = write_model(
        models_dir,
        window_ms=window_ms,
        sliding_window=sliding_window,
        warmup_ms=warmup_ms,
        **changes,
    )
    fakes = library(stride=stride, respond=respond, features=features, zero_point=zero_point)
    spotter = MicroNameSpotter.load(
        placement(models_dir, kws_threads=threads), library_loader=fakes
    )
    return Rig(spotter, fakes, path)


def levels(*values: int) -> bytes:
    """PCM16 of 160-sample chunks, chunk k holding `values[k]` in every sample."""
    return b"".join(np.full(CHUNK, value, dtype="<i2").tobytes() for value in values)


def push_frames(
    stream: kws.MicroNameSpotterStream, pcm: bytes, first_pos: int = 0
) -> list[NameHit]:
    """Push `pcm` as 20 ms frames at positions `first_pos...`, the remainder as one more."""
    hits: list[NameHit] = []
    for number, start in enumerate(range(0, len(pcm), FRAME_BYTES)):
        hits.extend(stream.push(pcm[start : start + FRAME_BYTES], first_pos + number))
    return hits


def silence(frames: int) -> bytes:
    return bytes(FRAME_BYTES * frames)


# -- Manifest and load ------------------------------------------------------------------------


def test_load_reads_the_manifest_and_checks_the_model_once(tmp_path: Path) -> None:
    rig = build(tmp_path, stride=3, threads=3, window_ms=1000, sliding_window=5, warmup_ms=1000)
    spotter = rig.spotter
    contract: NameSpotter = spotter
    stream: NameSpotterStream = contract.open_stream(0.5)
    assert stream.push(silence(1), 0) == []
    assert spotter.model_id == MODEL_ID
    assert spotter.names == ("mist",)
    assert spotter.window_ms == 1000
    assert spotter.hit_lag_ms == 120
    assert spotter.training == "ab" * 32
    assert spotter.stride == 3
    assert spotter.cooldown_steps == math.ceil(1000 / 30) == 34
    assert dict(spotter.calibration) == {
        "server": SpotterCalibration(
            false_accepts=((0.5, 2.0), (0.7, 0.5), (0.9, 0.5)),
            positive_quantile=((0.5, 0.1), (0.7, 0.1), (0.9, 0.8)),
        )
    }
    assert rig.library.loads == 1
    checker, opened = rig.library.interpreters
    assert (checker.model, checker.threads, checker.invokes) == (MODEL_BYTES, 3, 0)
    assert (opened.model, opened.threads, opened.allocated) == (MODEL_BYTES, 3, 1)


def test_the_manifest_reads_the_device_file_in_place(tmp_path: Path) -> None:
    path = write_model(tmp_path)
    manifest = kws.read_manifest(path, MODEL_ID)
    assert manifest.model_file == str(path.parent / "device" / "mist.tflite")
    assert manifest.model_sha256 == hashlib.sha256(MODEL_BYTES).hexdigest()
    assert (manifest.window_ms, manifest.sliding_window, manifest.warmup_ms) == (1000, 5, 1000)
    assert (manifest.hit_lag_ms, manifest.names, manifest.path) == (120, ("mist",), str(path))


def test_load_accepts_both_calibration_entries_or_none(tmp_path: Path) -> None:
    entry = {"false_accepts": [[0.5, 1.0]], "positive_quantile": [[0.5, 0.5]]}
    rig = build(tmp_path, calibration={"server": entry, "endpoint": entry})
    assert set(rig.spotter.calibration) == {"server", "endpoint"}
    rig = build(tmp_path, calibration={})
    assert dict(rig.spotter.calibration) == {}


def test_the_model_bytes_are_read_once_and_every_stream_runs_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_model(tmp_path, window_ms=60, sliding_window=1, warmup_ms=0)
    model = manifest_path(tmp_path).parent / MODEL_FILE
    reads: list[Path] = []
    original = pathlib.Path.read_bytes

    def counting(self: Path) -> bytes:
        reads.append(self)
        return original(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", counting)
    fakes = library()
    spotter = MicroNameSpotter.load(placement(tmp_path, kws_threads=2), library_loader=fakes)
    assert [Path(read) for read in reads] == [model]

    # A file swapped after load is never run: the streams build from the bytes that were checked.
    model.write_bytes(b"another model")
    spotter.open_stream(0.5)
    spotter.open_stream(0.5).reset()
    assert [built.model for built in fakes.interpreters] == [MODEL_BYTES] * 4
    assert [built.threads for built in fakes.interpreters] == [2] * 4
    assert len(reads) == 1


def test_load_without_a_model_setting_is_refused(tmp_path: Path) -> None:
    write_model(tmp_path)
    with pytest.raises(SpeechEngineError, match="MIST_VOICE_KWS_MODEL"):
        MicroNameSpotter.load(PlacementConfig(models_dir=str(tmp_path)), library_loader=library())


def test_a_model_id_leaving_the_wake_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SpeechEngineError, match="MIST_VOICE_KWS_MODEL"):
        MicroNameSpotter.load(
            PlacementConfig(models_dir=str(tmp_path), kws_model="../mist"),
            library_loader=library(),
        )


def test_a_missing_manifest_is_refused_naming_it(tmp_path: Path) -> None:
    with pytest.raises(SpeechEngineError) as caught:
        MicroNameSpotter.load(placement(tmp_path), library_loader=library())
    assert str(caught.value).startswith(str(manifest_path(tmp_path)))


def test_a_manifest_that_is_not_json_is_refused_naming_it(tmp_path: Path) -> None:
    path = write_model(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SpeechEngineError) as caught:
        MicroNameSpotter.load(placement(tmp_path), library_loader=library())
    assert str(caught.value).startswith(str(path))


def test_a_missing_model_file_is_refused_naming_it(tmp_path: Path) -> None:
    path = write_model(tmp_path)
    model = path.parent / MODEL_FILE
    model.unlink()
    with pytest.raises(SpeechEngineError) as caught:
        MicroNameSpotter.load(placement(tmp_path), library_loader=library())
    assert str(caught.value).startswith(str(path))
    assert str(model) in str(caught.value)


def _top(key: str, value: object) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        manifest[key] = value

    return mutate


def _drop(key: str) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        del manifest[key]

    return mutate


def _model(key: str, value: object) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        manifest["model"][key] = value

    return mutate


def _drop_model(key: str) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        del manifest["model"][key]

    return mutate


def _server_entry(**entry: object) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        manifest["calibration"] = {"server": entry}

    return mutate


GOOD_FA = [[0.5, 2.0], [0.9, 0.1]]
GOOD_PQ = [[0.5, 0.1], [0.9, 0.8]]
KEYS = sorted(base_manifest("0" * 64))

REFUSALS = {
    **{f"missing {key}": _drop(key) for key in KEYS},
    "unknown top-level key": _top("notes", "x"),
    "old threshold key": _top("threshold", 0.5),
    "old graphs key": _top("graphs", []),
    "family of another runtime": _top("family", "onnx-chain"),
    "family not a string": _top("family", 1),
    "sample rate": _top("sample_rate", 22050),
    "sample rate as a float": _top("sample_rate", 16000.0),
    "feature step": _top("feature_step_ms", 20),
    "feature step as a bool": _top("feature_step_ms", True),
    "window zero": _top("window_ms", 0),
    "window fractional": _top("window_ms", 1000.5),
    "window a string": _top("window_ms", "1000"),
    "sliding window zero": _top("sliding_window", 0),
    "sliding window fractional": _top("sliding_window", 2.5),
    "sliding window a bool": _top("sliding_window", True),
    "negative warm-up": _top("warmup_ms", -1),
    "fractional warm-up": _top("warmup_ms", 10.5),
    "negative hit lag": _top("hit_lag_ms", -1),
    "fractional hit lag": _top("hit_lag_ms", 1.5),
    "model not an object": _top("model", "device/mist.tflite"),
    "model without a file": _drop_model("file"),
    "model without a sha256": _drop_model("sha256"),
    "model with an unknown key": _model("notes", "x"),
    "model file absolute": _model("file", "/models/mist.tflite"),
    "model file with a drive": _model("file", "C:/models/mist.tflite"),
    "model file empty": _model("file", ""),
    "model file not a string": _model("file", 3),
    "model file with an empty part": _model("file", "device//mist.tflite"),
    "model file with a dot part": _model("file", "./device/mist.tflite"),
    "model file with a dot-dot part": _model("file", "device/../../mist.tflite"),
    "model file with a backslash": _model("file", "device\\mist.tflite"),
    "model file missing": _model("file", "device/other.tflite"),
    "sha256 upper-case": _model("sha256", "AB" * 32),
    "sha256 short": _model("sha256", "ab" * 31),
    "sha256 differs": _model("sha256", "0" * 64),
    "false accepts scores not ascending": _server_entry(
        false_accepts=[[0.9, 2.0], [0.5, 0.1]], positive_quantile=GOOD_PQ
    ),
    "false accepts rates increasing": _server_entry(
        false_accepts=[[0.5, 0.1], [0.9, 2.0]], positive_quantile=GOOD_PQ
    ),
    "quantile shares decreasing": _server_entry(
        false_accepts=GOOD_FA, positive_quantile=[[0.5, 0.8], [0.9, 0.1]]
    ),
    "quantile share above one": _server_entry(
        false_accepts=GOOD_FA, positive_quantile=[[0.5, 0.1], [0.9, 1.5]]
    ),
    "entry without quantile": _server_entry(false_accepts=GOOD_FA),
    "entry without false accepts": _server_entry(positive_quantile=GOOD_PQ),
    "unknown calibration spotter": _top(
        "calibration", {"phone": {"false_accepts": GOOD_FA, "positive_quantile": GOOD_PQ}}
    ),
    "unknown calibration entry key": _server_entry(
        false_accepts=GOOD_FA, positive_quantile=GOOD_PQ, notes=[]
    ),
    "empty names": _top("names", []),
    "model id not the directory": _top("model_id", "other"),
    "training not a sha256": _top("training", "abc"),
}


@pytest.mark.parametrize("mutate", REFUSALS.values(), ids=REFUSALS.keys())
def test_a_bad_manifest_is_refused_naming_it(
    tmp_path: Path, mutate: Callable[[dict], None]
) -> None:
    path = write_model(tmp_path, mutate)
    fakes = library()
    with pytest.raises(SpeechEngineError) as caught:
        MicroNameSpotter.load(placement(tmp_path), library_loader=fakes)
    assert str(caught.value).startswith(f"{path}: ")
    with pytest.raises(SpeechEngineError) as again:
        kws.read_manifest(path, MODEL_ID)
    assert str(again.value) == str(caught.value)
    assert fakes.interpreters == []


def test_a_model_whose_sha256_differs_is_refused_naming_the_file(tmp_path: Path) -> None:
    path = write_model(tmp_path, _model("sha256", "0" * 64))
    with pytest.raises(SpeechEngineError) as caught:
        kws.read_manifest(path, MODEL_ID)
    message = str(caught.value)
    assert str(path.parent / MODEL_FILE) in message
    assert hashlib.sha256(MODEL_BYTES).hexdigest() in message


SHAPES = {
    "input float32": ([details(0, (1, 3, 40), np.float32)], None, "float32 [1, 3, 40]"),
    "input uint8": ([details(0, (1, 3, 40), np.uint8)], None, "uint8 [1, 3, 40]"),
    "input of 32 features": ([details(0, (1, 3, 32), np.int8)], None, "int8 [1, 3, 32]"),
    "input stride zero": ([details(0, (1, 0, 40), np.int8)], None, "int8 [1, 0, 40]"),
    "input of rank 2": ([details(0, (3, 40), np.int8)], None, "int8 [3, 40]"),
    "input batch of 2": ([details(0, (2, 3, 40), np.int8)], None, "int8 [2, 3, 40]"),
    "input of rank 4": ([details(0, (1, 3, 40, 1), np.int8)], None, "int8 [1, 3, 40, 1]"),
    "two inputs": (
        [details(0, (1, 3, 40), np.int8), details(1, (1, 3, 40), np.int8)],
        None,
        "int8 [1, 3, 40], int8 [1, 3, 40]",
    ),
    "output int8": (None, [details(7, (1, 1), np.int8)], "int8 [1, 1]"),
    "output float32": (None, [details(7, (1, 1), np.float32)], "float32 [1, 1]"),
    "output of 2": (None, [details(7, (1, 2), np.uint8)], "uint8 [1, 2]"),
    "two outputs": (
        None,
        [details(7, (1, 1), np.uint8), details(8, (1, 1), np.uint8)],
        "uint8 [1, 1], uint8 [1, 1]",
    ),
}


@pytest.mark.parametrize(("inputs", "outputs", "found"), SHAPES.values(), ids=SHAPES.keys())
def test_a_model_of_another_shape_or_type_is_refused_naming_the_file(
    tmp_path: Path, inputs: list[dict] | None, outputs: list[dict] | None, found: str
) -> None:
    path = write_model(tmp_path)
    fakes = library(inputs=inputs, outputs=outputs)
    with pytest.raises(SpeechEngineError) as caught:
        MicroNameSpotter.load(placement(tmp_path), library_loader=fakes)
    message = str(caught.value)
    assert message.startswith(f"{path}: ")
    assert str(path.parent / MODEL_FILE) in message
    assert f"got {found}" in message


def test_a_model_the_interpreter_cannot_open_is_refused_naming_the_file(tmp_path: Path) -> None:
    path = write_model(tmp_path)
    fakes = library()
    fakes.fail = ValueError("Model provided has model identifier 'plac', should be 'TFL3'")
    with pytest.raises(SpeechEngineError) as caught:
        MicroNameSpotter.load(placement(tmp_path), library_loader=fakes)
    assert str(caught.value).startswith(f"{path}: ")
    assert str(path.parent / MODEL_FILE) in str(caught.value)


# -- Features and the input formula -----------------------------------------------------------

TABLE = (0, 1, 2, 600, 670)
QUANTIZED = (-128, -128, -127, 103, 127)


def test_the_input_is_the_devices_integer_formula_on_the_frontend_value(tmp_path: Path) -> None:
    """V = 600 gives 103, where the model's input scale of 26/255 would give 102."""

    def table(call: int, pcm: bytes) -> list[float]:
        return [float(np.float32(value * SCALE)) for value in TABLE] * 8

    rig = build(tmp_path, stride=3, features=table)
    stream, interpreter = rig.stream()
    stream.push(bytes(2 * 3 * CHUNK), 0)
    (step,) = interpreter.inputs
    assert step.dtype == np.int8
    assert step.shape == (1, 3, 40)
    np.testing.assert_array_equal(step, np.array([[list(QUANTIZED) * 8] * 3], dtype=np.int8))


def test_a_buffer_gives_the_same_steps_however_it_is_split(tmp_path: Path) -> None:
    rig = build(tmp_path, stride=3, features=by_level, respond=by_content)
    pcm = levels(*range(1, 61))
    whole, whole_interpreter = rig.stream()
    whole_hits = whole.push(pcm, 0)
    whole_frontend = rig.library.frontends[-1]

    pieces, piece_interpreter = rig.stream()
    piece_frontend = rig.library.frontends[-1]
    piece_hits: list[NameHit] = []
    start = 0
    for size in (7, 333, 1, 159, 160, 3000):
        piece_hits.extend(pieces.push(pcm[2 * start : 2 * (start + size)], 0))
        start += size
    piece_hits.extend(pieces.push(pcm[2 * start :], 0))

    assert len(whole_interpreter.inputs) == 20
    assert len(piece_interpreter.inputs) == 20
    for one, other in zip(whole_interpreter.inputs, piece_interpreter.inputs, strict=True):
        np.testing.assert_array_equal(one, other)
    assert whole_frontend.chunks == piece_frontend.chunks
    assert [hit.score for hit in whole_hits] == [hit.score for hit in piece_hits]


def test_samples_left_over_from_a_push_wait_for_the_next(tmp_path: Path) -> None:
    rig = build(tmp_path, stride=1, features=by_level)
    stream, interpreter = rig.stream()
    pcm = levels(7, 9)
    stream.push(pcm[:300], 0)
    assert rig.library.frontends[-1].chunks == []
    stream.push(pcm[300:322], 1)
    assert rig.library.frontends[-1].chunks == [pcm[:320]]
    assert len(interpreter.inputs) == 1


def test_a_frontend_call_without_features_completes_no_step(tmp_path: Path) -> None:
    def filling(call: int, pcm: bytes) -> list[float]:
        return [] if call <= 2 else [SCALE] * 40

    rig = build(tmp_path, stride=2, features=filling)
    stream, interpreter = rig.stream()
    stream.push(silence(2), 0)
    assert len(interpreter.inputs) == 1


def test_a_frontend_giving_another_feature_count_is_refused(tmp_path: Path) -> None:
    rig = build(tmp_path, features=lambda call, pcm: [0.0] * 32)
    stream, _ = rig.stream()
    with pytest.raises(SpeechEngineError, match="40"):
        stream.push(silence(1), 0)


def test_odd_byte_pcm_is_refused(tmp_path: Path) -> None:
    stream, _ = build(tmp_path).stream()
    with pytest.raises(SpeechEngineError):
        stream.push(b"\x00\x01\x02", 0)
    with pytest.raises(SpeechEngineError):
        stream.warm(b"\x00", 0)


# -- Probabilities, scores and hits -----------------------------------------------------------


@pytest.mark.parametrize(
    ("zero_point", "output", "expected"), [(0, 255, 1.0), (10, 137, 127 / 255)]
)
def test_the_output_is_a_probability_over_255(
    tmp_path: Path, zero_point: int, output: int, expected: float
) -> None:
    rig = build(tmp_path, zero_point=zero_point, respond=constant(output))
    assert rig.spotter.probabilities(silence(2)) == [expected, expected]


def test_the_score_is_the_mean_of_a_zero_filled_window(tmp_path: Path) -> None:
    rig = build(tmp_path, sliding_window=4, respond=scripted(255, 255, 0, 0, 0, 0))
    assert rig.spotter.score_series(silence(6)) == [0.25, 0.5, 0.5, 0.5, 0.25, 0.0]


def test_a_score_equal_to_the_threshold_hits(tmp_path: Path) -> None:
    rig = build(tmp_path, respond=constant(51))
    stream, _ = rig.stream(51 / 255)
    assert [hit.end_pos for hit in push_frames(stream, silence(1))] == [0]
    above, _ = rig.stream(51 / 255 + 1e-9)
    assert push_frames(above, silence(1)) == []


def test_no_hit_within_the_cooldown_of_the_last(tmp_path: Path) -> None:
    # window_ms 60 over 20 ms steps is a cooldown of 3: after a hit at step s, s+1 and s+2
    # cannot hit, and s+3 can.
    outputs = (255, 0, 255, 255, 255, 0, 0, 255, 255, 0, 255)
    rig = build(tmp_path, window_ms=60, respond=scripted(*outputs))
    assert rig.spotter.cooldown_steps == 3
    stream, _ = rig.stream()
    hits = push_frames(stream, silence(len(outputs)))
    assert [hit.end_pos for hit in hits] == [0, 3, 7, 10]
    assert all(hit.keyword == "mist" and hit.score == 1.0 for hit in hits)
    series = [output / 255 for output in outputs]
    assert hit_steps(series, 0.5, 3) == [0, 3, 7, 10]


def test_a_cooldown_rounds_up_to_whole_steps(tmp_path: Path) -> None:
    rig = build(tmp_path, stride=3, window_ms=61)
    assert rig.spotter.cooldown_steps == math.ceil(61 / 30) == 3


def test_a_hits_end_pos_is_the_push_that_completed_its_step(tmp_path: Path) -> None:
    rig = build(tmp_path, window_ms=20, respond=constant(255))
    stream, _ = rig.stream()
    hits = stream.push(silence(4), 9)
    assert hits == [NameHit(end_pos=9, keyword="mist", score=1.0)] * 4


def test_hit_steps_takes_the_greedy_spaced_steps_at_or_above_the_threshold() -> None:
    assert hit_steps([0.6, 0.6, 0.1, 0.7, 0.5], 0.5, 2) == [0, 3]
    assert hit_steps([0.6, 0.6, 0.1, 0.7, 0.5], 0.5, 1) == [0, 1, 3, 4]
    assert hit_steps([], 0.5, 2) == []
    with pytest.raises(ValueError):
        hit_steps([0.5], 0.5, -1)


# -- Warm, reset and the warm-up --------------------------------------------------------------


def test_warm_reports_nothing_but_its_hit_starts_the_cooldown(tmp_path: Path) -> None:
    rig = build(tmp_path, window_ms=60, respond=constant(255))
    stream, _ = rig.stream()
    stream.warm(silence(1), 0)
    assert stream.warm(silence(1), 1) is None
    assert push_frames(stream, silence(1), 2) == []
    assert [hit.end_pos for hit in push_frames(stream, silence(2), 3)] == [3]


def test_a_stream_opens_on_the_warm_up_silence_and_reports_none_of_it(tmp_path: Path) -> None:
    rig = build(tmp_path, warmup_ms=100, features=by_level, respond=constant(255))
    stream, interpreter = rig.stream()
    frontend = rig.library.frontends[-1]
    assert frontend.chunks == [bytes(2 * CHUNK)] * 10
    assert interpreter.invokes == 5
    hits = push_frames(stream, levels(*[5] * 8))
    assert [hit.end_pos for hit in hits] == [0, 3]


def test_reset_starts_a_new_context_with_its_own_interpreter(tmp_path: Path) -> None:
    rig = build(tmp_path, window_ms=60, warmup_ms=40, features=by_level, respond=constant(255))
    stream, first = rig.stream()
    assert [hit.end_pos for hit in push_frames(stream, levels(5, 5))] == [0]
    stream.push(levels(5)[:100], 1)
    stream.reset()
    second = rig.library.interpreters[-1]
    assert second is not first and second.threads == 1
    frontend = rig.library.frontends[-1]
    # The new context warms on silence again; the cooldown and the leftover samples are gone.
    assert [hit.end_pos for hit in push_frames(stream, levels(5, 5), 2)] == [2]
    assert frontend.chunks == [bytes(2 * CHUNK)] * 4 + [levels(5)] * 2


def test_reset_without_warm_up_reports_no_hit_until_warmup_ms_is_fed(tmp_path: Path) -> None:
    rig = build(tmp_path, window_ms=20, warmup_ms=100, features=by_level, respond=constant(255))
    stream, _ = rig.stream()
    stream.reset(warm_up=False)
    frontend = rig.library.frontends[-1]
    hits = push_frames(stream, levels(*[5] * 16))
    # 100 ms is 1600 samples: the step completed by the fifth frame is the first that may hit.
    assert [hit.end_pos for hit in hits] == [4, 5, 6, 7]
    assert frontend.chunks[0] == levels(5)
    assert len(frontend.chunks) == 16

    stream.reset()
    assert rig.library.frontends[-1].chunks == [bytes(2 * CHUNK)] * 10
    assert [hit.end_pos for hit in push_frames(stream, levels(5, 5))] == [0]


def test_probabilities_start_from_a_fresh_context_with_no_warm_up(tmp_path: Path) -> None:
    rig = build(tmp_path, sliding_window=3, warmup_ms=100, respond=scripted(255, 0, 51))
    assert rig.spotter.probabilities(silence(3)) == [1.0, 0.0, 0.2]
    assert rig.library.frontends[-1].chunks == [bytes(2 * CHUNK)] * 6
    assert rig.spotter.probabilities(silence(3)) == [1.0, 0.0, 0.2]


def test_score_window_is_the_highest_score_from_a_fresh_warmed_context(tmp_path: Path) -> None:
    rig = build(tmp_path, sliding_window=2, warmup_ms=40, respond=scripted(255, 255, 0, 255, 51))
    # The warm-up's two steps take the first two outputs, and the window still holds them.
    assert rig.spotter.score_series(silence(3)) == pytest.approx([0.5, 0.5, 0.6])
    assert rig.spotter.score_window(silence(3)) == pytest.approx(0.6)
    assert rig.spotter.score_window(bytes(2 * CHUNK)) == 0.0
    assert rig.spotter.score_window(b"") == 0.0


# -- The seeded property ----------------------------------------------------------------------


def _random_case(rng: random.Random) -> tuple[dict[str, int], bytes]:
    settings = {
        "stride": rng.randint(1, 3),
        "sliding_window": rng.randint(1, 6),
        "window_ms": rng.randint(10, 150),
        "warmup_ms": rng.choice((0, 30, 45, 200)),
    }
    pcm = levels(*(rng.randint(0, 700) for _ in range(rng.randint(0, 80))))
    return settings, pcm[: len(pcm) - 2 * rng.randint(0, 100)] if pcm else pcm


@pytest.mark.parametrize("seed", range(40))
def test_hit_steps_is_monotone_and_equals_a_stream_split_between_warm_and_push(
    tmp_path: Path, seed: int
) -> None:
    rng = random.Random(seed)
    settings, pcm = _random_case(rng)
    rig = build(tmp_path, features=by_level, respond=by_content, **settings)
    spotter = rig.spotter
    series = spotter.score_series(pcm)
    cooldown = spotter.cooldown_steps

    thresholds = sorted({rng.random() for _ in range(12)} | set(series))
    counts = [len(hit_steps(series, threshold, cooldown)) for threshold in thresholds]
    assert counts == sorted(counts, reverse=True)

    # The same audio, split at random between `warm` and `push`.
    threshold = rng.choice(thresholds) if thresholds else 0.5
    stream, _ = rig.stream(threshold)
    pieces: list[tuple[int, bool]] = []
    reported: list[tuple[int, float]] = []
    total = len(pcm) // 2
    start = 0
    while start < total:
        end = min(total, start + rng.randint(1, 700))
        piece = pcm[2 * start : 2 * end]
        pos, pushed = len(pieces), rng.random() < 0.5
        if pushed:
            reported.extend((hit.end_pos, hit.score) for hit in stream.push(piece, pos))
        else:
            stream.warm(piece, pos)
        start = end
        pieces.append((end, pushed))

    # Steps completed by the end of each piece: one feature per 160 samples, silence included.
    silence_samples = settings["warmup_ms"] * 16
    before = silence_samples // CHUNK // settings["stride"]
    done = [(silence_samples + end) // CHUNK // settings["stride"] - before for end, _ in pieces]
    assert len(series) == (done[-1] if done else 0)
    expected = [
        (piece, series[step])
        for step in hit_steps(series, threshold, cooldown)
        for piece in [bisect_right(done, step)]
        if pieces[piece][1]
    ]
    assert reported == expected

    # `score_window` reaches a threshold exactly when a fresh stream pushed the audio hits.
    for level in (rng.uniform(0.01, 1.0) for _ in range(3)):
        fresh, _ = rig.stream(level)
        assert (spotter.score_window(pcm) >= level) == bool(push_frames(fresh, pcm))


# -- The runtime import -----------------------------------------------------------------------


class FakeLiteInterpreter:
    def __init__(self, *, model_content: bytes, num_threads: int) -> None:
        self.model_content = model_content
        self.num_threads = num_threads


@dataclass
class FakeResult:
    features: list[float]
    samples_read: int


class SnakeFrontend:
    def process_samples(self, audio: bytes) -> FakeResult:
        return FakeResult(features=[SCALE] * 40, samples_read=len(audio) // 2)


class CamelFrontend:
    def ProcessSamples(self, audio: bytes) -> FakeResult:  # noqa: N802 - pymicro-features 1.x
        return FakeResult(features=[2 * SCALE] * 40, samples_read=len(audio) // 2)


class ShortFrontend:
    def process_samples(self, audio: bytes) -> FakeResult:
        return FakeResult(features=[], samples_read=len(audio) // 4)


def _install(
    monkeypatch: pytest.MonkeyPatch, *, litert: bool = True, frontend: type | None = SnakeFrontend
) -> None:
    if litert:
        package = types.ModuleType("ai_edge_litert")
        module = types.ModuleType("ai_edge_litert.interpreter")
        module.Interpreter = FakeLiteInterpreter  # type: ignore[attr-defined]
        package.interpreter = module  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ai_edge_litert", package)
        monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", module)
    else:
        monkeypatch.setitem(sys.modules, "ai_edge_litert", None)
        monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", None)
    if frontend is None:
        monkeypatch.setitem(sys.modules, "pymicro_features", None)
    else:
        micro = types.ModuleType("pymicro_features")
        micro.MicroFrontend = frontend  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pymicro_features", micro)


@pytest.mark.parametrize(
    ("litert", "frontend", "package"),
    [(False, SnakeFrontend, "ai-edge-litert"), (True, None, "pymicro-features")],
)
def test_import_micro_runtime_names_the_missing_package(
    monkeypatch: pytest.MonkeyPatch, litert: bool, frontend: type | None, package: str
) -> None:
    _install(monkeypatch, litert=litert, frontend=frontend)
    with pytest.raises(SpeechEngineError) as caught:
        kws.import_micro_runtime()
    assert package in str(caught.value)
    assert "requirements-voice.txt" in str(caught.value)


@pytest.mark.parametrize(
    ("frontend", "value"), [(SnakeFrontend, SCALE), (CamelFrontend, 2 * SCALE)]
)
def test_import_micro_runtime_binds_the_interpreter_and_the_frontend(
    monkeypatch: pytest.MonkeyPatch, frontend: type, value: float
) -> None:
    _install(monkeypatch, frontend=frontend)
    runtime = kws.import_micro_runtime()
    interpreter = runtime.interpreter(MODEL_BYTES, 3)
    assert isinstance(interpreter, FakeLiteInterpreter)
    assert (interpreter.model_content, interpreter.num_threads) == (MODEL_BYTES, 3)
    assert list(runtime.frontend().features(bytes(2 * CHUNK))) == [value] * 40


def test_a_frontend_that_reads_part_of_a_call_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, frontend=ShortFrontend)
    with pytest.raises(SpeechEngineError, match="160"):
        kws.import_micro_runtime().frontend().features(bytes(2 * CHUNK))


def test_the_module_imports_neither_runtime_package() -> None:
    source = Path(kws.__file__).read_text(encoding="utf-8")
    assert "import ai_edge_litert" not in source
    assert "import pymicro_features" not in source
    assert "from ai_edge_litert" not in source
    assert "from pymicro_features" not in source
