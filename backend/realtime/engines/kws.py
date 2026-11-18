"""The server name spotter: microWakeWord's streaming model behind `NameSpotter`, on LiteRT.

One model serves the room device and the server, so one training serves both and their two
calibrations describe the same weights. The server runs the device's own streaming `.tflite`
file in place: the manifest's `model.file` is a path inside the model directory
(`device/mist.tflite` for `mist-mww-1`), so both spotters run one file with one sha256.

The input is quantized as the device quantizes it. pymicro-features' microfrontend gives each
of its 40 features per 10 ms as the integer value v times 0.0390625; the adapter recovers v
exactly and applies ESPHome's fixed integer formula, `(v * 256 + 333) // 666 - 128` clamped to
int8 (`micro_wake_word.cpp`), not the model's input scale, so the server scores what the device
scores. The uint8 output becomes a probability as microWakeWord's `dequantize_output_data`
computes it, `(q - zero_point) / 255`.

A hit is the server's own rule, not the device's: a step whose score, the mean of the last
`sliding_window` probabilities, is at least the threshold, when no hit came in the last
`cooldown_steps` steps, which span `window_ms` of audio. The rule is greedy, so a higher threshold
never gives more hits on the same scores (its hits are spaced picks from a subset of the steps a
lower one picks from, and greedy picks the most), and the converter that writes
`calibration.server` refuses counts that rise with the threshold. One `_Cooldown` holds the rule
for the stream and for `hit_steps`, the offline evaluation's count, so what is measured is what
ships.

The streaming model keeps its state in the interpreter, so each stream, each reset and each
offline score builds its own interpreter, from the bytes `load` read and checked once.

Weights are never downloaded, never committed, and stay under `<models_dir>/speech/wake/<id>/`,
because the trained weights are for personal use only. ai-edge-litert and pymicro-features are
imported only by `import_micro_runtime`, which `load` is handed, so importing this module loads
neither and tests pass a fake library.
"""

import hashlib
import importlib
import json
import logging
import math
import re
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Protocol, TypedDict

import numpy as np

from ..budget import PlacementConfig
from ..errors import SpeechEngineError
from ..protocol import Spotter
from ..speech import NameHit, SpotterCalibration

logger = logging.getLogger(__name__)

FAMILY = "microwakeword"
SAMPLE_RATE = 16000
FEATURE_STEP_MS = 10
FEATURES = 40
FRONTEND_SAMPLES = SAMPLE_RATE * FEATURE_STEP_MS // 1000
"""Samples per frontend call, 10 ms, as microWakeWord's `generate_features_for_clip` feeds it."""
FEATURE_SCALE = 0.0390625
"""pymicro-features' `FLOAT32_SCALE`: a feature is the microfrontend's integer value times it."""
INPUT_VALUE_SCALE = 256
INPUT_VALUE_DIV = 666
"""ESPHome's `value_scale` and `value_div`, the device's int8 input formula on the value."""
OUTPUT_SCALE = 255.0
"""microWakeWord's `dequantize_output_data` divides by 255 whatever the output's own scale."""

_FRONTEND_BYTES = 2 * FRONTEND_SAMPLES
_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_KEYS = frozenset(
    {
        "model_id",
        "names",
        "family",
        "sample_rate",
        "feature_step_ms",
        "window_ms",
        "sliding_window",
        "warmup_ms",
        "model",
        "hit_lag_ms",
        "calibration",
        "training",
    }
)
_MODEL_KEYS = frozenset({"file", "sha256"})
_SPOTTERS: tuple[Spotter, ...] = ("server", "endpoint")
_CALIBRATION_KEYS = frozenset({"false_accepts", "positive_quantile"})
_RUNTIME = (
    ("ai_edge_litert.interpreter", "ai-edge-litert"),
    ("pymicro_features", "pymicro-features"),
)
"""Each runtime module `import_micro_runtime` imports, with the package that provides it."""
_INTERPRETER_ERRORS = (ValueError, RuntimeError)
"""What LiteRT's `Interpreter` raises on a file it cannot open or allocate."""


class TensorDetails(TypedDict):
    """The keys of LiteRT's tensor details the spotter reads; the real dicts hold more."""

    index: int
    shape: np.ndarray
    dtype: type
    quantization: tuple[float, int]


class MicroInterpreter(Protocol):
    """The part of LiteRT's `Interpreter` the spotter calls, so tests can fake it."""

    def allocate_tensors(self) -> None:
        """Allocate the tensors; needed once before the first `set_tensor`."""
        ...

    def get_input_details(self) -> Sequence[TensorDetails]:
        """The model's inputs."""
        ...

    def get_output_details(self) -> Sequence[TensorDetails]:
        """The model's outputs."""
        ...

    def set_tensor(self, tensor_index: int, value: np.ndarray) -> None:
        """Copy `value` into the input tensor at `tensor_index`."""
        ...

    def invoke(self) -> None:
        """Run the model once, advancing its streaming state."""
        ...

    def get_tensor(self, tensor_index: int) -> np.ndarray:
        """A copy of the tensor at `tensor_index`."""
        ...


class FeatureFrontend(Protocol):
    """The microfrontend, fed one 10 ms call at a time as microWakeWord feeds it."""

    def features(self, pcm: bytes) -> Sequence[float]:
        """Take 160 samples of PCM16 and return the 40 features they complete.

        Returns no features while the frontend's analysis window is still filling.
        """
        ...


@dataclass(frozen=True, slots=True)
class MicroLibrary:
    """What the spotter needs from the runtime, bound once by `import_micro_runtime`.

    `interpreter(model, threads)` builds a LiteRT interpreter over the model's bytes;
    `frontend()` a fresh microfrontend, whose filterbank and noise state are per stream.
    """

    interpreter: Callable[[bytes, int], MicroInterpreter]
    frontend: Callable[[], FeatureFrontend]


class _FrontendResult(Protocol):
    features: Sequence[float]
    samples_read: int


class _PyMicroFrontend:
    """pymicro-features' `MicroFrontend` behind `FeatureFrontend`.

    pymicro-features 1.x named the call `ProcessSamples` and the pinned 2.x line names it
    `process_samples` (unverified here, since the runner has neither); the adapter takes either,
    so the pin can move without a code change. A call that reads less than the 160 samples it is
    given would drop audio, so it is refused.
    """

    __slots__ = ("_process",)

    def __init__(self, frontend: object) -> None:
        process = getattr(frontend, "process_samples", None) or getattr(
            frontend, "ProcessSamples", None
        )
        if not callable(process):
            raise SpeechEngineError(
                "pymicro-features' MicroFrontend has no process_samples call; "
                "requirements-voice.txt pins the version the spotter is written for"
            )
        self._process: Callable[[bytes], _FrontendResult] = process

    def features(self, pcm: bytes) -> Sequence[float]:
        """Run one 10 ms call; see `FeatureFrontend.features`."""
        result = self._process(pcm)
        if result.samples_read != len(pcm) // 2:
            raise SpeechEngineError(
                f"the microfrontend read {result.samples_read} of the {FRONTEND_SAMPLES} samples "
                "it was given"
            )
        return result.features


def import_micro_runtime() -> MicroLibrary:
    """Import ai-edge-litert and pymicro-features and bind what the spotter calls.

    The interpreter is LiteRT's `ai_edge_litert.interpreter.Interpreter`, as microWakeWord's own
    `inference.py` imports it, and the frontend pymicro-features' `MicroFrontend`, as its
    `generate_features_for_clip` builds it.

    Raises:
        SpeechEngineError: A package is not installed; the message names it.
    """
    modules = []
    for module, package in _RUNTIME:
        try:
            modules.append(importlib.import_module(module))
        except ImportError as exc:
            raise SpeechEngineError(
                f"{package} is not installed ({exc.name or module}); requirements-voice.txt "
                "pins ai-edge-litert and pymicro-features for the server name spotter"
            ) from exc
    litert, micro = modules
    interpreter_class = litert.Interpreter
    frontend_class = micro.MicroFrontend

    def interpreter(model: bytes, threads: int) -> MicroInterpreter:
        return interpreter_class(model_content=model, num_threads=threads)

    def frontend() -> FeatureFrontend:
        return _PyMicroFrontend(frontend_class())

    return MicroLibrary(interpreter=interpreter, frontend=frontend)


@dataclass(frozen=True, slots=True)
class WakeManifest:
    """A model directory's `manifest.json` after validation.

    `model_file` is the model's full path and `model_sha256` its checked digest. `training` is
    the sha256 of the training-split manifest the model was trained from, kept so that the
    evaluation can check which split a measured model came from.
    """

    path: str
    model_id: str
    names: tuple[str, ...]
    window_ms: int
    sliding_window: int
    warmup_ms: int
    model_file: str
    model_sha256: str
    hit_lag_ms: int
    calibration: Mapping[Spotter, SpotterCalibration]
    training: str


# -- Manifest validation ----------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class _Reader:
    """Validation helpers that prefix every refusal with the manifest's path."""

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        self.path = path

    def fail(self, reason: str) -> SpeechEngineError:
        return SpeechEngineError(f"{self.path}: {reason}")

    def keys(
        self,
        value: object,
        where: str,
        allowed: set[str] | frozenset[str],
        *,
        optional: bool = False,
    ) -> dict:
        if not isinstance(value, dict):
            raise self.fail(f"{where} must be an object")
        unknown = sorted(set(value) - set(allowed))
        missing = [] if optional else sorted(set(allowed) - set(value))
        if unknown:
            raise self.fail(f"{where} has unknown keys {unknown}")
        if missing:
            raise self.fail(f"{where} is missing keys {missing}")
        return value

    def integer(self, value: object, where: str, minimum: int) -> int:
        if not _is_int(value) or value < minimum:
            raise self.fail(f"{where} must be an integer of at least {minimum}, got {value!r}")
        return value

    def exactly(self, value: object, where: str, expected: int) -> None:
        if not _is_int(value) or value != expected:
            raise self.fail(f"{where} must be {expected} for {FAMILY}, got {value!r}")

    def pairs(self, value: object, where: str) -> tuple[tuple[float, float], ...]:
        if not isinstance(value, list) or not value:
            raise self.fail(f"{where} must be a non-empty list of [score, value] pairs")
        points: list[tuple[float, float]] = []
        for point in value:
            if not (isinstance(point, list) and len(point) == 2 and all(map(_is_number, point))):
                raise self.fail(f"{where} must hold [score, value] pairs of numbers, got {point!r}")
            points.append((float(point[0]), float(point[1])))
        scores = [score for score, _ in points]
        if any(later <= earlier for earlier, later in zip(scores, scores[1:], strict=False)):
            raise self.fail(f"{where} scores must be strictly ascending")
        return tuple(points)


def _calibration_entry(reader: _Reader, value: object, where: str) -> SpotterCalibration:
    entry = reader.keys(value, where, _CALIBRATION_KEYS)
    false_accepts = reader.pairs(entry["false_accepts"], f"{where}.false_accepts")
    rates = [rate for _, rate in false_accepts]
    if rates[-1] < 0 or any(b > a for a, b in zip(rates, rates[1:], strict=False)):
        raise reader.fail(f"{where}.false_accepts rates must be at least 0 and not increase")
    quantile = reader.pairs(entry["positive_quantile"], f"{where}.positive_quantile")
    shares = [share for _, share in quantile]
    if not (shares[0] >= 0.0 and shares[-1] <= 1.0) or any(
        b < a for a, b in zip(shares, shares[1:], strict=False)
    ):
        raise reader.fail(f"{where}.positive_quantile shares must run from 0 to 1, not decreasing")
    return SpotterCalibration(false_accepts=false_accepts, positive_quantile=quantile)


def _model_file(reader: _Reader, value: object, directory: Path) -> tuple[Path, str, bytes]:
    """The model file's path, its checked sha256 and its bytes, read once.

    The path is split on `/` by hand: `PurePosixPath` folds an empty or `.` part away, and a
    backslash or a drive would leave the directory on Windows.
    """
    entry = reader.keys(value, "model", _MODEL_KEYS)
    file = entry["file"]
    if not isinstance(file, str) or not file:
        raise reader.fail("model.file must be a relative path inside the model directory")
    parts = file.split("/")
    if (
        "\\" in file
        or PureWindowsPath(file).drive
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise reader.fail(
            f"model.file {file!r} must be a relative path inside {directory}, with no empty, "
            "'.' or '..' part"
        )
    digest = entry["sha256"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise reader.fail("model.sha256 must be 64 lower-case hex digits")
    path = directory.joinpath(*parts)
    try:
        content = path.read_bytes()
    except FileNotFoundError as exc:
        raise reader.fail(f"wake model {path} not found") from exc
    except OSError as exc:
        raise reader.fail(f"wake model {path} could not be read: {exc}") from exc
    actual = hashlib.sha256(content).hexdigest()
    if actual != digest:
        raise reader.fail(f"wake model {path} has sha256 {actual}, the manifest gives {digest}")
    return path, digest, content


def _read(path: Path, model_id: str) -> tuple[WakeManifest, bytes]:
    """The checked manifest and the model's bytes, which `load` runs without reading again."""
    reader = _Reader(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise reader.fail("wake model manifest not found") from exc
    except OSError as exc:
        raise reader.fail(f"wake model manifest could not be read: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise reader.fail(f"manifest is not valid JSON: {exc}") from exc
    manifest = reader.keys(raw, "manifest", _MANIFEST_KEYS)
    if manifest["model_id"] != model_id:
        raise reader.fail(f"model_id {manifest['model_id']!r} is not the directory's {model_id!r}")
    names = manifest["names"]
    if not (isinstance(names, list) and names and all(isinstance(n, str) and n for n in names)):
        raise reader.fail("names must be a non-empty list of non-empty strings")
    if manifest["family"] != FAMILY:
        raise reader.fail(f"family must be {FAMILY!r}, got {manifest['family']!r}")
    reader.exactly(manifest["sample_rate"], "sample_rate", SAMPLE_RATE)
    reader.exactly(manifest["feature_step_ms"], "feature_step_ms", FEATURE_STEP_MS)
    window_ms = reader.integer(manifest["window_ms"], "window_ms", 1)
    sliding_window = reader.integer(manifest["sliding_window"], "sliding_window", 1)
    warmup_ms = reader.integer(manifest["warmup_ms"], "warmup_ms", 0)
    hit_lag_ms = reader.integer(manifest["hit_lag_ms"], "hit_lag_ms", 0)
    model_path, digest, content = _model_file(reader, manifest["model"], path.parent)
    # Each spotter's entry is optional: a spotter with none ranks as uncalibrated.
    calibration = reader.keys(manifest["calibration"], "calibration", set(_SPOTTERS), optional=True)
    table = {
        spotter: _calibration_entry(reader, calibration[spotter], f"calibration.{spotter}")
        for spotter in _SPOTTERS
        if spotter in calibration
    }
    training = manifest["training"]
    if not isinstance(training, str) or not _SHA256.fullmatch(training):
        raise reader.fail("training must be the training-split manifest's sha256")
    checked = WakeManifest(
        path=str(path),
        model_id=model_id,
        names=tuple(names),
        window_ms=window_ms,
        sliding_window=sliding_window,
        warmup_ms=warmup_ms,
        model_file=str(model_path),
        model_sha256=digest,
        hit_lag_ms=hit_lag_ms,
        calibration=MappingProxyType(table),
        training=training,
    )
    return checked, content


def read_manifest(path: Path, model_id: str) -> WakeManifest:
    """Read and check `<models_dir>/speech/wake/<model_id>/manifest.json` and its model file.

    Every key is required and an unknown key is refused, so a field the code does not read
    cannot be mistaken for one it honours.

    Raises:
        SpeechEngineError: The manifest or the model file is missing, the model's sha256
            differs, or a value is outside what this spotter implements. The message starts
            with the manifest's path, and names the model file where it is at fault.
    """
    return _read(path, model_id)[0]


# -- The streaming context --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Model:
    """What every context of one loaded model shares: the checked bytes and the step's shape."""

    content: bytes
    library: MicroLibrary
    threads: int
    stride: int
    input_index: int
    output_index: int
    output_zero_point: int
    sliding_window: int
    warmup_samples: int
    cooldown_steps: int
    keyword: str


@dataclass(frozen=True, slots=True)
class _Step:
    """One invoke: its probability, the window's score after it, and whether it may hit."""

    probability: float
    score: float
    eligible: bool


class _Cooldown:
    """The hit rule: at or above the threshold, and no hit in the last `steps` steps.

    A hit at step s blocks steps s+1 to s+steps-1, so two hits are at least `steps` steps, and
    so at least `window_ms` of audio, apart. Greedy, so the count never rises with the
    threshold.
    """

    __slots__ = ("_since", "_steps")

    def __init__(self, steps: int) -> None:
        if steps < 0:
            raise ValueError(f"a cooldown is a count of steps, got {steps}")
        self._steps = steps
        self._since: int | None = None

    def step(self, reaches: bool) -> bool:
        """Advance one step; True when the step is a hit, which restarts the cooldown."""
        if self._since is not None:
            self._since += 1
        if reaches and (self._since is None or self._since >= self._steps):
            self._since = 0
            return True
        return False


def hit_steps(series: Sequence[float], threshold: float, cooldown_steps: int) -> list[int]:
    """The steps of a score series that hit at `threshold`, by the stream's own rule.

    A stream fed the same audio from a fresh warmed context, in any split between `warm` and
    `push`, finds these hits and reports those in pushed audio.

    Raises:
        ValueError: `cooldown_steps` is negative.
    """
    cooldown = _Cooldown(cooldown_steps)
    return [index for index, score in enumerate(series) if cooldown.step(score >= threshold)]


def _quantize(values: Sequence[float]) -> np.ndarray:
    """One step's features as the device quantizes them: v recovered, then ESPHome's formula."""
    features = np.asarray(values, dtype=np.float64)
    if features.shape != (FEATURES,):
        raise SpeechEngineError(
            f"the microfrontend gave {features.size} features, expected {FEATURES}"
        )
    value = np.rint(features / FEATURE_SCALE).astype(np.int64)
    quantized = (value * INPUT_VALUE_SCALE + INPUT_VALUE_DIV // 2) // INPUT_VALUE_DIV - 128
    return np.clip(quantized, -128, 127).astype(np.int8)


def _open_interpreter(library: MicroLibrary, content: bytes, threads: int) -> MicroInterpreter:
    interpreter = library.interpreter(content, threads)
    interpreter.allocate_tensors()
    return interpreter


class _Context:
    """One fresh streaming state: interpreter, frontend, pending samples and features, window.

    The window starts as zeros, as the device's sliding window does. With `warm_up` the context
    is first fed `warmup_ms` of digital silence, whose steps are dropped, so a short buffer
    starts from a settled state; without it, no step is eligible to hit until `warmup_ms` of
    audio has been fed, as the device ignores its first slices after a reset.
    """

    __slots__ = (
        "_fed",
        "_features",
        "_frontend",
        "_interpreter",
        "_model",
        "_pending",
        "_quiet",
        "_window",
    )

    def __init__(self, model: _Model, *, warm_up: bool) -> None:
        self._model = model
        self._interpreter = _open_interpreter(model.library, model.content, model.threads)
        self._frontend = model.library.frontend()
        self._pending = b""
        self._features: list[np.ndarray] = []
        self._window: deque[float] = deque(
            [0.0] * model.sliding_window, maxlen=model.sliding_window
        )
        self._fed = 0
        self._quiet = 0 if warm_up else model.warmup_samples
        if warm_up and model.warmup_samples:
            self.feed(bytes(2 * model.warmup_samples))

    def feed(self, pcm: bytes) -> list[_Step]:
        """Run every step `pcm` completes; samples past the last whole 10 ms wait.

        Raises:
            SpeechEngineError: The PCM is not whole 16-bit samples, or the frontend gives
                other than 40 features.
        """
        if len(pcm) % 2:
            raise SpeechEngineError(f"name spotter takes PCM16, got {len(pcm)} bytes")
        pending = self._pending + pcm
        steps: list[_Step] = []
        start = 0
        while len(pending) - start >= _FRONTEND_BYTES:
            chunk = pending[start : start + _FRONTEND_BYTES]
            start += _FRONTEND_BYTES
            self._fed += FRONTEND_SAMPLES
            values = self._frontend.features(chunk)
            if not len(values):
                continue
            self._features.append(_quantize(values))
            if len(self._features) == self._model.stride:
                steps.append(self._invoke())
        self._pending = pending[start:]
        return steps

    def _invoke(self) -> _Step:
        model = self._model
        block = np.stack(self._features)[np.newaxis, :, :]
        self._features = []
        self._interpreter.set_tensor(model.input_index, block)
        self._interpreter.invoke()
        output = int(np.asarray(self._interpreter.get_tensor(model.output_index)).reshape(-1)[0])
        probability = (output - model.output_zero_point) / OUTPUT_SCALE
        self._window.append(probability)
        score = sum(self._window) / model.sliding_window
        return _Step(probability=probability, score=score, eligible=self._fed >= self._quiet)


# -- The spotter ------------------------------------------------------------------------------


class MicroNameSpotterStream:
    """One audio source's spotting state: the model's own state, the window and the cooldown.

    `warm` and `push` run the same steps and the same cooldown, so a hit in warmed audio blocks
    the pushed hits after it as one in pushed audio would; only `push` reports.
    """

    __slots__ = ("_context", "_cooldown", "_model", "_threshold")

    def __init__(self, model: _Model, threshold: float) -> None:
        self._model = model
        self._threshold = threshold
        self.reset()

    def push(self, pcm: bytes, pos: int) -> list[NameHit]:
        """Score PCM16 mono 16 kHz audio and report each step that hits.

        A hit's `end_pos` is `pos`, the frame whose push completed the step, and `keyword` the
        manifest's first name: the model gives one probability, not one per name.

        Raises:
            SpeechEngineError: The PCM is not whole 16-bit samples, or the frontend gives
                other than 40 features.
        """
        return [
            NameHit(end_pos=pos, keyword=self._model.keyword, score=score)
            for score in self._hits(pcm)
        ]

    def warm(self, pcm: bytes, pos: int) -> None:
        """Score audio as `push` does and report nothing; a hit it finds starts the cooldown.

        Raises:
            SpeechEngineError: As `push`.
        """
        self._hits(pcm)

    def reset(self, *, warm_up: bool = True) -> None:
        """Start a new context: a new interpreter and frontend, an empty window and cooldown.

        With `warm_up` the context first hears `warmup_ms` of silence; without it, no hit is
        reported until `warmup_ms` of audio has been fed, so a reset while behind adds no work.
        """
        self._context = _Context(self._model, warm_up=warm_up)
        self._cooldown = _Cooldown(self._model.cooldown_steps)

    def _hits(self, pcm: bytes) -> list[float]:
        return [
            step.score
            for step in self._context.feed(pcm)
            if self._cooldown.step(step.eligible and step.score >= self._threshold)
        ]


class MicroNameSpotter:
    """The microWakeWord spotter over one checked model file that every stream shares.

    `stride` is the feature steps one invoke reads, from the model's input shape, and
    `cooldown_steps` the invokes `window_ms` spans, rounded up.
    """

    __slots__ = (
        "_model",
        "calibration",
        "cooldown_steps",
        "hit_lag_ms",
        "model_id",
        "names",
        "stride",
        "training",
        "window_ms",
    )

    def __init__(self, manifest: WakeManifest, model: _Model) -> None:
        self._model = model
        self.model_id = manifest.model_id
        self.names = manifest.names
        self.window_ms = manifest.window_ms
        self.hit_lag_ms = manifest.hit_lag_ms
        self.calibration = manifest.calibration
        self.training = manifest.training
        self.stride = model.stride
        self.cooldown_steps = model.cooldown_steps

    @classmethod
    def load(
        cls, placement: PlacementConfig, *, library_loader: Callable[[], MicroLibrary]
    ) -> "MicroNameSpotter":
        """Open `placement.kws_model` from `<models_dir>/speech/wake/<id>/` on the CPU.

        The model's bytes are read once and their sha256 checked; one interpreter built from
        them checks the model's shape, and every stream later builds its own from the same
        bytes with `placement.kws_threads` threads.

        Raises:
            SpeechEngineError: No model is set, the id is malformed, the manifest is refused
                (`read_manifest`), a runtime package is missing, the interpreter cannot open
                the file, or its input is not int8 `[1, stride, 40]` or its output not uint8
                `[1, 1]`; a refusal of the file names it and the shape found.
        """
        model_id = placement.kws_model
        if model_id is None:
            raise SpeechEngineError("MIST_VOICE_KWS_MODEL is not set; the server spotter is off")
        if not _ID.fullmatch(model_id):
            raise SpeechEngineError(
                f"MIST_VOICE_KWS_MODEL={model_id!r}: expected a model id [a-z0-9][a-z0-9-]*"
            )
        path = Path(placement.models_dir, "speech", "wake", model_id, "manifest.json")
        manifest, content = _read(path, model_id)
        library = library_loader()
        where = f"{manifest.path}: wake model {manifest.model_file}"
        try:
            interpreter = _open_interpreter(library, content, placement.kws_threads)
        except _INTERPRETER_ERRORS as exc:
            raise SpeechEngineError(
                f"{where} could not be opened ({type(exc).__name__}: {exc})"
            ) from exc
        stride, input_index = _check_input(interpreter.get_input_details(), where)
        output_index, zero_point = _check_output(interpreter.get_output_details(), where)
        model = _Model(
            content=content,
            library=library,
            threads=placement.kws_threads,
            stride=stride,
            input_index=input_index,
            output_index=output_index,
            output_zero_point=zero_point,
            sliding_window=manifest.sliding_window,
            warmup_samples=manifest.warmup_ms * SAMPLE_RATE // 1000,
            cooldown_steps=math.ceil(manifest.window_ms / (stride * FEATURE_STEP_MS)),
            keyword=manifest.names[0],
        )
        spotter = cls(manifest, model)
        logger.info(
            "Name spotter %s loaded from %s with %d threads, stride %d, cooldown %d steps",
            model_id,
            manifest.model_file,
            placement.kws_threads,
            stride,
            model.cooldown_steps,
        )
        return spotter

    def open_stream(self, threshold: float) -> MicroNameSpotterStream:
        """Open a stream on a fresh warmed context that reports scores at or above `threshold`."""
        return MicroNameSpotterStream(self._model, threshold)

    def score_window(self, pcm: bytes) -> float:
        """The highest score over a finished buffer from a fresh warmed context, else 0.0.

        For a threshold above 0, the result reaches it exactly when a fresh stream pushed the
        same audio reports a hit. A buffer that completes no step scores 0.0.

        Raises:
            SpeechEngineError: As `MicroNameSpotterStream.push`.
        """
        return max(self.score_series(pcm), default=0.0)

    def score_series(self, pcm: bytes) -> list[float]:
        """Every step's score over a buffer from a fresh warmed context, for `hit_steps`.

        Raises:
            SpeechEngineError: As `MicroNameSpotterStream.push`.
        """
        return [step.score for step in _Context(self._model, warm_up=True).feed(pcm)]

    def probabilities(self, pcm: bytes) -> list[float]:
        """Every step's raw probability from a fresh context with no warm-up, for parity.

        This is what microWakeWord's own inference gives on the same audio, before the
        sliding window.

        Raises:
            SpeechEngineError: As `MicroNameSpotterStream.push`.
        """
        return [step.probability for step in _Context(self._model, warm_up=False).feed(pcm)]


def _describe(details: Sequence[TensorDetails]) -> str:
    found = [
        f"{np.dtype(item['dtype']).name} {[int(size) for size in item['shape']]}"
        for item in details
    ]
    return ", ".join(found) or "none"


def _check_input(details: Sequence[TensorDetails], where: str) -> tuple[int, int]:
    """The stride and the input's index, from one int8 input of shape `[1, stride, 40]`."""
    shape = [int(size) for size in details[0]["shape"]] if len(details) == 1 else []
    if (
        len(details) != 1
        or np.dtype(details[0]["dtype"]) != np.int8
        or len(shape) != 3
        or shape[0] != 1
        or shape[1] < 1
        or shape[2] != FEATURES
    ):
        raise SpeechEngineError(
            f"{where}: the input must be one int8 tensor of shape [1, stride, {FEATURES}] with a "
            f"positive stride, got {_describe(details)}"
        )
    return shape[1], int(details[0]["index"])


def _check_output(details: Sequence[TensorDetails], where: str) -> tuple[int, int]:
    """The output's index and zero point, from one uint8 output of shape `[1, 1]`."""
    shape = [int(size) for size in details[0]["shape"]] if len(details) == 1 else []
    if len(details) != 1 or np.dtype(details[0]["dtype"]) != np.uint8 or shape != [1, 1]:
        raise SpeechEngineError(
            f"{where}: the output must be one uint8 tensor of shape [1, 1], "
            f"got {_describe(details)}"
        )
    return int(details[0]["index"]), int(details[0]["quantization"][1])
