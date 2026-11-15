"""The server name spotter: an openWakeWord-style chain of three ONNX graphs behind `NameSpotter`.

Wake by name needs MIST's name scored on the uplink as it arrives. The model is three graphs, a
melspectrogram, a shared speech embedding and a classifier trained for her names, and research
R2a found that each reads a different window: the melspectrogram 1280 new samples behind 480 of
left context, scaled by `x/10 + 2`; the embedding the last 76 mel frames, moving by 8 per hop;
the classifier the last N embeddings, with N read from its own input shape. An ordered list of
files cannot carry that, so the model directory's `manifest.json` states each graph's input
spec, and this module refuses any spec but the one the code implements. A manifest that loads
is one whose scores the code can reproduce.

Weights are never downloaded, never committed, and stay under `<models_dir>/speech/wake/<id>/`,
because the trained weights are for personal use only. Each graph's sha256 is checked at load,
so a file swapped under a manifest is refused rather than scored.

The first `warmup_predictions` predictions after a reset are zeroed (R2a). How openWakeWord
fills its windows before they are full is not in R2a, so two choices here are unverified: a
window not yet full is zero-padded, and its score is never reported; and the melspectrogram is
fed int16-scale samples as float32. T36's parity check on the real model settles them.

onnxruntime is never imported here: the sessions come from the injected `session_factory`, and
the real factory is `silero_vad.open_onnx_session`.
"""

import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol

import numpy as np

from ..budget import PlacementConfig
from ..errors import SpeechEngineError
from ..protocol import Spotter
from ..speech import NameHit, SpotterCalibration
from .silero_vad import CPU_PROVIDER, SessionSpec

logger = logging.getLogger(__name__)

FAMILY = "openwakeword"
SAMPLE_RATE = 16000
HOP_MS = 80
HOP_SAMPLES = SAMPLE_RATE * HOP_MS // 1000
LEFT_CONTEXT_SAMPLES = 480
FRAMES_PER_HOP = 8
MEL_BINS = 32
MEL_SCALE = 0.1
MEL_OFFSET = 2.0
EMBEDDING_FRAMES = 76
EMBEDDING_STRIDE = 8
EMBEDDING_DIMS = 96
ROLES = ("melspectrogram", "embedding", "classifier")

_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_KEYS = frozenset(
    {
        "model_id",
        "names",
        "family",
        "sample_rate",
        "hop_ms",
        "window_ms",
        "threshold",
        "warmup_predictions",
        "graphs",
        "hit_lag_ms",
        "calibration",
        "training",
    }
)
_GRAPH_COMMON = {"role", "file", "sha256", "input"}
_EXPECTED_GRAPHS: dict[str, dict[str, object]] = {
    "melspectrogram": {
        "input": {"new_samples": HOP_SAMPLES, "left_context_samples": LEFT_CONTEXT_SAMPLES},
        "frames_per_hop": FRAMES_PER_HOP,
        "bins": MEL_BINS,
        "output_transform": {"scale": MEL_SCALE, "offset": MEL_OFFSET},
    },
    "embedding": {
        "input": {"window_frames": EMBEDDING_FRAMES, "stride_frames": EMBEDDING_STRIDE},
        "dims": EMBEDDING_DIMS,
    },
    "classifier": {"input": {"window_embeddings": "from_model"}},
}
"""The one input spec per role the pipeline implements; a manifest must state exactly this."""
_SPOTTERS: tuple[Spotter, ...] = ("server", "endpoint")
_CALIBRATION_KEYS = frozenset({"false_accepts", "positive_quantile"})


class ShapedInput(Protocol):
    """A model input with its shape, which the classifier's window is read from.

    A symbolic dimension is a string or `None`, which is why the window check refuses both.
    """

    @property
    def name(self) -> str:
        """The input's name, which every feed is keyed by."""
        ...

    @property
    def shape(self) -> Sequence[object]:
        """The input's declared shape."""
        ...


class KwsSession(Protocol):
    """The part of `onnxruntime.InferenceSession` the spotter uses, so tests can fake it."""

    def get_inputs(self) -> Sequence[ShapedInput]:
        """The model's inputs; the spotter feeds the first."""
        ...

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, np.ndarray]
    ) -> Sequence[np.ndarray]:
        """Run the model; `None` for `output_names` returns every output in model order."""
        ...


KwsSessionFactory = Callable[[SessionSpec], KwsSession]


@dataclass(frozen=True, slots=True)
class WakeGraph:
    """One checked graph file of the chain, in its role's place."""

    role: str
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class WakeManifest:
    """A model directory's `manifest.json` after validation.

    `training` is the sha256 of the training-split manifest the model was trained from, kept so
    that H11 can check which split a measured model came from.
    """

    path: str
    model_id: str
    names: tuple[str, ...]
    window_ms: int
    threshold: float
    warmup_predictions: int
    graphs: tuple[WakeGraph, ...]
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


def _graph(reader: _Reader, value: object, index: int, directory: Path) -> WakeGraph:
    where = f"graphs[{index}]"
    role = ROLES[index]
    if not isinstance(value, dict) or value.get("role") != role:
        found = value.get("role") if isinstance(value, dict) else value
        raise reader.fail(f"{where} must be the {role} graph (order {list(ROLES)}), got {found!r}")
    expected = _EXPECTED_GRAPHS[role]
    entry = reader.keys(value, where, _GRAPH_COMMON | set(expected))
    for key, spec in expected.items():
        if entry[key] != spec or not _same_types(entry[key], spec):
            raise reader.fail(f"{where} ({role}) {key} must be {spec!r}, got {entry[key]!r}")
    file = entry["file"]
    if not isinstance(file, str) or not file:
        raise reader.fail(f"{where}.file must be a relative path")
    relative = PurePosixPath(file)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise reader.fail(f"{where}.file {file!r} must stay inside {directory}")
    digest = entry["sha256"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise reader.fail(f"{where}.sha256 must be 64 lower-case hex digits")
    path = directory.joinpath(*relative.parts)
    if not path.is_file():
        raise reader.fail(f"{role} graph {path} not found")
    actual = _sha256(path)
    if actual != digest:
        raise reader.fail(f"{role} graph {path} has sha256 {actual}, the manifest gives {digest}")
    return WakeGraph(role=role, path=str(path), sha256=digest)


def _same_types(value: object, spec: object) -> bool:
    """Equal values of the wrong JSON type (`true` for 1, `1` for "1") are still refused."""
    if isinstance(spec, dict):
        return isinstance(value, dict) and all(
            _same_types(value[key], item) for key, item in spec.items()
        )
    if isinstance(spec, float):
        return _is_number(value)
    if isinstance(spec, int):
        return _is_int(value)
    return type(value) is type(spec)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path, model_id: str) -> WakeManifest:
    """Read and check `<models_dir>/speech/wake/<model_id>/manifest.json`.

    Every key is required and an unknown key is refused, so a field the code does not read
    cannot be mistaken for one it honours.

    Raises:
        SpeechEngineError: The manifest or a graph file is missing, a sha256 differs, or a
            value is outside what the openWakeWord pipeline here implements. The message starts
            with the manifest's path, and names the graph file where one is at fault.
    """
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
    if not _is_int(manifest["sample_rate"]) or manifest["sample_rate"] != SAMPLE_RATE:
        raise reader.fail(f"sample_rate must be {SAMPLE_RATE}, got {manifest['sample_rate']!r}")
    if not _is_int(manifest["hop_ms"]) or manifest["hop_ms"] != HOP_MS:
        raise reader.fail(f"hop_ms must be {HOP_MS} for {FAMILY}, got {manifest['hop_ms']!r}")
    window_ms = reader.integer(manifest["window_ms"], "window_ms", 1)
    if not _is_number(manifest["threshold"]):
        raise reader.fail(f"threshold must be a number, got {manifest['threshold']!r}")
    warmup = reader.integer(manifest["warmup_predictions"], "warmup_predictions", 0)
    hit_lag_ms = reader.integer(manifest["hit_lag_ms"], "hit_lag_ms", 0)
    graphs = manifest["graphs"]
    if not isinstance(graphs, list) or len(graphs) != len(ROLES):
        raise reader.fail(f"graphs must list the {len(ROLES)} graphs {list(ROLES)} in order")
    checked = tuple(_graph(reader, graph, index, path.parent) for index, graph in enumerate(graphs))
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
    return WakeManifest(
        path=str(path),
        model_id=model_id,
        names=tuple(names),
        window_ms=window_ms,
        threshold=float(manifest["threshold"]),
        warmup_predictions=warmup,
        graphs=checked,
        hit_lag_ms=hit_lag_ms,
        calibration=MappingProxyType(table),
        training=training,
    )


# -- The pipeline -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Chain:
    """The three loaded sessions and the names of the inputs each is fed through."""

    mel: KwsSession
    mel_input: str
    embedding: KwsSession
    embedding_input: str
    classifier: KwsSession
    classifier_input: str
    window: int
    warmup: int
    model_id: str


class _HopScorer:
    """One context's buffers: audio left context, mel frames and embeddings.

    The buffers start as zeros, so a window not yet full reads as zero-padded; `step` reports
    0.0 for such a hop, and for each warm-up prediction, so neither can reach a threshold.
    """

    __slots__ = (
        "_chain",
        "_context",
        "_embeddings",
        "_full_embeddings",
        "_mel",
        "_mel_frames",
        "_predictions",
    )

    def __init__(self, chain: _Chain) -> None:
        self._chain = chain
        self._context = np.zeros(LEFT_CONTEXT_SAMPLES, dtype=np.float32)
        self._mel = np.zeros((EMBEDDING_FRAMES, MEL_BINS), dtype=np.float32)
        self._embeddings = np.zeros((chain.window, EMBEDDING_DIMS), dtype=np.float32)
        self._mel_frames = 0
        self._full_embeddings = 0
        self._predictions = 0

    def step(self, hop: np.ndarray) -> float:
        """Run the chain once over one 1280-sample hop and return its reportable score.

        The melspectrogram is fed int16-scale float32 samples (unverified; see the module).
        """
        chain = self._chain
        framed = np.concatenate([self._context, hop.astype(np.float32)])[np.newaxis, :]
        self._context = framed[0, -LEFT_CONTEXT_SAMPLES:]
        frames = self._output(chain.mel.run(None, {chain.mel_input: framed}), "melspectrogram")
        if frames.size != FRAMES_PER_HOP * MEL_BINS:
            raise SpeechEngineError(
                f"wake model {chain.model_id}: the melspectrogram gave {frames.size} values per "
                f"hop, expected {FRAMES_PER_HOP} frames of {MEL_BINS} bins"
            )
        frames = frames.reshape(FRAMES_PER_HOP, MEL_BINS) * MEL_SCALE + MEL_OFFSET
        self._mel = np.concatenate([self._mel[FRAMES_PER_HOP:], frames])
        self._mel_frames += FRAMES_PER_HOP

        window = self._mel[np.newaxis, :, :, np.newaxis]
        vector = self._output(
            chain.embedding.run(None, {chain.embedding_input: window}), "embedding"
        )
        if vector.size != EMBEDDING_DIMS:
            raise SpeechEngineError(
                f"wake model {chain.model_id}: the embedding gave {vector.size} values, "
                f"expected {EMBEDDING_DIMS}"
            )
        self._embeddings = np.concatenate([self._embeddings[1:], vector[np.newaxis, :]])
        if self._mel_frames >= EMBEDDING_FRAMES:
            self._full_embeddings += 1

        features = self._embeddings[np.newaxis, :, :]
        scores = self._output(
            chain.classifier.run(None, {chain.classifier_input: features}), "classifier"
        )
        if scores.size != 1:
            raise SpeechEngineError(
                f"wake model {chain.model_id}: the classifier gave {scores.size} scores, expected 1"
            )
        self._predictions += 1
        if self._predictions <= chain.warmup or self._full_embeddings < chain.window:
            return 0.0
        return float(scores[0])

    def _output(self, outputs: Sequence[np.ndarray], role: str) -> np.ndarray:
        if not outputs:
            raise SpeechEngineError(f"wake model {self._chain.model_id}: the {role} gave no output")
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)


def _samples(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:
        raise SpeechEngineError(f"name spotter takes PCM16, got {len(pcm)} bytes")
    return np.frombuffer(pcm, dtype="<i2")


class OnnxNameSpotterStream:
    """One audio source's spotting state: the pending samples, the windows, and the edge.

    A hit is a rising edge, the score reaching the threshold from below, so a name held across
    several hops reports once. After a hit, no other hit is reported for `window_ms` of audio,
    because the classifier window still holds the same name until then.
    """

    __slots__ = (
        "_chain",
        "_hops_since_hit",
        "_keyword",
        "_pending",
        "_previous",
        "_refractory_hops",
        "_scorer",
        "_threshold",
    )

    def __init__(self, chain: _Chain, threshold: float, keyword: str, window_ms: int) -> None:
        self._chain = chain
        self._threshold = threshold
        self._keyword = keyword
        self._refractory_hops = -(-window_ms // HOP_MS)
        self.reset()

    def push(self, pcm: bytes, pos: int) -> list[NameHit]:
        """Buffer PCM16 mono 16 kHz audio and score each hop it completes.

        A 20 ms frame is 320 samples, so a hop completes on every fourth; the samples past the
        last whole hop wait for the next push. When one push completes several hops the
        classifier runs once per hop and the push scores the highest, so a hit is judged once
        per push however the frames arrive. A hit's `end_pos` is `pos`, the frame the hop ended on.

        Raises:
            SpeechEngineError: The PCM is not whole 16-bit samples, or a graph's output has an
                unexpected size.
        """
        pending = np.concatenate([self._pending, _samples(pcm)])
        scores: list[float] = []
        while len(pending) >= HOP_SAMPLES:
            scores.append(self._scorer.step(pending[:HOP_SAMPLES]))
            pending = pending[HOP_SAMPLES:]
            if self._hops_since_hit is not None:
                self._hops_since_hit += 1
        self._pending = pending
        if not scores:
            return []
        score = max(scores)
        rising = score >= self._threshold > self._previous
        self._previous = score
        refractory = (
            self._hops_since_hit is not None and self._hops_since_hit < self._refractory_hops
        )
        if not rising or refractory:
            return []
        self._hops_since_hit = 0
        return [NameHit(end_pos=pos, keyword=self._keyword, score=score)]

    def reset(self) -> None:
        """Drop the pending samples, the windows, the warm-up count and the edge state."""
        self._scorer = _HopScorer(self._chain)
        self._pending = np.zeros(0, dtype="<i2")
        self._previous = 0.0
        self._hops_since_hit: int | None = None


class OnnxNameSpotter:
    """The openWakeWord-style spotter over three onnxruntime sessions that every stream shares.

    `keyword` on each hit is the manifest's first name: the classifier gives one score for the
    model, not one per name. `threshold` is the manifest's default, which the wake settings may
    override per stream.
    """

    __slots__ = (
        "_chain",
        "calibration",
        "classifier_window",
        "hit_lag_ms",
        "model_id",
        "names",
        "threshold",
        "training",
        "warmup_predictions",
        "window_ms",
    )

    def __init__(self, manifest: WakeManifest, sessions: Sequence[KwsSession]) -> None:
        mel, embedding, classifier = sessions
        classifier_path = manifest.graphs[2].path
        window = _classifier_window(classifier, manifest.path, classifier_path)
        self._chain = _Chain(
            mel=mel,
            mel_input=_first_input(mel, manifest.path, manifest.graphs[0].path).name,
            embedding=embedding,
            embedding_input=_first_input(embedding, manifest.path, manifest.graphs[1].path).name,
            classifier=classifier,
            classifier_input=_first_input(classifier, manifest.path, classifier_path).name,
            window=window,
            warmup=manifest.warmup_predictions,
            model_id=manifest.model_id,
        )
        self.model_id = manifest.model_id
        self.names = manifest.names
        self.window_ms = manifest.window_ms
        self.hit_lag_ms = manifest.hit_lag_ms
        self.calibration = manifest.calibration
        self.threshold = manifest.threshold
        self.warmup_predictions = manifest.warmup_predictions
        self.training = manifest.training
        self.classifier_window = window

    @classmethod
    def load(
        cls, placement: PlacementConfig, *, session_factory: KwsSessionFactory
    ) -> "OnnxNameSpotter":
        """Open `placement.kws_model` from `<models_dir>/speech/wake/<id>/` on the CPU.

        Each session runs `placement.kws_threads` intra-op threads and one inter-op thread:
        the chain is three single-path graphs run in turn.

        Raises:
            SpeechEngineError: No model is set, the id is malformed, the manifest is refused
                (`read_manifest`), or the classifier's window is not a positive integer.
        """
        model_id = placement.kws_model
        if model_id is None:
            raise SpeechEngineError("MIST_VOICE_KWS_MODEL is not set; the server spotter is off")
        if not _ID.fullmatch(model_id):
            raise SpeechEngineError(
                f"MIST_VOICE_KWS_MODEL={model_id!r}: expected a model id [a-z0-9][a-z0-9-]*"
            )
        path = Path(placement.models_dir, "speech", "wake", model_id, "manifest.json")
        manifest = read_manifest(path, model_id)
        sessions = [
            session_factory(
                SessionSpec(
                    path=graph.path,
                    providers=(CPU_PROVIDER,),
                    device_id=0,
                    intra_op_num_threads=placement.kws_threads,
                    inter_op_num_threads=1,
                )
            )
            for graph in manifest.graphs
        ]
        spotter = cls(manifest, sessions)
        logger.info(
            "Name spotter %s loaded from %s with %d threads, classifier window %d",
            model_id,
            path.parent,
            placement.kws_threads,
            spotter.classifier_window,
        )
        return spotter

    def open_stream(self, threshold: float) -> OnnxNameSpotterStream:
        """Open a stream with fresh context that reports scores at or above `threshold`."""
        return OnnxNameSpotterStream(self._chain, threshold, self.names[0], self.window_ms)

    def score_window(self, pcm: bytes) -> float:
        """The highest reportable score over a finished buffer, from a fresh context.

        This is the stream's pipeline, warm-up and fill rules included, so a buffer shorter
        than the windows scores 0.0; samples past the last whole hop are not scored.

        Raises:
            SpeechEngineError: The PCM is not whole 16-bit samples, or a graph's output has an
                unexpected size.
        """
        samples = _samples(pcm)
        scorer = _HopScorer(self._chain)
        best = 0.0
        for start in range(0, len(samples) - HOP_SAMPLES + 1, HOP_SAMPLES):
            best = max(best, scorer.step(samples[start : start + HOP_SAMPLES]))
        return best


def _first_input(session: KwsSession, manifest: str, graph: str) -> ShapedInput:
    inputs = session.get_inputs()
    if not inputs:
        raise SpeechEngineError(f"{manifest}: graph {graph} has no inputs")
    return inputs[0]


def _classifier_window(session: KwsSession, manifest: str, graph: str) -> int:
    """N, the embeddings one classifier call reads, from its first input's second dimension."""
    shape = list(_first_input(session, manifest, graph).shape)
    window = shape[1] if len(shape) > 1 else None
    if not _is_int(window) or window < 1:
        raise SpeechEngineError(
            f"{manifest}: classifier graph {graph} input shape {shape} has no positive integer "
            "window in its second dimension"
        )
    return window
