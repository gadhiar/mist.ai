"""The acoustic wake verifier behind `WakeVerifier`: a second model that scores a name hit's audio.

The name spotter answers only "was the name heard", so "Mist, what's the time" and "the mist
rolled in" both hit it. The verifier scores the audio around the hit, by default from 2000 ms
before it to 300 ms after it, which holds the name, what came before it and whether the same
voice runs on past it. A hit is accepted when that score reaches the manifest's threshold for
the spotter cutoff that produced it (`threshold_at`).

The window (`verify_window`) and the PCM16 conversion (`pcm_to_waveform`) are defined here once,
for training, the evaluation harness and ingress to share, so the model is served the audio it
is trained on: never padded, never normalised, int16 divided by 32768.

Weights are never downloaded. `<models_dir>/speech/wake/<id>/` holds `model.onnx`,
`manifest.json` and `provenance.json`, staged by hand beside the spotter they were trained
against, and `load` refuses a verifier whose spotter's model file is not the one installed.
onnxruntime is reached only through the session factory and provider lister `load` is handed
(`silero_vad.open_onnx_session` and `silero_vad.onnxruntime_providers`), so importing this
module loads neither onnxruntime nor torch, and the unit tier injects fakes.
"""

import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..budget import PlacementConfig
from ..errors import SpeechEngineError
from .kws import read_manifest
from .silero_vad import (
    CPU_PROVIDER,
    OnnxSession,
    ProviderLister,
    SessionFactory,
    SessionSpec,
    open_checked_session,
)

logger = logging.getLogger(__name__)

VERSION = 1
KIND = "wake_verifier"
SAMPLE_RATE = 16000
FRAME_MS = 20
"""One uplink position: the window's bounds are whole frames."""
MODEL_FILE = "model.onnx"
PROVENANCE_FILE = "provenance.json"
PCM_SCALE = 32768.0
"""int16 to float by 1/32768, the conversion the model is trained with; not 32767."""
GRID: tuple[float, ...] = tuple(step / 100 for step in range(1, 100))
"""The threshold table's cutoffs, 0.01 to 0.99 by 0.01, one row each."""
VERIFY_INPUTS = frozenset({"waveform"})

_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KEYS = frozenset(
    {
        "version",
        "kind",
        "model_id",
        "file",
        "sha256",
        "sample_rate",
        "frame_ms",
        "before_ms",
        "after_ms",
        "min_window_ms",
        "spotter",
        "thresholds",
        "training",
        "provenance_sha256",
    }
)
_SPOTTER_KEYS = frozenset({"model_id", "model_sha256"})
_THRESHOLD_KEYS = frozenset({"cutoffs", "scores", "per_hour", "hours"})
_EPSILON = 1e-9
"""Keeps a grid cutoff whose product with 100 lands a hair under its integer on its own row."""


@dataclass(frozen=True, slots=True)
class VerifierManifest:
    """A verifier directory's `manifest.json` after validation.

    `model_file` is the model's full path and `model_sha256` its checked digest. The spotter
    fields name the spotter and the model file the verifier was trained and calibrated against.
    `scores` is the threshold table, one row per `cutoffs` entry, measured over `hours` of
    calibration audio to pass at most `per_hour` hits an hour. `training` is the sha256 of the
    training split, and `provenance_sha256` the checked digest of `provenance.json`.
    """

    path: str
    model_id: str
    model_file: str
    model_sha256: str
    sample_rate: int
    frame_ms: int
    before_ms: int
    after_ms: int
    min_window_ms: int
    spotter_model_id: str
    spotter_model_sha256: str
    cutoffs: tuple[float, ...]
    scores: tuple[float, ...]
    per_hour: float
    hours: float
    training: str
    provenance_sha256: str


# -- Manifest validation ----------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class _Checks:
    """Validation helpers that prefix every refusal with the manifest's path."""

    __slots__ = ("path",)

    def __init__(self, path: Path) -> None:
        self.path = path

    def fail(self, reason: str) -> SpeechEngineError:
        return SpeechEngineError(f"{self.path}: {reason}")

    def keys(self, value: object, where: str, expected: frozenset[str]) -> dict:
        if not isinstance(value, dict):
            raise self.fail(f"{where} must be an object")
        unknown = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        if unknown:
            raise self.fail(f"{where} has unknown keys {unknown}")
        if missing:
            raise self.fail(f"{where} is missing keys {missing}")
        return value

    def exactly(self, value: object, where: str, expected: int) -> int:
        if not _is_int(value) or value != expected:
            raise self.fail(f"{where} must be {expected}, got {value!r}")
        return expected

    def window_ms(self, value: object, where: str, minimum: int) -> int:
        if not _is_int(value) or value < minimum or value % FRAME_MS:
            raise self.fail(
                f"{where} must be an integer multiple of {FRAME_MS} of at least {minimum}, "
                f"got {value!r}"
            )
        return value

    def model_id(self, value: object, where: str) -> str:
        if not isinstance(value, str) or not _ID.fullmatch(value):
            raise self.fail(f"{where} must be a model id [a-z0-9][a-z0-9-]*, got {value!r}")
        return value

    def sha256(self, value: object, where: str) -> str:
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise self.fail(f"{where} must be 64 lower-case hex digits")
        return value

    def amount(self, value: object, where: str) -> float:
        if not _is_number(value) or value < 0:
            raise self.fail(f"{where} must be a finite number of at least 0, got {value!r}")
        return float(value)

    def digest_of(self, path: Path, what: str, expected: str, key: str) -> None:
        try:
            with path.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
        except FileNotFoundError as exc:
            raise self.fail(f"{what} {path} not found") from exc
        except OSError as exc:
            raise self.fail(f"{what} {path} could not be read: {exc}") from exc
        if actual != expected:
            raise self.fail(
                f"{what} {path} has sha256 {actual}, the manifest's {key} is {expected}"
            )


def _table(checks: _Checks, thresholds: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The checked cutoffs and scores: the grid, and finite scores in [0, 1] not increasing."""
    cutoffs = thresholds["cutoffs"]
    if not (
        isinstance(cutoffs, list)
        and len(cutoffs) == len(GRID)
        and all(_is_number(c) and float(c) == g for c, g in zip(cutoffs, GRID))
    ):
        raise checks.fail("thresholds.cutoffs must be 0.01 to 0.99 by 0.01, in order")
    scores = thresholds["scores"]
    if not (isinstance(scores, list) and len(scores) == len(GRID)):
        raise checks.fail(f"thresholds.scores must hold {len(GRID)} scores, one per cutoff")
    if not all(_is_number(score) and 0.0 <= score <= 1.0 for score in scores):
        raise checks.fail("thresholds.scores must be finite numbers from 0 to 1")
    if any(later > earlier for earlier, later in zip(scores, scores[1:])):
        raise checks.fail("thresholds.scores must not increase with the cutoff")
    return GRID, tuple(float(score) for score in scores)


def read_verifier_manifest(path: Path, model_id: str) -> VerifierManifest:
    """Read and check `<models_dir>/speech/wake/<model_id>/manifest.json` and its two files.

    Every key is required and an unknown key is refused, so a field the code does not read
    cannot be mistaken for one it honours. The model file is hashed last, after every cheaper
    check, because it is the large one.

    Raises:
        SpeechEngineError: The manifest, the model or `provenance.json` is missing, a digest
            differs, or a value is outside what this verifier implements. The message starts
            with the manifest's path and names the key or check that failed.
    """
    checks = _Checks(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise checks.fail("verifier manifest not found") from exc
    except OSError as exc:
        raise checks.fail(f"verifier manifest could not be read: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise checks.fail(f"manifest is not valid JSON: {exc}") from exc
    manifest = checks.keys(raw, "manifest", _KEYS)
    checks.exactly(manifest["version"], "version", VERSION)
    if manifest["kind"] != KIND:
        raise checks.fail(f"kind must be {KIND!r}, got {manifest['kind']!r}")
    checks.model_id(manifest["model_id"], "model_id")
    if manifest["model_id"] != model_id:
        raise checks.fail(f"model_id {manifest['model_id']!r} is not the directory's {model_id!r}")
    if manifest["file"] != MODEL_FILE:
        raise checks.fail(f"file must be {MODEL_FILE!r}, got {manifest['file']!r}")
    model_sha256 = checks.sha256(manifest["sha256"], "sha256")
    sample_rate = checks.exactly(manifest["sample_rate"], "sample_rate", SAMPLE_RATE)
    frame_ms = checks.exactly(manifest["frame_ms"], "frame_ms", FRAME_MS)
    before_ms = checks.window_ms(manifest["before_ms"], "before_ms", 0)
    after_ms = checks.window_ms(manifest["after_ms"], "after_ms", 0)
    min_window_ms = checks.window_ms(manifest["min_window_ms"], "min_window_ms", FRAME_MS)
    if min_window_ms > before_ms + after_ms + FRAME_MS:
        raise checks.fail(
            f"min_window_ms {min_window_ms} is longer than the longest window, "
            f"before_ms + after_ms + {FRAME_MS} = {before_ms + after_ms + FRAME_MS}"
        )
    spotter = checks.keys(manifest["spotter"], "spotter", _SPOTTER_KEYS)
    spotter_id = checks.model_id(spotter["model_id"], "spotter.model_id")
    spotter_sha256 = checks.sha256(spotter["model_sha256"], "spotter.model_sha256")
    thresholds = checks.keys(manifest["thresholds"], "thresholds", _THRESHOLD_KEYS)
    cutoffs, scores = _table(checks, thresholds)
    per_hour = checks.amount(thresholds["per_hour"], "thresholds.per_hour")
    hours = checks.amount(thresholds["hours"], "thresholds.hours")
    training = checks.sha256(manifest["training"], "training")
    provenance_sha256 = checks.sha256(manifest["provenance_sha256"], "provenance_sha256")
    directory = path.parent
    checks.digest_of(
        directory / PROVENANCE_FILE, "provenance", provenance_sha256, "provenance_sha256"
    )
    model_path = directory / MODEL_FILE
    checks.digest_of(model_path, "verifier model", model_sha256, "sha256")
    return VerifierManifest(
        path=str(path),
        model_id=model_id,
        model_file=str(model_path),
        model_sha256=model_sha256,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        before_ms=before_ms,
        after_ms=after_ms,
        min_window_ms=min_window_ms,
        spotter_model_id=spotter_id,
        spotter_model_sha256=spotter_sha256,
        cutoffs=cutoffs,
        scores=scores,
        per_hour=per_hour,
        hours=hours,
        training=training,
        provenance_sha256=provenance_sha256,
    )


# -- The window, the conversion and the table -------------------------------------------------


def verify_window(
    has: Callable[[int], bool],
    hit_pos: int,
    *,
    before_frames: int,
    after_frames: int,
    min_frames: int,
) -> range | None:
    """The positions to score for a hit: the run of held positions around it, clipped.

    The window is the longest run of contiguous positions `p` with `has(p)` that contains
    `hit_pos`, clipped to `[hit_pos - before_frames, hit_pos + after_frames]`. It is never
    padded: a clip edge, a lost frame or a mute shortens it from that side, and what `has`
    means is the caller's. None when `hit_pos` itself is not held, or the window is shorter
    than `min_frames`. No position outside the clip bounds is asked about.

    Raises:
        ValueError: A frame count is negative.
    """
    if min(before_frames, after_frames, min_frames) < 0:
        raise ValueError("verify_window: frame counts must not be negative")
    if not has(hit_pos):
        return None
    first = hit_pos
    while first > hit_pos - before_frames and has(first - 1):
        first -= 1
    last = hit_pos
    while last < hit_pos + after_frames and has(last + 1):
        last += 1
    window = range(first, last + 1)
    return window if len(window) >= min_frames else None


def pcm_to_waveform(pcm: bytes) -> np.ndarray:
    """PCM16 mono audio as the model's input: float32 of shape `[1, n]`, int16 over 32768.

    Raises:
        SpeechEngineError: The byte count is odd, so the buffer is not whole samples.
    """
    if len(pcm) % 2:
        raise SpeechEngineError(f"PCM16 audio must be whole samples, got an odd {len(pcm)} bytes")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    return (samples / np.float32(PCM_SCALE)).reshape(1, -1)


def threshold_at(table: Sequence[float], cutoff: float) -> float | None:
    """The table's threshold for a spotter cutoff, or None for a cutoff below 0.01 or NaN.

    The cutoff floors to the grid, `floor(100 * cutoff + 1e-9) / 100`, capped at 0.99: the row
    at or below the cutoff, whose threshold is at least as strict since the table does not
    increase. The device's 0.85 cutoff arrives as 216/255 = 0.847 and reads the 0.84 row.

    Raises:
        ValueError: The table does not have one row per grid cutoff.
    """
    if len(table) != len(GRID):
        raise ValueError(f"a threshold table has {len(GRID)} rows, got {len(table)}")
    if math.isnan(cutoff) or cutoff <= 0:
        return None
    row = len(GRID) if math.isinf(cutoff) else min(math.floor(100 * cutoff + _EPSILON), len(GRID))
    if row < 1:
        return None
    return float(table[row - 1])


# -- The engine -------------------------------------------------------------------------------


class OnnxWakeVerifier:
    """The verifier's ONNX graph over one onnxruntime session, on the CPU.

    The graph takes `waveform`, float32 `[1, n]` with `n` dynamic, and returns one score. A
    session's `run` is safe to call from several threads, and the model keeps no state between
    windows, so one loaded verifier serves every session.
    """

    __slots__ = (
        "_session",
        "after_ms",
        "before_ms",
        "manifest",
        "min_window_ms",
        "model_id",
        "spotter_model_id",
    )

    def __init__(self, manifest: VerifierManifest, session: OnnxSession) -> None:
        self._session: OnnxSession | None = session
        self.manifest = manifest
        self.model_id = manifest.model_id
        self.spotter_model_id = manifest.spotter_model_id
        self.before_ms = manifest.before_ms
        self.after_ms = manifest.after_ms
        self.min_window_ms = manifest.min_window_ms

    @classmethod
    def load(
        cls,
        placement: PlacementConfig,
        *,
        session_factory: SessionFactory,
        available_providers: ProviderLister,
    ) -> "OnnxWakeVerifier":
        """Open `placement.verify_model` from `<models_dir>/speech/wake/<id>/` on the CPU.

        The manifest and both files are checked first, then the spotter it names: its
        `spotter.model_sha256` must be the installed spotter's model sha256, since the
        threshold table was calibrated on that model's hits. Only then is a session opened,
        with `placement.verify_threads` intra-op threads and one inter-op thread.

        Raises:
            SpeechEngineError: No model is set, the id is malformed, the thread count is below
                one, the manifest is refused (`read_verifier_manifest`), the spotter's manifest
                is refused (`kws.read_manifest`) or holds another model, onnxruntime is missing
                or has no CPU provider, or the model's inputs are not `waveform`.
        """
        model_id = placement.verify_model
        if model_id is None:
            raise SpeechEngineError("MIST_VOICE_VERIFY_MODEL is not set; the wake verifier is off")
        if not _ID.fullmatch(model_id):
            raise SpeechEngineError(
                f"MIST_VOICE_VERIFY_MODEL={model_id!r}: expected a model id [a-z0-9][a-z0-9-]*"
            )
        if placement.verify_threads < 1:
            raise SpeechEngineError(
                f"MIST_VOICE_VERIFY_THREADS={placement.verify_threads}: expected at least 1"
            )
        wake = Path(placement.models_dir, "speech", "wake")
        manifest = read_verifier_manifest(wake / model_id / "manifest.json", model_id)
        spotter_id = manifest.spotter_model_id
        spotter = read_manifest(wake / spotter_id / "manifest.json", spotter_id)
        if spotter.model_sha256 != manifest.spotter_model_sha256:
            raise SpeechEngineError(
                f"{manifest.path}: the verifier was trained against spotter {spotter_id} with "
                f"model sha256 {manifest.spotter_model_sha256}, but the installed {spotter_id} "
                f"has {spotter.model_sha256}"
            )
        if CPU_PROVIDER not in available_providers():
            raise SpeechEngineError(
                f"this onnxruntime build has no {CPU_PROVIDER}, which the wake verifier runs on"
            )
        spec = SessionSpec(
            path=manifest.model_file,
            providers=(CPU_PROVIDER,),
            device_id=0,
            intra_op_num_threads=placement.verify_threads,
            inter_op_num_threads=1,
        )
        session = open_checked_session(
            spec, session_factory, VERIFY_INPUTS, "MIST_VOICE_VERIFY_MODEL"
        )
        logger.info(
            "Wake verifier %s loaded from %s for spotter %s with %d threads",
            model_id,
            manifest.model_file,
            spotter_id,
            placement.verify_threads,
        )
        return cls(manifest, session)

    def threshold_at(self, cutoff: float) -> float | None:
        """The manifest table's threshold for a spotter cutoff; see the module's `threshold_at`."""
        return threshold_at(self.manifest.scores, cutoff)

    def score(self, pcm: bytes) -> float:
        """Score one window of PCM16 mono 16 kHz audio.

        Raises:
            SpeechEngineError: The verifier is closed, the window is empty or not whole
                samples, or the model's output is not one finite score.
        """
        session = self._session
        if session is None:
            raise SpeechEngineError(f"wake verifier {self.model_id!r} is closed")
        waveform = pcm_to_waveform(pcm)
        if waveform.shape[1] == 0:
            raise SpeechEngineError(f"wake verifier {self.model_id!r} was given an empty window")
        outputs = session.run(None, {"waveform": waveform})
        values = np.asarray(outputs[0]).reshape(-1) if len(outputs) == 1 else np.array([])
        if values.size != 1:
            raise SpeechEngineError(
                f"wake verifier {self.model_id!r} must output one score, got "
                f"{len(outputs)} outputs of {values.size} values"
            )
        value = float(values[0])
        if not math.isfinite(value):
            raise SpeechEngineError(
                f"wake verifier {self.model_id!r} output a score that is not finite: {value}"
            )
        return value

    def close(self) -> None:
        """Drop the session, so every later `score` raises; a second call does nothing."""
        self._session = None
