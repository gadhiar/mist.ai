"""Tests for the acoustic wake verifier over a fake onnxruntime session and staged placeholder files.

The unit tier has no onnxruntime, no torch and no verifier weights, so every test writes a
manifest, a placeholder `model.onnx` and a `provenance.json` under a temporary `models_dir`, beside
a name spotter's manifest, and hands `OnnxWakeVerifier.load` a fake session factory. That pins what
the module owns: the manifest it accepts and refuses, the check that the installed spotter is the
one the verifier was trained against, the session it asks for, the window it cuts around a hit,
the PCM16 conversion it feeds the model, and the threshold row it reads for a spotter cutoff. The
real model's parity with torch is H38's to measure.
"""

import ast
import hashlib
import importlib
import inspect
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import PlacementConfig
from backend.realtime.engines import verifier
from backend.realtime.engines.silero_vad import SessionSpec
from backend.realtime.engines.verifier import (
    OnnxWakeVerifier,
    VerifierManifest,
    pcm_to_waveform,
    read_verifier_manifest,
    threshold_at,
    verify_window,
)
from backend.realtime.errors import SpeechEngineError
from backend.realtime.speech import WakeVerifier

MODEL_ID = "mist-verify-test"
SPOTTER_ID = "mist-mww-test"
MODEL_BYTES = b"placeholder verifier graph"
SPOTTER_BYTES = b"placeholder streaming spotter"
PROVENANCE_BYTES = b'{"version": 1}'
CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
GRID = [step / 100 for step in range(1, 100)]
SCORES = [round(0.95 - 0.004 * index, 6) for index in range(99)]
"""A table that falls by 0.004 a row, so each row's threshold is distinct."""


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# -- Staged files -----------------------------------------------------------------------------


def wake_dir(models_dir: Path, model_id: str) -> Path:
    return models_dir / "speech" / "wake" / model_id


def write_spotter(models_dir: Path, content: bytes = SPOTTER_BYTES) -> Path:
    """Stage a name spotter's manifest and model, as `kws.read_manifest` accepts them."""
    directory = wake_dir(models_dir, SPOTTER_ID)
    (directory / "device").mkdir(parents=True, exist_ok=True)
    (directory / "device" / "mist.tflite").write_bytes(content)
    manifest = {
        "model_id": SPOTTER_ID,
        "names": ["mist"],
        "family": "microwakeword",
        "sample_rate": 16000,
        "feature_step_ms": 10,
        "window_ms": 1000,
        "sliding_window": 5,
        "warmup_ms": 1000,
        "model": {"file": "device/mist.tflite", "sha256": sha256(content)},
        "hit_lag_ms": 120,
        "calibration": {},
        "training": "ab" * 32,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def base_manifest() -> dict[str, object]:
    return {
        "version": 1,
        "kind": "wake_verifier",
        "model_id": MODEL_ID,
        "file": "model.onnx",
        "sha256": sha256(MODEL_BYTES),
        "sample_rate": 16000,
        "frame_ms": 20,
        "before_ms": 2000,
        "after_ms": 300,
        "min_window_ms": 800,
        "spotter": {"model_id": SPOTTER_ID, "model_sha256": sha256(SPOTTER_BYTES)},
        "thresholds": {
            "cutoffs": list(GRID),
            "scores": list(SCORES),
            "per_hour": 0.2,
            "hours": 0.0,
        },
        "training": "cd" * 32,
        "provenance_sha256": sha256(PROVENANCE_BYTES),
    }


def write_verifier(models_dir: Path, mutate: Callable[[dict], None] | None = None) -> Path:
    """Stage the verifier's three files; `mutate` alters the manifest before it is written."""
    directory = wake_dir(models_dir, MODEL_ID)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.onnx").write_bytes(MODEL_BYTES)
    (directory / "provenance.json").write_bytes(PROVENANCE_BYTES)
    manifest = base_manifest()
    if mutate is not None:
        mutate(manifest)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def stage(models_dir: Path, mutate: Callable[[dict], None] | None = None) -> Path:
    write_spotter(models_dir)
    return write_verifier(models_dir, mutate)


def placement(models_dir: Path, **changes: object) -> PlacementConfig:
    return PlacementConfig(models_dir=str(models_dir), verify_model=MODEL_ID, **changes)


# -- Fake session -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FakeInput:
    name: str


@dataclass
class FakeSession:
    """A scripted onnxruntime session: records every feed and answers `outputs`."""

    outputs: Sequence[np.ndarray] = (np.array([0.75], dtype=np.float32),)
    input_names: Sequence[str] = ("waveform",)
    feeds: list[dict[str, np.ndarray]] = field(default_factory=list)

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput(name) for name in self.input_names]

    def get_providers(self) -> list[str]:
        return [CPU]

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names is None
        self.feeds.append({name: np.array(value, copy=True) for name, value in input_feed.items()})
        return list(self.outputs)


@dataclass
class FakeFactory:
    session: FakeSession = field(default_factory=FakeSession)
    specs: list[SessionSpec] = field(default_factory=list)

    def __call__(self, spec: SessionSpec) -> FakeSession:
        self.specs.append(spec)
        return self.session


def only_cpu() -> list[str]:
    return [CPU]


def load(
    place: PlacementConfig,
    session: FakeSession | None = None,
    providers: Callable[[], list[str]] = only_cpu,
) -> tuple[OnnxWakeVerifier, FakeFactory]:
    factory = FakeFactory(session or FakeSession())
    loaded = OnnxWakeVerifier.load(place, session_factory=factory, available_providers=providers)
    return loaded, factory


def pcm(*samples: int) -> bytes:
    return np.array(samples, dtype="<i2").tobytes()


# -- The manifest -----------------------------------------------------------------------------


def _top(key: str, value: object) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        manifest[key] = value

    return mutate


def _drop(key: str) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        del manifest[key]

    return mutate


def _nested(section: str, key: str, value: object) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        manifest[section][key] = value

    return mutate


def _drop_nested(section: str, key: str) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        del manifest[section][key]

    return mutate


def _score(index: int, value: object) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        manifest["thresholds"]["scores"][index] = value

    return mutate


def _cutoff(index: int, value: object) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        manifest["thresholds"]["cutoffs"][index] = value

    return mutate


KEYS = sorted(base_manifest())

REFUSALS = {
    **{f"missing {key}": (_drop(key), key) for key in KEYS},
    "unknown key": (_top("notes", "x"), "notes"),
    "version 2": (_top("version", 2), "version"),
    "version a string": (_top("version", "1"), "version"),
    "version a bool": (_top("version", True), "version"),
    "kind of another model": (_top("kind", "name_spotter"), "kind"),
    "model id not the directory": (_top("model_id", "mist-verify-other"), "model_id"),
    "model id not a string": (_top("model_id", 7), "model_id"),
    "file another name": (_top("file", "weights.onnx"), "file"),
    "file in a subdirectory": (_top("file", "onnx/model.onnx"), "file"),
    "sha256 upper-case": (_top("sha256", "AB" * 32), "sha256"),
    "sha256 short": (_top("sha256", "ab" * 31), "sha256"),
    "sha256 differs": (_top("sha256", "0" * 64), "sha256"),
    "sample rate": (_top("sample_rate", 22050), "sample_rate"),
    "frame length": (_top("frame_ms", 10), "frame_ms"),
    "before not a multiple of 20": (_top("before_ms", 2010), "before_ms"),
    "before negative": (_top("before_ms", -20), "before_ms"),
    "before a float": (_top("before_ms", 2000.0), "before_ms"),
    "after not a multiple of 20": (_top("after_ms", 310), "after_ms"),
    "after a bool": (_top("after_ms", True), "after_ms"),
    "min window not a multiple of 20": (_top("min_window_ms", 810), "min_window_ms"),
    "min window zero": (_top("min_window_ms", 0), "min_window_ms"),
    "min window past the longest window": (_top("min_window_ms", 2340), "min_window_ms"),
    "spotter not an object": (_top("spotter", SPOTTER_ID), "spotter"),
    "spotter without a sha256": (_drop_nested("spotter", "model_sha256"), "spotter"),
    "spotter with an unknown key": (_nested("spotter", "notes", "x"), "spotter"),
    "spotter id malformed": (_nested("spotter", "model_id", "../mist"), "spotter.model_id"),
    "spotter sha256 malformed": (_nested("spotter", "model_sha256", "abc"), "spotter.model_sha256"),
    "thresholds without hours": (_drop_nested("thresholds", "hours"), "thresholds"),
    "thresholds with an unknown key": (_nested("thresholds", "notes", 1), "thresholds"),
    "cutoffs one short": (_nested("thresholds", "cutoffs", GRID[:-1]), "thresholds.cutoffs"),
    "cutoffs off the grid": (_cutoff(41, 0.425), "thresholds.cutoffs"),
    "cutoffs out of order": (
        _nested("thresholds", "cutoffs", [GRID[1], GRID[0], *GRID[2:]]),
        "thresholds.cutoffs",
    ),
    "cutoffs as strings": (_cutoff(0, "0.01"), "thresholds.cutoffs"),
    "scores one short": (_nested("thresholds", "scores", SCORES[:-1]), "thresholds.scores"),
    "score not finite": (_score(10, math.nan), "thresholds.scores"),
    "score above one": (_score(0, 1.5), "thresholds.scores"),
    "score below zero": (_score(98, -0.1), "thresholds.scores"),
    "score a bool": (_score(98, False), "thresholds.scores"),
    "scores increasing": (_score(50, 0.99), "thresholds.scores"),
    "per hour negative": (_nested("thresholds", "per_hour", -0.2), "thresholds.per_hour"),
    "hours a string": (_nested("thresholds", "hours", "9"), "thresholds.hours"),
    "training not a sha256": (_top("training", "abc"), "training"),
    "provenance sha256 malformed": (_top("provenance_sha256", "xyz"), "provenance_sha256"),
    "provenance sha256 differs": (_top("provenance_sha256", "0" * 64), "provenance"),
}


class TestReadVerifierManifest:
    def test_a_good_manifest_reads_into_its_fields(self, tmp_path: Path) -> None:
        path = write_verifier(tmp_path)

        manifest = read_verifier_manifest(path, MODEL_ID)

        assert manifest == VerifierManifest(
            path=str(path),
            model_id=MODEL_ID,
            model_file=str(path.parent / "model.onnx"),
            model_sha256=sha256(MODEL_BYTES),
            sample_rate=16000,
            frame_ms=20,
            before_ms=2000,
            after_ms=300,
            min_window_ms=800,
            spotter_model_id=SPOTTER_ID,
            spotter_model_sha256=sha256(SPOTTER_BYTES),
            cutoffs=tuple(GRID),
            scores=tuple(SCORES),
            per_hour=0.2,
            hours=0.0,
            training="cd" * 32,
            provenance_sha256=sha256(PROVENANCE_BYTES),
        )

    def test_the_manifest_is_frozen_and_slotted(self) -> None:
        params = VerifierManifest.__dataclass_params__

        assert params.frozen
        assert "__slots__" in VerifierManifest.__dict__

    def test_the_longest_min_window_is_the_whole_window(self, tmp_path: Path) -> None:
        path = write_verifier(tmp_path, _top("min_window_ms", 2320))

        assert read_verifier_manifest(path, MODEL_ID).min_window_ms == 2320

    @pytest.mark.parametrize(("mutate", "named"), REFUSALS.values(), ids=REFUSALS.keys())
    def test_a_bad_manifest_is_refused_naming_the_key_or_check(
        self, tmp_path: Path, mutate: Callable[[dict], None], named: str
    ) -> None:
        path = write_verifier(tmp_path, mutate)

        with pytest.raises(SpeechEngineError) as caught:
            read_verifier_manifest(path, MODEL_ID)

        assert str(caught.value).startswith(f"{path}: ")
        assert named in str(caught.value)

    def test_a_model_id_outside_the_id_form_is_refused(self, tmp_path: Path) -> None:
        path = write_verifier(tmp_path, _top("model_id", "Mist_Verify"))

        with pytest.raises(SpeechEngineError, match=r"model_id.*\[a-z0-9\]"):
            read_verifier_manifest(path, "Mist_Verify")

    def test_a_missing_model_file_is_refused_naming_it(self, tmp_path: Path) -> None:
        path = write_verifier(tmp_path)
        (path.parent / "model.onnx").unlink()

        with pytest.raises(SpeechEngineError) as caught:
            read_verifier_manifest(path, MODEL_ID)

        assert str(path.parent / "model.onnx") in str(caught.value)

    def test_a_missing_provenance_file_is_refused_naming_it(self, tmp_path: Path) -> None:
        path = write_verifier(tmp_path)
        (path.parent / "provenance.json").unlink()

        with pytest.raises(SpeechEngineError) as caught:
            read_verifier_manifest(path, MODEL_ID)

        assert str(path.parent / "provenance.json") in str(caught.value)

    def test_a_missing_manifest_is_refused_naming_it(self, tmp_path: Path) -> None:
        path = wake_dir(tmp_path, MODEL_ID) / "manifest.json"

        with pytest.raises(SpeechEngineError) as caught:
            read_verifier_manifest(path, MODEL_ID)

        assert str(caught.value).startswith(f"{path}: ")

    def test_a_manifest_that_is_not_json_is_refused_naming_it(self, tmp_path: Path) -> None:
        path = write_verifier(tmp_path)
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(SpeechEngineError, match="JSON") as caught:
            read_verifier_manifest(path, MODEL_ID)

        assert str(caught.value).startswith(f"{path}: ")


# -- The window -------------------------------------------------------------------------------

HIT = 500
BEFORE = 100
AFTER = 15
MIN = 40


def _present(*runs: range) -> Callable[[int], bool]:
    held = {pos for run in runs for pos in run}
    return held.__contains__


WINDOWS = {
    "a full window": (_present(range(0, 1000)), MIN, range(400, 516)),
    "a gap before the hit": (_present(range(0, 450), range(451, 1000)), MIN, range(451, 516)),
    "a gap after the hit": (_present(range(0, 505), range(506, 1000)), MIN, range(400, 505)),
    "a run under 40 frames": (_present(range(480, 511)), MIN, None),
    "a run of exactly 40 frames": (_present(range(476, 516)), MIN, range(476, 516)),
    "the hit at the run's first frame": (_present(range(500, 1000)), 10, range(500, 516)),
    "the hit at the run's last frame": (_present(range(0, 501)), MIN, range(400, 501)),
    "a missing hit frame": (_present(range(0, 500), range(501, 1000)), MIN, None),
}


class TestVerifyWindow:
    @pytest.mark.parametrize(
        ("has", "min_frames", "expected"), WINDOWS.values(), ids=WINDOWS.keys()
    )
    def test_the_window_is_the_run_holding_the_hit_clipped_to_its_bounds(
        self, has: Callable[[int], bool], min_frames: int, expected: range | None
    ) -> None:
        window = verify_window(
            has, HIT, before_frames=BEFORE, after_frames=AFTER, min_frames=min_frames
        )

        assert window == expected

    def test_the_default_window_is_116_frames(self) -> None:
        window = verify_window(
            _present(range(0, 1000)), HIT, before_frames=BEFORE, after_frames=AFTER, min_frames=MIN
        )

        assert window is not None
        assert len(window) == 116

    def test_positions_outside_the_bounds_are_never_asked_for(self) -> None:
        asked: list[int] = []

        def has(pos: int) -> bool:
            asked.append(pos)
            return True

        verify_window(has, HIT, before_frames=BEFORE, after_frames=AFTER, min_frames=MIN)

        assert min(asked) == HIT - BEFORE
        assert max(asked) == HIT + AFTER


# -- The conversion ---------------------------------------------------------------------------


class TestPcmToWaveform:
    def test_samples_are_divided_by_32768_into_one_float32_row(self) -> None:
        waveform = pcm_to_waveform(pcm(-32768, -1, 0, 1, 16384, 32767))

        assert waveform.dtype == np.float32
        assert waveform.shape == (1, 6)
        np.testing.assert_array_equal(
            waveform[0],
            np.array([-1.0, -1 / 32768, 0.0, 1 / 32768, 0.5, 32767 / 32768], dtype=np.float32),
        )

    def test_an_odd_byte_count_is_refused(self) -> None:
        with pytest.raises(SpeechEngineError, match="odd"):
            pcm_to_waveform(b"\x00\x01\x02")


# -- The threshold table ----------------------------------------------------------------------


class TestThresholdAt:
    @pytest.mark.parametrize("row", range(1, 100))
    def test_each_grid_cutoff_reads_its_own_row(self, row: int) -> None:
        cutoff = json.loads(json.dumps(row / 100))

        assert threshold_at(SCORES, cutoff) == SCORES[row - 1]

    @pytest.mark.parametrize(
        ("cutoff", "row"),
        [
            pytest.param(0.555, 55, id="between-rows"),
            pytest.param(0.5599999, 55, id="just-under-the-next-row"),
            pytest.param(0.01, 1, id="the-first-row"),
            pytest.param(0.0199, 1, id="inside-the-first-row"),
        ],
    )
    def test_a_cutoff_floors_to_the_row_at_or_below_it(self, cutoff: float, row: int) -> None:
        assert threshold_at(SCORES, cutoff) == SCORES[row - 1]

    @pytest.mark.parametrize("cutoff", [0.847, 216 / 255], ids=["0.847", "216/255"])
    def test_the_device_cutoff_reads_the_084_row(self, cutoff: float) -> None:
        assert threshold_at(SCORES, cutoff) == SCORES[83]

    @pytest.mark.parametrize("cutoff", [0.0099, 0.005, 0.0, -0.5, -math.inf])
    def test_a_cutoff_below_001_has_no_threshold(self, cutoff: float) -> None:
        assert threshold_at(SCORES, cutoff) is None

    @pytest.mark.parametrize("cutoff", [0.99, 0.995, 1.0, 3.0, math.inf])
    def test_a_cutoff_at_or_above_099_reads_the_last_row(self, cutoff: float) -> None:
        assert threshold_at(SCORES, cutoff) == SCORES[98]

    def test_a_nan_cutoff_has_no_threshold(self) -> None:
        assert threshold_at(SCORES, math.nan) is None

    def test_a_table_of_another_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="99"):
            threshold_at(SCORES[:-1], 0.5)


# -- Loading ----------------------------------------------------------------------------------


class TestPlacementFields:
    def test_the_verifier_is_off_with_two_threads_by_default(self) -> None:
        place = PlacementConfig()

        assert place.verify_model is None
        assert place.verify_threads == 2

    def test_the_placement_docstring_names_both_fields(self) -> None:
        assert "verify_model" in PlacementConfig.__doc__
        assert "verify_threads" in PlacementConfig.__doc__


class TestLoad:
    def test_load_opens_the_model_on_the_cpu_with_the_verify_threads(self, tmp_path: Path) -> None:
        stage(tmp_path)

        _, factory = load(placement(tmp_path))

        assert factory.specs == [
            SessionSpec(
                path=str(wake_dir(tmp_path, MODEL_ID) / "model.onnx"),
                providers=(CPU,),
                device_id=0,
                intra_op_num_threads=2,
                inter_op_num_threads=1,
            )
        ]

    @pytest.mark.parametrize("threads", [1, 4])
    def test_intra_op_threads_follow_the_placement(self, tmp_path: Path, threads: int) -> None:
        stage(tmp_path)

        _, factory = load(placement(tmp_path, verify_threads=threads))

        assert factory.specs[0].intra_op_num_threads == threads
        assert factory.specs[0].inter_op_num_threads == 1

    def test_a_cuda_capable_runtime_still_runs_the_verifier_on_the_cpu(
        self, tmp_path: Path
    ) -> None:
        stage(tmp_path)

        _, factory = load(placement(tmp_path), providers=lambda: [CUDA, CPU])

        assert factory.specs[0].providers == (CPU,)

    def test_the_loaded_verifier_carries_the_manifest_values(self, tmp_path: Path) -> None:
        stage(tmp_path)

        loaded, _ = load(placement(tmp_path))
        contract: WakeVerifier = loaded

        assert contract.model_id == MODEL_ID
        assert contract.spotter_model_id == SPOTTER_ID
        assert (contract.before_ms, contract.after_ms, contract.min_window_ms) == (2000, 300, 800)
        assert contract.threshold_at(0.847) == SCORES[83]
        assert contract.threshold_at(0.005) is None
        for name in WakeVerifier.__annotations__:
            assert hasattr(loaded, name)

    @pytest.mark.parametrize("method", ["threshold_at", "score", "close"])
    def test_each_protocol_method_has_the_protocols_parameters(self, method: str) -> None:
        def params(cls: type) -> list[str]:
            return list(inspect.signature(getattr(cls, method)).parameters)

        assert params(OnnxWakeVerifier) == params(WakeVerifier)

    def test_without_a_model_setting_load_is_refused(self, tmp_path: Path) -> None:
        stage(tmp_path)

        with pytest.raises(SpeechEngineError, match="MIST_VOICE_VERIFY_MODEL"):
            load(PlacementConfig(models_dir=str(tmp_path)))

    @pytest.mark.parametrize("model_id", ["../mist-verify", "Mist", "mist/verify", ""])
    def test_a_model_id_outside_the_id_form_is_refused(self, tmp_path: Path, model_id: str) -> None:
        with pytest.raises(SpeechEngineError, match="MIST_VOICE_VERIFY_MODEL"):
            load(PlacementConfig(models_dir=str(tmp_path), verify_model=model_id))

    def test_fewer_than_one_thread_is_refused(self, tmp_path: Path) -> None:
        stage(tmp_path)

        with pytest.raises(SpeechEngineError, match="MIST_VOICE_VERIFY_THREADS"):
            load(placement(tmp_path, verify_threads=0))

    def test_a_refused_manifest_opens_no_session(self, tmp_path: Path) -> None:
        stage(tmp_path, _top("version", 2))
        factory = FakeFactory()

        with pytest.raises(SpeechEngineError, match="version"):
            OnnxWakeVerifier.load(
                placement(tmp_path), session_factory=factory, available_providers=only_cpu
            )

        assert factory.specs == []

    def test_a_verifier_trained_against_another_spotter_model_is_refused(
        self, tmp_path: Path
    ) -> None:
        write_spotter(tmp_path, content=b"a retrained spotter")
        write_verifier(tmp_path)
        factory = FakeFactory()

        with pytest.raises(SpeechEngineError) as caught:
            OnnxWakeVerifier.load(
                placement(tmp_path), session_factory=factory, available_providers=only_cpu
            )

        message = str(caught.value)
        assert SPOTTER_ID in message
        assert sha256(SPOTTER_BYTES) in message
        assert sha256(b"a retrained spotter") in message
        assert factory.specs == []

    def test_a_verifier_whose_spotter_is_not_installed_is_refused(self, tmp_path: Path) -> None:
        write_verifier(tmp_path)

        with pytest.raises(SpeechEngineError) as caught:
            load(placement(tmp_path))

        assert str(wake_dir(tmp_path, SPOTTER_ID) / "manifest.json") in str(caught.value)

    def test_a_runtime_without_the_cpu_provider_is_refused(self, tmp_path: Path) -> None:
        stage(tmp_path)

        with pytest.raises(SpeechEngineError, match=CPU):
            load(placement(tmp_path), providers=lambda: [CUDA])

    def test_a_model_with_other_inputs_is_refused_naming_them(self, tmp_path: Path) -> None:
        stage(tmp_path)

        with pytest.raises(SpeechEngineError) as caught:
            load(placement(tmp_path), session=FakeSession(input_names=("audio",)))

        assert "waveform" in str(caught.value)
        assert "'audio'" in str(caught.value)


# -- Scoring ----------------------------------------------------------------------------------


class TestScore:
    def test_the_window_reaches_the_model_as_one_float32_row_over_32768(
        self, tmp_path: Path
    ) -> None:
        stage(tmp_path)
        session = FakeSession()
        loaded, _ = load(placement(tmp_path), session=session)

        loaded.score(pcm(-32768, 0, 16384, 32767))

        (feed,) = session.feeds
        assert list(feed) == ["waveform"]
        assert feed["waveform"].dtype == np.float32
        assert feed["waveform"].shape == (1, 4)
        np.testing.assert_array_equal(
            feed["waveform"][0], np.array([-1.0, 0.0, 0.5, 32767 / 32768], dtype=np.float32)
        )

    @pytest.mark.parametrize(
        "output",
        [np.array([0.75], dtype=np.float32), np.array([[0.75]], dtype=np.float32)],
        ids=["shape-1", "shape-1x1"],
    )
    def test_the_score_is_the_models_one_output(self, tmp_path: Path, output: np.ndarray) -> None:
        stage(tmp_path)
        loaded, _ = load(placement(tmp_path), session=FakeSession(outputs=(output,)))

        score = loaded.score(pcm(1, 2, 3))

        assert score == pytest.approx(0.75)
        assert isinstance(score, float)
        assert not isinstance(score, np.floating)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_output_raises(self, tmp_path: Path, value: float) -> None:
        stage(tmp_path)
        output = np.array([value], dtype=np.float32)
        loaded, _ = load(placement(tmp_path), session=FakeSession(outputs=(output,)))

        with pytest.raises(SpeechEngineError, match="finite"):
            loaded.score(pcm(1, 2, 3))

    def test_an_output_of_more_than_one_value_raises(self, tmp_path: Path) -> None:
        stage(tmp_path)
        output = np.array([0.1, 0.2], dtype=np.float32)
        loaded, _ = load(placement(tmp_path), session=FakeSession(outputs=(output,)))

        with pytest.raises(SpeechEngineError, match="one score"):
            loaded.score(pcm(1, 2, 3))

    def test_an_empty_window_raises_without_running_the_model(self, tmp_path: Path) -> None:
        stage(tmp_path)
        session = FakeSession()
        loaded, _ = load(placement(tmp_path), session=session)

        with pytest.raises(SpeechEngineError, match="empty"):
            loaded.score(b"")

        assert session.feeds == []

    def test_a_closed_verifier_refuses_to_score(self, tmp_path: Path) -> None:
        stage(tmp_path)
        session = FakeSession()
        loaded, _ = load(placement(tmp_path), session=session)

        loaded.close()
        loaded.close()

        with pytest.raises(SpeechEngineError, match="closed"):
            loaded.score(pcm(1, 2, 3))
        assert session.feeds == []


# -- Imports ----------------------------------------------------------------------------------


class TestImports:
    def test_importing_the_verifier_loads_neither_onnxruntime_nor_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for blocked in ("onnxruntime", "torch"):
            monkeypatch.setitem(sys.modules, blocked, None)
        monkeypatch.delitem(sys.modules, verifier.__name__)

        fresh = importlib.import_module(verifier.__name__)

        assert fresh.OnnxWakeVerifier.__name__ == "OnnxWakeVerifier"

    def test_the_module_names_neither_onnxruntime_nor_torch_in_an_import(self) -> None:
        tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and not node.level
        }

        assert "onnxruntime" not in imported
        assert "torch" not in imported
