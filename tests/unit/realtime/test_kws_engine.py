"""Tests for the ONNX name spotter over an injected session factory and a temporary manifest.

The unit tier has no onnxruntime, no wake model and no audio, so every test writes a manifest
and three placeholder graph files under a temporary `models_dir` and hands the spotter recording
fake sessions. That pins what the adapter owns: the manifest it accepts and refuses, the windows
each of the three graphs reads per 80 ms hop, the warm-up and fill rules, and the rising edge
and refractory period. Score parity with openWakeWord on the real model is T36's.
"""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from backend.realtime.budget import PlacementConfig
from backend.realtime.engines import kws
from backend.realtime.engines.kws import OnnxNameSpotter
from backend.realtime.engines.silero_vad import SessionSpec
from backend.realtime.errors import SpeechEngineError
from backend.realtime.speech import NameHit, NameSpotter, NameSpotterStream, SpotterCalibration

CPU = "CPUExecutionProvider"
MODEL_ID = "mist"
HOP = 1280
FRAME = 320
FILES = {
    "melspectrogram": "melspectrogram.onnx",
    "embedding": "embedding_model.onnx",
    "classifier": "mist.onnx",
}


# -- Fakes ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FakeInput:
    name: str
    shape: tuple[object, ...]


@dataclass
class RecordingSession:
    """A scripted session: records each input it is fed and answers `respond(call, input)`."""

    input_name: str
    shape: tuple[object, ...]
    respond: Callable[[int, np.ndarray], np.ndarray]
    feeds: list[np.ndarray] = field(default_factory=list)

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput(self.input_name, self.shape)]

    def get_providers(self) -> list[str]:
        return [CPU]

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names is None
        assert list(input_feed) == [self.input_name]
        value = np.array(input_feed[self.input_name], copy=True)
        self.feeds.append(value)
        return [self.respond(len(self.feeds), value)]


@dataclass
class FakeFactory:
    """Opens the fake session whose graph file name the spec's path ends with."""

    sessions: dict[str, RecordingSession]
    specs: list[SessionSpec] = field(default_factory=list)

    def __call__(self, spec: SessionSpec) -> RecordingSession:
        self.specs.append(spec)
        return self.sessions[Path(spec.path).name]


def mel_constant(value: float) -> Callable[[int, np.ndarray], np.ndarray]:
    def respond(call: int, value_in: np.ndarray) -> np.ndarray:
        return np.full((1, 1, 8, 32), value, dtype=np.float32)

    return respond


def mel_by_call(call: int, value_in: np.ndarray) -> np.ndarray:
    """Each hop's 8 frames are `10 * call`, so after `x/10 + 2` they read `call + 2`."""
    return np.full((1, 1, 8, 32), 10.0 * call, dtype=np.float32)


def embedding_by_call(call: int, value_in: np.ndarray) -> np.ndarray:
    return np.full((1, 1, 1, 96), float(call), dtype=np.float32)


def scripted(scores: Sequence[float]) -> Callable[[int, np.ndarray], np.ndarray]:
    """The classifier's score for its n-th call is `scores[n - 1]`, then 0.0."""

    def respond(call: int, value_in: np.ndarray) -> np.ndarray:
        score = scores[call - 1] if call <= len(scores) else 0.0
        return np.array([[score]], dtype=np.float32)

    return respond


@dataclass
class Rig:
    spotter: OnnxNameSpotter
    factory: FakeFactory
    mel: RecordingSession
    embedding: RecordingSession
    classifier: RecordingSession
    manifest: Path


# -- Manifest helpers -------------------------------------------------------------------------


def manifest_path(models_dir: Path) -> Path:
    return models_dir / "speech" / "wake" / MODEL_ID / "manifest.json"


def base_manifest(hashes: Mapping[str, str]) -> dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "names": ["mist"],
        "family": "openwakeword",
        "sample_rate": 16000,
        "hop_ms": 80,
        "window_ms": 2000,
        "threshold": 0.5,
        "warmup_predictions": 5,
        "graphs": [
            {
                "role": "melspectrogram",
                "file": FILES["melspectrogram"],
                "sha256": hashes["melspectrogram"],
                "input": {"new_samples": 1280, "left_context_samples": 480},
                "frames_per_hop": 8,
                "bins": 32,
                "output_transform": {"scale": 0.1, "offset": 2.0},
            },
            {
                "role": "embedding",
                "file": FILES["embedding"],
                "sha256": hashes["embedding"],
                "input": {"window_frames": 76, "stride_frames": 8},
                "dims": 96,
            },
            {
                "role": "classifier",
                "file": FILES["classifier"],
                "sha256": hashes["classifier"],
                "input": {"window_embeddings": "from_model"},
            },
        ],
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
    models_dir: Path, mutate: Callable[[dict], None] | None = None, **changes: object
) -> Path:
    """Stage three placeholder graphs and a manifest; `changes` and `mutate` alter the manifest."""
    directory = manifest_path(models_dir).parent
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for role, name in FILES.items():
        content = f"{role} graph".encode()
        (directory / name).write_bytes(content)
        hashes[role] = hashlib.sha256(content).hexdigest()
    manifest = base_manifest(hashes)
    manifest.update(changes)
    if mutate is not None:
        mutate(manifest)
    path = manifest_path(models_dir)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def placement(models_dir: Path, **changes: object) -> PlacementConfig:
    return PlacementConfig(models_dir=str(models_dir), kws_model=MODEL_ID, **changes)


def build(
    models_dir: Path,
    *,
    window: int = 16,
    scores: Sequence[float] = (),
    mel: Callable[[int, np.ndarray], np.ndarray] | None = None,
    threads: int = 1,
    **changes: object,
) -> Rig:
    path = write_model(models_dir, **changes)
    mel_session = RecordingSession("input", (1, "samples"), mel or mel_constant(10.0))
    embedding = RecordingSession("input_1", (1, 76, 32, 1), embedding_by_call)
    classifier = RecordingSession("x", (1, window, 96), scripted(scores))
    factory = FakeFactory(
        {
            FILES["melspectrogram"]: mel_session,
            FILES["embedding"]: embedding,
            FILES["classifier"]: classifier,
        }
    )
    spotter = OnnxNameSpotter.load(
        placement(models_dir, kws_threads=threads), session_factory=factory
    )
    return Rig(spotter, factory, mel_session, embedding, classifier, path)


def audio(samples: int, start: int = 0) -> bytes:
    """PCM16 whose sample `i` (counted from the stream start) is `i % 20000 - 10000`."""
    values = (np.arange(start, start + samples) % 20000) - 10000
    return values.astype("<i2").tobytes()


def push_frames(stream: kws.OnnxNameSpotterStream, count: int, first_pos: int = 0) -> list:
    """Push `count` 20 ms frames at positions `first_pos...`; return each push's hits by pos."""
    hits: list[NameHit] = []
    for pos in range(first_pos, first_pos + count):
        hits.extend(stream.push(audio(FRAME, pos * FRAME), pos))
    return hits


def hop_of(hit: NameHit) -> int:
    """The 1-based hop a hit ended on, for frames pushed from position 0."""
    return (hit.end_pos + 1) // 4


# -- Manifest and load ------------------------------------------------------------------------


def test_load_reads_the_manifest(tmp_path: Path) -> None:
    rig = build(tmp_path, threads=3)
    spotter = rig.spotter
    contract: NameSpotter = spotter
    stream: NameSpotterStream = contract.open_stream(0.5)
    assert stream.push(audio(FRAME), 0) == []
    assert spotter.model_id == MODEL_ID
    assert spotter.names == ("mist",)
    assert spotter.window_ms == 2000
    assert spotter.hit_lag_ms == 120
    assert spotter.threshold == 0.5
    assert spotter.warmup_predictions == 5
    assert spotter.classifier_window == 16
    assert spotter.training == "ab" * 32
    assert dict(spotter.calibration) == {
        "server": SpotterCalibration(
            false_accepts=((0.5, 2.0), (0.7, 0.5), (0.9, 0.5)),
            positive_quantile=((0.5, 0.1), (0.7, 0.1), (0.9, 0.8)),
        )
    }
    directory = rig.manifest.parent
    assert [spec.path for spec in rig.factory.specs] == [
        str(directory / FILES["melspectrogram"]),
        str(directory / FILES["embedding"]),
        str(directory / FILES["classifier"]),
    ]
    for spec in rig.factory.specs:
        assert spec.providers == (CPU,)
        assert spec.intra_op_num_threads == 3
        assert spec.inter_op_num_threads == 1


def test_load_accepts_both_calibration_entries_or_none(tmp_path: Path) -> None:
    entry = {"false_accepts": [[0.5, 1.0]], "positive_quantile": [[0.5, 0.5]]}
    rig = build(tmp_path, calibration={"server": entry, "endpoint": entry})
    assert set(rig.spotter.calibration) == {"server", "endpoint"}
    rig = build(tmp_path, calibration={})
    assert dict(rig.spotter.calibration) == {}


def test_load_without_a_model_setting_is_refused(tmp_path: Path) -> None:
    write_model(tmp_path)
    with pytest.raises(SpeechEngineError, match="MIST_VOICE_KWS_MODEL"):
        OnnxNameSpotter.load(
            PlacementConfig(models_dir=str(tmp_path)), session_factory=FakeFactory({})
        )


def test_a_model_id_leaving_the_wake_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SpeechEngineError, match="MIST_VOICE_KWS_MODEL"):
        OnnxNameSpotter.load(
            PlacementConfig(models_dir=str(tmp_path), kws_model="../mist"),
            session_factory=FakeFactory({}),
        )


def test_a_missing_manifest_is_refused_naming_it(tmp_path: Path) -> None:
    with pytest.raises(SpeechEngineError) as caught:
        OnnxNameSpotter.load(placement(tmp_path), session_factory=FakeFactory({}))
    assert str(manifest_path(tmp_path)) in str(caught.value)


def test_a_manifest_that_is_not_json_is_refused_naming_it(tmp_path: Path) -> None:
    path = write_model(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SpeechEngineError) as caught:
        OnnxNameSpotter.load(placement(tmp_path), session_factory=FakeFactory({}))
    assert str(path) in str(caught.value)


def test_a_missing_graph_file_is_refused_naming_it(tmp_path: Path) -> None:
    path = write_model(tmp_path)
    graph = path.parent / FILES["embedding"]
    graph.unlink()
    with pytest.raises(SpeechEngineError) as caught:
        OnnxNameSpotter.load(placement(tmp_path), session_factory=FakeFactory({}))
    assert str(path) in str(caught.value)
    assert str(graph) in str(caught.value)


def _graph(role: str) -> Callable[[dict], dict]:
    index = {"melspectrogram": 0, "embedding": 1, "classifier": 2}[role]
    return lambda manifest: manifest["graphs"][index]


def _set(getter: Callable[[dict], dict], key: str, value: object) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        getter(manifest)[key] = value

    return mutate


def _set_input(role: str, key: str, value: object) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        _graph(role)(manifest)["input"][key] = value

    return mutate


def _top(key: str, value: object) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        manifest[key] = value

    return mutate


def _drop(key: str) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        del manifest[key]

    return mutate


def _swap_graphs(manifest: dict) -> None:
    graphs = manifest["graphs"]
    graphs[0], graphs[1] = graphs[1], graphs[0]


def _server_entry(**entry: object) -> Callable[[dict], None]:
    def mutate(manifest: dict) -> None:
        manifest["calibration"] = {"server": entry}

    return mutate


GOOD_FA = [[0.5, 2.0], [0.9, 0.1]]
GOOD_PQ = [[0.5, 0.1], [0.9, 0.8]]

REFUSALS = {
    "sha256 mismatch": _set(_graph("classifier"), "sha256", "0" * 64),
    "sample rate": _top("sample_rate", 22050),
    "hop": _top("hop_ms", 100),
    "family": _top("family", "other"),
    "graphs out of order": _swap_graphs,
    "two graphs": lambda manifest: manifest["graphs"].pop(),
    "mel new samples": _set_input("melspectrogram", "new_samples", 1600),
    "mel left context": _set_input("melspectrogram", "left_context_samples", 320),
    "mel frames per hop": _set(_graph("melspectrogram"), "frames_per_hop", 10),
    "mel bins": _set(_graph("melspectrogram"), "bins", 40),
    "mel transform": _set(_graph("melspectrogram"), "output_transform", {"scale": 1, "offset": 0}),
    "embedding window": _set_input("embedding", "window_frames", 64),
    "embedding stride": _set_input("embedding", "stride_frames", 4),
    "embedding dims": _set(_graph("embedding"), "dims", 128),
    "classifier window": _set_input("classifier", "window_embeddings", 16),
    "unknown input key": _set_input("embedding", "channels", 1),
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
    "quantile scores not ascending": _server_entry(
        false_accepts=GOOD_FA, positive_quantile=[[0.9, 0.1], [0.5, 0.8]]
    ),
    "entry without quantile": _server_entry(false_accepts=GOOD_FA),
    "entry without false accepts": _server_entry(positive_quantile=GOOD_PQ),
    "unknown calibration spotter": _top(
        "calibration", {"phone": {"false_accepts": GOOD_FA, "positive_quantile": GOOD_PQ}}
    ),
    "unknown calibration entry key": _server_entry(
        false_accepts=GOOD_FA, positive_quantile=GOOD_PQ, notes=[]
    ),
    "unknown top-level key": _top("notes", "x"),
    "unknown graph key": _set(_graph("embedding"), "notes", "x"),
    "missing key": _drop("training"),
    "negative hit lag": _top("hit_lag_ms", -1),
    "fractional hit lag": _top("hit_lag_ms", 1.5),
    "empty names": _top("names", []),
    "model id not the directory": _top("model_id", "other"),
    "training not a sha256": _top("training", "abc"),
    "graph file leaving the model directory": _set(_graph("classifier"), "file", "../mist.onnx"),
}


@pytest.mark.parametrize("mutate", REFUSALS.values(), ids=REFUSALS.keys())
def test_a_bad_manifest_is_refused_naming_it(
    tmp_path: Path, mutate: Callable[[dict], None]
) -> None:
    path = write_model(tmp_path, mutate)
    factory = FakeFactory({})
    with pytest.raises(SpeechEngineError) as caught:
        OnnxNameSpotter.load(placement(tmp_path), session_factory=factory)
    assert str(path) in str(caught.value)


@pytest.mark.parametrize("shape", [(1, 0, 96), (1, "N", 96), (1, None, 96), (1,), (1, True, 96)])
def test_a_classifier_window_that_is_not_a_positive_integer_is_refused(
    tmp_path: Path, shape: tuple[object, ...]
) -> None:
    path = write_model(tmp_path)
    factory = FakeFactory(
        {
            FILES["melspectrogram"]: RecordingSession("input", (1, 1760), mel_constant(10.0)),
            FILES["embedding"]: RecordingSession("input_1", (1, 76, 32, 1), embedding_by_call),
            FILES["classifier"]: RecordingSession("x", shape, scripted(())),
        }
    )
    with pytest.raises(SpeechEngineError) as caught:
        OnnxNameSpotter.load(placement(tmp_path), session_factory=factory)
    assert str(path.parent / FILES["classifier"]) in str(caught.value)


# -- Hops and windows -------------------------------------------------------------------------


def test_frames_are_buffered_into_hops_keeping_the_remainder(tmp_path: Path) -> None:
    rig = build(tmp_path)
    stream = rig.spotter.open_stream(0.5)
    push_frames(stream, 3)
    assert rig.mel.feeds == []
    push_frames(stream, 1, first_pos=3)
    assert len(rig.mel.feeds) == 1
    # Pushes that do not fall on hop boundaries: 960 + 640 samples is one hop and 320 over.
    stream.push(audio(960, HOP), 6)
    assert len(rig.mel.feeds) == 1
    stream.push(audio(640, HOP + 960), 8)
    assert len(rig.mel.feeds) == 2
    stream.push(audio(960, 2 * HOP + 320), 11)
    assert len(rig.mel.feeds) == 3
    third = rig.mel.feeds[2][0, 480:]
    expected = np.frombuffer(audio(HOP, 2 * HOP), dtype="<i2").astype(np.float32)
    np.testing.assert_array_equal(third, expected)


def test_odd_byte_pcm_is_refused(tmp_path: Path) -> None:
    stream = build(tmp_path).spotter.open_stream(0.5)
    with pytest.raises(SpeechEngineError):
        stream.push(b"\x00\x01\x02", 0)


def test_the_melspectrogram_reads_1760_samples_with_480_of_left_context(tmp_path: Path) -> None:
    rig = build(tmp_path)
    stream = rig.spotter.open_stream(0.5)
    push_frames(stream, 8)
    first, second = rig.mel.feeds
    assert first.shape == (1, 1760)
    assert first.dtype == np.float32
    whole = np.frombuffer(audio(2 * HOP), dtype="<i2").astype(np.float32)
    np.testing.assert_array_equal(first[0, :480], np.zeros(480, dtype=np.float32))
    np.testing.assert_array_equal(first[0, 480:], whole[:HOP])
    np.testing.assert_array_equal(second[0], whole[HOP - 480 : 2 * HOP])


def test_the_melspectrogram_output_is_scaled_by_x_over_10_plus_2(tmp_path: Path) -> None:
    rig = build(tmp_path, mel=mel_constant(10.0))
    push_frames(rig.spotter.open_stream(0.5), 4)
    (window,) = rig.embedding.feeds
    np.testing.assert_allclose(window[0, -8:, :, 0], np.full((8, 32), 3.0))


def test_the_embedding_reads_76_frames_moving_by_8_per_hop(tmp_path: Path) -> None:
    rig = build(tmp_path, mel=mel_by_call)
    push_frames(rig.spotter.open_stream(0.5), 4 * 12)
    assert len(rig.embedding.feeds) == 12
    for hop, window in enumerate(rig.embedding.feeds, start=1):
        assert window.shape == (1, 76, 32, 1)
        assert window.dtype == np.float32
        np.testing.assert_allclose(window[0, -8:, :, 0], np.full((8, 32), hop + 2.0))
        if hop > 1:
            previous = rig.embedding.feeds[hop - 2]
            np.testing.assert_array_equal(window[0, :-8], previous[0, 8:])
    # Hop 12 sees hops 3 to 12: the oldest 4 frames of hop 3, then all of hops 4 to 12.
    last = rig.embedding.feeds[-1][0, :, 0, 0]
    np.testing.assert_allclose(last[:4], np.full(4, 5.0))
    np.testing.assert_allclose(last[4:12], np.full(8, 6.0))


@pytest.mark.parametrize("window", [16, 28])
def test_the_classifier_reads_the_last_n_embeddings(tmp_path: Path, window: int) -> None:
    rig = build(tmp_path, window=window)
    assert rig.spotter.classifier_window == window
    hops = window + 5
    push_frames(rig.spotter.open_stream(0.5), 4 * hops)
    assert len(rig.classifier.feeds) == hops
    first = rig.classifier.feeds[0]
    assert first.shape == (1, window, 96)
    np.testing.assert_array_equal(first[0, :-1], np.zeros((window - 1, 96)))
    np.testing.assert_allclose(first[0, -1], np.full(96, 1.0))
    last = rig.classifier.feeds[-1]
    assert last.dtype == np.float32
    expected = np.arange(hops - window + 1, hops + 1, dtype=np.float32)
    np.testing.assert_allclose(last[0, :, 0], expected)


# -- Scores and hits --------------------------------------------------------------------------

FULL_16 = 25
"""With N 16 the classifier window is full from hop 25: 10 hops fill the 76 mel frames, and 15
more give the 16 embeddings of full mel windows."""


def test_more_than_one_pending_hop_reports_the_maximum(tmp_path: Path) -> None:
    scores = [0.0] * FULL_16 + [0.2, 0.9, 0.4]
    rig = build(tmp_path, scores=scores)
    stream = rig.spotter.open_stream(0.5)
    assert push_frames(stream, 4 * FULL_16) == []
    hits = stream.push(audio(3 * HOP, FULL_16 * HOP), 4 * FULL_16 + 11)
    assert len(rig.classifier.feeds) == FULL_16 + 3
    assert hits == [NameHit(end_pos=4 * FULL_16 + 11, keyword="mist", score=pytest.approx(0.9))]


def test_the_first_predictions_after_reset_are_zeroed(tmp_path: Path) -> None:
    # A warm-up longer than the fill shows the warm-up itself: hops 25 to 30 are full but warm.
    scores = [0.0] * 24 + [0.99] * 6 + [0.0, 0.99]
    rig = build(tmp_path, scores=scores, warmup_predictions=30)
    stream = rig.spotter.open_stream(0.5)
    hits = push_frames(stream, 4 * 32)
    assert [hop_of(hit) for hit in hits] == [32]
    rig.classifier.respond = scripted([0.99] * 31)
    rig.classifier.feeds.clear()
    stream.reset()
    hits = push_frames(stream, 4 * 31)
    assert [hop_of(hit) for hit in hits] == [31]


def test_a_scripted_099_in_the_first_5_predictions_gives_no_hit(tmp_path: Path) -> None:
    rig = build(tmp_path, scores=[0.99] * 5)
    stream = rig.spotter.open_stream(0.5)
    assert push_frames(stream, 4 * 5) == []
    stream.reset()
    rig.classifier.feeds.clear()
    assert push_frames(stream, 4 * 5) == []


@pytest.mark.parametrize(("window", "full"), [(16, 25), (28, 37)])
def test_no_hit_before_the_windows_are_full(tmp_path: Path, window: int, full: int) -> None:
    rig = build(tmp_path, window=window, scores=[0.99] * (full + 2), warmup_predictions=0)
    stream = rig.spotter.open_stream(0.5)
    hits = push_frames(stream, 4 * (full + 2))
    assert [hop_of(hit) for hit in hits] == [full]
    first = rig.embedding.feeds[0]
    np.testing.assert_array_equal(first[0, :-8], np.zeros((68, 32, 1)))


def test_reset_starts_a_new_context(tmp_path: Path) -> None:
    rig = build(tmp_path, scores=[0.99] * 100, warmup_predictions=0)
    stream = rig.spotter.open_stream(0.5)
    push_frames(stream, 4 * FULL_16 + 2)
    stream.reset()
    rig.mel.feeds.clear()
    hits = push_frames(stream, 4 * FULL_16, first_pos=200)
    assert len(hits) == 1 and hits[0].end_pos == 200 + 4 * FULL_16 - 1
    first = rig.mel.feeds[0]
    np.testing.assert_array_equal(first[0, :480], np.zeros(480, dtype=np.float32))
    expected = np.frombuffer(audio(HOP, 200 * FRAME), dtype="<i2").astype(np.float32)
    np.testing.assert_array_equal(first[0, 480:], expected)


def test_a_hit_is_a_rising_edge_with_a_refractory_period(tmp_path: Path) -> None:
    # window_ms 400 is 5 hops: after a hit at hop h, hops h+1 to h+4 cannot hit.
    # Offsets from hop 25: 0 hits; 1 is not a rising edge; 3 rises inside the refractory
    # period; 5 rises as it ends; 10 rises to exactly the threshold, which counts.
    after = [0.9, 0.9, 0.1, 0.9, 0.1, 0.9, 0.95, 0.1, 0.1, 0.1, 0.5]
    rig = build(tmp_path, scores=[0.0] * (FULL_16 - 1) + after, window_ms=400)
    stream = rig.spotter.open_stream(0.5)
    hits = push_frames(stream, 4 * (FULL_16 - 1 + len(after)))
    assert [(hop_of(hit) - FULL_16, hit.score) for hit in hits] == [
        (0, pytest.approx(0.9)),
        (5, pytest.approx(0.9)),
        (10, pytest.approx(0.5)),
    ]
    assert all(hit.keyword == "mist" for hit in hits)
    assert hits[0].end_pos == 4 * FULL_16 - 1


def test_a_score_held_above_the_threshold_through_the_refractory_period_does_not_hit_again(
    tmp_path: Path,
) -> None:
    rig = build(tmp_path, scores=[0.0] * (FULL_16 - 1) + [0.9] * 12, window_ms=400)
    hits = push_frames(rig.spotter.open_stream(0.5), 4 * (FULL_16 - 1 + 12))
    assert [hop_of(hit) for hit in hits] == [FULL_16]


def test_the_stream_threshold_is_its_own(tmp_path: Path) -> None:
    rig = build(tmp_path, scores=[0.0] * (FULL_16 - 1) + [0.7])
    assert push_frames(rig.spotter.open_stream(0.8), 4 * FULL_16) == []


# -- score_window -----------------------------------------------------------------------------


def test_score_window_runs_the_same_pipeline_and_returns_the_maximum(tmp_path: Path) -> None:
    scores = [0.99] * (FULL_16 - 1) + [0.3, 0.8, 0.6]
    rig = build(tmp_path, scores=scores)
    pcm = audio((FULL_16 + 2) * HOP + 100)
    assert rig.spotter.score_window(pcm) == pytest.approx(0.8)
    assert len(rig.classifier.feeds) == FULL_16 + 2
    np.testing.assert_array_equal(rig.mel.feeds[0][0, :480], np.zeros(480, dtype=np.float32))
    # A second call starts from a fresh context, as the stream does after a reset.
    rig.classifier.respond = scripted(scores)
    rig.classifier.feeds.clear()
    rig.mel.feeds.clear()
    assert rig.spotter.score_window(pcm) == pytest.approx(0.8)
    np.testing.assert_array_equal(rig.mel.feeds[0][0, :480], np.zeros(480, dtype=np.float32))


def test_score_window_over_a_buffer_shorter_than_the_windows_is_zero(tmp_path: Path) -> None:
    rig = build(tmp_path, scores=[0.99] * 10)
    assert rig.spotter.score_window(audio(10 * HOP)) == 0.0
    assert rig.spotter.score_window(b"") == 0.0
