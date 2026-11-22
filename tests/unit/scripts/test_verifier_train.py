"""Tests for the verifier's trainer, exporter, calibrator and pre-flight.

Every input is generated here: low noise and tone bursts as WAV files, a train-only recording
manifest, a dataset index over them with its `provenance.json`, an AMI directory with its shares,
and fold runs with out-of-fold scores. The trainer runs a tiny stand-in encoder; the WavLM tests
build the real Base+ shape or a two-layer one from random initialisation, and the export is
checked through `onnx.reference`, since the runner image has no onnxruntime. No test reads a
recording, needs a network or the GPU.
"""

import json
import random
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from backend.realtime.engines.verifier import GRID, read_verifier_manifest
from scripts.wake.verifier import calibrate, dataset, export, model, preflight, train

REPO = Path(__file__).resolve().parents[3]
RATE = 16000
FRAME = 320
WINDOW = 116 * FRAME


# -- Audio and the generated world ------------------------------------------------------------


def noise(seconds: float, seed: int, level: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-level, level + 1, int(round(seconds * RATE))).astype("<i2")


def burst(seconds: float) -> np.ndarray:
    times = np.arange(int(round(seconds * RATE))) / RATE
    return (16000 * np.sin(2 * np.pi * 440 * times)).astype("<i2")


class TinyEncoder(model.StateEncoder):
    """A stand-in for WavLM: a strided convolution and three layers, the last one trainable."""

    def __init__(self) -> None:
        super().__init__()
        self.frontend = nn.Conv1d(1, 8, kernel_size=FRAME, stride=FRAME)
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.dropout = nn.Dropout(0.5)
        self.dim = 8
        self.num_states = 4

    def trainable_modules(self) -> list[nn.Module]:
        return [self.layers[2]]

    def hidden_states(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        x = self.frontend(waveform.unsqueeze(1)).transpose(1, 2)
        states = [x]
        for layer in self.layers:
            x = torch.tanh(layer(self.dropout(x)))
            states.append(x)
        return states


def tiny_model() -> model.VerifierModel:
    return model.VerifierModel(TinyEncoder())


RECORDINGS = (
    ("rec/pos-1.wav", "positive", "mist", "s1"),
    ("rec/pos-2.wav", "positive", "mist", "s2"),
    ("rec/pos-3.wav", "positive", "mist", "s3"),
    ("rec/hom-1.wav", "homophone", "I missed the bus", "s4"),
    ("rec/hom-2.wav", "homophone", "the mist rolled in", "s5"),
    ("rec/pos-4.wav", "positive", "mist", "s6"),
)


@dataclass
class World:
    root: Path
    manifest: Path
    full_manifest: Path
    dataset: Path
    ami: Path
    weights: Path
    windows: list[dataset.Window]

    def train_args(self, out: Path, *extra: str, manifest: Path | None = None) -> list[str]:
        return [
            "--dataset",
            str(self.dataset),
            "--manifest",
            str(manifest or self.manifest),
            "--weights",
            str(self.weights),
            "--out",
            str(out),
            *extra,
        ]


def _entries(recordings: Path, splits: list[str]) -> list[dict]:
    entries = []
    for number, ((file, label, phrase, session), split) in enumerate(zip(RECORDINGS, splits)):
        samples = np.concatenate([noise(2.0, number, 300), burst(0.4), noise(0.6, 50 + number)])
        dataset.write_wav(recordings / file, samples)
        entries.append(
            {
                "file": file,
                "label": label,
                "split": split,
                "phrase": phrase,
                "condition": "headset",
                "duration_ms": len(samples) * 1000 // RATE,
                "session": session,
            }
        )
    return entries


def _write_manifest(path: Path, recordings: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"root": str(recordings), "entries": entries}), encoding="utf-8")
    return path


def write_dataset(directory: Path, windows: list[dataset.Window], training: str, roots: dict):
    directory.mkdir(parents=True, exist_ok=True)
    dataset.write_windows(directory / dataset.INDEX_FILE, windows)
    meta = {"version": 1, "training": training, "roots": roots, "windows": len(windows)}
    (directory / dataset.INDEX_META).write_text(json.dumps(meta), encoding="utf-8")
    digests = sorted({ref.sha256 for window in windows for ref in window.refs()})
    provenance = {
        "version": 1,
        "training": training,
        "file_sha256": digests,
        "sources": {"manifest": {"licence": "x", "version": "y", "files": 6, "hours": 0.005}},
        "excluded": {"ami_dev_sha256sums": "a" * 64, "ami_eval_sha256sums": "b" * 64},
    }
    (directory / dataset.PROVENANCE_FILE).write_text(json.dumps(provenance), encoding="utf-8")


@pytest.fixture
def world(tmp_path: Path) -> World:
    recordings = tmp_path / "recordings"
    entries = _entries(recordings, ["train"] * len(RECORDINGS))
    manifest = _write_manifest(tmp_path / "train_manifest.json", recordings, entries)
    full_entries = [dict(entry) for entry in entries]
    full_entries[-1]["split"] = "tune"
    full = _write_manifest(tmp_path / "manifest.json", recordings, full_entries)
    training = dataset.sha256_file(manifest)
    ami = tmp_path / "ami"
    mined = np.concatenate([noise(3.0, 99), burst(0.4), noise(3.0, 98)])
    dataset.write_wav(ami / "ES2002a.Mix-Headset.wav", mined)
    windows = []
    for number, (file, label, _, session) in enumerate(RECORDINGS):
        path = recordings / file
        samples = dataset.wav_info(path).frames
        ref = dataset.FileRef("manifest", dataset.sha256_file(path), None, number, samples)
        kind = "real_positive" if label == "positive" else "homophone"
        cls = "positive" if label == "positive" else "negative"
        windows.append(dataset.Window(ref, 0, WINDOW, cls, kind, f"manifest:{session}", (0.5, 0.6)))
    ami_path = ami / "ES2002a.Mix-Headset.wav"
    ami_ref = dataset.FileRef(
        "ami",
        dataset.sha256_file(ami_path),
        "ES2002a.Mix-Headset.wav",
        None,
        dataset.wav_info(ami_path).frames,
    )
    for start in (0, 3 * RATE - 100 * FRAME):
        windows.append(
            dataset.Window(
                ami_ref, start, start + WINDOW, "negative", "mined_hit", "ami:ES2002a", ()
            )
        )
    data = tmp_path / "dataset"
    write_dataset(data, windows, training, {"ami": str(ami)})
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"not read by the tiny builder")
    return World(tmp_path, manifest, full, data, ami, weights, windows)


def tiny_builder(built: list[model.VerifierModel]) -> train.ModelBuilder:
    def build(config: train.TrainConfig, weights: Path) -> model.VerifierModel:
        del config, weights
        made = tiny_model()
        built.append(made)
        return made

    return build


# -- The model --------------------------------------------------------------------------------


class TestModel:
    def test_the_head_maps_any_length_to_one_logit(self) -> None:
        verifier = tiny_model().eval()
        for samples in (40 * FRAME, WINDOW):
            assert verifier(torch.zeros(2, samples)).shape == (2,)
        score = model.ScoreGraph(verifier)(torch.randn(1, 50 * FRAME) * 0.1)
        assert score.shape == (1,)
        assert 0.0 <= float(score[0]) <= 1.0

    def test_the_head_pools_1536_values_for_768_wide_states(self) -> None:
        head = model.VerifierHead(13, 768)
        assert head.mlp[0].in_features == 1536
        assert head.mlp[0].out_features == 256
        states = [torch.randn(1, 20, 768) for _ in range(13)]
        assert head(states).shape == (1,)

    def test_base_plus_is_built_from_the_bundles_params(self) -> None:
        params = model.base_plus_params()
        from torchaudio.pipelines import WAVLM_BASE_PLUS

        assert params == WAVLM_BASE_PLUS._params
        assert params["encoder_layer_drop"] == 0.05
        assert params["encoder_ff_interm_dropout"] == 0.0
        assert model.normalizes_waveform() is False
        verifier = model.build_verifier(params)
        layers = verifier.encoder.wavlm.encoder.transformer.layers
        assert len(layers) == 12
        trainable = {id(p) for p in verifier.parameters() if p.requires_grad}
        expected = {id(p) for layer in layers[8:] for p in layer.parameters()}
        expected |= {id(p) for p in verifier.head.parameters()}
        assert trainable == expected
        frozen = [p for p in verifier.encoder.parameters() if not p.requires_grad]
        assert sum(p.numel() for p in frozen) + sum(
            p.numel() for layer in layers[8:] for p in layer.parameters()
        ) == sum(p.numel() for p in verifier.encoder.parameters())
        verifier.train()
        assert all(not module.training for module in verifier.frozen_modules())
        assert verifier.encoder.wavlm.feature_extractor.training is False
        assert all(not layer.training for layer in layers[:8])
        assert all(layer.training for layer in layers[8:])
        assert verifier.head.training
        verifier.eval()
        assert not any(module.training for module in verifier.modules())

    def test_the_frozen_part_of_the_stand_in_stays_in_eval(self) -> None:
        verifier = tiny_model()
        verifier.train()
        assert verifier.encoder.dropout.training is False
        assert verifier.encoder.frontend.training is False
        assert verifier.encoder.layers[2].training
        frozen = [p for p in verifier.encoder.parameters() if not p.requires_grad]
        assert len(frozen) == 6

    def test_the_weights_load_strictly_and_never_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*args: object, **kwargs: object) -> None:
            raise AssertionError("a downloader was called")

        monkeypatch.setattr(torch.hub, "load_state_dict_from_url", refuse)
        monkeypatch.setattr(torch.hub, "download_url_to_file", refuse)
        params = {**model.base_plus_params(), "encoder_num_layers": 2}
        source = model.build_wavlm(params)
        path = tmp_path / "wavlm.pth"
        torch.save(source.state_dict(), path)
        loaded = model.load_wavlm(path, params)
        for key, value in source.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value)
        state = source.state_dict()
        state.pop(next(iter(state)))
        torch.save(state, tmp_path / "short.pth")
        with pytest.raises(model.ModelError, match="not a WavLM state dict"):
            model.load_wavlm(tmp_path / "short.pth", params)

    def test_the_trainer_builds_base_plus_from_the_staged_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        params = {**model.base_plus_params(), "encoder_num_layers": 2}
        torch.save(model.build_wavlm(params).state_dict(), tmp_path / "w.pth")
        monkeypatch.setattr(train, "base_plus_params", lambda: dict(params))
        config = train.TrainConfig(first_trainable_layer=2)
        built = train.build_wavlm_verifier(config, tmp_path / "w.pth")
        assert len(built.encoder.trainable_modules()) == 1


# -- The trainer ------------------------------------------------------------------------------


class TestTrainer:
    def test_a_fold_trains_and_keeps_its_last_epoch(
        self, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        built: list[model.VerifierModel] = []
        runs = world.root / "runs"
        assert train.main(world.train_args(runs, "--fold", "0"), build=tiny_builder(built)) == 0
        run = runs / "fold-0"
        saved = torch.load(run / train.RUN_MODEL, weights_only=True)
        for key, value in built[0].state_dict().items():
            assert torch.equal(saved[key], value.cpu())
        config = json.loads((run / train.RUN_CONFIG).read_text(encoding="utf-8"))
        assert config["optimizer"] == "AdamW"
        assert (config["layers_lr"], config["head_lr"]) == (2e-5, 1e-3)
        assert (config["batch"], config["epochs"], config["seed"]) == (32, 6, 17)
        assert config["fp16"] is True
        assert config["keep"] == "last_epoch" and config["kept_epoch"] == 6
        assert config["normalize_waveform"] is False
        assert config["params"] == json.loads(json.dumps(model.base_plus_params()))
        history = json.loads((run / train.RUN_HISTORY).read_text(encoding="utf-8"))
        assert [row["epoch"] for row in history] == [1, 2, 3, 4, 5, 6]
        folds = train.assign_folds(world.windows)
        oof = [
            json.loads(line)
            for line in (run / train.RUN_OOF).read_text(encoding="utf-8").splitlines()
        ]
        assert oof
        assert {row["window"] for row in oof} == {
            i for i, w in enumerate(world.windows) if folds[w.group] == 0
        }
        assert all(0.0 <= row["score"] <= 1.0 for row in oof)
        assert "trained on" in capsys.readouterr().out

    def test_the_final_run_trains_on_everything_and_writes_no_oof(self, world: World) -> None:
        runs = world.root / "runs"
        assert train.main(world.train_args(runs, "--final"), build=tiny_builder([])) == 0
        config = json.loads((runs / "final" / train.RUN_CONFIG).read_text(encoding="utf-8"))
        assert config["windows"] == {"train": len(world.windows), "held_out": 0}
        assert not (runs / "final" / train.RUN_OOF).exists()
        assert train.main(world.train_args(runs, "--final"), build=tiny_builder([])) == 2

    def test_fit_runs_the_fixed_schedule_and_selects_nothing(self) -> None:
        verifier = tiny_model()
        torch.manual_seed(0)
        audio = torch.randn(4, 50 * FRAME) * 0.1
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        snapshots = []

        def batches(epoch: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
            snapshots.append({k: v.clone() for k, v in verifier.state_dict().items()})
            return [(audio, labels)]

        config = train.TrainConfig(epochs=3, head_lr=1e-2)
        history = train.fit(verifier, batches, config, torch.device("cpu"))
        assert [row["epoch"] for row in history] == [1, 2, 3]
        final = verifier.state_dict()
        weights = "head.mlp.3.weight"
        assert not torch.equal(final[weights], snapshots[-1][weights])
        assert not torch.equal(snapshots[1][weights], snapshots[2][weights])
        assert not verifier.training

    def test_the_run_refuses_when_the_bundle_normalises(
        self, world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bundle = types.SimpleNamespace(_params=model.base_plus_params(), _normalize_waveform=True)
        monkeypatch.setattr(model, "WAVLM_BASE_PLUS", bundle)
        runs = world.root / "runs"
        assert train.main(world.train_args(runs, "--final"), build=tiny_builder([])) == 2
        assert "_normalize_waveform reads True" in capsys.readouterr().err
        assert not runs.exists()

    def test_an_index_row_outside_the_provenance_is_refused(
        self, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = world.dataset / dataset.PROVENANCE_FILE
        provenance = json.loads(path.read_text(encoding="utf-8"))
        provenance["file_sha256"] = provenance["file_sha256"][1:]
        path.write_text(json.dumps(provenance), encoding="utf-8")
        runs = world.root / "runs"
        assert train.main(world.train_args(runs, "--final"), build=tiny_builder([])) == 2
        assert "not in provenance.json" in capsys.readouterr().err

    @pytest.mark.parametrize("cli", ["train", "export"])
    def test_a_manifest_with_a_tune_entry_is_refused(
        self, world: World, cli: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        training = dataset.sha256_file(world.full_manifest)
        write_dataset(world.dataset, world.windows, training, {"ami": str(world.ami)})
        code = run_cli(cli, world, world.full_manifest)
        assert code == 2
        assert "outside the train split" in capsys.readouterr().err

    @pytest.mark.parametrize("cli", ["train", "export"])
    def test_a_manifest_that_is_not_the_indexs_training_is_refused(
        self, world: World, cli: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        other = world.root / "other.json"
        other.write_text(world.manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        assert run_cli(cli, world, other) == 2
        assert "is not the index's training" in capsys.readouterr().err

    def test_the_fold_split_keeps_groups_apart(self) -> None:
        ref = dataset.FileRef("ami", "c" * 64, "x.wav", None, 10 * WINDOW)
        windows = [
            dataset.Window(ref, 0, WINDOW, "negative", "mined_hit", group, ())
            for number in range(40)
            for group in (f"ami:g{number % 7}", f"manifest:s{number % 9}", f"piper:v{number % 3}")
        ]
        folds = train.assign_folds(windows)
        assert set(folds) == {window.group for window in windows}
        assert set(folds.values()) == set(range(train.FOLDS))
        sessions = {folds[f"manifest:s{n}"] for n in range(9)}
        assert sessions == set(range(train.FOLDS))
        assert folds == train.assign_folds(list(reversed(windows)))

    def test_truncation_is_the_same_for_either_class(self) -> None:
        ref = dataset.FileRef("ami", "c" * 64, "x.wav", None, 100 * WINDOW)
        positive = dataset.Window(ref, WINDOW, 2 * WINDOW, "positive", "synthetic", "g", ())
        negative = dataset.Window(ref, WINDOW, 2 * WINDOW, "negative", "synthetic", "g", ())
        config = train.TrainConfig(real_repeat=1)
        first = train.plan_epoch([positive, negative], [0, 1], config, 1)
        items = [item for batch in first for item in batch]
        by_window = {item.window: (item.first, item.last) for item in items}
        assert len(by_window) == 2
        lengths = {0: [], 1: []}
        for epoch in range(400):
            for batch in train.plan_epoch([positive, negative], [0, 1], config, epoch):
                for item in batch:
                    lengths[item.window].append(item.frames)
                    assert item.first <= 100 <= item.last
                    assert item.frames >= dataset.MIN_WINDOW_FRAMES
        assert min(lengths[0]) < 60 and max(lengths[0]) == 116
        assert np.mean(lengths[0]) == pytest.approx(np.mean(lengths[1]), abs=4)
        rng_a, rng_b = random.Random(5), random.Random(5)
        assert train.truncate(116, 100, rng_a, 1.0) == train.truncate(116, 100, rng_b, 1.0)
        audio = noise(2.32, 3).astype(np.float32) / 32768
        rirs = [np.array([1.0, 0.3, 0.1])]
        settings = train.Augment(rir_p=1.0, lowpass_p=1.0, noise_p=1.0, agc_p=1.0)
        one = train.augment(audio, np.random.default_rng(7), settings, rirs)
        two = train.augment(audio, np.random.default_rng(7), settings, rirs)
        assert np.array_equal(one, two)
        assert "label" not in train.augment.__code__.co_varnames
        assert "label" not in train.truncate.__code__.co_varnames

    def test_the_hit_offset_follows_window_bounds(self) -> None:
        for hit, frames in ((150, 400), (30, 400), (395, 400), (45, 50)):
            start, end = dataset.window_bounds(hit, frames)
            ref = dataset.FileRef("ami", "c" * 64, "x.wav", None, frames * FRAME)
            window = dataset.Window(ref, start, end, "negative", "random", "g", ())
            assert start // FRAME + train.hit_offset(window) == hit or end == frames * FRAME


def run_cli(cli: str, world: World, manifest: Path) -> int:
    if cli == "train":
        return train.main(
            world.train_args(world.root / "runs", "--final", manifest=manifest),
            build=tiny_builder([]),
        )
    return export.main(
        [
            "--run",
            str(world.root / "runs" / "final"),
            "--dataset",
            str(world.dataset),
            "--manifest",
            str(manifest),
            "--out",
            str(world.root / "export" / "model.onnx"),
        ]
    )


# -- The exporter -----------------------------------------------------------------------------


class TestExport:
    def test_a_two_layer_wavlm_matches_torch_through_onnx_reference(self, tmp_path: Path) -> None:
        import onnx
        from onnx.reference import ReferenceEvaluator

        torch.manual_seed(3)
        params = {**model.base_plus_params(), "encoder_num_layers": 2}
        verifier = model.build_verifier(params, first_trainable=2).eval()
        path = tmp_path / "model.onnx"
        export.export_onnx(verifier, path)
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        assert [item.name for item in graph.graph.input] == ["waveform"]
        assert [item.name for item in graph.graph.output] == ["score"]
        evaluator = ReferenceEvaluator(graph)
        rng = np.random.default_rng(11)
        for samples in (12800, 20480):
            waveform = (rng.standard_normal((1, samples)) * 0.1).astype(np.float32)
            (got,) = evaluator.run(None, {"waveform": waveform})
            with torch.no_grad():
                expected = model.ScoreGraph(verifier)(torch.from_numpy(waveform)).numpy()
            assert got.shape == (1,)
            assert np.max(np.abs(got - expected)) <= 1e-4

    def test_export_exits_2_naming_onnx_when_it_cannot_import_it(
        self, world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setitem(sys.modules, "onnx", None)
        assert run_cli("export", world, world.manifest) == 2
        assert "the onnx package cannot be imported" in capsys.readouterr().err

    def test_parity_compares_each_length_around_the_hit(self, world: World) -> None:
        verifier = tiny_model().eval()
        reader = train.WindowReader(
            train.train_manifest(world.manifest, {}, dataset.sha256_file(world.manifest)),
            {"ami": str(world.ami)},
        )
        lengths = []

        def drifting(waveform: np.ndarray) -> float:
            lengths.append(waveform.shape[1])
            with torch.no_grad():
                return float(model.ScoreGraph(verifier)(torch.from_numpy(waveform))[0]) + 0.002

        worst, compared = export.parity(verifier, drifting, world.windows, reader)
        assert compared == 3 * len(world.windows)
        assert set(lengths) == {40 * FRAME, 80 * FRAME, 116 * FRAME}
        assert worst == pytest.approx(0.002, abs=1e-6)
        assert worst > export.PARITY_BOUND


# -- Calibration ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    cutoffs: tuple[float, ...]
    score: float


class TestTable:
    def test_at_most_the_hourly_allowance_passes(self) -> None:
        hits = [Scored(GRID, score) for score in (0.95, 0.9, 0.8, 0.7, 0.6)]
        table = calibrate.threshold_table(hits, hours=10.0)
        assert table[0] == pytest.approx(0.8, abs=1e-12) and table[0] > 0.8
        assert calibrate.gated_counts(hits, table)[0] == 2

    def test_the_floor_is_half(self) -> None:
        hits = [Scored(GRID, 0.2), Scored(GRID, 0.3)]
        assert calibrate.threshold_table(hits, hours=1.0) == [0.5] * len(GRID)
        assert calibrate.threshold_table([], hours=1.0) == [0.5] * len(GRID)

    def test_a_lower_row_is_never_more_lenient(self) -> None:
        strict = Scored((0.8,), 0.97)
        loose = Scored(tuple(c for c in GRID if c <= 0.3), 0.6)
        table = calibrate.threshold_table([strict, loose], hours=1.0)
        row = dict(zip(GRID, table))
        assert row[0.8] > 0.97
        assert row[0.3] == row[0.8] and row[0.01] == row[0.8]
        assert row[0.81] == 0.5
        assert all(a >= b for a, b in zip(table, table[1:]))

    def test_no_audio_is_refused(self) -> None:
        with pytest.raises(calibrate.CalibrateError):
            calibrate.threshold_table([], hours=0.0)


def _rms(pcm: bytes) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    frames = len(samples) // FRAME
    return np.sqrt(np.mean(samples[: frames * FRAME].reshape(frames, FRAME) ** 2, axis=1))


@dataclass
class FakeSpotter:
    stride: int = 2
    cooldown_steps: int = 5

    def score_series(self, pcm: bytes) -> list[float]:
        return [float(value) for value in np.minimum(1.0, _rms(pcm) / 8000.0)]

    def probabilities(self, pcm: bytes) -> list[float]:
        return [0.0] * (len(pcm) // 320 // self.stride)


def fake_engines(training: str, score: float) -> Callable[[], calibrate.Engines]:
    def load() -> calibrate.Engines:
        return calibrate.Engines(
            spotter=FakeSpotter(),
            score=lambda pcm: score,
            warmup_samples=0,
            model_id="mist-mww-test",
            model_sha256="cd" * 32,
            training=training,
        )

    return load


def stage_shares(world: World) -> None:
    meetings = {"ES2005a": 1, "IS1000a": 2}
    for meeting, seed in meetings.items():
        samples = np.concatenate([noise(4.0, seed, 50), burst(0.4), noise(4.0, seed + 9, 50)])
        dataset.write_wav(world.ami / f"{meeting}.Mix-Headset.wav", samples)
    names = sorted(path.name for path in world.ami.glob("*.wav"))
    (world.ami / "SHA256SUMS").write_text(
        "".join(f"{dataset.sha256_file(world.ami / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    split = {
        "shares": {"calibration": ["ES2005a"], "check": ["IS1000a"], "mining": ["ES2002a"]},
        "groups": [["ES2002a"], ["ES2005a"], ["IS1000a"]],
        "excluded": [],
    }
    (world.ami / "split.json").write_text(json.dumps(split), encoding="utf-8")


def calibrate_args(world: World, out: Path) -> list[str]:
    model_file = world.root / "model.onnx"
    if not model_file.exists():
        model_file.write_bytes(b"an exported graph")
    return [
        "--model",
        str(model_file),
        "--dataset",
        str(world.dataset),
        "--models-dir",
        str(world.root / "models"),
        "--spotter",
        "mist-mww-test",
        "--model-id",
        out.name,
        "--threads",
        "2",
        "--workers",
        "1",
        "--out",
        str(out),
    ]


def run_calibrate(world: World) -> Path:
    stage_shares(world)
    out = world.root / "export" / "mist-verify-test"
    training = dataset.sha256_file(world.manifest)
    assert calibrate.main(calibrate_args(world, out), engines=fake_engines(training, 0.3)) == 0
    return out


@pytest.fixture
def verifier_dir(world: World) -> Path:
    return run_calibrate(world)


class TestCalibrate:
    def test_it_writes_the_manifest_the_verifier_reads(
        self, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        verifier_dir = run_calibrate(world)
        manifest = read_verifier_manifest(verifier_dir / "manifest.json", "mist-verify-test")
        assert manifest.training == dataset.sha256_file(world.manifest)
        assert manifest.spotter_model_id == "mist-mww-test"
        assert (manifest.before_ms, manifest.after_ms, manifest.min_window_ms) == (2000, 300, 800)
        assert manifest.per_hour == 0.2
        assert manifest.scores == tuple([0.5] * len(GRID))
        assert (verifier_dir / "provenance.json").read_bytes() == (
            world.dataset / "provenance.json"
        ).read_bytes()
        record = json.loads((verifier_dir / "calibration.json").read_text(encoding="utf-8"))
        assert record["hits"]["calibration"] and record["hits"]["check"]
        assert all(hit["meeting"] == "IS1000a" for hit in record["hits"]["check"])
        out = capsys.readouterr().out
        assert "cutoff 0.50: threshold 0.5000, check gated 0 (0.000 an hour)" in out
        assert "ES2005a" not in out and "IS1000a" not in out

    def test_a_share_meeting_read_in_training_is_refused(
        self, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage_shares(world)
        split = json.loads((world.ami / "split.json").read_text(encoding="utf-8"))
        split["shares"]["check"] = ["ES2002a"]
        (world.ami / "split.json").write_text(json.dumps(split), encoding="utf-8")
        out = world.root / "export" / "mist-verify-test"
        training = dataset.sha256_file(world.manifest)
        code = calibrate.main(calibrate_args(world, out), engines=fake_engines(training, 0.3))
        assert code == 2
        assert "was read in training" in capsys.readouterr().err

    def test_a_spotter_from_another_training_is_refused(self, world: World) -> None:
        stage_shares(world)
        out = world.root / "export" / "mist-verify-test"
        assert calibrate.main(calibrate_args(world, out), engines=fake_engines("e" * 64, 0.3)) == 2


# -- The pre-flight ---------------------------------------------------------------------------


def write_runs(runs: Path, positives: dict[int, list[float]], homophone: float = 0.1) -> None:
    for fold in range(train.FOLDS):
        run = runs / f"fold-{fold}"
        run.mkdir(parents=True)
        rows = [
            {
                "window": number,
                "kind": "real_positive",
                "label": "positive",
                "group": f"manifest:s{fold}",
                "file_sha256": f"{fold}{number:063d}",
                "cutoffs": list(GRID),
                "score": score,
            }
            for number, score in enumerate(positives.get(fold, []))
        ]
        rows.append(
            {
                "window": 99,
                "kind": "homophone",
                "label": "negative",
                "group": f"manifest:h{fold}",
                "file_sha256": "f" * 64,
                "cutoffs": [0.5],
                "score": homophone,
            }
        )
        (run / train.RUN_OOF).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


class TestPreflight:
    def _setup(self, world: World, verifier_dir: Path, *, wakes: int, drift: float) -> list[str]:
        runs = world.root / "runs"
        write_runs(runs, {fold: [0.9, 0.95, 0.99, 0.2] for fold in range(train.FOLDS)})
        report = {"parity": {"max_abs_diff": drift}}
        (verifier_dir / "export.json").write_text(json.dumps(report), encoding="utf-8")
        evaluation = world.root / "spotter_eval.json"
        evaluation.write_text(json.dumps({"positive_scores": [0.9] * wakes}), encoding="utf-8")
        return [
            "--runs",
            str(runs),
            "--verifier",
            str(verifier_dir),
            "--spotter-eval",
            str(evaluation),
            "--out",
            str(world.root / "preflight.json"),
        ]

    @staticmethod
    def scorer(seen: list[int]) -> preflight.FoldScorer:
        def score(run: Path, hits: list, ami_root: str) -> list[float]:
            seen.append(len(hits))
            return [0.0] * len(hits)

        return score

    def test_a_pass(
        self, world: World, verifier_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seen: list[int] = []
        args = self._setup(world, verifier_dir, wakes=20, drift=1e-5)
        assert preflight.main(args, scorer=self.scorer(seen)) == 0
        record = json.loads((verifier_dir / "calibration.json").read_text(encoding="utf-8"))
        assert seen == [len(record["hits"]["calibration"])] * train.FOLDS
        result = json.loads((world.root / "preflight.json").read_text(encoding="utf-8"))
        row = result["rows"][GRID.index(0.9)]
        assert (row["positives_with_hit"], row["positives_passed"]) == (20, 15)
        assert row["predicted_wakes"] == pytest.approx(15.0)
        assert result["c1"] == 0.9 and result["pass"] is True
        assert result["raj_channel"]["homophone"][GRID.index(0.5)] == 0
        out = capsys.readouterr().out
        assert "pre-flight passed" in out
        assert "provenance manifest: 6 files" in out

    def test_too_few_predicted_wakes_fail(self, world: World, verifier_dir: Path) -> None:
        args = self._setup(world, verifier_dir, wakes=19, drift=1e-5)
        assert preflight.main(args, scorer=self.scorer([])) == 1
        result = json.loads((world.root / "preflight.json").read_text(encoding="utf-8"))
        assert result["c1"] is None
        assert result["failures"] == ["no cutoff predicts 15 gated tune wakes"]

    def test_parity_outside_the_bound_fails(self, world: World, verifier_dir: Path) -> None:
        args = self._setup(world, verifier_dir, wakes=20, drift=2e-3)
        assert preflight.main(args, scorer=self.scorer([])) == 1

    def test_a_missing_fold_is_refused(self, world: World, verifier_dir: Path) -> None:
        args = self._setup(world, verifier_dir, wakes=20, drift=1e-5)
        (world.root / "runs" / "fold-4" / train.RUN_OOF).unlink()
        assert preflight.main(args, scorer=self.scorer([])) == 2

    def test_only_positive_scores_is_read(self, world: World, verifier_dir: Path) -> None:
        args = self._setup(world, verifier_dir, wakes=20, drift=1e-5)
        evaluation = Path(args[args.index("--spotter-eval") + 1])
        evaluation.write_text(
            json.dumps({"positive_scores": [0.9] * 20, "cutoffs": "not read"}), encoding="utf-8"
        )
        assert preflight.main(args, scorer=self.scorer([])) == 0
        evaluation.write_text(json.dumps({"cutoffs": [0.5]}), encoding="utf-8")
        assert preflight.main(args, scorer=self.scorer([])) == 2

    def test_pass_rates_compare_each_positive_with_its_folds_table(self) -> None:
        oof = [
            preflight.OutOfFold(0, "real_positive", "a" * 64, (0.5,), 0.7),
            preflight.OutOfFold(0, "real_positive", "a" * 64, (0.5,), 0.55),
            preflight.OutOfFold(1, "real_positive", "b" * 64, (0.5,), 0.65),
            preflight.OutOfFold(1, "homophone", "c" * 64, (0.5,), 0.99),
        ]
        tables = {0: [0.6] * len(GRID), 1: [0.8] * len(GRID)}
        rates = preflight.pass_rates(oof, tables)
        assert rates[GRID.index(0.5)] == (2, 1)
        assert rates[GRID.index(0.6)] == (0, 0)
        raj = preflight.raj_channel(oof, tables)
        assert raj["homophone"][GRID.index(0.5)] == 1
        assert sum(raj["homophone"]) == 1


def test_notice_names_wavlm_base_plus() -> None:
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert "WavLM Base+" in notice
    assert "provenance.json" in notice
