"""Tests for the verifier's trainer, exporter, calibrator and pre-flight.

Every input is generated here: low noise and tone bursts as WAV files, a train-only recording
manifest, a dataset index over them with its `provenance.json`, an AMI directory with its shares,
and fold runs with out-of-fold scores. The trainer runs a tiny stand-in encoder; the WavLM tests
build the real Base+ shape or a two-layer one from random initialisation, and the export is
checked through `onnx.reference`, since the runner image has no onnxruntime. No test reads a
recording, needs a network or the GPU.
"""

import collections
import itertools
import json
import random
import re
import shutil
import sys
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from backend.realtime.engines.verifier import GRID, read_verifier_manifest
from scripts.wake.verifier import (
    augment,
    calibrate,
    dataset,
    export,
    model,
    preflight,
    store,
    train,
)

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


@pytest.fixture(autouse=True)
def _no_free_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build every store with no spare space required beyond the store itself.

    The stores here hold a few windows, but `store.FREE_MARGIN` asks for 1 GiB more, which a
    small temporary directory (a 1 GiB tmpfs at `/tmp`) does not have.
    """
    monkeypatch.setattr(store, "FREE_MARGIN", 0)


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
    def build(config: train.TrainConfig, weights: Path | None) -> model.VerifierModel:
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
        assert train.Augment is augment.Augment
        assert "label" not in augment.draw.__code__.co_varnames
        assert "label" not in augment.augment_batch.__code__.co_varnames
        assert "label" not in train.truncate.__code__.co_varnames

    def test_the_hit_offset_follows_window_bounds(self) -> None:
        for hit, frames in ((150, 400), (30, 400), (395, 400), (45, 50)):
            start, end = dataset.window_bounds(hit, frames)
            ref = dataset.FileRef("ami", "c" * 64, "x.wav", None, frames * FRAME)
            window = dataset.Window(ref, start, end, "negative", "random", "g", ())
            assert start // FRAME + train.hit_offset(window) == hit or end == frames * FRAME

    def test_the_reader_takes_channel_0_of_a_room_response_and_each_census_format(
        self, world: World, tmp_path: Path
    ) -> None:
        import soundfile

        from tests.unit.scripts.test_verifier_data import (
            room_response,
            write_channels,
            write_extensible,
        )

        rirs = tmp_path / "rirs"
        responses = {"r8.wav": room_response(8), "r30.wav": room_response(30)}
        for name, frames in responses.items():
            write_extensible(rirs / name, frames)
        responses["r2.wav"] = room_response(2)
        write_channels(rirs / "r2.wav", responses["r2.wav"])
        musan = tmp_path / "musan"
        mono = noise(1.0, 5)
        write_extensible(musan / "n.wav", mono[:, None])
        librispeech = tmp_path / "librispeech"
        librispeech.mkdir()
        soundfile.write(str(librispeech / "19-198-0001.flac"), mono, RATE, subtype="PCM_16")
        reader = train.WindowReader(
            train.train_manifest(world.manifest, {}, dataset.sha256_file(world.manifest)),
            {"rirs": str(rirs), "musan": str(musan), "librispeech": str(librispeech)},
        )
        for name, frames in responses.items():
            ref = dataset.FileRef("rirs", "a" * 64, name, None, len(frames))
            first = frames[:, 0].astype(np.float32) / np.float32(32768)
            np.testing.assert_allclose(reader.rir(ref), first / np.max(np.abs(first)), rtol=1e-6)
        for source, file in (("musan", "n.wav"), ("librispeech", "19-198-0001.flac")):
            ref = dataset.FileRef(source, "b" * 64, file, None, len(mono))
            window = dataset.Window(ref, 2 * FRAME, 50 * FRAME, "negative", "random", "g", ())
            np.testing.assert_array_equal(reader.samples(window), mono[2 * FRAME : 50 * FRAME])


# -- The store, the loader, scoring and progress -----------------------------------------------


CONFIG_KEYS = {
    "version",
    "run",
    "fold",
    "optimizer",
    "layers_lr",
    "head_lr",
    "weight_decay",
    "batch",
    "accumulate",
    "epochs",
    "fp16",
    "seed",
    "first_trainable_layer",
    "folds",
    "real_repeat",
    "keep",
    "loader_workers",
    "augment",
    "fp16_autocast",
    "device",
    "params",
    "normalize_waveform",
    "training",
    "provenance_sha256",
    "weights_sha256",
    "kept_epoch",
    "windows",
}
"""`config.json`'s keys as `train.py` wrote them at dc98960c."""
AUGMENT_KEYS = {
    "truncate_p",
    "gain_db",
    "noise_p",
    "noise_snr_db",
    "rir_p",
    "lowpass_p",
    "lowpass_hz",
    "agc_p",
    "agc_target_rms",
    "agc_max_gain_db",
}
OOF_KEYS = {"window", "kind", "label", "group", "file_sha256", "cutoffs", "score"}
FIXTURE_NAMES = (
    "rec/",
    "pos-",
    "hom-",
    "ES2002a",
    ".wav",
    "recordings",
    "dataset",
    "train_manifest",
    "weights",
    "window-cache",
    "runs",
    "rirs",
)


def _private(window: dataset.Window) -> bool:
    return any(ref.source == "manifest" for ref in window.refs())


def _reader(world: World, roots: dict[str, str] | None = None) -> train.WindowReader:
    manifest = train.train_manifest(world.manifest, {}, dataset.sha256_file(world.manifest))
    return train.WindowReader(manifest, roots or {"ami": str(world.ami)})


def _decay(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 8000, count) * np.exp(-np.arange(count) / (count / 6))).astype("<i2")


def _responses(directory: Path) -> list[dataset.FileRef]:
    refs = []
    for name, count in (("short.wav", 600), ("mid.wav", 2000), ("long.wav", 3 * RATE)):
        path = directory / name
        dataset.write_wav(path, _decay(count, count))
        refs.append(dataset.FileRef("rirs", dataset.sha256_file(path), name, None, count))
    return refs


def _open(root: Path, windows, responses, reader) -> store.WindowStore:
    return store.open_store(
        root,
        "a" * 64,
        windows,
        responses,
        reader,
        provenance_sha256="b" * 64,
        workers=1,
        progress=lambda line: None,
    )


class TestStoreAndLoader:
    def test_a_fold_builds_the_store_and_the_final_run_reuses_it(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = collections.Counter()
        samples = train.WindowReader.samples

        def counting(self: train.WindowReader, window: dataset.Window) -> np.ndarray:
            calls["private" if _private(window) else "public"] += 1
            return samples(self, window)

        monkeypatch.setattr(train.WindowReader, "samples", counting)
        runs = world.root / "runs"
        one = ("--store-workers", "1")
        assert train.main(world.train_args(runs, "--fold", "0", *one), build=tiny_builder([])) == 0
        key = dataset.sha256_file(world.dataset / dataset.INDEX_FILE)
        cache = runs / train.CACHE_DIR
        assert sorted(path.name for path in cache.iterdir()) == [".lock", key]
        public = [window for window in world.windows if not _private(window)]
        private = [window for window in world.windows if _private(window)]
        assert public and private
        assert calls["public"] == len(public)
        calls.clear()
        assert train.main(world.train_args(runs, "--final", *one), build=tiny_builder([])) == 0
        assert calls["public"] == 0
        assert calls["private"] == len(private)
        stored = (cache / key / store.WINDOWS_FILE).read_bytes()
        assert len(stored) == sum(2 * (window.end - window.start) for window in public)
        reader = _reader(world)
        for window in private:
            assert samples(reader, window).tobytes() not in stored
            assert samples(reader, window)[: RATE // 2].tobytes() not in stored

    def test_config_and_oof_keep_their_keys_and_record_the_options(
        self, world: World, tmp_path: Path
    ) -> None:
        runs = world.root / "runs"
        cache = tmp_path / "elsewhere"
        options = ("--cache", str(cache), "--loader-workers", "2", "--progress-seconds", "5")
        args = world.train_args(runs, "--fold", "1", *options)
        assert train.main(args, build=tiny_builder([])) == 0
        assert not (runs / train.CACHE_DIR).exists()
        assert (cache / store.index_key(world.dataset) / store.RECORD_FILE).exists()
        config = json.loads((runs / "fold-1" / train.RUN_CONFIG).read_text(encoding="utf-8"))
        assert set(config) == CONFIG_KEYS
        assert set(config["augment"]) == AUGMENT_KEYS
        assert config["loader_workers"] == 2
        lines = (runs / "fold-1" / train.RUN_OOF).read_text(encoding="utf-8").splitlines()
        assert lines and all(set(json.loads(line)) == OOF_KEYS for line in lines)

    def test_a_store_that_cannot_be_built_exits_2(
        self, world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        usage = collections.namedtuple("usage", "total used free")
        monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(1 << 30, 1 << 30, 0))
        runs = world.root / "runs"
        assert train.main(world.train_args(runs, "--final"), build=tiny_builder([])) == 2
        assert "train: the store needs" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "option",
        [("--loader-workers", "-1"), ("--store-workers", "0"), ("--progress-seconds", "0")],
    )
    def test_a_bad_option_is_refused(self, world: World, option: tuple[str, str]) -> None:
        runs = world.root / "runs"
        assert train.main(world.train_args(runs, "--final", *option), build=tiny_builder([])) == 2
        assert not runs.exists()

    def test_batched_scores_equal_batch_one_fp32_scores(self, world: World, tmp_path: Path) -> None:
        ami = world.windows[-1].file
        extra = [
            dataset.Window(ami, start, start + frames * FRAME, "negative", "mined_hit", "ami:x", ())
            for start, frames in ((0, 40), (7 * FRAME, 40), (3 * FRAME, 80), (50 * FRAME, 80))
        ]
        windows = [*world.windows, *extra, extra[0]]
        reader = _reader(world)
        opened = _open(tmp_path / "cache", windows, [], reader)
        torch.manual_seed(5)
        verifier = tiny_model()
        verifier.train()
        held = [9, 0, 3, 10, 6, 12, 8, 1, 11, 2, 7]
        lines: list[str] = []
        scores = train.score_windows(
            verifier, opened, held, torch.device("cpu"), batch=3, progress=lines.append
        )
        assert len(scores) == len(held)
        verifier.eval()
        for index, score in zip(held, scores, strict=True):
            audio = reader.samples(windows[index]).astype(np.float32) / np.float32(32768.0)
            with torch.no_grad():
                logit = verifier(torch.from_numpy(audio).reshape(1, -1))
            assert score == pytest.approx(float(torch.sigmoid(logit.float())[0]), abs=1e-5)
        assert scores[held.index(8)] == pytest.approx(scores[held.index(12)], abs=1e-6)
        assert f"{len(held)}/{len(held)} windows" in lines[-1]
        assert "windows/s" in lines[-1] and "ETA" in lines[-1]

    def test_the_loaders_batches_are_the_stores_kept_frames_with_each_items_draws(
        self, world: World, tmp_path: Path
    ) -> None:
        responses = _responses(tmp_path / "rirs")
        reader = _reader(world, {"ami": str(world.ami), "rirs": str(tmp_path / "rirs")})
        opened = _open(tmp_path / "cache", world.windows, responses, reader)
        config = train.TrainConfig(batch=4, augment=augment.Augment(rir_p=0.5))
        plan = train.plan_epoch(world.windows, range(len(world.windows)), config, 1)
        labels = [1.0 if window.label == "positive" else 0.0 for window in world.windows]
        data = train.EpochBatches(plan, labels, opened, config.augment)
        assert len(data) == len(plan)
        rows = collections.Counter()
        for index, items in enumerate(plan):
            batch = data[index]
            length = items[0].frames * FRAME
            assert batch.audio.dtype == torch.int16
            assert tuple(batch.audio.shape) == (len(items), length)
            assert batch.labels.dtype == torch.float32
            draws = [augment.draw(item.seed, config.augment, len(responses)) for item in items]
            expected = augment.batch_draws(draws)
            for name in (item.name for item in fields(augment.BatchDraws)):
                assert torch.equal(getattr(batch.draws, name), getattr(expected, name)), name
            taps = [opened.response(d.rir)[:length] for d in draws if d.rir is not None]
            width = max((len(row) for row in taps), default=0)
            assert tuple(batch.responses.shape) == (len(items), width)
            for row, (item, choice) in enumerate(zip(items, draws, strict=True)):
                window = world.windows[item.window]
                kept = reader.samples(window)[item.first * FRAME : (item.last + 1) * FRAME]
                np.testing.assert_array_equal(batch.audio[row].numpy(), kept)
                assert float(batch.labels[row]) == labels[item.window]
                got = batch.responses[row].numpy()
                if choice.rir is None:
                    assert not got.any()
                    rows["none"] += 1
                    continue
                response = opened.response(choice.rir)[:length]
                np.testing.assert_array_equal(got[: len(response)], response)
                assert not got[len(response) :].any()
                rows["room"] += 1
        assert rows["room"] and rows["none"]
        quiet = train.TrainConfig(batch=4, augment=augment.Augment(rir_p=0.0))
        silent = train.EpochBatches(plan, labels, opened, quiet.augment)
        assert silent[0].responses.shape == (len(plan[0]), 0)

    def test_loader_workers_hand_over_the_same_batches_and_the_device_augments_them(
        self, world: World, tmp_path: Path
    ) -> None:
        responses = _responses(tmp_path / "rirs")
        reader = _reader(world, {"ami": str(world.ami), "rirs": str(tmp_path / "rirs")})
        opened = _open(tmp_path / "cache", world.windows, responses, reader)
        config = train.TrainConfig(batch=8, augment=augment.Augment(rir_p=0.5))
        plan = train.plan_epoch(world.windows, range(len(world.windows)), config, 2)
        labels = [1.0 if window.label == "positive" else 0.0 for window in world.windows]
        data = train.EpochBatches(plan, labels, opened, config.augment)
        loader = train.epoch_loader(data, 2, pin=False)
        assert loader.prefetch_factor == train.PREFETCH_BATCHES
        assert loader.worker_init_fn is not None
        handed = list(loader)
        assert len(handed) == len(plan)
        for index, batch in enumerate(handed):
            direct = data[index]
            assert torch.equal(batch.audio, direct.audio)
            assert torch.equal(batch.responses, direct.responses)
            assert torch.equal(batch.draws.noise_seeds, direct.draws.noise_seeds)
        examples = sum(len(items) for items in plan)
        epoch = train.DeviceEpoch(handed, examples, torch.device("cpu"), config.augment)
        for (waveforms, batch_labels), direct in zip(epoch, handed, strict=True):
            expected = augment.augment_batch(
                direct.audio, direct.draws, direct.responses, config.augment
            )
            assert waveforms.dtype == torch.float32
            assert torch.equal(waveforms, expected)
            assert torch.equal(batch_labels, direct.labels)

    def test_progress_lines_carry_a_rate_and_an_eta_and_name_no_file(
        self, world: World, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ticks = itertools.count(0.0, 61.0)
        runs = world.root / "runs"
        args = world.train_args(runs, "--fold", "0", "--store-workers", "1")
        code = train.main(args, build=tiny_builder([]), clock=lambda: float(next(ticks)))
        assert code == 0
        lines = capsys.readouterr().out.splitlines()
        ends = [line for line in lines if re.match(r"epoch \d+: loss ", line)]
        steps = [line for line in lines if re.match(r"epoch \d+: \d+/\d+ examples", line)]
        assert len(ends) == 6 and steps
        end = r"epoch \d+: loss \d+\.\d{4} over \d+ examples, \d+\.\d examples/s, run ETA \S+$"
        assert all(re.match(end, line) for line in ends)
        assert ends[-1].endswith("run ETA 0:00:00")
        for line in steps:
            assert "examples/s" in line and "epoch ETA" in line and "run ETA" in line
            assert "unknown" not in line
        scoring = [line for line in lines if line.startswith("oof: ")]
        assert scoring and "windows/s" in scoring[-1] and "ETA" in scoring[-1]
        assert any(line.startswith("store: ") for line in lines)
        key = store.index_key(world.dataset)
        for line in lines:
            assert str(world.root) not in line
            assert key not in line
            for name in FIXTURE_NAMES:
                assert name not in line, line

    def test_fit_reads_the_loss_only_for_a_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        verifier = tiny_model()
        audio = torch.randn(4, 50 * FRAME) * 0.1
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        reads = []
        item = torch.Tensor.item

        def counting(self: torch.Tensor) -> float:
            reads.append(tuple(self.shape))
            return float(item(self))

        monkeypatch.setattr(torch.Tensor, "__float__", counting)
        lines: list[str] = []
        config = train.TrainConfig(epochs=2, accumulate=2)
        history = train.fit(
            verifier,
            lambda epoch: [(audio, labels)] * 3,
            config,
            torch.device("cpu"),
            progress=lines.append,
            clock=lambda: 0.0,
        )
        assert len(reads) == config.epochs
        assert [row["examples"] for row in history] == [12, 12]
        assert lines == [
            f"epoch {row['epoch']}: loss {row['loss']:.4f} over 12 examples, 0.0 examples/s, "
            f"run ETA {'unknown' if row['epoch'] == 1 else '0:00:00'}"
            for row in history
        ]


# -- DeviceEpoch -------------------------------------------------------------------------------


DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="the side stream needs an accelerator"
        ),
    ),
]
EPOCH_AUGMENT = augment.Augment(rir_p=0.5, noise_p=0.5, lowpass_p=0.5, agc_p=0.5)
TAPS = 64


def fixture_batches(count: int = 5) -> list[train.Batch]:
    """`count` synthetic batches of differing lengths and sizes, each item's choices from a seed."""
    rng = np.random.default_rng(11)
    decay = np.exp(-np.arange(TAPS) / 8.0)
    batches = []
    for number in range(count):
        size, length = 3 + number % 2, (40 + 7 * number) * FRAME
        draws = [augment.draw(100 * number + row, EPOCH_AUGMENT, 2) for row in range(size)]
        responses = np.zeros((size, TAPS), dtype=np.float32)
        for row, choice in enumerate(draws):
            if choice.rir is not None:
                responses[row] = (rng.standard_normal(TAPS) * decay).astype(np.float32)
        batches.append(
            train.Batch(
                torch.from_numpy(rng.integers(-2000, 2001, (size, length)).astype(np.int16)),
                torch.tensor([float((number + row) % 2) for row in range(size)]),
                augment.batch_draws(draws),
                torch.from_numpy(responses),
            )
        )
    return batches


def _hosted(batches: list[train.Batch], device: torch.device) -> list[train.Batch]:
    """The batches as the trainer's loader hands them over: pinned on CUDA."""
    return [batch.pin_memory() for batch in batches] if device.type == "cuda" else batches


def _examples(batches: list[train.Batch]) -> int:
    return sum(len(batch.labels) for batch in batches)


class _Tracked:
    """A loader iterator the test keeps a reference to, so only an explicit close closes it."""

    def __init__(self, batches: list[train.Batch]) -> None:
        self.batches = iter(batches)
        self.closed = False

    def __iter__(self) -> "_Tracked":
        return self

    def __next__(self) -> train.Batch:
        return next(self.batches)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("device", DEVICES)
def test_device_epoch_hands_over_a_plain_loops_batches_in_order(device: str) -> None:
    target = torch.device(device)
    batches = fixture_batches()
    expected = []
    for batch in batches:
        moved = batch.to(target)
        with torch.no_grad():
            waveforms = augment.augment_batch(
                moved.audio, moved.draws, moved.responses, EPOCH_AUGMENT
            )
        expected.append((waveforms.cpu(), moved.labels.cpu()))
    epoch = train.DeviceEpoch(_hosted(batches, target), _examples(batches), target, EPOCH_AUGMENT)
    handed = []
    for waveforms, labels in epoch:
        assert waveforms.device.type == labels.device.type == target.type
        handed.append((waveforms.cpu(), labels.cpu()))
    assert len(handed) == len(expected)
    for (waveforms, labels), (want, want_labels) in zip(handed, expected, strict=True):
        assert waveforms.dtype == torch.float32
        assert torch.equal(waveforms, want)
        assert torch.equal(labels, want_labels)


def test_a_loader_error_mid_epoch_propagates_out_of_the_device_epoch() -> None:
    batches = fixture_batches(3)

    def failing() -> Iterator[train.Batch]:
        yield batches[0]
        yield batches[1]
        raise OSError("the store went away")

    epoch = train.DeviceEpoch(failing(), _examples(batches), torch.device("cpu"), EPOCH_AUGMENT)
    handed = []
    with pytest.raises(OSError, match="the store went away"):
        for waveforms, _ in epoch:
            handed.append(waveforms)
    assert len(handed) == 2


@pytest.mark.parametrize("device", DEVICES)
def test_a_consumer_that_stops_early_closes_the_loaders_iterator(device: str) -> None:
    target = torch.device(device)
    batches = fixture_batches(4)
    loader = _Tracked(_hosted(batches, target))
    epoch = iter(train.DeviceEpoch(loader, _examples(batches), target, EPOCH_AUGMENT))
    next(epoch)
    assert not loader.closed
    epoch.close()
    assert loader.closed


# -- The command line and the benchmark --------------------------------------------------------


H37_FOLD = [
    "--dataset",
    "d",
    "--manifest",
    "m.json",
    "--root-map",
    "/from=/to",
    "--weights",
    "w.pth",
    "--out",
    "o",
    "--fold",
    "3",
    "--batch",
    "32",
    "--accumulate",
    "2",
    "--seed",
    "17",
    "--cache",
    "c",
    "--store-workers",
    "8",
    "--loader-workers",
    "4",
    "--progress-seconds",
    "60",
]
TINY_BENCHMARK = (
    "--benchmark",
    "--benchmark-seconds",
    "0.05",
    "--benchmark-workers",
    "0,1",
    "--benchmark-windows",
    "48",
    "--benchmark-decode",
    "2000",
    "--loader-workers",
    "1",
    "--store-workers",
    "1",
)


class TestCommandLine:
    def test_h37s_commands_parse_as_before(self) -> None:
        args = train.parse_args(H37_FOLD)
        assert args.benchmark is False
        assert (args.dataset, args.manifest, args.weights, args.out) == (
            Path("d"),
            Path("m.json"),
            Path("w.pth"),
            Path("o"),
        )
        assert (args.fold, args.final, args.root_map, args.cache) == (
            3,
            False,
            ["/from=/to"],
            Path("c"),
        )
        assert (args.batch, args.accumulate, args.seed) == (32, 2, 17)
        assert (args.store_workers, args.loader_workers, args.progress_seconds) == (8, 4, 60.0)
        final = [*H37_FOLD[: H37_FOLD.index("--fold")], "--final"]
        args = train.parse_args(final)
        assert (args.fold, args.final) == (None, True)
        args = train.parse_args(H37_FOLD[:10] + ["--final"])
        assert (args.batch, args.cache, args.store_workers) == (32, None, train.STORE_WORKERS)
        assert args.loader_workers == 4 and args.progress_seconds == train.PROGRESS_SECONDS

    @pytest.mark.parametrize(
        ("drop", "message"),
        [
            ("--manifest", "the following arguments are required: --manifest"),
            ("--weights", "the following arguments are required: --weights"),
            ("--fold", "one of the arguments --fold --final is required"),
        ],
    )
    def test_a_missing_argument_still_exits_2_with_argparses_message(
        self, drop: str, message: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = list(H37_FOLD)
        at = argv.index(drop)
        del argv[at : at + 2]
        with pytest.raises(SystemExit) as exited:
            train.main(argv, build=tiny_builder([]))
        assert exited.value.code == 2
        assert message in capsys.readouterr().err

    def test_the_benchmark_and_its_flags_are_optional(self) -> None:
        args = train.parse_args(["--benchmark"])
        assert args.benchmark is True
        assert args.dataset is None and args.out is None
        assert (args.benchmark_seconds, args.benchmark_workers) == (30.0, (0, 1, 2, 4, 8))
        assert (args.benchmark_windows, args.benchmark_decode, args.batch) == (16384, 2000, 32)
        args = train.parse_args(
            ["--benchmark", "--dataset", "d", "--out", "o", "--benchmark-workers", "2,4"]
        )
        assert (args.dataset, args.out, args.benchmark_workers) == (Path("d"), Path("o"), (2, 4))

    @pytest.mark.parametrize(
        "option",
        [
            ("--manifest", "m.json"),
            ("--weights", "w.pth"),
            ("--fold", "0"),
            ("--final",),
            ("--cache", "c"),
        ],
    )
    def test_the_benchmark_refuses_a_training_input(
        self, option: tuple[str, ...], capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exited:
            train.parse_args(["--benchmark", *option])
        assert exited.value.code == 2
        assert f"{option[0]} is refused" in capsys.readouterr().err

    def test_a_benchmark_flag_without_the_benchmark_is_refused(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exited:
            train.parse_args([*H37_FOLD, "--benchmark-seconds", "5"])
        assert exited.value.code == 2
        assert "--benchmark-seconds is read only with --benchmark" in capsys.readouterr().err


def _refuse_private_reads(world: World, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make the manifest loaders and any read of a recording fail; returns the files read."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the benchmark read a manifest")

    monkeypatch.setattr(train, "load_manifest", refuse)
    monkeypatch.setattr(train, "train_manifest", refuse)
    read = train.read_wav_segment
    opened: list[str] = []

    def segment(path: Path, start: int, count: int) -> np.ndarray:
        assert world.ami in Path(path).parents, "the benchmark read a recording"
        opened.append(Path(path).name)
        return read(path, start, count)

    monkeypatch.setattr(train, "read_wav_segment", segment)
    return opened


class TestBenchmark:
    def test_it_prints_each_measurement_and_the_projection(
        self, world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        opened = _refuse_private_reads(world, monkeypatch)
        out = world.root / "bench"
        argv = [*TINY_BENCHMARK, "--dataset", str(world.dataset), "--out", str(out)]
        assert train.main(argv, build=tiny_builder([])) == 0
        assert opened and set(opened) == {"ES2002a.Mix-Headset.wav"}
        assert not (out / train.BENCHMARK_CACHE).exists()
        lines = capsys.readouterr().out.splitlines()
        starts = [
            "device: cpu",
            "synthetic store: 48 windows",
            "store cold read: ",
            "store cold read as examples: ",
            "loader: 0 workers, ",
            "loader: 1 workers, ",
            "step: cpu, ",
            "end-to-end: 0 loader workers, ",
            "scoring: ",
            "index: 8 windows, 2 public",
            "store build: 2 public windows",
            "projection: training rate min(",
            "projection: final training 6 x 44 / ",
            "projection: store build 2 / ",
            "projection: total ",
            "ratio: loader / step = ",
        ]
        for start in starts:
            assert any(line.startswith(start) for line in lines), start
        assert not any(line.startswith("step: from pinned host batches") for line in lines)
        folds = train.assign_folds(world.windows)
        for fold in range(train.FOLDS):
            held = sum(1 for window in world.windows if folds[window.group] == fold)
            trained = sum(
                train.REAL_REPEAT if window.kind == "real_positive" else 1
                for window in world.windows
                if folds[window.group] != fold
            )
            assert any(
                line.startswith(f"projection: fold {fold} training 6 x {trained} / ")
                for line in lines
            ), fold
            assert any(
                line.startswith(f"projection: fold {fold} scoring {held} / ") for line in lines
            ), fold
        assert re.search(r"^projection: total \d+ s = \d+\.\d\d h$", lines[-2])
        assert re.search(r"^ratio: loader / step = \S+ / \S+ = \d+\.\d\d$", lines[-1])
        for line in lines:
            assert str(world.root) not in line and str(out) not in line
            assert train.BENCHMARK_CACHE not in line
            for name in FIXTURE_NAMES:
                assert name not in line, line

    def test_without_an_index_it_prints_seconds_per_million_examples(
        self,
        world: World,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _refuse_private_reads(world, monkeypatch)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(train.tempfile, "tempdir", str(scratch))
        assert train.main(list(TINY_BENCHMARK), build=tiny_builder([])) == 0
        assert not [path for path in scratch.iterdir() if "verifier-benchmark" in path.name]
        lines = capsys.readouterr().out.splitlines()
        assert not any(line.startswith(("index:", "store build:")) for line in lines)
        assert re.search(
            r"^projection: 1000000 / \S+ = \d+ s a million training examples$", lines[-2]
        )
        assert lines[-1].startswith("ratio: loader / step = ")

    def test_it_removes_its_cache_when_it_fails(
        self, world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _refuse_private_reads(world, monkeypatch)
        out = world.root / "bench"
        seen: list[bool] = []

        def failing(*args: object, **kwargs: object) -> list[float]:
            seen.append((out / train.BENCHMARK_CACHE).is_dir())
            raise store.StoreError("scoring failed")

        monkeypatch.setattr(train, "score_windows", failing)
        argv = [*TINY_BENCHMARK, "--dataset", str(world.dataset), "--out", str(out)]
        assert train.main(argv, build=tiny_builder([])) == 2
        assert seen == [True]
        assert not (out / train.BENCHMARK_CACHE).exists()
        assert "train: scoring failed" in capsys.readouterr().err

    def test_the_projection_is_printed_as_arithmetic(self) -> None:
        workload = train.Workload(folds=((100, 10), (200, 20)), final=300, public=50)
        rates = train.Rates(step=10.0, loader=20.0, cold_read=40.0, scoring=5.0, build=25.0)
        lines = train.projection(workload, rates, epochs=6)
        assert lines == [
            "projection: training rate min(step 10.0, loader 20.0, cold read 40.0) "
            "= 10.0 examples/s",
            "projection: fold 0 training 6 x 100 / 10.0 = 60 s",
            "projection: fold 0 scoring 10 / 5.0 = 2 s",
            "projection: fold 1 training 6 x 200 / 10.0 = 120 s",
            "projection: fold 1 scoring 20 / 5.0 = 4 s",
            "projection: final training 6 x 300 / 10.0 = 180 s",
            "projection: store build 50 / 25.0 = 2 s",
            "projection: total 368 s = 0.10 h",
        ]
        quick_loader = train.Rates(step=10.0, loader=4.0, cold_read=40.0, scoring=5.0, build=1.0)
        assert train.projection(None, quick_loader, epochs=6) == [
            "projection: training rate min(step 10.0, loader 4.0, cold read 40.0) "
            "= 4.0 examples/s",
            "projection: 1000000 / 4.0 = 250000 s a million training examples",
        ]


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
