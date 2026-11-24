"""The seed stages' model bindings over fake model packages (T11).

Every model package is a fake module put into `sys.modules`, with the signatures the entry-point
sheets recorded from the built images, so a binding that passes a keyword the real call lacks
fails here with `TypeError`. `huggingface_hub` is a fake too: it serves a snapshot folder for every
model of the real `families.json` at its pinned revision, under a temporary `HF_HOME`. torch,
numpy and scipy are real, and no weight is loaded. Audio is synthetic.

The tests check the call shapes, that a snapshot path (never a repository id) reaches each
library, sample-rate handling, device choice, offline mode, errors-only logging and the mapping of
library errors to `JobError`.
"""

from __future__ import annotations

import ast
import logging
import math
import os
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.voice_data import jobs
from scripts.voice_data.bindings import (
    arousal,
    asr,
    diarizers,
    embedders,
    enhancers,
    quality,
    separators,
    vad,
)
from scripts.voice_data.bindings.protocols import QualityTrack, Span, Turn, Word
from scripts.voice_data.errors import JobError
from scripts.voice_data.stages import extract as extract_stage
from scripts.voice_data.stages import fetch

FAMILIES = jobs.load_families()
LOGGERS = (
    "huggingface_hub",
    "transformers",
    "pyannote",
    "speechbrain",
    "whisperx",
    "faster_whisper",
    "crisperwhisper",
    "silero_vad",
    "lightning",
    "pytorch_lightning",
    "lightning_fabric",
    "look2hear",
    "sidon",
)
QUIET_ENV = ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_PROGRESS_BARS", "TQDM_DISABLE")


def tone(seconds: float, rate: int, hz: float = 220.0) -> np.ndarray:
    """A synthetic tone, float32 mono."""
    t = np.arange(int(round(seconds * rate)), dtype=np.float64) / rate
    return (0.3 * np.sin(2.0 * np.pi * hz * t)).astype(np.float32)


# The fake cache.


class LocalEntryNotFoundError(FileNotFoundError):
    """The hub's "not cached" error; the real one is also a `FileNotFoundError`."""


@dataclass
class Cache:
    """The fake hub and the snapshot folder of every extract and score model."""

    hf_home: Path
    calls: list[dict[str, object]] = field(default_factory=list)
    snapshots: dict[str, Path] = field(default_factory=dict)

    def folder(self, repo_id: str, revision: str, repo_type: str = "model") -> Path:
        hub = self.hf_home / "hub"
        return hub / f"{repo_type}s--{repo_id.replace('/', '--')}" / "snapshots" / revision

    def snapshot_download(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        repo_type: str | None = None,
        local_files_only: bool = False,
        **kwargs: object,
    ) -> str:
        self.calls.append({"repo_id": repo_id, "revision": revision, "local": local_files_only})
        assert local_files_only, "a binding never downloads"
        folder = self.folder(repo_id, str(revision), repo_type or "model")
        if not folder.is_dir():
            raise LocalEntryNotFoundError(repo_id)
        return str(folder)

    def path(self, model_id: str) -> str:
        return str(self.snapshots[model_id])


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Cache]:
    """A cache holding every extract and score model; env, `sys.path` and loggers restored after."""
    root = tmp_path / "cache"
    fake = Cache(root / "hf")
    for family in ("extract", "score"):
        for model_id, model in FAMILIES[family]["models"].items():
            if model["kind"] == "url":
                dest = root / model["dest"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"checkpoint")
                continue
            folder = fake.folder(model["repository"], model["revision"])
            for name in model.get("files", ["config.json", "pytorch_model.bin"]):
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes(b"weights")
            fake.snapshots[model_id] = folder
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = fake.snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("HF_HOME", str(root / "hf"))
    monkeypatch.setenv("TORCH_HOME", str(root / "torch"))
    for name in QUIET_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    levels = {name: logging.getLogger(name).level for name in LOGGERS}
    yield fake
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def install(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> types.ModuleType:
    """Put a fake module and fake parents into `sys.modules`; real parents are shadowed too."""
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    parts = name.split(".")
    parent: types.ModuleType | None = None
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        existing = sys.modules.get(parent_name)
        if existing is None or not getattr(existing, "__fake__", False):
            existing = types.ModuleType(parent_name)
            existing.__path__ = []  # type: ignore[attr-defined]
            existing.__fake__ = True  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, parent_name, existing)
            if parent is not None:
                monkeypatch.setattr(parent, parts[i - 1], existing, raising=False)
        parent = existing
    module.__fake__ = True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    if parent is not None:
        monkeypatch.setattr(parent, parts[-1], module, raising=False)
    return module


def assert_offline_and_quiet(*loggers: str) -> None:
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TQDM_DISABLE"] == "1"
    for name in ("huggingface_hub", *loggers):
        assert logging.getLogger(name).level == logging.ERROR, name


@dataclass
class Recorder:
    calls: list[tuple[str, tuple, dict]] = field(default_factory=list)

    def add(self, name: str, *args: object, **kwargs: object) -> None:
        self.calls.append((name, args, kwargs))

    def named(self, name: str) -> list[tuple[tuple, dict]]:
        return [(a, k) for n, a, k in self.calls if n == name]

    def one(self, name: str) -> tuple[tuple, dict]:
        (call,) = self.named(name)
        return call


class FakeModule:
    """A model whose `to` and `eval` return itself, recording the device."""

    def __init__(self, rec: Recorder, name: str) -> None:
        self.rec = rec
        self.name = name

    def to(self, device: object) -> FakeModule:
        self.rec.add(f"{self.name}.to", device)
        return self

    def eval(self) -> FakeModule:
        return self


# Shared helpers in stages/fetch.py.


def test_model_dir_resolves_the_pinned_snapshot(cache: Cache) -> None:
    path = fetch.model_dir("extract", "separator-dme")
    model = FAMILIES["extract"]["models"]["separator-dme"]
    assert path == cache.snapshots["separator-dme"]
    assert cache.calls == [
        {"repo_id": model["repository"], "revision": model["revision"], "local": True}
    ]


def test_missing_snapshot_is_a_job_error(cache: Cache) -> None:
    for item in cache.snapshots["wavlm-sv"].iterdir():
        item.unlink()
    cache.snapshots["wavlm-sv"].rmdir()
    with pytest.raises(JobError, match="run fetch"):
        fetch.model_dir("score", "wavlm-sv")


def test_model_file_and_cache_file_need_the_file(cache: Cache) -> None:
    with pytest.raises(JobError, match="decoder.bin"):
        fetch.model_file("extract", "enhancer-sidon", "decoder.bin")
    path = fetch.cache_file("extract", "whisperx-align-en")
    assert path.name == "wav2vec2_fairseq_base_ls960_asr_ls960.pth" and path.is_file()
    path.unlink()
    with pytest.raises(JobError, match="whisperx-align-en is not in the cache"):
        fetch.cache_file("extract", "whisperx-align-en")


def test_unknown_model_or_device_is_a_job_error(cache: Cache) -> None:
    with pytest.raises(JobError, match="no model nope"):
        fetch.model_dir("extract", "nope")
    with pytest.raises(JobError, match="device tpu"):
        fetch.prepare("tpu")


def test_library_errors_map_to_job_error_with_the_model_id_only() -> None:
    with (
        pytest.raises(JobError, match=r"^model-x: RuntimeError$") as info,
        fetch.library_errors("model-x"),
    ):
        raise RuntimeError("a library message that may hold a path")
    assert isinstance(info.value.__cause__, RuntimeError)
    with pytest.raises(JobError, match="^already$"), fetch.library_errors("model-x"):
        raise JobError("already")
    with pytest.raises(KeyError), fetch.library_errors("model-x"):
        raise KeyError("a bug surfaces as itself")


# Separators.


@dataclass
class MsstFakes:
    rec: Recorder
    target: str = "vocals"
    rate: int = 44100


@pytest.fixture
def msst(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> MsstFakes:
    fakes = MsstFakes(Recorder())
    rec = fakes.rec

    class Roformer(FakeModule):
        def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
            rec.add("load_state_dict", sorted(state_dict))

    def get_model_from_config(model_type: str, config_path: str) -> tuple:
        rec.add("get_model_from_config", model_type, config_path)
        config = SimpleNamespace(
            audio=SimpleNamespace(sample_rate=fakes.rate, num_channels=2),
            training=SimpleNamespace(target_instrument=None, instruments=[fakes.target, "other"]),
        )
        return Roformer(rec, "roformer"), config

    def demix(
        config: object,
        model: object,
        mix: torch.Tensor,
        device: torch.device,
        model_type: str,
        pbar: bool = False,
    ) -> dict:
        rec.add("demix", tuple(mix.shape), device, model_type, pbar)
        assert isinstance(mix, torch.Tensor) and mix.dtype == torch.float32
        # MSST's demix looks the function up as a module global at call time (model_utils.py:161).
        accumulate = sys.modules["utils.model_utils"]._get_accumulation_device
        rec.add("accumulation", accumulate(device, 4.0 * 1.5 * mix.numel()))
        data = mix.numpy()
        return {fakes.target: 0.5 * data, "other": 0.5 * data}

    def _get_accumulation_device(device: object, required_bytes: float) -> torch.device:
        # Upstream's choice when VRAM is plentiful: the model's own device for CUDA.
        device = torch.device(device)
        return device if device.type == "cuda" else torch.device("cpu")

    def prefer_target_instrument(config: SimpleNamespace) -> list[str]:
        return list(config.training.instruments)

    def fake_load(f: str, map_location: object = None, weights_only: bool | None = None) -> dict:
        rec.add("torch.load", f, map_location, weights_only)
        return {"state_dict": {"layer.weight": torch.zeros(1)}}

    install(monkeypatch, "utils.settings", get_model_from_config=get_model_from_config)
    install(
        monkeypatch,
        "utils.model_utils",
        demix=demix,
        prefer_target_instrument=prefer_target_instrument,
        _get_accumulation_device=_get_accumulation_device,
    )
    monkeypatch.setattr(torch, "load", fake_load)
    return fakes


# TIGER-DnR's chunking at its sample rate: 12 s windows, a 4 s hop, 8 s pads.
TIGER_RATE = 44100
WINDOW = 12 * TIGER_RATE
HOP = 4 * TIGER_RATE


class FakeDialog:
    """TIGER-DnR's `dialog` sub-model: [1, 1, T] in, [1, 3, T] out, deterministic.

    With `ramp` set, each track also depends on a sample's position in its window, so a window
    cut, or added back, at the wrong offset changes the output.
    """

    def __init__(self, rec: Recorder, ramp: float = 0.0) -> None:
        self.rec = rec
        self.ramp = ramp

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.rec.add("dialog", tuple(x.shape))
        pos = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device) / x.shape[-1]
        mono = x[:, 0, :]
        shaped = self.ramp * (pos + 3.0 * mono * pos)
        tracks = [0.2 * mono - shaped, 0.3 * mono + 2.0 * shaped, 0.5 * mono + shaped]
        return torch.stack(tracks, dim=1)


class FakeTigerModel(FakeModule):
    """TIGERDNR's shape: three sub-models, of which only `dialog` may run."""

    def __init__(self, rec: Recorder, name: str, ramp: float = 0.0) -> None:
        super().__init__(rec, name)
        self.dialog = FakeDialog(rec, ramp)

    def forward(self, mixture_tensor: torch.Tensor) -> tuple:
        raise AssertionError("TIGERDNR.forward runs the effect and music models too")

    def __call__(self, mixture_tensor: torch.Tensor) -> tuple:
        return self.forward(mixture_tensor)


@pytest.fixture
def tiger(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> Recorder:
    rec = Recorder()

    class TIGERDNR(FakeTigerModel):
        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path: str | Path,
            *,
            force_download: bool = False,
            resume_download: bool | None = None,
            proxies: dict | None = None,
            token: str | bool | None = None,
            cache_dir: str | Path | None = None,
            local_files_only: bool = False,
            revision: str | None = None,
            **model_kwargs: object,
        ) -> TIGERDNR:
            rec.add(
                "from_pretrained", pretrained_model_name_or_path, local_files_only=local_files_only
            )
            return cls(rec, "tiger")

    install(monkeypatch, "look2hear.models", TIGERDNR=TIGERDNR)
    return rec


def test_separators_load_tiger_and_melband_from_snapshots(
    cache: Cache, msst: MsstFakes, tiger: Recorder
) -> None:
    loaded = separators.load("cpu")
    assert [s.name for s in loaded] == ["separator-dme", "separator-melband-vocals"]
    (path,), kwargs = tiger.one("from_pretrained")
    assert path == cache.path("separator-dme") and kwargs["local_files_only"] is True
    assert tiger.one("tiger.to")[0] == ("cpu",)
    (model_type, cfg), _ = msst.rec.one("get_model_from_config")
    assert model_type == "mel_band_roformer"
    assert cfg == str(cache.snapshots["separator-melband-vocals"] / "config_vocals_becruily.yaml")
    (ckpt, location, weights_only), _ = msst.rec.one("torch.load")
    assert ckpt.endswith("mel_band_roformer_vocals_becruily.ckpt") and weights_only is True
    assert location == "cpu"
    assert msst.rec.one("load_state_dict")[0] == (["layer.weight"],)
    assert str(separators.TIGER_DIR) in sys.path and str(separators.MSST_DIR) in sys.path
    assert_offline_and_quiet("look2hear")


def test_tiger_runs_at_44k_and_returns_the_input_rate_and_length(
    cache: Cache, msst: MsstFakes, tiger: Recorder
) -> None:
    dme = separators.load("cpu")[0]
    audio = tone(2.0, 48000)
    out = dme.dialogue(audio, 48000)
    assert out.dtype == np.float32 and out.shape == audio.shape
    # 2 s at 44.1 kHz, padded by 8 s at each end, is three 12 s windows on a 4 s hop.
    assert [shape for (shape,), _ in tiger.named("dialog")] == [(1, 1, WINDOW)] * 3
    assert np.allclose(out[1000:-1000], 0.5 * audio[1000:-1000], atol=0.02)


def test_melband_demixes_a_stereo_copy_and_takes_the_target_stem(
    cache: Cache, msst: MsstFakes, tiger: Recorder
) -> None:
    melband = separators.load("cuda")[1]
    audio = tone(1.0, 24000)
    out = melband.dialogue(audio, 24000)
    assert out.shape == audio.shape and out.dtype == np.float32
    (shape, device, model_type, pbar), _ = msst.rec.one("demix")
    assert shape == (2, 44100) and device == torch.device("cuda")
    assert model_type == "mel_band_roformer" and pbar is False
    assert msst.rec.one("roformer.to")[0] == (torch.device("cuda"),)
    assert np.allclose(out[500:-500], 0.5 * audio[500:-500], atol=0.02)


def test_separator_library_error_is_a_job_error(
    cache: Cache, msst: MsstFakes, tiger: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("size mismatch")

    monkeypatch.setattr(sys.modules["utils.settings"], "get_model_from_config", broken)
    with pytest.raises(JobError, match="separator-melband-vocals: RuntimeError"):
        separators.load("cpu")


# TIGER-DnR's dialogue stem in bounded memory, against upstream's overlap-add.


# Transcribed from TIGER (https://github.com/JusperLee/TIGER) at
# 9f18d4a10a7137e1ce8052cfb62215179f1287b6, look2hear/models/tiger_dnr.py:803-868, licensed
# under the Apache License, Version 2.0. Modified: the method became a function, `self.sr` became
# the `sr` argument, and its comments and docstring are omitted.
def upstream_wav_chunk_inference(
    model, mixture_tensor, sr=44100, target_length=12.0, hop_length=4.0, batch_size=1, n_tracks=3
):
    batch_mixture = mixture_tensor  # [1, nch, T]
    batch_length = batch_mixture.shape[-1]
    session = int(sr * target_length)
    target = int(sr * target_length)
    ignore = (session - target) // 2
    hop = int(sr * hop_length)
    tr_ratio = target_length / hop_length
    if ignore > 0:
        zero_pad = (
            torch.zeros(batch_mixture.shape[0], batch_mixture.shape[1], ignore)
            .type(batch_mixture.dtype)
            .to(batch_mixture.device)
        )
        batch_mixture_pad = torch.cat([zero_pad, batch_mixture, zero_pad], -1)
    else:
        batch_mixture_pad = batch_mixture
    if target - hop > 0:
        hop_pad = (
            torch.zeros(batch_mixture.shape[0], batch_mixture.shape[1], target - hop)
            .type(batch_mixture.dtype)
            .to(batch_mixture.device)
        )
        batch_mixture_pad = torch.cat([hop_pad, batch_mixture_pad, hop_pad], -1)
    skip_idx = ignore + target - hop
    zero_pad = (
        torch.zeros(batch_mixture.shape[0], batch_mixture.shape[1], session)
        .type(batch_mixture.dtype)
        .to(batch_mixture.device)
    )
    num_session = (batch_mixture_pad.shape[-1] - session) // hop + 2
    all_target = torch.zeros(
        batch_mixture_pad.shape[0],
        n_tracks,
        batch_mixture_pad.shape[1],
        batch_mixture_pad.shape[2],
    ).to(batch_mixture_pad.device)
    all_input = []
    all_segment_length = []
    for i in range(num_session):
        this_input = batch_mixture_pad[:, :, i * hop : i * hop + session]
        segment_length = this_input.shape[-1]
        if segment_length < session:
            this_input = torch.cat([this_input, zero_pad[:, :, : session - segment_length]], -1)
        all_input.append(this_input)
        all_segment_length.append(segment_length)
    all_input = torch.cat(all_input, 0)
    num_batch = num_session // batch_size
    if num_session % batch_size > 0:
        num_batch += 1
    for i in range(num_batch):
        this_input = all_input[i * batch_size : (i + 1) * batch_size]
        actual_batch_size = this_input.shape[0]
        with torch.no_grad():
            est_target = model(this_input)
            est_target = est_target.unsqueeze(2)
        for j in range(actual_batch_size):
            this_est_target = est_target[j, :, :, : all_segment_length[i * batch_size + j]][
                :, :, ignore : ignore + target
            ].unsqueeze(0)
            at = ignore + (i * batch_size + j) * hop
            all_target[:, :, :, at : at + target] += this_est_target  # [batch, ntrack, nch, T]
    all_target = all_target[:, :, :, skip_idx : skip_idx + batch_length].contiguous() / tr_ratio
    return all_target.squeeze(0)


def _ramp(samples: int) -> np.ndarray:
    """A clip whose every sample differs, so an offset error shows."""
    t = np.arange(samples, dtype=np.float64) / TIGER_RATE
    return (0.4 * np.sin(2.0 * np.pi * 3.1 * t) + 0.1 * np.cos(7.0 * t * t)).astype(np.float32)


@pytest.mark.parametrize(
    ("case", "samples"),
    [("shorter than one window", TIGER_RATE), ("a hop multiple", 5 * HOP), ("odd", 600_001)],
)
def test_tiger_dialogue_equals_upstream_chunking(case: str, samples: int) -> None:
    rec = Recorder()
    model = FakeTigerModel(rec, "tiger", ramp=0.01)
    audio = _ramp(samples)
    mixture = torch.from_numpy(audio)[None, None]
    expected = upstream_wav_chunk_inference(model.dialog, mixture)[2].numpy().reshape(-1)
    upstream_calls = len(rec.named("dialog"))
    rec.calls.clear()
    out = separators.TigerDnr(model, "cpu").dialogue(audio, TIGER_RATE)
    assert out.dtype == np.float32 and out.shape == audio.shape
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-6)
    assert [shape for (shape,), _ in rec.named("dialog")] == [(1, 1, WINDOW)] * upstream_calls


def test_tiger_moves_one_window_at_a_time_to_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    moved: list[int] = []
    real = torch.Tensor.to

    def recording_to(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        moved.append(int(self.shape[-1]) if self.dim() else 1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", recording_to)
    rec = Recorder()
    audio = _ramp(5 * WINDOW + 17)
    separators.TigerDnr(FakeTigerModel(rec, "tiger"), "cpu").dialogue(audio, TIGER_RATE)
    windows = len(rec.named("dialog"))
    assert windows == (len(audio) + 2 * (WINDOW - HOP) - WINDOW) // HOP + 2
    assert len(moved) >= windows and max(moved) <= WINDOW


def test_tiger_never_runs_the_whole_model() -> None:
    model = FakeTigerModel(Recorder(), "tiger")
    with pytest.raises(AssertionError, match="forward"):
        model(torch.zeros(1, 1, 10))
    separators.TigerDnr(model, "cpu").dialogue(_ramp(TIGER_RATE), TIGER_RATE)


def test_tiger_reports_each_window_to_its_progress_callable() -> None:
    calls: list[tuple[int, int]] = []
    tiger_dnr = separators.TigerDnr(
        FakeTigerModel(Recorder(), "tiger"), "cpu", progress=lambda d, t: calls.append((d, t))
    )
    tiger_dnr.dialogue(_ramp(2 * WINDOW), TIGER_RATE)
    total = (2 * WINDOW + 2 * (WINDOW - HOP) - WINDOW) // HOP + 2
    assert calls == [(n, total) for n in range(1, total + 1)]


def test_progress_passes_through_every_loader(
    cache: Cache, msst: MsstFakes, tiger: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, int]] = []

    def progress(done: int, total: int) -> None:
        calls.append((done, total))

    separators.load_tiger("cpu", progress).dialogue(tone(1.0, 44100), 44100)
    assert calls[-1] == (3, 3)
    calls.clear()
    separators.load("cpu", progress=progress)[0].dialogue(tone(1.0, 44100), 44100)
    assert calls[-1] == (3, 3)
    seen: list[tuple[str, object]] = []

    def separators_load(device: str, progress: object = None) -> list:
        seen.append((device, progress))
        return []

    monkeypatch.setattr(separators, "load", separators_load)
    for module in (arousal, diarizers, embedders, quality, vad):
        monkeypatch.setattr(module, "load", lambda device: [])
    extract_stage.load_models("cuda", progress=progress)
    assert seen == [("cuda", progress)]
    extract_stage.load_models("cpu")
    assert seen[-1] == ("cpu", None)


# Mel-Band RoFormer's demix accumulates on the host.


def test_melband_keeps_the_demix_accumulator_on_the_host(
    cache: Cache, msst: MsstFakes, tiger: Recorder
) -> None:
    melband = separators.load("cuda")[1]
    hook = sys.modules["utils.model_utils"]._get_accumulation_device
    for device in ("cuda", "cpu", torch.device("cuda"), torch.device("cpu")):
        for size in (1.0, 1e15):
            assert hook(device, size) == torch.device("cpu")
    melband.dialogue(tone(1.0, 44100), 44100)
    assert msst.rec.one("accumulation")[0] == (torch.device("cpu"),)


def test_dereverb_keeps_the_demix_accumulator_on_the_host_too(
    cache: Cache, msst: MsstFakes, sidon: SidonFakes
) -> None:
    msst.target = "noreverb"
    dereverb = enhancers.load("cuda")[1]
    dereverb.apply(tone(1.0, 44100), 44100)
    assert msst.rec.one("accumulation")[0] == (torch.device("cpu"),)


def test_melband_without_the_accumulation_hook_is_a_job_error(
    cache: Cache, msst: MsstFakes, tiger: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(sys.modules["utils.model_utils"], "_get_accumulation_device")
    with pytest.raises(JobError, match="separator-melband-vocals"):
        separators.load("cpu")


# Enhancers.


@dataclass
class SidonFakes:
    rec: Recorder
    chunks: list[int] = field(default_factory=list)


@pytest.fixture
def sidon(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> SidonFakes:
    fakes = SidonFakes(Recorder())
    rec = fakes.rec

    def extract_seamless_m4t_features(
        raw_speech: torch.Tensor,
        sampling_rate: int = 16000,
        num_mel_bins: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        preemphasis_coefficient: float = 0.97,
        dither: float = 0.0,
        window_type: str = "povey",
        do_normalize_per_mel_bins: bool = True,
        stride: int = 2,
        padding: str = "longest",
        max_length: int | None = None,
        pad_to_multiple_of: int | None = 2,
        return_tensors: str | None = "pt",
        return_attention_mask: bool = True,
        padding_value: float = 0.0,
        device: str = "cuda",
    ) -> dict:
        fakes.chunks.append(int(raw_speech.shape[-1]))
        rec.add("features", sampling_rate=sampling_rate, device=device)
        frames = int(raw_speech.shape[-1]) // 320
        return {"input_features": torch.ones(1, frames, 160)}

    class Script(FakeModule):
        def __call__(self, x: torch.Tensor) -> object:
            if self.name == "fe":
                return {"last_hidden_state": torch.ones(1, x.shape[1], 8)}
            assert x.shape[1] == 8
            return torch.full((1, 1, x.shape[2] * 960), 0.25)

    def jit_load(f: str, map_location: object = None) -> Script:
        rec.add("jit.load", f, map_location)
        return Script(rec, "fe" if "feature_extractor" in f else "decoder")

    install(
        monkeypatch,
        "sidon.cleansing.audio",
        extract_seamless_m4t_features=extract_seamless_m4t_features,
    )
    monkeypatch.setattr(torch.jit, "load", jit_load)
    return fakes


def test_enhancers_are_sidon_then_dereverb(
    cache: Cache, msst: MsstFakes, sidon: SidonFakes
) -> None:
    msst.target = "noreverb"
    loaded = enhancers.load("cuda")
    assert [e.name for e in loaded] == ["enhancer-sidon", "dereverb-melband"]
    loads = sidon.rec.named("jit.load")
    snapshot = cache.snapshots["enhancer-sidon"]
    assert [a for a, _ in loads] == [
        (str(snapshot / "feature_extractor_cuda.pt"), "cuda"),
        (str(snapshot / "decoder_cuda.pt"), "cuda"),
    ]
    (_, cfg), _ = msst.rec.one("get_model_from_config")
    assert cfg.endswith("dereverb_mel_band_roformer_anvuew.yaml")
    assert str(enhancers.SIDON_DIR) in sys.path
    audio = tone(1.0, 24000)
    out = loaded[1].apply(audio, 24000)
    assert out.shape == audio.shape
    assert np.allclose(out[500:-500], 0.5 * audio[500:-500], atol=0.02)


def test_sidon_refuses_cpu(cache: Cache, msst: MsstFakes, sidon: SidonFakes) -> None:
    with pytest.raises(JobError, match="enhancer-sidon runs on cuda only"):
        enhancers.load("cpu")
    assert sidon.rec.named("jit.load") == []


def test_sidon_recipe_chunks_pads_and_returns_the_input_rate(sidon: SidonFakes) -> None:
    fe = sidon_script(sidon, "fe")
    decoder = sidon_script(sidon, "decoder")
    extract = sys.modules["sidon.cleansing.audio"].extract_seamless_m4t_features
    step = enhancers.Sidon(fe, decoder, extract, "cpu")
    audio = tone(100.0, 24000)
    out = step.apply(audio, 24000)
    assert out.shape == audio.shape and out.dtype == np.float32
    samples_16k = 100 * 16000 + enhancers.SIDON_PAD
    assert sidon.chunks == [enhancers.SIDON_CHUNK + 320, samples_16k - enhancers.SIDON_CHUNK + 320]
    assert {k["sampling_rate"] for _, k in sidon.rec.named("features")} == {16000}
    assert {k["device"] for _, k in sidon.rec.named("features")} == {"cpu"}
    assert np.allclose(out[100:-100], 0.25, atol=1e-3)


def sidon_script(fakes: SidonFakes, name: str) -> object:
    return torch.jit.load(f"/x/{'feature_extractor' if name == 'fe' else 'decoder'}_cuda.pt")


# Diarizer.


@dataclass
class FakeSegment:
    start: float
    end: float


class FakeAnnotation:
    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        self.tracks = tracks

    def itertracks(self, yield_label: bool = False) -> Iterator[tuple]:
        for n, (start, end, label) in enumerate(self.tracks):
            yield (
                (FakeSegment(start, end), n, label)
                if yield_label
                else (
                    FakeSegment(start, end),
                    n,
                )
            )


@pytest.fixture
def pyannote(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> Recorder:
    rec = Recorder()

    class Pipeline:
        @classmethod
        def from_pretrained(
            cls,
            checkpoint: str | Path | dict,
            revision: str | None = None,
            hparams_file: str | Path | None = None,
            subfolder: str | None = None,
            token: str | bool | None = None,
            cache_dir: Path | str | None = None,
        ) -> Pipeline:
            rec.add("Pipeline.from_pretrained", checkpoint, revision=revision, token=token)
            return cls()

        def to(self, device: torch.device) -> Pipeline:
            rec.add("Pipeline.to", device)
            return self

        def __call__(self, file: dict, preload: bool = False, **kwargs: object) -> object:
            rec.add("Pipeline.call", tuple(file["waveform"].shape), file["sample_rate"])
            return SimpleNamespace(
                speaker_diarization=FakeAnnotation(
                    [(0.0, 2.0, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01"), (2.8, 4.0, "SPEAKER_00")]
                ),
                exclusive_speaker_diarization=FakeAnnotation(
                    [(2.0, 2.8, "SPEAKER_01"), (0.0, 1.5, "SPEAKER_00"), (3.0, 4.0, "SPEAKER_00")]
                ),
                speaker_embeddings=None,
            )

    class Model(FakeModule):
        @classmethod
        def from_pretrained(
            cls,
            checkpoint: Path | str,
            map_location: object = None,
            strict: bool = True,
            subfolder: str | None = None,
            revision: str | None = None,
            token: str | bool | None = None,
            cache_dir: Path | str | None = None,
            **kwargs: object,
        ) -> Model:
            rec.add("Model.from_pretrained", checkpoint, strict=strict)
            return cls(rec, "brouhaha")

    class Inference:
        def __init__(
            self,
            model: object,
            window: str = "sliding",
            duration: float | None = None,
            step: float | None = None,
            pre_aggregation_hook: object = None,
            skip_aggregation: bool = False,
            skip_conversion: bool = False,
            device: torch.device | None = None,
            batch_size: int = 32,
        ) -> None:
            rec.add("Inference", model, device=device)

        def __call__(self, file: dict, hook: object = None) -> object:
            rec.add("Inference.call", tuple(file["waveform"].shape), file["sample_rate"])
            frames = [
                (FakeSegment(i * 0.0169, i * 0.0169 + 0.0169), np.array([0.9, 20.0 + i, 40.0 - i]))
                for i in range(5)
            ]
            return SimpleNamespace(
                __iter__=None, frames=frames, sliding_window=SimpleNamespace(step=0.0169)
            )

    install(monkeypatch, "pyannote.audio", Pipeline=Pipeline, Model=Model, Inference=Inference)
    return rec


def test_diarizer_loads_community_1_from_its_snapshot_on_the_device(
    cache: Cache, pyannote: Recorder
) -> None:
    diarizer = diarizers.load("cuda")
    (checkpoint,), kwargs = pyannote.one("Pipeline.from_pretrained")
    assert checkpoint == cache.path("diarizer-community-1")
    assert kwargs == {"revision": None, "token": None}
    assert pyannote.one("Pipeline.to")[0] == (torch.device("cuda"),)
    assert_offline_and_quiet("pyannote")
    audio = tone(4.0, 48000)
    assert diarizer.turns(audio, 48000) == [
        Turn(0.0, 1.5, "SPEAKER_00"),
        Turn(2.0, 2.8, "SPEAKER_01"),
        Turn(3.0, 4.0, "SPEAKER_00"),
    ]
    assert diarizer.overlaps(audio, 48000) == [Span(1.5, 2.0), Span(2.8, 3.0)]
    (shape, rate), _ = pyannote.one("Pipeline.call")
    assert shape == (1, 64000) and rate == 16000


def test_diarizer_runs_the_pipeline_again_for_new_audio(cache: Cache, pyannote: Recorder) -> None:
    diarizer = diarizers.load("cpu")
    diarizer.turns(tone(4.0, 16000), 16000)
    diarizer.overlaps(tone(4.0, 16000), 16000)
    diarizer.turns(tone(4.0, 16000, hz=330.0), 16000)
    assert len(pyannote.named("Pipeline.call")) == 2


def test_diarizer_pipeline_that_does_not_load_is_a_job_error(
    cache: Cache, pyannote: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = sys.modules["pyannote.audio"].Pipeline
    monkeypatch.setattr(pipeline, "from_pretrained", classmethod(lambda cls, checkpoint: None))
    with pytest.raises(JobError, match="diarizer-community-1"):
        diarizers.load("cpu")


# Quality.


def test_quality_reads_snr_and_c50_frames_with_the_window_step(
    cache: Cache, pyannote: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    inference = sys.modules["pyannote.audio"].Inference
    real_call = inference.__call__

    class Output:
        def __init__(self, wrapped: SimpleNamespace) -> None:
            self.sliding_window = wrapped.sliding_window
            self._frames = wrapped.frames

        def __iter__(self) -> Iterator[tuple]:
            return iter(self._frames)

    monkeypatch.setattr(
        inference, "__call__", lambda self, file, hook=None: Output(real_call(self, file))
    )
    estimator = quality.load("cuda")
    (checkpoint,), kwargs = pyannote.one("Model.from_pretrained")
    assert checkpoint == str(cache.snapshots["quality-brouhaha"] / "pytorch_model.bin")
    assert kwargs == {"strict": False}
    (_model,), kwargs = pyannote.one("Inference")
    assert kwargs["device"] == torch.device("cuda")
    assert str(quality.BROUHAHA_DIR) in sys.path
    track = estimator.frames(tone(3.0, 44100), 44100)
    assert isinstance(track, QualityTrack)
    assert math.isclose(track.hop_s, 0.0169)
    assert track.snr_db.tolist() == [20.0, 21.0, 22.0, 23.0, 24.0]
    assert track.c50_db.tolist() == [40.0, 39.0, 38.0, 37.0, 36.0]
    (shape, rate), _ = pyannote.one("Inference.call")
    assert shape == (1, 48000) and rate == 16000
    assert_offline_and_quiet("pyannote")


# VAD.


@pytest.fixture
def silero(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> Recorder:
    rec = Recorder()

    def load_silero_vad(
        onnx: bool = False,
        opset_version: int = 16,
        sequence: bool = False,
        sampling_rate: int = 16000,
        max_frames: int = 512,
    ) -> object:
        rec.add("load", onnx=onnx)
        return "silero-model"

    def get_speech_timestamps(
        audio: torch.Tensor,
        model: object,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = math.inf,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        return_seconds: bool = False,
        time_resolution: int = 1,
        visualize_probs: bool = False,
        progress_tracking_callback: object = None,
        neg_threshold: float | None = None,
        window_size_samples: int = 512,
        min_silence_at_max_speech: int = 98,
        use_max_poss_sil_at_max_speech: bool = True,
    ) -> list[dict]:
        rec.add("timestamps", tuple(audio.shape), model, sampling_rate, return_seconds)
        return [{"start": 8000, "end": 24000}, {"start": 40000, "end": 99999}]

    install(
        monkeypatch,
        "silero_vad",
        load_silero_vad=load_silero_vad,
        get_speech_timestamps=get_speech_timestamps,
    )
    return rec


def test_vad_resamples_to_16k_and_returns_seconds(cache: Cache, silero: Recorder) -> None:
    detector = vad.load("cuda")
    spans = detector.speech(tone(3.0, 48000), 48000)
    assert spans == [Span(0.5, 1.5), Span(2.5, 3.0)]
    (shape, model, rate, seconds), _ = silero.one("timestamps")
    assert shape == (48000,) and model == "silero-model" and rate == 16000 and seconds is False
    assert_offline_and_quiet("silero_vad")


# Embedders.


@dataclass
class Transformers:
    rec: Recorder
    module: types.ModuleType


@pytest.fixture
def transformers_fake(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> Transformers:
    rec = Recorder()

    class Wav2Vec2FeatureExtractor:
        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            cache_dir: str | None = None,
            force_download: bool = False,
            local_files_only: bool = False,
            token: str | bool | None = None,
            revision: str = "main",
            **kwargs: object,
        ) -> Wav2Vec2FeatureExtractor:
            rec.add("extractor", pretrained_model_name_or_path, local_files_only=local_files_only)
            return cls()

        def __call__(
            self,
            raw_speech: np.ndarray,
            padding: bool | str = False,
            max_length: int | None = None,
            truncation: bool = False,
            pad_to_multiple_of: int | None = None,
            return_attention_mask: bool | None = None,
            return_tensors: str | None = None,
            sampling_rate: int | None = None,
            **kwargs: object,
        ) -> dict:
            rec.add("extract", len(raw_speech), sampling_rate=sampling_rate, tensors=return_tensors)
            values = np.asarray(raw_speech, dtype=np.float32)[None]
            return {"input_values": torch.from_numpy(values) if return_tensors == "pt" else values}

    class WavLMForXVector(FakeModule):
        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            *model_args: object,
            config: object = None,
            cache_dir: str | None = None,
            ignore_mismatched_sizes: bool = False,
            force_download: bool = False,
            local_files_only: bool = False,
            token: str | bool | None = None,
            revision: str = "main",
            use_safetensors: bool | None = None,
            weights_only: bool = True,
            **kwargs: object,
        ) -> WavLMForXVector:
            rec.add("wavlm", pretrained_model_name_or_path, local_files_only=local_files_only)
            return cls(rec, "wavlm")

        def __call__(
            self,
            input_values: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            labels: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            rec.add("wavlm.call", tuple(input_values.shape))
            return SimpleNamespace(embeddings=torch.tensor([[3.0, 4.0, 0.0]]))

    def pipeline(
        task: str | None = None,
        model: object = None,
        config: object = None,
        tokenizer: object = None,
        feature_extractor: object = None,
        image_processor: object = None,
        processor: object = None,
        framework: str | None = None,
        revision: str | None = None,
        use_fast: bool = True,
        token: str | bool | None = None,
        device: object = None,
        device_map: object = None,
        dtype: object = "auto",
        trust_remote_code: bool | None = None,
        model_kwargs: dict | None = None,
        pipeline_class: object = None,
        **kwargs: object,
    ) -> object:
        rec.add("pipeline", task, model, device=device, dtype=dtype)

        def call(inputs: dict, **kwargs: object) -> dict:
            rec.add("pipe.call", len(inputs["raw"]), inputs["sampling_rate"], **kwargs)
            return {"text": " Hello there. "}

        return call

    class Wav2Vec2PreTrainedModel(torch.nn.Module):
        missing: list[str] = []

        def __init__(self, config: object, *inputs: object, **kwargs: object) -> None:
            super().__init__()
            self.config = config

        def post_init(self) -> None:
            rec.add("post_init")

        @classmethod
        def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            *model_args: object,
            config: object = None,
            cache_dir: str | None = None,
            ignore_mismatched_sizes: bool = False,
            force_download: bool = False,
            local_files_only: bool = False,
            token: str | bool | None = None,
            revision: str = "main",
            use_safetensors: bool | None = None,
            weights_only: bool = True,
            **kwargs: object,
        ) -> object:
            rec.add(
                "emotion",
                pretrained_model_name_or_path,
                local_files_only=local_files_only,
                **kwargs,
            )
            model = cls(SimpleNamespace(hidden_size=4, final_dropout=0.0, num_labels=3))
            info = {"missing_keys": list(cls.missing), "unexpected_keys": []}
            return (model, info) if kwargs.get("output_loading_info") else model

    class Wav2Vec2Model(torch.nn.Module):
        def __init__(self, config: SimpleNamespace) -> None:
            super().__init__()
            self.size = config.hidden_size

        def forward(self, input_values: torch.Tensor) -> tuple:
            rec.add("wav2vec2", tuple(input_values.shape))
            return (torch.ones(1, 5, self.size),)

    module = install(
        monkeypatch,
        "transformers",
        Wav2Vec2FeatureExtractor=Wav2Vec2FeatureExtractor,
        WavLMForXVector=WavLMForXVector,
        pipeline=pipeline,
    )
    install(
        monkeypatch,
        "transformers.models.wav2vec2.modeling_wav2vec2",
        Wav2Vec2PreTrainedModel=Wav2Vec2PreTrainedModel,
        Wav2Vec2Model=Wav2Vec2Model,
    )
    return Transformers(rec, module)


@pytest.fixture
def speechbrain(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> Recorder:
    rec = Recorder()

    class EncoderClassifier:
        @classmethod
        def from_hparams(
            cls, source: str, hparams_file: str = "hyperparams.yaml", **kwargs: object
        ) -> EncoderClassifier:
            rec.add("from_hparams", source, hparams_file=hparams_file, **kwargs)
            return cls()

        def encode_batch(
            self, wavs: torch.Tensor, wav_lens: torch.Tensor | None = None, normalize: bool = False
        ) -> torch.Tensor:
            rec.add("encode_batch", tuple(wavs.shape), wavs.device)
            return torch.tensor([[[0.0, 6.0, 8.0]]])

    install(monkeypatch, "speechbrain.inference.speaker", EncoderClassifier=EncoderClassifier)
    return rec


def test_embedders_are_wavlm_then_ecapa_from_snapshots(
    cache: Cache, transformers_fake: Transformers, speechbrain: Recorder
) -> None:
    loaded = embedders.load("cuda")
    assert [e.name for e in loaded] == ["wavlm-sv", "ecapa"]
    rec = transformers_fake.rec
    (path,), kwargs = rec.one("wavlm")
    assert path == cache.path("wavlm-sv") and kwargs["local_files_only"] is True
    assert rec.one("extractor")[0] == (cache.path("wavlm-sv"),)
    assert rec.one("wavlm.to")[0] == ("cuda",)
    (source,), kwargs = speechbrain.one("from_hparams")
    assert source == cache.path("ecapa")
    assert kwargs["overrides"] == {"pretrained_path": cache.path("ecapa")}
    assert kwargs["run_opts"] == {"device": "cuda"}
    assert Path(kwargs["savedir"]).is_dir()
    assert not Path(kwargs["savedir"]).is_relative_to(cache.hf_home.parent)
    assert_offline_and_quiet("transformers", "speechbrain")


def test_embeddings_are_unit_vectors_from_16k_audio(
    cache: Cache, transformers_fake: Transformers, speechbrain: Recorder
) -> None:
    wavlm, ecapa = embedders.load("cpu")
    audio = tone(2.0, 24000)
    assert np.allclose(wavlm.embed(audio, 24000), [0.6, 0.8, 0.0])
    assert np.allclose(ecapa.embed(audio, 24000), [0.0, 0.6, 0.8])
    (length,), kwargs = transformers_fake.rec.one("extract")
    assert length == 32000 and kwargs["sampling_rate"] == 16000
    assert transformers_fake.rec.one("wavlm.call")[0] == ((1, 32000),)
    (shape, device), _ = speechbrain.one("encode_batch")
    assert shape == (1, 32000) and device == torch.device("cpu")


def test_missing_speechbrain_is_a_job_error(
    cache: Cache, transformers_fake: Transformers, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "speechbrain", None)
    with pytest.raises(JobError, match="ecapa: ModuleNotFoundError"):
        embedders.load("cpu")


def test_zero_embedding_is_a_job_error(
    cache: Cache, transformers_fake: Transformers, speechbrain: Recorder
) -> None:
    wavlm, _ = embedders.load("cpu")
    model = wavlm._model
    model.__class__.__call__ = lambda self, input_values: SimpleNamespace(
        embeddings=torch.zeros(1, 3)
    )
    with pytest.raises(JobError, match="wavlm-sv"):
        wavlm.embed(tone(1.0, 16000), 16000)


# ASR.


@pytest.fixture
def asr_fakes(monkeypatch: pytest.MonkeyPatch, cache: Cache) -> Recorder:
    rec = Recorder()

    class WhisperModel:
        def __init__(
            self,
            model_size_or_path: str,
            device: str = "auto",
            device_index: int | list[int] = 0,
            compute_type: str = "default",
            cpu_threads: int = 0,
            num_workers: int = 1,
            download_root: str | None = None,
            local_files_only: bool = False,
            files: dict | None = None,
            revision: str | None = None,
            use_auth_token: str | bool | None = None,
            **model_kwargs: object,
        ) -> None:
            rec.add(
                "WhisperModel",
                model_size_or_path,
                device=device,
                compute_type=compute_type,
                local_files_only=local_files_only,
            )

        def transcribe(
            self,
            audio: np.ndarray,
            language: str | None = None,
            task: str = "transcribe",
            log_progress: bool = False,
            beam_size: int = 5,
            best_of: int = 5,
            patience: float = 1,
            length_penalty: float = 1,
            repetition_penalty: float = 1,
            no_repeat_ngram_size: int = 0,
            temperature: object = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold: float | None = 2.4,
            log_prob_threshold: float | None = -1.0,
            no_speech_threshold: float | None = 0.6,
            condition_on_previous_text: bool = True,
            prompt_reset_on_temperature: float = 0.5,
            initial_prompt: object = None,
            prefix: str | None = None,
            suppress_blank: bool = True,
            suppress_tokens: list[int] | None = None,
            without_timestamps: bool = False,
            max_initial_timestamp: float = 1.0,
            word_timestamps: bool = False,
            prepend_punctuations: str = "",
            append_punctuations: str = "",
            multilingual: bool = False,
            vad_filter: bool = False,
            vad_parameters: object = None,
            max_new_tokens: int | None = None,
            chunk_length: int | None = None,
            clip_timestamps: object = "0",
            hallucination_silence_threshold: float | None = None,
            hotwords: str | None = None,
            language_detection_threshold: float | None = 0.5,
            language_detection_segments: int = 1,
        ) -> tuple:
            rec.add("transcribe", len(audio), audio.dtype, language=language)
            segments = [
                SimpleNamespace(text=" Hello there.", start=0.0, end=1.0),
                SimpleNamespace(text=" Again 2024.", start=1.2, end=2.0),
            ]
            return iter(segments), SimpleNamespace(language="en")

    def load_align_model(
        language_code: str,
        device: str,
        model_name: str | None = None,
        model_dir: object = None,
        model_cache_only: bool = False,
    ) -> tuple:
        rec.add("load_align_model", language_code, device)
        return "align-model", {"language": "en"}

    def align(
        transcript: list,
        model: object,
        align_model_metadata: dict,
        audio: np.ndarray,
        device: str,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        print_progress: bool = False,
        combined_progress: bool = False,
        progress_callback: object = None,
    ) -> dict:
        rec.add("align", transcript, model, len(audio), device, print_progress=print_progress)
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello there.",
                    "words": [
                        {"word": "Hello", "start": 0.1, "end": 0.4, "score": 0.9},
                        {"word": "there.", "start": 0.5, "end": 0.9, "score": 0.8},
                    ],
                },
                {
                    "start": 1.2,
                    "end": 2.0,
                    "text": "Again 2024.",
                    "words": [
                        {"word": "Again", "start": 1.3, "end": 1.6, "score": 0.9},
                        {"word": "2024."},
                    ],
                },
            ],
            "word_segments": [],
        }

    def setup_logging(level: str = "info", log_file: str | None = None) -> None:
        rec.add("setup_logging", level=level, log_file=log_file)

    class CrisperWhisperModel:
        def __init__(
            self,
            model_name_or_path: str = "nyralabs/CrisperWhisper2.0_large",
            *,
            backend: str = "auto",
            compute_type: str = "float16",
            device: str = "auto",
            device_index: int = 0,
            draft_model: str | None = None,
            speculative_k: int | str = "auto",
            num_speculative_tokens: int | None = None,
            min_speculative_tokens: int = 0,
            max_speculative_tokens: int = 0,
            cache_dir: str | Path | None = None,
            forced_aligner: object = None,
        ) -> None:
            rec.add("Crisper", model_name_or_path, device=device, compute_type=compute_type)

        def transcribe(
            self,
            audio: np.ndarray,
            *,
            language: str = "en",
            mode: str = "verbatim",
            hotwords: list[str] | None = None,
            sr: int | None = None,
            longform_strategy: str = "continuation",
            chunk_duration: float = 30.0,
            stride: float = 26.0,
            context_words: int = 12,
            drop_words: int = 2,
            timestamp_aware_drop: bool = True,
            temperature_fallback: bool = True,
            max_new_tokens: int = 256,
            speculative_decoding: bool = False,
            speculative_mode: str = "strict",
            hallucination_mitigation: bool = True,
            early_eot_recovery: bool = True,
            word_timestamps: bool = False,
            alignment_heads: object = None,
            suppress_tokens: list[int] | None = None,
            word_confidence: bool = False,
            timing_method: str = "attention",
        ) -> SimpleNamespace:
            rec.add("Crisper.transcribe", len(audio), language=language, mode=mode, sr=sr)
            return SimpleNamespace(text=" Um, hello there. ", words=None)

    install(monkeypatch, "faster_whisper", WhisperModel=WhisperModel)
    install(monkeypatch, "whisperx.alignment", load_align_model=load_align_model, align=align)
    install(monkeypatch, "whisperx.log_utils", setup_logging=setup_logging)
    install(monkeypatch, "crisperwhisper", CrisperWhisperModel=CrisperWhisperModel)
    return rec


def test_asr_load_returns_timing_then_verbatim(cache: Cache, asr_fakes: Recorder) -> None:
    timing, verbatim = asr.load("cuda")
    assert (timing.name, verbatim.name) == ("whisperx", "crisperwhisper")
    (path,), kwargs = asr_fakes.one("WhisperModel")
    assert path == cache.path("whisperx-asr-large-v3")
    assert kwargs == {"device": "cuda", "compute_type": "default", "local_files_only": True}
    assert asr_fakes.one("load_align_model")[0] == ("en", "cuda")
    assert asr_fakes.one("setup_logging")[1]["level"] == "error"
    (path,), kwargs = asr_fakes.one("Crisper")
    assert path == cache.path("crisperwhisper")
    assert kwargs == {"device": "cuda", "compute_type": "float16"}
    assert_offline_and_quiet("whisperx", "faster_whisper", "crisperwhisper")


def test_whisperx_aligns_words_from_16k_audio(cache: Cache, asr_fakes: Recorder) -> None:
    timing, _ = asr.load("cpu")
    result = timing.transcribe(tone(2.0, 24000), 24000)
    assert result.text == "Hello there. Again 2024."
    assert result.words == (
        Word("Hello", 0.1, 0.4),
        Word("there.", 0.5, 0.9),
        Word("Again", 1.3, 1.6),
        Word("2024.", 1.6, 1.6),
    )
    (length, dtype), kwargs = asr_fakes.one("transcribe")
    assert length == 32000 and dtype == np.float32 and kwargs["language"] == "en"
    (pieces, model, length, device), kwargs = asr_fakes.one("align")
    assert pieces == [
        {"text": " Hello there.", "start": 0.0, "end": 1.0},
        {"text": " Again 2024.", "start": 1.2, "end": 2.0},
    ]
    assert model == "align-model" and length == 32000 and device == "cpu"
    assert kwargs["print_progress"] is False
    assert asr_fakes.one("Crisper")[1]["compute_type"] == "float32"


def test_crisperwhisper_transcribes_verbatim_english(cache: Cache, asr_fakes: Recorder) -> None:
    _, verbatim = asr.load("cpu")
    result = verbatim.transcribe(tone(1.0, 48000), 48000)
    assert result.text == "Um, hello there." and result.words == ()
    (length,), kwargs = asr_fakes.one("Crisper.transcribe")
    assert length == 16000 and kwargs == {"language": "en", "mode": "verbatim", "sr": 16000}


def test_whisperx_needs_the_alignment_checkpoint(cache: Cache, asr_fakes: Recorder) -> None:
    fetch.cache_file("extract", "whisperx-align-en").unlink()
    with pytest.raises(JobError, match="whisperx-align-en is not in the cache"):
        asr.load("cpu")


def test_asr_library_error_is_a_job_error(
    cache: Cache, asr_fakes: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("no such file")

    monkeypatch.setattr(sys.modules["crisperwhisper"], "CrisperWhisperModel", broken)
    with pytest.raises(JobError, match="crisperwhisper: OSError"):
        asr.load("cpu")


def test_whisper_turbo_is_a_separate_loader(cache: Cache, transformers_fake: Transformers) -> None:
    turbo = asr.load_turbo("cuda")
    assert turbo.name == "whisper-large-v3-turbo"
    (task, model), kwargs = transformers_fake.rec.one("pipeline")
    assert task == "automatic-speech-recognition" and model == cache.path("whisper-large-v3-turbo")
    assert kwargs == {"device": "cuda", "dtype": torch.float16}
    result = turbo.transcribe(tone(1.0, 24000), 24000)
    assert result.text == "Hello there." and result.words == ()
    (length, rate), kwargs = transformers_fake.rec.one("pipe.call")
    assert length == 16000 and rate == 16000
    assert kwargs == {"generate_kwargs": {"language": "en", "task": "transcribe"}}
    assert_offline_and_quiet("transformers")


# Arousal.


def test_arousal_is_logit_zero_of_the_card_model(
    cache: Cache, transformers_fake: Transformers
) -> None:
    monitor = arousal.load("cpu")
    (path,), kwargs = transformers_fake.rec.one("emotion")
    assert path == cache.path("arousal")
    assert kwargs == {"local_files_only": True, "output_loading_info": True}
    assert transformers_fake.rec.one("extractor")[0] == (cache.path("arousal"),)
    model = monitor._model
    assert [n for n, _ in model.named_children()] == ["wav2vec2", "classifier"]
    head = model.classifier
    assert [n for n, _ in head.named_children()] == ["dense", "dropout", "out_proj"]
    with torch.no_grad():
        head.out_proj.weight.zero_()
        head.out_proj.bias.copy_(torch.tensor([0.7, 0.2, 0.1]))
    assert math.isclose(monitor.arousal(tone(1.0, 24000), 24000), 0.7, rel_tol=1e-6)
    (length,), kwargs = transformers_fake.rec.one("extract")
    assert length == 16000 and kwargs["sampling_rate"] == 16000
    assert transformers_fake.rec.one("wav2vec2")[0] == ((1, 16000),)
    with torch.no_grad():
        head.out_proj.bias.copy_(torch.tensor([1.4, 0.2, 0.1]))
    assert monitor.arousal(tone(1.0, 16000), 16000) == 1.0
    assert_offline_and_quiet("transformers")


def test_arousal_missing_keys_are_a_job_error(
    cache: Cache, transformers_fake: Transformers, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = sys.modules["transformers.models.wav2vec2.modeling_wav2vec2"].Wav2Vec2PreTrainedModel
    monkeypatch.setattr(base, "missing", ["classifier.dense.weight"])
    with pytest.raises(JobError, match="arousal: 1 weights are missing"):
        arousal.load("cpu")


# Every binding.


@pytest.mark.parametrize(
    "module", [arousal, asr, diarizers, embedders, enhancers, quality, separators, vad]
)
def test_importing_a_binding_imports_no_model_package(module: types.ModuleType) -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    top: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top.append(node.module)
    assert {name.split(".")[0] for name in top} <= {
        "__future__",
        "collections",
        "hashlib",
        "numpy",
        "tempfile",
    }
    assert callable(module.load)
