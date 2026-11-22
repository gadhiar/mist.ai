"""Train the wake verifier on the dataset index, one cross-validation fold or the final model.

    python -m scripts.wake.verifier.train --dataset <dir> --manifest <train-only manifest> \
        --root-map <from>=<to> --weights <wavlm_base_plus.pth> --out <dir> \
        (--fold <k> | --final) [--batch 32] [--accumulate 1] [--seed 17] [--cache <dir>] \
        [--store-workers 8] [--loader-workers 4] [--progress-seconds 60]

The model is `model.build_verifier` over the staged `wavlm_base_plus.pth`, loaded strictly; the
run refuses to start if the installed torchaudio's `WAVLM_BASE_PLUS._normalize_waveform` reads
True, since the exported graph applies no normalisation.

Data. Only what `build.py` indexed is read, and only after every file a window was read or
mixed from is checked against the dataset's `provenance.json`: a row outside it is refused. A
recording is read through the train-only manifest (`--root-map` translating its root), which is
refused unless its sha256 is the index's `training` and it lists train entries alone, so no tune
or test file is opened. Real positives are repeated `REAL_REPEAT` times an epoch.

The store. `WindowReader` opens, parses and decodes a window's source file on every read, and an
epoch reads every window, so the run reads them from `store.open_store` instead: each window
decoded once into one flat file under `--cache` (default `<out>/window-cache`), kept under the
sha256 of `index.jsonl` and reused by every later run of the same index with the same cache, the
other folds and the final run included. A window that reads or mixes a recording is never
written there; each run reads those through the manifest into memory. The store is opened before
the model is built: its decoding processes, `--store-workers` of them, are started by spawn, and
CUDA must not have started before them.

Folds. Five folds grouped by each window's `group` as `mine.py` set it (recording session, AMI
speaker group, LibriSpeech speaker, which a Chatterbox clone shares, piper speaker, MUSAN file),
balanced within each kind of group, so no group is on
both sides of a fold. `--fold k` trains on the other four and scores fold k's windows into
`oof.jsonl`; `--final` trains on all of them. Every run trains the same fixed schedule and keeps
its last epoch: nothing is chosen on out-of-fold data, which would bias the pre-flight's `r(c)`.

Augmentation, for both classes alike from a per-item seed: the truncation of Design 3 (either
side cut to a random length down to `MIN_WINDOW_FRAMES`, the hit kept), then gain, a room
response from the index's own `rirs` files, a low-pass band limit, white noise and an AGC-like
compressor (`augment.augment_batch`). No step reads the label, so a window's length or level
carries none. The steps run on the device, a batch at a time, outside autocast: done per item in
numpy, the room response's convolution alone held the loader workers long enough that the GPU
waited. A loader worker, `--loader-workers` of them on CUDA, only reads a batch's kept frames
from the store and draws each item's choices from its seed.

Progress. Every `--progress-seconds` (default 60) and at each epoch's end the run prints the
examples done, examples a second and the time left of the epoch and of the run; the out-of-fold
scoring prints the same for windows. A line names no file or path, so a run's log says nothing
of the recordings it read.

Defaults (recorded in the run's `config.json`): AdamW at 2e-5 for layers 9 to 12 and 1e-3 for
the head, batch 32 (split into `--accumulate` micro-batches when VRAM is short), 6 epochs, fp16
autocast on CUDA, seed 17. A batch holds windows of one length, because the encoder's group
norm would see padding. The run writes `<out>/fold-<k>/` or `<out>/final/`: `model.pt`, the
verifier's state dict; `config.json`; `history.json`, each epoch's mean loss; and for a fold
`oof.jsonl`. Exit 0 trained, 2 refused (an existing run directory and a store that cannot be
built included).

Benchmark. The runner that tests this file has no GPU, so the step rate a training projection
needs comes from one run on the training host:

    python -m scripts.wake.verifier.train --benchmark [--dataset <dir>] [--out <dir>] \
        [--batch 32] [--loader-workers N] [--store-workers N] [--benchmark-seconds 30] \
        [--benchmark-workers 0,1,2,4,8] [--benchmark-windows 16384] [--benchmark-decode 2000]

It reads no manifest, no recording and no training store, so `--manifest`, `--weights`,
`--fold`, `--final` and `--cache` are refused. Its windows are int16 noise and its room
responses decaying noise, decoded by `open_store` into `<out>/benchmark-cache` (a temporary
directory without `--out`), a root no training store shares since `open_store` sweeps its root;
the directory is removed at the end, failed or not. The model is the Base+ verifier from random
initialisation. It prints one line a measurement, each over `--benchmark-seconds` after a
warm-up: the loader's rate for each worker count, the store's cold read rate, the step rate of
`fit`'s own loop over batches already on the device (cudnn benchmark off, the trainer's
setting, and on), the end-to-end rate over the real loader, and the scoring rate. With
`--dataset` it reads the index's three files alone, times a store build over a sample of its
public windows, and projects the five folds and the final run; without it, the seconds a
million training examples take. Exit 0 measured, 2 refused.
"""

import argparse
import hashlib
import itertools
import json
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from ..manifest import ManifestError, RecordingManifest, load_manifest
from .augment import Augment, BatchDraws, augment_batch, batch_draws, draw
from .dataset import (
    AFTER_FRAMES,
    BEFORE_FRAMES,
    FRAME_SAMPLES,
    INDEX_FILE,
    INDEX_META,
    MIN_WINDOW_FRAMES,
    PROVENANCE_FILE,
    SAMPLE_RATE,
    DatasetError,
    FileRef,
    Window,
    parse_root_map,
    read_rir,
    read_wav_segment,
    read_windows,
    sha256_file,
    to_16k,
)
from .model import (
    FIRST_TRAINABLE_LAYER,
    ModelError,
    VerifierModel,
    base_plus_params,
    build_verifier,
    normalizes_waveform,
    sha256_path,
)
from .store import (
    RESPONSES_FILE,
    WINDOWS_FILE,
    StoreError,
    WindowStore,
    index_key,
    is_private,
    open_store,
)

FOLDS = 5
REAL_REPEAT = 10
"""How many times a real positive window is seen an epoch, against the synthetic thousands."""
RUN_CONFIG = "config.json"
RUN_MODEL = "model.pt"
RUN_HISTORY = "history.json"
RUN_OOF = "oof.jsonl"
FINAL_RUN = "final"
CACHE_DIR = "window-cache"
"""The store's root under `--out` when `--cache` is not given."""
STORE_WORKERS = 8
"""Decoding processes the store is built with: one per physical core of the training host (8
cores, 16 hardware threads, all of them visible to the training container). Each process decodes
on one thread; whether a core's second hardware thread speeds decoding has not been measured."""
PREFETCH_BATCHES = 4
"""Batches each loader worker keeps ready, so one slow read does not leave the device waiting; at
batch 32 a batch holds about 7 MB of pinned memory at most."""
SCORE_BATCH = 64
"""Held-out windows scored at once; a 116-frame batch of them is 9.5 MB of fp32 input."""
PROGRESS_SECONDS = 60.0


class TrainError(Exception):
    """A refusal; the command exits 2 with the message."""


def fold_run(fold: int) -> str:
    """A fold's run directory name under `--out`."""
    return f"fold-{fold}"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """A run's settings; `config.json` records them with what the run read."""

    optimizer: str = "AdamW"
    layers_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    batch: int = 32
    accumulate: int = 1
    epochs: int = 6
    fp16: bool = True
    seed: int = 17
    first_trainable_layer: int = FIRST_TRAINABLE_LAYER
    folds: int = FOLDS
    real_repeat: int = REAL_REPEAT
    keep: str = "last_epoch"
    loader_workers: int = 4
    augment: Augment = field(default_factory=Augment)


_DEFAULTS = TrainConfig()

# -- The dataset and the manifest -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Indexed:
    """The index's windows, its meta, and the provenance every window was checked against."""

    windows: tuple[Window, ...]
    training: str
    roots: Mapping[str, str]
    provenance_sha256: str


def _json(path: Path, what: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainError(f"{what} could not be read: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainError(f"{what} must be a JSON object")
    return value


def load_index(dataset: Path) -> Indexed:
    """Read the index and refuse any window whose files `provenance.json` does not list.

    Raises:
        TrainError: The meta or provenance is malformed, their `training` differ, or a window
            reads or mixes a file outside the provenance.
        DatasetError: A window record is malformed.
    """
    meta = _json(dataset / INDEX_META, f"dataset {INDEX_META}")
    provenance = _json(dataset / PROVENANCE_FILE, f"dataset {PROVENANCE_FILE}")
    training = meta.get("training")
    if not isinstance(training, str) or provenance.get("training") != training:
        raise TrainError(
            f"the index's training {training!r} is not provenance.json's "
            f"{provenance.get('training')!r}"
        )
    listed = provenance.get("file_sha256")
    if not isinstance(listed, list):
        raise TrainError("provenance.json has no file_sha256 list")
    allowed = frozenset(listed)
    windows = read_windows(dataset / INDEX_FILE)
    for number, window in enumerate(windows, start=1):
        for ref in window.refs():
            if ref.sha256 not in allowed:
                raise TrainError(
                    f"{INDEX_FILE} line {number}: a {ref.source} file is not in provenance.json"
                )
    roots = meta.get("roots") or {}
    if not isinstance(roots, dict):
        raise TrainError(f"dataset {INDEX_META} roots must be an object")
    return Indexed(tuple(windows), training, dict(roots), sha256_file(dataset / PROVENANCE_FILE))


def train_manifest(path: Path, root_map: Mapping[str, str], training: str) -> RecordingManifest:
    """The train-only manifest, checked before any recording is read.

    Its sha256 is checked first, before `load_manifest` opens and hashes every entry, so a
    full manifest is refused without opening a tune or test file.

    Raises:
        TrainError: The sha256 is not the index's `training`, or an entry is outside train.
        ManifestError: The manifest fails its own checks.
    """
    digest = sha256_file(path)
    if digest != training:
        raise TrainError(
            f"the manifest's sha256 {digest} is not the index's training {training}; read the "
            "train-only manifest `python -m scripts.wake.manifest split --split train` wrote"
        )
    manifest = load_manifest(path, root_map=dict(root_map) or None)
    outside = [item for item in manifest.entries if item.split != "train"]
    if outside:
        raise TrainError(
            f"the manifest lists {len(outside)} entries outside the train split; only the "
            "train-only manifest is read"
        )
    return manifest


class WindowReader:
    """Reads a window's samples: a recording through the manifest, a file under its root."""

    def __init__(self, manifest: RecordingManifest, roots: Mapping[str, str]) -> None:
        self._paths = {item.index: (str(item.path), item.sha256) for item in manifest.entries}
        self._roots = dict(roots)

    def path(self, ref: FileRef) -> Path:
        """Where `ref` is read from.

        Raises:
            DatasetError: The entry is not in the manifest, its sha256 differs, or the source
                has no root.
        """
        if ref.source == "manifest":
            found = self._paths.get(ref.entry if ref.entry is not None else -1)
            if found is None or found[1] != ref.sha256:
                raise DatasetError(f"manifest entry {ref.entry} is not the one indexed")
            return Path(found[0])
        root = self._roots.get(ref.source)
        if root is None:
            raise DatasetError(f"the dataset meta has no root for source {ref.source}")
        return Path(root, ref.path or "")

    def samples(self, window: Window) -> np.ndarray:
        """The window's int16 samples, `end - start` of them."""
        path = self.path(window.file)
        count = window.end - window.start
        if path.suffix.lower() == ".flac":
            import soundfile  # noqa: PLC0415 - only LibriSpeech's FLAC needs it

            data, rate = soundfile.read(
                str(path), start=window.start, stop=window.end, dtype="int16", always_2d=True
            )
            if rate != SAMPLE_RATE:
                raise DatasetError(f"a {window.file.source} file is {rate} Hz, not 16 kHz")
            samples = data.mean(axis=1).astype("<i2")
        else:
            samples = read_wav_segment(path, window.start, count)
        if len(samples) != count:
            raise DatasetError(f"a {window.file.source} file is shorter than its window")
        return samples

    def rir(self, ref: FileRef) -> np.ndarray:
        """A room response at 16 kHz, peak-normalised."""
        samples, rate = read_rir(self.path(ref))
        response = to_16k(samples, rate)
        peak = float(np.max(np.abs(response))) if len(response) else 0.0
        return response / peak if peak > 0 else response


# -- Folds ------------------------------------------------------------------------------------


def assign_folds(windows: Sequence[Window], folds: int = FOLDS) -> dict[str, int]:
    """Each group's fold: whole groups, balanced by window count within each kind of group.

    The kind is the group's prefix (`manifest`, `ami`, `librispeech`, ...). Groups go, largest
    first and then by name, to the fold holding the fewest windows of their kind, ties to the
    lowest fold, so every fold has some of each kind and the split is the same on every run.
    """
    sizes = Counter(window.group for window in windows)
    by_kind: dict[str, list[str]] = defaultdict(list)
    for group in sizes:
        by_kind[group.partition(":")[0]].append(group)
    assigned: dict[str, int] = {}
    for kind in sorted(by_kind):
        held = [0] * folds
        for group in sorted(by_kind[kind], key=lambda name: (-sizes[name], name)):
            fold = min(range(folds), key=lambda number: (held[number], number))
            assigned[group] = fold
            held[fold] += sizes[group]
    return assigned


# -- Epochs -----------------------------------------------------------------------------------


def hit_offset(window: Window) -> int:
    """The hit's frame within the window, as `window_bounds` placed it.

    A window starts `BEFORE_FRAMES` before the hit unless the file's start clipped it, and ends
    `AFTER_FRAMES` after it unless the file's end did; when both clipped, the hit is taken as
    `AFTER_FRAMES` before the end, the latest it can be.
    """
    frames = (window.end - window.start) // FRAME_SAMPLES
    if window.start > 0:
        return min(BEFORE_FRAMES, frames - 1)
    return max(0, frames - 1 - AFTER_FRAMES)


def truncate(frames: int, hit: int, rng: random.Random, p: float) -> tuple[int, int]:
    """The frames `[first, last]` kept: with probability `p` a random length down to the minimum.

    The length is uniform from `MIN_WINDOW_FRAMES` to `frames`, and the cut is placed uniformly
    among the placements that keep the hit. It reads no label, so both classes see the same
    distribution of lengths and edges.
    """
    if frames <= MIN_WINDOW_FRAMES or rng.random() >= p:
        return 0, frames - 1
    length = rng.randint(MIN_WINDOW_FRAMES, frames)
    lowest = max(0, hit - length + 1)
    highest = min(hit, frames - length)
    first = rng.randint(lowest, highest) if lowest <= highest else max(0, frames - length)
    return first, first + length - 1


@dataclass(frozen=True, slots=True)
class Item:
    """One training example of an epoch: a window, the frames kept, and its augmentation seed."""

    window: int
    first: int
    last: int
    seed: int

    @property
    def frames(self) -> int:
        """The frames kept."""
        return self.last - self.first + 1


def plan_epoch(
    windows: Sequence[Window],
    chosen: Sequence[int],
    config: TrainConfig,
    epoch: int,
) -> list[list[Item]]:
    """The epoch's batches: each chosen window truncated, grouped by length, shuffled.

    Real positives appear `real_repeat` times. Windows of one kept length share a batch, so no
    window is padded.
    """
    rng = random.Random(f"{config.seed}:{epoch}")  # nosec B311 - reproducible draws only
    items = []
    for index in chosen:
        window = windows[index]
        repeat = config.real_repeat if window.kind == "real_positive" else 1
        frames = (window.end - window.start) // FRAME_SAMPLES
        for _ in range(repeat):
            first, last = truncate(frames, hit_offset(window), rng, config.augment.truncate_p)
            items.append(Item(index, first, last, rng.getrandbits(63)))
    rng.shuffle(items)
    by_length: dict[int, list[Item]] = defaultdict(list)
    for item in items:
        by_length[item.frames].append(item)
    batches = [
        group[start : start + config.batch]
        for _, group in sorted(by_length.items())
        for start in range(0, len(group), config.batch)
    ]
    rng.shuffle(batches)
    return batches


@dataclass(frozen=True, slots=True)
class Batch:
    """One planned batch as a loader worker hands it over, before the device augments it.

    It defines `pin_memory` because DataLoader pins only tensors, sequences, mappings and objects
    that define it.

    Attributes:
        audio: [B, L] int16, each item's kept frames.
        labels: [B] float32, 1 for a positive.
        draws: The items' augmentation choices.
        responses: [B, R] float32, row i item i's room response cut to `L`, zero for an item
            without one; `R` is the longest selected, 0 when no item has a room.
    """

    audio: torch.Tensor
    labels: torch.Tensor
    draws: BatchDraws
    responses: torch.Tensor

    def pin_memory(self) -> "Batch":
        """The same batch in pinned memory, so the copy to the device can be asynchronous."""
        return Batch(
            self.audio.pin_memory(),
            self.labels.pin_memory(),
            self.draws.pin_memory(),
            self.responses.pin_memory(),
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        """The same batch on `device`; `draws` keeps its noise seeds on the host."""
        return Batch(
            self.audio.to(device, non_blocking=non_blocking),
            self.labels.to(device, non_blocking=non_blocking),
            self.draws.to(device, non_blocking=non_blocking),
            self.responses.to(device, non_blocking=non_blocking),
        )


class EpochBatches(torch.utils.data.Dataset):
    """An epoch's plan read from the store, one `Batch` an item.

    A worker reads each item's kept frames, draws its choices from its seed and reads the room
    responses chosen; every step that touches the samples is left to the device.
    """

    def __init__(
        self,
        plan: Sequence[Sequence[Item]],
        labels: Sequence[float],
        store: WindowStore,
        settings: Augment,
    ) -> None:
        self.plan = plan
        self.labels = labels
        self.store = store
        self.settings = settings

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> Batch:
        items = self.plan[index]
        audio = np.stack(
            [
                self.store.read(
                    item.window, item.first * FRAME_SAMPLES, item.frames * FRAME_SAMPLES
                )
                for item in items
            ]
        )
        draws = [draw(item.seed, self.settings, self.store.responses) for item in items]
        length = audio.shape[1]
        rooms = {
            choice.rir: self.store.response(choice.rir)[:length]
            for choice in draws
            if choice.rir is not None
        }
        width = max((len(taps) for taps in rooms.values()), default=0)
        responses = np.zeros((len(items), width), dtype=np.float32)
        for row, choice in enumerate(draws):
            if choice.rir is not None:
                taps = rooms[choice.rir]
                responses[row, : len(taps)] = taps
        return Batch(
            torch.from_numpy(audio),
            torch.tensor([self.labels[item.window] for item in items], dtype=torch.float32),
            batch_draws(draws),
            torch.from_numpy(responses),
        )


_worker_limits: list[object] = []


def _one_thread(worker: int) -> None:
    """A loader worker's setup: one intra-op thread, for torch and for BLAS alike.

    Each worker reads and draws for whole batches, so parallel work comes from the workers; a
    thread pool in each one would only contend with the others and with the main process.
    """
    del worker
    from threadpoolctl import threadpool_limits  # noqa: PLC0415 - loader workers only

    torch.set_num_threads(1)
    _worker_limits.append(threadpool_limits(1))


def epoch_loader(data: EpochBatches, workers: int, pin: bool) -> torch.utils.data.DataLoader:
    """The DataLoader over an epoch's batches, in plan order, `workers` processes reading ahead."""
    return torch.utils.data.DataLoader(
        data,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        worker_init_fn=_one_thread,
        prefetch_factor=PREFETCH_BATCHES if workers > 0 else None,
    )


class DeviceEpoch:
    """An epoch's batches moved to the device and augmented there, as `fit` takes them."""

    def __init__(
        self,
        loader: Iterable[Batch],
        examples: int,
        device: torch.device,
        settings: Augment,
    ) -> None:
        self.loader = loader
        self.examples = examples
        self.device = device
        self.settings = settings

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for batch in self.loader:
            moved = batch.to(self.device, non_blocking=True)
            with torch.no_grad():
                waveforms = augment_batch(moved.audio, moved.draws, moved.responses, self.settings)
            yield waveforms, moved.labels


# -- Progress ---------------------------------------------------------------------------------


Clock = Callable[[], float]
Progress = Callable[[str], None]


def _duration(seconds: float) -> str:
    whole = int(round(seconds))
    return f"{whole // 3600}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def _eta(left: int, rate: float) -> str:
    if left <= 0:
        return "0:00:00"
    return _duration(left / rate) if rate > 0 else "unknown"


class _EpochMeter:
    """An epoch's progress lines: counts, rate and the time left, never a file name or a path.

    The run's time left takes every later epoch to be as long as this one, as `plan_epoch`'s are.
    """

    def __init__(
        self, epoch: int, epochs: int, total: int, progress: Progress, interval: float, clock: Clock
    ) -> None:
        self.epoch = epoch
        self.epochs = epochs
        self.total = total
        self.progress = progress
        self.interval = interval
        self.clock = clock
        self.started = clock()
        self.last = self.started

    def _rate(self, done: int, now: float) -> float:
        elapsed = now - self.started
        return done / elapsed if elapsed > 0 else 0.0

    def _run_left(self, done: int) -> int:
        return self.total - done + (self.epochs - self.epoch) * self.total

    def step(self, done: int) -> None:
        now = self.clock()
        if now - self.last < self.interval:
            return
        self.last = now
        rate = self._rate(done, now)
        self.progress(
            f"epoch {self.epoch}: {done}/{self.total} examples, {rate:.1f} examples/s, "
            f"epoch ETA {_eta(self.total - done, rate)}, run ETA {_eta(self._run_left(done), rate)}"
        )

    def finish(self, seen: int, loss: float) -> None:
        rate = self._rate(seen, self.clock())
        self.progress(
            f"epoch {self.epoch}: loss {loss:.4f} over {seen} examples, {rate:.1f} examples/s, "
            f"run ETA {_eta(self._run_left(seen), rate)}"
        )


class _ScoreMeter:
    """The out-of-fold scoring's progress lines, never a file name or a path."""

    def __init__(self, total: int, progress: Progress, interval: float, clock: Clock) -> None:
        self.total = total
        self.progress = progress
        self.interval = interval
        self.clock = clock
        self.done = 0
        self.started = clock()
        self.last = self.started

    def add(self, count: int) -> None:
        self.done += count
        now = self.clock()
        if now - self.last >= self.interval:
            self.last = now
            self._emit(now)

    def finish(self) -> None:
        self._emit(self.clock())

    def _emit(self, now: float) -> None:
        elapsed = now - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        self.progress(
            f"oof: {self.done}/{self.total} windows, {rate:.1f} windows/s, "
            f"ETA {_eta(self.total - self.done, rate)}"
        )


# -- Training ---------------------------------------------------------------------------------


Batches = Callable[[int], Iterable[tuple[torch.Tensor, torch.Tensor]]]
"""An epoch's batches by epoch number, each `(waveforms, labels)`: a sequence of them, or an
iterable such as `DeviceEpoch` that carries its `examples`."""


def _examples(epoch: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> int:
    examples = getattr(epoch, "examples", None)
    if examples is not None:
        return int(examples)
    if isinstance(epoch, Sequence):
        return sum(len(labels) for _, labels in epoch)
    raise TypeError("an epoch's batches are a sequence or carry their `examples`")


def fit(
    model: VerifierModel,
    batches: Batches,
    config: TrainConfig,
    device: torch.device,
    *,
    progress: Progress = print,
    interval: float = PROGRESS_SECONDS,
    clock: Clock = time.monotonic,
) -> list[dict[str, float]]:
    """Train the fixed schedule; the model is left at its last epoch, which the run keeps.

    The loss is summed on the device and read only for the epoch's line, so the loop waits for
    the device no more often than GradScaler's own step does.

    Args:
        model: The verifier.
        batches: Each epoch's batches.
        config: The schedule, optimiser and precision.
        device: Where the model trains.
        progress: Takes a line every `interval` seconds of `clock` and one at each epoch's end.
        interval: Seconds between progress lines.
        clock: The clock `interval` and the rates are measured on.

    Returns:
        Each epoch's mean loss and example count, in order; no epoch is selected from them.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(layers_lr=config.layers_lr, head_lr=config.head_lr),
        weight_decay=config.weight_decay,
    )
    half = config.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=half)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_batches = batches(epoch)
        meter = _EpochMeter(
            epoch, config.epochs, _examples(epoch_batches), progress, interval, clock
        )
        total = torch.zeros((), device=device)
        seen = 0
        for waveforms, labels in epoch_batches:
            optimizer.zero_grad(set_to_none=True)
            size = len(labels)
            for part in range(config.accumulate):
                chunk = slice(
                    part * size // config.accumulate, (part + 1) * size // config.accumulate
                )
                if chunk.start == chunk.stop:
                    continue
                audio = waveforms[chunk].to(device)
                target = labels[chunk].to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=half):
                    logits = model(audio)
                loss = loss_fn(logits.float(), target) / size
                scaler.scale(loss).backward()
                total += loss.detach() * size
            scaler.step(optimizer)
            scaler.update()
            seen += size
            meter.step(seen)
        mean = float(total) / max(seen, 1)
        history.append({"epoch": epoch, "loss": mean, "examples": seen})
        meter.finish(seen, mean)
    model.eval()
    return history


@torch.no_grad()
def score_windows(
    model: VerifierModel,
    store: WindowStore,
    held: Sequence[int],
    device: torch.device,
    *,
    batch: int = SCORE_BATCH,
    progress: Progress = print,
    interval: float = PROGRESS_SECONDS,
    clock: Clock = time.monotonic,
) -> list[float]:
    """Each held window's score, whole and unaugmented, as the exported graph would give it.

    Windows of one length are scored `batch` at a time, in eval mode and fp32 without autocast,
    so a window's score depends on its batch only by float rounding; one length a batch means no
    window is padded.

    Args:
        model: The trained verifier.
        store: The run's window store.
        held: The held-out windows' positions in the index.
        device: Where the model runs.
        batch: Windows scored at once.
        progress: Takes a line every `interval` seconds of `clock` and one at the end.
        interval: Seconds between progress lines.
        clock: The clock `interval` and the rate are measured on.

    Returns:
        The scores in `held`'s order.
    """
    model.eval()
    model.to(device)
    by_length: dict[int, list[int]] = defaultdict(list)
    for position, window in enumerate(held):
        by_length[store.length(window)].append(position)
    scores = [0.0] * len(held)
    meter = _ScoreMeter(len(held), progress, interval, clock)
    for length, positions in sorted(by_length.items()):
        for start in range(0, len(positions), batch):
            chunk = positions[start : start + batch]
            audio = np.stack([store.read(held[position], 0, length) for position in chunk])
            waveform = torch.from_numpy(audio.astype(np.float32) / np.float32(32768.0))
            with torch.autocast(device.type, enabled=False):
                logits = model(waveform.to(device))
            values = torch.sigmoid(logits.float()).cpu().tolist()
            for position, value in zip(chunk, values, strict=True):
                scores[position] = float(value)
            meter.add(len(chunk))
    meter.finish()
    return scores


ModelBuilder = Callable[[TrainConfig, Path | None], VerifierModel]


def build_wavlm_verifier(config: TrainConfig, weights: Path | None) -> VerifierModel:
    """The verifier over the staged Base+ weights, loaded strictly; random with no weights."""
    return build_verifier(
        base_plus_params(), weights=weights, first_trainable=config.first_trainable_layer
    )


def run(args: argparse.Namespace, build: ModelBuilder, clock: Clock = time.monotonic) -> int:
    """Check the inputs, train one fold or the final model, and write the run."""
    if normalizes_waveform():
        raise TrainError(
            "torchaudio's WAVLM_BASE_PLUS._normalize_waveform reads True; the exported graph "
            "applies no normalisation, so the verifier would be served audio it was not "
            "trained on"
        )
    name = FINAL_RUN if args.final else fold_run(args.fold)
    if args.fold is not None and not 0 <= args.fold < FOLDS:
        raise TrainError(f"--fold must be 0 to {FOLDS - 1}, got {args.fold}")
    out = args.out / name
    if out.exists():
        raise TrainError(f"the run {name} already exists; a run is written new")
    if args.batch < 1 or args.accumulate < 1 or args.accumulate > args.batch:
        raise TrainError("--batch and --accumulate must be at least 1, accumulate <= batch")
    if args.loader_workers < 0 or args.store_workers < 1 or not args.progress_seconds > 0:
        raise TrainError(
            "--loader-workers must be at least 0, --store-workers at least 1, and "
            "--progress-seconds above 0"
        )
    config = TrainConfig(
        batch=args.batch,
        accumulate=args.accumulate,
        seed=args.seed,
        loader_workers=args.loader_workers,
    )
    indexed = load_index(args.dataset)
    manifest = train_manifest(args.manifest, parse_root_map(args.root_map), indexed.training)
    reader = WindowReader(manifest, indexed.roots)
    folds = assign_folds(indexed.windows, config.folds)
    windows = indexed.windows
    held = [i for i, w in enumerate(windows) if not args.final and folds[w.group] == args.fold]
    held_set = set(held)
    chosen = [i for i in range(len(windows)) if i not in held_set]
    if not chosen:
        raise TrainError("no window is left to train on")
    rirs = sorted(
        {ref.sha256: ref for w in windows for ref in w.inputs if ref.source == "rirs"}.values(),
        key=lambda ref: ref.sha256,
    )
    # Before the model and the device: the store's decoding processes start by spawn, and
    # nothing may have started CUDA before them.
    store = open_store(
        args.cache if args.cache is not None else args.out / CACHE_DIR,
        index_key(args.dataset),
        windows,
        rirs,
        reader,
        provenance_sha256=indexed.provenance_sha256,
        workers=args.store_workers,
        interval=args.progress_seconds,
        clock=clock,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = build(config, args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = device.type == "cuda"
    labels = [1.0 if window.label == "positive" else 0.0 for window in windows]

    def batches(epoch: int) -> DeviceEpoch:
        plan = plan_epoch(windows, chosen, config, epoch)
        data = EpochBatches(plan, labels, store, config.augment)
        loader = epoch_loader(data, config.loader_workers if on_cuda else 0, pin=on_cuda)
        return DeviceEpoch(loader, sum(len(batch) for batch in plan), device, config.augment)

    history = fit(model, batches, config, device, interval=args.progress_seconds, clock=clock)
    out.mkdir(parents=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / RUN_MODEL)
    if not args.final:
        scores = score_windows(
            model, store, held, device, interval=args.progress_seconds, clock=clock
        )
        with (out / RUN_OOF).open("w", encoding="utf-8", newline="\n") as handle:
            for index, score in zip(held, scores, strict=True):
                window = windows[index]
                row = {
                    "window": index,
                    "kind": window.kind,
                    "label": window.label,
                    "group": window.group,
                    "file_sha256": window.file.sha256,
                    "cutoffs": list(window.cutoffs),
                    "score": score,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    record = {
        "version": 1,
        "run": name,
        "fold": None if args.final else args.fold,
        **asdict(config),
        "fp16_autocast": config.fp16 and device.type == "cuda",
        "device": device.type,
        "params": base_plus_params(),
        "normalize_waveform": normalizes_waveform(),
        "training": indexed.training,
        "provenance_sha256": indexed.provenance_sha256,
        "weights_sha256": sha256_path(args.weights),
        "kept_epoch": config.epochs,
        "windows": {"train": len(chosen), "held_out": len(held)},
    }
    (out / RUN_CONFIG).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (out / RUN_HISTORY).write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    counts = Counter(windows[i].label for i in chosen)
    print(f"{name}: trained on {counts['positive']} positive, {counts['negative']} negative")
    return 0


# -- Benchmark --------------------------------------------------------------------------------


BENCHMARK_CACHE = "benchmark-cache"
"""The benchmark's store root under `--out`. `open_store` removes every other store under its
root, so this root is the benchmark's alone, never `--cache` or `<out>/window-cache`."""
BENCHMARK_SECONDS = 30.0
BENCHMARK_WORKERS = (0, 1, 2, 4, 8)
BENCHMARK_WINDOWS = 16384
"""Synthetic windows: 16,384 whole windows are 1.2 GB, which the page cache holds where the real
store's 30 GB outgrows it, so the cold read line is the storage bound."""
BENCHMARK_DECODE = 2000
BENCHMARK_FRAMES = BEFORE_FRAMES + AFTER_FRAMES + 1
"""A whole window's frames, 116."""
BENCHMARK_ROOMS = (8000, 16000, 32000)
"""The synthetic room responses' lengths in taps, `ROOMS_EACH` of each."""
ROOMS_EACH = 20
NOISE_LEVEL = 3000
WARM_BATCHES = 8
"""Batches handed over before a loader's measurement starts, so its workers have started and
filled their prefetch."""
STEP_POOL = 32
"""Distinct batches the step rate cycles through on the device, about 7 MB each at most."""
PLAN_REPEATS = 10_000
"""How often one loader cycles the synthetic plan, so no measurement restarts its workers."""
HONOURED = 0.8
"""A cold pass at this fraction of the warm pass's rate or above reads as a cache not dropped."""
MILLION = 1_000_000


def _say(line: str) -> None:
    print(line, flush=True)


def _quiet(line: str) -> None:
    del line


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SyntheticReader:
    """Noise windows and decaying room responses made in memory for `open_store` to decode.

    Each is drawn from the seed and its own position, so a decoding process makes the same
    samples as any other; nothing is read from a file.
    """

    seed: int

    def samples(self, window: Window) -> np.ndarray:
        """The window's int16 noise, `end - start` samples."""
        rng = np.random.default_rng((self.seed, window.start))
        count = window.end - window.start
        noise = rng.integers(-NOISE_LEVEL, NOISE_LEVEL + 1, count, dtype=np.int16)
        return noise.astype("<i2", copy=False)

    def rir(self, ref: FileRef) -> np.ndarray:
        """An exponentially decaying noise response of `ref.samples` taps, peak-normalised."""
        rng = np.random.default_rng((self.seed, int(ref.sha256[:8], 16)))
        decay = np.exp(-np.arange(ref.samples) / (ref.samples / 6))
        taps = rng.standard_normal(ref.samples) * decay
        return (taps / np.max(np.abs(taps))).astype(np.float32)


def synthetic_windows(count: int) -> list[Window]:
    """`count` whole windows of one noise file, alternately negative and positive."""
    length = BENCHMARK_FRAMES * FRAME_SAMPLES
    ref = FileRef("benchmark", _digest("benchmark noise"), None, None, count * length)
    return [
        Window(
            ref,
            number * length,
            (number + 1) * length,
            "positive" if number % 2 else "negative",
            "synthetic",
            f"benchmark:{number % 64}",
            (),
        )
        for number in range(count)
    ]


def synthetic_rooms() -> list[FileRef]:
    """`ROOMS_EACH` room responses of each length in `BENCHMARK_ROOMS`."""
    return [
        FileRef("benchmark", _digest(f"benchmark room {taps} {number}"), None, None, taps)
        for taps in BENCHMARK_ROOMS
        for number in range(ROOMS_EACH)
    ]


class _Cycle(Sequence[Sequence[Item]]):
    """A plan repeated `times` over, read lazily, so one loader outlasts any measurement."""

    def __init__(self, plan: Sequence[Sequence[Item]], times: int) -> None:
        self.plan = plan
        self.times = times

    def __len__(self) -> int:
        return len(self.plan) * self.times

    def __getitem__(self, index: int) -> Sequence[Item]:  # type: ignore[override]
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.plan[index % len(self.plan)]


class _Timed:
    """Hands over `batches` until `seconds` have passed after the first `warm` of them.

    The clock starts as batch `warm` is asked for and is checked before each later one is, so
    the examples counted are those handed over inside the time measured. A consumer that
    finishes its work on a batch before asking for the next, as `fit`'s loop does, has done the
    counted work by the time `rate` is read.
    """

    def __init__(self, batches: Iterable[Batch], warm: int, seconds: float, clock: Clock) -> None:
        self.batches = batches
        self.warm = warm
        self.seconds = seconds
        self.clock = clock
        self.examples = 0
        self.started: float | None = None

    def __iter__(self) -> Iterator[Batch]:
        iterator = iter(self.batches)
        try:
            for number in itertools.count():
                if number == self.warm:
                    self.started = self.clock()
                elif self.started is not None and self.clock() - self.started >= self.seconds:
                    return
                batch = next(iterator, None)
                if batch is None:
                    return
                if number >= self.warm:
                    self.examples += len(batch.labels)
                yield batch
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def rate(self, now: float) -> float:
        """Examples a second from the start to `now`."""
        if self.started is None or now <= self.started:
            return 0.0
        return self.examples / (now - self.started)


def _examples_in(batches: Sequence[Batch]) -> int:
    return sum(len(batch.labels) for batch in batches)


def _drop_cache(directory: Path) -> str | None:
    """Ask the kernel to drop the store files' cached pages; why it could not, or None."""
    advise = getattr(os, "posix_fadvise", None)
    if advise is None:
        return "posix_fadvise is not available"
    for name in (WINDOWS_FILE, RESPONSES_FILE):
        fd = os.open(directory / name, os.O_RDONLY)
        try:
            advise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError as exc:
            return f"posix_fadvise was refused ({exc.strerror})"
        finally:
            os.close(fd)
    return None


def _read_pass(
    store: WindowStore, order: Sequence[int], seconds: float, clock: Clock
) -> tuple[int, float]:
    started = clock()
    read = 0
    for window in order:
        store.read(window, 0, store.length(window))
        read += 1
        if clock() - started >= seconds:
            break
    return read, max(clock() - started, 1e-9)


def _cold_read(
    store: WindowStore, mean_frames: float, seed: int, seconds: float, clock: Clock
) -> float:
    """One pass of whole-window reads in random order after the store's pages are dropped.

    A warm pass over the same windows follows, so a drop the file system did not honour shows
    as a cold rate close to the warm one. Returns the cold rate as training examples a second:
    an example reads its kept frames alone.
    """
    order = list(range(len(store)))
    random.Random(seed).shuffle(order)  # nosec B311 - a reproducible order only
    refused = _drop_cache(store.directory)
    read, elapsed = _read_pass(store, order, seconds, clock)
    _, again = _read_pass(store, order[:read], float("inf"), clock)
    cold, warm = read / elapsed, read / again
    if refused is not None:
        note = f"page cache not dropped: {refused}"
    elif cold >= HONOURED * warm:
        note = "page cache dropped, but the cold pass was not slower than the warm one"
    else:
        note = "page cache dropped"
    megabytes = cold * BENCHMARK_FRAMES * FRAME_SAMPLES * 2 / 1e6
    _say(
        f"store cold read: {read} windows in random order in {elapsed:.1f} s, "
        f"{cold:.1f} windows/s, {megabytes:.1f} MB/s; warm {warm:.1f} windows/s; {note}"
    )
    examples = cold * BENCHMARK_FRAMES / mean_frames
    _say(
        f"store cold read as examples: {cold:.1f} windows/s x {BENCHMARK_FRAMES} / "
        f"{mean_frames:.1f} mean kept frames = {examples:.1f} examples/s"
    )
    return examples


def _step_rate(
    model: VerifierModel,
    warm: Sequence[Batch],
    pool: Sequence[Batch],
    config: TrainConfig,
    device: torch.device,
    seconds: float,
    clock: Clock,
) -> float:
    """`fit`'s own loop over batches already on the device: augmentation in, loader out.

    The warm-up epoch runs `warm` once; the measured one cycles `pool` and skips its first
    batch, which pays for the new optimiser's state.
    """
    one = replace(config, epochs=1)
    fit(
        model,
        lambda epoch: DeviceEpoch(warm, _examples_in(warm), device, config.augment),
        one,
        device,
        progress=_quiet,
    )
    timed = _Timed(itertools.cycle(pool), 1, seconds, clock)
    fit(
        model,
        lambda epoch: DeviceEpoch(timed, _examples_in(pool), device, config.augment),
        one,
        device,
        progress=_quiet,
    )
    return timed.rate(clock())


def _step_rates(
    model: VerifierModel,
    data: EpochBatches,
    config: TrainConfig,
    device: torch.device,
    seconds: float,
    clock: Clock,
) -> float:
    """The step rate with cudnn's benchmark off (the trainer's setting) and on; the first.

    On CUDA the warm-up covers every kept length once, so each shape's kernels are chosen
    before the clock starts. The CPU chooses none, so there it is one batch.
    """
    pool = [data[index].to(device) for index in range(min(STEP_POOL, len(data)))]
    if device.type != "cuda":
        rate = _step_rate(model, [data[0]], pool, config, device, seconds, clock)
        _say(f"step: cpu, {rate:.1f} examples/s; one warm-up batch, no cudnn")
        return rate
    firsts: dict[int, int] = {}
    for index, items in enumerate(data.plan):
        firsts.setdefault(items[0].frames, index)
    warm = [data[index] for index in sorted(firsts.values())]
    previous = torch.backends.cudnn.benchmark
    rates: dict[bool, float] = {}
    try:
        for flag in (False, True):
            torch.backends.cudnn.benchmark = flag
            torch.cuda.reset_peak_memory_stats(device)
            rates[flag] = _step_rate(model, warm, pool, config, device, seconds, clock)
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            _say(
                f"step: cudnn benchmark {'on' if flag else 'off'}, {rates[flag]:.1f} "
                f"examples/s over {len(warm)} warmed lengths, peak {peak:.2f} GiB allocated"
            )
    finally:
        torch.backends.cudnn.benchmark = previous
    return rates[False]


def _scoring_rate(
    model: VerifierModel, store: WindowStore, device: torch.device, seconds: float, clock: Clock
) -> float:
    """`score_windows`' rate over whole windows, one batch of them warming up."""
    chunk = min(SCORE_BATCH, len(store))
    windows = itertools.cycle(range(len(store)))
    score_windows(model, store, list(itertools.islice(windows, chunk)), device, progress=_quiet)
    started = clock()
    scored = 0
    while True:
        held = list(itertools.islice(windows, chunk))
        score_windows(model, store, held, device, progress=_quiet)
        scored += len(held)
        now = clock()
        if now - started >= seconds:
            return scored / max(now - started, 1e-9)


@dataclass(frozen=True, slots=True)
class Workload:
    """What the five folds and the final run read.

    Attributes:
        folds: Each fold's training examples an epoch (`REAL_REPEAT` included) and its
            held-out windows.
        final: The final run's training examples an epoch.
        public: The windows a store build decodes.
    """

    folds: tuple[tuple[int, int], ...]
    final: int
    public: int


def workload(windows: Sequence[Window], config: TrainConfig = _DEFAULTS) -> Workload:
    """The workload of the index's real fold split."""
    assigned = assign_folds(windows, config.folds)
    repeats = [config.real_repeat if w.kind == "real_positive" else 1 for w in windows]
    split = []
    for fold in range(config.folds):
        held = [assigned[window.group] == fold for window in windows]
        trained = sum(repeat for repeat, out in zip(repeats, held, strict=True) if not out)
        split.append((trained, sum(held)))
    public = sum(1 for window in windows if not is_private(window))
    return Workload(tuple(split), sum(repeats), public)


@dataclass(frozen=True, slots=True)
class Rates:
    """The measured rates: examples a second, but windows a second for `scoring` and `build`."""

    step: float
    loader: float
    cold_read: float
    scoring: float
    build: float

    @property
    def training(self) -> float:
        """The bound on training: the least of the step, the loader and the cold read."""
        return min(self.step, self.loader, self.cold_read)


def _over(amount: float, rate: float) -> float:
    if amount == 0:
        return 0.0
    return amount / rate if rate > 0 else float("inf")


def projection(work: Workload | None, rates: Rates, epochs: int) -> list[str]:
    """The projection's lines, each with its arithmetic.

    With a workload: each run's training and each fold's scoring, the store build once, and
    the total. Without one: the seconds a million training examples take.
    """
    training = rates.training
    lines = [
        f"projection: training rate min(step {rates.step:.1f}, loader {rates.loader:.1f}, "
        f"cold read {rates.cold_read:.1f}) = {training:.1f} examples/s"
    ]
    if work is None:
        lines.append(
            f"projection: {MILLION} / {training:.1f} = {_over(MILLION, training):.0f} s "
            "a million training examples"
        )
        return lines
    total = 0.0
    for fold, (examples, held) in enumerate(work.folds):
        trained = _over(epochs * examples, training)
        scored = _over(held, rates.scoring)
        total += trained + scored
        lines.append(
            f"projection: fold {fold} training {epochs} x {examples} / {training:.1f} "
            f"= {trained:.0f} s"
        )
        lines.append(
            f"projection: fold {fold} scoring {held} / {rates.scoring:.1f} = {scored:.0f} s"
        )
    final = _over(epochs * work.final, training)
    built = _over(work.public, rates.build)
    total += final + built
    lines += [
        f"projection: final training {epochs} x {work.final} / {training:.1f} = {final:.0f} s",
        f"projection: store build {work.public} / {rates.build:.1f} = {built:.0f} s",
        f"projection: total {total:.0f} s = {total / 3600:.2f} h",
    ]
    return lines


def _sample_build(indexed: Indexed, root: Path, args: argparse.Namespace, clock: Clock) -> float:
    """Windows a second of a store build over a random sample of the index's public windows.

    The reader has no manifest entries, so no recording can be read; the sample's windows read
    their files from the source roots `meta.json` gives.
    """
    public = [window for window in indexed.windows if not is_private(window)]
    count = min(args.benchmark_decode, len(public))
    sample = random.Random(args.seed).sample(public, count)  # nosec B311 - a sample only
    reader = WindowReader(RecordingManifest(Path(), "", ()), indexed.roots)
    started = clock()
    built = open_store(
        root,
        _digest(f"benchmark sample {indexed.provenance_sha256} {count}"),
        sample,
        [],
        reader,
        provenance_sha256=indexed.provenance_sha256,
        workers=args.store_workers,
        progress=_quiet,
    )
    elapsed = max(clock() - started, 1e-9)
    built.close()
    shutil.rmtree(root, ignore_errors=True)
    rate = count / elapsed
    _say(
        f"store build: {count} public windows decoded in {elapsed:.1f} s with "
        f"{args.store_workers} workers, {rate:.1f} windows/s"
    )
    return rate


def _check_benchmark(args: argparse.Namespace) -> None:
    if args.batch < 1 or args.accumulate < 1 or args.accumulate > args.batch:
        raise TrainError("--batch and --accumulate must be at least 1, accumulate <= batch")
    if args.loader_workers < 0 or args.store_workers < 1:
        raise TrainError("--loader-workers must be at least 0 and --store-workers at least 1")
    if not args.benchmark_seconds > 0 or args.benchmark_windows < 1 or args.benchmark_decode < 1:
        raise TrainError(
            "--benchmark-seconds must be above 0, --benchmark-windows and --benchmark-decode "
            "at least 1"
        )


def benchmark(args: argparse.Namespace, build: ModelBuilder, clock: Clock = time.monotonic) -> int:
    """Measure the trainer's rates on synthetic windows and print the projection.

    Its store root is removed at the end, whether the measurements finished or not.
    """
    _check_benchmark(args)
    indexed = load_index(args.dataset) if args.dataset is not None else None
    if args.out is not None:
        root = args.out / BENCHMARK_CACHE
        root.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(tempfile.mkdtemp(prefix="verifier-benchmark-"))
    try:
        _measure(args, build, clock, indexed, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return 0


def _measure(
    args: argparse.Namespace,
    build: ModelBuilder,
    clock: Clock,
    indexed: Indexed | None,
    root: Path,
) -> None:
    config = TrainConfig(
        batch=args.batch,
        accumulate=args.accumulate,
        seed=args.seed,
        loader_workers=args.loader_workers,
    )
    seconds = args.benchmark_seconds
    windows = synthetic_windows(args.benchmark_windows)
    rooms = synthetic_rooms()
    # Both stores before the device: their decoding processes start by spawn, as in `run`.
    started = clock()
    store = open_store(
        root / "synthetic",
        _digest(f"benchmark {len(windows)} {len(rooms)}"),
        windows,
        rooms,
        SyntheticReader(config.seed),
        provenance_sha256=_digest("benchmark"),
        workers=args.store_workers,
        progress=_quiet,
    )
    size = sum((store.directory / name).stat().st_size for name in (WINDOWS_FILE, RESPONSES_FILE))
    _say(
        f"synthetic store: {len(windows)} windows of {BENCHMARK_FRAMES} frames and {len(rooms)} "
        f"room responses of {', '.join(str(taps) for taps in BENCHMARK_ROOMS)} taps, "
        f"{size / 1e9:.2f} GB, built in {clock() - started:.1f} s with {args.store_workers} "
        "workers"
    )
    work = None
    build_rate = 0.0
    if indexed is not None:
        work = workload(indexed.windows, config)
        _say(
            f"index: {len(indexed.windows)} windows, {work.public} public, held-out windows by "
            f"fold {', '.join(str(held) for _, held in work.folds)}"
        )
        build_rate = _sample_build(indexed, root / "sample", args, clock)
    labels = [1.0 if window.label == "positive" else 0.0 for window in windows]
    plan = plan_epoch(windows, range(len(windows)), config, 1)
    mean_frames = float(np.mean([item.frames for items in plan for item in items]))
    cold_read = _cold_read(store, mean_frames, config.seed, seconds, clock)

    on_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if on_cuda else "cpu")
    if on_cuda:
        _say(f"device: cuda, {torch.cuda.get_device_name(device)}")
    else:
        _say("device: cpu; no CUDA, so the step, end-to-end and scoring rates are the CPU's")
    cycled = EpochBatches(_Cycle(plan, PLAN_REPEATS), labels, store, config.augment)
    loader_rates: dict[int, float] = {}
    for workers in dict.fromkeys([*args.benchmark_workers, config.loader_workers]):
        timed = _Timed(epoch_loader(cycled, workers, pin=on_cuda), WARM_BATCHES, seconds, clock)
        for _ in timed:
            pass
        loader_rates[workers] = timed.rate(clock())
        pinned = ", pinned" if on_cuda else ""
        _say(f"loader: {workers} workers, {loader_rates[workers]:.1f} examples/s{pinned}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = build(config, None)
    data = EpochBatches(plan, labels, store, config.augment)
    step = _step_rates(model, data, config, device, seconds, clock)
    if on_cuda:
        torch.cuda.empty_cache()

    workers = config.loader_workers if on_cuda else 0
    timed = _Timed(
        epoch_loader(cycled, workers, pin=on_cuda), WARM_BATCHES if on_cuda else 1, seconds, clock
    )
    examples = sum(len(items) for items in plan)
    fit(
        model,
        lambda epoch: DeviceEpoch(timed, examples, device, config.augment),
        replace(config, epochs=1),
        device,
        progress=_quiet,
    )
    reads = "" if on_cuda else ", as the trainer reads on the CPU"
    _say(f"end-to-end: {workers} loader workers{reads}, {timed.rate(clock()):.1f} examples/s")

    scoring = _scoring_rate(model, store, device, seconds, clock)
    _say(f"scoring: {BENCHMARK_FRAMES}-frame windows, {scoring:.1f} windows/s")
    store.close()

    loader = loader_rates[config.loader_workers]
    rates = Rates(step, loader, cold_read, scoring, build_rate)
    for line in projection(work, rates, config.epochs):
        _say(line)
    ratio = loader / step if step > 0 else float("inf")
    _say(f"ratio: loader / step = {loader:.1f} / {step:.1f} = {ratio:.2f}")


# -- The command line -------------------------------------------------------------------------


_BENCHMARK_DEFAULTS: dict[str, object] = {
    "benchmark_seconds": BENCHMARK_SECONDS,
    "benchmark_workers": BENCHMARK_WORKERS,
    "benchmark_windows": BENCHMARK_WINDOWS,
    "benchmark_decode": BENCHMARK_DECODE,
}
_TRAINING_INPUTS = ("manifest", "weights", "fold", "final", "cache")
"""What the benchmark refuses: it reads no manifest, no weights and no training store."""


def _worker_counts(text: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(part) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of counts: {text}") from exc
    if any(count < 0 for count in counts):
        raise argparse.ArgumentTypeError("a worker count is at least 0")
    return counts


def _parser(benchmark: bool = False) -> argparse.ArgumentParser:
    """The command line; with `benchmark` no training input is required."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.verifier.train", description=__doc__
    )
    training = not benchmark
    parser.add_argument("--dataset", required=training, type=Path)
    parser.add_argument("--manifest", required=training, type=Path)
    parser.add_argument("--root-map", action="append", default=[])
    parser.add_argument("--weights", required=training, type=Path)
    parser.add_argument("--out", required=training, type=Path)
    which = parser.add_mutually_exclusive_group(required=training)
    which.add_argument("--fold", type=int)
    which.add_argument("--final", action="store_true")
    parser.add_argument("--batch", type=int, default=_DEFAULTS.batch)
    parser.add_argument("--accumulate", type=int, default=_DEFAULTS.accumulate)
    parser.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    parser.add_argument("--cache", type=Path, help=f"the store's root; <out>/{CACHE_DIR} if unset")
    parser.add_argument("--store-workers", type=int, default=STORE_WORKERS)
    parser.add_argument("--loader-workers", type=int, default=_DEFAULTS.loader_workers)
    parser.add_argument("--progress-seconds", type=float, default=PROGRESS_SECONDS)
    parser.add_argument("--benchmark", action="store_true", help="measure; train nothing")
    parser.add_argument("--benchmark-seconds", type=float, help=f"default {BENCHMARK_SECONDS:g}")
    parser.add_argument(
        "--benchmark-workers",
        type=_worker_counts,
        help=f"default {','.join(str(count) for count in BENCHMARK_WORKERS)}",
    )
    parser.add_argument("--benchmark-windows", type=int, help=f"default {BENCHMARK_WINDOWS}")
    parser.add_argument("--benchmark-decode", type=int, help=f"default {BENCHMARK_DECODE}")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The arguments; argparse exits 2 on a missing, refused or malformed one.

    `--benchmark` is found first, so a training run still requires every input it did, and a
    benchmark refuses the training inputs and fills its own flags' defaults.
    """
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--benchmark", action="store_true")
    known, _ = probe.parse_known_args(argv)
    parser = _parser(benchmark=known.benchmark)
    args = parser.parse_args(argv)
    if args.benchmark:
        for name in _TRAINING_INPUTS:
            value = getattr(args, name)
            if value is not None and value is not False:
                parser.error(
                    f"--benchmark reads no manifest, weights or training store: --{name} is "
                    "refused"
                )
        for name, default in _BENCHMARK_DEFAULTS.items():
            if getattr(args, name) is None:
                setattr(args, name, default)
        return args
    for name in _BENCHMARK_DEFAULTS:
        if getattr(args, name) is not None:
            parser.error(f"--{name.replace('_', '-')} is read only with --benchmark")
    return args


def main(
    argv: Sequence[str] | None = None,
    *,
    build: ModelBuilder | None = None,
    clock: Clock = time.monotonic,
) -> int:
    """Run the trainer or its benchmark; 0 trained or measured, 2 refused."""
    args = parse_args(argv)
    try:
        if args.benchmark:
            return benchmark(args, build or build_wavlm_verifier, clock)
        return run(args, build or build_wavlm_verifier, clock)
    except (TrainError, ManifestError, DatasetError, ModelError, StoreError) as exc:
        print(f"train: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
