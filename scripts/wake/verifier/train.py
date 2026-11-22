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
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
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
from .store import StoreError, WindowStore, index_key, open_store

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


ModelBuilder = Callable[[TrainConfig, Path], VerifierModel]


def build_wavlm_verifier(config: TrainConfig, weights: Path) -> VerifierModel:
    """The verifier over the staged Base+ weights, loaded strictly."""
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.verifier.train", description=__doc__
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root-map", action="append", default=[])
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--fold", type=int)
    which.add_argument("--final", action="store_true")
    parser.add_argument("--batch", type=int, default=_DEFAULTS.batch)
    parser.add_argument("--accumulate", type=int, default=_DEFAULTS.accumulate)
    parser.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    parser.add_argument("--cache", type=Path, help=f"the store's root; <out>/{CACHE_DIR} if unset")
    parser.add_argument("--store-workers", type=int, default=STORE_WORKERS)
    parser.add_argument("--loader-workers", type=int, default=_DEFAULTS.loader_workers)
    parser.add_argument("--progress-seconds", type=float, default=PROGRESS_SECONDS)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    build: ModelBuilder | None = None,
    clock: Clock = time.monotonic,
) -> int:
    """Run the trainer; 0 trained, 2 refused."""
    args = _parser().parse_args(argv)
    try:
        return run(args, build or build_wavlm_verifier, clock)
    except (TrainError, ManifestError, DatasetError, ModelError, StoreError) as exc:
        print(f"train: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
