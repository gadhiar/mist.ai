"""Train the wake verifier on the dataset index, one cross-validation fold or the final model.

    python -m scripts.wake.verifier.train --dataset <dir> --manifest <train-only manifest> \
        --root-map <from>=<to> --weights <wavlm_base_plus.pth> --out <dir> \
        (--fold <k> | --final) [--batch 32] [--accumulate 1] [--seed 17]

The model is `model.build_verifier` over the staged `wavlm_base_plus.pth`, loaded strictly; the
run refuses to start if the installed torchaudio's `WAVLM_BASE_PLUS._normalize_waveform` reads
True, since the exported graph applies no normalisation.

Data. Only what `build.py` indexed is read, and only after every file a window was read or
mixed from is checked against the dataset's `provenance.json`: a row outside it is refused. A
recording is read through the train-only manifest (`--root-map` translating its root), which is
refused unless its sha256 is the index's `training` and it lists train entries alone, so no tune
or test file is opened. Real positives are repeated `REAL_REPEAT` times an epoch.

Folds. Five folds grouped by each window's `group` as `mine.py` set it (recording session, AMI
speaker group, LibriSpeech speaker, which a Chatterbox clone shares, piper speaker, MUSAN file),
balanced within each kind of group, so no group is on
both sides of a fold. `--fold k` trains on the other four and scores fold k's windows into
`oof.jsonl`; `--final` trains on all of them. Every run trains the same fixed schedule and keeps
its last epoch: nothing is chosen on out-of-fold data, which would bias the pre-flight's `r(c)`.

Augmentation, for both classes alike from a per-item seed: the truncation of Design 3 (either
side cut to a random length down to `MIN_WINDOW_FRAMES`, the hit kept), gain, white noise, a room
response from the index's own `rirs` files, a low-pass band limit, and an AGC-like compressor.
No step reads the label, so a window's length or level carries none.

Defaults (recorded in the run's `config.json`): AdamW at 2e-5 for layers 9 to 12 and 1e-3 for
the head, batch 32 (split into `--accumulate` micro-batches when VRAM is short), 6 epochs, fp16
autocast on CUDA, seed 17. A batch holds windows of one length, because the encoder's group
norm would see padding. The run writes `<out>/fold-<k>/` or `<out>/final/`: `model.pt`, the
verifier's state dict; `config.json`; `history.json`, each epoch's mean loss; and for a fold
`oof.jsonl`. Exit 0 trained, 2 refused (an existing run directory included).
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..manifest import ManifestError, RecordingManifest, load_manifest
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
    read_audio,
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

FOLDS = 5
REAL_REPEAT = 10
"""How many times a real positive window is seen an epoch, against the synthetic thousands."""
RUN_CONFIG = "config.json"
RUN_MODEL = "model.pt"
RUN_HISTORY = "history.json"
RUN_OOF = "oof.jsonl"
FINAL_RUN = "final"


class TrainError(Exception):
    """A refusal; the command exits 2 with the message."""


def fold_run(fold: int) -> str:
    """A fold's run directory name under `--out`."""
    return f"fold-{fold}"


@dataclass(frozen=True, slots=True)
class Augment:
    """The augmentation's probabilities and ranges, the same for both classes."""

    truncate_p: float = 0.5
    gain_db: tuple[float, float] = (-6.0, 6.0)
    noise_p: float = 0.5
    noise_snr_db: tuple[float, float] = (10.0, 40.0)
    rir_p: float = 0.25
    lowpass_p: float = 0.3
    lowpass_hz: tuple[float, float] = (3000.0, 7500.0)
    agc_p: float = 0.3
    agc_target_rms: float = 0.1
    agc_max_gain_db: float = 20.0


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
        samples, rate = read_audio(self.path(ref))
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


# -- Augmentation -----------------------------------------------------------------------------


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


def _agc(samples: np.ndarray, target: float, max_gain_db: float) -> np.ndarray:
    frames = max(1, len(samples) // FRAME_SAMPLES)
    level = np.sqrt(
        np.mean(samples[: frames * FRAME_SAMPLES].reshape(frames, -1) ** 2, axis=1) + 1e-10
    )
    envelope = np.maximum.accumulate(level) * 0.5 + level * 0.5
    gain = np.minimum(target / np.maximum(envelope, 1e-6), 10 ** (max_gain_db / 20))
    per_sample = np.repeat(gain, FRAME_SAMPLES)[: len(samples)]
    per_sample = np.pad(per_sample, (0, len(samples) - len(per_sample)), mode="edge")
    return samples * per_sample


def augment(
    samples: np.ndarray, rng: np.random.Generator, settings: Augment, rirs: Sequence[np.ndarray]
) -> np.ndarray:
    """Gain, noise, a room response, a band limit and AGC, each drawn from `rng`; no label read."""
    out = samples.astype(np.float64)
    out = out * 10 ** (rng.uniform(*settings.gain_db) / 20)
    if rirs and rng.random() < settings.rir_p:
        response = rirs[int(rng.integers(len(rirs)))]
        out = np.convolve(out, response)[: len(out)]
    if rng.random() < settings.lowpass_p:
        from scipy.signal import butter, sosfilt  # noqa: PLC0415 - training only

        cutoff = rng.uniform(*settings.lowpass_hz)
        out = sosfilt(butter(6, cutoff, fs=SAMPLE_RATE, output="sos"), out)
    if rng.random() < settings.noise_p:
        power = float(np.mean(out**2)) + 1e-10
        snr = rng.uniform(*settings.noise_snr_db)
        out = out + rng.normal(0.0, math.sqrt(power / 10 ** (snr / 10)), len(out))
    if rng.random() < settings.agc_p:
        out = _agc(out, settings.agc_target_rms, settings.agc_max_gain_db)
    return np.clip(out, -1.0, 32767 / 32768).astype(np.float32)


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


class EpochData(torch.utils.data.Dataset):
    """An epoch's items read and augmented; a batch sampler hands it one length at a time."""

    def __init__(
        self,
        windows: Sequence[Window],
        items: Sequence[Item],
        reader: WindowReader,
        settings: Augment,
        rirs: Sequence[FileRef],
    ) -> None:
        self.windows = windows
        self.items = items
        self.reader = reader
        self.settings = settings
        self.rir_refs = tuple(rirs)
        self._rirs: list[np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rirs is None:
            self._rirs = [self.reader.rir(ref) for ref in self.rir_refs]
        item = self.items[position]
        window = self.windows[item.window]
        samples = self.reader.samples(window)
        kept = samples[item.first * FRAME_SAMPLES : (item.last + 1) * FRAME_SAMPLES]
        audio = kept.astype(np.float32) / np.float32(32768.0)
        audio = augment(audio, np.random.default_rng(item.seed), self.settings, self._rirs)
        label = 1.0 if window.label == "positive" else 0.0
        return torch.from_numpy(audio), torch.tensor(label)


def _collate(batch: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
    return torch.stack([audio for audio, _ in batch]), torch.stack([label for _, label in batch])


# -- Training ---------------------------------------------------------------------------------


Batches = Callable[[int], Sequence[tuple[torch.Tensor, torch.Tensor]]]
"""An epoch's batches by epoch number, each `(waveforms, labels)`."""


def fit(
    model: VerifierModel,
    batches: Batches,
    config: TrainConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """Train the fixed schedule; the model is left at its last epoch, which the run keeps.

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
        total, seen = 0.0, 0
        for waveforms, labels in batches(epoch):
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
                total += float(loss) * size
            scaler.step(optimizer)
            scaler.update()
            seen += size
        history.append({"epoch": epoch, "loss": total / max(seen, 1), "examples": seen})
        print(f"epoch {epoch}: loss {total / max(seen, 1):.4f} over {seen} examples")
    model.eval()
    return history


@torch.no_grad()
def score_windows(
    model: VerifierModel,
    windows: Sequence[Window],
    reader: WindowReader,
    device: torch.device,
) -> list[float]:
    """Each whole window's score, unaugmented, as the exported graph would give it."""
    model.eval()
    model.to(device)
    scores = []
    for window in windows:
        audio = reader.samples(window).astype(np.float32) / np.float32(32768.0)
        logit = model(torch.from_numpy(audio).reshape(1, -1).to(device))
        scores.append(float(torch.sigmoid(logit.float())[0]))
    return scores


ModelBuilder = Callable[[TrainConfig, Path], VerifierModel]


def build_wavlm_verifier(config: TrainConfig, weights: Path) -> VerifierModel:
    """The verifier over the staged Base+ weights, loaded strictly."""
    return build_verifier(
        base_plus_params(), weights=weights, first_trainable=config.first_trainable_layer
    )


def run(args: argparse.Namespace, build: ModelBuilder) -> int:
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
    config = TrainConfig(batch=args.batch, accumulate=args.accumulate, seed=args.seed)
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
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = build(config, args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def batches(epoch: int) -> torch.utils.data.DataLoader:
        plan = plan_epoch(windows, chosen, config, epoch)
        flat = [item for batch in plan for item in batch]
        starts = np.cumsum([0] + [len(batch) for batch in plan])
        sampler = [list(range(starts[i], starts[i + 1])) for i in range(len(plan))]
        data = EpochData(windows, flat, reader, config.augment, rirs)
        return torch.utils.data.DataLoader(
            data,
            batch_sampler=sampler,
            num_workers=config.loader_workers if device.type == "cuda" else 0,
            collate_fn=_collate,
        )

    history = fit(model, batches, config, device)
    out.mkdir(parents=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / RUN_MODEL)
    if not args.final:
        scores = score_windows(model, [windows[i] for i in held], reader, device)
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
    return parser


def main(argv: Sequence[str] | None = None, *, build: ModelBuilder | None = None) -> int:
    """Run the trainer; 0 trained, 2 refused."""
    args = _parser().parse_args(argv)
    try:
        return run(args, build or build_wavlm_verifier)
    except (TrainError, ManifestError, DatasetError, ModelError) as exc:
        print(f"train: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
