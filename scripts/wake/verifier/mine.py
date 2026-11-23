"""Mine the spotter's hits into training windows: real, synthetic and ambient.

    python -m scripts.wake.verifier.mine --manifest <train-only manifest> --root-map <from>=<to> \
        --models-dir <dir> --spotter <id> --ami <ami/train> --librispeech <dir> --musan <dir> \
        --rirs <dir> --synth <dir> --out <dir> --workers <n> [--seed 17]

The verifier is trained on the audio around the spotter's hits, because that is all it is ever
asked to judge. Each source is scored once, one `score_series` from a fresh warmed context, and
`kws.hit_positions` places its hits at every grid cutoff, as a stream pushed the source frame
by frame reports them; each distinct hit frame becomes one window, `dataset.window_bounds`
around it, carrying the cutoffs it is a hit at.

- The manifest's train positives at their hits (positive); its homophones and media at their
  hits (negative); and, for each positive, the window that ends where the VAD segment holding
  its first hit begins (negative, `pre_name`: the name is not in it). The VAD is Silero with
  ingress's `SpeechSegmentTracker`, as `eval_wake` runs it.
- Each synthetic clip (`<synth>/piper` and `<synth>/chatterbox`, listed in `clips.jsonl`),
  resampled to 16 kHz mono PCM16 (`synth.load_clip`), mixed onto an AMI mining-share or MUSAN
  noise or music background (`synth.plan_mix` from `numpy.random.default_rng([seed, i])` and
  `synth.mix`), and written to `<out>/mixed/`; windows at its hits from the clip's start to
  `AFTER_MS` past its end take the clip's label, and hits on the background alone are left to
  the background's own mining.
- Hard negatives: the hits on the AMI mining share, LibriSpeech train-clean-100 and MUSAN, and
  one random window a minute of each file (at least one), drawn from `seed` and the file's
  sha256.

Which manifest: only the train-only manifest that `python -m scripts.wake.manifest split
--split train` writes is read, so no tune or test recording is opened by this process (Design
8); a manifest with any entry outside the train split is refused, exit 2. Its sha256 is
recorded in `meta.json`, and `build.py` refuses the output unless it is the full manifest's
train `subset_sha256`. A spotter whose `training` is not that sha256 is refused too.

The frontend's lead (the calls it makes before its first feature) is measured from the loaded
spotter (`frontend_lead`), since the stream's positions depend on it and the spotter does not
report it. Every source file's header is read before the spotter loads (`check_headers`), so
a file no reader takes is refused in seconds. A multichannel room response is read as its
first channel (`dataset.read_rir`); every other source as the mean of its channels. `--workers`
processes score the sources; the output is the same for any count. The
printed lines are counts, never a file name. Exit 0 written, 1 an engine failed, 2 refused
(an existing `--out` included).
"""

import argparse
import functools
import json
import multiprocessing
import random
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np

from backend.realtime.budget import EndpointingConfig, PlacementConfig
from backend.realtime.engines.kws import (
    FRONTEND_SAMPLES,
    MicroNameSpotter,
    hit_positions,
    import_micro_runtime,
    read_manifest,
)
from backend.realtime.engines.silero_vad import (
    SileroVad,
    onnxruntime_providers,
    open_onnx_session,
)
from backend.realtime.engines.verifier import GRID
from backend.realtime.errors import SpeechEngineError
from backend.realtime.ingress import VAD_WINDOW_SAMPLES, SpeechSegmentTracker

from ..manifest import ManifestError, load_manifest, read_pcm16
from .dataset import (
    AFTER_FRAMES,
    CLIPS_FILE,
    FRAME_SAMPLES,
    MINED_FILE,
    MINED_META,
    SAMPLE_RATE,
    SYNTH_META,
    DatasetError,
    FileRef,
    WavInfo,
    Window,
    audio_format,
    is_pcm16_16k_mono,
    parse_root_map,
    read_audio,
    read_pcm16_mono,
    read_rir,
    read_wav_segment,
    sha256_file,
    to_16k,
    to_pcm16,
    wav_info,
    window_bounds,
    write_wav,
    write_windows,
)
from .stage import SPLIT, WAV_SUFFIX
from .synth import load_clip, mix, plan_mix

SYNTHESISERS = ("piper", "chatterbox")
RANDOM_EVERY_SAMPLES = 60 * SAMPLE_RATE
"""One random window a minute of each ambient file, and at least one."""
MAX_LEAD_CALLS = 64
MAX_NAMED = 5
"""The unreadable source files a refusal names; the rest are counted."""
DEFAULT_SEED = 17
_MANIFEST_KINDS = {
    "positive": ("positive", "real_positive"),
    "homophone": ("negative", "homophone"),
    "media": ("negative", "media"),
}


class MineError(Exception):
    """A refusal; the command exits 2 with the message."""


class Spotter(Protocol):
    """The part of `MicroNameSpotter` the miner calls, so tests can script it."""

    stride: int
    cooldown_steps: int

    def score_series(self, pcm: bytes) -> list[float]:
        """Every step's score over a buffer from a fresh warmed context."""
        ...

    def probabilities(self, pcm: bytes) -> list[float]:
        """Every step's probability from a fresh context with no warm-up."""
        ...


Segmenter = Callable[[bytes], Sequence[tuple[int, int]]]
"""A recording's VAD speech segments, as (first, last) frames."""


@dataclass(frozen=True, slots=True)
class Engines:
    """The spotter, the segmenter and the facts of the spotter's manifest the miner needs."""

    spotter: Spotter
    segments: Segmenter
    warmup_samples: int
    model_id: str
    model_sha256: str
    training: str


def vad_segments(vad: SileroVad, cfg: EndpointingConfig, pcm: bytes) -> list[tuple[int, int]]:
    """Speech segments of a recording, from Silero's windows through ingress's tracker."""
    frames = len(pcm) // (2 * FRAME_SAMPLES)
    samples = np.frombuffer(pcm[: frames * 2 * FRAME_SAMPLES], dtype="<i2")
    samples = samples.astype(np.float32) / 32768.0
    stream = vad.open_stream()
    tracker = SpeechSegmentTracker(cfg)
    segments = []
    for window in range(len(samples) // VAD_WINDOW_SAMPLES):
        start = window * VAD_WINDOW_SAMPLES
        speech = stream.score(samples[start : start + VAD_WINDOW_SAMPLES]) >= cfg.vad_threshold
        first = start // FRAME_SAMPLES
        last = (start + VAD_WINDOW_SAMPLES - 1) // FRAME_SAMPLES
        step = tracker.window(speech, first, last)
        if step.closed is not None:
            segments.append(step.closed)
    if tracker.open_segment is not None:
        segments.append(tracker.open_segment)
    return segments


def load_real_engines(models_dir: str, spotter_id: str) -> Engines:
    """The spotter and Silero VAD from `models_dir`, as `eval_wake.real_engines` loads them.

    Raises:
        SpeechEngineError: A model or runtime is missing or refused.
    """
    placement = PlacementConfig(models_dir=models_dir, kws_model=spotter_id)
    spotter = MicroNameSpotter.load(placement, library_loader=import_micro_runtime)
    manifest = read_manifest(
        Path(models_dir, "speech", "wake", spotter_id, "manifest.json"), spotter_id
    )
    vad = SileroVad.load(
        placement, session_factory=open_onnx_session, available_providers=onnxruntime_providers
    )
    return Engines(
        spotter=spotter,
        segments=functools.partial(vad_segments, vad, EndpointingConfig()),
        warmup_samples=manifest.warmup_ms * SAMPLE_RATE // 1000,
        model_id=spotter_id,
        model_sha256=manifest.model_sha256,
        training=manifest.training,
    )


def frontend_lead(spotter: Spotter) -> int:
    """The frontend calls before its first feature: the first silence giving a step, less one.

    A fresh context with no warm-up completes its first step after `lead + stride` calls.

    Raises:
        MineError: No step within `MAX_LEAD_CALLS` calls.
    """
    for calls in range(1, MAX_LEAD_CALLS + spotter.stride + 1):
        if spotter.probabilities(bytes(2 * FRONTEND_SAMPLES * calls)):
            return calls - spotter.stride
    raise MineError(f"the spotter completed no step in {MAX_LEAD_CALLS} frontend calls")


# -- Tasks ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordingTask:
    """A train recording of the manifest."""

    entry: int
    label: str
    session: str
    path: str
    sha256: str
    samples: int


@dataclass(frozen=True, slots=True)
class AmbientTask:
    """An ambient file: AMI mining share, LibriSpeech or MUSAN, by its path under `root`."""

    source: str
    root: str
    path: str
    group: str


@dataclass(frozen=True, slots=True)
class SynthTask:
    """Synthetic clip `index`, a path under the synth root, to mix and mine."""

    index: int
    synthesiser: str
    root: str
    path: str
    label: str
    group: str


@dataclass(frozen=True, slots=True)
class Background:
    """A background file a clip may be mixed onto."""

    ref: FileRef
    root: str


@dataclass(frozen=True, slots=True)
class Context:
    """What every worker shares: the seed, the output, the backgrounds and room responses."""

    seed: int
    lead_calls: int
    out: str
    backgrounds: tuple[Background, ...] = ()
    rirs_root: str = ""
    rirs: tuple[str, ...] = ()


_STATE: dict[str, object] = {}


def _init(factory: Callable[[], Engines], context: Context) -> None:
    _STATE["engines"] = factory()
    _STATE["context"] = context


def _hits(pcm: bytes) -> dict[int, list[float]]:
    """Each hit frame of a source with the grid cutoffs it is a hit at."""
    engines: Engines = _STATE["engines"]  # type: ignore[assignment]
    context: Context = _STATE["context"]  # type: ignore[assignment]
    spotter = engines.spotter
    series = spotter.score_series(pcm)
    found: dict[int, list[float]] = {}
    for cutoff in GRID:
        for pos in hit_positions(
            series,
            cutoff,
            stride=spotter.stride,
            cooldown_steps=spotter.cooldown_steps,
            first_pos=0,
            warmup_samples=engines.warmup_samples,
            lead_calls=context.lead_calls,
        ):
            found.setdefault(pos, []).append(cutoff)
    return found


def _windows(
    ref: FileRef,
    hits: dict[int, list[float]],
    *,
    label: str,
    kind: str,
    group: str,
    inputs: tuple[FileRef, ...] = (),
) -> list[Window]:
    frames = ref.samples // FRAME_SAMPLES
    windows = []
    for pos in sorted(hits):
        bounds = window_bounds(pos, frames)
        if bounds is not None:
            windows.append(Window(ref, *bounds, label, kind, group, tuple(hits[pos]), inputs))
    return windows


def _recording(task: RecordingTask) -> tuple[FileRef, list[Window]]:
    engines: Engines = _STATE["engines"]  # type: ignore[assignment]
    pcm = read_pcm16(Path(task.path))
    ref = FileRef("manifest", task.sha256, None, task.entry, task.samples)
    label, kind = _MANIFEST_KINDS[task.label]
    group = f"manifest:{task.session}"
    hits = _hits(pcm)
    windows = _windows(ref, hits, label=label, kind=kind, group=group)
    if task.label == "positive" and hits:
        first = min(hits)
        before = [segment for segment in engines.segments(pcm) if segment[0] <= first]
        if before:
            anchor = before[-1][0] - 1 - AFTER_FRAMES
            bounds = window_bounds(anchor, task.samples // FRAME_SAMPLES) if anchor >= 0 else None
            if bounds is not None:
                windows.append(Window(ref, *bounds, "negative", "pre_name", group, ()))
    return ref, windows


def _check_wav_ambient(path: Path) -> WavInfo:
    info = wav_info(path)
    if not is_pcm16_16k_mono(info):
        raise DatasetError(
            f"{path.name} is {info.rate} Hz, {info.channels} channels, {8 * info.width}-bit; "
            "mining reads 16 kHz mono PCM16"
        )
    return info


def _check_ambient(path: Path) -> None:
    """Refuse an ambient file whose header `_ambient_pcm` would refuse.

    Raises:
        DatasetError: It is not a 16 kHz FLAC, or a 16 kHz mono PCM16 WAV.
    """
    if path.suffix.lower() != ".flac":
        _check_wav_ambient(path)
        return
    rate, _ = audio_format(path)
    if rate != SAMPLE_RATE:
        raise DatasetError(f"{path.name} is {rate} Hz; mining reads 16 kHz sources only")


def _ambient_pcm(path: Path) -> bytes:
    if path.suffix.lower() == ".flac":
        samples, rate = read_audio(path)
        if rate != SAMPLE_RATE:
            raise DatasetError(f"{path.name} is {rate} Hz; mining reads 16 kHz sources only")
        return to_pcm16(samples).tobytes()
    _check_wav_ambient(path)
    return read_pcm16_mono(path)


def _ambient(task: AmbientTask) -> tuple[FileRef, list[Window]]:
    context: Context = _STATE["context"]  # type: ignore[assignment]
    path = Path(task.root, task.path)
    pcm = _ambient_pcm(path)
    ref = FileRef(task.source, sha256_file(path), task.path, None, len(pcm) // 2)
    windows = _windows(ref, _hits(pcm), label="negative", kind="mined_hit", group=task.group)
    frames = ref.samples // FRAME_SAMPLES
    if frames:
        # Not a security use: the seed only makes the draw reproducible.
        rng = random.Random(f"{context.seed}:{ref.sha256}")  # nosec B311
        count = max(1, ref.samples // RANDOM_EVERY_SAMPLES)
        anchors = sorted({rng.randrange(frames) for _ in range(count)})
        windows += _windows(
            ref, {pos: [] for pos in anchors}, label="negative", kind="random", group=task.group
        )
    return ref, windows


def _synthetic(task: SynthTask) -> tuple[FileRef, list[Window]]:
    context: Context = _STATE["context"]  # type: ignore[assignment]
    clip_path = Path(task.root, task.path)
    clip = load_clip(clip_path)
    clip_ref = FileRef(task.synthesiser, sha256_file(clip_path), task.path, None, len(clip))
    lengths = [item.ref.samples for item in context.backgrounds]
    plan = plan_mix(
        np.random.default_rng([context.seed, task.index]), len(clip), lengths, len(context.rirs)
    )
    background = context.backgrounds[plan.background]
    segment = read_wav_segment(
        Path(background.root, background.ref.path or ""), plan.offset, plan.total(len(clip))
    )
    inputs = [clip_ref, background.ref]
    rir = None
    if plan.rir is not None:
        rir_path = Path(context.rirs_root, context.rirs[plan.rir])
        samples, rate = read_rir(rir_path)
        rir = to_16k(samples, rate)
        inputs.append(
            FileRef("rirs", sha256_file(rir_path), context.rirs[plan.rir], None, max(1, len(rir)))
        )
    mixed = mix(clip, segment, rir, plan)
    relative = f"mixed/{task.synthesiser}/{task.index:06d}.wav"
    write_wav(Path(context.out, relative), mixed)
    ref = FileRef("mixed", sha256_file(Path(context.out, relative)), relative, None, len(mixed))
    first = plan.lead // FRAME_SAMPLES
    last = (plan.lead + len(clip)) // FRAME_SAMPLES + AFTER_FRAMES
    hits = {pos: cutoffs for pos, cutoffs in _hits(mixed.tobytes()).items() if first <= pos <= last}
    windows = _windows(
        ref, hits, label=task.label, kind="synthetic", group=task.group, inputs=tuple(inputs)
    )
    return ref, windows


def _map(
    function: Callable[[object], tuple[FileRef, list[Window]]],
    tasks: Sequence[object],
    factory: Callable[[], Engines],
    context: Context,
    workers: int,
    engines: Engines,
) -> list[tuple[FileRef, list[Window]]]:
    """Run tasks in order, in this process or in `workers` forked processes."""
    if workers <= 1 or len(tasks) <= 1:
        _STATE["engines"] = engines
        _STATE["context"] = context
        return [function(task) for task in tasks]
    pool_context = multiprocessing.get_context("fork")
    with pool_context.Pool(workers, initializer=_init, initargs=(factory, context)) as pool:
        return list(pool.imap(function, tasks, chunksize=1))


# -- Inputs -----------------------------------------------------------------------------------


def _relative(path: Path, root: Path) -> str:
    return PurePosixPath(*path.relative_to(root).parts).as_posix()


def _ami_tasks(ami: Path) -> list[AmbientTask]:
    try:
        split = json.loads((ami / SPLIT).read_text(encoding="utf-8"))
        mining = split["shares"]["mining"]
        groups = split["groups"]
    except (OSError, ValueError, KeyError) as exc:
        raise MineError(f"--ami has no staged {SPLIT}: {exc}") from exc
    group_of = {meeting: group[0] for group in groups for meeting in group}
    return [
        AmbientTask("ami", str(ami), f"{meeting}{WAV_SUFFIX}", f"ami:{group_of[meeting]}")
        for meeting in sorted(mining)
    ]


def _librispeech_tasks(root: Path) -> list[AmbientTask]:
    return [
        AmbientTask(
            "librispeech",
            str(root),
            _relative(path, root),
            f"librispeech:{path.name.split('-')[0]}",
        )
        for path in sorted(root.rglob("*.flac"))
    ]


def _musan_tasks(root: Path) -> list[AmbientTask]:
    return [
        AmbientTask("musan", str(root), _relative(path, root), f"musan:{_relative(path, root)}")
        for path in sorted(root.rglob("*.wav"))
    ]


def _rirs(root: Path) -> tuple[str, ...]:
    """OpenSLR 28's room responses: the simulated rooms and the real `*rir*` files, no noise."""
    return tuple(
        _relative(path, root)
        for path in sorted(root.rglob("*.wav"))
        if "rir" in path.name.lower() or "simulated_rirs" in path.parts
    )


def _synth_tasks(root: Path) -> tuple[list[SynthTask], dict[str, dict]]:
    tasks: list[SynthTask] = []
    metas: dict[str, dict] = {}
    for name in SYNTHESISERS:
        directory = root / name
        if not (directory / CLIPS_FILE).is_file():
            continue
        try:
            metas[name] = json.loads((directory / SYNTH_META).read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in (directory / CLIPS_FILE).read_text(encoding="utf-8").splitlines()
            ]
        except (OSError, ValueError) as exc:
            raise MineError(f"--synth {name} is not a driver's output: {exc}") from exc
        labels = [row.get("label") for row in rows]
        if labels.count("positive") != labels.count("negative") or len(labels) % 2:
            raise MineError(f"--synth {name} must hold as many positive clips as negative ones")
        for row in rows:
            group = (
                f"librispeech:{row['speaker']}"
                if name == "chatterbox"
                else f"{name}:{row['speaker']}"
            )
            tasks.append(
                SynthTask(len(tasks), name, str(root), f"{name}/{row['file']}", row["label"], group)
            )
    return tasks, metas


def check_headers(
    ambient: Sequence[AmbientTask],
    synth_tasks: Sequence[SynthTask],
    rirs_root: Path,
    rirs: Sequence[str],
) -> None:
    """Read every source file's header before the spotter loads and any file is scored.

    Each file is checked as the reader mining uses for it would take it: an ambient file as
    `_ambient_pcm` reads it, a clip as `synth.load_clip` (`read_audio`), a room response as
    `read_rir`. A file none reads fails the run in seconds, not hours into the pool.

    Raises:
        MineError: A file's header is not one its reader takes; the first few are named.
    """
    problems: list[str] = []
    checks: list[tuple[Callable[[Path], object], Path]] = [
        (_check_ambient, Path(task.root, task.path)) for task in ambient
    ]
    checks += [(audio_format, Path(task.root, task.path)) for task in synth_tasks]
    checks += [(audio_format, Path(rirs_root, rir)) for rir in rirs]
    for check, path in checks:
        try:
            check(path)
        except DatasetError as exc:
            problems.append(str(exc))
    if problems:
        more = f"; and {len(problems) - MAX_NAMED} more" if len(problems) > MAX_NAMED else ""
        raise MineError(
            f"{len(problems)} source files cannot be read: "
            + "; ".join(problems[:MAX_NAMED])
            + more
        )


# -- The run ----------------------------------------------------------------------------------


def _flatten(results: Iterable[tuple[FileRef, list[Window]]]) -> list[Window]:
    return [window for _, windows in results for window in windows]


def mine(args: argparse.Namespace, factory: Callable[[], Engines]) -> int:
    """Check the manifest and the spotter, mine every source and write the output."""
    if args.out.exists():
        raise MineError("--out already exists; the mined output is written new")
    manifest = load_manifest(args.manifest, root_map=parse_root_map(args.root_map))
    outside = [item for item in manifest.entries if item.split != "train"]
    if outside:
        raise MineError(
            f"the manifest lists {len(outside)} entries outside the train split; mine reads "
            "the train-only manifest `python -m scripts.wake.manifest split --split train` writes"
        )
    manifest_sha256 = sha256_file(args.manifest)
    recordings = [
        RecordingTask(
            item.index, item.label, item.session, str(item.path), item.sha256, item.frames
        )
        for item in manifest.entries
    ]
    ami = _ami_tasks(args.ami)
    ambient = ami + _librispeech_tasks(args.librispeech) + _musan_tasks(args.musan)
    synth_tasks, synth_metas = _synth_tasks(args.synth)
    rirs = _rirs(args.rirs)
    check_headers(ambient, synth_tasks, args.rirs, rirs)
    engines = factory()
    if engines.training != manifest_sha256:
        raise MineError(
            f"spotter {engines.model_id} was trained from {engines.training}, not this "
            f"manifest's {manifest_sha256}"
        )
    lead = frontend_lead(engines.spotter)
    args.out.mkdir(parents=True)
    context = Context(seed=args.seed, lead_calls=lead, out=str(args.out))
    first = _map(_recording, recordings, factory, context, args.workers, engines)
    mined = _map(_ambient, ambient, factory, context, args.workers, engines)
    backgrounds = tuple(
        Background(ref, task.root)
        for (ref, _), task in zip(mined, ambient)
        if ref.source == "ami"
        or (ref.source == "musan" and PurePosixPath(task.path).parts[0] in ("noise", "music"))
    )
    if synth_tasks and not backgrounds:
        raise MineError("no AMI mining-share or MUSAN noise or music file to mix clips onto")
    mix_context = Context(args.seed, lead, str(args.out), backgrounds, str(args.rirs), rirs)
    synthetic = _map(_synthetic, synth_tasks, factory, mix_context, args.workers, engines)
    windows = _flatten(first) + _flatten(mined) + _flatten(synthetic)
    write_windows(args.out / MINED_FILE, windows)
    meta = {
        "version": 1,
        "manifest_sha256": manifest_sha256,
        "spotter": {"model_id": engines.model_id, "model_sha256": engines.model_sha256},
        "seed": args.seed,
        "lead_calls": lead,
        "warmup_samples": engines.warmup_samples,
        "ami_split_sha256": sha256_file(args.ami / SPLIT),
        "roots": {
            "ami": str(args.ami),
            "librispeech": str(args.librispeech),
            "musan": str(args.musan),
            "rirs": str(args.rirs),
            "piper": str(args.synth),
            "chatterbox": str(args.synth),
            "mixed": str(args.out),
        },
        "synth": synth_metas,
    }
    (args.out / MINED_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    counts = Counter((window.file.source, window.label) for window in windows)
    for (source, label), count in sorted(counts.items()):
        print(f"{source} {label}: {count}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.verifier.mine", description=__doc__
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root-map", action="append", default=[])
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--spotter", required=True)
    parser.add_argument("--ami", required=True, type=Path)
    parser.add_argument("--librispeech", required=True, type=Path)
    parser.add_argument("--musan", required=True, type=Path)
    parser.add_argument("--rirs", required=True, type=Path)
    parser.add_argument("--synth", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None, *, engines: Callable[[], Engines] | None = None) -> int:
    """Run the miner; see the module docstring for the exit codes."""
    args = _parser().parse_args(argv)
    factory = engines or functools.partial(load_real_engines, str(args.models_dir), args.spotter)
    try:
        return mine(args, factory)
    except (MineError, ManifestError, DatasetError) as exc:
        print(f"mine: {exc}", file=sys.stderr)
        return 2
    except SpeechEngineError as exc:
        print(f"mine: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
