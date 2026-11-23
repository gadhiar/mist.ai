"""Calibrate the verifier's threshold table on the AMI calibration share and write its manifest.

    python -m scripts.wake.verifier.calibrate --model <model.onnx> --dataset <dir> \
        --models-dir <dir> --spotter <id> --model-id <id> --threads 2 --workers <n> --out <dir>

The spotter scores every calibration- and check-share meeting of `ami/train/split.json` (the
AMI root the dataset's meta records), and `kws.hit_positions` places its hits at every grid
cutoff, as the miner places them. Each distinct hit frame is one window (`window_bounds`),
scored by the exported model in ONNX Runtime on the CPU (`--threads` intra-op threads, one
inter-op), in `--workers` processes. onnxruntime is imported only here, at run time.

The table (Design 7): at each cutoff c, `v*(c)` is the lowest threshold at which at most
`PER_HOUR` of the calibration share's hits an hour pass, never below `FLOOR`; then a running
maximum from 0.99 down makes it non-increasing in c, so a lower row is never more lenient than a
higher one. The check share is never calibrated on: each cutoff's gated rate on it is printed.

Refused, exit 2: a meeting that is AMI dev or eval, whose sha256 is not the one
`ami/train/SHA256SUMS` lists, or that a training window read; a spotter trained from another
manifest than the dataset's `training`; an `--out` that already holds a manifest.

It writes to `--out`: `model.onnx` (copied unless it is already there), `provenance.json`
copied from the dataset byte for byte, `manifest.json` with exactly the keys
`verifier.read_verifier_manifest` reads, and `calibration.json`: both shares' hours, every
scored hit (meeting, samples, cutoffs, score) for the pre-flight, and the check share's gated
counts. The printed lines are counts and rates, never a file name. Exit 0 written, 1 an engine
failed, 2 refused.
"""

import argparse
import functools
import json
import math
import multiprocessing
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend.realtime.budget import PlacementConfig
from backend.realtime.engines.kws import (
    MicroNameSpotter,
    hit_positions,
    import_micro_runtime,
    read_manifest,
)
from backend.realtime.engines.silero_vad import CPU_PROVIDER, SessionSpec, open_onnx_session
from backend.realtime.engines.verifier import (
    FRAME_MS,
    GRID,
    KIND,
    MODEL_FILE,
    PROVENANCE_FILE,
    VERSION,
    pcm_to_waveform,
)
from backend.realtime.errors import SpeechEngineError

from .dataset import (
    AFTER_MS,
    BEFORE_MS,
    FRAME_SAMPLES,
    INDEX_META,
    MIN_WINDOW_MS,
    SAMPLE_RATE,
    DatasetError,
    read_pcm16_mono,
    read_sums,
    sha256_file,
    window_bounds,
)
from .mine import Spotter, frontend_lead
from .stage import AMI_DEV, AMI_EVAL, SPLIT, SUMS, WAV_SUFFIX

PER_HOUR = 0.2
"""Calibration hits allowed through an hour: at a true 0.2, AMI dev's 9.67 h rarely fails 0.5."""
FLOOR = 0.5
SHARES = ("calibration", "check")
MANIFEST_FILE = "manifest.json"
CALIBRATION_FILE = "calibration.json"
_EPSILON = 1e-9


class CalibrateError(Exception):
    """A refusal; the command exits 2 with the message."""


# -- The table --------------------------------------------------------------------------------


class Hit(Protocol):
    """A scored hit: the grid cutoffs it is a hit at, and the verifier's score."""

    cutoffs: Sequence[float]
    score: float


@dataclass(frozen=True, slots=True)
class ScoredHit:
    """One hit window of a meeting, `[start, end)` in samples, and its score."""

    meeting: str
    start: int
    end: int
    cutoffs: tuple[float, ...]
    score: float

    def to_json(self) -> dict[str, object]:
        """The record's JSON form."""
        return {
            "meeting": self.meeting,
            "start": self.start,
            "end": self.end,
            "cutoffs": list(self.cutoffs),
            "score": self.score,
        }


def cutoff_threshold(
    scores: Sequence[float], hours: float, *, per_hour: float = PER_HOUR, floor: float = FLOOR
) -> float:
    """The lowest threshold passing at most `per_hour * hours` of `scores`, at least `floor`.

    A hit passes at a score at or above the threshold, so the threshold sits just above the
    first score that must be stopped: the next float after it, capped at 1.
    """
    allowed = math.floor(per_hour * hours + _EPSILON)
    ranked = sorted(scores, reverse=True)
    threshold = 0.0 if len(ranked) <= allowed else min(1.0, math.nextafter(ranked[allowed], 2.0))
    return max(threshold, floor)


def threshold_table(
    hits: Sequence[Hit], hours: float, *, per_hour: float = PER_HOUR, floor: float = FLOOR
) -> list[float]:
    """`v*(c)` for every grid cutoff, made non-increasing by a running maximum from 0.99 down.

    Raises:
        CalibrateError: `hours` is not positive.
    """
    if not hours > 0:
        raise CalibrateError(f"the calibration share must hold audio, got {hours} h")
    table = [
        cutoff_threshold(
            [hit.score for hit in hits if cutoff in hit.cutoffs],
            hours,
            per_hour=per_hour,
            floor=floor,
        )
        for cutoff in GRID
    ]
    for row in range(len(table) - 2, -1, -1):
        table[row] = max(table[row], table[row + 1])
    return table


def gated_counts(hits: Sequence[Hit], table: Sequence[float]) -> list[int]:
    """At each cutoff, the hits there whose score reaches that cutoff's threshold."""
    return [
        sum(1 for hit in hits if cutoff in hit.cutoffs and hit.score >= threshold)
        for cutoff, threshold in zip(GRID, table, strict=True)
    ]


# -- Engines ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Engines:
    """The spotter, the verifier's scorer and the spotter manifest's facts."""

    spotter: Spotter
    score: Callable[[bytes], float]
    warmup_samples: int
    model_id: str
    model_sha256: str
    training: str


def load_real_engines(models_dir: str, spotter_id: str, model: str, threads: int) -> Engines:
    """The installed spotter and an ONNX Runtime session of `model` on the CPU.

    Raises:
        SpeechEngineError: A model or runtime is missing or refused.
    """
    placement = PlacementConfig(models_dir=models_dir, kws_model=spotter_id)
    spotter = MicroNameSpotter.load(placement, library_loader=import_micro_runtime)
    manifest = read_manifest(
        Path(models_dir, "speech", "wake", spotter_id, "manifest.json"), spotter_id
    )
    session = open_onnx_session(
        SessionSpec(
            path=model,
            providers=(CPU_PROVIDER,),
            device_id=0,
            intra_op_num_threads=threads,
            inter_op_num_threads=1,
        )
    )

    def score(pcm: bytes) -> float:
        outputs = session.run(None, {"waveform": pcm_to_waveform(pcm)})
        return float(outputs[0].reshape(-1)[0])

    return Engines(
        spotter=spotter,
        score=score,
        warmup_samples=manifest.warmup_ms * SAMPLE_RATE // 1000,
        model_id=spotter_id,
        model_sha256=manifest.model_sha256,
        training=manifest.training,
    )


@dataclass(frozen=True, slots=True)
class Task:
    """One meeting of a share to score."""

    share: str
    meeting: str
    path: str
    sha256: str


_STATE: dict[str, object] = {}


def _init(factory: Callable[[], Engines], lead_calls: int) -> None:
    _STATE["engines"] = factory()
    _STATE["lead_calls"] = lead_calls


def _score_meeting(task: Task) -> tuple[str, int, list[ScoredHit]]:
    engines: Engines = _STATE["engines"]  # type: ignore[assignment]
    lead_calls: int = _STATE["lead_calls"]  # type: ignore[assignment]
    path = Path(task.path)
    if sha256_file(path) != task.sha256:
        raise CalibrateError(f"{task.meeting}'s sha256 is not the one ami/train/{SUMS} lists")
    pcm = read_pcm16_mono(path)
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
            lead_calls=lead_calls,
        ):
            found.setdefault(pos, []).append(cutoff)
    frames = len(pcm) // (2 * FRAME_SAMPLES)
    hits = []
    for pos in sorted(found):
        bounds = window_bounds(pos, frames)
        if bounds is None:
            continue
        start, end = bounds
        score = engines.score(pcm[2 * start : 2 * end])
        hits.append(ScoredHit(task.meeting, start, end, tuple(found[pos]), score))
    return task.share, len(pcm) // 2, hits


def _map(
    tasks: Sequence[Task],
    factory: Callable[[], Engines],
    engines: Engines,
    lead_calls: int,
    workers: int,
) -> list[tuple[str, int, list[ScoredHit]]]:
    if workers <= 1 or len(tasks) <= 1:
        _STATE["engines"] = engines
        _STATE["lead_calls"] = lead_calls
        return [_score_meeting(task) for task in tasks]
    context = multiprocessing.get_context("fork")
    with context.Pool(workers, initializer=_init, initargs=(factory, lead_calls)) as pool:
        return list(pool.imap(_score_meeting, tasks, chunksize=1))


# -- The run ----------------------------------------------------------------------------------


def _json(path: Path, what: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CalibrateError(f"{what} could not be read: {exc}") from exc
    if not isinstance(value, dict):
        raise CalibrateError(f"{what} must be a JSON object")
    return value


def _tasks(ami: Path, trained_on: frozenset[str]) -> list[Task]:
    split = _json(ami / SPLIT, f"ami/train/{SPLIT}")
    sums = read_sums(ami / SUMS)
    tasks = []
    for share in SHARES:
        try:
            meetings = sorted(split["shares"][share])
        except (KeyError, TypeError) as exc:
            raise CalibrateError(f"ami/train/{SPLIT} has no {share} share") from exc
        for meeting in meetings:
            if meeting in AMI_DEV or meeting in AMI_EVAL:
                raise CalibrateError(f"{meeting} is an AMI dev or eval meeting")
            name = f"{meeting}{WAV_SUFFIX}"
            if name not in sums:
                raise CalibrateError(f"{meeting} is not listed in ami/train/{SUMS}")
            if sums[name] in trained_on:
                raise CalibrateError(f"{meeting} of the {share} share was read in training")
            tasks.append(Task(share, meeting, str(ami / name), sums[name]))
    return tasks


def calibrate(args: argparse.Namespace, factory: Callable[[], Engines]) -> int:
    """Score both shares, write the table, the manifest and the calibration record."""
    if (args.out / MANIFEST_FILE).exists():
        raise CalibrateError("--out already holds a manifest.json; a calibration is written new")
    meta = _json(args.dataset / INDEX_META, f"dataset {INDEX_META}")
    provenance_path = args.dataset / PROVENANCE_FILE
    provenance = _json(provenance_path, f"dataset {PROVENANCE_FILE}")
    training = meta.get("training")
    if not isinstance(training, str) or provenance.get("training") != training:
        raise CalibrateError("the dataset's meta and provenance.json name different trainings")
    roots = meta.get("roots") or {}
    if not isinstance(roots, dict) or "ami" not in roots:
        raise CalibrateError(f"dataset {INDEX_META} records no AMI root")
    ami = Path(roots["ami"])
    tasks = _tasks(ami, frozenset(provenance.get("file_sha256") or ()))
    engines = factory()
    if engines.training != training:
        raise CalibrateError(
            f"spotter {engines.model_id} was trained from {engines.training}, not the "
            f"dataset's {training}"
        )
    lead = frontend_lead(engines.spotter)
    results = _map(tasks, factory, engines, lead, args.workers)
    hits: dict[str, list[ScoredHit]] = {share: [] for share in SHARES}
    samples = dict.fromkeys(SHARES, 0)
    for share, count, found in results:
        hits[share].extend(found)
        samples[share] += count
    hours = {share: samples[share] / SAMPLE_RATE / 3600 for share in SHARES}
    table = threshold_table(hits["calibration"], hours["calibration"])
    if hours["check"] <= 0:
        raise CalibrateError("the check share holds no audio")
    gated = gated_counts(hits["check"], table)
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / MODEL_FILE
    if args.model.resolve() != model_path.resolve():
        shutil.copyfile(args.model, model_path)
    shutil.copyfile(provenance_path, args.out / PROVENANCE_FILE)
    manifest = {
        "version": VERSION,
        "kind": KIND,
        "model_id": args.model_id,
        "file": MODEL_FILE,
        "sha256": sha256_file(model_path),
        "sample_rate": SAMPLE_RATE,
        "frame_ms": FRAME_MS,
        "before_ms": BEFORE_MS,
        "after_ms": AFTER_MS,
        "min_window_ms": MIN_WINDOW_MS,
        "spotter": {"model_id": engines.model_id, "model_sha256": engines.model_sha256},
        "thresholds": {
            "cutoffs": list(GRID),
            "scores": table,
            "per_hour": PER_HOUR,
            "hours": hours["calibration"],
        },
        "training": training,
        "provenance_sha256": sha256_file(args.out / PROVENANCE_FILE),
    }
    (args.out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    record = {
        "version": 1,
        "ami_root": str(ami),
        "hours": hours,
        "hits": {share: [hit.to_json() for hit in hits[share]] for share in SHARES},
        "check": {
            "gated": gated,
            "rates": [count / hours["check"] for count in gated],
        },
    }
    (args.out / CALIBRATION_FILE).write_text(json.dumps(record) + "\n", encoding="utf-8")
    for share in SHARES:
        print(f"{share}: {hours[share]:.2f} h, {len(hits[share])} hit windows")
    for cutoff, threshold, count in zip(GRID, table, gated, strict=True):
        print(
            f"cutoff {cutoff:.2f}: threshold {threshold:.4f}, check gated {count} "
            f"({count / hours['check']:.3f} an hour)"
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.verifier.calibrate", description=__doc__
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--spotter", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None, *, engines: Callable[[], Engines] | None = None) -> int:
    """Run the calibration; see the module docstring for the exit codes."""
    args = _parser().parse_args(argv)
    factory = engines or functools.partial(
        load_real_engines, str(args.models_dir), args.spotter, str(args.model), args.threads
    )
    try:
        return calibrate(args, factory)
    except (CalibrateError, DatasetError) as exc:
        print(f"calibrate: {exc}", file=sys.stderr)
        return 2
    except SpeechEngineError as exc:
        print(f"calibrate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
