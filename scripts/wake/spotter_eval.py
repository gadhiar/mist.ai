"""The server spotter's pass bar, measured on the model over a directory of ambient audio.

The bar, set by the lead on 18 Nov 2026: at one cutoff, the model wakes on at least
ceil(14n/18) of the n `tune` positives of the recording manifest, and wakes falsely at most 0.5
times an hour on the ambient set. The lowest cutoff meeting both is the pass.

    python -m scripts.wake.spotter_eval --manifest <recordings> --model <id> --ambient <dir> \
        --out <eval.json> [--cutoffs <list>] [--min-hours 8] [--models-dir <dir>]

A positive wakes at cutoff c when its `score_window`, the highest score over the recording from a
fresh warmed context, is at least c: for a c above 0, exactly when a fresh stream pushed the
same audio at threshold c reports a hit. Only `tune` positives are scored; no `train` or `test` entry's audio
is read, and a model whose `training` is not the sha256 of this manifest's `train` subset is
refused, as `eval_wake` refuses it.

Every frame of each ambient file is scored as one stream: one `score_series` per file, from a
fresh warmed context at the file's start, then `hit_steps` at each cutoff with the model's own
`cooldown_steps`, the rule the stream applies. Ingress's stream is fed every frame of a session
as one stream too, warming the frames outside speech and reporting only the hits in a speech
segment or its pre-roll (`backend/realtime/ingress.py`), and a warmed hit starts the cooldown as
a pushed one does. So on the same audio from the same start, with no reset or backlog drop in
between, the hits ingress reports are a subset of the hits counted here, and the rate measured
here bounds the shipped one. Every cutoff comes from the one scoring pass.

The output is the converter's evaluation, exactly `cutoffs`, `false_accepts_per_hour` and
`positive_scores`, which `device_calibration` turns into `calibration.server`. The converter
refuses false-accept rates that rise with the cutoff, and so does this, writing nothing.

What it prints is counts, rates, the ambient hours, the scoring's wall time and its real-time
factor (the cost, per session, of ingress feeding the spotter every frame), each cutoff's wakes
and false wakes, then `bar: pass at <c>` or `bar: no cutoff passes`; never a file name. Exit 0
when the evaluation is written and the bar passes, 1 when it is written and no cutoff passes,
2 when the run is refused and nothing is written (an existing `--out` included). A refusal of an
ambient WAV names the file.
"""

import argparse
import dataclasses
import json
import math
import os
import re
import sys
import time
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend.realtime.budget import PlacementConfig, load_voice_settings
from backend.realtime.engines.kws import (
    MicroNameSpotter,
    hit_steps,
    import_micro_runtime,
    read_manifest,
)
from backend.realtime.errors import SpeechEngineError

from .manifest import ManifestError, load_manifest, read_pcm16, subset_sha256

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
DEFAULT_CUTOFFS = tuple(round(step / 100, 2) for step in range(1, 100))
"""0.01 to 0.99 by 0.01."""
DEFAULT_MIN_HOURS = 8.0
"""The least ambient audio a run accepts unless `--min-hours` asks for less."""
MAX_FALSE_WAKES_PER_HOUR = 0.5
BAR_WAKES = (14, 18)
"""At least 14 wakes in every 18 tune positives, rounded up."""
EVAL_KEYS = ("cutoffs", "false_accepts_per_hour", "positive_scores")

_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
_OUT_EXISTS = "--out already exists; the evaluation is written new, never over one"


class RefusedError(Exception):
    """The run cannot go ahead; the message says why, and nothing is written."""


class AmbientSpotter(Protocol):
    """The part of `MicroNameSpotter` the evaluation calls, so tests can script it."""

    cooldown_steps: int

    def score_window(self, pcm: bytes) -> float:
        """The highest score over a buffer from a fresh warmed context."""
        ...

    def score_series(self, pcm: bytes) -> list[float]:
        """Every step's score over a buffer from a fresh warmed context."""
        ...


SpotterLoader = Callable[[PlacementConfig], AmbientSpotter]
"""Builds the spotter from the placement, with `kws_model` set to the model under test."""


@dataclass(frozen=True, slots=True)
class AmbientFile:
    """One checked ambient WAV: its path and its sample count."""

    path: Path
    samples: int


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One run's measurement: the converter's three lists and what the printed lines report."""

    cutoffs: tuple[float, ...]
    false_wakes: tuple[int, ...]
    false_accepts_per_hour: tuple[float, ...]
    positive_scores: tuple[float, ...]
    files: int
    hours: float
    wall_s: float

    @property
    def passing(self) -> float | None:
        """The bar's cutoff, or None when no cutoff passes."""
        return passing_cutoff(self.cutoffs, self.false_accepts_per_hour, self.positive_scores)

    def payload(self) -> dict[str, list[float]]:
        """The evaluation as the converter reads it, with exactly its three keys."""
        return {
            "cutoffs": list(self.cutoffs),
            "false_accepts_per_hour": list(self.false_accepts_per_hour),
            "positive_scores": list(self.positive_scores),
        }


# -- The bar ----------------------------------------------------------------------------------


def required_wakes(positives: int) -> int:
    """ceil(14n/18) for n positives, in integers so no rounding error can move it."""
    wakes, of = BAR_WAKES
    return -(-wakes * positives // of)


def wakes_at(cutoff: float, scores: Sequence[float]) -> int:
    """The positives that wake at `cutoff`: a score at or above it, as the stream's rule is."""
    return sum(score >= cutoff for score in scores)


def passing_cutoff(
    cutoffs: Sequence[float], rates: Sequence[float], scores: Sequence[float]
) -> float | None:
    """The lowest cutoff with enough wakes and at most 0.5 false wakes an hour, else None."""
    needed = required_wakes(len(scores))
    for cutoff, rate in sorted(zip(cutoffs, rates, strict=True)):
        if wakes_at(cutoff, scores) >= needed and rate <= MAX_FALSE_WAKES_PER_HOUR:
            return cutoff
    return None


# -- Inputs -----------------------------------------------------------------------------------


def _cutoffs(value: str | None) -> tuple[float, ...]:
    if value is None:
        return DEFAULT_CUTOFFS
    try:
        parsed = [float(part) for part in value.split(",")]
    except ValueError as exc:
        raise RefusedError(f"--cutoffs must be comma-separated numbers, got {value!r}") from exc
    outside = [item for item in parsed if not 0.0 <= item <= 1.0]
    if outside:
        raise RefusedError(f"--cutoffs must run from 0 to 1, the scores' scale; got {outside}")
    if len(set(parsed)) != len(parsed):
        raise RefusedError(f"--cutoffs lists a cutoff twice: {value!r}")
    return tuple(sorted(parsed))


def _ambient_file(path: Path) -> AmbientFile:
    """Check one ambient WAV's format, naming it in a refusal, and count its samples."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            samples = handle.getnframes()
    except (wave.Error, EOFError, OSError) as exc:
        raise RefusedError(f"ambient {path.name} is not a PCM WAV file: {exc}") from exc
    if rate != SAMPLE_RATE:
        raise RefusedError(f"ambient {path.name} has sample rate {rate} Hz, expected 16000")
    if channels != 1:
        raise RefusedError(f"ambient {path.name} has {channels} channels, expected 1 (mono)")
    if width != SAMPLE_WIDTH:
        raise RefusedError(f"ambient {path.name} has {8 * width}-bit samples, expected 16-bit")
    return AmbientFile(path=path, samples=samples)


def _ambient(directory: Path, min_hours: float) -> tuple[AmbientFile, ...]:
    """Every `.wav` directly in `directory`, in name order, each checked, with enough hours."""
    if not directory.is_dir():
        raise RefusedError("--ambient is not a directory")
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not paths:
        raise RefusedError("the ambient directory holds no .wav file")
    files = tuple(_ambient_file(path) for path in paths)
    hours = _hours(files)
    if hours <= 0:
        raise RefusedError("the ambient set holds no audio")
    if hours < min_hours:
        raise RefusedError(
            f"the ambient set holds {hours:.3f} hours, under --min-hours {min_hours:g}"
        )
    return files


def _hours(files: Sequence[AmbientFile]) -> float:
    return sum(item.samples for item in files) / SAMPLE_RATE / 3600


# -- The measurement --------------------------------------------------------------------------


def evaluate(
    spotter: AmbientSpotter,
    positives: Sequence[Path],
    ambient: Sequence[AmbientFile],
    cutoffs: Sequence[float],
    clock: Callable[[], float],
) -> Evaluation:
    """Score the positives and the ambient set once, and count every cutoff's false wakes.

    The wall time covers the ambient `score_series` calls only, the work ingress's stream does
    for every frame of a session, so the real-time factor is that feed's cost.

    Raises:
        RefusedError: The false-accept rates rise with the cutoff.
        SpeechEngineError: As the spotter's scoring.
    """
    scores = tuple(spotter.score_window(read_pcm16(path)) for path in positives)
    hits = dict.fromkeys(cutoffs, 0)
    wall_s = 0.0
    for item in ambient:
        pcm = read_pcm16(item.path)
        start = clock()
        series = spotter.score_series(pcm)
        wall_s += clock() - start
        for cutoff in cutoffs:
            hits[cutoff] += len(hit_steps(series, cutoff, spotter.cooldown_steps))
    hours = _hours(ambient)
    rates = tuple(hits[cutoff] / hours for cutoff in cutoffs)
    points = list(zip(cutoffs, rates, strict=True))
    for (low, low_rate), (high, high_rate) in zip(points, points[1:], strict=False):
        if high_rate > low_rate:
            raise RefusedError(
                f"false wakes per hour rise with the cutoff, from {low_rate} at {low} to "
                f"{high_rate} at {high}; the converter refuses such an evaluation"
            )
    return Evaluation(
        cutoffs=tuple(cutoffs),
        false_wakes=tuple(hits[cutoff] for cutoff in cutoffs),
        false_accepts_per_hour=rates,
        positive_scores=scores,
        files=len(ambient),
        hours=hours,
        wall_s=wall_s,
    )


def report_lines(evaluation: Evaluation) -> list[str]:
    """The printed summary: counts, rates, hours and timing, never a file name."""
    count = len(evaluation.positive_scores)
    audio_s = evaluation.hours * 3600
    lines = [
        f"ambient: {evaluation.files} files, {evaluation.hours:.3f} hours",
        f"scoring: {evaluation.wall_s:.1f} s wall, "
        f"real-time factor {evaluation.wall_s / audio_s:.4f}",
        f"tune positives: {count}",
    ]
    for cutoff, wakes, rate in zip(
        evaluation.cutoffs,
        evaluation.false_wakes,
        evaluation.false_accepts_per_hour,
        strict=True,
    ):
        lines.append(
            f"cutoff {cutoff}: wakes {wakes_at(cutoff, evaluation.positive_scores)}/{count}, "
            f"false wakes {wakes} ({rate:.3f} per hour)"
        )
    passed = evaluation.passing
    lines.append("bar: no cutoff passes" if passed is None else f"bar: pass at {passed}")
    return lines


# -- The command ------------------------------------------------------------------------------


def _real_spotter(placement: PlacementConfig) -> AmbientSpotter:
    return MicroNameSpotter.load(placement, library_loader=import_micro_runtime)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.spotter_eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, type=Path, help="the recording manifest")
    parser.add_argument("--model", required=True, help="the wake model id under test")
    parser.add_argument(
        "--ambient", required=True, type=Path, help="a directory of 16 kHz mono PCM16 WAVs"
    )
    parser.add_argument("--out", required=True, type=Path, help="the evaluation JSON to write")
    parser.add_argument(
        "--cutoffs", help="comma-separated, from 0 to 1 (default: 0.01 to 0.99 by 0.01)"
    )
    parser.add_argument(
        "--min-hours",
        type=float,
        default=DEFAULT_MIN_HOURS,
        help="the least ambient audio to accept, in hours (default: 8)",
    )
    parser.add_argument(
        "--models-dir", help="the models directory (default: the voice settings' models_dir)"
    )
    return parser


def _run(
    args: argparse.Namespace,
    load_spotter: SpotterLoader,
    environ: Mapping[str, str],
    clock: Callable[[], float],
) -> Evaluation:
    if args.out.exists():
        raise RefusedError(_OUT_EXISTS)
    cutoffs = _cutoffs(args.cutoffs)
    if not math.isfinite(args.min_hours) or args.min_hours < 0:
        raise RefusedError(f"--min-hours must be a finite number of at least 0: {args.min_hours}")
    if not _MODEL_ID.fullmatch(args.model):
        raise RefusedError(f"--model {args.model!r}: expected a model id [a-z0-9][a-z0-9-]*")
    try:
        settings = load_voice_settings(environ)
    except ValueError as exc:
        raise RefusedError(f"the voice settings are invalid: {exc}") from exc
    models_dir = args.models_dir or settings.placement.models_dir
    try:
        model = read_manifest(
            Path(models_dir, "speech", "wake", args.model, "manifest.json"), args.model
        )
        recordings = load_manifest(args.manifest)
    except (SpeechEngineError, ManifestError) as exc:
        raise RefusedError(str(exc)) from exc

    training = subset_sha256(recordings, "train")
    if model.training != training:
        raise RefusedError(
            f"model {model.model_id} records training {model.training}, but this manifest's "
            f"train split has sha256 {training}; it was not trained from this split"
        )
    positives = [item.path for item in recordings.split("tune") if item.label == "positive"]
    if not positives:
        raise RefusedError("the manifest has no tune positive to measure wakes on")
    ambient = _ambient(args.ambient, args.min_hours)

    placement = dataclasses.replace(
        settings.placement, models_dir=str(models_dir), kws_model=args.model
    )
    try:
        spotter = load_spotter(placement)
        evaluation = evaluate(spotter, positives, ambient, cutoffs, clock)
    except SpeechEngineError as exc:
        raise RefusedError(f"the spotter could not score: {exc}") from exc

    data = json.dumps(evaluation.payload(), indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise RefusedError(_OUT_EXISTS) from exc
    return evaluation


def main(
    argv: Sequence[str] | None = None,
    *,
    load_spotter: SpotterLoader | None = None,
    environ: Mapping[str, str] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    """Run the evaluation; 0 when the bar passes, 1 when no cutoff does, 2 when refused.

    `load_spotter` replaces the real microWakeWord spotter (tests pass a scripted one),
    `environ` the process environment the voice settings are read from, and `clock` the timer.
    """
    args = _parser().parse_args(argv)
    try:
        evaluation = _run(
            args,
            _real_spotter if load_spotter is None else load_spotter,
            os.environ if environ is None else environ,
            clock,
        )
    except RefusedError as exc:
        print(f"spotter_eval: {exc}", file=sys.stderr)
        return 2
    for line in report_lines(evaluation):
        print(line)
    return 1 if evaluation.passing is None else 0


if __name__ == "__main__":
    sys.exit(main())
