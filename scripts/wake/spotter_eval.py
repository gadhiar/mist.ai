"""The server spotter's pass bar, measured after the addressee gate over ambient audio.

The bar, set by the lead on 18 Nov 2026 and measured after the gate on Raj's go of 18 Nov: at one
cutoff c, the gate accepts a hit on at least ceil(14n/18) of the n `tune` positives of the
recording manifest (the gated wakes), and accepts at most 0.5 hits an hour on the ambient set
(the gated false wakes). A hit counts by what she does with it, not by what the spotter hears:
each runs through `eval_wake`'s offline driver of the live path, the VAD's segments, the spotter
pushed every frame as ingress pushes it, the gate's pre-decode checks, the gate decode with the
`--stt-factory` STT, and `decide` on its text.

    python -m scripts.wake.spotter_eval --manifest <recordings> --model <id> --ambient <dir> \
        --out <eval.json> --stt-factory <module>:<callable> [--cutoffs <list>] \
        [--min-hours 8] [--models-dir <dir>]

Which cutoff, as Raj ruled on 18 Nov (A60, option A): the gated wakes are measured at every
cutoff, and c is the highest cutoff with enough of them. Every cutoff is checked, because gated
wakes need not fall as the cutoff rises: an early hit can be decoded before the words that
address her. The ambient set is gated at c alone, and the bar passes when c's gated rate is at
most 0.5 an hour. Its failure mode: when no cutoff has enough gated wakes, or c's gated rate is
above 0.5, no cutoff passes, and a lower cutoff with enough wakes that would pass is never run.
`--cutoffs` narrows the grid for both measurements, so `--cutoffs <x>` gates x alone when x has
enough gated wakes. A cutoff not gated prints "not run".

The gate runs as the live session runs it with nothing playing and nothing said by her: with no
reference score and no recent text of hers, the self-media and self-speech checks pass every hit,
as they do live in a room where she is idle and the PC plays nothing, and `decide` reads the gate
decode of ingress's span. Every frame of an ambient file is pushed once and in order to one
stream at c, as ingress pushes it, and every hit the stream reports, in speech or in silence, is
gated. The gated count counts accepted hits, the count `eval_wake` reports. It is neither a
strict upper nor a strict lower bound on the turns she would open: live, a hit is
dropped while she is held, while another candidate waits or within a published utterance, which
makes the count high; and live she reads a finished utterance's final transcript where this
decodes ingress's span, and that text can be accepted where the span's was not.

The spotter-only figures are T63's, printed beside the gated ones. A positive wakes at cutoff c
when its `score_window`, the highest score over the recording from a fresh warmed context, is at
least c: for a c above 0, exactly when a fresh stream pushed the same audio at threshold c reports
a hit. Every frame of each ambient file is scored as one stream: one `score_series` per file, from
a fresh warmed context at the file's start, then `hit_steps` at each cutoff with the model's own
`cooldown_steps`, the rule the stream applies. Ingress pushes its stream every frame of a session
as one stream too, and reports every hit it finds (`backend/realtime/ingress.py`). So on the
same audio from the same start, with no reset or backlog drop in between, the hits ingress
reports equal the hits counted here, and the spotter-only rate is the shipped one. Every server
hit that passes the self-speech check sends `attend`, ducks her while she speaks and costs a
gate decode, so the spotter-only rate at c bounds the `attend` cues an hour from ambient sound;
the gated bar does not limit them.

The output is the converter's evaluation, spotter-only and from the one scoring pass, so it keeps
that equality: exactly `cutoffs`, `false_accepts_per_hour` and `positive_scores`, which
`device_calibration` turns into `calibration.server`. The converter refuses false-accept rates
that rise with the cutoff, and so does this, writing nothing.

Only `tune` positives are scored. `load_manifest` checks every entry's file, reading it for its
format and its sha256, so `train` and `test` audio is read there, but no `train` or `test` entry
is scored. A model whose `training` is not the sha256 of this manifest's `train` subset is
refused, as `eval_wake` refuses it.

What it prints is counts, rates, hours and timings, never a file name or a transcript: the
ambient hours, the scoring's wall time and its real-time factor (the cost, per session, of
ingress feeding the spotter every frame), a gate line for the tune positives, the ambient VAD
pass's wall time and a gate line for the gated cutoff, each gate line with its decodes, failed
decodes and wall time, then each cutoff's wakes and false wakes, spotter-only and gated, and
`bar: pass at <c>` or `bar: no cutoff passes`. Each progress line is flushed as its pass ends. A
log line names an ambient file only as "ambient file <i>", its place in name order. Exit 0 when
the evaluation is written and the bar passes, 1 when it is written and no cutoff passes, 2 when
the run is refused and nothing is written (an existing `--out`, or a missing or bad
`--stt-factory`, included). A refusal of an ambient WAV names the file.
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

from backend.realtime.budget import VoiceSettings, load_voice_settings
from backend.realtime.engines.kws import hit_steps, read_manifest
from backend.realtime.errors import SpeechEngineError

from .eval_wake import (
    AmbientGate,
    DecodeCounts,
    EngineLoader,
    Engines,
    RefusedError,
    accepted_wakes,
    real_engines,
)
from .manifest import Entry, ManifestError, load_manifest, read_pcm16, subset_sha256

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


class AmbientSpotter(Protocol):
    """The part of `MicroNameSpotter` the spotter-only scoring calls, so tests can script it."""

    cooldown_steps: int

    def score_window(self, pcm: bytes) -> float:
        """The highest score over a buffer from a fresh warmed context."""
        ...

    def score_series(self, pcm: bytes) -> list[float]:
        """Every step's score over a buffer from a fresh warmed context."""
        ...


@dataclass(frozen=True, slots=True)
class AmbientFile:
    """One checked ambient WAV: its path and its sample count."""

    path: Path
    samples: int


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One run's measurement: the converter's three lists and what the printed lines report.

    `gated_wakes` holds every cutoff's gated wakes, and `gated_false_wakes` the gated false
    wakes of the cutoffs gated on the ambient set, None for the rest. `gate_wall_s`, `decodes`
    and `decode_errors` cover both gated passes.
    """

    cutoffs: tuple[float, ...]
    false_wakes: tuple[int, ...]
    false_accepts_per_hour: tuple[float, ...]
    positive_scores: tuple[float, ...]
    files: int
    hours: float
    wall_s: float
    gated_wakes: tuple[int, ...]
    gated_false_wakes: tuple[int | None, ...]
    gate_wall_s: float
    decodes: int
    decode_errors: int

    @property
    def gated_rates(self) -> tuple[float | None, ...]:
        """Gated false wakes per hour of ambient audio, None where not run."""
        return tuple(
            None if count is None else count / self.hours for count in self.gated_false_wakes
        )

    @property
    def passing(self) -> float | None:
        """The bar's cutoff, or None when no cutoff passes."""
        return passing_cutoff(
            self.cutoffs, self.gated_wakes, self.gated_rates, len(self.positive_scores)
        )

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


def gated_cutoff(
    cutoffs: Sequence[float], gated_wakes: Sequence[int], positives: int
) -> float | None:
    """The cutoff the ambient set is gated at: the highest with enough gated wakes, else None."""
    needed = required_wakes(positives)
    return max(
        (cutoff for cutoff, wakes in zip(cutoffs, gated_wakes, strict=True) if wakes >= needed),
        default=None,
    )


def passing_cutoff(
    cutoffs: Sequence[float],
    gated_wakes: Sequence[int],
    gated_rates: Sequence[float | None],
    positives: int,
) -> float | None:
    """The cutoff the bar names, else None.

    That is the highest cutoff with enough gated wakes, when its gated rate is at most 0.5 an
    hour. A lower cutoff's rate does not count, even when it was measured.
    """
    cutoff = gated_cutoff(cutoffs, gated_wakes, positives)
    if cutoff is None:
        return None
    rate = dict(zip(cutoffs, gated_rates, strict=True))[cutoff]
    return cutoff if rate is not None and rate <= MAX_FALSE_WAKES_PER_HOUR else None


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


@dataclass(frozen=True, slots=True)
class _Scored:
    """The spotter-only figures from the one scoring pass."""

    false_wakes: tuple[int, ...]
    rates: tuple[float, ...]
    scores: tuple[float, ...]
    wall_s: float


def _score(
    spotter: AmbientSpotter,
    positives: Sequence[Path],
    ambient: Sequence[AmbientFile],
    cutoffs: Sequence[float],
    clock: Callable[[], float],
) -> _Scored:
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
    return _Scored(
        false_wakes=tuple(hits[cutoff] for cutoff in cutoffs),
        rates=rates,
        scores=scores,
        wall_s=wall_s,
    )


def evaluate(
    engines: Engines,
    settings: VoiceSettings,
    positives: Sequence[Entry],
    ambient: Sequence[AmbientFile],
    cutoffs: Sequence[float],
    clock: Callable[[], float],
    progress: Callable[[str], None],
) -> Evaluation:
    """Score once for the spotter-only figures, then measure the bar after the gate.

    `engines.spotter` scores as `AmbientSpotter` too, as `MicroNameSpotter` does. The gated
    wakes come from `accepted_wakes` at every cutoff; the ambient set is gated at the highest
    cutoff with enough of them, if any, through `AmbientGate`. `progress` takes each progress
    line as its pass ends.

    Raises:
        RefusedError: The spotter-only false-accept rates rise with the cutoff.
        SpeechEngineError: As the engines.
    """
    paths = [item.path for item in positives]
    scored = _score(engines.spotter, paths, ambient, cutoffs, clock)
    hours = _hours(ambient)
    progress(f"ambient: {len(ambient)} files, {hours:.3f} hours")
    factor = scored.wall_s / (hours * 3600)
    progress(f"scoring: {scored.wall_s:.1f} s wall, real-time factor {factor:.4f}")

    counts = DecodeCounts()
    start = clock()
    woken = accepted_wakes(positives, engines, settings, cutoffs, clock, counts=counts)
    gate_wall_s = clock() - start
    progress(
        f"gate, tune positives: {len(cutoffs)} cutoffs, {counts.decodes} decodes, "
        f"{counts.decode_errors} failed, {gate_wall_s:.1f} s wall"
    )
    gated_wakes = tuple(woken[cutoff] for cutoff in cutoffs)
    false_wakes: dict[float, int | None] = dict.fromkeys(cutoffs)

    cutoff = gated_cutoff(cutoffs, gated_wakes, len(positives))
    if cutoff is not None:
        gate = AmbientGate(engines, settings, clock)
        start = clock()
        false_wakes[cutoff] = sum(gate.accepted_hits(item.path, cutoff) for item in ambient)
        ambient_s = clock() - start
        gate_wall_s += ambient_s
        counts.decodes += gate.decodes
        counts.decode_errors += gate.decode_errors
        progress(f"ambient VAD: {len(ambient)} files, {gate.listen_s:.1f} s wall")
        progress(
            f"gate at {cutoff}: {false_wakes[cutoff]} false wakes, {gate.decodes} decodes, "
            f"{gate.decode_errors} failed, {ambient_s - gate.listen_s:.1f} s wall"
        )

    return Evaluation(
        cutoffs=tuple(cutoffs),
        false_wakes=scored.false_wakes,
        false_accepts_per_hour=scored.rates,
        positive_scores=scored.scores,
        files=len(ambient),
        hours=hours,
        wall_s=scored.wall_s,
        gated_wakes=gated_wakes,
        gated_false_wakes=tuple(false_wakes[item] for item in cutoffs),
        gate_wall_s=gate_wall_s,
        decodes=counts.decodes,
        decode_errors=counts.decode_errors,
    )


def report_lines(evaluation: Evaluation) -> list[str]:
    """The printed summary after the progress lines: counts and rates, never a file name."""
    count = len(evaluation.positive_scores)
    lines = [f"tune positives: {count}"]
    for cutoff, wakes, rate, gated, gated_false, gated_rate in zip(
        evaluation.cutoffs,
        evaluation.false_wakes,
        evaluation.false_accepts_per_hour,
        evaluation.gated_wakes,
        evaluation.gated_false_wakes,
        evaluation.gated_rates,
        strict=True,
    ):
        after_gate = (
            "not run"
            if gated_false is None or gated_rate is None
            else f"{gated_false} ({gated_rate:.3f} per hour)"
        )
        lines.append(
            f"cutoff {cutoff}: wakes {wakes_at(cutoff, evaluation.positive_scores)}/{count}, "
            f"false wakes {wakes} ({rate:.3f} per hour), gated wakes {gated}/{count}, "
            f"gated false wakes {after_gate}"
        )
    passed = evaluation.passing
    lines.append("bar: no cutoff passes" if passed is None else f"bar: pass at {passed}")
    return lines


# -- The command ------------------------------------------------------------------------------


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
        "--stt-factory", help="<module>:<callable> building the gate's STT from the placement"
    )
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
        "--models-dir",
        help="the models directory for the spotter, the VAD and the STT "
        "(default: the voice settings' models_dir)",
    )
    return parser


def _run(
    args: argparse.Namespace,
    load_engines: EngineLoader | None,
    environ: Mapping[str, str],
    clock: Callable[[], float],
    progress: Callable[[str], None],
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
    positives = [item for item in recordings.split("tune") if item.label == "positive"]
    if not positives:
        raise RefusedError("the manifest has no tune positive to measure wakes on")
    ambient = _ambient(args.ambient, args.min_hours)

    if load_engines is None:
        if not args.stt_factory:
            raise RefusedError("--stt-factory is required to load the real engines")
        load_engines = real_engines(args.stt_factory)
    placement = dataclasses.replace(
        settings.placement, models_dir=str(models_dir), kws_model=args.model
    )
    try:
        engines = load_engines(placement)
    except SpeechEngineError as exc:
        raise RefusedError(f"the engines could not be loaded: {exc}") from exc
    try:
        evaluation = evaluate(engines, settings, positives, ambient, cutoffs, clock, progress)
    except SpeechEngineError as exc:
        raise RefusedError(f"the engines could not run: {exc}") from exc

    data = json.dumps(evaluation.payload(), indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise RefusedError(_OUT_EXISTS) from exc
    return evaluation


def _print(line: str) -> None:
    print(line, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    load_engines: EngineLoader | None = None,
    environ: Mapping[str, str] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    """Run the evaluation; 0 when the bar passes, 1 when no cutoff does, 2 when refused.

    `load_engines` replaces `eval_wake.real_engines` (tests pass scripted engines), `environ`
    the process environment the voice settings are read from, and `clock` the timer.
    """
    args = _parser().parse_args(argv)
    try:
        evaluation = _run(
            args, load_engines, os.environ if environ is None else environ, clock, _print
        )
    except RefusedError as exc:
        print(f"spotter_eval: {exc}", file=sys.stderr)
        return 2
    for line in report_lines(evaluation):
        _print(line)
    return 1 if evaluation.passing is None else 0


if __name__ == "__main__":
    sys.exit(main())
