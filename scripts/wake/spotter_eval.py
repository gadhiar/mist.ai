"""The server spotter's pass bar, measured through the wake verifier over ambient audio.

The bar, set by the lead on 18 Nov 2026 and measured after the gate on Raj's go of 18 Nov: at one
cutoff c, the gate accepts a hit on at least ceil(14n/18) of the n `tune` positives of the
recording manifest (the gated wakes), and accepts at most 0.5 hits an hour on the ambient set
(the gated false wakes). A hit counts by what she does with it, not by what the spotter hears:
each runs through `eval_wake`'s offline driver of the live path, the VAD's segments, the spotter
pushed every frame as ingress pushes it, the gate's pre-decode checks, and the `--verifier`
wake verifier's score of the window `verify_window` takes around the hit, against its
threshold for the cutoff.

    python -m scripts.wake.spotter_eval --manifest <recordings> --model <id> --verifier <id> \
        --ambient <dir> --out <eval.json> [--cutoffs <list>] [--endpoint-cutoff <x>] \
        [--min-hours 8] [--models-dir <dir>]

Which cutoff, as Raj ruled on 18 Nov (A60, option A): the gated wakes are measured at every
cutoff, and c is the highest cutoff with enough of them. Every cutoff is checked, because gated
wakes need not fall as the cutoff rises: the verifier's threshold table does not increase with
the cutoff, so a higher cutoff can read a lower threshold. The ambient set is gated at c alone,
and the bar passes when c's gated rate is at most 0.5 an hour. Its failure mode: when no cutoff
has enough gated wakes, or c's gated rate is above 0.5, no cutoff passes, and a lower cutoff
with enough wakes that would pass is never run. `--cutoffs` narrows the grid for both
measurements, so `--cutoffs <x>` gates x alone when x has enough gated wakes. A cutoff not gated
prints "not run".

The gate runs as the live session runs it with nothing playing and nothing said by her: with no
reference score and no recent text of hers, the self-media and self-speech checks pass every hit,
as they do live in a room where she is idle and the PC plays nothing, and the verifier scores
each hit's window. Every frame of an ambient file is pushed once and in order to one stream at
c, as ingress pushes it, and every hit the stream reports, in speech or in silence, is gated.
The gated count counts accepted hits, the count `eval_wake` reports. It is not the count of
turns she would open: live, a hit is dropped while she is held, while another candidate waits or
within a published utterance, which makes the count high; and live a lost frame or a mute can
shorten a hit's window, which this never sees.

`--endpoint-cutoff x` (H11 passes the device's 0.847) adds one line beside the bar: the server
spotter's figures at x, spotter-only and gated, over the tune positives and the ambient set. It
is a proxy for the endpoint path, whose device spotter fires at x and whose hits the verifier
judges: here the server spotter's hits at x are judged, on the audio the server hears. The
ambient set is gated at x through the same `AmbientGate` as at c, so a window scored at c is
not scored again. It changes neither the evaluation nor the bar.

The spotter-only figures are T63's, printed beside the gated ones. A positive wakes at cutoff c
when its `score_window`, the highest score over the recording from a fresh warmed context, is at
least c: for a c above 0, exactly when a fresh stream pushed the same audio at threshold c reports
a hit. Every frame of each ambient file is scored as one stream: one `score_series` per file, from
a fresh warmed context at the file's start, then `hit_steps` at each cutoff with the model's own
`cooldown_steps`, the rule the stream applies. Ingress pushes its stream every frame of a session
as one stream too, and reports every hit it finds (`backend/realtime/ingress.py`). So on the
same audio from the same start, with no reset or backlog drop in between, the hits ingress
reports equal the hits counted here, and the spotter-only rate is the shipped one. Every server
hit that passes the self-speech check ducks her playback while she speaks and costs a verify
call, so the spotter-only rate at c bounds those ducks and verify calls an hour from ambient
sound; the gated bar does not limit them.

The output is the converter's evaluation, spotter-only and from the one scoring pass, so it keeps
that equality: exactly `cutoffs`, `false_accepts_per_hour` and `positive_scores`, which
`device_calibration` turns into `calibration.server`. The converter refuses false-accept rates
that rise with the cutoff, and so does this, writing nothing.

Only `tune` positives are scored. `load_manifest` checks every entry's file, reading it for its
format and its sha256, so `train` and `test` audio is read there, but no `train` or `test` entry
is scored. A model whose `training` is not the sha256 of this manifest's `train` subset is
refused, as `eval_wake` refuses it, and so is a verifier `eval_wake.checked_verifier` refuses.
Before anything is scored, every `tune` and `test` entry's sha256, and every ambient file's,
read whole for it, is checked against the files the verifier's provenance lists: a recording
that trained the verifier is refused, by count, never by name.

What it prints is counts, rates, hours and timings, never a file name or a transcript: what the
provenance lists and what was checked against it, the ambient hours, the scoring's wall time
and its real-time factor (the cost, per session, of ingress feeding the spotter every frame), a
gate line for the tune positives, the ambient VAD pass's wall time and a gate line for the gated
cutoff, each gate line with its verifies, failed verifies, short windows and wall time; the
endpoint proxy line when asked for; the verifier's windows with their p50 and p90 times, and the
accept estimate, `after_ms` plus those times; then the gate checks (`check_lines`) at the bar's
cutoff when it passes, else at the highest cutoff with enough spotter-only wakes: each tune
positive's first hit by its gate check, overall and for the name said alone or in a sentence,
and where it landed against the VAD's segments; and, when the ambient set was gated, its
reported hits split into those from a segment's pre-roll to 500 ms past its speech and the
rest. Then each cutoff's wakes and false wakes, spotter-only and gated, and
`bar: pass at <c>` or `bar: no cutoff passes`. Each progress line is flushed as its pass ends. A
log line names an ambient file only as "ambient file <i>", its place in name order. Exit 0 when
the evaluation is written and the bar passes, 1 when it is written and no cutoff passes, 2 when
the run is refused and nothing is written (an existing `--out`, or a missing, bad or absent
`--verifier`, included). A refusal of an ambient WAV's format names the file.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import statistics
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
    EVALUATED_SPLITS,
    SPLIT_POST_ROLL_MS,
    AmbientGate,
    EngineLoader,
    Engines,
    GateChecks,
    Provenance,
    RefusedError,
    VerifyCounts,
    accepted_wakes,
    checked_verifier,
    gate_checks,
    real_engines,
    verifier_id,
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
    wakes of the cutoffs gated on the ambient set, None for the rest. `gate_wall_s` covers every
    gated pass. `decodes` and `decode_errors` count the verifier's windows scored and its failed
    scores over the bar's gated passes and the endpoint proxy's; they keep the names the
    converter's test builds an evaluation with.
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


def checks_cutoff(
    cutoffs: Sequence[float],
    gated_wakes: Sequence[int],
    gated_rates: Sequence[float | None],
    scores: Sequence[float],
) -> float | None:
    """The cutoff the gate checks are printed at, else None.

    The bar's cutoff when it passes; else the highest cutoff whose spotter-only wakes reach
    ceil(14n/18), the cutoff a spotter good enough for the bar would run at, so the checks say
    what the gate does there.
    """
    passed = passing_cutoff(cutoffs, gated_wakes, gated_rates, len(scores))
    if passed is not None:
        return passed
    needed = required_wakes(len(scores))
    return max((cutoff for cutoff in cutoffs if wakes_at(cutoff, scores) >= needed), default=None)


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{key} {value}" for key, value in sorted(counts.items())) or "none"


def check_lines(cutoff: float, checks: GateChecks) -> list[str]:
    """The gate-check lines at `cutoff`: counts and milliseconds, never a phrase or a file."""
    prefix = f"gate checks at {cutoff}:"
    alone = checks.first_hit_by_group["name_alone"]
    sentence = checks.first_hit_by_group["sentence"]
    lags = sorted(lag for place, lag in checks.places if place == "after" and lag is not None)
    after = f"after a segment {len(lags)}"
    if lags:
        median = round(float(statistics.median(lags)))
        after += f" (min {lags[0]}, median {median}, max {lags[-1]} ms)"
    inside = sum(place == "inside" for place, _ in checks.places)
    unsegmented = sum(place == "no_segment" for place, _ in checks.places)
    return [
        f"{prefix} first hit of {sum(checks.first_hit.values())} tune positives: "
        f"{_counts(checks.first_hit)}",
        f"{prefix} name alone {sum(alone.values())}: {_counts(alone)}; "
        f"in a sentence {sum(sentence.values())}: {_counts(sentence)}",
        f"{prefix} first hit inside a segment {inside}, {after}, "
        f"with no segment before it {unsegmented}",
    ]


def _gate_line(counts: VerifyCounts) -> str:
    """A gate line's verifier counts."""
    return (
        f"{counts.verifies} verifies, {counts.failures} failed, "
        f"{counts.short_windows} short windows"
    )


def _ms(value: object) -> str:
    return "none" if value is None else f"{value} ms"


def verify_lines(counts: VerifyCounts, after_ms: int) -> list[str]:
    """The verifier's windows and times over the gated passes, and the accept estimate."""
    accept = counts.accept_ms(after_ms)
    return [
        f"verify: {counts.verifies} windows, {counts.failures} failed, "
        f"{counts.short_windows} short windows, {counts.wall_s:.1f} s wall, "
        f"p50 {_ms(counts.p50_ms)}, p90 {_ms(counts.p90_ms)}",
        f"accept: p50 {_ms(accept['p50'])}, p90 {_ms(accept['p90'])}, "
        f"after_ms {after_ms} plus the measured verify time",
    ]


def provenance_line(provenance: Provenance, entries: int, ambient: int) -> str:
    """What the verifier's provenance lists and what was checked against it, by count."""
    return (
        f"provenance: {len(provenance.files)} files listed from {len(provenance.sources)} "
        f"sources, {provenance.hours:.3f} hours; {entries} tune and test entries and {ambient} "
        "ambient files checked, none listed"
    )


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


def _endpoint_cutoff(value: str | None) -> float | None:
    """The `--endpoint-cutoff`, above 0 and at most 1, or None when not given."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RefusedError(f"--endpoint-cutoff must be a number, got {value!r}") from exc
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise RefusedError(
            f"--endpoint-cutoff must be above 0 and at most 1, the scores' scale; got {value!r}"
        )
    return parsed


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


def _ambient_sha256(item: AmbientFile, number: int) -> str:
    """The sha256 of one ambient file's bytes, as the provenance lists a file."""
    try:
        with item.path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise RefusedError(f"ambient file {number} could not be read: {exc.strerror}") from exc


def _hours(files: Sequence[AmbientFile]) -> float:
    return sum(item.samples for item in files) / SAMPLE_RATE / 3600


# -- The measurement --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Scored:
    """The spotter-only figures from the one scoring pass, and the endpoint cutoff's hits."""

    false_wakes: tuple[int, ...]
    rates: tuple[float, ...]
    scores: tuple[float, ...]
    wall_s: float
    endpoint_false_wakes: int | None


def _score(
    spotter: AmbientSpotter,
    positives: Sequence[Path],
    ambient: Sequence[AmbientFile],
    cutoffs: Sequence[float],
    clock: Callable[[], float],
    endpoint: float | None = None,
) -> _Scored:
    """Score the positives and the ambient set once, and count every cutoff's false wakes.

    The wall time covers the ambient `score_series` calls only, the work ingress's stream does
    for every frame of a session, so the real-time factor is that feed's cost. The endpoint
    cutoff's hits are counted from the same series and kept out of the rates.

    Raises:
        RefusedError: The false-accept rates rise with the cutoff.
        SpeechEngineError: As the spotter's scoring.
    """
    scores = tuple(spotter.score_window(read_pcm16(path)) for path in positives)
    counted = list(cutoffs)
    if endpoint is not None and endpoint not in counted:
        counted.append(endpoint)
    hits = dict.fromkeys(counted, 0)
    wall_s = 0.0
    for item in ambient:
        pcm = read_pcm16(item.path)
        start = clock()
        series = spotter.score_series(pcm)
        wall_s += clock() - start
        for cutoff in counted:
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
        endpoint_false_wakes=None if endpoint is None else hits[endpoint],
    )


def _gate_ambient(
    gate: AmbientGate, ambient: Sequence[AmbientFile], cutoff: float, clock: Callable[[], float]
) -> tuple[int, float, float]:
    """The ambient set's accepted hits at `cutoff`, the pass's wall time and its VAD part."""
    listened = gate.listen_s
    start = clock()
    accepted = sum(gate.accepted_hits(item.path, cutoff) for item in ambient)
    return accepted, clock() - start, gate.listen_s - listened


def evaluate(
    engines: Engines,
    settings: VoiceSettings,
    positives: Sequence[Entry],
    ambient: Sequence[AmbientFile],
    cutoffs: Sequence[float],
    clock: Callable[[], float],
    progress: Callable[[str], None],
    *,
    endpoint: float | None = None,
    header: Sequence[str] = (),
) -> Evaluation:
    """Score once for the spotter-only figures, then measure the bar through the verifier.

    `engines.spotter` scores as `AmbientSpotter` too, as `MicroNameSpotter` does. The gated
    wakes come from `accepted_wakes` at every cutoff, and at `endpoint` when given; the ambient
    set is gated at the highest cutoff with enough of them, if any, and at `endpoint`, through
    one `AmbientGate`. The gate checks then run over the positives once more, at
    `checks_cutoff`; their verifies are not in the evaluation's counts. `header` is printed once
    the scoring has passed the converter's check, and `progress` takes each progress line as its
    pass ends.

    Raises:
        RefusedError: The spotter-only false-accept rates rise with the cutoff.
        SpeechEngineError: As the engines.
    """
    paths = [item.path for item in positives]
    scored = _score(engines.spotter, paths, ambient, cutoffs, clock, endpoint)
    for line in header:
        progress(line)
    hours = _hours(ambient)
    progress(f"ambient: {len(ambient)} files, {hours:.3f} hours")
    factor = scored.wall_s / (hours * 3600)
    progress(f"scoring: {scored.wall_s:.1f} s wall, real-time factor {factor:.4f}")

    gated = sorted({*cutoffs, *(() if endpoint is None else (endpoint,))})
    counts = VerifyCounts()
    start = clock()
    woken = accepted_wakes(positives, engines, settings, gated, clock, counts=counts)
    gate_wall_s = clock() - start
    progress(
        f"gate, tune positives: {len(gated)} cutoffs, {_gate_line(counts)}, "
        f"{gate_wall_s:.1f} s wall"
    )
    gated_wakes = tuple(woken[cutoff] for cutoff in cutoffs)
    false_wakes: dict[float, int | None] = dict.fromkeys(cutoffs)

    cutoff = gated_cutoff(cutoffs, gated_wakes, len(positives))
    gate = AmbientGate(engines, settings, clock)
    split: tuple[int, int] | None = None
    if cutoff is not None:
        accepted, wall_s, listen_s = _gate_ambient(gate, ambient, cutoff, clock)
        false_wakes[cutoff] = accepted
        gate_wall_s += wall_s
        progress(f"ambient VAD: {len(ambient)} files, {listen_s:.1f} s wall")
        progress(
            f"gate at {cutoff}: {accepted} false wakes, {_gate_line(gate.counts)}, "
            f"{wall_s - listen_s:.1f} s wall"
        )
        places = [gate.hit_places(item.path, cutoff) for item in ambient]
        split = (sum(near for near, _ in places), sum(rest for _, rest in places))
    if endpoint is not None and scored.endpoint_false_wakes is not None:
        accepted, wall_s, listen_s = _gate_ambient(gate, ambient, endpoint, clock)
        gate_wall_s += wall_s
        if cutoff is None:
            progress(f"ambient VAD: {len(ambient)} files, {listen_s:.1f} s wall")
        threshold = engines.verifier.threshold_at(endpoint)
        spotted = scored.endpoint_false_wakes
        count = len(positives)
        progress(
            f"endpoint proxy at {endpoint}, on the server spotter's audio: verifier threshold "
            f"{'none' if threshold is None else threshold}, "
            f"wakes {wakes_at(endpoint, scored.scores)}/{count}, "
            f"false wakes {spotted} ({spotted / hours:.3f} per hour), "
            f"gated wakes {woken[endpoint]}/{count}, "
            f"gated false wakes {accepted} ({accepted / hours:.3f} per hour)"
        )
    counts.add(gate.counts)
    for line in verify_lines(counts, engines.verifier.after_ms):
        progress(line)

    gated_rates = [None if count is None else count / hours for count in false_wakes.values()]
    at = checks_cutoff(cutoffs, gated_wakes, gated_rates, scored.scores)
    if at is None:
        progress("gate checks: not run, no cutoff has enough spotter-only wakes")
    else:
        for line in check_lines(at, gate_checks(positives, engines, settings, at, clock)):
            progress(line)
    if split is not None:
        near, rest = split
        progress(
            f"ambient split at {cutoff}: {near} hits from a segment's pre-roll to "
            f"{SPLIT_POST_ROLL_MS} ms past its speech ({near / hours:.3f} per hour), "
            f"{rest} others ({rest / hours:.3f} per hour)"
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
        decodes=counts.verifies,
        decode_errors=counts.failures,
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
        "--verifier", help="the wake verifier model id the hits are judged by (required)"
    )
    parser.add_argument(
        "--cutoffs", help="comma-separated, from 0 to 1 (default: 0.01 to 0.99 by 0.01)"
    )
    parser.add_argument(
        "--endpoint-cutoff",
        help="a cutoff above 0 and at most 1 to print the endpoint proxy line at "
        "(the device's is 0.847)",
    )
    parser.add_argument(
        "--min-hours",
        type=float,
        default=DEFAULT_MIN_HOURS,
        help="the least ambient audio to accept, in hours (default: 8)",
    )
    parser.add_argument(
        "--models-dir",
        help="the models directory for the spotter, the VAD and the verifier "
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
    verifier = verifier_id(args.verifier)
    if not _MODEL_ID.fullmatch(args.model):
        raise RefusedError(f"--model {args.model!r}: expected a model id [a-z0-9][a-z0-9-]*")
    endpoint = _endpoint_cutoff(args.endpoint_cutoff)
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
    _, provenance = checked_verifier(str(models_dir), verifier, model, training)
    positives = [item for item in recordings.split("tune") if item.label == "positive"]
    if not positives:
        raise RefusedError("the manifest has no tune positive to measure wakes on")
    ambient = _ambient(args.ambient, args.min_hours)
    # Neither a measured entry nor the ambient set may have trained the verifier.
    entries = provenance.refuse_listed(
        (item.sha256 for name in EVALUATED_SPLITS for item in recordings.split(name)),
        "tune and test entries",
    )
    files = provenance.refuse_listed(
        (_ambient_sha256(item, number) for number, item in enumerate(ambient, start=1)),
        "ambient files",
    )

    placement = dataclasses.replace(
        settings.placement, models_dir=str(models_dir), kws_model=args.model, verify_model=verifier
    )
    if load_engines is None:
        load_engines = real_engines(verifier)
    try:
        engines = load_engines(placement)
    except SpeechEngineError as exc:
        raise RefusedError(f"the engines could not be loaded: {exc}") from exc
    try:
        evaluation = evaluate(
            engines,
            settings,
            positives,
            ambient,
            cutoffs,
            clock,
            progress,
            endpoint=endpoint,
            header=(provenance_line(provenance, entries, files),),
        )
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
