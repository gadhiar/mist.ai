"""The wake model evaluation harness: the spotter and the wake verifier, offline, over one split.

A wake model is judged end to end, by what reaches a verdict, not by its raw scores: a name hit
the gate rejects costs a duck of her playback and a verify call, and a hit it accepts on a film
is a false wake. So each recording runs through an offline driver of the live path: the VAD
opens speech segments and the spotter is pushed every frame as ingress pushes it
(`backend/realtime/ingress.py`); every hit passes the gate's pre-decode checks, with the PC's own
output scored when the entry has a reference; and the acoustic wake verifier scores the window
`verify_window` takes around the hit, which `AddresseeGate.verified` reads against the
verifier's threshold for the spotter's cutoff.

    python -m scripts.wake.eval_wake --manifest <path> --model <id> --verifier <id> \
        --split tune|test --thresholds 0.3,0.4,0.5 --out <json> [--models-dir <dir>]

The tuning pass (`--split tune`) sweeps the thresholds and writes, besides the rates, each
threshold's measured `hit_lag_ms` and the `calibration` tables in the model manifest's format.
The test pass (`--split test`) takes exactly one threshold, reads `hit_lag_ms` and the
calibration from the model manifest, and writes no tables, so nothing is ever chosen on the test
split. `train` is never evaluated. A model whose `training` hash is not the sha256 of this
manifest's own `train` subset is refused, because its scores on `tune` or `test` would not be
scores on unseen recordings.

The verifier is refused on the same ground, before any audio is scored (`checked_verifier`): its
manifest's `training` and its `provenance.json`'s must both be that sha256, the provenance must
be the file the manifest pins, and the verifier must name the model under test and its model
sha256 as its spotter, since its threshold table was calibrated on that model's hits. No `tune`
or `test` entry's sha256 may be among the files its provenance lists, whichever split runs. A
refusal names no recording.

The report holds file ids, labels, counts, rates and milliseconds: besides the rates, the
verifier's id, its windows scored, failed and too short with their times (`verify`), the accept
estimate (`accept_ms`), and what its provenance lists (`provenance`). No transcript is made. A
log line names an entry by its place in the manifest, never by its file.

`spotter_eval` measures the server bar through the same driver: `accepted_wakes` counts the
recordings the gate wakes on at each threshold, `gate_checks` what the gate made of each first
hit at one, and `AmbientGate` the hits it accepts in ambient audio that is not in the manifest.
The engines come from `real_engines`, as here.

Timings follow the live path rather than the offline wall clock. Audio arrives at real time on
the uplink, so a hit cannot be verified before the verifier's `after_ms` of audio after it has
arrived, or before the pre-decode checks finish if they take longer; the measured verify time
follows. The screen moves at acceptance, so `wake_screen_ms` is each woken recording's hit lag
plus its `wake_accept_ms` value.
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from backend.realtime.addressee import AddresseeGate, GateVerdict
from backend.realtime.budget import (
    EndpointingConfig,
    PlacementConfig,
    VoiceSettings,
    load_voice_settings,
)
from backend.realtime.engines.kws import (
    MicroNameSpotter,
    WakeManifest,
    import_micro_runtime,
    read_manifest,
)
from backend.realtime.engines.silero_vad import (
    SileroVad,
    onnxruntime_providers,
    open_onnx_session,
)
from backend.realtime.engines.verifier import (
    PROVENANCE_FILE,
    OnnxWakeVerifier,
    VerifierManifest,
    read_verifier_manifest,
    verify_window,
)
from backend.realtime.errors import SpeechEngineError
from backend.realtime.ingress import (
    VAD_WINDOW_SAMPLES,
    SpeechSegmentTracker,
    onset_lead_frames,
)
from backend.realtime.speech import (
    NameHit,
    NameSpotter,
    VoiceActivityDetector,
    WakeVerifier,
)

from .manifest import (
    LABELS,
    Entry,
    ManifestError,
    load_manifest,
    phrase_key,
    read_pcm16,
    subset_sha256,
)
from .record import NAME_ALONE

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
EVALUATED_SPLITS = ("tune", "test")
SPLIT_POST_ROLL_MS = 500
"""How far past a segment's last speech frame `AmbientGate.hit_places` counts a hit as near it.

It is the longest post-roll weighed for a spotter that reports only hits near speech, and the
only one the split measures."""

_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class Engines:
    """The engines one run drives: the spotter under test, the VAD and the wake verifier."""

    spotter: NameSpotter
    vad: VoiceActivityDetector
    verifier: WakeVerifier


EngineLoader = Callable[[PlacementConfig], Engines]
"""Builds the engines from the placement, with `kws_model` and `verify_model` those under test."""


class RefusedError(Exception):
    """The run cannot go ahead; the message says why, and nothing is written."""


# -- Calibration tables -----------------------------------------------------------------------


def false_accept_table(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Score to false accepts per hour, made monotone by a running minimum from low scores up.

    Each rate comes from its own run, and a higher threshold can measure more false accepts
    than a lower one: the verifier's threshold table does not increase with the cutoff, so a
    higher cutoff can read a lower threshold, and a higher cutoff can place a hit at a later
    step, whose window can pass where the lower cutoff's did not. A table the arbitration reads
    must not rank a higher score as more likely false, so each rate is capped by every rate
    below.
    """
    table: list[tuple[float, float]] = []
    lowest = math.inf
    for score, rate in sorted(points):
        lowest = min(lowest, rate)
        table.append((score, lowest))
    return table


def positive_quantile_table(
    scores: Sequence[float], buckets: Sequence[float]
) -> list[tuple[float, float]]:
    """Score to the share of positive hits scoring at or below it, which rises with the score."""
    count = len(scores)
    return [
        (bucket, sum(score <= bucket for score in scores) / count) for bucket in sorted(buckets)
    ]


# -- The offline driver -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Heard:
    """One recording as ingress would hear it: its frames and the VAD's segments.

    A segment is `(onset frame, last speech frame)`, inclusive. The spotter is pushed every
    frame, so the frames are all it needs.
    """

    frames: tuple[bytes, ...]
    segments: tuple[tuple[int, int], ...]


def _listen(pcm: bytes, vad: VoiceActivityDetector, cfg: EndpointingConfig) -> _Heard:
    """Segment one recording as ingress does, to place its hits against the speech.

    Each VAD window at or above `vad_threshold` is speech, and ingress's own
    `SpeechSegmentTracker` turns the windows into segments. The segments decide no frame the
    spotter hears: ingress pushes it every frame once and in order as each is routed, and so
    does `_Driver._spot`, from the first frame of the recording to its last. They place each
    hit against the speech: the hit lag (`_hit_lag`), a first hit's place (`_place`) and the
    ambient split (`AmbientGate.hit_places`).
    """
    frames = _frames(pcm)
    usable = len(frames) * FRAME_BYTES
    samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
    stream = vad.open_stream()
    tracker = SpeechSegmentTracker(cfg)
    segments: list[tuple[int, int]] = []
    window = 0

    for pos in range(len(frames)):
        while (window + 1) * VAD_WINDOW_SAMPLES <= (pos + 1) * FRAME_SAMPLES:
            start = window * VAD_WINDOW_SAMPLES
            first = start // FRAME_SAMPLES
            last = (start + VAD_WINDOW_SAMPLES - 1) // FRAME_SAMPLES
            window += 1
            speech = stream.score(samples[start : start + VAD_WINDOW_SAMPLES]) >= cfg.vad_threshold
            step = tracker.window(speech, first, last)
            if step.closed is not None:
                segments.append(step.closed)
    if tracker.open_segment is not None:
        segments.append(tracker.open_segment)
    return _Heard(frames=frames, segments=tuple(segments))


def _frames(pcm: bytes) -> tuple[bytes, ...]:
    """A recording's whole 20 ms frames; a partial last frame is never routed."""
    usable = len(pcm) // FRAME_BYTES * FRAME_BYTES
    return tuple(pcm[start : start + FRAME_BYTES] for start in range(0, usable, FRAME_BYTES))


@dataclass(frozen=True, slots=True)
class _HitResult:
    """One name hit's verdict and timings; `position` is the frame within the recording.

    `verify_score` is the verifier's score of the hit's window, None when the gate rejected the
    hit before verifying it, the window was too short or the score failed.
    """

    position: int
    score: float
    verdict: GateVerdict
    accept_ms: float
    verify_score: float | None


@dataclass(frozen=True, slots=True)
class _Outcome:
    """One recording at one threshold."""

    entry: Entry
    heard: _Heard
    hits: tuple[_HitResult, ...]

    @property
    def accepted(self) -> tuple[_HitResult, ...]:
        return tuple(item for item in self.hits if item.verdict.accepted)

    @property
    def reason(self) -> str:
        """Why the recording opened no turn: no hit, or the first hit's gate check."""
        return self.hits[0].verdict.check if self.hits else "no_hit"


@dataclass(frozen=True, slots=True)
class _Verified:
    """One window's verifier score, None when it was too short or failed, and its time in ms."""

    score: float | None
    ms: float


@dataclass(slots=True)
class VerifyCounts:
    """The verifier's work over a run, which `accepted_wakes` adds to for a caller's report.

    `verifies` counts every window scored, a failed one included, and `failures` the scores that
    raised, each rejecting its hit; `short_windows` counts the hits whose window was under the
    verifier's `min_window_ms`, rejected and never scored. `times_ms` holds each score's
    measured time, failed ones included.
    """

    verifies: int = 0
    failures: int = 0
    short_windows: int = 0
    times_ms: list[float] = field(default_factory=list)

    def add(self, other: "VerifyCounts") -> None:
        """Add another run's counts and times to these."""
        self.verifies += other.verifies
        self.failures += other.failures
        self.short_windows += other.short_windows
        self.times_ms.extend(other.times_ms)

    @property
    def wall_s(self) -> float:
        """The scores' summed time in seconds."""
        return sum(self.times_ms) / 1000

    @property
    def p50_ms(self) -> float | None:
        """The median score time, or None with no window scored."""
        return _percentile(self.times_ms, 50)

    @property
    def p90_ms(self) -> float | None:
        """The 90th percentile score time, or None with no window scored."""
        return _percentile(self.times_ms, 90)

    def summary(self) -> dict[str, object]:
        """The counts and times as the report holds them."""
        return {
            "count": self.verifies,
            "failures": self.failures,
            "short_windows": self.short_windows,
            "wall_s": round(self.wall_s, 3),
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
        }

    def accept_ms(self, after_ms: int) -> dict[str, object]:
        """The accept estimate: `after_ms` of audio after the hit, then the measured verify.

        A hit's verdict cannot come before the verifier's `after_ms` has arrived at real time,
        so the percentiles are the verify times' plus `after_ms`; None with no window scored.
        """
        return {
            "p50": None if self.p50_ms is None else round(after_ms + self.p50_ms, 1),
            "p90": None if self.p90_ms is None else round(after_ms + self.p90_ms, 1),
            "after_ms": after_ms,
        }


class _Driver:
    """Runs recordings through stage 1 and the gate at every threshold of the run.

    A recording is named in log lines by a source label: a manifest entry by its index, an
    ambient file by its number (`AmbientGate`). `counts` gathers the verifier's work.
    """

    def __init__(
        self,
        engines: Engines,
        settings: VoiceSettings,
        thresholds: Sequence[float],
        clock: Callable[[], float],
    ) -> None:
        self._engines = engines
        self._settings = settings
        self._endpointing = settings.endpointing
        self._wake = settings.wake
        self._clock = clock
        self._lead = onset_lead_frames(settings.endpointing)
        self._thresholds = tuple(thresholds)
        self._gates: dict[float, AddresseeGate] = {}
        self.counts = VerifyCounts()

    def gate(self, threshold: float) -> AddresseeGate:
        """The gate at `threshold`, built once, reading the verifier's threshold table.

        The live gate reads the stream's threshold for the self-media check, and the threshold
        under test is the one the daily instance would run with.
        """
        if threshold not in self._gates:
            self._gates[threshold] = AddresseeGate(
                dataclasses.replace(self._wake, kws_threshold=threshold),
                echo_guard_ms=self._settings.barge_in.echo_guard_ms,
                thresholds=self._engines.verifier.threshold_at,
            )
        return self._gates[threshold]

    def run(self, item: Entry, offset: int) -> dict[float, _Outcome]:
        """Every threshold's outcome for one recording, its frames numbered from `offset`.

        A hit's window does not depend on the threshold, so each hit position is scored once
        per recording and its score and time reused at every threshold.
        """
        heard = _listen(read_pcm16(item.path), self._engines.vad, self._endpointing)
        reference = read_pcm16(item.reference_path) if item.reference_path is not None else None
        scores: dict[int, _Verified] = {}
        source = f"entry {item.index}"
        return {
            threshold: _Outcome(
                item, heard, self._spot(source, heard, offset, threshold, reference, scores)
            )
            for threshold in self._thresholds
        }

    def _spot(
        self,
        source: str,
        heard: _Heard,
        offset: int,
        threshold: float,
        reference: bytes | None,
        scores: dict[int, _Verified],
    ) -> tuple[_HitResult, ...]:
        # Every frame once and in order, as ingress pushes them, so every hit is judged. The
        # harness never resets the stream: it has no mute and no backlog, the only causes of a
        # reset in ingress, and each recording gets a stream of its own.
        gate = self.gate(threshold)
        stream = self._engines.spotter.open_stream(threshold)
        results: list[_HitResult] = []
        for position, frame in enumerate(heard.frames):
            arrival = self._clock()
            for found in stream.push(frame, offset + position):
                results.append(
                    self._judge(
                        source, heard, found, offset, arrival, gate, threshold, reference, scores
                    )
                )
        return tuple(results)

    def _judge(
        self,
        source: str,
        heard: _Heard,
        found: NameHit,
        offset: int,
        arrival: float,
        gate: AddresseeGate,
        threshold: float,
        reference: bytes | None,
        scores: dict[int, _Verified],
    ) -> _HitResult:
        position = min(max(found.end_pos - offset, 0), len(heard.frames) - 1)
        # Check a: her own recent speech, a memory read. Offline she has said nothing, so it
        # never rejects. The screen does not move here: it moves at acceptance.
        verdict = gate.pre_decode(score=found.score, reference_score=None, recent_tts_text="")
        reference_score = None
        if verdict is None and reference is not None:
            reference_score = self._reference_score(source, reference, position)
            verdict = gate.pre_decode(
                score=found.score, reference_score=reference_score, recent_tts_text=""
            )
        wait_ms = max(float(self._engines.verifier.after_ms), (self._clock() - arrival) * 1000)
        if verdict is not None:
            return _HitResult(position, found.score, verdict, wait_ms, None)

        if position not in scores:
            scores[position] = self._verify(source, heard, position)
        verified = scores[position]
        verdict = gate.verified(
            score=found.score,
            reference_score=reference_score,
            verify_score=verified.score,
            cutoff=threshold,
        )
        return _HitResult(position, found.score, verdict, wait_ms + verified.ms, verified.score)

    def _verify(self, source: str, heard: _Heard, position: int) -> _Verified:
        """The verifier's score of the hit's window, or None, and the score's measured time.

        The window is `verify_window`'s over the recording's frames, the one training and
        ingress take: `before_ms` before the hit through `after_ms` after it, clipped to the
        recording and never padded. A window under `min_window_ms` is not scored, and a failed
        score rejects its hit, as each does live.
        """
        verifier = self._engines.verifier
        window = verify_window(
            lambda pos: 0 <= pos < len(heard.frames),
            position,
            before_frames=verifier.before_ms // FRAME_MS,
            after_frames=verifier.after_ms // FRAME_MS,
            min_frames=-(-verifier.min_window_ms // FRAME_MS),
        )
        if window is None:
            self.counts.short_windows += 1
            logger.info(
                "The window of a hit in %s is under %d ms; counted as not addressed",
                source,
                verifier.min_window_ms,
            )
            return _Verified(None, 0.0)
        pcm = b"".join(heard.frames[window.start : window.stop])
        self.counts.verifies += 1
        before = self._clock()
        score: float | None
        try:
            score = verifier.score(pcm)
        except SpeechEngineError:
            score = None
            self.counts.failures += 1
            logger.warning("Verify failed for %s; counted as not addressed", source)
        ms = (self._clock() - before) * 1000
        self.counts.times_ms.append(ms)
        return _Verified(score, ms)

    def _reference_score(self, source: str, reference: bytes, position: int) -> float | None:
        """The spotter's score over the PC's output in the window ending at the hit.

        The window is ingress's: the spotter's own window plus `reference_tolerance_ms` before
        it, one-sided, because what played after the name cannot have caused the hit.
        """
        if position >= len(reference) // FRAME_BYTES:
            return None
        window = -(-self._engines.spotter.window_ms // FRAME_MS)
        lead = self._wake.reference_tolerance_ms // FRAME_MS
        first = max(position - window - lead + 1, 0)
        try:
            return self._engines.spotter.score_window(
                reference[first * FRAME_BYTES : (position + 1) * FRAME_BYTES]
            )
        except SpeechEngineError:
            logger.warning("Reference scoring failed for %s; scored as absent", source)
            return None


def _outcomes(driver: _Driver, entries: Sequence[Entry]) -> Iterator[dict[float, _Outcome]]:
    """Each entry's outcomes at every threshold, in manifest order.

    Frames are numbered on one uplink running from the first recording to the last, so a hit's
    position names its recording.
    """
    offset = 0
    for number, item in enumerate(entries, start=1):
        outcomes = driver.run(item, offset)
        heard = next(iter(outcomes.values())).heard
        offset += len(heard.frames)
        logger.debug(
            "Entry %d of %d: %d frames, %d segments",
            number,
            len(entries),
            len(heard.frames),
            len(heard.segments),
        )
        yield outcomes


def accepted_wakes(
    entries: Sequence[Entry],
    engines: Engines,
    settings: VoiceSettings,
    thresholds: Sequence[float],
    clock: Callable[[], float],
    *,
    counts: VerifyCounts | None = None,
) -> dict[float, int]:
    """At each threshold, the entries with at least one hit the gate accepts.

    Over positives this is the count the tune and test passes report as woken, their total less
    their missed: the same driver over the same entries, numbered the same way. `counts`, when
    given, gains the run's verifier work.
    """
    driver = _Driver(engines, settings, thresholds, clock)
    woken = dict.fromkeys(thresholds, 0)
    for outcomes in _outcomes(driver, entries):
        for threshold, outcome in outcomes.items():
            woken[threshold] += bool(outcome.accepted)
    if counts is not None:
        counts.add(driver.counts)
    return woken


Place = Literal["inside", "after", "no_segment"]


@dataclass(frozen=True, slots=True)
class GateChecks:
    """What the gate made of each entry's first hit at one threshold, as counts.

    `first_hit` counts the first hit's gate check, or `no_hit`, and `first_hit_by_group` the
    same for the entries that say the name alone (`name_alone`, a phrase of `record.NAME_ALONE`)
    and the rest (`sentence`). Over the entries with a hit, `places` gives each first hit's
    place in entry order: in a segment (`inside`), after the last speech frame of the latest
    segment to open before it (`after`, with the lag in milliseconds), or before any segment
    (`no_segment`).
    """

    first_hit: Mapping[str, int]
    first_hit_by_group: Mapping[str, Mapping[str, int]]
    places: tuple[tuple[Place, int | None], ...]


def gate_checks(
    entries: Sequence[Entry],
    engines: Engines,
    settings: VoiceSettings,
    threshold: float,
    clock: Callable[[], float],
) -> GateChecks:
    """The gate checks at `threshold` of each entry's first hit, through the same driver.

    These say why a positive did not wake her and where the spotter placed the name against
    the VAD's speech.
    """
    alone = frozenset(phrase_key(phrase) for phrase in NAME_ALONE)
    driver = _Driver(engines, settings, [threshold], clock)
    first_hit: Counter[str] = Counter()
    groups: dict[str, Counter[str]] = {"name_alone": Counter(), "sentence": Counter()}
    places: list[tuple[Place, int | None]] = []
    for outcomes in _outcomes(driver, entries):
        outcome = outcomes[threshold]
        first_hit[outcome.reason] += 1
        group = "name_alone" if phrase_key(outcome.entry.phrase) in alone else "sentence"
        groups[group][outcome.reason] += 1
        if outcome.hits:
            places.append(_place(outcome.heard.segments, outcome.hits[0].position))
    return GateChecks(
        first_hit=dict(sorted(first_hit.items())),
        first_hit_by_group={name: dict(sorted(counts.items())) for name, counts in groups.items()},
        places=tuple(places),
    )


def _place(segments: Sequence[tuple[int, int]], position: int) -> tuple[Place, int | None]:
    """Where a hit landed against the VAD's segments, with its lag past the speech if after."""
    before = [segment for segment in segments if segment[0] <= position]
    if not before:
        return "no_segment", None
    last = before[-1][1]
    if position <= last:
        return "inside", None
    return "after", (position - last) * FRAME_MS


class AmbientGate:
    """Counts the hits the gate accepts in ambient audio, one file and one threshold at a time.

    Each file goes through the driver as a recording does: the VAD's segments, every frame
    pushed once and in order to a fresh stream at the threshold as ingress pushes it, and every
    hit, in speech or in silence, judged as the live gate judges it with no reference and
    nothing she said, which is the room where she is idle and the PC plays nothing.

    A file is heard by the VAD once: its segments are kept, and its audio is read again at each
    threshold, so one file's audio is held at a time. Each hit position is scored at most once
    per file, whatever the threshold, and a failed score rejects its hit as it does live.
    `counts` gathers the verifier's work across files and thresholds, and `listen_s` is the
    clock's time spent in the VAD. A log line names a file only as "ambient file <i>", its place
    in the order the files were first given. The positions of each file's reported hits at each
    threshold are kept too, for `hit_places`.
    """

    def __init__(
        self, engines: Engines, settings: VoiceSettings, clock: Callable[[], float]
    ) -> None:
        self._driver = _Driver(engines, settings, (), clock)
        self._vad = engines.vad
        self._endpointing = settings.endpointing
        self._clock = clock
        self._heard: dict[Path, tuple[str, _Heard]] = {}
        self._scores: dict[Path, dict[int, _Verified]] = {}
        self._reported: dict[tuple[Path, float], tuple[int, ...]] = {}
        self.listen_s = 0.0

    @property
    def counts(self) -> VerifyCounts:
        """The verifier's work so far, across every file and threshold."""
        return self._driver.counts

    def accepted_hits(self, path: Path, threshold: float) -> int:
        """The hits in one ambient file at `threshold` that the gate accepts."""
        pcm = read_pcm16(path)
        if path in self._heard:
            source, kept = self._heard[path]
            heard = dataclasses.replace(kept, frames=_frames(pcm))
        else:
            source = f"ambient file {len(self._heard) + 1}"
            start = self._clock()
            heard = _listen(pcm, self._vad, self._endpointing)
            self.listen_s += self._clock() - start
            self._heard[path] = (source, dataclasses.replace(heard, frames=()))
            self._scores[path] = {}
        hits = self._driver._spot(source, heard, 0, threshold, None, self._scores[path])
        self._reported[(path, threshold)] = tuple(item.position for item in hits)
        return sum(item.verdict.accepted for item in hits)

    def hit_places(self, path: Path, threshold: float) -> tuple[int, int]:
        """The file's reported hits at `threshold`, near speech and not, accepted or not.

        Near speech is from a segment's first frame less the pre-roll through
        `SPLIT_POST_ROLL_MS` past its last speech frame: the hits a spotter that reported only
        there would still report. The positions are those `accepted_hits` recorded at
        `threshold`, which runs first if it has not.
        """
        if (path, threshold) not in self._reported:
            self.accepted_hits(path, threshold)
        _, kept = self._heard[path]
        lead = self._driver._lead
        post = SPLIT_POST_ROLL_MS // FRAME_MS
        positions = self._reported[(path, threshold)]
        near = sum(
            any(first - lead <= position <= last + post for first, last in kept.segments)
            for position in positions
        )
        return near, len(positions) - near


# -- The verifier's provenance ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Provenance:
    """A verifier's checked `provenance.json`: what it was trained from.

    `files` holds the sha256 of every file a training window was read from, and `sources` each
    source's file count and hours.
    """

    verifier_id: str
    training: str
    files: frozenset[str]
    sources: Mapping[str, tuple[int, float]]

    @property
    def hours(self) -> float:
        """The hours of every source together."""
        return sum(hours for _, hours in self.sources.values())

    def refuse_listed(self, digests: Iterable[str], what: str) -> int:
        """Refuse when any of these sha256 values is listed; return how many were checked.

        The message counts the listed files and names none, so a refusal names no recording.
        """
        checked = list(digests)
        listed = sum(digest in self.files for digest in checked)
        if listed:
            raise RefusedError(
                f"{listed} of the {len(checked)} {what} appear in verifier {self.verifier_id}'s "
                "provenance; a measured recording must never train the verifier"
            )
        return len(checked)

    def report(self, checked: Mapping[str, int]) -> dict[str, object]:
        """The provenance as the report holds it, with the counts of what was checked."""
        return {
            "files": len(self.files),
            "sources": {
                name: {"files": files, "hours": hours}
                for name, (files, hours) in sorted(self.sources.items())
            },
            "checked": dict(checked),
        }


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_hours(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def read_provenance(manifest: VerifierManifest) -> Provenance:
    """Read the verifier's `provenance.json` and check it against its manifest.

    The bytes read here are hashed again, so the file checked is the one the manifest pins by
    `provenance_sha256`, and its `training` must be the manifest's.

    Raises:
        RefusedError: The file cannot be read, is not the pinned file, is not the provenance
            format, or records another training split.
    """
    where = f"verifier {manifest.model_id}'s provenance"
    try:
        data = (Path(manifest.path).parent / PROVENANCE_FILE).read_bytes()
    except OSError as exc:
        raise RefusedError(f"{where} could not be read: {exc}") from exc
    if hashlib.sha256(data).hexdigest() != manifest.provenance_sha256:
        raise RefusedError(f"{where} is not the file its manifest pins by provenance_sha256")
    try:
        raw = json.loads(data)
    except ValueError as exc:
        raise RefusedError(f"{where} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise RefusedError(f"{where} must be a version 1 object")
    training = raw.get("training")
    files = raw.get("file_sha256")
    sources = raw.get("sources")
    if not isinstance(training, str) or not _SHA256.fullmatch(training):
        raise RefusedError(f"{where} must record its training as a sha256")
    if not isinstance(files, list) or not all(
        isinstance(item, str) and _SHA256.fullmatch(item) for item in files
    ):
        raise RefusedError(f"{where} must list file_sha256 as sha256 values")
    if not isinstance(sources, dict) or not all(
        isinstance(value, dict) and _is_count(value.get("files")) and _is_hours(value.get("hours"))
        for value in sources.values()
    ):
        raise RefusedError(f"{where} must give each source its files and hours")
    if training != manifest.training:
        raise RefusedError(
            f"{where} records training {training}, but its manifest records training "
            f"{manifest.training}"
        )
    return Provenance(
        verifier_id=manifest.model_id,
        training=training,
        files=frozenset(files),
        sources={
            str(name): (int(value["files"]), float(value["hours"]))
            for name, value in sources.items()
        },
    )


def checked_verifier(
    models_dir: str, verifier_id: str, model: WakeManifest, training: str
) -> tuple[VerifierManifest, Provenance]:
    """Read the verifier's manifest and provenance, refusing one the run cannot score with.

    The verifier must have been trained from this manifest's `train` split, as the model must,
    and against the model under test, since its threshold table was calibrated on that model's
    hits.

    Raises:
        RefusedError: `read_verifier_manifest` refuses the manifest, the verifier records
            another training split or another spotter, or `read_provenance` refuses.
    """
    try:
        manifest = read_verifier_manifest(
            Path(models_dir, "speech", "wake", verifier_id, "manifest.json"), verifier_id
        )
    except SpeechEngineError as exc:
        raise RefusedError(str(exc)) from exc
    if manifest.training != training:
        raise RefusedError(
            f"verifier {verifier_id} records training {manifest.training}, but this manifest's "
            f"train split has sha256 {training}; it was not trained from this split"
        )
    if manifest.spotter_model_id != model.model_id:
        raise RefusedError(
            f"verifier {verifier_id} was trained against spotter {manifest.spotter_model_id}, "
            f"not the model under test {model.model_id}"
        )
    if manifest.spotter_model_sha256 != model.model_sha256:
        raise RefusedError(
            f"verifier {verifier_id} was trained against spotter {model.model_id} with model "
            f"sha256 {manifest.spotter_model_sha256}, but {model.model_id}'s model has sha256 "
            f"{model.model_sha256}"
        )
    return manifest, read_provenance(manifest)


def verifier_id(value: str | None) -> str:
    """The `--verifier` value, which every run needs and which must be a model id."""
    if not value:
        raise RefusedError("--verifier is required: the bar is measured through the wake verifier")
    if not _MODEL_ID.fullmatch(value):
        raise RefusedError(f"--verifier {value!r}: expected a model id [a-z0-9][a-z0-9-]*")
    return value


# -- The report -------------------------------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def _rate(rejected: int, total: int) -> dict[str, object]:
    return {
        "rate": round(rejected / total, 6) if total else None,
        "rejected": rejected,
        "total": total,
    }


def _grouped(outcomes: Sequence[_Outcome], key: Callable[[Entry], str]) -> dict[str, object]:
    groups: dict[str, list[_Outcome]] = {}
    for outcome in outcomes:
        groups.setdefault(key(outcome.entry), []).append(outcome)
    return {
        name: _rate(sum(not item.accepted for item in members), len(members))
        for name, members in sorted(groups.items())
    }


def _hit_lag(outcomes: Sequence[_Outcome], names: frozenset[str]) -> list[float]:
    """Hit position less the VAD's speech end, over positives that say a name alone.

    There the name is the last speech, so the speech end is where the name ended.
    """
    lags: list[float] = []
    for outcome in outcomes:
        entry = outcome.entry
        if entry.label != "positive" or phrase_key(entry.phrase) not in names or not outcome.hits:
            continue
        position = outcome.hits[0].position
        before = [segment for segment in outcome.heard.segments if segment[0] <= position]
        if before:
            lags.append(float((position + 1 - (before[-1][1] + 1)) * FRAME_MS))
    return lags


def _row(
    threshold: float,
    outcomes: Sequence[_Outcome],
    media_hours: float,
    lag_ms: int | None,
    measured_lags: Sequence[float] | None,
) -> dict[str, object]:
    positives = [item for item in outcomes if item.entry.label == "positive"]
    homophones = [item for item in outcomes if item.entry.label == "homophone"]
    media = [item for item in outcomes if item.entry.label == "media"]
    missed = [item for item in positives if not item.accepted]
    woken = [item.accepted[0] for item in positives if item.accepted]

    media_accepts = sum(len(item.accepted) for item in media)
    media_reasons = Counter(
        hit.verdict.check for item in media for hit in item.hits if not hit.verdict.accepted
    )
    homophone_accepted = [item for item in homophones if item.accepted]
    # The screen moves at acceptance: the hit lag, then the accept time after the hit.
    screen = [lag_ms + hit.accept_ms for hit in woken] if lag_ms is not None else []

    row: dict[str, object] = {
        "threshold": threshold,
        "false_reject": {
            "overall": _rate(len(missed), len(positives)),
            "by_condition": _grouped(positives, lambda entry: entry.condition),
            "by_phrase": _grouped(positives, lambda entry: entry.phrase),
            "files": [item.entry.file for item in missed],
            "reasons": dict(sorted(Counter(item.reason for item in missed).items())),
        },
        "media_false_accepts": {
            "count": media_accepts,
            "per_hour": media_accepts / media_hours if media_hours else None,
            "files": [item.entry.file for item in media if item.accepted],
            "reasons": dict(sorted(media_reasons.items())),
        },
        "homophone_false_accepts": {
            "accepted": len(homophone_accepted),
            "total": len(homophones),
            "share": round(len(homophone_accepted) / len(homophones), 6) if homophones else None,
            "files": [item.entry.file for item in homophone_accepted],
        },
        "wake_accept_ms": {
            "p50": _percentile([hit.accept_ms for hit in woken], 50),
            "p90": _percentile([hit.accept_ms for hit in woken], 90),
            "count": len(woken),
        },
        "wake_screen_ms": {
            "p50": _percentile(screen, 50),
            "p90": _percentile(screen, 90),
            "count": len(screen),
            "hit_lag_ms": lag_ms,
        },
    }
    if measured_lags is not None:
        row["hit_lag_ms"] = {
            "median": _lag_median(measured_lags),
            "p90": _percentile(measured_lags, 90),
            "count": len(measured_lags),
        }
    return row


def _lag_median(lags: Sequence[float]) -> int | None:
    """The median lag as the manifest holds it: whole milliseconds, at least 0."""
    return max(0, round(float(np.median(lags)))) if lags else None


def evaluate(
    entries: Sequence[Entry],
    engines: Engines,
    settings: VoiceSettings,
    model: WakeManifest,
    split: str,
    thresholds: Sequence[float],
    clock: Callable[[], float],
    *,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Run one split's recordings at each threshold and build the report.

    Frames are numbered on one uplink running from the first recording to the last, in
    manifest order, so a hit's position names its recording. `provenance` is the checked
    provenance as the report holds it (`Provenance.report`).
    """
    driver = _Driver(engines, settings, thresholds, clock)
    by_threshold: dict[float, list[_Outcome]] = {threshold: [] for threshold in thresholds}
    for outcomes in _outcomes(driver, entries):
        for threshold, outcome in outcomes.items():
            by_threshold[threshold].append(outcome)

    names = frozenset(name.lower() for name in model.names)
    media_hours = sum(item.duration_ms for item in entries if item.label == "media") / 3_600_000
    tuning = split == "tune"
    rows = []
    for threshold in thresholds:
        outcomes = by_threshold[threshold]
        if tuning:
            lags = _hit_lag(outcomes, names)
            rows.append(_row(threshold, outcomes, media_hours, _lag_median(lags), lags))
        else:
            rows.append(_row(threshold, outcomes, media_hours, model.hit_lag_ms, None))

    report: dict[str, object] = {
        "model_id": model.model_id,
        "split": split,
        "training": model.training,
        "verifier_id": engines.verifier.model_id,
        "entries": {label: sum(item.label == label for item in entries) for label in LABELS},
        "media_hours": media_hours,
        "verify": driver.counts.summary(),
        "accept_ms": driver.counts.accept_ms(engines.verifier.after_ms),
        "provenance": dict(provenance),
        "thresholds": rows,
    }
    if tuning:
        report["calibration"] = _calibration(by_threshold, media_hours)
    else:
        report["hit_lag_ms"] = model.hit_lag_ms
    return report


def _calibration(
    by_threshold: Mapping[float, Sequence[_Outcome]], media_hours: float
) -> dict[str, object]:
    """The server spotter's tables, bucketed at the swept thresholds, or none.

    Positive hits are every hit on a positive recording at the lowest threshold, the run that
    finds the most. With no media or no positive hit a table cannot be built, and the model
    manifest refuses one table without the other, so then neither is written.
    """
    lowest = min(by_threshold)
    scores = [
        hit.score
        for outcome in by_threshold[lowest]
        if outcome.entry.label == "positive"
        for hit in outcome.hits
    ]
    if not media_hours or not scores:
        logger.warning("No calibration written: the tune split has no media or no positive hit")
        return {}
    points = [
        (
            threshold,
            sum(len(item.accepted) for item in outcomes if item.entry.label == "media")
            / media_hours,
        )
        for threshold, outcomes in by_threshold.items()
    ]
    return {
        "server": {
            "false_accepts": [list(point) for point in false_accept_table(points)],
            "positive_quantile": [
                list(point) for point in positive_quantile_table(scores, list(by_threshold))
            ],
        }
    }


# -- The command ------------------------------------------------------------------------------


def _thresholds(value: str) -> list[float]:
    try:
        parsed = [float(part) for part in value.split(",")]
    except ValueError as exc:
        raise RefusedError(f"--thresholds must be comma-separated numbers, got {value!r}") from exc
    if not all(math.isfinite(item) for item in parsed):
        raise RefusedError(f"--thresholds must be finite numbers, got {value!r}")
    if len(set(parsed)) != len(parsed):
        raise RefusedError(f"--thresholds lists a threshold twice: {value!r}")
    return sorted(parsed)


def real_engines(verifier: str) -> EngineLoader:
    """The spotter, the VAD and the wake verifier `verifier` from the placement.

    The verifier loads as the daily instance loads it, on the CPU through the same onnxruntime
    session factory as the VAD, with the placement's `verify_model` set to `verifier`.
    """

    def load(placement: PlacementConfig) -> Engines:
        spotter = MicroNameSpotter.load(placement, library_loader=import_micro_runtime)
        vad = SileroVad.load(
            placement,
            session_factory=open_onnx_session,
            available_providers=onnxruntime_providers,
        )
        wake_verifier = OnnxWakeVerifier.load(
            dataclasses.replace(placement, verify_model=verifier),
            session_factory=open_onnx_session,
            available_providers=onnxruntime_providers,
        )
        return Engines(spotter=spotter, vad=vad, verifier=wake_verifier)

    return load


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.wake.eval_wake", description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="the recording manifest")
    parser.add_argument("--model", required=True, help="the wake model id under test")
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", *EVALUATED_SPLITS),
        help="tune or test; train is refused",
    )
    parser.add_argument(
        "--thresholds", required=True, help="comma-separated; exactly one for the test split"
    )
    parser.add_argument("--out", required=True, type=Path, help="the JSON report to write")
    parser.add_argument(
        "--models-dir", help="the models directory (default: the voice settings' models_dir)"
    )
    parser.add_argument(
        "--verifier", help="the wake verifier model id the hits are judged by (required)"
    )
    return parser


def _run(
    args: argparse.Namespace,
    load_engines: EngineLoader | None,
    environ: Mapping[str, str],
    clock: Callable[[], float],
) -> None:
    if args.split not in EVALUATED_SPLITS:
        raise RefusedError("the train split is never evaluated: use --split tune or --split test")
    thresholds = _thresholds(args.thresholds)
    if args.split == "test" and len(thresholds) != 1:
        raise RefusedError(
            "a test run takes exactly one threshold, the tuned one, so the test split "
            f"never selects anything; got {len(thresholds)}"
        )
    verifier = verifier_id(args.verifier)
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
    _, provenance = checked_verifier(str(models_dir), verifier, model, training)
    # Every tune and test entry, whichever split runs: neither may have trained the verifier.
    checked = {name: len(recordings.split(name)) for name in EVALUATED_SPLITS}
    provenance.refuse_listed(
        (item.sha256 for name in EVALUATED_SPLITS for item in recordings.split(name)),
        "tune and test entries",
    )
    if args.split == "test" and "server" not in model.calibration:
        raise RefusedError(
            f"model {model.model_id} has no server calibration: run and write the tuning pass "
            "before the test pass"
        )
    entries = recordings.split(args.split)
    if not entries:
        raise RefusedError(f"the manifest has no {args.split} entries")

    placement = dataclasses.replace(
        settings.placement, models_dir=str(models_dir), kws_model=args.model, verify_model=verifier
    )
    if load_engines is None:
        load_engines = real_engines(verifier)
    try:
        engines = load_engines(placement)
    except SpeechEngineError as exc:
        raise RefusedError(f"the engines could not be loaded: {exc}") from exc
    logger.info(
        "Evaluating model %s with verifier %s on %d %s entries at %d thresholds",
        model.model_id,
        verifier,
        len(entries),
        args.split,
        len(thresholds),
    )
    report = evaluate(
        entries,
        engines,
        settings,
        model,
        args.split,
        thresholds,
        clock,
        provenance=provenance.report(checked),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote the %s report to %s", args.split, args.out)


def main(
    argv: Sequence[str] | None = None,
    *,
    load_engines: EngineLoader | None = None,
    environ: Mapping[str, str] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    """Run the harness; 0 when the report is written, 2 when the run is refused.

    `load_engines` replaces the real engines (tests pass the scripted fakes), `environ` the
    process environment the voice settings are read from, and `clock` the timer.
    """
    args = _parser().parse_args(argv)
    try:
        _run(args, load_engines, os.environ if environ is None else environ, clock)
    except RefusedError as exc:
        print(f"eval_wake: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
