"""The wake model evaluation harness: stage 1 and the addressee gate, offline, over one split.

A wake model is judged end to end, by what reaches a verdict, not by its raw scores: a name hit
the gate rejects costs nothing but a cue, and a hit it accepts on a film is a false wake. So
each recording runs through an offline driver of the live path: the VAD opens speech segments
and the spotter is fed every frame as ingress feeds it (`backend/realtime/ingress.py`); every hit
passes the gate's pre-decode checks, with the PC's own output scored when the entry has a
reference; the gate decode runs through the `StreamingTranscriber` contract; and `decide` reads
its text.

    python -m scripts.wake.eval_wake --manifest <path> --model <id> --split tune|test \
        --thresholds 0.3,0.4,0.5 --out <json> --stt-factory <module>:<callable>

The tuning pass (`--split tune`) sweeps the thresholds and writes, besides the rates, each
threshold's measured `hit_lag_ms` and the `calibration` tables in the model manifest's format.
The test pass (`--split test`) takes exactly one threshold, reads `hit_lag_ms` and the
calibration from the model manifest, and writes no tables, so nothing is ever chosen on the test
split. `train` is never evaluated. A model whose `training` hash is not the sha256 of this
manifest's own `train` subset is refused, because its scores on `tune` or `test` would not be
scores on unseen recordings.

The report holds file ids, labels, counts, rates and milliseconds. Transcripts stay in memory
while their recording is judged at each threshold (an ambient file's for as long as its
`AmbientGate` lives), for the `decide` calls that read them, and are never written or logged.

`spotter_eval` measures the server bar through the same driver: `accepted_wakes` counts the
recordings the gate wakes on at each threshold, and `AmbientGate` the hits it accepts in ambient
audio that is not in the manifest. The engines come from `real_engines`, as here.

Timings follow the live path rather than the offline wall clock. Audio arrives at real time on
the uplink, so a hit's verdict cannot come before `gate_tail_ms` of audio after it, or before
the pre-decode checks finish if they take longer; the measured decode and `decide` follow.
`wake_screen_ms` is the hit lag plus the measured time from the hit frame's push to the end of
the self-speech check, after which the live controller sends `attend`.
"""

import argparse
import dataclasses
import importlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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
from backend.realtime.engines.parakeet_stt import ParakeetTranscriber, import_onnx_asr
from backend.realtime.engines.silero_vad import (
    SileroVad,
    onnxruntime_providers,
    open_onnx_session,
)
from backend.realtime.errors import SpeechEngineError
from backend.realtime.ingress import (
    VAD_WINDOW_SAMPLES,
    SpeechSegmentTracker,
    onset_lead_frames,
    spotter_feed,
)
from backend.realtime.speech import (
    CancelToken,
    NameHit,
    NameSpotter,
    StreamingTranscriber,
    VoiceActivityDetector,
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

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
EVALUATED_SPLITS = ("tune", "test")

_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9-]*")


@dataclass(frozen=True, slots=True)
class Engines:
    """The engines one run drives: the spotter under test, the VAD and the gate's STT."""

    spotter: NameSpotter
    vad: VoiceActivityDetector
    transcriber: StreamingTranscriber


EngineLoader = Callable[[PlacementConfig], Engines]
"""Builds the engines from the placement, with `kws_model` set to the model under test."""


class RefusedError(Exception):
    """The run cannot go ahead; the message says why, and nothing is written."""


# -- Calibration tables -----------------------------------------------------------------------


def false_accept_table(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Score to false accepts per hour, made monotone by a running minimum from low scores up.

    Each rate comes from its own run, and a higher threshold can measure more false accepts
    than a lower one: the gate decides on a decode of the span around each hit, a higher
    threshold can place a hit at a later step, and that span's decode can be accepted where the
    lower threshold's was not. A table the arbitration reads must not rank a higher score as
    more likely false, so each rate is capped by every rate below.
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
class _Feed:
    """Frames the spotter scores in turn: `warm` scored silently, then `push` reported.

    The harness never resets the stream: it has no mute and no backlog, the only causes of a
    reset in ingress, and each recording gets a stream of its own.
    """

    warm: range
    push: range


def _released(feeds: list[_Feed], warm: range, push: range) -> None:
    """Add one `spotter_feed` release to `feeds`, joined to the last feed where it can be.

    Releases are contiguous, so a release after a feed that pushed nothing, or one that warms
    nothing after a feed that pushed, extends that feed and the stream hears the same frames in
    the same order. A recording's feeds then grow with its segments, not its frames, which keeps
    hours of ambient audio's feeds small enough to hold.
    """
    if feeds:
        last = feeds[-1]
        if not last.push:
            feeds[-1] = _Feed(warm=range(last.warm.start, warm.stop), push=push)
            return
        if not warm:
            feeds[-1] = _Feed(warm=last.warm, push=range(last.push.start, push.stop))
            return
    feeds.append(_Feed(warm=warm, push=push))


@dataclass(frozen=True, slots=True)
class _Heard:
    """One recording as ingress would hear it: its frames, the spotter feeds and the segments.

    A segment is `(onset frame, last speech frame)`, inclusive.
    """

    frames: tuple[bytes, ...]
    feeds: tuple[_Feed, ...]
    segments: tuple[tuple[int, int], ...]


def _listen(pcm: bytes, vad: VoiceActivityDetector, cfg: EndpointingConfig) -> _Heard:
    """Segment one recording and decide which frames reach the spotter, as ingress does.

    Each VAD window at or above `vad_threshold` is speech, and ingress's own
    `SpeechSegmentTracker` turns the windows into segments. After each frame, as ingress does
    once it routes a frame, `spotter_feed` releases the frames the spotter scores next: every
    frame of the recording once and in order, a segment and its pre-roll pushed and the rest
    warmed. Frames still waiting when the recording ends, which an onset could yet have made
    pre-roll, are not scored, as ingress would not have scored them yet.
    """
    frames = _frames(pcm)
    usable = len(frames) * FRAME_BYTES
    samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
    stream = vad.open_stream()
    tracker = SpeechSegmentTracker(cfg)
    lead = onset_lead_frames(cfg)
    feeds: list[_Feed] = []
    segments: list[tuple[int, int]] = []
    window = 0
    next_pos = 0

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
        # The first frame the VAD has not scored: the one holding the next window's first
        # sample, which is what ingress's `_vad_from` names.
        next_unscored = window * VAD_WINDOW_SAMPLES // FRAME_SAMPLES
        segment = tracker.open_segment
        released = spotter_feed(
            next_pos,
            pos,
            segment[0] if segment is not None else None,
            tracker.earliest_onset(next_unscored),
            lead,
        )
        if released.end > next_pos:
            _released(feeds, released.warm, released.push)
            next_pos = released.end
    if tracker.open_segment is not None:
        segments.append(tracker.open_segment)
    return _Heard(frames=frames, feeds=tuple(feeds), segments=tuple(segments))


def _frames(pcm: bytes) -> tuple[bytes, ...]:
    """A recording's whole 20 ms frames; a partial last frame is never routed."""
    usable = len(pcm) // FRAME_BYTES * FRAME_BYTES
    return tuple(pcm[start : start + FRAME_BYTES] for start in range(0, usable, FRAME_BYTES))


@dataclass(frozen=True, slots=True)
class _HitResult:
    """One name hit's verdict and timings; `position` is the frame within the recording."""

    position: int
    score: float
    verdict: GateVerdict
    accept_ms: float
    screen_server_ms: float


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


class _Driver:
    """Runs recordings through stage 1 and the gate at every threshold of the run.

    A recording is named in log lines by a source label: a manifest entry by its file id, an
    ambient file by its number (`AmbientGate`).
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
        self.decodes = 0
        self.decode_errors = 0

    def gate(self, threshold: float) -> AddresseeGate:
        """The gate at `threshold`, built once.

        The live gate reads the stream's threshold for the self-media check, and the threshold
        under test is the one the daily instance would run with.
        """
        if threshold not in self._gates:
            self._gates[threshold] = AddresseeGate(
                dataclasses.replace(self._wake, kws_threshold=threshold),
                echo_guard_ms=self._settings.barge_in.echo_guard_ms,
            )
        return self._gates[threshold]

    def run(self, item: Entry, offset: int) -> dict[float, _Outcome]:
        """Every threshold's outcome for one recording, its frames numbered from `offset`.

        The gate decode of a span is the same text at every threshold, so each distinct span is
        decoded once per recording and its text and decode time reused.
        """
        heard = _listen(read_pcm16(item.path), self._engines.vad, self._endpointing)
        reference = read_pcm16(item.reference_path) if item.reference_path is not None else None
        decodes: dict[tuple[int, int], tuple[str, float] | None] = {}
        return {
            threshold: _Outcome(
                item,
                heard,
                self._spot(
                    item.file, heard, offset, self.gate(threshold), reference, decodes, threshold
                ),
            )
            for threshold in self._thresholds
        }

    def _spot(
        self,
        source: str,
        heard: _Heard,
        offset: int,
        gate: AddresseeGate,
        reference: bytes | None,
        decodes: dict[tuple[int, int], tuple[str, float] | None],
        threshold: float,
    ) -> tuple[_HitResult, ...]:
        stream = self._engines.spotter.open_stream(threshold)
        results: list[_HitResult] = []
        for batch in heard.feeds:
            for position in batch.warm:
                stream.warm(heard.frames[position], offset + position)
            for position in batch.push:
                arrival = self._clock()
                for found in stream.push(heard.frames[position], offset + position):
                    results.append(
                        self._judge(source, heard, found, offset, arrival, gate, reference, decodes)
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
        reference: bytes | None,
        decodes: dict[tuple[int, int], tuple[str, float] | None],
    ) -> _HitResult:
        position = min(max(found.end_pos - offset, 0), len(heard.frames) - 1)
        # Check a: her own recent speech, a memory read; `attend` goes out after it. Offline she
        # has said nothing, so it never rejects, but its cost is part of the screen latency.
        verdict = gate.pre_decode(score=found.score, reference_score=None, recent_tts_text="")
        screen_ms = (self._clock() - arrival) * 1000
        reference_score = None
        if verdict is None and reference is not None:
            reference_score = self._reference_score(source, reference, position)
            verdict = gate.pre_decode(
                score=found.score, reference_score=reference_score, recent_tts_text=""
            )
        tail_ms = max(float(self._wake.gate_tail_ms), (self._clock() - arrival) * 1000)
        if verdict is not None:
            return _HitResult(position, found.score, verdict, tail_ms, screen_ms)

        span = self._span(heard, position)
        if span not in decodes:
            decodes[span] = self._decode(source, heard, span)
        decoded = decodes[span]
        if decoded is None:
            # A failed gate decode rejects the candidate live, and so it does here.
            verdict = GateVerdict(
                accepted=False,
                check="not_addressed",
                score=found.score,
                reference_score=reference_score,
            )
            return _HitResult(position, found.score, verdict, tail_ms, screen_ms)
        text, decode_ms = decoded
        before = self._clock()
        verdict = gate.decide(text=text, score=found.score, reference_score=reference_score)
        decide_ms = (self._clock() - before) * 1000
        return _HitResult(
            position, found.score, verdict, tail_ms + decode_ms + decide_ms, screen_ms
        )

    def _span(self, heard: _Heard, position: int) -> tuple[int, int]:
        """The frames the gate decodes for a hit: ingress's `from_pos` through the gate tail.

        The start is the earliest segment in the chain reaching the hit, each gap at most
        `join_gap_ms`, less the pre-roll, and no further back than `lookback_ms`.
        """
        segments = [segment for segment in heard.segments if segment[0] <= position]
        start = position
        if segments:
            index = len(segments) - 1
            while index > 0:
                gap = segments[index][0] - segments[index - 1][1] - 1
                if gap * FRAME_MS > self._wake.join_gap_ms:
                    break
                index -= 1
            start = segments[index][0] - self._lead
        start = max(start, position - self._wake.lookback_ms // FRAME_MS, 0)
        end = min(position + self._wake.gate_tail_ms // FRAME_MS, len(heard.frames) - 1)
        return start, end

    def _decode(
        self, source: str, heard: _Heard, span: tuple[int, int]
    ) -> tuple[str, float] | None:
        start, end = span
        pcm = b"".join(heard.frames[start : end + 1])
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.decodes += 1
        stream = self._engines.transcriber.open_stream()
        try:
            stream.push(audio)
            before = self._clock()
            update = stream.decode_partial(CancelToken())
            decode_ms = (self._clock() - before) * 1000
        except SpeechEngineError:
            self.decode_errors += 1
            logger.warning("Gate decode failed for %s; counted as not addressed", source)
            return None
        finally:
            stream.close()
        return update.text, decode_ms

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
            "Entry %d of %d (%s): %d frames, %d segments",
            number,
            len(entries),
            item.file,
            len(heard.frames),
            len(heard.segments),
        )
        yield outcomes


@dataclass(slots=True)
class DecodeCounts:
    """Gate decodes run and failed, which `accepted_wakes` adds to for a caller's report."""

    decodes: int = 0
    decode_errors: int = 0


def accepted_wakes(
    entries: Sequence[Entry],
    engines: Engines,
    settings: VoiceSettings,
    thresholds: Sequence[float],
    clock: Callable[[], float],
    *,
    counts: DecodeCounts | None = None,
) -> dict[float, int]:
    """At each threshold, the entries with at least one hit the gate accepts.

    Over positives this is the count the tune and test passes report as woken, their total less
    their missed: the same driver over the same entries, numbered the same way. `counts`, when
    given, gains the run's gate decodes and failed decodes.
    """
    driver = _Driver(engines, settings, thresholds, clock)
    woken = dict.fromkeys(thresholds, 0)
    for outcomes in _outcomes(driver, entries):
        for threshold, outcome in outcomes.items():
            woken[threshold] += bool(outcome.accepted)
    if counts is not None:
        counts.decodes += driver.decodes
        counts.decode_errors += driver.decode_errors
    return woken


class AmbientGate:
    """Counts the hits the gate accepts in ambient audio, one file and one threshold at a time.

    Each file goes through the driver as a recording does: the VAD's segments, every frame fed
    through `spotter_feed` to a fresh stream at the threshold, warmed or pushed as ingress feeds
    it, and each pushed hit judged as the live gate judges it with no reference and nothing she
    said, which is the room where she is idle and the PC plays nothing. A warmed hit is never
    reported, so never gated.

    A file is heard by the VAD once: its segments and feeds are kept, and its audio is read
    again at each threshold, so one file's audio is held at a time. Each span is decoded at most
    once per file, whatever the threshold, and a failed decode rejects its hit as it does live.
    `decodes` and `decode_errors` count both across files, and `listen_s` is the clock's time
    spent in the VAD. A log line names a file only as "ambient file <i>", its place in the order
    the files were first given.
    """

    def __init__(
        self, engines: Engines, settings: VoiceSettings, clock: Callable[[], float]
    ) -> None:
        self._driver = _Driver(engines, settings, (), clock)
        self._vad = engines.vad
        self._endpointing = settings.endpointing
        self._clock = clock
        self._heard: dict[Path, tuple[str, _Heard]] = {}
        self._decoded: dict[Path, dict[tuple[int, int], tuple[str, float] | None]] = {}
        self.listen_s = 0.0

    @property
    def decodes(self) -> int:
        """Gate decodes run, failed ones included."""
        return self._driver.decodes

    @property
    def decode_errors(self) -> int:
        """Gate decodes that failed, each rejecting its hit."""
        return self._driver.decode_errors

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
            self._decoded[path] = {}
        hits = self._driver._spot(
            source,
            heard,
            0,
            self._driver.gate(threshold),
            None,
            self._decoded[path],
            threshold,
        )
        return sum(item.verdict.accepted for item in hits)


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
    screen = [lag_ms + hit.screen_server_ms for hit in woken] if lag_ms is not None else []

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
) -> dict[str, object]:
    """Run one split's recordings at each threshold and build the report.

    Frames are numbered on one uplink running from the first recording to the last, in
    manifest order, so a hit's position names its recording.
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
        "entries": {label: sum(item.label == label for item in entries) for label in LABELS},
        "media_hours": media_hours,
        "decode_errors": driver.decode_errors,
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


def real_engines(spec: str) -> EngineLoader:
    """The spotter and VAD from the placement, and the STT from a named factory.

    The factory is `<module>:<callable>`, called with the placement and returning a
    `StreamingTranscriber`, so the harness drives the STT the daily instance runs.
    """
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise RefusedError(f"--stt-factory must be <module>:<callable>, got {spec!r}")

    def load(placement: PlacementConfig) -> Engines:
        try:
            factory = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as exc:
            raise RefusedError(f"--stt-factory {spec!r} could not be loaded: {exc}") from exc
        spotter = MicroNameSpotter.load(placement, library_loader=import_micro_runtime)
        vad = SileroVad.load(
            placement,
            session_factory=open_onnx_session,
            available_providers=onnxruntime_providers,
        )
        return Engines(spotter=spotter, vad=vad, transcriber=factory(placement))

    return load


def parakeet_cpu(placement: PlacementConfig) -> StreamingTranscriber:
    """The `parakeet-cpu` engine as the gate's STT: `scripts.wake.eval_wake:parakeet_cpu`.

    `--stt-factory` is required and takes a `<module>:<callable>`; this one loads the adapter
    as `backend/factories.py` loads the `parakeet-cpu` chain, so a host run with no GPU needs
    no module of its own.

    Raises:
        SpeechEngineError: As `ParakeetTranscriber.load`.
    """
    return ParakeetTranscriber.load(placement, library_loader=import_onnx_asr)


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
        "--stt-factory", help="<module>:<callable> building the STT from the placement"
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
    if load_engines is None:
        if not args.stt_factory:
            raise RefusedError("--stt-factory is required to load the real engines")
        load_engines = real_engines(args.stt_factory)
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
    if args.split == "test" and "server" not in model.calibration:
        raise RefusedError(
            f"model {model.model_id} has no server calibration: run and write the tuning pass "
            "before the test pass"
        )
    entries = recordings.split(args.split)
    if not entries:
        raise RefusedError(f"the manifest has no {args.split} entries")

    placement = dataclasses.replace(
        settings.placement, models_dir=str(models_dir), kws_model=args.model
    )
    try:
        engines = load_engines(placement)
    except SpeechEngineError as exc:
        raise RefusedError(f"the engines could not be loaded: {exc}") from exc
    logger.info(
        "Evaluating model %s on %d %s entries at %d thresholds",
        model.model_id,
        len(entries),
        args.split,
        len(thresholds),
    )
    report = evaluate(entries, engines, settings, model, args.split, thresholds, clock)
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
