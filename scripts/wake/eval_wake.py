"""The wake model evaluation harness: stage 1 and the addressee gate, offline, over one split.

A wake model is judged end to end, by what reaches a verdict, not by its raw scores: a name hit
the gate rejects costs nothing but a cue, and a hit it accepts on a film is a false wake. So
each recording runs through an offline driver of the live path: the VAD opens speech segments
and feeds the spotter from each onset as ingress does (`backend/realtime/ingress.py`); every hit
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
for the one `decide` call that reads them, and are never written or logged.

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
from collections.abc import Callable, Mapping, Sequence
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
from backend.realtime.engines.kws import OnnxNameSpotter, WakeManifest, read_manifest
from backend.realtime.engines.silero_vad import (
    SileroVad,
    onnxruntime_providers,
    open_onnx_session,
)
from backend.realtime.errors import SpeechEngineError
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
VAD_WINDOW_SAMPLES = 512
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
    than a lower one: a score that dips below it and returns is two rising edges there and one
    below, and each hit's decode covers a different span. A table the arbitration reads must
    not rank a higher score as more likely false, so each rate is capped by every rate below.
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
    """Frames the spotter scores in one go, after a context reset when `reset` is set."""

    reset: bool
    positions: tuple[int, ...]


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

    Speech starts after `speech_start_windows` consecutive VAD windows at or above
    `vad_threshold`; the spotter then gets a fresh context and the frames from the onset less
    the pre-roll, and every frame after it while the segment is open. A segment closes after
    `endpoint_silence_ms` of silence. Ingress feeds the spotter only during speech, so audio it
    never scores live is not scored here either.
    """
    usable = len(pcm) // FRAME_BYTES * FRAME_BYTES
    frames = tuple(pcm[start : start + FRAME_BYTES] for start in range(0, usable, FRAME_BYTES))
    samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
    stream = vad.open_stream()
    pre_roll = cfg.pre_roll_ms // FRAME_MS
    feeds: list[tuple[bool, list[int]]] = []
    segments: list[tuple[int, int]] = []
    segment: list[int] | None = None
    run = run_start = silence = window = 0
    fed: int | None = None
    reset_pending = False

    def feed(positions: range, reset: bool) -> None:
        if not reset and feeds and feeds[-1][1][-1] == positions[0] - 1:
            feeds[-1][1].extend(positions)
        else:
            feeds.append((reset, list(positions)))

    for pos in range(len(frames)):
        while (window + 1) * VAD_WINDOW_SAMPLES <= (pos + 1) * FRAME_SAMPLES:
            start = window * VAD_WINDOW_SAMPLES
            first = start // FRAME_SAMPLES
            last = (start + VAD_WINDOW_SAMPLES - 1) // FRAME_SAMPLES
            window += 1
            speech = stream.score(samples[start : start + VAD_WINDOW_SAMPLES]) >= cfg.vad_threshold
            if speech and segment is not None:
                segment[1] = last
                silence = 0
                continue
            silence += VAD_WINDOW_SAMPLES
            if not speech:
                run = 0
                if segment is not None and silence * 1000 // SAMPLE_RATE >= cfg.endpoint_silence_ms:
                    segments.append((segment[0], segment[1]))
                    segment = None
                continue
            if run == 0:
                run_start = first
            run += 1
            if run >= cfg.speech_start_windows:
                run = silence = 0
                segment = [run_start, last]
                reset_pending = True
                begin = max(run_start - pre_roll, 0 if fed is None else fed + 1)
                if begin <= pos:
                    feed(range(begin, pos + 1), True)
                    reset_pending = False
                    fed = pos
        if segment is not None and (fed is None or pos > fed):
            feed(range(pos, pos + 1), reset_pending)
            reset_pending = False
            fed = pos
    if segment is not None:
        segments.append((segment[0], segment[1]))
    return _Heard(
        frames=frames,
        feeds=tuple(_Feed(reset, tuple(positions)) for reset, positions in feeds),
        segments=tuple(segments),
    )


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
    """Runs recordings through stage 1 and the gate at every threshold of the run."""

    def __init__(
        self,
        engines: Engines,
        settings: VoiceSettings,
        thresholds: Sequence[float],
        clock: Callable[[], float],
    ) -> None:
        self._engines = engines
        self._endpointing = settings.endpointing
        self._wake = settings.wake
        self._clock = clock
        self._pre_roll = settings.endpointing.pre_roll_ms // FRAME_MS
        # The live gate reads the stream's threshold for the self-media check, and the threshold
        # under test is the one the daily instance would run with.
        self._gates = {
            threshold: AddresseeGate(
                dataclasses.replace(settings.wake, kws_threshold=threshold),
                echo_guard_ms=settings.barge_in.echo_guard_ms,
            )
            for threshold in thresholds
        }
        self.decode_errors = 0

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
                item, heard, self._spot(item, heard, offset, gate, reference, decodes, threshold)
            )
            for threshold, gate in self._gates.items()
        }

    def _spot(
        self,
        item: Entry,
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
            if batch.reset:
                stream.reset()
            for position in batch.positions:
                arrival = self._clock()
                for found in stream.push(heard.frames[position], offset + position):
                    results.append(
                        self._judge(item, heard, found, offset, arrival, gate, reference, decodes)
                    )
        return tuple(results)

    def _judge(
        self,
        item: Entry,
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
            reference_score = self._reference_score(item, reference, position)
            verdict = gate.pre_decode(
                score=found.score, reference_score=reference_score, recent_tts_text=""
            )
        tail_ms = max(float(self._wake.gate_tail_ms), (self._clock() - arrival) * 1000)
        if verdict is not None:
            return _HitResult(position, found.score, verdict, tail_ms, screen_ms)

        span = self._span(heard, position)
        if span not in decodes:
            decodes[span] = self._decode(item, heard, span)
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
            start = segments[index][0] - self._pre_roll
        start = max(start, position - self._wake.lookback_ms // FRAME_MS, 0)
        end = min(position + self._wake.gate_tail_ms // FRAME_MS, len(heard.frames) - 1)
        return start, end

    def _decode(
        self, item: Entry, heard: _Heard, span: tuple[int, int]
    ) -> tuple[str, float] | None:
        start, end = span
        pcm = b"".join(heard.frames[start : end + 1])
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        stream = self._engines.transcriber.open_stream()
        try:
            stream.push(audio)
            before = self._clock()
            update = stream.decode_partial(CancelToken())
            decode_ms = (self._clock() - before) * 1000
        except SpeechEngineError:
            self.decode_errors += 1
            logger.warning("Gate decode failed for %s; counted as not addressed", item.file)
            return None
        finally:
            stream.close()
        return update.text, decode_ms

    def _reference_score(self, item: Entry, reference: bytes, position: int) -> float | None:
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
            logger.warning("Reference scoring failed for %s; scored as absent", item.file)
            return None


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
    offset = 0
    for number, item in enumerate(entries, start=1):
        outcomes = driver.run(item, offset)
        heard = next(iter(outcomes.values())).heard
        offset += len(heard.frames)
        for threshold, outcome in outcomes.items():
            by_threshold[threshold].append(outcome)
        logger.debug(
            "Entry %d of %d (%s): %d frames, %d segments",
            number,
            len(entries),
            item.file,
            len(heard.frames),
            len(heard.segments),
        )

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


def _real_engines(spec: str) -> EngineLoader:
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
        spotter = OnnxNameSpotter.load(placement, session_factory=open_onnx_session)
        vad = SileroVad.load(
            placement,
            session_factory=open_onnx_session,
            available_providers=onnxruntime_providers,
        )
        return Engines(spotter=spotter, vad=vad, transcriber=factory(placement))

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
        load_engines = _real_engines(args.stt_factory)
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
