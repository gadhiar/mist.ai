"""Voice latency bench: drive a running backend over `/ws` and report each stage's P50 and P90.

The bench speaks the wake fixtures that `scripts/voice_fixtures/generate_fixtures.py` wrote, as an
endpoint would: `voice_start` with `spotter` `endpoint`, then per fixture a `wake` whose
`backlog_ms` is the audio up to the name's end, then the fixture's frames, the backlog at once and
the rest in real time, then silence until the turn's `turn_metrics` arrives. It answers each turn's
audio with `playback_progress` and `playback_done` as the audio arrives, since a spoken turn
commits only once the client reports it played; the bench plays nothing, so arrival stands in for
playback.

Every `turn_metrics` is printed to stdout as one JSON line, as received. After the last turn the
bench prints, per `power_level` and turn `kind`, the P50 and P90 of each budget stage and of the
caption, wake and screen timings, each beside its row of the turn's active preset and of the
`memo` preset, whose total gates are 850 ms and 1.2 s, and the mean share of prompt tokens the
LLM backend reported cached. `--scene-log` adds the frontend's event-to-pixel reports. The bench
exits 0 once it has finished, whatever the numbers; judging them is the reader's job.

`--replay <file>` prints the tables from recorded lines instead, for example from a saved run:

    python scripts/voice_bench/latency_bench.py --url ws://localhost:8001/ws > run.jsonl
    python scripts/voice_bench/latency_bench.py --replay run.jsonl --scene-log scene.log
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import re
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Run as a file, the repository root is not on the path, and `backend` would not import.
    sys.path.insert(0, str(REPO_ROOT))

from backend.realtime.budget import BUDGET_FIELDS, BUDGET_PRESETS, STAGE_FIELDS
from backend.realtime.errors import ProtocolError
from backend.realtime.protocol import (
    UPLINK_SAMPLE_RATE,
    Direction,
    Frame,
    MsgType,
    TurnMetricsMessage,
    decode_frame,
    encode_frame,
    parse_server_message,
)
from backend.realtime.turn import STAGE_TIMESTAMPS

BUDGET_ROWS: tuple[str, ...] = BUDGET_FIELDS
"""Every row a `turn_metrics.budget` carries besides `preset`."""

FRAME_MS = 20
FRAME_SAMPLES = UPLINK_SAMPLE_RATE * FRAME_MS // 1000
DOWNLINK_SAMPLES_PER_MS = 24
WAKE_KEYWORD = "mist"
IDLE_MS = 1000
"""Silence sent between turns, so each wake meets an idle controller."""
DEFAULT_GENERATED = REPO_ROOT / "tests" / "fixtures" / "voice" / "generated"

POWER_ORDER = ("full", "low", "off")
KIND_ORDER = ("plain", "tool")

# Rows reported straight from `timings`, each with the budget row it is judged beside; None
# where the budget has no row for it. `scene_anchor_lead_ms` is a lead, the least time between
# an anchored operation's send and its sample's play, so it has no ceiling to sit beside.
PASSTHROUGH_ROWS: tuple[tuple[str, str | None], ...] = (
    ("caption_lag_ms", "caption_lag_ms"),
    ("wake_screen_ms", "wake_screen_ms"),
    ("wake_accept_ms", None),
    ("screen_command_ms", "screen_command_ms"),
    ("scene_anchor_lead_ms", None),
)
TOTAL_ROW = "total_ms"
EVENT_TO_PIXEL = "event_to_pixel_ms"


class BenchError(RuntimeError):
    """The bench cannot go on: no voice on the server, a refusal, or an invalid message."""


# -- Rows and percentiles ---------------------------------------------------------------------


def percentile(values: Sequence[int], p: float) -> int:
    """The nearest-rank percentile: the smallest value with at least `p` percent at or below it.

    Raises:
        ValueError: `values` is empty.
    """
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def turn_rows(message: TurnMetricsMessage) -> dict[str, int]:
    """One turn's measured rows, in milliseconds.

    Each budget stage is its timestamp less the nearest earlier stage timestamp present, floored
    at 0, which is how the server judges `over_budget`; `total_ms` is `first_played`, the time
    `total_p50_ms` and `total_p90_ms` bound. `tool_call_ms` is a tool turn's first token to its
    parsed call, and `barge_in_reaction_ms` a keyword to the playback stop. The caption, wake and
    screen timings are already durations and pass through.
    """
    timings = message.timings
    rows: dict[str, int] = {}
    previous = 0
    for stage, key in STAGE_TIMESTAMPS:
        if key in timings:
            rows[stage] = max(timings[key] - previous, 0)
            previous = timings[key]
    if "first_played" in timings:
        rows[TOTAL_ROW] = timings["first_played"]
    if message.kind == "tool" and "tool_call_parsed" in timings and "first_token" in timings:
        rows["tool_call_ms"] = timings["tool_call_parsed"] - timings["first_token"]
    if "keyword_heard" in timings and "playback_stopped" in timings:
        rows["barge_in_reaction_ms"] = timings["playback_stopped"] - timings["keyword_heard"]
    for key, _ in PASSTHROUGH_ROWS:
        if key in timings:
            rows[key] = timings[key]
    return rows


@dataclass(frozen=True, slots=True)
class TurnSample:
    """What the report needs from one `turn_metrics`."""

    turn_id: str
    power_level: str
    kind: str
    preset: str
    budget: Mapping[str, int]
    rows: Mapping[str, int]
    prompt_tokens: int | None
    cached_tokens: int | None


def sample_from(message: TurnMetricsMessage) -> TurnSample:
    """A report sample from a validated `turn_metrics`."""
    budget = message.budget.model_dump()
    return TurnSample(
        turn_id=message.turn_id,
        power_level=message.power_level,
        kind=message.kind,
        preset=budget.pop("preset"),
        budget=budget,
        rows=turn_rows(message),
        prompt_tokens=message.prompt.prompt_tokens,
        cached_tokens=message.prompt.cached_tokens,
    )


def parse_metrics_line(text: str) -> TurnMetricsMessage | None:
    """The `turn_metrics` a line holds, or None for any other line.

    Raises:
        BenchError: The line is a `turn_metrics` the protocol refuses; the message names the
            field.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        raw = json.loads(stripped)
    except ValueError:
        return None
    if not isinstance(raw, dict) or raw.get("type") != "turn_metrics":
        return None
    try:
        message = parse_server_message(stripped)
    except ProtocolError as exc:
        raise BenchError(f"invalid turn_metrics: {exc}") from exc
    if not isinstance(message, TurnMetricsMessage):
        return None
    return message


# -- Scene log --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PixelReport:
    """One event-to-pixel report from the frontend, with its turn when the line names one."""

    turn_id: str | None
    event_to_pixel_ms: int


_PIXEL = re.compile(r"\bevent_to_pixel_ms[=:]\s*(\d+)")
_TURN = re.compile(r"\bturn_id[=:]\s*([A-Za-z0-9_.:-]+)")


def read_scene_log(lines: Iterable[str]) -> list[PixelReport]:
    """Every event-to-pixel report in a log, in order.

    The backend logs the frontend's report at INFO under `backend.realtime.scene` from stage 2
    (T27), whose line format is not fixed yet, so two shapes are read: a JSON object with
    `event_to_pixel_ms` and optionally `turn_id`, anywhere from the line's first `{`, and
    `event_to_pixel_ms=<int>` with optionally `turn_id=<id>` anywhere in the line. Other lines
    are skipped.
    """
    reports: list[PixelReport] = []
    for text in lines:
        brace = text.find("{")
        if brace >= 0:
            try:
                raw = json.loads(text[brace:])
            except ValueError:
                raw = None
            if isinstance(raw, dict) and isinstance(raw.get("event_to_pixel_ms"), int):
                turn = raw.get("turn_id")
                reports.append(
                    PixelReport(turn if isinstance(turn, str) else None, raw["event_to_pixel_ms"])
                )
                continue
        pixel = _PIXEL.search(text)
        if pixel is None:
            continue
        turn_match = _TURN.search(text)
        reports.append(
            PixelReport(turn_match.group(1) if turn_match else None, int(pixel.group(1)))
        )
    return reports


# -- The report -------------------------------------------------------------------------------


def _row_order(samples: Sequence[TurnSample], pixels: bool) -> list[tuple[str, str | None]]:
    """Each row with data in the group, in table order, with the budget row it sits beside."""
    present = {row for sample in samples for row in sample.rows}
    order: list[tuple[str, str | None]] = [(stage, stage) for stage in STAGE_FIELDS]
    order.append((TOTAL_ROW, TOTAL_ROW))
    order += [("tool_call_ms", "tool_call_ms"), ("barge_in_reaction_ms", "barge_in_reaction_ms")]
    order += list(PASSTHROUGH_ROWS)
    rows = [(row, budget) for row, budget in order if row in present]
    if pixels:
        rows.append((EVENT_TO_PIXEL, EVENT_TO_PIXEL))
    return rows


def _budget_cell(budget: Mapping[str, int], row: str | None) -> str:
    if row is None:
        return "-"
    if row == TOTAL_ROW:
        return f"{budget.get('total_p50_ms', '-')}/{budget.get('total_p90_ms', '-')}"
    value = budget.get(row)
    return "-" if value is None else str(value)


def _table(samples: Sequence[TurnSample], pixels: Mapping[str, list[int]]) -> tuple[list[str], str]:
    """The group's table lines, and the preset its budget column shows."""
    presets = sorted({sample.preset for sample in samples})
    preset = presets[0] if len(presets) == 1 else "mixed:" + ",".join(presets)
    budget = samples[0].budget
    memo = {field: getattr(BUDGET_PRESETS["memo"], field) for field in BUDGET_ROWS}
    group_pixels = [value for s in samples for value in pixels.get(s.turn_id, [])]
    header = f"{'row':<24}{'n':>5}{'p50':>8}{'p90':>8}  {'budget':>12}  {'memo':>10}"
    lines = [header]
    for row, budget_row in _row_order(samples, bool(group_pixels)):
        if row == EVENT_TO_PIXEL:
            values = group_pixels
        else:
            values = [sample.rows[row] for sample in samples if row in sample.rows]
        lines.append(
            f"{row:<24}{len(values):>5}{percentile(values, 50):>8}{percentile(values, 90):>8}"
            f"  {_budget_cell(budget, budget_row):>12}  {_budget_cell(memo, budget_row):>10}"
        )
    return lines, preset


def _cached_share(samples: Sequence[TurnSample]) -> str:
    shares = [
        s.cached_tokens / s.prompt_tokens
        for s in samples
        if s.prompt_tokens and s.cached_tokens is not None
    ]
    missing = len(samples) - len(shares)
    if not shares:
        return f"cached_tokens share: no turn reported usage ({missing} without usage)"
    mean = sum(shares) / len(shares)
    return (
        f"cached_tokens share: mean {mean:.3f} over {len(shares)} turns ({missing} without usage)"
    )


def report(samples: Sequence[TurnSample], pixels: Sequence[PixelReport]) -> list[str]:
    """The tables, one per `power_level` and `kind` with turns, then the scene log's summary."""
    by_turn: dict[str, list[int]] = {}
    for item in pixels:
        if item.turn_id is not None:
            by_turn.setdefault(item.turn_id, []).append(item.event_to_pixel_ms)
    groups: dict[tuple[str, str], list[TurnSample]] = {}
    for sample in samples:
        groups.setdefault((sample.power_level, sample.kind), []).append(sample)

    def order(key: tuple[str, str]) -> tuple[int, int, str, str]:
        level, kind = key
        rank = POWER_ORDER.index(level) if level in POWER_ORDER else len(POWER_ORDER)
        kind_rank = KIND_ORDER.index(kind) if kind in KIND_ORDER else len(KIND_ORDER)
        return rank, kind_rank, level, kind

    lines: list[str] = []
    for level, kind in sorted(groups, key=order):
        group = groups[(level, kind)]
        table, preset = _table(group, by_turn)
        lines.append(f"== power_level={level} kind={kind} turns={len(group)} preset={preset} ==")
        lines += table
        lines.append(_cached_share(group))
    if not samples:
        lines.append("no turn_metrics to report")
    if pixels:
        known = {sample.turn_id for sample in samples}
        matched = sum(1 for item in pixels if item.turn_id in known)
        values = [item.event_to_pixel_ms for item in pixels]
        lines.append(
            f"scene log: {len(pixels)} event_to_pixel reports, {matched} matched to a turn"
        )
        lines.append(
            f"{EVENT_TO_PIXEL} over every report: n={len(values)} "
            f"p50={percentile(values, 50)} p90={percentile(values, 90)}"
        )
    return lines


# -- Driving `/ws` ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchFixture:
    """A wake fixture: its audio at 16 kHz and the backlog up to the name's end."""

    id: str
    audio: np.ndarray
    backlog_ms: int


def load_bench_fixtures(generated: Path) -> list[BenchFixture]:
    """The wake fixtures of a generated set; the other scenarios are the host tier's.

    Raises:
        BenchError: The manifest or a WAV cannot be read.
    """
    from scripts.voice_fixtures.generate_fixtures import FixtureError, read_manifest, read_wav

    try:
        manifest = read_manifest(generated)
        return [
            BenchFixture(entry.id, read_wav(generated / entry.file), entry.wake.name_end_ms)
            for entry in manifest.fixtures
            if entry.wake is not None
        ]
    except (FixtureError, OSError) as exc:
        raise BenchError(f"cannot read the fixtures in {generated}: {exc}") from exc


def uplink_frames(audio: np.ndarray, *, first_seq: int) -> list[bytes]:
    """The audio as `mic_audio` frames of 20 ms, the last one padded with silence."""
    count = math.ceil(len(audio) / FRAME_SAMPLES)
    padded = np.zeros(count * FRAME_SAMPLES, dtype=np.float32)
    padded[: len(audio)] = audio
    pcm = np.rint(np.clip(padded, -1.0, 1.0) * 32767).astype("<i2")
    frames = []
    for n in range(count):
        payload = pcm[n * FRAME_SAMPLES : (n + 1) * FRAME_SAMPLES].tobytes()
        frame = Frame(MsgType.MIC_AUDIO, 0, first_seq + n, UPLINK_SAMPLE_RATE, payload)
        frames.append(encode_frame(frame))
    return frames


class PlaybackTracker:
    """Counts each turn's downlink audio and answers it as a client that played it on arrival."""

    def __init__(self) -> None:
        self._turns: dict[int, str] = {}
        self._samples: dict[int, int] = {}

    def stream_start(self, message: Mapping[str, object]) -> None:
        """Note a turn's audio tag, from its `stream_start`."""
        tag = message.get("audio_tag")
        if isinstance(tag, int):
            self._turns[tag] = str(message["turn_id"])
            self._samples[tag] = 0

    def frame(self, frame: Frame) -> list[dict[str, object]]:
        """The messages a downlink frame calls for: progress per chunk, done at the end."""
        turn_id = self._turns.get(frame.turn_tag)
        if turn_id is None:
            return []
        if frame.msg_type == MsgType.AUDIO_CHUNK:
            self._samples[frame.turn_tag] += len(frame.payload) // 2
            played = self._samples[frame.turn_tag] // DOWNLINK_SAMPLES_PER_MS
            return [{"type": "playback_progress", "turn_id": turn_id, "played_ms": played}]
        if frame.msg_type == MsgType.AUDIO_COMPLETE:
            played = self._samples.pop(frame.turn_tag) // DOWNLINK_SAMPLES_PER_MS
            del self._turns[frame.turn_tag]
            return [{"type": "playback_done", "turn_id": turn_id, "played_ms": played}]
        return []


class Socket(Protocol):
    """The part of a WebSocket connection the bench uses."""

    async def send(self, data: str | bytes) -> None:
        """Send one text or binary message."""
        ...

    async def recv(self) -> str | bytes:
        """Receive the next message."""
        ...


VOICE_START = {
    "type": "voice_start",
    "req": 1,
    "mode": "wake",
    "sample_rate": UPLINK_SAMPLE_RATE,
    "frame_ms": FRAME_MS,
    "aec": "headset",
    "playback": True,
    "endpoint": "desktop",
    "spotter": "endpoint",
}


async def _handshake(socket: Socket) -> None:
    """Wait for `session_started`, start voice, and wait for the answer."""
    while True:
        raw = await socket.recv()
        if isinstance(raw, bytes):
            continue
        message = json.loads(raw)
        if message.get("type") == "session_started":
            voice = (message.get("capabilities") or {}).get("voice")
            if not voice or "endpoint" not in voice.get("spotters", []):
                raise BenchError(
                    "the server offers no voice with an endpoint spotter; check that VAD and "
                    "STT loaded"
                )
            break
    await socket.send(json.dumps(VOICE_START))
    while True:
        raw = await socket.recv()
        if isinstance(raw, bytes):
            continue
        message = json.loads(raw)
        if message.get("type") == "voice_state" and message.get("cause") == "answer":
            if not message.get("active") or message.get("refused"):
                raise BenchError(f"voice_start refused: {message.get('refused')}")
            return


class _Receiver:
    """Reads every server message, answers playback, and collects each `turn_metrics`."""

    def __init__(self, socket: Socket, emit: Callable[[str], None], warn: Callable[[str], None]):
        self.socket = socket
        self.emit = emit
        self.warn = warn
        self.tracker = PlaybackTracker()
        self.results: list[TurnMetricsMessage] = []
        self.ended = asyncio.Event()

    async def run(self) -> None:
        while True:
            raw = await self.socket.recv()
            if isinstance(raw, bytes):
                try:
                    frame = decode_frame(raw, direction=Direction.DOWN)
                except ProtocolError as exc:
                    self.warn(f"latency_bench: bad downlink frame ignored: {exc}")
                    continue
                for reply in self.tracker.frame(frame):
                    await self.socket.send(json.dumps(reply))
                continue
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "stream_start":
                self.tracker.stream_start(message)
            elif kind == "turn_metrics":
                metrics = parse_metrics_line(raw)
                if metrics is not None:
                    self.emit(raw)
                    self.results.append(metrics)
                    self.ended.set()
            elif kind == "voice_decision" and message.get("kind") in ("wake_rejected", "dropped"):
                # Either closes the utterance without a turn, so no `turn_metrics` will come.
                self.warn(
                    f"latency_bench: {message.get('kind')} {message.get('reason')}: "
                    f"{json.dumps(message.get('detail'))}"
                )
                self.ended.set()


async def drive(
    socket: Socket,
    fixtures: Sequence[BenchFixture],
    *,
    rounds: int,
    turn_timeout_s: float,
    emit: Callable[[str], None],
    warn: Callable[[str], None],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[TurnMetricsMessage]:
    """Speak every fixture `rounds` times, one turn at a time; returns each `turn_metrics`.

    Raises:
        BenchError: The server has no endpoint voice, or refused `voice_start`.
    """
    await _handshake(socket)
    receiver = _Receiver(socket, emit, warn)
    task = asyncio.ensure_future(receiver.run())
    seq = 0
    silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    idle = np.zeros(UPLINK_SAMPLE_RATE * IDLE_MS // 1000, dtype=np.float32)
    due = clock()

    async def paced(frame: bytes) -> None:
        # One frame per 20 ms slot, as a microphone delivers them. A late frame goes at once and
        # the schedule restarts from now, so the bench never bursts to catch up. The sleep runs
        # even when the slot is due, so the receiver gets its turn between frames.
        nonlocal due
        await sleep(max(due - clock(), 0.0))
        await socket.send(frame)
        due = max(due, clock()) + FRAME_MS / 1000

    try:
        for _ in range(rounds):
            for fixture in fixtures:
                receiver.ended.clear()
                await socket.send(
                    json.dumps(
                        {
                            "type": "wake",
                            "keyword": WAKE_KEYWORD,
                            "score": 1.0,
                            "backlog_ms": fixture.backlog_ms,
                        }
                    )
                )
                frames = uplink_frames(fixture.audio, first_seq=seq)
                seq += len(frames)
                backlog = fixture.backlog_ms // FRAME_MS
                # The backlog was buffered before the wake, so it goes at once, as an endpoint
                # flushes its ring buffer; the rest is live speech.
                for frame in frames[:backlog]:
                    await socket.send(frame)
                due = clock()
                for frame in frames[backlog:]:
                    await paced(frame)
                deadline = clock() + turn_timeout_s
                while not receiver.ended.is_set() and clock() < deadline:
                    await paced(uplink_frames(silence, first_seq=seq)[0])
                    seq += 1
                if not receiver.ended.is_set():
                    warn(f"latency_bench: {fixture.id}: no turn_metrics in {turn_timeout_s:g} s")
                for frame in uplink_frames(idle, first_seq=seq):
                    await paced(frame)
                    seq += 1
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return receiver.results


# -- The command line -------------------------------------------------------------------------


def _print_report(samples: Sequence[TurnSample], scene_log: Path | None) -> None:
    pixels = []
    if scene_log is not None:
        pixels = read_scene_log(scene_log.read_text(encoding="utf-8").splitlines())
    print()
    for text in report(samples, pixels):
        print(text)


def _replay(path: Path) -> list[TurnSample]:
    samples = []
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            message = parse_metrics_line(text)
        except BenchError as exc:
            print(f"latency_bench: {path.name}:{number}: {exc}; skipped", file=sys.stderr)
            continue
        if message is not None:
            samples.append(sample_from(message))
    return samples


async def _live(args: argparse.Namespace) -> list[TurnSample]:
    from websockets.asyncio.client import connect

    fixtures = load_bench_fixtures(args.fixtures)
    if not fixtures:
        raise BenchError(f"no wake fixtures in {args.fixtures}")
    async with connect(args.url, max_size=None) as socket:
        results = await drive(
            socket,
            fixtures,
            rounds=args.rounds,
            turn_timeout_s=args.turn_timeout_s,
            emit=lambda text: print(text, flush=True),
            warn=lambda text: print(text, file=sys.stderr, flush=True),
        )
    return [sample_from(message) for message in results]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench or replay a recorded run; 0 once it finished, whatever the numbers."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="the backend's /ws, for example ws://localhost:8001/ws")
    source.add_argument("--replay", type=Path, help="recorded turn_metrics lines to report")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--rounds", type=int, default=4, help="passes over the wake fixtures")
    parser.add_argument("--turn-timeout-s", type=float, default=60.0)
    parser.add_argument("--scene-log", type=Path, help="log lines with event_to_pixel reports")
    args = parser.parse_args(argv)
    try:
        samples = _replay(args.replay) if args.replay else asyncio.run(_live(args))
        _print_report(samples, args.scene_log)
    except (BenchError, OSError) as exc:
        print(f"latency_bench: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
