"""The spike listener: a stand-in for MIST's backend that measures one device against G1.

    python -m scripts.room_endpoint.spike_listener --devices <dir> --port 8443 --out <json>
        [--host 0.0.0.0] [--fresh]

At H4 it runs in a container from the worker image, started with `docker run -i` so stdin
reaches it, with `-p <PC LAN IP>:8443:8443`, the devices dir mounted read-only and the result
directory mounted. It serves `wss://<host>:<port>/v2/device` with the devices dir's
`tls/server.pem` and `tls/server.key`, and admits a device only when
`backend.devices.registry.DeviceRegistry.verify` accepts its `Authorization: Bearer
<name>.<secret>`; anything else is refused with HTTP 403 before the WebSocket is accepted, and a
path other than `/v2/device` with 404. A newer connection replaces the current one (close 4001).

Frames are decoded and messages built with `backend.realtime.protocol` (protocol 2). It answers
`session_started` on connect, a `voice_state` (`cause` answer, `active` true) to each
`voice_start`, and a `voice_state` broadcast to each `mute`. It sends nothing else unless told to.

Operator commands, one per line on stdin and never on the TLS port; each prints a one-line reply:

    state <idle|attend|listen|think|speak>
        Send `state_cycle` with that state. `speak` and `idle` close the device's uplink, so the
        listener marks the open wake closed by it.
    reject [self_speech|self_media|not_addressed|lost_arbitration|no_backlog]
        Send `voice_decision` `wake_rejected` (default `not_addressed`), with the open wake's
        score, and mark the wake closed by it.
    tone
        Stream a generated 400 Hz tone for 60 s: `stream_start` with a fresh `audio_tag`, then
        1500 `0x01` frames of 40 ms of 24 kHz PCM16, paced to the clock with 200 ms of lead,
        then `0x02`. A `barge_in` from the device does not stop it, so the 60 s always plays.
    status
        Print what has been counted so far.
    quit
        Write the result and stop. SIGINT and SIGTERM (`docker stop`) do the same.

What it measures, all on its own monotonic clock:

- Per wake: `backlog_ms`; the backlog's frame count, taken as the first `backlog_ms / 20`
  uplink frames after the `wake` (item 4 has the device send its whole backlog before any live
  frame); whether their `seq` rises by one each frame; the time from the `wake` to the last
  backlog frame; and the live frames after it, with each forward `seq` jump counted as a gap and
  the frames it skipped as missing. A wake is open until the next `wake` or the connection ends.
- Uplink frames that arrive with no wake open on the connection (G3).
- The `device_stats` series, every sample with its time.
- Per tone: the uplink frames received from its start to 61 s later (the tone's 60 s and 1 s
  for frames in flight), how many of them were live, their `seq` gaps, and the device's
  `playback_underruns` from the last sample at or before the start to the first at or after the
  tone's end.
- From the listener's start (its socket bound) to the device's first authenticated, valid
  `voice_start`. G1's reconnection check reads this from every run after the first: restart
  the listener at least once with the same `--out`.

The result is the one file `--out`, rewritten every 10 s, after `quit` and on a signal. Each run
appends to the runs already in it (`--fresh` discards them), and G1's checks and a pass or fail
for each are computed over all runs. Payloads are counted and dropped as they arrive: no audio
is kept, and nothing but that JSON is written (its rewrite goes through one temporary file in the
same directory, renamed over it).

Exit codes: 0 stopped; 1 failed to serve (the port could not be bound); 2 refused, nothing
written (the devices dir or an existing `--out` could not be used).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import signal
import ssl
import struct
import sys
import tempfile
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from backend.devices.registry import DeviceRecord, DeviceRegistry, claimed_name
from backend.errors import MistError
from backend.realtime.errors import ProtocolError
from backend.realtime.protocol import (
    DOWNLINK_SAMPLE_RATE,
    PROTOCOL_VERSION,
    UPLINK_SAMPLE_RATE,
    BargeInSettings,
    Direction,
    EngineStatus,
    EngineStatuses,
    Frame,
    LatencyBudget,
    MsgType,
    MuteMessage,
    PlaybackDoneMessage,
    PlaybackSettings,
    SessionCapabilities,
    SessionStartedMessage,
    StateCycleMessage,
    StreamFormat,
    StreamStartMessage,
    TtsVoice,
    VoiceCapabilities,
    VoiceDecisionWakeRejected,
    VoiceStartMessage,
    VoiceStateMessage,
    WakeMessage,
    WakeRejectedDetail,
    decode_frame,
    encode_frame,
    encode_message,
    next_turn_tag,
    parse_client_message,
    peek_voice_start,
)
from scripts.daily.devices import (
    DEFAULT_PORT,
    REGISTRY_FILE,
    SERVER_CERT,
    SERVER_KEY,
    TLS_DIR,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

DEVICE_PATH = "/v2/device"
BIND_ALL = "0.0.0.0"  # nosec B104 - inside its container; Docker publishes it on one LAN address
WS_MAX_SIZE = 65536  # the device listener's `ws_max_size`
RESULT_FILE_SCHEMA = "mist-spike-result/1"
RESULT_WRITE_INTERVAL_S = 10.0

UPLINK_FRAME_MS = 20
TONE_HZ = 400  # 16 whole cycles per 40 ms frame, so every frame is the same samples
TONE_AMPLITUDE = 6000
TONE_FRAME_MS = 40
TONE_S = 60.0
TONE_LEAD_S = 0.2  # sent this far ahead of the clock, above the device's 80 ms prebuffer
PLAYBACK_GRACE_S = 1.0

# G1's thresholds (plan mist-voice-pe v1, Goals).
HEAP_FREE_MIN_B = 30_000
PSRAM_FREE_MIN_B = 1_000_000
MEMORY_WINDOW_S = 1800.0
WAKES_REQUIRED = 20
BACKLOG_MIN_MS = 5000
BACKLOG_DELIVERY_MS = 1000
PLAYBACK_LIVE_FRAMES = round(TONE_S * 1000 / UPLINK_FRAME_MS)
RECONNECT_LIMIT_MS = 15_000

STATES = ("idle", "attend", "listen", "think", "speak")
UPLINK_CLOSING_STATES = ("idle", "speak")
REJECT_REASONS = ("self_speech", "self_media", "not_addressed", "lost_arbitration", "no_backlog")
COMMANDS = "state <state> | reject [reason] | tone | status | quit"

# The keys a run must carry for the result to be computed over it; an older file lacking one is
# not this listener's result.
_RUN_KEYS = frozenset(
    {"memory", "wakes", "playbacks", "first_voice_start_ms", "frames_without_wake"}
)

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class SpikeListenerError(MistError):
    """The devices dir or the result file cannot be used, so the listener does not start."""


class DeviceStatsMessage(BaseModel):
    """The spike build's `device_stats`, sent every 10 s; not part of protocol 2."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    type: Literal["device_stats"]
    heap_free_min: int = Field(ge=0)
    psram_free: int = Field(ge=0)
    mic_overruns: int = Field(ge=0)
    ring_overruns: int = Field(ge=0)
    playback_underruns: int = Field(ge=0)
    uptime_s: int = Field(ge=0)
    reconnects: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class StatsSample:
    """One `device_stats`, with when it arrived on the listener's clock."""

    at: float
    stats: DeviceStatsMessage


@dataclass(slots=True)
class WakeRecord:
    """What one wake's uplink delivered; counts and times only."""

    at: float
    keyword: str
    score: float
    backlog_ms: int
    frames: int = 0
    backlog_frames: int = 0
    backlog_contiguous: bool = True
    last_backlog_at: float | None = None
    live_frames: int = 0
    live_gaps: int = 0
    live_missing: int = 0
    live_resets: int = 0
    last_seq: int | None = None
    closed_by: str | None = None
    frames_after_close: int = 0

    @property
    def expected_backlog_frames(self) -> int:
        """The frames `backlog_ms` promises, at 20 ms each."""
        return self.backlog_ms // UPLINK_FRAME_MS


@dataclass(slots=True)
class PlaybackRecord:
    """One tone, and the uplink received while it played."""

    audio_tag: int
    turn_id: str
    start: float
    frames_sent: int = 0
    completed: bool = False
    ended: float | None = None
    played_ms: int | None = None
    barge_ins: int = 0
    frames_in_window: int = 0
    live_frames: int = 0
    seq_gaps: int = 0
    missing_frames: int = 0
    seq_resets: int = 0
    last_seq: int | None = None

    def covers(self, at: float) -> bool:
        """Whether an uplink frame arriving at `at` counts as sent during this tone."""
        return self.start <= at <= self.start + TONE_S + PLAYBACK_GRACE_S


def _seq_step(last: int | None, seq: int) -> Literal["first", "next", "gap", "reset"]:
    if last is None:
        return "first"
    if seq == last + 1:
        return "next"
    return "gap" if seq > last + 1 else "reset"


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


class RunRecorder:
    """Everything one run of the listener counts, on an injected monotonic clock."""

    def __init__(self, clock: Clock, started_at: datetime) -> None:
        self._clock = clock
        self._started_at = started_at
        self._start = clock()
        self.device: str | None = None
        self.auth_refused = 0
        self.connections = 0
        self.disconnect_codes: list[int | None] = []
        self.protocol_errors = 0
        self.messages: Counter[str] = Counter()
        self.first_voice_start_ms: int | None = None
        self.uplink_frames = 0
        self.uplink_bytes = 0
        self.frames_without_wake = 0
        self.wakes: list[WakeRecord] = []
        self.stats: list[StatsSample] = []
        self.playbacks: list[PlaybackRecord] = []
        self._open_wake: WakeRecord | None = None

    @property
    def wakes_opened(self) -> int:
        """How many `wake` messages this run has recorded."""
        return len(self.wakes)

    @property
    def stats_samples(self) -> int:
        """How many `device_stats` this run has recorded."""
        return len(self.stats)

    @property
    def open_wake(self) -> WakeRecord | None:
        """The wake whose uplink frames are arriving now, if any."""
        return self._open_wake

    def listener_started(self) -> None:
        """Restart the run's clock at the moment the socket is bound."""
        self._start = self._clock()

    def refused(self) -> None:
        """Count one connection refused for its token."""
        self.auth_refused += 1

    def connected(self, device: str) -> None:
        """Count an authenticated connection; a new connection opens with no wake."""
        self.device = device
        self.connections += 1
        self._open_wake = None

    def disconnected(self, code: int | None) -> None:
        """End the connection, and with it any open wake."""
        self.disconnect_codes.append(code)
        self._open_wake = None

    def message(self, kind: str) -> None:
        """Count one control message by type."""
        self.messages[kind] += 1

    def protocol_error(self) -> None:
        """Count one frame or message the protocol refused."""
        self.protocol_errors += 1

    def voice_start(self) -> None:
        """Note the first valid `voice_start` of the run, the reconnection measurement."""
        if self.first_voice_start_ms is None:
            self.first_voice_start_ms = _ms(self._clock() - self._start)

    def wake(self, message: WakeMessage) -> None:
        """Open a wake; the previous one, if still open, ends here."""
        record = WakeRecord(
            at=self._clock(),
            keyword=message.keyword,
            score=message.score,
            backlog_ms=message.backlog_ms,
        )
        self.wakes.append(record)
        self._open_wake = record

    def closing_command(self, command: str) -> None:
        """Mark the open wake closed by an operator command; later frames still count."""
        wake = self._open_wake
        if wake is not None and wake.closed_by is None:
            wake.closed_by = command

    def uplink_frame(self, frame: Frame) -> None:
        """Count one `0x10` frame; its payload is not kept."""
        now = self._clock()
        self.uplink_frames += 1
        self.uplink_bytes += len(frame.payload)
        wake = self._open_wake
        live = False
        if wake is None:
            self.frames_without_wake += 1
        else:
            live = self._wake_frame(wake, frame.seq, now)
        for playback in self.playbacks:
            if playback.covers(now):
                self._playback_frame(playback, frame.seq, live)

    def _wake_frame(self, wake: WakeRecord, seq: int, now: float) -> bool:
        previous = wake.last_seq
        step = _seq_step(previous, seq)
        wake.last_seq = seq
        wake.frames += 1
        if wake.closed_by is not None:
            wake.frames_after_close += 1
        if wake.frames <= wake.expected_backlog_frames:
            wake.backlog_frames += 1
            if step not in ("first", "next"):
                wake.backlog_contiguous = False
            if wake.frames == wake.expected_backlog_frames:
                wake.last_backlog_at = now
            return False
        wake.live_frames += 1
        if step == "gap" and previous is not None:
            wake.live_gaps += 1
            wake.live_missing += seq - previous - 1
        elif step == "reset":
            wake.live_resets += 1
        return True

    def _playback_frame(self, playback: PlaybackRecord, seq: int, live: bool) -> None:
        step = _seq_step(playback.last_seq, seq)
        if step == "gap" and playback.last_seq is not None:
            playback.seq_gaps += 1
            playback.missing_frames += seq - playback.last_seq - 1
        elif step == "reset":
            playback.seq_resets += 1
        playback.last_seq = seq
        playback.frames_in_window += 1
        if live:
            playback.live_frames += 1

    def device_stats(self, message: DeviceStatsMessage) -> None:
        """Record one `device_stats` sample."""
        self.stats.append(StatsSample(self._clock(), message))

    def tone_started(self, audio_tag: int, turn_id: str) -> PlaybackRecord:
        """Open a tone's record at the moment its `stream_start` went out."""
        playback = PlaybackRecord(audio_tag=audio_tag, turn_id=turn_id, start=self._clock())
        self.playbacks.append(playback)
        return playback

    def tone_finished(self, playback: PlaybackRecord, completed: bool) -> None:
        """Close a tone's record; `completed` when every frame and the `0x02` went out."""
        playback.completed = completed
        playback.ended = self._clock()

    def playback_done(self, message: PlaybackDoneMessage) -> None:
        """Note the device's `played_ms` on the tone it names."""
        for playback in self.playbacks:
            if playback.turn_id == message.turn_id:
                playback.played_ms = message.played_ms

    def barge_in(self) -> None:
        """Count a `barge_in` against the tone playing, if one is."""
        if self.playbacks and self.playbacks[-1].ended is None:
            self.playbacks[-1].barge_ins += 1

    def _t(self, at: float) -> float:
        return round(at - self._start, 3)

    def _wake_json(self, wake: WakeRecord) -> dict[str, object]:
        delivery = None if wake.last_backlog_at is None else _ms(wake.last_backlog_at - wake.at)
        failures = []
        if wake.backlog_ms < BACKLOG_MIN_MS:
            failures.append("backlog_ms")
        if delivery is None:
            failures.append("backlog_incomplete")
        if not wake.backlog_contiguous:
            failures.append("backlog_seq")
        if delivery is not None and delivery > BACKLOG_DELIVERY_MS:
            failures.append("backlog_late")
        return {
            "t_s": self._t(wake.at),
            "keyword": wake.keyword,
            "score": wake.score,
            "backlog_ms": wake.backlog_ms,
            "backlog_frames_expected": wake.expected_backlog_frames,
            "backlog_frames": wake.backlog_frames,
            "backlog_seq_contiguous": wake.backlog_contiguous,
            "wake_to_last_backlog_ms": delivery,
            "live_frames": wake.live_frames,
            "live_seq_gaps": wake.live_gaps,
            "live_missing_frames": wake.live_missing,
            "live_seq_resets": wake.live_resets,
            "closed_by": wake.closed_by,
            "frames_after_close": wake.frames_after_close,
            "failures": failures,
            "pass": not failures,
        }

    def _underruns(self, playback: PlaybackRecord) -> int | None:
        before = [s for s in self.stats if s.at <= playback.start]
        after = [s for s in self.stats if s.at >= playback.start + TONE_S]
        if not before or not after:
            return None
        delta = after[0].stats.playback_underruns - before[-1].stats.playback_underruns
        return delta if delta >= 0 else None  # a reboot between the two resets the counter

    def _playback_json(self, playback: PlaybackRecord) -> dict[str, object]:
        underruns = self._underruns(playback)
        passed = (
            playback.completed
            and playback.live_frames >= PLAYBACK_LIVE_FRAMES
            and playback.seq_gaps == 0
            and playback.seq_resets == 0
            and underruns == 0
        )
        return {
            "t_s": self._t(playback.start),
            "audio_tag": playback.audio_tag,
            "turn_id": playback.turn_id,
            "frames_sent": playback.frames_sent,
            "completed": playback.completed,
            "duration_s": (
                None if playback.ended is None else round(playback.ended - playback.start, 3)
            ),
            "played_ms": playback.played_ms,
            "barge_ins": playback.barge_ins,
            "uplink_frames": playback.frames_in_window,
            "live_frames": playback.live_frames,
            "seq_gaps": playback.seq_gaps,
            "missing_frames": playback.missing_frames,
            "seq_resets": playback.seq_resets,
            "underruns": underruns,
            "pass": passed,
        }

    def _memory_json(self) -> dict[str, object]:
        if not self.stats:
            return {"window_s": None, "heap_free_min": None, "psram_free_min": None}
        return {
            "window_s": round(self.stats[-1].at - self.stats[0].at, 3),
            "heap_free_min": min(s.stats.heap_free_min for s in self.stats),
            "psram_free_min": min(s.stats.psram_free for s in self.stats),
        }

    def snapshot(self) -> dict[str, object]:
        """This run as the result file records it: counts and timings, never audio."""
        return {
            "started_at": self._started_at.isoformat(timespec="seconds"),
            "duration_s": self._t(self._clock()),
            "device": self.device,
            "auth_refused": self.auth_refused,
            "connections": self.connections,
            "disconnect_codes": list(self.disconnect_codes),
            "protocol_errors": self.protocol_errors,
            "messages": dict(sorted(self.messages.items())),
            "first_voice_start_ms": self.first_voice_start_ms,
            "uplink_frames": self.uplink_frames,
            "uplink_bytes": self.uplink_bytes,
            "frames_without_wake": self.frames_without_wake,
            "wakes": [self._wake_json(wake) for wake in self.wakes],
            "playbacks": [self._playback_json(playback) for playback in self.playbacks],
            "memory": self._memory_json(),
            "device_stats": [
                {"t_s": self._t(sample.at), **sample.stats.model_dump(exclude={"type"})}
                for sample in self.stats
            ],
        }


# -- G1 over every run -------------------------------------------------------------------------


def _memory_check(runs: Sequence[dict], key: str, threshold: int) -> dict[str, object]:
    observed = [run["memory"][key] for run in runs if run["memory"][key] is not None]
    windows = [run["memory"]["window_s"] or 0.0 for run in runs]
    qualifying = [
        index
        for index, run in enumerate(runs)
        if windows[index] >= MEMORY_WINDOW_S
        and any(playback["completed"] for playback in run["playbacks"])
    ]
    observed_min = min(observed, default=None)
    return {
        "measured_as": (
            f"the lowest `{key}` in any `device_stats` of any run; it passes when that is at "
            f"least the threshold and some run's samples span {MEMORY_WINDOW_S:.0f} s with a "
            "completed tone"
        ),
        "threshold_b": threshold,
        "observed_b": observed_min,
        "required_window_s": MEMORY_WINDOW_S,
        "longest_window_s": max(windows, default=0.0),
        "qualifying_runs": qualifying,
        "pass": observed_min is not None and observed_min >= threshold and bool(qualifying),
    }


def _wakes_check(runs: Sequence[dict]) -> dict[str, object]:
    wakes = [wake for run in runs for wake in run["wakes"]]
    failing = [
        {"run": run_index, "wake": wake_index, "failures": wake["failures"]}
        for run_index, run in enumerate(runs)
        for wake_index, wake in enumerate(run["wakes"])
        if not wake["pass"]
    ]
    return {
        "measured_as": (
            f"every wake in every run carries `backlog_ms` of at least {BACKLOG_MIN_MS}, then "
            "that many ms of frames with `seq` rising by one, the last within "
            f"{BACKLOG_DELIVERY_MS} ms of the `wake`; at least {WAKES_REQUIRED} wakes"
        ),
        "required": WAKES_REQUIRED,
        "recorded": len(wakes),
        "passing": len(wakes) - len(failing),
        "failing": failing,
        "pass": len(wakes) >= WAKES_REQUIRED and not failing,
    }


def _playback_check(runs: Sequence[dict]) -> dict[str, object]:
    passing = [
        {"run": run_index, "playback": index}
        for run_index, run in enumerate(runs)
        for index, playback in enumerate(run["playbacks"])
        if playback["pass"]
    ]
    return {
        "measured_as": (
            f"a completed {TONE_S:.0f} s tone during which at least {PLAYBACK_LIVE_FRAMES} live "
            "uplink frames arrived with no `seq` gap or reset, and `playback_underruns` did not "
            "rise between the samples either side of it"
        ),
        "required_live_frames": PLAYBACK_LIVE_FRAMES,
        "playbacks": sum(len(run["playbacks"]) for run in runs),
        "passing": passing,
        "pass": bool(passing),
    }


def _reconnection_check(runs: Sequence[dict]) -> dict[str, object]:
    restarts = [run["first_voice_start_ms"] for run in runs[1:]]
    return {
        "measured_as": (
            "per run after the first: from the listener's start (its socket bound) to the "
            "device's first authenticated, valid `voice_start`; null when none arrived"
        ),
        "limit_ms": RECONNECT_LIMIT_MS,
        "restarts_ms": restarts,
        "pass": bool(restarts)
        and all(value is not None and value <= RECONNECT_LIMIT_MS for value in restarts),
    }


def build_result(runs: Sequence[dict]) -> dict[str, object]:
    """The result file: G1's checks, each with a pass or fail, over every run, and the runs."""
    checks = {
        "heap_free_min": _memory_check(runs, "heap_free_min", HEAP_FREE_MIN_B),
        "psram_free_min": _memory_check(runs, "psram_free_min", PSRAM_FREE_MIN_B),
        "wakes": _wakes_check(runs),
        "uplink_during_playback": _playback_check(runs),
        "reconnection": _reconnection_check(runs),
    }
    without_wake = sum(run["frames_without_wake"] for run in runs)
    return {
        "schema": RESULT_FILE_SCHEMA,
        "note": "counts and timings only; the listener keeps no audio",
        "g1": {"pass": all(check["pass"] for check in checks.values()), "checks": checks},
        "g3": {
            "measured_as": "uplink frames that arrived with no wake open on their connection",
            "frames_without_wake": without_wake,
            "pass": without_wake == 0,
        },
        "runs": list(runs),
    }


class ResultFile:
    """`spike_result.json`: the earlier runs it held, and the one file this listener writes."""

    def __init__(self, path: Path, previous_runs: list[dict]) -> None:
        self._path = path
        self._previous = previous_runs

    @classmethod
    def open(cls, path: Path, *, fresh: bool) -> ResultFile:
        """Read the runs already at `path`, unless `fresh`; writes nothing.

        Raises:
            SpikeListenerError: When `path` exists, `fresh` is not set, and it is not a spike
                result this listener can append to.
        """
        if fresh or not path.exists():
            return cls(path, [])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise SpikeListenerError(f"cannot read {path}: {exc}") from exc
        runs = payload.get("runs") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != RESULT_FILE_SCHEMA
            or not isinstance(runs, list)
            or not all(isinstance(run, dict) and run.keys() >= _RUN_KEYS for run in runs)
        ):
            raise SpikeListenerError(
                f"{path} is not a {RESULT_FILE_SCHEMA} result; move it or pass --fresh"
            )
        return cls(path, runs)

    def write(self, current_run: dict) -> None:
        """Replace the file with the result over the earlier runs and `current_run`."""
        text = json.dumps(build_result([*self._previous, current_run]), indent=2) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(text)
            os.replace(temporary, self._path)
        except OSError:
            Path(temporary).unlink(missing_ok=True)
            raise


def load_devices_dir(directory: Path) -> tuple[DeviceRegistry, ssl.SSLContext]:
    """The registry and the server's TLS context from `devices.py`'s devices dir.

    Raises:
        DeviceRegistryError: When `registry.json` is missing or malformed.
        SpikeListenerError: When `tls/server.pem` or `tls/server.key` cannot be loaded.
    """
    registry = DeviceRegistry.load(directory / REGISTRY_FILE)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    tls = directory / TLS_DIR
    try:
        context.load_cert_chain(tls / SERVER_CERT, tls / SERVER_KEY)
    except (OSError, ssl.SSLError) as exc:
        raise SpikeListenerError(f"cannot load the server certificate in {tls}: {exc}") from exc
    return registry, context


# -- Messages the listener sends ---------------------------------------------------------------


def _session_started(session_id: str) -> str:
    return encode_message(
        SessionStartedMessage(
            session_id=session_id,
            protocol_version=PROTOCOL_VERSION,
            mist_state="idle",
            capabilities=SessionCapabilities(
                tts_enabled=True,
                vad_enabled=False,
                voice=VoiceCapabilities(
                    protocol=2,
                    modes=["wake"],
                    spotters=["endpoint"],
                    reference=False,
                    uplink_sample_rate=UPLINK_SAMPLE_RATE,
                    downlink_sample_rate=DOWNLINK_SAMPLE_RATE,
                    barge_in="keyword",
                    tts=True,
                    stt=False,
                    eot=False,
                ),
            ),
        )
    )


def _voice_state(
    *,
    cause: Literal["answer", "broadcast"],
    req: int | None,
    active: bool,
    muted: bool,
    refused: str | None,
) -> str:
    off = EngineStatus(loaded=False, location="off", model_id=None, vram_mib=None)
    return encode_message(
        VoiceStateMessage(
            cause=cause,
            req=req,
            active=active,
            mode="wake" if active else None,
            uplink=StreamFormat(sample_rate=UPLINK_SAMPLE_RATE, frame_ms=UPLINK_FRAME_MS),
            downlink=StreamFormat(sample_rate=DOWNLINK_SAMPLE_RATE, frame_ms=TONE_FRAME_MS),
            playback=PlaybackSettings(client_prebuffer_ms=80, fade_ms=30, progress_interval_ms=100),
            barge_in=BargeInSettings(policy="keyword", keywords=["stop"], arm_delay_ms=300),
            budget=LatencyBudget(preset="spike"),
            engines=EngineStatuses(vad=off, eot=off, stt=off, tts=off, kws=off),
            tts_voice=TtsVoice(engine="tone", voice="tone", profile="spike"),
            vram_budget_mib=0,
            power_level="full",
            muted=muted,
            refused=refused,
        )
    )


def tone_frame_payload() -> bytes:
    """40 ms of the tone as 24 kHz PCM16 little-endian, generated in memory."""
    samples = DOWNLINK_SAMPLE_RATE * TONE_FRAME_MS // 1000
    return struct.pack(
        f"<{samples}h",
        *(
            round(TONE_AMPLITUDE * math.sin(2 * math.pi * TONE_HZ * n / DOWNLINK_SAMPLE_RATE))
            for n in range(samples)
        ),
    )


def _message_type(raw: str) -> str | None:
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    kind = data.get("type") if isinstance(data, dict) else None
    return kind if isinstance(kind, str) else None


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SpikeListener:
    """The `/v2/device` server, its operator commands, and the run it records."""

    def __init__(
        self,
        *,
        registry: DeviceRegistry,
        ssl_context: ssl.SSLContext,
        result: ResultFile,
        clock: Clock,
        sleep: Sleep,
        wall_clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._registry = registry
        self._ssl = ssl_context
        self._result = result
        self._clock = clock
        self._sleep = sleep
        self.recorder = RunRecorder(clock, wall_clock())
        self.stop_requested = asyncio.Event()
        self._server: Server | None = None
        self._authenticated: dict[ServerConnection, DeviceRecord] = {}
        self._active: ServerConnection | None = None
        self._tone_task: asyncio.Task[None] | None = None
        self._next_tag = 1
        self._tones = 0
        self._sessions = 0
        self._tone_payload = tone_frame_payload()

    @property
    def tone_playing(self) -> bool:
        """Whether a tone is streaming now."""
        return self._tone_task is not None and not self._tone_task.done()

    async def start(self, host: str, port: int) -> int:
        """Bind and serve; returns the bound port, and writes the result once.

        Raises:
            OSError: When the port cannot be bound.
        """
        self._server = await serve(
            self._handle,
            host,
            port,
            ssl=self._ssl,
            process_request=self._process_request,
            max_size=WS_MAX_SIZE,
        )
        self.recorder.listener_started()
        self.write_result()
        return self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """Stop the tone and the server, then write the result."""
        if self._tone_task is not None:
            self._tone_task.cancel()
            await asyncio.gather(self._tone_task, return_exceptions=True)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.write_result()

    def snapshot(self) -> dict[str, object]:
        """This run as the result file records it."""
        return self.recorder.snapshot()

    def write_result(self) -> None:
        """Rewrite the result file with this run as it stands."""
        self._result.write(self.recorder.snapshot())

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        if urlsplit(request.path).path != DEVICE_PATH:
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        self._registry.reload_if_changed()
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        record = self._registry.verify(token) if scheme == "Bearer" and token else None
        if record is None:
            self.recorder.refused()
            logger.warning(
                "spike_auth_refused device=%s peer=%s",
                claimed_name(token) or "?",
                connection.remote_address,
            )
            return connection.respond(HTTPStatus.FORBIDDEN, "refused\n")
        self._authenticated[connection] = record
        return None

    async def _handle(self, connection: ServerConnection) -> None:
        record = self._authenticated.pop(connection)
        previous = self._active
        if previous is not None:
            await previous.close(4001, "replaced")
        self._active = connection
        self.recorder.connected(record.name)
        self._sessions += 1
        logger.info("spike_device_connected device=%s", record.name)
        try:
            await connection.send(_session_started(f"spike-{self._sessions}"))
            async for message in connection:
                if isinstance(message, bytes):
                    self._on_binary(message)
                else:
                    await self._on_text(connection, message)
        except ConnectionClosed:
            pass
        finally:
            if self._active is connection:
                self._active = None
            self.recorder.disconnected(connection.close_code)
            logger.info(
                "spike_device_disconnected device=%s code=%s", record.name, connection.close_code
            )

    def _on_binary(self, data: bytes) -> None:
        try:
            frame = decode_frame(data, direction=Direction.UP)
        except ProtocolError:
            self.recorder.protocol_error()
            return
        if frame.msg_type is not MsgType.MIC_AUDIO:
            self.recorder.protocol_error()
            return
        self.recorder.uplink_frame(frame)

    async def _on_text(self, connection: ServerConnection, raw: str) -> None:
        kind = _message_type(raw)
        if kind is not None:
            self.recorder.message(kind)
        if kind == "device_stats":
            try:
                stats = DeviceStatsMessage.model_validate_json(raw, strict=True)
            except ValidationError:
                self.recorder.protocol_error()
                return
            self.recorder.device_stats(stats)
            return
        try:
            message = parse_client_message(raw)
        except ProtocolError:
            self.recorder.protocol_error()
            peek = peek_voice_start(raw)
            if peek is not None:
                await connection.send(
                    _voice_state(
                        cause="answer", req=peek.req, active=False, muted=False, refused="invalid"
                    )
                )
            return
        if isinstance(message, VoiceStartMessage):
            self.recorder.voice_start()
            await connection.send(
                _voice_state(
                    cause="answer", req=message.req, active=True, muted=message.muted, refused=None
                )
            )
        elif isinstance(message, WakeMessage):
            self.recorder.wake(message)
        elif isinstance(message, MuteMessage):
            await connection.send(
                _voice_state(
                    cause="broadcast", req=None, active=True, muted=message.active, refused=None
                )
            )
        elif isinstance(message, PlaybackDoneMessage):
            self.recorder.playback_done(message)
        elif message.type == "barge_in":
            self.recorder.barge_in()

    async def command(self, line: str) -> str:
        """Run one operator command and return its one-line reply."""
        verb, _, rest = line.strip().partition(" ")
        argument = rest.strip()
        if verb == "quit":
            self.stop_requested.set()
            return "stopping"
        if verb == "status":
            return self._status()
        if verb not in ("state", "reject", "tone"):
            return f"unknown command: {line.strip()} (commands: {COMMANDS})"
        connection = self._active
        if connection is None:
            return "refused: no device connected"
        if verb == "state":
            if argument not in STATES:
                return f"refused: state is one of {', '.join(STATES)}"
            sent = await self._send(connection, encode_message(StateCycleMessage(state=argument)))
            if sent and argument in UPLINK_CLOSING_STATES:
                self.recorder.closing_command(f"state_cycle {argument}")
            return f"sent state_cycle {argument}" if sent else "refused: device disconnected"
        if verb == "reject":
            reason = argument or "not_addressed"
            if reason not in REJECT_REASONS:
                return f"refused: reason is one of {', '.join(REJECT_REASONS)}"
            wake = self.recorder.open_wake
            message = VoiceDecisionWakeRejected(
                turn_id=None,
                reason=reason,
                detail=WakeRejectedDetail(
                    source="endpoint",
                    score=wake.score if wake is not None else 0.0,
                    reference_score=None,
                ),
            )
            sent = await self._send(connection, encode_message(message))
            if sent:
                self.recorder.closing_command("wake_rejected")
            return f"sent wake_rejected {reason}" if sent else "refused: device disconnected"
        if self.tone_playing:
            return "refused: a tone is playing"
        tag = self._next_tag
        self._next_tag = next_turn_tag(tag)
        self._tones += 1
        self._tone_task = asyncio.create_task(
            self._play_tone(connection, tag, f"spike-tone-{self._tones}")
        )
        return f"tone started (audio_tag {tag}, {TONE_S:.0f} s)"

    async def _send(self, connection: ServerConnection, text: str) -> bool:
        try:
            await connection.send(text)
        except ConnectionClosed:
            return False
        return True

    async def _play_tone(self, connection: ServerConnection, tag: int, turn_id: str) -> None:
        start_message = StreamStartMessage(
            turn_id=turn_id, channel="tts", origin="system", audio_tag=tag
        )
        if not await self._send(connection, encode_message(start_message)):
            return
        playback = self.recorder.tone_started(tag, turn_id)
        frame_s = TONE_FRAME_MS / 1000
        frames = round(TONE_S / frame_s)
        completed = False
        try:
            for index in range(frames):
                delay = playback.start + index * frame_s - TONE_LEAD_S - self._clock()
                if delay > 0:
                    await self._sleep(delay)
                chunk = Frame(
                    MsgType.AUDIO_CHUNK, tag, index, DOWNLINK_SAMPLE_RATE, self._tone_payload
                )
                await connection.send(encode_frame(chunk))
                playback.frames_sent += 1
            end = Frame(MsgType.AUDIO_COMPLETE, tag, frames, DOWNLINK_SAMPLE_RATE)
            await connection.send(encode_frame(end))
            completed = True
        except ConnectionClosed:
            pass
        finally:
            self.recorder.tone_finished(playback, completed)

    def _status(self) -> str:
        recorder = self.recorder
        return (
            f"device={recorder.device if self._active is not None else 'none'} "
            f"wakes={recorder.wakes_opened} uplink_frames={recorder.uplink_frames} "
            f"without_wake={recorder.frames_without_wake} stats={recorder.stats_samples} "
            f"tone={'playing' if self.tone_playing else 'idle'} "
            f"refused={recorder.auth_refused}"
        )


async def read_commands(
    reader: asyncio.StreamReader, listener: SpikeListener, reply: Callable[[str], None]
) -> None:
    """Run commands line by line until `quit` or the end of input.

    The end of input stops the commands, not the listener: a container started without `-i`
    has an empty stdin and still serves.
    """
    while not listener.stop_requested.is_set():
        line = await reader.readline()
        if not line:
            return
        text = line.decode("utf-8", errors="replace").strip()
        if text:
            reply(await listener.command(text))


async def _stdin_commands(listener: SpikeListener) -> None:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    try:
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    except (OSError, ValueError, NotImplementedError) as exc:
        print(f"spike_listener: no commands, stdin is not readable: {exc}", flush=True)
        return
    await read_commands(reader, listener, lambda text: print(text, flush=True))


async def _write_periodically(listener: SpikeListener) -> None:
    while True:
        await asyncio.sleep(RESULT_WRITE_INTERVAL_S)
        listener.write_result()


async def _serve(
    args: argparse.Namespace,
    registry: DeviceRegistry,
    context: ssl.SSLContext,
    result: ResultFile,
) -> None:
    listener = SpikeListener(
        registry=registry,
        ssl_context=context,
        result=result,
        clock=asyncio.get_running_loop().time,
        sleep=asyncio.sleep,
    )
    port = await listener.start(args.host, args.port)
    print(
        f"spike_listener: serving wss://{args.host}:{port}{DEVICE_PATH}, writing {args.out}; "
        f"commands on stdin: {COMMANDS}",
        flush=True,
    )
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        # Windows' loop has no signal handlers; Ctrl+C still ends the run there.
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signum, listener.stop_requested.set)
    tasks = [
        asyncio.create_task(_write_periodically(listener)),
        asyncio.create_task(_stdin_commands(listener)),
    ]
    try:
        await listener.stop_requested.wait()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await listener.stop()
        print(f"spike_listener: stopped; result in {args.out}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """The command line, as T4's README and H4 run it."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.room_endpoint.spike_listener",
        description="Serve /v2/device to one room device and measure it against G1.",
    )
    parser.add_argument("--devices", required=True, help="the devices dir, read only")
    parser.add_argument("--out", required=True, help="the result file, spike_result.json")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=BIND_ALL, help="the address to bind in the container")
    parser.add_argument("--fresh", action="store_true", help="discard the runs already in --out")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Serve until `quit` or a signal; see the module docstring for the exit codes."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        registry, context = load_devices_dir(Path(args.devices))
        result = ResultFile.open(Path(args.out), fresh=args.fresh)
    except MistError as exc:
        print(f"spike_listener: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        asyncio.run(_serve(args, registry, context, result))
    except OSError as exc:
        print(f"spike_listener: failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        pass
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
