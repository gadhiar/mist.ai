"""Per-session audio ingress: uplink frames to speech segments, name hits and utterances.

Every engine call runs on an inference lane (`lanes.py`), so ingress itself only keeps state and
schedules work. One worker task consumes the frames in arrival order: it cuts them into
512-sample VAD windows, marks speech segments, routes frames into the open utterance's
transcriber stream, and feeds the name spotter on the `kws` lane every frame the ring holds,
once and in order, reporting names only from a segment and its pre-roll (`spotter_feed`).
Partial decodes, the final decode and the end-of-turn score run as their own tasks, so silence
keeps being measured while they are in flight, and a resume can cancel them.

Ingress listens in one of two ways (Design, wake by name, item 1). Dormant, VAD marks segments
and the spotter hears them, but nothing is transcribed. Engaged, VAD speech also opens an
utterance. A wake utterance is opened by the controller through `open_utterance`, from the
`from_pos` of a `NameHeard`, with backfill from the ring of recent frames.

Audio is kept, stamped and cut by the session's frame position, never by the uplink `seq`, which
only counts gaps. Each position is one 20 ms frame. Events go to the `emit` callback; ingress
sends nothing to the client itself. Nothing here logs transcript text: an utterance the addressee
gate rejects was never meant for MIST, so its words stay out of the logs.
"""

import asyncio
import bisect
import logging
from collections import OrderedDict, deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

import numpy as np

from .budget import EndpointingConfig, TurnConfig, WakeConfig
from .errors import RealtimeError, SpeechEngineError
from .lanes import LaneJobCancelled
from .protocol import Frame, Spotter
from .speech import (
    CancelToken,
    NameHit,
    NameSpotter,
    NameSpotterStream,
    StreamingTranscriber,
    TranscriptStream,
    TranscriptUpdate,
    TurnEndDetector,
    VadStream,
    VoiceActivityDetector,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

FRAME_MS = 20
"""The audio one uplink position covers."""

VAD_WINDOW_SAMPLES = 512
MAX_RETENTION_MS = 10000
"""The most audio the ring may hold: the wire's longest wake backlog."""

KWS_BACKLOG_FRAMES = 50
"""The most frames released to the name spotter that may wait behind its running job: 1000 ms."""

_SAMPLE_RATE = 16000
_RETENTION_HEADROOM_MS = 1000
_PCM16_SCALE = 32768.0

EndReason = Literal["semantic", "silence_cap", "silence_cap_no_model"]
DropReason = Literal[
    "noise",
    "no_speech",
    "stt_error",
    "text_input",
    "wake_lost",
    "not_addressed",
    "voice_stop",
    "muted",
]
ControllerDropReason = Literal["text_input", "wake_lost", "not_addressed"]


def ring_retention_ms(turn: TurnConfig, wake: WakeConfig) -> int:
    """How much uplink the ring keeps: enough for a teardown's wait and for a wake's lookback.

    The tool grace covers the longest wait inside a teardown, and the lookback is how far back a
    name hit may open its utterance; each gets a second of headroom. 7000 ms at the defaults.
    """
    return min(
        MAX_RETENTION_MS,
        max(
            turn.tool_cancel_grace_ms + _RETENTION_HEADROOM_MS,
            wake.lookback_ms + _RETENTION_HEADROOM_MS,
        ),
    )


# -- Speech segments ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentStep:
    """What one VAD window did: a segment it opened, or the one its silence closed.

    Each is `(onset frame, last speech frame)`, inclusive. One window never does both.
    """

    started: tuple[int, int] | None
    closed: tuple[int, int] | None


_NO_STEP = SegmentStep(started=None, closed=None)


class SpeechSegmentTracker:
    """VAD windows to speech segments: the one place ingress's run and silence rules live.

    A segment opens after `speech_start_windows` consecutive speech windows, at the first frame
    of the run, and closes once `endpoint_silence_ms` of windows without speech follow its last
    speech. The wake harness drives the same tracker, so it scores audio as ingress hears it.
    """

    __slots__ = ("_cfg", "_run", "_run_start", "_silence_samples", "_open")

    def __init__(self, cfg: EndpointingConfig) -> None:
        self._cfg = cfg
        self._run = 0
        self._run_start = 0
        self._silence_samples = 0
        self._open: tuple[int, int] | None = None

    def window(self, speech: bool, first: int, last: int) -> SegmentStep:
        """Take one window's verdict; `first` and `last` are the frames the window spans."""
        segment = self._open
        if speech and segment is not None:
            self._open = (segment[0], last)
            self._silence_samples = 0
            return _NO_STEP
        self._silence_samples += VAD_WINDOW_SAMPLES
        if not speech:
            self._run = 0
            if segment is not None and self.silence_ms() >= self._cfg.endpoint_silence_ms:
                self._open = None
                return SegmentStep(started=None, closed=segment)
            return _NO_STEP
        if self._run == 0:
            self._run_start = first
        self._run += 1
        if self._run < self._cfg.speech_start_windows:
            return _NO_STEP
        self._run = 0
        self._silence_samples = 0
        self._open = (self._run_start, last)
        return SegmentStep(started=self._open, closed=None)

    @property
    def open_segment(self) -> tuple[int, int] | None:
        """The segment still open, with its last speech frame so far, or None."""
        return self._open

    def silence_ms(self) -> int:
        """The silence since the last speech window, or since the last restart."""
        return self._silence_samples * 1000 // _SAMPLE_RATE

    def restart_silence(self) -> None:
        """Count the pause from now: ingress does so when a lever opens with no segment open."""
        self._silence_samples = 0

    def reset(self) -> tuple[int, int] | None:
        """Forget the run and the open segment, which is returned, as when the VAD restarts.

        The silence count is kept, as ingress's VAD restart on a mute always kept it.
        """
        segment = self._open
        self._open = None
        self._run = 0
        return segment

    def earliest_onset(self, next_unscored: int) -> int:
        """The earliest frame a segment not yet open can start at.

        A segment opens at the first frame of its run, so while a run is under way that is the
        run's first frame; with none, a run can only begin in a window not yet scored, which
        starts at or after `next_unscored`, the first frame the VAD has not scored.
        """
        return self._run_start if self._run else next_unscored


@dataclass(frozen=True, slots=True)
class SpotterFeed:
    """The frames one `spotter_feed` call releases: `warm` first, then `push`, both in order."""

    warm: range
    push: range

    @property
    def end(self) -> int:
        """The first frame after those released, from which the next call starts."""
        return self.push.stop if self.push else self.warm.stop


def _span(start: int, stop: int) -> range:
    return range(start, max(start, stop))


def spotter_feed(
    next_pos: int, newest: int, onset: int | None, earliest_onset: int, pre_roll_frames: int
) -> SpotterFeed:
    """Which frames the name spotter scores next: some warmed, some pushed, none twice.

    The spotter hears every frame, once and in order, as one stream. A streaming model carries
    its state from step to step and runs without a break on the device, so starting it cold at
    each speech onset would score every name from a state it never has there. And fed one
    stretch of audio from one start, the stream finds the hits the server's measured bar counts
    over that audio; it reports the ones in pushed frames, so the hits it reports are a subset
    of the measured ones. Only a speech segment and its pre-roll are pushed, since ingress
    never reports a name heard outside speech; every other frame is warmed, scored and never
    reported.

    `next_pos` is the first frame not yet fed, `newest` the newest routed frame, `onset` the
    open segment's first frame or None, and `earliest_onset` the earliest frame a segment not
    yet open can start at (`SpeechSegmentTracker.earliest_onset`). With a segment open, frames
    before its pre-roll are warmed and the rest pushed through `newest`. With none open,
    frames are warmed only up to `pre_roll_frames` before `earliest_onset`; later ones wait,
    because an onset may still make them pre-roll, and a warmed frame can never be reported.
    While a segment is open nothing waits, so the wait delays no reported hit.
    """
    if onset is not None:
        start = onset - pre_roll_frames
        return SpotterFeed(
            warm=_span(next_pos, min(newest + 1, start)),
            push=_span(max(start, next_pos), newest + 1),
        )
    warm = _span(next_pos, min(newest, earliest_onset - pre_roll_frames - 1) + 1)
    return SpotterFeed(warm=warm, push=range(warm.stop, warm.stop))


def onset_lead_frames(cfg: EndpointingConfig) -> int:
    """The pre-roll, `pre_roll_ms`, in whole frames.

    Ingress opens an utterance, pushes to the spotter and starts a wake chain this many frames
    before a speech onset, and the wake harness reads the same number from here.
    """
    return cfg.pre_roll_ms // FRAME_MS


# -- Events ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """VAD speech opened an utterance in engaged listening; a lever's open emits none."""

    epoch: int
    event_seq: int
    utterance_id: str
    first_pos: int


@dataclass(frozen=True, slots=True)
class SpeechResumed:
    """Speech came back before the end was confirmed, so the utterance continues."""

    epoch: int
    event_seq: int
    utterance_id: str
    first_pos: int


@dataclass(frozen=True, slots=True)
class TranscriptPartial:
    """An interim transcript of the utterance so far.

    `audio_end_pos` is the last frame the decode covered and `heard_at_ms` that frame's arrival.
    `uncovered_since_ms` is the arrival of the oldest frame the previous partial did not cover,
    from which the session measures caption lag. `gate` marks the out-of-cadence decode the
    controller asked for through `through_pos`.
    """

    epoch: int
    event_seq: int
    utterance_id: str
    first_pos: int
    text: str
    stable_chars: int
    heard_at_ms: int
    audio_end_pos: int
    uncovered_since_ms: int
    gate: bool = False
    through_pos: int | None = None


@dataclass(frozen=True, slots=True)
class EndpointCandidate:
    """The final transcript at the first pause, before the end is confirmed (speculative start)."""

    epoch: int
    event_seq: int
    utterance_id: str
    first_pos: int
    text: str
    heard_at_ms: int


@dataclass(frozen=True, slots=True)
class UtteranceEnded:
    """A confirmed end, with why it ended and the final transcript's scores; text is never empty."""

    epoch: int
    event_seq: int
    utterance_id: str
    first_pos: int
    text: str
    reason: EndReason
    eot_probability: float | None
    silence_ms: int
    duration_ms: int
    no_speech_prob: float | None
    avg_logprob: float | None
    heard_at_ms: int


@dataclass(frozen=True, slots=True)
class UtteranceDropped:
    """An utterance closed without a turn.

    Ingress emits the drops it decides (`noise`, `no_speech`, `stt_error`), and `muted` when a
    mute closes the open utterance or a wake is asked for while muted: that drop is the
    controller's signal that the utterance is gone, so the controller has no close of its own to
    make. `drop` and `close` return theirs instead of emitting them, carrying the `event_seq` of
    the last emitted event, so the emitted sequence stays 1, 2, ... with no hole.
    """

    epoch: int
    event_seq: int
    utterance_id: str
    first_pos: int
    reason: DropReason
    text: str | None
    duration_ms: int
    no_speech_prob: float | None
    avg_logprob: float | None


@dataclass(frozen=True, slots=True)
class IngressOverload:
    """Audio was lost before it could be used, on the named lane.

    `vad` is the queue cap, `capture` a backfill reaching past the ring, and `kws` frames
    released to the name spotter and dropped, oldest first, because more than
    `KWS_BACKLOG_FRAMES` waited behind its running job; the spotter's stream then starts a fresh
    context. `kws` is sent once per episode, which ends when a spotter job ends with no frame
    waiting, and `dropped_ms` is the first drop's.
    """

    epoch: int
    event_seq: int
    lane: str
    dropped_ms: int
    utterance_id: str | None
    first_pos: int | None


@dataclass(frozen=True, slots=True)
class NameHeard:
    """A name hit, from the server's spotter or an endpoint's wake (Design, item 2).

    `from_pos` is where a wake utterance would open, `utterance_id` the open utterance the hit
    fell in, and `arrival` the clock time of the hit's last frame (for an endpoint, the `wake`
    message's arrival as the controller gave it).
    """

    epoch: int
    event_seq: int
    hit: NameHit
    arrival: float
    from_pos: int
    utterance_id: str | None
    reference_score: float | None
    spotter: Spotter = "server"


IngressEvent = (
    SpeechStarted
    | SpeechResumed
    | TranscriptPartial
    | EndpointCandidate
    | UtteranceEnded
    | UtteranceDropped
    | IngressOverload
    | NameHeard
)


@dataclass(frozen=True, slots=True)
class OpenedUtterance:
    """The utterance a lever opened or continued."""

    utterance_id: str
    first_pos: int


@dataclass(frozen=True, slots=True)
class IngressSeed:
    """What one ingress hands the next across a reconfigure (M46).

    `frames` are the retained `(position, float32 audio)` pairs, oldest first; `floor` is one past
    the last frame of the last utterance, or one past the newest frame received before the last
    mute, whichever is later; `vad_from` the first frame the VAD has not scored, or None when the
    VAD is not running.
    """

    frames: tuple[tuple[int, np.ndarray], ...]
    floor: int
    vad_from: int | None


@dataclass(frozen=True, slots=True)
class IngressStats:
    """Counters the session reports; none of them holds audio or text.

    `skipped_kws_hops` counts frames, not jobs: the frames the name spotter's backlog bound
    (`KWS_BACKLOG_FRAMES`) dropped unscored. A frame waiting behind a running spotter job is
    not counted; the next job scores it.
    """

    frames_received: int
    gap_frames: int
    muted_frames: int
    ignored_frames: int
    queue_dropped_frames: int
    skipped_partials: int
    skipped_kws_hops: int
    reference_uncovered: int
    reference_frames: int
    engine_errors: int


class Lane(Protocol):
    """What ingress needs from an inference lane; `InferenceLane` is the production one."""

    async def run(self, fn: Callable[..., T], *args: object, token: CancelToken) -> T:
        """Run `fn(*args)` off the event loop; a set `token` fails it with `LaneJobCancelled`."""
        ...


@dataclass(frozen=True, slots=True)
class IngressLanes:
    """The lanes ingress submits to: VAD, STT, end of turn and the name spotter's `kws`."""

    vad: Lane
    stt: Lane
    eot: Lane
    kws: Lane


@dataclass(frozen=True, slots=True)
class IngressEngines:
    """The engines ingress drives; no end-of-turn model or no spotter turns that part off."""

    vad: VoiceActivityDetector
    transcriber: StreamingTranscriber
    turn_end: TurnEndDetector | None
    spotter: NameSpotter | None


# -- Internal state -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Frame:
    pos: int
    seq: int | None
    pcm: np.ndarray
    payload: bytes
    arrival_ms: int


class _EndpointWake:
    __slots__ = ("keyword", "score", "arrival", "remaining", "first_pos", "last_pos")

    def __init__(self, keyword: str, score: float, arrival: float, frames: int) -> None:
        self.keyword = keyword
        self.score = score
        self.arrival = arrival
        self.remaining = frames
        self.first_pos: int | None = None
        self.last_pos: int | None = None


class _Utterance:
    """The open utterance: its stream, the frames it holds and the jobs it has in flight.

    `state` is `speaking` while frames go into the stream and `pending` from the endpoint
    candidate's silence on, when frames are held back so the final decode covers exactly the
    utterance; a resume pushes the held frames and continues. Partial and gate decodes use
    `decode_token`, which only the utterance's close sets, and the final decode and end-of-turn
    score use `end_token`, which a resume also sets.
    """

    __slots__ = (
        "id",
        "first_pos",
        "stream",
        "state",
        "last_pushed",
        "positions",
        "arrivals",
        "audio",
        "held",
        "next_partial_ms",
        "decoding",
        "partial_end",
        "last_partial_text",
        "decode_token",
        "end_token",
        "tasks",
        "end_pos",
        "final",
        "final_end",
        "final_from_gate",
        "decided",
        "eot_running",
        "eot_checked_ms",
        "eot_prob",
        "gate_state",
        "gate_through",
        "gate_end",
        "gate_result",
    )

    def __init__(self, utterance_id: str, first_pos: int, stream: TranscriptStream) -> None:
        self.id = utterance_id
        self.first_pos = first_pos
        self.stream = stream
        self.state: Literal["speaking", "pending"] = "speaking"
        self.last_pushed = first_pos - 1
        self.positions: list[int] = []
        self.arrivals: list[int] = []
        self.audio: list[np.ndarray] = []
        self.held: list[_Frame] = []
        self.next_partial_ms = 0
        self.decoding = False
        self.partial_end = first_pos - 1
        self.last_partial_text: str | None = None
        self.decode_token = CancelToken()
        self.end_token = CancelToken()
        self.tasks: set[asyncio.Task[None]] = set()
        self.end_pos = first_pos - 1
        self.final: TranscriptUpdate | None = None
        self.final_end = first_pos - 1
        self.final_from_gate = False
        self.decided: tuple[EndReason, int] | None = None
        self.eot_running = False
        self.eot_checked_ms = 0
        self.eot_prob: float | None = None
        self.gate_state: Literal["none", "waiting", "running", "done"] = "none"
        self.gate_through = 0
        self.gate_end = first_pos - 1
        self.gate_result: TranscriptUpdate | None = None

    def arrival(self, pos: int) -> int:
        index = bisect.bisect_left(self.positions, pos)
        index = min(index, len(self.positions) - 1)
        return self.arrivals[index]

    def first_arrival_after(self, pos: int) -> int:
        index = min(bisect.bisect_right(self.positions, pos), len(self.positions) - 1)
        return self.arrivals[index]

    def duration_ms(self, last_pos: int) -> int:
        return max(0, last_pos - self.first_pos + 1) * FRAME_MS


def _spotter_job(
    stream: NameSpotterStream,
    frames: tuple[tuple[bytes, int, bool], ...],
    reset_warm_up: bool | None,
) -> list[NameHit]:
    """One `kws` lane job: a fresh context first when asked, then each frame warmed or pushed.

    `reset_warm_up` is None for no reset, else the reset's `warm_up`. Each frame is
    `(pcm, pos, push)`; only pushed frames can give hits.
    """
    if reset_warm_up is not None:
        stream.reset(warm_up=reset_warm_up)
    hits: list[NameHit] = []
    for pcm, pos, push in frames:
        if push:
            hits.extend(stream.push(pcm, pos))
        else:
            stream.warm(pcm, pos)
    return hits


def _to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


_UNREPORTED = object()


class AudioIngress:
    """One session's ingress (Design: Ingress and endpointing; wake by name, items 1, 2 and 6).

    Args:
        engines: The VAD, transcriber, optional end-of-turn model and optional name spotter.
        lanes: The lanes each engine's calls run on.
        endpointing: Speech start, partial cadence and end-of-turn settings.
        wake: The spotter threshold and the `from_pos` chain rule's limits.
        emit: Receives every event; the controller's mailbox `post`.
        retention_ms: How much recent uplink the ring keeps (`ring_retention_ms`), at most
            10000.
        epoch: Stamped on every event, so the controller can tell this ingress's events from a
            replaced one's.
        new_utterance_id: The session's id source; ingress mints no id of its own.
        clock: The session's monotonic millisecond clock.
        seed: The previous ingress's `handoff()` across a reconfigure.

    Raises:
        ValueError: `retention_ms` is not between 1 and 10000.
    """

    def __init__(
        self,
        *,
        engines: IngressEngines,
        lanes: IngressLanes,
        endpointing: EndpointingConfig,
        wake: WakeConfig,
        emit: Callable[[IngressEvent], None],
        retention_ms: int,
        epoch: int,
        new_utterance_id: Callable[[], str],
        clock: Callable[[], int],
        seed: IngressSeed | None = None,
    ) -> None:
        if not 0 < retention_ms <= MAX_RETENTION_MS:
            raise ValueError(
                f"retention_ms must be between 1 and {MAX_RETENTION_MS}, got {retention_ms}"
            )
        self._engines = engines
        self._lanes = lanes
        self._cfg = endpointing
        self._wake = wake
        self._emit_cb = emit
        self._epoch = epoch
        self._new_id = new_utterance_id
        self._clock = clock
        self._token = CancelToken()

        self._ring: deque[_Frame] = deque()
        self._by_pos: dict[int, _Frame] = {}
        self._ring_frames = max(1, retention_ms // FRAME_MS)
        self._queue: deque[_Frame] = deque()
        self._queue_cap = max(1, endpointing.max_backlog_ms // FRAME_MS)
        self._pre_roll = onset_lead_frames(endpointing)
        self._wakeup: asyncio.Event | None = None
        self._worker: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

        self._closed = False
        self._muted = False
        self._engaged = False
        self._speaking = False
        self._seq_last: int | None = None

        self._vad_stream: VadStream | None = None
        self._vad_frames: deque[_Frame] = deque()
        self._vad_offset = 0
        self._vad_buffered = 0
        self._vad_generation = 0
        self._tracker = SpeechSegmentTracker(endpointing)
        self._segments: deque[tuple[int, int]] = deque()
        self._processed_pos: int | None = None

        self._utt: _Utterance | None = None
        self._floor = 0
        self._event_seq = 0
        self._last_emitted = 0
        self._queue_overload_key: object = _UNREPORTED

        # The spotter's feed: `_kws_next` is the first frame not yet released to it (None until
        # the stream's start is known), `_kws_waiting` the released frames no job has taken,
        # each with whether it is pushed, and `_kws_reset` a pending reset's `warm_up`.
        # `_kws_off` is set when the stream could not be opened: nothing more is fed.
        self._kws_stream: NameSpotterStream | None = None
        self._kws_next: int | None = None
        self._kws_waiting: deque[tuple[_Frame, bool]] = deque()
        self._kws_busy = False
        self._kws_off = False
        self._kws_reset: bool | None = None
        self._kws_generation = 0
        self._kws_overload_reported = False
        self._reference: OrderedDict[int, bytes] = OrderedDict()
        self._reference_seen = False
        spotter = engines.spotter
        lead_ms = (spotter.window_ms if spotter else 0) + wake.reference_tolerance_ms
        self._reference_cap = -(-lead_ms // FRAME_MS)
        self._endpoint_wake: _EndpointWake | None = None

        self._frames_received = 0
        self._gap_frames = 0
        self._muted_frames = 0
        self._ignored_frames = 0
        self._queue_dropped = 0
        self._skipped_partials = 0
        self._skipped_kws = 0
        self._reference_uncovered = 0
        self._engine_errors = 0

        if seed is not None:
            self._apply_seed(seed)

    # -- Public surface ---------------------------------------------------------------------------

    def push_frame(self, frame: Frame, pos: int) -> None:
        """Take one uplink frame at the session's position `pos`.

        A muted ingress discards the frame before the VAD and the spotter; a closed one counts
        and ignores it. Frames queued for the VAD past `max_backlog_ms` are dropped oldest
        first, with one `vad` overload per utterance.
        """
        if self._closed:
            self._ignored_frames += 1
            return
        self._count_gap(frame.seq)
        if self._muted:
            self._muted_frames += 1
            return
        self._frames_received += 1
        pcm = np.frombuffer(frame.payload, dtype="<i2").astype(np.float32) / _PCM16_SCALE
        item = _Frame(pos, frame.seq, pcm, frame.payload, self._clock())
        self._remember(item)
        if self._kws_next is None:
            # The spotter's stream starts at the first frame this ingress receives, so frames
            # the VAD queue drops before any is routed are fed all the same.
            self._kws_next = pos
        wake = self._endpoint_wake
        if wake is not None and wake.last_pos is None:
            if wake.first_pos is None:
                wake.first_pos = pos
            wake.remaining -= 1
            if wake.remaining <= 0:
                wake.last_pos = pos
        self._queue.append(item)
        dropped = 0
        while len(self._queue) > self._queue_cap:
            self._queue.popleft()
            dropped += 1
        if dropped:
            self._queue_dropped += dropped
            self._report_queue_overload(dropped)
        self._kick()

    def push_reference(self, seq: int, pcm: bytes) -> None:
        """Keep a frame of the PC's own output, matched to the microphone frame with this `seq`.

        Only the last `window_ms` + `reference_tolerance_ms` are kept: a hit's reference window
        ends at the hit, so nothing older can be asked for.
        """
        if self._closed or self._muted or self._engines.spotter is None:
            return
        self._reference_seen = True
        self._reference[seq] = pcm
        self._reference.move_to_end(seq)
        while len(self._reference) > self._reference_cap:
            self._reference.popitem(last=False)

    def set_engaged(self, engaged: bool) -> None:
        """Engaged, VAD speech opens an utterance; dormant, it only marks segments."""
        self._engaged = engaged

    def set_speaking(self, speaking: bool) -> None:
        """While MIST speaks, partials come at `stt_partial_interval_speaking_ms`."""
        self._speaking = speaking

    def set_muted(self, muted: bool) -> None:
        """Muted, every uplink frame is discarded before the VAD and the spotter.

        A mute closes the open utterance here and emits its `UtteranceDropped` with reason
        `muted` before returning, so the controller needs no close of its own. An armed endpoint wake is discarded, and the floor moves past every
        frame received so far, including frames still queued for the VAD: no utterance or name
        opened after the unmute can reach back into audio from before the mute. Unmuting starts
        the VAD afresh at the next frame, since the frames between are gone.

        The name spotter skips to the floor too: frames waiting for it are discarded, its next
        fed frame starts a fresh, warmed context, and a spotter job still running reports none
        of its hits, even when the mute is over by the time it ends.
        """
        if muted == self._muted or self._closed:
            return
        self._muted = muted
        if muted:
            utt = self._utt
            if utt is not None:
                self._drop_emitted(utt, "muted", utt.last_partial_text, None, utt.last_pushed)
            self._endpoint_wake = None
            if self._ring:
                # A frame enters the ring before its queue, so this covers the queued ones too.
                self._floor = max(self._floor, self._ring[-1].pos + 1)
            self._muted_frames += len(self._queue)
            self._queue.clear()
            self._reset_vad()
            self._mute_spotter()

    def expect_endpoint_wake(
        self, *, keyword: str, score: float, backlog_ms: int, arrival: float
    ) -> None:
        """Treat the next `backlog_ms` of frames as an endpoint's backlog ending at its name.

        When the backlog's last frame has been scored, ingress emits a `NameHeard` with spotter
        `endpoint`, the hit at that frame, and `from_pos` by the chain rule over the backlog,
        never before its first frame.
        """
        if self._closed or self._muted:
            return
        self._endpoint_wake = _EndpointWake(keyword, score, arrival, max(1, backlog_ms // FRAME_MS))

    def open_utterance(self, mode: Literal["wake"], from_pos: int) -> OpenedUtterance:
        """Open a wake utterance from `from_pos`, or continue the open one.

        A new utterance starts at the first retained frame at or after the later of `from_pos`
        and the floor, with the retained frames backfilled into its stream at once, outside the
        queue cap. A start older than the ring begins at the oldest frame and reports the lost
        audio as a `capture` overload. It emits no `SpeechStarted` and endpoints by VAD and end
        of turn.

        While muted there is no audio to hear, so it opens no stream: it mints an id, emits that
        id's `UtteranceDropped` with reason `muted` at once, and returns the id at the later of
        `from_pos` and the floor. The controller closes it on that drop, as on any other.

        Raises:
            RealtimeError: The ingress is closed.
            ValueError: `mode` is not `wake`.
        """
        if self._closed:
            raise RealtimeError("audio ingress is closed: cannot open an utterance")
        if mode != "wake":
            raise ValueError(f"unknown utterance mode {mode!r}")
        if self._muted:
            return self._muted_wake(from_pos)
        utt = self._utt
        if utt is not None:
            return OpenedUtterance(utt.id, utt.first_pos)
        wanted = max(from_pos, self._floor)
        missing = 0
        if self._ring and wanted < self._ring[0].pos:
            missing = self._ring[0].pos - wanted
        utt = self._open(wanted)
        if self._tracker.open_segment is None:
            # A lever can open before the speech it was opened for, so the pause is counted
            # from the open, not from the last speech before it.
            self._tracker.restart_silence()
        if missing:
            self._send(
                IngressOverload(
                    self._epoch,
                    self._stamp(),
                    "capture",
                    missing * FRAME_MS,
                    utt.id,
                    utt.first_pos,
                )
            )
        return OpenedUtterance(utt.id, utt.first_pos)

    def request_gate_partial(self, utterance_id: str, through_pos: int) -> None:
        """Decode the open utterance once frames through `through_pos` have arrived.

        The result is one `TranscriptPartial` with `gate` true. It is the one partial in flight,
        so a cadence tick meanwhile is skipped, and when the utterance ends within it, it is
        the final transcript too.
        """
        utt = self._utt
        if self._closed or utt is None or utt.id != utterance_id:
            return
        utt.gate_state = "waiting"
        utt.gate_through = through_pos
        utt.gate_result = None
        self._maybe_start_gate(utt)

    def drop(self, utterance_id: str, reason: ControllerDropReason) -> UtteranceDropped | None:
        """Close the named open utterance without a final decode and return its drop.

        Its pending lane jobs are cancelled and its stream closed. Nothing is emitted. Returns
        None, changing nothing, unless ingress still holds that id open.
        """
        utt = self._utt
        if self._closed or utt is None or utt.id != utterance_id:
            return None
        return self._drop_returned(utt, reason)

    def close(self) -> UtteranceDropped | None:
        """Stop for good: the open utterance's drop (`voice_stop`), or None, and nothing emitted.

        Every pending job is cancelled and the ring freed. Afterwards frames are counted and
        ignored, and no event is emitted.
        """
        if self._closed:
            return None
        utt = self._utt
        dropped = self._drop_returned(utt, "voice_stop") if utt is not None else None
        self._closed = True
        self._token.set()
        self._endpoint_wake = None
        current = asyncio.current_task() if self._has_loop() else None
        for task in [*self._tasks, self._worker]:
            if task is not None and task is not current and not task.done():
                task.cancel()
        self._queue.clear()
        self._ring.clear()
        self._by_pos.clear()
        self._reference.clear()
        self._vad_frames.clear()
        return dropped

    def idle_since(self) -> int | None:
        """The last emitted `event_seq` while no utterance is open (0 before any); else None."""
        if self._utt is not None:
            return None
        return self._last_emitted

    def handoff(self) -> IngressSeed:
        """The seed for the next ingress across a reconfigure; reads only."""
        frames = tuple((item.pos, item.pcm) for item in self._ring)
        return IngressSeed(frames=frames, floor=self._floor, vad_from=self._vad_from())

    def stats(self) -> IngressStats:
        """The counters so far: uplink gaps, discarded frames, skipped jobs, missing reference."""
        return IngressStats(
            frames_received=self._frames_received,
            gap_frames=self._gap_frames,
            muted_frames=self._muted_frames,
            ignored_frames=self._ignored_frames,
            queue_dropped_frames=self._queue_dropped,
            skipped_partials=self._skipped_partials,
            skipped_kws_hops=self._skipped_kws,
            reference_uncovered=self._reference_uncovered,
            reference_frames=len(self._reference),
            engine_errors=self._engine_errors,
        )

    # -- Frames, ring and worker ------------------------------------------------------------------

    def _apply_seed(self, seed: IngressSeed) -> None:
        arrival = self._clock()
        for pos, pcm in seed.frames:
            audio = np.asarray(pcm, dtype=np.float32)
            self._remember(_Frame(pos, None, audio, _to_pcm16(audio), arrival))
        self._floor = seed.floor
        if seed.vad_from is None:
            # The spotter starts at the first frame received, as a new ingress's does.
            self._processed_pos = seed.frames[-1][0] if seed.frames else None
            return
        # No stream state crosses a reconfigure: this ingress's stream starts at the pre-roll
        # before the first frame its VAD scores, never below the floor.
        self._kws_next = max(seed.floor, seed.vad_from - self._pre_roll)
        before = [item.pos for item in self._ring if item.pos < seed.vad_from]
        self._processed_pos = before[-1] if before else None
        self._queue.extend(item for item in self._ring if item.pos >= seed.vad_from)

    def _remember(self, item: _Frame) -> None:
        self._ring.append(item)
        self._by_pos[item.pos] = item
        while len(self._ring) > self._ring_frames:
            old = self._ring.popleft()
            if self._by_pos.get(old.pos) is old:
                del self._by_pos[old.pos]

    def _count_gap(self, seq: int) -> None:
        last = self._seq_last
        if last is not None and seq > last + 1:
            self._gap_frames += seq - last - 1
        self._seq_last = seq

    @staticmethod
    def _has_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def _kick(self) -> None:
        if self._worker is None:
            self._wakeup = asyncio.Event()
            self._worker = asyncio.get_running_loop().create_task(self._work())
        assert self._wakeup is not None
        self._wakeup.set()

    def _spawn(self, coro: Coroutine[object, object, None]) -> asyncio.Task[None]:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _work(self) -> None:
        assert self._wakeup is not None
        while not self._closed:
            if not self._queue:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            item = self._queue.popleft()
            await self._process(item)
            # Let the lane results and spotter jobs this frame started settle before the next
            # frame, so each is judged against the frame it was started on.
            await asyncio.sleep(0)

    async def _process(self, item: _Frame) -> None:
        generation = self._vad_generation
        self._vad_frames.append(item)
        self._vad_buffered += len(item.pcm)
        while self._vad_buffered >= VAD_WINDOW_SAMPLES:
            window, first, last = self._take_window()
            if self._vad_stream is None:
                self._vad_stream = self._engines.vad.open_stream()
            try:
                prob = await self._lanes.vad.run(self._vad_stream.score, window, token=self._token)
            except LaneJobCancelled:
                return
            except SpeechEngineError:
                logger.warning("Ingress VAD window failed; scored as silence", exc_info=True)
                self._engine_errors += 1
                prob = 0.0
            if self._closed or generation != self._vad_generation:
                return
            self._on_window(prob >= self._cfg.vad_threshold, first, last)
        if self._closed or generation != self._vad_generation:
            return
        self._route(item)
        self._after_frame(item)

    def _take_window(self) -> tuple[np.ndarray, int, int]:
        parts: list[np.ndarray] = []
        need = VAD_WINDOW_SAMPLES
        first: int | None = None
        last = 0
        while need:
            item = self._vad_frames[0]
            if first is None:
                first = item.pos
            available = len(item.pcm) - self._vad_offset
            take = min(need, available)
            parts.append(item.pcm[self._vad_offset : self._vad_offset + take])
            last = item.pos
            need -= take
            if take == available:
                self._vad_frames.popleft()
                self._vad_offset = 0
            else:
                self._vad_offset += take
        self._vad_buffered -= VAD_WINDOW_SAMPLES
        assert first is not None
        return np.concatenate(parts), first, last

    def _reset_vad(self) -> None:
        self._vad_generation += 1
        self._vad_stream = None
        self._vad_frames.clear()
        self._vad_offset = 0
        self._vad_buffered = 0
        closed = self._tracker.reset()
        if closed is not None:
            self._segments.append(closed)

    def _vad_from(self) -> int | None:
        if self._closed or self._muted:
            return None
        if self._vad_frames:
            return self._vad_frames[0].pos
        if self._queue:
            return self._queue[0].pos
        if self._processed_pos is None:
            return None
        return self._processed_pos + 1

    def _silence_ms(self) -> int:
        return self._tracker.silence_ms()

    # -- Speech segments --------------------------------------------------------------------------

    def _on_window(self, speech: bool, first: int, last: int) -> None:
        step = self._tracker.window(speech, first, last)
        if step.closed is not None:
            self._segments.append(step.closed)
        if step.started is not None:
            self._start_segment(step.started[0])

    def _start_segment(self, onset: int) -> None:
        oldest = self._ring[0].pos if self._ring else onset
        while self._segments and self._segments[0][1] < oldest:
            self._segments.popleft()
        utt = self._utt
        if utt is not None:
            if utt.state == "pending":
                self._resume(utt)
        elif self._engaged:
            utt = self._open(max(onset - self._pre_roll, self._floor))
            self._send(SpeechStarted(self._epoch, self._stamp(), utt.id, utt.first_pos))

    def _chain_from(self, hit_pos: int, lower: int | None) -> int:
        """Where a wake utterance for a hit at `hit_pos` opens (Design, item 2).

        The start of the earliest segment in the chain reaching the hit, each gap at most
        `join_gap_ms`, less the pre-roll; bounded below by the lookback, the floor, the ring's
        oldest frame and, for an endpoint, the backlog's first frame.
        """
        segments = [segment for segment in self._segments if segment[0] <= hit_pos]
        current = self._tracker.open_segment
        if current is not None and current[0] <= hit_pos:
            segments.append(current)
        start = hit_pos
        if segments:
            index = len(segments) - 1
            while index > 0:
                gap = segments[index][0] - segments[index - 1][1] - 1
                if gap * FRAME_MS > self._wake.join_gap_ms:
                    break
                index -= 1
            start = segments[index][0] - self._pre_roll
        bounds = [start, hit_pos - self._wake.lookback_ms // FRAME_MS, self._floor]
        if self._ring:
            bounds.append(self._ring[0].pos)
        if lower is not None:
            bounds.append(lower)
        return max(bounds)

    # -- Name spotting ----------------------------------------------------------------------------

    def _feed_spotter(self) -> None:
        """Release the frames `spotter_feed` allows, from the ring by position, to the spotter.

        Called after each routed frame. Frames the VAD queue dropped, and a seeded ingress's
        frames before `vad_from`, are in the ring though never routed, and are released all
        the same. A position the ring no longer holds is skipped.
        """
        if self._engines.spotter is None or self._kws_off or self._kws_next is None:
            return
        newest = self._processed_pos
        next_unscored = self._vad_from()
        if newest is None or next_unscored is None:
            return
        segment = self._tracker.open_segment
        feed = spotter_feed(
            self._kws_next,
            newest,
            segment[0] if segment is not None else None,
            self._tracker.earliest_onset(next_unscored),
            self._pre_roll,
        )
        if feed.end <= self._kws_next:
            return
        for positions, push in ((feed.warm, False), (feed.push, True)):
            for pos in positions:
                frame = self._by_pos.get(pos)
                if frame is not None:
                    self._kws_waiting.append((frame, push))
        self._kws_next = feed.end
        if self._kws_busy:
            self._bound_spotter_backlog()
        else:
            self._start_spotter_job()

    def _bound_spotter_backlog(self) -> None:
        """Drop the oldest frames waiting past `KWS_BACKLOG_FRAMES`, and restart the context.

        The reset skips the silence warm-up, which would only add to the backlog; the fresh
        context reports nothing until it has heard `warmup_ms` of audio.
        """
        dropped = 0
        while len(self._kws_waiting) > KWS_BACKLOG_FRAMES:
            self._kws_waiting.popleft()
            dropped += 1
        if not dropped:
            return
        self._skipped_kws += dropped
        self._kws_reset = False
        if not self._kws_overload_reported:
            self._kws_overload_reported = True
            self._send_overload("kws", dropped * FRAME_MS)

    def _start_spotter_job(self) -> None:
        """Start one job with every waiting frame; at most one runs per session.

        With no stream yet, the job opens one first on the `kws` lane (`_run_spotter`).
        """
        spotter = self._engines.spotter
        if spotter is None or self._kws_busy or self._closed or not self._kws_waiting:
            return
        frames = tuple((item.payload, item.pos, push) for item, push in self._kws_waiting)
        self._kws_waiting.clear()
        reset, self._kws_reset = self._kws_reset, None
        stream = self._kws_stream
        if stream is None:
            # A new stream is a fresh context, so a reset asked for before it adds nothing.
            reset = None
        self._kws_busy = True
        self._spawn(self._run_spotter(stream, frames, reset, self._kws_generation))

    async def _run_spotter(
        self,
        stream: NameSpotterStream | None,
        frames: tuple[tuple[bytes, int, bool], ...],
        reset: bool | None,
        generation: int,
    ) -> None:
        """Feed one job's frames on the `kws` lane, opening the stream first when `stream` is None.

        The open is a lane job of its own, so a failed open, which turns the spotter off, is
        told apart from a failed push, which fails only its frames job. However the job ends,
        `_kws_busy` is cleared and the next job starts unless the ingress is closed, so no
        exception, from an engine or from reporting a hit, leaves the spotter waiting for good.
        """
        try:
            if stream is None:
                stream = await self._open_spotter()
                if stream is None:
                    return
                if generation != self._kws_generation:
                    # A mute landed while no stream existed, so it asked for no reset. Every
                    # frame here is from before it, and the fresh stream has fed nothing, so
                    # there is no gap for a backlog reset to restart from either.
                    self._kws_reset = None
                    return
            try:
                hits = await self._lanes.kws.run(
                    _spotter_job, stream, frames, reset, token=self._token
                )
            except SpeechEngineError:
                logger.warning("Ingress name spotter job failed", exc_info=True)
                self._engine_errors += 1
                hits = []
            for hit in hits:
                await self._on_hit(hit, generation)
        except LaneJobCancelled:
            return
        finally:
            self._kws_busy = False
            if not self._closed:
                if self._kws_waiting:
                    self._start_spotter_job()
                else:
                    # The spotter has caught up, so the next drop is a new overload episode.
                    self._kws_overload_reported = False

    async def _open_spotter(self) -> NameSpotterStream | None:
        """Open the spotter's stream as its own `kws` job and store it as soon as it returns.

        Opening builds an interpreter and feeds it `warmup_ms` of silence, engine work that
        belongs on the lane. A `SpeechEngineError` turns the spotter off for this ingress: it is
        logged once and counted, the waiting frames are cleared and no later frame is queued.
        `load` has already built an interpreter from the same bytes, so a failed open is not
        expected to be transient, and retrying at every job would log and rebuild up to 50
        times a second. A new ingress (a reconfigure) opens again. Returns None when off.
        """
        spotter = self._engines.spotter
        assert spotter is not None
        try:
            stream = await self._lanes.kws.run(
                spotter.open_stream, self._wake.kws_threshold, token=self._token
            )
        except SpeechEngineError:
            logger.warning(
                "Ingress name spotter could not open its stream; it is off for this ingress",
                exc_info=True,
            )
            self._engine_errors += 1
            self._kws_off = True
            self._kws_waiting.clear()
            return None
        self._kws_stream = stream
        return stream

    def _mute_spotter(self) -> None:
        """Skip the spotter to the floor; its next fed frame starts a fresh, warmed context."""
        self._kws_generation += 1
        self._kws_waiting.clear()
        if self._kws_next is not None:
            self._kws_next = max(self._kws_next, self._floor)
        if self._kws_stream is not None:
            self._kws_reset = True

    def _hearing(self, generation: int) -> bool:
        """Whether a name hit may be reported: never while muted, after close, or after a mute.

        A job holds up to `KWS_BACKLOG_FRAMES` frames, and a mute and an unmute can both land
        while it runs, so a hit from a job the spotter generation has moved past is dropped.
        """
        return not (self._closed or self._muted or generation != self._kws_generation)

    async def _on_hit(self, hit: NameHit, generation: int) -> None:
        if not self._hearing(generation):
            return
        frame = self._by_pos.get(hit.end_pos)
        arrival = float(frame.arrival_ms if frame is not None else self._clock())
        reference = await self._reference_score(hit)
        # Scoring the reference awaits a lane, and a mute or a close may land meanwhile.
        if not self._hearing(generation):
            return
        utt = self._utt
        in_utterance = utt is not None and hit.end_pos >= utt.first_pos
        self._send(
            NameHeard(
                epoch=self._epoch,
                event_seq=self._stamp(),
                hit=hit,
                arrival=arrival,
                from_pos=self._chain_from(hit.end_pos, None),
                utterance_id=utt.id if in_utterance and utt is not None else None,
                reference_score=reference,
            )
        )

    async def _reference_score(self, hit: NameHit) -> float | None:
        """The spotter's score over the PC's output in a window ending at the hit.

        One-sided: audio the PC played after the microphone heard the name cannot have caused
        the hit, so the window ends at the hit's frame and reaches `reference_tolerance_ms`
        before the spotter's own window, and the score is ready with no wait for later frames.
        """
        spotter = self._engines.spotter
        if spotter is None or not self._reference_seen:
            return None
        window = -(-spotter.window_ms // FRAME_MS)
        lead = self._wake.reference_tolerance_ms // FRAME_MS
        positions = range(hit.end_pos - window - lead + 1, hit.end_pos + 1)
        seqs: list[int | None] = []
        for pos in positions:
            frame = self._by_pos.get(pos)
            seqs.append(frame.seq if frame is not None else None)
        if any(seq is None or seq not in self._reference for seq in seqs[-window:]):
            self._reference_uncovered += 1
            return None
        pcm = b"".join(
            self._reference[seq] for seq in seqs if seq is not None and seq in self._reference
        )
        try:
            return await self._lanes.kws.run(spotter.score_window, pcm, token=self._token)
        except LaneJobCancelled:
            return None
        except SpeechEngineError:
            logger.warning("Ingress reference scoring failed", exc_info=True)
            self._engine_errors += 1
            return None

    def _endpoint_heard(self, wake: _EndpointWake) -> None:
        assert wake.last_pos is not None
        self._endpoint_wake = None
        utt = self._utt
        self._send(
            NameHeard(
                epoch=self._epoch,
                event_seq=self._stamp(),
                hit=NameHit(end_pos=wake.last_pos, keyword=wake.keyword, score=wake.score),
                arrival=wake.arrival,
                from_pos=self._chain_from(wake.last_pos, wake.first_pos),
                utterance_id=utt.id if utt is not None else None,
                reference_score=None,
                spotter="endpoint",
            )
        )

    # -- Utterances -------------------------------------------------------------------------------

    def _open(self, start: int) -> _Utterance:
        first_pos = next((item.pos for item in self._ring if item.pos >= start), start)
        utt = _Utterance(self._new_id(), first_pos, self._engines.transcriber.open_stream())
        utt.next_partial_ms = self._clock() + self._partial_interval()
        self._utt = utt
        if self._processed_pos is not None:
            for item in self._ring:
                if first_pos <= item.pos <= self._processed_pos:
                    self._push(utt, item)
        return utt

    def _push(self, utt: _Utterance, item: _Frame) -> None:
        utt.stream.push(item.pcm)
        utt.last_pushed = item.pos
        utt.positions.append(item.pos)
        utt.arrivals.append(item.arrival_ms)
        utt.audio.append(item.pcm)

    def _route(self, item: _Frame) -> None:
        self._processed_pos = item.pos
        utt = self._utt
        if utt is not None and item.pos >= utt.first_pos and item.pos > utt.last_pushed:
            if utt.state == "speaking":
                self._push(utt, item)
            elif not utt.held or item.pos > utt.held[-1].pos:
                utt.held.append(item)
        self._feed_spotter()

    def _after_frame(self, item: _Frame) -> None:
        utt = self._utt
        if utt is not None:
            if utt.state == "speaking":
                self._tick_partial(utt)
                if (
                    self._tracker.open_segment is None
                    and utt.last_pushed >= utt.first_pos
                    and self._silence_ms() >= self._cfg.endpoint_silence_ms
                ):
                    self._candidate(utt)
            else:
                self._tick_pending(utt)
            if self._utt is utt:
                self._maybe_start_gate(utt)
        wake = self._endpoint_wake
        if wake is not None and wake.last_pos is not None and item.pos >= wake.last_pos:
            self._endpoint_heard(wake)

    def _partial_interval(self) -> int:
        if self._speaking:
            return self._cfg.stt_partial_interval_speaking_ms
        return self._cfg.stt_partial_interval_ms

    def _tick_partial(self, utt: _Utterance) -> None:
        now = self._clock()
        if now < utt.next_partial_ms:
            return
        while utt.next_partial_ms <= now:
            utt.next_partial_ms += self._partial_interval()
        if utt.decoding:
            self._skipped_partials += 1
            return
        if utt.last_pushed <= utt.partial_end:
            return
        self._start_decode(utt, gate=False)

    def _maybe_start_gate(self, utt: _Utterance) -> None:
        if (
            utt.gate_state == "waiting"
            and not utt.decoding
            and self._processed_pos is not None
            and self._processed_pos >= utt.gate_through
        ):
            utt.gate_state = "running"
            utt.gate_end = utt.last_pushed
            self._start_decode(utt, gate=True)

    def _start_decode(self, utt: _Utterance, *, gate: bool) -> None:
        utt.decoding = True
        task = self._spawn(self._decode_partial(utt, utt.last_pushed, gate))
        utt.tasks.add(task)
        task.add_done_callback(utt.tasks.discard)

    async def _decode_partial(self, utt: _Utterance, end_pos: int, gate: bool) -> None:
        token = utt.decode_token
        try:
            update = await self._lanes.stt.run(utt.stream.decode_partial, token, token=token)
        except LaneJobCancelled:
            return
        except SpeechEngineError:
            logger.warning("Ingress partial decode failed for %s", utt.id, exc_info=True)
            self._engine_errors += 1
            if gate and self._utt is utt:
                utt.gate_state = "none"
                if utt.final_from_gate:
                    utt.final_from_gate = False
                    self._start_final(utt)
            return
        finally:
            utt.decoding = False
        if self._utt is not utt or self._closed:
            return
        uncovered = utt.first_arrival_after(utt.partial_end)
        utt.partial_end = end_pos
        utt.last_partial_text = update.text
        through = utt.gate_through if gate else None
        self._send(
            TranscriptPartial(
                self._epoch,
                self._stamp(),
                utt.id,
                utt.first_pos,
                update.text,
                update.stable_chars,
                utt.arrival(end_pos),
                end_pos,
                uncovered,
                gate,
                through,
            )
        )
        if gate:
            utt.gate_state = "done"
            utt.gate_result = update
            if utt.final_from_gate and utt.state == "pending":
                utt.final_from_gate = False
                self._on_final(utt, update, end_pos)
        elif self._utt is utt:
            self._maybe_start_gate(utt)

    def _gate_covers_end(self, utt: _Utterance) -> bool:
        if utt.gate_state in ("done", "running"):
            return utt.gate_end >= utt.end_pos
        if utt.gate_state == "waiting":
            return utt.gate_through >= utt.end_pos
        return False

    def _candidate(self, utt: _Utterance) -> None:
        """The endpoint silence started: the final decode and the end-of-turn score start now.

        Both are submitted in this step, before either can complete, so the prefill a candidate
        triggers overlaps the end-of-turn decision instead of following it.
        """
        utt.state = "pending"
        utt.end_pos = utt.last_pushed
        utt.final = None
        utt.decided = None
        utt.eot_prob = None
        utt.eot_checked_ms = self._silence_ms()
        reuse_gate = self._gate_covers_end(utt)
        if not reuse_gate:
            self._start_final(utt)
        if self._engines.turn_end is not None:
            self._start_eot(utt)
        if reuse_gate:
            if utt.gate_state == "done" and utt.gate_result is not None:
                self._on_final(utt, utt.gate_result, utt.end_pos)
            else:
                utt.final_from_gate = True

    def _start_final(self, utt: _Utterance) -> None:
        task = self._spawn(self._final(utt, utt.end_token))
        utt.tasks.add(task)
        task.add_done_callback(utt.tasks.discard)

    def _start_eot(self, utt: _Utterance) -> None:
        turn_end = self._engines.turn_end
        assert turn_end is not None
        utt.eot_running = True
        window = self._cfg.eot_window_ms * _SAMPLE_RATE // 1000
        audio = np.concatenate([*utt.audio, *(item.pcm for item in utt.held)])[-window:]
        task = self._spawn(self._eot(utt, utt.end_token, turn_end, audio))
        utt.tasks.add(task)
        task.add_done_callback(utt.tasks.discard)

    async def _final(self, utt: _Utterance, token: CancelToken) -> None:
        try:
            update = await self._lanes.stt.run(utt.stream.decode_final, token, token=token)
        except LaneJobCancelled:
            return
        except SpeechEngineError:
            if self._utt is utt and not token.is_set():
                logger.warning("Ingress final decode failed for %s", utt.id, exc_info=True)
                self._engine_errors += 1
                self._drop_emitted(utt, "stt_error", None, None, utt.end_pos)
            return
        if self._utt is not utt or token.is_set():
            return
        self._on_final(utt, update, utt.end_pos)

    async def _eot(
        self, utt: _Utterance, token: CancelToken, turn_end: TurnEndDetector, audio: np.ndarray
    ) -> None:
        try:
            prob = await self._lanes.eot.run(turn_end.probability_complete, audio, token=token)
        except LaneJobCancelled:
            return
        except SpeechEngineError:
            logger.warning("Ingress end-of-turn score failed for %s", utt.id, exc_info=True)
            self._engine_errors += 1
            if utt.end_token is token:
                utt.eot_running = False
            return
        if self._utt is not utt or token.is_set():
            return
        utt.eot_running = False
        utt.eot_prob = prob
        if utt.decided is None and prob > self._cfg.eot_threshold:
            utt.decided = ("semantic", self._silence_ms())
        self._try_end(utt)

    def _tick_pending(self, utt: _Utterance) -> None:
        if utt.decided is not None:
            return
        silence = self._silence_ms()
        if silence >= self._cfg.eot_max_silence_ms:
            reason: EndReason = (
                "silence_cap" if self._engines.turn_end is not None else "silence_cap_no_model"
            )
            utt.decided = (reason, silence)
            self._try_end(utt)
            return
        if (
            self._engines.turn_end is not None
            and not utt.eot_running
            and silence >= utt.eot_checked_ms + self._cfg.eot_recheck_ms
        ):
            utt.eot_checked_ms = silence
            self._start_eot(utt)

    def _on_final(self, utt: _Utterance, update: TranscriptUpdate, end_pos: int) -> None:
        if not update.text.strip():
            self._drop_emitted(utt, "noise", update.text, update, end_pos)
            return
        if (
            update.no_speech_prob is not None
            and update.avg_logprob is not None
            and update.no_speech_prob > self._cfg.no_speech_threshold
            and update.avg_logprob <= self._cfg.logprob_threshold
        ):
            self._drop_emitted(utt, "no_speech", update.text, update, end_pos)
            return
        utt.final = update
        utt.final_end = end_pos
        if self._cfg.speculative_start:
            self._send(
                EndpointCandidate(
                    self._epoch,
                    self._stamp(),
                    utt.id,
                    utt.first_pos,
                    update.text,
                    utt.arrival(end_pos),
                )
            )
        self._try_end(utt)

    def _try_end(self, utt: _Utterance) -> None:
        if utt.decided is None or utt.final is None or self._utt is not utt:
            return
        reason, silence = utt.decided
        final = utt.final
        probability = None if reason == "silence_cap_no_model" else utt.eot_prob
        self._send(
            UtteranceEnded(
                self._epoch,
                self._stamp(),
                utt.id,
                utt.first_pos,
                final.text,
                reason,
                probability,
                silence,
                utt.duration_ms(utt.final_end),
                final.no_speech_prob,
                final.avg_logprob,
                utt.arrival(utt.final_end),
            )
        )
        self._finish(utt, utt.final_end)

    def _resume(self, utt: _Utterance) -> None:
        utt.end_token.set()
        utt.end_token = CancelToken()
        utt.state = "speaking"
        utt.final = None
        utt.decided = None
        utt.eot_prob = None
        utt.eot_running = False
        utt.final_from_gate = False
        held, utt.held = utt.held, []
        for item in held:
            self._push(utt, item)
        self._send(SpeechResumed(self._epoch, self._stamp(), utt.id, utt.first_pos))

    def _finish(self, utt: _Utterance, last_pos: int) -> None:
        utt.decode_token.set()
        utt.end_token.set()
        current = asyncio.current_task() if self._has_loop() else None
        for task in list(utt.tasks):
            if task is not current and not task.done():
                task.cancel()
        utt.stream.close()
        utt.held = []
        self._floor = max(self._floor, last_pos + 1)
        self._utt = None

    def _drop_emitted(
        self,
        utt: _Utterance,
        reason: DropReason,
        text: str | None,
        update: TranscriptUpdate | None,
        last_pos: int,
    ) -> None:
        logger.info("Ingress dropped utterance %s: %s", utt.id, reason)
        self._send(
            UtteranceDropped(
                self._epoch,
                self._stamp(),
                utt.id,
                utt.first_pos,
                reason,
                text,
                utt.duration_ms(last_pos),
                update.no_speech_prob if update is not None else None,
                update.avg_logprob if update is not None else None,
            )
        )
        self._finish(utt, last_pos)

    def _drop_returned(self, utt: _Utterance, reason: DropReason) -> UtteranceDropped:
        logger.info("Ingress dropped utterance %s: %s", utt.id, reason)
        dropped = UtteranceDropped(
            self._epoch,
            self._last_emitted,
            utt.id,
            utt.first_pos,
            reason,
            utt.last_partial_text,
            utt.duration_ms(utt.last_pushed),
            None,
            None,
        )
        self._finish(utt, utt.last_pushed)
        return dropped

    def _muted_wake(self, from_pos: int) -> OpenedUtterance:
        utterance_id = self._new_id()
        first_pos = max(from_pos, self._floor)
        logger.info("Ingress dropped utterance %s: muted", utterance_id)
        self._send(
            UtteranceDropped(
                self._epoch,
                self._stamp(),
                utterance_id,
                first_pos,
                "muted",
                None,
                0,
                None,
                None,
            )
        )
        return OpenedUtterance(utterance_id, first_pos)

    # -- Emission ---------------------------------------------------------------------------------

    def _stamp(self) -> int:
        self._event_seq += 1
        return self._event_seq

    def _send(self, event: IngressEvent) -> None:
        if self._closed:
            return
        self._last_emitted = event.event_seq
        self._emit_cb(event)

    def _send_overload(self, lane: str, dropped_ms: int) -> None:
        utt = self._utt
        self._send(
            IngressOverload(
                self._epoch,
                self._stamp(),
                lane,
                dropped_ms,
                utt.id if utt is not None else None,
                utt.first_pos if utt is not None else None,
            )
        )

    def _report_queue_overload(self, dropped: int) -> None:
        key = self._utt.id if self._utt is not None else None
        if self._queue_overload_key == key:
            return
        self._queue_overload_key = key
        self._send_overload("vad", dropped * FRAME_MS)
