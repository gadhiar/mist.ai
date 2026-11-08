"""Downlink playback: the session's `turn` lane, the pacer that feeds it and the ledger.

`TurnLane` is the queue of a session's `turn` lane, which the session's sender drains. Its purge
reports what it removed, so a cut knows whether the sender already took the turn's `0x02`, its
`stream_start` and which of its tokens.

`PlaybackPacer` frames one tag's synthesized audio into 40 ms `0x01` frames and queues them
through a turn sink no more than `max_lead_ms` ahead of the client's playback position, so a
barge-in has little queued audio to flush. It queues exactly one terminal frame per tag and
nothing of that tag after it. Its `stop` sends a `0x03` exactly when the client may still hold
audio of the tag, that is unless the sender took the `0x02`. Before a segment's first frame and
before the terminal frame it calls a `PlaybackListener`, which is how the scene seam (T11) learns
the sample at which an anchored intent plays.

`PlaybackLedger` maps played time back to the reply's text: the spoken prefix that an
interrupted turn records, and the recently played text the echo guard reads. It records a frame
only once it is queued to the client, so dropped or kept-back audio never counts as heard.
"""

import asyncio
import math
import threading
import time
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

from .budget import PlaybackConfig
from .errors import RealtimeError
from .protocol import (
    DOWNLINK_SAMPLE_RATE,
    PCM16_SAMPLE_BYTES,
    Frame,
    MsgType,
    PlaybackStopMessage,
)
from .speech import SynthChunk, TextSegment

LaneItemKind = Literal["stream_start", "token"]
LedgerSource = Literal["client", "estimate"]


# -- The turn lane -------------------------------------------------------------


# N818 asks for an `Error` suffix; the name is the interface the plan fixes, and it says what
# happened to the lane, as `asyncio.QueueFull` does.
class LaneFull(RealtimeError):  # noqa: N818
    """The `turn` lane is at `max_items`; the session closes with code 1013."""


@dataclass(frozen=True, slots=True)
class LanePurge:
    """What one purge removed of a turn's queued items.

    Attributes:
        terminal: An item put with `terminal` set, the tag's `0x02`, was still queued.
        stream_start: The turn's `stream_start` was still queued, so the client never saw the
            turn open.
        first_token: The `token_index` of the oldest removed `stream_token`, None when no
            token was removed. A turn's items leave the lane in put order, so the removed
            tokens run from it to the last one put.
    """

    terminal: bool
    stream_start: bool
    first_token: int | None


@dataclass(frozen=True, slots=True)
class _LaneEntry:
    turn_id: str
    item: object
    terminal: bool
    kind: LaneItemKind | None
    token_index: int | None


class TurnLane:
    """The bounded queue of a session's `turn` lane.

    Put, take and purge each hold the lane's lock for the whole call and never await, so a take
    and a purge are each one atomic step: a queued `0x02` is either taken by the sender or
    reported by the purge, never both and never neither.
    """

    def __init__(
        self, max_items: int = 2048, lock: AbstractContextManager[object] | None = None
    ) -> None:
        self._max_items = max_items
        self._lock: AbstractContextManager[object] = (
            lock if lock is not None else threading.Lock()
        )
        self._items: deque[_LaneEntry] = deque()

    def put(
        self,
        turn_id: str,
        item: object,
        terminal: bool = False,
        kind: LaneItemKind | None = None,
        token_index: int | None = None,
    ) -> bool:
        """Queue one item of a turn.

        Args:
            turn_id: The turn the item belongs to, which a purge selects by.
            item: The message or frame the sender sends.
            terminal: Set only for a tag's `0x02`.
            kind: `stream_start` for the turn's `stream_start`, `token` for each
                `stream_token`.
            token_index: The token's 0-based index among the turn's `stream_token`s, set
                exactly when `kind` is `token`.

        Returns:
            True: the item is queued. The lane never keeps an item back, so it can serve as a
            pacer's turn sink.

        Raises:
            LaneFull: The lane already holds `max_items`.
            ValueError: `token_index` is set when `kind` is not `token`, or missing when it is.
        """
        if (kind == "token") != (token_index is not None):
            raise ValueError("token_index is set exactly when kind is 'token'")
        entry = _LaneEntry(turn_id, item, terminal, kind, token_index)
        with self._lock:
            if len(self._items) >= self._max_items:
                raise LaneFull(f"turn lane is full at {self._max_items} items")
            self._items.append(entry)
        return True

    def take(self) -> object | None:
        """Remove and return the oldest item, None when the lane is empty.

        A taken item counts as sent: a later purge does not report it.
        """
        with self._lock:
            if not self._items:
                return None
            return self._items.popleft().item

    def purge(self, turn_id: str) -> LanePurge:
        """Remove every queued item of `turn_id` and report what was removed."""
        with self._lock:
            entries = list(self._items)
            self._items.clear()
            self._items.extend(entry for entry in entries if entry.turn_id != turn_id)
        removed = [entry for entry in entries if entry.turn_id == turn_id]
        tokens = [entry.token_index for entry in removed if entry.kind == "token"]
        return LanePurge(
            terminal=any(entry.terminal for entry in removed),
            stream_start=any(entry.kind == "stream_start" for entry in removed),
            first_token=tokens[0] if tokens else None,
        )


# -- Seams ------------------------------------------------------------------------


class TurnSink(Protocol):
    """Where the pacer queues a tag's frames: the session's `TurnLane`, or T11's sink in front
    of it, which keeps frames back during a hold and returns False for them."""

    def put(
        self,
        turn_id: str,
        item: object,
        terminal: bool = False,
        kind: LaneItemKind | None = None,
        token_index: int | None = None,
    ) -> bool:
        """Queue `item` and return True, or keep it back and return False."""
        ...


class PlaybackListener(Protocol):
    """Hooks the pacer calls synchronously, each just before it queues the frame it names.

    `ScenePort` (T11) satisfies this structurally. Sample offsets count 24 kHz samples of the
    tag's downlink stream as the server queued them, from 0 at its first frame.
    """

    def segment_starting(self, audio_tag: int, segment_index: int, first_sample: int) -> None:
        """Segment `segment_index`'s first frame, starting at `first_sample`, is next."""
        ...

    def stream_ending(self, audio_tag: int, last_sample: int) -> None:
        """The tag's terminal frame is next; `last_sample` is the offset just past its last
        sample, so 0 for a tag that carried no frame."""
        ...


class NullPlaybackListener:
    """The default listener: stage 1 has no scene to anchor."""

    def segment_starting(self, audio_tag: int, segment_index: int, first_sample: int) -> None:
        """Do nothing."""

    def stream_ending(self, audio_tag: int, last_sample: int) -> None:
        """Do nothing."""


class PlaybackClock(Protocol):
    """The session clock, in milliseconds, which tests replace with a fake."""

    def now_ms(self) -> int:
        """The current time."""
        ...

    async def sleep_ms(self, ms: int) -> None:
        """Wait `ms` milliseconds."""
        ...


class MonotonicPlaybackClock:
    """The real session clock: `time.monotonic` and `asyncio.sleep`."""

    def now_ms(self) -> int:
        """The monotonic clock in whole milliseconds."""
        return int(time.monotonic() * 1000)

    async def sleep_ms(self, ms: int) -> None:
        """Wait on the event loop."""
        await asyncio.sleep(ms / 1000)


# -- The ledger ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SegmentAudio:
    segment: TextSegment
    synthesized: int  # samples of the segment's audio synthesized so far


@dataclass(frozen=True, slots=True)
class _RecordedFrame:
    segment_index: int
    segment_offset: int  # the frame's first sample within its segment's audio
    samples: int
    first_sample: int  # the frame's first sample within the recorded stream
    recorded_ms: int
    start_ms: float  # when the estimate has the frame start playing


class PlaybackLedger:
    """Maps one tag's played audio back to the reply text it voices.

    The unit is the segment. Every chunk of a segment carries the segment's whole span, so a
    played share is taken of the segment's synthesized audio: whole segments, plus a
    proportional share of the current one, rounded down to a word boundary. While a segment is
    still being synthesized its share is taken of the audio synthesized so far.

    The position comes from the client's `playback_progress` reports when there are any (the
    source is `client`), each taken as reported rather than extrapolated, so a cut never counts
    audio the client had not yet confirmed. Without reports it is estimated from the clock (the
    source is `estimate`): the first recorded frame starts after the client's prebuffer, and
    each later frame when the one before it ends or, after an underrun, when it was recorded.
    """

    def __init__(
        self,
        display_text: Callable[[], str],
        clock: PlaybackClock,
        *,
        sample_rate: int = DOWNLINK_SAMPLE_RATE,
        prebuffer_ms: int = 80,
    ) -> None:
        """Build an empty ledger for one tag.

        Args:
            display_text: Returns the turn's text as the client displays it, every Token text
                concatenated, which the segment spans index (`SpeechSegmenter.text`).
            clock: The session clock.
            sample_rate: The downlink sample rate.
            prebuffer_ms: How much audio the client buffers before it starts playing.
        """
        self._display_text = display_text
        self._clock = clock
        self._sample_rate = sample_rate
        self._prebuffer_ms = prebuffer_ms
        self._segments: dict[int, _SegmentAudio] = {}
        self._frames: list[_RecordedFrame] = []
        self._firsts: list[int] = []
        self._starts: list[float] = []
        self._total = 0
        self._ended = False
        self._report_times: list[int] = []
        self._report_samples: list[int] = []

    @property
    def source(self) -> LedgerSource:
        """`client` once the client has reported its progress, else `estimate`."""
        return "client" if self._report_times else "estimate"

    @property
    def recorded_samples(self) -> int:
        """Samples of every frame recorded as queued to the client."""
        return self._total

    def note_synthesized(self, segment_index: int, segment: TextSegment, samples: int) -> None:
        """Record how much audio segment `segment_index` has synthesized in total so far."""
        self._segments[segment_index] = _SegmentAudio(segment, samples)

    def record(self, segment_index: int, segment_offset: int, samples: int) -> None:
        """Record one frame queued to the client.

        Raises:
            ValueError: The segment's synthesized audio was never noted.
        """
        if segment_index not in self._segments:
            raise ValueError(f"segment {segment_index} has no synthesized audio noted")
        now = self._clock.now_ms()
        if self._frames:
            before = self._frames[-1]
            start = max(before.start_ms + self._ms(before.samples), float(now))
        else:
            start = float(now + self._prebuffer_ms)
        frame = _RecordedFrame(segment_index, segment_offset, samples, self._total, now, start)
        self._frames.append(frame)
        self._firsts.append(frame.first_sample)
        self._starts.append(start)
        self._total += samples

    def mark_ended(self) -> None:
        """The tag's terminal frame is queued: no frame follows."""
        self._ended = True

    def report(self, played_ms: int) -> None:
        """Take the client's report of how much of the tag it has played."""
        self._report_times.append(self._clock.now_ms())
        self._report_samples.append(played_ms * self._sample_rate // 1000)

    def played_samples(self, at_ms: int | None = None) -> int:
        """Samples of recorded audio played by `at_ms` (now when None)."""
        at = self._clock.now_ms() if at_ms is None else at_ms
        if self._report_times:
            index = bisect_right(self._report_times, at) - 1
            return min(self._report_samples[index], self._total) if index >= 0 else 0
        index = bisect_right(self._starts, at) - 1
        if index < 0:
            return 0
        frame = self._frames[index]
        into = int((at - frame.start_ms) * self._sample_rate / 1000)
        return frame.first_sample + min(frame.samples, into)

    def position_ms(self) -> int:
        """Milliseconds of the tag's audio played now."""
        return self.played_samples() * 1000 // self._sample_rate

    def played_through(self) -> bool:
        """The tag has ended and all its recorded audio has played, by the current source."""
        return self._ended and self.played_samples() >= self._total

    def spoken_chars(self) -> int:
        """The length of the spoken prefix of the displayed text."""
        return self._spoken_offset(self.played_samples())

    def spoken_text(self) -> str:
        """The displayed text the user has heard, ending at a word boundary."""
        return self._display_text()[: self.spoken_chars()]

    def recent_tts_text(self, window_ms: int) -> str:
        """The text played in the last `window_ms`, for the echo guard.

        A word partly played at either edge of the window is included whole, because its
        sound may still reach the microphone.
        """
        now = self._clock.now_ms()
        end = self.played_samples(now)
        start = self.played_samples(now - window_ms)
        if end <= start:
            return ""
        return self._display_text()[self._window_start(start) : self._window_end(end)].strip()

    def scheduled_play_ms(self, sample: int) -> int:
        """When `sample` of the tag's stream is scheduled to play, on the session clock.

        With client reports it is placed from the latest report, and not before the frame
        holding it was queued; without, it is the estimate's schedule. A sample not yet
        recorded is placed after the recorded audio.
        """
        frame = self._frame_holding(sample)
        if self._report_times:
            at = self._report_times[-1] + self._ms(sample - self._report_samples[-1])
            if frame is not None:
                at = max(at, float(frame.recorded_ms))
            return math.floor(at)
        if frame is not None:
            return math.floor(frame.start_ms + self._ms(sample - frame.first_sample))
        now = self._clock.now_ms()
        if self._frames:
            last = self._frames[-1]
            next_start = max(last.start_ms + self._ms(last.samples), float(now))
        else:
            next_start = float(now + self._prebuffer_ms)
        return math.floor(next_start + self._ms(sample - self._total))

    def _ms(self, samples: int) -> float:
        return samples * 1000 / self._sample_rate

    def _frame_holding(self, sample: int) -> _RecordedFrame | None:
        if not 0 <= sample < self._total:
            return None
        return self._frames[bisect_right(self._firsts, sample) - 1]

    def _share(self, frame: _RecordedFrame, played_in_frame: int) -> tuple[_SegmentAudio, float]:
        """The segment a frame voices and the fraction of its characters played."""
        audio = self._segments[frame.segment_index]
        played = frame.segment_offset + played_in_frame
        fraction = 1.0 if played >= audio.synthesized else played / audio.synthesized
        return audio, fraction * len(audio.segment.text)

    def _spoken_offset(self, played: int) -> int:
        if played <= 0 or not self._frames:
            return 0
        index = bisect_left(self._firsts, played) - 1
        frame = self._frames[index]
        audio, chars = self._share(frame, played - frame.first_sample)
        text = audio.segment.text
        cut = math.floor(chars)
        if cut >= len(text):
            return audio.segment.end_char
        if 0 < cut < len(text) and not text[cut].isspace():
            while cut > 0 and not text[cut - 1].isspace():
                cut -= 1
        word_end = len(text[:cut].rstrip())
        if word_end > 0:
            return audio.segment.start_char + word_end
        # Nothing of this segment counts yet: the prefix ends with the segment before it, all
        # of whose recorded frames have played.
        first = index
        while first > 0 and self._frames[first - 1].segment_index == frame.segment_index:
            first -= 1
        if first == 0:
            return 0
        return self._segments[self._frames[first - 1].segment_index].segment.end_char

    def _window_start(self, played: int) -> int:
        index = bisect_right(self._firsts, played) - 1
        frame = self._frames[index]
        audio, chars = self._share(frame, played - frame.first_sample)
        text = audio.segment.text
        cut = min(math.floor(chars), len(text))
        while cut > 0 and not text[cut - 1].isspace():
            cut -= 1
        return audio.segment.start_char + cut

    def _window_end(self, played: int) -> int:
        index = bisect_left(self._firsts, played) - 1
        frame = self._frames[index]
        audio, chars = self._share(frame, played - frame.first_sample)
        text = audio.segment.text
        cut = min(math.ceil(chars), len(text))
        if 0 < cut < len(text) and not text[cut - 1].isspace():
            while cut < len(text) and not text[cut].isspace():
                cut += 1
        return audio.segment.start_char + cut


# -- The pacer ------------------------------------------------------------------------


class _PendingSegment:
    """A segment's synthesized audio not yet framed: a working buffer, mutated in place."""

    __slots__ = ("index", "segment", "audio", "framed", "synthesized", "closed")

    def __init__(self, index: int, segment: TextSegment) -> None:
        self.index = index
        self.segment = segment
        self.audio = bytearray()
        self.framed = 0  # samples of the segment already framed
        self.synthesized = 0
        self.closed = False  # no more audio will arrive for the segment


@dataclass(frozen=True, slots=True)
class _FrameAudio:
    segment_index: int
    segment_offset: int
    samples: int


class PlaybackPacer:
    """Frames and paces one tag's synthesized audio onto the `turn` lane.

    The synthesize step hands audio over with `submit` and `finish`, and the turn's play task
    runs `run`, which queues the frames as the lead allows and the `0x02` once all audio is
    queued. A frame never spans two segments, so each segment starts on a frame boundary; a
    segment's last frame may be shorter than 40 ms.

    Each 0x01 is recorded in the ledger only when the turn sink's `put` returns True; a frame
    kept back is recorded by `record_flushed` when the controller later queues it. Audio that
    `end_unsent` or `stop` drops is never recorded.
    """

    def __init__(
        self,
        *,
        turn_id: str,
        audio_tag: int,
        turn_sink: TurnSink,
        control: Callable[[object], None],
        ledger: PlaybackLedger,
        clock: PlaybackClock,
        config: PlaybackConfig,
        listener: PlaybackListener | None = None,
        sample_rate: int = DOWNLINK_SAMPLE_RATE,
    ) -> None:
        """Build the pacer for one tag.

        Args:
            turn_id: The turn the tag belongs to, which the lane purges by.
            audio_tag: The turn's downlink tag, 1..65535.
            turn_sink: Where `0x01` and `0x02` frames are queued.
            control: Queues an item on the session's `control` lane: the `0x03` and the
                `playback_stop` of a stop.
            ledger: The tag's ledger.
            clock: The session clock.
            config: Frame length, lead and fade.
            listener: The scene seam's hooks; `NullPlaybackListener` when None.
            sample_rate: The downlink sample rate.
        """
        self._turn_id = turn_id
        self._tag = audio_tag
        self._sink = turn_sink
        self._control = control
        self._ledger = ledger
        self._clock = clock
        self._config = config
        self._listener: PlaybackListener = listener or NullPlaybackListener()
        self._sample_rate = sample_rate
        self._frame_samples = config.frame_ms * sample_rate // 1000
        self._max_lead_samples = config.max_lead_ms * sample_rate // 1000
        self._pending: deque[_PendingSegment] = deque()
        self._last_index: int | None = None
        self._kept: dict[int, _FrameAudio] = {}
        self._seq = 0
        self._handed = 0  # samples handed to the turn sink: the stream's sample offset
        self._finished = False  # no more audio will be submitted
        self._ended = False  # nothing more of the tag will be queued
        self._terminal_queued = False
        self._stopped = False
        self._wake = asyncio.Event()

    @property
    def listener(self) -> PlaybackListener:
        """The hooks the pacer calls before a segment's first frame and the terminal frame."""
        return self._listener

    def submit(self, segment_index: int, chunk: SynthChunk) -> None:
        """Hand over a chunk of segment `segment_index`'s audio, in synthesis order.

        A chunk of a later segment closes the segments before it. Audio arriving after
        `finish`, `end_unsent` or `stop` is dropped.

        Raises:
            ValueError: `segment_index` is lower than the last segment's.
        """
        if self._finished or self._ended:
            return
        if self._last_index is not None and segment_index < self._last_index:
            raise ValueError(f"segment {segment_index} arrived after segment {self._last_index}")
        tail = self._pending[-1] if self._pending else None
        if tail is None or tail.index != segment_index:
            if tail is not None:
                tail.closed = True
            tail = _PendingSegment(segment_index, chunk.segment)
            self._pending.append(tail)
        tail.audio += chunk.pcm16
        tail.synthesized += len(chunk.pcm16) // PCM16_SAMPLE_BYTES
        self._last_index = segment_index
        self._ledger.note_synthesized(segment_index, chunk.segment, tail.synthesized)
        self._wake.set()

    def finish(self) -> None:
        """No more audio: `run` queues the `0x02` once it has queued every frame."""
        if self._finished or self._ended:
            return
        self._finished = True
        if self._pending:
            self._pending[-1].closed = True
        self._wake.set()

    async def run(self) -> None:
        """The play task: queue frames within the lead, then the terminal frame.

        It returns after the `0x02` is queued, or at its next step once `end_unsent` or `stop`
        has been called, queueing nothing more even if it was suspended between frames.
        """
        while not self._ended:
            step = self._next_frame()
            if step is None:
                if self._finished:
                    self._queue_terminal()
                    return
                self._wake.clear()
                await self._wake.wait()
                continue
            segment, samples = step
            wait_ms = self._lead_wait_ms(samples)
            if wait_ms > 0:
                await self._clock.sleep_ms(min(wait_ms, self._config.frame_ms))
                continue
            self._hand_off(segment, samples)

    def report_progress(self, played_ms: int) -> None:
        """Take the client's `playback_progress` for this tag."""
        self._ledger.report(played_ms)

    def scheduled_play_ms(self, sample: int) -> int:
        """When `sample` of the tag's stream is scheduled to play, from the client's progress
        or, without any, the ledger's estimate (T27's `scene_anchor_lead_ms` reads it)."""
        return self._ledger.scheduled_play_ms(sample)

    def stop(self, terminal_purged: bool, *, reason: str = "interrupt") -> None:
        """Stop the tag at a cut: send `0x03` and then `playback_stop` on the control lane.

        The `0x03` is left out only when the pacer queued its `0x02` and the purge did not
        remove it, because then the sender took it and the client already knows the tag ended
        cleanly. The pacer is inert afterwards; a second stop sends nothing.

        Args:
            terminal_purged: The cut's `LanePurge.terminal`, from the purge run in the same
                synchronous step.
            reason: The `playback_stop` reason.
        """
        if self._stopped:
            return
        self._stopped = True
        self._ended = True
        self._pending.clear()
        self._wake.set()
        if not (self._terminal_queued and not terminal_purged):
            self._control(self._frame(MsgType.INTERRUPT_FADE))
        self._control(
            PlaybackStopMessage(
                turn_id=self._turn_id,
                audio_tag=self._tag,
                fade_ms=self._config.fade_ms,
                reason=reason,
            )
        )

    def end_unsent(self) -> None:
        """End the tag as a text reply: drop the audio not yet handed over and queue the `0x02`.

        The controller calls it after a `degraded tts_failed` when no `0x01` of the tag has
        reached the client. No `0x01` is queued afterwards; a call after the terminal frame or
        a stop does nothing, and a later `stop` follows its own rule.
        """
        if self._ended:
            return
        self._finished = True
        self._queue_terminal()

    def record_flushed(self, frame: object) -> None:
        """Record a frame the turn sink kept back, now that the controller has queued it.

        Raises:
            ValueError: The frame is not one of this pacer's kept-back frames.
        """
        if (
            not isinstance(frame, Frame)
            or frame.turn_tag != self._tag
            or frame.seq not in self._kept
        ):
            raise ValueError("record_flushed takes a frame this pacer's sink kept back")
        audio = self._kept.pop(frame.seq)
        self._ledger.record(audio.segment_index, audio.segment_offset, audio.samples)

    def _next_frame(self) -> tuple[_PendingSegment, int] | None:
        """The segment and length of the next frame, or None when it is not complete yet."""
        while self._pending:
            head = self._pending[0]
            available = len(head.audio) // PCM16_SAMPLE_BYTES
            if available >= self._frame_samples:
                return head, self._frame_samples
            if not head.closed:
                return None
            if available > 0:
                return head, available
            self._pending.popleft()
        return None

    def _lead_wait_ms(self, samples: int) -> int:
        """How long until a frame of `samples` fits within the lead, 0 when it fits now."""
        ahead = self._handed + samples - self._ledger.played_samples()
        excess = ahead - self._max_lead_samples
        return math.ceil(excess * 1000 / self._sample_rate) if excess > 0 else 0

    def _hand_off(self, segment: _PendingSegment, samples: int) -> None:
        size = samples * PCM16_SAMPLE_BYTES
        payload = bytes(segment.audio[:size])
        del segment.audio[:size]
        offset = segment.framed
        segment.framed += samples
        if offset == 0:
            self._listener.segment_starting(self._tag, segment.index, self._handed)
        frame = self._frame(MsgType.AUDIO_CHUNK, payload)
        audio = _FrameAudio(segment.index, offset, samples)
        self._handed += samples
        if self._sink.put(self._turn_id, frame):
            self._ledger.record(audio.segment_index, audio.segment_offset, audio.samples)
        else:
            self._kept[frame.seq] = audio

    def _queue_terminal(self) -> None:
        self._ended = True
        self._pending.clear()
        self._wake.set()
        self._listener.stream_ending(self._tag, self._handed)
        self._sink.put(self._turn_id, self._frame(MsgType.AUDIO_COMPLETE), terminal=True)
        self._terminal_queued = True
        self._ledger.mark_ended()

    def _frame(self, msg_type: MsgType, payload: bytes = b"") -> Frame:
        frame = Frame(
            msg_type=msg_type,
            turn_tag=self._tag,
            seq=self._seq,
            sample_rate=self._sample_rate,
            payload=payload,
        )
        self._seq += 1
        return frame
