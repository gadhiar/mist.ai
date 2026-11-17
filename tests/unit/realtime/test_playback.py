"""Tests for the session's `turn` lane, the playback pacer and the playback ledger.

The pacer is the only writer of a turn's downlink audio, so it carries the rules that keep a cut
clean: exactly one terminal frame per tag, nothing of a tag after it, and a `0x03` exactly when
the client may still hold audio the sender passed on. The lane's purge is what tells the cut
whether the sender took the `0x02`, so the pacer's terminal frames are tested against the real
`TurnLane`. The ledger turns played time into the spoken prefix that is recorded for an
interrupted turn, so it must never count audio that was dropped or kept back. Every test runs on
a fake clock that moves only when the test advances it.
"""

import asyncio
import heapq
import struct
from collections import deque
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio

from backend.realtime.budget import PlaybackConfig
from backend.realtime.errors import RealtimeError
from backend.realtime.playback import (
    LaneFull,
    LanePurge,
    MonotonicPlaybackClock,
    NullPlaybackListener,
    PlaybackLedger,
    PlaybackListener,
    PlaybackPacer,
    TurnLane,
)
from backend.realtime.protocol import (
    DOWNLINK_SAMPLE_RATE,
    Frame,
    MsgType,
    PlaybackStopMessage,
)
from backend.realtime.speech import SynthChunk, TextSegment

TURN = "turn-1"
TAG = 7
SAMPLES_PER_MS = DOWNLINK_SAMPLE_RATE // 1000
FRAME_SAMPLES = 40 * SAMPLES_PER_MS


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


class FakeClock:
    """A session clock that moves only when the test calls `advance`."""

    def __init__(self) -> None:
        self.now = 0
        self._sleepers: list[tuple[int, int, asyncio.Future[None]]] = []
        self._order = 0

    def now_ms(self) -> int:
        return self.now

    async def sleep_ms(self, ms: int) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._order += 1
        heapq.heappush(self._sleepers, (self.now + ms, self._order, future))
        await future

    @property
    def sleeping(self) -> int:
        return sum(1 for _, _, future in self._sleepers if not future.done())

    async def advance(self, ms: int) -> None:
        target = self.now + ms
        await _settle()
        while self._sleepers and self._sleepers[0][0] <= target:
            wake, _, future = heapq.heappop(self._sleepers)
            self.now = wake
            if not future.done():
                future.set_result(None)
            await _settle()
        self.now = target
        await _settle()


def _segment(text: str, start: int = 0) -> TextSegment:
    return TextSegment(text, start, start + len(text))


def _chunk(segment: TextSegment, ms: int, fill: int = 1) -> SynthChunk:
    samples = ms * SAMPLES_PER_MS
    return SynthChunk(struct.pack("<h", fill) * samples, samples, segment)


def _drain(lane: TurnLane) -> list[object]:
    items = []
    while (item := lane.take()) is not None:
        items.append(item)
    return items


def _kinds(items: list[object]) -> list[str]:
    names = []
    for item in items:
        if isinstance(item, Frame):
            names.append(f"0x{int(item.msg_type):02X}")
        elif isinstance(item, PlaybackStopMessage):
            names.append("playback_stop")
        else:
            names.append(repr(item))
    return names


class RecordingListener:
    """Satisfies `PlaybackListener` structurally, as T11's `ScenePort` will."""

    def __init__(self, log: list[tuple[object, ...]]) -> None:
        self.log = log

    def segment_starting(self, audio_tag: int, segment_index: int, first_sample: int) -> None:
        self.log.append(("segment_starting", audio_tag, segment_index, first_sample))

    def stream_ending(self, audio_tag: int, last_sample: int) -> None:
        self.log.append(("stream_ending", audio_tag, last_sample))


class RecordingSink:
    """A turn sink in front of a real lane that logs each put, with the clock and the lead."""

    def __init__(
        self,
        lane: TurnLane,
        log: list[tuple[object, ...]],
        clock: FakeClock,
        ledger: PlaybackLedger,
    ) -> None:
        self.lane = lane
        self.log = log
        self.clock = clock
        self.ledger = ledger
        self.audio_samples = 0
        self.leads_ms: list[float] = []
        self.put_times: list[int] = []

    def put(self, turn_id, item, terminal=False, kind=None, token_index=None) -> bool:
        if isinstance(item, Frame) and item.msg_type is MsgType.AUDIO_CHUNK:
            self.audio_samples += len(item.payload) // 2
            played = self.ledger.position_ms()
            self.leads_ms.append(self.audio_samples * 1000 / DOWNLINK_SAMPLE_RATE - played)
            self.put_times.append(self.clock.now_ms())
        self.log.append(("put", item))
        return self.lane.put(turn_id, item, terminal=terminal, kind=kind, token_index=token_index)


class HoldingSink:
    """T11's hold gate: keeps every item back and reports that it did not queue it."""

    def __init__(self) -> None:
        self.held: list[object] = []

    def put(self, turn_id, item, terminal=False, kind=None, token_index=None) -> bool:
        self.held.append(item)
        return False


class Harness:
    def __init__(
        self,
        *,
        config: PlaybackConfig | None = None,
        sink: object | None = None,
        listener: PlaybackListener | None = None,
        display_text: str = "",
    ) -> None:
        self.clock = FakeClock()
        self.lane = TurnLane()
        self.control: list[object] = []
        self.log: list[tuple[object, ...]] = []
        self.display_text = display_text
        self.config = config or PlaybackConfig()
        self.ledger = PlaybackLedger(
            lambda: self.display_text,
            self.clock,
            prebuffer_ms=self.config.client_prebuffer_ms,
        )
        self.sink = sink or RecordingSink(self.lane, self.log, self.clock, self.ledger)
        kwargs = {} if listener is None else {"listener": listener}
        self.pacer = PlaybackPacer(
            turn_id=TURN,
            audio_tag=TAG,
            turn_sink=self.sink,
            control=self.control.append,
            ledger=self.ledger,
            clock=self.clock,
            config=self.config,
            **kwargs,
        )
        self.task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self.pacer.run())
        await _settle()

    async def finish(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await self.task


@pytest_asyncio.fixture
async def harness() -> AsyncIterator[Harness]:
    h = Harness()
    yield h
    await h.finish()


# -- TurnLane ------------------------------------------------------------------


class TestTurnLane:
    def test_take_returns_items_oldest_first_and_none_when_empty(self):
        lane = TurnLane()

        for item in ("a", "b", "c"):
            assert lane.put(TURN, item) is True

        assert _drain(lane) == ["a", "b", "c"]
        assert lane.take() is None

    def test_put_raises_lane_full_at_max_items(self):
        lane = TurnLane(max_items=3)
        for item in ("a", "b", "c"):
            lane.put(TURN, item)

        with pytest.raises(LaneFull):
            lane.put(TURN, "d")

        assert isinstance(LaneFull("x"), RealtimeError)
        assert _drain(lane) == ["a", "b", "c"]

    def test_purge_removes_only_that_turns_items(self):
        lane = TurnLane()
        lane.put("t1", "a")
        lane.put("t2", "x")
        lane.put("t1", "b")
        lane.put("t2", "y")

        lane.purge("t1")

        assert _drain(lane) == ["x", "y"]

    @pytest.mark.parametrize(
        ("put_terminal", "taken", "expected"),
        [
            pytest.param(True, False, True, id="queued"),
            pytest.param(True, True, False, id="taken"),
            pytest.param(False, False, False, id="never-put"),
        ],
    )
    def test_purge_reports_a_queued_terminal_item(self, put_terminal, taken, expected):
        lane = TurnLane()
        lane.put(TURN, "frame")
        if put_terminal:
            lane.put(TURN, "0x02", terminal=True)
        if taken:
            _drain(lane)

        purge = lane.purge(TURN)

        assert purge.terminal is expected

    @pytest.mark.parametrize(
        ("put_start", "taken", "expected"),
        [
            pytest.param(True, False, True, id="queued"),
            pytest.param(True, True, False, id="taken"),
            pytest.param(False, False, False, id="never-put"),
        ],
    )
    def test_purge_reports_a_queued_stream_start(self, put_start, taken, expected):
        lane = TurnLane()
        if put_start:
            lane.put(TURN, "stream_start", kind="stream_start")
        if taken:
            _drain(lane)
        lane.put(TURN, "other")

        purge = lane.purge(TURN)

        assert purge.stream_start is expected

    @pytest.mark.parametrize(
        ("taken", "expected"),
        [
            pytest.param(3, 3, id="three-taken"),
            pytest.param(6, None, id="all-taken"),
            pytest.param(0, 0, id="none-taken"),
        ],
    )
    def test_purge_reports_the_first_removed_token(self, taken, expected):
        lane = TurnLane()
        for index in range(6):
            lane.put(TURN, f"token-{index}", kind="token", token_index=index)
        for _ in range(taken):
            lane.take()

        purge = lane.purge(TURN)

        assert purge.first_token == expected

    def test_another_turns_tokens_are_neither_removed_nor_counted(self):
        lane = TurnLane()
        for index in range(6):
            lane.put(TURN, f"token-{index}", kind="token", token_index=index)
            lane.put("other", f"other-{index}", kind="token", token_index=index + 10)
        for _ in range(4):
            lane.take()

        purge = lane.purge(TURN)

        assert purge == LanePurge(terminal=False, stream_start=False, first_token=2)
        assert _drain(lane) == [f"other-{index}" for index in range(2, 6)]

    @pytest.mark.parametrize(
        ("kind", "token_index"),
        [
            pytest.param("token", None, id="token-without-index"),
            pytest.param(None, 3, id="index-without-token"),
            pytest.param("stream_start", 0, id="index-on-stream-start"),
        ],
    )
    def test_token_index_is_set_exactly_for_tokens(self, kind, token_index):
        lane = TurnLane()

        with pytest.raises(ValueError):
            lane.put(TURN, "item", kind=kind, token_index=token_index)

    def test_a_queued_terminal_is_taken_or_purged_never_both(self):
        taken_first = TurnLane()
        taken_first.put(TURN, "0x02", terminal=True)
        purged_first = TurnLane()
        purged_first.put(TURN, "0x02", terminal=True)

        taken = taken_first.take()
        after_take = taken_first.purge(TURN)
        after_purge = purged_first.purge(TURN)
        taken_after_purge = purged_first.take()

        assert taken == "0x02"
        assert after_take.terminal is False
        assert after_purge.terminal is True
        assert taken_after_purge is None


class RecordingLock:
    def __init__(self, log: list[tuple[str, bool]]) -> None:
        self.log = log
        self.held = False

    def __enter__(self) -> "RecordingLock":
        self.log.append(("acquire", self.held))
        self.held = True
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.held = False
        self.log.append(("release", self.held))


class RecordingStore:
    """Stands in for the lane's `collections.deque` and logs whether each access held the lock."""

    def __init__(self, lock: RecordingLock, log: list[tuple[str, bool]]) -> None:
        self._items: deque[object] = deque()
        self._lock = lock
        self._log = log

    def _touch(self, name: str) -> None:
        self._log.append((f"store.{name}", self._lock.held))

    def append(self, item: object) -> None:
        self._touch("append")
        self._items.append(item)

    def popleft(self) -> object:
        self._touch("popleft")
        return self._items.popleft()

    def clear(self) -> None:
        self._touch("clear")
        self._items.clear()

    def extend(self, items: object) -> None:
        self._touch("extend")
        self._items.extend(items)

    def __len__(self) -> int:
        self._touch("len")
        return len(self._items)

    def __bool__(self) -> bool:
        self._touch("bool")
        return bool(self._items)

    def __iter__(self) -> Iterator[object]:
        self._touch("iter")
        return iter(list(self._items))


class TestTurnLaneLocking:
    def _lane(self, max_items: int = 2048) -> tuple[TurnLane, list[tuple[str, bool]]]:
        log: list[tuple[str, bool]] = []
        lock = RecordingLock(log)
        lane = TurnLane(max_items=max_items, lock=lock)
        lane._items = RecordingStore(lock, log)
        return lane, log

    def _assert_one_locked_call(self, log: list[tuple[str, bool]]) -> None:
        assert log[0] == ("acquire", False)
        assert log[-1] == ("release", False)
        assert [entry for entry in log if entry[0] in ("acquire", "release")] == [
            ("acquire", False),
            ("release", False),
        ]
        touches = log[1:-1]
        assert touches
        assert all(name.startswith("store.") and held for name, held in touches)

    def test_put_holds_the_lock_for_every_store_access(self):
        lane, log = self._lane()

        lane.put(TURN, "a")

        self._assert_one_locked_call(log)

    def test_take_holds_the_lock_for_every_store_access(self):
        lane, log = self._lane()
        lane.put(TURN, "a")
        log.clear()

        lane.take()

        self._assert_one_locked_call(log)

    def test_take_of_an_empty_lane_holds_the_lock(self):
        lane, log = self._lane()

        assert lane.take() is None

        self._assert_one_locked_call(log)

    def test_purge_holds_the_lock_for_every_store_access(self):
        lane, log = self._lane()
        lane.put(TURN, "a", kind="token", token_index=0)
        lane.put("other", "b")
        lane.put(TURN, "0x02", terminal=True)
        log.clear()

        purge = lane.purge(TURN)

        self._assert_one_locked_call(log)
        assert purge == LanePurge(terminal=True, stream_start=False, first_token=0)
        assert _drain(lane) == ["b"]

    def test_put_on_a_full_lane_raises_and_still_releases_the_lock(self):
        lane, log = self._lane(max_items=1)
        lane.put(TURN, "a")
        log.clear()

        with pytest.raises(LaneFull):
            lane.put(TURN, "b")

        self._assert_one_locked_call(log)


# -- PlaybackPacer: frames and pacing ------------------------------------------


class TestFraming:
    @pytest.mark.asyncio
    async def test_audio_goes_out_as_40_ms_frames_with_the_turns_tag(self, harness):
        segment = _segment("Hello there.")
        harness.pacer.submit(0, _chunk(segment, 60, fill=1))
        harness.pacer.submit(0, _chunk(segment, 40, fill=2))
        harness.pacer.finish()

        await harness.start()

        items = _drain(harness.lane)
        assert _kinds(items) == ["0x01", "0x01", "0x01", "0x02"]
        assert all(item.turn_tag == TAG for item in items)
        assert [item.seq for item in items] == [0, 1, 2, 3]
        assert [len(item.payload) // 2 for item in items[:3]] == [960, 960, 480]
        audio = b"".join(item.payload for item in items[:3])
        assert audio == _chunk(segment, 60, fill=1).pcm16 + _chunk(segment, 40, fill=2).pcm16
        assert all(item.sample_rate == DOWNLINK_SAMPLE_RATE for item in items)

    @pytest.mark.asyncio
    async def test_a_frame_never_spans_two_segments(self, harness):
        first = _segment("One.")
        second = _segment("Two.", 5)
        harness.pacer.submit(0, _chunk(first, 50))
        harness.pacer.submit(1, _chunk(second, 50))
        harness.pacer.finish()

        await harness.start()

        frames = _drain(harness.lane)[:-1]
        assert [len(item.payload) // 2 for item in frames] == [960, 240, 960, 240]

    @pytest.mark.asyncio
    async def test_the_tail_of_a_segment_waits_until_the_segment_is_known_to_end(self, harness):
        segment = _segment("One.")
        harness.pacer.submit(0, _chunk(segment, 60))
        await harness.start()

        before = _drain(harness.lane)
        harness.pacer.submit(1, _chunk(_segment("Two.", 5), 40))
        await _settle()
        after = _drain(harness.lane)

        assert [len(item.payload) // 2 for item in before] == [960]
        assert [len(item.payload) // 2 for item in after] == [480, 960]

    @pytest.mark.asyncio
    async def test_a_segment_index_lower_than_the_last_is_refused(self, harness):
        harness.pacer.submit(1, _chunk(_segment("Two."), 40))

        with pytest.raises(ValueError):
            harness.pacer.submit(0, _chunk(_segment("One."), 40))


def _sizes(items: list[object]) -> list[int | str]:
    """Each `0x01`'s sample count, and every other frame by its kind."""
    return [
        (
            len(item.payload) // 2
            if isinstance(item, Frame) and item.msg_type is MsgType.AUDIO_CHUNK
            else kind
        )
        for item, kind in zip(items, _kinds(items), strict=True)
    ]


class TestEndSegment:
    @pytest.mark.asyncio
    async def test_end_segment_sends_the_short_last_frame_before_the_next_segment(self, harness):
        harness.pacer.submit(0, _chunk(_segment("One."), 60))
        await harness.start()
        held = _drain(harness.lane)

        harness.pacer.end_segment(0)
        await _settle()
        ended = _drain(harness.lane)
        harness.pacer.submit(1, _chunk(_segment("Two.", 5), 40))
        await _settle()

        assert _sizes(held + ended) == [960, 480]
        assert _sizes(_drain(harness.lane)) == [960]
        assert harness.ledger.recorded_samples == 960 + 480 + 960

    @pytest.mark.asyncio
    async def test_end_segment_after_finish_changes_nothing(self, harness):
        harness.pacer.submit(0, _chunk(_segment("One."), 60))
        harness.pacer.finish()

        harness.pacer.end_segment(0)
        await harness.start()

        assert _sizes(_drain(harness.lane)) == [960, 480, "0x02"]

    @pytest.mark.asyncio
    async def test_end_segment_after_end_unsent_queues_no_audio(self, harness):
        harness.pacer.submit(0, _chunk(_segment("Never heard."), 60))
        harness.pacer.end_unsent()

        harness.pacer.end_segment(0)
        await harness.start()

        assert _sizes(_drain(harness.lane)) == ["0x02"]
        assert harness.ledger.recorded_samples == 0

    @pytest.mark.asyncio
    async def test_end_segment_after_stop_queues_nothing(self, harness):
        harness.pacer.submit(0, _chunk(_segment("One."), 60))
        await harness.start()
        before = len(harness.log)
        harness.pacer.stop(harness.lane.purge(TURN).terminal)

        harness.pacer.end_segment(0)
        await harness.clock.advance(1000)

        assert len(harness.log) == before
        assert harness.lane.take() is None
        assert _kinds(harness.control) == ["0x03", "playback_stop"]

    @pytest.mark.asyncio
    async def test_end_segment_of_an_index_with_no_pending_audio_changes_nothing(self, harness):
        harness.pacer.end_segment(0)  # nothing submitted yet
        harness.pacer.submit(0, _chunk(_segment("One."), 60))
        await harness.start()

        harness.pacer.end_segment(1)  # not submitted
        await _settle()
        before = _drain(harness.lane)
        harness.pacer.submit(1, _chunk(_segment("Two.", 5), 60))
        harness.pacer.end_segment(0)  # already closed by segment 1
        await _settle()
        after = _drain(harness.lane)

        assert _sizes(before) == [960]
        assert _sizes(after) == [480, 960]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("framed", [False, True], ids=["pending", "framed"])
    async def test_a_submit_of_a_segment_that_has_ended_is_refused(self, harness, framed):
        harness.pacer.submit(0, _chunk(_segment("One."), 80))
        harness.pacer.end_segment(0)
        if framed:
            await harness.start()
            assert _sizes(_drain(harness.lane)) == [960, 960]

        with pytest.raises(ValueError):
            harness.pacer.submit(0, _chunk(_segment("One."), 40))
        harness.pacer.submit(1, _chunk(_segment("Two.", 5), 40))


class TestPacing:
    @pytest.mark.asyncio
    async def test_with_the_reported_position_held_the_pacer_stops_at_max_lead(self, harness):
        harness.pacer.submit(0, _chunk(_segment("A long reply."), 2000))
        await harness.start()
        harness.pacer.report_progress(0)

        await harness.clock.advance(1000)
        held = len(_drain(harness.lane))
        harness.pacer.report_progress(100)
        await harness.clock.advance(40)
        resumed = len(_drain(harness.lane))

        # At 100 ms played, the lead allows audio up to 600 ms: three more 40 ms frames.
        assert held == 12
        assert resumed == 3
        assert max(harness.sink.leads_ms) <= 500

    @pytest.mark.asyncio
    async def test_never_runs_more_than_max_lead_ahead_of_a_reporting_client(self, harness):
        harness.pacer.submit(0, _chunk(_segment("A long reply."), 3000))
        harness.pacer.finish()
        await harness.start()

        for _ in range(60):
            played = max(0, min(harness.clock.now - 80, 3000))
            harness.pacer.report_progress(played)
            await harness.clock.advance(100)

        items = _drain(harness.lane)
        assert _kinds(items)[-1] == "0x02"
        assert len(items) == 3000 // 40 + 1
        assert max(harness.sink.leads_ms) <= 500
        assert harness.ledger.source == "client"

    @pytest.mark.asyncio
    async def test_without_reports_the_position_is_estimated_from_the_clock(self, harness):
        harness.pacer.submit(0, _chunk(_segment("A long reply."), 2000))
        await harness.start()

        await harness.clock.advance(220)

        # 12 frames (480 ms) fit in the 500 ms lead at once. The estimate starts playing at the
        # 80 ms prebuffer, so the 13th frame fits once 20 ms have played, at 100 ms, and one
        # more frame fits every 40 ms after that.
        assert harness.sink.put_times == [0] * 12 + [100, 140, 180, 220]
        assert max(harness.sink.leads_ms) <= 500
        assert harness.ledger.source == "estimate"


class TestTerminalFrames:
    @pytest.mark.asyncio
    async def test_the_end_of_the_stream_queues_0x02_marked_terminal_after_the_last_frame(
        self, harness
    ):
        harness.pacer.submit(0, _chunk(_segment("Hi."), 80))
        harness.pacer.finish()
        await harness.start()

        purge = harness.lane.purge(TURN)

        assert [entry[1].msg_type for entry in harness.log] == [
            MsgType.AUDIO_CHUNK,
            MsgType.AUDIO_CHUNK,
            MsgType.AUDIO_COMPLETE,
        ]
        assert purge.terminal is True
        assert harness.task.done()

    @pytest.mark.asyncio
    async def test_a_tag_with_no_frame_queues_0x02_alone(self, harness):
        harness.pacer.finish()
        await harness.start()

        assert _kinds(_drain(harness.lane)) == ["0x02"]

    @pytest.mark.asyncio
    async def test_a_cut_that_purges_the_queued_0x02_sends_0x03_then_playback_stop(self, harness):
        harness.pacer.submit(0, _chunk(_segment("Hi."), 80))
        harness.pacer.finish()
        await harness.start()

        purged = harness.lane.purge(TURN)
        harness.pacer.stop(purged.terminal)

        assert purged.terminal is True
        assert harness.lane.take() is None
        assert _kinds(harness.control) == ["0x03", "playback_stop"]
        assert harness.control[0].turn_tag == TAG
        stop = harness.control[1]
        assert (stop.turn_id, stop.audio_tag, stop.fade_ms) == (TURN, TAG, 30)

    @pytest.mark.asyncio
    async def test_a_cut_after_the_sender_took_the_0x02_sends_playback_stop_alone(self, harness):
        harness.pacer.submit(0, _chunk(_segment("Hi."), 80))
        harness.pacer.finish()
        await harness.start()
        taken = _drain(harness.lane)

        purged = harness.lane.purge(TURN)
        harness.pacer.stop(purged.terminal)

        assert _kinds(taken) == ["0x01", "0x01", "0x02"]
        assert purged.terminal is False
        assert _kinds(harness.control) == ["playback_stop"]

    @pytest.mark.asyncio
    async def test_a_cut_after_the_sender_took_the_frames_but_not_the_0x02_sends_0x03(
        self, harness
    ):
        harness.pacer.submit(0, _chunk(_segment("Hi."), 80))
        harness.pacer.finish()
        await harness.start()
        taken = [harness.lane.take(), harness.lane.take()]

        purged = harness.lane.purge(TURN)
        harness.pacer.stop(purged.terminal)

        assert _kinds(taken) == ["0x01", "0x01"]
        assert purged.terminal is True
        assert _kinds(harness.control) == ["0x03", "playback_stop"]

    @pytest.mark.asyncio
    async def test_a_stop_before_any_0x02_sends_0x03_then_playback_stop(self, harness):
        harness.pacer.submit(0, _chunk(_segment("Hi."), 80))
        await harness.start()
        _drain(harness.lane)

        harness.pacer.stop(harness.lane.purge(TURN).terminal)
        await _settle()

        assert _kinds(harness.control) == ["0x03", "playback_stop"]
        assert harness.lane.take() is None
        assert harness.task.done()

    @pytest.mark.asyncio
    async def test_a_stop_with_nothing_queued_sends_no_frame_before_0x03(self, harness):
        await harness.start()

        harness.pacer.stop(harness.lane.purge(TURN).terminal, reason="barge_in")
        await _settle()

        assert harness.log == []
        assert _kinds(harness.control) == ["0x03", "playback_stop"]
        assert harness.control[1].reason == "barge_in"

    @pytest.mark.asyncio
    async def test_a_second_end_or_stop_sends_nothing_and_no_0x01_follows_a_terminal(self, harness):
        segment = _segment("Hi.")
        harness.pacer.submit(0, _chunk(segment, 40))
        harness.pacer.finish()
        await harness.start()

        harness.pacer.finish()
        harness.pacer.submit(1, _chunk(_segment("More.", 4), 40))
        await harness.clock.advance(1000)
        harness.pacer.stop(False)
        harness.pacer.stop(True)

        assert _kinds(_drain(harness.lane)) == ["0x01", "0x02"]
        assert _kinds(harness.control) == ["playback_stop"]

    @pytest.mark.asyncio
    async def test_a_play_task_suspended_between_frames_queues_nothing_after_stop(self, harness):
        harness.pacer.submit(0, _chunk(_segment("A long reply."), 2000))
        harness.pacer.finish()
        await harness.start()
        assert harness.clock.sleeping == 1
        before = len(harness.log)

        harness.pacer.stop(harness.lane.purge(TURN).terminal)
        await harness.clock.advance(5000)

        assert len(harness.log) == before
        assert harness.lane.take() is None
        assert _kinds(harness.control) == ["0x03", "playback_stop"]
        assert harness.task.done()


class TestEndUnsent:
    @pytest.mark.asyncio
    async def test_end_unsent_queues_0x02_alone_and_drops_the_unsent_audio(self, harness):
        harness.pacer.submit(0, _chunk(_segment("Never heard."), 400))

        harness.pacer.end_unsent()
        await harness.start()
        await harness.clock.advance(1000)

        assert _kinds(_drain(harness.lane)) == ["0x02"]
        assert harness.lane.purge(TURN).terminal is False
        assert harness.ledger.recorded_samples == 0
        assert harness.ledger.spoken_text() == ""
        assert harness.ledger.played_through() is True

    @pytest.mark.asyncio
    async def test_end_unsent_stops_a_play_task_suspended_between_frames(self):
        sink = HoldingSink()
        h = Harness(sink=sink, display_text="Kept back.")
        h.pacer.submit(0, _chunk(_segment("Kept back."), 2000))
        await h.start()
        assert h.clock.sleeping == 1
        held = len(sink.held)

        h.pacer.end_unsent()
        await h.clock.advance(5000)

        assert held == 12
        assert _kinds(sink.held[held:]) == ["0x02"]
        assert h.ledger.recorded_samples == 0
        assert h.ledger.spoken_text() == ""
        assert h.task.done()

    @pytest.mark.asyncio
    async def test_a_stop_after_end_unsent_follows_the_terminal_rule(self, harness):
        harness.pacer.submit(0, _chunk(_segment("Never heard."), 400))
        harness.pacer.end_unsent()

        harness.pacer.stop(harness.lane.purge(TURN).terminal)

        assert _kinds(harness.control) == ["0x03", "playback_stop"]

    @pytest.mark.asyncio
    async def test_a_stop_after_the_sender_took_the_end_unsent_0x02_sends_playback_stop(
        self, harness
    ):
        harness.pacer.end_unsent()
        taken = _drain(harness.lane)

        harness.pacer.stop(harness.lane.purge(TURN).terminal)

        assert _kinds(taken) == ["0x02"]
        assert _kinds(harness.control) == ["playback_stop"]

    @pytest.mark.asyncio
    async def test_a_second_end_unsent_or_one_after_stop_queues_nothing(self, harness):
        harness.pacer.end_unsent()
        harness.pacer.end_unsent()
        stopped = Harness()
        stopped.pacer.stop(False)

        stopped.pacer.end_unsent()

        assert _kinds(_drain(harness.lane)) == ["0x02"]
        assert stopped.lane.take() is None
        assert stopped.log == []


# -- The ledger, as the pacer feeds it ------------------------------------------


class TestLedgerRecording:
    @pytest.mark.asyncio
    async def test_frames_a_sink_keeps_back_are_never_recorded(self):
        sink = HoldingSink()
        h = Harness(sink=sink, display_text="Kept back.")
        h.pacer.submit(0, _chunk(_segment("Kept back."), 200))
        h.pacer.finish()

        await h.start()
        await h.clock.advance(1000)

        assert _kinds(sink.held) == ["0x01"] * 5 + ["0x02"]
        assert h.ledger.recorded_samples == 0
        assert h.ledger.spoken_text() == ""

    @pytest.mark.asyncio
    async def test_record_flushed_records_exactly_the_frames_it_is_given(self):
        sink = HoldingSink()
        h = Harness(sink=sink, display_text="Kept back.")
        h.pacer.submit(0, _chunk(_segment("Kept back."), 200))
        h.pacer.finish()
        await h.start()
        frames = [item for item in sink.held if item.msg_type is MsgType.AUDIO_CHUNK]

        h.pacer.record_flushed(frames[0])
        h.pacer.record_flushed(frames[1])

        assert h.ledger.recorded_samples == 2 * FRAME_SAMPLES

    @pytest.mark.asyncio
    async def test_record_flushed_refuses_a_frame_it_did_not_keep_back(self, harness):
        harness.pacer.submit(0, _chunk(_segment("Hi."), 40))
        harness.pacer.finish()
        await harness.start()
        queued = _drain(harness.lane)[0]

        with pytest.raises(ValueError):
            harness.pacer.record_flushed(queued)

    @pytest.mark.asyncio
    async def test_frames_the_sink_queued_are_recorded(self, harness):
        harness.pacer.submit(0, _chunk(_segment("Hi."), 200))
        harness.pacer.finish()

        await harness.start()

        assert harness.ledger.recorded_samples == 200 * SAMPLES_PER_MS


# -- PlaybackListener -------------------------------------------------------------


class TestPlaybackListener:
    def _harness(self) -> Harness:
        log: list[tuple[object, ...]] = []
        h = Harness(listener=RecordingListener(log))
        h.log = log
        h.sink.log = log
        return h

    @pytest.mark.asyncio
    async def test_each_hook_is_called_just_before_its_frame_is_queued(self):
        h = self._harness()
        h.pacer.submit(0, _chunk(_segment("One."), 50))
        h.pacer.submit(2, _chunk(_segment("Two.", 5), 40))
        h.pacer.finish()

        await h.start()

        steps = [
            entry if entry[0] != "put" else ("put", _kinds([entry[1]])[0], entry[1].seq)
            for entry in h.log
        ]
        assert steps == [
            ("segment_starting", TAG, 0, 0),
            ("put", "0x01", 0),
            ("put", "0x01", 1),
            ("segment_starting", TAG, 2, 1200),
            ("put", "0x01", 2),
            ("stream_ending", TAG, 2160),
            ("put", "0x02", 3),
        ]

    @pytest.mark.asyncio
    async def test_a_frameless_tag_calls_stream_ending_at_sample_zero(self):
        h = self._harness()
        h.pacer.finish()

        await h.start()

        assert h.log[0] == ("stream_ending", TAG, 0)
        assert _kinds([h.log[1][1]]) == ["0x02"]

    @pytest.mark.asyncio
    async def test_a_purged_tag_calls_neither_hook_again(self):
        h = self._harness()
        h.pacer.submit(0, _chunk(_segment("A long reply."), 2000))
        await h.start()
        h.pacer.stop(h.lane.purge(TURN).terminal)
        calls = len(h.log)

        h.pacer.submit(1, _chunk(_segment("More.", 14), 400))
        h.pacer.finish()
        await h.clock.advance(5000)

        assert len(h.log) == calls
        await h.finish()

    @pytest.mark.asyncio
    async def test_a_segment_with_no_queued_frame_never_calls_segment_starting(self):
        h = self._harness()
        h.pacer.submit(0, _chunk(_segment("A long reply."), 600))
        h.pacer.submit(1, _chunk(_segment("Dropped.", 14), 400))
        await h.start()

        h.pacer.end_unsent()
        await h.clock.advance(5000)

        starts = [entry for entry in h.log if entry[0] == "segment_starting"]
        assert starts == [("segment_starting", TAG, 0, 0)]
        assert h.log[-2] == ("stream_ending", TAG, 12 * FRAME_SAMPLES)

    def test_the_default_listener_does_nothing(self):
        listener = NullPlaybackListener()

        assert listener.segment_starting(TAG, 0, 0) is None
        assert listener.stream_ending(TAG, 0) is None

    @pytest.mark.asyncio
    async def test_a_pacer_built_without_a_listener_uses_the_null_listener(self, harness):
        harness.pacer.submit(0, _chunk(_segment("Hi."), 40))
        harness.pacer.finish()

        await harness.start()

        assert isinstance(harness.pacer.listener, NullPlaybackListener)
        assert _kinds(_drain(harness.lane)) == ["0x01", "0x02"]


# -- Scheduled play time -----------------------------------------------------------


class TestScheduledPlay:
    @pytest.mark.asyncio
    async def test_without_reports_each_frame_is_scheduled_by_the_estimate(self, harness):
        harness.pacer.submit(0, _chunk(_segment("A long reply."), 2000))
        await harness.start()
        await harness.clock.advance(100)

        scheduled = [harness.pacer.scheduled_play_ms(k * FRAME_SAMPLES) for k in range(13)]

        # The first frame plays after the 80 ms prebuffer, the rest back to back.
        assert scheduled == [80 + 40 * k for k in range(13)]

    @pytest.mark.asyncio
    async def test_an_underrun_frame_is_scheduled_when_it_is_queued(self, harness):
        harness.pacer.submit(0, _chunk(_segment("One."), 40))
        await harness.start()
        await harness.clock.advance(1000)

        harness.pacer.submit(1, _chunk(_segment("Two.", 5), 40))
        await _settle()

        assert harness.pacer.scheduled_play_ms(FRAME_SAMPLES) == 1000

    @pytest.mark.asyncio
    async def test_with_reports_frames_are_scheduled_from_the_clients_progress(self, harness):
        harness.pacer.submit(0, _chunk(_segment("A long reply."), 2000))
        await harness.start()
        await harness.clock.advance(200)

        harness.pacer.report_progress(100)

        assert harness.pacer.scheduled_play_ms(12 * FRAME_SAMPLES) == 200 + (480 - 100)


# -- PlaybackLedger ------------------------------------------------------------------


FIRST = "Hello there, friend."
SECOND = "How are you today?"
DISPLAY = f"{FIRST} {SECOND}"


class LedgerHarness:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.ledger = PlaybackLedger(lambda: DISPLAY, self.clock, prebuffer_ms=80)
        self.first = _segment(FIRST)
        self.second = _segment(SECOND, len(FIRST) + 1)

    def record_ms(self, index: int, segment: TextSegment, start_ms: int, ms: int) -> None:
        for offset in range(start_ms, start_ms + ms, 40):
            self.ledger.record(index, offset * SAMPLES_PER_MS, FRAME_SAMPLES)

    def two_segments(self) -> None:
        self.ledger.note_synthesized(0, self.first, 1000 * SAMPLES_PER_MS)
        self.ledger.note_synthesized(1, self.second, 1000 * SAMPLES_PER_MS)
        self.record_ms(0, self.first, 0, 1000)
        self.record_ms(1, self.second, 0, 1000)


class TestSpokenText:
    def test_whole_segments_plus_a_share_rounded_down_to_a_word(self):
        h = LedgerHarness()
        h.two_segments()

        h.ledger.report(1500)

        # Half of "How are you today?" is 9 characters, inside "you", so the prefix ends at
        # "are".
        assert h.ledger.spoken_text() == "Hello there, friend. How are"
        assert h.ledger.spoken_chars() == len("Hello there, friend. How are")
        assert h.ledger.source == "client"

    def test_a_share_ending_on_a_word_end_keeps_that_word(self):
        h = LedgerHarness()
        h.two_segments()

        # 389 ms of the second segment's 1000 ms is just past 7 of its 18 characters: the
        # space after "are".
        h.ledger.report(1389)

        assert h.ledger.spoken_text() == "Hello there, friend. How are"

    def test_a_whole_segment_played_gives_its_full_text(self):
        h = LedgerHarness()
        h.two_segments()

        h.ledger.report(1000)

        assert h.ledger.spoken_text() == FIRST

    def test_a_share_inside_the_first_word_of_a_segment_ends_at_the_previous_segment(self):
        h = LedgerHarness()
        h.two_segments()

        h.ledger.report(1100)

        assert h.ledger.spoken_text() == FIRST

    def test_nothing_played_is_the_empty_prefix(self):
        h = LedgerHarness()
        h.two_segments()

        assert h.ledger.spoken_text() == ""

    def test_a_report_past_the_recorded_audio_is_capped_at_it(self):
        h = LedgerHarness()
        h.two_segments()

        h.ledger.report(9000)

        assert h.ledger.spoken_text() == DISPLAY

    def test_the_share_is_of_the_segments_synthesized_audio_not_its_recorded_frames(self):
        h = LedgerHarness()
        h.ledger.note_synthesized(0, h.first, 2000 * SAMPLES_PER_MS)
        h.record_ms(0, h.first, 0, 600)

        h.ledger.report(500)

        # 500 ms of a 2000 ms segment is a quarter: 5 of 20 characters, the end of "Hello".
        # Measured against the 600 ms recorded it would be 16 characters.
        assert h.ledger.spoken_text() == "Hello"

    @pytest.mark.asyncio
    async def test_without_reports_the_estimate_counts_from_the_first_frame_plus_prebuffer(self):
        h = LedgerHarness()
        h.two_segments()

        await h.clock.advance(80 + 1000)

        assert h.ledger.source == "estimate"
        assert h.ledger.position_ms() == 1000
        assert h.ledger.spoken_text() == FIRST

    def test_recording_a_segment_it_was_not_told_of_is_refused(self):
        h = LedgerHarness()

        with pytest.raises(ValueError):
            h.ledger.record(0, 0, FRAME_SAMPLES)


class TestRecentTtsText:
    def test_returns_the_words_played_in_the_window(self):
        h = LedgerHarness()
        h.two_segments()
        h.clock.now = 1000
        h.ledger.report(1000)
        h.clock.now = 2000
        h.ledger.report(2000)

        assert h.ledger.recent_tts_text(1000) == SECOND

    def test_includes_the_word_playing_at_each_edge_of_the_window(self):
        h = LedgerHarness()
        h.two_segments()
        h.clock.now = 1000
        h.ledger.report(500)
        h.clock.now = 2000
        h.ledger.report(1500)

        # 500 ms is inside "there," and 1500 ms inside "you".
        assert h.ledger.recent_tts_text(1000) == "there, friend. How are you"

    def test_is_empty_once_playback_has_ended_for_longer_than_the_window(self):
        h = LedgerHarness()
        h.two_segments()
        h.clock.now = 2000
        h.ledger.report(2000)
        h.clock.now = 6000

        assert h.ledger.recent_tts_text(1500) == ""

    def test_is_empty_before_anything_played(self):
        h = LedgerHarness()
        h.two_segments()

        assert h.ledger.recent_tts_text(1500) == ""


class TestPlayedThrough:
    def test_a_frameless_tag_is_played_through_once_it_ends(self):
        h = LedgerHarness()

        before = h.ledger.played_through()
        h.ledger.mark_ended()

        assert before is False
        assert h.ledger.played_through() is True

    @pytest.mark.asyncio
    async def test_the_estimate_passes_the_end_after_the_last_frame_plays(self):
        h = LedgerHarness()
        h.two_segments()
        h.ledger.mark_ended()

        await h.clock.advance(80 + 1999)
        early = h.ledger.played_through()
        await h.clock.advance(1)

        assert early is False
        assert h.ledger.played_through() is True


class TestMonotonicClock:
    @pytest.mark.asyncio
    async def test_now_never_goes_back_and_sleep_returns(self):
        clock = MonotonicPlaybackClock()

        first = clock.now_ms()
        await clock.sleep_ms(0)

        assert clock.now_ms() >= first
