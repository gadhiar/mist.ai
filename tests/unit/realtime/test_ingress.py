"""Tests for `AudioIngress`, which turns uplink frames into speech segments, name hits and utterances.

Ingress is where the realtime path decides when the user spoke, whether they said MIST's name,
what they said and when they finished, so every rule here is one a user notices when it breaks:
a clipped first word, a reply that starts before the sentence ended, a name heard in silence. The
tests run on a fake clock with the deterministic engines from `tests/fakes/speech.py` and lanes
that run each job on the event loop, so every outcome is fixed by the frames pushed. Audio comes
from `scripted_audio`; speech starts on multiples of 160 ms (8 frames, 5 VAD windows), so a
speech onset falls on a window boundary and its position is exact.
"""

import asyncio
import itertools
import logging
import typing
from collections.abc import AsyncIterator, Callable, Sequence

import numpy as np
import pytest
import pytest_asyncio

from backend.realtime import ingress as ingress_module
from backend.realtime.budget import EndpointingConfig, TurnConfig, WakeConfig
from backend.realtime.errors import SpeechEngineError
from backend.realtime.ingress import (
    KWS_BACKLOG_FRAMES,
    AudioIngress,
    ControllerDropReason,
    DropReason,
    EndpointCandidate,
    IngressEngines,
    IngressLanes,
    IngressOverload,
    IngressSeed,
    IngressStats,
    NameHeard,
    OpenedUtterance,
    SegmentStep,
    SpeechResumed,
    SpeechSegmentTracker,
    SpeechStarted,
    SpotterFeed,
    TranscriptPartial,
    UtteranceDropped,
    UtteranceEnded,
    onset_lead_frames,
    ring_retention_ms,
    spotter_feed,
)
from backend.realtime.lanes import InferenceLane, LaneJobCancelled
from backend.realtime.protocol import Frame, MsgType
from backend.realtime.speech import CancelToken, NameHit, TranscriptUpdate
from tests.fakes.scripted_audio import scripted_audio, silence, speech_burst, to_pcm16
from tests.fakes.speech import (
    EnergyVad,
    ScriptedNameSpotter,
    ScriptedNameSpotterStream,
    ScriptedTranscriber,
    ScriptedTranscriptStream,
    ScriptedTurnEnd,
    ScriptedUtterance,
)

FRAME_MS = 20
FRAME_SAMPLES = 320
PRE_ROLL = 15  # 300 ms of 20 ms frames


class FakeClock:
    """The session clock in milliseconds; tests advance it 20 ms per frame pushed."""

    def __init__(self, now: int = 1000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


class StepLane:
    """A lane that runs each job on the event loop, in the order submitted.

    Unheld, a job runs inside the awaiting task with no suspension, so a result lands at the next
    yield and the ordering is fixed. Held, each job waits at the lane until `release`, the way a
    slow engine keeps a real lane busy. A set token fails the job with `LaneJobCancelled`, as
    `InferenceLane` does.
    """

    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.started: list[str] = []
        self.holding = False
        self._lock: asyncio.Lock | None = None
        self._parked: list[asyncio.Event] = []

    async def run(self, fn: Callable[..., object], *args: object, token: CancelToken) -> object:
        if token.is_set():
            raise LaneJobCancelled("cancelled before submission")
        name = getattr(fn, "__name__", type(fn).__name__)
        self.submitted.append(name)
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self.holding:
                gate = asyncio.Event()
                self._parked.append(gate)
                await gate.wait()
            if token.is_set():
                raise LaneJobCancelled("cancelled before it started")
            self.started.append(name)
            result = fn(*args)
        if token.is_set():
            raise LaneJobCancelled("cancelled while it ran")
        return result

    def release(self) -> None:
        self.holding = False
        for gate in self._parked:
            gate.set()
        self._parked.clear()


async def settle(rounds: int = 20) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


def counter_ids() -> Callable[[], str]:
    numbers = itertools.count(1)
    return lambda: f"u{next(numbers)}"


def marked_pcm(pos: int) -> bytes:
    """A quiet frame whose samples all hold `pos + 1`, so a recording stream can tell positions."""
    return np.full(FRAME_SAMPLES, pos + 1, dtype="<i2").tobytes()


def positions_in(pushes: Sequence[np.ndarray]) -> list[int]:
    audio = np.concatenate(pushes) if pushes else np.zeros(0, dtype=np.float32)
    return [
        int(round(float(audio[start]) * 32768)) - 1
        for start in range(0, len(audio) - FRAME_SAMPLES + 1, FRAME_SAMPLES)
    ]


class RecordingStream(ScriptedTranscriptStream):
    """A scripted stream that keeps the audio pushed into it."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.audio: list[np.ndarray] = []

    def push(self, pcm: np.ndarray) -> None:
        super().push(pcm)
        self.audio.append(np.array(pcm, copy=True))


class RecordingTranscriber(ScriptedTranscriber):
    def open_stream(self) -> RecordingStream:  # type: ignore[override]
        index = len(self.streams)
        utterance = self.script[index] if index < len(self.script) else ScriptedUtterance("")
        stream = RecordingStream(utterance, ms_per_word=self.ms_per_word, on_decode=self.on_decode)
        self.streams.append(stream)
        return stream


class FailingFinalTranscriber(ScriptedTranscriber):
    """The first stream's final decode fails as a broken engine would; later streams work."""

    def open_stream(self) -> ScriptedTranscriptStream:
        stream = super().open_stream()
        if len(self.streams) == 1:

            def fail(token: CancelToken) -> TranscriptUpdate:
                raise SpeechEngineError("decoder crashed")

            stream.decode_final = fail  # type: ignore[method-assign]
        return stream


_OPEN_RIGS: list["Rig"] = []


@pytest_asyncio.fixture(autouse=True)
async def close_rigs() -> AsyncIterator[None]:
    """Close every ingress a test built, so no worker task outlives its test's event loop."""
    yield
    for rig in _OPEN_RIGS:
        rig.ingress.close()
    _OPEN_RIGS.clear()
    await settle()


class Rig:
    """One ingress wired to fakes, with helpers that push scripted audio frame by frame."""

    def __init__(
        self,
        *,
        transcriber: ScriptedTranscriber | None = None,
        turn_end: ScriptedTurnEnd | None | str = "default",
        spotter: ScriptedNameSpotter | None = None,
        endpointing: EndpointingConfig | None = None,
        wake: WakeConfig | None = None,
        retention_ms: int = 7000,
        epoch: int = 3,
        new_utterance_id: Callable[[], str] | None = None,
        seed: IngressSeed | None = None,
        engaged: bool = True,
        lanes: IngressLanes | None = None,
    ) -> None:
        self.clock = FakeClock()
        self.vad = EnergyVad()
        self.transcriber = transcriber or ScriptedTranscriber(["hello there"], ms_per_word=20)
        self.turn_end = ScriptedTurnEnd([0.9]) if turn_end == "default" else turn_end
        self.spotter = spotter
        self.lanes = lanes or IngressLanes(
            vad=StepLane(), stt=StepLane(), eot=StepLane(), kws=StepLane()
        )
        self.events: list[object] = []
        self.emitted_at: list[int] = []
        self.ingress = AudioIngress(
            engines=IngressEngines(
                vad=self.vad,
                transcriber=self.transcriber,
                turn_end=self.turn_end,  # type: ignore[arg-type]
                spotter=self.spotter,
            ),
            lanes=self.lanes,
            endpointing=endpointing or EndpointingConfig(),
            wake=wake or WakeConfig(),
            emit=self._emit,
            retention_ms=retention_ms,
            epoch=epoch,
            new_utterance_id=new_utterance_id or counter_ids(),
            clock=self.clock,
            seed=seed,
        )
        _OPEN_RIGS.append(self)
        if engaged:
            self.ingress.set_engaged(True)
        self.pos = seed.frames[-1][0] + 1 if seed is not None and seed.frames else 0
        self.seq = 0
        self.arrivals: dict[int, int] = {}

    def _emit(self, event: object) -> None:
        self.events.append(event)
        self.emitted_at.append(self.clock())

    def of(self, kind: type) -> list:
        return [event for event in self.events if isinstance(event, kind)]

    async def push_pcm(
        self, pcm: bytes, *, reference: bytes | None = None, settle_after: bool = True
    ) -> int:
        pos = self.pos
        self.clock.now += FRAME_MS
        if reference is not None:
            self.ingress.push_reference(self.seq, reference)
        self.ingress.push_frame(Frame(MsgType.MIC_AUDIO, 0, self.seq, 16000, pcm), pos)
        self.arrivals[pos] = self.clock.now
        self.pos += 1
        self.seq += 1
        if settle_after:
            await settle()
        return pos

    async def push_audio(
        self,
        audio: np.ndarray,
        *,
        reference: Callable[[int], bytes | None] | None = None,
        settle_after: bool = True,
    ) -> None:
        for start in range(0, len(audio), FRAME_SAMPLES):
            chunk = audio[start : start + FRAME_SAMPLES]
            ref = reference(self.pos) if reference is not None else None
            await self.push_pcm(to_pcm16(chunk), reference=ref, settle_after=settle_after)

    async def speak(self, ms: int, **kwargs: object) -> None:
        await self.push_audio(scripted_audio([speech_burst(ms)], seed=self.pos), **kwargs)  # type: ignore[arg-type]

    async def quiet(self, ms: int, **kwargs: object) -> None:
        await self.push_audio(scripted_audio([silence(ms)], seed=self.pos), **kwargs)  # type: ignore[arg-type]

    async def marked(self, count: int) -> None:
        for _ in range(count):
            await self.push_pcm(marked_pcm(self.pos))


def last_frame(ended: UtteranceEnded) -> int:
    return ended.first_pos + ended.duration_ms // FRAME_MS - 1


def hit_at(pos: int, score: float = 0.9) -> NameHit:
    return NameHit(end_pos=pos, keyword="mist", score=score)


# -- Events ---------------------------------------------------------------------------------------


class TestEventStamps:
    @pytest.mark.asyncio
    async def test_every_event_carries_the_epoch_and_consecutive_event_seq(self):
        rig = Rig(
            transcriber=ScriptedTranscriber(["hello there", ""], ms_per_word=20),
            turn_end=ScriptedTurnEnd([0.1, 0.9]),
            spotter=ScriptedNameSpotter([hit_at(60)]),
            retention_ms=1000,
        )

        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(320)
        await rig.speak(320)
        await rig.quiet(1600)
        await rig.speak(320)
        await rig.quiet(2400)
        rig.ingress.open_utterance("wake", 0)

        kinds = {type(event) for event in rig.events}
        assert kinds == {
            NameHeard,
            SpeechStarted,
            TranscriptPartial,
            EndpointCandidate,
            SpeechResumed,
            UtteranceEnded,
            UtteranceDropped,
            IngressOverload,
        }
        assert [event.event_seq for event in rig.events] == list(range(1, len(rig.events) + 1))
        assert {event.epoch for event in rig.events} == {3}

    @pytest.mark.asyncio
    async def test_every_event_about_an_utterance_names_it_and_its_first_frame(self):
        rig = Rig(turn_end=ScriptedTurnEnd([0.1, 0.9]))

        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(320)
        await rig.speak(320)
        await rig.quiet(800)

        started = rig.of(SpeechStarted)[0]
        about = [
            event
            for event in rig.events
            if isinstance(
                event,
                SpeechStarted
                | TranscriptPartial
                | EndpointCandidate
                | SpeechResumed
                | UtteranceEnded,
            )
        ]
        assert {(event.utterance_id, event.first_pos) for event in about} == {
            (started.utterance_id, started.first_pos)
        }

    @pytest.mark.asyncio
    async def test_two_ingresses_sharing_one_factory_never_repeat_an_id(self):
        ids = counter_ids()
        first = Rig(new_utterance_id=ids)
        second = Rig(new_utterance_id=ids)

        for rig in (first, second, first):
            await rig.marked(4)
            opened = rig.ingress.open_utterance("wake", rig.pos - 2)
            rig.ingress.drop(opened.utterance_id, "text_input")
        seen = [first.ingress.open_utterance("wake", first.pos).utterance_id]
        seen.append(second.ingress.open_utterance("wake", second.pos).utterance_id)

        assert seen == ["u4", "u5"]


# -- Listening: dormant and engaged ---------------------------------------------------------------


class TestDormantListening:
    @pytest.mark.asyncio
    async def test_speech_opens_no_utterance_and_no_stt_stream(self):
        rig = Rig(engaged=False)

        await rig.quiet(640)
        await rig.speak(960)
        await rig.quiet(800)

        assert rig.of(SpeechStarted) == []
        assert rig.transcriber.streams == []
        assert rig.ingress.idle_since() == 0

    @pytest.mark.asyncio
    async def test_the_spotter_hears_the_speech_and_its_pre_roll_and_no_other_silence(self):
        spotter = ScriptedNameSpotter()
        rig = Rig(engaged=False, spotter=spotter)

        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(1600)

        pushed = spotter.streams[0].pushed
        assert pushed == list(range(32 - PRE_ROLL, pushed[-1] + 1))
        # The segment closes once 200 ms of silence follow the last speech window (frame 63).
        assert pushed[-1] < 64 + 12

    @pytest.mark.asyncio
    async def test_set_engaged_makes_the_next_speech_open_an_utterance(self):
        rig = Rig(engaged=False)
        await rig.quiet(640)
        await rig.speak(320)
        await rig.quiet(800)

        rig.ingress.set_engaged(True)
        await rig.speak(320)

        assert len(rig.of(SpeechStarted)) == 1
        assert len(rig.transcriber.streams) == 1


class TestSpeechStart:
    @pytest.mark.asyncio
    async def test_three_speech_windows_start_an_utterance_with_300_ms_of_pre_roll(self):
        rig = Rig(transcriber=RecordingTranscriber(["hi"], ms_per_word=20))
        await rig.quiet(640)

        await rig.speak(80)
        assert rig.of(SpeechStarted) == []
        await rig.speak(80)

        started = rig.of(SpeechStarted)
        assert len(started) == 1
        assert started[0].first_pos == 32 - PRE_ROLL
        stream = rig.transcriber.streams[0]
        assert sum(len(push) for push in stream.audio) == (rig.pos - started[0].first_pos) * 320

    @pytest.mark.asyncio
    async def test_vad_speech_while_a_lever_holds_an_utterance_open_opens_nothing_new(self):
        rig = Rig()
        await rig.quiet(320)
        opened = rig.ingress.open_utterance("wake", 10)

        await rig.speak(640)

        assert rig.of(SpeechStarted) == []
        assert len(rig.transcriber.streams) == 1
        assert rig.ingress.idle_since() is None
        assert opened.first_pos == 10


# -- Partials -------------------------------------------------------------------------------------


class TestPartials:
    @pytest.mark.asyncio
    async def test_first_partial_comes_250_ms_after_speech_starts_then_every_250_ms(self):
        rig = Rig(transcriber=ScriptedTranscriber(["one two three four five six"], ms_per_word=20))
        await rig.quiet(640)

        await rig.speak(1600)

        start_index = rig.events.index(rig.of(SpeechStarted)[0])
        start = rig.emitted_at[start_index]
        partial_times = [
            at
            for event, at in zip(rig.events, rig.emitted_at, strict=True)
            if isinstance(event, TranscriptPartial)
        ]
        expected = [-(-(start + 250 * k) // FRAME_MS) * FRAME_MS for k in range(1, 7)]
        assert partial_times[:6] == expected

    @pytest.mark.asyncio
    async def test_set_speaking_switches_the_interval_between_400_and_250_ms(self):
        rig = Rig(
            transcriber=ScriptedTranscriber(["one two three"], ms_per_word=20),
            endpointing=EndpointingConfig(
                stt_partial_interval_ms=400, stt_partial_interval_speaking_ms=250
            ),
        )
        await rig.quiet(640)
        await rig.speak(1280)
        quiet_times = [
            at for e, at in zip(rig.events, rig.emitted_at) if isinstance(e, TranscriptPartial)
        ]

        rig.ingress.set_speaking(True)
        await rig.speak(1600)
        all_times = [
            at for e, at in zip(rig.events, rig.emitted_at) if isinstance(e, TranscriptPartial)
        ]
        speaking_times = all_times[len(quiet_times) :]

        assert {b - a for a, b in itertools.pairwise(quiet_times)} == {400}
        assert {b - a for a, b in itertools.pairwise(speaking_times[1:])} <= {240, 260}

    @pytest.mark.asyncio
    async def test_partials_carry_stable_chars_from_the_transcriber(self):
        rig = Rig(transcriber=ScriptedTranscriber(["one two three four"], ms_per_word=200))
        await rig.quiet(640)

        await rig.speak(1600)

        partials = rig.of(TranscriptPartial)
        stream = rig.transcriber.streams[0]
        assert [(p.text, p.stable_chars) for p in partials] == [
            (u.text, u.stable_chars) for u in stream.partials
        ]
        assert any(p.stable_chars > 0 for p in partials)

    @pytest.mark.asyncio
    async def test_a_partial_carries_its_last_position_and_the_oldest_uncovered_arrival(self):
        rig = Rig()
        await rig.quiet(640)

        await rig.speak(960)

        first, second = rig.of(TranscriptPartial)[:2]
        assert first.uncovered_since_ms == rig.arrivals[first.first_pos]
        assert second.uncovered_since_ms == rig.arrivals[first.audio_end_pos + 1]
        assert second.heard_at_ms == rig.arrivals[second.audio_end_pos]
        assert first.gate is False and first.through_pos is None

    @pytest.mark.asyncio
    async def test_heard_at_is_the_arrival_of_the_last_frame_the_partial_covers(self):
        rig = Rig()
        await rig.quiet(360)
        rig.ingress.open_utterance("wake", 10)

        await rig.speak(320)

        covering_30 = [p for p in rig.of(TranscriptPartial) if p.audio_end_pos == 30]
        assert len(covering_30) == 1
        assert covering_30[0].heard_at_ms == rig.arrivals[30] == 1000 + 31 * FRAME_MS

    @pytest.mark.asyncio
    async def test_no_partial_is_requested_before_new_audio_reaches_the_utterance(self):
        rig = Rig()
        await rig.marked(4)
        rig.ingress.open_utterance("wake", 40)

        await rig.marked(30)

        assert rig.of(TranscriptPartial) == []
        assert rig.transcriber.streams[0].partials == []

    @pytest.mark.asyncio
    async def test_a_tick_while_a_partial_decodes_is_skipped_and_counted(self):
        rig = Rig(endpointing=EndpointingConfig(stt_partial_interval_ms=100))
        await rig.quiet(640)
        await rig.speak(320)
        rig.lanes.stt.holding = True
        skipped_before = rig.ingress.stats().skipped_partials

        await rig.speak(480)

        assert rig.lanes.stt.submitted.count("decode_partial") == len(rig.of(TranscriptPartial)) + 1
        assert rig.ingress.stats().skipped_partials - skipped_before >= 3

    @pytest.mark.asyncio
    async def test_the_final_decode_waits_behind_at_most_the_partial_in_flight(self):
        rig = Rig(endpointing=EndpointingConfig(stt_partial_interval_ms=100))
        await rig.quiet(640)
        await rig.speak(320)
        rig.lanes.stt.holding = True
        submitted_before = len(rig.lanes.stt.submitted)

        await rig.speak(320)
        await rig.quiet(1600)

        assert rig.lanes.stt.submitted[submitted_before:] == ["decode_partial", "decode_final"]
        rig.lanes.stt.release()
        await settle()
        assert len(rig.of(UtteranceEnded)) == 1


# -- Endpointing ----------------------------------------------------------------------------------


class TestEndpointing:
    @pytest.mark.asyncio
    async def test_after_200_ms_of_silence_the_final_decode_and_eot_start_together(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)
        rig.lanes.stt.holding = True
        rig.lanes.eot.holding = True

        await rig.quiet(320)

        assert "decode_final" in rig.lanes.stt.submitted
        assert rig.lanes.eot.submitted == ["probability_complete"]
        assert rig.of(EndpointCandidate) == [] and rig.of(UtteranceEnded) == []
        rig.lanes.stt.release()
        rig.lanes.eot.release()
        await settle()
        assert len(rig.of(UtteranceEnded)) == 1

    @pytest.mark.asyncio
    async def test_speculative_start_yields_a_candidate_with_the_final_text(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)

        await rig.quiet(480)

        (candidate,) = rig.of(EndpointCandidate)
        (ended,) = rig.of(UtteranceEnded)
        assert candidate.text == "hello there"
        assert candidate.utterance_id == ended.utterance_id
        assert candidate.heard_at_ms == ended.heard_at_ms == rig.arrivals[last_frame(ended)]
        assert rig.events.index(candidate) < rig.events.index(ended)

    @pytest.mark.asyncio
    async def test_without_speculative_start_no_candidate_is_emitted(self):
        rig = Rig(endpointing=EndpointingConfig(speculative_start=False))
        await rig.quiet(640)
        await rig.speak(640)

        await rig.quiet(480)

        assert rig.of(EndpointCandidate) == []
        assert len(rig.of(UtteranceEnded)) == 1

    @pytest.mark.asyncio
    async def test_a_probability_above_the_threshold_ends_the_utterance_as_semantic(self):
        rig = Rig(
            transcriber=ScriptedTranscriber(
                [ScriptedUtterance("hello there", 0.1, -0.2)], ms_per_word=20
            ),
            turn_end=ScriptedTurnEnd([0.8]),
        )
        await rig.quiet(640)
        await rig.speak(640)

        await rig.quiet(480)

        (ended,) = rig.of(UtteranceEnded)
        assert (ended.reason, ended.eot_probability, ended.text) == ("semantic", 0.8, "hello there")
        assert 200 <= ended.silence_ms < 300
        assert ended.duration_ms == (last_frame(ended) - ended.first_pos + 1) * FRAME_MS
        assert (ended.no_speech_prob, ended.avg_logprob) == (0.1, -0.2)

    @pytest.mark.asyncio
    async def test_the_end_of_turn_is_rechecked_every_300_ms_until_it_passes(self):
        rig = Rig(turn_end=ScriptedTurnEnd([0.1, 0.2, 0.3, 0.9]))
        await rig.quiet(640)
        await rig.speak(640)

        await rig.quiet(1600)

        (ended,) = rig.of(UtteranceEnded)
        assert len(rig.turn_end.calls) == 4
        assert ended.reason == "semantic"
        assert 1100 <= ended.silence_ms < 1300

    @pytest.mark.asyncio
    async def test_silence_reaching_1500_ms_ends_it_as_silence_cap(self):
        rig = Rig(turn_end=ScriptedTurnEnd([0.1]))
        await rig.quiet(640)
        await rig.speak(640)

        await rig.quiet(2000)

        (ended,) = rig.of(UtteranceEnded)
        assert ended.reason == "silence_cap"
        assert ended.eot_probability == 0.1
        assert 1500 <= ended.silence_ms < 1540

    @pytest.mark.asyncio
    async def test_with_no_end_of_turn_engine_the_cap_ends_it_as_silence_cap_no_model(self):
        rig = Rig(turn_end=None)
        await rig.quiet(640)
        await rig.speak(640)

        await rig.quiet(2000)

        (ended,) = rig.of(UtteranceEnded)
        assert (ended.reason, ended.eot_probability) == ("silence_cap_no_model", None)
        assert ended.silence_ms >= 1500

    @pytest.mark.asyncio
    async def test_speech_resuming_before_the_end_continues_the_utterance(self):
        rig = Rig(turn_end=ScriptedTurnEnd([0.1]))
        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(480)
        assert len(rig.of(EndpointCandidate)) == 1

        await rig.speak(640)
        await rig.quiet(2000)

        (resumed,) = rig.of(SpeechResumed)
        (ended,) = rig.of(UtteranceEnded)
        assert resumed.utterance_id == ended.utterance_id == rig.of(SpeechStarted)[0].utterance_id
        assert len(rig.transcriber.streams) == 1

    @pytest.mark.asyncio
    async def test_after_a_resume_the_next_candidate_and_end_cover_the_audio_before_the_pause(
        self,
    ):
        rig = Rig(turn_end=ScriptedTurnEnd([0.1]))
        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(480)

        await rig.speak(640)
        await rig.quiet(2000)

        first, second = rig.of(EndpointCandidate)
        (ended,) = rig.of(UtteranceEnded)
        stream = rig.transcriber.streams[0]
        assert len(stream.finals) == 2
        assert stream.finals[-1].audio_ms == ended.duration_ms
        assert stream.finals[-1].audio_ms > stream.finals[0].audio_ms + 640
        assert second.text == ended.text == "hello there"

    @pytest.mark.asyncio
    async def test_speech_after_a_confirmed_end_starts_a_new_utterance(self):
        rig = Rig(transcriber=ScriptedTranscriber(["one", "two"], ms_per_word=20))
        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(640)

        await rig.speak(640)

        first, second = rig.of(SpeechStarted)
        (ended,) = rig.of(UtteranceEnded)
        assert first.utterance_id != second.utterance_id
        assert second.first_pos > last_frame(ended)


class TestDrops:
    @pytest.mark.parametrize(
        "speech_ms", [pytest.param(100, id="100ms"), pytest.param(2000, id="2s")]
    )
    @pytest.mark.asyncio
    async def test_an_empty_final_drops_the_utterance_as_noise_at_any_length(self, speech_ms):
        rig = Rig(transcriber=ScriptedTranscriber([""], ms_per_word=20))
        await rig.quiet(640)
        await rig.speak(speech_ms)

        await rig.quiet(800)

        assert rig.of(UtteranceEnded) == [] and rig.of(EndpointCandidate) == []
        (dropped,) = rig.of(UtteranceDropped)
        assert dropped.reason == "noise"

    @pytest.mark.asyncio
    async def test_a_final_flagged_both_ways_as_non_speech_is_dropped_as_no_speech(self):
        rig = Rig(
            transcriber=ScriptedTranscriber(
                [ScriptedUtterance("thanks for watching", 0.7, -1.0)], ms_per_word=20
            )
        )
        await rig.quiet(640)
        await rig.speak(640)

        await rig.quiet(800)

        (dropped,) = rig.of(UtteranceDropped)
        assert (dropped.reason, dropped.text) == ("no_speech", "thanks for watching")
        assert (dropped.no_speech_prob, dropped.avg_logprob) == (0.7, -1.0)
        assert rig.of(UtteranceEnded) == [] and rig.of(EndpointCandidate) == []

    @pytest.mark.parametrize(
        ("no_speech_prob", "avg_logprob"),
        [
            pytest.param(0.7, -0.5, id="only-no-speech-prob"),
            pytest.param(0.5, -1.5, id="only-low-logprob"),
        ],
    )
    @pytest.mark.asyncio
    async def test_a_final_meeting_only_one_non_speech_test_is_kept(
        self, no_speech_prob, avg_logprob
    ):
        rig = Rig(
            transcriber=ScriptedTranscriber(
                [ScriptedUtterance("turn it up", no_speech_prob, avg_logprob)], ms_per_word=20
            )
        )
        await rig.quiet(640)
        await rig.speak(640)

        await rig.quiet(800)

        assert rig.of(UtteranceDropped) == []
        assert [e.text for e in rig.of(UtteranceEnded)] == ["turn it up"]

    @pytest.mark.asyncio
    async def test_a_failed_final_decode_drops_as_stt_error_and_ingress_keeps_running(self):
        rig = Rig(transcriber=FailingFinalTranscriber(["broken", "works"], ms_per_word=20))
        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(800)

        await rig.speak(640)
        await rig.quiet(800)

        assert [d.reason for d in rig.of(UtteranceDropped)] == ["stt_error"]
        assert [e.text for e in rig.of(UtteranceEnded)] == ["works"]


# -- Queue overload and uplink gaps ---------------------------------------------------------------


class TestOverloadAndGaps:
    @pytest.mark.asyncio
    async def test_audio_queued_past_2_s_is_dropped_oldest_first_with_one_event_per_utterance(
        self,
    ):
        rig = Rig(
            turn_end=ScriptedTurnEnd([0.1]),
            endpointing=EndpointingConfig(eot_max_silence_ms=60000),
        )
        await rig.marked(4)
        opened = rig.ingress.open_utterance("wake", 0)

        for _ in range(150):
            await rig.push_pcm(marked_pcm(rig.pos), settle_after=False)
        await settle(500)
        for _ in range(150):
            await rig.push_pcm(marked_pcm(rig.pos), settle_after=False)
        await settle(500)

        (overload,) = rig.of(IngressOverload)
        assert (overload.lane, overload.utterance_id) == ("vad", opened.utterance_id)
        assert rig.ingress.stats().queue_dropped_frames == 100

    @pytest.mark.parametrize("first_seq", [0, 500])
    @pytest.mark.asyncio
    async def test_the_first_frame_sets_the_base_and_counts_no_gap(self, first_seq):
        rig = Rig()
        rig.seq = first_seq

        await rig.marked(3)

        assert rig.ingress.stats().gap_frames == 0

    @pytest.mark.asyncio
    async def test_a_forward_jump_counts_the_missing_frames(self):
        rig = Rig()
        await rig.marked(8)

        rig.seq = 10
        await rig.marked(1)

        assert rig.ingress.stats().gap_frames == 2

    @pytest.mark.asyncio
    async def test_a_seq_restart_sets_a_new_base_with_no_gap(self):
        rig = Rig()
        rig.seq = 500
        await rig.marked(3)

        rig.seq = 0
        await rig.marked(3)

        assert rig.ingress.stats().gap_frames == 0


# -- Levers ---------------------------------------------------------------------------------------


class TestOpenUtterance:
    @pytest.mark.asyncio
    async def test_it_starts_at_from_pos_within_the_ring(self):
        rig = Rig(transcriber=RecordingTranscriber([""]), retention_ms=400)
        await rig.marked(30)

        opened = rig.ingress.open_utterance("wake", 15)

        assert opened == OpenedUtterance(opened.utterance_id, 15)
        assert positions_in(rig.transcriber.streams[0].audio) == list(range(15, 30))

    @pytest.mark.asyncio
    async def test_it_starts_one_past_the_previous_utterance(self):
        rig = Rig(transcriber=RecordingTranscriber(["", ""]))
        await rig.marked(21)
        first = rig.ingress.open_utterance("wake", 0)
        rig.ingress.drop(first.utterance_id, "text_input")

        second = rig.ingress.open_utterance("wake", 15)

        assert second.first_pos == 21
        assert positions_in(rig.transcriber.streams[0].audio) == list(range(0, 21))
        assert positions_in(rig.transcriber.streams[1].audio) == []

    @pytest.mark.asyncio
    async def test_vad_pre_roll_starts_no_earlier_than_one_past_the_previous_utterance(self):
        rig = Rig()
        await rig.quiet(420)
        first = rig.ingress.open_utterance("wake", 0)
        rig.ingress.drop(first.utterance_id, "text_input")

        await rig.quiet(60)
        await rig.speak(320)

        assert rig.of(SpeechStarted)[0].first_pos == 21

    @pytest.mark.asyncio
    async def test_a_from_pos_older_than_the_ring_starts_at_the_oldest_frame_and_reports_it(self):
        rig = Rig(transcriber=RecordingTranscriber([""]), retention_ms=400)
        await rig.marked(30)

        opened = rig.ingress.open_utterance("wake", 4)

        assert opened.first_pos == 10
        (overload,) = rig.of(IngressOverload)
        assert (overload.lane, overload.dropped_ms) == ("capture", 6 * FRAME_MS)
        assert overload.utterance_id == opened.utterance_id

    @pytest.mark.asyncio
    async def test_a_2800_ms_backfill_reaches_the_stream_whole_in_position_order(self):
        rig = Rig(transcriber=RecordingTranscriber([""]))
        await rig.marked(150)

        rig.ingress.open_utterance("wake", 10)

        assert positions_in(rig.transcriber.streams[0].audio) == list(range(10, 150))
        assert rig.of(IngressOverload) == []

    @pytest.mark.asyncio
    async def test_wake_with_an_utterance_open_continues_it(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)
        started = rig.of(SpeechStarted)[0]

        opened = rig.ingress.open_utterance("wake", rig.pos - 3)

        assert opened == OpenedUtterance(started.utterance_id, started.first_pos)
        assert len(rig.transcriber.streams) == 1

    @pytest.mark.asyncio
    async def test_an_utterance_a_lever_opens_emits_no_speech_started_and_endpoints_by_vad(self):
        rig = Rig()
        await rig.quiet(320)
        opened = rig.ingress.open_utterance("wake", 10)

        await rig.speak(640)
        await rig.quiet(480)

        assert rig.of(SpeechStarted) == []
        (ended,) = rig.of(UtteranceEnded)
        assert (ended.utterance_id, ended.reason) == (opened.utterance_id, "semantic")


class TestDrop:
    @pytest.mark.parametrize("reason", ["text_input", "wake_lost", "not_addressed"])
    @pytest.mark.asyncio
    async def test_drop_returns_the_last_partial_and_duration_and_emits_nothing(self, reason):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)
        partial = rig.of(TranscriptPartial)[-1]
        started = rig.of(SpeechStarted)[0]
        count = len(rig.events)

        dropped = rig.ingress.drop(started.utterance_id, reason)

        assert dropped is not None
        assert (dropped.reason, dropped.text) == (reason, partial.text)
        assert dropped.duration_ms == (rig.pos - started.first_pos) * FRAME_MS
        assert len(rig.events) == count
        assert rig.transcriber.streams[0].closed
        assert rig.transcriber.streams[0].finals == []

    @pytest.mark.asyncio
    async def test_drop_cancels_the_pending_lane_jobs(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)
        rig.lanes.stt.holding = True
        rig.lanes.eot.holding = True
        await rig.quiet(320)
        started = rig.of(SpeechStarted)[0]

        rig.ingress.drop(started.utterance_id, "text_input")
        rig.lanes.stt.release()
        rig.lanes.eot.release()
        await settle()

        assert "decode_final" not in rig.lanes.stt.started
        assert rig.lanes.eot.started == []
        assert rig.of(EndpointCandidate) == [] and rig.of(UtteranceEnded) == []

    @pytest.mark.asyncio
    async def test_drop_naming_an_ended_utterance_returns_none_and_the_open_one_runs_on(self):
        rig = Rig(transcriber=ScriptedTranscriber(["one", "two"], ms_per_word=20))
        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(640)
        (ended,) = rig.of(UtteranceEnded)
        await rig.speak(640)

        result = rig.ingress.drop(ended.utterance_id, "wake_lost")
        await rig.quiet(640)

        assert result is None
        assert [e.text for e in rig.of(UtteranceEnded)] == ["one", "two"]

    @pytest.mark.asyncio
    async def test_a_not_addressed_drop_logs_none_of_the_utterances_text(self, caplog):
        rig = Rig(transcriber=ScriptedTranscriber(["the morning mist rolled"], ms_per_word=20))
        caplog.set_level(logging.DEBUG)
        await rig.quiet(640)
        await rig.speak(960)
        started = rig.of(SpeechStarted)[0]

        dropped = rig.ingress.drop(started.utterance_id, "not_addressed")

        assert dropped is not None and dropped.text
        assert all("morning" not in record.getMessage() for record in caplog.records)
        assert any("not_addressed" in record.getMessage() for record in caplog.records)


class TestClose:
    @pytest.mark.asyncio
    async def test_close_returns_the_open_utterance_as_voice_stop_and_emits_nothing(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)
        count = len(rig.events)

        dropped = rig.ingress.close()

        assert dropped is not None and dropped.reason == "voice_stop"
        assert len(rig.events) == count
        assert rig.transcriber.streams[0].closed

    @pytest.mark.asyncio
    async def test_close_with_nothing_open_returns_none(self):
        rig = Rig()
        await rig.quiet(320)

        assert rig.ingress.close() is None

    @pytest.mark.asyncio
    async def test_after_close_frames_are_counted_and_ignored_and_no_event_comes(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)
        rig.lanes.stt.holding = True
        await rig.quiet(320)
        rig.ingress.close()
        count = len(rig.events)

        rig.lanes.stt.release()
        await rig.speak(640)
        await rig.quiet(800)

        assert len(rig.events) == count
        assert rig.ingress.stats().ignored_frames == 32 + 40


class TestIdleSince:
    @pytest.mark.asyncio
    async def test_it_is_zero_before_any_event_and_none_while_an_utterance_is_open(self):
        rig = Rig()
        assert rig.ingress.idle_since() == 0
        await rig.quiet(640)
        await rig.speak(640)

        assert rig.ingress.idle_since() is None

    @pytest.mark.asyncio
    async def test_it_is_the_ends_event_seq_after_an_utterance_ended(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)

        await rig.quiet(480)

        (ended,) = rig.of(UtteranceEnded)
        assert rig.ingress.idle_since() == ended.event_seq == rig.events[-1].event_seq

    @pytest.mark.asyncio
    async def test_it_is_the_drops_event_seq_after_an_utterance_dropped(self):
        rig = Rig(transcriber=ScriptedTranscriber([""]))
        await rig.quiet(640)
        await rig.speak(640)

        await rig.quiet(480)

        (dropped,) = rig.of(UtteranceDropped)
        assert rig.ingress.idle_since() == dropped.event_seq

    @pytest.mark.asyncio
    async def test_after_drop_returns_a_drop_it_is_the_last_emitted_event_seq(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)
        last = rig.events[-1].event_seq

        rig.ingress.drop(rig.of(SpeechStarted)[0].utterance_id, "text_input")

        assert rig.ingress.idle_since() == last

    @pytest.mark.asyncio
    async def test_it_is_none_from_a_levers_open_to_the_end(self):
        rig = Rig()
        await rig.quiet(320)
        rig.ingress.open_utterance("wake", 10)
        assert rig.ingress.idle_since() is None

        await rig.speak(640)
        assert rig.ingress.idle_since() is None
        await rig.quiet(480)

        assert rig.ingress.idle_since() == rig.of(UtteranceEnded)[0].event_seq


# -- Ring, retention, handoff and seed ------------------------------------------------------------


class TestRetention:
    def test_the_designs_retention_is_7000_ms_at_the_defaults(self):
        assert ring_retention_ms(TurnConfig(), WakeConfig()) == 7000

    def test_a_retention_above_10000_ms_raises(self):
        with pytest.raises(ValueError, match="retention_ms"):
            Rig(retention_ms=10020)

    @pytest.mark.asyncio
    async def test_the_ring_keeps_the_last_retention_ms_of_frames(self):
        rig = Rig(retention_ms=3000, engaged=False)

        await rig.marked(200)

        seed = rig.ingress.handoff()
        assert [pos for pos, _ in seed.frames] == list(range(50, 200))


class TestHandoff:
    @pytest.mark.asyncio
    async def test_handoff_reports_the_floor_and_the_first_unscored_frame(self):
        rig = Rig(retention_ms=1000)
        await rig.marked(21)
        opened = rig.ingress.open_utterance("wake", 5)
        rig.ingress.drop(opened.utterance_id, "text_input")
        await rig.marked(3)

        seed = rig.ingress.handoff()

        assert seed.floor == 21
        # 24 frames are 7680 samples, exactly 15 windows, so every frame pushed was scored.
        assert seed.vad_from == 24
        assert seed.frames[-1][0] == 23

    @pytest.mark.asyncio
    async def test_after_handoff_the_ingress_still_routes_scores_and_emits(self):
        rig = Rig()
        await rig.quiet(640)
        rig.ingress.handoff()

        await rig.speak(640)
        await rig.quiet(480)

        assert len(rig.of(SpeechStarted)) == 1
        assert len(rig.of(UtteranceEnded)) == 1


def seed_from(
    audio: np.ndarray, *, first_pos: int, floor: int, vad_from: int | None
) -> IngressSeed:
    frames = tuple(
        (first_pos + index, audio[start : start + FRAME_SAMPLES].astype(np.float32))
        for index, start in enumerate(range(0, len(audio), FRAME_SAMPLES))
    )
    return IngressSeed(frames=frames, floor=floor, vad_from=vad_from)


class TestSeed:
    @pytest.mark.asyncio
    async def test_a_seeded_ring_lets_an_utterance_start_inside_the_seed(self):
        seed = seed_from(
            np.zeros(61 * FRAME_SAMPLES, dtype=np.float32), first_pos=100, floor=0, vad_from=None
        )
        rig = Rig(transcriber=RecordingTranscriber([""]), seed=seed)

        rig.ingress.open_utterance("wake", 150)
        await rig.marked(5)

        assert rig.pos == 166
        assert positions_in(rig.transcriber.streams[0].audio)[11:] == list(range(161, 166))
        assert len(rig.transcriber.streams[0].audio) == 16

    @pytest.mark.asyncio
    async def test_no_utterance_starts_before_the_seeds_floor(self):
        seed = seed_from(
            np.zeros(40 * FRAME_SAMPLES, dtype=np.float32), first_pos=0, floor=30, vad_from=None
        )
        rig = Rig(seed=seed)

        opened = rig.ingress.open_utterance("wake", 20)

        assert opened.first_pos == 30

    @pytest.mark.asyncio
    async def test_vad_pre_roll_stops_at_the_seeds_floor(self):
        seed = seed_from(
            np.zeros(40 * FRAME_SAMPLES, dtype=np.float32), first_pos=0, floor=35, vad_from=None
        )
        rig = Rig(seed=seed)

        await rig.speak(320)

        assert rig.of(SpeechStarted)[0].first_pos == 35

    @pytest.mark.asyncio
    async def test_open_mode_scores_the_seed_from_vad_from_and_keeps_its_pre_roll(self):
        audio = scripted_audio([silence(1600), speech_burst(400)], seed=7)
        seed = seed_from(audio, first_pos=0, floor=0, vad_from=83)
        rig = Rig(seed=seed)

        await rig.speak(320)

        started = rig.of(SpeechStarted)[0]
        assert started.first_pos == 83 - PRE_ROLL

    @pytest.mark.asyncio
    async def test_with_vad_from_none_the_vad_starts_at_the_first_frame_pushed(self):
        audio = scripted_audio([speech_burst(800)], seed=7)
        seed = seed_from(audio, first_pos=0, floor=0, vad_from=None)
        rig = Rig(seed=seed)

        await rig.marked(8)

        assert rig.of(SpeechStarted) == []
        assert len(rig.vad.streams[0].scores) == 5


# -- Name hits ------------------------------------------------------------------------------------


class TestNameHeard:
    @pytest.mark.asyncio
    async def test_a_name_last_in_one_segment_opens_at_the_segment_less_the_pre_roll(self):
        rig = Rig(engaged=False, spotter=ScriptedNameSpotter([hit_at(78)]))
        await rig.quiet(640)

        await rig.speak(960)

        (heard,) = rig.of(NameHeard)
        assert heard.from_pos == 32 - PRE_ROLL
        assert heard.hit == hit_at(78)
        assert (heard.utterance_id, heard.reference_score, heard.spotter) == (None, None, "server")
        assert heard.arrival == rig.arrivals[78]

    @pytest.mark.asyncio
    async def test_segments_800_ms_apart_chain_to_the_first(self):
        rig = Rig(engaged=False, spotter=ScriptedNameSpotter([hit_at(130)]))
        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(800)

        await rig.speak(640)

        assert [h.from_pos for h in rig.of(NameHeard)] == [32 - PRE_ROLL]

    @pytest.mark.asyncio
    async def test_a_1200_ms_gap_breaks_the_chain(self):
        rig = Rig(engaged=False, spotter=ScriptedNameSpotter([hit_at(140)]))
        await rig.quiet(640)
        await rig.speak(400)
        await rig.quiet(1200)

        await rig.speak(640)

        assert [h.from_pos for h in rig.of(NameHeard)] == [112 - PRE_ROLL]

    @pytest.mark.asyncio
    async def test_a_preamble_longer_than_the_lookback_opens_6000_ms_before_the_hit(self):
        rig = Rig(engaged=False, spotter=ScriptedNameSpotter([hit_at(366)]))
        await rig.quiet(160)

        await rig.speak(7200)

        assert [h.from_pos for h in rig.of(NameHeard)] == [366 - 300]

    @pytest.mark.asyncio
    async def test_a_previous_utterance_bounds_it_and_an_engaged_hit_names_the_open_one(self):
        spotter = ScriptedNameSpotter()
        rig = Rig(spotter=spotter)
        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(480)
        (ended,) = rig.of(UtteranceEnded)
        while rig.pos < last_frame(ended) + 16:
            await rig.quiet(20)
        spotter.hits.append(hit_at(rig.pos + 20))

        await rig.speak(640)

        (heard,) = rig.of(NameHeard)
        second = rig.of(SpeechStarted)[1]
        assert heard.from_pos == last_frame(ended) + 1 == second.first_pos
        assert heard.utterance_id == second.utterance_id

    @pytest.mark.asyncio
    async def test_no_name_is_heard_while_muted(self):
        spotter = ScriptedNameSpotter([hit_at(40)])
        rig = Rig(engaged=False, spotter=spotter)
        await rig.quiet(640)
        rig.ingress.set_muted(True)

        await rig.speak(640)

        assert rig.of(NameHeard) == []
        assert spotter.streams == [] or spotter.streams[0].pushed == []


class LoggedStream(ScriptedNameSpotterStream):
    """A scripted stream that also logs its calls in order, so warm, push and reset interleave."""

    def __init__(self, hits: Sequence[NameHit], threshold: float, log: list) -> None:
        super().__init__(hits, threshold)
        self.log = log

    def push(self, pcm: bytes, pos: int) -> list[NameHit]:
        self.log.append(("push", pos))
        return super().push(pcm, pos)

    def warm(self, pcm: bytes, pos: int) -> None:
        self.log.append(("warm", pos))
        super().warm(pcm, pos)

    def reset(self, *, warm_up: bool = True) -> None:
        self.log.append(("reset", warm_up))
        super().reset(warm_up=warm_up)


class LoggedSpotter(ScriptedNameSpotter):
    """A scripted spotter whose streams share one ordered log of their calls."""

    def __init__(self, hits: Sequence[NameHit] = ()) -> None:
        super().__init__(hits)
        self.log: list[tuple[str, object]] = []

    def open_stream(self, threshold: float) -> LoggedStream:
        stream = LoggedStream(self.hits, threshold, self.log)
        self.streams.append(stream)
        return stream


def fed_positions(log: Sequence[tuple[str, object]]) -> list[int]:
    """Every position warmed or pushed, in the order the spotter was fed them."""
    return [typing.cast(int, pos) for kind, pos in log if kind != "reset"]


class TestTheSpotterFeed:
    """Ingress feeds the spotter every frame once and in order, reporting only from speech."""

    @pytest.mark.asyncio
    async def test_frames_between_two_segments_are_warmed_with_no_reset(self):
        spotter = LoggedSpotter()
        rig = Rig(engaged=False, spotter=spotter)

        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(1600)
        await rig.speak(640)
        await rig.quiet(400)

        (stream,) = spotter.streams
        assert stream.resets == 0
        fed = fed_positions(spotter.log)
        assert fed == list(range(len(fed)))
        assert len(fed) > 176
        second = 144
        assert set(range(32 - PRE_ROLL, 64)) <= set(stream.pushed)
        assert set(range(80, second - PRE_ROLL)) <= set(stream.warmed)
        assert set(range(second - PRE_ROLL, second + 32)) <= set(stream.pushed)

    @pytest.mark.asyncio
    async def test_the_pre_roll_is_pushed_never_warmed_though_the_vad_scored_it_first(self):
        spotter = LoggedSpotter()
        rig = Rig(engaged=False, spotter=spotter)
        await rig.quiet(640)
        # Frames 32 to 35 are scored as two speech windows: a run, not yet a segment.
        await rig.speak(80)
        (stream,) = spotter.streams
        assert stream.warmed == list(range(32 - PRE_ROLL))
        assert stream.pushed == []

        # Frame 36 completes the third window, and the segment opens at frame 32.
        await rig.speak(20)

        assert stream.warmed == list(range(32 - PRE_ROLL))
        assert stream.pushed == list(range(32 - PRE_ROLL, 37))

    @pytest.mark.asyncio
    async def test_a_scripted_hit_on_a_warmed_frame_is_not_reported(self):
        spotter = LoggedSpotter([hit_at(5)])
        rig = Rig(engaged=False, spotter=spotter)

        await rig.quiet(640)

        assert 5 in spotter.streams[0].warmed
        assert rig.of(NameHeard) == []

    @pytest.mark.asyncio
    async def test_a_mute_resets_the_stream_and_no_frame_from_before_it_is_fed_after_it(self):
        spotter = LoggedSpotter()
        rig = Rig(engaged=False, spotter=spotter)
        await rig.quiet(640)

        rig.ingress.set_muted(True)
        await rig.quiet(200)
        rig.ingress.set_muted(False)
        unmuted_at = rig.pos
        await rig.quiet(640)

        log = spotter.log
        assert [entry for entry in log if entry[0] == "reset"] == [("reset", True)]
        reset_at = log.index(("reset", True))
        before = fed_positions(log[:reset_at])
        after = fed_positions(log[reset_at + 1 :])
        assert before == list(range(32 - PRE_ROLL))
        assert after and after == list(range(unmuted_at, unmuted_at + len(after)))
        assert len(spotter.streams) == 1

    @pytest.mark.asyncio
    async def test_a_hit_from_a_job_that_ran_across_a_mute_and_a_short_unmute_is_dropped(self):
        spotter = LoggedSpotter([hit_at(48)])
        rig = Rig(engaged=False, spotter=spotter)
        await rig.quiet(640)
        await rig.speak(320)
        rig.lanes.kws.holding = True
        await rig.speak(20)

        rig.ingress.set_muted(True)
        rig.ingress.set_muted(False)
        rig.lanes.kws.release()
        await settle()

        assert 48 in spotter.streams[0].pushed
        assert rig.of(NameHeard) == []

    @pytest.mark.asyncio
    async def test_frames_the_vad_queue_dropped_are_fed(self):
        spotter = LoggedSpotter()
        rig = Rig(engaged=False, spotter=spotter)
        for _ in range(150):
            await rig.quiet(20, settle_after=False)
        await settle(500)

        await rig.quiet(640)

        assert rig.ingress.stats().queue_dropped_frames == 50
        fed = fed_positions(spotter.log)
        assert fed[:60] == list(range(60))

    @pytest.mark.asyncio
    async def test_frames_that_arrive_during_a_slow_job_are_fed_by_the_next_job(self):
        spotter = LoggedSpotter()
        rig = Rig(engaged=False, spotter=spotter)
        await rig.quiet(640)
        await rig.speak(320)
        jobs = len(rig.lanes.kws.submitted)
        rig.lanes.kws.holding = True

        await rig.speak(100)
        rig.lanes.kws.release()
        await settle()

        assert spotter.streams[0].pushed[-5:] == [48, 49, 50, 51, 52]
        assert len(rig.lanes.kws.submitted) - jobs == 2
        assert rig.ingress.stats().skipped_kws_hops == 0
        assert rig.of(IngressOverload) == []

    @pytest.mark.asyncio
    async def test_a_backlog_past_the_bound_drops_the_oldest_resets_cold_and_reports_once(self):
        spotter = LoggedSpotter()
        rig = Rig(engaged=False, spotter=spotter)
        await rig.quiet(640)
        await rig.speak(320)
        rig.lanes.kws.holding = True
        await rig.speak(20)

        await rig.speak((KWS_BACKLOG_FRAMES + 10) * FRAME_MS)
        rig.lanes.kws.release()
        await settle()

        assert rig.ingress.stats().skipped_kws_hops == 10
        (overload,) = rig.of(IngressOverload)
        assert (overload.lane, overload.dropped_ms) == ("kws", FRAME_MS)
        log = spotter.log
        reset_at = log.index(("reset", False))
        assert log[reset_at - 1] == ("push", 48)
        assert log[reset_at + 1 :] == [("push", pos) for pos in range(59, 109)]
        fed = fed_positions(log)
        assert fed == sorted(set(fed))
        assert not set(range(49, 59)) & set(fed)

    @pytest.mark.parametrize(("floor", "first"), [(0, 83 - PRE_ROLL), (75, 75)])
    @pytest.mark.asyncio
    async def test_a_seeded_ingress_feeds_from_the_pre_roll_before_vad_from_above_the_floor(
        self, floor, first
    ):
        audio = scripted_audio([silence(1600), speech_burst(400)], seed=7)
        spotter = LoggedSpotter()
        rig = Rig(
            seed=seed_from(audio, first_pos=0, floor=floor, vad_from=83),
            spotter=spotter,
            engaged=False,
        )

        await rig.speak(320)

        fed = fed_positions(spotter.log)
        assert fed and fed == list(range(first, first + len(fed)))
        assert spotter.streams[0].resets == 0

    @pytest.mark.asyncio
    async def test_a_seed_with_no_vad_from_feeds_from_the_first_frame_pushed(self):
        seed = seed_from(
            np.zeros(40 * FRAME_SAMPLES, dtype=np.float32), first_pos=0, floor=0, vad_from=None
        )
        spotter = LoggedSpotter()
        rig = Rig(seed=seed, spotter=spotter, engaged=False)

        await rig.quiet(640)

        fed = fed_positions(spotter.log)
        assert fed and fed == list(range(40, 40 + len(fed)))

    def test_the_docstrings_say_what_kws_overload_and_skipped_kws_hops_count(self):
        assert "KWS_BACKLOG_FRAMES" in (IngressOverload.__doc__ or "")
        assert "skipped_kws_hops" in (IngressStats.__doc__ or "")
        assert "every frame" in (spotter_feed.__doc__ or "")


class UnopenableSpotter(LoggedSpotter):
    """A spotter whose stream cannot be built, as when its interpreter fails to load."""

    def __init__(self) -> None:
        super().__init__()
        self.opens = 0

    def open_stream(self, threshold: float) -> LoggedStream:
        self.opens += 1
        raise SpeechEngineError("the interpreter could not be built")


class FirstPushFailsStream(LoggedStream):
    """A logged stream whose first push fails as a broken engine would; later pushes work."""

    def __init__(self, hits: Sequence[NameHit], threshold: float, log: list) -> None:
        super().__init__(hits, threshold, log)
        self.failed = False

    def push(self, pcm: bytes, pos: int) -> list[NameHit]:
        if not self.failed:
            self.failed = True
            raise SpeechEngineError("the first push failed")
        return super().push(pcm, pos)


class FirstPushFailsSpotter(LoggedSpotter):
    def open_stream(self, threshold: float) -> FirstPushFailsStream:
        stream = FirstPushFailsStream(self.hits, threshold, self.log)
        self.streams.append(stream)
        return stream


class TestTheSpotterLane:
    """The stream opens as its own `kws` job, and no failure leaves the spotter stuck busy."""

    @pytest.mark.asyncio
    async def test_the_stream_is_opened_by_its_own_kws_job_before_the_first_frames_job(self):
        spotter = LoggedSpotter()
        rig = Rig(engaged=False, spotter=spotter)

        await rig.quiet(640)

        submitted = rig.lanes.kws.submitted
        assert submitted[:2] == ["open_stream", "_spotter_job"]
        assert submitted.count("open_stream") == 1
        (stream,) = spotter.streams
        assert stream.threshold == WakeConfig().kws_threshold
        assert stream.warmed

    @pytest.mark.asyncio
    async def test_an_open_that_fails_turns_the_spotter_off_and_the_utterance_still_ends(
        self, caplog
    ):
        caplog.set_level(logging.WARNING, logger="backend.realtime.ingress")
        spotter = UnopenableSpotter()
        rig = Rig(spotter=spotter)

        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(960)

        assert [ended.text for ended in rig.of(UtteranceEnded)] == ["hello there"]
        assert spotter.opens == 1
        assert rig.lanes.kws.submitted == ["open_stream"]
        assert rig.ingress.stats().engine_errors == 1
        failures = [r for r in caplog.records if "spotter" in r.getMessage()]
        assert len(failures) == 1
        assert failures[0].exc_info is not None
        # No frame is queued for a spotter that is off, so none waits for a job that never runs.
        assert not rig.ingress._kws_waiting

    @pytest.mark.asyncio
    async def test_a_first_push_that_raises_keeps_the_stream_and_later_frames_are_fed(self):
        spotter = FirstPushFailsSpotter()
        rig = Rig(engaged=False, spotter=spotter)

        await rig.speak(640)

        assert rig.lanes.kws.submitted[:2] == ["open_stream", "_spotter_job"]
        assert rig.lanes.kws.submitted.count("open_stream") == 1
        (stream,) = spotter.streams
        assert stream.failed
        assert len(stream.pushed) > 20
        assert stream.pushed == list(range(stream.pushed[0], stream.pushed[0] + len(stream.pushed)))
        assert rig.ingress.stats().engine_errors == 1

    @pytest.mark.asyncio
    async def test_a_mute_during_the_open_skips_its_frames_and_feeds_the_fresh_stream_unreset(
        self,
    ):
        spotter = LoggedSpotter([hit_at(2)])
        rig = Rig(engaged=False, spotter=spotter)
        rig.lanes.kws.holding = True
        await rig.speak(320)
        assert rig.lanes.kws.submitted == ["open_stream"]

        rig.ingress.set_muted(True)
        rig.ingress.set_muted(False)
        unmuted_at = rig.pos
        rig.lanes.kws.release()
        await settle()
        await rig.speak(320)

        (stream,) = spotter.streams
        assert stream.resets == 0
        fed = fed_positions(spotter.log)
        assert fed and fed == list(range(unmuted_at, unmuted_at + len(fed)))
        assert 2 not in stream.pushed
        assert rig.of(NameHeard) == []

    @pytest.mark.asyncio
    async def test_a_backlog_reset_asked_for_during_an_open_a_mute_skipped_is_cleared(self):
        spotter = LoggedSpotter()
        rig = Rig(engaged=False, spotter=spotter)
        rig.lanes.kws.holding = True
        await rig.speak(320)
        rig.ingress.set_muted(True)
        rig.ingress.set_muted(False)
        unmuted_at = rig.pos

        await rig.quiet((KWS_BACKLOG_FRAMES + 30) * FRAME_MS)
        assert rig.ingress.stats().skipped_kws_hops > 0
        rig.lanes.kws.release()
        await settle()

        (stream,) = spotter.streams
        assert stream.resets == 0
        fed = fed_positions(spotter.log)
        assert fed and min(fed) >= unmuted_at
        assert fed == sorted(set(fed))

    @pytest.mark.asyncio
    async def test_an_on_hit_that_raises_once_leaves_the_frames_after_it_fed(self, monkeypatch):
        spotter = LoggedSpotter([hit_at(40), hit_at(60)])
        rig = Rig(engaged=False, spotter=spotter)
        report = rig.ingress._on_hit
        calls: list[int] = []

        async def on_hit_failing_once(hit: NameHit, generation: int) -> None:
            calls.append(hit.end_pos)
            if len(calls) == 1:
                raise RuntimeError("a bug in reporting a hit")
            await report(hit, generation)

        monkeypatch.setattr(rig.ingress, "_on_hit", on_hit_failing_once)
        await rig.quiet(640)

        await rig.speak(960)

        assert calls == [40, 60]
        (stream,) = spotter.streams
        assert stream.pushed[-1] == rig.pos - 1
        assert [heard.hit for heard in rig.of(NameHeard)] == [hit_at(60)]
        assert rig.ingress.stats().skipped_kws_hops == 0


def ref_frame(marker: bool) -> bytes:
    body = bytearray(640)
    if marker:
        body[100:104] = b"MARK"
    return bytes(body)


class TestReference:
    def rig(self) -> Rig:
        return Rig(
            engaged=False,
            spotter=ScriptedNameSpotter([hit_at(120)], markers={b"MARK": 0.8}, window_ms=1280),
            wake=WakeConfig(reference_tolerance_ms=400),
        )

    async def play(self, rig: Rig, reference: Callable[[int], bytes | None] | None) -> None:
        await rig.quiet(1600, reference=reference)
        await rig.speak(960, reference=reference)

    @pytest.mark.asyncio
    async def test_a_marker_within_the_tolerance_before_the_window_scores_at_the_hit(self):
        rig = self.rig()

        await self.play(rig, lambda pos: ref_frame(pos == 120 - 64 - 20 + 1))

        (heard,) = rig.of(NameHeard)
        assert heard.reference_score == 0.8

    @pytest.mark.asyncio
    async def test_a_marker_only_after_the_hit_gives_the_no_marker_score(self):
        rig = self.rig()

        await self.play(rig, lambda pos: ref_frame(pos > 120))

        (heard,) = rig.of(NameHeard)
        assert heard.reference_score == 0.0

    @pytest.mark.asyncio
    async def test_reference_frames_that_miss_the_window_give_none_and_count(self):
        rig = self.rig()

        await self.play(rig, lambda pos: ref_frame(False) if pos < 60 else None)

        (heard,) = rig.of(NameHeard)
        assert heard.reference_score is None
        assert rig.ingress.stats().reference_uncovered == 1

    @pytest.mark.asyncio
    async def test_with_no_reference_ever_pushed_it_is_none_and_nothing_is_counted(self):
        rig = self.rig()

        await self.play(rig, None)

        (heard,) = rig.of(NameHeard)
        assert heard.reference_score is None
        assert rig.ingress.stats().reference_uncovered == 0

    @pytest.mark.asyncio
    async def test_the_reference_ring_holds_at_most_window_ms_plus_tolerance(self):
        rig = self.rig()

        await self.play(rig, lambda pos: ref_frame(False))

        assert rig.ingress.stats().reference_frames == (1280 + 400) // FRAME_MS


class TestEndpointWake:
    def rig(self) -> Rig:
        return Rig(engaged=False)

    def expect(self, rig: Rig) -> None:
        rig.ingress.expect_endpoint_wake(
            keyword="hey mist", score=0.7, backlog_ms=4000, arrival=55.5
        )

    @pytest.mark.asyncio
    async def test_the_name_is_heard_with_the_backlogs_last_frame_and_not_before(self):
        rig = self.rig()
        self.expect(rig)
        await rig.quiet(640)
        await rig.speak(400)
        await rig.quiet(1200)
        await rig.speak(1740)
        assert rig.pos == 199 and rig.of(NameHeard) == []

        await rig.speak(20)

        (heard,) = rig.of(NameHeard)
        assert heard.hit == NameHit(end_pos=199, keyword="hey mist", score=0.7)
        assert (heard.spotter, heard.arrival, heard.reference_score) == ("endpoint", 55.5, None)
        assert heard.from_pos == 112 - PRE_ROLL

    @pytest.mark.asyncio
    async def test_an_800_ms_gap_in_the_backlog_chains_to_the_earlier_segment(self):
        rig = self.rig()
        self.expect(rig)
        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(800)

        await rig.speak(1920)

        assert [h.from_pos for h in rig.of(NameHeard)] == [32 - PRE_ROLL]

    @pytest.mark.asyncio
    async def test_speech_from_the_backlogs_first_frame_opens_at_that_frame(self):
        rig = self.rig()
        await rig.speak(800)
        self.expect(rig)

        await rig.speak(4000)

        assert [h.from_pos for h in rig.of(NameHeard)] == [40]

    @pytest.mark.asyncio
    async def test_a_connection_closing_inside_the_backlog_emits_nothing(self):
        rig = self.rig()
        self.expect(rig)
        await rig.speak(3000)

        rig.ingress.close()
        await rig.speak(1000)

        assert rig.of(NameHeard) == []


# -- The gate partial -----------------------------------------------------------------------------


class TestGatePartial:
    @pytest.mark.asyncio
    async def test_it_decodes_once_frames_through_through_pos_arrive(self):
        rig = Rig(endpointing=EndpointingConfig(stt_partial_interval_ms=10000))
        await rig.quiet(320)
        await rig.speak(640)
        started = rig.of(SpeechStarted)[0]

        rig.ingress.request_gate_partial(started.utterance_id, rig.pos + 5)
        await rig.speak(100)
        assert rig.of(TranscriptPartial) == []
        await rig.speak(20)

        (gate,) = rig.of(TranscriptPartial)
        assert (gate.gate, gate.through_pos, gate.audio_end_pos) == (True, rig.pos - 1, rig.pos - 1)

    @pytest.mark.asyncio
    async def test_it_is_the_one_partial_in_flight_and_cadence_ticks_skip(self):
        rig = Rig(endpointing=EndpointingConfig(stt_partial_interval_ms=100))
        await rig.quiet(320)
        await rig.speak(640)
        started = rig.of(SpeechStarted)[0]
        partials = len(rig.of(TranscriptPartial))
        skipped = rig.ingress.stats().skipped_partials
        rig.lanes.stt.holding = True

        rig.ingress.request_gate_partial(started.utterance_id, rig.pos - 1)
        await rig.speak(400)
        rig.lanes.stt.release()
        await settle()

        new = rig.of(TranscriptPartial)[partials:]
        assert [p.gate for p in new][:1] == [True]
        assert rig.ingress.stats().skipped_partials - skipped >= 3

    @pytest.mark.asyncio
    async def test_an_utterance_ending_within_the_gate_reuses_its_decode_as_the_final(self):
        rig = Rig(endpointing=EndpointingConfig(stt_partial_interval_ms=10000))
        await rig.quiet(320)
        await rig.speak(960)
        started = rig.of(SpeechStarted)[0]
        rig.lanes.stt.holding = True

        rig.ingress.request_gate_partial(started.utterance_id, rig.pos + 30)
        await rig.quiet(640)
        rig.lanes.stt.release()
        await settle()

        (gate,) = rig.of(TranscriptPartial)
        (ended,) = rig.of(UtteranceEnded)
        stream = rig.transcriber.streams[0]
        assert ended.text == gate.text == "hello there"
        assert (len(stream.partials), len(stream.finals)) == (1, 0)
        assert last_frame(ended) <= gate.through_pos


# -- Mute -----------------------------------------------------------------------------------------


class TestMute:
    @pytest.mark.asyncio
    async def test_muted_frames_are_discarded_before_the_vad_and_counted(self):
        rig = Rig()
        rig.ingress.set_muted(True)

        await rig.quiet(320)
        await rig.speak(640)

        assert rig.of(SpeechStarted) == []
        assert sum(len(stream.scores) for stream in rig.vad.streams) == 0
        assert rig.ingress.stats().muted_frames == 48

    @pytest.mark.asyncio
    async def test_unmuting_lets_speech_start_an_utterance_again(self):
        rig = Rig()
        rig.ingress.set_muted(True)
        await rig.speak(640)

        rig.ingress.set_muted(False)
        await rig.quiet(320)
        await rig.speak(640)

        assert len(rig.of(SpeechStarted)) == 1


def events_after(rig: Rig, event: object) -> list:
    return rig.events[rig.events.index(event) + 1 :]


class TestMuteClosesTheUtterance:
    """A mute closes the open utterance inside ingress and emits its drop (ruling of 12 Nov).

    The emitted drop is the controller's signal, so it needs no mute-specific close of its own,
    and audio from before the mute can never reach an utterance or a name opened after it.
    """

    def test_muted_is_an_ingress_reason_and_the_controllers_reasons_are_unchanged(self):
        assert "muted" in typing.get_args(DropReason)
        assert typing.get_args(ControllerDropReason) == ("text_input", "wake_lost", "not_addressed")

    @pytest.mark.asyncio
    async def test_mute_emits_one_drop_with_the_last_partial_and_the_next_event_seq(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)
        started = rig.of(SpeechStarted)[0]
        partial = rig.of(TranscriptPartial)[-1]
        last_seq = rig.events[-1].event_seq

        rig.ingress.set_muted(True)

        (dropped,) = rig.of(UtteranceDropped)
        assert dropped == UtteranceDropped(
            epoch=3,
            event_seq=last_seq + 1,
            utterance_id=started.utterance_id,
            first_pos=started.first_pos,
            reason="muted",
            text=partial.text,
            duration_ms=(rig.pos - started.first_pos) * FRAME_MS,
            no_speech_prob=None,
            avg_logprob=None,
        )
        assert rig.events[-1] is dropped
        assert [event.event_seq for event in rig.events] == list(range(1, len(rig.events) + 1))
        assert rig.ingress.idle_since() == dropped.event_seq
        assert rig.transcriber.streams[0].closed

    @pytest.mark.asyncio
    async def test_mute_before_any_partial_drops_with_no_text(self):
        rig = Rig(endpointing=EndpointingConfig(stt_partial_interval_ms=10000))
        await rig.marked(10)
        opened = rig.ingress.open_utterance("wake", 2)

        rig.ingress.set_muted(True)

        (dropped,) = rig.of(UtteranceDropped)
        assert (dropped.utterance_id, dropped.first_pos) == (opened.utterance_id, 2)
        assert (dropped.reason, dropped.text) == ("muted", None)
        assert dropped.duration_ms == (rig.pos - 2) * FRAME_MS
        assert rig.of(TranscriptPartial) == []

    @pytest.mark.asyncio
    async def test_a_partial_decode_in_flight_at_mute_yields_nothing_after_the_drop(self):
        rig = Rig(endpointing=EndpointingConfig(stt_partial_interval_ms=100))
        await rig.quiet(640)
        await rig.speak(320)
        rig.lanes.stt.holding = True
        await rig.speak(160)
        started = rig.of(SpeechStarted)[0]
        assert rig.lanes.stt.submitted[-1] == "decode_partial"

        rig.ingress.set_muted(True)
        rig.lanes.stt.release()
        await settle()
        rig.ingress.set_muted(False)
        await rig.quiet(640)

        (dropped,) = rig.of(UtteranceDropped)
        assert dropped.utterance_id == started.utterance_id
        assert [
            event
            for event in events_after(rig, dropped)
            if getattr(event, "utterance_id", None) == started.utterance_id
        ] == []

    @pytest.mark.asyncio
    async def test_a_final_decode_in_flight_at_mute_yields_no_candidate_and_no_end(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)
        rig.lanes.stt.holding = True
        await rig.quiet(320)
        started = rig.of(SpeechStarted)[0]
        assert "decode_final" in rig.lanes.stt.submitted

        rig.ingress.set_muted(True)
        rig.lanes.stt.release()
        await settle()
        rig.ingress.set_muted(False)
        await rig.quiet(640)

        (dropped,) = rig.of(UtteranceDropped)
        assert (dropped.utterance_id, dropped.reason) == (started.utterance_id, "muted")
        assert rig.of(EndpointCandidate) == [] and rig.of(UtteranceEnded) == []
        assert "decode_final" not in rig.lanes.stt.started

    @pytest.mark.asyncio
    async def test_mute_with_no_utterance_open_emits_nothing(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)
        await rig.quiet(480)
        assert len(rig.of(UtteranceEnded)) == 1
        count = len(rig.events)

        rig.ingress.set_muted(True)

        assert len(rig.events) == count

    @pytest.mark.asyncio
    async def test_a_second_mute_emits_nothing(self):
        rig = Rig()
        await rig.quiet(640)
        await rig.speak(640)
        rig.ingress.set_muted(True)
        count = len(rig.events)

        rig.ingress.set_muted(True)

        assert len(rig.events) == count
        assert len(rig.of(UtteranceDropped)) == 1

    @pytest.mark.asyncio
    async def test_a_wake_while_muted_opens_nothing_and_is_dropped_at_once(self):
        rig = Rig(transcriber=RecordingTranscriber(["hello there"]), retention_ms=400)
        await rig.quiet(640)
        await rig.speak(640)
        before = rig.of(SpeechStarted)[0]
        rig.ingress.set_muted(True)
        count = len(rig.events)

        opened = rig.ingress.open_utterance("wake", 4)

        assert opened.utterance_id != before.utterance_id
        assert opened.first_pos == rig.pos
        assert len(rig.transcriber.streams) == 1
        (dropped,) = rig.events[count:]
        assert dropped == UtteranceDropped(
            epoch=3,
            event_seq=rig.events[count - 1].event_seq + 1,
            utterance_id=opened.utterance_id,
            first_pos=opened.first_pos,
            reason="muted",
            text=None,
            duration_ms=0,
            no_speech_prob=None,
            avg_logprob=None,
        )
        assert rig.of(IngressOverload) == []
        assert rig.ingress.idle_since() == dropped.event_seq

    @pytest.mark.asyncio
    async def test_an_endpoint_wake_partly_counted_at_mute_is_never_heard(self):
        rig = Rig(engaged=False)
        rig.ingress.expect_endpoint_wake(
            keyword="hey mist", score=0.7, backlog_ms=4000, arrival=55.5
        )
        await rig.speak(1000)

        rig.ingress.set_muted(True)
        await rig.speak(400)
        rig.ingress.set_muted(False)
        await rig.speak(4000)
        await rig.quiet(400)

        assert rig.of(NameHeard) == []

    @pytest.mark.asyncio
    async def test_an_endpoint_wake_whose_last_frame_is_unscored_at_mute_is_never_heard(self):
        rig = Rig(engaged=False)
        rig.ingress.expect_endpoint_wake(keyword="hey mist", score=0.7, backlog_ms=400, arrival=5.0)
        await rig.quiet(380)
        await rig.quiet(20, settle_after=False)
        assert rig.of(NameHeard) == []

        rig.ingress.set_muted(True)
        await settle()
        rig.ingress.set_muted(False)
        await rig.quiet(400)

        assert rig.of(NameHeard) == []

    @pytest.mark.asyncio
    async def test_after_unmute_a_wake_from_before_the_mute_starts_after_the_unmute(self):
        rig = Rig(transcriber=RecordingTranscriber([""]))
        await rig.marked(10)
        for _ in range(5):
            await rig.push_pcm(marked_pcm(rig.pos), settle_after=False)
        assert rig.ingress.handoff().vad_from <= 10

        rig.ingress.set_muted(True)
        await rig.marked(5)
        rig.ingress.set_muted(False)
        first_after = rig.pos
        await rig.marked(5)
        opened = rig.ingress.open_utterance("wake", 2)

        assert opened.first_pos >= first_after
        assert min(positions_in(rig.transcriber.streams[0].audio)) >= first_after

    @pytest.mark.asyncio
    async def test_after_unmute_vad_speech_opens_a_fresh_utterance(self):
        rig = Rig(transcriber=ScriptedTranscriber(["one", "two"], ms_per_word=20))
        await rig.quiet(640)
        await rig.speak(640)
        first = rig.of(SpeechStarted)[0]

        rig.ingress.set_muted(True)
        rig.ingress.set_muted(False)
        unmuted_at = rig.pos
        # Speech straight after the unmute: before the fix the old utterance was still open and
        # took it, so no second utterance started.
        await rig.speak(640)

        started = rig.of(SpeechStarted)
        assert len(started) == 2
        assert started[1].utterance_id != first.utterance_id
        assert started[1].first_pos >= unmuted_at
        assert len(rig.transcriber.streams) == 2

    def test_the_docstrings_say_ingress_closes_the_utterance_on_mute(self):
        assert "controller's to close" not in (AudioIngress.set_muted.__doc__ or "")
        for documented in (AudioIngress.set_muted, AudioIngress.open_utterance, UtteranceDropped):
            assert "muted" in (documented.__doc__ or "")
        assert "mute" in (IngressSeed.__doc__ or "")


# -- Real lanes -----------------------------------------------------------------------------------


class TestWithInferenceLanes:
    @pytest.mark.asyncio
    async def test_an_utterance_runs_end_to_end_on_real_lane_threads(self):
        lanes = IngressLanes(
            vad=InferenceLane("vad"),
            stt=InferenceLane("stt"),
            eot=InferenceLane("eot"),
            kws=InferenceLane("kws"),
        )
        rig = Rig(lanes=lanes)
        audio = scripted_audio([silence(640), speech_burst(640), silence(960)], seed=1)
        try:
            for start in range(0, len(audio), FRAME_SAMPLES * 8):
                await rig.push_audio(audio[start : start + FRAME_SAMPLES * 8], settle_after=False)
                windows = (rig.pos * FRAME_SAMPLES) // 512
                for _ in range(500):
                    if sum(len(s.scores) for s in rig.vad.streams) >= windows:
                        break
                    await asyncio.sleep(0.01)
            for _ in range(500):
                if rig.of(UtteranceEnded):
                    break
                await asyncio.sleep(0.01)
        finally:
            for lane in (lanes.vad, lanes.stt, lanes.eot, lanes.kws):
                lane.shutdown()

        assert [e.text for e in rig.of(UtteranceEnded)] == ["hello there"]


# -- Speech segments ------------------------------------------------------------------------------


def windows(
    tracker: SpeechSegmentTracker, start: int, count: int, *, speech: bool
) -> list[SegmentStep]:
    """Score VAD windows `start` to `start + count - 1`, each spanning the frames it covers."""
    return [
        tracker.window(speech, index * 512 // FRAME_SAMPLES, (index * 512 + 511) // FRAME_SAMPLES)
        for index in range(start, start + count)
    ]


NOTHING = SegmentStep(started=None, closed=None)


def opened_tracker() -> SpeechSegmentTracker:
    """Two silent windows, then three speech windows (2 to 4) opening a segment at frame 3."""
    tracker = SpeechSegmentTracker(EndpointingConfig())
    windows(tracker, 0, 2, speech=False)
    *_, step = windows(tracker, 2, 3, speech=True)
    assert step == SegmentStep(started=(3, 7), closed=None)
    return tracker


class TestSpeechSegmentTracker:
    def test_a_run_shorter_than_speech_start_windows_opens_nothing(self):
        tracker = SpeechSegmentTracker(EndpointingConfig())

        steps = windows(tracker, 0, 2, speech=True)
        steps += windows(tracker, 2, 1, speech=False)
        steps += windows(tracker, 3, 2, speech=True)

        assert steps == [NOTHING] * 5
        assert tracker.open_segment is None

    def test_speech_start_windows_in_a_row_open_a_segment_at_the_runs_first_frame(self):
        tracker = opened_tracker()

        assert tracker.open_segment == (3, 7)
        assert tracker.silence_ms() == 0

    def test_speech_extends_the_open_segment_and_clears_its_silence(self):
        tracker = opened_tracker()
        windows(tracker, 5, 3, speech=False)
        assert tracker.silence_ms() == 96

        assert windows(tracker, 8, 1, speech=True) == [NOTHING]

        assert tracker.open_segment == (3, (8 * 512 + 511) // FRAME_SAMPLES)
        assert tracker.silence_ms() == 0

    def test_silence_closes_the_segment_at_endpoint_silence_ms(self):
        tracker = opened_tracker()

        steps = windows(tracker, 5, 7, speech=False)

        # Six 32 ms windows are 192 ms, short of 200; the seventh closes it.
        assert steps[:6] == [NOTHING] * 6
        assert steps[6] == SegmentStep(started=None, closed=(3, 7))
        assert tracker.open_segment is None
        assert tracker.silence_ms() == 224

    def test_restart_silence_counts_the_pause_again_from_zero(self):
        tracker = opened_tracker()
        windows(tracker, 5, 5, speech=False)

        tracker.restart_silence()

        assert tracker.silence_ms() == 0
        steps = windows(tracker, 10, 7, speech=False)
        assert steps[:6] == [NOTHING] * 6
        assert steps[6].closed == (3, 7)

    def test_reset_returns_the_open_segment_and_forgets_the_run(self):
        tracker = opened_tracker()

        assert tracker.reset() == (3, 7)
        assert tracker.open_segment is None
        assert tracker.reset() is None

        windows(tracker, 5, 2, speech=True)
        tracker.reset()
        assert windows(tracker, 7, 2, speech=True) == [NOTHING] * 2
        (step,) = windows(tracker, 9, 1, speech=True)
        assert step.started == (7 * 512 // FRAME_SAMPLES, (9 * 512 + 511) // FRAME_SAMPLES)

    def test_with_no_run_the_earliest_onset_is_the_first_unscored_frame(self):
        tracker = SpeechSegmentTracker(EndpointingConfig())
        windows(tracker, 0, 2, speech=False)

        assert tracker.earliest_onset(4) == 4

    def test_while_a_run_is_under_way_the_earliest_onset_is_its_first_frame(self):
        tracker = SpeechSegmentTracker(EndpointingConfig())
        windows(tracker, 0, 2, speech=False)
        windows(tracker, 2, 2, speech=True)

        assert tracker.earliest_onset(7) == 3

        windows(tracker, 4, 1, speech=False)
        assert tracker.earliest_onset(8) == 8


class TestSpotterFeed:
    def test_with_a_segment_open_frames_before_its_pre_roll_are_warmed_and_the_rest_pushed(self):
        assert spotter_feed(10, 60, 40, 40, PRE_ROLL) == SpotterFeed(
            warm=range(10, 40 - PRE_ROLL), push=range(40 - PRE_ROLL, 61)
        )

    def test_with_a_segment_open_a_pre_roll_partly_fed_is_pushed_from_the_next_frame(self):
        feed = spotter_feed(30, 60, 40, 40, PRE_ROLL)

        assert (list(feed.warm), feed.push) == ([], range(30, 61))

    def test_with_none_open_frames_an_onset_could_make_pre_roll_wait(self):
        feed = spotter_feed(10, 60, None, 50, PRE_ROLL)

        assert (feed.warm, list(feed.push)) == (range(10, 50 - PRE_ROLL), [])

    def test_with_none_open_and_no_onset_possible_by_the_newest_frame_all_are_warmed(self):
        feed = spotter_feed(10, 60, None, 100, PRE_ROLL)

        assert (feed.warm, list(feed.push)) == (range(10, 61), [])

    @pytest.mark.parametrize(
        ("next_pos", "onset", "earliest"), [(61, 40, 40), (40, None, 50), (35, None, 50)]
    )
    def test_nothing_is_released_twice_or_early(self, next_pos, onset, earliest):
        feed = spotter_feed(next_pos, 60, onset, earliest, PRE_ROLL)

        assert (list(feed.warm), list(feed.push)) == ([], [])

    def test_the_pre_roll_in_frames_comes_from_the_endpointing_config(self):
        assert onset_lead_frames(EndpointingConfig()) == PRE_ROLL
        assert onset_lead_frames(EndpointingConfig(pre_roll_ms=210)) == 10


class TestIngressSegmentsThroughTheTracker:
    @pytest.mark.asyncio
    async def test_every_vad_window_goes_through_the_shared_tracker(self, monkeypatch):
        steps: list[SegmentStep] = []

        class Recording(SpeechSegmentTracker):
            def window(self, speech: bool, first: int, last: int) -> SegmentStep:
                step = super().window(speech, first, last)
                steps.append(step)
                return step

        monkeypatch.setattr(ingress_module, "SpeechSegmentTracker", Recording)
        rig = Rig()

        await rig.quiet(640)
        await rig.speak(320)

        assert len(steps) == sum(len(stream.scores) for stream in rig.vad.streams)
        assert [step.started[0] for step in steps if step.started is not None] == [32]
        assert rig.of(SpeechStarted)[0].first_pos == 32 - PRE_ROLL
