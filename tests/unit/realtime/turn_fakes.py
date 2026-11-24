"""Fakes the turn controller's tests share (T11), and that T11b's model check reads.

This is not a `conftest.py`: `tests/unit/realtime/conftest.py` belongs to another task. Its name
does not start with `test_`, so pytest does not collect it. Each test file imports what it needs.

- `FakeClock`: the session clock, which moves only when a test calls `advance`.
- `IdCounter`: the one utterance id source the fake ingress and the controller share (v4).
- `InlineLane`: a TTS lane that runs each job on the event loop, so synthesis is deterministic;
  a test can hold its jobs back until it opens it.
- `FailingSynthesizer`: a `FakeSynthesizer` that fails one call, at once or after some chunks.
- `RecordingOutbound`: the recording transport. It models the `turn` lane's queue and purge with
  the real `TurnLane`, and a sender that takes `control` before `turn`, either at once
  (`auto_take`) or when the test says so. What the sender took is the client-facing record. It
  notes a violation for any `turn` item queued before its turn's `stream_start`, and for a
  frame or `stream_token` of a turn queued after that turn's purge.
- `RecordingScenePort`: records every scene call.
- `RecordingIngress`: the controller's levers on ingress, recorded, and builders for the events
  ingress emits, numbered by epoch and `event_seq`.
- `FakeResponder`: scripted turn streams; records requests, prefetches, commits and closes. A
  `WaitGate` step waits on the scope's dispatch gate, as the handler does before a tool. Its
  capability-only path records each call and returns the next scripted result, None without one.
- `FakeAlarm`: the controller's `alarm_ringing`, which a test rings and stops by hand.
- `FakeHost`: a `ReconfigureHost` that records its answers.
- `FakeWakeArbiter`: a `WakeArbitration` whose verdicts a test grants or denies by hand; a
  denial carries a winner's score.
- `FakeCallLater`: the loop's `call_later` for a real `WakeArbiter`: it records each timer, and
  a test fires it by hand.
- `Harness` and `make_harness`: a controller wired to all of the above.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass, field

from backend.chat.conversation_handler import CapabilityOnlyTurn
from backend.chat.stream_events import StreamEvent, Token, WSEvent
from backend.chat.turn_scope import CommitResult, TurnDelivery, TurnRequest, TurnScope
from backend.realtime.addressee import AddresseeGate
from backend.realtime.bargein import AecMode, KeywordBargeIn
from backend.realtime.budget import (
    BUDGET_PRESETS,
    BargeInConfig,
    PlaybackConfig,
    SegmenterConfig,
    TurnConfig,
    WakeConfig,
)
from backend.realtime.errors import SpeechEngineError
from backend.realtime.ingress import (
    EndpointCandidate,
    IngressOverload,
    IngressSeed,
    NameHeard,
    OpenedUtterance,
    SpeechResumed,
    SpeechStarted,
    TranscriptPartial,
    UtteranceDropped,
    UtteranceEnded,
    WakeVerified,
)
from backend.realtime.lanes import LaneJobCancelled
from backend.realtime.playback import LanePurge, TurnLane
from backend.realtime.protocol import Frame, MsgType, VoiceStartMessage
from backend.realtime.scene import SceneAnchor
from backend.realtime.speech import (
    CancelToken,
    NameHit,
    SpotterCalibration,
    SynthChunk,
    TextSegment,
)
from backend.realtime.turn import (
    Reconfigure,
    TurnController,
    TurnPlacement,
    VoiceStart,
    WakeArbitration,
    WakeRank,
)
from tests.fakes.speech import FakeSynthesizer, ScriptedWakeVerifier

# -- Clock and ids -------------------------------------------------------------


async def settle(rounds: int = 60) -> None:
    """Let every ready task run until the loop is quiet."""
    for _ in range(rounds):
        await asyncio.sleep(0)


class FakeClock:
    """A session clock that moves only when the test calls `advance`."""

    def __init__(self, now: int = 0) -> None:
        self.now = now
        self._sleepers: list[tuple[int, int, asyncio.Future[None]]] = []
        self._order = 0

    def now_ms(self) -> int:
        return self.now

    def __call__(self) -> int:
        return self.now

    async def sleep_ms(self, ms: int) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._order += 1
        heapq.heappush(self._sleepers, (self.now + ms, self._order, future))
        await future

    async def advance(self, ms: int) -> None:
        target = self.now + ms
        await settle()
        while self._sleepers and self._sleepers[0][0] <= target:
            wake, _, future = heapq.heappop(self._sleepers)
            self.now = max(self.now, wake)
            if not future.done():
                future.set_result(None)
            await settle()
        self.now = target
        await settle()


class IdCounter:
    """One id source for ingress and the controller, so no two ids ever repeat (v4)."""

    def __init__(self, prefix: str = "u") -> None:
        self._prefix = prefix
        self._next = itertools.count(1)
        self.issued: list[str] = []

    def __call__(self) -> str:
        value = f"{self._prefix}{next(self._next)}"
        self.issued.append(value)
        return value


class InlineLane:
    """A lane that runs each job inline on the loop; a set token fails it as the real lane does.

    A job that raises anything but `SpeechEngineError` or `LaneJobCancelled` fails with
    `SpeechEngineError`, as the real lane's worker does when such an error ends its thread, so a
    test sees the error the controller would.

    `hold()` makes every later job wait until `open()`, as a busy TTS lane would, so a test can
    queue a turn's tokens before any of their audio exists.
    """

    def __init__(self) -> None:
        self.jobs = 0
        self._gate: asyncio.Event | None = None

    def hold(self) -> None:
        self._gate = asyncio.Event()

    def open(self) -> None:
        if self._gate is not None:
            self._gate.set()
            self._gate = None

    async def run(self, fn: Callable[..., object], *args: object, token: CancelToken) -> object:
        if token.is_set():
            raise LaneJobCancelled("token set before the job ran")
        self.jobs += 1
        await asyncio.sleep(0)
        while self._gate is not None:
            await self._gate.wait()
        if token.is_set():
            raise LaneJobCancelled("token set while the job waited")
        try:
            return fn(*args)
        except (SpeechEngineError, LaneJobCancelled):
            raise
        except Exception as exc:  # the real lane's worker dies on any other error
            raise SpeechEngineError("inline lane: worker thread died") from exc


class FailingSynthesizer(FakeSynthesizer):
    """A `FakeSynthesizer` whose `fail_call`-th `synthesize` (0-based) fails.

    With `after_chunks` 0 the call itself raises; otherwise its iterator yields that many
    chunks and then raises, as an engine failing mid-segment does.
    """

    def __init__(self, *, fail_call: int = 0, after_chunks: int = 0, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.fail_call = fail_call
        self.after_chunks = after_chunks
        self.calls = 0

    def synthesize(self, segment: TextSegment, token: CancelToken) -> Iterator[SynthChunk]:
        call = self.calls
        self.calls += 1
        if call != self.fail_call:
            return super().synthesize(segment, token)
        if self.after_chunks == 0:
            raise SpeechEngineError("fake synthesis failure")
        return self._failing(super().synthesize(segment, token))

    def _failing(self, chunks: Iterator[SynthChunk]) -> Iterator[SynthChunk]:
        for _ in range(self.after_chunks):
            yield next(chunks)
        raise SpeechEngineError("fake synthesis failure")


# -- The recording transport -----------------------------------------------------


def describe(item: object) -> str:
    """A short name for an outbound item: `0x01`, `state_cycle:think`, `stream_token`, ..."""
    if isinstance(item, Frame):
        return f"0x{int(item.msg_type):02x}"
    if isinstance(item, dict):
        return str(item.get("type"))
    kind = getattr(item, "type", type(item).__name__)
    if kind == "voice_decision":
        return f"voice_decision:{item.kind}"
    if kind == "state_cycle":
        return f"state_cycle:{item.state}"
    if kind == "vad_status":
        return f"vad_status:{item.status}"
    return str(kind)


def turn_id_of(item: object) -> str | None:
    if isinstance(item, dict):
        value = item.get("turn_id")
        return value if isinstance(value, str) else None
    return getattr(item, "turn_id", None)


@dataclass
class Sent:
    lane: str
    item: object
    turn_id: str | None = None


class RecordingOutbound:
    """The recording `Outbound`: control and turn lanes, a sender and the client-facing record."""

    def __init__(self, *, auto_take: bool = True) -> None:
        self.auto_take = auto_take
        self.lane = TurnLane()
        self.control: list[object] = []
        self.client: list[object] = []
        self.sent: list[Sent] = []
        self.purges: list[tuple[str, LanePurge]] = []
        self.violations: list[str] = []
        self._started: set[str] = set()
        self._purged: set[str] = set()

    # Outbound
    def send_control(self, item: object) -> None:
        self.sent.append(Sent("control", item, turn_id_of(item)))
        self.control.append(item)
        if self.auto_take:
            self.take_all()

    def send_turn(
        self,
        turn_id: str,
        item: object,
        terminal: bool = False,
        kind: str | None = None,
        token_index: int | None = None,
    ) -> bool:
        if kind == "stream_start":
            self._started.add(turn_id)
        elif turn_id not in self._started:
            self.violations.append(f"{describe(item)} queued before stream_start of {turn_id}")
        if turn_id in self._purged and (
            kind == "token"
            or (
                isinstance(item, Frame)
                and item.msg_type in (MsgType.AUDIO_CHUNK, MsgType.AUDIO_COMPLETE)
            )
        ):
            self.violations.append(f"{describe(item)} queued after the purge of {turn_id}")
        self.sent.append(Sent("turn", item, turn_id))
        self.lane.put(turn_id, item, terminal=terminal, kind=kind, token_index=token_index)
        if self.auto_take:
            self.take_all()
        return True

    def purge_turn(self, turn_id: str, *, audio_only: bool = False) -> LanePurge:
        purge = self.lane.purge(turn_id, audio_only=audio_only)
        self._purged.add(turn_id)
        self.purges.append((turn_id, purge))
        return purge

    # The sender
    def take(self, count: int = 1) -> None:
        """Take up to `count` items, `control` first, as the session's sender does."""
        for _ in range(count):
            if self.control:
                self.client.append(self.control.pop(0))
                continue
            item = self.lane.take()
            if item is None:
                return
            self.client.append(item)

    def take_until(self, done: Callable[[object], bool]) -> None:
        """Take items one at a time until one that `done` accepts has been taken."""
        while True:
            before = len(self.client)
            self.take()
            if len(self.client) == before:
                raise AssertionError("the lanes ran empty before the awaited item was taken")
            if done(self.client[-1]):
                return

    def take_all(self) -> None:
        while True:
            if self.control:
                self.client.append(self.control.pop(0))
                continue
            item = self.lane.take()
            if item is None:
                return
            self.client.append(item)

    # Views
    def names(self) -> list[str]:
        return [describe(item) for item in self.client]

    def of(self, name: str) -> list[object]:
        return [item for item in self.client if describe(item) == name]

    def turn_ids(self) -> list[str]:
        return [item.turn_id for item in self.of("stream_start")]

    def queued_turn_ids(self) -> list[str]:
        """Every turn whose `stream_start` was queued, taken by the sender or not."""
        return [s.item.turn_id for s in self.sent if describe(s.item) == "stream_start"]

    def tokens(self, turn_id: str | None = None) -> list[str]:
        return [
            item["token"]
            for item in self.of("stream_token")
            if turn_id is None or item["turn_id"] == turn_id
        ]


# -- Scene --------------------------------------------------------------------------


class RecordingScenePort:
    """Records each `ScenePort` call as a tuple, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.on_call: Callable[[tuple[object, ...]], object] | None = None

    def _record(self, call: tuple[object, ...]) -> None:
        self.calls.append(call)
        if self.on_call is not None:
            self.on_call(call)

    def phase(self, phase: str, *, turn_id: str | None, at: SceneAnchor) -> None:
        self._record(("phase", phase, turn_id, at))

    def tool_view(
        self,
        turn_id: str,
        call_id: str,
        view: str,
        data: object,
        *,
        at: SceneAnchor,
    ) -> None:
        self._record(("tool_view", turn_id, call_id, view, data, at))

    def cancel_turn(self, turn_id: str) -> None:
        self._record(("cancel_turn", turn_id))

    def turn_ended(self, turn_id: str) -> None:
        self._record(("turn_ended", turn_id))

    def segment_starting(self, audio_tag: int, segment_index: int, first_sample: int) -> None:
        self._record(("segment_starting", audio_tag, segment_index, first_sample))

    def stream_ending(self, audio_tag: int, last_sample: int) -> None:
        self._record(("stream_ending", audio_tag, last_sample))

    def phases(self) -> list[tuple[object, ...]]:
        return [call for call in self.calls if call[0] == "phase"]


# -- Ingress -------------------------------------------------------------------------


class RecordingIngress:
    """The ingress levers the controller calls, recorded, and builders for ingress events."""

    def __init__(self, ids: IdCounter, *, epoch: int = 1) -> None:
        self.ids = ids
        self.epoch = epoch
        self.calls: list[tuple[object, ...]] = []
        self._counter = itertools.count(1)
        self.emitted = 0
        self.holding = False
        self.close_result: UtteranceDropped | None = None
        self.first_pos: dict[str, int] = {}

    def _next_seq(self) -> int:
        """The next event's `event_seq`; a built event counts as emitted by ingress."""
        self.emitted = next(self._counter)
        return self.emitted

    # Levers
    def open_utterance(self, mode: str, from_pos: int) -> OpenedUtterance:
        uid = self.ids()
        self.calls.append(("open_utterance", mode, from_pos, uid))
        self.first_pos[uid] = from_pos
        return OpenedUtterance(uid, from_pos)

    def request_verify(self, utterance_id: str, hit_pos: int) -> None:
        self.calls.append(("request_verify", utterance_id, hit_pos))

    def set_engaged(self, engaged: bool) -> None:
        self.calls.append(("set_engaged", engaged))

    def set_speaking(self, speaking: bool) -> None:
        self.calls.append(("set_speaking", speaking))

    def set_muted(self, muted: bool) -> None:
        self.calls.append(("set_muted", muted))

    def expect_endpoint_wake(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("expect_endpoint_wake", args, kwargs))

    def drop(self, utterance_id: str, reason: str) -> UtteranceDropped | None:
        self.calls.append(("drop", utterance_id, reason))
        return None

    def close(self) -> UtteranceDropped | None:
        self.calls.append(("close",))
        return self.close_result

    def idle_since(self) -> int | None:
        """None while a test says ingress holds an utterance, else the last event's seq."""
        return None if self.holding else self.emitted

    def handoff(self) -> IngressSeed:
        self.calls.append(("handoff",))
        return IngressSeed(frames=(), floor=0, vad_from=None)

    # Events
    def new_id(self) -> str:
        return self.ids()

    def _pos(self, uid: str) -> int:
        return self.first_pos.setdefault(uid, 100)

    def started(self, uid: str) -> SpeechStarted:
        return SpeechStarted(self.epoch, self._next_seq(), uid, self._pos(uid))

    def resumed(self, uid: str) -> SpeechResumed:
        return SpeechResumed(self.epoch, self._next_seq(), uid, self._pos(uid))

    def partial(self, uid: str, text: str, *, heard_at_ms: int = 0) -> TranscriptPartial:
        return TranscriptPartial(
            epoch=self.epoch,
            event_seq=self._next_seq(),
            utterance_id=uid,
            first_pos=self._pos(uid),
            text=text,
            stable_chars=0,
            heard_at_ms=heard_at_ms,
            audio_end_pos=self._pos(uid) + 10,
            uncovered_since_ms=heard_at_ms,
        )

    def heard(
        self,
        uid: str | None = None,
        *,
        from_pos: int = 100,
        hit_pos: int = 110,
        arrival: float = 0.0,
        score: float = 0.9,
        reference_score: float | None = None,
        spotter: str = "server",
        keyword: str = "mist",
    ) -> NameHeard:
        """A name hit; `uid` is the open utterance it fell in, None when ingress was dormant."""
        return NameHeard(
            epoch=self.epoch,
            event_seq=self._next_seq(),
            hit=NameHit(end_pos=hit_pos, keyword=keyword, score=score),
            arrival=arrival,
            from_pos=from_pos,
            utterance_id=uid,
            reference_score=reference_score,
            spotter=spotter,  # type: ignore[arg-type]
        )

    def candidate(self, uid: str, text: str, *, heard_at_ms: int = 0) -> EndpointCandidate:
        return EndpointCandidate(
            self.epoch, self._next_seq(), uid, self._pos(uid), text, heard_at_ms
        )

    def ended(
        self, uid: str, text: str, *, heard_at_ms: int = 0, silence_ms: int = 0
    ) -> UtteranceEnded:
        return UtteranceEnded(
            epoch=self.epoch,
            event_seq=self._next_seq(),
            utterance_id=uid,
            first_pos=self._pos(uid),
            text=text,
            reason="semantic",
            eot_probability=0.9,
            silence_ms=silence_ms,
            duration_ms=800,
            no_speech_prob=0.01,
            avg_logprob=-0.2,
            heard_at_ms=heard_at_ms,
        )

    def dropped(self, uid: str, reason: str = "noise", text: str | None = None) -> UtteranceDropped:
        return UtteranceDropped(
            epoch=self.epoch,
            event_seq=self._next_seq(),
            utterance_id=uid,
            first_pos=self._pos(uid),
            reason=reason,  # type: ignore[arg-type]
            text=text,
            duration_ms=300,
            no_speech_prob=None,
            avg_logprob=None,
        )

    def overload(self, lane: str = "vad", dropped_ms: int = 120) -> IngressOverload:
        return IngressOverload(self.epoch, self._next_seq(), lane, dropped_ms, None, None)

    def verified(self, uid: str, *, hit_pos: int = 110, score: float | None = 0.9) -> WakeVerified:
        """The verifier's score for the hit at `hit_pos`; None as ingress sends on any failure."""
        return WakeVerified(self.epoch, self._next_seq(), uid, hit_pos, score)


def default_gate(
    config: WakeConfig | None = None, thresholds: AddresseeGate.Thresholds | None = None
) -> AddresseeGate:
    """An addressee gate built from the defaults (v5), with the verifier's `thresholds`."""
    return AddresseeGate(
        config or WakeConfig(),
        echo_guard_ms=BargeInConfig().echo_guard_ms,
        thresholds=thresholds,
    )


# -- The responder ------------------------------------------------------------------------


@dataclass
class Pause:
    """A script step: the fake stream waits here until the test releases it."""

    event: asyncio.Event = field(default_factory=asyncio.Event)
    reached: asyncio.Event = field(default_factory=asyncio.Event)

    def release(self) -> None:
        self.event.set()


@dataclass
class Call:
    """A script step: run `fn(scope)` inside the stream, as the handler's own work would."""

    fn: Callable[[TurnScope], None]


@dataclass
class WaitGate:
    """A script step: wait for the scope's dispatch gate, as the handler does before a tool."""


ScriptStep = StreamEvent | Pause | Call | WaitGate


def tool_started(call_id: str = "call-1", name: str = "set_timer") -> WSEvent:
    return WSEvent({"type": "tool_call_started", "tool_call_id": call_id, "name": name})


def tool_completed(
    call_id: str = "call-1", name: str = "set_timer", view: str | None = None
) -> WSEvent:
    payload: dict[str, object] = {
        "type": "tool_call_completed",
        "tool_call_id": call_id,
        "name": name,
        "duration_ms": 5,
        "result_summary": "ok",
        "error": None,
    }
    if view is not None:
        payload["view"] = view
        payload["data"] = {"label": "pasta"}
    return WSEvent(payload)


def dispatch_tool(scope: TurnScope) -> None:
    scope.tool_dispatched = True
    scope.tool_calls.append({"name": "set_timer", "args": {}})


class FakeResponder:
    """A `TurnResponder` whose turns stream scripted events.

    Each `stream_turn` takes the next script, or the default one. `log` records `open`, each
    event as it is yielded (`yield:<kind>:<text>`), `closed` when a stream is closed before its
    end, and `commit:<outcome>`.
    """

    DEFAULT = (Token("Hello there."),)

    def __init__(
        self,
        *scripts: Iterable[ScriptStep],
        commit_result: CommitResult | None = None,
        clock: FakeClock | None = None,
    ) -> None:
        self.scripts: list[list[ScriptStep]] = [list(script) for script in scripts]
        self.commit_result = commit_result or CommitResult(True, "event-1", None)
        self.clock = clock
        self.requests: list[TurnRequest] = []
        self.scopes: list[TurnScope] = []
        self.prefetches: list[tuple[str, str | None]] = []
        self.commits: list[TurnDelivery] = []
        self.commit_times: list[int] = []
        self.log: list[str] = []
        self.closed: list[int] = []
        self.capability_results: list[CapabilityOnlyTurn | None] = []
        self.capability_calls: list[tuple[str, str]] = []
        self.on_capability: Callable[[str], None] | None = None

    def add(self, *steps: ScriptStep) -> None:
        self.scripts.append(list(steps))

    def open_turn(self, request: TurnRequest) -> TurnScope:
        scope = TurnScope(
            session_id=request.session_id, turn_index=len(self.scopes), request=request
        )
        if not request.held:
            scope.dispatch_gate.set()
        self.requests.append(request)
        self.scopes.append(scope)
        self.log.append("open")
        return scope

    async def stream_turn(self, scope: TurnScope) -> AsyncIterator[StreamEvent]:
        script = self.scripts.pop(0) if self.scripts else list(self.DEFAULT)
        finished = False
        try:
            for step in script:
                if isinstance(step, Pause):
                    step.reached.set()
                    await step.event.wait()
                    continue
                if isinstance(step, Call):
                    step.fn(scope)
                    continue
                if isinstance(step, WaitGate):
                    await scope.dispatch_gate.wait()
                    self.log.append("gate")
                    continue
                if isinstance(step, Token):
                    scope.stream.emitted += step.text
                self.log.append(f"yield:{type(step).__name__}:{getattr(step, 'text', '')}")
                yield step
            finished = True
        finally:
            if not finished:
                self.closed.append(scope.turn_index)
                self.log.append("closed")

    def commit_turn(self, scope: TurnScope, delivery: TurnDelivery) -> CommitResult:
        self.commits.append(delivery)
        if self.clock is not None:
            self.commit_times.append(self.clock.now)
        self.log.append(f"commit:{delivery.outcome}")
        return self.commit_result

    def prefetch(self, session_id: str, partial_text: str | None) -> None:
        self.prefetches.append((session_id, partial_text))

    async def handle_capability_only(self, text: str, session_id: str) -> CapabilityOnlyTurn | None:
        """Record the call, run `on_capability` (a capability's effect), return the next result."""
        self.capability_calls.append((text, session_id))
        self.log.append(f"capability:{text}")
        if self.on_capability is not None:
            self.on_capability(text)
        return self.capability_results.pop(0) if self.capability_results else None


class FakeAlarm:
    """The controller's `alarm_ringing`: rings until a test, or a fake capability, stops it."""

    def __init__(self, *, ringing: bool = True) -> None:
        self.ringing = ringing

    def __call__(self) -> bool:
        return self.ringing

    def stop(self) -> None:
        self.ringing = False


# -- Host and arbiter ------------------------------------------------------------------------


class FakeHost:
    """A `ReconfigureHost` that records its answers and applies nothing unless told."""

    def __init__(self, apply_result: VoiceStart | None = None) -> None:
        self.apply_result = apply_result
        self.calls: list[tuple[object, ...]] = []

    def apply(self, request: Reconfigure, seed: IngressSeed) -> VoiceStart | None:
        self.calls.append(("apply", request, seed))
        return self.apply_result

    def refuse(self, request: Reconfigure, reason: str) -> None:
        self.calls.append(("refuse", request, reason))


@dataclass
class Claim:
    claim_id: int
    session_id: str
    rank: WakeRank
    verdict: Callable[[bool, float | None], None]
    answered: bool = False


DENYING_WINNER_SCORE = 0.9
"""The winner's raw score `FakeWakeArbiter.deny` reports when a test names none."""


class FakeWakeArbiter:
    """A `WakeArbitration` that records claims and settles; a test grants or denies by hand.

    With `solo=True` every claim is granted before `claim` returns, as the real arbiter does for
    a lone session. A denial carries a winner's score, as the real arbiter's always does.
    """

    def __init__(self, *, solo: bool = False) -> None:
        self.solo = solo
        self.claims: list[Claim] = []
        self.settles: list[tuple[int, bool]] = []
        self._ids = itertools.count(1)

    def claim(
        self, session_id: str, rank: WakeRank, verdict: Callable[[bool, float | None], None]
    ) -> int:
        entry = Claim(next(self._ids), session_id, rank, verdict)
        self.claims.append(entry)
        if self.solo:
            self.grant(entry.claim_id)
        return entry.claim_id

    def settle(self, claim_id: int, accepted: bool) -> None:
        self.settles.append((claim_id, accepted))

    def _answer(self, claim_id: int, proceed: bool, winner_score: float | None) -> None:
        entry = next(c for c in self.claims if c.claim_id == claim_id)
        if entry.answered:
            raise AssertionError(f"claim {claim_id} already has a verdict")
        entry.answered = True
        entry.verdict(proceed, winner_score)

    def grant(self, claim_id: int) -> None:
        self._answer(claim_id, True, None)

    def deny(self, claim_id: int, winner_score: float = DENYING_WINNER_SCORE) -> None:
        self._answer(claim_id, False, winner_score)


class FakeCallLater:
    """A `call_later` that records each `(delay, callback)` and runs one only when told."""

    def __init__(self) -> None:
        self.timers: list[tuple[float, Callable[[], None]]] = []
        self.fired = 0

    def __call__(self, delay: float, callback: Callable[[], None]) -> object:
        self.timers.append((delay, callback))
        return object()

    def fire(self) -> None:
        """Run the oldest timer not yet run."""
        _, callback = self.timers[self.fired]
        self.fired += 1
        callback()

    @property
    def pending(self) -> int:
        return len(self.timers) - self.fired


# -- The harness ---------------------------------------------------------------------------


def floor_placement() -> TurnPlacement:
    return TurnPlacement(mode="primary", stt="cuda:0", tts="cuda:0", budget=BUDGET_PRESETS["floor"])


def voice_settings(*, playback: bool = True, aec: AecMode = "headset") -> VoiceStartMessage:
    return VoiceStartMessage(
        req=1,
        mode="wake",
        sample_rate=16000,
        frame_ms=20,
        aec=aec,
        playback=playback,
        endpoint="desktop",
        spotter="server",
    )


# Lead large enough that the pacer never waits on the fake clock, unless a test wants it to.
TEST_PLAYBACK = PlaybackConfig(max_lead_ms=600_000)
TEST_SEGMENTER = SegmenterConfig(first_segment_min_chars=1)


@dataclass
class Harness:
    controller: TurnController
    outbound: RecordingOutbound
    responder: FakeResponder
    scene: RecordingScenePort
    ingress: RecordingIngress
    clock: FakeClock
    synth: FakeSynthesizer | None
    arbiter: FakeWakeArbiter
    host: FakeHost
    ids: IdCounter
    policy: KeywordBargeIn
    kwargs: dict[str, object]
    aec: AecMode = "headset"
    verifier: ScriptedWakeVerifier | None = None

    @property
    def lane(self) -> InlineLane:
        lane = self.kwargs["tts_lane"]
        assert isinstance(lane, InlineLane)
        return lane

    def post(self, *messages: object) -> None:
        for message in messages:
            self.controller.post(message)  # type: ignore[arg-type]

    async def settle(self) -> None:
        await settle()

    def start_voice(self, *, playback: bool = True) -> None:
        self.post(
            VoiceStart(
                ingress=self.ingress,  # type: ignore[arg-type]
                policy=self.policy,
                settings=voice_settings(playback=playback, aec=self.aec),
                epoch=self.ingress.epoch,
            )
        )

    async def speak(self, text: str, *, partials: Iterable[str] = (), **ended: int) -> str:
        """Publish an utterance in idle and end it: the turn it opens starts."""
        uid = self.ids()
        self.post(self.ingress.started(uid))
        for partial in partials:
            self.post(self.ingress.partial(uid, partial))
        self.post(self.ingress.ended(uid, text, **ended))
        await self.settle()
        return uid

    def last_turn(self) -> str:
        return self.outbound.turn_ids()[-1]


LIVE_CONTROLLERS: list[TurnController] = []
"""Every controller `make_harness` built, so a fixture can stop them before the loop closes."""


async def close_live_controllers() -> None:
    """Close each controller a test left running, so no mailbox task outlives its loop."""
    while LIVE_CONTROLLERS:
        controller = LIVE_CONTROLLERS.pop()
        task = controller.task
        if task is None or task.done():
            continue
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)


def make_harness(
    *scripts: Iterable[ScriptStep],
    synthesizer: FakeSynthesizer | None | bool = True,
    auto_take: bool = True,
    commit_result: CommitResult | None = None,
    placement: Callable[[], TurnPlacement] | None = None,
    tts_enabled: bool = True,
    scene: object | None = None,
    barge_in: BargeInConfig | None = None,
    playback: PlaybackConfig = TEST_PLAYBACK,
    segmenter: SegmenterConfig = TEST_SEGMENTER,
    turn_config: TurnConfig | None = None,
    start: bool = True,
    wake: WakeConfig | None = None,
    recent_tts_text: Callable[[int], str] | None = None,
    calibration: SpotterCalibration | None = None,
    hit_lag_ms: int = 150,
    solo: bool = True,
    arbiter: WakeArbitration | None = None,
    session_id: str = "session-1",
    id_prefix: str = "u",
    aec: AecMode = "headset",
    alarm_ringing: Callable[[], bool] | None = None,
    verifier: ScriptedWakeVerifier | None | bool = True,
) -> Harness:
    """A controller wired to every fake; `synthesizer=False` builds one with no TTS.

    `wake` configures the gate and the controller alike. With `solo=False` every wake claim waits
    until the test grants or denies it, as it would with two sessions registered. `arbiter`, when
    given, is wired in place of the fake one, so two harnesses can share one real `WakeArbiter`;
    `Harness.arbiter` is then a `FakeWakeArbiter` nothing calls. `verifier` supplies the
    thresholds the gate and the controller read: by default a `ScriptedWakeVerifier`, 0.5 for
    every cutoff; `verifier=None` wires none, so every wake fails closed. Its scores are never
    read here: a test posts each `WakeVerified` itself.
    """
    clock = FakeClock()
    ids = IdCounter(id_prefix)
    outbound = RecordingOutbound(auto_take=auto_take)
    responder = FakeResponder(*scripts, commit_result=commit_result, clock=clock)
    recording_scene = RecordingScenePort()
    ingress = RecordingIngress(ids)
    synth: FakeSynthesizer | None
    if synthesizer is True:
        synth = FakeSynthesizer()
    elif synthesizer is False or synthesizer is None:
        synth = None
    else:
        synth = synthesizer
    fake_arbiter = FakeWakeArbiter(solo=solo)
    host = FakeHost()
    wake_config = wake or WakeConfig()
    barge = barge_in or BargeInConfig(arm_delay_ms=0)
    wake_verifier: ScriptedWakeVerifier | None
    if verifier is True:
        wake_verifier = ScriptedWakeVerifier()
    elif verifier is False or verifier is None:
        wake_verifier = None
    else:
        wake_verifier = verifier
    thresholds = wake_verifier.threshold_at if wake_verifier is not None else None
    built: list[TurnController] = []
    # The session's wiring: the policy's echo guard reads the controller's played text.
    policy = KeywordBargeIn(
        barge,
        aec=aec,
        names=wake_config.names,
        recent_tts_text=lambda ms: built[0].recent_tts_text(ms),
        clock=clock,
    )
    kwargs: dict[str, object] = {
        "session_id": session_id,
        "responder": responder,
        "outbound": outbound,
        "host": host,
        "placement": placement or floor_placement,
        "arbiter": fake_arbiter if arbiter is None else arbiter,
        "gate": default_gate(wake_config, thresholds),
        "thresholds": thresholds,
        "wake": wake_config,
        "calibration": calibration,
        "hit_lag_ms": hit_lag_ms,
        "recent_tts_text": recent_tts_text or (lambda _ms: ""),
        "new_utterance_id": ids,
        "clock": clock,
        "synthesizer": synth,
        "tts_lane": InlineLane() if synth is not None else None,
        "tts_enabled": tts_enabled,
        "scene": scene if scene is not None else recording_scene,
        "segmenter_config": segmenter,
        "playback_config": playback,
        "turn_config": turn_config or TurnConfig(),
        "barge_in_config": barge,
    }
    if alarm_ringing is not None:
        kwargs["alarm_ringing"] = alarm_ringing
    controller = TurnController(**kwargs)  # type: ignore[arg-type]
    built.append(controller)
    if start:
        controller.start()
    LIVE_CONTROLLERS.append(controller)
    return Harness(
        controller=controller,
        outbound=outbound,
        responder=responder,
        scene=recording_scene,
        ingress=ingress,
        clock=clock,
        synth=synth,
        arbiter=fake_arbiter,
        host=host,
        ids=ids,
        policy=policy,
        kwargs=kwargs,
        aec=aec,
        verifier=wake_verifier,
    )
