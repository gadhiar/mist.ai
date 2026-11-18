"""T11b: an exhaustive bounded model check of the per-session turn controller.

The search drives the real `TurnController`, with the real `KeywordBargeIn` policy and
`PlaybackPacer`, on a fake clock, over every sequence of environment actions up to
`MODEL_CHECK_DEPTH` (`DEFAULT_DEPTHS` when unset) from each start state, and checks the invariants after every
step and after a drain at every leaf. It reuses `turn_fakes.py` and wraps its fakes here.

Spec: plan v2 :1866, as v3 :1703, v4 :330, v5 :1751, v6 :1658, v8 :273 and v9 :320 change it.
v5 removed push-to-talk, the hold in `thinking` and its supersede, so the alphabet has no press
and I1 is gone, and M67 holds only at a held turn's release. D8 (the alarm rule) is deferred.

The harness:

- `GatedMailbox` replaces the controller's mailbox. Every post, the turn's own tasks' included,
  queues there in order, and the controller handles the head only when the search takes a
  controller step (`deliver`). A step runs until its handling ends or suspends at the stall.
- `ModelIngress` follows T6's contract on top of `RecordingIngress`: one utterance open at a
  time, opened only while engaged, events numbered by epoch and `event_seq`, a gate decode that
  returns only on its own action, a mute that drops the open utterance ahead of `Mute`, and an
  `open_utterance` on a muted ingress whose drop follows the current step (v8). An ingress the
  session has stopped keeps emitting its open utterance's events (the `late_` actions) until the
  controller calls its `close()` at `VoiceStop`'s handling, as the real ingress does
  (`ingress.py` `close()`; `turn.py` `_close_ingress`): this is how B1(b)'s old-epoch end
  reaches the mailbox behind `VoiceStart`.
- `ModelResponder` streams what the search feeds it, one step per action; `ModelHost` answers
  reconfigures and checks swap safety at each `apply`; `ModelOutbound` sends `control` at once
  and queues `turn` items on the real `TurnLane` until the sender's `take`.

Reductions. Each is sound for the reason given; the search uses no other.

- Positions as ranks. The snapshot holds each frame position it refers to (each utterance's
  `first_pos`, each gate decode's `through_pos`, the ingress floor and the next position) as its
  rank among them, never as its value. The controller and the fake ingress compare positions
  only by order (the record's audio order, `max` with the floor), and the gate's `through_pos`
  is only passed back; so states whose positions rank alike decide alike. Retention never acts
  within the bound (the preconditions check it), so no frame leaves the ring.
- No pure counters and no reported-only values. Drop counts, `stale_events`, the fakes' call
  counts, timings and durations stay out of the snapshot. A counter a rule reads enters only as
  the relation it is read through: `event_seq` as whether the controller has handled every event
  its ingress emitted, the epoch as whether an event's ingress is the adopted one, the tag
  counter as which turn's tag a frame carries. No transition and no checked property reads the
  values left out, so two states that differ only in them have the same successors and verdicts.
- Eager controller steps, only where they meet C1. With a message in the mailbox and the
  controller free, the step and the actions its handling can observe form a persistent set only
  if no action outside the set, taken first, depends on one inside it (the ample-set condition
  C1). Nearly every environment action appends to the mailbox: the ingress actions, the `late_`
  events, `name`, `endpoint_name` and the gate answers post an ingress event; `mute`, `unmute`,
  `voice_stop`, `voice_start`, the wakes, the reconfigures, `interrupt`, `playback_done` and
  `text` post their message; `grant` and `deny` settle a claim whose verdict the controller posts;
  `say`, `hint`, `hint_out`, `handoff` and `done` feed the responder, whose stream task posts a
  `_StreamItem`; `advance` and `tool_end` are counted as appending, which can only widen the
  search. Two appends do not commute (the mailbox then holds their messages in the other order),
  so an append outside the set depends on every append inside it. The step can also disable an
  action outside it (end the turn `interrupt` and `playback_done` need, or push a claim out of
  the two `grant` and `deny` answer), and `voice_start` enables the ingress actions it observes.
  So the step is eager only when every enabled action is the step, `take`, `synth_fail` or
  `text` (`EAGER_SAFE`), and then only `text` waits for it. `take` and `synth_fail` post
  nothing: `TurnLane.take` wakes no task, and the failure is a field the next synthesis reads.
  `text` appends a message the step does not read (the controller reads its mailbox only at
  the head, `turn.py` `_run`), no other action's guard reads what it changes, and no step
  invariant reads the mailbox. So `text` commutes with the step when the step posts nothing,
  and the state it skips has the verdicts of the one before it. A step can post, though: from
  `idle`, after `reconfigure_refused`, `reconfigure_refused`, `name`, `mute` and `voice_stop`,
  the step is eager (no `voice_start` is left), and the step on the name opens on the
  controller's ingress, which it holds until it handles the `VoiceStop`; that ingress is muted,
  so the open posts a muted drop after the step. Which steps post is not listed here (the
  controller's own tasks post `_AudioStarted`, `_SynthFailed`, `_PlayEnded` and `_AckPlayed`
  too): the search sees it, as the step's mailbox no shorter than before it, and then takes
  the actions that waited before the step as well (`Waited`). The parent's visited entry stays
  sound, since the step is the parent's first child and is taken right after it, so every
  action of the parent is on the stack before any other path can reach it. With that, the eager step
  meets C1 at every depth. `text ahead of a step's muted drop` is the witness; an earlier
  filter tried before the step every action the step could observe, and missed each order in
  which an append outside that set came before one inside it, with `reconfigure behind a voice
  restart` as its witness.

Every path is replayed from its start state: no controller state is copied. A state's first child
continues on its parent's world, which is the world its replay would build; the others replay.
Replay checks the invariants of a path's last step only: the search checked every earlier step
as the last step of its own edge, and replay is deterministic. Visited states are kept with the
most depth left they were reached with, and one is searched again only with more.

Reach. `REACH` names each order of v2 :2026-2031 that v5 :1751 keeps, with its start state and
the depth it needs, and a reach run asserts that the search visited it at exactly that depth. An
order is a sequence of handled mailbox messages (`World.handled`), each with the phase before
and after its handling and the facts it saw. An order counts only on a concrete path the search
replayed. The snapshot leaves out history an order reads, so each order's progress (how many of
its steps the path holds) joins the visited key: a path pruned as visited then has a twin with
the same progress and at least as much depth left, whose continuations finish the order if the
pruned path's would. Without it, B1(c)'s text step was pruned: handling the text before or
after v opens reaches one snapshot, and only one of the two holds the step. Two more entries,
`reconfigure behind a voice restart` and `text ahead of a step's muted drop`, are not v2
orders: they show that the eager step meets C1.

Three orders need more than their start's default depth. B8 needs 8 from `thinking_tool`: its
frameless `0x02` is fifth on the lane and each take moves one item. B1(d) needs 4 from
`speaking`: a name, the lost verdict, w's end and v's VAD start. The posting-step witness needs
6 from `idle`. Each one's reach run searches only the actions that the order and its
alternatives use (`B8_ALPHABET`, `B1D_ALPHABET`, `POSTING_ALPHABET`) to its own depth: every
sequence of those actions, with the same reductions and invariants, and no other action. A
search keys its visited states on the progress of its start's full-alphabet orders and of its
own narrowed ones only. The default depths stay as ruled.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import pytest
import pytest_asyncio

from backend.chat.stream_events import HandOff, StateHint, Token
from backend.chat.turn_scope import TurnScope
from backend.realtime.bargein import KeywordBargeIn
from backend.realtime.budget import (
    BargeInConfig,
    EndpointingConfig,
    TurnConfig,
    WakeConfig,
)
from backend.realtime.ingress import (
    FRAME_MS,
    IngressSeed,
    OpenedUtterance,
    SpeechStarted,
    UtteranceDropped,
    UtteranceEnded,
    ring_retention_ms,
)
from backend.realtime.protocol import (
    Frame,
    MsgType,
    TranscriptionMessage,
    VoiceDecisionDegraded,
)
from backend.realtime.speech import NEUTRAL_STATE, SPEAKING_STATES
from backend.realtime.turn import (
    Interrupt,
    Mute,
    PlaybackDone,
    Reconfigure,
    Text,
    TurnController,
    VoiceStart,
    VoiceStop,
    Wake,
    WakeVerdict,
)

from .turn_fakes import (
    TEST_PLAYBACK,
    TEST_SEGMENTER,
    FailingSynthesizer,
    FakeClock,
    FakeHost,
    FakeResponder,
    FakeWakeArbiter,
    IdCounter,
    InlineLane,
    RecordingIngress,
    RecordingOutbound,
    RecordingScenePort,
    default_gate,
    describe,
    floor_placement,
    tool_started,
    voice_settings,
)

pytestmark = pytest.mark.asyncio

DEFAULT_DEPTHS = {"idle": 4, "off": 4}
"""Environment actions per path when `MODEL_CHECK_DEPTH` is unset; 3 from every other start.

The lead's ruling under v2 :2043-2045: depth 8 existed for the push-to-talk orders (v2 :2032),
which v5 removed, and it does not finish (each level multiplies the edges by about 12); depth 3
from every start state, and 4 from `idle` and `off`, fits the 120 s cap.
"""


def depth_for(start: str) -> int:
    """`MODEL_CHECK_DEPTH` when set, for every start state; else the ruled default."""
    value = os.environ.get("MODEL_CHECK_DEPTH")
    return int(value) if value else DEFAULT_DEPTHS.get(start, 3)


REQUIRED_FRAME_MS = 20
REQUIRED_PRE_ROLL_FRAMES = 15
MAX_START_FRAMES = 6
RETENTION_FRAMES = ring_retention_ms(TurnConfig(), WakeConfig()) // FRAME_MS
"""The fake ingress's retention in frames: the real ring's, at the default configuration."""

PLANT: Callable[[World], None] | None = None
"""A fault the teeth tests plant in each world's controller before it starts."""

MAX_UTTERANCES = 3
MAX_STARTS = 2
MAX_TEXTS = 1
ADDRESSED = "mist what's up"
UNADDRESSED = "I missed the bus"
STOP = "stop"
REPLY = "Hi. "
ACK = "One moment. "
CLOSING = ("transcription", "voice_decision:dropped", "voice_decision:wake_lost")
WAKE_ENDS = ("voice_decision:wake_rejected", "voice_decision:wake_lost")
"""The messages that end a name without accepting it; neither may cut or idle a turn."""

EAGER_SAFE = frozenset({"deliver", "take", "synth_fail", "text"})
"""The actions under which the controller step is eager; only `text` then waits for it (C1)."""

OPEN_EVENTS = ("partial", "partial_stop", "candidate", "resumed", "end", "drop")
"""The events of an ingress's open utterance; a stopped ingress emits them as `late_` actions."""


class Violation(AssertionError):  # noqa: N818 - a violation, not an error in the harness
    """An invariant failed; `name` says which."""

    def __init__(self, name: str, detail: str) -> None:
        super().__init__(f"{name}: {detail}")
        self.name = name


async def quiesce() -> None:
    """Run every ready task until the loop has nothing ready twice in a row."""
    loop = asyncio.get_running_loop()
    quiet = 0
    for _ in range(400):
        await asyncio.sleep(0)
        ready = getattr(loop, "_ready", None)
        quiet = quiet + 1 if ready is not None and len(ready) == 0 else 0
        if quiet >= 2:
            return


def named_utterance(item: object) -> str | None:
    uid = getattr(item, "utterance_id", None)
    if uid is None:
        uid = getattr(getattr(item, "detail", None), "utterance_id", None)
    return uid if isinstance(uid, str) else None


def end_then_start(queued: list[object], uid: str) -> bool:
    """Whether `uid`'s `UtteranceEnded` is queued, and another utterance's start behind it."""
    ends = [
        n for n, m in enumerate(queued) if isinstance(m, UtteranceEnded) and m.utterance_id == uid
    ]
    return bool(ends) and any(
        isinstance(m, SpeechStarted) and m.utterance_id != uid for m in queued[ends[0] + 1 :]
    )


def item_turn(item: object) -> str | None:
    if isinstance(item, dict):
        value = item.get("turn_id")
        return value if isinstance(value, str) else None
    value = getattr(item, "turn_id", None)
    return value if isinstance(value, str) else None


# -- The fakes, wrapped ---------------------------------------------------------------------


class GatedMailbox:
    """The controller's mailbox: FIFO, handed out one message per granted step."""

    def __init__(self) -> None:
        self.items: collections.deque[object] = collections.deque()
        self.permits = 0
        self._waiter: asyncio.Future[None] | None = None

    def put_nowait(self, item: object) -> None:
        self.items.append(item)

    def grant(self) -> None:
        self.permits += 1
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(None)

    async def get(self) -> object:
        while not (self.permits and self.items):
            self._waiter = asyncio.get_running_loop().create_future()
            await self._waiter
        self.permits -= 1
        return self.items.popleft()


class ModelIngress(RecordingIngress):
    """T6's contract over the recording levers: one utterance open at a time."""

    def __init__(self, world: World, epoch: int) -> None:
        super().__init__(world.ids, epoch=epoch)
        self.world = world
        self.open_uid: str | None = None
        self.engaged = False
        self.gate_asks: list[tuple[str, int]] = []
        self.expecting = False
        self.closed = False
        self.floor = world.pos
        self.known: set[str] = set()

    def open_utterance(self, mode: str, from_pos: int) -> OpenedUtterance:
        if self.open_uid is not None:
            return OpenedUtterance(self.open_uid, self.first_pos[self.open_uid])
        first = max(from_pos, self.floor)
        uid = self.ids()
        self.calls.append(("open_utterance", mode, from_pos, uid))
        self.first_pos[uid] = first
        self.known.add(uid)
        self.world.utterances += 1
        if self.world.muted:
            # Opened on a muted ingress: minted and dropped at once, after this step (v8).
            self.world.after_step.append(self.dropped(uid, "muted"))
        else:
            self.open_uid = uid
        return OpenedUtterance(uid, first)

    def open_vad(self) -> str:
        uid = self.ids()
        self.first_pos[uid] = max(self.world.pos, self.floor)
        self.known.add(uid)
        self.open_uid = uid
        self.world.utterances += 1
        return uid

    def request_gate_partial(self, utterance_id: str, through_pos: int) -> None:
        super().request_gate_partial(utterance_id, through_pos)
        if utterance_id in self.known:
            self.gate_asks.append((utterance_id, through_pos))

    def set_engaged(self, engaged: bool) -> None:
        super().set_engaged(engaged)
        self.engaged = engaged

    def expect_endpoint_wake(self, *args: object, **kwargs: object) -> None:
        super().expect_endpoint_wake(*args, **kwargs)
        self.expecting = not self.world.muted

    def forget(self, uid: str) -> None:
        self.known.discard(uid)
        self.gate_asks = [ask for ask in self.gate_asks if ask[0] != uid]
        if self.open_uid == uid:
            self.open_uid = None

    def drop(self, utterance_id: str, reason: str) -> None:
        super().drop(utterance_id, reason)
        self.forget(utterance_id)
        return None

    def close(self) -> None:
        super().close()
        self.closed = True
        if self.world.stopped is self:
            self.world.stopped = None
        self.open_uid = None
        self.gate_asks = []
        self.known.clear()
        self.expecting = False
        return None

    def idle_since(self) -> int | None:
        return None if self.open_uid is not None else self.emitted

    def handoff(self) -> IngressSeed:
        self.calls.append(("handoff",))
        return IngressSeed(frames=(), floor=self.world.pos, vad_from=None)


class ModelOutbound(RecordingOutbound):
    """`control` reaches the client at once; `turn` items wait for the sender's `take`."""

    def __init__(self, world: World) -> None:
        super().__init__(auto_take=False)
        self.world = world

    def send_control(self, item: object) -> None:
        self.world.on_send(item)
        super().send_control(item)
        self.client.append(self.control.pop())

    def send_turn(self, turn_id, item, terminal=False, kind=None, token_index=None) -> bool:
        self.world.on_send(item)
        return super().send_turn(turn_id, item, terminal, kind, token_index)


class ModelResponder(FakeResponder):
    """A responder whose stream yields one step per search action."""

    def __init__(self, world: World) -> None:
        super().__init__(clock=world.clock)
        self.world = world
        self.feeds: dict[int, asyncio.Queue[object]] = {}
        self.awaiting: set[int] = set()
        self.open: set[int] = set()
        self.finished: set[int] = set()
        self.tools: list[asyncio.Future[str]] = []

    def open_turn(self, request):
        world = self.world
        if self.open:
            world.fail("M66", f"a turn opened while turn {sorted(self.open)} is open")
        world.check_acceptance("a turn opened")
        scope = super().open_turn(request)
        self.open.add(scope.turn_index)
        return scope

    def prefetch(self, session_id, partial_text):
        self.world.check_acceptance("prefetch")
        super().prefetch(session_id, partial_text)

    async def stream_turn(self, scope: TurnScope):
        index = scope.turn_index
        feed: asyncio.Queue[object] = asyncio.Queue()
        self.feeds[index] = feed
        while True:
            self.awaiting.add(index)
            try:
                step = await feed.get()
            finally:
                self.awaiting.discard(index)
            if step is None:
                self.finished.add(index)
                return
            if step == "tool":
                await scope.dispatch_gate.wait()
                future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
                scope.tool_dispatched = True
                scope.tool_calls.append({"name": "set_timer", "args": {}})
                scope.tool_tasks.append(future)  # type: ignore[arg-type]
                self.tools.append(future)
                step = tool_started()
            if isinstance(step, Token):
                scope.stream.emitted += step.text
            yield step

    def commit_turn(self, scope, delivery):
        world = self.world
        index = scope.turn_index
        if index in world.commits:
            world.fail("M66", f"turn {index} committed twice")
        world.commits.add(index)
        if delivery.outcome == "completed":
            if index not in self.finished:
                world.fail("M66", f"turn {index} completed before the responder completed it")
            turn = world.controller._turn
            if (
                turn is not None
                and turn.scope is scope
                and turn.audio_tag is not None
                and turn.turn_id not in world.done_handled
                and not turn.tts_failed
            ):
                world.fail("M66", f"turn {index} completed before its playback_done")
        self.open.discard(index)
        return super().commit_turn(scope, delivery)


class ModelHost(FakeHost):
    """Answers every reconfigure, building a seeded ingress for a servable one."""

    def __init__(self, world: World) -> None:
        super().__init__()
        self.world = world

    def apply(self, request, seed):
        """Check the swap against the ingress the controller replaces (`_maybe_reconfigure`).

        That is the session's latest ingress unless a `VoiceStart` already queued behind the
        request gave the session a newer one; the environment then keeps driving the newer one,
        and the controller closes this request's ingress when it handles the `VoiceStop` before it.
        """
        world = self.world
        old = world.controller._ingress
        if old is not None and old.open_uid is not None:
            world.fail("swap", f"apply while the old ingress holds {old.open_uid} open")
        if old is not None and any(
            getattr(m, "epoch", None) == old.epoch for m in world.mailbox.items
        ):
            world.fail("swap", "apply while the mailbox holds an event of the old ingress")
        if world.responder.open:
            world.fail("M66", "a reconfigure applied while a turn is open")
        super().apply(request, seed)
        start = world.new_start()
        if world.ingress is old:
            world.ingress = start.ingress  # type: ignore[assignment]
        return start


# -- One world: a controller and its fakes on one path ----------------------------------------


@dataclass
class Step:
    """What the handling of one mailbox message did, for the per-message invariants."""

    message: object
    client_from: int
    scene_from: int
    purges_from: int
    published: frozenset[str]
    phase: str


@dataclass(frozen=True, slots=True)
class Handled:
    """One handled mailbox message, as the reach runs read it."""

    kind: str
    before: str
    after: str
    facts: frozenset[str]


@dataclass
class World:
    """One path's world, built afresh for every replay."""

    clock: FakeClock = field(default_factory=FakeClock)
    ids: IdCounter = field(default_factory=IdCounter)
    mailbox: GatedMailbox = field(default_factory=GatedMailbox)
    pos: int = 0
    env: int = 0
    utterances: int = 0
    starts: int = 0
    texts: int = 0
    muted: bool = False
    voice_on: bool = False
    epochs: int = 0
    after_step: list[object] = field(default_factory=list)
    failures: list[Violation] = field(default_factory=list)
    commits: set[int] = field(default_factory=set)
    done_handled: set[str] = field(default_factory=set)
    done_sent: set[str] = field(default_factory=set)
    reconfigures: list[Reconfigure] = field(default_factory=list)
    rejected_wakes: list[object] = field(default_factory=list)
    lost_wakes: list[object] = field(default_factory=list)
    segments: dict[str, list[tuple[int, str, str]]] = field(default_factory=dict)
    hinted: dict[str, bool] = field(default_factory=dict)
    acks: collections.Counter[str] = field(default_factory=collections.Counter)
    handling: Step | None = None
    labels: list[str] = field(default_factory=list)
    accepted: set[str] = field(default_factory=set)
    handled: list[Handled] = field(default_factory=list)
    stopped: ModelIngress | None = None
    checking: bool = True
    """Whether a step's invariants run; `replay` clears it for a prefix the search has checked."""

    def __post_init__(self) -> None:
        self.outbound = ModelOutbound(self)
        self.responder = ModelResponder(self)
        self.scene = RecordingScenePort()
        self.synth = FailingSynthesizer(fail_call=-1, ms_per_char=10)
        self.arbiter = FakeWakeArbiter(solo=False)
        self.host = ModelHost(self)
        self.policy = KeywordBargeIn(
            BargeInConfig(), aec="headset", recent_tts_text=lambda _ms: "", clock=self.clock
        )
        self.ingress: ModelIngress | None = None
        self.ingresses: list[ModelIngress] = []
        wake = WakeConfig()
        self.controller = TurnController(
            session_id="s",
            responder=self.responder,
            outbound=self.outbound,
            host=self.host,
            placement=floor_placement,
            arbiter=self.arbiter,
            gate=default_gate(wake),
            wake=wake,
            calibration=None,
            hit_lag_ms=150,
            recent_tts_text=lambda _ms: "",
            new_utterance_id=self.ids,
            clock=self.clock,
            synthesizer=self.synth,
            tts_lane=InlineLane(),
            scene=self.scene,
            segmenter_config=TEST_SEGMENTER,
            playback_config=TEST_PLAYBACK,
            turn_config=TurnConfig(),
            barge_in_config=BargeInConfig(),
        )
        self.controller._mailbox = self.mailbox  # type: ignore[assignment]
        if PLANT is not None:
            PLANT(self)
        self._instrument()
        self.controller.start()

    # -- Instrumentation ----------------------------------------------------------------------

    def fail(self, name: str, detail: str) -> None:
        self.failures.append(Violation(name, detail))

    def _instrument(self) -> None:
        c = self.controller
        handlers = c._handlers
        for kind, handler in list(handlers.items()):
            handlers[kind] = self._wrap_handler(handler)
        send_rejected = c._send_wake_rejected

        def wake_rejected(wake, reason):
            # The wakes are kept, not their ids, so no id is reused by a later name.
            if any(seen is wake for seen in self.rejected_wakes):
                self.fail("wake_rejected once", f"a second wake_rejected for one name ({reason})")
            if any(seen is wake for seen in self.lost_wakes):
                self.fail("wake_lost once", f"a wake_rejected for a name that lost ({reason})")
            self.rejected_wakes.append(wake)
            before = len(self.outbound.of("voice_decision:wake_rejected"))
            send_rejected(wake, reason)
            if len(self.outbound.of("voice_decision:wake_rejected")) - before != 1:
                self.fail("wake_rejected once", f"one rejection sent other than one ({reason})")

        c._send_wake_rejected = wake_rejected  # type: ignore[method-assign]
        send_lost = c._send_wake_lost

        def wake_lost(wake, winner_score, utterance_id):
            # One loss per name, and never a loss for a name already rejected, or the reverse.
            if any(seen is wake for seen in (*self.lost_wakes, *self.rejected_wakes)):
                self.fail("wake_lost once", "a second end (wake_lost) for one name")
            self.lost_wakes.append(wake)
            before = len(self.outbound.of("voice_decision:wake_lost"))
            send_lost(wake, winner_score, utterance_id)
            if len(self.outbound.of("voice_decision:wake_lost")) - before != 1:
                self.fail("wake_lost once", "one loss sent other than one wake_lost")

        c._send_wake_lost = wake_lost  # type: ignore[method-assign]
        settle = self.arbiter.settle

        def settle_once(claim_id, accepted):
            if any(cid == claim_id for cid, _ in self.arbiter.settles):
                self.fail("settle once", f"claim {claim_id} settled twice")
            settle(claim_id, accepted)

        self.arbiter.settle = settle_once  # type: ignore[method-assign]
        submit = c._submit_segment

        def submitted(turn, segment):
            before = turn.segments.qsize()
            submit(turn, segment)
            if turn.segments.qsize() > before:
                index, text_segment = turn.segments._queue[-1]  # type: ignore[attr-defined]
                self.segments.setdefault(turn.turn_id, []).append(
                    (index, text_segment.text, text_segment.state)
                )

        c._submit_segment = submitted  # type: ignore[method-assign]
        on_hint = c._on_state_hint

        def hinted(turn, event):
            self.hinted[turn.turn_id] = True
            on_hint(turn, event)

        c._on_state_hint = hinted  # type: ignore[method-assign]
        on_handoff = c._on_handoff

        def handoff(turn, event):
            seen = len(self.segments.get(turn.turn_id, []))
            on_handoff(turn, event)
            if len(self.segments.get(turn.turn_id, [])) > seen:
                self.acks[turn.turn_id] += 1

        c._on_handoff = handoff  # type: ignore[method-assign]
        cancel_held = c._cancel_held

        async def cancelled_held(turn):
            await cancel_held(turn)
            self.responder.open.discard(turn.scope.turn_index)

        c._cancel_held = cancelled_held  # type: ignore[method-assign]
        accept_wake = c._accept_wake

        async def accepted(wake):
            # The gate's verdict is the acceptance; what the accepting step sends may name it.
            self.accepted.add(wake.utterance_id)
            await accept_wake(wake)

        c._accept_wake = accepted  # type: ignore[method-assign]
        synthesize = self.synth.synthesize

        def synth(segment, token):
            turn = c._turn
            if turn is not None:
                recorded = {i: s for i, _, s in self.segments.get(turn.turn_id, [])}
                states = {s for _, t, s in self.segments.get(turn.turn_id, []) if t == segment.text}
                if recorded and states and segment.state not in states:
                    self.fail("v6 segments", f"a segment's state became {segment.state}")
            return synthesize(segment, token)

        self.synth.synthesize = synth  # type: ignore[method-assign]

    def _wrap_handler(self, handler):
        async def handle(message):
            step = Step(
                message=message,
                client_from=len(self.outbound.client),
                scene_from=len(self.scene.calls),
                purges_from=len(self.outbound.purges),
                published=self.published(),
                phase=self.controller._phase,
            )
            self.handling = step
            footprint = self.footprint()
            settled = isinstance(message, WakeVerdict) and any(
                cid == message.claim_id for cid, _ in self.arbiter.settles
            )
            if isinstance(message, PlaybackDone):
                self.done_handled.add(message.turn_id)
            facts = self.facts_before(message)
            queued = list(self.mailbox.items)
            ingress = self.ingress
            levers = len(ingress.calls) if ingress is not None else 0
            try:
                await handler(message)
            finally:
                self.handling = None
            if any(purge.terminal for _, purge in self.outbound.purges[step.purges_from :]):
                facts.add("purged 0x02")
            if isinstance(message, WakeVerdict) and not message.proceed and ingress is not None:
                # B1(d): the lost claim's drop names w, whose end and then v's start are queued.
                lost = [
                    call[1]
                    for call in ingress.calls[levers:]
                    if call[0] == "drop" and call[2] == "wake_lost"
                ]
                if any(end_then_start(queued, w) for w in lost):
                    facts.add("lost w, w's end and v's start queued")
            self.handled.append(
                Handled(
                    type(message).__name__, step.phase, self.controller._phase, frozenset(facts)
                )
            )
            if settled and self.footprint() != footprint:
                self.fail("settle once", f"the verdict for settled claim {message.claim_id} acted")
            if self.checking:
                self.check_message(step)

        return handle

    def footprint(self) -> tuple[int, ...]:
        levers = sum(len(ingress.calls) for ingress in self.ingresses)
        return (
            len(self.outbound.sent),
            len(self.scene.calls),
            levers,
            len(self.arbiter.settles),
            len(self.responder.requests),
        )

    def facts_before(self, message: object) -> set[str]:
        """What a reach order needs from the mailbox and the fakes as `message` is handled."""
        c = self.controller
        facts: set[str] = set()
        queued = list(self.mailbox.items)
        turn = c._turn
        if isinstance(message, Reconfigure) and message.refusal is None:
            facts.add("servable")
            stops = [n for n, m in enumerate(queued) if isinstance(m, VoiceStop)]
            if stops and any(isinstance(m, VoiceStart) for m in queued[stops[0] + 1 :]):
                facts.add("voice restart queued")
        if (
            isinstance(message, Text)
            and self.starts >= MAX_STARTS
            and any(isinstance(m, UtteranceDropped) and m.reason == "muted" for m in queued)
        ):
            # The posting-step witness: a muted drop queued behind the text, with no `voice_start`
            # left to keep the step before the text from being eager.
            facts.add("muted drop queued, starts spent")
        if isinstance(message, Text) and turn is not None and turn.held:
            # B1(c): the held utterance's end is queued behind the text; ingress is on the next.
            held = turn.utterance_id
            ingress = self.ingress
            ended = any(isinstance(m, UtteranceEnded) and m.utterance_id == held for m in queued)
            if ended and ingress is not None and ingress.open_uid not in (None, held):
                facts.add("held end queued, next open")
        if isinstance(message, VoiceStop):
            # B1(b): a `VoiceStart`, then the closing ingress's end of a published utterance.
            starts = [n for n, m in enumerate(queued) if isinstance(m, VoiceStart)]
            published = self.published()
            if starts and any(
                isinstance(m, UtteranceEnded)
                and m.epoch == c._epoch
                and m.utterance_id in published
                for m in queued[starts[0] + 1 :]
            ):
                facts.add("old end behind VoiceStart")
        if (
            isinstance(message, PlaybackDone)
            and turn is not None
            and turn.scope.turn_index not in self.responder.finished
        ):
            facts.add("generation open")
        return facts

    def published(self) -> frozenset[str]:
        return frozenset(
            u.utterance_id for u in self.controller._utterances.values() if u.published
        )

    def on_send(self, item: object) -> None:
        uid = named_utterance(item)
        if uid is None:
            return
        utterance = self.controller._utterances.get(uid)
        minted = all(
            uid not in ingress.known and uid not in ingress.first_pos for ingress in self.ingresses
        )
        if utterance is not None and (utterance.published or uid in self.accepted):
            return
        if utterance is None and (minted or uid in self.ever_published):
            return
        # A lost claim's verdict is its `wake_lost`, which names the candidate it has just
        # closed unpublished (A52); no other message may name it.
        if (
            describe(item) == "voice_decision:wake_lost"
            and utterance is not None
            and utterance.closed
            and not utterance.published
            and any(getattr(wake, "utterance_id", None) == uid for wake in self.lost_wakes)
        ):
            return
        self.fail("acceptance", f"{describe(item)} names {uid} before its acceptance")

    def check_acceptance(self, what: str) -> None:
        step = self.handling
        uid = getattr(step.message, "utterance_id", None) if step is not None else None
        if not isinstance(uid, str):
            return
        utterance = self.controller._utterances.get(uid)
        if utterance is not None and utterance.candidate:
            self.fail("acceptance", f"{what} for candidate {uid} before its acceptance")

    # -- Ingress and sessions -----------------------------------------------------------------

    def new_start(self) -> VoiceStart:
        self.epochs += 1
        ingress = ModelIngress(self, self.epochs)
        self.ingresses.append(ingress)
        return VoiceStart(
            ingress=ingress,  # type: ignore[arg-type]
            policy=self.policy,
            settings=voice_settings(),
            epoch=ingress.epoch,
        )

    def post(self, message: object) -> None:
        self.mailbox.put_nowait(message)

    @property
    def ever_published(self) -> set[str]:
        return self._ever

    _ever: set[str] = field(default_factory=set)

    # -- Actions ------------------------------------------------------------------------------

    def turn(self):
        turn = self.controller._turn
        return None if turn is None or turn.ending else turn

    def stream_waiting(self) -> int | None:
        turn = self.turn()
        if turn is None:
            return None
        index = turn.scope.turn_index
        return index if index in self.responder.awaiting else None

    def tag_terminal_taken(self, turn) -> bool:
        tag = turn.audio_tag
        return any(
            isinstance(item, Frame)
            and item.turn_tag == tag
            and item.msg_type == MsgType.AUDIO_COMPLETE
            for item in self.outbound.client
        )

    def enabled(self, *, search: bool = True) -> list[str]:
        """The actions enabled now, controller step first; eager when the mailbox holds one."""
        task = self.controller.task
        if task is None or task.done():
            return []
        actions: list[str] = []
        can_deliver = self.handling is None and bool(self.mailbox.items)
        if can_deliver:
            actions.append("deliver")
        ingress = self.ingress if self.voice_on else None
        room = self.utterances < MAX_UTTERANCES
        if ingress is not None:
            if ingress.open_uid is None:
                if ingress.engaged and not self.muted and room:
                    actions.append("vad")
            else:
                actions += ["partial", "partial_stop", "candidate", "resumed", "end", "drop"]
            if not self.muted and (room or ingress.open_uid is not None):
                actions.append("name")
            if ingress.expecting and not self.muted:
                actions.append("endpoint_name")
            if ingress.gate_asks:
                actions += ["gate_yes", "gate_no"]
            if room:
                actions.append("wake")
            actions.append("wake_short")
            actions.append("unmute" if self.muted else "mute")
            actions.append("voice_stop")
            if self.starts < MAX_STARTS:
                actions += ["reconfigure", "reconfigure_refused"]
        elif self.starts < MAX_STARTS:
            actions.append("voice_start")
        late = self.stopped is not None and self.stopped.open_uid is not None
        if late:
            actions += [f"late_{event}" for event in OPEN_EVENTS]
        for index, claim in enumerate(self.arbiter.claims[-2:]):
            if not claim.answered:
                number = len(self.arbiter.claims[:-2]) + index
                actions += [f"grant:{number}", f"deny:{number}"]
        if self.texts < MAX_TEXTS:
            actions.append("text")
        turn = self.turn()
        if turn is not None:
            actions.append("interrupt")
            actions.append("advance")
            if self.stream_waiting() is not None:
                actions += ["say", "hint", "hint_out", "handoff", "done"]
            if turn.audio_tag is not None and not turn.tts_failed and self.synth.fail_call < 0:
                actions.append("synth_fail")
            if (
                turn.audio_tag is not None
                and turn.turn_id not in self.done_sent
                and self.tag_terminal_taken(turn)
            ):
                actions.append("playback_done")
        elif self.controller._turn is not None:
            actions.append("advance")  # the stall's grace runs on the clock
        if any(not tool.done() for tool in self.responder.tools):
            actions.append("tool_end")
        if self.outbound.lane_items():
            actions.append("take")
        if search and can_deliver and set(actions) <= EAGER_SAFE:
            return [a for a in actions if a != "text"]
        return actions

    async def act(self, label: str) -> None:
        if label == "deliver":
            self.mailbox.grant()
            await quiesce()
            while self.after_step:
                self.post(self.after_step.pop(0))
        else:
            self.env += 1
            self.pos += 1
            await self._env(label)
            await quiesce()
        self.labels.append(label)
        self._ever |= self.published()
        if self.checking:
            self.check_step(label)

    def _feed(self, step: object) -> None:
        index = self.stream_waiting()
        assert index is not None
        self.responder.feeds[index].put_nowait(step)

    async def _env(self, label: str) -> None:  # noqa: C901 - one branch per action
        ingress = self.ingress
        if label.startswith("late_"):
            # The stopped ingress, still emitting until the controller closes it.
            ingress, label = self.stopped, label.removeprefix("late_")
        name, _, arg = label.partition(":")
        if name == "vad":
            assert ingress is not None
            self.post(ingress.started(ingress.open_vad()))
        elif name in ("partial", "partial_stop"):
            assert ingress is not None and ingress.open_uid is not None
            text = STOP if name == "partial_stop" else ADDRESSED
            ingress.last_text = text  # type: ignore[attr-defined]
            self.post(ingress.partial(ingress.open_uid, text, heard_at_ms=self.clock.now))
        elif name == "candidate":
            assert ingress is not None and ingress.open_uid is not None
            text = getattr(ingress, "last_text", ADDRESSED)
            self.post(ingress.candidate(ingress.open_uid, text, heard_at_ms=self.clock.now))
        elif name == "resumed":
            assert ingress is not None and ingress.open_uid is not None
            self.post(ingress.resumed(ingress.open_uid))
        elif name == "end":
            assert ingress is not None and ingress.open_uid is not None
            uid, ingress.open_uid = ingress.open_uid, None
            text = getattr(ingress, "last_text", ADDRESSED)
            self.post(ingress.ended(uid, text, heard_at_ms=self.clock.now))
        elif name == "drop":
            assert ingress is not None and ingress.open_uid is not None
            uid, ingress.open_uid = ingress.open_uid, None
            ingress.forget(uid)
            self.post(ingress.dropped(uid, "noise"))
        elif name == "name":
            assert ingress is not None
            uid = ingress.open_uid
            start = ingress.first_pos[uid] if uid is not None else self.pos
            self.post(
                ingress.heard(uid, from_pos=start, hit_pos=self.pos, arrival=float(self.clock.now))
            )
        elif name == "endpoint_name":
            assert ingress is not None
            ingress.expecting = False
            self.post(
                ingress.heard(
                    ingress.open_uid,
                    from_pos=self.pos,
                    hit_pos=self.pos,
                    arrival=float(self.clock.now),
                    spotter="endpoint",
                )
            )
        elif name in ("gate_yes", "gate_no"):
            assert ingress is not None
            uid, through = ingress.gate_asks.pop(0)
            text = ADDRESSED if name == "gate_yes" else UNADDRESSED
            ingress.last_text = text  # type: ignore[attr-defined]
            self.post(ingress.partial(uid, text, gate=True, through_pos=through))
        elif name in ("wake", "wake_short"):
            backlog = 1500 if name == "wake" else 0
            self.post(
                Wake(keyword="mist", score=0.8, backlog_ms=backlog, arrival=float(self.clock.now))
            )
        elif name in ("grant", "deny"):
            claim = self.arbiter.claims[int(arg)]
            if name == "grant":
                self.arbiter.grant(claim.claim_id)
            else:
                self.arbiter.deny(claim.claim_id)
        elif name == "text":
            self.texts += 1
            self.post(Text("what's the time"))
        elif name == "interrupt":
            self.post(Interrupt())
        elif name == "voice_start":
            self.starts += 1
            start = self.new_start()
            self.ingress = start.ingress  # type: ignore[assignment]
            self.voice_on = True
            self.post(start)
        elif name == "voice_stop":
            self.voice_on = False
            if ingress is not None:
                ingress.expecting = False
                self.stopped = None if ingress.closed else ingress
            self.post(VoiceStop())
        elif name in ("reconfigure", "reconfigure_refused"):
            self.starts += 1
            request = Reconfigure(
                settings=voice_settings(),
                refusal="unavailable" if name == "reconfigure_refused" else None,
            )
            self.reconfigures.append(request)
            self.post(request)
        elif name == "mute":
            self.muted = True
            if ingress is not None:
                ingress.expecting = False
                if ingress.open_uid is not None:
                    uid, ingress.open_uid = ingress.open_uid, None
                    ingress.forget(uid)
                    self.post(ingress.dropped(uid, "muted"))
            self.post(Mute(active=True))
        elif name == "unmute":
            self.muted = False
            self.post(Mute(active=False))
        elif name == "say":
            self._feed(Token(REPLY))
        elif name == "hint":
            self._feed(StateHint(state="calm"))
        elif name == "hint_out":
            self._feed(StateHint(state="shouty"))
        elif name == "handoff":
            self._feed(HandOff(ack_text=ACK))
        elif name == "done":
            self._feed(None)
        elif name == "tool":
            self._feed("tool")
        elif name == "synth_fail":
            self.synth.fail_call = self.synth.calls
        elif name == "tool_end":
            for tool in self.responder.tools:
                if not tool.done():
                    tool.set_result("ok")
        elif name == "playback_done":
            turn = self.turn()
            assert turn is not None
            self.done_sent.add(turn.turn_id)
            self.post(PlaybackDone(turn_id=turn.turn_id, played_ms=60_000))
        elif name == "advance":
            await self.clock.advance(1000)
        elif name == "take":
            self.outbound.take()
        else:
            raise AssertionError(f"unknown action {label}")

    async def run_eager(self, labels: Iterable[str]) -> None:
        """A start state's prefix: each action, then every controller step it allows."""
        for label in labels:
            await self.act(label)
            while "deliver" in self.enabled(search=False):
                await self.act("deliver")

    async def close(self) -> None:
        for tool in self.responder.tools:
            if not tool.done():
                tool.set_result("ok")
        task = self.controller.task
        if task is not None and not task.done():
            task.cancel()
            for _ in range(5):
                await quiesce()
                if task.done():
                    break
                await self.clock.advance(5000)
        for feed in self.responder.feeds.values():
            feed.put_nowait(None)
        await quiesce()

    # -- Invariants ---------------------------------------------------------------------------

    def raise_failures(self) -> None:
        if self.failures:
            raise self.failures[0]

    def check_message(self, step: Step) -> None:
        """Invariants over what one handled message did."""
        new = self.outbound.client[step.client_from :]
        names = [describe(item) for item in new]
        scene = self.scene.calls[step.scene_from :]
        newly = self.published() - step.published
        for end in (n for n in WAKE_ENDS if n in names):
            if "voice_decision:barge_in" in names or "playback_stop" in names:
                self.fail("rejection cuts nothing", f"{names}")
            if len(self.outbound.purges) > step.purges_from:
                self.fail("rejection cuts nothing", f"a purge in the step of a {end}")
            if step.phase in ("thinking", "speaking"):
                after = names[names.index(end) + 1 :]
                cycles = [n for n in after if n.startswith("state_cycle")]
                if cycles and cycles[0] == "state_cycle:idle":
                    self.fail("rejection restores", f"{step.phase} {end} then {cycles[0]}")
        for call in scene:
            if call[0] == "phase" and call[1] in ("wake", "listening") and not newly:
                self.fail("scene anchors", f"{call[1]} intent with no acceptance in its step")
        message = step.message
        if getattr(message, "reason", None) == "muted" and hasattr(message, "utterance_id"):
            held = self.controller._utterances.get(message.utterance_id)
            if held is not None and not held.closed:
                self.fail("mute", f"{message.utterance_id} still open after its muted drop")
        if isinstance(step.message, VoiceStop):
            c = self.controller
            if c._utterances or c._pending is not None or c._candidate is not None:
                self.fail("L2", "the record or a pending request survived voice_stop")

    def check_step(self, label: str) -> None:  # noqa: C901 - one block per invariant
        """Invariants after every step."""
        c = self.controller
        task = c.task
        if task is not None and task.done():
            error = task.exception() if not task.cancelled() else None
            self.fail("mailbox", f"the mailbox task ended: {error!r}")
        for violation in self.outbound.violations:
            self.fail("purge", violation)
        self.outbound.violations.clear()
        client = self.outbound.client
        # I3: no second closing message, and nothing after the closing one.
        closed: set[str] = set()
        releases: set[str] = set()
        last_pos = -1
        for item in client:
            uid = named_utterance(item)
            name = describe(item)
            if uid is None:
                continue
            if uid in closed:
                self.fail("I3", f"{name} names {uid} after its closing message")
            if name in CLOSING:
                closed.add(uid)
            if name == "voice_decision:release":
                if uid in releases:
                    self.fail("M67", f"a second release for {uid}")
                releases.add(uid)
            if isinstance(item, TranscriptionMessage):
                pos = self.first_pos(uid)
                if pos is None or pos <= last_pos:
                    self.fail("I2", f"transcription of {uid} at {pos} not after {last_pos}")
                last_pos = pos if pos is not None else last_pos
        # M67: in `text` the dispatch gate is set and nothing is kept back.
        turn = c._turn
        if (
            c._phase == "text"
            and turn is not None
            and (not turn.scope.dispatch_gate.is_set() or turn.kept)
        ):
            self.fail("M67", "a turn in text with its gate clear or output kept")
        self._check_degraded()
        # v3: no scene intent names a turn after its `cancel_turn`.
        cancelled: set[object] = set()
        for call in self.scene.calls:
            if call[0] == "cancel_turn":
                cancelled.add(call[1])
            elif (
                call[0] == "phase"
                and call[2] in cancelled
                or call[0] == "tool_view"
                and call[1] in cancelled
            ):
                self.fail("scene cancel", f"{call[:3]} after its turn's cancel_turn")
        # v8: no utterance is open while ingress is muted; an attend cue is never left standing.
        if self.muted and self.ingress is not None and self.ingress.open_uid is not None:
            self.fail("mute", f"ingress holds {self.ingress.open_uid} open while muted")
        if self.handling is None and c._candidate is None:
            cycles = [describe(i) for i in client if describe(i).startswith("state_cycle")]
            if cycles and cycles[-1] == "state_cycle:attend":
                self.fail("attend", "attend left standing with no candidate")
        # v6: segments after the ack, in the vocabulary, neutral first unless hinted.
        for turn_id, segments in self.segments.items():
            for _, _, state in segments:
                if state not in SPEAKING_STATES:
                    self.fail("v6 segments", f"segment state {state} outside the vocabulary")
            if segments and segments[0][2] != NEUTRAL_STATE and not self.hinted.get(turn_id):
                self.fail("v6 segments", "a first segment not neutral with no hint before it")
            if (
                self.acks[turn_id]
                and segments
                and segments[0][1] != ACK.strip()
                and not segments[0][1].startswith("One")
            ):
                self.fail("v6 ack", f"a segment before the ack: {segments[0][1]!r}")
        for turn_id, count in self.acks.items():
            if count > 1:
                self.fail("v6 ack", f"{count} acks spoken in {turn_id}")
        self.raise_failures()

    def first_pos(self, uid: str) -> int | None:
        for ingress in self.ingresses:
            if uid in ingress.first_pos:
                return ingress.first_pos[uid]
        return None

    def _check_degraded(self) -> None:
        """M68: `degraded` order on the client record."""
        client = self.outbound.client
        started: set[str] = set()
        tokens: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
        tags: dict[int, str] = {}
        degraded_at: dict[str, VoiceDecisionDegraded] = {}
        for item in client:
            name = describe(item)
            turn_id = item_turn(item)
            if name == "stream_start" and turn_id is not None:
                started.add(turn_id)
                if item.audio_tag is not None:  # type: ignore[attr-defined]
                    tags[item.audio_tag] = turn_id  # type: ignore[attr-defined]
            elif name == "stream_token" and turn_id is not None:
                tokens[turn_id].append(item["token"])  # type: ignore[index]
            elif isinstance(item, VoiceDecisionDegraded):
                if item.turn_id not in started:
                    self.fail(
                        "M68", f"degraded {item.reason} before stream_start of {item.turn_id}"
                    )
                if item.reason == "tts_failed":
                    detail = item.detail
                    if len(tokens[item.turn_id]) != detail.text_from_token:
                        self.fail("M68", "degraded tts_failed not after exactly its tokens")
                    degraded_at[item.turn_id] = item
            elif isinstance(item, Frame) and item.turn_tag in tags:
                owner = tags[item.turn_tag]
                degraded = degraded_at.get(owner)
                if (
                    item.msg_type == MsgType.AUDIO_CHUNK
                    and degraded is not None
                    and degraded.detail.voiced_text == ""
                ):
                    self.fail("M68", "a 0x01 reached the client after a voiceless degraded")
        for turn_id, degraded in degraded_at.items():
            sent = "".join(
                s.item["token"]  # type: ignore[index]
                for s in self.outbound.sent
                if describe(s.item) == "stream_token" and s.turn_id == turn_id
            )
            if not sent.startswith(degraded.detail.voiced_text or ""):
                self.fail("M68", "voiced_text is not a prefix of the turn's tokens")
            if degraded.detail.voiced_text == "":
                tag = next((t for t, owner in tags.items() if owner == turn_id), None)
                index = client.index(degraded)
                if any(
                    isinstance(i, Frame) and i.turn_tag == tag and i.msg_type == MsgType.AUDIO_CHUNK
                    for i in client
                ):
                    self.fail("M68", "a 0x01 of a voiceless degraded's tag reached the client")
                if any(
                    isinstance(i, Frame)
                    and i.turn_tag == tag
                    and i.msg_type == MsgType.AUDIO_COMPLETE
                    for i in client[:index]
                ):
                    self.fail("M68", "a 0x02 before its voiceless degraded")

    async def drain(self) -> None:
        """End everything in flight, then check the leaf invariants."""
        order = (
            "deliver",
            "tool_end",
            "gate_yes",
            "endpoint_name",
            "end",
            "grant",
            "done",
            "take",
            "playback_done",
            "advance",
        )
        for _ in range(120):
            actions = self.enabled(search=False)
            pick = next((a for kind in order for a in actions if a.partition(":")[0] == kind), None)
            if pick is None:
                break
            await self.act(pick)
        self.check_leaf()

    def check_leaf(self) -> None:
        client = self.outbound.client
        # L3, from the fakes: with no turn open in the responder, nothing is left open.
        if not self.responder.open:
            ingress = self.ingress
            if ingress is not None and ingress.open_uid is not None:
                self.fail("L3", f"ingress still holds {ingress.open_uid} open")
            if any(getattr(m, "epoch", None) is not None for m in self.mailbox.items):
                self.fail("L3", "the mailbox still holds an ingress event")
        # Every published utterance has exactly one closing message.
        named = {uid for uid in map(named_utterance, client) if uid is not None}
        closings = collections.Counter(
            named_utterance(item) for item in client if describe(item) in CLOSING
        )
        for uid in sorted(named | self._ever):
            if closings[uid] != 1:
                self.fail("closing", f"{uid} has {closings[uid]} closing messages")
        # Every Reconfigure has exactly one answer, in posting order.
        answered = [call[1] for call in self.host.calls]
        if [id(r) for r in answered] != [id(r) for r in self.reconfigures]:
            self.fail("reconfigure", f"{len(answered)} answers for {len(self.reconfigures)}")
        # M47: every shown tag has exactly one terminal frame, after every 0x01 of the tag.
        for start in [i for i in client if describe(i) == "stream_start"]:
            tag = start.audio_tag  # type: ignore[attr-defined]
            if tag is None:
                continue
            frames = [i for i in client if isinstance(i, Frame) and i.turn_tag == tag]
            ends = [i for i, f in enumerate(frames) if f.msg_type != MsgType.AUDIO_CHUNK]
            if len(ends) != 1 or ends[0] != len(frames) - 1:
                self.fail("M47", f"tag {tag}: {[describe(f) for f in frames]}")
        # v9: every claim is settled exactly once.
        settles = collections.Counter(cid for cid, _ in self.arbiter.settles)
        if settles != collections.Counter(c.claim_id for c in self.arbiter.claims):
            self.fail("settle once", f"settles {dict(settles)} for {len(self.arbiter.claims)}")
        self.raise_failures()

    # -- The snapshot -------------------------------------------------------------------------

    def snapshot(self) -> tuple[object, ...]:  # noqa: C901 - one entry per part of the state
        c = self.controller
        issued = {uid: n for n, uid in enumerate(self.ids.issued)}

        def u(uid: str | None) -> int | None:
            return None if uid is None else issued.get(uid, -1)

        def t(turn_id: str | None) -> str | None:
            return None if turn_id is None else turn_id.rpartition("-")[2]

        positions = {self.pos}
        for ingress in self.ingresses:
            positions |= set(ingress.first_pos.values()) | {ingress.floor}
            positions |= {through for _, through in ingress.gate_asks}
        ranks = {p: n for n, p in enumerate(sorted(positions | {p + 1 for p in positions}))}
        turn = c._turn
        turn_state: object = None
        if turn is not None:
            turn_state = (
                t(turn.turn_id),
                turn.held,
                turn.ending,
                turn.channel,
                turn.stream_done,
                turn.tag_ended,
                turn.playback_done,
                turn.tts_failed,
                turn.frames_queued,
                turn.handoff_waiting,
                turn.handoff_seen,
                turn.state,
                tuple(turn.token_texts),
                turn.complete_sent,
                turn.stretch_open,
                turn.audio_started,
                u(turn.utterance_id),
                len(turn.kept),
                turn.scope.dispatch_gate.is_set(),
                tuple(turn.open_tools),
                turn.voiced_text,
                turn.text_from_token,
                turn.error,
                turn.audio_tag is None,
                turn.ack_last_index,
                turn.synth_done_through,
                turn.segments.qsize(),
                tuple(task.done() for task in turn.tasks),
                fingerprint(turn.pacer),
                fingerprint(turn.ledger),
                fingerprint(turn.segmenter),
                turn.scope.tool_dispatched,
            )
        record = tuple(
            sorted(
                (
                    u(x.utterance_id),
                    ranks[x.first_pos],
                    x.published,
                    x.closed,
                    x.partial_text,
                    x.keyword,
                    x.idle_ignored,
                    x.candidate,
                    x.followup,
                    x.latest.text if x.latest else None,
                    tuple(type(e).__name__ for e in x.held_ends),
                )
                for x in c._utterances.values()
            )
        )
        wake = c._candidate
        wake_state = None
        if wake is not None:
            wake_state = (
                wake.source,
                wake.waiting_name,
                wake.proceed,
                wake.gate_requested,
                wake.ducked is not None,
                wake.attend_at is not None,
                wake.claim_id,
                u(wake.utterance_id),
                wake.reference_score,
                t(wake.turn_id),
            )
        ingress = c._ingress
        ingresses = tuple(
            (
                i.epoch,
                u(i.open_uid),
                i.engaged,
                i.expecting,
                i.closed,
                ranks[i.floor],
                tuple((u(a), ranks[b]) for a, b in i.gate_asks),
                getattr(i, "last_text", None),
                i.emitted == c._handled_seq if i is ingress else None,
                i is ingress,
                i is self.ingress,
                i is self.stopped,
                tuple(sorted(u(k) for k in i.known)),
            )
            for i in self.ingresses
        )
        mailbox = tuple(self.describe_message(m, u, t) for m in self.mailbox.items)
        after = tuple(self.describe_message(m, u, t) for m in self.after_step)
        lane = tuple((t(owner), describe(item)) for owner, item in self.outbound.lane_items())
        client = self.outbound.client
        history = (
            tuple(sorted(u(x) for x in self._ever)),
            tuple(
                sorted(
                    {u(named_utterance(i)) for i in client if describe(i) in CLOSING} - {None},
                    key=str,
                )
            ),
            tuple(
                sorted(
                    {
                        u(named_utterance(i))
                        for i in client
                        if describe(i) == "voice_decision:release"
                    }
                    - {None},
                    key=str,
                )
            ),
            tuple(t(i.turn_id) for i in client if describe(i) == "stream_start"),  # type: ignore[attr-defined]
            tuple(sorted(str(cl[1]) for cl in self.scene.calls if cl[0] == "cancel_turn")),
            tuple(t(turn_id) for turn_id, _ in self.outbound.purges),
            tuple(sorted(self.commits)),
            tuple(sorted(self.responder.open)),
            tuple(sorted(self.responder.finished)),
            tuple(sorted(self.responder.awaiting)),
            tuple(sorted(t(x) for x in self.done_handled)),
            tuple(sorted(t(x) for x in self.done_sent)),
            tuple((cl.claim_id, cl.answered) for cl in self.arbiter.claims),
            tuple(self.arbiter.settles),
            tuple(sorted(c._claims)),
            len(self.host.calls),
            len(self.reconfigures),
            tuple((t(k), tuple(v)) for k, v in sorted(self.segments.items())),
            tuple(sorted((t(k), v) for k, v in self.hinted.items())),
            tuple(sorted((t(k), v) for k, v in self.acks.items())),
            tuple(t(i.turn_id) for i in client if isinstance(i, VoiceDecisionDegraded)),
            tuple(self.last_cycles()),
            tuple(
                (describe(i), t(item_turn(i)))
                for i in client
                if isinstance(i, Frame) or describe(i) in ("stream_token", "stream_start")
            ),
        )
        return (
            c._phase,
            turn_state,
            c._pending is not None and c._pending.refusal,
            record,
            wake_state,
            c._muted,
            c._engaged,
            c._window_until is not None,
            c._note,
            c._closed,
            ingress is not None,
            c._epoch,
            mailbox,
            after,
            self.mailbox.permits,
            self.handling is not None,
            ingresses,
            lane,
            history,
            self.clock.now,
            fingerprint(self.policy),
            self.utterances,
            self.starts,
            self.texts,
            self.muted,
            self.voice_on,
            self.synth.fail_call >= 0,
            self.synth.calls - max(self.synth.fail_call, 0),
            tuple(tool.done() for tool in self.responder.tools),
            ranks[self.pos],
        )

    def last_cycles(self) -> list[str]:
        cycles = [
            describe(i) for i in self.outbound.client if describe(i).startswith("state_cycle")
        ]
        return cycles[-1:]

    def describe_message(self, message: object, u, t) -> tuple[object, ...]:
        fields: list[object] = [type(message).__name__]
        for key in (
            "epoch",
            "text",
            "reason",
            "gate",
            "spotter",
            "proceed",
            "active",
            "backlog_ms",
            "refusal",
        ):
            if hasattr(message, key):
                fields.append((key, getattr(message, key)))
        if hasattr(message, "utterance_id"):
            fields.append(u(message.utterance_id))  # type: ignore[attr-defined]
        if hasattr(message, "turn_id"):
            fields.append(t(message.turn_id))  # type: ignore[attr-defined]
        if hasattr(message, "claim_id"):
            fields.append(message.claim_id)  # type: ignore[attr-defined]
        event = getattr(message, "event", None)
        if event is not None:
            fields.append(
                (type(event).__name__, getattr(event, "text", None), getattr(event, "state", None))
            )
        return tuple(fields)


def _lane_items(self: RecordingOutbound) -> list[tuple[str, object]]:
    """The `turn` lane's queued items, oldest first."""
    return [(entry.turn_id, entry.item) for entry in self.lane._items]


ModelOutbound.lane_items = _lane_items  # type: ignore[attr-defined]


def fingerprint(obj: object, depth: int = 2) -> object:
    """The primitive part of an object's state, for the snapshot; callables and tasks left out."""
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if depth == 0:
        return type(obj).__name__
    if isinstance(obj, list | tuple | collections.deque):
        return tuple(fingerprint(x, depth - 1) for x in obj)
    if isinstance(obj, set | frozenset):
        return tuple(sorted(repr(fingerprint(x, depth - 1)) for x in obj))
    if isinstance(obj, dict):
        return tuple(sorted((repr(k), repr(fingerprint(v, depth - 1))) for k, v in obj.items()))
    if callable(obj) or isinstance(obj, asyncio.Future | asyncio.Event | asyncio.Queue):
        return None
    names: list[str] = []
    for klass in type(obj).__mro__:
        names += list(getattr(klass, "__slots__", ()))
    state = dict(getattr(obj, "__dict__", {}))
    for name in names:
        if hasattr(obj, name):
            state[name] = getattr(obj, name)
    return tuple(
        (key, fingerprint(value, depth - 1))
        for key, value in sorted(state.items())
        if not callable(value) and not key.startswith("_clock")
    )


# -- Start states ------------------------------------------------------------------------------


START_STATES: dict[str, tuple[str, ...]] = {
    "off": (),
    "idle": ("voice_start",),
    "held": ("voice_start", "name", "grant:0", "gate_yes", "candidate"),
    "thinking_tool": ("voice_start", "name", "grant:0", "gate_yes", "end", "tool"),
    "speaking": ("voice_start", "name", "grant:0", "gate_yes", "end", "say"),
    "text_tool": ("voice_start", "text", "tool"),
}
"""Each start state's prefix, run eagerly. Every one but `off` has voice on, unmuted."""


async def build(start: str, *, checking: bool = True) -> World:
    world = World(checking=checking)
    await world.run_eager(START_STATES[start])
    world.checking = True
    world.env = 0
    world.starts = 0
    world.texts = 0
    world.labels = []
    world.handled = []
    return world


def check_preconditions(depth: int) -> None:
    """The bounds the reductions rely on; each failure names its bound."""
    endpointing = EndpointingConfig()
    if FRAME_MS != REQUIRED_FRAME_MS:
        raise AssertionError(f"precondition: frames are {FRAME_MS} ms, not 20")
    if endpointing.pre_roll_ms // FRAME_MS != REQUIRED_PRE_ROLL_FRAMES:
        raise AssertionError("precondition: VAD pre-roll is not 15 frames")
    largest = max(len(prefix) for prefix in START_STATES.values())
    if largest > MAX_START_FRAMES:
        raise AssertionError(f"precondition: a start state holds {largest} frames, above 6")
    if largest + depth >= REQUIRED_PRE_ROLL_FRAMES:
        raise AssertionError(f"precondition: {largest} start frames plus depth {depth} reach 15")
    if largest + depth >= RETENTION_FRAMES:
        raise AssertionError("precondition: start frames plus depth reach the ingress retention")


# -- The search -------------------------------------------------------------------------------


@dataclass
class Stats:
    states: int = 0
    edges: int = 0
    leaves: int = 0
    seconds: float = 0.0
    reached: dict[str, int] = field(default_factory=dict)
    """Each named order the search visited, with the fewest environment actions it took."""


# -- Reach: the named orders ------------------------------------------------------------------


def handled(
    kind: str, *, before: str | None = None, after: str | None = None, fact: str | None = None
) -> Callable[[Handled], bool]:
    """One handled message of an order: its type, and the phase and fact it must show."""

    def matches(entry: Handled) -> bool:
        return (
            entry.kind == kind
            and (before is None or entry.before == before)
            and (after is None or entry.after == after)
            and (fact is None or fact in entry.facts)
        )

    return matches


@dataclass(frozen=True)
class Reach:
    """A named order: handled messages in this order, from `start`, at `depth` actions."""

    name: str
    start: str
    depth: int
    steps: tuple[Callable[[Handled], bool], ...]
    alphabet: frozenset[str] | None = None
    """None for the full alphabet at the start's default depth; else a narrowed search."""

    def progress(self, world: World) -> int:
        """How many steps the path's handled messages hold in order, others between.

        Matching each step at its earliest entry is never worse than a later one, so this count
        is all a continuation needs to know about the path's past for this order.
        """
        done = 0
        for entry in world.handled:
            if done < len(self.steps) and self.steps[done](entry):
                done += 1
        return done


B8_DEPTH = 8
B8_ALPHABET = frozenset({"synth_fail", "say", "take", "playback_done", "done"})
"""B8's actions and its alternatives': arm the failure, a token to fail on, the sender's takes up
to the frameless `0x02`, the client's `playback_done`, and the responder's completion."""

B1D_DEPTH = 4
B1D_ALPHABET = frozenset({"name", "grant", "deny", "end", "drop", "vad"})
"""B1(d)'s actions and its alternatives': a name that opens w and claims, the claim's verdict
either way, w's end or its drop by ingress, and the VAD start of v."""

POSTING_DEPTH = 6
POSTING_ALPHABET = frozenset({"reconfigure_refused", "name", "mute", "voice_stop", "text"})
"""The posting-step witness's actions: two refused reconfigures to spend the voice starts, a
name, the mute and `voice_stop` that leave the controller a muted ingress, and the text."""

REACH: tuple[Reach, ...] = (
    # v2 :2027-2028, a servable `Reconfigure` while speaking and while a turn is held, which
    # waits for a quiescent point.
    Reach(
        "reconfigure while speaking",
        "speaking",
        1,
        (handled("Reconfigure", before="speaking", after="speaking", fact="servable"),),
    ),
    Reach(
        "reconfigure while held",
        "held",
        1,
        (handled("Reconfigure", before="held", after="held", fact="servable"),),
    ),
    # v2 :1479-1482, B5: the last frames and the `0x02` queued and not taken, then a cut that
    # purges the `0x02` (the stream's end, then an interrupt).
    Reach(
        "B5",
        "speaking",
        2,
        (
            handled("_StreamEnded", before="speaking"),
            handled("Interrupt", before="speaking", fact="purged 0x02"),
        ),
    ),
    # v2 :1655-1656, B1(b): `VoiceStop`, `VoiceStart`, then the old ingress's end of a published
    # utterance, which is dropped as an old-epoch event.
    Reach(
        "B1(b)",
        "held",
        3,
        (
            handled("VoiceStop", before="held", fact="old end behind VoiceStart"),
            handled("VoiceStart"),
            handled("UtteranceEnded"),
        ),
    ),
    # v2 :1657-1659, B1(c): `held` on u, ingress has ended u and opened v, and the mailbox holds
    # the text, then u's end.
    Reach(
        "B1(c)",
        "held",
        3,
        (
            handled("Text", before="held", fact="held end queued, next open"),
            handled("UtteranceEnded", before="text"),
        ),
    ),
    # v2 :1660-1662, B1(d) in its v5 form: v5 has no `WakeLost`; a lost claim drops w with
    # `wake_lost` and sends one `voice_decision wake_lost` (`turn.py` `_lose`). The verdict is
    # handled with w's end and v's start queued behind it. Ingress is dormant in `idle`
    # (v5 :1177), so this runs from `speaking`, and it needs depth 4: name, deny, end, vad.
    Reach(
        "B1(d)",
        "speaking",
        B1D_DEPTH,
        (
            handled("WakeVerdict", before="speaking", fact="lost w, w's end and v's start queued"),
            handled("UtteranceEnded", before="speaking"),
            handled("SpeechStarted", before="speaking"),
        ),
        alphabet=B1D_ALPHABET,
    ),
    # v2 :2029-2030, B9: from `thinking`, a VAD start, then the failure, which moves to `text`.
    Reach(
        "B9",
        "thinking_tool",
        3,
        (
            handled("SpeechStarted", before="thinking"),
            handled("_SynthFailed", before="thinking", after="text"),
        ),
    ),
    # v2 :2030-2031, N13 and B10: from `held`, the failure, then the utterance's end, which
    # releases the turn into `text`.
    Reach(
        "B10/N13",
        "held",
        3,
        (
            handled("_SynthFailed", before="held", after="held"),
            handled("UtteranceEnded", before="held", after="text"),
        ),
    ),
    # Not a v2 order: the witness that the eager step meets C1. A servable `Reconfigure` is handled
    # with `VoiceStop` and then `VoiceStart` queued behind it, the race whose apply exposed the
    # harness's swap check. A reduction that skips `voice_start` while the request waits never
    # reaches it.
    Reach(
        "reconfigure behind a voice restart",
        "idle",
        3,
        (handled("Reconfigure", before="idle", fact="voice restart queued"),),
    ),
    # Not a v2 order: the witness that an eager step which posts still lets the text go first.
    # After two refused reconfigures, a name, a mute and `voice_stop`, the step on the name opens
    # on the controller's muted ingress and posts its drop; the text queued before that step is
    # handled ahead of the drop only if the search takes it before the eager step.
    Reach(
        "text ahead of a step's muted drop",
        "idle",
        POSTING_DEPTH,
        (
            handled("VoiceStop"),
            handled("Text", fact="muted drop queued, starts spent"),
        ),
        alphabet=POSTING_ALPHABET,
    ),
    # v2 :2028-2029, B8: from `thinking`, the failure, then `playback_done` before the
    # responder's completion. Its frameless `0x02` is fifth on the lane, so it needs depth 8.
    Reach(
        "B8",
        "thinking_tool",
        B8_DEPTH,
        (
            handled("_SynthFailed", before="thinking", after="text"),
            handled("PlaybackDone", fact="generation open"),
        ),
        alphabet=B8_ALPHABET,
    ),
)


async def replay(start: str, path: tuple[str, ...]) -> World:
    """Replay `path`, checking the invariants of its last step only.

    The search replays `path[:-1]` (and the start's prefix, at `()`) before it pushes `path`,
    and replay is deterministic, so every earlier step was checked as the last step of its own
    edge; checking it again cannot fail.
    """
    world = await build(start, checking=not path)
    try:
        for index, label in enumerate(path):
            world.checking = index == len(path) - 1
            await world.act(label)
    except Violation as error:
        await world.close()
        raise AssertionError(f"{start} after {list(path)}: {error}") from error
    return world


async def extend(world: World, start: str, path: tuple[str, ...]) -> World:
    """Take `path`'s last action on the world at `path[:-1]`, with its invariants checked."""
    try:
        await world.act(path[-1])
    except Violation as error:
        await world.close()
        raise AssertionError(f"{start} after {list(path)}: {error}") from error
    return world


@dataclass(frozen=True, slots=True)
class Waited:
    """An eager step's parent: its mailbox length, and the actions that waited for the step."""

    mailbox: int
    actions: tuple[str, ...]


async def search(start: str, depth: int, alphabet: frozenset[str] | None = None) -> Stats:
    """Depth-first over every bounded path from `start`; raises on the first violation.

    `alphabet`, when given, narrows the environment actions to those named; controller steps
    are always taken. Each of the start's named orders over the full alphabet, and each over
    `alphabet`, adds its progress to the visited key, so a path pruned as visited has a twin
    with the same progress and at least its depth left.
    """
    check_preconditions(depth)
    stats = Stats()
    orders = [
        order for order in REACH if order.start == start and order.alphabet in (None, alphabet)
    ]
    began = time.perf_counter()
    visited: dict[tuple[object, ...], int] = {}

    def bounded(actions: list[str], left: int) -> list[str]:
        if alphabet is not None:
            actions = [a for a in actions if a == "deliver" or a.partition(":")[0] in alphabet]
        return [a for a in actions if a == "deliver" or left > 0]

    # A path; its parent's world when this is the parent's first child, which continues on the
    # parent's world (the world its replay would build) while the others replay; and, when the
    # parent stepped eagerly, the parent's mailbox length and the actions that waited for the
    # step, which are explored before the step after all if the step posts.
    stack: list[tuple[tuple[str, ...], World | None, Waited | None]] = [((), None, None)]
    while stack:
        path, parent, waited = stack.pop()
        world = await replay(start, path) if parent is None else await extend(parent, start, path)
        stats.edges += 1
        if waited is not None and len(world.mailbox.items) >= waited.mailbox:
            # The step posted (the mailbox lost its head and still did not shrink), so it does
            # not commute with an append: take each waiting action before it as well.
            for label in reversed(waited.actions):
                stack.append(((*path[:-1], label), None, None))
        kept = False
        try:
            progress = tuple(order.progress(world) for order in orders)
            for order, done in zip(orders, progress):
                if done == len(order.steps):
                    stats.reached[order.name] = min(stats.reached.get(order.name, depth), world.env)
            left = depth - world.env
            snap = (world.snapshot(), progress)
            if visited.get(snap, -1) >= left:
                continue
            visited[snap] = left
            stats.states += 1
            actions = bounded(world.enabled(), left)
            waiting = tuple(
                a for a in bounded(world.enabled(search=False), left) if a not in actions
            )
            if not actions:
                stats.leaves += 1
                try:
                    await world.drain()
                except Violation as error:
                    raise AssertionError(
                        f"{start} after {list(path)}, drained {world.labels[len(path):]}: {error}"
                    ) from error
                continue
            for label in reversed(actions[1:]):
                stack.append(((*path, label), None, None))
            step = Waited(len(world.mailbox.items), waiting) if waiting else None
            stack.append(((*path, actions[0]), world, step))
            kept = True
        finally:
            if not kept:
                await world.close()
    stats.seconds = time.perf_counter() - began
    return stats


@pytest_asyncio.fixture(autouse=True)
async def _quiet_loop():
    yield
    await quiesce()


@pytest.mark.parametrize("start", sorted(START_STATES))
async def test_every_bounded_path_keeps_every_invariant(start):
    depth = depth_for(start)
    stats = await searched(start, depth)
    print(
        f"\nmodel-check start={start} depth={depth} states={stats.states} "
        f"edges={stats.edges} leaves={stats.leaves} seconds={stats.seconds:.1f}"
    )
    assert stats.states > 0


async def test_a_depth_past_the_bounds_fails_before_any_state_is_visited():
    with pytest.raises(AssertionError, match="precondition: 6 start frames plus depth 9 reach 15"):
        await search("held", 9)


_SEARCHED: dict[tuple[str, int, frozenset[str] | None], Stats] = {}


async def searched(start: str, depth: int, alphabet: frozenset[str] | None = None) -> Stats:
    """`search` with no fault planted, run once per bound, so the reach runs reuse its result."""
    assert PLANT is None, "a planted search is never shared"
    key = (start, depth, alphabet)
    if key not in _SEARCHED:
        _SEARCHED[key] = await search(start, depth, alphabet)
    return _SEARCHED[key]


@pytest.mark.parametrize("order", REACH, ids=[order.name for order in REACH])
async def test_the_search_visits_each_named_order(order):
    depth = depth_for(order.start) if order.alphabet is None else order.depth
    stats = await searched(order.start, depth, order.alphabet)
    if order.alphabet is not None:
        print(
            f"\nmodel-check start={order.start} depth={depth} narrowed={sorted(order.alphabet)} "
            f"states={stats.states} edges={stats.edges} leaves={stats.leaves} "
            f"seconds={stats.seconds:.1f}"
        )
    assert (
        order.name in stats.reached
    ), f"{order.name} not visited from {order.start} at depth {depth}; it needs {order.depth}"
    assert (
        stats.reached[order.name] == order.depth
    ), f"{order.name} first visited at depth {stats.reached[order.name]}, stated {order.depth}"


# -- Teeth: each plants one fault in the controller, and the search must name its invariant ----


def _handler(world: World, kind: type, replace: Callable[..., object]) -> None:
    world.controller._handlers[kind] = replace  # type: ignore[assignment]


def _twice(world: World, name: str) -> None:
    original = getattr(world.controller, name)

    def both(*args):
        original(*args)
        original(*args)

    setattr(world.controller, name, both)


def _quiescent_without_idle_since(world: World) -> None:
    c = world.controller
    c._quiescent = lambda: c._phase == "idle" and c._turn is None and c._ingress is not None  # type: ignore[method-assign]


def _ignore_purge_result(world: World) -> None:
    class Flipped:
        def __getattr__(self, name):
            return getattr(world.outbound, name)

        def purge_turn(self, turn_id):
            purge = world.outbound.purge_turn(turn_id)
            return dataclasses.replace(purge, terminal=not purge.terminal)

    world.controller._outbound = Flipped()  # type: ignore[assignment]


def _settle_twice(world: World) -> None:
    c = world.controller
    settle = c._settle

    def twice(wake, accepted):
        claim = wake.claim_id
        settle(wake, accepted)
        if claim is not None:
            c._arbiter.settle(claim, False)

    c._settle = twice  # type: ignore[method-assign]


def _name_the_candidate_at_its_end(world: World) -> None:
    c = world.controller
    hold_end = c._hold_end

    async def hold(utterance, event):
        await hold_end(utterance, event)
        c._send_control(TranscriptionMessage(text="x", utterance_id=utterance.utterance_id))

    c._hold_end = hold  # type: ignore[method-assign]


def _ignore_muted_drops(world: World) -> None:
    c = world.controller
    from backend.realtime.ingress import UtteranceDropped

    async def dropped(event):
        if event.reason != "muted":
            await c._on_utterance_dropped(event)
        else:
            c._accept(event)

    _handler(world, UtteranceDropped, dropped)


def _complete_without_playback(world: World) -> None:
    c = world.controller

    async def maybe(turn):
        if not turn.ending and not turn.held and turn.stream_done:
            await c._complete(turn)

    c._maybe_complete = maybe  # type: ignore[method-assign]


def _scene_after_cut(world: World) -> None:
    c = world.controller
    cut = c._cut

    async def cutting(turn, **kwargs):
        await cut(turn, **kwargs)
        c._scene.phase("idle", turn_id=turn.turn_id, at=None)  # type: ignore[arg-type]

    c._cut = cutting  # type: ignore[method-assign]


def _ignore_voice_stop(world: World) -> None:
    async def nothing(message):
        return None

    _handler(world, VoiceStop, nothing)


def _degraded_on_control(world: World) -> None:
    c = world.controller
    send_turn = c._send_turn

    def send(turn, item, **kwargs):
        if isinstance(item, VoiceDecisionDegraded):
            c._send_control(item)
        else:
            send_turn(turn, item, **kwargs)

    c._send_turn = send  # type: ignore[method-assign]


def _no_restore(world: World) -> None:
    world.controller._restore = lambda wake: None  # type: ignore[method-assign]


def _restore_to_idle(world: World) -> None:
    c = world.controller
    from backend.realtime.protocol import StateCycleMessage

    c._restore = lambda wake: c._send_control(StateCycleMessage(state="idle"))  # type: ignore[method-assign]


def _any_hinted_state(world: World) -> None:
    def hint(turn, event):
        turn.state = event.state

    world.controller._on_state_hint = hint  # type: ignore[method-assign]


def _listening_at_every_name(world: World) -> None:
    c = world.controller
    from backend.realtime.ingress import NameHeard
    from backend.realtime.scene import Immediate

    async def heard(event):
        await c._on_name_heard(event)
        c._scene.phase("listening", turn_id=None, at=Immediate())

    _handler(world, NameHeard, heard)


def _purge_at_rejection(world: World) -> None:
    c = world.controller
    reject = c._reject

    def rejecting(wake, reason):
        reject(wake, reason)
        if c._turn is not None:
            c._outbound.purge_turn(c._turn.turn_id)

    c._reject = rejecting  # type: ignore[method-assign]


def _transcription_names_a_ghost(world: World) -> None:
    c = world.controller
    send = c._send_end_of_utterance

    def ghost(turn_id, event, text=None):
        send(turn_id, dataclasses.replace(event, utterance_id="ghost"), text)

    c._send_end_of_utterance = ghost  # type: ignore[method-assign]


def _release_twice(world: World) -> None:
    c = world.controller
    from backend.realtime.protocol import VoiceDecisionRelease

    send = c._send_control

    def sending(item):
        send(item)
        if isinstance(item, VoiceDecisionRelease):
            send(item)

    c._send_control = sending  # type: ignore[method-assign]


def _refusal_unanswered(world: World) -> None:
    c = world.controller

    async def reconfigure(message):
        if message.refusal is None:
            await c._on_reconfigure(message)

    _handler(world, Reconfigure, reconfigure)


def _token_after_cut(world: World) -> None:
    c = world.controller
    cut = c._cut

    async def cutting(turn, **kwargs):
        await cut(turn, **kwargs)
        item = {"type": "stream_token", "turn_id": turn.turn_id, "token": "late"}
        c._outbound.send_turn(turn.turn_id, item, kind="token", token_index=99)

    c._cut = cutting  # type: ignore[method-assign]


def _published_drop_unsent(world: World) -> None:
    c = world.controller
    from backend.realtime.ingress import UtteranceDropped

    async def dropped(event):
        utterance = c._utterances.get(event.utterance_id)
        if utterance is not None and utterance.published:
            c._accept(event)
            utterance.closed = True
            return
        await c._on_utterance_dropped(event)

    _handler(world, UtteranceDropped, dropped)


def _every_handoff_speaks(world: World) -> None:
    """The guard that ignores a second `HandOff` struck out: each one speaks its ack."""
    c = world.controller
    on_handoff = c._on_handoff

    def handoff(turn, event):
        texts = turn.token_texts
        turn.token_texts, turn.handoff_seen = [], False
        on_handoff(turn, event)
        turn.token_texts = texts + turn.token_texts

    c._on_handoff = handoff  # type: ignore[method-assign]


TEETH: list[tuple[str, str, int, Callable[[World], None]]] = [
    ("v6 ack", "held", 2, _every_handoff_speaks),
    ("swap", "idle", 3, _quiescent_without_idle_since),
    ("M47", "speaking", 3, _ignore_purge_result),
    ("settle once", "idle", 2, _settle_twice),
    ("wake_rejected once", "idle", 2, lambda w: _twice(w, "_send_wake_rejected")),
    ("wake_lost once", "idle", 2, lambda w: _twice(w, "_send_wake_lost")),
    ("acceptance", "idle", 2, _name_the_candidate_at_its_end),
    ("mute", "held", 1, _ignore_muted_drops),
    ("I3", "held", 1, lambda w: _twice(w, "_send_end_of_utterance")),
    ("M66", "speaking", 1, _complete_without_playback),
    ("scene cancel", "speaking", 1, _scene_after_cut),
    ("L2", "held", 1, _ignore_voice_stop),
    ("M68", "speaking", 2, _degraded_on_control),
    ("attend", "idle", 2, _no_restore),
    ("rejection restores", "speaking", 2, _restore_to_idle),
    ("v6 segments", "held", 2, _any_hinted_state),
    ("scene anchors", "idle", 1, _listening_at_every_name),
    ("rejection cuts nothing", "speaking", 2, _purge_at_rejection),
    ("I2", "held", 1, _transcription_names_a_ghost),
    ("M67", "held", 1, _release_twice),
    ("reconfigure", "idle", 1, _refusal_unanswered),
    ("purge", "speaking", 1, _token_after_cut),
    ("closing", "held", 1, _published_drop_unsent),
]


@pytest.mark.parametrize(("invariant", "start", "depth", "plant"), TEETH, ids=[t[0] for t in TEETH])
async def test_a_planted_fault_breaks_its_invariant(invariant, start, depth, plant, monkeypatch):
    monkeypatch.setattr(f"{__name__}.PLANT", plant)
    with pytest.raises(AssertionError) as caught:
        await search(start, depth)
    message = str(caught.value)
    assert f": {invariant}: " in message, message
