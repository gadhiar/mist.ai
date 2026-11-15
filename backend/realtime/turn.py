"""The per-session turn controller: one mailbox task owns a voice session's turns.

Every input reaches the controller through `post`: the session's ingress events, the client's
playback reports and controls, and the events the controller's own turn tasks produce. One
mailbox task handles them in order and is the only code that moves a turn through its phases,
so a cut and the event it races with are never handled at the same time.

A turn runs as a small task group the mailbox task owns:

- the producer reads the responder's stream inside `contextlib.aclosing` and posts each event
  to the mailbox, waiting until the mailbox has handled it before it reads the next. So a cut
  handled in the mailbox lands before the next event is consumed, and cancelling the producer
  closes the stream;
- for a `tts` turn, the synthesize task runs each segment through the TTS lane and hands the
  audio to the pacer, and the play task runs the pacer, which queues the tag's frames on the
  `turn` lane through the controller's sink.

A cut cancels the group, waits for it with the wait shielded, purges the turn's queued `turn`
items, stops the pacer with the purge's `terminal` in the same synchronous step, and commits
what the client received. The commit is always the last step of a turn, and a turn is
committed at most once, whichever path ends it: completion, a cut, `close`, or the mailbox
task's own cancellation at server shutdown.

A speculative turn starts held at an `EndpointCandidate`: it generates and synthesizes, but
everything it would send to the client or the screen waits in its buffer, and its scope's
dispatch gate stays clear, so it dispatches no tool. The utterance's end releases it: `release`,
then the gate set, then the buffer flushed in order. Speech that resumes, or anything else that
ends it while held, cancels it without trace: nothing was sent, so nothing is purged, sent or
committed.

MIST wakes by name. A `NameHeard` from the server's spotter, or an endpoint's `wake` and the
`NameHeard` its backlog yields, opens an unpublished wake candidate: nothing names it on the
wire, in a log line or to the responder until the addressee gate accepts it. The controller
checks her own recent speech first, sends `attend`, opens the candidate, claims it from the
cross-session arbiter, checks the reference score once the claim may proceed, and only then asks
ingress for the one gate decode. Acceptance publishes the candidate in one step and replays any
end of turn held meanwhile; rejection drops it and sends one `wake_rejected`. While a turn is in
flight an accepted name interrupts it with `barge_in` reason `wake`. After a name-opened turn a
follow-up window, when configured, keeps ingress engaged so the next utterance needs no name.

T11a builds the core: the mailbox, the channel rule, completed turns, the commit rule, the
interruption note, cancellation, prefetch, `turn_metrics`, the scene anchors not tied to a
wake, `StateHint` and `HandOff`. T11c adds the record a cut keeps, the move to text on
`tts_failed`, completion after both ends and the tool grace's orphan closing. T11d adds the
`VoiceStart`, `voice_stop` and `Reconfigure` rows, the held turn, and the keyword row: a
transcript's keyword interrupt publishes its utterance, whose end then starts a turn from the
text after the keyword or closes with `dropped keyword`. T11e adds the addressee rows, the
handling of arbitration verdicts and the follow-up window, then `outranks` and `WakeArbiter`, the
one cross-session arbiter every voice session's controller shares.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar

from backend.chat.stream_events import Complete, HandOff, StateHint, StreamEvent, Token, WSEvent
from backend.chat.turn_scope import CommitResult, TurnDelivery, TurnRequest, TurnScope

from .addressee import AddresseeGate
from .bargein import BargeInPolicy, BargeInVerdict
from .budget import (
    BUDGET_FIELDS,
    BargeInConfig,
    PlaybackConfig,
    SegmenterConfig,
    TurnConfig,
    VoiceLatencyBudget,
    WakeConfig,
)
from .errors import SpeechEngineError
from .ingress import (
    FRAME_MS,
    EndpointCandidate,
    IngressOverload,
    IngressSeed,
    Lane,
    NameHeard,
    OpenedUtterance,
    SpeechResumed,
    SpeechStarted,
    TranscriptPartial,
    UtteranceDropped,
    UtteranceEnded,
)
from .lanes import LaneJobCancelled
from .playback import LanePurge, PlaybackClock, PlaybackLedger, PlaybackPacer
from .protocol import (
    MAX_TURN_TAG,
    BargeInDetail,
    BargeInIgnoredDetail,
    DegradedDetail,
    DroppedDetail,
    EndpointDetail,
    EnginePlacement,
    Frame,
    LatencyBudget,
    MsgType,
    OverloadDetail,
    PlaybackDuckMessage,
    PowerLevel,
    Spotter,
    StateCycleMessage,
    StreamCancelledMessage,
    StreamChannel,
    StreamStartMessage,
    TranscriptionMessage,
    TranscriptPartialMessage,
    TurnLaneStats,
    TurnMetricsMessage,
    TurnOrigin,
    TurnPromptStats,
    UtteranceDetail,
    VadStatusMessage,
    VoiceDecisionBargeIn,
    VoiceDecisionBargeInIgnored,
    VoiceDecisionDegraded,
    VoiceDecisionDropped,
    VoiceDecisionEndpoint,
    VoiceDecisionOverload,
    VoiceDecisionRelease,
    VoiceDecisionWakeRejected,
    VoiceStartMessage,
    WakeRejectedDetail,
    next_turn_tag,
)
from .scene import (
    AtSegment,
    AtStreamEnd,
    Immediate,
    NullScenePort,
    SceneAnchor,
    ScenePhase,
    ScenePort,
    ScreenView,
)
from .segmenter import SpeechSegment, SpeechSegmenter
from .speech import (
    NEUTRAL_STATE,
    SPEAKING_STATES,
    CancelToken,
    Phase,
    SpeechSynthesizer,
    SpotterCalibration,
    SynthChunk,
    TextSegment,
    TranscriptUpdate,
    false_accepts_per_hour,
    positive_quantile,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

CancelReason = Literal[
    "user_interrupt", "barge_in", "superseded", "new_input", "disconnect", "voice_stop"
]
Outcome = Literal["completed", "interrupted", "superseded", "error"]

STAGE_TIMESTAMPS: tuple[tuple[str, str], ...] = (
    ("end_of_turn_ms", "eot_decided"),
    ("stt_final_ms", "stt_final"),
    ("context_build_ms", "context_built"),
    ("llm_first_token_ms", "first_token"),
    ("first_segment_ms", "first_segment"),
    ("tts_first_chunk_ms", "first_tts_chunk"),
    ("transport_playback_ms", "first_played"),
)
"""Each budget stage and the `timings` key that ends it, in order (v2 Design changes)."""

DUCK_DB = 12.0
"""How far a name hit lowers her playback until its verdict (v1's 12 dB duck)."""

WakeRejectReason = Literal[
    "self_speech", "self_media", "not_addressed", "lost_arbitration", "no_backlog"
]


# -- Ports ----------------------------------------------------------------------------------


class TurnResponder(Protocol):
    """The stream-and-commit turn API: `ConversationHandler` and the degraded responder."""

    def open_turn(self, request: TurnRequest) -> TurnScope:
        """Create the turn's scope; nothing durable is written."""
        ...

    def stream_turn(self, scope: TurnScope) -> AsyncIterator[StreamEvent]:
        """Produce the reply as stream events."""
        ...

    def commit_turn(self, scope: TurnScope, delivery: TurnDelivery) -> CommitResult:
        """Record what the client received."""
        ...

    def prefetch(self, session_id: str, partial_text: str | None) -> None:
        """Hint that a turn is forming; returns at once."""
        ...


class Outbound(Protocol):
    """What the controller sends through; the session implements it over its lanes.

    The controller never touches a WebSocket. `send_turn` takes a `turn` lane item with the
    `TurnLane.put` keywords and serves as the pacer's turn sink; `purge_turn` removes the turn's
    queued items and reports what it removed.
    """

    def send_control(self, item: object) -> None:
        """Queue an item on the `control` lane."""
        ...

    def send_turn(
        self,
        turn_id: str,
        item: object,
        terminal: bool = False,
        kind: Literal["stream_start", "token"] | None = None,
        token_index: int | None = None,
    ) -> bool:
        """Queue an item of a turn on the `turn` lane."""
        ...

    def purge_turn(self, turn_id: str) -> LanePurge:
        """Remove every queued `turn` item of the turn."""
        ...


class ControllerIngress(Protocol):
    """The levers the controller pulls on the session's `AudioIngress`."""

    def open_utterance(self, mode: Literal["wake"], from_pos: int) -> OpenedUtterance:
        """Open or continue a wake utterance from `from_pos`."""
        ...

    def request_gate_partial(self, utterance_id: str, through_pos: int) -> None:
        """Decode the utterance through `through_pos` out of cadence."""
        ...

    def set_engaged(self, engaged: bool) -> None:
        """Listen for speech without the name, or stop."""
        ...

    def expect_endpoint_wake(
        self, *, keyword: str, score: float, backlog_ms: int, arrival: float
    ) -> None:
        """Treat the next `backlog_ms` of frames as an endpoint's backlog ending at its name."""
        ...

    def drop(self, utterance_id: str, reason: Literal["text_input", "wake_lost", "not_addressed"]):
        """Close the utterance without a turn and return its drop."""
        ...

    def close(self) -> UtteranceDropped | None:
        """Stop ingress and return the open utterance's drop, if any."""
        ...

    def idle_since(self) -> int | None:
        """The position since which no utterance is open, None while one is."""
        ...

    def handoff(self) -> IngressSeed:
        """What the next ingress needs across a reconfigure."""
        ...


@dataclass(frozen=True, slots=True)
class TurnPlacement:
    """Where the engines run for a turn and the budget it is judged against, read at its open."""

    mode: Literal["primary", "yield"]
    stt: str
    tts: str
    budget: VoiceLatencyBudget


@dataclass(frozen=True, slots=True)
class WakeRank:
    """How strongly one candidate heard the name, for cross-session arbitration."""

    arrival: float
    false_accepts_per_hour: float | None
    positive_quantile: float | None
    margin: float


@dataclass(frozen=True, slots=True)
class WakeVerdict:
    """The mailbox event a wake arbitration verdict posts: may the claim proceed."""

    claim_id: int
    proceed: bool


class WakeArbitration(Protocol):
    """Cross-session wake arbitration; one instance is shared by every voice session."""

    def claim(self, session_id: str, rank: WakeRank, verdict: Callable[[bool], None]) -> int:
        """Register a candidate; `verdict` posts its answer into the owner's mailbox."""
        ...

    def settle(self, claim_id: int, accepted: bool) -> None:
        """Report the gate's verdict for a winner, or withdraw a waiting claim."""
        ...


def _calibrated(rank: WakeRank) -> bool:
    return rank.false_accepts_per_hour is not None and rank.positive_quantile is not None


def _rank_key(rank: WakeRank) -> tuple[int, float, float, float]:
    """The sort key `outranks` compares; the lower key is the stronger hit."""
    if _calibrated(rank):
        assert rank.false_accepts_per_hour is not None  # noqa: S101 - `_calibrated` checked it
        assert rank.positive_quantile is not None  # noqa: S101 - `_calibrated` checked it
        return (0, rank.false_accepts_per_hour, -rank.positive_quantile, rank.arrival)
    return (1, -rank.margin, 0.0, rank.arrival)


def outranks(a: WakeRank, b: WakeRank) -> bool:
    """Whether `a` heard the name more surely than `b` (v5 :336-351).

    The two spotters' raw scores come from different models, so a calibrated rank is compared on
    the shared scales: the lower false accepts per hour, then the higher positive quantile. Any
    calibrated rank outranks an uncalibrated one, and two uncalibrated ranks compare by margin.
    Arrival breaks a tie last, because it favours the transport with the shorter path, not the
    better hit. A rank missing either scale counts as uncalibrated.
    """
    return _rank_key(a) < _rank_key(b)


@dataclass(slots=True, eq=False)
class _Claim:
    """One candidate's entry in the arbiter, from its claim to its settle or its verdict."""

    claim_id: int
    session_id: str
    rank: WakeRank
    verdict: Callable[[bool], None]


class WakeArbiter:
    """Cross-session wake arbitration: of the names several sessions heard at once, one proceeds.

    One per registry (T12); every voice session's controller holds the same instance. Claims made
    within `window_ms` of the first compete, and when the window closes the best-ranked one is
    told it may proceed. The rest wait on that winner's gate verdict: they lose when it is
    accepted, and the next-ranked proceeds when it is rejected (v5 :352-360, v9 :132-151). A lone
    session with nothing pending never waits.

    `verdict` callbacks only post into their controller's mailbox, and every call runs on the
    event loop's thread, so the arbiter takes no lock. Its state is updated before any callback
    runs, so a callback that settles at once sees a consistent arbiter.

    Args:
        window_ms: How long the first claim waits for others (`wake_arbitration_ms`).
        sessions: The number of voice sessions registered now.
        call_later: Runs a callback after a delay in seconds; the running loop's `call_later`.
    """

    def __init__(
        self,
        *,
        window_ms: int,
        sessions: Callable[[], int],
        call_later: Callable[[float, Callable[[], None]], object],
    ) -> None:
        self._window_ms = window_ms
        self._sessions = sessions
        self._call_later = call_later
        self._next_id = 0
        self._waiting: dict[int, _Claim] = {}
        self._winner: _Claim | None = None
        self._window_open = False

    def claim(self, session_id: str, rank: WakeRank, verdict: Callable[[bool], None]) -> int:
        """Register a candidate and return its claim id.

        With one session and no claim pending, `verdict(True)` runs before this returns and no
        timer is set. Otherwise the claim joins the open window, or waits on an unsettled
        winner, or opens a window.
        """
        self._next_id += 1
        entry = _Claim(self._next_id, session_id, rank, verdict)
        if self._sessions() <= 1 and not self._waiting and self._winner is None:
            self._winner = entry
            verdict(True)
            return entry.claim_id
        self._waiting[entry.claim_id] = entry
        if not self._window_open and self._winner is None:
            self._window_open = True
            self._call_later(self._window_ms / 1000, self._close_window)
        return entry.claim_id

    def settle(self, claim_id: int, accepted: bool) -> None:
        """Take the gate's verdict for the winner, or withdraw a claim.

        Accepting the winner sends every other pending claim `verdict(False)`; rejecting it gives
        the next-ranked claim `verdict(True)`. `accepted=False` on a waiting claim withdraws it
        with no verdict to anyone, and on a claim the arbiter no longer holds (one that lost, or
        is already settled) does nothing.

        Raises:
            ValueError: `accepted` is True for a claim that is not the unsettled winner.
        """
        winner = self._winner
        if winner is not None and winner.claim_id == claim_id:
            self._winner = None
            if accepted:
                losers = list(self._waiting.values())
                self._waiting.clear()
                for loser in losers:
                    loser.verdict(False)
            else:
                self._grant_next()
            return
        if accepted:
            raise ValueError(f"wake claim {claim_id} is not the winner and cannot be accepted")
        self._waiting.pop(claim_id, None)

    def _close_window(self) -> None:
        self._window_open = False
        self._grant_next()

    def _grant_next(self) -> None:
        """Give the best-ranked waiting claim `verdict(True)`, unless a window or winner stands."""
        if self._window_open or self._winner is not None or not self._waiting:
            return
        best: _Claim | None = None
        for entry in self._waiting.values():
            if best is None or outranks(entry.rank, best.rank):
                best = entry
        assert best is not None  # noqa: S101 - `_waiting` is not empty
        del self._waiting[best.claim_id]
        self._winner = best
        best.verdict(True)


# -- Messages -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Text:
    """A typed message: it becomes a typed turn."""

    text: str


@dataclass(frozen=True, slots=True)
class Interrupt:
    """The UI's stop."""


@dataclass(frozen=True, slots=True)
class VoiceStart:
    """Voice turned on: adopt this ingress, its epoch, the barge-in policy and the settings."""

    ingress: ControllerIngress
    policy: BargeInPolicy
    settings: VoiceStartMessage
    epoch: int


@dataclass(frozen=True, slots=True)
class VoiceStop:
    """Voice turned off."""


@dataclass(frozen=True, slots=True)
class Reconfigure:
    """A `voice_start` received while voice is active; `refusal` is why it cannot be served."""

    settings: VoiceStartMessage
    refusal: str | None


@dataclass(frozen=True, slots=True)
class Mute:
    """The client's mute state changed; the session has already told ingress."""

    active: bool


@dataclass(frozen=True, slots=True)
class Wake:
    """An endpoint's `wake` message, stamped with its arrival on the session clock."""

    keyword: str
    score: float
    backlog_ms: int
    arrival: float


@dataclass(frozen=True, slots=True)
class BargeInReport:
    """An endpoint's spotter heard a barge-in keyword during the named turn."""

    turn_id: str
    keyword: str


@dataclass(frozen=True, slots=True)
class PlaybackProgress:
    """How much of the named turn's audio the client has played."""

    turn_id: str
    played_ms: int


@dataclass(frozen=True, slots=True)
class PlaybackDone:
    """The client played all of the named turn's audio."""

    turn_id: str
    played_ms: int


@dataclass(frozen=True, slots=True)
class Close:
    """The connection closed: end the turn as a disconnect and stop the mailbox."""


IngressMessage = (
    SpeechStarted
    | SpeechResumed
    | TranscriptPartial
    | EndpointCandidate
    | UtteranceEnded
    | UtteranceDropped
    | IngressOverload
    | NameHeard
)

ControllerMessage = (
    IngressMessage
    | Text
    | Interrupt
    | VoiceStart
    | VoiceStop
    | Reconfigure
    | Mute
    | Wake
    | WakeVerdict
    | BargeInReport
    | PlaybackProgress
    | PlaybackDone
    | Close
)


class ReconfigureHost(Protocol):
    """The session's side of a reconfigure, applied only at a quiescent point (T11d)."""

    def apply(self, request: Reconfigure, seed: IngressSeed) -> VoiceStart | None:
        """Build the seeded ingress and policy, switch routing and answer; None when refused."""
        ...

    def refuse(self, request: Reconfigure, reason: str) -> None:
        """Answer the request refused."""
        ...


# Events the turn's own tasks post. Each names its turn; one for a turn that is no longer the
# current one is dropped without counting, since it is the controller's own.


@dataclass(frozen=True, slots=True)
class _StreamItem:
    turn_id: str
    event: StreamEvent
    ack: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _StreamEnded:
    turn_id: str
    failed: bool


@dataclass(frozen=True, slots=True)
class _AudioStarted:
    turn_id: str
    at_ms: int


@dataclass(frozen=True, slots=True)
class _PlayEnded:
    turn_id: str


@dataclass(frozen=True, slots=True)
class _SynthFailed:
    turn_id: str
    error: SpeechEngineError


@dataclass(frozen=True, slots=True)
class _AckPlayed:
    turn_id: str


_TurnEvent = _StreamItem | _StreamEnded | _AudioStarted | _PlayEnded | _SynthFailed | _AckPlayed


@dataclass(frozen=True, slots=True)
class _WindowLapsed:
    """The follow-up window `seq` has run its length."""

    seq: int


# -- State --------------------------------------------------------------------------------


@dataclass(slots=True)
class _Utterance:
    """One utterance in the record (v2 M39).

    `keyword` is the keyword a transcript verdict interrupted a turn with: the barge-in
    published the utterance, and its turn text is then what follows that keyword (v2, v5).
    `idle_ignored` notes the one `barge_in_ignored idle` it got.

    A `candidate` is unpublished and waits for a verdict: a name's (`_Wake`) or, in a
    follow-up window (`followup`), its first transcript's. Until then its latest partial is kept
    in `latest`, and its `EndpointCandidate` and `UtteranceEnded` in `held_ends`, the latest of
    each in arrival order (v5 :1176). `wake_timings` are the name's, for the turn it opens.
    """

    utterance_id: str
    first_pos: int
    published: bool = False
    closed: bool = False
    partial_text: str = ""
    partial_ms: int = 0
    keyword: str | None = None
    idle_ignored: bool = False
    candidate: bool = False
    followup: bool = False
    latest: TranscriptPartial | None = None
    held_ends: list[EndpointCandidate | UtteranceEnded] = field(default_factory=list)
    wake_timings: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True, eq=False)
class _Wake:
    """One name, from its hit (or an endpoint's `wake`) to the verdict on its candidate.

    `arrival_ms` is the hit's arrival, or the `wake` message's; `name_end_ms` is when the name
    ended by estimate, from which `wake_screen_ms` runs. `turn_id` is the turn in flight at the
    hit, which a `wake_rejected` names. `utterance_id` is None while an endpoint wake waits for
    the `NameHeard` its backlog yields (`waiting_name`). `proceed` records a winning verdict;
    `claim_id` stays in the controller's claims until the claim is settled.
    """

    source: Spotter
    score: float
    arrival_ms: int
    name_end_ms: int
    turn_id: str | None
    reference_score: float | None = None
    attend_at: int | None = None
    ducked: str | None = None
    claim_id: int | None = None
    proceed: bool = False
    waiting_name: bool = False
    utterance_id: str | None = None
    hit_pos: int = 0
    gate_requested: bool = False


@dataclass(frozen=True, slots=True)
class _Kept:
    """One output a held turn keeps back until its release; `frame` marks a kept `0x01`.

    `send` queues the output as it would have been queued had the turn not been held. A `0x01`
    is marked so that a synthesis failure while held can discard it (v2 M65).
    """

    send: Callable[[], None]
    frame: Frame | None = None


@dataclass(slots=True, eq=False)
class _Turn:
    """One turn's state, owned by the mailbox task.

    A held turn (`held`) answers `utterance_id`, whose end is not yet confirmed; until its
    release every output it makes waits in `kept`, in the order it was made.
    """

    turn_id: str
    origin: TurnOrigin
    channel: StreamChannel
    scope: TurnScope
    placement: TurnPlacement
    power_level: PowerLevel
    opened_ms: int
    base_ms: int
    marks: dict[str, int]
    text: str = ""
    held: bool = False
    utterance_id: str | None = None
    kept: list[_Kept] = field(default_factory=list)
    scene: ScenePort = field(default_factory=NullScenePort)
    audio_tag: int | None = None
    segmenter: SpeechSegmenter | None = None
    ledger: PlaybackLedger | None = None
    pacer: PlaybackPacer | None = None
    token: CancelToken = field(default_factory=CancelToken)
    segments: asyncio.Queue[tuple[int, TextSegment] | None] = field(default_factory=asyncio.Queue)
    tasks: list[asyncio.Task[None]] = field(default_factory=list)
    token_texts: list[str] = field(default_factory=list)
    state: str = NEUTRAL_STATE
    state_warned: bool = False
    requested_states: int = 0
    voiced_states: set[int] = field(default_factory=set)
    stream_done: bool = False
    complete: Complete | None = None
    complete_sent: bool = False
    tag_ended: bool = False
    playback_done: bool = False
    frames_queued: bool = False
    voiced_end: int = 0
    tts_failed: bool = False
    voiced_text: str = ""
    text_from_token: int = 0
    disconnected: bool = False
    open_tools: dict[str, tuple[str, int]] = field(default_factory=dict)
    tool: bool = False
    error: bool = False
    error_sent: bool = False
    handoff_seen: bool = False
    handoff_waiting: bool = False
    handoff_ack_ms: int | None = None
    wake_timings: dict[str, int] = field(default_factory=dict)
    ack_last_index: int | None = None
    synth_done_through: int = -1
    stretch_open: bool = False
    audio_started: bool = False
    scene_ended: bool = False
    stale_events: int = 0
    ending: bool = False
    committed: bool = False
    commit_result: CommitResult | None = None
    outcome: Outcome = "completed"

    def mark(self, key: str, at_ms: int) -> None:
        """Record a timestamp the first time its stage is reached."""
        self.marks.setdefault(key, at_ms)


# -- Metrics ----------------------------------------------------------------------------


def judge_over_budget(
    timings: Mapping[str, int], budget: VoiceLatencyBudget, *, tool: bool
) -> list[str]:
    """The budget rows a turn's timings exceed (v2 Design changes, `turn_metrics.timings`).

    Each stage is its timestamp less the nearest earlier stage timestamp present, floored at 0;
    a stage whose timestamp is absent is not judged. `total_p50_ms` is judged against
    `first_played`, `tool_call_ms` only for a tool turn, and `barge_in_reaction_ms` only when a
    keyword cut the turn, which is when both of its timestamps are present.
    """
    over: list[str] = []
    previous = 0
    for stage, key in STAGE_TIMESTAMPS:
        if key not in timings:
            continue
        if max(timings[key] - previous, 0) > getattr(budget, stage):
            over.append(stage)
        previous = timings[key]
    if "first_played" in timings and timings["first_played"] > budget.total_p50_ms:
        over.append("total_p50_ms")
    if (
        tool
        and "tool_call_parsed" in timings
        and "first_token" in timings
        and timings["tool_call_parsed"] - timings["first_token"] > budget.tool_call_ms
    ):
        over.append("tool_call_ms")
    if "keyword_heard" in timings and "playback_stopped" in timings:
        reaction = timings["playback_stopped"] - timings["keyword_heard"]
        if reaction > budget.barge_in_reaction_ms:
            over.append("barge_in_reaction_ms")
    return over


def _prompt_stats(stats: object | None) -> TurnPromptStats:
    """The wire form of the scope's prompt statistics; zeros before the prompt was built.

    The scope holds `backend.chat.context_budget.PromptStats`, read by attribute so the
    realtime package does not import the knowledge layer behind it.
    """
    if stats is None:
        return TurnPromptStats(
            prefix_tokens_est=0,
            history_tokens_est=0,
            delta_tokens_est=0,
            prompt_tokens=None,
            cached_tokens=None,
            prefix_changed=False,
            history_compacted=False,
        )
    return TurnPromptStats(
        prefix_tokens_est=stats.prefix_tokens_est,  # type: ignore[attr-defined]
        history_tokens_est=stats.history_tokens_est,  # type: ignore[attr-defined]
        delta_tokens_est=stats.delta_tokens_est,  # type: ignore[attr-defined]
        prompt_tokens=stats.prompt_tokens,  # type: ignore[attr-defined]
        cached_tokens=stats.cached_tokens,  # type: ignore[attr-defined]
        prefix_changed=stats.prefix_changed,  # type: ignore[attr-defined]
        history_compacted=stats.history_compacted != "none",  # type: ignore[attr-defined]
    )


def _final_update(event: UtteranceEnded) -> TranscriptUpdate:
    """The end's final transcript as the barge-in policy reads it."""
    return TranscriptUpdate(
        event.text,
        len(event.text),
        True,
        event.duration_ms,
        event.no_speech_prob,
        event.avg_logprob,
    )


# The policy's word rule (`bargein.py`): a run of letters or digits, joined across an apostrophe.
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_LEADING_SEPARATORS = re.compile(r"^[\s,.;:!?\-–—…]+")


def _after_keyword(text: str, keyword: str) -> str:
    """The text after the first occurrence of `keyword` in a final transcript, as a request.

    The keyword was matched in an earlier partial; the final's own words are searched again,
    since a final transcript may punctuate or case them differently. A final that no longer holds
    the keyword is taken whole: what the user said is not discarded.
    """
    parts = keyword.split()
    matches = list(_WORD.finditer(text))
    words = [match.group().lower() for match in matches]
    for start in range(len(words) - len(parts) + 1):
        if words[start : start + len(parts)] == parts:
            end = matches[start + len(parts) - 1].end()
            return _LEADING_SEPARATORS.sub("", text[end:]).strip()
    return text.strip()


def _ended_detail(event: UtteranceEnded) -> DroppedDetail:
    """The `dropped` detail of an utterance closed at its end: its final transcript."""
    return DroppedDetail(
        utterance_id=event.utterance_id,
        duration_ms=event.duration_ms,
        text=event.text,
        no_speech_prob=event.no_speech_prob,
        avg_logprob=event.avg_logprob,
    )


def _closing_detail(utterance: _Utterance, drop: UtteranceDropped | None) -> DroppedDetail:
    """The detail of a `dropped` the controller sends for an utterance it closes itself (v2).

    It is the drop the lever returned when that drop names the utterance; otherwise it comes
    from the record, the last transcript handled and its length, since ingress may already have
    ended an utterance whose end is still queued.
    """
    if drop is not None and drop.utterance_id == utterance.utterance_id:
        return DroppedDetail(
            utterance_id=utterance.utterance_id,
            duration_ms=drop.duration_ms,
            text=drop.text,
            no_speech_prob=drop.no_speech_prob,
            avg_logprob=drop.avg_logprob,
        )
    return DroppedDetail(
        utterance_id=utterance.utterance_id,
        duration_ms=utterance.partial_ms,
        text=utterance.partial_text,
        no_speech_prob=None,
        avg_logprob=None,
    )


def _wire_budget(budget: VoiceLatencyBudget) -> LatencyBudget:
    rows = {name: getattr(budget, name) for name in BUDGET_FIELDS}
    return LatencyBudget(preset=budget.preset, **rows)


# -- The turn sink ------------------------------------------------------------------------


def _is_audio(item: object) -> bool:
    return isinstance(item, Frame) and item.msg_type == MsgType.AUDIO_CHUNK


class _TurnSink:
    """The pacer's turn sink: forwards to `Outbound.send_turn` and notes a stretch's first audio.

    Before the first `0x01` of a stretch of audio it queues `state_cycle speak`, so the client
    sees MIST speak no later than the frame, and posts `_AudioStarted` so the mailbox moves the
    turn to `speaking`. While the turn is held it keeps each frame back and returns False, so
    the pacer leaves it out of the ledger until the release queues it (v2 N26).
    """

    def __init__(self, controller: TurnController, turn: _Turn) -> None:
        self._controller = controller
        self._turn = turn

    def put(
        self,
        turn_id: str,
        item: object,
        terminal: bool = False,
        kind: Literal["stream_start", "token"] | None = None,
        token_index: int | None = None,
    ) -> bool:
        turn = self._turn
        controller = self._controller
        if turn.held:
            frame = item if isinstance(item, Frame) and _is_audio(item) else None
            turn.kept.append(_Kept(lambda: controller._flush_frame(turn, item, terminal), frame))
            return False
        return controller._queue_frame(turn, item, terminal, kind, token_index)


class _TurnScene:
    """The scene port as one turn reaches it, and that turn's pacer listener.

    A held turn has not reached the screen either, so its intents and its pacer's hooks wait in
    its buffer with its other output and reach the port at the release, in the order made.
    """

    def __init__(self, scene: ScenePort, turn: _Turn) -> None:
        self._scene = scene
        self._turn = turn

    def _run(self, call: Callable[[], None]) -> None:
        if self._turn.held:
            self._turn.kept.append(_Kept(call))
        else:
            call()

    def phase(self, phase: ScenePhase, *, turn_id: str | None, at: SceneAnchor) -> None:
        self._run(lambda: self._scene.phase(phase, turn_id=turn_id, at=at))

    def tool_view(
        self,
        turn_id: str,
        call_id: str,
        view: ScreenView,
        data: Mapping[str, object] | None,
        *,
        at: SceneAnchor,
    ) -> None:
        self._run(lambda: self._scene.tool_view(turn_id, call_id, view, data, at=at))

    def cancel_turn(self, turn_id: str) -> None:
        self._run(lambda: self._scene.cancel_turn(turn_id))

    def turn_ended(self, turn_id: str) -> None:
        self._run(lambda: self._scene.turn_ended(turn_id))

    def segment_starting(self, audio_tag: int, segment_index: int, first_sample: int) -> None:
        self._run(lambda: self._scene.segment_starting(audio_tag, segment_index, first_sample))

    def stream_ending(self, audio_tag: int, last_sample: int) -> None:
        self._run(lambda: self._scene.stream_ending(audio_tag, last_sample))


# -- The controller -------------------------------------------------------------------------


class TurnController:
    """Owns one session's turns: every input is posted, and one mailbox task handles it.

    Args:
        session_id: The session the turns belong to.
        responder: Opens, streams and commits turns.
        outbound: Where messages and frames go.
        host: The session's side of a reconfigure (T11d).
        placement: Read once when a turn opens; the turn's engines and budget come from it.
        arbiter: Cross-session wake arbitration (T11e).
        gate: The addressee gate (T11e).
        wake: Wake by name settings (T11e).
        calibration: The spotter's calibration, None without one (T11e).
        hit_lag_ms: The spotter's median lag from the name's end to its hit (T11e).
        recent_tts_text: Every session's text played within the given milliseconds (T11e).
        new_utterance_id: The session's utterance id source, shared with ingress.
        clock: The session clock.
        synthesizer: The TTS engine; None means every turn uses the `text` channel.
        tts_lane: The lane synthesis runs on; given exactly when `synthesizer` is.
        tts_enabled: The deployment's TTS switch.
        scene: The scene port; `NullScenePort` when None.
        segmenter_config: How reply text is cut into TTS segments.
        playback_config: Downlink pacing.
        turn_config: The teardown's tool grace.
        barge_in_config: Keywords and the alarm stop words (T11d).
        alarm_ringing: Whether an alarm rings now (T11d).
        power_level: The power level a turn's metrics report, read when it opens.

    Raises:
        ValueError: `synthesizer` is given without `tts_lane`, or the reverse.
    """

    def __init__(
        self,
        *,
        session_id: str,
        responder: TurnResponder,
        outbound: Outbound,
        host: ReconfigureHost,
        placement: Callable[[], TurnPlacement],
        arbiter: WakeArbitration,
        gate: AddresseeGate,
        wake: WakeConfig,
        calibration: SpotterCalibration | None,
        hit_lag_ms: int,
        recent_tts_text: Callable[[int], str],
        new_utterance_id: Callable[[], str],
        clock: PlaybackClock,
        synthesizer: SpeechSynthesizer | None = None,
        tts_lane: Lane | None = None,
        tts_enabled: bool = True,
        scene: ScenePort | None = None,
        segmenter_config: SegmenterConfig | None = None,
        playback_config: PlaybackConfig | None = None,
        turn_config: TurnConfig | None = None,
        barge_in_config: BargeInConfig | None = None,
        alarm_ringing: Callable[[], bool] | None = None,
        power_level: Callable[[], PowerLevel] | None = None,
    ) -> None:
        if (synthesizer is None) != (tts_lane is None):
            raise ValueError("synthesizer and tts_lane are given together or not at all")
        self._session_id = session_id
        self._responder = responder
        self._outbound = outbound
        self._host = host
        self._placement = placement
        self._arbiter = arbiter
        self._gate = gate
        self._wake = wake
        self._calibration = calibration
        self._hit_lag_ms = hit_lag_ms
        self._recent_tts_text = recent_tts_text
        self._new_utterance_id = new_utterance_id
        self._clock = clock
        self._synthesizer = synthesizer
        self._tts_lane = tts_lane
        self._tts_enabled = tts_enabled
        self._scene: ScenePort = scene if scene is not None else NullScenePort()
        self._segmenter_config = segmenter_config or SegmenterConfig()
        self._playback_config = playback_config or PlaybackConfig()
        self._turn_config = turn_config or TurnConfig()
        self._barge_in_config = barge_in_config or BargeInConfig()
        self._alarm_ringing = alarm_ringing or (lambda: False)
        self._power_level: Callable[[], PowerLevel] = power_level or (lambda: "full")

        self._mailbox: asyncio.Queue[object] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._phase: Phase = "idle"
        self._turn: _Turn | None = None
        self._note: Literal["mid_reply", "before_reply"] | None = None
        self._tag = MAX_TURN_TAG
        self._turn_seq = 0
        self._ingress: ControllerIngress | None = None
        self._epoch: int | None = None
        self._policy: BargeInPolicy | None = None
        self._settings: VoiceStartMessage | None = None
        self._utterances: dict[str, _Utterance] = {}
        self._handled_seq = 0
        self._pending: Reconfigure | None = None
        self._stale = 0
        self._closed = False
        self._cancel_pending = False
        self._candidate: _Wake | None = None
        self._claims: dict[int, _Wake] = {}
        self._muted = False
        self._engaged = False
        self._window_until: int | None = None
        self._window_seq = 0
        self._window_task: asyncio.Task[None] | None = None
        self._silent_closes = 0
        self._handlers: dict[type, Callable[[object], Awaitable[None]]] = {
            SpeechStarted: self._on_speech_started,
            SpeechResumed: self._on_speech_resumed,
            TranscriptPartial: self._on_partial,
            EndpointCandidate: self._on_candidate,
            UtteranceEnded: self._on_utterance_ended,
            UtteranceDropped: self._on_utterance_dropped,
            IngressOverload: self._on_overload,
            NameHeard: self._on_name_heard,
            Text: self._on_text,
            Interrupt: self._on_interrupt,
            VoiceStart: self._on_voice_start,
            VoiceStop: self._on_voice_stop,
            Reconfigure: self._on_reconfigure,
            Mute: self._on_mute,
            Wake: self._on_wake,
            WakeVerdict: self._on_wake_verdict,
            BargeInReport: self._on_barge_in_report,
            PlaybackProgress: self._on_playback_progress,
            PlaybackDone: self._on_playback_done,
            Close: self._on_close,
            _StreamItem: self._on_stream_item,
            _StreamEnded: self._on_stream_ended,
            _AudioStarted: self._on_audio_started,
            _PlayEnded: self._on_play_ended,
            _SynthFailed: self._on_synth_failed,
            _AckPlayed: self._on_ack_played,
            _WindowLapsed: self._on_window_lapsed,
        }  # type: ignore[dict-item]

    # -- Public surface -------------------------------------------------------------------

    @property
    def phase(self) -> Phase:
        """The current phase, as the barge-in policy reads it."""
        return self._phase

    @property
    def task(self) -> asyncio.Task[None] | None:
        """The mailbox task, once started."""
        return self._task

    @property
    def scene(self) -> ScenePort:
        """The scene port; the session hands it to nothing else, the pacers get it from here."""
        return self._scene

    @property
    def stale_events(self) -> int:
        """Events dropped because they named another epoch, utterance, turn or settled claim."""
        return self._stale

    @property
    def silent_closes(self) -> int:
        """Unnamed utterances that ended unpublished, closed with no message (v5 :421-426)."""
        return self._silent_closes

    def start(self) -> None:
        """Start the mailbox task on the running loop; a second call does nothing."""
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(
                self._run(), name=f"turn-controller-{self._session_id}"
            )

    def post(self, message: ControllerMessage) -> None:
        """Queue one input for the mailbox task; never blocks."""
        self._mailbox.put_nowait(message)

    async def close(self) -> None:
        """Post `Close` and wait until the mailbox task has committed the turn and stopped.

        A caller cancelled while it waits cancels the mailbox task too, which still tears the
        turn down and commits it before it re-raises.
        """
        self.post(Close())
        if self._task is not None:
            await self._task

    # -- The mailbox --------------------------------------------------------------------

    async def _run(self) -> None:
        try:
            while not self._closed:
                message = await self._mailbox.get()
                await self._handle(message)
                if self._cancel_pending:
                    raise asyncio.CancelledError
                self._sync_engaged()
                self._maybe_reconfigure()
        except asyncio.CancelledError:
            self._closed = True
            self._close_window()
            self._drop_candidates(None)
            turn = self._turn
            if turn is not None and not turn.ending:
                await self._cut(turn, reason="disconnect", notify=False)
            raise

    async def _handle(self, message: object) -> None:
        handler = self._handlers.get(type(message))
        if handler is None:
            logger.warning("turn controller %s: unknown message dropped", self._session_id)
            return
        await handler(message)

    def _post_internal(self, event: _TurnEvent) -> None:
        self._mailbox.put_nowait(event)

    async def _uncancellable(self, awaitable: Awaitable[_T]) -> _T:
        """Await `awaitable` to its end even if the mailbox task is cancelled meanwhile.

        A cancellation that arrives is remembered and raised once the current step is done, so
        a teardown wait and the commit after it always finish.
        """
        inner = asyncio.ensure_future(awaitable)
        while True:
            try:
                return await asyncio.shield(inner)
            except asyncio.CancelledError:
                if inner.cancelled():
                    raise
                self._cancel_pending = True

    def _count_stale(self) -> None:
        self._stale += 1
        if self._turn is not None:
            self._turn.stale_events += 1

    def _current(self, turn_id: str) -> _Turn | None:
        turn = self._turn
        if turn is None or turn.turn_id != turn_id or turn.ending:
            return None
        return turn

    # -- Sending --------------------------------------------------------------------------

    def _send_control(self, item: object) -> None:
        self._outbound.send_control(item)

    def _keep_or_send(self, turn: _Turn, send: Callable[[], None]) -> None:
        """Send now, or keep it in a held turn's buffer until the release."""
        if turn.held:
            turn.kept.append(_Kept(send))
        else:
            send()

    def _send_turn_control(self, turn: _Turn, item: object) -> None:
        """A `control` item that names the turn, so a held turn keeps it back."""
        self._keep_or_send(turn, lambda: self._outbound.send_control(item))

    def _send_turn(
        self,
        turn: _Turn,
        item: object,
        *,
        kind: Literal["stream_start", "token"] | None = None,
        token_index: int | None = None,
    ) -> None:
        self._keep_or_send(
            turn,
            lambda: self._outbound.send_turn(
                turn.turn_id, item, kind=kind, token_index=token_index
            ),
        )

    def _send_token(self, turn: _Turn, text: str) -> None:
        index = len(turn.token_texts)
        turn.token_texts.append(text)
        self._send_turn(
            turn,
            {"type": "stream_token", "turn_id": turn.turn_id, "token": text},
            kind="token",
            token_index=index,
        )

    def _queue_frame(
        self,
        turn: _Turn,
        item: object,
        terminal: bool = False,
        kind: Literal["stream_start", "token"] | None = None,
        token_index: int | None = None,
    ) -> bool:
        """Queue a pacer frame, with `state_cycle speak` before a stretch's first `0x01`."""
        if _is_audio(item):
            now = self._clock.now_ms()
            turn.frames_queued = True
            if not turn.stretch_open:
                turn.stretch_open = True
                self._outbound.send_turn(
                    turn.turn_id, StateCycleMessage(state="speak", turn_id=turn.turn_id)
                )
                self._post_internal(_AudioStarted(turn.turn_id, now))
            turn.mark("first_frame_sent", now)
            if turn.handoff_seen and turn.handoff_ack_ms is None:
                turn.handoff_ack_ms = now - turn.opened_ms
        return self._outbound.send_turn(
            turn.turn_id, item, terminal=terminal, kind=kind, token_index=token_index
        )

    def _flush_frame(self, turn: _Turn, item: object, terminal: bool) -> None:
        """Queue a frame a held turn kept back; a `0x01` enters the ledger only now (N26)."""
        self._queue_frame(turn, item, terminal)
        if _is_audio(item) and turn.pacer is not None:
            turn.pacer.record_flushed(item)

    def _send_degraded(self, turn: _Turn, reason: str, error: str | None = None) -> None:
        self._send_turn(
            turn,
            VoiceDecisionDegraded(
                turn_id=turn.turn_id,
                reason=reason,  # type: ignore[arg-type]
                detail=DegradedDetail(
                    engine=None, error=error, voiced_text=None, text_from_token=None
                ),
            ),
        )

    # -- Ingress events ------------------------------------------------------------------

    def _accept(self, event: IngressMessage) -> bool:
        """Whether the event is from the adopted ingress; any other is dropped and counted."""
        if self._ingress is None or event.epoch != self._epoch:
            self._count_stale()
            return False
        # Every event of the adopted ingress counts as handled, one later dropped as naming a
        # closed utterance included, so the reconfigure's quiescent check can trust it (M59).
        self._handled_seq = event.event_seq
        return True

    def _utterance(self, utterance_id: str, first_pos: int) -> _Utterance | None:
        """The utterance an event names, opened in the record if new; None once it closed."""
        utterance = self._utterances.get(utterance_id)
        if utterance is None:
            utterance = _Utterance(utterance_id, first_pos)
            self._utterances[utterance_id] = utterance
        elif utterance.closed:
            self._count_stale()
            return None
        return utterance

    def _publish(self, utterance: _Utterance) -> None:
        """Show the utterance: the start events and the `listening` intent (v3 D3)."""
        utterance.published = True
        self._send_control(VadStatusMessage(status="speech_started"))
        self._send_control(StateCycleMessage(state="listen"))
        self._scene.phase("listening", turn_id=None, at=Immediate())
        self._responder.prefetch(self._session_id, None)

    async def _on_speech_started(self, event: SpeechStarted) -> None:
        """Speech that opened an engaged utterance, which no name opened (v5 :1177).

        In `thinking` and `speaking` it stays unpublished: only the keyword policy judges it,
        and in `text` it is ignored. In `idle` it is the follow-up window's candidate while the
        window is open. Ingress opens no utterance while dormant, so the `idle` cell is otherwise
        unreachable; it keeps the earlier rows' publishing at speech start.
        """
        if not self._accept(event):
            return
        utterance = self._utterance(event.utterance_id, event.first_pos)
        if utterance is None or utterance.published or utterance.candidate:
            return
        if self._phase != "idle":
            return
        if self._turn is None and self._window_until is not None:
            utterance.candidate = True
            utterance.followup = True
            return
        self._publish(utterance)

    async def _on_speech_resumed(self, event: SpeechResumed) -> None:
        """The utterance continued: a turn held on it is cancelled without trace (v1).

        A candidate's held `EndpointCandidate` no longer stands: its end is not near.
        """
        if not self._accept(event):
            return
        utterance = self._utterance(event.utterance_id, event.first_pos)
        if utterance is None:
            return
        if utterance.candidate:
            utterance.held_ends = [
                e for e in utterance.held_ends if not isinstance(e, EndpointCandidate)
            ]
        turn = self._held_turn(event.utterance_id)
        if turn is not None:
            await self._cancel_held(turn)

    def _held_turn(self, utterance_id: str) -> _Turn | None:
        """The turn held on the utterance, if one is."""
        turn = self._turn
        if turn is None or not turn.held or turn.ending or turn.utterance_id != utterance_id:
            return None
        return turn

    async def _on_partial(self, event: TranscriptPartial) -> None:
        if not self._accept(event):
            return
        utterance = self._utterance(event.utterance_id, event.first_pos)
        if utterance is None:
            return
        utterance.partial_text = event.text
        utterance.partial_ms = max(event.audio_end_pos - event.first_pos + 1, 0) * self._frame_ms
        if utterance.candidate:
            # Unpublished: the gate reads it, not the keyword policy, and nothing is sent.
            utterance.latest = event
            wake = self._candidate_of(utterance.utterance_id)
            if wake is not None:
                if event.gate and wake.gate_requested:
                    await self._decide(wake, event.text)
            elif utterance.followup:
                await self._judge_followup(utterance)
            return
        if utterance.published:
            self._send_control(
                TranscriptPartialMessage(
                    utterance_id=event.utterance_id,
                    text=event.text,
                    stable_chars=event.stable_chars,
                )
            )
            self._responder.prefetch(self._session_id, event.text)
        if self._policy is None:
            return
        update = TranscriptUpdate(event.text, event.stable_chars, False, 0)
        if self._phase == "speaking":
            verdict = self._policy.evaluate(update, self._clock.now_ms(), self._phase)
            if verdict is not None and verdict.action == "interrupt":
                await self._barge_in(verdict, event.utterance_id, event.heard_at_ms)
        elif self._phase == "idle" and utterance.published and utterance.keyword is None:
            verdict = self._policy.evaluate(update, self._clock.now_ms(), self._phase)
            self._send_idle_ignored(utterance, verdict)

    def _send_idle_ignored(self, utterance: _Utterance, verdict: BargeInVerdict | None) -> None:
        """The `idle` cell of the keyword verdict row: `barge_in_ignored idle`, once (v1).

        The policy judges every partial and the end alike, so a keyword early in the utterance
        would give the same verdict again at each; the client is told once per utterance.
        """
        if verdict is None or verdict.action != "ignored" or utterance.idle_ignored:
            return
        utterance.idle_ignored = True
        self._send_ignored(None, verdict, utterance.utterance_id)

    def _send_ignored(
        self, turn_id: str | None, verdict: BargeInVerdict, utterance_id: str | None
    ) -> None:
        self._send_control(
            VoiceDecisionBargeInIgnored(
                turn_id=turn_id,
                reason=verdict.reason,  # type: ignore[arg-type]
                detail=BargeInIgnoredDetail(
                    keyword=verdict.keyword, text=verdict.text, utterance_id=utterance_id
                ),
            )
        )

    async def _on_candidate(self, event: EndpointCandidate) -> None:
        """Prefetch, and in `idle` start the turn held on a published utterance (speculative).

        Ingress emits a candidate only with speculative start on. The candidate's text is the
        last transcript the record handled for the utterance. In every other phase the cell is
        ignored: the utterance's end decides.
        """
        if not self._accept(event):
            return
        utterance = self._utterance(event.utterance_id, event.first_pos)
        if utterance is None:
            return
        if utterance.candidate:
            await self._hold_end(utterance, event)
            return
        await self._candidate_step(utterance, event)

    async def _candidate_step(self, utterance: _Utterance, event: EndpointCandidate) -> None:
        utterance.partial_text = event.text
        if not utterance.published:
            return
        self._responder.prefetch(self._session_id, event.text)
        if self._phase == "idle" and self._turn is None:
            self._open_turn(
                self._next_turn_id(),
                event.text,
                "voice",
                base_ms=event.heard_at_ms,
                marks={"stt_final": self._clock.now_ms()},
                held_utterance=event.utterance_id,
            )
            self._take_wake_timings(event.utterance_id)

    async def _on_utterance_ended(self, event: UtteranceEnded) -> None:
        if not self._accept(event):
            return
        utterance = self._utterance(event.utterance_id, event.first_pos)
        if utterance is None:
            return
        if utterance.candidate:
            await self._hold_end(utterance, event)
            return
        await self._ended_step(utterance, event)

    async def _ended_step(self, utterance: _Utterance, event: UtteranceEnded) -> None:
        utterance.closed = True
        held = self._held_turn(event.utterance_id)
        if held is not None:
            await self._end_held(held, event)
            return
        text = event.text
        if utterance.keyword is not None:
            text = _after_keyword(event.text, utterance.keyword)
        elif self._phase in ("thinking", "speaking") and self._policy is not None:
            verdict = self._policy.on_utterance_end(_final_update(event), self._phase)
            if verdict is not None and verdict.action == "interrupt":
                await self._barge_in(verdict, event.utterance_id, event.heard_at_ms)
                text = verdict.remainder
        elif self._phase == "idle" and utterance.published and self._policy is not None:
            verdict = self._policy.on_utterance_end(_final_update(event), self._phase)
            self._send_idle_ignored(utterance, verdict)
        if not utterance.published:
            # Unnamed speech that interrupted nothing closes silently, counted (v5 :421-426).
            self._silent_closes += 1
            return
        if utterance.keyword is not None and not text:
            # A keyword interrupt consumed it with nothing after the keyword (v2's closing
            # messages), in whichever phase it ends: no turn starts, and a typed turn goes on.
            self._send_control(
                VoiceDecisionDropped(turn_id=None, reason="keyword", detail=_ended_detail(event))
            )
            return
        if self._turn is not None:
            await self._cut(self._turn, reason="new_input")
        self._start_voice_turn(event, text)

    async def _end_held(self, turn: _Turn, event: UtteranceEnded) -> None:
        """The held utterance's end: a stop keyword cancels, any other end releases.

        A stop-keyword utterance means "do not answer": the held turn is cancelled without trace,
        since it never reached the client (no `barge_in`, no teardown), and the utterance closes
        with `dropped keyword`. An end whose text is not the text the turn answers is not
        released: the turn is cancelled without trace and a turn starts from the end.
        """
        if self._policy is not None:
            verdict = self._policy.on_utterance_end(_final_update(event), "held")
            if verdict is not None and verdict.action == "interrupt":
                await self._cancel_held(turn)
                self._send_control(
                    VoiceDecisionDropped(
                        turn_id=None, reason="keyword", detail=_ended_detail(event)
                    )
                )
                return
        if event.text != turn.text:
            await self._cancel_held(turn)
            self._start_voice_turn(event)
            return
        await self._release(turn, event)

    async def _release(self, turn: _Turn, event: UtteranceEnded) -> None:
        """Release a held turn in v1's order: `release`, the dispatch gate, the buffer (v1, M67).

        The one `release` naming the utterance goes before its `transcription`, its closing
        message. The buffered output then follows in the order it was made, each `0x01` entering
        the ledger as it is queued, so the estimate runs from the release. A synthesis failure
        while held put its `degraded` and frameless `0x02` in the buffer, so the turn is then a
        text reply (M65).
        """
        self._send_control(
            VoiceDecisionRelease(
                turn_id=turn.turn_id,
                reason=event.reason,
                detail=UtteranceDetail(utterance_id=event.utterance_id),
            )
        )
        turn.scope.dispatch_gate.set()
        self._send_end_of_utterance(turn.turn_id, event)
        turn.marks["eot_decided"] = event.heard_at_ms
        turn.base_ms = event.heard_at_ms - event.silence_ms
        turn.held = False
        self._phase = "thinking" if turn.audio_tag is not None else "text"
        kept, turn.kept = turn.kept, []
        for item in kept:
            item.send()
        if turn.tts_failed and not turn.frames_queued:
            self._phase = "text"
        await self._maybe_complete(turn)

    async def _cancel_held(self, turn: _Turn) -> None:
        """Cancel a held turn without trace (v1).

        It sent nothing and dispatched no tool, so nothing is purged, sent or committed, and the
        interruption note is left for the turn that follows. Its utterance is untouched.
        """
        if turn.ending:
            return
        turn.ending = True
        turn.token.set()
        turn.kept.clear()
        await self._cancel_tasks(turn)
        self._finish(turn)

    async def _on_utterance_dropped(self, event: UtteranceDropped) -> None:
        if not self._accept(event):
            return
        utterance = self._utterance(event.utterance_id, event.first_pos)
        if utterance is None:
            return
        utterance.closed = True
        if utterance.published:
            self._send_control(
                VoiceDecisionDropped(
                    turn_id=None,
                    reason=event.reason,
                    detail=DroppedDetail(
                        utterance_id=event.utterance_id,
                        duration_ms=event.duration_ms,
                        text=event.text,
                        no_speech_prob=event.no_speech_prob,
                        avg_logprob=event.avg_logprob,
                    ),
                )
            )
        elif utterance.candidate:
            # Nothing named it, so no message names its close; a name's cue is undone (v8).
            utterance.candidate = False
            utterance.held_ends.clear()
            wake = self._candidate_of(event.utterance_id)
            if wake is not None:
                self._abandon(wake, restore=True)
        held = self._held_turn(event.utterance_id)
        if held is not None:
            await self._cancel_held(held)

    async def _on_overload(self, event: IngressOverload) -> None:
        """`overload` names the turn in flight, but never a held one, which sent nothing (v1)."""
        if not self._accept(event):
            return
        turn = self._turn
        self._send_control(
            VoiceDecisionOverload(
                turn_id=turn.turn_id if turn is not None and not turn.held else None,
                reason=event.lane,
                detail=OverloadDetail(dropped_ms=event.dropped_ms, lane=event.lane),
            )
        )

    async def _on_name_heard(self, event: NameHeard) -> None:
        """The `NameHeard` row (v5 :1174): a name hit opens a wake candidate.

        Check a, `attend` (and a duck while she speaks), the candidate, then a claim; checks b
        and c follow when the claim may proceed. A held utterance is already addressed, so a
        name in it is ignored, as is a name in a published utterance or while another candidate
        waits. An endpoint's `NameHeard` continues the `wake` that armed it.
        """
        if not self._accept(event):
            return
        if event.spotter == "endpoint":
            await self._on_endpoint_name(event)
            return
        if self._phase == "held" or self._candidate is not None:
            return
        named = self._utterances.get(event.utterance_id) if event.utterance_id else None
        if named is not None and named.published:
            return
        arrival = round(event.arrival)
        wake = _Wake(
            source="server",
            score=event.hit.score,
            arrival_ms=arrival,
            name_end_ms=arrival - self._hit_lag_ms,
            turn_id=self._turn_id_now(),
            reference_score=event.reference_score,
            hit_pos=event.hit.end_pos,
        )
        if not self._screen_self_speech(wake):
            return
        self._candidate = wake
        self._attend(wake)
        if not self._open_candidate(wake, event.from_pos, named):
            return
        self._claim(wake, self._wake.kws_threshold)

    async def _on_endpoint_name(self, event: NameHeard) -> None:
        """The backlog of a waiting endpoint wake has arrived: open its candidate there.

        One with no wake waiting for it is dropped and counted (v8 :256-260), as after a mute
        abandoned the wake that armed it.
        """
        wake = self._candidate
        if wake is None or not wake.waiting_name:
            self._count_stale()
            return
        wake.waiting_name = False
        wake.hit_pos = event.hit.end_pos
        wake.reference_score = event.reference_score
        named = self._utterances.get(event.utterance_id) if event.utterance_id else None
        if not self._open_candidate(wake, event.from_pos, named):
            return
        await self._screen(wake)

    # -- Wake by name ---------------------------------------------------------------------

    @property
    def _frame_ms(self) -> int:
        return self._settings.frame_ms if self._settings is not None else FRAME_MS

    def _turn_id_now(self) -> str | None:
        """The turn a message may name now: none while it is held, since it sent nothing (v1)."""
        turn = self._turn
        if turn is None or turn.ending or turn.held:
            return None
        return turn.turn_id

    def _candidate_of(self, utterance_id: str) -> _Wake | None:
        wake = self._candidate
        if wake is None or wake.utterance_id != utterance_id:
            return None
        return wake

    def _screen_self_speech(self, wake: _Wake) -> bool:
        """Check a, which reads memory only and so runs before `attend`: False when it rejects.

        `pre_decode` is asked about her recent speech alone here; the reference score is check
        b's, which runs after `attend`.
        """
        recent = self._recent_tts_text(self._gate.echo_guard_ms)
        verdict = self._gate.pre_decode(
            score=wake.score, reference_score=None, recent_tts_text=recent
        )
        if verdict is None:
            return True
        self._send_wake_rejected(wake, "self_speech")
        return False

    def _attend(self, wake: _Wake) -> None:
        """`attend` at the hit, before any wait, and the duck while she speaks (v5 :1174)."""
        now = self._clock.now_ms()
        if self._wake.attend_cue:
            wake.attend_at = now
            self._send_control(StateCycleMessage(state="attend"))
        turn = self._turn
        if self._phase == "speaking" and turn is not None and not turn.ending:
            wake.ducked = turn.turn_id
            self._send_control(PlaybackDuckMessage(turn_id=turn.turn_id, db=DUCK_DB))

    def _open_candidate(self, wake: _Wake, from_pos: int, named: _Utterance | None) -> bool:
        """Open the candidate at `from_pos`; in a turn's phase it is the open engaged utterance.

        A name that falls in an utterance already published is already addressed: the hit is
        abandoned with its cue undone, and False is returned.
        """
        utterance = named if named is not None and not named.closed else None
        if utterance is None:
            assert self._ingress is not None  # noqa: S101 - `_accept` checked it
            opened = self._ingress.open_utterance("wake", from_pos)
            utterance = self._utterances.get(opened.utterance_id)
            if utterance is None:
                utterance = _Utterance(opened.utterance_id, opened.first_pos)
                self._utterances[opened.utterance_id] = utterance
        if utterance.published or utterance.closed:
            self._abandon(wake, restore=True)
            return False
        utterance.candidate = True
        utterance.followup = False
        wake.utterance_id = utterance.utterance_id
        return True

    def _claim(self, wake: _Wake, threshold: float) -> None:
        """Enter arbitration; the verdict comes back through the mailbox as a `WakeVerdict`.

        A lone session's arbiter answers before `claim` returns, when the id is not yet known,
        so that verdict is posted once it is.
        """
        calibration = self._calibration
        fa: float | None = None
        quantile: float | None = None
        if calibration is not None:
            fa = false_accepts_per_hour(wake.score, calibration.false_accepts)
            quantile = positive_quantile(wake.score, calibration.positive_quantile)
            if fa is None or quantile is None:
                fa = quantile = None
        rank = WakeRank(
            arrival=float(wake.arrival_ms),
            false_accepts_per_hour=fa,
            positive_quantile=quantile,
            margin=wake.score - threshold,
        )
        known: list[int] = []
        early: list[bool] = []

        def verdict(proceed: bool) -> None:
            if known:
                self._mailbox.put_nowait(WakeVerdict(known[0], proceed))
            else:
                early.append(proceed)

        claim_id = self._arbiter.claim(self._session_id, rank, verdict)
        known.append(claim_id)
        wake.claim_id = claim_id
        self._claims[claim_id] = wake
        for proceed in early:
            self._mailbox.put_nowait(WakeVerdict(claim_id, proceed))

    def _settle(self, wake: _Wake, accepted: bool) -> None:
        """Settle the candidate's claim, once: later verdicts for it are stale."""
        claim_id = wake.claim_id
        if claim_id is None or self._claims.pop(claim_id, None) is None:
            return
        self._arbiter.settle(claim_id, accepted)

    async def _screen(self, wake: _Wake) -> None:
        """Checks b and c once the claim may proceed and the candidate is open.

        Check b needs no decode. Check c reads the one gate decode through the hit plus
        `gate_tail_ms`; when ingress has already ended the candidate, no decode can come, and its
        final transcript, held meanwhile, is read instead.
        """
        if not wake.proceed or wake.utterance_id is None or wake.gate_requested:
            return
        media = self._gate.pre_decode(
            score=wake.score, reference_score=wake.reference_score, recent_tts_text=""
        )
        if media is not None:
            self._reject(wake, "self_media")
            return
        utterance = self._utterances[wake.utterance_id]
        ended = [e for e in utterance.held_ends if isinstance(e, UtteranceEnded)]
        if ended:
            await self._decide(wake, ended[-1].text)
            return
        wake.gate_requested = True
        assert self._ingress is not None  # noqa: S101 - a candidate is open on it
        through = wake.hit_pos + self._wake.gate_tail_ms // self._frame_ms
        self._ingress.request_gate_partial(wake.utterance_id, through)

    async def _decide(self, wake: _Wake, text: str) -> None:
        verdict = self._gate.decide(
            text=text, score=wake.score, reference_score=wake.reference_score
        )
        if verdict.accepted:
            await self._accept_wake(wake)
        else:
            self._reject(wake, "not_addressed")

    async def _accept_wake(self, wake: _Wake) -> None:
        """Accept: interrupt a turn in flight, publish in one step, replay a held end (v5)."""
        self._candidate = None
        self._settle(wake, True)
        assert wake.utterance_id is not None  # noqa: S101 - only an open candidate is decided
        utterance = self._utterances[wake.utterance_id]
        utterance.candidate = False
        now = self._clock.now_ms()
        screened_at = wake.attend_at if wake.attend_at is not None else now
        utterance.wake_timings = {
            "wake_accept_ms": max(now - wake.arrival_ms, 0),
            "wake_screen_ms": max(screened_at - wake.name_end_ms, 0),
        }
        turn = self._turn
        if turn is not None and not turn.ending:
            if turn.held:
                await self._cancel_held(turn)  # it never reached the client
            else:
                self._send_control(
                    VoiceDecisionBargeIn(
                        turn_id=turn.turn_id,
                        reason="wake",
                        detail=BargeInDetail(
                            keyword=None, phase=self._phase, utterance_id=wake.utterance_id
                        ),
                    )
                )
                await self._cut(turn, reason="barge_in")
        self._publish_candidate(utterance, wake=True)
        await self._replay(utterance)

    def _publish_candidate(self, utterance: _Utterance, *, wake: bool) -> None:
        """Publish an accepted candidate in one step (v5 :411-416, the D3 anchors at :1196-1197).

        `vad_status`, `listen`, the scene `wake` for a name, `listening`, then the latest
        partial; from here it is an ordinary utterance.
        """
        utterance.published = True
        utterance.candidate = False
        self._send_control(VadStatusMessage(status="speech_started"))
        self._send_control(StateCycleMessage(state="listen"))
        if wake:
            self._scene.phase("wake", turn_id=None, at=Immediate())
        self._scene.phase("listening", turn_id=None, at=Immediate())
        latest = utterance.latest
        if latest is not None:
            self._send_control(
                TranscriptPartialMessage(
                    utterance_id=utterance.utterance_id,
                    text=latest.text,
                    stable_chars=latest.stable_chars,
                )
            )
        self._responder.prefetch(self._session_id, latest.text if latest is not None else None)
        self._set_engaged(True)

    async def _replay(self, utterance: _Utterance) -> None:
        """Replay the held ends after the publishing step, in arrival order, as if just come."""
        ends, utterance.held_ends = utterance.held_ends, []
        for event in ends:
            if utterance.closed:
                return
            if isinstance(event, EndpointCandidate):
                await self._candidate_step(utterance, event)
            else:
                await self._ended_step(utterance, event)

    async def _hold_end(
        self, utterance: _Utterance, event: EndpointCandidate | UtteranceEnded
    ) -> None:
        """Keep the latest of each end until the verdict; no prefetch (v5 :1176, :401-405).

        A follow-up candidate whose end comes before any partial is judged at its end.
        """
        utterance.held_ends = [e for e in utterance.held_ends if type(e) is not type(event)]
        utterance.held_ends.append(event)
        if utterance.followup and isinstance(event, UtteranceEnded):
            await self._judge_followup(utterance)

    def _reject(self, wake: _Wake, reason: WakeRejectReason) -> None:
        """Drop the candidate, send one `wake_rejected`, and put the client back (v5 :417-422)."""
        drop = "wake_lost" if reason == "lost_arbitration" else "not_addressed"
        self._abandon(wake, drop_reason=drop, restore=False)
        self._send_wake_rejected(wake, reason)
        self._restore(wake)

    def _abandon(
        self,
        wake: _Wake,
        *,
        drop_reason: Literal["text_input", "wake_lost", "not_addressed"] | None = None,
        restore: bool,
    ) -> None:
        """End the candidate with its claim settled not accepted and no message naming it.

        `drop_reason` closes its utterance through ingress's `drop`; None leaves ingress alone,
        when ingress dropped it itself or is closing. `restore` undoes the hit's cue.
        """
        if self._candidate is wake:
            self._candidate = None
        self._settle(wake, False)
        utterance = self._utterances.get(wake.utterance_id) if wake.utterance_id else None
        if utterance is not None and not utterance.closed:
            utterance.closed = True
            utterance.candidate = False
            utterance.held_ends.clear()
            if drop_reason is not None and self._ingress is not None:
                self._ingress.drop(utterance.utterance_id, drop_reason)
        if restore:
            self._restore(wake)

    def _restore(self, wake: _Wake) -> None:
        """Unduck if the hit ducked, and resend the current phase's state after `attend`."""
        turn = self._turn
        ducked = wake.ducked is not None and turn is not None and turn.turn_id == wake.ducked
        if ducked and turn is not None and not turn.ending:
            self._send_control(PlaybackDuckMessage(turn_id=turn.turn_id, db=0.0))
        wake.ducked = None
        if wake.attend_at is None:
            return
        wake.attend_at = None
        if turn is None or turn.ending or self._phase == "idle":
            self._send_control(StateCycleMessage(state="idle"))
        elif self._phase == "speaking":
            self._send_control(StateCycleMessage(state="speak", turn_id=turn.turn_id))
        elif self._phase == "held":
            self._send_control(StateCycleMessage(state="listen"))
        else:
            self._send_control(StateCycleMessage(state="think", turn_id=turn.turn_id))

    def _send_wake_rejected(self, wake: _Wake, reason: WakeRejectReason) -> None:
        self._send_control(
            VoiceDecisionWakeRejected(
                turn_id=wake.turn_id,
                reason=reason,
                detail=WakeRejectedDetail(
                    source=wake.source, score=wake.score, reference_score=wake.reference_score
                ),
            )
        )

    def _drop_candidates(
        self, drop_reason: Literal["text_input", "wake_lost", "not_addressed"] | None
    ) -> _Wake | None:
        """Drop every unpublished candidate silently, settling a name's claim; returns the name's.

        `text` and `voice_stop` drop a candidate with no `wake_rejected` (v5 :1189-1190), and a
        disconnect drops it with nothing at all.
        """
        wake = self._candidate
        if wake is not None:
            self._abandon(wake, drop_reason=drop_reason, restore=False)
        for utterance in self._utterances.values():
            if utterance.candidate and not utterance.closed:
                utterance.closed = True
                utterance.candidate = False
                utterance.held_ends.clear()
                if drop_reason is not None and self._ingress is not None:
                    self._ingress.drop(utterance.utterance_id, drop_reason)
        return wake

    async def _judge_followup(self, utterance: _Utterance) -> None:
        """A follow-up window's candidate: checks a and b on its first transcript (A36).

        It carries no name, so check a reads her recent speech, and check b has no reference
        score to read. Accepted as `followup`, it is published; rejected, it closes silently.
        """
        utterance.followup = False
        recent = self._recent_tts_text(self._gate.echo_guard_ms)
        verdict = self._gate.pre_decode(score=0.0, reference_score=None, recent_tts_text=recent)
        if verdict is not None:
            utterance.candidate = False
            utterance.closed = True
            utterance.held_ends.clear()
            if self._ingress is not None:
                self._ingress.drop(utterance.utterance_id, "not_addressed")
            return
        self._publish_candidate(utterance, wake=False)
        await self._replay(utterance)

    def _take_wake_timings(self, utterance_id: str) -> None:
        """Give the turn just opened from the utterance the name's wake timings, if any."""
        utterance = self._utterances.get(utterance_id)
        if utterance is not None and self._turn is not None and utterance.wake_timings:
            self._turn.wake_timings = dict(utterance.wake_timings)

    # -- Listening: engaged or dormant, and the follow-up window ------------------------------

    def _set_engaged(self, engaged: bool) -> None:
        if self._ingress is None or engaged == self._engaged:
            return
        self._engaged = engaged
        self._ingress.set_engaged(engaged)

    def _busy(self) -> bool:
        """A turn, a waiting name, or an addressed or waiting utterance still open."""
        if self._turn is not None or self._candidate is not None:
            return True
        return any(not u.closed and (u.published or u.candidate) for u in self._utterances.values())

    def _sync_engaged(self) -> None:
        """At the end of each step: dormant again once nothing is in flight and no window is open.

        Ingress is engaged at an acceptance (v5 :1202-1203) and stays so through the turn it
        opens, so a stop keyword during her reply is heard.
        """
        if self._engaged and self._window_until is None and not self._busy():
            self._set_engaged(False)

    def _open_window(self) -> None:
        """Open the follow-up window at a turn's end, when one is configured (A36)."""
        length = self._wake.followup_window_ms
        if not self._engaged or length <= 0 or self._closed or self._ingress is None:
            return
        self._close_window()
        self._window_seq += 1
        self._window_until = self._clock.now_ms() + length
        self._window_task = asyncio.ensure_future(self._lapse(self._window_seq, length))

    async def _lapse(self, seq: int, length: int) -> None:
        await self._clock.sleep_ms(length)
        self._mailbox.put_nowait(_WindowLapsed(seq))

    def _close_window(self) -> None:
        task, self._window_task = self._window_task, None
        self._window_until = None
        if task is not None and not task.done():
            task.cancel()

    async def _on_window_lapsed(self, event: _WindowLapsed) -> None:
        if event.seq == self._window_seq:
            self._window_task = None
            self._window_until = None

    # -- Controls ---------------------------------------------------------------------------

    async def _on_text(self, message: Text) -> None:
        """A typed turn; a held turn is cancelled without trace and its utterance closed first.

        The held utterance was never shown as a turn, so it closes visibly (`dropped
        text_input`) rather than ending later and cutting the typed turn it preceded (v1). An
        unpublished candidate is dropped silently (v5 :1189-1190); the typed turn's `think`
        follows the `attend` its name sent.
        """
        self._drop_candidates("text_input")
        turn = self._turn
        if turn is not None and turn.held and turn.utterance_id is not None:
            await self._cancel_held(turn)
            self._drop_for_text(turn.utterance_id)
        elif turn is not None:
            await self._cut(turn, reason="new_input")
        now = self._clock.now_ms()
        self._open_turn(self._next_turn_id(), message.text, "text", base_ms=now, marks={})

    def _drop_for_text(self, utterance_id: str) -> None:
        """Close the utterance through the `drop` lever naming it, with `dropped text_input`."""
        utterance = self._utterances.get(utterance_id)
        if utterance is None or utterance.closed:
            return
        utterance.closed = True
        drop = self._ingress.drop(utterance_id, "text_input") if self._ingress else None
        self._send_control(
            VoiceDecisionDropped(
                turn_id=None, reason="text_input", detail=_closing_detail(utterance, drop)
            )
        )

    async def _on_interrupt(self, message: Interrupt) -> None:
        turn = self._turn
        if turn is None or turn.ending:
            return
        if turn.held:
            await self._cancel_held(turn)  # it never reached the client: no `barge_in`
            return
        self._send_control(
            VoiceDecisionBargeIn(
                turn_id=turn.turn_id,
                reason="ui",
                detail=BargeInDetail(keyword=None, phase=self._phase, utterance_id=None),
            )
        )
        await self._cut(turn, reason="user_interrupt")

    async def _on_voice_start(self, message: VoiceStart) -> None:
        """Adopt the ingress, its epoch, the policy and the settings; nothing is sent.

        The session posts `VoiceStart` only while voice is off, so an ingress is adopted here
        only when a test drives the row directly: the `voice_stop` cell runs first, and then
        the `idle` cell. A running typed turn goes on, and keeps its channel.
        """
        if self._ingress is not None:
            await self._stop_voice()
        self._adopt(message)

    def _adopt(self, start: VoiceStart) -> None:
        """Adopt an ingress; a new one starts dormant, and a follow-up window carries over."""
        self._ingress = start.ingress
        self._epoch = start.epoch
        self._policy = start.policy
        self._settings = start.settings
        self._handled_seq = 0
        self._engaged = False
        if self._window_until is not None:
            self._set_engaged(True)

    async def _on_voice_stop(self, message: VoiceStop) -> None:
        """Answer a pending reconfigure first, then interrupt by phase and close ingress."""
        await self._stop_voice()

    async def _stop_voice(self) -> None:
        """An unpublished candidate is dropped silently with ingress (v5 :1189-1190).

        When its name sent `attend` and no cut follows to send `idle`, the phase's state is sent
        again, so the client is not left attending.
        """
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self._host.refuse(pending, "voice_stop")
        wake = self._drop_candidates(None)
        turn = self._turn
        cut = False
        if turn is not None and turn.held:
            await self._cancel_held(turn)
        elif turn is not None and not turn.ending and self._phase in ("thinking", "speaking"):
            cut = True
            self._send_control(
                VoiceDecisionBargeIn(
                    turn_id=turn.turn_id,
                    reason="voice_stop",
                    detail=BargeInDetail(keyword=None, phase=self._phase, utterance_id=None),
                )
            )
            await self._cut(turn, reason="voice_stop")
        if wake is not None and not cut:
            self._restore(wake)
        self._close_window()
        self._close_ingress()

    def _close_ingress(self) -> None:
        """Close the adopted ingress and every utterance in the record, in audio order.

        Each published utterance closes with `dropped voice_stop`. Its detail is the drop
        `close()` returned when that drop names it; otherwise it comes from the record, the
        last partial's text and length, since ingress may already have ended an utterance whose
        end is still queued. An unpublished one closes silently: nothing named it.
        """
        ingress = self._ingress
        if ingress is None:
            return
        drop = ingress.close()
        self._ingress = None
        self._epoch = None
        self._policy = None
        open_utterances = sorted(
            (u for u in self._utterances.values() if not u.closed), key=lambda u: u.first_pos
        )
        self._utterances.clear()
        for utterance in open_utterances:
            utterance.closed = True
            if not utterance.published:
                continue
            self._send_control(
                VoiceDecisionDropped(
                    turn_id=None, reason="voice_stop", detail=_closing_detail(utterance, drop)
                )
            )

    async def _on_reconfigure(self, message: Reconfigure) -> None:
        """Hold a servable request until a quiescent point; answer the others at once (M46).

        A newer request replaces a pending one, which is answered `superseded` first, even when
        the newer one cannot be served.
        """
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self._host.refuse(pending, "superseded")
        if message.refusal is not None:
            self._host.refuse(message, message.refusal)
            return
        if self._ingress is None:
            self._host.refuse(message, "voice_stop")
            return
        self._pending = message

    def _quiescent(self) -> bool:
        """No turn, no utterance in the record, and ingress idle as of the last event handled.

        Ingress runs ahead of the mailbox, so an utterance it opened can be invisible to the
        record; comparing its `idle_since()` with the last `event_seq` handled sees it.
        """
        if self._phase != "idle" or self._turn is not None or self._ingress is None:
            return False
        if self._candidate is not None:
            return False  # an endpoint wake may still be waiting for its backlog
        if any(not u.closed for u in self._utterances.values()):
            return False
        return self._ingress.idle_since() == self._handled_seq

    def _maybe_reconfigure(self) -> None:
        """Apply a pending reconfigure in one step, with no await, so no frame is routed between.

        The old ingress is closed only once the host has built and adopted its successor; a
        host that refuses (None) leaves it adopted and open.
        """
        request = self._pending
        if request is None or self._closed or not self._quiescent():
            return
        self._pending = None
        ingress = self._ingress
        assert ingress is not None  # noqa: S101 - `_quiescent` checked it
        seed = ingress.handoff()
        start = self._host.apply(request, seed)
        if start is None:
            return
        ingress.close()
        self._utterances.clear()
        self._adopt(start)

    async def _on_mute(self, message: Mute) -> None:
        """Mute closes nothing itself: ingress emits the open utterance's drop (v8).

        An endpoint wake still waiting for its backlog has no utterance to drop: ingress
        discarded the backlog at mute, so the wake is abandoned here, with its claim settled and
        its cue undone, and no message (v8 :253-255).
        """
        self._muted = message.active
        wake = self._candidate
        if message.active and wake is not None and wake.waiting_name:
            self._abandon(wake, restore=True)

    async def _on_wake(self, message: Wake) -> None:
        """The `wake` row (v5 :1179): an endpoint's name, through the addressee gate.

        With no voice session it is ignored. Muted, it gets a fresh id and one `dropped muted`
        (v4). With too little backlog it is rejected `no_backlog` before anything opens.
        Otherwise check a, `attend` and `expect_endpoint_wake` run now, and the claim is made
        now, from the message's arrival, so a slow backlog still takes part in arbitration; the
        candidate opens when the backlog's `NameHeard` comes. A held, thinking, speaking or typed
        turn is interrupted only after acceptance.
        """
        if self._ingress is None:
            return
        if self._muted:
            self._send_control(
                VoiceDecisionDropped(
                    turn_id=None,
                    reason="muted",
                    detail=DroppedDetail(
                        utterance_id=self._new_utterance_id(),
                        duration_ms=0,
                        text=None,
                        no_speech_prob=None,
                        avg_logprob=None,
                    ),
                )
            )
            return
        arrival = round(message.arrival)
        wake = _Wake(
            source="endpoint",
            score=message.score,
            arrival_ms=arrival,
            name_end_ms=arrival,
            turn_id=self._turn_id_now(),
        )
        if message.backlog_ms < self._wake.endpoint_min_backlog_ms:
            self._send_wake_rejected(wake, "no_backlog")
            return
        if self._candidate is not None or not self._screen_self_speech(wake):
            return
        self._candidate = wake
        wake.waiting_name = True
        self._attend(wake)
        self._ingress.expect_endpoint_wake(
            keyword=message.keyword,
            score=message.score,
            backlog_ms=message.backlog_ms,
            arrival=message.arrival,
        )
        self._claim(wake, self._wake.endpoint_threshold)

    async def _on_wake_verdict(self, message: WakeVerdict) -> None:
        """An arbitration verdict for one of this controller's claims (v9 :153-163).

        One for a claim already settled is ignored and counted: a drop was handled before a
        verdict already in the mailbox. A losing claim closes its candidate
        `lost_arbitration`; a winning one goes on to checks b and c, at once or when an endpoint
        wake's `NameHeard` opens its candidate.
        """
        wake = self._claims.get(message.claim_id)
        if wake is None:
            self._count_stale()
            return
        if not message.proceed:
            self._reject(wake, "lost_arbitration")
            return
        wake.proceed = True
        await self._screen(wake)

    async def _on_barge_in_report(self, message: BargeInReport) -> None:
        """An endpoint's keyword: it interrupts only audio playing now (v1); every other verdict
        goes out as `barge_in_ignored`, naming no utterance, since no transcript produced it
        (v2 :931-935).

        A report naming any turn but the one in flight is counted stale and answered
        `stale_turn`, whatever the policy would say: the controller knows that turn ended.
        """
        turn = self._turn
        now = self._clock.now_ms()
        if turn is None or turn.turn_id != message.turn_id or turn.ending:
            self._count_stale()
            if self._policy is not None:
                verdict = self._policy.on_endpoint_report(message.turn_id, message.keyword, now)
                stale = BargeInVerdict(
                    action="ignored",
                    reason="stale_turn",
                    keyword=verdict.keyword if verdict is not None else message.keyword,
                    text="",
                    remainder="",
                    phase=None,
                )
                self._send_ignored(message.turn_id, stale, None)
            return
        if self._policy is None or self._phase != "speaking":
            # An endpoint verdict carries no phase, so the controller checks it: only audio
            # the client is playing can be barged in on by an endpoint keyword.
            return
        verdict = self._policy.on_endpoint_report(message.turn_id, message.keyword, now)
        if verdict is None:
            return
        if verdict.action == "interrupt":
            await self._barge_in(verdict, None, None)
        else:
            self._send_ignored(turn.turn_id, verdict, None)

    async def _on_playback_progress(self, message: PlaybackProgress) -> None:
        turn = self._current(message.turn_id)
        if turn is None or turn.ledger is None:
            self._count_stale()
            return
        self._report(turn, message.played_ms)
        if turn.handoff_waiting and self._ack_played(turn):
            await self._on_ack_played(_AckPlayed(turn.turn_id))

    async def _on_playback_done(self, message: PlaybackDone) -> None:
        turn = self._current(message.turn_id)
        if turn is None or turn.ledger is None:
            self._count_stale()
            return
        self._report(turn, message.played_ms)
        turn.playback_done = True
        await self._maybe_complete(turn)
        if turn.tts_failed and not turn.ending and self._turn is turn:
            # The voiced part has drained before the generation ended: the rest is text (M65).
            self._phase = "text"

    def _report(self, turn: _Turn, played_ms: int) -> None:
        assert turn.ledger is not None  # noqa: S101 - the callers checked it
        turn.ledger.report(played_ms)
        if played_ms > 0:
            turn.mark("first_played", self._clock.now_ms() - played_ms)

    async def _on_close(self, message: Close) -> None:
        self._closed = True
        self._pending = None  # a disconnect drops a pending reconfigure with no answer
        self._close_window()
        self._drop_candidates(None)  # settled, with nothing sent and no lever pulled
        turn = self._turn
        if turn is not None and not turn.ending:
            await self._cut(turn, reason="disconnect", notify=False)

    # -- Opening a turn ---------------------------------------------------------------------

    def _next_turn_id(self) -> str:
        self._turn_seq += 1
        return f"{self._session_id}-turn-{self._turn_seq}"

    def _channel_for(self, origin: TurnOrigin) -> StreamChannel:
        """The channel rule: speak a voice turn when TTS is on, loaded, and wanted."""
        if (
            origin == "voice"
            and self._tts_enabled
            and self._synthesizer is not None
            and self._settings is not None
            and self._settings.playback
        ):
            return "tts"
        return "text"

    def _send_end_of_utterance(
        self, turn_id: str, event: UtteranceEnded, text: str | None = None
    ) -> None:
        """The end's `endpoint`, `speech_ended` and the utterance's `transcription`.

        The `transcription` carries what the turn takes from the utterance (v2's closing
        messages): its whole text, or `text`, the part after a keyword that interrupted.
        """
        self._send_control(
            VoiceDecisionEndpoint(
                turn_id=turn_id,
                reason=event.reason,
                detail=EndpointDetail(
                    utterance_id=event.utterance_id,
                    eot_probability=event.eot_probability,
                    silence_ms=event.silence_ms,
                ),
            )
        )
        self._send_control(VadStatusMessage(status="speech_ended"))
        self._send_control(
            TranscriptionMessage(
                text=event.text if text is None else text, utterance_id=event.utterance_id
            )
        )

    def _start_voice_turn(self, event: UtteranceEnded, text: str | None = None) -> None:
        """Start the turn the end opens; `text`, when given, is its turn text (the remainder)."""
        turn_id = self._next_turn_id()
        turn_text = event.text if text is None else text
        self._send_end_of_utterance(turn_id, event, turn_text)
        # Timings count from the end of speech: the utterance's last frame less its silence.
        marks = {"eot_decided": event.heard_at_ms, "stt_final": self._clock.now_ms()}
        base = event.heard_at_ms - event.silence_ms
        self._open_turn(turn_id, turn_text, "voice", base_ms=base, marks=marks)
        self._take_wake_timings(event.utterance_id)

    def _open_turn(
        self,
        turn_id: str,
        text: str,
        origin: TurnOrigin,
        *,
        base_ms: int,
        marks: dict[str, int],
        held_utterance: str | None = None,
    ) -> None:
        """Open a turn; with `held_utterance` it starts held, answering that utterance."""
        placement = self._placement()
        channel = self._channel_for(origin)
        held = held_utterance is not None
        request = TurnRequest(
            session_id=self._session_id,
            user_text=text,
            origin=origin,
            channel=channel,
            interruption_note=self._note,
            held=held,
        )
        scope = self._responder.open_turn(request)
        now = self._clock.now_ms()
        turn = _Turn(
            turn_id=turn_id,
            origin=origin,
            channel=channel,
            scope=scope,
            placement=placement,
            power_level=self._power_level(),
            opened_ms=now,
            base_ms=base_ms,
            marks=marks,
            text=text,
            held=held,
            utterance_id=held_utterance,
        )
        turn.scene = _TurnScene(self._scene, turn)
        self._turn = turn
        self._send_turn_control(turn, StateCycleMessage(state="think", turn_id=turn_id))
        if channel == "tts":
            self._tag = next_turn_tag(self._tag)
            turn.audio_tag = self._tag
        self._send_turn(
            turn,
            StreamStartMessage(
                turn_id=turn_id, channel=channel, origin=origin, audio_tag=turn.audio_tag
            ),
            kind="stream_start",
        )
        turn.scene.phase("thinking", turn_id=turn_id, at=Immediate())
        if turn.audio_tag is not None:
            self._start_speech(turn, turn.audio_tag)
            self._phase = "thinking"
        else:
            self._phase = "text"
        if held:
            self._phase = "held"
        turn.mark("request_sent", now)
        producer = self._spawn(turn, self._produce(turn), "stream")
        producer.add_done_callback(lambda task: self._producer_done(turn, task))

    def _start_speech(self, turn: _Turn, tag: int) -> None:
        turn.scene.phase("speaking", turn_id=turn.turn_id, at=AtSegment(tag, 0))
        turn.scene.phase("idle", turn_id=turn.turn_id, at=AtStreamEnd(tag))
        segmenter = SpeechSegmenter(self._segmenter_config)
        turn.segmenter = segmenter
        turn.ledger = PlaybackLedger(
            lambda: segmenter.text,
            self._clock,
            prebuffer_ms=self._playback_config.client_prebuffer_ms,
        )
        turn.pacer = PlaybackPacer(
            turn_id=turn.turn_id,
            audio_tag=tag,
            turn_sink=_TurnSink(self, turn),
            control=lambda item: self._pacer_control(turn, item),
            ledger=turn.ledger,
            clock=self._clock,
            config=self._playback_config,
            listener=turn.scene,
        )
        self._spawn(turn, self._synthesize(turn), "synthesize")
        self._spawn(turn, self._play(turn), "play")

    def _pacer_control(self, turn: _Turn, item: object) -> None:
        """The pacer's `control` sink; after a disconnect no client is left to send to."""
        if not turn.disconnected:
            self._outbound.send_control(item)

    def _spawn(self, turn: _Turn, coro: Awaitable[None], name: str) -> asyncio.Task[None]:
        task = asyncio.ensure_future(coro)
        task.set_name(f"{turn.turn_id}-{name}")
        turn.tasks.append(task)
        return task

    # -- The turn's tasks -------------------------------------------------------------------

    async def _produce(self, turn: _Turn) -> None:
        loop = asyncio.get_running_loop()
        stream = self._responder.stream_turn(turn.scope)
        async with contextlib.aclosing(stream):  # type: ignore[type-var]
            async for event in stream:
                ack: asyncio.Future[None] = loop.create_future()
                self._post_internal(_StreamItem(turn.turn_id, event, ack))
                await ack
        self._post_internal(_StreamEnded(turn.turn_id, failed=False))

    def _producer_done(self, turn: _Turn, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "turn %s: the reply stream failed with %s", turn.turn_id, type(error).__name__
            )
            self._post_internal(_StreamEnded(turn.turn_id, failed=True))

    async def _synthesize(self, turn: _Turn) -> None:
        assert self._synthesizer is not None and self._tts_lane is not None  # noqa: S101
        assert turn.pacer is not None  # noqa: S101
        synthesizer = self._synthesizer
        lane = self._tts_lane
        try:
            while True:
                item = await turn.segments.get()
                if item is None:
                    turn.pacer.finish()
                    return
                index, segment = item
                chunks = await lane.run(
                    synthesizer.synthesize, segment, turn.token, token=turn.token
                )
                while True:
                    chunk = await lane.run(next, chunks, None, token=turn.token)
                    if chunk is None:
                        break
                    self._on_chunk(turn, index, segment, chunk)
                turn.pacer.end_segment(index)
                # A play task waiting for audio was woken by `end_segment`, which put it ahead
                # of this task on the loop's ready queue: one yield lets it queue the segment's
                # short last frame before the segment counts as synthesized, so `_ack_played`
                # never sees a finished ack whose last frame is not yet queued.
                await asyncio.sleep(0)
                turn.synth_done_through = index
        except LaneJobCancelled:
            return
        except SpeechEngineError as exc:
            self._post_internal(_SynthFailed(turn.turn_id, exc))

    def _on_chunk(self, turn: _Turn, index: int, segment: TextSegment, chunk: SynthChunk) -> None:
        assert turn.pacer is not None  # noqa: S101
        turn.mark("first_tts_chunk", self._clock.now_ms())
        if segment.state != NEUTRAL_STATE and chunk.state == segment.state:
            turn.voiced_states.add(index)
        turn.voiced_end = segment.end_char
        turn.pacer.submit(index, chunk)

    async def _play(self, turn: _Turn) -> None:
        assert turn.pacer is not None  # noqa: S101
        await turn.pacer.run()
        self._post_internal(_PlayEnded(turn.turn_id))

    async def _watch_ack(self, turn: _Turn) -> None:
        """Wait, by the ledger's estimate, until the hand-off's ack has played."""
        while turn.handoff_waiting:
            if self._ack_played(turn):
                self._post_internal(_AckPlayed(turn.turn_id))
                return
            await self._clock.sleep_ms(self._playback_config.frame_ms)

    def _ack_played(self, turn: _Turn) -> bool:
        """Every frame of the ack queued so far has played and its synthesis is done.

        `_synthesize` ends each segment in the pacer, and lets the play task step, before it
        counts the segment as synthesized. By then the ack's last frame, even one shorter than
        `frame_ms`, is queued or waits on the lead behind queued audio not yet played, so the
        ack counts as played only once that frame has played too.
        """
        ledger = turn.ledger
        if ledger is None or turn.ack_last_index is None:
            return False
        if turn.synth_done_through < turn.ack_last_index or ledger.recorded_samples == 0:
            return False
        return ledger.played_samples() >= ledger.recorded_samples

    # -- Turn events --------------------------------------------------------------------------

    async def _on_stream_item(self, item: _StreamItem) -> None:
        try:
            turn = self._current(item.turn_id)
            if turn is None:
                return
            event = item.event
            if isinstance(event, Token):
                self._on_token(turn, event)
            elif isinstance(event, HandOff):
                self._on_handoff(turn, event)
            elif isinstance(event, StateHint):
                self._on_state_hint(turn, event)
            elif isinstance(event, WSEvent):
                self._on_ws_event(turn, event)
            elif isinstance(event, Complete):
                self._send_complete(turn, event)
        finally:
            if not item.ack.done():
                item.ack.set_result(None)

    def _on_token(self, turn: _Turn, event: Token) -> None:
        now = self._clock.now_ms()
        turn.handoff_waiting = False
        turn.mark("first_token", now)
        self._send_token(turn, event.text)
        if event.is_error:
            turn.error = True
            if not turn.error_sent:
                turn.error_sent = True
                self._send_degraded(turn, "turn_error")
        if self._speaks(turn):
            assert turn.segmenter is not None  # noqa: S101 - `_speaks` checked it
            for segment in turn.segmenter.feed(event.text, event.is_error):
                self._submit_segment(turn, segment)

    @staticmethod
    def _speaks(turn: _Turn) -> bool:
        """Whether the turn's text still goes to TTS: a `tts` turn not moved to text."""
        return turn.segmenter is not None and not turn.tts_failed

    def _submit_segment(self, turn: _Turn, segment: SpeechSegment) -> None:
        turn.mark("first_segment", self._clock.now_ms())
        text_segment = segment.segment
        if turn.state != NEUTRAL_STATE:
            text_segment = dataclasses.replace(text_segment, state=turn.state)
            turn.requested_states += 1
        turn.segments.put_nowait((segment.index, text_segment))

    def _on_handoff(self, turn: _Turn, event: HandOff) -> None:
        """Speak the ack as its own segment at once, then wait in `thinking` (v6)."""
        if turn.token_texts or turn.handoff_seen:
            logger.warning("turn %s: a HandOff after the reply began was ignored", turn.turn_id)
            return
        turn.handoff_seen = True
        self._send_token(turn, event.ack_text)
        if not self._speaks(turn):
            return
        assert turn.segmenter is not None  # noqa: S101 - `_speaks` checked it
        segments = turn.segmenter.feed(event.ack_text, False) + turn.segmenter.flush()
        for segment in segments:
            self._submit_segment(turn, segment)
        if segments:
            turn.ack_last_index = segments[-1].index
            turn.handoff_waiting = True
            self._spawn(turn, self._watch_ack(turn), "ack")

    def _on_state_hint(self, turn: _Turn, event: StateHint) -> None:
        if event.state in SPEAKING_STATES:
            turn.state = event.state
            return
        if not turn.state_warned:
            turn.state_warned = True
            logger.warning(
                "turn %s: a speaking state outside the vocabulary was read as neutral",
                turn.turn_id,
            )
        turn.state = NEUTRAL_STATE

    def _on_ws_event(self, turn: _Turn, event: WSEvent) -> None:
        payload = event.payload
        self._send_turn(turn, payload)
        kind = payload.get("type")
        now = self._clock.now_ms()
        call_id = str(payload.get("tool_call_id"))
        if kind == "tool_call_started":
            turn.tool = True
            turn.mark("tool_call_parsed", now)
            turn.open_tools[call_id] = (str(payload.get("name")), now)
        elif kind == "tool_call_completed":
            turn.marks["tool_done"] = now
            turn.open_tools.pop(call_id, None)
            view = payload.get("view")
            if isinstance(view, str):
                data = payload.get("data")
                at: Immediate | AtSegment = Immediate()
                if turn.audio_tag is not None and self._speaks(turn):
                    assert turn.segmenter is not None  # noqa: S101 - `_speaks` checked it
                    at = AtSegment(turn.audio_tag, turn.segmenter.next_index)
                turn.scene.tool_view(
                    turn.turn_id,
                    str(payload.get("tool_call_id")),
                    view,  # type: ignore[arg-type]
                    data if isinstance(data, Mapping) else None,
                    at=at,
                )

    def _send_complete(self, turn: _Turn, event: Complete | None) -> None:
        if turn.complete_sent:
            return
        turn.complete_sent = True
        full_text = event.final_response if event is not None else "".join(turn.token_texts)
        tool_calls = event.tool_calls_used if event is not None else len(turn.scope.tool_calls)
        self._send_turn(
            turn,
            {
                "type": "stream_complete",
                "turn_id": turn.turn_id,
                "full_text": full_text,
                "duration_ms": max(self._clock.now_ms() - turn.opened_ms, 0),
                "tool_calls_used": tool_calls,
            },
        )

    async def _on_stream_ended(self, event: _StreamEnded) -> None:
        turn = self._current(event.turn_id)
        if turn is None or turn.stream_done:
            return
        turn.stream_done = True
        turn.handoff_waiting = False
        if event.failed:
            turn.error = True
            if not turn.error_sent:
                turn.error_sent = True
                self._send_degraded(turn, "turn_error")
        self._send_complete(turn, None)
        if turn.segmenter is not None:
            flushed = turn.segmenter.flush() if self._speaks(turn) else []
            for segment in flushed:
                self._submit_segment(turn, segment)
            turn.segments.put_nowait(None)
        await self._maybe_complete(turn)

    async def _on_audio_started(self, event: _AudioStarted) -> None:
        turn = self._current(event.turn_id)
        if turn is None:
            return
        if not turn.audio_started:
            turn.audio_started = True
            if self._policy is not None:
                self._policy.on_turn_audio_started(turn.turn_id, event.at_ms)
        if self._phase == "thinking":
            self._phase = "speaking"

    async def _on_ack_played(self, event: _AckPlayed) -> None:
        turn = self._current(event.turn_id)
        if turn is None or not turn.handoff_waiting:
            return
        turn.handoff_waiting = False
        if self._phase == "speaking":
            self._phase = "thinking"
            turn.stretch_open = False
            self._send_turn(turn, StateCycleMessage(state="think", turn_id=turn.turn_id))

    async def _on_play_ended(self, event: _PlayEnded) -> None:
        turn = self._current(event.turn_id)
        if turn is None:
            return
        turn.tag_ended = True
        self._end_scene(turn)
        await self._maybe_complete(turn)

    async def _on_synth_failed(self, event: _SynthFailed) -> None:
        """The synthesizer failed: the rest of the reply goes on as text (v2 M65, M68).

        A held turn's failure goes into its buffer, so it reaches the client at the release, in
        its buffered order; until then the `held` cells apply.
        """
        turn = self._current(event.turn_id)
        if turn is None or turn.pacer is None or turn.tts_failed:
            return
        logger.error("turn %s: synthesis failed with %s", turn.turn_id, type(event.error).__name__)
        self._switch_to_text(turn, event.error)

    def _switch_to_text(self, turn: _Turn, error: SpeechEngineError) -> None:
        """Send `degraded tts_failed` in stream order and end the tag as the audio allows.

        The `degraded` follows exactly the `stream_token`s sent so far, so `text_from_token`
        counts them and `voiced_text`, the text of the audio that still reaches the client, is a
        prefix of their text. With no `0x01` queued yet, the audio the pacer holds is dropped,
        its frameless `0x02` follows the `degraded`, and the turn is a text reply from here.
        Otherwise the pacer hands on the audio it holds, then its `0x02`, and the turn moves to
        `text` when the client has played it (`_on_playback_done`), unless it completes first.

        A held turn has queued no `0x01`: the frames it kept back are discarded, and the
        `degraded` and the frameless `0x02` join its buffer after the tokens kept before them.
        It stays `held`; its release moves it to `text` (v2 :392-401).
        """
        assert turn.pacer is not None and turn.segmenter is not None  # noqa: S101
        turn.tts_failed = True
        turn.handoff_waiting = False
        turn.text_from_token = len(turn.token_texts)
        turn.voiced_text = turn.segmenter.text[: turn.voiced_end] if turn.frames_queued else ""
        if turn.held:
            turn.kept = [kept for kept in turn.kept if kept.frame is None]
        self._send_turn(
            turn,
            VoiceDecisionDegraded(
                turn_id=turn.turn_id,
                reason="tts_failed",
                detail=DegradedDetail(
                    engine=self._synthesizer.engine_id if self._synthesizer is not None else None,
                    error=type(error).__name__,
                    voiced_text=turn.voiced_text,
                    text_from_token=turn.text_from_token,
                ),
            ),
        )
        if turn.frames_queued:
            turn.pacer.finish()
            return
        turn.pacer.end_unsent()
        if not turn.held:
            self._phase = "text"

    # -- Ending a turn ---------------------------------------------------------------------

    def _end_scene(self, turn: _Turn) -> None:
        if not turn.scene_ended:
            turn.scene_ended = True
            turn.scene.turn_ended(turn.turn_id)

    async def _maybe_complete(self, turn: _Turn) -> None:
        """Complete the turn once its generation and its tag have both ended; never while held."""
        if turn.ending or turn.held or not turn.stream_done:
            return
        if turn.channel == "tts":
            assert turn.ledger is not None  # noqa: S101
            if not turn.tag_ended:
                return
            if not (turn.playback_done or (turn.tts_failed and turn.ledger.played_through())):
                return
        await self._complete(turn)

    async def _complete(self, turn: _Turn) -> None:
        """Commit the full reply: a completion purges nothing, so the client has every token.

        A turn moved to text records its voiced text plus every token from `text_from_token`
        on (v2 M65, M69): the tokens before the switch that no audio voiced were never heard.
        """
        turn.ending = True
        await self._cancel_tasks(turn)
        self._close_orphan_tools(turn)
        outcome: Outcome = "error" if turn.error else "completed"
        delivered = "".join(turn.token_texts)
        if turn.tts_failed:
            delivered = turn.voiced_text + "".join(turn.token_texts[turn.text_from_token :])
        result = self._commit(turn, outcome, delivered, tool_effect_unknown=False)
        if turn.channel == "text" or turn.tts_failed:
            self._scene.phase("idle", turn_id=turn.turn_id, at=Immediate())
        self._end_scene(turn)
        if result.error is not None:
            self._send_degraded(turn, "record_failed", result.error)
        self._send_turn(turn, StateCycleMessage(state="idle", turn_id=turn.turn_id))
        self._send_turn(turn, self._metrics(turn))
        self._finish(turn)

    async def _barge_in(
        self, verdict: BargeInVerdict, utterance_id: str | None, heard_at_ms: int | None
    ) -> None:
        """Cut the turn for a policy's `interrupt` verdict.

        A transcript's verdict names its utterance, and that `barge_in` publishes it (v5
        :421-424): from then on it is an ordinary utterance, whose turn text, when it ends, is
        what follows the keyword.
        """
        turn = self._turn
        if turn is None or turn.ending:
            return
        utterance = self._utterances.get(utterance_id) if utterance_id is not None else None
        if utterance is not None:
            utterance.published = True
            utterance.keyword = verdict.keyword
        self._send_control(
            VoiceDecisionBargeIn(
                turn_id=turn.turn_id,
                reason=verdict.reason,  # type: ignore[arg-type]
                detail=BargeInDetail(
                    keyword=verdict.keyword, phase=self._phase, utterance_id=utterance_id
                ),
            )
        )
        if heard_at_ms is not None:
            turn.marks["keyword_heard"] = heard_at_ms
        await self._cut(turn, reason="barge_in")

    async def _cut(self, turn: _Turn, *, reason: CancelReason, notify: bool = True) -> None:
        """Cancel the turn and commit what the client received (v2 M62, M69).

        The task group is cancelled and awaited first, so no token or frame of the turn is
        queued after the purge; the purge and the pacer's stop then run with no await between
        them. A disconnect (`notify` false) sends nothing from the purge on, the pacer's
        `0x03` and `playback_stop` included, since no client is left.

        The record holds what the sender took: the tokens before the purge's `first_token`.
        A `tts` turn's spoken prefix is the ledger's, capped at the end of their text, since an
        estimate made without client reports runs over frames the client may never get. A turn
        moved to text adds the taken tokens from `text_from_token` on; when the purge removed
        its `degraded`, it removed token `text_from_token` too, so that adds nothing.

        A held turn never reached the client, so any cut of one, a disconnect included, is a
        cancel without trace.
        """
        if turn.held:
            await self._cancel_held(turn)
            return
        if turn.ending:
            return
        turn.ending = True
        turn.disconnected = not notify
        turn.token.set()
        await self._cancel_tasks(turn)
        purge = self._outbound.purge_turn(turn.turn_id)
        if turn.pacer is not None:
            turn.pacer.stop(purge.terminal, reason=reason)
            turn.marks["playback_stopped"] = self._clock.now_ms() + self._playback_config.fade_ms
        self._scene.cancel_turn(turn.turn_id)
        taken_count = len(turn.token_texts) if purge.first_token is None else purge.first_token
        taken_text = "".join(turn.token_texts[:taken_count])
        spoken: str | None = None
        delivered = taken_text
        if turn.channel == "tts":
            assert turn.ledger is not None  # noqa: S101
            spoken = turn.ledger.spoken_text()[: len(taken_text)]
            delivered = spoken
            if turn.tts_failed:
                delivered += "".join(turn.token_texts[turn.text_from_token : taken_count])
        if notify:
            self._send_turn(
                turn,
                StreamCancelledMessage(
                    turn_id=turn.turn_id,
                    partial_text=taken_text,
                    reason=reason,
                    spoken_text=spoken,
                ),
            )
        finished = await self._uncancellable(self._wait_tools(turn.scope))
        if notify:
            self._close_orphan_tools(turn)
        result = self._commit(turn, "interrupted", delivered, tool_effect_unknown=not finished)
        if notify:
            if result.error is not None:
                if purge.stream_start:
                    logger.error("turn %s: record_failed", turn.turn_id)
                else:
                    self._send_degraded(turn, "record_failed", result.error)
            self._scene.phase("idle", turn_id=None, at=Immediate())
            self._send_control(StateCycleMessage(state="idle"))
            self._send_turn(turn, self._metrics(turn))
        self._finish(turn)

    async def _cancel_tasks(self, turn: _Turn) -> None:
        tasks = [task for task in turn.tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await self._uncancellable(asyncio.gather(*tasks, return_exceptions=True))

    def _close_orphan_tools(self, turn: _Turn) -> None:
        """Close every `tool_call_started` the client got no `tool_call_completed` for (v1).

        The stream that would have carried the completion is closed, so the client is told the
        call was cancelled whether or not the tool finished within the grace; the record keeps
        the real result of one that did.
        """
        now = self._clock.now_ms()
        for call_id, (name, started_ms) in turn.open_tools.items():
            self._send_turn(
                turn,
                {
                    "type": "tool_call_completed",
                    "tool_call_id": call_id,
                    "name": name,
                    "duration_ms": max(now - started_ms, 0),
                    "result_summary": "",
                    "error": "cancelled",
                },
            )
        turn.open_tools.clear()

    async def _wait_tools(self, scope: TurnScope) -> bool:
        """Wait up to the tool grace, on the session clock, for every dispatched tool."""
        pending = [task for task in scope.tool_tasks if not task.done()]
        if not pending:
            return True
        grace = asyncio.ensure_future(self._clock.sleep_ms(self._turn_config.tool_cancel_grace_ms))
        try:
            while pending and not grace.done():
                await asyncio.wait([*pending, grace], return_when=asyncio.FIRST_COMPLETED)
                pending = [task for task in pending if not task.done()]
            return not pending
        finally:
            grace.cancel()

    def _commit(
        self, turn: _Turn, outcome: Outcome, delivered: str, *, tool_effect_unknown: bool
    ) -> CommitResult:
        """Commit the turn once; a second request returns the first result and writes nothing."""
        if turn.committed and turn.commit_result is not None:
            return turn.commit_result
        turn.committed = True
        turn.outcome = outcome
        delivery = TurnDelivery(
            outcome=outcome,
            delivered_text=delivered,
            stt_model=None,
            tts_model=(
                self._synthesizer.model_id
                if turn.channel == "tts" and self._synthesizer is not None
                else None
            ),
            tool_effect_unknown=tool_effect_unknown,
        )
        result = self._responder.commit_turn(turn.scope, delivery)
        turn.commit_result = result
        # Recording semantics: the next turn's request says how this one ended, never its text.
        if outcome in ("interrupted", "superseded"):
            self._note = "mid_reply" if delivered else "before_reply"
        else:
            self._note = None
        return result

    def _finish(self, turn: _Turn) -> None:
        """The turn is over: back to `idle`, with the follow-up window open if configured."""
        self._log_states(turn)
        if self._turn is turn:
            self._turn = None
            self._phase = "idle"
            self._open_window()

    def _log_states(self, turn: _Turn) -> None:
        """One INFO line per turn that asked for a speaking state, with counts and no text."""
        if turn.requested_states == 0 or self._synthesizer is None:
            return
        logger.info(
            "turn %s speaking states: requested=%d voiced=%d engine=%s",
            turn.turn_id,
            turn.requested_states,
            len(turn.voiced_states),
            self._synthesizer.engine_id,
        )

    def _metrics(self, turn: _Turn) -> TurnMetricsMessage:
        timings = {key: value - turn.base_ms for key, value in turn.marks.items()}
        ledger = turn.ledger
        if "first_played" not in timings and ledger is not None and ledger.recorded_samples > 0:
            timings["first_played"] = ledger.scheduled_play_ms(0) - turn.base_ms
        if turn.handoff_ack_ms is not None:
            timings["handoff_ack_ms"] = turn.handoff_ack_ms
        timings.update(turn.wake_timings)
        kind: Literal["plain", "tool"] = (
            "tool" if turn.tool or turn.scope.tool_dispatched else "plain"
        )
        budget = turn.placement.budget
        result = turn.commit_result
        return TurnMetricsMessage(
            turn_id=turn.turn_id,
            origin=turn.origin,
            channel=turn.channel,
            kind=kind,
            outcome=turn.outcome,
            recorded=result.recorded if result is not None else False,
            power_level=turn.power_level,
            engines=EnginePlacement(stt=turn.placement.stt, tts=turn.placement.tts),
            timings=timings,
            budget=_wire_budget(budget),
            over_budget=judge_over_budget(timings, budget, tool=kind == "tool"),
            ledger_source=turn.ledger.source if turn.ledger is not None else "estimate",
            prompt=_prompt_stats(turn.scope.prompt_stats),
            lanes=TurnLaneStats(stt_wait_ms=0, tts_wait_ms=0, tts_busy_after_cancel_ms=0),
            stale_events=turn.stale_events,
            bulk_dropped=0,
        )
