"""Per-turn state and the value objects of the stream-and-commit turn API.

`ConversationHandler` is shared by every session in the process. Before this
module, the FE-bound event buffer, the session id and the turn index lived on
the handler itself, so a turn starting while another was still in flight
cleared the first turn's buffer and overwrote the session id its graph seed
read. A `TurnScope` is created at the start of each turn and passed explicitly
to every method that reads or writes that state, so turns in different
sessions can run concurrently without seeing each other's events.

A turn runs in three steps: `open_turn(TurnRequest)` creates the scope,
`stream_turn(scope)` produces the reply and writes nothing durable, and
`commit_turn(scope, TurnDelivery)` writes session history and the event-log
row. Splitting the write from the stream lets a caller cancel a turn part way
(a barge-in) and still record what the user actually received. `TurnRequest`,
`TurnDelivery` and `CommitResult` live here, in the chat package, so the
realtime layer imports them from `backend.chat` and chat never imports
realtime.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from backend.knowledge.models import RetrievalResult
    from backend.llm.models import ToolCall, UsageMetadata


@dataclass(frozen=True, slots=True)
class TurnRequest:
    """What one turn is asked to answer.

    Attributes:
        session_id: Session the turn belongs to.
        user_text: The user's utterance or typed message.
        origin: Who produced the input. `voice` for an utterance, `text` for a
            typed message; `system` (a turn MIST starts herself) is reserved
            and refused by `open_turn` until something produces it.
        channel: Where the reply goes: `tts` (spoken) or `text` (displayed).
        user_id: User the session is created for on its first turn.
        max_history: Upper bound on history messages in the prompt.
        interruption_note: Set by the turn controller from the session's
            last committed turn: `mid_reply` when that turn was cut off after
            some of its reply was delivered, `before_reply` when nothing was.
            The prompt layout renders it as a fixed line; the handler only
            carries it on the scope.
        held: True while the turn may still be cancelled without trace (the
            utterance might continue). A held turn's `dispatch_gate` starts
            clear.
    """

    session_id: str
    user_text: str
    origin: Literal["voice", "text", "system"]
    channel: Literal["tts", "text"]
    user_id: str = "User"
    max_history: int = 10
    interruption_note: Literal["mid_reply", "before_reply"] | None = None
    held: bool = False


@dataclass(frozen=True, slots=True)
class TurnDelivery:
    """What the caller delivered of a turn, handed to `commit_turn`.

    Attributes:
        outcome: `completed`, `interrupted`, `superseded` or `error`.
        delivered_text: What reached the user: the spoken prefix for a `tts`
            turn, every streamed token for a `text` turn. Recorded as the
            turn's `system_response` and kept in session history.
        stt_model: Speech-to-text model that transcribed the utterance, if any.
        tts_model: Text-to-speech model that voiced the reply, if any.
        tool_effect_unknown: True when a dispatched tool was still running at
            commit time, so its effect is not known; its `tool_calls` entry
            in the event row is marked `"effect_unknown": true`.
    """

    outcome: Literal["completed", "interrupted", "superseded", "error"]
    delivered_text: str
    stt_model: str | None
    tts_model: str | None
    tool_effect_unknown: bool = False


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Result of `commit_turn`.

    With the event store disabled (`handler.event_store is None`): `recorded`
    False and `error` None, which is not a failure. A failed write: `recorded`
    False and `error` set, which the controller reports as `record_failed`.
    """

    recorded: bool
    event_id: str | None
    error: str | None


@dataclass(slots=True)
class StreamTurnState:
    """Mutable accumulator threaded through both LLM passes of one streaming turn.

    Async generators cannot `return` a value (PEP 525 forbids `return expr`
    inside one), so `ConversationHandler._stream_llm_pass` communicates its
    result by mutating an instance of this class rather than returning one.

    `emitted` is the turn-wide concatenation of every Token piece yielded so
    far, across both passes -- it becomes `Complete.final_response` verbatim
    (T3/v2 contract: final_response is exactly what was streamed, not a
    separately post-filtered copy of it).

    The `pass_*` dicts are keyed by pass_num (1 or 2) and reconstruct what
    the pre-streaming code carried in a single `LLMResponse` per pass, built
    from what actually streamed and passed the slop gate. A pass's keys are
    written only when the pass finishes, so a cancelled pass leaves none.
    """

    emitted: str = ""
    pass_text: dict[int, str] = field(default_factory=dict)
    pass_tool_calls: dict[int, list[ToolCall] | None] = field(default_factory=dict)
    pass_finish_reason: dict[int, str | None] = field(default_factory=dict)
    pass_usage: dict[int, UsageMetadata | None] = field(default_factory=dict)
    pass_reasoning: dict[int, str | None] = field(default_factory=dict)
    pass_ttft_ms: dict[int, float | None] = field(default_factory=dict)
    pass_duration_ms: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class TurnScope:
    """State owned by exactly one conversation turn.

    Mutable, unlike the request-side value objects, because the pipeline
    appends to it while the turn runs. It is created per turn and never
    shared between turns, so no locking is needed.

    Attributes:
        session_id: Session the turn belongs to.
        turn_index: The handler's turn counter at the start of this turn.
            One counter per handler across all sessions, as before T4, so
            seeds match the pre-T4 handler; a turn in another session
            advances it.
        ws_events: ADR-017 FE-bound event payloads (tool_call_started and
            tool_call_completed, cards_summon and cards_dismiss,
            graph_subgraph, vault_results, form_switch) appended during tool
            dispatch and drained by the streaming pipeline as `WSEvent`s.
        request: The request the turn was opened with, interruption note
            included. None only for a scope built outside `open_turn`.
        dispatch_gate: Set by `open_turn` unless the request is held.
        tool_dispatched: True once any tool call has been dispatched. A turn
            that dispatched a tool is committed even when superseded,
            because the tool may have had effects.
        tool_calls: `{"name", "args"}` per dispatched tool call, in dispatch
            order, appended at dispatch time so a cancelled turn still lists
            a tool that was running.
        tool_results: `{"name", "result", "tool_call_id"}` per finished tool
            call, in dispatch order.
        prompt_stats: Prompt-size statistics for the turn. None until the
            prompt-layout work fills it.
        error: The error message when the stream ended on its error path,
            else None.
        stream: Per-pass accumulator the stream writes as it goes.
        context_window: The message list sent to the LLM, kept for the
            event-log row.
        retrieval_result: The auto-inject retrieval result, kept for the
            event-log row.
        commit_result: The first `commit_turn` result; a second commit
            returns it and writes nothing.
    """

    session_id: str
    turn_index: int
    ws_events: list[dict[str, object]] = field(default_factory=list)
    request: TurnRequest | None = None
    dispatch_gate: asyncio.Event = field(default_factory=asyncio.Event)
    tool_dispatched: bool = False
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    tool_results: list[dict[str, object]] = field(default_factory=list)
    prompt_stats: object | None = None
    error: str | None = None
    stream: StreamTurnState = field(default_factory=StreamTurnState)
    context_window: list[dict[str, object]] | None = None
    retrieval_result: RetrievalResult | None = None
    commit_result: CommitResult | None = None

    @property
    def generated_text(self) -> str:
        """Every Token piece the stream has yielded so far, across both passes."""
        return self.stream.emitted

    @property
    def graph_seed(self) -> str:
        """Seed for the graph_subgraph distant-point layout.

        Session plus turn index, so a repeated query within one turn lays out
        identically and different turns lay out differently.
        """
        return f"{self.session_id}:{self.turn_index}"

    def emit(self, payload: dict[str, object]) -> None:
        """Append one FE-bound event payload to this turn's buffer."""
        self.ws_events.append(payload)

    def drain_ws_events(self) -> list[dict[str, object]]:
        """Return the buffered payloads in emit order and empty the buffer.

        No await sits between the read and the reset, so a concurrent task
        cannot append to the list being returned.
        """
        events = self.ws_events
        self.ws_events = []
        return events
