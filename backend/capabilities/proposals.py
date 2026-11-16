"""Cross-turn confirmation: the proposal store and the yes-or-no matcher.

A turn cannot wait for the user's "yes": a turn holds `generation_lock` while it generates, and
an utterance that arrives meanwhile is queued rather than processed (`voice_processor.py`). So
a call that needs confirmation is stored as a proposal, the turn ends, and a later utterance or
an `action_decision` message resolves it. A spoken "yes" and a button press can race on the main
loop, so `ProposalStore.pop` is synchronous: whichever runs first takes the proposal, and the
other finds nothing to run.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from backend.errors import CapabilityInvocationError

from .clock import Clock
from .contracts import RiskClass

logger = logging.getLogger(__name__)

Decision = Literal["approve", "reject"]
_Fate = Literal["resolved", "superseded", "expired"]

# How many finished proposal ids the store remembers, so a late `pop` of a resolved id answers
# None rather than "unknown". An older id is forgotten and then reads as unknown.
_FINISHED_MEMORY = 1024

_APPROVE = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "confirm",
        "do it",
        "go ahead",
        "please do",
        # `normalize_utterance` strips a leading "please", so the listed approval "please do"
        # reaches the matcher as "do". Accepting the residue keeps that phrase approving.
        "do",
    }
)
_REJECT = frozenset({"no", "nope", "cancel", "stop", "dont", "never mind"})
_FILLERS = frozenset({"please", "thanks", "thank you", "it", "that", "now"})
_HEDGES = frozenset(
    {"but", "wait", "actually", "instead", "only", "except", "not", "hold on"} | _REJECT
)
_CONNECTIVES = frozenset({"and", "then", "also"})
# Every listed phrase has two words, so taking a two-word phrase before a single word is
# longest-first.
_PHRASES = frozenset(phrase for phrase in _APPROVE | _REJECT | _FILLERS | _HEDGES if " " in phrase)


@dataclass(frozen=True, slots=True)
class ProposedCall:
    """One validated call waiting for confirmation.

    Attributes:
        capability: The capability that runs the call.
        tool: The tool name, for example `"lights.power"`.
        args: The validated arguments; approval runs exactly these.
        summary: The tool's `summarize` output ("turn off the bedroom lights").
        risk: Shown with the proposal.
    """

    capability: str
    tool: str
    args: BaseModel
    summary: str
    risk: RiskClass


@dataclass(frozen=True, slots=True)
class Proposal:
    """Every call of one turn that needs confirmation, asked about once.

    Attributes:
        proposal_id: Names the proposal in `action_decision` and `action_resolved`.
        session_id: Only this session may resolve it.
        calls: Run in this order on approval.
        summary: The call summaries joined with ", then ".
        expires_at: Aware UTC; after it the proposal can no longer be approved.
    """

    proposal_id: str
    session_id: str
    calls: tuple[ProposedCall, ...]
    summary: str
    expires_at: datetime


class DecisionMatcher:
    """Reads a yes or a no to a live proposal from `normalize_utterance` output.

    It is deliberately narrow. A proposal runs only on an explicit yes, so anything mixed or
    hedged ("okay, never mind", "yeah but only the lamp") is not a decision; the design has
    such an utterance supersede the proposal rather than run it.
    """

    def match(self, normalized: str) -> tuple[Decision, str | None] | None:
        """Classify `normalized` as an approval, a rejection or neither.

        The text is split left to right into tokens, a listed two-word phrase ("do it", "never
        mind") counting as one. It approves when every token is an approve word or a filler
        and one is an approve word, or when the first token is an approve word, the second is
        "and", "then" or "also", and the rest holds no hedge or negation word. It rejects when
        every token is a reject word or a filler and one is a reject word.

        Args:
            normalized: The output of `normalize_utterance`.

        Returns:
            `(decision, remainder)`, where the remainder is the text after a leading approval
            and its connective ("yes, and dim them" gives `"dim them"`), or None when nothing
            follows the connective or there is no connective; or None when the text is not a
            decision.
        """
        tokens = _tokenise(normalized)
        if not tokens:
            return None
        if _all_of(tokens, _APPROVE):
            return ("approve", None)
        if _all_of(tokens, _REJECT):
            return ("reject", None)
        if len(tokens) >= 2 and tokens[0] in _APPROVE and tokens[1] in _CONNECTIVES:
            rest = tokens[2:]
            if not any(token in _HEDGES for token in rest):
                return ("approve", " ".join(rest) or None)
        return None


def _tokenise(text: str) -> list[str]:
    words = text.split()
    tokens: list[str] = []
    index = 0
    while index < len(words):
        pair = " ".join(words[index : index + 2])
        if pair in _PHRASES:
            tokens.append(pair)
            index += 2
        else:
            tokens.append(words[index])
            index += 1
    return tokens


def _all_of(tokens: list[str], words: frozenset[str]) -> bool:
    return all(t in words or t in _FILLERS for t in tokens) and any(t in words for t in tokens)


class ProposalStore:
    """Holds at most one live proposal per session, and expires it on a timer.

    `put` and `pop` must run on the event loop thread: `put` schedules the expiry timer as a
    task there, and both rely on running without an await to stay atomic.

    Expiry calls `on_expire` exactly once per expired proposal, so the caller can announce it
    (`action_resolved` with `expired`) without a turn running. Usually the timer task makes the
    call. But a `pop` that finds a proposal past its deadline before the timer has run expires
    it there, so a proposal is never approved late, and then `on_expire` runs synchronously
    inside that `pop`. `on_expire` must therefore be safe to call from inside `pop`: it must
    not await, and must not call back into the store expecting the proposal to be live.

    The store remembers the last 1024 finished ids. A resolved or superseded id older than
    that is forgotten, and `pop` then raises "unknown" for it rather than returning None.
    """

    def __init__(self, clock: Clock, ttl_s: float, on_expire: Callable[[Proposal], None]) -> None:
        """Create an empty store.

        Args:
            clock: Supplies `now()` for deadlines and `sleep` for the expiry timer.
            ttl_s: Seconds a proposal stays live (`MIST_CAPABILITY_PROPOSAL_TTL_S`).
            on_expire: Called once with each proposal that expires unresolved, from the timer
                task or synchronously inside `pop` (class docstring).
        """
        self._clock = clock
        self._ttl = timedelta(seconds=ttl_s)
        self._ttl_s = ttl_s
        self._on_expire = on_expire
        self._live: dict[str, Proposal] = {}
        self._timers: dict[str, asyncio.Task[None]] = {}
        # Every timer task not yet ended, whether or not its proposal is still live: `_finish`
        # pops a cancelled timer from `_timers` before it has unwound, and `close` waits for it.
        self._tasks: set[asyncio.Task[None]] = set()
        self._finished: OrderedDict[str, tuple[str, _Fate]] = OrderedDict()
        self._closed = False

    def put(self, session_id: str, calls: Sequence[ProposedCall]) -> Proposal:
        """Store a turn's calls as one proposal, superseding the session's live one.

        Args:
            session_id: The session that must resolve it.
            calls: The turn's calls that need confirmation, in the order they were chosen.

        Returns:
            The new proposal.

        Raises:
            CapabilityInvocationError: The store is closed; nothing changed and no timer started.
            ValueError: `calls` is empty.
            RuntimeError: No event loop is running in this thread, so no timer can start.
        """
        if self._closed:
            raise CapabilityInvocationError("proposal store is closed")
        if not calls:
            raise ValueError("a proposal needs at least one call")
        loop = asyncio.get_running_loop()
        previous = self._live.pop(session_id, None)
        if previous is not None:
            self._finish(previous, "superseded")
        proposal = Proposal(
            proposal_id=uuid.uuid4().hex,
            session_id=session_id,
            calls=tuple(calls),
            summary=", then ".join(call.summary for call in calls),
            expires_at=self._clock.now() + self._ttl,
        )
        self._live[session_id] = proposal
        timer = loop.create_task(self._expire_after_ttl(proposal))
        timer.add_done_callback(self._on_timer_done)
        timer.add_done_callback(self._tasks.discard)
        self._timers[proposal.proposal_id] = timer
        self._tasks.add(timer)
        return proposal

    def pop(self, session_id: str, proposal_id: str | None) -> Proposal | None:
        """Take the session's live proposal, so it can run or be dropped exactly once.

        Args:
            session_id: The caller's session; another session's proposal is never returned.
            proposal_id: The proposal the caller means, or None for whatever is live (a
                spoken decision names no id).

        Returns:
            The proposal, now removed; or None when nothing is live, when the id was already
            resolved or superseded, when it belongs to another session, or for any session and
            id once the store is closed.

        Raises:
            CapabilityInvocationError: `proposal_id` has expired, or is unknown to the store.
        """
        if self._closed:
            return None
        live = self._live.get(session_id)
        if live is not None and self._clock.now() >= live.expires_at:
            self._expire(live)
            live = None
        if proposal_id is None or (live is not None and live.proposal_id == proposal_id):
            if live is None:
                return None
            del self._live[session_id]
            self._finish(live, "resolved")
            return live
        for other in self._live.values():
            if other.proposal_id == proposal_id:
                return None
        finished = self._finished.get(proposal_id)
        if finished is None:
            raise CapabilityInvocationError(f"proposal {proposal_id} is unknown")
        owner, fate = finished
        if owner != session_id:
            return None
        if fate == "expired":
            raise CapabilityInvocationError(f"proposal {proposal_id} has expired")
        return None

    async def close(self) -> None:
        """Drop every live proposal and wait for every expiry timer to end.

        Before its first await it marks the store closed, cancels each live proposal's timer and
        drops the proposals without calling `on_expire`: shutting down is not an expiry, so
        nothing is announced or counted. A timer cancelled at its sleep, or whose sleep had
        finished but which had not yet run, raises `CancelledError` at that sleep when it
        resumes, so none reaches its expiry check. It then waits, without a bound, for every
        timer the store created that has not ended, including one a `pop` cancelled that has
        not yet unwound; a timer's only await is the clock's sleep, so a cancelled one always
        ends.

        After close, `put` raises and `pop` returns None. A second call returns at once and
        logs nothing.
        """
        if self._closed:
            return
        self._closed = True
        dropped = len(self._live)
        self._live.clear()
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()
        logger.info("Proposal store closed; dropped %d live proposal(s)", dropped)
        pending = [task for task in self._tasks if not task.done()]
        # `asyncio.wait` rather than `gather` or `await task`, which would re-raise a timer's
        # `CancelledError` here; and it is skipped when empty, as `wait` refuses an empty set.
        if pending:
            await asyncio.wait(pending)

    async def _expire_after_ttl(self, proposal: Proposal) -> None:
        await self._clock.sleep(self._ttl_s)
        if self._live.get(proposal.session_id) is proposal:
            self._expire(proposal)

    def _expire(self, proposal: Proposal) -> None:
        del self._live[proposal.session_id]
        self._finish(proposal, "expired")
        self._on_expire(proposal)

    def _finish(self, proposal: Proposal, fate: _Fate) -> None:
        timer = self._timers.pop(proposal.proposal_id, None)
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()
        self._finished[proposal.proposal_id] = (proposal.session_id, fate)
        while len(self._finished) > _FINISHED_MEMORY:
            self._finished.popitem(last=False)

    @staticmethod
    def _on_timer_done(task: asyncio.Task[None]) -> None:
        # The timer runs outside any turn, so an exception from `on_expire` would otherwise be
        # dropped with the task. Log it with its traceback instead.
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "proposal expiry callback failed",
                exc_info=(type(error), error, error.__traceback__),
            )
