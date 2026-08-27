"""Checks that must pass BEFORE a hydration run drives its first turn.

A hydration run costs 87 LLM turns. Every check here is one that would
otherwise surface partway through -- or, worse, not at all.

The ordering principle is the same one the closure design applies to the gate:
a run that fails at turn 60 has wasted an hour, but a run that COMPLETES while
quietly wrong has produced an artifact someone will compare against and
believe. Two of these four target the second case specifically.

Preconditions live here; `postconditions.py` holds what must be true after.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class HydrationPreflightError(RuntimeError):
    """Raised when a hydration run must not start."""


class _SessionReader(Protocol):
    def get_session(self, session_id: str) -> Any: ...


def assert_hydration_isolation() -> None:
    """Refuse unless this process is the hydration target.

    In-process hydration writes through the same code paths live traffic uses,
    so the only thing separating "hydrating the dev graph" from "writing 87
    fictional turns into live" is which container the command runs in. The dev
    stack sets MIST_HYDRATION_ISOLATION and the live stack never does.
    """
    from backend.knowledge.eval_isolation import is_hydration_isolation_active

    if not is_hydration_isolation_active():
        raise HydrationPreflightError(
            "MIST_HYDRATION_ISOLATION is not set, so this process is not the "
            "hydration target. Run inside mist-backend-dev (see "
            "docker-compose.dev-hydration.yml). Refusing: in-process hydration "
            "writes through the same paths live traffic uses, and the dev stack "
            "sets MIST_SESSION_ORIGIN=real, so contaminating turns would be "
            "indistinguishable from genuine usage."
        )


def assert_clock_present(clock: Any) -> None:
    """Refuse when no hydration clock was wired.

    THE gap this module exists for. Without a clock every turn is stamped with
    the wall clock, which collapses the corpus's authored gap ladder onto the
    moment the run happened -- and both sides of `live == rebuilt` then read
    that same flattened timeline back out of the event store, so the gate
    passes GREEN over a history that never existed.

    Nothing else catches this. The clock is optional by design (production and
    eval both run without one), so its absence is silent unless asserted here.
    """
    if clock is None:
        raise HydrationPreflightError(
            "No hydration clock is wired: MIST_HYDRATION_CLOCK is unset on this "
            "container. Every turn would be stamped with the wall clock, "
            "collapsing the corpus's authored timeline onto the moment of the "
            "run. The gate would still pass -- both sides read the timestamp "
            "back out of the event store -- which is exactly why this is "
            "refused rather than warned about. Set MIST_HYDRATION_CLOCK to the "
            "corpus path."
        )


def assert_clock_covers_corpus(clock: Any, rows: list[dict[str, Any]]) -> None:
    """Refuse unless the clock has a timestamp for every turn about to be driven.

    The clock raises on a miss, so an uncovered turn would fail the run anyway
    -- but at the turn it occurs, after everything before it has already been
    written and the LLM time spent. Checking totality up front turns a
    partial-graph failure an hour in into a refusal in milliseconds.

    Args:
        clock: A `HydrationClock`.
        rows: The corpus rows about to be driven, each with `session_id` and
            `turn_index`.
    """
    missing = [
        (row["session_id"], row["turn_index"])
        for row in rows
        if (str(row["session_id"]), int(row["turn_index"])) not in clock.timestamps
    ]
    if missing:
        shown = ", ".join(f"{s}#{t}" for s, t in missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise HydrationPreflightError(
            f"The hydration clock does not cover {len(missing)} of {len(rows)} "
            f"corpus turns: {shown}{more}. The clock was built from "
            f"{clock.source_path!r}. Driving anyway would write every covered "
            "turn and then fail partway, leaving a partial graph that is not a "
            "function of the whole corpus."
        )


def assert_sessions_sequential(rows: list[dict[str, Any]]) -> None:
    """Refuse unless each session's rows appear as turn_index 0, 1, 2, ... in FILE order.

    Found by the cloud review, and it is a fail-GREEN gap in the mechanism
    built to prevent fail-green.

    `assert_clock_covers_corpus` checks that every corpus key EXISTS in the
    clock. It says nothing about order. But the runtime lookup does not use the
    corpus row's `turn_index` at all -- `_record_turn_event` derives the key
    from the event store's `turn_count`, a sequential 0,1,2,... counter, and
    `run_replay` drives rows in FILE order. The two agree only when the corpus
    is authored in strict per-session order.

    Reshuffle a session's rows and every lookup still finds a real key, so the
    clock's fail-closed miss never fires: every turn gets a plausible authored
    timestamp attached to the wrong utterance. `recorded_at` is the fact-time
    authority for every bitemporal edge the turn produces, so the error
    propagates into valid-time, supersession ordering and currency -- and both
    sides of the gate then read those same false stamps back out of the event
    store and agree.

    The cloud review rated this a nit because today's golden log happens to be
    strictly ascending. That is a property of the current DATA, not of the
    code, and MIS-134's work will author new corpora.

    Raises:
        HydrationPreflightError: naming the session, the expected index and
            what was found, since a corpus author needs all three.
    """
    seen: dict[str, int] = defaultdict(int)
    for row in rows:
        session = str(row["session_id"])
        expected = seen[session]
        actual = int(row["turn_index"])
        if actual != expected:
            raise HydrationPreflightError(
                f"Session {session!r} is out of order: expected turn_index={expected} "
                f"at this position, found {actual}. The runtime keys the hydration "
                "clock on the event store's sequential turn_count, not on the "
                "corpus's turn_index, so rows must appear in strict per-session "
                "order starting from 0. Out of order, every lookup still finds a "
                "real key and every turn gets the WRONG authored timestamp -- which "
                "both sides of the gate would then agree on."
            )
        seen[session] += 1


def assert_event_store_present(event_store: Any) -> None:
    """Refuse when the conversation handler has no event store.

    `ConversationHandler.__init__` sets `self.event_store = None` on ANY
    initialization exception, logging an error and continuing. In that state a
    hydration run burns every corpus turn through full inference, writes
    nothing (`_record_turn_event` returns early, so no extraction task is even
    spawned), skips `assert_sessions_unused`, skips the curation postcondition,
    and exits 0 reporting "Complete".

    That is the silent-green shape this module exists to refuse, arrived at
    from a direction none of the other checks cover.
    """
    if event_store is None:
        raise HydrationPreflightError(
            "The conversation handler has no event store. `ConversationHandler` "
            "degrades to `event_store = None` on any init failure and logs an "
            "ERROR rather than raising, so this is usually a misconfigured "
            "EVENT_STORE_DB_PATH or an unwritable dev-state volume -- check the "
            "backend log for 'Failed to initialize event store'. Refusing: a run "
            "in this state would drive every turn through inference, write "
            "nothing, and report success."
        )


def assert_sessions_unused(event_store: _SessionReader, rows: list[dict[str, Any]]) -> None:
    """Refuse when any corpus session already has turns in the event store.

    The clock keys on `(session_id, turn_index)`, and `turn_index` comes from
    the event store's `turn_count` -- not from the corpus. Those agree only
    when each session starts empty. Re-running hydration onto a non-empty store
    would start session `golden-2025-09` at turn_index 3 instead of 0, and
    every subsequent lookup would miss.

    The clock's fail-closed miss would catch that on the first turn, so this
    check is about the MESSAGE more than the safety: "session already has 3
    turns, wipe dev-state first" is actionable, where "no authored timestamp
    for turn_index=3" sends someone to inspect the corpus.
    """
    used = []
    for session_id in dict.fromkeys(str(row["session_id"]) for row in rows):
        session = event_store.get_session(session_id)
        if session is not None and getattr(session, "turn_count", 0) > 0:
            used.append((session_id, session.turn_count))
    if used:
        detail = ", ".join(f"{s} ({n} turns)" for s, n in used)
        raise HydrationPreflightError(
            f"Corpus sessions already carry turns in this event store: {detail}. "
            "turn_index is read from the store's turn_count, so a non-empty "
            "session shifts every clock lookup and the run would fail on its "
            "first turn. Reset the dev state (`rm -rf dev-state`, then bring the "
            "dev stack back up) before re-running hydration."
        )


def run_all(*, clock: Any, rows: list[dict[str, Any]], event_store: _SessionReader | None) -> None:
    """Run every precondition, cheapest and most-fundamental first.

    Order is deliberate: isolation before anything that could write, clock
    presence before clock coverage (coverage on a None clock is a TypeError,
    not a diagnosis), ordering after coverage (an uncovered key is the simpler
    diagnosis of the two), and the store checks last since they are the only
    ones that touch I/O.

    A missing event store is now a REFUSAL rather than a skipped check. It was
    the latter, which meant an init failure silently disabled the two checks
    most likely to catch the resulting mess.
    """
    assert_hydration_isolation()
    assert_clock_present(clock)
    assert_clock_covers_corpus(clock, rows)
    assert_sessions_sequential(rows)
    assert_event_store_present(event_store)
    assert_sessions_unused(event_store, rows)
    logger.info("Hydration preflight passed: %d turns, clock %s", len(rows), clock.source_path)
