"""Catch-up must be cheap. Its cost is LLM calls, so the tests assert on calls.

R1.3.1 fix round 1: dedup moved from frontmatter `session_id` matching (which
never actually worked -- the vault writer's frontmatter `session_id` is
always the path-derived slug, never any session identifier) to a direct
check of the note's own canonical path. The harness below seeds "existing
notes" by writing a real file at exactly the path `session_path_for` would
compute, mirroring how the production dedup check actually works now.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.chat.session_synthesizer import SessionSynthesis
from backend.vault.session_catchup import _MAX_ATTEMPTS_PER_PASS, SessionNoteCatchup

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

_DEFAULT_TURN_DATE = "2026-07-29"


class _FakeEventStore:
    """Stands in for `EventStore`. `sessions` is the harness's hook for
    `list_sessions_with_turns`; `get_turns` always returns two turns so a
    session that reaches synthesis has real transcript content to work with.
    """

    def __init__(self) -> None:
        self.sessions: list[str] = []

    def list_sessions_with_turns(self) -> list[str]:
        return list(self.sessions)

    def get_turns(self, session_id: str) -> list[dict]:
        # `session_id` rides along on each turn dict so `_FakeSynthesizer`
        # (which only receives `turns`, matching the real
        # `SessionSynthesizer.synthesize` signature) can still record which
        # session it was called for.
        return [
            {
                "session_id": session_id,
                "user_utterance": "tell me about the vault architecture",
                "system_response": "It has three layers.",
                "timestamp": f"{_DEFAULT_TURN_DATE}T10:00:00+00:00",
            },
            {
                "session_id": session_id,
                "user_utterance": "thanks, that helps",
                "system_response": "Any time.",
                "timestamp": f"{_DEFAULT_TURN_DATE}T10:05:00+00:00",
            },
        ]


class _FakeSynthesizer:
    """Records the session_id of every synthesize() call. `always_fail`
    simulates a synthesizer that never produces usable output (matching
    `SessionSynthesizer.synthesize`'s real "returns None on failure"
    contract -- catch-up must treat that the same as an exception).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.always_fail = False

    async def synthesize(self, turns: list[dict]) -> SessionSynthesis | None:
        self.calls.append(turns[0]["session_id"])
        if self.always_fail:
            return None
        return SessionSynthesis(
            title="Catch-up session",
            body="### What Was Accomplished\n- Discussed vault architecture\n",
        )


class _FakeVaultWriter:
    """Records every write_session_note call, mirroring the real
    `VaultWriter.write_session_note` keyword signature.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, dict]] = []

    async def write_session_note(
        self,
        vault_note_path: str,
        synthesis: SessionSynthesis | None,
        related_entities: list[str] | None = None,
        status: str = "completed",
    ) -> str:
        self.writes.append(
            (
                vault_note_path,
                {
                    "synthesis": synthesis,
                    "related_entities": related_entities,
                    "status": status,
                },
            )
        )
        return vault_note_path


def _make_session_path_for(sessions_dir: Path):
    """Build a `session_path_for` fake anchored inside the real `tmp_path`.

    Routes through the SAME directory `seed_existing_note` writes into, so
    the production path-based dedup check (which opens exactly the path
    this returns) can actually find a seeded note.
    """

    def _path_for(session_id: str, first_utterance: str, date: str) -> str:
        return str(sessions_dir / f"{date}-{session_id}.md")

    return _path_for


def _seed_existing_note(
    sessions_dir: Path,
    session_id: str,
    status: str,
    date: str = _DEFAULT_TURN_DATE,
    authored_by: str | None = None,
) -> None:
    """Write a minimal real note at the exact path `session_path_for` would
    compute for `session_id`. Deliberately carries no `session_id`
    frontmatter field -- the production dedup check no longer reads one
    (R1.3.1 fix round 1, C1); writing one here would risk a test that
    passes for the wrong reason, matching a shape the real writer never
    emits either.

    `authored_by` is omitted entirely when None rather than defaulting to
    `mist`, so the user-authoritative tests exercise a note shape that
    differs from the default in exactly one field.
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    note_path = sessions_dir / f"{date}-{session_id}.md"
    authored_line = f"authored_by: {authored_by}\n" if authored_by else ""
    note_path.write_text(f"---\nstatus: {status}\n{authored_line}---\n\nbody\n", encoding="utf-8")


async def _always_ready() -> bool:
    return True


async def _never_ready() -> bool:
    return False


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def catchup_harness(tmp_path: Path) -> SimpleNamespace:
    """Assemble fakes for every `SessionNoteCatchup` collaborator.

    `sessions_with_graph_state` and `is_conversation_active` are exposed on
    the harness as plain mutable attributes (a set and a callable
    respectively) and wired into `kwargs` via closures, so a test can
    reassign `h.sessions_with_graph_state = {...}` or
    `h.is_conversation_active = lambda: True` *after* the fixture runs and
    have `SessionNoteCatchup(**h.kwargs)` see the new value at `.run()`
    time. `is_llm_ready` is the same pattern but async.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    h = SimpleNamespace(
        event_store=_FakeEventStore(),
        synthesizer=_FakeSynthesizer(),
        writer=_FakeVaultWriter(),
        sessions_dir=sessions_dir,
        sessions_with_graph_state=set(),
        is_conversation_active=(lambda: False),
        is_llm_ready=_always_ready,
    )
    h.seed_existing_note = lambda session_id, status, authored_by=None: _seed_existing_note(
        sessions_dir, session_id, status, authored_by=authored_by
    )
    h.kwargs = {
        "event_store": h.event_store,
        "synthesizer": h.synthesizer,
        "vault_writer": h.writer,
        "sessions_with_graph_state": lambda: h.sessions_with_graph_state,
        "session_path_for": _make_session_path_for(sessions_dir),
        "is_conversation_active": lambda: h.is_conversation_active(),
        "is_llm_ready": lambda: h.is_llm_ready(),
    }
    return h


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skips_sessions_that_produced_no_graph_state(catchup_harness):
    """The largest efficiency win, reusing the gate that already existed:
    a session that put nothing in the graph has nothing worth remembering.
    """
    h = catchup_harness
    h.event_store.sessions = ["s-empty"]
    h.sessions_with_graph_state = set()  # s-empty produced nothing

    await SessionNoteCatchup(**h.kwargs).run()

    assert h.synthesizer.calls == [], "no LLM call may fire for a zero-extraction session"
    assert h.writer.writes == []


@pytest.mark.asyncio
async def test_synthesizes_a_session_that_produced_graph_state(catchup_harness):
    h = catchup_harness
    h.event_store.sessions = ["s-real"]
    h.sessions_with_graph_state = {"s-real"}

    await SessionNoteCatchup(**h.kwargs).run()

    assert h.synthesizer.calls == ["s-real"]
    assert len(h.writer.writes) == 1


@pytest.mark.asyncio
async def test_skips_sessions_that_already_have_a_completed_note(catchup_harness):
    h = catchup_harness
    h.event_store.sessions = ["s-done"]
    h.sessions_with_graph_state = {"s-done"}
    h.seed_existing_note("s-done", status="completed")

    await SessionNoteCatchup(**h.kwargs).run()

    assert h.synthesizer.calls == []


@pytest.mark.asyncio
async def test_skips_sessions_already_marked_skipped(catchup_harness):
    """Bounded retry: a session that failed twice is not reconsidered."""
    h = catchup_harness
    h.event_store.sessions = ["s-bad"]
    h.sessions_with_graph_state = {"s-bad"}
    h.seed_existing_note("s-bad", status="skipped")

    await SessionNoteCatchup(**h.kwargs).run()

    assert h.synthesizer.calls == []


@pytest.mark.parametrize("authored_by", ["user", "user-edit"])
@pytest.mark.asyncio
async def test_never_resynthesizes_a_user_authoritative_note(catchup_harness, authored_by):
    """A user-authoritative note is skipped on EVERY pass, not just the first.

    `VaultWriter.write_session_note` refuses to overwrite a `user` /
    `user-edit` note (ADR-010 Invariant 5), and that refusal is silent to
    catch-up -- no exception, no status written back. So a user-edited note
    left at a non-terminal `status` (`in-progress` here, the legacy default
    for a session that crashed before completion) would be re-synthesized on
    every `run_forever` tick forever if the gate only consulted `status`:
    a real LLM call each time, with the write refused each time.

    Asserting across THREE passes rather than one is the point. A gate that
    regressed to firing only once -- or one that consulted `status` alone --
    would still leave a single-pass assertion green.
    """
    h = catchup_harness
    h.event_store.sessions = ["s-user"]
    h.sessions_with_graph_state = {"s-user"}
    h.seed_existing_note("s-user", status="in-progress", authored_by=authored_by)

    catchup = SessionNoteCatchup(**h.kwargs)
    for _ in range(3):
        await catchup.run()

    assert h.synthesizer.calls == []
    assert h.writer.writes == []


@pytest.mark.asyncio
async def test_marks_skipped_after_repeated_synthesis_failure(catchup_harness):
    """The skip must persist in the vault so it survives a restart."""
    h = catchup_harness
    h.event_store.sessions = ["s-fails"]
    h.sessions_with_graph_state = {"s-fails"}
    h.synthesizer.always_fail = True

    await SessionNoteCatchup(**h.kwargs).run()

    statuses = [kw.get("status") for _, kw in h.writer.writes]
    assert "skipped" in statuses


@pytest.mark.asyncio
async def test_retries_exactly_the_bounded_number_of_times(catchup_harness):
    """I6: the prior test suite never asserted the retry count itself --
    a reviewer set _MAX_ATTEMPTS_PER_PASS to 50 and every test stayed
    green, a 25x cost regression that would have shipped undetected. `2`
    is hardcoded rather than compared against the module constant
    deliberately: comparing against the constant would make this test
    track any future change to it instead of catching one.
    """
    h = catchup_harness
    h.event_store.sessions = ["s-fails"]
    h.sessions_with_graph_state = {"s-fails"}
    h.synthesizer.always_fail = True

    await SessionNoteCatchup(**h.kwargs).run()

    assert h.synthesizer.calls.count("s-fails") == 2
    assert _MAX_ATTEMPTS_PER_PASS == 2, (
        "this constant changed -- update the hardcoded expectation above "
        "deliberately, don't just make it track the constant"
    )


@pytest.mark.asyncio
async def test_defers_while_a_conversation_is_active(catchup_harness):
    """Boot is exactly when the user is most likely to start talking, and a
    synthesis call holding the GPU would delay their first token.
    """
    h = catchup_harness
    h.event_store.sessions = ["s-real"]
    h.sessions_with_graph_state = {"s-real"}
    h.is_conversation_active = lambda: True

    await SessionNoteCatchup(**h.kwargs).run()

    assert h.synthesizer.calls == [], "catch-up must yield to live traffic"


@pytest.mark.asyncio
async def test_yields_mid_pass_when_a_conversation_starts_during_the_pass(catchup_harness):
    """The check above only proves the *combination* of the pass-start check
    and the mid-pass check yields -- either one alone satisfies it, since
    the pass-start check already returns before any candidate is examined.
    This test isolates the mid-pass check specifically: the conversation is
    NOT active when the pass starts (so candidate 1 is processed), then
    becomes active partway through (so candidate 2 must not be).
    """
    h = catchup_harness
    h.event_store.sessions = ["s-first", "s-second"]
    h.sessions_with_graph_state = {"s-first", "s-second"}
    call_count = {"n": 0}

    def _active() -> bool:
        call_count["n"] += 1
        # False for the pass-start check and the check before candidate 1;
        # True from the check before candidate 2 onward.
        return call_count["n"] > 2

    h.is_conversation_active = _active

    await SessionNoteCatchup(**h.kwargs).run()

    assert h.synthesizer.calls == ["s-first"], "candidate 2 must not run once a call starts"


@pytest.mark.asyncio
async def test_does_not_probe_llm_readiness_when_there_is_nothing_to_do(catchup_harness):
    """N2 (fix round 2): an empty backlog must not make an LLM health-check
    HTTP call every tick forever just to decide it has nothing to attempt.
    The candidate check must run BEFORE the readiness probe, not after.
    """
    h = catchup_harness
    h.event_store.sessions = []  # nothing to do
    probe_calls = {"n": 0}

    async def _counting_ready() -> bool:
        probe_calls["n"] += 1
        return True

    h.is_llm_ready = _counting_ready

    await SessionNoteCatchup(**h.kwargs).run()

    assert probe_calls["n"] == 0, "no LLM readiness probe may fire when there is nothing to do"


@pytest.mark.asyncio
async def test_defers_the_whole_pass_when_llm_is_not_ready(catchup_harness):
    """I4: a cold LLM at boot must be indistinguishable from "come back
    later," never from "synthesis genuinely failed." Distinct from the
    always_fail tests above: this asserts NEITHER a synthesis call NOR a
    skip-marker write happens -- a cold backend must not spend even one of
    the bounded retry attempts, or permanently mark the session skipped
    before the model has finished loading.
    """
    h = catchup_harness
    h.event_store.sessions = ["s-real"]
    h.sessions_with_graph_state = {"s-real"}
    h.is_llm_ready = _never_ready

    await SessionNoteCatchup(**h.kwargs).run()

    assert h.synthesizer.calls == [], "no synthesis attempt while the LLM is not ready"
    assert h.writer.writes == [], "must not persist a skip stub while merely waiting"


@pytest.mark.asyncio
async def test_a_later_pass_picks_up_a_session_deferred_for_llm_readiness(catchup_harness):
    """I5's underlying property, exercised directly on `run()`: a session
    deferred because the LLM was not ready must be picked up by a LATER
    pass, not permanently abandoned. This is what makes periodic retry
    (`run_forever`) actually solve the cold-boot problem rather than just
    delaying the same permanent-abandonment bug by one tick.
    """
    h = catchup_harness
    h.event_store.sessions = ["s-real"]
    h.sessions_with_graph_state = {"s-real"}

    h.is_llm_ready = _never_ready
    await SessionNoteCatchup(**h.kwargs).run()
    assert h.synthesizer.calls == []

    h.is_llm_ready = _always_ready
    await SessionNoteCatchup(**h.kwargs).run()
    assert h.synthesizer.calls == ["s-real"], "the session must not have been abandoned"


@pytest.mark.asyncio
async def test_run_forever_calls_run_repeatedly_until_cancelled(catchup_harness):
    """Structural check that `run_forever` is actually a loop and not a
    single pass -- I5 requires catch-up to retry later, which is
    meaningless if it only ever runs once. `run()` itself is stubbed out
    here (already covered by every other test in this file) so this test
    isolates looping behavior alone.
    """
    h = catchup_harness
    catchup = SessionNoteCatchup(**h.kwargs)
    call_count = {"n": 0}

    async def _counting_run() -> None:
        call_count["n"] += 1

    catchup.run = _counting_run  # type: ignore[method-assign]

    task = asyncio.create_task(catchup.run_forever(interval_seconds=0))
    for _ in range(200):
        if call_count["n"] >= 3:
            break
        await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert call_count["n"] >= 3, "run_forever must invoke run() more than once"


@pytest.mark.asyncio
async def test_one_bad_session_does_not_abort_the_rest_of_the_backlog(catchup_harness):
    """I1: `session_path_for` can raise (the real `VaultWriter.session_path`
    raises `ValueError` on a pathological slug -- reachable from an
    utterance whose significant tokens produce a double-hyphen). The live
    note-writing path survives this inside `handle_message`'s envelope;
    catch-up must not let one bad session take the rest of an unbounded
    backlog down with it.
    """
    h = catchup_harness
    h.event_store.sessions = ["s-bad-slug", "s-good"]
    h.sessions_with_graph_state = {"s-bad-slug", "s-good"}

    real_path_for = h.kwargs["session_path_for"]

    def _flaky_path_for(session_id: str, first_utterance: str, date: str) -> str:
        if session_id == "s-bad-slug":
            raise ValueError("session_slug must be lowercase kebab-case, got: '--'")
        return real_path_for(session_id, first_utterance, date)

    h.kwargs["session_path_for"] = _flaky_path_for

    await SessionNoteCatchup(**h.kwargs).run()

    assert h.synthesizer.calls == ["s-good"], "the good session must still be processed"
