"""Catch-up must be cheap. Its cost is LLM calls, so the tests assert on calls."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.chat.session_synthesizer import SessionSynthesis
from backend.vault.session_catchup import SessionNoteCatchup

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


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
                "timestamp": "2026-07-29T10:00:00+00:00",
            },
            {
                "session_id": session_id,
                "user_utterance": "thanks, that helps",
                "system_response": "Any time.",
                "timestamp": "2026-07-29T10:05:00+00:00",
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


class _NoteStatusRegistry(dict):
    """Dict-like stand-in for `existing_note_status`.

    `SessionNoteCatchup._recorded_statuses` reads the vault directory
    directly (glob + frontmatter parse) rather than through an injected
    hook, so the only way to make an assignment like
    `h.existing_note_status["s-done"] = "completed"` visible to a
    subsequent `.run()` is to materialize a real note on disk at
    assignment time. The note's frontmatter carries exactly the two fields
    `_recorded_statuses` reads (`session_id`, `status`); the body content
    is irrelevant.
    """

    def __init__(self, sessions_dir: Path) -> None:
        super().__init__()
        self._sessions_dir = sessions_dir

    def __setitem__(self, session_id: str, status: str) -> None:
        super().__setitem__(session_id, status)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        note_path = self._sessions_dir / f"2026-07-29-{session_id}.md"
        note_path.write_text(
            f"---\nsession_id: {session_id}\nstatus: {status}\n---\n\nbody\n",
            encoding="utf-8",
        )


def _fake_session_path_for(session_id: str, first_utterance: str, date: str) -> str:
    return f"/vault/sessions/{date}-{session_id}.md"


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
    time.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    h = SimpleNamespace(
        event_store=_FakeEventStore(),
        synthesizer=_FakeSynthesizer(),
        writer=_FakeVaultWriter(),
        sessions_dir=sessions_dir,
        existing_note_status=_NoteStatusRegistry(sessions_dir),
        sessions_with_graph_state=set(),
        is_conversation_active=(lambda: False),
    )
    h.kwargs = {
        "event_store": h.event_store,
        "synthesizer": h.synthesizer,
        "vault_writer": h.writer,
        "vault_sessions_dir": sessions_dir,
        "sessions_with_graph_state": lambda: h.sessions_with_graph_state,
        "session_path_for": _fake_session_path_for,
        "is_conversation_active": lambda: h.is_conversation_active(),
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
    h.existing_note_status["s-done"] = "completed"

    await SessionNoteCatchup(**h.kwargs).run()

    assert h.synthesizer.calls == []


@pytest.mark.asyncio
async def test_skips_sessions_already_marked_skipped(catchup_harness):
    """Bounded retry: a session that failed twice is not reconsidered."""
    h = catchup_harness
    h.event_store.sessions = ["s-bad"]
    h.sessions_with_graph_state = {"s-bad"}
    h.existing_note_status["s-bad"] = "skipped"

    await SessionNoteCatchup(**h.kwargs).run()

    assert h.synthesizer.calls == []


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
