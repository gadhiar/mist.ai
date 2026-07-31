"""Unit tests for the session-note catch-up background-task lifecycle
wiring in backend.server -- scheduling at startup, cancellation at
shutdown.

R1.3.1 fix round 2 (N1): this wiring previously had zero test coverage,
even though the surrounding gates (SessionNoteCatchup itself) were heavily
tested. Two mutations against the live lifespan() code each left the full
unit suite at 2376 passed:

  - scheduling `catchup.run()` instead of `catchup.run_forever()` --
    I5's production effect (that catch-up ever retries a deferred pass)
    silently reverts.
  - removing `add_done_callback(_log_catchup_task_exception)` and moving
    `catchup_task.cancel()` back to the shutdown tail, unawaited -- I2 and
    I3 both silently revert together.

`lifespan()` itself cannot be driven directly in a test without faking the
entire model stack (VoiceProcessor.initialize, Neo4j, llama-server), so the
scheduling and shutdown logic were extracted into
`_schedule_session_note_catchup` / `_shutdown_session_note_catchup` --
small, pure functions testable in isolation. The one property extraction
alone cannot prove -- that the shutdown call is textually the FIRST
shutdown action, before anything that could race it -- is checked via
source inspection, the same technique already used for import-boundary
guards in tests/unit/vault/test_invalidation_bus.py and
test_filewatcher_graph_noop.py.

Final-review fix (C1, CRITICAL): production wired `is_conversation_active`
to `bool(catchup_handler.sessions)`. `ConversationHandler.sessions` only
ever grows in the production call graph (nothing removes from it on a
normal WebSocket disconnect), so after the FIRST conversation this
predicate is True forever -- `run()` returns at its first line for the
life of the process, nullifying `run_forever`'s entire retry purpose (I5).
The construction logic (which collaborator each callable is wired to) is
now also extracted, into `_build_session_note_catchup`, so this wiring is
directly testable the same way the scheduling/shutdown mechanics are.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from types import SimpleNamespace

import pytest

from backend import server


class _FakeCatchup:
    """Records which coroutine method was invoked, and how many times.

    `run_forever` mirrors the real method's shape: it can be made to raise
    (simulating a bug escaping its own internal exception handling) or,
    when it does not, loops forever until cancelled -- exactly the
    long-running-background-task shape `_schedule_session_note_catchup`
    is meant to manage.
    """

    def __init__(self, *, run_forever_raises: bool = False) -> None:
        self.run_calls = 0
        self.run_forever_calls = 0
        self._run_forever_raises = run_forever_raises

    async def run(self) -> None:
        self.run_calls += 1

    async def run_forever(self) -> None:
        self.run_forever_calls += 1
        if self._run_forever_raises:
            raise RuntimeError("simulated bug escaping run_forever's own handling")
        while True:
            await asyncio.sleep(3600)


class TestScheduleSessionNoteCatchup:
    @pytest.mark.asyncio
    async def test_schedules_run_forever_not_run(self):
        """I5 regression guard: catch-up must be scheduled to retry
        periodically, not fire once -- a single `catchup.run()` call would
        never pick a session deferred for a cold LLM or a live conversation
        back up.
        """
        catchup = _FakeCatchup()

        task = server._schedule_session_note_catchup(catchup)
        await asyncio.sleep(0)  # let the task start running
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert catchup.run_forever_calls == 1
        assert catchup.run_calls == 0

    @pytest.mark.asyncio
    async def test_attaches_a_callback_that_logs_an_escaping_exception(self, caplog):
        """I2 regression guard: an exception escaping run_forever's own
        internal handling must be logged, not surface only as "Task
        exception was never retrieved" at garbage collection.
        """
        catchup = _FakeCatchup(run_forever_raises=True)

        with caplog.at_level(logging.ERROR, logger="backend.server"):
            task = server._schedule_session_note_catchup(catchup)
            with contextlib.suppress(RuntimeError):
                await task

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any(
            "Session-note catch-up task failed" in msg for msg in error_messages
        ), f"Expected the done-callback to log the escaping exception. Got: {error_messages}"


class TestShutdownSessionNoteCatchup:
    @pytest.mark.asyncio
    async def test_cancels_and_awaits_the_task(self):
        """I3 regression guard: shutdown must not just call `.cancel()` and
        move on -- it must AWAIT cancellation completing, so nothing later
        in shutdown can race a still-resuming pass against an
        already-stopped vault_writer.
        """
        catchup = _FakeCatchup()
        task = server._schedule_session_note_catchup(catchup)
        await asyncio.sleep(0)  # let it actually start running

        await server._shutdown_session_note_catchup(task)

        assert task.done()
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_is_a_no_op_when_no_task_was_scheduled(self):
        """The vault layer can be disabled, in which case lifespan() never
        schedules a task and catchup_task is None; shutdown must not raise.
        """
        await server._shutdown_session_note_catchup(None)  # must not raise


class TestShutdownOrdering:
    """I3's other half: the shutdown CALL SITE, not just the extracted
    function's own behavior. Testing `_shutdown_session_note_catchup` in
    isolation (above) cannot catch a regression where the call is simply
    moved to the wrong place in `lifespan()`'s shutdown block -- the
    function would still behave correctly wherever it is called from.
    """

    def test_catchup_shutdown_is_the_first_awaiting_action_in_shutdown(self):
        """Catch-up's shutdown must be the FIRST awaiting action, not merely
        earlier than two named calls.

        Fix round 2's version of this test asserted only "before
        curation_scheduler.stop() and before ch.aclose()" -- which two
        round-3 reorderings satisfy while still breaking the property that
        matters:

        - Position kept, `await` dropped (replace `await
          _shutdown_session_note_catchup(...)` with
          `asyncio.create_task(_shutdown_session_note_catchup(...))`):
          textually still first, so the old assertions held, but shutdown
          no longer waits for cancellation -- the exact race I3 exists to
          prevent.
        - A new awaiting step inserted ABOVE the catch-up call: both old
          relations still held (catch-up was still before curation and
          aclose), but catch-up was no longer the FIRST shutdown action,
          reopening the yield window. Only two named calls were checked,
          so any future awaiting step added above catch-up would slip
          through silently -- the same shape as the original N1 gap, one
          level down.

        Asserting catch-up's `await` is the first `await ` in the shutdown
        block subsumes both former assertions (the first `await` position
        is necessarily <= the position of any other `await`, including the
        ones embedded in `await curation_scheduler.stop()` and `await
        ch.aclose()`) while also catching both reorderings above, which
        neither former assertion did.
        """
        source = inspect.getsource(server.lifespan)
        # Split at the shutdown block's `yield`, not the "# Shutdown"
        # comment -- the explanatory comment above the catch-up shutdown
        # call itself mentions "curation_scheduler.stop()" and "ch.aclose()"
        # in prose ("...both await, which yields...", comma not space, so
        # it does not collide with the `"await "` search below either).
        shutdown_source = source.split("\n    yield\n", 1)[1]

        assert shutdown_source.index("await ") == shutdown_source.index(
            "await _shutdown_session_note_catchup"
        ), "catch-up shutdown must be the first awaiting action in shutdown"


class _RecordingEventStore:
    """Records whether `list_sessions_with_turns` was ever reached.

    `SessionNoteCatchup.run()` checks `is_conversation_active()` before
    anything else and returns immediately if it is True -- `list_sessions_
    with_turns` is the very next thing `run()` calls if it is NOT deferred.
    So whether this was reached is a direct, public-API-level probe of
    what `is_conversation_active` actually evaluated to, without reaching
    into `SessionNoteCatchup`'s private attributes.
    """

    def __init__(self) -> None:
        self.list_sessions_with_turns_called = False

    def list_sessions_with_turns(self) -> list[str]:
        self.list_sessions_with_turns_called = True
        return []


class _FakeSynthesizer:
    async def is_ready(self) -> bool:
        return True


def _fake_voice_processor(handler) -> SimpleNamespace:
    return SimpleNamespace(
        models=SimpleNamespace(knowledge=SimpleNamespace(conversation_handler=handler))
    )


def _fake_handler() -> SimpleNamespace:
    """Minimal ConversationHandler-shaped double for `_build_session_note_catchup`.

    `sessions` is seeded non-empty by default -- exactly the regression
    shape: a conversation that ended but whose entry production never
    evicts from `ConversationHandler.sessions`. A correct wiring must
    ignore this entirely; the buggy wiring this replaces would defer
    forever because of it.
    """
    return SimpleNamespace(
        sessions={"leftover-from-an-ended-conversation": object()},
        event_store=_RecordingEventStore(),
        session_synthesizer=_FakeSynthesizer(),
        graph_store=SimpleNamespace(sessions_with_graph_state=lambda: set()),
        derive_session_note_path=lambda session_id, first_utterance, date: None,
    )


class TestBuildSessionNoteCatchup:
    """C1 (CRITICAL, final review): proves the WIRING, not just that some
    constructed `SessionNoteCatchup` behaves correctly when told to defer.
    """

    def test_returns_none_when_vault_writer_is_none(self):
        handler = _fake_handler()
        voice_processor = _fake_voice_processor(handler)

        assert server._build_session_note_catchup(voice_processor, None) is None

    def test_returns_none_when_no_conversation_handler_is_available(self):
        voice_processor = SimpleNamespace(models=None)

        assert server._build_session_note_catchup(voice_processor, vault_writer=object()) is None

    def test_returns_none_when_event_store_is_unavailable(self):
        handler = _fake_handler()
        handler.event_store = None
        voice_processor = _fake_voice_processor(handler)

        assert server._build_session_note_catchup(voice_processor, vault_writer=object()) is None

    @pytest.mark.asyncio
    async def test_is_conversation_active_reflects_websocket_connections_not_handler_sessions(
        self, monkeypatch
    ):
        """The core C1 regression guard. `handler.sessions` is non-empty
        (simulating the leftover-entry bug this wiring must NOT depend on)
        while there is no live WebSocket connection -- catch-up must still
        proceed past the conversation-active gate.
        """
        handler = _fake_handler()
        voice_processor = _fake_voice_processor(handler)
        monkeypatch.setattr(server, "active_connections", set())

        catchup = server._build_session_note_catchup(voice_processor, vault_writer=object())
        assert catchup is not None
        await catchup.run()

        assert handler.event_store.list_sessions_with_turns_called, (
            "is_conversation_active must reflect active_connections, not "
            "ConversationHandler.sessions (which only ever grows in production)"
        )

    @pytest.mark.asyncio
    async def test_defers_when_there_is_an_active_websocket_connection(self, monkeypatch):
        """Positive counterpart: a genuinely live connection must still defer."""
        handler = _fake_handler()
        handler.sessions = {}  # empty -- proves this is NOT what gates it either
        voice_processor = _fake_voice_processor(handler)
        monkeypatch.setattr(server, "active_connections", {object()})

        catchup = server._build_session_note_catchup(voice_processor, vault_writer=object())
        assert catchup is not None
        await catchup.run()

        assert not handler.event_store.list_sessions_with_turns_called

    @pytest.mark.asyncio
    async def test_full_wiring_reaches_synthesis_and_write_for_a_real_candidate(self, monkeypatch):
        """Closes the broader review concern ("no test asserts anything
        about what production passes as any of the four injected
        callables") for the other three: drives a full successful pass and
        checks the exact path and write that result, proving
        `session_path_for`, `sessions_with_graph_state`, and `is_llm_ready`
        are wired to the real collaborators, not just SOME collaborators.
        """

        class _EventStoreWithOneCandidate(_RecordingEventStore):
            def list_sessions_with_turns(self):
                super().list_sessions_with_turns()
                return ["s-1"]

            def get_turns(self, session_id):
                return [
                    {
                        "user_utterance": "hello",
                        "system_response": "hi",
                        "timestamp": "2026-07-30T10:00:00+00:00",
                    }
                ]

        class _RecordingVaultWriter:
            def __init__(self) -> None:
                self.writes: list[tuple[str, dict]] = []

            async def write_session_note(self, vault_note_path, synthesis, **kwargs):
                self.writes.append((vault_note_path, kwargs))
                return vault_note_path

        class _SucceedingSynthesizer:
            async def is_ready(self) -> bool:
                return True

            async def synthesize(self, turns):
                from backend.chat.session_synthesizer import SessionSynthesis

                return SessionSynthesis(title="T", body="### What Was Accomplished\n- x\n")

        handler = _fake_handler()
        handler.sessions = {}
        handler.event_store = _EventStoreWithOneCandidate()
        handler.session_synthesizer = _SucceedingSynthesizer()
        handler.graph_store = SimpleNamespace(sessions_with_graph_state=lambda: {"s-1"})
        handler.derive_session_note_path = (
            lambda session_id, first_utterance, date: f"/vault/sessions/{date}-{session_id}.md"
        )
        voice_processor = _fake_voice_processor(handler)
        vault_writer = _RecordingVaultWriter()
        monkeypatch.setattr(server, "active_connections", set())

        catchup = server._build_session_note_catchup(voice_processor, vault_writer)
        assert catchup is not None
        await catchup.run()

        assert vault_writer.writes == [
            ("/vault/sessions/2026-07-30-s-1.md", {"related_entities": []})
        ]
