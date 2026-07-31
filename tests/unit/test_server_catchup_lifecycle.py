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
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging

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

    def test_catchup_shutdown_precedes_curation_and_conversation_handler(self):
        source = inspect.getsource(server.lifespan)
        # Split at the shutdown block's `yield`, not the "# Shutdown"
        # comment -- the explanatory comment above the catch-up shutdown
        # call itself mentions "curation_scheduler.stop()" and "ch.aclose()"
        # in prose, and an `await `-prefixed search alone still finds that
        # comment text if it happens to precede the yield (it does not
        # here, but searching only the post-yield slice is the more
        # robust anchor).
        shutdown_source = source.split("\n    yield\n", 1)[1]

        catchup_pos = shutdown_source.index("_shutdown_session_note_catchup")
        # `await `-prefixed, not the bare method call: the explanatory
        # comment directly above the catch-up shutdown call also contains
        # the bare "curation_scheduler.stop()" and "ch.aclose()" strings in
        # prose form, which would otherwise match first.
        curation_pos = shutdown_source.index("await curation_scheduler.stop()")
        aclose_pos = shutdown_source.index("await ch.aclose()")

        assert catchup_pos < curation_pos, (
            "catch-up must be cancelled and awaited before the curation " "scheduler stops"
        )
        assert catchup_pos < aclose_pos, (
            "catch-up must be cancelled and awaited before " "ConversationHandler.aclose()"
        )
