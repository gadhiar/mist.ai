"""The turn-result side-channel must not be readable across turns (T0).

`last_complete` / `last_error` were instance attributes on
`KnowledgeIntegration`, which is a process-wide singleton reached through the
module-level `voice_processor`. They are per-TURN values, so an attribute is
the wrong shape: it is safe only while a lock invariant enforced in
`voice_processor.py` keeps the write and the read inside one hold.

These tests pin the ContextVar behaviour that replaces it, including the case
the old `if knowledge else None` guard was quietly covering: a pooled executor
thread carrying a previous turn's value into a turn that never runs the
producer at all.
"""

from __future__ import annotations

import concurrent.futures

from backend.chat.stream_events import Complete
from backend.request_context import current_turn_complete, current_turn_error


def test_context_vars_default_to_none():
    assert current_turn_complete.get() is None
    assert current_turn_error.get() is None


def test_a_reused_pool_thread_carries_a_previous_turns_value():
    """The hazard the consumer-side reset exists to close.

    Not a bug in the ContextVars -- it is the documented behaviour of a
    reused executor thread, which is what `run_in_executor(None, ...)` uses.
    This test exists so that if someone later removes the turn-start reset in
    `_process_conversation_turn`, the reason it was there is stated in a
    failing test rather than lost.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        complete = Complete(final_response="prior turn", tool_calls_used=7)
        pool.submit(current_turn_complete.set, complete).result()
        leaked = pool.submit(current_turn_complete.get).result()
        assert leaked is complete, "expected the pool thread to retain the value"
    finally:
        pool.shutdown(wait=True)


def test_a_generator_body_sets_into_its_callers_context():
    """The mechanism the whole design rests on.

    The consumer drives the producer's generator, and a generator body runs in
    its CALLER's context -- so the producer's `.set()` is visible to the
    consumer without any shared object. If a future Python gives generators
    their own context (PEP 568, never implemented), this is the test that
    catches it, and the ContextVar approach would have to be revisited.
    """
    complete = Complete(final_response="from the generator", tool_calls_used=1)

    def producer():
        current_turn_complete.set(complete)
        yield "token"

    current_turn_complete.set(None)
    list(producer())
    try:
        assert current_turn_complete.get() is complete
    finally:
        current_turn_complete.set(None)
