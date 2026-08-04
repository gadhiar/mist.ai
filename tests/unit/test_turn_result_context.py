"""The turn-result side-channel must not be readable across turns (T0).

`last_complete` / `last_error` were instance attributes on
`KnowledgeIntegration`, which is a process-wide singleton reached through the
module-level `voice_processor`. They are per-TURN values, so an attribute is
the wrong shape: it is safe only while a lock invariant enforced in
`voice_processor.py` keeps the write and the read inside one hold.

These tests pin the ContextVar behaviour that replaces it: the default, the
generator-body-runs-in-caller's-context mechanism the whole design rests on,
and the producer-clears-before-yielding invariant that makes a consumer-side
reset unnecessary (see `backend/request_context.py` for why one was tried and
removed -- `KnowledgeIntegration.enabled` never flips at runtime, so no turn
can skip the producer after a prior turn set a value).
"""

from __future__ import annotations

import pytest

from backend.chat.knowledge_integration import KnowledgeIntegration
from backend.chat.stream_events import Complete
from backend.request_context import current_turn_complete, current_turn_error


@pytest.fixture(autouse=True)
def reset_turn_result_context():
    """Keep a leaked context var from one test satisfying the next.

    Matches the `reset_session_context` pattern in
    `test_session_id_propagation.py` / `test_singleton_session_state.py`.
    Every test in this file that drives a non-None value does so on the main
    pytest thread (unlike the production code, which always runs on a worker
    thread), so without this a value set here would otherwise leak into
    whichever test runs next.
    """
    complete_token = current_turn_complete.set(None)
    error_token = current_turn_error.set(None)
    yield
    current_turn_complete.reset(complete_token)
    current_turn_error.reset(error_token)


def test_context_vars_default_to_none():
    assert current_turn_complete.get() is None
    assert current_turn_error.get() is None


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


def test_the_producer_clears_both_vars_before_yielding_anything():
    """The invariant that makes a consumer-side reset unnecessary.

    The reset is the FIRST statement of the generator body, so it runs on the
    first `__next__()` -- before any item reaches the consumer, and therefore
    before the consumer's post-loop reads. If anyone moves it below a yield, a
    turn becomes able to read the previous turn's value, and this fails.

    Uses the disabled-knowledge branch deliberately: it exercises the reset
    with no handler, loop, or session, so the test pins ordering and nothing
    else.
    """
    ki = object.__new__(KnowledgeIntegration)
    ki.enabled = False
    ki.conversation_handler = None

    current_turn_complete.set(Complete(final_response="previous turn", tool_calls_used=9))
    current_turn_error.set(("server", "previous turn's error"))

    gen = ki.generate_response_streaming("hi", event_loop=None)
    # Generator bodies are lazy -- creating it must not have run anything yet.
    assert current_turn_error.get() == ("server", "previous turn's error")

    next(gen)

    assert current_turn_complete.get() is None
    assert current_turn_error.get() is None
