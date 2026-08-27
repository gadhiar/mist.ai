"""B2, the wiring half: `_record_turn_event` must stop bypassing the clock seam.

`ConversationHandler` has had an injectable clock (`self._now_fn`,
`conversation_handler.py:768`) since the replay work, and uses it for the
user-snapshot `rendered_at`. `_record_turn_event` did not: it read
`datetime.now(UTC)` directly, three lines under a comment stating the value
must never be a wall-clock read at write time.

These tests pin three things, in increasing order of what they cost if wrong:

1. Production behaviour is unchanged -- the seam's default is still wall-clock.
2. An injected `now_fn` now reaches the turn event, which is what closes the
   bypass and is worth having independently of hydration.
3. A hydration clock stamps the CORPUS's authored timestamp, and the resulting
   event carries it into the event store -- where both sides of
   `live == rebuilt` read it back from.

Point 3 is the one with the dangerous failure direction. If the stamp were
wall-clock, both sides of the gate would agree and it would pass GREEN over a
timeline that never existed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.chat.hydration_clock import HydrationClock, HydrationClockError
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.unit.conftest import make_test_conventions_loader

_AUTHORED = datetime(2025, 9, 2, 8, 0, tzinfo=UTC)
_PINNED = datetime(2020, 1, 1, 12, 0, tzinfo=UTC)


class _FakeExtractionPipeline:
    async def extract_from_utterance(self, *a, **k):  # pragma: no cover
        return None


def _handler(*, now_fn=None, hydration_clock=None, event_store=True):
    config = build_test_config(event_store_enabled=event_store, event_store_db_path=":memory:")
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=_FakeExtractionPipeline(),
        retriever=KnowledgeRetriever(config=config, graph_store=gs),
        llm_provider=FakeLLM(),
        conventions_loader=make_test_conventions_loader(),
        now_fn=now_fn,
        hydration_clock=hydration_clock,
    )


class TestProductionUnchanged:
    def test_default_stamps_wall_clock(self):
        """Adding the seam must not change what live does."""
        before = datetime.now(UTC)
        handler = _handler()
        _, recorded_at = handler._record_turn_event(
            session_id="s1", user_message="hi", assistant_message="hey"
        )
        after = datetime.now(UTC)
        assert before <= datetime.fromisoformat(recorded_at) <= after

    def test_no_clock_means_no_hydration_dependency(self):
        """A handler built without a clock must not reference one."""
        assert _handler()._hydration_clock is None


class TestSeamBypassClosed:
    def test_an_injected_now_fn_reaches_the_turn_event(self):
        """The bypass itself, independent of hydration.

        Before B2 this failed: `_record_turn_event` read the wall clock no
        matter what was injected, so MIST_FIXED_CLOCK could not pin the turn
        event even though it pinned the user snapshot on the same handler.
        """
        handler = _handler(now_fn=lambda: _PINNED)
        _, recorded_at = handler._record_turn_event(
            session_id="s1", user_message="hi", assistant_message="hey"
        )
        assert datetime.fromisoformat(recorded_at) == _PINNED


class TestHydrationClockWins:
    def test_the_authored_timestamp_is_stamped(self):
        clock = HydrationClock(timestamps={("s1", 0): _AUTHORED}, source_path="fixture")
        handler = _handler(hydration_clock=clock)
        _, recorded_at = handler._record_turn_event(
            session_id="s1", user_message="hi", assistant_message="hey"
        )
        assert datetime.fromisoformat(recorded_at) == _AUTHORED

    def test_it_beats_an_injected_now_fn(self):
        """Both seams present: the corpus is the more specific authority."""
        clock = HydrationClock(timestamps={("s1", 0): _AUTHORED}, source_path="fixture")
        handler = _handler(now_fn=lambda: _PINNED, hydration_clock=clock)
        _, recorded_at = handler._record_turn_event(
            session_id="s1", user_message="hi", assistant_message="hey"
        )
        assert datetime.fromisoformat(recorded_at) == _AUTHORED

    def test_the_authored_stamp_reaches_the_event_store(self):
        """Where the gate reads it back from -- a stamp that stops at the
        return value would leave the timeline wrong in the store regardless.
        """
        clock = HydrationClock(timestamps={("s1", 0): _AUTHORED}, source_path="fixture")
        handler = _handler(hydration_clock=clock)
        handler._record_turn_event(session_id="s1", user_message="hi", assistant_message="hey")
        turns = handler.event_store.get_all_turns_for_reextraction()
        assert datetime.fromisoformat(turns[-1]["timestamp"]) == _AUTHORED

    def test_successive_turns_advance_with_the_corpus(self):
        """The gap ladder, end to end.

        `turn_index` comes from the event store's `turn_count`, so this also
        pins that the two agree -- the exact property the fail-closed miss
        below exists to police.
        """
        second = datetime(2026, 7, 15, 10, 0, tzinfo=UTC)
        clock = HydrationClock(
            timestamps={("s1", 0): _AUTHORED, ("s1", 1): second}, source_path="fixture"
        )
        handler = _handler(hydration_clock=clock)
        _, first_iso = handler._record_turn_event(
            session_id="s1", user_message="a", assistant_message="b"
        )
        _, second_iso = handler._record_turn_event(
            session_id="s1", user_message="c", assistant_message="d"
        )
        assert datetime.fromisoformat(first_iso) == _AUTHORED
        assert datetime.fromisoformat(second_iso) == second
        assert (datetime.fromisoformat(second_iso) - datetime.fromisoformat(first_iso)).days > 300

    def test_a_corpus_miss_raises_rather_than_stamping_wall_clock(self):
        """`_record_turn_event` swallows failures by design -- this must not be swallowed.

        The method's contract is that event-store failures never break the
        conversation. A missing authored timestamp is a different thing: it
        means the run has desynced from the corpus, and continuing would write
        a plausible-looking false timeline. It has to surface.
        """
        clock = HydrationClock(timestamps={("s1", 0): _AUTHORED}, source_path="fixture")
        handler = _handler(hydration_clock=clock)
        handler._record_turn_event(session_id="s1", user_message="a", assistant_message="b")
        with pytest.raises(HydrationClockError):
            handler._record_turn_event(session_id="s1", user_message="c", assistant_message="d")
