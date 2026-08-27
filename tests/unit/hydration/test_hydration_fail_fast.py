"""Review fix: a failed turn must abort a hydration run, not be recorded as a real one.

Found by `/code-review high` on PR #6, and it is the most serious defect the
branch had -- it is the exact failure class the branch exists to prevent.

`handle_message`'s blanket handler catches any exception, builds
`"I encountered an error: ..."`, calls `_record_turn_event` ON THE ERROR PATH,
and returns that string normally. `run_chat` sees a string return, so
`ok=True`, `fail_count` stays 0, and `cmd_hydrate` prints "Complete."

Concretely: llama-server times out on corpus turn 40, and a fabricated
error-text turn lands in the event store stamped with turn 40's AUTHORED
timestamp (from the hydration clock) and `origin='real'` -- structurally
indistinguishable from a genuine turn. The `live == rebuilt` gate would then
compare it happily.

That behaviour is correct for live: an error turn is what the user actually
saw, and recording it keeps the transcript honest. It is pure contamination
during hydration. So the guard is gated on the hydration clock's presence
rather than changing live behaviour.

The same guard closes a second finding, reported independently by both
reviewers: `_record_turn_event`'s `except HydrationClockError: raise` was
being caught by this same blanket handler, which then called
`_record_turn_event` a SECOND time -- raising the same error from inside the
except block, masking the original traceback. The fail-loud the docstring
promised became "fail every remaining turn one at a time", costing an hour of
inference on an 87-turn corpus and polluting session history with error
messages that feed subsequent turns' LLM context.
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


class _BoomLLM(FakeLLM):
    """Fails the way llama-server does under load: the first invoke raises."""

    async def invoke(self, request):  # noqa: ARG002
        raise TimeoutError("llama-server did not respond")


class _FakeExtractionPipeline:
    async def extract_from_utterance(self, *a, **k):  # pragma: no cover
        return None


def _handler(*, hydration_clock=None, llm=None):
    config = build_test_config(event_store_enabled=True, event_store_db_path=":memory:")
    gs = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=_FakeExtractionPipeline(),
        retriever=KnowledgeRetriever(config=config, graph_store=gs),
        llm_provider=llm or FakeLLM(),
        conventions_loader=make_test_conventions_loader(),
        hydration_clock=hydration_clock,
    )


def _clock(timestamps=None):
    return HydrationClock(timestamps=timestamps or {("s1", 0): _AUTHORED}, source_path="fixture")


class TestLiveBehaviourUnchanged:
    """An error turn is what the user saw. Live must keep recording it."""

    @pytest.mark.asyncio
    async def test_a_failed_turn_returns_an_error_string(self):
        handler = _handler(llm=_BoomLLM())
        result = await handler.handle_message("hi", session_id="s1")
        assert "encountered an error" in result

    @pytest.mark.asyncio
    async def test_a_failed_turn_is_still_recorded_live(self):
        """The transcript stays honest when there is a human on the other end."""
        handler = _handler(llm=_BoomLLM())
        await handler.handle_message("hi", session_id="s1")
        assert (
            handler.event_store.get_all_turns_for_reextraction()
        ), "live must still record the error turn"


class TestHydrationAborts:
    @pytest.mark.asyncio
    async def test_a_failed_turn_raises_instead_of_returning(self):
        """No string return means `run_chat` cannot score it ok=True."""
        handler = _handler(hydration_clock=_clock(), llm=_BoomLLM())
        with pytest.raises(TimeoutError):
            await handler.handle_message("hi", session_id="s1")

    @pytest.mark.asyncio
    async def test_no_fabricated_turn_reaches_the_event_store(self):
        """THE finding. A turn written here is indistinguishable from a real one.

        It would carry the corpus's authored timestamp and origin='real', and
        the gate would compare it without complaint.
        """
        handler = _handler(hydration_clock=_clock(), llm=_BoomLLM())
        with pytest.raises(TimeoutError):
            await handler.handle_message("hi", session_id="s1")
        assert handler.event_store.get_all_turns_for_reextraction() == []

    @pytest.mark.asyncio
    async def test_the_original_exception_survives(self):
        """Not wrapped, not replaced by a masked second raise.

        Before the fix, a HydrationClockError raised in `_record_turn_event`
        was caught here and the handler called `_record_turn_event` again,
        raising the same error from inside the except block -- so the
        traceback an operator saw pointed at the recovery path rather than the
        original failure.
        """
        handler = _handler(hydration_clock=_clock(), llm=_BoomLLM())
        with pytest.raises(TimeoutError, match="llama-server did not respond"):
            await handler.handle_message("hi", session_id="s1")

    @pytest.mark.asyncio
    async def test_a_clock_miss_propagates_unmasked(self):
        """The second reviewer finding, from the other direction.

        A corpus/event-store desync must surface as HydrationClockError at the
        first divergent turn, not as an error-string return that lets the run
        continue through every remaining turn.
        """
        handler = _handler(hydration_clock=_clock({("other", 0): _AUTHORED}))
        with pytest.raises(HydrationClockError):
            await handler.handle_message("hi", session_id="s1")
