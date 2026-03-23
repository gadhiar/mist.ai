"""Tests for SelfReflectionJob batch internal derivation."""

import json
from datetime import UTC, datetime

import pytest

from backend.knowledge.curation.self_reflection import ReflectionResult, SelfReflectionJob
from backend.knowledge.extraction.internal_derivation import InternalKnowledgeDeriver
from backend.knowledge.extraction.signal_detector import SignalDetector
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM


class FakeEventStore:
    """Minimal event store fake that returns canned turns from get_turns_since."""

    def __init__(self, turns: list[dict] | None = None):
        self._turns = turns or []

    def get_turns_since(self, since: datetime) -> list[dict]:
        return self._turns


def _make_turn(
    *,
    utterance: str = "hello",
    response: str = "hi there",
    session_id: str = "sess-1",
    event_id: str = "evt-1",
) -> dict:
    """Build a conversation turn dict matching EventStore output format."""
    return {
        "event_id": event_id,
        "session_id": session_id,
        "user_utterance": utterance,
        "system_response": response,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _build_job(
    *,
    llm_response: str | None = None,
    event_store: FakeEventStore | None = None,
    connection: FakeNeo4jConnection | None = None,
) -> tuple[SelfReflectionJob, FakeLLM, FakeGraphExecutor]:
    """Build a SelfReflectionJob with test doubles."""
    conn = connection or FakeNeo4jConnection()
    executor = FakeGraphExecutor(conn)
    llm = FakeLLM(default_response=llm_response or '{"operations": []}')
    deriver = InternalKnowledgeDeriver(llm=llm, executor=executor)
    job = SelfReflectionJob(
        executor=executor,
        internal_deriver=deriver,
        signal_detector=SignalDetector(),
        event_store=event_store,
    )
    return job, llm, executor


class TestProcessRecentEvents:
    @pytest.mark.asyncio
    async def test_calls_deriver_for_each_event(self):
        turns = [
            _make_turn(
                utterance="I prefer concise answers",
                response="Noted.",
                event_id="evt-1",
            ),
            _make_turn(
                utterance="I like when you explain things clearly",
                response="Thank you.",
                event_id="evt-2",
            ),
        ]
        store = FakeEventStore(turns=turns)

        llm_resp = json.dumps(
            {
                "operations": [
                    {
                        "op": "CREATE_PREFERENCE",
                        "id": "pref-concise",
                        "display_name": "Concise Answers",
                        "description": "User prefers concise answers",
                        "confidence": 0.8,
                    },
                ]
            }
        )

        job, llm, _ = _build_job(llm_response=llm_resp, event_store=store)

        result = await job.run(lookback_hours=24)

        assert isinstance(result, ReflectionResult)
        assert result.events_processed == 2
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_skips_turns_without_utterance(self):
        turns = [
            _make_turn(utterance="", response="system boot"),
            _make_turn(utterance="hello", response="hi"),
        ]
        store = FakeEventStore(turns=turns)
        job, llm, _ = _build_job(event_store=store)

        result = await job.run()

        assert result.events_processed == 1

    @pytest.mark.asyncio
    async def test_aggregates_operations_across_events(self):
        turns = [
            _make_turn(
                utterance="I prefer detailed explanations when debugging",
                response="Understood, I will be more detailed.",
                event_id="evt-1",
            ),
        ]
        store = FakeEventStore(turns=turns)

        llm_resp = json.dumps(
            {
                "operations": [
                    {
                        "op": "CREATE_CAPABILITY",
                        "id": "cap-debug",
                        "display_name": "Debugging",
                        "description": "Good at debugging",
                        "confidence": 0.8,
                    },
                    {
                        "op": "CREATE_TRAIT",
                        "id": "trait-analytical",
                        "display_name": "Analytical",
                        "description": "Analytical thinker",
                        "confidence": 0.7,
                    },
                ]
            }
        )

        job, _, _ = _build_job(llm_response=llm_resp, event_store=store)

        result = await job.run()

        assert result.operations_applied == 2


class TestNoEventStore:
    @pytest.mark.asyncio
    async def test_returns_zero_result_when_no_event_store(self):
        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(conn)
        llm = FakeLLM()
        deriver = InternalKnowledgeDeriver(llm=llm, executor=executor)

        job = SelfReflectionJob(
            executor=executor,
            internal_deriver=deriver,
            signal_detector=SignalDetector(),
            event_store=None,
        )

        result = await job.run()

        assert result.events_processed == 0
        assert result.operations_applied == 0
        assert result.duration_ms == 0.0
        llm.assert_not_called()


class TestEmptyEvents:
    @pytest.mark.asyncio
    async def test_returns_zero_when_no_recent_events(self):
        store = FakeEventStore(turns=[])
        job, llm, _ = _build_job(event_store=store)

        result = await job.run()

        assert result.events_processed == 0
        assert result.operations_applied == 0
        assert result.duration_ms > 0
        llm.assert_not_called()
