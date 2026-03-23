"""Tests for InternalKnowledgeDeriver."""

import json

import pytest

from backend.knowledge.extraction.internal_derivation import InternalKnowledgeDeriver
from backend.knowledge.extraction.signal_detector import SignalDetectionResult
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM


class TestDeriveWithSignals:
    @pytest.mark.asyncio
    async def test_creates_trait_from_feedback(self):
        llm = FakeLLM(
            default_response=json.dumps(
                {
                    "operations": [
                        {
                            "op": "CREATE_TRAIT",
                            "id": "trait-concise-communication",
                            "display_name": "Concise Communication",
                            "trait_category": "communication",
                            "description": "User prefers concise responses",
                            "evidence": "Stop summarizing everything",
                            "confidence": 0.85,
                        }
                    ]
                }
            )
        )
        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)

        deriver = InternalKnowledgeDeriver(llm=llm, executor=executor)

        signals = SignalDetectionResult(
            has_signals=True,
            signal_types=frozenset({"feedback"}),
            matched_patterns=("feedback:stop summarizing",),
        )

        result = await deriver.derive(
            utterance="Stop summarizing everything, just give me the answer",
            assistant_response="I understand, I'll be more concise.",
            signals=signals,
            session_id="sess-001",
            event_id="evt-001",
        )

        assert len(result.operations) == 1
        assert result.operations[0]["op"] == "CREATE_TRAIT"
        assert result.operations[0]["id"] == "trait-concise-communication"
        assert result.llm_called
        # Verify graph write happened (MistIdentity link)
        assert len(conn.writes) >= 1


class TestDeriveNoSignals:
    @pytest.mark.asyncio
    async def test_skips_llm_call_without_signals(self):
        llm = FakeLLM()
        executor = FakeGraphExecutor()

        deriver = InternalKnowledgeDeriver(llm=llm, executor=executor)

        signals = SignalDetectionResult(has_signals=False)

        result = await deriver.derive(
            utterance="What's the weather?",
            assistant_response="I don't have weather data.",
            signals=signals,
            session_id="sess-001",
            event_id="evt-001",
        )

        assert len(result.operations) == 0
        assert not result.llm_called
        llm.assert_not_called()


class TestDeriveEmptyResponse:
    @pytest.mark.asyncio
    async def test_handles_empty_operations(self):
        llm = FakeLLM(default_response='{"operations": []}')
        executor = FakeGraphExecutor()

        deriver = InternalKnowledgeDeriver(llm=llm, executor=executor)

        signals = SignalDetectionResult(
            has_signals=True,
            signal_types=frozenset({"feedback"}),
        )

        result = await deriver.derive(
            utterance="good job",
            assistant_response="Thanks!",
            signals=signals,
            session_id="sess-001",
            event_id="evt-001",
        )

        assert len(result.operations) == 0
        assert result.llm_called


class TestDeriveLLMFailure:
    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        llm = FakeLLM(default_response="not json at all")
        executor = FakeGraphExecutor()
        deriver = InternalKnowledgeDeriver(llm=llm, executor=executor)

        signals = SignalDetectionResult(
            has_signals=True,
            signal_types=frozenset({"feedback"}),
        )

        result = await deriver.derive(
            utterance="you're great at this",
            assistant_response="Thanks!",
            signals=signals,
            session_id="sess-001",
            event_id="evt-001",
        )

        # Graceful degradation -- no operations, no crash
        assert len(result.operations) == 0
        assert result.llm_called


class TestUpdateOperation:
    @pytest.mark.asyncio
    async def test_update_forwards_fields(self):
        llm = FakeLLM(
            default_response=json.dumps(
                {
                    "operations": [
                        {
                            "op": "UPDATE",
                            "entity_id": "trait-concise-communication",
                            "fields": {"confidence": 0.95, "description": "Updated desc"},
                        }
                    ]
                }
            )
        )
        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        deriver = InternalKnowledgeDeriver(llm=llm, executor=executor)

        signals = SignalDetectionResult(has_signals=True, signal_types=frozenset({"feedback"}))

        result = await deriver.derive(
            utterance="yes exactly like that, keep it concise",
            assistant_response="Got it.",
            signals=signals,
            session_id="sess-001",
            event_id="evt-001",
        )

        assert len(result.operations) == 1
        assert result.operations[0]["op"] == "UPDATE"
        # entity_id is passed as a parameter, not in the query string
        conn.assert_write_executed("__Entity__")
        _, params = conn.writes[0]
        assert params["entity_id"] == "trait-concise-communication"


class TestDeprecateOperation:
    @pytest.mark.asyncio
    async def test_deprecate_sets_status(self):
        llm = FakeLLM(
            default_response=json.dumps(
                {
                    "operations": [
                        {
                            "op": "DEPRECATE",
                            "entity_id": "preference-formal-tone",
                            "reason": "User now prefers casual communication",
                        }
                    ]
                }
            )
        )
        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        deriver = InternalKnowledgeDeriver(llm=llm, executor=executor)

        signals = SignalDetectionResult(has_signals=True, signal_types=frozenset({"preference"}))

        result = await deriver.derive(
            utterance="actually, you can be more casual with me",
            assistant_response="Sure thing!",
            signals=signals,
            session_id="sess-001",
            event_id="evt-001",
        )

        assert len(result.operations) == 1
        assert result.operations[0]["op"] == "DEPRECATE"
        conn.assert_write_executed("deprecated")
