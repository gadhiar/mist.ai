"""Tests for the Stage 9 render/parse split and the public apply_operations/
fetch_existing_internal_entities methods (T1a).

`render_derivation_messages` and `parse_derivation_output` are the pure
functions the extraction service reuses; `apply_operations` is the single
apply path `derive()` now goes through, and the same path the extraction
service's caller (the backend dispatcher) will call directly with
operations parsed from `DerivationOut.operations`.
"""

from __future__ import annotations

import pytest

from backend.errors import ExtractionValidationError
from backend.knowledge.extraction.internal_derivation import (
    InternalKnowledgeDeriver,
    parse_derivation_output,
    render_derivation_messages,
)
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM


class TestRenderDerivationMessages:
    def test_renders_system_and_user_messages(self):
        messages = render_derivation_messages(
            utterance="Stop summarizing everything",
            assistant_response="Got it.",
            signal_types=["feedback"],
            matched_patterns=["feedback:stop"],
            existing_internal_entities="No existing internal entities.",
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_carries_all_fields(self):
        messages = render_derivation_messages(
            utterance="Stop summarizing everything",
            assistant_response="Got it.",
            signal_types=["feedback"],
            matched_patterns=["feedback:stop"],
            existing_internal_entities="No existing internal entities.",
        )

        content = messages[1]["content"]
        assert "Stop summarizing everything" in content
        assert "Got it." in content
        assert "feedback" in content
        assert "feedback:stop" in content
        assert "No existing internal entities." in content

    def test_accepts_frozenset_and_tuple_inputs(self):
        """SignalDetectionResult carries frozenset/tuple, not list."""
        messages = render_derivation_messages(
            utterance="u",
            assistant_response="a",
            signal_types=frozenset({"feedback"}),
            matched_patterns=("feedback:x",),
            existing_internal_entities="none",
        )

        assert "feedback" in messages[1]["content"]


class TestParseDerivationOutputNonStrict:
    def test_valid_json_returns_operations(self):
        raw = '{"operations": [{"op": "CREATE_TRAIT", "id": "trait-x"}]}'

        assert parse_derivation_output(raw, strict=False) == [
            {"op": "CREATE_TRAIT", "id": "trait-x"}
        ]

    def test_empty_string_returns_empty_list(self):
        assert parse_derivation_output("", strict=False) == []

    def test_none_returns_empty_list(self):
        assert parse_derivation_output(None, strict=False) == []

    def test_unparseable_garbage_returns_empty_list(self):
        assert parse_derivation_output("not json", strict=False) == []

    def test_non_dict_json_returns_empty_list(self):
        assert parse_derivation_output("[1, 2]", strict=False) == []

    def test_non_list_operations_returns_empty_list(self):
        assert parse_derivation_output('{"operations": "oops"}', strict=False) == []


class TestParseDerivationOutputStrict:
    def test_valid_json_returns_operations(self):
        raw = '{"operations": []}'

        assert parse_derivation_output(raw, strict=True) == []

    def test_empty_string_raises(self):
        with pytest.raises(ExtractionValidationError):
            parse_derivation_output("", strict=True)

    def test_none_raises(self):
        with pytest.raises(ExtractionValidationError):
            parse_derivation_output(None, strict=True)

    def test_unparseable_garbage_raises(self):
        with pytest.raises(ExtractionValidationError):
            parse_derivation_output("not json", strict=True)

    def test_non_dict_json_raises(self):
        with pytest.raises(ExtractionValidationError):
            parse_derivation_output("[1, 2]", strict=True)

    def test_non_list_operations_raises(self):
        with pytest.raises(ExtractionValidationError):
            parse_derivation_output('{"operations": "oops"}', strict=True)


class TestApplyOperations:
    @pytest.mark.asyncio
    async def test_applies_valid_operation_and_returns_it(self):
        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        deriver = InternalKnowledgeDeriver(llm=FakeLLM(), executor=executor)

        result = await deriver.apply_operations(
            [
                {
                    "op": "CREATE_TRAIT",
                    "id": "trait-concise",
                    "display_name": "Concise",
                    "description": "Prefers concise responses",
                    "confidence": 0.85,
                }
            ],
            session_id="s1",
            event_id="e1",
        )

        assert len(result) == 1
        assert result[0]["id"] == "trait-concise"
        conn.assert_write_executed("__SelfModel__")

    @pytest.mark.asyncio
    async def test_skips_invalid_op_type_without_writing(self):
        conn = FakeNeo4jConnection()
        executor = FakeGraphExecutor(connection=conn)
        deriver = InternalKnowledgeDeriver(llm=FakeLLM(), executor=executor)

        result = await deriver.apply_operations(
            [{"op": "NOT_A_REAL_OP", "id": "x"}],
            session_id="s1",
            event_id="e1",
        )

        assert result == ()
        conn.assert_no_writes()

    @pytest.mark.asyncio
    async def test_skips_op_that_fails_to_apply_but_keeps_others(self):
        """A write error on one op does not abort the batch."""
        conn = FakeNeo4jConnection(write_errors={"MERGE": RuntimeError("boom")})
        executor = FakeGraphExecutor(connection=conn)
        deriver = InternalKnowledgeDeriver(llm=FakeLLM(), executor=executor)

        result = await deriver.apply_operations(
            [
                {"op": "CREATE_TRAIT", "id": "trait-a", "display_name": "A"},
                {
                    "op": "DEPRECATE",
                    "entity_id": "preference-old",
                    "reason": "superseded",
                },
            ],
            session_id="s1",
            event_id="e1",
        )

        # The CREATE_TRAIT write raises (MERGE pattern); DEPRECATE's write
        # query has no MERGE, so it succeeds and survives in the result.
        assert [op["op"] for op in result] == ["DEPRECATE"]

    @pytest.mark.asyncio
    async def test_returns_empty_tuple_for_empty_input(self):
        executor = FakeGraphExecutor()
        deriver = InternalKnowledgeDeriver(llm=FakeLLM(), executor=executor)

        result = await deriver.apply_operations([], session_id="s1", event_id="e1")

        assert result == ()


class TestFetchExistingInternalEntitiesAlias:
    @pytest.mark.asyncio
    async def test_underscore_alias_matches_public_method(self):
        executor = FakeGraphExecutor()
        deriver = InternalKnowledgeDeriver(llm=FakeLLM(), executor=executor)

        public_result = await deriver.fetch_existing_internal_entities()
        alias_result = await deriver._fetch_existing_internal_entities()

        assert public_result == alias_result == "No existing internal entities."
