"""Tests for the Stage 2 render/parse split (T1a).

`render_extraction_messages` and `parse_extraction_output` are the pure
functions the extraction service reuses. `OntologyConstrainedExtractor`
already exercises the non-strict path indirectly via `test_stage_purity.py`
and the pipeline tests; this file targets the split functions directly,
including the `strict=True` path the in-process pipeline never uses.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from backend.errors import ExtractionValidationError
from backend.knowledge.extraction.ontology_extractor import (
    parse_extraction_output,
    render_extraction_messages,
)
from backend.knowledge.extraction.preprocessor import PreProcessedInput

REF_DATE = datetime(2026, 4, 21)


def _pre_processed(
    utterance: str = "I use Rust", scope: str = "user-scope", context: list[str] | None = None
) -> PreProcessedInput:
    return PreProcessedInput(
        original_text=utterance,
        resolved_text=utterance,
        conversation_context=context or [],
        reference_date=REF_DATE,
        turn_index=0,
        metadata={"subject_scope": scope} if scope else {},
    )


class TestRenderExtractionMessages:
    def test_renders_system_and_user_messages(self):
        messages = render_extraction_messages(_pre_processed())

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_substitutes_reference_date(self):
        messages = render_extraction_messages(_pre_processed())

        assert "2026-04-21" in messages[0]["content"]

    def test_user_message_carries_scope_and_utterance(self):
        messages = render_extraction_messages(_pre_processed(utterance="MIST uses LanceDB"))

        assert "MIST uses LanceDB" in messages[1]["content"]
        assert "user-scope" in messages[1]["content"]

    def test_missing_subject_scope_defaults_to_unknown(self):
        pre = _pre_processed(scope="")
        pre.metadata = {}

        messages = render_extraction_messages(pre)

        assert "Subject scope: unknown" in messages[1]["content"]

    def test_no_context_renders_placeholder(self):
        messages = render_extraction_messages(_pre_processed(context=[]))

        assert "(no prior context)" in messages[1]["content"]

    def test_context_lines_are_joined(self):
        messages = render_extraction_messages(
            _pre_processed(context=["[user]: hi", "[assistant]: hello"])
        )

        assert "[user]: hi\n[assistant]: hello" in messages[1]["content"]


class TestParseExtractionOutputNonStrict:
    """strict=False: the in-process pipeline's historical fall-back-to-empty behavior."""

    def test_valid_json_returns_parsed_dict(self):
        raw = '{"entities": [{"id": "user", "name": "User", "type": "User"}], "relationships": []}'

        result = parse_extraction_output(raw, strict=False)

        assert result["entities"] == [{"id": "user", "name": "User", "type": "User"}]
        assert result["relationships"] == []

    def test_json_embedded_in_prose_is_extracted_via_regex_fallback(self):
        raw = 'Sure, here is the JSON: {"entities": [], "relationships": []} Hope that helps!'

        result = parse_extraction_output(raw, strict=False)

        assert result == {"entities": [], "relationships": []}

    def test_empty_string_returns_empty_result(self):
        assert parse_extraction_output("", strict=False) == {"entities": [], "relationships": []}

    def test_unparseable_garbage_returns_empty_result(self):
        assert parse_extraction_output("not json at all {{{", strict=False) == {
            "entities": [],
            "relationships": [],
        }

    def test_dict_missing_required_keys_is_returned_as_is(self):
        """Non-strict never validates shape -- only strict does."""
        result = parse_extraction_output('{"other_key": 1}', strict=False)

        assert result == {"other_key": 1}


class TestParseExtractionOutputStrict:
    """strict=True: the extraction service's never-silently-empty behavior."""

    def test_valid_well_shaped_output_returns_parsed_dict(self):
        raw = '{"entities": [], "relationships": []}'

        result = parse_extraction_output(raw, strict=True)

        assert result == {"entities": [], "relationships": []}

    def test_valid_output_with_real_entities_is_not_treated_as_a_failure(self):
        raw = (
            '{"entities": [{"id": "rust", "name": "Rust", "type": "Technology"}], '
            '"relationships": []}'
        )

        result = parse_extraction_output(raw, strict=True)

        assert len(result["entities"]) == 1

    def test_empty_string_raises(self):
        with pytest.raises(ExtractionValidationError):
            parse_extraction_output("", strict=True)

    def test_unparseable_garbage_raises(self):
        with pytest.raises(ExtractionValidationError):
            parse_extraction_output("not json at all {{{", strict=True)

    def test_non_dict_json_raises(self):
        with pytest.raises(ExtractionValidationError):
            parse_extraction_output("[1, 2, 3]", strict=True)

    def test_dict_missing_entities_key_raises(self):
        with pytest.raises(ExtractionValidationError):
            parse_extraction_output('{"relationships": []}', strict=True)

    def test_dict_with_non_list_entities_raises(self):
        with pytest.raises(ExtractionValidationError):
            parse_extraction_output('{"entities": "oops", "relationships": []}', strict=True)
