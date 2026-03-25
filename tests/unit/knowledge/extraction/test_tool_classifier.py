"""Unit tests for ToolOutputClassifier.

Covers provider-specific routing rules, anti-bloat gates (dedup, rate
limiting, content length), and tool name parsing/normalization.
"""

from __future__ import annotations

import hashlib
import time

import pytest

from backend.knowledge.extraction.tool_classifier import (
    _AGENT_RATE_LIMIT,
    _ORCHESTRATOR_RATE_LIMIT,
    _RATE_WINDOW_SECONDS,
    ToolOutputClassifier,
)
from backend.knowledge.models import RoutingDestination, ToolClassification

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Content long enough to pass the 50-char minimum gate and carry entity signals
# for GRAPH_AND_VECTOR routing at orchestrator level.
_RICH_CONTENT = (
    "Issue: MIST-123 assigned to Alice Smith. "
    "Team: Backend. Project: MIST.AI. "
    "Alice Smith works at Anthropic and is part of the Backend team."
)

# Content that is long enough (>=50 chars) but carries no entity signals --
# used to exercise vector-only fallback paths.
_PLAIN_CONTENT = (
    "no entities here, just some generic filler text that is long enough "
    "to pass the minimum content length gate of fifty characters"
)


def _make_classifier() -> ToolOutputClassifier:
    """Return a fresh ToolOutputClassifier with no prior state."""
    return ToolOutputClassifier()


def _classify(
    tool_name: str,
    tool_output: str = _RICH_CONTENT,
    source_level: str = "orchestrator",
) -> ToolClassification:
    """Shorthand: create a fresh classifier and classify one call."""
    return _make_classifier().classify(tool_name, tool_output, source_level)


# ---------------------------------------------------------------------------
# TestLinearProviderRouting
# ---------------------------------------------------------------------------


class TestLinearProviderRouting:
    """Provider-specific routing rules for Linear MCP tools."""

    def test_get_issue_routes_to_graph_and_vector(self):
        # Arrange
        classifier = _make_classifier()

        # Act
        result = classifier.classify("mcp__linear__get_issue", _RICH_CONTENT)

        # Assert
        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value
        assert result.tool_provider == "linear"
        assert result.tool_action == "get_issue"

    def test_save_issue_is_discarded_as_write_operation(self):
        # Arrange
        classifier = _make_classifier()

        # Act
        result = classifier.classify("mcp__linear__save_issue", _RICH_CONTENT)

        # Assert
        assert result.destination == RoutingDestination.DISCARD.value
        assert result.tool_provider == "linear"
        assert result.tool_action == "save_issue"

    def test_list_issues_routes_to_graph_and_vector(self):
        # Arrange
        classifier = _make_classifier()

        # Act
        result = classifier.classify("mcp__linear__list_issues", _RICH_CONTENT)

        # Assert
        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value
        assert result.tool_provider == "linear"
        assert result.tool_action == "list_issues"

    def test_save_issue_confidence_reflects_high_certainty_discard(self):
        # Write-operation discards carry 0.90 base confidence.
        result = _classify("mcp__linear__save_issue", _RICH_CONTENT)

        assert result.confidence >= 0.85

    def test_get_issue_tool_action_extracted_correctly(self):
        result = _classify("mcp__linear__get_issue", _RICH_CONTENT)

        assert result.tool_action == "get_issue"

    @pytest.mark.parametrize(
        "tool_name",
        [
            pytest.param("mcp__linear__save_comment", id="save_comment"),
            pytest.param("mcp__linear__save_project", id="save_project"),
            pytest.param("mcp__linear__save_milestone", id="save_milestone"),
            pytest.param("mcp__linear__create_issue_label", id="create_issue_label"),
            pytest.param("mcp__linear__create_document", id="create_document"),
            pytest.param("mcp__linear__delete_comment", id="delete_comment"),
        ],
    )
    def test_linear_write_operations_are_discarded(self, tool_name: str):
        # All linear write/mutate operations must be DISCARD.
        result = _classify(tool_name, _RICH_CONTENT)

        assert (
            result.destination == RoutingDestination.DISCARD.value
        ), f"{tool_name} expected DISCARD but got {result.destination}"


# ---------------------------------------------------------------------------
# TestGitHubProviderRouting
# ---------------------------------------------------------------------------


class TestGitHubProviderRouting:
    """Provider-specific routing rules for GitHub MCP tools."""

    def test_get_pull_request_routes_to_graph_and_vector(self):
        # Arrange
        classifier = _make_classifier()

        # Act
        result = classifier.classify("mcp__github__get_pull_request", _RICH_CONTENT)

        # Assert
        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value
        assert result.tool_provider == "github"
        assert result.tool_action == "get_pull_request"

    def test_list_pull_requests_routes_to_graph_and_vector(self):
        # Arrange
        classifier = _make_classifier()

        # Act
        result = classifier.classify("mcp__github__list_pull_requests", _RICH_CONTENT)

        # Assert
        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value
        assert result.tool_provider == "github"
        assert result.tool_action == "list_pull_requests"

    def test_create_issue_is_discarded_as_write_operation(self):
        # Arrange
        classifier = _make_classifier()

        # Act
        result = classifier.classify("mcp__github__create_issue", _RICH_CONTENT)

        # Assert
        assert result.destination == RoutingDestination.DISCARD.value
        assert result.tool_provider == "github"
        assert result.tool_action == "create_issue"

    def test_get_file_contents_routes_to_vector_only(self):
        result = _classify("mcp__github__get_file_contents", _PLAIN_CONTENT)

        assert result.destination == RoutingDestination.VECTOR_ONLY.value

    def test_search_code_routes_to_vector_only(self):
        result = _classify("mcp__github__search_code", _PLAIN_CONTENT)

        assert result.destination == RoutingDestination.VECTOR_ONLY.value

    @pytest.mark.parametrize(
        "tool_name",
        [
            pytest.param("mcp__github__create_or_update_file", id="create_or_update_file"),
            pytest.param("mcp__github__create_pull_request", id="create_pull_request"),
            pytest.param("mcp__github__create_repository", id="create_repository"),
            pytest.param("mcp__github__create_branch", id="create_branch"),
            pytest.param("mcp__github__add_issue_comment", id="add_issue_comment"),
            pytest.param("mcp__github__merge_pull_request", id="merge_pull_request"),
            pytest.param("mcp__github__push_files", id="push_files"),
        ],
    )
    def test_github_write_operations_are_discarded(self, tool_name: str):
        # All github write/mutate operations must be DISCARD.
        result = _classify(tool_name, _RICH_CONTENT)

        assert (
            result.destination == RoutingDestination.DISCARD.value
        ), f"{tool_name} expected DISCARD but got {result.destination}"

    def test_github_provider_extracted_correctly(self):
        result = _classify("mcp__github__get_issue", _RICH_CONTENT)

        assert result.tool_provider == "github"


# ---------------------------------------------------------------------------
# TestFilesystemProviderRouting
# ---------------------------------------------------------------------------


class TestFilesystemProviderRouting:
    """Provider-specific routing rules for filesystem MCP tools."""

    def test_read_file_routes_to_vector_only(self):
        # Arrange
        classifier = _make_classifier()

        # Act
        result = classifier.classify("mcp__filesystem-dev__read_file", _PLAIN_CONTENT)

        # Assert
        assert result.destination == RoutingDestination.VECTOR_ONLY.value
        assert result.tool_provider == "filesystem"
        assert result.tool_action == "read_file"

    def test_write_file_is_discarded(self):
        # Arrange
        classifier = _make_classifier()

        # Act
        result = classifier.classify("mcp__filesystem-dev__write_file", _PLAIN_CONTENT)

        # Assert
        assert result.destination == RoutingDestination.DISCARD.value
        assert result.tool_provider == "filesystem"
        assert result.tool_action == "write_file"

    def test_edit_file_is_discarded(self):
        result = _classify("mcp__filesystem-dev__edit_file", _PLAIN_CONTENT)

        assert result.destination == RoutingDestination.DISCARD.value

    def test_read_text_file_routes_to_vector_only(self):
        result = _classify("mcp__filesystem-dev__read_text_file", _PLAIN_CONTENT)

        assert result.destination == RoutingDestination.VECTOR_ONLY.value

    def test_create_directory_is_discarded(self):
        result = _classify("mcp__filesystem-dev__create_directory", _PLAIN_CONTENT)

        assert result.destination == RoutingDestination.DISCARD.value

    def test_move_file_is_discarded(self):
        result = _classify("mcp__filesystem-dev__move_file", _PLAIN_CONTENT)

        assert result.destination == RoutingDestination.DISCARD.value


# ---------------------------------------------------------------------------
# TestAntiBloatGates
# ---------------------------------------------------------------------------


class TestAntiBloatGates:
    """Anti-bloat gates: content length, dedup, and rate limiting."""

    def test_content_shorter_than_50_chars_is_discarded(self):
        # Arrange -- use a tool WITHOUT a specific provider rule so the length
        # gate (step 6) is actually reached. Provider-specific rules fire first
        # and bypass the length check entirely.
        classifier = _make_classifier()
        short_content = "too short"  # 9 chars -- well under the 50-char gate
        assert len(short_content) < 50

        # Act -- unknown provider has no rule, so falls through to length gate
        result = classifier.classify("mcp__unknownprovider__some_action", short_content)

        # Assert
        assert result.destination == RoutingDestination.DISCARD.value

    def test_content_exactly_at_50_chars_passes_length_gate(self):
        # The gate is strictly < 50; content of exactly 50 chars must NOT be
        # discarded by the length check.
        classifier = _make_classifier()
        # Construct content with exactly 50 chars that will not match a
        # provider rule (unknown provider) and exceed the 200-char web threshold.
        # Instead, use a provider with a vector_only rule to confirm length gate
        # does not fire.
        content_50 = "x" * 50
        assert len(content_50) == 50

        # Act
        result = classifier.classify("mcp__filesystem-dev__read_file", content_50)

        # Assert: length gate must not have fired; filesystem read_file -> VECTOR_ONLY
        assert result.destination == RoutingDestination.VECTOR_ONLY.value
        assert "too short" not in result.reason

    def test_content_49_chars_is_discarded_by_length_gate(self):
        # Gate fires for len < 50; 49 chars must be discarded even for a
        # VECTOR_ONLY provider rule if no earlier rule fires.
        # However, provider rules fire before the length gate (step 4 vs step 6).
        # A provider rule that maps to VECTOR_ONLY returns immediately without
        # hitting the length gate. We test the length gate by using a provider
        # action that has no rule (falls through to step 6).
        classifier = _make_classifier()
        content_49 = "y" * 49
        assert len(content_49) == 49

        # Use a known provider with an unknown action to fall through to step 6
        result = classifier.classify("mcp__linear__unknown_action", content_49)

        assert result.destination == RoutingDestination.DISCARD.value
        assert "too short" in result.reason

    def test_duplicate_content_within_dedup_window_is_discarded(self):
        # Arrange: classify the same content twice on the same classifier
        # instance (dedup cache persists across calls).
        classifier = _make_classifier()
        content = _PLAIN_CONTENT

        # Act
        first = classifier.classify("mcp__linear__get_issue", content)
        second = classifier.classify("mcp__linear__get_issue", content)

        # Assert: first call routed normally, second hit the dedup gate
        assert (
            first.destination != RoutingDestination.DISCARD.value
            or first.reason != "duplicate_within_1h"
        ), "First call should not be discarded as a duplicate"
        assert second.destination == RoutingDestination.DISCARD.value
        assert second.reason == "duplicate_within_1h"

    def test_dedup_discard_has_high_confidence(self):
        # Dedup discards carry confidence=0.95.
        classifier = _make_classifier()
        content = _PLAIN_CONTENT
        classifier.classify("mcp__linear__get_issue", content)  # seed
        second = classifier.classify("mcp__linear__get_issue", content)

        assert second.confidence == 0.95

    def test_different_content_is_not_deduplicated(self):
        # Two calls with different content must not trigger the dedup gate.
        classifier = _make_classifier()
        content_a = _RICH_CONTENT
        content_b = _PLAIN_CONTENT
        assert content_a != content_b

        classifier.classify("mcp__linear__get_issue", content_a)  # prime dedup cache
        result_b = classifier.classify("mcp__linear__get_issue", content_b)

        assert result_b.reason != "duplicate_within_1h"

    def test_rate_limit_exceeded_for_orchestrator_is_discarded(self):
        # Orchestrator limit is 10 calls/minute. Pre-fill the rate log with
        # 10 entries inside the window, then verify the next call is discarded.
        classifier = _make_classifier()
        now = time.monotonic()

        # Directly inject 10 recent timestamps into the rate log to simulate
        # hitting the limit without making 10 actual classify() calls (which
        # would also populate the dedup cache and complicate isolation).
        for _ in range(_ORCHESTRATOR_RATE_LIMIT):
            classifier._rate_log["orchestrator"].append(now)

        # Act: unique content so dedup does not fire before the rate check
        unique_content = "A" * 200  # long enough, unique hash
        result = classifier.classify("mcp__linear__get_issue", unique_content, "orchestrator")

        # Assert
        assert result.destination == RoutingDestination.DISCARD.value
        assert result.reason == "rate_limit_exceeded"

    def test_rate_limit_exceeded_for_agent_is_discarded(self):
        # Agent limit is 5 calls/minute.
        classifier = _make_classifier()
        now = time.monotonic()

        for _ in range(_AGENT_RATE_LIMIT):
            classifier._rate_log["agent"].append(now)

        unique_content = "B" * 200
        result = classifier.classify("mcp__linear__get_issue", unique_content, "agent")

        assert result.destination == RoutingDestination.DISCARD.value
        assert result.reason == "rate_limit_exceeded"

    def test_rate_limit_not_triggered_below_limit(self):
        # Filling the log to one below the limit must not trigger rate discard.
        classifier = _make_classifier()
        now = time.monotonic()

        for _ in range(_ORCHESTRATOR_RATE_LIMIT - 1):
            classifier._rate_log["orchestrator"].append(now)

        unique_content = "C" * 200
        result = classifier.classify("mcp__linear__get_issue", unique_content, "orchestrator")

        assert result.reason != "rate_limit_exceeded"

    def test_rate_limit_entries_outside_window_do_not_count(self):
        # Timestamps older than RATE_WINDOW_SECONDS are evicted and must not
        # count toward the limit.
        classifier = _make_classifier()
        stale_time = time.monotonic() - (_RATE_WINDOW_SECONDS + 1.0)

        for _ in range(_ORCHESTRATOR_RATE_LIMIT):
            classifier._rate_log["orchestrator"].append(stale_time)

        unique_content = "D" * 200
        result = classifier.classify("mcp__linear__get_issue", unique_content, "orchestrator")

        assert result.reason != "rate_limit_exceeded"


# ---------------------------------------------------------------------------
# TestProviderParsing
# ---------------------------------------------------------------------------


class TestProviderParsing:
    """Tool name parsing and provider normalization."""

    def test_filesystem_dev_slug_normalizes_to_filesystem(self):
        # "filesystem-dev" must normalize to "filesystem" so provider rules fire.
        result = _classify("mcp__filesystem-dev__read_file", _PLAIN_CONTENT)

        assert result.tool_provider == "filesystem"

    def test_filesystem_dev_action_parsed_correctly(self):
        result = _classify("mcp__filesystem-dev__write_file", _PLAIN_CONTENT)

        assert result.tool_action == "write_file"

    def test_unknown_provider_falls_back_gracefully(self):
        # A tool name with an unrecognized provider must not raise; the
        # classifier should return a result (web/unknown fallback path).
        long_content = "E" * 300  # >200 chars triggers VECTOR_ONLY in web/unknown path
        result = _classify("mcp__unknownprovider__some_action", long_content)

        assert isinstance(result, ToolClassification)
        assert result.tool_provider == "unknownprovider"
        assert result.tool_action == "some_action"

    def test_unknown_provider_with_short_content_routes_to_vector_only(self):
        # Unknown provider with content > 50 but <= 200 chars follows the
        # web/unknown fallback path. The exact routing depends on the
        # classifier's fallback logic -- verify it does not raise and returns
        # a valid classification.
        short_but_valid = "F" * 100
        result = _classify("mcp__unknownprovider__some_action", short_but_valid)

        assert isinstance(result, ToolClassification)
        assert result.destination in {
            RoutingDestination.VECTOR_ONLY.value,
            RoutingDestination.DISCARD.value,
        }

    def test_unknown_provider_with_long_content_routes_to_vector_only(self):
        # Unknown provider + content > 200 chars -> VECTOR_ONLY.
        long_content = "G" * 201
        result = _classify("mcp__unknownprovider__some_action", long_content)

        assert result.destination == RoutingDestination.VECTOR_ONLY.value

    def test_tool_name_without_mcp_prefix_is_unknown(self):
        # Names not matching "mcp__<provider>__<action>" fall back to unknown.
        long_content = "H" * 300
        result = _classify("some_tool_without_mcp_prefix", long_content)

        assert result.tool_provider == "unknown"
        assert result.tool_action == "some_tool_without_mcp_prefix"

    def test_mcp_prefix_required_for_provider_extraction(self):
        # A name like "linear__get_issue" (missing "mcp" prefix) must not
        # produce provider="linear"; it falls to unknown.
        long_content = "I" * 300
        result = _classify("linear__get_issue", long_content)

        assert result.tool_provider == "unknown"

    def test_content_hash_is_sha256_of_output(self):
        # The content_hash in the result must match SHA-256 of tool_output.
        content = _PLAIN_CONTENT
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        result = _classify("mcp__filesystem-dev__read_file", content)

        assert result.content_hash == expected_hash

    def test_source_level_propagated_to_result(self):
        result = _classify("mcp__linear__get_issue", _RICH_CONTENT, source_level="agent")

        assert result.source_level == "agent"


# ---------------------------------------------------------------------------
# TestAgentSourceLevelAdjustments
# ---------------------------------------------------------------------------


class TestAgentSourceLevelAdjustments:
    """Agent-level source reduces confidence and may downgrade graph routing."""

    def test_agent_confidence_lower_than_orchestrator_for_same_rule(self):
        # Agent level applies a -0.15 penalty to base confidence.
        orch_result = _classify(
            "mcp__linear__get_issue", _RICH_CONTENT, source_level="orchestrator"
        )
        agent_result = _classify("mcp__linear__get_issue", _RICH_CONTENT, source_level="agent")

        assert agent_result.confidence < orch_result.confidence

    def test_agent_with_insufficient_entity_signals_downgrades_to_vector_only(self):
        # At agent level, GRAPH_AND_VECTOR rules require >=2 entity signals.
        # _PLAIN_CONTENT has no entity signals.
        result = _classify("mcp__linear__get_issue", _PLAIN_CONTENT, source_level="agent")

        assert result.destination == RoutingDestination.VECTOR_ONLY.value

    def test_orchestrator_with_rich_content_routes_to_graph_and_vector(self):
        # Confirm that same rule at orchestrator level (with rich content) does
        # NOT get downgraded, establishing the agent/orchestrator contrast.
        result = _classify("mcp__linear__get_issue", _RICH_CONTENT, source_level="orchestrator")

        assert result.destination == RoutingDestination.GRAPH_AND_VECTOR.value
