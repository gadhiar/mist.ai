"""Tests for ToolUsageTracker -- tool call recording and pattern detection."""

from datetime import UTC, datetime, timedelta

import pytest

from backend.knowledge.config import SkillDerivationConfig
from backend.knowledge.extraction.tool_usage_tracker import (
    ToolCallRecord,
    ToolUsageTracker,
    classify_tool_type,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_record(
    *,
    tool_name: str = "file_search",
    tool_type: str = "search",
    context: str = "searching for files",
    success: bool = True,
    timestamp: datetime | None = None,
    session_id: str = "session-test-001",
    event_id: str = "event-test-001",
) -> ToolCallRecord:
    """Build a valid ToolCallRecord with overridable fields."""
    return ToolCallRecord(
        tool_name=tool_name,
        tool_type=tool_type,
        context=context,
        success=success,
        timestamp=timestamp if timestamp is not None else datetime.now(UTC),
        session_id=session_id,
        event_id=event_id,
    )


def build_skill_config(
    *,
    window_size: int = 10,
    skill_threshold: int = 3,
    capability_threshold: int = 5,
    lookback_days: int = 7,
    similarity_threshold: float = 0.7,
    enabled: bool = True,
) -> SkillDerivationConfig:
    """Build a SkillDerivationConfig with explicit test values."""
    return SkillDerivationConfig(
        window_size=window_size,
        skill_threshold=skill_threshold,
        capability_threshold=capability_threshold,
        lookback_days=lookback_days,
        similarity_threshold=similarity_threshold,
        enabled=enabled,
    )


# ---------------------------------------------------------------------------
# TestRecord
# ---------------------------------------------------------------------------


class TestRecord:
    def test_appends_record_to_internal_list(self):
        # Arrange
        config = build_skill_config(window_size=5)
        tracker = ToolUsageTracker(config)
        record = build_record(tool_name="bash_exec")

        # Act
        tracker.record(record)

        # Assert
        assert len(tracker._records) == 1
        assert tracker._records[0] is record

    def test_evicts_oldest_when_at_max_capacity(self):
        # Arrange
        config = build_skill_config(window_size=3)
        tracker = ToolUsageTracker(config)

        oldest = build_record(tool_name="old_tool", event_id="event-old")
        second = build_record(tool_name="second_tool", event_id="event-second")
        third = build_record(tool_name="third_tool", event_id="event-third")
        fourth = build_record(tool_name="fourth_tool", event_id="event-fourth")

        # Act
        tracker.record(oldest)
        tracker.record(second)
        tracker.record(third)
        tracker.record(fourth)

        # Assert -- window_size=3, so oldest is dropped
        assert len(tracker._records) == 3
        assert all(r.tool_name != "old_tool" for r in tracker._records)
        assert tracker._records[-1].tool_name == "fourth_tool"


# ---------------------------------------------------------------------------
# TestDetectPatterns
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    def test_returns_empty_when_no_records(self):
        # Arrange
        config = build_skill_config()
        tracker = ToolUsageTracker(config)

        # Act
        result = tracker.detect_patterns()

        # Assert
        assert result == []

    def test_returns_pattern_when_occurrences_meet_skill_threshold(self):
        # Arrange
        config = build_skill_config(skill_threshold=2, lookback_days=7)
        tracker = ToolUsageTracker(config)
        now = datetime.now(UTC)

        for i in range(2):
            tracker.record(
                build_record(
                    tool_name="file_search",
                    tool_type="search",
                    context="searching codebase",
                    timestamp=now - timedelta(hours=i),
                    event_id=f"event-{i}",
                )
            )

        # Act
        patterns = tracker.detect_patterns()

        # Assert
        assert len(patterns) == 1
        assert patterns[0].tool_type == "search"
        assert patterns[0].occurrence_count == 2

    def test_filters_out_patterns_below_threshold(self):
        # Arrange
        config = build_skill_config(skill_threshold=3, lookback_days=7)
        tracker = ToolUsageTracker(config)
        now = datetime.now(UTC)

        # Only 2 records -- below threshold of 3
        for i in range(2):
            tracker.record(
                build_record(
                    tool_name="file_search",
                    tool_type="search",
                    context="searching codebase",
                    timestamp=now - timedelta(hours=i),
                    event_id=f"event-{i}",
                )
            )

        # Act
        patterns = tracker.detect_patterns()

        # Assert
        assert patterns == []

    def test_respects_lookback_days_window_excludes_old_records(self):
        # Arrange
        config = build_skill_config(skill_threshold=2, lookback_days=3)
        tracker = ToolUsageTracker(config)
        now = datetime.now(UTC)

        # Two records older than the lookback window
        for i in range(2):
            tracker.record(
                build_record(
                    tool_name="file_search",
                    tool_type="search",
                    context="searching codebase",
                    timestamp=now - timedelta(days=5),
                    event_id=f"event-old-{i}",
                )
            )

        # Act -- records are outside the 3-day window
        patterns = tracker.detect_patterns()

        # Assert
        assert patterns == []


# ---------------------------------------------------------------------------
# TestClusterByContext
# ---------------------------------------------------------------------------


class TestClusterByContext:
    def test_identical_contexts_merge_into_one_cluster(self):
        # Arrange
        config = build_skill_config(similarity_threshold=0.7)
        tracker = ToolUsageTracker(config)
        now = datetime.now(UTC)

        records = [
            build_record(
                context="searching the codebase for files",
                timestamp=now - timedelta(seconds=i),
                event_id=f"ev-{i}",
            )
            for i in range(3)
        ]

        # Act
        clusters = tracker._cluster_by_context(records)

        # Assert
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_similar_contexts_merge_when_jaccard_above_threshold(self):
        # Arrange -- threshold 0.5: two overlapping word sets should merge
        config = build_skill_config(similarity_threshold=0.5)
        tracker = ToolUsageTracker(config)
        now = datetime.now(UTC)

        record_a = build_record(
            context="reading python files in project", timestamp=now, event_id="ev-a"
        )
        # Shares "reading", "python", "files" with record_a -- high overlap
        record_b = build_record(
            context="reading python files in editor",
            timestamp=now - timedelta(seconds=1),
            event_id="ev-b",
        )

        # Act
        clusters = tracker._cluster_by_context([record_a, record_b])

        # Assert -- should merge into one cluster
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_dissimilar_contexts_stay_in_separate_clusters(self):
        # Arrange -- threshold 0.7 means low-overlap contexts stay separate
        config = build_skill_config(similarity_threshold=0.7)
        tracker = ToolUsageTracker(config)
        now = datetime.now(UTC)

        record_a = build_record(
            context="searching python codebase files", timestamp=now, event_id="ev-a"
        )
        record_b = build_record(
            context="running bash terminal command",
            timestamp=now - timedelta(seconds=1),
            event_id="ev-b",
        )

        # Act
        clusters = tracker._cluster_by_context([record_a, record_b])

        # Assert -- no shared words, separate clusters
        assert len(clusters) == 2

    def test_empty_records_returns_empty_clusters(self):
        # Arrange
        config = build_skill_config()
        tracker = ToolUsageTracker(config)

        # Act
        clusters = tracker._cluster_by_context([])

        # Assert
        assert clusters == []


# ---------------------------------------------------------------------------
# TestClassifyToolType
# ---------------------------------------------------------------------------


class TestClassifyToolType:
    @pytest.mark.parametrize(
        "tool_name, expected_type",
        [
            pytest.param("file_search", "search", id="file-search-classified-as-search"),
            pytest.param(
                "bash_exec", "code_execution", id="bash-exec-classified-as-code-execution"
            ),
            pytest.param(
                "read_file", "file_management", id="read-file-classified-as-file-management"
            ),
            pytest.param("unknown_tool", "general", id="unknown-tool-classified-as-general"),
        ],
    )
    def test_classifies_tool_name_to_expected_type(self, tool_name: str, expected_type: str):
        # Act
        result = classify_tool_type(tool_name)

        # Assert
        assert result == expected_type


# ---------------------------------------------------------------------------
# TestBuildPattern
# ---------------------------------------------------------------------------


class TestBuildPattern:
    def test_aggregates_tool_names_as_frozenset(self):
        # Arrange
        now = datetime.now(UTC)
        cluster = [
            build_record(tool_name="file_search", timestamp=now, event_id="ev-1"),
            build_record(
                tool_name="grep_search", timestamp=now - timedelta(seconds=1), event_id="ev-2"
            ),
            build_record(
                tool_name="file_search", timestamp=now - timedelta(seconds=2), event_id="ev-3"
            ),
        ]

        # Act
        pattern = ToolUsageTracker._build_pattern("search", cluster)

        # Assert
        assert pattern.tool_names == frozenset({"file_search", "grep_search"})

    def test_counts_successes_correctly(self):
        # Arrange
        now = datetime.now(UTC)
        cluster = [
            build_record(success=True, timestamp=now, event_id="ev-1"),
            build_record(success=False, timestamp=now - timedelta(seconds=1), event_id="ev-2"),
            build_record(success=True, timestamp=now - timedelta(seconds=2), event_id="ev-3"),
        ]

        # Act
        pattern = ToolUsageTracker._build_pattern("search", cluster)

        # Assert
        assert pattern.success_count == 2

    def test_uses_most_common_context_as_summary(self):
        # Arrange
        now = datetime.now(UTC)
        cluster = [
            build_record(context="context alpha", timestamp=now, event_id="ev-1"),
            build_record(
                context="context beta", timestamp=now - timedelta(seconds=1), event_id="ev-2"
            ),
            build_record(
                context="context alpha", timestamp=now - timedelta(seconds=2), event_id="ev-3"
            ),
        ]

        # Act
        pattern = ToolUsageTracker._build_pattern("search", cluster)

        # Assert -- "context alpha" appears twice vs "context beta" once
        assert pattern.context_summary == "context alpha"

    def test_captures_first_seen_and_last_seen_timestamps(self):
        # Arrange
        now = datetime.now(UTC)
        earliest = now - timedelta(hours=2)
        latest = now

        cluster = [
            build_record(timestamp=now - timedelta(hours=1), event_id="ev-1"),
            build_record(timestamp=earliest, event_id="ev-2"),
            build_record(timestamp=latest, event_id="ev-3"),
        ]

        # Act
        pattern = ToolUsageTracker._build_pattern("search", cluster)

        # Assert
        assert pattern.first_seen == earliest
        assert pattern.last_seen == latest
