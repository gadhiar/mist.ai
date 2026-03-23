"""Tests for TemporalResolver extraction pipeline stage."""

from datetime import datetime

import pytest

from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.extraction.temporal import TemporalResolver


def _make_extraction(temporal_expression: str) -> ExtractionResult:
    """Build an ExtractionResult with one relationship carrying a temporal expression."""
    return ExtractionResult(
        entities=[],
        relationships=[
            {
                "type": "WORKS_AT",
                "source": "user",
                "target": "acme",
                "properties": {"temporal_expression": temporal_expression},
            }
        ],
        source_utterance="test",
    )


def _get_props(result: ExtractionResult) -> dict:
    """Extract the properties dict from the first relationship."""
    return result.relationships[0]["properties"]


class TestRelativeDatePatterns:
    """Relative temporal expressions resolve to absolute ISO-8601 dates."""

    @pytest.mark.parametrize(
        "expression, ref_date, expected_start, expected_end, expected_status",
        [
            pytest.param(
                "last year",
                datetime(2025, 6, 15),
                "2024-01-01",
                "2024-12-31",
                "past",
                id="last-year",
            ),
            pytest.param(
                "next year",
                datetime(2025, 6, 15),
                "2026-01-01",
                "2026-12-31",
                "future",
                id="next-year",
            ),
            pytest.param(
                "this year",
                datetime(2025, 6, 15),
                "2025-01-01",
                None,
                "current",
                id="this-year",
            ),
            pytest.param(
                "last month",
                datetime(2025, 6, 15),
                "2025-05-01",
                "2025-05-31",
                "past",
                id="last-month",
            ),
            pytest.param(
                "next week",
                datetime(2025, 6, 15),
                None,
                None,
                None,
                id="next-week-no-pattern",
            ),
        ],
    )
    def test_resolves_relative_expression(
        self,
        expression: str,
        ref_date: datetime,
        expected_start: str | None,
        expected_end: str | None,
        expected_status: str | None,
    ):
        # Arrange
        resolver = TemporalResolver()
        extraction = _make_extraction(expression)

        # Act
        result = resolver.resolve(extraction, ref_date)

        # Assert
        props = _get_props(result)
        if expected_start is not None:
            assert props["start_date"] == expected_start
        else:
            assert "start_date" not in props
        if expected_end is not None:
            assert props["end_date"] == expected_end
        else:
            assert "end_date" not in props
        if expected_status is not None:
            assert props["temporal_status"] == expected_status
        else:
            assert "temporal_status" not in props

    @pytest.mark.parametrize(
        "expression, ref_date, expected_start",
        [
            pytest.param(
                "2 months ago",
                datetime(2025, 6, 15),
                "2025-04-01",
                id="2-months-ago",
            ),
            pytest.param(
                "3 years ago",
                datetime(2025, 6, 15),
                "2022-01-01",
                id="3-years-ago",
            ),
        ],
    )
    def test_resolves_n_units_ago(
        self,
        expression: str,
        ref_date: datetime,
        expected_start: str,
    ):
        # Arrange
        resolver = TemporalResolver()
        extraction = _make_extraction(expression)

        # Act
        result = resolver.resolve(extraction, ref_date)

        # Assert
        props = _get_props(result)
        assert props["start_date"] == expected_start
        assert props["temporal_status"] == "past"

    def test_yesterday_resolves_to_previous_day(self):
        # Arrange
        resolver = TemporalResolver()
        extraction = _make_extraction("yesterday")
        ref = datetime(2025, 3, 15)

        # Act
        result = resolver.resolve(extraction, ref)

        # Assert
        props = _get_props(result)
        assert props["start_date"] == "2025-03-14"
        assert props["end_date"] == "2025-03-14"
        assert props["temporal_status"] == "past"


class TestEdgeCases:
    """Edge cases for temporal resolution."""

    def test_january_last_month_wraps_to_previous_december(self):
        # Arrange -- reference date in January
        resolver = TemporalResolver()
        extraction = _make_extraction("last month")
        ref = datetime(2025, 1, 10)

        # Act
        result = resolver.resolve(extraction, ref)

        # Assert -- should wrap to December of previous year
        props = _get_props(result)
        assert props["start_date"] == "2024-12-01"
        assert props["end_date"] == "2024-12-31"
        assert props["temporal_status"] == "past"

    def test_months_ago_wraps_across_year_boundary(self):
        # Arrange -- 3 months ago from February 2025 -> November 2024
        resolver = TemporalResolver()
        extraction = _make_extraction("3 months ago")
        ref = datetime(2025, 2, 10)

        # Act
        result = resolver.resolve(extraction, ref)

        # Assert
        props = _get_props(result)
        assert props["start_date"] == "2024-11-01"
        assert props["temporal_status"] == "past"


class TestNoTemporalExpression:
    """When no temporal expression is present, dates are not added."""

    def test_no_temporal_expression_returns_no_dates(self):
        # Arrange
        resolver = TemporalResolver()
        extraction = ExtractionResult(
            entities=[],
            relationships=[
                {
                    "type": "WORKS_AT",
                    "source": "user",
                    "target": "acme",
                    "properties": {"confidence": 0.9},
                }
            ],
            source_utterance="test",
        )
        ref = datetime(2025, 6, 15)

        # Act
        result = resolver.resolve(extraction, ref)

        # Assert
        props = result.relationships[0]["properties"]
        assert "start_date" not in props
        assert "end_date" not in props

    def test_none_temporal_expression_returns_no_dates(self):
        # Arrange
        resolver = TemporalResolver()
        extraction = ExtractionResult(
            entities=[],
            relationships=[
                {
                    "type": "WORKS_AT",
                    "source": "user",
                    "target": "acme",
                    "properties": {"temporal_expression": None},
                }
            ],
            source_utterance="test",
        )
        ref = datetime(2025, 6, 15)

        # Act
        result = resolver.resolve(extraction, ref)

        # Assert
        props = result.relationships[0]["properties"]
        assert "start_date" not in props
        assert "end_date" not in props
