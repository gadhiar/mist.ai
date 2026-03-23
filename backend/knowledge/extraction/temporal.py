"""Temporal resolution stage for the extraction pipeline.

Stage 4: Resolves relative temporal expressions ("last year", "3 years ago")
into absolute ISO-8601 date ranges. No LLM call, target <5ms.
"""

import calendar
import logging
import re
from datetime import datetime

from backend.knowledge.extraction.ontology_extractor import ExtractionResult

logger = logging.getLogger(__name__)


class TemporalResolver:
    """Resolves relative temporal expressions to absolute dates.

    Scans each relationship's temporal_expression property and, where
    it matches a known pattern, fills in start_date and/or end_date
    with ISO-8601 date strings.

    Also infers temporal_status (current/past/future) when possible.
    """

    def __init__(self) -> None:
        """Initialize temporal resolver with compiled patterns."""
        # Each entry: (compiled pattern, resolver callable).
        # The callable receives (match, reference_date) and returns
        # (start_date_str | None, end_date_str | None, temporal_status | None).
        self._patterns: list[tuple[re.Pattern, callable]] = [
            (
                re.compile(r"\blast\s+year\b", re.IGNORECASE),
                self._resolve_last_year,
            ),
            (
                re.compile(r"\bthis\s+year\b", re.IGNORECASE),
                self._resolve_this_year,
            ),
            (
                re.compile(r"\bnext\s+year\b", re.IGNORECASE),
                self._resolve_next_year,
            ),
            (
                re.compile(r"\blast\s+month\b", re.IGNORECASE),
                self._resolve_last_month,
            ),
            (
                re.compile(r"\bthis\s+month\b", re.IGNORECASE),
                self._resolve_this_month,
            ),
            (
                re.compile(r"\bnext\s+month\b", re.IGNORECASE),
                self._resolve_next_month,
            ),
            (
                re.compile(r"\bnext\s+quarter\b", re.IGNORECASE),
                self._resolve_next_quarter,
            ),
            (
                re.compile(r"\blast\s+quarter\b", re.IGNORECASE),
                self._resolve_last_quarter,
            ),
            (
                re.compile(r"\b(\d+)\s+years?\s+ago\b", re.IGNORECASE),
                self._resolve_n_years_ago,
            ),
            (
                re.compile(r"\b(\d+)\s+months?\s+ago\b", re.IGNORECASE),
                self._resolve_n_months_ago,
            ),
            (
                re.compile(r"\blast\s+week\b", re.IGNORECASE),
                self._resolve_last_week,
            ),
            (
                re.compile(r"\byesterday\b", re.IGNORECASE),
                self._resolve_yesterday,
            ),
        ]

    def resolve(self, extraction: ExtractionResult, reference_date: datetime) -> ExtractionResult:
        """Resolve temporal expressions in all relationships.

        Modifies the ExtractionResult in place and returns it.

        Args:
            extraction: The ExtractionResult with relationships that may
                contain temporal_expression strings.
            reference_date: The reference date for resolving relative
                expressions (typically today's date).

        Returns:
            The same ExtractionResult with resolved dates.
        """
        resolutions = 0

        for rel in extraction.relationships:
            props = rel.get("properties", {})
            if props is None:
                continue

            temporal_expr = props.get("temporal_expression")
            if not temporal_expr or not isinstance(temporal_expr, str):
                continue

            # Try each pattern
            for pattern, resolver in self._patterns:
                match = pattern.search(temporal_expr)
                if match:
                    start_date, end_date, status = resolver(match, reference_date)

                    if start_date is not None:
                        props["start_date"] = start_date
                    if end_date is not None:
                        props["end_date"] = end_date
                    if status is not None and not props.get("temporal_status"):
                        props["temporal_status"] = status

                    resolutions += 1
                    break  # First matching pattern wins

        if resolutions > 0:
            logger.debug("Resolved %d temporal expressions", resolutions)

        return extraction

    # --- Resolver methods ---
    # Each returns (start_date_str | None, end_date_str | None, temporal_status | None)

    @staticmethod
    def _resolve_last_year(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        year = ref.year - 1
        return (f"{year}-01-01", f"{year}-12-31", "past")

    @staticmethod
    def _resolve_this_year(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        return (f"{ref.year}-01-01", None, "current")

    @staticmethod
    def _resolve_next_year(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        year = ref.year + 1
        return (f"{year}-01-01", f"{year}-12-31", "future")

    @staticmethod
    def _resolve_last_month(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        if ref.month == 1:
            year, month = ref.year - 1, 12
        else:
            year, month = ref.year, ref.month - 1
        last_day = calendar.monthrange(year, month)[1]
        return (f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}", "past")

    @staticmethod
    def _resolve_this_month(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        return (f"{ref.year}-{ref.month:02d}-01", None, "current")

    @staticmethod
    def _resolve_next_month(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        if ref.month == 12:
            year, month = ref.year + 1, 1
        else:
            year, month = ref.year, ref.month + 1
        last_day = calendar.monthrange(year, month)[1]
        return (f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}", "future")

    @staticmethod
    def _resolve_next_quarter(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        current_quarter = (ref.month - 1) // 3 + 1
        if current_quarter == 4:
            next_q_start_month = 1
            year = ref.year + 1
        else:
            next_q_start_month = current_quarter * 3 + 1
            year = ref.year
        return (f"{year}-{next_q_start_month:02d}-01", None, "future")

    @staticmethod
    def _resolve_last_quarter(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        current_quarter = (ref.month - 1) // 3 + 1
        if current_quarter == 1:
            prev_q_start_month = 10
            year = ref.year - 1
        else:
            prev_q_start_month = (current_quarter - 2) * 3 + 1
            year = ref.year
        end_month = prev_q_start_month + 2
        last_day = calendar.monthrange(year, end_month)[1]
        return (
            f"{year}-{prev_q_start_month:02d}-01",
            f"{year}-{end_month:02d}-{last_day:02d}",
            "past",
        )

    @staticmethod
    def _resolve_n_years_ago(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        n = int(match.group(1))
        year = ref.year - n
        return (f"{year}-01-01", None, "past")

    @staticmethod
    def _resolve_n_months_ago(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        n = int(match.group(1))
        year = ref.year
        month = ref.month - n
        while month <= 0:
            month += 12
            year -= 1
        return (f"{year}-{month:02d}-01", None, "past")

    @staticmethod
    def _resolve_last_week(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        from datetime import timedelta

        # Monday of last week
        days_since_monday = ref.weekday()
        this_monday = ref - timedelta(days=days_since_monday)
        last_monday = this_monday - timedelta(days=7)
        last_sunday = last_monday + timedelta(days=6)
        return (
            last_monday.strftime("%Y-%m-%d"),
            last_sunday.strftime("%Y-%m-%d"),
            "past",
        )

    @staticmethod
    def _resolve_yesterday(
        match: re.Match, ref: datetime
    ) -> tuple[str | None, str | None, str | None]:
        from datetime import timedelta

        yesterday = ref - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")
        return (date_str, date_str, "past")
