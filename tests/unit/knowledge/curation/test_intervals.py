"""C1 valid-time interval normalization (design 4.5)."""

from datetime import UTC, datetime

from backend.knowledge.curation.intervals import (
    ALWAYS,
    Interval,
    is_empty,
    overlaps,
    parse_from_bound,
    parse_to_bound,
)


def _dt(*args):
    return datetime(*args, tzinfo=UTC)


class TestParseBounds:
    def test_year_precision_closed_open(self):
        assert parse_from_bound("2024") == _dt(2024, 1, 1)
        assert parse_to_bound("2024") == _dt(2025, 1, 1)

    def test_month_precision(self):
        assert parse_from_bound("2026-05") == _dt(2026, 5, 1)
        assert parse_to_bound("2026-05") == _dt(2026, 6, 1)
        assert parse_to_bound("2026-12") == _dt(2027, 1, 1)

    def test_day_precision(self):
        assert parse_from_bound("2026-05-20") == _dt(2026, 5, 20)
        assert parse_to_bound("2026-05-20") == _dt(2026, 5, 21)

    def test_full_datetime_passthrough_and_naive_becomes_utc(self):
        assert parse_from_bound("2026-05-20T10:30:00+00:00") == _dt(2026, 5, 20, 10, 30)
        assert parse_from_bound("2026-05-20T10:30:00") == _dt(2026, 5, 20, 10, 30)

    def test_always_sentinel_is_open_start(self):
        assert parse_from_bound(ALWAYS) is None

    def test_unparseable_returns_none(self):
        assert parse_from_bound("sometime soonish") is None
        assert parse_to_bound("") is None


class TestOverlaps:
    def test_disjoint(self):
        a = Interval(_dt(2020, 1, 1), _dt(2021, 1, 1))
        b = Interval(_dt(2022, 1, 1), _dt(2023, 1, 1))
        assert not overlaps(a, b)

    def test_closed_open_touching_is_not_overlap(self):
        a = Interval(_dt(2020, 1, 1), _dt(2021, 1, 1))
        b = Interval(_dt(2021, 1, 1), None)
        assert not overlaps(a, b)

    def test_open_ended_overlaps_later_open(self):
        a = Interval(_dt(2024, 1, 1), None)
        b = Interval(_dt(2027, 1, 1), None)
        assert overlaps(a, b)

    def test_contained(self):
        a = Interval(_dt(2020, 1, 1), None)
        b = Interval(_dt(2021, 1, 1), _dt(2022, 1, 1))
        assert overlaps(a, b)

    def test_none_start_is_minus_infinity(self):
        a = Interval(None, _dt(2021, 1, 1))
        b = Interval(_dt(1990, 1, 1), _dt(1991, 1, 1))
        assert overlaps(a, b)


class TestEmpty:
    def test_zero_width_is_empty(self):
        t = _dt(2026, 6, 10)
        assert is_empty(Interval(t, t))

    def test_open_is_not_empty(self):
        assert not is_empty(Interval(_dt(2026, 6, 10), None))
