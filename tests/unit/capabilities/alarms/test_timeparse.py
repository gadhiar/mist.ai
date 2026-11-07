"""Tests for spoken alarm times (`parse_alarm_time`) and repeat rules (`next_occurrence`)."""

from __future__ import annotations

import re
import zoneinfo
from datetime import UTC, datetime, timedelta, timezone, tzinfo

import pytest

from backend.capabilities.alarms.timeparse import (
    RepeatRule,
    next_occurrence,
    parse_alarm_time,
    parse_repeat,
)
from backend.errors import TimeParseError

# A fixed-offset zone five hours behind UTC; the image has no IANA zone data until H1.
TZ = timezone(timedelta(hours=-5))


def local(day: int, hour: int, minute: int = 0, *, month: int = 1, zone: tzinfo = TZ) -> datetime:
    """An instant given as wall time in `zone` in 2026, returned as aware UTC."""
    return datetime(2026, month, day, hour, minute, tzinfo=zone).astimezone(UTC)


# 2026-01-05 is a Monday.
MON_0600 = local(5, 6)
MON_0800 = local(5, 8)
MON_2000 = local(5, 20)


def parse(text: str, day: str | None = None, now: datetime = MON_0600) -> datetime:
    return parse_alarm_time(text, day, now, TZ)


def new_york() -> tzinfo:
    try:
        return zoneinfo.ZoneInfo("America/New_York")
    except zoneinfo.ZoneInfoNotFoundError:
        pytest.skip("tzdata absent until H1")


class TestClockTimes:
    @pytest.mark.parametrize("text", ["7am", "7 am", "7AM", "7:00am", "7:00 am"])
    def test_morning_forms(self, text):
        assert parse(text) == local(5, 7)

    @pytest.mark.parametrize("text", ["7:30 pm", "7:30pm", "7:30 PM", "19:30"])
    def test_evening_forms(self, text):
        assert parse(text) == local(5, 19, 30)

    def test_noon(self):
        assert parse("noon") == local(5, 12)

    def test_midnight_is_the_next_midnight(self):
        assert parse("midnight") == local(6, 0)

    def test_twelve_am_is_midnight_and_twelve_pm_is_noon(self):
        assert parse("12am") == local(6, 0)
        assert parse("12 pm") == local(5, 12)

    def test_surrounding_space_and_case_are_ignored(self):
        assert parse("  Noon ") == local(5, 12)

    def test_result_is_aware_utc(self):
        result = parse("7am")

        assert result.tzinfo is UTC

    def test_reads_wall_time_in_the_given_zone(self):
        india = timezone(timedelta(hours=5, minutes=30))
        now = datetime(2026, 1, 5, 6, 0, tzinfo=india)

        result = parse_alarm_time("7am", None, now, india)

        assert result == datetime(2026, 1, 5, 7, 0, tzinfo=india)


class TestBareHour:
    def test_morning_hour_still_ahead_is_today_morning(self):
        assert parse("7", now=MON_0600) == local(5, 7)

    def test_morning_hour_passed_is_this_evening(self):
        assert parse("7", now=MON_0800) == local(5, 19)

    def test_both_passed_is_tomorrow_morning(self):
        assert parse("7", now=MON_2000) == local(6, 7)

    def test_hour_and_minutes_without_meridiem_take_the_next_occurrence(self):
        assert parse("6:30", now=MON_0800) == local(5, 18, 30)

    def test_twelve_is_the_next_of_noon_and_midnight(self):
        assert parse("12", now=MON_0800) == local(5, 12)
        assert parse("12", now=MON_2000) == local(6, 0)

    def test_hours_past_twelve_are_a_24_hour_clock(self):
        assert parse("19", now=MON_0800) == local(5, 19)

    def test_bare_hour_on_a_day_is_that_morning(self):
        assert parse("7", "tomorrow", MON_2000) == local(6, 7)

    def test_hour_without_leading_zero_keeps_the_bare_hour_rule(self):
        assert parse("7:00", now=MON_0800) == local(5, 19)


class TestLeadingZero:
    """A leading-zero hour is a 24-hour clock, never the bare-hour next occurrence."""

    def test_leading_zero_hour_still_ahead_is_today(self):
        assert parse("07:00", now=MON_0600) == local(5, 7)

    def test_leading_zero_hour_passed_rolls_to_tomorrow_morning(self):
        assert parse("07:00", now=MON_0800) == local(6, 7)
        assert parse("07:30", now=MON_0800) == local(6, 7, 30)

    def test_leading_zero_bare_hour_is_24_hour(self):
        assert parse("07", now=MON_0800) == local(6, 7)


class TestSpacedMinutes:
    """T3's normaliser turns "seven thirty pm" into "7 30 pm"."""

    def test_spaced_minutes_with_meridiem(self):
        assert parse("7 30 pm", now=MON_0800) == local(5, 19, 30)
        assert parse("7 30pm", now=MON_0800) == local(5, 19, 30)
        assert parse("7 30 am", now=MON_0600) == local(5, 7, 30)

    def test_spaced_minutes_without_meridiem_follow_the_bare_hour_rule(self):
        assert parse("7 30", now=MON_0600) == local(5, 7, 30)
        assert parse("7 30", now=MON_0800) == local(5, 19, 30)

    def test_spaced_minutes_with_a_leading_zero_hour_are_24_hour(self):
        assert parse("07 30", now=MON_0800) == local(6, 7, 30)


class TestRelative:
    def test_minutes(self):
        assert parse("in 20 minutes") == MON_0600 + timedelta(minutes=20)

    def test_hours(self):
        assert parse("in 2 hours") == MON_0600 + timedelta(hours=2)

    @pytest.mark.parametrize(
        ("text", "delta"),
        [
            ("in 1 minute", timedelta(minutes=1)),
            ("in 1 hour", timedelta(hours=1)),
            ("in an hour", timedelta(hours=1)),
            ("in a minute", timedelta(minutes=1)),
        ],
    )
    def test_singular_units(self, text, delta):
        assert parse(text) == MON_0600 + delta

    def test_relative_crosses_midnight(self):
        assert parse("in 5 hours", now=MON_2000) == local(6, 1)

    def test_relative_accepts_today_as_the_day(self):
        assert parse("in 20 minutes", "today") == MON_0600 + timedelta(minutes=20)

    def test_relative_rejects_another_day(self):
        with pytest.raises(TimeParseError, match="tomorrow"):
            parse("in 20 minutes", "tomorrow")

    def test_zero_is_rejected(self):
        with pytest.raises(TimeParseError, match=re.escape("'in 0 minutes'")):
            parse("in 0 minutes")


class TestDays:
    def test_today(self):
        assert parse("7am", "today") == local(5, 7)

    def test_today_with_a_passed_time_is_rejected_naming_the_input(self):
        with pytest.raises(TimeParseError, match=re.escape("'7am'")):
            parse("7am", "today", MON_0800)

    def test_tomorrow(self):
        assert parse("7am", "tomorrow") == local(6, 7)

    def test_day_is_case_insensitive(self):
        assert parse("7am", "Tomorrow") == local(6, 7)

    def test_weekday_name_is_the_coming_one(self):
        assert parse("7am", "friday") == local(9, 7)

    def test_todays_weekday_with_the_time_still_ahead_is_today(self):
        assert parse("7am", "monday", MON_0600) == local(5, 7)

    def test_todays_weekday_with_the_time_passed_is_next_week(self):
        assert parse("7am", "monday", MON_0800) == local(12, 7)

    def test_iso_date(self):
        assert parse("7:30 pm", "2026-01-20") == local(20, 19, 30)

    def test_past_iso_date_is_rejected(self):
        with pytest.raises(TimeParseError, match="2026-01-01"):
            parse("7am", "2026-01-01")

    @pytest.mark.parametrize("day", ["someday", "next week", "2026-02-30", "funday"])
    def test_unknown_day_is_rejected_naming_it(self, day):
        with pytest.raises(TimeParseError, match=re.escape(repr(day))):
            parse("7am", day)

    def test_empty_day_means_no_day(self):
        assert parse("7am", "") == local(5, 7)


class TestRollover:
    def test_explicit_time_passed_today_rolls_to_tomorrow(self):
        assert parse("7am", now=MON_0800) == local(6, 7)

    def test_explicit_time_equal_to_now_rolls_to_tomorrow(self):
        assert parse("6am", now=MON_0600) == local(6, 6)

    def test_noon_passed_rolls_to_tomorrow(self):
        assert parse("noon", now=MON_2000) == local(6, 12)


class TestRejected:
    @pytest.mark.parametrize(
        "text",
        [
            "whenever",
            "",
            "25:00",
            "7:60",
            "13pm",
            "0am",
            "24",
            "7 75",
            "7 5",
            "7 75 pm",
            "in minutes",
            "in 20 days",
            "7am tomorrow",
            "-1",
        ],
    )
    def test_raises_naming_the_input(self, text):
        with pytest.raises(TimeParseError, match=re.escape(repr(text))):
            parse(text)

    def test_naive_now_is_a_caller_error(self):
        with pytest.raises(ValueError, match="aware"):
            parse_alarm_time("7am", None, datetime(2026, 1, 5, 6, 0), TZ)


class TestRepeatRule:
    def test_json_round_trip(self):
        rule = RepeatRule(kind="weekdays", hour=6, minute=45)

        assert RepeatRule.from_json(rule.to_json()) == rule

    def test_json_is_an_object_with_kind_hour_and_minute(self):
        assert RepeatRule(kind="daily", hour=7, minute=5).to_json() == (
            '{"kind": "daily", "hour": 7, "minute": 5}'
        )

    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            "[1, 2]",
            '{"kind": "hourly", "hour": 7, "minute": 0}',
            '{"kind": "daily", "hour": 24, "minute": 0}',
            '{"kind": "daily", "hour": 7, "minute": 60}',
            '{"kind": "daily", "hour": "7", "minute": 0}',
            '{"kind": "daily", "hour": true, "minute": 0}',
            '{"kind": "daily", "hour": 7}',
            '{"kind": ["daily"], "hour": 7, "minute": 0}',
            '{"kind": {"daily": 1}, "hour": 7, "minute": 0}',
            '{"kind": "daily", "hour": [7], "minute": 0}',
            '{"kind": "daily", "hour": 7, "minute": 0, "extra": 1}',
            "null",
            "7",
            '"daily"',
        ],
    )
    def test_from_json_rejects_malformed_text(self, text):
        with pytest.raises(ValueError):
            RepeatRule.from_json(text)

    @pytest.mark.parametrize("value", [b"{}", None, 7])
    def test_from_json_rejects_a_value_that_is_not_text_with_value_error(self, value):
        # A stored cell can hold any SQLite type. `json.loads` accepts bytes, so without the
        # guard bytes holding a valid rule would decode; a stored rule must be text. For None
        # and an int, `json.loads` raises `TypeError`, which a reader of stored rules does not
        # treat as a damaged row. "JSON text" names the guard's message, so the bytes case
        # fails when the guard is gone even though `{}` would be rejected later.
        with pytest.raises(ValueError, match="JSON text"):
            RepeatRule.from_json(value)  # type: ignore[arg-type]

    def test_from_json_rejects_deeply_nested_json_with_value_error(self):
        nested = "[" * 100000 + "]" * 100000

        with pytest.raises(ValueError, match="nested too deeply"):
            RepeatRule.from_json(nested)

    def test_constructor_rejects_an_unknown_kind(self):
        with pytest.raises(ValueError, match="hourly"):
            RepeatRule(kind="hourly", hour=7, minute=0)  # type: ignore[arg-type]

    def test_constructor_rejects_an_unhashable_kind_with_value_error(self):
        with pytest.raises(ValueError, match="daily"):
            RepeatRule(kind=["daily"], hour=7, minute=0)  # type: ignore[arg-type]


class TestNextOccurrence:
    def test_daily_later_today(self):
        assert next_occurrence(RepeatRule("daily", 7, 0), MON_0600, TZ) == local(5, 7)

    def test_daily_after_todays_time_is_tomorrow(self):
        assert next_occurrence(RepeatRule("daily", 7, 0), MON_0800, TZ) == local(6, 7)

    def test_strictly_after(self):
        assert next_occurrence(RepeatRule("daily", 7, 0), local(5, 7), TZ) == local(6, 7)

    def test_weekdays_skip_the_weekend(self):
        friday_0800 = local(9, 8)

        assert next_occurrence(RepeatRule("weekdays", 7, 0), friday_0800, TZ) == local(12, 7)

    def test_weekdays_from_saturday_is_monday(self):
        assert next_occurrence(RepeatRule("weekdays", 7, 0), local(10, 6), TZ) == local(12, 7)

    def test_weekends_from_monday_is_saturday(self):
        assert next_occurrence(RepeatRule("weekends", 9, 30), MON_0800, TZ) == local(10, 9, 30)

    def test_weekends_after_saturdays_time_is_sunday(self):
        assert next_occurrence(RepeatRule("weekends", 9, 30), local(10, 10), TZ) == local(11, 9, 30)

    def test_weekends_after_sundays_time_is_next_saturday(self):
        assert next_occurrence(RepeatRule("weekends", 9, 30), local(11, 10), TZ) == local(17, 9, 30)

    def test_result_is_aware_utc(self):
        assert next_occurrence(RepeatRule("daily", 7, 0), MON_0600, TZ).tzinfo is UTC


class TestDaylightSaving:
    """America/New_York: clocks spring forward 2026-03-08 02:00 and fall back 2026-11-01 02:00."""

    def test_daily_keeps_its_wall_time_across_spring_forward(self):
        zone = new_york()
        saturday_0800 = local(7, 8, month=3, zone=zone)

        result = next_occurrence(RepeatRule("daily", 7, 0), saturday_0800, zone)

        assert result == local(8, 7, month=3, zone=zone)
        assert result == datetime(2026, 3, 8, 11, 0, tzinfo=UTC)

    def test_time_in_the_spring_forward_gap_shifts_forward(self):
        zone = new_york()
        before_gap = datetime(2026, 3, 8, 6, 0, tzinfo=UTC)  # 01:00 EST

        result = next_occurrence(RepeatRule("daily", 2, 30), before_gap, zone)

        assert result == datetime(2026, 3, 8, 7, 30, tzinfo=UTC)  # 03:30 EDT

    def test_ambiguous_fall_back_time_takes_the_first_occurrence(self):
        zone = new_york()
        before_fold = datetime(2026, 11, 1, 4, 0, tzinfo=UTC)  # 00:00 EDT

        result = next_occurrence(RepeatRule("daily", 1, 30), before_fold, zone)

        assert result == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)  # 01:30 EDT

    def test_fall_back_does_not_repeat_the_same_wall_time(self):
        zone = new_york()
        first_0130 = datetime(2026, 11, 1, 5, 30, tzinfo=UTC)

        result = next_occurrence(RepeatRule("daily", 1, 30), first_0130, zone)

        assert result == datetime(2026, 11, 2, 6, 30, tzinfo=UTC)  # 01:30 EST next day

    def test_parsed_time_in_the_gap_shifts_forward(self):
        zone = new_york()
        before_gap = datetime(2026, 3, 8, 6, 0, tzinfo=UTC)

        result = parse_alarm_time("2:30am", None, before_gap, zone)

        assert result == datetime(2026, 3, 8, 7, 30, tzinfo=UTC)

    def test_relative_time_counts_elapsed_time_across_fall_back(self):
        zone = new_york()
        now = datetime(2026, 11, 1, 0, 30, tzinfo=zone)  # 00:30 EDT, 04:30 UTC

        result = parse_alarm_time("in 2 hours", None, now, zone)

        assert result == datetime(2026, 11, 1, 6, 30, tzinfo=UTC)  # 01:30 EST

    def test_relative_time_counts_elapsed_time_across_spring_forward(self):
        zone = new_york()
        now = datetime(2026, 3, 8, 1, 30, tzinfo=zone)  # 01:30 EST, 06:30 UTC

        result = parse_alarm_time("in 2 hours", None, now, zone)

        assert result == datetime(2026, 3, 8, 8, 30, tzinfo=UTC)  # 04:30 EDT

    def test_parsed_ambiguous_time_takes_the_first_occurrence(self):
        zone = new_york()
        before_fold = datetime(2026, 11, 1, 4, 0, tzinfo=UTC)

        result = parse_alarm_time("1:30am", None, before_fold, zone)

        assert result == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)


MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(7)


class TestCustomRepeatRule:
    """`custom` rings on any non-empty set of weekdays, Monday 0 to Sunday 6."""

    def test_custom_accepts_a_weekday_set(self):
        rule = RepeatRule("custom", 7, 0, frozenset({MONDAY, THURSDAY}))

        assert rule.days == frozenset({MONDAY, THURSDAY})

    def test_custom_takes_days_as_a_keyword(self):
        rule = RepeatRule(kind="custom", hour=7, minute=0, days=frozenset({SUNDAY}))

        assert rule.days_set() == frozenset({SUNDAY})

    def test_custom_stores_any_collection_of_days_as_a_frozenset(self):
        rule = RepeatRule("custom", 7, 0, [THURSDAY, MONDAY, THURSDAY])  # type: ignore[arg-type]

        assert rule.days == frozenset({MONDAY, THURSDAY})
        assert isinstance(rule.days, frozenset)

    def test_custom_rule_is_hashable_and_compares_by_its_set(self):
        first = RepeatRule("custom", 7, 0, frozenset({MONDAY, THURSDAY}))
        second = RepeatRule("custom", 7, 0, frozenset({THURSDAY, MONDAY}))

        assert first == second
        assert hash(first) == hash(second)

    def test_custom_accepts_every_day_from_0_to_6(self):
        assert RepeatRule("custom", 7, 0, frozenset(range(7))).days_set() == frozenset(range(7))

    @pytest.mark.parametrize("days", [None, frozenset()])
    def test_custom_without_days_is_refused(self, days):
        with pytest.raises(ValueError, match="custom"):
            RepeatRule("custom", 7, 0, days)

    @pytest.mark.parametrize("kind", ["daily", "weekdays", "weekends"])
    def test_days_on_a_preset_are_refused(self, kind):
        with pytest.raises(ValueError, match=kind):
            RepeatRule(kind, 7, 0, frozenset({MONDAY}))

    @pytest.mark.parametrize("day", [-1, 7, 10])
    def test_an_out_of_range_day_is_refused(self, day):
        with pytest.raises(ValueError, match="0 to 6"):
            RepeatRule("custom", 7, 0, frozenset({MONDAY, day}))

    @pytest.mark.parametrize("day", [True, "0", 1.0, None])
    def test_a_day_that_is_not_an_integer_is_refused(self, day):
        with pytest.raises(ValueError, match="0 to 6"):
            RepeatRule("custom", 7, 0, [day])  # type: ignore[list-item]

    @pytest.mark.parametrize("days", ["03", 3, {"monday": 0}, [[0]]])
    def test_days_that_are_not_a_collection_of_integers_are_refused(self, days):
        with pytest.raises(ValueError):
            RepeatRule("custom", 7, 0, days)  # type: ignore[arg-type]

    def test_hour_and_minute_are_still_checked(self):
        with pytest.raises(ValueError, match="hour"):
            RepeatRule("custom", 24, 0, frozenset({MONDAY}))


class TestDaysSet:
    @pytest.mark.parametrize(
        ("kind", "days"),
        [
            ("daily", frozenset(range(7))),
            ("weekdays", frozenset(range(5))),
            ("weekends", frozenset({SATURDAY, SUNDAY})),
        ],
    )
    def test_presets(self, kind, days):
        assert RepeatRule(kind, 7, 0).days_set() == days

    def test_custom(self):
        rule = RepeatRule("custom", 7, 0, frozenset({TUESDAY, FRIDAY}))

        assert rule.days_set() == frozenset({TUESDAY, FRIDAY})

    def test_presets_hold_no_days(self):
        assert RepeatRule("daily", 7, 0).days is None


class TestCustomRepeatRuleJson:
    @pytest.mark.parametrize(
        ("kind", "text"),
        [
            ("daily", '{"kind": "daily", "hour": 7, "minute": 5}'),
            ("weekdays", '{"kind": "weekdays", "hour": 7, "minute": 5}'),
            ("weekends", '{"kind": "weekends", "hour": 7, "minute": 5}'),
        ],
    )
    def test_preset_json_carries_no_days(self, kind, text):
        assert RepeatRule(kind, 7, 5).to_json() == text

    def test_custom_json_holds_sorted_days(self):
        rule = RepeatRule("custom", 6, 45, frozenset({THURSDAY, MONDAY, SUNDAY}))

        assert rule.to_json() == '{"kind": "custom", "hour": 6, "minute": 45, "days": [0, 3, 6]}'

    def test_custom_round_trip(self):
        rule = RepeatRule("custom", 6, 45, frozenset({SUNDAY, WEDNESDAY}))

        assert RepeatRule.from_json(rule.to_json()) == rule

    @pytest.mark.parametrize(
        "text",
        [
            '{"kind": "custom", "hour": 7, "minute": 0}',
            '{"kind": "custom", "hour": 7, "minute": 0, "days": []}',
            '{"kind": "custom", "hour": 7, "minute": 0, "days": null}',
            '{"kind": "custom", "hour": 7, "minute": 0, "days": [7]}',
            '{"kind": "custom", "hour": 7, "minute": 0, "days": [-1]}',
            '{"kind": "custom", "hour": 7, "minute": 0, "days": [true]}',
            '{"kind": "custom", "hour": 7, "minute": 0, "days": ["0"]}',
            '{"kind": "custom", "hour": 7, "minute": 0, "days": [[0]]}',
            '{"kind": "custom", "hour": 7, "minute": 0, "days": "03"}',
            '{"kind": "custom", "hour": 7, "minute": 0, "days": {"0": 1}}',
            '{"kind": "custom", "hour": 7, "minute": 0, "days": 3}',
            '{"kind": "daily", "hour": 7, "minute": 0, "days": [0]}',
            '{"kind": "daily", "hour": 7, "minute": 0, "days": null}',
            '{"kind": "custom", "hour": 7, "minute": 0, "days": [0], "extra": 1}',
        ],
    )
    def test_from_json_rejects_malformed_custom_text(self, text):
        with pytest.raises(ValueError):
            RepeatRule.from_json(text)


class TestNextOccurrenceCustom:
    MONDAY_THURSDAY = RepeatRule("custom", 7, 0, frozenset({MONDAY, THURSDAY}))

    def test_later_today_on_a_listed_day(self):
        assert next_occurrence(self.MONDAY_THURSDAY, MON_0600, TZ) == local(5, 7)

    def test_skips_unlisted_days(self):
        assert next_occurrence(self.MONDAY_THURSDAY, MON_0800, TZ) == local(8, 7)

    def test_crosses_the_week_boundary(self):
        friday_0800 = local(9, 8)

        assert next_occurrence(self.MONDAY_THURSDAY, friday_0800, TZ) == local(12, 7)

    def test_after_the_last_listed_day_of_the_week(self):
        thursday_0800 = local(8, 8)

        assert next_occurrence(self.MONDAY_THURSDAY, thursday_0800, TZ) == local(12, 7)

    def test_single_day_after_its_time_is_a_week_later(self):
        mondays = RepeatRule("custom", 7, 0, frozenset({MONDAY}))

        assert next_occurrence(mondays, MON_0800, TZ) == local(12, 7)

    def test_sunday_to_monday_crosses_the_week(self):
        sunday_monday = RepeatRule("custom", 7, 0, frozenset({SUNDAY, MONDAY}))
        sunday_0800 = local(11, 8)

        assert next_occurrence(sunday_monday, sunday_0800, TZ) == local(12, 7)

    def test_strictly_after(self):
        assert next_occurrence(self.MONDAY_THURSDAY, local(5, 7), TZ) == local(8, 7)

    def test_every_day_custom_matches_daily(self):
        every_day = RepeatRule("custom", 7, 0, frozenset(range(7)))

        assert next_occurrence(every_day, MON_0800, TZ) == next_occurrence(
            RepeatRule("daily", 7, 0), MON_0800, TZ
        )


class TestCustomDaylightSaving:
    """America/New_York: 2026-03-08 and 2026-11-01 are Sundays."""

    def test_custom_keeps_its_wall_time_across_spring_forward(self):
        zone = new_york()
        thursday_0800 = local(5, 8, month=3, zone=zone)
        mondays = RepeatRule("custom", 7, 0, frozenset({MONDAY}))

        result = next_occurrence(mondays, thursday_0800, zone)

        assert result == datetime(2026, 3, 9, 11, 0, tzinfo=UTC)  # 07:00 EDT

    def test_custom_time_in_the_spring_forward_gap_shifts_forward(self):
        zone = new_york()
        friday = datetime(2026, 3, 6, 12, 0, tzinfo=UTC)
        sundays = RepeatRule("custom", 2, 30, frozenset({SUNDAY}))

        result = next_occurrence(sundays, friday, zone)

        assert result == datetime(2026, 3, 8, 7, 30, tzinfo=UTC)  # 03:30 EDT

    def test_custom_ambiguous_fall_back_time_takes_the_first_occurrence(self):
        zone = new_york()
        friday = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
        sundays = RepeatRule("custom", 1, 30, frozenset({SUNDAY}))

        result = next_occurrence(sundays, friday, zone)

        assert result == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)  # 01:30 EDT

    def test_custom_fall_back_does_not_repeat_the_same_wall_time(self):
        zone = new_york()
        first_0130 = datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
        sundays = RepeatRule("custom", 1, 30, frozenset({SUNDAY}))

        result = next_occurrence(sundays, first_0130, zone)

        assert result == datetime(2026, 11, 8, 6, 30, tzinfo=UTC)  # 01:30 EST a week on


# `parse_repeat` reads `normalize_utterance` output: lower case, punctuation gone, so
# "every Monday, Thursday and Friday" arrives as "every monday thursday and friday".
REPEAT_PHRASES = [
    ("every day", "daily", None),
    ("every weekday", "weekdays", None),
    ("weekdays", "weekdays", None),
    ("on weekends", "weekends", None),
    ("every monday", "custom", {MONDAY}),
    ("every sunday", "custom", {SUNDAY}),
    ("every monday and thursday", "custom", {MONDAY, THURSDAY}),
    ("every saturday and sunday", "custom", {SATURDAY, SUNDAY}),
    ("every monday wednesday", "custom", {MONDAY, WEDNESDAY}),
    ("every monday wednesday and friday", "custom", {MONDAY, WEDNESDAY, FRIDAY}),
    ("every tuesday thursday saturday", "custom", {TUESDAY, THURSDAY, SATURDAY}),
    (
        "every monday tuesday wednesday thursday friday saturday and sunday",
        "custom",
        set(range(7)),
    ),
    ("every friday and monday", "custom", {MONDAY, FRIDAY}),
    ("  every   monday and thursday ", "custom", {MONDAY, THURSDAY}),
]

NOT_REPEAT_PHRASES = [
    "every now and then",
    "every 5 minutes",
    "every 2 days",
    "every other day",
    "every week",
    "every month",
    "every",
    "",
    "monday",
    "on monday",
    "every mon",
    "every funday",
    "every monday and",
    "every and monday",
    "every monday and tuesday and friday",
    "every day at 7am",
    "set an alarm every day",
    "every weekday and monday",
    "weekday",
    "every weekend",
]


class TestParseRepeat:
    def test_fixture_size(self):
        assert len(REPEAT_PHRASES) >= 12
        assert len(NOT_REPEAT_PHRASES) >= 12

    @pytest.mark.parametrize(("text", "kind", "days"), REPEAT_PHRASES)
    def test_reads_a_repeat_phrase(self, text, kind, days):
        rule = parse_repeat(text, hour=7, minute=0)

        assert rule is not None
        assert rule.kind == kind
        assert rule.days == (None if days is None else frozenset(days))

    @pytest.mark.parametrize("text", NOT_REPEAT_PHRASES)
    def test_anything_else_is_none(self, text):
        assert parse_repeat(text, hour=7, minute=0) is None

    def test_reads_case_and_commas_too(self):
        rule = parse_repeat("Every Monday, Thursday", hour=7, minute=0)

        assert rule is not None
        assert rule.days_set() == frozenset({MONDAY, THURSDAY})

    def test_the_rule_rings_at_the_given_time(self):
        rule = parse_repeat("every monday and thursday", hour=6, minute=30)

        assert rule == RepeatRule("custom", 6, 30, frozenset({MONDAY, THURSDAY}))

    @pytest.mark.parametrize("kwargs", [{}, {"hour": 7}, {"minute": 0}])
    def test_the_time_is_required(self, kwargs):
        # A caller that forgets the time must fail at the call, not arm a midnight alarm.
        with pytest.raises(TypeError):
            parse_repeat("every day", **kwargs)

    def test_the_time_is_keyword_only(self):
        with pytest.raises(TypeError):
            parse_repeat("every day", 7, 0)  # type: ignore[misc]

    def test_an_out_of_range_time_is_refused(self):
        with pytest.raises(ValueError, match="hour"):
            parse_repeat("every day", hour=24, minute=0)
