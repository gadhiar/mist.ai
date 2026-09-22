"""Tests for backend/event_store/uptime.py -- the historical uptime derivation.

See that module's docstring for the full spec this pins: gap classification
into continuous/outage/restart, the longest-continuous-run lower bound, and
the null-not-zero rule for an empty or single-row ledger.

Per tests/CLAUDE.md's rule that expectations must not be circular, no expected
value here is produced by calling the function under test. They come from two
different routes, and the distinction matters when reading a failure:

- `test_longest_continuous_run_matches_independent_boundary_arithmetic`
  subtracts two timestamp literals taken from the CSV, so it recomputes the
  answer by an independent route.
- Every other expectation -- the row count, the event count, the plateau
  parameters -- is a bare literal, hand-derived from the fixture once and
  written down. It is non-circular but NOT independently recomputed: if the
  fixture is ever regenerated, these literals must be re-derived by hand
  rather than pasted from a run of this module.
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.event_store.uptime import (
    INTERVAL_SECONDS,
    TOLERANCE_SECONDS,
    build_uptime_report,
    derive_uptime,
    format_uptime_report,
    parse_started_at,
    read_curation_job_rows,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "uptime"
    / "curation_job_runs_2026-09-22.csv"
)


def _load_fixture_rows(job_name: str = "confidence_decay") -> list[tuple[str, str]]:
    """Read (started_at, trigger_source) for one job straight off the CSV.

    Independent of `read_curation_job_rows` -- this is the ground truth the
    SQLite-backed reader is checked against, not a shared implementation.
    """
    with open(FIXTURE_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            (row["started_at"], row["trigger_source"])
            for row in reader
            if row["job_name"] == job_name
        ]


def _build_fixture_db(tmp_path: Path) -> Path:
    """Build a real SQLite store carrying curation_job_runs from the fixture CSV."""
    db_path = tmp_path / "event_store.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE curation_job_runs (
            run_id TEXT PRIMARY KEY,
            job_name TEXT NOT NULL,
            trigger_source TEXT NOT NULL,
            started_at TEXT NOT NULL,
            duration_ms REAL NOT NULL,
            outcome TEXT NOT NULL,
            result_type TEXT,
            examined INTEGER,
            produced INTEGER,
            metrics TEXT,
            error TEXT
        )
        """
    )
    with open(FIXTURE_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            conn.execute(
                "INSERT INTO curation_job_runs "
                "(run_id, job_name, trigger_source, started_at, duration_ms, outcome, "
                "examined, produced) VALUES (?, ?, ?, ?, 0, ?, ?, ?)",
                (
                    f"row-{i}",
                    row["job_name"],
                    row["trigger_source"],
                    row["started_at"],
                    row["outcome"],
                    row["examined"] or None,
                    row["produced"] or None,
                ),
            )
    conn.commit()
    conn.close()
    return db_path


class TestRealFixture:
    """End-to-end regression test against the committed live-ledger export."""

    def test_confidence_decay_scheduled_row_count_is_39(self):
        rows = _load_fixture_rows()
        scheduled = [r for r in rows if r[1] == "scheduled"]

        assert len(scheduled) == 39

    def test_window_matches_first_and_last_scheduled_row(self):
        rows = _load_fixture_rows()

        report = build_uptime_report(rows, job_name="confidence_decay")

        assert report.window_start == datetime.fromisoformat("2026-08-04T15:19:26.173441+00:00")
        assert report.window_end == datetime.fromisoformat("2026-09-21T17:42:38.016488+00:00")

    def test_event_count_at_default_tolerance_is_27(self):
        rows = _load_fixture_rows()

        report = build_uptime_report(rows, job_name="confidence_decay")

        assert report.event_count == 27

    def test_longest_continuous_run_matches_independent_boundary_arithmetic(self):
        rows = _load_fixture_rows()

        report = build_uptime_report(rows, job_name="confidence_decay")

        # Independent route: the two boundary timestamps, subtracted directly,
        # not derived by calling derive_uptime or build_uptime_report again.
        expected_start = datetime.fromisoformat("2026-09-15T15:24:45.705708+00:00")
        expected_end = datetime.fromisoformat("2026-09-18T15:26:53.688059+00:00")
        expected_duration = (expected_end - expected_start).total_seconds()

        assert report.longest_run is not None
        assert report.longest_run.start_at == expected_start
        assert report.longest_run.end_at == expected_end
        assert report.longest_run.duration_seconds == expected_duration

    def test_manual_and_unparseable_exclusion_counts_are_zero_on_the_real_fixture(self):
        rows = _load_fixture_rows()

        report = build_uptime_report(rows, job_name="confidence_decay")

        # Documented in the module brief: trigger_source is 100% 'scheduled'
        # and outcome 100% 'completed' in this fixture, so the manual filter
        # and the parse filter both exclude nothing today.
        assert report.rows_excluded_manual == 0
        assert report.rows_excluded_unparseable == 0


class TestTolerancePlateau:
    """The event count is identical across a wide tolerance range -- the knob is not load-bearing."""

    @pytest.mark.parametrize(
        "tolerance_seconds, expected_count",
        [
            pytest.param(60, 27, id="tolerance-60s"),
            pytest.param(120, 27, id="tolerance-120s"),
            pytest.param(300, 27, id="tolerance-300s-default"),
            pytest.param(500, 27, id="tolerance-500s"),
            pytest.param(600, 26, id="tolerance-600s-plateau-edge"),
        ],
    )
    def test_event_count_at_tolerance(self, tolerance_seconds, expected_count):
        rows = _load_fixture_rows()
        timestamps = [
            parse_started_at(started_at) for started_at, trigger in rows if trigger == "scheduled"
        ]

        derivation = derive_uptime(timestamps, tolerance_seconds=tolerance_seconds)

        assert len(derivation.events) == expected_count


class TestGoldenSyntheticShape:
    """A small, fully hand-designed timeline exercising every gap kind."""

    def _ts(self, iso: str) -> datetime:
        return datetime.fromisoformat(iso)

    def test_designed_gaps_produce_exact_kinds_and_bounds(self):
        # Day 0: baseline.
        # Day 1: +86400s + 40s drift -> continuous (well inside 300s tolerance).
        # Then: +200000s (~55.6h) -> outage. Downtime bounded
        #   [200000 - 86400, 200000] = [113600, 200000] seconds.
        # Then: +1800s (30min) -> restart. Downtime bounded [None, 1800].
        # Then: +86400s exactly -> continuous, closing a second short run.
        t0 = self._ts("2026-01-01T00:00:00+00:00")
        t1 = t0 + _seconds(86440)  # continuous
        t2 = t1 + _seconds(200000)  # outage
        t3 = t2 + _seconds(1800)  # restart
        t4 = t3 + _seconds(86400)  # continuous

        derivation = derive_uptime([t0, t1, t2, t3, t4])

        assert len(derivation.events) == 2
        outage, restart = derivation.events

        assert outage.kind == "outage"
        assert outage.previous_at == t1
        assert outage.resumed_at == t2
        assert outage.downtime_lower_seconds == pytest.approx(113600.0)
        assert outage.downtime_upper_seconds == pytest.approx(200000.0)

        assert restart.kind == "restart"
        assert restart.previous_at == t2
        assert restart.resumed_at == t3
        assert restart.downtime_lower_seconds is None
        assert restart.downtime_upper_seconds == pytest.approx(1800.0)

        # Two continuous chains of length 1 gap each (t0->t1 and t3->t4), tied.
        # Either is a valid "longest" run; assert the invariant rather than
        # picking one arbitrarily -- both have the same duration.
        assert derivation.longest_run is not None
        assert derivation.longest_run.duration_seconds == pytest.approx(86440.0)


class TestGapBoundaries:
    """Exact classification boundaries, one gap per case."""

    def _derive_two_point_gap(self, gap_seconds: float):
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        t1 = t0 + _seconds(gap_seconds)
        return derive_uptime([t0, t1])

    def test_gap_exactly_at_interval_is_continuous(self):
        derivation = self._derive_two_point_gap(INTERVAL_SECONDS)

        assert derivation.events == ()

    def test_gap_at_interval_plus_tolerance_is_continuous(self):
        derivation = self._derive_two_point_gap(INTERVAL_SECONDS + TOLERANCE_SECONDS)

        assert derivation.events == ()

    def test_gap_one_second_past_interval_plus_tolerance_is_outage(self):
        derivation = self._derive_two_point_gap(INTERVAL_SECONDS + TOLERANCE_SECONDS + 1)

        assert len(derivation.events) == 1
        assert derivation.events[0].kind == "outage"

    def test_gap_at_interval_minus_tolerance_is_continuous(self):
        derivation = self._derive_two_point_gap(INTERVAL_SECONDS - TOLERANCE_SECONDS)

        assert derivation.events == ()

    def test_gap_one_second_past_interval_minus_tolerance_is_restart(self):
        derivation = self._derive_two_point_gap(INTERVAL_SECONDS - TOLERANCE_SECONDS - 1)

        assert len(derivation.events) == 1
        assert derivation.events[0].kind == "restart"


class TestWindowEdge:
    """The trailing span since the last row is reported as unmeasured, never an outage."""

    def test_events_and_longest_run_are_unaffected_by_how_stale_the_window_is(self):
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        t1 = t0 + _seconds(INTERVAL_SECONDS)
        t2 = t1 + _seconds(INTERVAL_SECONDS)

        derivation = derive_uptime([t0, t1, t2])

        # derive_uptime takes no `now` at all -- staleness cannot affect it.
        assert derivation.events == ()
        assert derivation.longest_run is not None
        assert derivation.longest_run.duration_seconds == pytest.approx(2 * INTERVAL_SECONDS)

    def test_formatter_reports_trailing_span_as_unmeasured_not_an_outage(self):
        rows = [
            ("2026-01-01T00:00:00+00:00", "scheduled"),
            ("2026-01-02T00:00:00+00:00", "scheduled"),
        ]
        report = build_uptime_report(rows, job_name="confidence_decay")
        now = datetime.fromisoformat("2026-01-12T00:00:00+00:00")  # 10 days after last row

        text = format_uptime_report(report, now=now)

        assert "NOT an outage and NOT uptime" in text
        assert "240.00h" in text  # 10 days of trailing, unmeasured span


class TestSingleRow:
    def test_longest_run_and_event_count_are_none_not_zero(self):
        derivation = derive_uptime([datetime(2026, 1, 1, tzinfo=UTC)])

        assert derivation.longest_run is None
        assert derivation.events == ()
        assert derivation.reason is not None

        report = build_uptime_report(
            [("2026-01-01T00:00:00+00:00", "scheduled")], job_name="confidence_decay"
        )

        assert report.event_count is None
        assert report.event_count != 0
        assert report.longest_run is None
        assert report.reason is not None


class TestEmptyTable:
    def test_every_metric_is_none_not_zero(self):
        report = build_uptime_report([], job_name="confidence_decay")

        assert report.window_start is None
        assert report.window_end is None
        assert report.event_count is None
        assert report.event_count != 0
        assert report.longest_run is None
        assert report.reason is not None
        assert report.events == ()


class TestCaveatNamesTheActualCount:
    """The deliberate-shutdown caveat quotes this run's count, never a baked-in literal.

    An earlier draft hardcoded the count the derivation produces on the committed
    fixture, so the caveat would have printed that same number against any other
    store -- a confident number published without the thing it claims to measure,
    which is exactly what the caveat exists to prevent.
    """

    def test_caveat_quotes_the_derived_event_count(self):
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        rows = [
            (t0.isoformat(), "scheduled"),
            ((t0 + _seconds(INTERVAL_SECONDS)).isoformat(), "scheduled"),
            # 1h after the previous row: far below the interval, so one restart.
            ((t0 + _seconds(INTERVAL_SECONDS + 3600)).isoformat(), "scheduled"),
        ]
        report = build_uptime_report(rows, job_name="confidence_decay")

        text = format_uptime_report(report, now=t0 + _seconds(3 * INTERVAL_SECONDS))

        assert report.event_count == 1
        assert '"1 events" as "1 failures"' in text
        assert '"27 events"' not in text

    def test_caveat_drops_the_count_when_nothing_was_measured(self):
        report = build_uptime_report([], job_name="confidence_decay")

        text = format_uptime_report(report, now=datetime(2026, 1, 1, tzinfo=UTC))

        assert "Do not read the event count below as a failure count." in text
        assert 'events" as "' not in text


class TestManualRowsExcluded:
    def test_manual_row_positioned_to_manufacture_false_restarts_is_excluded(self):
        # Two clean 24h-interval scheduled rows.
        t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        t1 = t0 + _seconds(INTERVAL_SECONDS)
        # A manual row dropped in the middle of that gap would, if NOT
        # filtered, split one continuous 24h gap into two ~12h gaps -- both
        # well under interval - tolerance, i.e. two false "restart" events.
        t_manual = t0 + _seconds(INTERVAL_SECONDS / 2)

        rows = [
            (t0.isoformat(), "scheduled"),
            (t_manual.isoformat(), "manual"),
            (t1.isoformat(), "scheduled"),
        ]

        report = build_uptime_report(rows, job_name="confidence_decay")

        assert report.rows_excluded_manual == 1
        assert report.event_count == 0  # the real 24h gap is continuous
        assert report.events == ()


class TestFailedRowRetained:
    def test_failed_outcome_scheduled_row_still_counts_as_liveness(self):
        # build_uptime_report never reads `outcome` at all -- this test
        # documents that a failed row is retained by construction (the
        # filter is on trigger_source only), not merely by absence of a
        # counter-example.
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        t1 = t0 + _seconds(INTERVAL_SECONDS)
        rows = [(t0.isoformat(), "scheduled"), (t1.isoformat(), "scheduled")]

        report = build_uptime_report(rows, job_name="confidence_decay")

        assert report.rows_used == 2
        assert report.window_end == t1


class TestReadCurationJobRows:
    """The SQLite-backed reader, checked against the same fixture the pure tests use."""

    def test_reads_same_rows_as_the_independent_csv_route(self, tmp_path: Path):
        db_path = _build_fixture_db(tmp_path)
        expected = _load_fixture_rows("confidence_decay")

        rows = read_curation_job_rows(str(db_path), "confidence_decay")

        # Full tuple equality, not a length plus a set of trigger_source values:
        # every row in this fixture is 'scheduled', so that set comparison reduced
        # to {"scheduled"} == {"scheduled"} and never once compared a started_at.
        # A SELECT that returned the wrong column, or the right columns in the
        # wrong order, passed it -- while making build_uptime_report filter every
        # row away and report "nothing measured" for a populated store.
        assert rows == expected

    def test_does_not_create_a_file_at_a_missing_path(self, tmp_path: Path):
        missing = tmp_path / "does-not-exist.db"

        with pytest.raises(sqlite3.OperationalError):
            read_curation_job_rows(str(missing), "confidence_decay")

        assert not missing.exists()


def _seconds(n: float):
    from datetime import timedelta

    return timedelta(seconds=n)
