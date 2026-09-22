"""Historical uptime, derived from the curation job ledger.

Nothing in the tree measures backend process uptime directly:
`grep -rniE "uptime|boot_time|process_start" backend/ --include=*.py` returns
nothing. What exists by accident is `curation_job_runs`: the curation
scheduler's loop seeds `last_run.get(name, 0.0)` for every job, so every
enabled job is "due" on the loop's very first pass
(`backend/knowledge/curation/scheduler.py:305-311`, comment and code agree).
A row in that table therefore marks a moment the backend process was alive,
and the first row after a gap marks a restart.

That is an inference, not a measurement, and the derivation below is built to
say so at every step:

- A gap close to the nominal interval is CONTINUOUS: nothing to report.
- A gap much longer than the interval is an OUTAGE. The process was alive at
  the row before it and went down somewhere in the interval that followed --
  so downtime is reported as a BOUND (`gap - interval` to `gap`), never a
  point estimate.
- A gap much shorter than the interval is a RESTART: the job re-seeded early
  because the process came back up. Downtime here has NO lower bound (the
  process could have been down for a second or an hour before restarting)
  and an upper bound of `gap`.
- Nothing outside `[first row, last row]` is measured. The span from the
  last row to "now" is reported as unmeasured, explicitly NOT an outage --
  the ledger goes quiet with a perfectly healthy process whenever
  `MIST_CURATION_SCHEDULER_ENABLED` is off
  (`backend/knowledge/curation/scheduler.py:271-273`) or hydration isolation
  is active (`backend/knowledge/curation/scheduler.py:260-270`).
- An empty ledger or a single row produces `None` metrics plus a reason
  string, never a `0`. Zero measured outages and zero rows measured are
  different facts.

Constants
---------
`INTERVAL_SECONDS` mirrors `backend/factories.py:741`
(`JobConfig(name="confidence_decay", interval_seconds=86400)`), which
hardcodes the nominal interval for the default job. This is a duplication --
the two files are not linked by any import -- so if the factory's interval
for `confidence_decay` ever changes, this constant must change with it by
hand.

`POLL_SECONDS` mirrors `backend/knowledge/curation/scheduler.py:326-327`
(`await asyncio.sleep(60)`, "1 minute granularity"). No gap this module
computes can resolve finer than one poll.

`TOLERANCE_SECONDS = 300.0` is not a judgement call; it sits in the middle of
a measured plateau. On the committed fixture
(`tests/fixtures/uptime/curation_job_runs_2026-09-22.csv`), drift on the
genuine ~24h gaps is, in seconds:
3.4, 12.7, 20.0, 22.3, 26.9, 36.6, 37.9, 46.1, 46.6, 49.0, 55.0 -- a maximum
of 55s, comfortably inside one 60s poll. The next deviation up is 521.2s, so
every tolerance from 60s to 500s classifies every gap in the fixture
identically; only at 600s does an outage that was 521.2s over the interval
flip to being read as continuous. Measured event-count sensitivity across the
fixture: 60->27, 120->27, 300->27, 500->27, 600->26, 1800->23, 3600->22.
300 sits in the middle of the flat part of that curve, not at either edge.

Reproduction: `python -c` loading the fixture CSV and running
`derive_uptime` at each tolerance reproduces the table above; the same
computation is pinned as a regression test in
`tests/unit/event_store/test_uptime.py` (the "tolerance plateau" test class).

Module structure
-----------------
Three separately testable concerns, deliberately not one function:

- `derive_uptime`: pure, over already-parsed, already-filtered timestamps.
  No I/O, no clock reads.
- `read_curation_job_rows`: the only I/O. Opens the event store strictly
  read-only via a `file:...?mode=ro` URI (mirroring
  `scripts/mist_admin.py:929`'s `_assert_replay_source_exists` idiom), never
  through `EventStore._get_connection`
  (`backend/event_store/store.py:50-67`), which is a plain read-write
  `sqlite3.connect` that issues `PRAGMA journal_mode=WAL` and creates the
  file if it is absent -- both unacceptable side effects for a report that
  must never write to the store it is reading.
- `format_uptime_report`: pure string formatting, given a report and an
  injected `now` (so tests never depend on the wall clock, per
  `tests/CLAUDE.md`'s determinism rule).

`build_uptime_report` composes the filtering/parsing step and `derive_uptime`
into one `UptimeReport`; it is pure over the raw `(started_at, trigger_source)`
tuples `read_curation_job_rows` returns, so it can be tested without SQLite.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from backend.knowledge.curation.run_record import TRIGGER_SCHEDULED

# Mirrors backend/factories.py:741 -- see module docstring for why this is a
# hand-maintained duplication rather than an import.
INTERVAL_SECONDS = 86400.0

# Mirrors backend/knowledge/curation/scheduler.py:326-327.
POLL_SECONDS = 60.0

# Empirically justified in the module docstring: the fixture's genuine gaps
# drift at most 55s from the nominal interval, the next deviation is 521.2s,
# and the event count is identical for every tolerance from 60s to 500s.
TOLERANCE_SECONDS = 300.0

EventKind = Literal["outage", "restart"]


@dataclass(frozen=True, slots=True)
class UptimeEvent:
    """One gap classified as an outage or a restart. Continuous gaps are not events."""

    kind: EventKind
    previous_at: datetime
    resumed_at: datetime
    gap_seconds: float
    # None for a restart: downtime before an early re-seed has no lower bound.
    downtime_lower_seconds: float | None
    downtime_upper_seconds: float


@dataclass(frozen=True, slots=True)
class LongestRun:
    """The longest maximal chain of continuous gaps.

    `duration_seconds` is a LOWER bound: the run began within one poll of
    `start_at` and ended somewhere in the unmeasured interval after `end_at`.
    """

    start_at: datetime
    end_at: datetime
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class UptimeDerivation:
    """Output of the pure `derive_uptime` step."""

    window_start: datetime | None
    window_end: datetime | None
    events: tuple[UptimeEvent, ...]
    longest_run: LongestRun | None
    # None when events/longest_run were actually computed. Set, with events
    # and longest_run left empty/None, when there was nothing to compute
    # (0 or 1 usable timestamps) -- the null-not-zero rule.
    reason: str | None


@dataclass(frozen=True, slots=True)
class UptimeReport:
    """Full result: derivation plus the read-side counts that fed it."""

    job_name: str
    interval_seconds: float
    tolerance_seconds: float
    rows_used: int
    rows_excluded_manual: int
    rows_excluded_unparseable: int
    window_start: datetime | None
    window_end: datetime | None
    # None exactly when `reason` is set (0 or 1 usable rows). Never 0 as a
    # stand-in for "not measured".
    event_count: int | None
    events: tuple[UptimeEvent, ...]
    longest_run: LongestRun | None
    reason: str | None


def parse_started_at(raw: str) -> datetime | None:
    """Parse one `started_at` value, returning None rather than raising.

    `CurationScheduler._execute_and_record` writes
    `datetime.now(UTC).isoformat()` (scheduler.py:175), which `fromisoformat`
    round-trips on every Python version this project targets (3.11+). A
    value that does not parse is excluded and counted by the caller, never
    silently dropped.
    """
    try:
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None


def derive_uptime(
    timestamps: Sequence[datetime],
    *,
    interval_seconds: float = INTERVAL_SECONDS,
    tolerance_seconds: float = TOLERANCE_SECONDS,
) -> UptimeDerivation:
    """Classify consecutive gaps between already-filtered, already-parsed timestamps.

    Pure: no I/O, no clock reads. Sorts defensively (`read_curation_job_rows`
    already orders by rowid, but this function's contract does not depend on
    caller ordering).

    Classification, applied to `gap = t[i+1] - t[i]` for every consecutive pair:

    - `abs(gap - interval_seconds) <= tolerance_seconds` -> continuous. No event.
    - `gap > interval_seconds + tolerance_seconds` -> outage. Downtime is at
      least `gap - interval_seconds` and at most `gap`: the process was alive
      at `t[i]` and went down somewhere in the interval after it.
    - `gap < interval_seconds - tolerance_seconds` -> restart. Downtime has no
      lower bound and is at most `gap`.

    The longest continuous run is the longest maximal chain of continuous
    gaps; `LongestRun.duration_seconds` is `t[last] - t[first]` of that chain,
    a lower bound (see `LongestRun` docstring).

    Args:
        timestamps: Parsed `started_at` values for one job, already filtered
            to the trigger_source(s) the caller wants included.
        interval_seconds: Nominal scheduled interval between runs.
        tolerance_seconds: Half-width of the "continuous" band around the
            nominal interval.

    Returns:
        `UptimeDerivation` with `reason` set (and `events`/`longest_run` left
        empty/None) when fewer than 2 timestamps were supplied -- 0 rows
        means nothing was measured, and 1 row bounds no interval.
    """
    ts = sorted(timestamps)
    n = len(ts)

    if n == 0:
        return UptimeDerivation(
            window_start=None,
            window_end=None,
            events=(),
            longest_run=None,
            reason="no rows -- nothing measured, not zero downtime",
        )
    if n == 1:
        return UptimeDerivation(
            window_start=ts[0],
            window_end=ts[0],
            events=(),
            longest_run=None,
            reason=(
                "exactly one row -- a single point bounds no interval, so "
                "longest run and event count are undefined, not zero"
            ),
        )

    events: list[UptimeEvent] = []
    run_start = ts[0]
    run_end = ts[0]
    best_start, best_end = ts[0], ts[0]

    for i in range(n - 1):
        prev, nxt = ts[i], ts[i + 1]
        gap = (nxt - prev).total_seconds()

        if abs(gap - interval_seconds) <= tolerance_seconds:
            run_end = nxt
            continue

        if (run_end - run_start) > (best_end - best_start):
            best_start, best_end = run_start, run_end
        run_start, run_end = nxt, nxt

        if gap > interval_seconds + tolerance_seconds:
            events.append(
                UptimeEvent(
                    kind="outage",
                    previous_at=prev,
                    resumed_at=nxt,
                    gap_seconds=gap,
                    downtime_lower_seconds=gap - interval_seconds,
                    downtime_upper_seconds=gap,
                )
            )
        else:
            events.append(
                UptimeEvent(
                    kind="restart",
                    previous_at=prev,
                    resumed_at=nxt,
                    gap_seconds=gap,
                    downtime_lower_seconds=None,
                    downtime_upper_seconds=gap,
                )
            )

    if (run_end - run_start) > (best_end - best_start):
        best_start, best_end = run_start, run_end

    longest_run = (
        LongestRun(
            start_at=best_start,
            end_at=best_end,
            duration_seconds=(best_end - best_start).total_seconds(),
        )
        if best_end > best_start
        else None
    )

    return UptimeDerivation(
        window_start=ts[0],
        window_end=ts[-1],
        events=tuple(events),
        longest_run=longest_run,
        reason=None,
    )


def build_uptime_report(
    rows: Sequence[tuple[str, str]],
    *,
    job_name: str,
    interval_seconds: float = INTERVAL_SECONDS,
    tolerance_seconds: float = TOLERANCE_SECONDS,
) -> UptimeReport:
    """Filter, parse, and derive uptime from raw `(started_at, trigger_source)` rows.

    Pure over its input: does no I/O, so it is testable without SQLite.

    Filtering:
        - Rows whose `trigger_source != 'scheduled'` are excluded and counted
          in `rows_excluded_manual`. `_record` (scheduler.py:198) runs
          regardless of `job.run()`'s outcome, so an `outcome='failed'` row
          still proves the process was alive and is deliberately NOT filtered
          on outcome.
        - Rows whose `started_at` does not parse are excluded and counted in
          `rows_excluded_unparseable`, never silently dropped.

    Args:
        rows: Every row for `job_name`, any trigger_source, as returned by
            `read_curation_job_rows`.
        job_name: The job these rows belong to (carried through for the
            formatter; not re-checked here).
        interval_seconds: Passed through to `derive_uptime`.
        tolerance_seconds: Passed through to `derive_uptime`.

    Returns:
        `UptimeReport` combining the exclusion counts with the derivation.
    """
    scheduled_raw = [started_at for started_at, trigger in rows if trigger == TRIGGER_SCHEDULED]
    rows_excluded_manual = len(rows) - len(scheduled_raw)

    timestamps: list[datetime] = []
    rows_excluded_unparseable = 0
    for raw in scheduled_raw:
        parsed = parse_started_at(raw)
        if parsed is None:
            rows_excluded_unparseable += 1
        else:
            timestamps.append(parsed)

    derivation = derive_uptime(
        timestamps, interval_seconds=interval_seconds, tolerance_seconds=tolerance_seconds
    )
    event_count = None if derivation.reason is not None else len(derivation.events)

    return UptimeReport(
        job_name=job_name,
        interval_seconds=interval_seconds,
        tolerance_seconds=tolerance_seconds,
        rows_used=len(timestamps),
        rows_excluded_manual=rows_excluded_manual,
        rows_excluded_unparseable=rows_excluded_unparseable,
        window_start=derivation.window_start,
        window_end=derivation.window_end,
        event_count=event_count,
        events=derivation.events,
        longest_run=derivation.longest_run,
        reason=derivation.reason,
    )


def _readonly_connect(db_path: str) -> sqlite3.Connection:
    """Open `db_path` strictly read-only via a percent-encoded `file:...?mode=ro` URI.

    Mirrors `scripts/mist_admin.py:929` (inside `_assert_replay_source_exists`),
    not `EventStore._get_connection` (`backend/event_store/store.py:50-67`),
    which is a plain read-write `sqlite3.connect` that issues
    `PRAGMA journal_mode=WAL` and creates the file if absent -- both
    unacceptable for a report that must never write to the store it reads.

    The path is percent-encoded before being embedded in the URI, not
    f-string interpolated: an un-encoded `#` or `?` in the path is URI
    syntax, not a filename character, and `scripts/mist_admin.py:723-737`
    documents the measured failure mode (a `#` in a directory name silently
    truncates the path at the URI fragment boundary).
    """
    uri = "file:" + quote(str(Path(db_path).resolve())) + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def read_curation_job_rows(db_path: str, job_name: str) -> list[tuple[str, str]]:
    """Read every `(started_at, trigger_source)` row for `job_name`, oldest first.

    Returns every trigger_source, not only 'scheduled': `build_uptime_report`
    needs the manual-trigger rows too, to report how many it excluded rather
    than having them vanish inside the query.

    Ordered by rowid, matching `EventStore.get_curation_job_runs`
    (`backend/event_store/store.py:706-708`, same reasoning): two rows can
    share a `started_at` to the microsecond, and rowid (insertion order) is
    the only total order guaranteed available. `derive_uptime` re-sorts by
    parsed timestamp regardless, so this ordering is a readability aid here,
    not a correctness dependency.

    Args:
        db_path: Filesystem path to the event store SQLite file. Caller is
            responsible for having already confirmed it exists and carries
            the expected schema (see `_assert_replay_source_exists` in
            `scripts/mist_admin.py`).
        job_name: Restrict to this job's rows.

    Returns:
        List of `(started_at, trigger_source)` tuples, oldest-inserted first.
    """
    conn = _readonly_connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT started_at, trigger_source FROM curation_job_runs "
            "WHERE job_name = ? ORDER BY rowid ASC",
            (job_name,),
        )
        return [(row[0], row[1]) for row in cursor.fetchall()]
    finally:
        conn.close()


def _fmt_hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f}h"


def format_uptime_report(report: UptimeReport, *, now: datetime) -> str:
    """Render a `UptimeReport` as the `[uptime]` CLI report text.

    `now` is injected rather than read from the clock so this function stays
    deterministic and testable (`tests/CLAUDE.md`'s "no time-dependent
    assertions" rule) -- the caller (`cmd_uptime`) passes `datetime.now(UTC)`.

    Args:
        report: Output of `build_uptime_report`.
        now: Wall-clock time to report the unmeasured trailing span against.

    Returns:
        Multi-line report text, no trailing newline.
    """
    # Phrased from the actual count rather than a baked-in literal: an earlier
    # draft hardcoded the number this derivation happened to produce on the
    # committed fixture, which would have printed that same number against any
    # other store. A caveat that misstates the count it is cautioning about is
    # the defect the caveat exists to prevent.
    if report.event_count is None:
        dont_read = "  Do not read the event count below as a failure count."
    else:
        dont_read = (
            f'  Do not read "{report.event_count} events" as "{report.event_count} failures".'
        )

    lines: list[str] = [
        "[uptime] HISTORICAL -- derived from the curation job ledger, not measured directly.",
        "",
        "WHAT THIS CANNOT TELL YOU",
        "  This report cannot distinguish a deliberate shutdown -- `docker compose",
        "  down`, a rebuild, a host reboot -- from a crash or an outage. Every event",
        "  below is one of those and this report does not know which. Where the",
        "  window covers active development, most of them are probably deliberate.",
        dont_read,
        "  A row that failed to write is also indistinguishable from downtime: the",
        "  scheduler swallows a recording error (scheduler.py:241-242), so a dropped",
        "  row appears here as an outage that never happened.",
        "",
        "HOW IT IS DERIVED",
        f"  Source:     curation_job_runs, job_name={report.job_name!r}, "
        f"trigger_source='scheduled', ascending by started_at.",
        "  Basis:      every enabled job fires on the scheduler's first pass "
        "(scheduler.py:305-311), so a row marks a moment the backend process was alive.",
        f"  Resolution: nominal interval {report.interval_seconds:.0f}s "
        f"({report.interval_seconds / 3600:.0f}h), scheduler poll {POLL_SECONDS:.0f}s -- "
        f"no gap resolves finer than {POLL_SECONDS:.0f}s.",
        f"  Tolerance:  {report.tolerance_seconds:.0f}s, used for every gap in this run.",
        "  Clock:      host wall clock, one started_at timestamp per row.",
        f"  Excluded:   {report.rows_excluded_manual} manual-trigger row(s), "
        f"{report.rows_excluded_unparseable} row(s) with an unparseable started_at.",
        "",
        "WINDOW",
    ]

    if report.window_start is None or report.window_end is None:
        lines.append(
            f"  (no scheduled {report.job_name!r} rows -- nothing measured, not zero downtime)"
        )
    else:
        lines.append(f"  First row: {report.window_start.isoformat()}")
        lines.append(f"  Last row:  {report.window_end.isoformat()}")
        if report.window_end > report.window_start:
            span = (report.window_end - report.window_start).total_seconds()
            lines.append(f"  Span:      {_fmt_hours(span)}")
        trailing = (now - report.window_end).total_seconds()
        lines.append(
            f"  Unmeasured before the first row and the {_fmt_hours(max(trailing, 0.0))} "
            f"since the last row (now: {now.isoformat()}); that is NOT an outage and NOT uptime."
        )

    lines.append("")
    lines.append("LONGEST CONTINUOUS RUN")
    if report.reason is not None:
        lines.append(f"  (undefined -- {report.reason})")
    elif report.longest_run is None:
        lines.append("  (no continuous run: every gap in the window was an outage or a restart)")
    else:
        lr = report.longest_run
        lines.append(
            f"  At least {lr.duration_seconds / 3600:.4f}h "
            f"({lr.start_at.isoformat()} -> {lr.end_at.isoformat()})"
        )
        lines.append(
            "  Lower bound: the run began within one poll of the first timestamp above "
            "and ended somewhere in the unmeasured interval after the last one -- the true "
            "run could be longer in either direction, never shorter."
        )

    lines.append("")
    if report.reason is not None:
        lines.append(f"EVENTS (undefined -- {report.reason})")
    else:
        lines.append(f"EVENTS ({report.event_count})")
        if not report.events:
            lines.append("  (none -- every gap in the window was continuous)")
        else:
            lines.append(f"  {'resumed at':<32} {'gap':>9} {'downtime':>24}  kind")
            for ev in report.events:
                if ev.downtime_lower_seconds is None:
                    downtime = f"no lower bound, <={_fmt_hours(ev.downtime_upper_seconds)}"
                else:
                    downtime = (
                        f">={_fmt_hours(ev.downtime_lower_seconds)}, "
                        f"<={_fmt_hours(ev.downtime_upper_seconds)}"
                    )
                lines.append(
                    f"  {ev.resumed_at.isoformat():<32} {_fmt_hours(ev.gap_seconds):>9} "
                    f"{downtime:>24}  {ev.kind}"
                )

    return "\n".join(lines)
