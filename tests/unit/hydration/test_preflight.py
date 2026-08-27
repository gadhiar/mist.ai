"""Preconditions for a hydration run: refuse before spending 87 LLM turns.

Two of these four exist because the failure they catch is SILENT. A run with
no clock completes successfully and produces a graph whose timeline is the
wall clock -- and the gate then passes, because both sides read that same
flattened timeline back out of the event store. Nothing downstream notices.

The other two are about turning a slow, partial failure into a fast, legible
one: an uncovered corpus turn and a re-used session both fail eventually, at
the turn they occur, after everything before them has been written.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.chat.hydration_clock import HydrationClock
from scripts.hydration.preflight import (
    HydrationPreflightError,
    assert_clock_covers_corpus,
    assert_clock_present,
    assert_event_store_present,
    assert_hydration_isolation,
    assert_sessions_sequential,
    assert_sessions_unused,
    run_all,
)

_ISOLATION = "MIST_HYDRATION_ISOLATION"
_T = datetime(2025, 9, 2, 8, 0, tzinfo=UTC)

_ROWS = [
    {"session_id": "golden-2025-09", "turn_index": 0},
    {"session_id": "golden-2025-09", "turn_index": 1},
    {"session_id": "golden-2026-07", "turn_index": 0},
]


def _clock(keys=None):
    keys = keys if keys is not None else [(r["session_id"], r["turn_index"]) for r in _ROWS]
    return HydrationClock(timestamps={k: _T for k in keys}, source_path="fixture")


class _Session:
    def __init__(self, turn_count):
        self.turn_count = turn_count


class _FakeStore:
    def __init__(self, sessions=None):
        self._sessions = sessions or {}

    def get_session(self, session_id):
        return self._sessions.get(session_id)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_ISOLATION, raising=False)


class TestAssertHydrationIsolation:
    def test_refuses_when_unset(self):
        with pytest.raises(HydrationPreflightError, match="MIST_HYDRATION_ISOLATION"):
            assert_hydration_isolation()

    def test_refuses_when_explicitly_off(self, monkeypatch):
        monkeypatch.setenv(_ISOLATION, "0")
        with pytest.raises(HydrationPreflightError):
            assert_hydration_isolation()

    def test_passes_when_set(self, monkeypatch):
        monkeypatch.setenv(_ISOLATION, "1")
        assert_hydration_isolation()


class TestAssertClockPresent:
    def test_refuses_none(self):
        """The silent failure: no clock means wall-clock stamps and a green gate."""
        with pytest.raises(HydrationPreflightError, match="MIST_HYDRATION_CLOCK"):
            assert_clock_present(None)

    def test_passes_with_a_clock(self):
        assert_clock_present(_clock())


class TestAssertClockCoversCorpus:
    def test_full_coverage_passes(self):
        assert_clock_covers_corpus(_clock(), _ROWS)

    def test_a_single_uncovered_turn_refuses(self):
        partial = _clock(keys=[("golden-2025-09", 0), ("golden-2025-09", 1)])
        with pytest.raises(HydrationPreflightError, match=r"golden-2026-07#0"):
            assert_clock_covers_corpus(partial, _ROWS)

    def test_it_reports_the_count_not_just_the_first(self):
        """An operator fixing a corpus needs the scale, not one example."""
        with pytest.raises(HydrationPreflightError, match="3 of 3"):
            assert_clock_covers_corpus(_clock(keys=[]), _ROWS)

    def test_many_misses_are_truncated_with_a_remainder(self):
        rows = [{"session_id": "s", "turn_index": i} for i in range(9)]
        with pytest.raises(HydrationPreflightError, match=r"\+4 more"):
            assert_clock_covers_corpus(_clock(keys=[]), rows)

    def test_string_and_int_key_forms_both_match(self):
        """The corpus is JSON; a turn_index could arrive as a string.

        The clock keys on `(str, int)`, so coverage must normalize the same way
        the loader does or a perfectly good corpus reads as uncovered.
        """
        rows = [{"session_id": "golden-2025-09", "turn_index": "0"}]
        assert_clock_covers_corpus(_clock(), rows)


class TestAssertSessionsUnused:
    def test_empty_store_passes(self):
        assert_sessions_unused(_FakeStore(), _ROWS)

    def test_a_session_with_zero_turns_passes(self):
        """Started but never used is not the same as used."""
        assert_sessions_unused(_FakeStore({"golden-2025-09": _Session(0)}), _ROWS)

    def test_a_used_session_refuses_with_its_turn_count(self):
        """The count is the actionable part -- it says how far a prior run got."""
        store = _FakeStore({"golden-2025-09": _Session(3)})
        with pytest.raises(HydrationPreflightError, match=r"golden-2025-09 \(3 turns\)"):
            assert_sessions_unused(store, _ROWS)

    def test_each_session_is_checked_once_despite_repeated_rows(self):
        store = _FakeStore({"golden-2026-07": _Session(1)})
        with pytest.raises(HydrationPreflightError) as exc:
            assert_sessions_unused(store, _ROWS)
        assert str(exc.value).count("golden-2026-07") == 1


class TestRunAll:
    def test_the_happy_path(self, monkeypatch):
        monkeypatch.setenv(_ISOLATION, "1")
        run_all(clock=_clock(), rows=_ROWS, event_store=_FakeStore())

    def test_isolation_is_checked_before_anything_else(self):
        """Ordering: nothing that could write runs before the isolation check.

        Passing a None clock AND no isolation must report the isolation
        failure -- if coverage ran first it would raise a TypeError on the
        None clock and bury the real cause.
        """
        with pytest.raises(HydrationPreflightError, match="MIST_HYDRATION_ISOLATION"):
            run_all(clock=None, rows=_ROWS, event_store=None)

    def test_clock_presence_is_checked_before_coverage(self, monkeypatch):
        """A coverage check on None is a TypeError, not a diagnosis."""
        monkeypatch.setenv(_ISOLATION, "1")
        with pytest.raises(HydrationPreflightError, match="MIST_HYDRATION_CLOCK"):
            run_all(clock=None, rows=_ROWS, event_store=None)

    def test_a_missing_event_store_now_refuses(self, monkeypatch):
        """Reversed by review, and the old reasoning was wrong.

        This previously read "event store disabled is a legitimate config, not
        a preflight failure" and SKIPPED both store-dependent checks.
        `ConversationHandler` sets `event_store = None` on any init exception
        and logs rather than raising, so in that state a hydration run drives
        every corpus turn through full inference, writes nothing, skips
        `assert_sessions_unused`, skips the curation postcondition, and exits 0
        reporting "Complete" -- the silent-green shape this module exists to
        refuse.
        """
        monkeypatch.setenv(_ISOLATION, "1")
        with pytest.raises(HydrationPreflightError, match="no event store"):
            run_all(clock=_clock(), rows=_ROWS, event_store=None)


class TestAssertSessionsSequential:
    """Cloud-review finding: coverage was checked, order never was.

    The runtime keys the clock on the event store's sequential `turn_count`,
    not on the corpus row's `turn_index`, and `run_replay` drives rows in FILE
    order. Reshuffle a session and every lookup still finds a real key -- so
    the clock's fail-closed miss never fires and every turn gets a plausible
    authored timestamp attached to the wrong utterance. Fails GREEN.
    """

    def test_ordered_rows_pass(self):
        assert_sessions_sequential(_ROWS)

    def test_interleaved_sessions_are_fine(self):
        """Order is per-session, not global. Two sessions may interleave."""
        assert_sessions_sequential(
            [
                {"session_id": "a", "turn_index": 0},
                {"session_id": "b", "turn_index": 0},
                {"session_id": "a", "turn_index": 1},
                {"session_id": "b", "turn_index": 1},
            ]
        )

    def test_a_reshuffled_session_is_refused(self):
        """The exact fail-green case: all keys covered, all stamps wrong."""
        rows = [
            {"session_id": "s", "turn_index": 2},
            {"session_id": "s", "turn_index": 0},
            {"session_id": "s", "turn_index": 1},
        ]
        with pytest.raises(HydrationPreflightError, match="expected turn_index=0"):
            assert_sessions_sequential(rows)

    def test_a_gap_is_refused(self):
        rows = [{"session_id": "s", "turn_index": 0}, {"session_id": "s", "turn_index": 2}]
        with pytest.raises(HydrationPreflightError, match="found 2"):
            assert_sessions_sequential(rows)

    def test_a_session_not_starting_at_zero_is_refused(self):
        """turn_count starts at 0 for a fresh session; the corpus must too."""
        with pytest.raises(HydrationPreflightError):
            assert_sessions_sequential([{"session_id": "s", "turn_index": 1}])

    def test_the_real_golden_log_is_in_order(self):
        """Non-vacuity against the corpus this actually guards.

        Also establishes that the fix does not reject today's data -- the
        reason the cloud review rated this a nit is that this assertion holds
        right now, which is a property of the data, not of the code.
        """
        import json
        from pathlib import Path

        rows = [
            json.loads(line)
            for line in Path("data/golden-log/golden-log.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert len(rows) >= 80
        assert_sessions_sequential(rows)


class TestAssertEventStorePresent:
    def test_a_store_passes(self):
        assert_event_store_present(_FakeStore())

    def test_none_refuses(self):
        with pytest.raises(HydrationPreflightError, match="no event store"):
            assert_event_store_present(None)
