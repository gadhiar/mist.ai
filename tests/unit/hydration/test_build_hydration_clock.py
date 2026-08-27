"""B2, the gating half: a keyed clock must be unreachable outside hydration.

`build_hydration_clock` is the only place a `HydrationClock` is constructed for
production wiring, and it refuses unless MIST_HYDRATION_ISOLATION is also set.

That second condition is not belt-and-braces, it is the point. The clock
rewrites `recorded_at`, which every bitemporal edge a turn produces inherits as
its fact-time authority. A clock active by accident on live would author a
false history that both sides of `live == rebuilt` would subsequently agree
on -- the failure mode that passes.

Requiring the isolation flag means the live backend cannot build one even if
MIST_HYDRATION_CLOCK leaks into its environment, because the live backend never
sets MIST_HYDRATION_ISOLATION.
"""

from __future__ import annotations

import json

import pytest

from backend.chat.hydration_clock import HydrationClockError
from backend.factories import build_hydration_clock

_CLOCK = "MIST_HYDRATION_CLOCK"
_ISOLATION = "MIST_HYDRATION_ISOLATION"

_ROW = {
    "session_id": "golden-2025-09",
    "turn_index": 0,
    "timestamp": "2025-09-02T08:00:00+00:00",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_CLOCK, raising=False)
    monkeypatch.delenv(_ISOLATION, raising=False)


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps(_ROW), encoding="utf-8")
    return path


class TestBuildHydrationClock:
    def test_unset_yields_none(self):
        """Production and eval both take this path."""
        assert build_hydration_clock() is None

    def test_blank_yields_none(self, monkeypatch):
        """An empty env var is 'unset', not 'a path called empty string'."""
        monkeypatch.setenv(_CLOCK, "   ")
        assert build_hydration_clock() is None

    def test_it_builds_when_both_flags_are_set(self, monkeypatch, corpus):
        monkeypatch.setenv(_ISOLATION, "1")
        monkeypatch.setenv(_CLOCK, str(corpus))
        clock = build_hydration_clock()
        assert clock is not None
        assert clock.recorded_at_for("golden-2025-09", 0).year == 2025

    def test_it_refuses_without_hydration_isolation(self, monkeypatch, corpus):
        """The gate. A leaked path variable on live must not build a clock."""
        monkeypatch.setenv(_CLOCK, str(corpus))
        with pytest.raises(HydrationClockError, match="MIST_HYDRATION_ISOLATION"):
            build_hydration_clock()

    def test_isolation_explicitly_off_also_refuses(self, monkeypatch, corpus):
        """Set-to-false is not the same as set, and must not satisfy the gate."""
        monkeypatch.setenv(_ISOLATION, "0")
        monkeypatch.setenv(_CLOCK, str(corpus))
        with pytest.raises(HydrationClockError, match="MIST_HYDRATION_ISOLATION"):
            build_hydration_clock()

    def test_isolation_alone_does_not_build_a_clock(self, monkeypatch):
        """A hydration container with no corpus configured gets None, not an error.

        The clock is opt-in per run; the dev stack sets the isolation flag
        permanently in its compose file.
        """
        monkeypatch.setenv(_ISOLATION, "1")
        assert build_hydration_clock() is None

    def test_a_bad_corpus_fails_the_process(self, monkeypatch, tmp_path):
        """Never degrade to wall-clock.

        A hydration run that silently fell back would produce a green gate over
        a timeline that never existed -- strictly worse than not running.
        """
        monkeypatch.setenv(_ISOLATION, "1")
        monkeypatch.setenv(_CLOCK, str(tmp_path / "missing.jsonl"))
        with pytest.raises(HydrationClockError, match="not found"):
            build_hydration_clock()
