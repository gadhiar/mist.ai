"""B1: the curation scheduler contaminates the compared surface, with no off switch.

`scheduler.py`'s loop makes every enabled job due on the FIRST pass -- the
source says so itself: "`last_run.get(name, 0.0)` makes every enabled job due
on the first pass, so all of them run at scheduler start rather than after
their interval. Long-standing behaviour, left alone deliberately." A 24-hour
interval therefore fires immediately, which is fine for live and fatal for a
`live == rebuilt` gate.

What lands inside the compared `:__Entity__` surface, none of it a function of
the log and none of it reproducible by any rebuild:

- `SkillDerivationJob` writes `MERGE (e:__Entity__ {id: $skill_id})` AND
  `MERGE (u:__Entity__ {id:'user'}) MERGE (u)-[:KNOWS]->(e)`
  (`skill_derivation.py:160,173-174`) -- a node and an edge.
- `orphan_detector.py:86` and `confidence_decay.py:39` both write `status`,
  which `canonical_serialize` does not exclude.

These push the gate RED, not green, and that is exactly the hazard: the cheap
way to make a red diff go away is to widen the exclusion set, which is how a
gate quietly stops proving anything.

Two layers, because they fail differently:

1. `curation_scheduler_enabled()` -- an explicit knob, default TRUE so live is
   unchanged. Gated in `CurationScheduler.start()` rather than at the server
   call site, so every caller inherits it.
2. Auto-disable under `MIST_HYDRATION_ISOLATION`. Forgetting the knob during a
   hydration run is the precise failure this exists to prevent, and nothing
   legitimately runs curation against a hydration target -- the same reasoning
   that gives `assert_neo4j_dev_isolated` no off switch.

The postcondition assertion is the belt to that pair of braces: it catches any
path that wrote a run, including the manual `run_all_once` ops trigger, which
`start()`'s gate deliberately does not cover.
"""

from __future__ import annotations

import pytest

from backend.knowledge.curation.scheduler import (
    CurationScheduler,
    JobConfig,
    curation_scheduler_enabled,
)
from scripts.hydration.postconditions import (
    HydrationPostconditionError,
    assert_no_curation_job_runs,
)

_KNOB = "MIST_CURATION_SCHEDULER_ENABLED"
_ISOLATION = "MIST_HYDRATION_ISOLATION"


class _StubJob:
    async def run(self):  # pragma: no cover - never reached in these tests
        raise AssertionError("job must not run")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_KNOB, raising=False)
    monkeypatch.delenv(_ISOLATION, raising=False)


def _scheduler():
    return CurationScheduler(jobs=[(JobConfig(name="stub", interval_seconds=86400), _StubJob())])


class TestCurationSchedulerEnabled:
    def test_defaults_to_true(self):
        """Live behaviour must be unchanged by adding a knob."""
        assert curation_scheduler_enabled() is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off"])
    def test_explicit_falsy_disables(self, monkeypatch, raw):
        monkeypatch.setenv(_KNOB, raw)
        assert curation_scheduler_enabled() is False

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_empty_means_unset_not_off(self, monkeypatch, raw):
        """Reversed by review, and it was a live-stack hazard.

        `parse_isolation_flag` reads with a "" default and "" is in _FALSY, so
        treating empty as a value meant `- MIST_CURATION_SCHEDULER_ENABLED=`
        in a compose file (or a bare trailing line in .env) silently disabled
        curation on LIVE, with only an INFO line as evidence. Docker Compose
        writes an empty value there, not an absent one.
        """
        monkeypatch.setenv(_KNOB, raw)
        assert curation_scheduler_enabled() is True

    @pytest.mark.parametrize("raw", ["1", "true", "on"])
    def test_explicit_truthy_enables(self, monkeypatch, raw):
        monkeypatch.setenv(_KNOB, raw)
        assert curation_scheduler_enabled() is True

    def test_unrecognized_value_disables_rather_than_guessing(self, monkeypatch):
        """Opposite default from the isolation flags, and deliberately so.

        `is_hydration_isolation_active` RAISES on a typo because it is read by
        a CLI that can print the refusal. This one is read during server
        startup, where raising would take the backend down over a typo. It
        fails toward OFF: a scheduler that did not run is a recoverable
        annoyance, a scheduler that ran during a gate run is a corrupted
        comparison.
        """
        monkeypatch.setenv(_KNOB, "enabld")
        assert curation_scheduler_enabled() is False


class TestStartIsGated:
    @pytest.mark.asyncio
    async def test_it_starts_by_default(self):
        sched = _scheduler()
        await sched.start()
        try:
            assert sched._task is not None
        finally:
            await sched.stop()

    @pytest.mark.asyncio
    async def test_the_knob_prevents_the_loop_from_being_created(self, monkeypatch):
        """Not started, not merely idle -- no task, so no first pass."""
        monkeypatch.setenv(_KNOB, "0")
        sched = _scheduler()
        await sched.start()
        assert sched._task is None
        assert sched._running is False

    @pytest.mark.asyncio
    async def test_hydration_isolation_disables_it_without_the_knob(self, monkeypatch):
        """Forgetting the knob during hydration is the failure this prevents."""
        monkeypatch.setenv(_ISOLATION, "1")
        sched = _scheduler()
        await sched.start()
        assert sched._task is None

    @pytest.mark.asyncio
    async def test_the_knob_cannot_re_enable_it_under_hydration_isolation(self, monkeypatch):
        """An explicit yes must not beat the structural no.

        Otherwise the dev compose could be 'fixed' into contaminating its own
        gate run by someone who set the knob for an unrelated reason.
        """
        monkeypatch.setenv(_ISOLATION, "1")
        monkeypatch.setenv(_KNOB, "1")
        sched = _scheduler()
        await sched.start()
        assert sched._task is None

    @pytest.mark.asyncio
    async def test_stop_is_safe_when_start_was_gated(self, monkeypatch):
        """Shutdown calls stop() unconditionally; a gated start must not break it."""
        monkeypatch.setenv(_KNOB, "0")
        sched = _scheduler()
        await sched.start()
        await sched.stop()


class _FakeStore:
    def __init__(self, rows):
        self._rows = rows

    def get_curation_job_runs(self, job_name=None, limit=100):  # noqa: ARG002
        return self._rows


class TestAssertNoCurationJobRuns:
    def test_an_empty_ledger_passes(self):
        assert_no_curation_job_runs(_FakeStore([]), baseline=frozenset())

    def test_any_run_refuses(self):
        store = _FakeStore([{"job_name": "skill_derivation", "outcome": "ok"}])
        with pytest.raises(HydrationPostconditionError, match="skill_derivation"):
            assert_no_curation_job_runs(store, baseline=frozenset())

    def test_it_names_every_job_that_ran(self):
        """An operator needs to know the blast radius, not just that it happened."""
        store = _FakeStore(
            [
                {"job_name": "skill_derivation", "outcome": "ok"},
                {"job_name": "orphan_detection", "outcome": "ok"},
            ]
        )
        with pytest.raises(HydrationPostconditionError) as exc:
            assert_no_curation_job_runs(store, baseline=frozenset())
        assert "skill_derivation" in str(exc.value)
        assert "orphan_detection" in str(exc.value)

    def test_a_pre_existing_run_in_the_baseline_is_ignored(self):
        """The windowing, and the reason for it.

        `dev-state/` is a named volume that survives `docker compose down`, so
        the ledger is not empty just because THIS run wrote nothing. Without a
        baseline, one dev boot without MIST_HYDRATION_ISOLATION would report
        every later hydration CONTAMINATED forever, with no way to tell a real
        contamination from a stale row.
        """
        store = _FakeStore([{"run_id": "old-1", "job_name": "confidence_decay"}])
        assert_no_curation_job_runs(store, baseline=frozenset({"old-1"}))

    def test_a_new_run_is_still_caught_alongside_a_baseline(self):
        """Non-vacuity for the test above: windowing must not disable the gate.

        Ignoring pre-existing rows is only safe if a row that appeared DURING
        the window still refuses. Without this, a baseline that swallowed
        everything would look identical to one that works.
        """
        store = _FakeStore(
            [
                {"run_id": "old-1", "job_name": "confidence_decay"},
                {"run_id": "new-1", "job_name": "skill_derivation"},
            ]
        )
        with pytest.raises(HydrationPostconditionError, match="skill_derivation"):
            assert_no_curation_job_runs(store, baseline=frozenset({"old-1"}))

    def test_the_message_names_only_the_new_run(self):
        """An operator chasing a stale row is chasing the wrong thing."""
        store = _FakeStore(
            [
                {"run_id": "old-1", "job_name": "confidence_decay"},
                {"run_id": "new-1", "job_name": "skill_derivation"},
            ]
        )
        with pytest.raises(HydrationPostconditionError) as exc:
            assert_no_curation_job_runs(store, baseline=frozenset({"old-1"}))
        assert "confidence_decay" not in str(exc.value)

    def test_it_catches_the_manual_trigger_that_start_does_not_gate(self):
        """`run_all_once` bypasses `start()` entirely; the ledger still records it.

        This is why the postcondition exists as well as the gate: the gate
        covers the automatic first pass, the ledger covers every path.
        """
        store = _FakeStore([{"job_name": "confidence_decay", "outcome": "ok"}])
        with pytest.raises(HydrationPostconditionError):
            assert_no_curation_job_runs(store, baseline=frozenset())
