"""Retention of curation JobResults to durable storage (D3).

Until 2026-08-03 `CurationScheduler._loop` was a bare `await job.run()`: every
`JobResult` the scheduler produced in production was discarded. `run_once()`,
which builds them, had five tests and zero production callers.

Every assertion in this file reads the persisted row BACK OUT of the store
after the run. None of them assert that a recording method was called -- a
call assertion would have passed against a recorder that dropped its argument
on the floor, which is the same shape as the defect under repair.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from backend.event_store.store import EventStore
from backend.knowledge.curation.confidence_decay import DecayResult
from backend.knowledge.curation.health import HealthScore
from backend.knowledge.curation.scheduler import CurationScheduler, JobConfig
from backend.knowledge.curation.self_reflection import SelfReflectionJob
from backend.knowledge.extraction.internal_derivation import InternalKnowledgeDeriver
from backend.knowledge.extraction.signal_detector import SignalDetector
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM


@pytest.fixture
def store() -> EventStore:
    """An initialized in-memory event store. Never touches the real DB."""
    event_store = EventStore(db_path=":memory:")
    event_store.initialize()
    yield event_store
    event_store.close()


class RecordingJob:
    """A job that reports whether it ran and returns a caller-chosen result."""

    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self.call_count = 0
        self._result = result
        self._error = error

    async def run(self):
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return self._result


class BrokenRecorder:
    """A recorder whose writes always fail, as a full disk would."""

    def __init__(self) -> None:
        import sqlite3

        self._error = sqlite3.OperationalError("database is locked")

    def append_curation_job_run(self, **kwargs) -> str:
        raise self._error

    def append_graph_health_event(self, **kwargs) -> str:
        raise self._error


async def _wait_for_runs(store: EventStore, job_name: str, count: int, timeout: float = 5.0):
    """Poll the ledger until `count` runs of `job_name` are readable.

    Polls rather than sleeping a fixed interval so the test is bounded by the
    condition, not by a guessed duration. Raises rather than returning short,
    so a caller cannot assert against a partial result.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = store.get_curation_job_runs(job_name=job_name)
        if len(rows) >= count:
            return rows
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"{job_name!r} did not reach {count} recorded run(s) within {timeout}s; "
        f"ledger holds {len(store.get_curation_job_runs(job_name=job_name))}"
    )


def _deriver() -> InternalKnowledgeDeriver:
    """A real deriver over fakes, for jobs that need one but must not use it."""
    return InternalKnowledgeDeriver(
        llm=FakeLLM(), executor=FakeGraphExecutor(FakeNeo4jConnection())
    )


class TestLoopRetention:
    """The defect itself: the SCHEDULED path must not discard results."""

    @pytest.mark.asyncio
    async def test_scheduled_run_is_readable_from_the_ledger_afterwards(self, store):
        """`_loop` fires each enabled job immediately at start (`last_run`
        defaults to 0.0). After that first pass the run must be durable.
        """
        job = RecordingJob(
            result=DecayResult(
                entities_scanned=41,
                entities_decayed=3,
                entities_archived=1,
                duration_ms=12.5,
            )
        )
        scheduler = CurationScheduler(
            jobs=[(JobConfig(name="confidence_decay", interval_seconds=86400), job)],
            run_recorder=store,
        )

        await scheduler.start()
        try:
            rows = await _wait_for_runs(store, "confidence_decay", 1)
        finally:
            await scheduler.stop()

        assert job.call_count == 1
        row = rows[0]
        assert row["job_name"] == "confidence_decay"
        assert row["trigger_source"] == "scheduled"
        assert row["outcome"] == "completed"
        assert row["result_type"] == "DecayResult"
        assert row["examined"] == 41
        assert row["produced"] == 4
        assert row["metrics"]["duration_ms"] == 12.5

    @pytest.mark.asyncio
    async def test_scheduled_failure_is_readable_from_the_ledger_afterwards(self, store):
        """A job that raises inside the loop must leave a row saying so,
        rather than only a log line that rotates away.
        """
        scheduler = CurationScheduler(
            jobs=[
                (
                    JobConfig(name="orphan_detection", interval_seconds=604800),
                    RecordingJob(error=RuntimeError("neo4j unreachable")),
                )
            ],
            run_recorder=store,
        )

        await scheduler.start()
        try:
            rows = await _wait_for_runs(store, "orphan_detection", 1)
        finally:
            await scheduler.stop()

        row = rows[0]
        assert row["outcome"] == "failed"
        assert row["error"] == "neo4j unreachable"
        assert row["result_type"] is None

    @pytest.mark.asyncio
    async def test_disabled_job_leaves_no_row(self, store):
        """The ledger must not claim a run that never happened."""
        enabled = RecordingJob(result=None)
        disabled = RecordingJob(result=None)
        scheduler = CurationScheduler(
            jobs=[
                (JobConfig(name="health_scoring", interval_seconds=86400), enabled),
                (
                    JobConfig(name="community_detection", interval_seconds=604800, enabled=False),
                    disabled,
                ),
            ],
            run_recorder=store,
        )

        await scheduler.start()
        try:
            await _wait_for_runs(store, "health_scoring", 1)
        finally:
            await scheduler.stop()

        assert disabled.call_count == 0
        assert store.get_curation_job_runs(job_name="community_detection") == []


class TestZeroResultVersusBroken:
    """The point of the exercise: tell "found nothing" from "broken"."""

    @pytest.mark.asyncio
    async def test_idle_failed_and_saw_nothing_are_three_distinct_rows(self, store):
        """Three runs that a bare log line renders identically -- all three
        report no output -- must be separable in the ledger.
        """
        scheduler = CurationScheduler(
            jobs=[
                (
                    JobConfig(name="idle", interval_seconds=60),
                    RecordingJob(
                        result=DecayResult(
                            entities_scanned=120,
                            entities_decayed=0,
                            entities_archived=0,
                            duration_ms=8.0,
                        )
                    ),
                ),
                (
                    JobConfig(name="saw_nothing", interval_seconds=60),
                    RecordingJob(
                        result=DecayResult(
                            entities_scanned=0,
                            entities_decayed=0,
                            entities_archived=0,
                            duration_ms=0.4,
                        )
                    ),
                ),
                (
                    JobConfig(name="broken", interval_seconds=60),
                    RecordingJob(error=RuntimeError("Neo4jQueryError: syntax error")),
                ),
            ],
            run_recorder=store,
        )

        await scheduler.run_once()

        idle = store.get_curation_job_runs(job_name="idle")[0]
        saw_nothing = store.get_curation_job_runs(job_name="saw_nothing")[0]
        broken = store.get_curation_job_runs(job_name="broken")[0]

        # Ran, looked at 120, changed none -- a healthy idle pass.
        assert (idle["outcome"], idle["examined"], idle["produced"]) == ("completed", 120, 0)

        # Ran, looked at nothing. Same produced=0, different diagnosis.
        assert (saw_nothing["outcome"], saw_nothing["examined"], saw_nothing["produced"]) == (
            "completed",
            0,
            0,
        )

        # Never got as far as looking.
        assert broken["outcome"] == "failed"
        assert broken["examined"] is None
        assert broken["produced"] is None
        assert "syntax error" in broken["error"]

    @pytest.mark.asyncio
    async def test_inert_reflection_is_separable_from_a_genuinely_empty_log(self, store):
        """The concrete instance the project's root finding names.

        `SelfReflectionJob` with no event store returns `ReflectionResult(0, 0,
        0.0)` on its FIRST line -- zeros without looking. The same job over a
        real but empty log also returns zeros. Both are `examined=0,
        produced=0` in the ledger, so the discriminator must be the job's own
        self-reported `duration_ms`, which the row preserves verbatim: exactly
        0.0 when the job short-circuited, measured and non-zero when it
        actually queried the store.
        """
        inert = SelfReflectionJob(
            executor=FakeGraphExecutor(FakeNeo4jConnection()),
            internal_deriver=_deriver(),
            signal_detector=SignalDetector(),
            event_store=None,
        )
        empty_log = SelfReflectionJob(
            executor=FakeGraphExecutor(FakeNeo4jConnection()),
            internal_deriver=_deriver(),
            signal_detector=SignalDetector(),
            event_store=store,
        )
        scheduler = CurationScheduler(
            jobs=[
                (JobConfig(name="reflection_inert", interval_seconds=86400), inert),
                (JobConfig(name="reflection_empty_log", interval_seconds=86400), empty_log),
            ],
            run_recorder=store,
        )

        await scheduler.run_once()

        inert_row = store.get_curation_job_runs(job_name="reflection_inert")[0]
        empty_row = store.get_curation_job_runs(job_name="reflection_empty_log")[0]

        # Indistinguishable on the counters, as the root finding says.
        assert (inert_row["examined"], inert_row["produced"]) == (0, 0)
        assert (empty_row["examined"], empty_row["produced"]) == (0, 0)

        # Separable on the preserved self-reported duration.
        assert inert_row["metrics"]["duration_ms"] == 0.0
        assert empty_row["metrics"]["duration_ms"] > 0.0

    @pytest.mark.asyncio
    async def test_read_only_job_records_produced_as_null_not_zero(self, store):
        """`GraphHealthScorer` measures and mutates nothing. Recording its
        output as `produced=0` would make it fire a zero-output alarm forever;
        NULL says "this job has no output to give".
        """
        scheduler = CurationScheduler(
            jobs=[
                (
                    JobConfig(name="health_scoring", interval_seconds=86400),
                    RecordingJob(
                        result=HealthScore(
                            overall=72.5,
                            freshness=80.0,
                            confidence=70.0,
                            connectivity=60.0,
                            consistency=100.0,
                            coverage=50.0,
                            self_model=100.0,
                            entity_count=214,
                            relationship_count=530,
                        )
                    ),
                )
            ],
            run_recorder=store,
        )

        await scheduler.run_once()

        row = store.get_curation_job_runs(job_name="health_scoring")[0]
        assert row["examined"] == 214
        assert row["produced"] is None


class TestGraphHealthEvents:
    """`graph_health_events` had no INSERT anywhere in the codebase."""

    @pytest.mark.asyncio
    async def test_health_score_lands_in_the_health_time_series(self, store):
        """All seven sub-scores reached `logger.info` and nothing else. After
        a run they must be readable from the table declared to hold them.
        """
        scheduler = CurationScheduler(
            jobs=[
                (
                    JobConfig(name="health_scoring", interval_seconds=86400),
                    RecordingJob(
                        result=HealthScore(
                            overall=72.5,
                            freshness=80.0,
                            confidence=70.0,
                            connectivity=60.0,
                            consistency=100.0,
                            coverage=50.0,
                            self_model=100.0,
                            entity_count=214,
                            relationship_count=530,
                        )
                    ),
                )
            ],
            run_recorder=store,
        )

        await scheduler.run_once()

        events = store.get_graph_health_events()
        assert len(events) == 1
        event = events[0]
        assert event["health_score"] == 72.5
        assert event["entity_count"] == 214
        assert event["relationship_count"] == 530
        assert event["metrics"] == {
            "freshness": 80.0,
            "confidence": 70.0,
            "connectivity": 60.0,
            "consistency": 100.0,
            "coverage": 50.0,
            "self_model": 100.0,
        }

    @pytest.mark.asyncio
    async def test_failed_health_run_has_a_ledger_row_and_no_series_row(self, store):
        """`graph_health_events.health_score` is NOT NULL, so a health run
        that raised is unrepresentable there. It must still be recorded -- in
        the run ledger, which is why the two tables both exist.
        """
        scheduler = CurationScheduler(
            jobs=[
                (
                    JobConfig(name="health_scoring", interval_seconds=86400),
                    RecordingJob(error=RuntimeError("executor closed")),
                )
            ],
            run_recorder=store,
        )

        await scheduler.run_once()

        assert store.get_graph_health_events() == []
        row = store.get_curation_job_runs(job_name="health_scoring")[0]
        assert row["outcome"] == "failed"
        assert row["error"] == "executor closed"

    @pytest.mark.asyncio
    async def test_non_health_job_writes_no_series_row(self, store):
        """Dispatch is on the RESULT type, so a job that is not the scorer
        must not contribute to the health series whatever it is named.
        """
        scheduler = CurationScheduler(
            jobs=[
                (
                    JobConfig(name="health_scoring", interval_seconds=86400),
                    RecordingJob(
                        result=DecayResult(
                            entities_scanned=5,
                            entities_decayed=0,
                            entities_archived=0,
                            duration_ms=1.0,
                        )
                    ),
                )
            ],
            run_recorder=store,
        )

        await scheduler.run_once()

        assert store.get_graph_health_events() == []
        assert len(store.get_curation_job_runs(job_name="health_scoring")) == 1


class TestSharedExecutionPath:
    """`run_once` and `_loop` must not be two paths that can drift."""

    @pytest.mark.asyncio
    async def test_manual_and_scheduled_runs_differ_only_in_trigger_source(self, store):
        """Same job, both entry points, same recorded facts except the
        discriminator that says which entry point it was.
        """
        result = DecayResult(
            entities_scanned=7, entities_decayed=2, entities_archived=0, duration_ms=3.0
        )
        scheduler = CurationScheduler(
            jobs=[
                (
                    JobConfig(name="confidence_decay", interval_seconds=86400),
                    RecordingJob(result=result),
                )
            ],
            run_recorder=store,
        )

        await scheduler.run_once()
        await scheduler.start()
        try:
            rows = await _wait_for_runs(store, "confidence_decay", 2)
        finally:
            await scheduler.stop()

        by_trigger = {row["trigger_source"]: row for row in rows}
        assert set(by_trigger) == {"manual", "scheduled"}
        for row in by_trigger.values():
            assert (row["outcome"], row["result_type"], row["examined"], row["produced"]) == (
                "completed",
                "DecayResult",
                7,
                2,
            )


class TestRecorderIsBestEffort:
    @pytest.mark.asyncio
    async def test_a_failing_recorder_does_not_fail_the_job(self, store):
        """Observability storage must never take down curation. The job's own
        result is still returned to the caller.
        """
        job = RecordingJob(
            result=DecayResult(
                entities_scanned=3, entities_decayed=1, entities_archived=0, duration_ms=2.0
            )
        )
        scheduler = CurationScheduler(
            jobs=[(JobConfig(name="confidence_decay", interval_seconds=86400), job)],
            run_recorder=BrokenRecorder(),
        )

        results = await scheduler.run_once()

        assert job.call_count == 1
        assert results[0].success is True
        assert results[0].result.entities_decayed == 1

    @pytest.mark.asyncio
    async def test_scheduler_without_a_recorder_still_runs_jobs(self, store):
        """`run_recorder` is optional -- `config.event_store.enabled=False` is
        a legitimate production configuration.
        """
        job = RecordingJob(result=None)
        scheduler = CurationScheduler(
            jobs=[(JobConfig(name="confidence_decay", interval_seconds=86400), job)]
        )

        results = await scheduler.run_once()

        assert job.call_count == 1
        assert results[0].success is True
        assert store.get_curation_job_runs() == []
