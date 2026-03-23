"""Tests for CurationScheduler."""

import pytest

from backend.knowledge.curation.scheduler import CurationScheduler, JobConfig


class FakeJob:
    """Test double for a curation job."""

    def __init__(self, *, should_fail: bool = False):
        self.call_count = 0
        self._should_fail = should_fail

    async def run(self):
        self.call_count += 1
        if self._should_fail:
            raise RuntimeError("job failed")
        return {"status": "ok"}


class TestRunOnce:
    @pytest.mark.asyncio
    async def test_executes_all_enabled_jobs(self):
        job_a = FakeJob()
        job_b = FakeJob()
        scheduler = CurationScheduler(
            jobs=[
                (JobConfig(name="job_a", interval_seconds=60), job_a),
                (JobConfig(name="job_b", interval_seconds=60), job_b),
            ]
        )

        results = await scheduler.run_once()

        assert len(results) == 2
        assert job_a.call_count == 1
        assert job_b.call_count == 1
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_disabled_job_skipped(self):
        job_a = FakeJob()
        job_b = FakeJob()
        scheduler = CurationScheduler(
            jobs=[
                (JobConfig(name="job_a", interval_seconds=60, enabled=True), job_a),
                (JobConfig(name="job_b", interval_seconds=60, enabled=False), job_b),
            ]
        )

        results = await scheduler.run_once()

        assert len(results) == 1
        assert job_a.call_count == 1
        assert job_b.call_count == 0

    @pytest.mark.asyncio
    async def test_job_failure_does_not_block_others(self):
        job_a = FakeJob(should_fail=True)
        job_b = FakeJob()
        scheduler = CurationScheduler(
            jobs=[
                (JobConfig(name="job_a", interval_seconds=60), job_a),
                (JobConfig(name="job_b", interval_seconds=60), job_b),
            ]
        )

        results = await scheduler.run_once()

        assert len(results) == 2
        assert not results[0].success
        assert results[0].error == "job failed"
        assert results[1].success
        assert job_b.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_scheduler(self):
        scheduler = CurationScheduler(jobs=[])
        results = await scheduler.run_once()
        assert results == []

    @pytest.mark.asyncio
    async def test_result_contains_job_output(self):
        job = FakeJob()
        scheduler = CurationScheduler(jobs=[(JobConfig(name="test_job", interval_seconds=60), job)])

        results = await scheduler.run_once()

        assert results[0].result == {"status": "ok"}
        assert results[0].duration_ms > 0
