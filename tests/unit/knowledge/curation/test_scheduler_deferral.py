"""Tests for deferring the LLM curation job while gaming mode holds the GPU.

While the GPU is yielded mist-llm is stopped, so `self_reflection` would fail
every call and still be recorded as a run, losing its window. The scheduler
skips a due `requires_llm` job instead, without moving its `last_run`, so it
runs on the first pass after the deferral ends.

Each test drives `run_due_jobs`, one scheduled pass, under an injected clock;
nothing sleeps.
"""

from __future__ import annotations

import logging

import pytest

from backend.knowledge.curation.scheduler import CurationScheduler, JobConfig

DAY = 86400
_START_LINE = "Deferring LLM curation jobs"
_END_LINE = "LLM curation job deferral ended"


class RecordingJob:
    def __init__(self) -> None:
        self.runs = 0

    async def run(self) -> str:
        self.runs += 1
        return "ok"


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Deferral:
    def __init__(self, active: bool = False) -> None:
        self.active = active

    def __call__(self) -> bool:
        return self.active


def _scheduler(jobs, *, defer_llm_jobs, clock) -> CurationScheduler:
    return CurationScheduler(jobs=jobs, defer_llm_jobs=defer_llm_jobs, clock=clock)


@pytest.mark.asyncio
async def test_due_llm_job_waits_for_the_deferral_to_end_then_runs():
    llm_job = RecordingJob()
    deferral = Deferral(active=True)
    clock = Clock()
    scheduler = _scheduler(
        [(JobConfig(name="self_reflection", interval_seconds=DAY, requires_llm=True), llm_job)],
        defer_llm_jobs=deferral,
        clock=clock,
    )

    await scheduler.run_due_jobs()
    clock.now += 60
    await scheduler.run_due_jobs()
    runs_while_deferred = llm_job.runs
    deferral.active = False
    clock.now += 60
    await scheduler.run_due_jobs()

    assert runs_while_deferred == 0
    assert llm_job.runs == 1


@pytest.mark.asyncio
async def test_deferral_does_not_move_last_run():
    """Ran at t0; due at t0 + 1 day but deferred; the deferral ends an hour later.

    Had a deferred pass recorded a run at t0 + 1 day, the job would not be due
    again until t0 + 2 days, and the pass an hour later would skip it.
    """
    llm_job = RecordingJob()
    deferral = Deferral(active=False)
    clock = Clock()
    scheduler = _scheduler(
        [(JobConfig(name="self_reflection", interval_seconds=DAY, requires_llm=True), llm_job)],
        defer_llm_jobs=deferral,
        clock=clock,
    )
    await scheduler.run_due_jobs()
    assert llm_job.runs == 1

    deferral.active = True
    clock.now += DAY
    await scheduler.run_due_jobs()
    deferral.active = False
    clock.now += 3600
    await scheduler.run_due_jobs()

    assert llm_job.runs == 2


@pytest.mark.asyncio
async def test_job_without_requires_llm_runs_while_deferred():
    plain_job = RecordingJob()
    llm_job = RecordingJob()
    scheduler = _scheduler(
        [
            (JobConfig(name="confidence_decay", interval_seconds=DAY), plain_job),
            (JobConfig(name="self_reflection", interval_seconds=DAY, requires_llm=True), llm_job),
        ],
        defer_llm_jobs=Deferral(active=True),
        clock=Clock(),
    )

    await scheduler.run_due_jobs()

    assert plain_job.runs == 1
    assert llm_job.runs == 0


@pytest.mark.asyncio
async def test_without_a_deferral_predicate_every_job_runs():
    plain_job = RecordingJob()
    llm_job = RecordingJob()
    clock = Clock()
    scheduler = _scheduler(
        [
            (JobConfig(name="confidence_decay", interval_seconds=DAY), plain_job),
            (JobConfig(name="self_reflection", interval_seconds=DAY, requires_llm=True), llm_job),
        ],
        defer_llm_jobs=None,
        clock=clock,
    )

    await scheduler.run_due_jobs()
    clock.now += 60
    await scheduler.run_due_jobs()
    clock.now += DAY
    await scheduler.run_due_jobs()

    assert (plain_job.runs, llm_job.runs) == (2, 2)


@pytest.mark.asyncio
async def test_one_line_marks_the_start_of_a_deferral_and_one_its_end(caplog):
    deferral = Deferral(active=True)
    clock = Clock()
    scheduler = _scheduler(
        [
            (
                JobConfig(name="self_reflection", interval_seconds=DAY, requires_llm=True),
                RecordingJob(),
            )
        ],
        defer_llm_jobs=deferral,
        clock=clock,
    )

    with caplog.at_level(logging.INFO, logger="backend.knowledge.curation.scheduler"):
        for _ in range(3):
            await scheduler.run_due_jobs()
            clock.now += 60
        deferral.active = False
        for _ in range(2):
            await scheduler.run_due_jobs()
            clock.now += 60

    messages = [record.getMessage() for record in caplog.records]
    assert sum(_START_LINE in m for m in messages) == 1
    assert sum(_END_LINE in m for m in messages) == 1
