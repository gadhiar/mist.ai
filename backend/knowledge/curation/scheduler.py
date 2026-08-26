"""Curation job scheduler.

Manages periodic execution of graph maintenance jobs via asyncio.
Each job is independent -- one failure does not block others.

Every execution -- scheduled or manual -- is recorded to the
`curation_job_runs` ledger via a `CurationRunRecorder`, and a health score is
additionally appended to the `graph_health_events` time series. See
`run_record.py` for why the ledger separates "the job ran" from "what it
examined" from "what it produced".
"""

import asyncio
import contextlib
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from backend.knowledge.eval_isolation import (
    EvalIsolationError,
    is_hydration_isolation_active,
    parse_isolation_flag,
)

from .run_record import (
    OUTCOME_COMPLETED,
    OUTCOME_FAILED,
    TRIGGER_MANUAL,
    TRIGGER_SCHEDULED,
    CurationRunRecorder,
    describe_result,
    health_event_fields,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JobConfig:
    """Configuration for a scheduled curation job."""

    name: str
    interval_seconds: int
    enabled: bool = True


@dataclass(slots=True)
class JobResult:
    """Result of a single job execution."""

    name: str
    success: bool
    duration_ms: float
    error: str | None = None
    result: Any = None


def curation_scheduler_enabled() -> bool:
    """Whether the background curation loop may start (B1). Defaults to TRUE.

    The scheduler makes every enabled job due on its FIRST pass, so a 24-hour
    interval fires immediately -- fine for live, fatal for a `live == rebuilt`
    gate. `SkillDerivationJob` writes a node AND an edge inside the compared
    `:__Entity__` surface (`skill_derivation.py:160,173-174`), and
    `orphan_detector.py:86` / `confidence_decay.py:39` write `status`, which
    `canonical_serialize` does not exclude. None of it is a function of the log,
    so no rebuild can reproduce any of it.

    Defaults to True so adding the knob changes nothing about live. Reads
    MIST_CURATION_SCHEDULER_ENABLED.

    Fails toward OFF on an unrecognized value, which is the OPPOSITE of
    `is_hydration_isolation_active`'s raise -- deliberately. That one is read by
    a CLI which can print a refusal and exit; this one is read during server
    startup, where raising would take the backend down over a typo. A scheduler
    that did not run is a recoverable annoyance; a scheduler that ran during a
    gate run is a corrupted comparison.
    """
    if os.getenv("MIST_CURATION_SCHEDULER_ENABLED") is None:
        return True
    try:
        return parse_isolation_flag("MIST_CURATION_SCHEDULER_ENABLED")
    except EvalIsolationError as exc:
        logger.warning(
            "MIST_CURATION_SCHEDULER_ENABLED unparsable, disabling the scheduler: %s", exc
        )
        return False


class CurationScheduler:
    """Asyncio-based scheduler for curation maintenance jobs.

    Jobs are registered as (config, job_instance) tuples. Each job must
    have an async `run()` method. The scheduler runs all enabled jobs
    on their configured intervals.
    """

    def __init__(
        self,
        jobs: list[tuple[JobConfig, Any]],
        run_recorder: CurationRunRecorder | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            jobs: List of (config, job_instance) tuples. Each job_instance
                must have an async `run()` method.
            run_recorder: Durable sink for job executions, satisfied in
                production by the live `EventStore`. When None every result is
                logged and then lost, which is precisely the defect this
                parameter exists to close -- so construction logs a warning
                rather than defaulting silently. A production None is
                legitimate only when `config.event_store.enabled` is False.
        """
        self._jobs = jobs
        self._run_recorder = run_recorder
        self._task: asyncio.Task | None = None
        self._running = False

        if run_recorder is None:
            logger.warning(
                "Curation scheduler built with no run recorder -- job results will be "
                "logged and discarded, and a zero-result run will stay indistinguishable "
                "from a broken one"
            )

    async def run_once(self) -> list[JobResult]:
        """Run all enabled jobs once. Manual/ops trigger.

        Shares its per-job execution and recording path with `_loop` via
        `_execute_and_record`, so a manual trigger and a scheduled one cannot
        drift in what they run or what they persist. Before 2026-08-03 this
        method had five tests and no production caller while `_loop` carried a
        near-identical inline copy that persisted nothing -- an instance of the
        exact defect class this codebase is remediating.

        Returns:
            List of JobResult for each enabled job.
        """
        results: list[JobResult] = []

        for config, job in self._jobs:
            if not config.enabled:
                logger.debug("Skipping disabled job: %s", config.name)
                continue

            results.append(await self._execute_and_record(config, job, TRIGGER_MANUAL))

        return results

    async def _execute_and_record(
        self, config: JobConfig, job: Any, trigger_source: str
    ) -> JobResult:
        """Run one job, then durably record what it did. The only run path.

        Args:
            config: The job's registration.
            job: The job instance to await.
            trigger_source: TRIGGER_SCHEDULED or TRIGGER_MANUAL.

        Returns:
            JobResult carrying the job's return value, or its error text.
        """
        started_at = datetime.now(UTC).isoformat()
        start = time.perf_counter()

        try:
            result = await job.run()
            elapsed = (time.perf_counter() - start) * 1000
            job_result = JobResult(
                name=config.name,
                success=True,
                duration_ms=elapsed,
                result=result,
            )
            logger.info("Job %s completed in %.1fms", config.name, elapsed)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            job_result = JobResult(
                name=config.name,
                success=False,
                duration_ms=elapsed,
                error=str(e),
            )
            logger.error("Job %s failed in %.1fms: %s", config.name, elapsed, e)

        self._record(job_result, trigger_source, started_at)
        return job_result

    def _record(self, job_result: JobResult, trigger_source: str, started_at: str) -> None:
        """Append one execution to the run ledger, plus the health series.

        Recording is best-effort by design: a curation job that did real work
        must not be rolled back, nor the loop killed, because observability
        storage failed. The failure is logged at error level with the job name
        so a missing row is itself diagnosable.

        Args:
            job_result: What `_execute_and_record` produced.
            trigger_source: TRIGGER_SCHEDULED or TRIGGER_MANUAL.
            started_at: ISO-8601 timestamp taken before the job was awaited.
        """
        if self._run_recorder is None:
            return

        facts = describe_result(job_result.result)

        try:
            self._run_recorder.append_curation_job_run(
                run_id=str(uuid.uuid4()),
                job_name=job_result.name,
                trigger_source=trigger_source,
                started_at=started_at,
                duration_ms=job_result.duration_ms,
                outcome=OUTCOME_COMPLETED if job_result.success else OUTCOME_FAILED,
                result_type=facts.result_type,
                examined=facts.examined,
                produced=facts.produced,
                metrics=facts.metrics,
                error=job_result.error,
            )

            health = health_event_fields(job_result.result)
            if health is not None:
                self._run_recorder.append_graph_health_event(
                    event_id=str(uuid.uuid4()),
                    timestamp=started_at,
                    **health,
                )
        except sqlite3.Error as e:
            logger.error("Failed to record curation run for job %s: %s", job_result.name, e)

    async def start(self) -> None:
        """Start the background scheduler loop.

        Gated here rather than at the server call site so every caller inherits
        the check (B1). Note `run_all_once` is deliberately NOT gated -- it is
        an explicit ops action, and the `curation_job_runs` postcondition
        (`scripts/hydration/postconditions.py`) is what covers it.
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        # Structural no beats explicit yes: nothing legitimately runs curation
        # against a hydration target, and forgetting the knob during a
        # hydration run is the precise failure B1 exists to prevent. Same
        # reasoning that gives `assert_neo4j_dev_isolated` no off switch.
        if is_hydration_isolation_active():
            logger.info(
                "Curation scheduler NOT started: MIST_HYDRATION_ISOLATION is set. Its "
                "jobs write nodes, edges and `status` inside the compared :__Entity__ "
                "surface, none of it derivable from the log."
            )
            return
        if not curation_scheduler_enabled():
            logger.info("Curation scheduler NOT started: MIST_CURATION_SCHEDULER_ENABLED is off.")
            return

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Curation scheduler started with %d jobs",
            sum(1 for c, _ in self._jobs if c.enabled),
        )

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("Curation scheduler stopped")

    async def _loop(self) -> None:
        """Main scheduler loop. Runs jobs at their configured intervals."""
        # Track last run time per job
        last_run: dict[str, float] = {}

        while self._running:
            now = time.time()

            for config, job in self._jobs:
                if not config.enabled:
                    continue

                # `last_run.get(name, 0.0)` makes every enabled job due on the
                # first pass, so all of them run at scheduler start rather
                # than after their interval. Long-standing behaviour, left
                # alone deliberately -- but it means the ledger's first rows
                # are written at boot.
                last = last_run.get(config.name, 0.0)
                if now - last >= config.interval_seconds:
                    # The pre-existing broad handler, kept. It no longer wraps
                    # the job itself -- `_execute_and_record` catches that and
                    # turns it into a `failed` row -- so what is left for it to
                    # guard is the recording step raising something other than
                    # `sqlite3.Error`. A curation pass must not die because
                    # observability storage misbehaved.
                    try:
                        result = await self._execute_and_record(config, job, TRIGGER_SCHEDULED)
                        if not result.success:
                            logger.error("Job %s failed: %s", config.name, result.error)
                    except Exception as e:
                        logger.error("Job %s could not be recorded: %s", config.name, e)
                    last_run[config.name] = now

            # Sleep before next check (1 minute granularity)
            await asyncio.sleep(60)
