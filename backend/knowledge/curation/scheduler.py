"""Curation job scheduler.

Manages periodic execution of graph maintenance jobs via asyncio.
Each job is independent -- one failure does not block others.
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

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


class CurationScheduler:
    """Asyncio-based scheduler for curation maintenance jobs.

    Jobs are registered as (config, job_instance) tuples. Each job must
    have an async `run()` method. The scheduler runs all enabled jobs
    on their configured intervals.
    """

    def __init__(self, jobs: list[tuple[JobConfig, Any]]) -> None:
        """Initialize the scheduler.

        Args:
            jobs: List of (config, job_instance) tuples. Each job_instance
                must have an async `run()` method.
        """
        self._jobs = jobs
        self._task: asyncio.Task | None = None
        self._running = False

    async def run_once(self) -> list[JobResult]:
        """Run all enabled jobs once. Useful for testing and manual trigger.

        Returns:
            List of JobResult for each enabled job.
        """
        results: list[JobResult] = []

        for config, job in self._jobs:
            if not config.enabled:
                logger.debug("Skipping disabled job: %s", config.name)
                continue

            start = time.perf_counter()
            try:
                result = await job.run()
                elapsed = (time.perf_counter() - start) * 1000
                results.append(
                    JobResult(
                        name=config.name,
                        success=True,
                        duration_ms=elapsed,
                        result=result,
                    )
                )
                logger.info("Job %s completed in %.1fms", config.name, elapsed)
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                results.append(
                    JobResult(
                        name=config.name,
                        success=False,
                        duration_ms=elapsed,
                        error=str(e),
                    )
                )
                logger.error("Job %s failed in %.1fms: %s", config.name, elapsed, e)

        return results

    async def start(self) -> None:
        """Start the background scheduler loop."""
        if self._running:
            logger.warning("Scheduler already running")
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

                last = last_run.get(config.name, 0.0)
                if now - last >= config.interval_seconds:
                    start = time.perf_counter()
                    try:
                        await job.run()
                        elapsed = (time.perf_counter() - start) * 1000
                        logger.info("Job %s completed in %.1fms", config.name, elapsed)
                    except Exception as e:
                        elapsed = (time.perf_counter() - start) * 1000
                        logger.error("Job %s failed in %.1fms: %s", config.name, elapsed, e)
                    last_run[config.name] = now

            # Sleep before next check (1 minute granularity)
            await asyncio.sleep(60)
