"""Postconditions a hydration run must satisfy before its output is usable.

These are checked AFTER the run, against observable state, and they exist
because every preventive gate is a claim about what code does rather than a
measurement of what happened. `CurationScheduler.start()` refuses to start
under hydration isolation (B1), but `run_all_once` bypasses `start()`
entirely, a future job could be wired somewhere else, and a stale container
could be running an image built before the gate existed. The ledger does not
care which of those it was.

Direction matters: a failed postcondition means the hydrated graph is
contaminated and must be discarded, not that the assertion is too strict. The
temptation at that point is to widen what counts as acceptable, which is the
same pressure the closure design names around the exclusion set -- and the
same wrong answer.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class HydrationPostconditionError(RuntimeError):
    """Raised when a completed hydration run produced unusable state."""


class _JobRunReader(Protocol):
    """The one method this module needs from an event store."""

    def get_curation_job_runs(
        self, job_name: str | None = ..., limit: int = ...
    ) -> list[dict[str, Any]]: ...


def snapshot_curation_run_ids(event_store: _JobRunReader, *, limit: int = 1000) -> frozenset[str]:
    """Capture which curation runs already exist, BEFORE a hydration run starts.

    `dev-state/` is a named volume that survives `docker compose down`, so the
    ledger is NOT empty just because this run has not written to it. If the dev
    backend was ever brought up once without MIST_HYDRATION_ISOLATION, or
    someone invoked `run_all_once`, those rows persist -- and a postcondition
    that simply asserts "the table is empty" reports CONTAMINATED on every
    subsequent run, forever, with no way to tell a real contamination from a
    stale one.

    `run_id` is the windowing key rather than `started_at`: it is exact, and it
    does not depend on clocks agreeing across a container restart.
    """
    return frozenset(
        row["run_id"] for row in event_store.get_curation_job_runs(limit=limit) if row.get("run_id")
    )


def assert_no_curation_job_runs(
    event_store: _JobRunReader, *, baseline: frozenset[str], limit: int = 1000
) -> None:
    """Refuse if any curation job ran during the hydration window.

    Curation writes into the compared `:__Entity__` surface and none of it is a
    function of the log: `SkillDerivationJob` writes a node and a `KNOWS` edge
    (`skill_derivation.py:160,173-174`), and `orphan_detector.py:86` /
    `confidence_decay.py:39` write `status`, which `canonical_serialize` does
    not exclude. A rebuild cannot reproduce any of it, so the gate would go RED
    for a reason that says nothing about whether the facts were reproduced.

    Args:
        event_store: Anything exposing `get_curation_job_runs`. The dev stack's
            store, read after the run completes.
        baseline: The `run_id` set captured by `snapshot_curation_run_ids`
            BEFORE the run. Required, not defaulted: an empty default would
            silently reproduce the "whole ledger must be empty" bug this
            parameter exists to fix, and a caller that has not captured a
            baseline cannot distinguish its own contamination from a stale
            row on a persistent volume.
        limit: Rows to read. The default is far above the handful a
            contaminated run would produce; this is a tripwire, not a census.

    Raises:
        HydrationPostconditionError: naming every job that ran, and how many
            times. The blast radius is what determines whether the graph can be
            salvaged or has to be rebuilt, so listing it beats a bare boolean.
    """
    runs = [
        row
        for row in event_store.get_curation_job_runs(limit=limit)
        if row.get("run_id") not in baseline
    ]
    if not runs:
        logger.info(
            "Hydration postcondition OK: no curation run appeared during the window "
            "(%d pre-existing run(s) ignored).",
            len(baseline),
        )
        return

    counts = Counter(row.get("job_name", "<unnamed>") for row in runs)
    detail = ", ".join(f"{name} x{count}" for name, count in sorted(counts.items()))
    raise HydrationPostconditionError(
        f"Curation ran DURING this hydration run: {detail}. These jobs write nodes, edges "
        "and `status` inside the compared :__Entity__ surface, and no rebuild can "
        "reproduce them -- the hydrated graph is contaminated and must be "
        "discarded rather than compared. Check MIST_HYDRATION_ISOLATION on the "
        "dev container and whether anything invoked `run_all_once`, which "
        "`CurationScheduler.start()`'s gate deliberately does not cover."
    )
