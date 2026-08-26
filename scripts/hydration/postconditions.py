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


def assert_no_curation_job_runs(event_store: _JobRunReader, *, limit: int = 100) -> None:
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
        limit: Rows to read. The default is far above the handful a
            contaminated run would produce; this is a tripwire, not a census.

    Raises:
        HydrationPostconditionError: naming every job that ran, and how many
            times. The blast radius is what determines whether the graph can be
            salvaged or has to be rebuilt, so listing it beats a bare boolean.
    """
    runs = event_store.get_curation_job_runs(limit=limit)
    if not runs:
        logger.info("Hydration postcondition OK: curation_job_runs is empty.")
        return

    counts = Counter(row.get("job_name", "<unnamed>") for row in runs)
    detail = ", ".join(f"{name} x{count}" for name, count in sorted(counts.items()))
    raise HydrationPostconditionError(
        f"Curation ran during hydration: {detail}. These jobs write nodes, edges "
        "and `status` inside the compared :__Entity__ surface, and no rebuild can "
        "reproduce them -- the hydrated graph is contaminated and must be "
        "discarded rather than compared. Check MIST_HYDRATION_ISOLATION on the "
        "dev container and whether anything invoked `run_all_once`, which "
        "`CurationScheduler.start()`'s gate deliberately does not cover."
    )
