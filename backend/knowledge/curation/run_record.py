"""Translation of curation job results into durable rows (D3).

Pure mapping, no I/O. `CurationScheduler` calls `describe_result` on whatever
`job.run()` returned and hands the outcome to a `CurationRunRecorder`.

WHY THIS EXISTS AS A SEPARATE CONCERN
-------------------------------------
Until 2026-08-03 every `JobResult` the scheduler loop produced was discarded.
That was harmless while `SelfReflectionJob` and `SkillDerivationJob` returned
all-zero results regardless of the graph (wrong tracker instance, absent event
store) -- there was nothing to lose. Now that both are correctly wired the
counts are real, and `GraphHealthScorer` computes seven sub-scores that reached
`logger.info` and nothing else.

The project's root finding is that "job ran, found no data, returned zeros" and
"job ran broken, returned zeros before looking" are indistinguishable in the
logs. A durable record only fixes that if it separates the facts: THAT the job
ran, WHAT it examined, and WHAT it produced. `ResultFacts` is that separation,
and `_RESULT_COUNTERS` is the explicit declaration of which field of each
result type carries which fact.

WHY AN EXPLICIT REGISTRY RATHER THAN FIELD-NAME SNIFFING
--------------------------------------------------------
A heuristic ("any field ending in `_scanned` is examined") would silently
mis-classify the next result type someone adds, producing a row that reads as
a diagnosis while being wrong -- worse than no row. The registry is keyed by
result class name and `tests/unit/knowledge/curation/test_run_record.py`
asserts that every job the composition root registers has an entry, so a tenth
job added without a mapping fails the suite rather than logging NULLs forever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Protocol

# Trigger discriminators written to `curation_job_runs.trigger_source`.
TRIGGER_SCHEDULED = "scheduled"
TRIGGER_MANUAL = "manual"

# Outcome discriminators written to `curation_job_runs.outcome`.
OUTCOME_COMPLETED = "completed"
OUTCOME_FAILED = "failed"

# result class name -> (examined field names, produced field names)
#
# Both sides are tuples and are SUMMED, because several jobs split one fact
# across several counters. An empty tuple means the result type carries no
# counter for that fact, and the column is written NULL -- which is itself
# information: it says this job cannot tell an empty input from a broken one,
# and no query should pretend otherwise.
#
# Per-entry notes where the choice is not self-evident:
#   StalenessResult   -- `active_count` is only the healthiest tier, so the
#                        examined universe is the sum of all three tiers.
#                        Produced is the two tiers it flags for follow-up.
#   HealthScore       -- read-only. It measures the graph and mutates nothing,
#                        so `produced` is deliberately NULL rather than 0. A
#                        zero-output alarm filtering on `produced = 0` must not
#                        fire on a scorer that has no output to give.
#   CentralityResult  -- `entities_scored` is both: PageRank scores every node
#                        it reads and writes the score back. examined ==
#                        produced always, correctly.
#   CommunityResult   -- carries NO examined counter. The GDS-unavailable path
#                        returns `CommunityResult(0, 0, elapsed)`, which is
#                        exactly the returns-zeros-without-looking shape, and
#                        this mapping cannot rescue it: the fix is a counter on
#                        the result type. Recorded as NULL so the gap is
#                        visible in the data instead of implied by a 0.
#   SkillDerivationResult -- `patterns_detected` is the closest thing to an
#                        examined counter; `ToolUsageTracker` reports no
#                        scanned-records count. Produced sums the three write
#                        counters, since an alarm cares that SOMETHING landed.
_RESULT_COUNTERS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "DecayResult": (("entities_scanned",), ("entities_decayed", "entities_archived")),
    "StalenessResult": (
        ("active_count", "stale_count", "very_stale_count"),
        ("stale_count", "very_stale_count"),
    ),
    "OrphanResult": (("entities_scanned",), ("orphans_archived",)),
    "HealthScore": (("entity_count",), ()),
    "ReflectionResult": (("events_processed",), ("operations_applied",)),
    "CommunityResult": ((), ("entities_labeled",)),
    "CentralityResult": (("entities_scored",), ("entities_scored",)),
    "EmbeddingMaintenanceResult": (("entities_scanned",), ("embeddings_regenerated",)),
    "SkillDerivationResult": (
        ("patterns_detected",),
        ("skills_created", "skills_updated", "capabilities_created"),
    ),
}

# Sub-score fields of `HealthScore` that belong in `graph_health_events.metrics`.
# `overall` goes to the `health_score` column and the two counts to their own
# columns, so they are excluded here rather than duplicated.
_HEALTH_SUBSCORES = (
    "freshness",
    "confidence",
    "connectivity",
    "consistency",
    "coverage",
    "self_model",
)


class CurationRunRecorder(Protocol):
    """Durable sink for curation job executions.

    Structurally satisfied by `backend.event_store.store.EventStore`. Declared
    here rather than in `backend/interfaces.py` because it is a curation-local
    contract, not one of the four application I/O boundaries.
    """

    def append_curation_job_run(
        self,
        *,
        run_id: str,
        job_name: str,
        trigger_source: str,
        started_at: str,
        duration_ms: float,
        outcome: str,
        result_type: str | None,
        examined: int | None,
        produced: int | None,
        metrics: str | None,
        error: str | None,
    ) -> str:
        """Append one curation-job execution row. Returns the stored run_id."""
        ...

    def append_graph_health_event(
        self,
        *,
        event_id: str,
        timestamp: str,
        health_score: float,
        metrics: str,
        entity_count: int | None,
        relationship_count: int | None,
        archived_count: int | None = None,
        community_count: int | None = None,
    ) -> str:
        """Append one graph-health metric sample. Returns the stored event_id."""
        ...


@dataclass(frozen=True, slots=True)
class ResultFacts:
    """The three separable facts about one job execution's output.

    Attributes:
        result_type: Class name of what `run()` returned, None if it returned
            None.
        examined: Units of input the result reports having looked at. None
            when the result type declares no such counter -- distinct from 0,
            which asserts the job looked and found nothing.
        produced: Units of output the result reports having written. None when
            the job is read-only or declares no such counter.
        metrics: JSON object holding EVERY field of the result verbatim,
            including the job's own self-reported `duration_ms`. Nothing the
            job returned is lost to the summarisation above.
    """

    result_type: str | None
    examined: int | None
    produced: int | None
    metrics: str | None


def describe_result(result: Any) -> ResultFacts:
    """Split a job result into the facts the run ledger stores separately.

    Args:
        result: Whatever `job.run()` returned. Any object is accepted; a
            non-dataclass is recorded by type name alone rather than rejected,
            since refusing to record an unfamiliar result would reintroduce
            the silence this table exists to end.

    Returns:
        ResultFacts. All fields are None when `result` is None.
    """
    if result is None:
        return ResultFacts(result_type=None, examined=None, produced=None, metrics=None)

    result_type = type(result).__name__

    if not is_dataclass(result) or isinstance(result, type):
        return ResultFacts(result_type=result_type, examined=None, produced=None, metrics=None)

    values = {f.name: getattr(result, f.name) for f in fields(result)}
    examined_fields, produced_fields = _RESULT_COUNTERS.get(result_type, ((), ()))

    return ResultFacts(
        result_type=result_type,
        examined=_sum_fields(values, examined_fields),
        produced=_sum_fields(values, produced_fields),
        # default=str so a field holding a non-JSON-native value (tuples of
        # dicts, datetimes) degrades to its repr instead of raising and
        # costing the whole row.
        metrics=json.dumps(values, default=str),
    )


def health_event_fields(result: Any) -> dict[str, Any] | None:
    """Project a `HealthScore` onto the `graph_health_events` columns.

    Args:
        result: A job result. Anything that is not a `HealthScore` returns
            None -- dispatch is on the result type rather than the job name so
            a renamed `JobConfig` cannot silently stop the time series.

    Returns:
        Keyword arguments for `CurationRunRecorder.append_graph_health_event`,
        minus `event_id` and `timestamp` which the caller mints. None when the
        result is not a health score.
    """
    if type(result).__name__ != "HealthScore" or not is_dataclass(result):
        return None

    present = {f.name for f in fields(result)}
    if not {"overall", "entity_count", "relationship_count"} <= present:
        return None

    return {
        "health_score": float(result.overall),
        "metrics": json.dumps(
            {name: getattr(result, name) for name in _HEALTH_SUBSCORES if name in present}
        ),
        "entity_count": result.entity_count,
        "relationship_count": result.relationship_count,
    }


def _sum_fields(values: dict[str, Any], names: tuple[str, ...]) -> int | None:
    """Sum the named integer fields, or None if none of them are present.

    Args:
        values: Field name -> value for one result dataclass.
        names: Field names to sum. Empty means the fact is not reported.

    Returns:
        The sum, or None when `names` is empty or no name resolves to an int.
    """
    present = [values[name] for name in names if isinstance(values.get(name), int)]
    if not present:
        return None
    return sum(present)
