"""R1.2 -- cache-driven log->graph regenerator (proof-first).

Replays the immutable event log (rowid order, epoch-pinned) into a fresh
staging Neo4j, deterministically, with NO in-loop LLM: every turn's extraction
result is pulled from the content-addressed ExtractionCache (coverage REQUIRED).
Each result is curated + reconciled into staging via the existing
`curate_and_store` (dedup + reconcile_turn under the Inv-A9 write lock). The
self-model copy-forward + cross-layer re-derivation (R1.2 Task 4) and the
build-then-swap CUTOVER (deferred) are separate.

The rebuild NEVER writes to the live graph: `assert_rebuild_target_not_live`
gates the staging URI, and the live `source` connection is read-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.errors import MistError
from backend.knowledge.eval_isolation import assert_rebuild_target_not_live
from backend.knowledge.extraction.validator import ValidationResult


class ColdCacheError(MistError):
    """Raised when the extraction cache does not cover 100% of the epoch's turns."""


@dataclass(frozen=True)
class RebuildReport:
    """Summary returned by LogRegenerator.rebuild on successful completion."""

    job_id: str
    turns_processed: int
    staging_uri: str
    epoch_id: int


class LogRegenerator:
    """Rebuilds a staging entity graph from the event log + extraction cache.

    Dependencies are injected (DI rule): the event store, the extraction cache,
    and a curation pipeline already wired to the STAGING graph store.
    """

    def __init__(self, *, event_store, extraction_cache, staging_curation_pipeline) -> None:
        self._events = event_store
        self._cache = extraction_cache
        self._curation = staging_curation_pipeline

    def _assert_cache_coverage(self, turns: list[dict], epoch: dict) -> None:
        uncached = [
            t["event_id"]
            for t in turns
            if self._cache.get(
                t["event_id"],
                epoch["ontology_version"],
                epoch["extraction_version"],
                epoch["model_hash"],
            )
            is None
        ]
        if uncached:
            raise ColdCacheError(
                f"{len(uncached)} of {len(turns)} turns are uncached for epoch "
                f"{epoch['epoch_id']} (ontology={epoch['ontology_version']}, "
                f"extraction={epoch['extraction_version']}). R1.2 is cache-driven; "
                f"warm the cache before rebuilding. First uncached: {uncached[:3]}"
            )

    async def rebuild(
        self,
        *,
        staging_uri: str,
        live_uri: str,
        epoch: dict,
        resume_from: str | None = None,
    ) -> RebuildReport:
        """Replay the log into staging from the cache. Never writes to live."""
        assert_rebuild_target_not_live(staging_uri, live_uri)

        turns = self._events.get_all_turns_for_reextraction(after_event_id=resume_from)
        self._assert_cache_coverage(turns, epoch)

        job_id = f"rebuild-{epoch['epoch_id']}-{turns[0]['event_id'] if turns else 'empty'}"
        started_at = turns[0]["timestamp"] if turns else epoch["activated_at"]
        self._events.create_reextraction_job(
            job_id=job_id,
            target_ontology_version=epoch["ontology_version"],
            source_ontology_version=None,
            total_events=len(turns),
            started_at=started_at,
        )

        processed = 0
        for turn in turns:
            cached = self._cache.get(
                turn["event_id"],
                epoch["ontology_version"],
                epoch["extraction_version"],
                epoch["model_hash"],
            )
            # coverage was asserted above, so cached is never None here
            vr = ValidationResult(
                valid=True,
                entities=cached["entities"],
                relationships=cached["relationships"],
            )
            await self._curation.curate_and_store(
                vr,
                event_id=turn["event_id"],
                session_id=turn["session_id"],
                recorded_at=turn["timestamp"],
            )
            processed += 1
            self._events.checkpoint_reextraction_job(
                job_id, turn["event_id"], processed, turn["timestamp"]
            )

        return RebuildReport(
            job_id=job_id,
            turns_processed=processed,
            staging_uri=staging_uri,
            epoch_id=epoch["epoch_id"],
        )
