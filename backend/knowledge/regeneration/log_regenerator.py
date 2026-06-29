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

import json
import uuid
from dataclasses import dataclass

from backend.errors import MistError
from backend.knowledge.curation.pipeline import CurationResult
from backend.knowledge.eval_isolation import assert_rebuild_target_not_live
from backend.knowledge.extraction.validator import ValidationResult


class ColdCacheError(MistError):
    """Raised when the extraction cache does not cover 100% of the epoch's turns."""


class RebuildError(MistError):
    """Raised when a rebuild operation cannot proceed."""


@dataclass(frozen=True)
class RebuildReport:
    """Summary returned by LogRegenerator.rebuild on successful completion."""

    job_id: str
    turns_processed: int
    turns_failed: int
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
        job_id: str | None = None,
        resume_from: str | None = None,
    ) -> RebuildReport:
        """Replay the log into staging from the cache. Never writes to live.

        Args:
            staging_uri: Bolt URI for the staging Neo4j (must not equal live_uri).
            live_uri: Bolt URI for the live Neo4j (guard only; never written to).
            epoch: Epoch dict with ontology_version, extraction_version, model_hash,
                epoch_id, activated_at.
            job_id: Optional caller-provided job identifier. For fresh runs this is
                optional (a unique id is generated). For resume runs this is REQUIRED
                (it must match the row created during the initial run).
            resume_from: Event ID to resume after (cursor-based); None for a full run.

        Returns:
            RebuildReport with job_id, turns_processed, turns_failed, staging_uri,
            and epoch_id.

        Raises:
            ColdCacheError: If any turn in the epoch is not in the extraction cache.
            RebuildError: If resume_from is set but job_id is None.
        """
        assert_rebuild_target_not_live(staging_uri, live_uri)

        turns = self._events.get_all_turns_for_reextraction(after_event_id=resume_from)
        self._assert_cache_coverage(turns, epoch)

        if resume_from is None:
            # Fresh run: generate a unique job_id so repeated rebuilds of the same
            # epoch do not collide on the primary key.
            if job_id is None:
                job_id = f"rebuild-{epoch['epoch_id']}-{uuid.uuid4().hex[:8]}"
            started_at = turns[0]["timestamp"] if turns else epoch["activated_at"]
            self._events.create_reextraction_job(
                job_id=job_id,
                target_ontology_version=epoch["ontology_version"],
                source_ontology_version=None,
                total_events=len(turns),
                started_at=started_at,
            )
        else:
            # Resume: the job row already exists from the initial run.
            # job_id MUST be provided by the caller -- we have no way to recover it.
            if job_id is None:
                raise RebuildError(
                    "job_id is required when resuming a rebuild (resume_from is set). "
                    "Pass the original job_id returned by the initial rebuild call."
                )

        processed = 0
        turns_failed = 0
        collected_errors: list[str] = []
        last_ts: str = epoch["activated_at"]

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
            result: CurationResult = await self._curation.curate_and_store(
                vr,
                event_id=turn["event_id"],
                session_id=turn["session_id"],
                recorded_at=turn["timestamp"],
            )
            if result.stage_errors:
                turns_failed += 1
                collected_errors.extend(result.stage_errors)
            processed += 1
            last_ts = turn["timestamp"]
            self._events.checkpoint_reextraction_job(job_id, turn["event_id"], processed, last_ts)

        final_status = "failed" if turns_failed else "completed"
        self._events.finalize_reextraction_job(
            job_id=job_id,
            status=final_status,
            failed=turns_failed,
            errors=json.dumps(collected_errors) if collected_errors else None,
            updated_at=last_ts,
        )

        return RebuildReport(
            job_id=job_id,
            turns_processed=processed,
            turns_failed=turns_failed,
            staging_uri=staging_uri,
            epoch_id=epoch["epoch_id"],
        )
