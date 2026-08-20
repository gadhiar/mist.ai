"""Where a rebuild records its OWN progress -- separate from the log it replays.

`LogRegenerator` reads the event log to replay it and writes job/checkpoint rows to
record how far it got. Those are two different roles, and collapsing them into one
injected `EventStore` is what let `mist_admin graph-rebuild-from-log --dry-run` write
to the LIVE ledger: the replay source has to BE the live store, so the job rows went
there too. The Neo4j leg of the same class never had this problem -- `source_conn`
reads live and `staging_conn` takes the writes.

Neither isolation guard could have caught it. `assert_rebuild_target_not_live` and
`assert_neo4j_isolated` both reason about bolt URIs; a SQLite path is invisible to
them. So the fix is structural rather than another guard: a rebuild cannot be
constructed without saying where its progress goes.

Two implementations, and the choice is the caller's:

- `EventStoreRebuildJournal` -- durable rows in an event store. What a rebuild of
  record wants (and what makes `resume_from` possible). Used by the golden-log replay
  and the integration tests, each against its own disposable store.
- `NullRebuildJournal` -- records nothing. What a determinism PROOF wants: the run is
  not a rebuild of record, and nothing reads its rows.

`durable` exists so `rebuild()` can refuse a resume it cannot honour. Without it, a
resume against a null journal would silently restart from the top and report success.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RebuildJournal(Protocol):
    """Sink for a rebuild's job/checkpoint rows."""

    @property
    def durable(self) -> bool:
        """True when rows survive the process and can support a resume."""
        ...

    def create(
        self,
        *,
        job_id: str,
        target_ontology_version: str,
        source_ontology_version: str | None,
        total_events: int,
        started_at: str,
    ) -> None:
        """Open a job in status 'running'."""
        ...

    def checkpoint(self, job_id: str, last_event_id: str, processed: int, updated_at: str) -> None:
        """Advance the cursor after a turn is replayed."""
        ...

    def finalize(
        self,
        *,
        job_id: str,
        status: str,
        failed: int,
        errors: str | None,
        updated_at: str,
    ) -> None:
        """Close the job in a terminal status ('completed' or 'failed')."""
        ...


class EventStoreRebuildJournal:
    """Durable journal backed by an `EventStore`'s `re_extraction_jobs` table.

    The store passed here is the WRITE target and need not be the store being
    replayed. Handing it the live event store is legitimate for a rebuild of record;
    it is exactly what a dry-run proof must not do.
    """

    def __init__(self, event_store) -> None:
        """Bind the journal to the event store its rows are written to."""
        self._events = event_store

    @property
    def durable(self) -> bool:
        """True: rows land in SQLite and survive the process, so resume works."""
        return True

    def create(
        self,
        *,
        job_id: str,
        target_ontology_version: str,
        source_ontology_version: str | None,
        total_events: int,
        started_at: str,
    ) -> None:
        """Insert the job row in status 'running'."""
        self._events.create_reextraction_job(
            job_id=job_id,
            target_ontology_version=target_ontology_version,
            source_ontology_version=source_ontology_version,
            total_events=total_events,
            started_at=started_at,
        )

    def checkpoint(self, job_id: str, last_event_id: str, processed: int, updated_at: str) -> None:
        """Advance the persisted cursor so a later run can resume from it."""
        self._events.checkpoint_reextraction_job(job_id, last_event_id, processed, updated_at)

    def finalize(
        self,
        *,
        job_id: str,
        status: str,
        failed: int,
        errors: str | None,
        updated_at: str,
    ) -> None:
        """Close the job row in a terminal status."""
        self._events.finalize_reextraction_job(
            job_id=job_id,
            status=status,
            failed=failed,
            errors=errors,
            updated_at=updated_at,
        )


class NullRebuildJournal:
    """Records nothing. For runs that are proofs, not rebuilds of record.

    Not merely "writes to a scratch store" -- it holds no state at all, so there is
    nothing to accidentally read back and mistake for a real job history.
    """

    @property
    def durable(self) -> bool:
        """False: nothing is persisted, so `rebuild()` refuses to resume against it."""
        return False

    def create(
        self,
        *,
        job_id: str,
        target_ontology_version: str,
        source_ontology_version: str | None,
        total_events: int,
        started_at: str,
    ) -> None:
        """Discard the job-open event."""
        return None

    def checkpoint(self, job_id: str, last_event_id: str, processed: int, updated_at: str) -> None:
        """Discard the checkpoint."""
        return None

    def finalize(
        self,
        *,
        job_id: str,
        status: str,
        failed: int,
        errors: str | None,
        updated_at: str,
    ) -> None:
        """Discard the terminal status."""
        return None
