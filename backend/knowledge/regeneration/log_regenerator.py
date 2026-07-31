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
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL


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

    # Intra-self-model edges (both endpoints in :__SelfModel__) -- copied verbatim.
    _INTRA_SELF_MODEL_EDGES = (
        "HAS_TRAIT",
        "HAS_CAPABILITY",
        "HAS_PREFERENCE",
        "IS_UNCERTAIN_ABOUT",
    )
    # Cross-layer edges (self-model -> :__Entity__) -- re-derived by canonical id.
    _CROSS_LAYER_EDGES = (
        "MIST_HAS_TRAIT",
        "MIST_HAS_CAPABILITY",
        "MIST_HAS_PREFERENCE",
        "IMPLEMENTED_WITH",
    )

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

    def copy_self_model_partition(self, source_conn, staging_conn) -> int:
        """Copy :__SelfModel__ nodes + intra-self-model edges from source to staging.

        Preserve (option A): nodes + their HAS_*/IS_UNCERTAIN_ABOUT edges are
        copied verbatim (by id + labels + properties). Cross-layer MIST_HAS_*
        edges are NOT copied here (re-derived separately -- their targets are
        fresh :__Entity__ nodes from the replay). Returns the node count copied.

        As of R1.4 the self-model is also authored as ordinary seed content
        (`mist-memory/seed/mist.md`, verified to cover all 21 :__SelfModel__
        nodes), so this copy is redundant belt-and-braces rather than the sole
        durable source. Retiring it requires `rebuild()` to gain a seed-apply
        step -- R1.6's composition work, not this method's contract. That step
        cannot simply call `apply_seed_documents` unmodified: that function
        hardcodes the `:__Entity__` label for every node it writes
        (`backend/knowledge/seed/applier.py`), with no routing to
        `:__SelfModel__` for self-model types the way `admin.py`'s older seed
        path does. Applied as-is against self-model content, it would create
        `:__Entity__` nodes colliding on id with the existing `:__SelfModel__`
        nodes (e.g. `mist-identity`) rather than replacing them.
        """
        nodes = source_conn.execute_query(
            f"MATCH (n:{SELF_MODEL_LABEL}) "
            "RETURN n.id AS id, labels(n) AS labels, properties(n) AS props",
            {},
        )
        for n in nodes:
            labels = ":".join(lbl for lbl in n["labels"])  # includes __SelfModel__ + typed label
            staging_conn.execute_write(
                f"MERGE (x:{labels} {{id: $id}}) SET x = $props",
                {"id": n["id"], "props": n["props"]},
            )
        for edge_type in self._INTRA_SELF_MODEL_EDGES:
            edges = source_conn.execute_query(
                f"MATCH (s:{SELF_MODEL_LABEL})-[r:{edge_type}]->(t:{SELF_MODEL_LABEL}) "
                "RETURN s.id AS s, t.id AS t, properties(r) AS props",
                {},
            )
            for e in edges:
                staging_conn.execute_write(
                    f"MATCH (s:{SELF_MODEL_LABEL} {{id: $s}}) "
                    f"MATCH (t:{SELF_MODEL_LABEL} {{id: $t}}) "
                    f"MERGE (s)-[r:{edge_type}]->(t) SET r = $props",
                    {"s": e["s"], "t": e["t"], "props": e["props"]},
                )
        return len(nodes)

    def rederive_self_model_cross_layer_edges(self, source_conn, staging_conn) -> dict[str, int]:
        """Re-create cross-layer self-model -> :__Entity__ edges in staging by id.

        Reads each cross-layer edge from source and MERGEs it in staging keyed on
        canonical (id) at both ends. Skips (and counts) edges whose target entity
        id is absent from staging (a target that the log-replay did not produce,
        e.g. a formerly vault-derived-only entity). Returns {edges, skipped}.
        """
        created, skipped = 0, 0
        for edge_type in self._CROSS_LAYER_EDGES:
            edges = source_conn.execute_query(
                f"MATCH (s:{SELF_MODEL_LABEL})-[r:{edge_type}]->(t:{ENTITY_LABEL}) "
                "RETURN s.id AS s, t.id AS t, properties(r) AS props",
                {},
            )
            for e in edges:
                result = staging_conn.execute_write(
                    f"MATCH (s:{SELF_MODEL_LABEL} {{id: $s}}) "
                    f"MATCH (t:{ENTITY_LABEL} {{id: $t}}) "
                    f"MERGE (s)-[r:{edge_type}]->(t) SET r = $props "
                    "RETURN count(r) AS n",
                    {"s": e["s"], "t": e["t"], "props": e["props"]},
                )
                if result and result[0]["n"] > 0:
                    created += 1
                else:
                    skipped += 1
        return {"edges": created, "skipped": skipped}

    async def rebuild(
        self,
        *,
        staging_uri: str,
        live_uri: str,
        epoch: dict,
        job_id: str | None = None,
        resume_from: str | None = None,
        source_conn=None,
        staging_conn=None,
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
            source_conn: Optional Neo4jConnection to the live source graph (read-only).
                When provided alongside staging_conn, the self-model partition is
                copied forward from source into staging after the replay loop.
            staging_conn: Optional Neo4jConnection to the staging graph (write). Must
                be provided together with source_conn to enable self-model copy-forward.

        Returns:
            RebuildReport with job_id, turns_processed, turns_failed, staging_uri,
            and epoch_id.

        Raises:
            ColdCacheError: If any turn in the epoch is not in the extraction cache.
            RebuildError: If resume_from is set but job_id is None.
        """
        assert_rebuild_target_not_live(staging_uri, live_uri)

        # Fail-fast resume guard: job_id is required when resuming. Checked here
        # so callers get an immediate error before any event-store reads occur.
        if resume_from is not None and job_id is None:
            raise RebuildError(
                "job_id is required when resuming a rebuild (resume_from is set). "
                "Pass the original job_id returned by the initial rebuild call."
            )

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
        # Resume path: job_id was validated above (non-None), so the row
        # already exists in the event store from the initial run.

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

        # Self-model copy-forward (R1.2 Task 4): optional -- only runs when both
        # source_conn and staging_conn are provided. Task 3 callers omit both and
        # are unaffected.
        # R1.4: the self-model is now also authored in mist-memory/seed/, but this
        # copy stays -- rebuild() has no seed-apply step, and apply_seed_documents
        # cannot stand in for it unmodified (see copy_self_model_partition
        # docstring). Retirement is R1.6's composition work.
        if source_conn is not None and staging_conn is not None:
            self.copy_self_model_partition(source_conn, staging_conn)
            self.rederive_self_model_cross_layer_edges(source_conn, staging_conn)

        return RebuildReport(
            job_id=job_id,
            turns_processed=processed,
            turns_failed=turns_failed,
            staging_uri=staging_uri,
            epoch_id=epoch["epoch_id"],
        )
