"""R1.2 -- cache-driven log->graph regenerator (proof-first).

Replays the immutable event log (rowid order, epoch-pinned) into a fresh
staging Neo4j, deterministically, with NO in-loop LLM: every turn's extraction
result is pulled from the content-addressed ExtractionCache (coverage REQUIRED).
Each result is curated + reconciled into staging via the existing
`curate_and_store` (dedup + reconcile_turn under the Inv-A9 write lock). The
self-model copy-forward + cross-layer re-derivation (R1.2 Task 4) and the
build-then-swap CUTOVER (deferred) are separate.

The rebuild NEVER writes to the live GRAPH: `assert_rebuild_target_not_live`
gates the staging URI, and the live `source` connection is read-only.

That sentence was true and too narrow. It said nothing about the two SQLite
stores this class also holds, and the event store is where the rebuild wrote its
own job/checkpoint rows -- into whichever store it was handed, which for the CLI
is the LIVE one. Progress now goes to an explicitly injected `journal` (see
rebuild_journal.py); the `event_store` dependency is the replay SOURCE and no
row is ever written to it.

Precisely "no rows", not "no bytes": `EventStore._get_connection` runs `PRAGMA
journal_mode=WAL` on every open, which rewrites a non-WAL database's header and
creates `-wal`/`-shm` sidecars. Harmless here -- the live store is already WAL --
but stated exactly, because a too-broad sentence in this same docstring is what
hid the original defect. Neither isolation guard could have caught that defect:
both reason about bolt URIs and cannot see a SQLite path.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from backend.errors import MistError
from backend.knowledge.curation.pipeline import CurationResult
from backend.knowledge.eval_isolation import assert_rebuild_target_not_live
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.extraction_cache import OUTCOME_SKIPPED
from backend.knowledge.ontologies import EDGE_TYPES_BY_NAME
from backend.knowledge.regeneration.rebuild_journal import RebuildJournal
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL

logger = logging.getLogger(__name__)

# Provenance values a rebuild of the CANONICAL graph replays. Fail-closed:
# `origin` is a guard R1.4 added so probe and eval traffic could be kept out of
# the graph, so the default excludes everything not marked genuine usage. A
# caller that genuinely wants fixture traffic replayed (the golden log, whose
# sessions are `origin='test'` by construction) must say so explicitly.
CANONICAL_ORIGINS: tuple[str, ...] = ("real",)


class ColdCacheError(MistError):
    """Raised when the extraction cache does not cover 100% of the epoch's turns."""


class RebuildScopeError(MistError):
    """The selection matched no turn in a populated log.

    A configuration failure, not a gate failure, and refused before any write.
    See the raise site in `rebuild` for why a warning was not enough. MIS-138.
    """


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
    # The DOMAIN the selection ran under, recorded so a run's evidence states it
    # rather than a plan asserting it. `ontology_version` and `origins` are not
    # merely stamps -- both RESTRICT which turns are in the function's domain, and
    # `turns_processed` alone cannot distinguish "replayed the whole log" from
    # "replayed the subset this epoch could see". MIS-138; feeds ADR-023 section 7.
    #
    # Defaulted so existing constructions stay valid; `rebuild` always populates
    # them.
    ontology_version: str | None = None
    origins: tuple[str, ...] = ()
    total_logged: int = 0


class LogRegenerator:
    """Rebuilds a staging entity graph from the event log + extraction cache.

    Dependencies are injected (DI rule): the event store to replay FROM, the
    extraction cache, a curation pipeline already wired to the STAGING graph
    store, and the journal its own progress rows are written TO. The last one is
    required precisely because the first must be the live store -- see
    rebuild_journal.py.
    """

    # Intra-self-model edges (both endpoints in :__SelfModel__) -- copied verbatim.
    _INTRA_SELF_MODEL_EDGES = (
        "HAS_TRAIT",
        "HAS_CAPABILITY",
        "HAS_PREFERENCE",
        "IS_UNCERTAIN_ABOUT",
    )
    # Cross-layer edges (self-model <-> :__Entity__) are discovered STRUCTURALLY --
    # see `rederive_self_model_cross_layer_edges`. There is deliberately no edge-type
    # tuple here. The retired `_CROSS_LAYER_EDGES` named four of the ten types the
    # ontology permits from a self-model source (`ADAPTED_FOR`, `DEPENDS_ON`,
    # `REFERENCES_DOCUMENT`, `USES`, `WORKS_WITH`, `RELATED_TO` were missing) and none
    # of the reverse direction, so every unlisted type was dropped on rebuild.
    # A hand-list that must agree with the ontology drifts from it on the next bump;
    # a longer hand-list only moves the next drift further out.
    _PARTITION_DIRECTIONS = (
        (SELF_MODEL_LABEL, ENTITY_LABEL),
        (ENTITY_LABEL, SELF_MODEL_LABEL),
    )

    def __init__(
        self,
        *,
        event_store,
        extraction_cache,
        staging_curation_pipeline,
        journal: RebuildJournal,
        confidence_scorer,
        temporal_resolver,
        normalizer,
        validator,
    ) -> None:
        self._events = event_store
        self._cache = extraction_cache
        self._curation = staging_curation_pipeline
        # `journal` is REQUIRED, not defaulted to `event_store`, and the REQUIREDNESS
        # is the whole of the protection. A default is what the bug looked like: the
        # replay source must be the LIVE store, so any implicit "journal into the
        # store you were given" sends a dry-run proof's job rows straight to the live
        # ledger. Every caller now states where its progress goes.
        #
        # The `: RebuildJournal` annotation above DOCUMENTS the contract; it does not
        # enforce it. Under `from __future__ import annotations` (:29) the annotation
        # is never evaluated, and nothing here runs `isinstance`, even though the
        # Protocol is `@runtime_checkable` (rebuild_journal.py:32). Verified rather
        # than assumed: `LogRegenerator.__init__.__annotations__` is
        # `{'journal': 'RebuildJournal', ...}` -- the string, unresolved -- and
        # constructing with `journal="not a journal at all"` succeeds, while omitting
        # it raises `TypeError: missing 1 required keyword-only argument: 'journal'`.
        # So a caller that passes the WRONG journal is caught by review and by
        # test_rebuild_cli_is_read_only.py, not here; a caller that passes NONE cannot
        # construct the object at all. Do not read the annotation as a runtime guard.
        self._journal = journal
        # confidence_scorer / temporal_resolver / normalizer / validator are REQUIRED
        # for the same reason `journal` is: a default is how a silently-skipped stage
        # survives. The cache holds RAW Stage-2 output (spec D2) -- replay's whole
        # point is to re-run Stages 3-6 against it under the CURRENT ontology, so an
        # ontology bump is re-derived rather than replaying stale post-Stage-6 state.
        # A caller that forgets one of these four cannot construct a LogRegenerator at
        # all; a caller that passes the wrong one still runs, same non-guarantee as
        # `journal` above.
        self._confidence_scorer = confidence_scorer
        self._temporal_resolver = temporal_resolver
        self._normalizer = normalizer
        self._validator = validator

    def _assert_cache_coverage(self, turns: list[dict], epoch: dict) -> None:
        uncached = [
            t["event_id"]
            for t in turns
            if self._cache.get(
                t["event_id"],
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
        step -- R1.6's composition work, not this method's contract. `R1.4
        whole-branch review, I1`: `apply_seed_documents` already routes each
        node to the partition its own `SeedDocument.partition` declares
        (`_assign_node_partitions`, R1.4 Task 4's rework, `5bbaac1`) --
        `:__SelfModel__` content lands on `:__SelfModel__`, not on a
        colliding `:__Entity__` copy. That is not the remaining gap. The
        remaining gap is that `rebuild()` has no seed-apply step of any kind
        yet -- nothing in this class calls `apply_seed_documents` or
        `reseed` against the staging graph at all, so there is nothing this
        copy could be replaced by until R1.6 adds one.
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
        """Re-create cross-layer self-model <-> :__Entity__ edges in staging by id.

        Reads every edge spanning the two partitions in EITHER direction and MERGEs it
        in staging keyed on canonical (id) at both ends. Skips (and counts) edges whose
        target id is absent from staging (a target the log-replay did not produce, e.g.
        a formerly vault-derived-only entity).

        The partition-pair predicate is deliberately the same one `dump_graph_json` uses
        to collect `self_model_cross_layer_edges` (`admin.py:972-981`, structural with no
        type filter, both directions). The comparison surface and the re-derivation must
        cover the same set of edges: an edge the comparison reads but the re-derivation
        does not re-create is a guaranteed RED on the `live == rebuilt` gate that says
        nothing about determinism. That cost is why this reads structurally rather than
        from a list -- and why the two queries must stay in step if either is edited.

        Edge types come from the DATA (Cypher cannot parameterize a relationship type,
        so the type is interpolated) and are therefore validated against the ontology
        first; an undeclared type is refused and counted rather than interpolated.
        Partition labels are module constants, never data.

        NOT handled here: self-model <-> :__Provenance__ edges, which the ontology permits
        as `LEARNED_SELF` (-> LearningEvent), `DERIVED_FROM` (-> VectorChunk /
        ExternalSource / VaultNote), and `RELATED_TO` to a provenance target. No gate key
        compares them either -- `cross_layer_edges` covers entity <-> provenance and this
        one covers entity <-> self-model, so self-model <-> provenance is in neither. That
        blind spot is a comparison-surface decision (ADR-023's residue table), not this
        method's contract, and is recorded rather than silently closed here.

        Returns {edges, skipped, unknown}.
        """
        created, skipped, unknown = 0, 0, 0
        for source_label, target_label in self._PARTITION_DIRECTIONS:
            edges = source_conn.execute_query(
                f"MATCH (s:{source_label})-[r]->(t:{target_label}) "
                "RETURN s.id AS s, type(r) AS type, t.id AS t, properties(r) AS props",
                {},
            )
            for e in edges:
                edge_type = e["type"]
                if edge_type not in EDGE_TYPES_BY_NAME:
                    logger.warning(
                        "Cross-layer edge type %r on %s->%s is not declared in the "
                        "ontology; refusing to re-derive it",
                        edge_type,
                        e["s"],
                        e["t"],
                    )
                    unknown += 1
                    continue
                result = staging_conn.execute_write(
                    f"MATCH (s:{source_label} {{id: $s}}) "
                    f"MATCH (t:{target_label} {{id: $t}}) "
                    f"MERGE (s)-[r:{edge_type}]->(t) SET r = $props "
                    "RETURN count(r) AS n",
                    {"s": e["s"], "t": e["t"], "props": e["props"]},
                )
                if result and result[0]["n"] > 0:
                    created += 1
                else:
                    skipped += 1
        return {"edges": created, "skipped": skipped, "unknown": unknown}

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
        origins: tuple[str, ...] = CANONICAL_ORIGINS,
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
            origins: Session provenance values to replay. Defaults to
                `CANONICAL_ORIGINS` (`('real',)`) -- a rebuild of the canonical
                graph must not absorb probe or eval traffic. Pass explicitly to
                replay fixture traffic (`('test',)` for the golden log).

        Returns:
            RebuildReport with job_id, turns_processed, turns_failed, staging_uri,
            and epoch_id.

        Raises:
            ColdCacheError: If any turn in the epoch is not in the extraction cache.
            RebuildError: If resume_from is set but job_id is None, or if resume_from
                is set while `journal` is non-durable (no checkpoint rows were ever
                persisted, so there is no cursor to resume against).
        """
        assert_rebuild_target_not_live(staging_uri, live_uri)

        # Fail-fast resume guard: job_id is required when resuming. Checked here
        # so callers get an immediate error before any event-store reads occur.
        if resume_from is not None and job_id is None:
            raise RebuildError(
                "job_id is required when resuming a rebuild (resume_from is set). "
                "Pass the original job_id returned by the initial rebuild call."
            )

        # A resume reads a cursor the journal was supposed to have persisted. A
        # non-durable journal never wrote one, so the resume would silently replay
        # from `resume_from` against a job row that does not exist and report
        # success. Refuse instead of half-honouring it.
        if resume_from is not None and not self._journal.durable:
            raise RebuildError(
                f"Cannot resume with a non-durable journal ({type(self._journal).__name__}): "
                "no checkpoint rows were ever written, so there is no cursor to resume "
                "against. Resume requires a durable journal (EventStoreRebuildJournal)."
            )

        # The scoping ontology_version comes from the epoch ROW, not from
        # `backend.knowledge.version_stamps.ONTOLOGY_VERSION`. This is not a
        # style preference. The cache keys below are derived from
        # `epoch["ontology_version"]`, so selecting turns under any other
        # authority would ask the cache for a triple the selection was not
        # made under -- exactly the two-authorities drift that caused the
        # 2026-08-02 incident. One epoch row, one triple, one selection.
        #
        # Until this call, `rebuild()` selected EVERY turn ever logged and then
        # demanded the current epoch's cache cover all of them. Any turn from a
        # superseded ontology epoch was a guaranteed miss -> ColdCacheError ->
        # abort before a single node was written. It passed only because the
        # live log is empty. The `epoch` in `graph = f(seed, log, epoch)` had
        # no scoping role at all.
        turns = self._events.get_all_turns_for_reextraction(
            ontology_version=epoch["ontology_version"],
            after_event_id=resume_from,
            origins=origins,
        )
        # Scoping can now select nothing from a non-empty log, which the cache
        # coverage gate would pass vacuously and every downstream assertion
        # would read as "the log was empty". Make that state auditable rather
        # than silent -- a rebuild that replays none of a populated log is
        # almost always a misconfigured epoch or origin, not a real no-op.
        total_logged = self._events.get_turn_count()
        logger.info(
            "Rebuild scope: %d of %d logged turns selected (epoch=%s, ontology=%s, origins=%s)",
            len(turns),
            total_logged,
            epoch["epoch_id"],
            epoch["ontology_version"],
            ",".join(origins),
        )
        if total_logged and not turns:
            # Refusal, not a warning. MIS-138.
            #
            # This was a `logger.warning` and the run continued: the replay no-ops,
            # `_assert_cache_coverage([])` passes vacuously over an empty selection,
            # and every downstream assertion reads the result as "the log was
            # empty". The rebuild reports success having produced nothing.
            #
            # The trigger is not an exotic misconfiguration. `ontology_version` is a
            # DOMAIN FILTER here, and turns are stamped with the version current when
            # they were written -- so the first ontology bump after any turns exist
            # makes this selection empty for every historical turn, permanently.
            # MIST has bumped 1.0.0 -> 1.1.0 -> 1.3.0 -> 1.4.0. A rebuild spanning a
            # bump is therefore the expected case, not the edge case, and warning
            # about it while proceeding is how it would reach a gate.
            #
            # Scoped to a POPULATED log deliberately: selecting nothing from an empty
            # log is correct, and refusing there would reject the one state this
            # cannot diagnose. An empty-log rebuild is the non-vacuity floors' job.
            raise RebuildScopeError(
                f"Rebuild selected 0 of {total_logged} logged turns: no turn matches "
                f"ontology_version={epoch['ontology_version']!r} AND origin in "
                f"({','.join(origins)}). Refusing rather than replaying nothing -- a "
                "no-op replay passes cache coverage vacuously and is indistinguishable "
                "downstream from an empty log. If the log spans an ontology bump, the "
                "historical turns carry the OLD version and this epoch cannot select "
                "them; if the corpus is probe or eval traffic, pass its origin "
                "explicitly."
            )

        self._assert_cache_coverage(turns, epoch)

        if resume_from is None:
            # Fresh run: generate a unique job_id so repeated rebuilds of the same
            # epoch do not collide on the primary key.
            if job_id is None:
                job_id = f"rebuild-{epoch['epoch_id']}-{uuid.uuid4().hex[:8]}"
            started_at = turns[0]["timestamp"] if turns else epoch["activated_at"]
            self._journal.create(
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

        # SEED-APPLY GOES HERE, ABOVE THIS LOOP -- not after it. MIS-130/MIS-138.
        #
        # `rebuild()` has no seed-apply step yet; the `:__SelfModel__` partition is
        # copied forward instead (see the calls after this loop, and
        # `copy_self_model_partition`'s docstring). When MIS-130 replaces that with a
        # real `apply_seed_documents` against staging, ORDER IS LOAD-BEARING and the
        # two writers do not commute:
        #
        #   seed      `ON MATCH SET n += $properties`      (seed/applier.py:62)
        #               -- unconditional clobber of every authored property
        #   extraction `display_name = CASE WHEN size(existing) < size(new) ...`
        #               (curation/graph_writer.py:251-256) -- longest-wins
        #
        # Live applies seed FIRST (`mist_admin seed` then `mist_admin hydrate`, per
        # docker-compose.dev-hydration.yml), so replayed facts reconcile ONTO seeded
        # nodes. Seeding AFTER this loop would let `n += $properties` overwrite values
        # the replay resolved by the longest-wins rule, producing a different graph
        # from identical inputs. `display_name` and `description` are in no exclusion
        # frozenset, so the gate sees it -- as a RED it will read as non-determinism.
        #
        # The existing post-loop composition step is the trap: it is the natural place
        # to add "one more" step, and it is the wrong side of the loop for this one.
        #
        # The matching gate assertion (seed node count non-zero BEFORE the first
        # replayed turn is applied) lands WITH the seed step, deliberately not before
        # it -- an assertion with no production caller is the defect MIS-137 existed
        # to fix, and adding a second one here would reproduce it.
        for turn in turns:
            cached = self._cache.get(
                turn["event_id"],
                epoch["extraction_version"],
                epoch["model_hash"],
            )
            # coverage was asserted above, so cached is never None here
            if cached["outcome"] == OUTCOME_SKIPPED:
                # A recorded decision, not a gap. The live pipeline looked at
                # this turn and declined; replaying it as a no-op is what makes
                # `live == rebuilt` reachable at all. Re-deciding here would need
                # the wall clock (rate limit) and a model (significance).
                processed += 1
                last_ts = turn["timestamp"]
                self._journal.checkpoint(job_id, turn["event_id"], processed, last_ts)
                continue

            extraction = ExtractionResult(
                entities=cached["entities"],
                relationships=cached["relationships"],
                # The cache holds RAW Stage-2 output (spec D2) -- no source_utterance,
                # since Stage 3's hedge/third-party confidence adjustment is one of the
                # things replay re-derives, not something the cache is trusted for.
                # `turn["user_utterance"]` is byte-identical to what fed the live
                # extraction: `conversation_handler._record_turn_event` writes
                # `user_utterance=user_message` to the event store, and the SAME
                # `user_message` is what `_extract_knowledge_async` passes as
                # `utterance=` on to `pre_process`, whose `original_text=utterance`
                # assignment is unconditional (`preprocessor.py`, no branch rewrites it
                # first). Omitting this left the adjustment permanently inert on
                # replay: `extraction.source_utterance` defaulted to `""`, so
                # `ConfidenceScorer.adjust_confidence`'s hedge-pattern search never
                # matched anything, and a rebuild would silently disagree with the live
                # graph on every hedged or third-party-attributed relationship.
                source_utterance=turn["user_utterance"],
            )
            # Stages 3-6, re-run rather than cached (spec D2). Pure code, no
            # model -- pinned by tests/unit/knowledge/extraction/test_stage_purity.py.
            reference_date = datetime.fromisoformat(turn["timestamp"])
            extraction = self._confidence_scorer.adjust_confidence(extraction)
            extraction = self._temporal_resolver.resolve(extraction, reference_date)
            extraction = await self._normalizer.normalize(extraction)
            vr = self._validator.validate(extraction)

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
            self._journal.checkpoint(job_id, turn["event_id"], processed, last_ts)

        final_status = "failed" if turns_failed else "completed"
        self._journal.finalize(
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
        # copy stays -- rebuild() has no seed-apply step at all yet (partition
        # routing itself is not the blocker; see copy_self_model_partition
        # docstring, corrected R1.4 whole-branch review I1). Retirement is R1.6's
        # composition work.
        if source_conn is not None and staging_conn is not None:
            self.copy_self_model_partition(source_conn, staging_conn)
            self.rederive_self_model_cross_layer_edges(source_conn, staging_conn)

        return RebuildReport(
            job_id=job_id,
            turns_processed=processed,
            turns_failed=turns_failed,
            staging_uri=staging_uri,
            epoch_id=epoch["epoch_id"],
            ontology_version=epoch["ontology_version"],
            origins=tuple(origins),
            total_logged=total_logged,
        )
