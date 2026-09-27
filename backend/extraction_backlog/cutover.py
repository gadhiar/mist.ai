"""Epoch cutover: re-extract the whole log under a new model, check it, promote it.

Decision 7 (MIS-171): changing the extraction model means a NEW epoch and
re-extraction of the WHOLE log through the backlog, with the old graph kept
until the new one passes a check. The lifecycle, one `epoch_cutover` row:

    begin     -> filling   the candidate stamps are recorded; nothing is live
    (fill)    -> ready     the dispatcher has a candidate cache row for every
                           logged turn (it keeps filling new turns after this)
    rebuild   -> checked   a staging graph built twice from the log under the
                           candidate passed the rebuild gates
    promote   -> promoted  the candidate is appended to `epoch_ledger` with a
                           first-hand activation, in one SQLite transaction
    abandon   -> abandoned deletes nothing

Filling is the dispatcher's job (`ExtractionDispatcher._fill_step`); this
module holds the operator steps the admin CLI runs, and the rebuild check.

What is live while a cutover is open: the OLD graph, frozen. The dispatcher
applies nothing while filling (it only caches under the candidate's stamps), so
the active epoch's apply-pending turns stay pending; the service now serves the
candidate model and cannot extract for the active epoch.

The code never writes the live Neo4j graph. `promote` requires
`--graph-swapped`, which the operator passes after doing the host swap in
`CUTOVER.md`.

Every transition logs one structured INFO line (`log_transition`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from backend.errors import MistError
from backend.event_store.store import EpochCutoverStateError

from .store import BacklogStore, Cutover, compose_epoch_model_hash

logger = logging.getLogger(__name__)

# Exit codes of `cutover rebuild`, the same families as
# `scripts/mist_admin.py::cmd_graph_rebuild_from_log`: 0 every gate green,
# 1 rebuild-twice disagreed (non-determinism), 2 refused before measuring,
# 4 a non-vacuity floor or the self-model gate failed. 3 (live != rebuilt) is
# not used: a new model is expected to produce a different graph, so that
# comparison is reported as information and gates nothing.
EXIT_OK = 0
EXIT_NONDETERMINISTIC = 1
EXIT_REFUSED = 2
EXIT_VACUOUS = 4

_FAILURE_TEXT_LIMIT = 4000


class CutoverRefusedError(MistError):
    """An operator cutover step was refused; nothing was changed."""


def log_transition(cutover: Cutover, from_state: str, to_state: str, **fields: Any) -> None:
    """The one structured INFO line every cutover transition logs."""
    extra = " ".join(f"{name}={value}" for name, value in fields.items())
    logger.info(
        "epoch_cutover transition cutover_id=%d from=%s to=%s ontology_version=%s "
        "extraction_version=%s model_hash=%s%s",
        cutover.cutover_id,
        from_state,
        to_state,
        cutover.ontology_version,
        cutover.extraction_version,
        cutover.model_hash,
        f" {extra}" if extra else "",
    )


# ---------------------------------------------------------------------------
# begin / abandon / promote
# ---------------------------------------------------------------------------


def begin_cutover(
    store: BacklogStore,
    *,
    bare_model_hash: str,
    extraction_version: str,
    ontology_version: str,
    embedding_model_name: str,
    now_iso: str,
) -> Cutover:
    """Open a cutover candidate in state 'filling'.

    The candidate's `model_hash` is composed from the bare service hash and the
    backend's embedding model through `compose_model_hash`, never inline.

    Raises:
        CutoverRefusedError: The ledger is empty, a cutover is already open, or
            the candidate's stamps equal the active epoch's (nothing to cut
            over to).
    """
    epoch = store.active_epoch()
    if epoch is None:
        raise CutoverRefusedError("the epoch ledger is empty; there is no epoch to cut over from")
    model_hash = compose_epoch_model_hash(bare_model_hash, embedding_model_name)
    if (ontology_version, extraction_version, model_hash) == (
        epoch.ontology_version,
        epoch.extraction_version,
        epoch.model_hash,
    ):
        raise CutoverRefusedError(
            f"the candidate stamps equal the active epoch {epoch.epoch_id}'s; nothing to cut over"
        )
    cutover = store.begin_cutover(
        ontology_version=ontology_version,
        extraction_version=extraction_version,
        model_hash=model_hash,
        bare_model_hash=bare_model_hash,
        source_epoch_id=epoch.epoch_id,
        requested_at=now_iso,
    )
    if cutover is None:
        existing = store.open_cutover()
        raise CutoverRefusedError(
            f"cutover {existing.cutover_id if existing else '?'} is already open "
            f"({existing.state if existing else '?'}); promote or abandon it first"
        )
    log_transition(cutover, "none", "filling", source_epoch_id=epoch.epoch_id)
    return cutover


def abandon_cutover(store: BacklogStore, *, now_iso: str) -> Cutover:
    """Close the open cutover as 'abandoned'. Deletes nothing.

    The candidate's cache rows stay: they are keyed by the candidate's stamps,
    so no other epoch reads them, and a later cutover to the same stamps
    reuses them.

    Raises:
        CutoverRefusedError: No cutover is open.
    """
    cutover = store.open_cutover()
    if cutover is None:
        raise CutoverRefusedError("no cutover is open")
    if not store.transition_cutover(
        cutover, from_states=(cutover.state,), to_state="abandoned", updated_at=now_iso
    ):
        raise CutoverRefusedError(f"cutover {cutover.cutover_id} changed state; re-run status")
    log_transition(cutover, cutover.state, "abandoned")
    return cutover


@dataclass(frozen=True, slots=True)
class Promotion:
    """What `promote_cutover` did."""

    cutover: Cutover
    epoch_id: int
    marked_applied: int
    turns_at_activation: int


def promote_cutover(store: BacklogStore, *, graph_swapped: bool, now_iso: str) -> Promotion:
    """Make the checked candidate the active epoch.

    Requires state 'checked' AND `graph_swapped`: the operator's statement that
    the live graph has been replaced by the checked staging graph (CUTOVER.md).
    The ledger append, the first-hand activation, the applied markers through
    `rebuilt_through_event_id` and the state change are one SQLite transaction
    (`EventStore.promote_epoch_cutover`).

    Raises:
        CutoverRefusedError: No cutover is open, it is not 'checked',
            `graph_swapped` is False, or the store refused the promotion.
    """
    cutover = store.open_cutover()
    if cutover is None:
        raise CutoverRefusedError("no cutover is open")
    if cutover.state != "checked":
        raise CutoverRefusedError(
            f"cutover {cutover.cutover_id} is {cutover.state!r}; run `cutover rebuild` until "
            "it records 'checked'"
        )
    if not graph_swapped:
        raise CutoverRefusedError(
            "--graph-swapped is required: pass it only after the live graph has been "
            "replaced by the checked staging graph (backend/extraction_backlog/CUTOVER.md)"
        )
    try:
        result = store.promote_cutover(cutover, activated_at=now_iso)
    except EpochCutoverStateError as exc:
        raise CutoverRefusedError(str(exc)) from exc
    log_transition(
        cutover,
        "checked",
        "promoted",
        epoch_id=result["epoch_id"],
        marked_applied=result["marked_applied"],
        turns_at_activation=result["turns_at_activation"],
        rebuilt_through_event_id=cutover.rebuilt_through_event_id,
    )
    return Promotion(
        cutover=cutover,
        epoch_id=int(result["epoch_id"]),
        marked_applied=int(result["marked_applied"]),
        turns_at_activation=int(result["turns_at_activation"]),
    )


# ---------------------------------------------------------------------------
# The check: `cutover rebuild`
# ---------------------------------------------------------------------------


@dataclass
class RebuildDeps:
    """What the check needs from the outside world; faked in unit tests.

    Attributes:
        live_uri: The live graph's bolt URI, a guard VALUE only.
        wipe_staging: Empties the staging graph before each build.
        build_regenerator: A fresh `LogRegenerator` over the staging graph.
        staging_form: The staging graph's canonical form, read after a build.
        live_form: The live graph's canonical form (read-only), for the
            self-model gate and the information-only live-vs-rebuilt diff.
        close: Releases connections.
    """

    live_uri: str
    wipe_staging: Callable[[], None]
    build_regenerator: Callable[[], Any]
    staging_form: Callable[[], str]
    live_form: Callable[[], str]
    close: Callable[[], None] = field(default=lambda: None)


@dataclass
class CheckResult:
    """The outcome of `check_cutover`."""

    exit_code: int
    report: dict[str, Any] = field(default_factory=dict)


def _in_scope_event_ids(store: BacklogStore, cutover: Cutover) -> list[str]:
    """The turns `LogRegenerator.rebuild` selects for this candidate, in replay order.

    The same call `rebuild` makes (`get_all_turns_for_reextraction` with the
    epoch's `ontology_version` and `CANONICAL_ORIGINS`), so its last element is
    the last turn the rebuild replays.
    """
    from backend.knowledge.regeneration.log_regenerator import CANONICAL_ORIGINS

    turns = store.event_store.get_all_turns_for_reextraction(
        ontology_version=cutover.ontology_version, origins=CANONICAL_ORIGINS
    )
    return [str(t["event_id"]) for t in turns]


class _ScopeChangedError(MistError):
    """A turn entered the rebuild's selection while the check ran."""


async def check_cutover(
    store: BacklogStore,
    deps_factory: Callable[[Cutover], RebuildDeps],
    *,
    staging_uri: str,
    min_seed_nodes: int,
    expect_turns: int,
    min_replay_edges: int,
    now_iso: str,
    gates: ModuleType | None = None,
) -> CheckResult:
    """Build the staging graph twice under the candidate and gate it.

    Gates, all from `backend.knowledge.regeneration.rebuild_gate` (none
    re-implemented): rebuild-twice identical, turns processed == expected (both
    builds), canonical form non-vacuous, replay-derived edges non-vacuous, and
    the self-model applied (present on both sides and equal). `live == rebuilt`
    is NOT gated -- a different model is expected to produce a different graph
    -- and its summary line is recorded as information.

    Every refusal `assert_rebuild_target_not_live` makes stands: it runs here
    before any staging write, and again inside every `LogRegenerator.rebuild`.
    `deps_factory` (which opens the Neo4j connections) is called only after the
    cutover's state and coverage have been checked.

    One check beyond the gates: the rebuild's turn selection is read before and
    after the two builds and must be unchanged and equal to what each build
    processed, because `rebuilt_through_event_id` (the last turn replayed) is
    what promotion marks applied through. A turn logged into the selection
    mid-check would otherwise be in the graph but not marked, and be applied a
    second time after promotion.

    On a pass the cutover goes to 'checked' with the report, the second build's
    job id and `rebuilt_through_event_id`. On any failure or in-run refusal it
    goes (or stays) 'ready' with the report and those two columns cleared.

    Returns:
        The exit code (EXIT_*) and the report.
    """
    from backend.knowledge.eval_isolation import (
        RebuildTargetError,
        assert_rebuild_target_not_live,
    )
    from backend.knowledge.regeneration import rebuild_gate
    from backend.knowledge.regeneration.log_regenerator import (
        ColdCacheError,
        RebuildScopeError,
    )
    from backend.knowledge.regeneration.rebuild_gate import (
        RebuildDeterminismError,
        RebuildVacuityError,
    )

    gate = gates if gates is not None else rebuild_gate
    cutover = store.open_cutover()
    if cutover is None:
        return CheckResult(EXIT_REFUSED, {"refused": "no cutover is open"})
    if cutover.state not in ("ready", "checked"):
        return CheckResult(
            EXIT_REFUSED,
            {"refused": f"cutover {cutover.cutover_id} is {cutover.state!r}; wait for 'ready'"},
        )
    fill = store.fill_scan(cutover)
    if fill.head is not None:
        return CheckResult(
            EXIT_REFUSED,
            {
                "refused": (
                    f"the candidate covers {fill.covered} of {fill.total} logged turns; "
                    "wait until the dispatcher has filled the rest"
                )
            },
        )
    missing = [
        name
        for name, value in (
            ("--min-seed-nodes", min_seed_nodes),
            ("--expect-turns", expect_turns),
            ("--min-replay-edges", min_replay_edges),
        )
        if value < 1
    ]
    if missing:
        return CheckResult(
            EXIT_REFUSED,
            {"refused": f"{', '.join(missing)} must be >= 1, sized from the corpus"},
        )

    # A re-check wipes the staging graph a previous pass certified, so the
    # previous 'checked' must not survive it: demote first. A check that then
    # dies part way (a Neo4j error propagates from here) leaves 'ready', and
    # promotion refuses, instead of 'checked' over a half-built staging graph.
    if cutover.state == "checked":
        if not store.transition_cutover(
            cutover,
            from_states=("checked",),
            to_state="ready",
            updated_at=now_iso,
            check_report={"in_progress": True, "started_at": now_iso},
            write_check=True,
        ):
            return CheckResult(EXIT_REFUSED, {"refused": "the cutover changed state; re-run"})
        log_transition(cutover, "checked", "ready", reason="re-check started")

    report: dict[str, Any] = {
        "cutover_id": cutover.cutover_id,
        "checked_at": now_iso,
        "staging_uri": staging_uri,
        "expect_turns": expect_turns,
        "min_replay_edges": min_replay_edges,
        "min_seed_nodes": min_seed_nodes,
        "gates": {},
        "passed": False,
    }
    exit_code = EXIT_REFUSED
    rebuilt_through: str | None = None
    job_id: str | None = None
    deps: RebuildDeps | None = None
    try:
        deps = deps_factory(cutover)
        assert_rebuild_target_not_live(staging_uri, deps.live_uri)
        epoch = cutover.epoch_dict()
        scope_before = _in_scope_event_ids(store, cutover)
        builds: list[tuple[str, Any]] = []
        for _ in range(2):
            deps.wipe_staging()
            regenerator = deps.build_regenerator()
            rebuild_report = await regenerator.rebuild(
                staging_uri=staging_uri,
                live_uri=deps.live_uri,
                epoch=epoch,
                min_seed_nodes=min_seed_nodes,
            )
            builds.append((deps.staging_form(), rebuild_report))
        scope_after = _in_scope_event_ids(store, cutover)
        (form_a, report_a), (form_b, report_b) = builds
        report["turns_processed"] = [report_a.turns_processed, report_b.turns_processed]
        report["total_logged"] = report_b.total_logged
        report["ontology_version"] = report_b.ontology_version
        report["origins"] = list(report_b.origins)
        report["seed_nodes_written"] = report_b.seed_nodes_written

        if scope_before != scope_after or any(
            r.turns_processed != len(scope_after) for r in (report_a, report_b)
        ):
            raise _ScopeChangedError(
                f"the rebuild's turn selection changed during the check ({len(scope_before)} "
                f"-> {len(scope_after)} turns; builds processed "
                f"{report_a.turns_processed} and {report_b.turns_processed}); re-run it"
            )
        report["gates"]["scope_stable"] = "passed"

        gate.assert_rebuild_twice_identical(form_a, form_b)
        report["gates"]["rebuild_twice_identical"] = "passed"
        for label, form, rebuild_report in (
            ("rebuild_1", form_a, report_a),
            ("rebuild_2", form_b, report_b),
        ):
            gate.assert_turns_processed(
                processed=rebuild_report.turns_processed, expected=expect_turns
            )
            gate.assert_canonical_form_non_vacuous(form, minimum_nodes=1)
            gate.assert_replay_derived_non_vacuous(form, minimum_edges=min_replay_edges)
            report["gates"][f"non_vacuous_{label}"] = "passed"

        live_form = deps.live_form()
        # Information only: a new model is expected to differ from live.
        report["live_vs_rebuilt"] = gate.live_vs_rebuilt_report(live_form, form_b).splitlines()[0]
        gate.assert_self_model_applied(live_form, form_b)
        report["gates"]["self_model_applied"] = "passed"

        rebuilt_through = scope_after[-1] if scope_after else None
        job_id = str(report_b.job_id)
        report["rebuilt_through_event_id"] = rebuilt_through
        report["rebuild_job_id"] = job_id
        report["passed"] = True
        exit_code = EXIT_OK
    except (
        RebuildTargetError,
        ColdCacheError,
        RebuildScopeError,
        CutoverRefusedError,
        _ScopeChangedError,
    ) as exc:
        exit_code = EXIT_REFUSED
        report["failure"] = _failure(exc)
    except RebuildDeterminismError as exc:
        exit_code = EXIT_NONDETERMINISTIC
        report["failure"] = _failure(exc)
    except RebuildVacuityError as exc:
        exit_code = EXIT_VACUOUS
        report["failure"] = _failure(exc)
    finally:
        if deps is not None:
            deps.close()

    report["exit_code"] = exit_code
    to_state = "checked" if exit_code == EXIT_OK else "ready"
    if not store.transition_cutover(
        cutover,
        from_states=("ready",),
        to_state=to_state,
        updated_at=now_iso,
        rebuild_job_id=job_id,
        rebuilt_through_event_id=rebuilt_through,
        check_report=report,
        write_check=True,
    ):
        report["recorded"] = False
        logger.warning(
            "cutover %d changed state during the check; its report was not recorded",
            cutover.cutover_id,
        )
        return CheckResult(EXIT_REFUSED, report)
    report["recorded"] = True
    log_transition(
        cutover,
        "ready",
        to_state,
        passed=report["passed"],
        exit_code=exit_code,
        rebuilt_through_event_id=rebuilt_through,
    )
    return CheckResult(exit_code, report)


def _failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= _FAILURE_TEXT_LIMIT else text[:_FAILURE_TEXT_LIMIT] + " [truncated]"


def format_report(report: dict[str, Any]) -> str:
    """The check report as indented JSON, for the CLI."""
    return json.dumps(report, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Real wiring (needs Neo4j; not exercised by the unit tier)
# ---------------------------------------------------------------------------


def build_rebuild_deps_from_env(
    store: BacklogStore, staging_uri: str, cutover: Cutover
) -> RebuildDeps:
    """Wire the real staging rebuild, as `scripts/mist_admin.py::_build_log_regenerator` does.

    Same construction, from the same backend factories (nothing is imported
    from `scripts/`), with two deliberate differences:

    - the curation pipeline is built from a config whose stamps are the
      CANDIDATE's (`dataclasses.replace` of `ontology_version`,
      `extraction_version` and the bare `model_hash`), so the staging graph's
      edges carry the stamps of the epoch it will become. The composed value is
      re-checked against the cutover row and refused if the embedding model has
      changed since `begin`;
    - the replay source is this admin's own event store and cache
      (`store.event_store`, `store.cache`); the regenerator writes no row to
      either (`NullRebuildJournal`).

    `assert_rebuild_target_not_live` runs before any connection is opened.

    Raises:
        CutoverRefusedError: The composed candidate hash no longer matches.
        RebuildTargetError: The staging URI is not an allowlisted staging
            endpoint, or resolves to live.
    """
    from dataclasses import replace
    from pathlib import Path

    from backend.factories import build_curation_pipeline
    from backend.knowledge.canonical_serialize import canonical_graph_form
    from backend.knowledge.config import Neo4jConfig, get_config
    from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
    from backend.knowledge.eval_isolation import assert_rebuild_target_not_live
    from backend.knowledge.extraction.confidence import ConfidenceScorer
    from backend.knowledge.extraction.normalizer import EntityNormalizer
    from backend.knowledge.extraction.temporal import TemporalResolver
    from backend.knowledge.extraction.validator import ExtractionValidator
    from backend.knowledge.regeneration.log_regenerator import LogRegenerator
    from backend.knowledge.regeneration.rebuild_journal import NullRebuildJournal
    from backend.knowledge.regeneration.staging_seeder import StagingSeeder
    from backend.knowledge.seed.loader import load_seed_documents
    from backend.knowledge.storage.graph_executor import GraphExecutor
    from backend.knowledge.storage.graph_store import GraphStore
    from backend.knowledge.storage.neo4j_connection import Neo4jConnection
    from backend.knowledge.version_stamps import compose_model_hash

    config = get_config()
    candidate_config = replace(
        config,
        ontology_version=cutover.ontology_version,
        extraction_version=cutover.extraction_version,
        model_hash=cutover.bare_model_hash,
    )
    if compose_model_hash(candidate_config) != cutover.model_hash:
        raise CutoverRefusedError(
            f"the candidate's composed model hash is {cutover.model_hash!r}, but this "
            f"environment composes {compose_model_hash(candidate_config)!r} (the embedding "
            "model changed since `cutover begin`); abandon and begin again"
        )

    live_uri = config.neo4j.uri
    assert_rebuild_target_not_live(staging_uri, live_uri)

    seed_documents = load_seed_documents(Path(config.vault.root) / "seed")
    embedding_provider = EmbeddingGenerator(config.embedding.model_name)

    staging_conn = Neo4jConnection(
        Neo4jConfig(uri=staging_uri, username=config.neo4j.username, password=config.neo4j.password)
    )
    staging_conn.connect()
    live_conn = Neo4jConnection(config.neo4j)  # read only: canonical_graph_form
    try:
        live_conn.connect()
    except MistError:
        staging_conn.disconnect()
        raise

    def build_regenerator() -> LogRegenerator:
        GraphStore(
            connection=staging_conn, embedding_generator=embedding_provider
        ).initialize_schema()
        pipeline = build_curation_pipeline(
            candidate_config, GraphExecutor(staging_conn), embedding_provider=embedding_provider
        )
        return LogRegenerator(
            event_store=store.event_store,
            extraction_cache=store.cache,
            staging_curation_pipeline=pipeline,
            confidence_scorer=ConfidenceScorer(),
            temporal_resolver=TemporalResolver(),
            normalizer=EntityNormalizer(embedding_generator=None, executor=None),
            validator=ExtractionValidator(
                min_confidence=config.extraction.min_extraction_confidence
            ),
            journal=NullRebuildJournal(),
            staging_seeder=StagingSeeder(
                connection=staging_conn,
                documents=seed_documents,
                seed_version=seed_documents[0].seed_version,
                embedding_generator=embedding_provider,
                expected_dimension=config.embedding.dimension,
            ),
        )

    def close() -> None:
        staging_conn.disconnect()
        live_conn.disconnect()

    return RebuildDeps(
        live_uri=live_uri,
        wipe_staging=lambda: staging_conn.execute_write("MATCH (n) DETACH DELETE n", {}),
        build_regenerator=build_regenerator,
        staging_form=lambda: canonical_graph_form(
            staging_conn, include_provenance=True, include_self_model=True
        ),
        live_form=lambda: canonical_graph_form(
            live_conn, include_provenance=True, include_self_model=True
        ),
        close=close,
    )
