"""Operator CLI for the extraction backlog and the epoch cutover.

    python -m backend.extraction_backlog.admin status
    python -m backend.extraction_backlog.admin retry-dead-letters [--event-id ID]
    python -m backend.extraction_backlog.admin cutover begin --model-hash BARE \
        [--extraction-version V] [--ontology-version O] [--settings-change [--refill]]
    python -m backend.extraction_backlog.admin cutover status
    python -m backend.extraction_backlog.admin cutover rebuild --staging-uri URI \
        --min-seed-nodes N --expect-turns N --min-replay-edges N
    python -m backend.extraction_backlog.admin cutover probe
    python -m backend.extraction_backlog.admin cutover promote --graph-swapped
    python -m backend.extraction_backlog.admin cutover promote --seed-only-graph
    python -m backend.extraction_backlog.admin cutover abandon
    python -m backend.extraction_backlog.admin redispatch-check --event-id ID
    python -m backend.extraction_backlog.admin epoch accept --serving-identity SI \
        --dispatch-identity DI [--cutover] --note TEXT

`status` reads the event store and extraction cache directly (it does not call
the extraction service or need the backend running) and prints the active
epoch's stamps, the backlog for it, including `legacy_unextracted` (turns
logged before the backlog first activated that had no cache row; never
dispatched, and covered by the next cutover's re-extraction), and any open
cutover with its candidate stamps. It also prints this code's
`ONTOLOGY_VERSION` and `EXTRACTION_VERSION`, the configured `MIST_MODEL_HASH`,
and `writer stamps: match` or `writer stamps: MISMATCH (<field>, ...)`: whether
the stamps a backend started from this environment would write
(`factories.writer_stamps_from_config`) equal the active epoch's, judged by
the dispatcher's own guard (`ExtractionDispatcher._writer_stamp_mismatch`).
For the active epoch and the open cutover it prints the binding
(`binding.py`): its source and rule, the serving and dispatch identities, the
cache namespace (3-part, or keyed with the key), and the setting names the
binding could not verify, or `binding: unbound`; and an operator acceptance on
record for the target.

`cutover begin --settings-change` opens a candidate with the active epoch's
stamps, for a change of a serving setting the extraction service keys or of a
backend dispatch setting; without the flag such a candidate is refused. While
the active epoch is unbound it also needs `--refill` (the change cannot be
checked against an identity the epoch does not have yet, so the candidate
re-extracts the log), and an ontology-only candidate (the active extraction
version and model hash) is refused. Both refusals exit 2.

`epoch accept` records an operator acceptance for the active epoch, or with
`--cutover` the open candidate (not a settings-change one: the 3-part rows of
its stamps are its source's). The dispatcher binds the target with source
'operator' at its next contact while the live identities equal the accepted
ones. Exit 2 with no target, a malformed identity, an empty note, a bound
target, or `--cutover` on a settings-change candidate.

`cutover probe` runs the seed-only checks read-only (`cutover.check_seed_only`):
the conversation log must be empty and the live-graph probe must pass. It
prints each result and every violation, and writes nothing to the event store,
the cache or the graph. It works with or without an open cutover.

`retry-dead-letters` puts dead-lettered turns (`extraction_failed` skips) back
into the backlog: it restarts the turn's failure count and deletes the skip row
for the active epoch. It leaves the turn's applied marker alone: a turn with no
cache row is inference-pending whatever its marker says, and the dispatcher
clears the stale marker itself before re-caching, so the CLI can run beside a
running dispatcher at any moment (`BacklogStore.retry_dead_letter` explains
the race this avoids). A running dispatcher picks the turn up on its next scan
(at most `MIST_EXTRACTION_IDLE_POLL_S` later); `retry-dead-letters` does not
wake it.

A retried turn is applied OUT OF LOG ORDER. Every turn logged after it has
already been applied, and graph outcomes that depend on order (which duplicate
wins dedup, which belief supersedes which) can then differ from a rebuild of
the same log until the next epoch cutover re-extracts it. The command says so
when it runs.

`cutover ...` drives the epoch cutover (`cutover.py`; runbook in `CUTOVER.md`).
`cutover promote` takes `--graph-swapped` or `--seed-only-graph`, never both
(argparse refuses the pair with exit 2 before anything is opened).
Exit codes: 0 done, 2 refused (nothing changed). `status` exits 1 when the
epoch ledger is empty. `cutover rebuild` also exits 1 (rebuild-twice disagreed)
or 4 (a non-vacuity or self-model gate failed), and records its report on the
cutover either way. `cutover probe` exits 0 when the log is empty and the
probe passes, 2 when either refuses, and 1 when the probe could not run (Neo4j
unreachable, or any `GraphProbeError`), whatever the log check found.

`redispatch-check --event-id ID` (FE-025) re-applies one applied turn from its
cache row to the LIVE graph, with the backend stopped, and compares
whole-graph fingerprints before and after; it writes no marker and no cache
row (`redispatch.py` has the operator command and the details). Exit 0 the
fingerprints are identical, 1 it could not run (Neo4j unreachable, the apply
raised, or curation stage errors), 2 refused with nothing applied (no active
epoch, no `applied` marker for the turn, no cache row under the active epoch,
a skip row, writer stamps differ from the epoch, the backend is not
positively down, or the pipeline's dispatch identity differs from the bound
epoch's), 3 the fingerprints differ.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TextIO

from backend.event_store.store import EventStore
from backend.knowledge.extraction_cache import ExtractionCache

from .store import BacklogStore, Cutover, Epoch

if TYPE_CHECKING:
    from backend.knowledge.config import KnowledgeConfig
    from backend.knowledge.curation.graph_writer import RebuildStamps

OUT_OF_ORDER_WARNING = (
    "[WARNING] A retried turn is applied OUT OF LOG ORDER: turns logged after it are "
    "already in the graph. Order-dependent results (dedup winners, supersession) may "
    "differ from a rebuild until the next epoch cutover re-extracts the log."
)


def build_store_from_env() -> BacklogStore:
    """Open the production event store and extraction cache named by the environment."""
    from backend.factories import production_cache_path
    from backend.knowledge.config import KnowledgeConfig

    config = KnowledgeConfig.from_env()
    event_store = EventStore(db_path=config.event_store.db_path)
    event_store.initialize()
    cache = ExtractionCache(production_cache_path(config))
    cache.initialize()
    return BacklogStore(event_store, cache)


def _embedding_model_from_env() -> str:
    from backend.knowledge.config import KnowledgeConfig

    return KnowledgeConfig.from_env().embedding.model_name


def _binding_rule_and_unverified(binding: dict[str, Any]) -> tuple[str, list[str]]:
    """The rule that bound a target and the setting names it left unverified.

    A promoted epoch's evidence nests the candidate's (`EventStore` promotion
    copies the cutover binding), and what was verified is the candidate's.
    """
    evidence = binding.get("evidence") or {}
    nested = evidence.get("cutover_binding_evidence")
    if binding.get("source") == "promoted" and isinstance(nested, dict):
        evidence = nested
    return str(evidence.get("rule") or "-"), [str(n) for n in evidence.get("unverified") or []]


def _print_binding(
    binding: dict[str, Any] | None, acceptance: dict[str, Any] | None, indent: str, out: TextIO
) -> None:
    """A target's binding: source, rule, both identities, the namespace, the unverified names.

    Then an operator acceptance on record for it, if any: the dispatcher binds
    with it (source 'operator') only while the live identities equal its.
    """
    if binding is None:
        print(f"{indent}binding: unbound", file=out)
    else:
        rule, unverified = _binding_rule_and_unverified(binding)
        namespace = binding.get("identity_namespace")
        shown = "3-part" if namespace is None else f"keyed ({namespace})"
        print(
            f"{indent}binding: source={binding['source']} rule={rule} namespace={shown}",
            file=out,
        )
        print(f"{indent}  serving_identity={binding['serving_identity']}", file=out)
        print(f"{indent}  dispatch_identity={binding['dispatch_identity']}", file=out)
        print(f"{indent}  bound_at={binding['bound_at']}", file=out)
        print(
            f"{indent}  unverified ({len(unverified)}): {', '.join(unverified) or '-'}",
            file=out,
        )
    if acceptance is not None:
        print(
            f"{indent}acceptance: serving_identity={acceptance['serving_identity']} "
            f"dispatch_identity={acceptance['dispatch_identity']} "
            f"(accepted {acceptance['accepted_at']}): {acceptance['note']}",
            file=out,
        )


def _print_cutover(store: BacklogStore, cutover: Cutover, out: TextIO) -> None:
    fill = store.fill_scan(cutover)
    print(
        f"cutover {cutover.cutover_id}: state={cutover.state} "
        f"(from epoch {cutover.source_epoch_id}, requested {cutover.requested_at})",
        file=out,
    )
    print(
        f"  candidate: ontology_version={cutover.ontology_version} "
        f"extraction_version={cutover.extraction_version} model_hash={cutover.model_hash} "
        f"(service model_hash={cutover.bare_model_hash})",
        file=out,
    )
    print(f"  covered={fill.covered} total={fill.total}", file=out)
    events = store.event_store
    _print_binding(
        events.get_cutover_identity(cutover.cutover_id),
        events.get_identity_acceptance(-cutover.cutover_id),
        "  ",
        out,
    )
    if cutover.rebuilt_through_event_id:
        print(
            f"  rebuilt_through_event_id={cutover.rebuilt_through_event_id} "
            f"rebuild_job_id={cutover.rebuild_job_id}",
            file=out,
        )
    if cutover.check_report is not None:
        report = cutover.check_report
        print(
            f"  last check: passed={report.get('passed')} exit_code={report.get('exit_code')} "
            f"at {report.get('checked_at') or report.get('started_at')}",
            file=out,
        )
        if report.get("failure"):
            print(f"  failure: {str(report['failure']).splitlines()[0]}", file=out)


_STAMP_FIELDS = ("ontology_version", "extraction_version", "model_hash")


def _writer_stamps_verdict(writer: RebuildStamps, epoch: Epoch) -> tuple[str, str | None]:
    """The `writer stamps: ...` line for `status`, and the guard's reason on a mismatch.

    Match or mismatch is decided by the dispatcher's own guard,
    `ExtractionDispatcher._writer_stamp_mismatch`, not a second comparison: it
    reads nothing of the dispatcher but `_writer_stamps`, so it is called on a
    stand-in holding just that. The field names listed on a mismatch are for
    the operator's eye only; a mismatch the guard reports with no differing
    field (not possible today) is still printed as MISMATCH.
    """
    from .dispatcher import ExtractionDispatcher

    reason = ExtractionDispatcher._writer_stamp_mismatch(
        SimpleNamespace(_writer_stamps=writer), epoch  # type: ignore[arg-type]
    )
    if reason is None:
        return "writer stamps: match", None
    differing = [name for name in _STAMP_FIELDS if getattr(writer, name) != getattr(epoch, name)]
    return f"writer stamps: MISMATCH ({', '.join(differing) or 'see reason'})", reason


def _status(store: BacklogStore, config: KnowledgeConfig, out: TextIO) -> int:
    from backend.factories import writer_stamps_from_config
    from backend.knowledge.version_stamps import EXTRACTION_VERSION, ONTOLOGY_VERSION

    epoch = store.active_epoch()
    if epoch is None:
        print("No epoch in the ledger; nothing to report.", file=out)
        return 1
    scan = store.scan(epoch)
    activation = store.get_activation(epoch)
    writer = writer_stamps_from_config(config)
    print(
        f"epoch {epoch.epoch_id}: ontology_version={epoch.ontology_version} "
        f"extraction_version={epoch.extraction_version} model_hash={epoch.model_hash}",
        file=out,
    )
    print(
        f"code: ONTOLOGY_VERSION={ONTOLOGY_VERSION} EXTRACTION_VERSION={EXTRACTION_VERSION} "
        f"configured model_hash={config.model_hash} (MIST_MODEL_HASH; composed "
        f"{writer.model_hash})",
        file=out,
    )
    verdict, reason = _writer_stamps_verdict(writer, epoch)
    print(verdict, file=out)
    if reason is not None:
        print(f"  {reason}", file=out)
    events = store.event_store
    _print_binding(
        events.get_epoch_identity(epoch.epoch_id),
        events.get_identity_acceptance(epoch.epoch_id),
        "",
        out,
    )
    if activation is None:
        print("activation: not yet activated (no dispatcher has run for this epoch)", file=out)
    else:
        print(
            f"activation: {activation.activated_at} "
            f"(turns_at_activation={activation.turns_at_activation}, "
            f"marked_applied={activation.marked_applied}, "
            f"legacy_unextracted={activation.legacy_unextracted})",
            file=out,
        )
    print(f"backlog_depth={scan.backlog_depth}", file=out)
    print(f"apply_pending={scan.apply_pending}", file=out)
    print(f"dead_lettered={scan.dead_lettered}", file=out)
    print(f"legacy_unextracted={scan.legacy_unextracted}", file=out)
    print(f"oldest_pending={scan.oldest_pending_timestamp or '-'}", file=out)
    for event_id in store.list_dead_letters(epoch):
        print(f"dead_letter event_id={event_id}", file=out)
    cutover = store.open_cutover()
    if cutover is not None:
        _print_cutover(store, cutover, out)
    return 0


def _retry(store: BacklogStore, event_id: str | None, out: TextIO) -> int:
    epoch = store.active_epoch()
    if epoch is None:
        print("No epoch in the ledger; nothing to retry.", file=out)
        return 1
    targets = [event_id] if event_id is not None else store.list_dead_letters(epoch)
    if not targets:
        print("No dead-lettered turns.", file=out)
        return 0
    retried = 0
    for target in targets:
        if store.retry_dead_letter(target, epoch):
            retried += 1
            print(f"retried event_id={target}", file=out)
        else:
            print(f"skipped event_id={target} (not dead-lettered under the active epoch)", file=out)
    if retried:
        print(OUT_OF_ORDER_WARNING, file=out)
    return 0 if retried == len(targets) else 2


# ---------------------------------------------------------------------------
# cutover
# ---------------------------------------------------------------------------


def _cutover_begin(
    store: BacklogStore, args: argparse.Namespace, embedding_model_name: str, now_iso: str, out
) -> int:
    from backend.knowledge.version_stamps import EXTRACTION_VERSION, ONTOLOGY_VERSION

    from .cutover import CutoverRefusedError, begin_cutover

    try:
        cutover = begin_cutover(
            store,
            bare_model_hash=args.model_hash,
            extraction_version=args.extraction_version or EXTRACTION_VERSION,
            ontology_version=args.ontology_version or ONTOLOGY_VERSION,
            embedding_model_name=embedding_model_name,
            now_iso=now_iso,
            settings_change=args.settings_change,
            refill=args.refill,
        )
    except CutoverRefusedError as exc:
        print(f"[cutover] REFUSED: {exc}", file=out)
        return 2
    print(f"[cutover] began cutover {cutover.cutover_id} (state=filling)", file=out)
    _print_cutover(store, cutover, out)
    if args.settings_change:
        print(
            "[cutover] Next: restart the extraction service (or the backend, for a dispatch "
            "setting) with the candidate settings. The dispatcher binds the candidate at its "
            "first contact with the changed service and fills it; while the service still "
            "reports the active epoch's identity it stalls in epoch_mismatch and fills nothing.",
            file=out,
        )
        return 0
    print(
        "[cutover] Next: point the extraction service at the candidate model (its /v1/info "
        f"must report extraction_version={cutover.extraction_version} and "
        f"model_hash={cutover.bare_model_hash}); the dispatcher fills while it matches.",
        file=out,
    )
    return 0


def _cutover_status(store: BacklogStore, out: TextIO) -> int:
    cutover = store.open_cutover()
    if cutover is not None:
        _print_cutover(store, cutover, out)
        return 0
    history = store.list_cutovers()
    if not history:
        print("No cutover has been started.", file=out)
        return 0
    last = history[-1]
    print(
        f"No cutover is open. Last: cutover {last.cutover_id} state={last.state} "
        f"(updated {last.updated_at}, promoted_epoch_id={last.promoted_epoch_id})",
        file=out,
    )
    return 0


def _cutover_abandon(store: BacklogStore, now_iso: str, out: TextIO) -> int:
    from .cutover import CutoverRefusedError, abandon_cutover

    try:
        cutover = abandon_cutover(store, now_iso=now_iso)
    except CutoverRefusedError as exc:
        print(f"[cutover] REFUSED: {exc}", file=out)
        return 2
    print(
        f"[cutover] abandoned cutover {cutover.cutover_id}; nothing was deleted. Point the "
        "extraction service back at the active epoch's model.",
        file=out,
    )
    return 0


def _cutover_promote(
    store: BacklogStore,
    args: argparse.Namespace,
    graph_probe: Callable | None,
    now_iso: str,
    out: TextIO,
) -> int:
    from backend.knowledge.version_stamps import EXTRACTION_VERSION

    from .cutover import (
        CutoverRefusedError,
        probe_live_graph_from_env,
        promote_cutover,
        promote_seed_only_cutover,
    )

    try:
        if args.seed_only_graph:
            promotion = promote_seed_only_cutover(
                store,
                graph_probe=graph_probe if graph_probe is not None else probe_live_graph_from_env,
                now_iso=now_iso,
            )
        else:
            promotion = promote_cutover(
                store, graph_swapped=bool(args.graph_swapped), now_iso=now_iso
            )
    except CutoverRefusedError as exc:
        print(f"[cutover] REFUSED: {exc}", file=out)
        return 2
    cutover = promotion.cutover
    if promotion.graph_probe is not None:
        probe = promotion.graph_probe
        print(
            f"[cutover] promoted cutover {cutover.cutover_id} to epoch {promotion.epoch_id} "
            f"over a seed-only live graph ({probe.node_count} node(s), "
            f"{probe.relationship_count} relationship(s), all seed-stamped) and an empty "
            "conversation log: no turn was marked applied. The dispatcher extracts and "
            "applies turns logged from now on, in log order, under the new epoch.",
            file=out,
        )
    else:
        print(
            f"[cutover] promoted cutover {cutover.cutover_id} to epoch {promotion.epoch_id}: "
            f"{promotion.marked_applied} of {promotion.turns_at_activation} logged turn(s) "
            f"marked applied through {cutover.rebuilt_through_event_id}; later turns stay "
            "apply-pending for the dispatcher.",
            file=out,
        )
    print(
        f"[cutover] Before starting the backend, set MIST_MODEL_HASH={cutover.bare_model_hash} "
        "in its environment: live graph writes are stamped from KnowledgeConfig, not from "
        "the epoch ledger.",
        file=out,
    )
    if cutover.extraction_version != EXTRACTION_VERSION:
        print(
            f"[WARNING] The new epoch's extraction_version is {cutover.extraction_version!r}, "
            f"but this code's EXTRACTION_VERSION is {EXTRACTION_VERSION!r}; live writes will "
            "be stamped with the code's value.",
            file=out,
        )
    return 0


def _cutover_probe(store: BacklogStore, graph_probe: Callable | None, out: TextIO) -> int:
    """`cutover probe`: the seed-only checks, read-only. Exit 0 pass, 2 refused, 1 not run."""
    from .cutover import check_seed_only, probe_live_graph_from_env

    check = check_seed_only(
        store, graph_probe=graph_probe if graph_probe is not None else probe_live_graph_from_env
    )
    if check.log_violation is None:
        print("[probe] conversation log: PASS (0 logged turns)", file=out)
    else:
        print(f"[probe] conversation log: REFUSED ({check.logged_turns} logged turn(s))", file=out)
        print(f"  violation: {check.log_violation}", file=out)
    if check.graph_probe is None:
        print(f"[probe] live graph: NOT RUN: {check.probe_error}", file=out)
    else:
        counts = " ".join(f"{name}={n}" for name, n in check.graph_probe.as_dict().items())
        verdict = "PASS" if not check.graph_violations else "REFUSED"
        print(f"[probe] live graph: {verdict} ({counts})", file=out)
        for violation in check.graph_violations:
            print(f"  violation: {violation}", file=out)
    if check.graph_probe is None:
        print("[probe] could not complete: the live graph probe did not run (exit 1)", file=out)
        return 1
    if check.log_violation is not None or check.graph_violations:
        print("[probe] seed-only promotion would be REFUSED (exit 2)", file=out)
        return 2
    print(
        "[probe] both checks pass (exit 0). The probe cannot see every write; the operator "
        "precondition in CUTOVER.md 6A still applies.",
        file=out,
    )
    return 0


def _cutover_rebuild(
    store: BacklogStore,
    args: argparse.Namespace,
    deps_factory: Callable,
    now_iso: str,
    out: TextIO,
) -> int:
    from .cutover import check_cutover, format_report

    result = asyncio.run(
        check_cutover(
            store,
            deps_factory,
            staging_uri=args.staging_uri,
            min_seed_nodes=args.min_seed_nodes,
            expect_turns=args.expect_turns,
            min_replay_edges=args.min_replay_edges,
            now_iso=now_iso,
        )
    )
    print(format_report(result.report), file=out)
    verdict = "PASSED -> checked" if result.exit_code == 0 else "did not pass"
    print(f"[cutover] rebuild check {verdict} (exit {result.exit_code})", file=out)
    return result.exit_code


# ---------------------------------------------------------------------------
# epoch accept
# ---------------------------------------------------------------------------


def _well_formed(identity: str, prefix: str) -> bool:
    """`prefix` and 64 lower-case hex digits: the form both identities are computed in."""
    digest = identity[len(prefix) :]
    return (
        identity.startswith(prefix)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest)
    )


def _epoch_accept(store: BacklogStore, args: argparse.Namespace, now_iso: str, out) -> int:
    """`epoch accept`: record the operator's acceptance of a live identity for one target.

    The target is the active epoch, or with `--cutover` the open candidate.
    Nothing is bound here: the dispatcher binds the target with source
    'operator' (rule 2) at its next contact, and only while the live serving
    and dispatch identities equal the accepted ones, so an acceptance cannot
    bind a configuration that is not running.
    """
    from backend.knowledge.extraction.serving_identity import SERVING_IDENTITY_PREFIX

    from .dispatch_identity import DISPATCH_IDENTITY_PREFIX

    def refuse(reason: str) -> int:
        print(f"[epoch] REFUSED: {reason}", file=out)
        return 2

    for flag, value, prefix in (
        ("--serving-identity", args.serving_identity, SERVING_IDENTITY_PREFIX),
        ("--dispatch-identity", args.dispatch_identity, DISPATCH_IDENTITY_PREFIX),
    ):
        if not _well_formed(value, prefix):
            return refuse(
                f"{flag} {value!r} is malformed: expected {prefix} and 64 lower-case hex "
                "digits, as the stall reason or admin status prints it"
            )
    note = args.note.strip()
    if not note:
        return refuse("--note is empty; say why this identity is accepted")

    if args.cutover:
        cutover = store.open_cutover()
        if cutover is None:
            return refuse("no cutover is open; omit --cutover to accept for the active epoch")
        source = (
            store.get_epoch(cutover.source_epoch_id)
            if cutover.source_epoch_id is not None
            else None
        )
        if source is not None and (
            cutover.ontology_version,
            cutover.extraction_version,
            cutover.model_hash,
        ) == (source.ontology_version, source.extraction_version, source.model_hash):
            return refuse(
                f"cutover {cutover.cutover_id} is a settings-change candidate: the 3-part rows "
                f"of its stamps are epoch {source.epoch_id}'s, so an acceptance cannot bind it "
                "there; it binds at its first fill contact with the changed service"
            )
        target_id = -cutover.cutover_id
        label = f"cutover {cutover.cutover_id}"
        bound = cutover.bound
    else:
        epoch = store.active_epoch()
        if epoch is None:
            return refuse("no epoch in the ledger; there is nothing to accept an identity for")
        target_id = epoch.epoch_id
        label = f"epoch {epoch.epoch_id}"
        bound = epoch.bound
    if bound:
        return refuse(
            f"{label} is already bound (admin status shows its binding); a binding is never "
            "moved, so an acceptance for it would never be read"
        )

    store.event_store.accept_identity(
        target_id,
        serving_identity=args.serving_identity,
        dispatch_identity=args.dispatch_identity,
        note=note,
        accepted_at=now_iso,
    )
    print(
        f"[epoch] accepted serving_identity={args.serving_identity} "
        f"dispatch_identity={args.dispatch_identity} for {label}. The dispatcher binds it, "
        "source 'operator', at its next contact while the live identities equal these; "
        "admin status shows the binding.",
        file=out,
    )
    return 0


def _add_epoch_parser(sub) -> None:
    epoch = sub.add_parser("epoch", help="epoch identity: accept a live identity")
    esub = epoch.add_subparsers(dest="epoch_command", required=True)
    accept = esub.add_parser(
        "accept",
        help=(
            "accept the live identities for the active epoch (or with --cutover the open "
            "candidate); the dispatcher binds it with source 'operator' while they hold"
        ),
    )
    accept.add_argument("--serving-identity", required=True, help="si1-..., the service's")
    accept.add_argument("--dispatch-identity", required=True, help="di1-..., the backend's")
    accept.add_argument(
        "--cutover", action="store_true", help="accept for the open cutover's candidate"
    )
    accept.add_argument("--note", required=True, help="why the operator accepts it (recorded)")


def _add_cutover_parser(sub) -> None:
    cutover = sub.add_parser(
        "cutover", help="epoch cutover: begin, status, probe, rebuild, promote, abandon"
    )
    csub = cutover.add_subparsers(dest="cutover_command", required=True)

    begin = csub.add_parser("begin", help="open a candidate epoch; the dispatcher starts filling")
    begin.add_argument(
        "--model-hash",
        required=True,
        help="the candidate model's BARE hash, as the extraction service's /v1/info reports it",
    )
    begin.add_argument(
        "--extraction-version",
        default=None,
        help="default: this code's EXTRACTION_VERSION (backend/knowledge/version_stamps.py)",
    )
    begin.add_argument(
        "--ontology-version",
        default=None,
        help="default: this code's ONTOLOGY_VERSION (backend/knowledge/version_stamps.py)",
    )
    begin.add_argument(
        "--settings-change",
        action="store_true",
        help=(
            "the candidate keeps the active epoch's stamps: a serving setting the extraction "
            "service keys, or a backend dispatch setting, changed (CUTOVER.md, settings "
            "changes)"
        ),
    )
    begin.add_argument(
        "--refill",
        action="store_true",
        help=(
            "with --settings-change while the active epoch is unbound: re-extract the whole "
            "log under the candidate, since the change cannot be checked against the epoch"
        ),
    )

    csub.add_parser("status", help="print the open cutover, or the last one")

    csub.add_parser(
        "probe",
        help=(
            "read-only: run the seed-only checks (empty conversation log, live-graph probe) "
            "and print every violation; exit 0 pass, 2 refused, 1 the probe could not run"
        ),
    )

    rebuild = csub.add_parser(
        "rebuild", help="build the staging graph twice under the candidate and gate it"
    )
    rebuild.add_argument("--staging-uri", required=True, help="disposable staging bolt URI")
    rebuild.add_argument("--min-seed-nodes", type=int, required=True)
    rebuild.add_argument("--expect-turns", type=int, required=True)
    rebuild.add_argument("--min-replay-edges", type=int, required=True)

    promote = csub.add_parser("promote", help="make the candidate the active epoch")
    how = promote.add_mutually_exclusive_group()
    how.add_argument(
        "--graph-swapped",
        action="store_true",
        help="the live graph has been replaced by the checked staging graph",
    )
    how.add_argument(
        "--seed-only-graph",
        action="store_true",
        help=(
            "no extraction has ever run against this live graph, so it holds seed data "
            "only (operator precondition, CUTOVER.md 6A: reseed with the backend stopped, "
            "then `cutover probe`); refused unless the conversation log is empty and a "
            "read-only probe finds no stamped and no unseeded element; promotes a 'ready' "
            "or 'checked' candidate without a swap"
        ),
    )

    csub.add_parser("abandon", help="close the open cutover; deletes nothing")


def main(
    argv: Sequence[str] | None = None,
    *,
    store: BacklogStore | None = None,
    out: TextIO | None = None,
    embedding_model_name: str | None = None,
    rebuild_deps_factory: Callable | None = None,
    graph_probe: Callable | None = None,
    clock: Callable[[], datetime] | None = None,
    knowledge_config: KnowledgeConfig | None = None,
    backend_probe: Callable | None = None,
    redispatch_graph: Callable | None = None,
) -> int:
    """Run the CLI. `store`, `out` and the cutover dependencies are injectable for tests.

    Args:
        embedding_model_name: For `cutover begin`; defaults to the environment's
            `KnowledgeConfig.embedding.model_name`.
        rebuild_deps_factory: For `cutover rebuild`, `Cutover -> RebuildDeps`;
            defaults to the real Neo4j wiring (`build_rebuild_deps_from_env`).
        graph_probe: For `cutover promote --seed-only-graph` and `cutover
            probe`, a `GraphProbe` (`() -> GraphProbeReport`), called at most
            once. `promote` calls it only after the cutover's state, fill,
            markers, source epoch and the empty log have passed; `probe` calls
            it unconditionally. Defaults to the real read-only live-graph probe
            (`probe_live_graph_from_env`).
        clock: Wall clock (tz-aware) for the timestamps cutover rows record.
        knowledge_config: For `status` and `redispatch-check`, the
            configuration whose writer stamps are compared with the active
            epoch (and, for `redispatch-check`, the graph it opens); defaults
            to `KnowledgeConfig.from_env()`, what a backend started from this
            environment reads.
        backend_probe: For `redispatch-check`, `() -> BackendProbe`; defaults
            to `redispatch.probe_backend`, the live `GET /health` probe.
        redispatch_graph: For `redispatch-check`, `() -> RedispatchGraph`,
            called only after every refusal check passes; defaults to
            `redispatch.open_live_graph_from_env(knowledge_config)`.

    Returns:
        Process exit code: 0 success, 1 no epoch, 2 refused or some requested
        turn was not dead-lettered; `cutover rebuild` adds 1 and 4,
        `cutover probe` returns 1 when the probe could not run, and
        `redispatch-check` returns 1 when it could not run and 3 when the
        fingerprints differ (see module docstring).
    """
    parser = argparse.ArgumentParser(prog="python -m backend.extraction_backlog.admin")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="print the backlog for the active epoch")
    retry = sub.add_parser(
        "retry-dead-letters", help="put dead-lettered turns back into the backlog"
    )
    retry.add_argument("--event-id", default=None, help="retry only this turn")
    _add_cutover_parser(sub)
    _add_epoch_parser(sub)
    redispatch = sub.add_parser(
        "redispatch-check",
        help=(
            "FE-025, backend stopped: re-apply one applied turn to the LIVE graph and compare "
            "whole-graph fingerprints; exit 0 identical, 1 could not run, 2 refused, 3 differ"
        ),
    )
    redispatch.add_argument("--event-id", required=True, help="the applied turn's event id")
    args = parser.parse_args(argv)

    stream = out if out is not None else sys.stdout
    backlog = store if store is not None else build_store_from_env()
    now_iso = (clock or (lambda: datetime.now(UTC)))().isoformat()
    if args.command in ("status", "redispatch-check") and knowledge_config is None:
        from backend.knowledge.config import KnowledgeConfig

        knowledge_config = KnowledgeConfig.from_env()
    if args.command == "status":
        return _status(backlog, knowledge_config, stream)
    if args.command == "retry-dead-letters":
        return _retry(backlog, args.event_id, stream)
    if args.command == "redispatch-check":
        return _redispatch_check(
            backlog, args.event_id, knowledge_config, backend_probe, redispatch_graph, stream
        )
    if args.command == "epoch":
        return _epoch_accept(backlog, args, now_iso, stream)

    command = args.cutover_command
    if command == "begin":
        model = embedding_model_name or _embedding_model_from_env()
        return _cutover_begin(backlog, args, model, now_iso, stream)
    if command == "status":
        return _cutover_status(backlog, stream)
    if command == "probe":
        return _cutover_probe(backlog, graph_probe, stream)
    if command == "abandon":
        return _cutover_abandon(backlog, now_iso, stream)
    if command == "promote":
        return _cutover_promote(backlog, args, graph_probe, now_iso, stream)
    # rebuild
    if rebuild_deps_factory is None:
        from .cutover import build_rebuild_deps_from_env

        def rebuild_deps_factory(cutover: Cutover):
            return build_rebuild_deps_from_env(backlog, args.staging_uri, cutover)

    return _cutover_rebuild(backlog, args, rebuild_deps_factory, now_iso, stream)


def _redispatch_check(
    store: BacklogStore,
    event_id: str,
    config: KnowledgeConfig,
    backend_probe: Callable | None,
    redispatch_graph: Callable | None,
    out: TextIO,
) -> int:
    """`redispatch-check`: see `redispatch.run_redispatch_check` for the exit codes."""
    from backend.factories import writer_stamps_from_config

    from .redispatch import open_live_graph_from_env, probe_backend, run_redispatch_check

    def open_graph():
        if redispatch_graph is not None:
            return redispatch_graph()
        return open_live_graph_from_env(config)

    return asyncio.run(
        run_redispatch_check(
            event_id,
            store=store,
            writer_stamps=writer_stamps_from_config(config),
            backend_probe=backend_probe if backend_probe is not None else probe_backend,
            open_graph=open_graph,
            out=out,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
