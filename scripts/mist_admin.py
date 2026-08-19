"""MIST Admin CLI — Tier 1, Tier 2, and Tier 3 operations.

Thin wrapper around backend.knowledge.admin functions and the production DI
composition root in backend.factories so the CLI exercises the same code paths
as the running backend.

Tier 1 subcommands (graph operations):
    seed [--seed-dir DIR]                   Wipe-then-apply the versioned seed
                                             source (mist-memory/seed/*.md)
                                             idempotently (R1.4 spec 2.0/O9).
    seed-verify [--seed-dir DIR]            Run the five gates (facts-present,
                                             node-definitions, containment,
                                             negation-proximity, embeddings)
                                             against the versioned seed source.
    graph-dump [--format json|cypher]       Dump full __Entity__ subgraph.
    graph-stats                             Node/rel counts, confidence, orphans.
    graph-reset [--confirm] [--dry-run]     Wipe graph with safety guards.
    stack-status                            Probe Neo4j + llama-server + backend.

Tier 2 subcommands (atomic pipeline operations):
    extract "<utterance>" [--commit]        Run extraction pipeline. Default
                                             is dry-run (no writes); --commit
                                             includes curation + internal
                                             derivation.
    retrieve "<query>"                      Run hybrid (graph + vector) retrieval
                                             and print facts with scores.

Tier 3 subcommands (end-to-end):
    chat "<message>" [--session-id X]       Full end-to-end turn through the
                                             production ConversationHandler.
                                             Retrieval + LLM + extraction +
                                             graph writes. Per-turn JSONL debug
                                             output via MIST_DEBUG_JSONL.
    replay <file> [--session-id X]          Replay utterances from a JSONL or
                                             plain-text file; aggregate results.
        [--extraction-only]                  Drive each utterance through the
                                             production extraction pipeline with
                                             NO chat reply (deterministic F2
                                             measurement path).

Usage:
    python scripts/mist_admin.py <subcommand> [options]

Spec: ~/.claude/plans/nimble-forage-cinder.md Part 3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.knowledge.regeneration.log_regenerator import LogRegenerator
    from backend.knowledge.seed.models import SeedDocument

# Make `backend` importable when running from the host (mist-ai is not pip-installed).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REPO_ROOT = _REPO_ROOT
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "data" / "graph_snapshots"

logger = logging.getLogger("mist_admin")


def _load_backend():
    """Lazy-import backend modules so `--help` works without the neo4j driver.

    Returns a namespace-like object with the imported modules attached.
    Actual command handlers call this; the top-level CLI does not. Tier 2
    factory imports (build_extraction_pipeline, build_knowledge_retriever)
    transitively pull in sentence_transformers and other heavy deps, so they
    are loaded only when extract/retrieve subcommands call `_load_factories`.
    """
    from backend.errors import MistError, Neo4jConnectionError, Neo4jQueryError
    from backend.knowledge import admin
    from backend.knowledge.config import get_config
    from backend.knowledge.storage.neo4j_connection import Neo4jConnection

    class _Backend:
        pass

    be = _Backend()
    be.MistError = MistError
    be.Neo4jConnectionError = Neo4jConnectionError
    be.Neo4jQueryError = Neo4jQueryError
    be.admin = admin
    be.get_config = get_config
    be.Neo4jConnection = Neo4jConnection
    return be


def _load_factories():
    """Import Tier 2 factories lazily. Pulls in heavy deps (sentence_transformers)."""
    from backend.factories import build_extraction_pipeline, build_knowledge_retriever

    return build_extraction_pipeline, build_knowledge_retriever


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_seed(args: argparse.Namespace) -> int:
    """Wipe-then-apply the versioned seed source (R1.4 spec 2.0/O9).

    Repointed in R1.4 Task 10 from the retired seed-YAML dict path
    (`load_seed_yaml` + `apply_seed`) onto `load_seed_documents` +
    `reseed` over `mist-memory/seed/*.md` -- `apply_seed`/`load_seed_yaml`
    remain in `backend/knowledge/admin.py` (with their own test coverage
    unaffected) as a follow-up cleanup candidate; nothing calls them
    anymore.

    `reseed` (not apply-only) matches spec O9: wipe-then-apply under one
    shared `seed_version` is the contract, and it is the only thing that
    exercises the wipe half at all -- an apply-only path would leave a
    fact deleted from the source silently stuck in the graph forever.
    """
    be = _load_backend()
    from backend.knowledge.seed.applier import reseed
    from backend.knowledge.seed.loader import load_seed_documents

    config = be.get_config()
    seed_dir = Path(args.seed_dir) if args.seed_dir else Path(config.vault.root) / "seed"
    print(f"[seed] Loading seed documents from {seed_dir}")
    documents = load_seed_documents(seed_dir)
    seed_version = documents[0].seed_version  # loader enforces exactly one shared version
    print(f"[seed] {len(documents)} document(s) at seed_version={seed_version!r}")

    now = datetime.now(UTC).isoformat()
    connection = _connect(be)
    try:
        counts = reseed(connection, documents, seed_version=seed_version, now_iso=now)
        print("[seed] Applied (wipe-then-apply, idempotent):")
        for layer, count in counts.items():
            print(f"  {layer}: {count}")
        print(f"[seed] Total writes: {sum(counts.values())}")

        # Embedding backfill: the new applier stamps seed_version/created_at/
        # updated_at only -- it never sets `embedding`, and a second `seed`
        # run's wipe-then-recreate cycle does not preserve properties the
        # applier itself does not set. _backfill_embeddings_for_seed matches
        # on seed_version across BOTH graph partitions (unlike the pre-R1.4
        # _backfill_embeddings, which is :__Entity__-only and provenance-
        # scoped -- neither survives this applier's write shape). Disabled
        # with --no-embeddings.
        if not args.no_embeddings:
            from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
            from backend.knowledge.seed.gates import check_embeddings

            print(f"[seed] Loading embedding model: {config.embedding.model_name}")
            embedding_generator = EmbeddingGenerator(model_name=config.embedding.model_name)
            embedded = be.admin._backfill_embeddings_for_seed(
                connection, embedding_generator, seed_version
            )
            print(f"[seed] Embeddings backfilled: {embedded}")

            # I7: verify here, not only in `seed-verify`. The backfill runs
            # AFTER the graph writes have already committed, so a failure in
            # it (model load, cache miss, OOM) leaves a fully-seeded,
            # fully-unembedded graph that every other gate passes -- which is
            # the shape of both historical live losses. Reporting the count
            # alone proves nothing: it counts rows the backfill THOUGHT it
            # wrote, from the same code that failed to write them.
            embedding_gate = check_embeddings(
                connection,
                documents,
                seed_version=seed_version,
                embedding_generator=embedding_generator,
                expected_dimension=config.embedding.dimension,
            )
            status = "PASS" if embedding_gate.passed else "FAIL"
            print(f"[seed] Embedding gate: {status} ({embedding_gate.examined} nodes examined)")
            for failure in embedding_gate.failures:
                print(f"  - {failure}")
            if not embedding_gate.passed:
                print("[seed] Seed applied but embeddings are NOT verified -- see above")
                return 1
    finally:
        connection.disconnect()

    # ADR-010 Cluster 8 Phase 10: vault bootstrap. Mirrors the seeded
    # identity/user documents into the vault as canonical markdown notes.
    # Disabled with --no-vault-bootstrap. Skipped automatically when the
    # vault subsystem is disabled in config. R1.4 Task 6 retired the
    # DERIVED_FROM->VaultNote provenance edge this used to also emit.
    if not getattr(args, "no_vault_bootstrap", False):
        if config.vault.enabled:
            _do_vault_bootstrap(be, config, documents)
        else:
            print("[seed] Vault bootstrap skipped: config.vault.enabled is False")

    return 0


def _do_vault_bootstrap(be: Any, config: Any, documents: list[SeedDocument]) -> None:
    """Run the vault bootstrap step for `cmd_seed` (Phase 10).

    Builds and starts a VaultWriter, writes identity/mist.md +
    users/<id>.md from the seed documents. Idempotent so re-running `seed`
    is safe. Vault errors are logged but never propagate -- graph seed
    already succeeded by the time this runs.
    """
    import asyncio

    from backend.factories import resolve_fixed_rendered_at
    from backend.vault.writer import VaultWriter

    # Replay-determinism clock seam: MIST_FIXED_CLOCK (when set) pins the seeded
    # notes' rendered_at so the seeded users/<id>.md the chat prompt reads is
    # byte-identical across replay runs. Unset in production -> wall-clock.
    rendered_at = resolve_fixed_rendered_at()
    if rendered_at is not None:
        print(f"[seed] Vault bootstrap: pinning rendered_at to {rendered_at} (fixed clock)")

    print("[seed] Vault bootstrap: writing identity/mist.md + users/<id>.md")

    async def _run() -> dict[str, str]:
        writer = VaultWriter(config.vault)
        await writer.start()
        try:
            return await be.admin.bootstrap_vault_from_seed(
                writer, documents, rendered_at=rendered_at
            )
        finally:
            await writer.stop()

    try:
        paths = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 -- ADR-010 Invariant 6
        print(f"[seed] Vault bootstrap failed (graph seed unaffected): {exc}")
        return

    print(f"[seed]   identity_path: {paths['identity_path']}")
    print(f"[seed]   user_path:     {paths['user_path']}")


def cmd_seed_verify(args: argparse.Namespace) -> int:
    """Run the five seed-verification gates against the versioned seed source.

    `facts-present`, `node-definitions` and `embeddings` are the three
    gates that touch the graph, and only to read -- none ever writes.
    `containment` and `negation-proximity` check the source against
    itself and need no connection at all. Exits non-zero if any gate
    fails, so this is safe to wire into a pre-rebuild check.

    `node-definitions` exists because `facts-present` alone was not
    enough: R1.4 Task 10's live wipe-and-recreate defect stripped every
    node's ontology label and descriptive property while leaving the
    edges those facts describe intact (MERGE recreated them from the
    source), so `facts-present` passed throughout on a graph that had
    lost everything else.

    `embeddings` (I7) exists because the same argument applies once more,
    to a property none of the other four read. Embeddings have been lost
    on live data TWICE, and no gate could see either loss:
    `canonical_serialize.py` excludes `embedding` from the canonical
    form, so `assert_rebuild_twice_identical` and `live_vs_rebuilt_report`
    are byte-identical whether every vector is present, absent or
    all-zero. The blindness is structural, not an oversight.

    Note `embeddings` FAILS after a `seed --no-embeddings` run. That is
    correct and intended -- that flag's own help says vector retrieval
    will miss, and a graph in that state genuinely is incomplete.
    """
    be = _load_backend()
    from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
    from backend.knowledge.seed.gates import (
        check_containment,
        check_embeddings,
        check_facts_present,
        check_negation_proximity,
        check_node_definitions,
    )
    from backend.knowledge.seed.loader import load_seed_documents

    config = be.get_config()
    seed_dir = Path(args.seed_dir) if args.seed_dir else Path(config.vault.root) / "seed"
    print(f"[seed-verify] Loading seed documents from {seed_dir}")
    documents = load_seed_documents(seed_dir)
    seed_version = documents[0].seed_version  # loader enforces exactly one shared version
    print(f"[seed-verify] {len(documents)} document(s) at seed_version={seed_version!r}")

    # `embeddings` is the only gate needing the embedding model, so it is
    # constructed here rather than at import: `backend.knowledge.embeddings`
    # exports `EmbeddingGenerator` lazily (I7 T1) precisely so the other four
    # gates do not pay a ~2.8s `sentence_transformers` import they never use.
    print(f"[seed-verify] Loading embedding model: {config.embedding.model_name}")
    embedding_generator = EmbeddingGenerator(model_name=config.embedding.model_name)

    connection = _connect(be)
    try:
        gate_results = [
            (
                "facts-present",
                check_facts_present(connection, documents, seed_version=seed_version),
            ),
            (
                "node-definitions",
                check_node_definitions(connection, documents, seed_version=seed_version),
            ),
            ("containment", check_containment(documents)),
            ("negation-proximity", check_negation_proximity(documents)),
            (
                "embeddings",
                check_embeddings(
                    connection,
                    documents,
                    seed_version=seed_version,
                    embedding_generator=embedding_generator,
                    expected_dimension=config.embedding.dimension,
                ),
            ),
        ]
    finally:
        connection.disconnect()

    all_passed = True
    for name, result in gate_results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[seed-verify] {name}: {status}")
        for failure in result.failures:
            print(f"  - {failure}")
        if not result.passed:
            all_passed = False

    if all_passed:
        print("[seed-verify] all gates passed")
    else:
        print("[seed-verify] one or more gates failed")
    return 0 if all_passed else 1


def cmd_graph_dump(args: argparse.Namespace) -> int:
    be = _load_backend()
    connection = _connect(be)
    include_provenance: bool = getattr(args, "include_provenance", False)
    try:
        if args.format == "json":
            payload = be.admin.dump_graph_json(connection, include_provenance=include_provenance)
            output = json.dumps(payload, indent=2, default=str)
        else:
            output = be.admin.dump_graph_cypher(connection, include_provenance=include_provenance)
    finally:
        connection.disconnect()
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"[graph-dump] Wrote {out_path} ({len(output)} bytes)")
    else:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def cmd_graph_stats(args: argparse.Namespace) -> int:
    be = _load_backend()
    connection = _connect(be)
    try:
        node_counts = be.admin.count_nodes_by_type(connection)
        rel_counts = be.admin.count_relationships_by_type(connection)
        confidence = be.admin.get_confidence_distribution(connection)
        orphans = be.admin.find_orphan_relationships(connection)
        provenance = be.admin.count_provenance(connection)
        prov_node_counts = be.admin.provenance_counts_by_type(connection)
        prov_rel_counts = be.admin.provenance_relationship_counts_by_type(connection)
        xlayer_counts = be.admin.cross_layer_relationship_counts(connection)
    finally:
        connection.disconnect()

    print("[graph-stats]")
    print(f"\nNodes by entity_type ({sum(r['count'] for r in node_counts)} total):")
    if not node_counts:
        print("  (empty graph)")
    for row in node_counts:
        print(f"  {row['entity_type']:<24} {row['count']:>6}")

    print(f"\nRelationships by type ({sum(r['count'] for r in rel_counts)} total):")
    if not rel_counts:
        print("  (none)")
    for row in rel_counts:
        print(f"  {row['rel_type']:<24} {row['count']:>6}")

    print("\nConfidence distribution:")
    for scope in ("nodes", "relationships"):
        stats = confidence.get(scope, {}) or {}
        n = stats.get("n", 0) or 0
        if n == 0:
            print(f"  {scope}: (no confidence data)")
            continue
        avg = stats.get("avg")
        mn = stats.get("min")
        mx = stats.get("max")
        print(f"  {scope}: n={n}, avg={_fmt(avg)}, min={_fmt(mn)}, max={_fmt(mx)}")

    print("\nProvenance breakdown:")
    if not provenance:
        print("  (empty graph)")
    for source, count in sorted(provenance.items()):
        print(f"  {source:<24} {count:>6}")

    print("\nOrphan relationships (endpoints not labelled __Entity__):")
    if not orphans:
        print("  none")
    for row in orphans:
        print(
            f"  {row['source_labels']} -[{row['rel_type']}]-> "
            f"{row['target_labels']}  x{row['count']}"
        )

    print(
        f"\nProvenance Nodes (:__Provenance__) ({sum(r['count'] for r in prov_node_counts)} total):"
    )
    if not prov_node_counts:
        print("  (none)")
    for row in prov_node_counts:
        print(f"  {row['entity_type']:<24} {row['count']:>6}")

    print(
        f"\nProvenance Relationships (:__Provenance__->:__Provenance__) ({sum(r['count'] for r in prov_rel_counts)} total):"
    )
    if not prov_rel_counts:
        print("  (none)")
    for row in prov_rel_counts:
        print(f"  {row['rel_type']:<24} {row['count']:>6}")

    print(
        f"\nCross-Layer Relationships (:__Entity__ <-> :__Provenance__) ({sum(r['count'] for r in xlayer_counts)} total):"
    )
    if not xlayer_counts:
        print("  (none)")
    for row in xlayer_counts:
        print(f"  {row['rel_type']:<24} {row['count']:>6}")

    return 0


def cmd_graph_backfill_bitemporal(args: argparse.Namespace) -> int:
    """One-shot idempotent C1 backfill: stamp bitemporal fields on legacy edges."""
    be = _load_backend()
    connection = _connect(be)
    try:
        result = be.admin.backfill_bitemporal(
            connection,
            ontology_version=be.get_config().ontology_version,
            dry_run=args.dry_run,
        )
        print(f"[backfill] {result}")
        return 0
    finally:
        connection.disconnect()


def _assert_replay_source_exists(
    db_path: str, label: str, required_tables: tuple[str, ...]
) -> None:
    """Refuse a rebuild whose replay source is absent or carries no schema.

    Checks the SCHEMA, not merely the path. What that buys over `Path.exists()` is two
    states, both of which pass an existence check:

    - A truncated, half-copied, or interrupted-`cp` file.
    - A file created by something other than `initialize()` -- `touch`, or a
      `sqlite3.connect()` that opened the path and wrote no schema (both leave a
      0-byte file).

    Neither is caught by an existence check, and on both the first read raises out of
    `sqlite3` rather than returning: `DatabaseError: file is not a database` for the
    truncated file, `OperationalError: no such table` for the schema-less one. Nothing
    catches either: `cmd_graph_rebuild_from_log` handles only `RebuildTargetError`,
    `ColdCacheError` and `RebuildDeterminismError` (:808, :811, :814 -- the only three
    `except` lines between that function's `def` at :748 and the next at :824), and the
    `main()` try that wraps the command dispatch handles only `ModuleNotFoundError`,
    `FileNotFoundError` and `MistError` (:2240, :2247, :2250). `main()`'s OTHER
    `ModuleNotFoundError` handler (:2230) sits above that try, guarding the lazy
    `MistError` import, and never sees a command's exception. So both escape as a
    traceback instead of a refusal.

    What the schema check does NOT buy, recorded because an earlier version of this
    docstring asserted the opposite and called it the common case: a store that a
    PRE-FIX run of this command created and left empty passes this check too.
    `EventStore.initialize()` executescripts `schema.sql`, which carries
    `CREATE TABLE IF NOT EXISTS epoch_ledger` (schema.sql:134), and
    `ExtractionCache.initialize()` executescripts a DDL whose sole statement is
    `CREATE TABLE IF NOT EXISTS extraction_cache` -- so on exactly those machines both
    required tables exist, are empty, and this guard passes. The run proceeds to
    `_build_log_regenerator`'s "No epochs found in the event store" refusal, which is
    a `ColdCacheError` and was never a traceback.

    The connection is opened READ-ONLY via a `file:...?mode=ro` URI. That is not
    decoration: `EventStore._get_connection` runs `PRAGMA journal_mode=WAL`
    (store.py:64), which mutates a non-WAL database's header and creates
    `-wal`/`-shm` sidecars, so checking through the normal path would make this
    guard a writer of database content.

    `mode=ro` is not, however, byte-free, and that is recorded here rather than left
    as the earlier implication that it made this guard a non-write. MEASURED against
    a temp-dir WAL store carrying this schema, on the container (sqlite 3.37.2,
    Linux) and on the host the CLI actually runs on (sqlite 3.45.1, Windows), with
    sha256 compared before and after: the main database and the `-wal` are
    byte-IDENTICAL, but where no `-shm` exists the first read CREATES a 32768-byte
    `-shm` wal-index beside the store and leaves it there after `close()`. SQLite
    needs that wal-index to read a WAL database. So the guard does not alter the
    store, and it does add a sidecar.

    The path is percent-encoded into that URI rather than f-string interpolated, and
    the bug that motivates it is not cosmetic. MEASURED on both platforms above, with
    a store under a directory named `release#2`: `f"file:{db_path}?mode=ro"` puts
    everything after the `#` into the URI FRAGMENT, so SQLite opened
    `.../release` -- a different path -- and, because `?mode=ro` was inside that
    discarded fragment, opened it read-write-CREATE. The run created a database file
    that did not exist, found an empty `sqlite_master`, and refused a healthy store
    for having no `epoch_ledger`. A dry-run command silently creating a database
    because of URI syntax is the same defect class this branch exists to remove.
    `%XX` misparses too (`v%41B` and `50%25off` both raised
    `OperationalError: unable to open database file`); a bare `%` not followed by two
    hex digits happens to survive. `pathlib.Path.as_uri()` was measured equally
    correct on every shape tested, but it emits the `file://<authority>/...` form,
    which changes the URI shape for path classes not tested here (UNC), so the
    narrower fix that touches only the escaping is the one used.

    That is also why `sqlite3.OperationalError` is handled apart from every other
    `sqlite3.Error` below. When the wal-index cannot be created the read raises
    `OperationalError` on a store whose schema is entirely intact -- MEASURED as
    "unable to open database file" against a read-only directory and against a `:ro`
    docker bind mount, and "attempt to write a readonly database" when no `-wal` is
    present either. Reporting that as "not a readable SQLite database" accuses a
    healthy store of corruption. Both branches still REFUSE, because the guard cannot
    read the schema either way and the replay's own reads would fail identically; only
    the diagnosis differs. The two are cleanly separable by type: a truncated file
    raises `sqlite3.DatabaseError` ("file is not a database") which is NOT an
    `OperationalError`, while the wal-index failure is. Note the raise surfaces from
    `execute()`, not from `sqlite3.connect()` -- `connect()` returns a Connection and
    the wal-index is materialised on first read -- but both sit inside the same `try`,
    so the handled outcome is the same.

    EVERY table the replay reads must be named, not just one. Gating on a single
    table leaves a store that has it but lacks a sibling passing the guard and then
    tracebacking on the first read -- the failure mode this function exists to convert
    into a decision. The event store's replay reads exactly three:
    `epoch_ledger` (`get_current_epoch` store.py:493, `list_epochs` :499, both called
    from `_build_log_regenerator`), and `conversation_turn_events` LEFT JOIN
    `conversation_sessions` (`get_all_turns_for_reextraction` :406-407, plus
    `get_turn_count` :454) -- the only two `EventStore` methods `LogRegenerator` calls
    (`grep -n "self._events" log_regenerator.py` -> :291, :301, and the :105 assignment).

    Args:
        db_path: Filesystem path to the SQLite store being replayed.
        label: Human name used in the refusal ("event store" / "extraction cache").
        required_tables: Every table the store must already have for the replay to
            read it. Order is preserved in the refusal message.

    Raises:
        ColdCacheError: When the file is missing, cannot be opened, is not a readable
            SQLite database, or lacks any of `required_tables`. Raised so the CLI's
            REFUSED branch reports a decision (exit 2) rather than a traceback.
    """
    import sqlite3
    from pathlib import Path as _Path
    from urllib.parse import quote as _quote

    from backend.knowledge.regeneration.log_regenerator import ColdCacheError

    _MISSING = (
        "A rebuild REPLAYS an existing log and will not create one -- creating it here "
        "would write to live state under a dry-run flag, and an empty store would then "
        "be indistinguishable from a missing one. Point the config at the real store, "
        "or run the traffic that populates it."
    )

    if not _Path(db_path).exists():
        raise ColdCacheError(f"No {label} at {db_path}. {_MISSING}")

    # The path is being interpolated into a URI, so it must be percent-encoded: `#`,
    # `?` and `%XX` are URI syntax, not filename characters. See the docstring for what
    # the unescaped f-string this replaces actually did.
    uri = "file:" + _quote(str(_Path(db_path).resolve())) + "?mode=ro"

    placeholders = ", ".join("?" * len(required_tables))
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            present = {
                row[0]
                for row in conn.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' "  # nosec B608
                    f"AND name IN ({placeholders})",
                    required_tables,
                ).fetchall()
            }
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        # NOT corruption. SQLite must create a `-shm` wal-index to read a WAL
        # database, and a read-only connection cannot create one where the
        # directory is not writable, so a healthy store on read-only media or a
        # `:ro` mount lands here with its schema fully intact. Refuse anyway -- the
        # replay's reads would fail the same way -- but do not call it corrupt.
        raise ColdCacheError(
            f"The {label} at {db_path} exists but could not be opened read-only "
            f"({exc}). This is NOT evidence that the store is corrupt. The usual "
            f"cause is the wal-index: SQLite needs a `-shm` to read a WAL database "
            f"and a read-only connection must create it when absent, so a store on "
            f"read-only media, a `:ro` bind mount, or a directory this process "
            f"cannot write fails here with its schema intact. Check that the store's "
            f"DIRECTORY is writable, or copy the store together with any `-wal` and "
            f"`-shm` siblings somewhere writable and point the config there. If the "
            f"directory is writable, the sqlite error above is the thing to read."
        ) from exc
    except sqlite3.Error as exc:
        raise ColdCacheError(
            f"The {label} at {db_path} is not a readable SQLite database ({exc}). {_MISSING}"
        ) from exc

    missing = [table for table in required_tables if table not in present]
    if missing:
        raise ColdCacheError(
            f"The {label} at {db_path} has no {', '.join(f'`{t}`' for t in missing)} "
            f"table(s), so it was not created by `initialize()` -- a 0-byte file left "
            f"by `touch` or by a bare `sqlite3.connect()` on the path looks exactly "
            f"like this. {_MISSING}"
        )


def _build_log_regenerator(
    be: Any, staging_conn: Any, epoch_id: int | None
) -> tuple[LogRegenerator, dict[str, Any]]:
    """Build a LogRegenerator wired to the staging graph, resolving the epoch.

    Wires the staging curation pipeline with the real all-MiniLM-L6-v2
    embedding provider: GraphStore.initialize_schema() on staging, then
    GraphExecutor(staging_conn), then build_curation_pipeline(config, executor,
    EmbeddingGenerator(model_name)). Using real embeddings ensures the merge
    decisions made by the deterministic deduper match production topology (fake
    embeddings produce different cosine distances and a different graph topology,
    which would block the R1.6 live==rebuilt closure gate). all-MiniLM-L6-v2
    embeddings are deterministic for identical input text, so rebuild-twice
    determinism still holds. EventStore and ExtractionCache are constructed from
    config paths (defaulting to ~/.mist/ siblings when the config carries no
    explicit override) and are READ-ONLY replay sources: the regenerator's own
    job/checkpoint rows go to a `NullRebuildJournal`, never to the live ledger.

    Returns (LogRegenerator, epoch_dict).
    """
    from pathlib import Path as _Path

    from backend.event_store.store import EventStore
    from backend.factories import build_curation_pipeline
    from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
    from backend.knowledge.extraction_cache import ExtractionCache
    from backend.knowledge.regeneration.log_regenerator import ColdCacheError, LogRegenerator
    from backend.knowledge.regeneration.rebuild_journal import NullRebuildJournal
    from backend.knowledge.storage.graph_executor import GraphExecutor
    from backend.knowledge.storage.graph_store import GraphStore

    config = be.get_config()

    # EventStore: use config path or fall back to the default ~/.mist/event_store.db.
    # Both spellings are LIVE state, so this store is a replay SOURCE only -- see the
    # `journal=` argument below.
    event_store_path = config.event_store.db_path or str(_Path.home() / ".mist" / "event_store.db")
    event_store = EventStore(event_store_path)

    # ExtractionCache: lives alongside the event store db.
    cache_path = str(_Path(event_store_path).parent / "extraction_cache.db")
    extraction_cache = ExtractionCache(cache_path)

    # NEITHER store is initialize()d here, and that is the point. `initialize()` is a
    # WRITE -- mkdir(parents=True) at store.py:75, executescript(schema.sql) at :80, and
    # TWO conditional `ALTER TABLE` migrations at :88-90 and :99-101 -- performed by a
    # command whose entire advertised contract is "proof-first, dry-run only". Calling it
    # also MANUFACTURED the absence it was meant to tolerate: on a machine with no event
    # store it created an empty one and the run then reported "No epochs found", which
    # reads identically to a store that exists and is empty. A rebuild replays an
    # existing log; it does not bring one into being.
    _assert_replay_source_exists(
        event_store_path,
        "event store",
        # Every table the replay reads, not just the epoch one -- see the guard's
        # docstring for the call sites that read each.
        ("epoch_ledger", "conversation_sessions", "conversation_turn_events"),
    )
    _assert_replay_source_exists(cache_path, "extraction cache", ("extraction_cache",))

    # Real embedding provider (all-MiniLM-L6-v2). Eager load is acceptable here:
    # this is an admin CLI (sync batch), NOT the async server event loop, so there
    # is no event-loop lazy-loading hazard. A single instance is shared across
    # GraphStore and the curation pipeline to avoid loading two copies.
    embedding_provider = EmbeddingGenerator(config.embedding.model_name)

    # Initialize staging schema (idempotent) via a transient GraphStore.
    staging_store = GraphStore(connection=staging_conn, embedding_generator=embedding_provider)
    staging_store.initialize_schema()

    # Curation pipeline wired to staging. Real embeddings ensure dedup merge
    # decisions match production topology; determinism holds because
    # all-MiniLM-L6-v2 is deterministic for identical input text.
    executor = GraphExecutor(staging_conn)
    pipeline = build_curation_pipeline(config, executor, embedding_provider=embedding_provider)

    regen = LogRegenerator(
        event_store=event_store,
        extraction_cache=extraction_cache,
        staging_curation_pipeline=pipeline,
        # The dry-run proof is not a rebuild of record: nothing reads its job rows
        # (`get_reextraction_job` has no production caller), and `_build_once` runs
        # twice per invocation, so a durable journal here would append two
        # `rebuild-<epoch>-<uuid>` rows plus a checkpoint per turn to the LIVE ledger
        # every time the determinism gate is run. Wired unconditionally rather than
        # behind `args.dry_run` because dry-run is the ONLY mode this command has
        # (`--dry-run` is `required=True`); a durable branch here would be a dead
        # branch justifying itself with a future caller.
        journal=NullRebuildJournal(),
    )

    # Resolve epoch -- raise ColdCacheError so the handler's REFUSED branch fires.
    if epoch_id is None:
        epoch = event_store.get_current_epoch()
        if epoch is None:
            raise ColdCacheError(
                "No epochs found in the event store. "
                "Run at least one conversation turn to register an epoch before rebuilding."
            )
    else:
        epochs = event_store.list_epochs()
        epoch = next((e for e in epochs if e["epoch_id"] == epoch_id), None)
        if epoch is None:
            raise ColdCacheError(
                f"Epoch {epoch_id} not found in the event store. "
                f"Available epoch IDs: {[e['epoch_id'] for e in epochs]}"
            )

    return regen, epoch


def cmd_graph_rebuild_from_log(args: argparse.Namespace) -> int:
    """Build a staging graph from the log twice; assert determinism; report live divergence."""
    import asyncio as _asyncio

    from backend.knowledge.canonical_serialize import canonical_graph_form
    from backend.knowledge.config import Neo4jConfig
    from backend.knowledge.eval_isolation import RebuildTargetError, assert_rebuild_target_not_live
    from backend.knowledge.regeneration.log_regenerator import ColdCacheError
    from backend.knowledge.regeneration.rebuild_gate import (
        RebuildDeterminismError,
        assert_rebuild_twice_identical,
        live_vs_rebuilt_report,
    )

    be = _load_backend()
    # Resolve live_uri early (no connection needed) so the isolation guard
    # fires before ANY connect() or execute_write() call.
    live_uri = be.get_config().neo4j.uri
    live_conn = None
    staging_conn = None
    try:
        # Guard FIRST: refuse if --staging-uri resolves to the same host:port as live.
        # Must be the first statement in the try so RebuildTargetError is caught below.
        assert_rebuild_target_not_live(args.staging_uri, live_uri)

        live_conn = _connect(be)  # reads NEO4J_URI (live) from env

        # Build staging connection from args.staging_uri with live credentials.
        config = be.get_config()
        staging_config = Neo4jConfig(
            uri=args.staging_uri,
            username=config.neo4j.username,
            password=config.neo4j.password,
        )
        staging_conn = be.Neo4jConnection(staging_config)
        staging_conn.connect()

        def _build_once() -> str:
            # Wipe staging so each build starts from a clean slate.
            staging_conn.execute_write("MATCH (n) DETACH DELETE n", {})
            regen, epoch = _build_log_regenerator(be, staging_conn, args.epoch)
            # Each call gets a unique job_id automatically (job_id left unset).
            _asyncio.run(
                regen.rebuild(
                    staging_uri=args.staging_uri,
                    live_uri=live_uri,
                    epoch=epoch,
                    source_conn=live_conn,
                    staging_conn=staging_conn,
                )
            )
            return canonical_graph_form(staging_conn, include_provenance=False)

        build_a = _build_once()
        build_b = _build_once()
        assert_rebuild_twice_identical(build_a, build_b)
        print("[rebuild] determinism gate PASSED (rebuild-twice byte-identical)")
        live_form = canonical_graph_form(live_conn, include_provenance=False)
        print(live_vs_rebuilt_report(live_form, build_b))
        return 0
    except RebuildTargetError as exc:
        print(f"[rebuild] REFUSED: {exc}")
        return 2
    except ColdCacheError as exc:
        print(f"[rebuild] REFUSED: {exc}")
        return 2
    except RebuildDeterminismError as exc:
        print(f"[rebuild] {exc}")
        return 1
    finally:
        if live_conn is not None:
            live_conn.disconnect()
        if staging_conn is not None:
            staging_conn.disconnect()


def cmd_graph_reset(args: argparse.Namespace) -> int:
    be = _load_backend()
    connection = _connect(be)
    try:
        non_seed = be.admin.count_non_seed_entities(connection)
        node_count = connection.execute_query("MATCH (n:__Entity__) RETURN count(n) AS count")[0][
            "count"
        ]
        rel_count = connection.execute_query(
            "MATCH (:__Entity__)-[r]->(:__Entity__) RETURN count(r) AS count"
        )[0]["count"]

        print(f"[graph-reset] Current graph: {node_count} nodes, {rel_count} relationships")
        print(f"[graph-reset] Non-seed entities: {non_seed}")

        if args.dry_run:
            print("[graph-reset] --dry-run: no changes written.")
            if non_seed > 0 and not args.include_derived:
                print(
                    f"[graph-reset] WOULD REFUSE: {non_seed} non-seed entities present. "
                    "Re-run with --include-derived to override."
                )
            elif node_count == 0:
                print("[graph-reset] WOULD NO-OP: graph is already empty.")
            else:
                print(f"[graph-reset] WOULD REMOVE: {node_count} nodes, {rel_count} relationships.")
            return 0

        if not args.confirm:
            print(
                "[graph-reset] Refusing to proceed: pass --confirm to execute. "
                "Pass --dry-run to preview."
            )
            return 2

        if non_seed > 0 and not args.include_derived:
            print(
                f"[graph-reset] REFUSING: {non_seed} non-seed entities present. "
                "Pass --include-derived to wipe anyway."
            )
            return 2

        if not args.no_snapshot and node_count > 0:
            snapshot_path = _resolve_snapshot_path(args.snapshot_to)
            print(f"[graph-reset] Snapshotting to {snapshot_path}")
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            cypher = be.admin.dump_graph_cypher(connection)
            snapshot_path.write_text(cypher, encoding="utf-8")
            print(f"[graph-reset] Snapshot written ({len(cypher)} bytes)")
        elif args.no_snapshot:
            print("[graph-reset] --no-snapshot: skipping pre-wipe snapshot.")

        counts = be.admin.reset_graph(connection, include_derived=args.include_derived)
    finally:
        connection.disconnect()

    prov_removed = counts.get("provenance_nodes_removed", 0)
    prov_msg = f", {prov_removed} provenance nodes" if prov_removed > 0 else ""
    print(
        f"[graph-reset] Removed {counts['nodes_removed']} nodes and "
        f"{counts['relationships_removed']} relationships{prov_msg}."
    )
    return 0


def cmd_stack_status(args: argparse.Namespace) -> int:
    be = _load_backend()
    config = be.get_config()
    connection = be.Neo4jConnection(config.neo4j)
    try:
        neo4j_status = be.admin.probe_neo4j(connection)
    finally:
        connection.disconnect()

    llm_status = be.admin.probe_llm(config.llm.base_url)
    backend_url = args.backend_url or "http://localhost:8001"
    backend_status = be.admin.probe_backend(backend_url)

    print("[stack-status]")
    for status in (neo4j_status, llm_status, backend_status):
        _print_status_line(status)
    healthy = all(s.get("status") == "healthy" for s in (neo4j_status, llm_status, backend_status))
    return 0 if healthy else 1


def cmd_extract(args: argparse.Namespace) -> int:
    """Run the extraction pipeline on a single utterance. Dry-run by default.

    Dry-run: `include_curation=False`, `include_internal_derivation=False` —
    returns `ValidationResult` without touching Neo4j or LanceDB.

    Commit: full pipeline with curation + MistIdentity derivation — writes
    entities, relationships, and provenance links through the same code path
    as the production backend.
    """
    be = _load_backend()
    build_extraction_pipeline, _ = _load_factories()
    config = be.get_config()
    event_id = f"admin-cli-{uuid.uuid4().hex[:12]}"
    session_id = args.session_id or "admin-cli"
    mode = "commit" if args.commit else "dry-run"
    print(f"[extract] mode={mode} event_id={event_id} session_id={session_id}")
    print(f"[extract] utterance: {args.utterance!r}")

    pipeline = build_extraction_pipeline(
        config,
        include_curation=args.commit,
        include_internal_derivation=args.commit,
    )
    result = asyncio.run(
        pipeline.extract_from_utterance(
            utterance=args.utterance,
            conversation_history=[],
            event_id=event_id,
            session_id=session_id,
            extraction_source="admin-cli",
        )
    )
    _print_extraction_result(result, mode=mode)
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    """Run hybrid retrieval on a query and print facts with scores + timing.

    Phase 9 / Phase 11: when the sidecar is enabled, the retriever's
    `historical` intent + three-way RRF hybrid path can return vault
    sidecar prose hits. Without the sidecar wired here, those branches
    return zero results -- the CLI would silently misrepresent what
    the production retriever can do. SQLite readers are race-safe with
    the server's writer (single-writer rule applies to writes only).
    """
    be = _load_backend()
    _, build_knowledge_retriever = _load_factories()
    config = be.get_config()
    print(
        f"[retrieve] query={args.query!r} user_id={args.user_id} "
        f"limit={args.limit} threshold={args.threshold}"
    )

    sidecar = _build_cli_sidecar(config)
    try:
        retriever = build_knowledge_retriever(config, vault_sidecar=sidecar)
        result = asyncio.run(
            retriever.retrieve(
                query=args.query,
                user_id=args.user_id,
                limit=args.limit,
                similarity_threshold=args.threshold,
            )
        )
    finally:
        _close_cli_sidecar(sidecar)
    _print_retrieval_result(result, show_context=args.show_context)
    return 0


# ---------------------------------------------------------------------------
# Tier 3 — core async logic (Protocol-injectable, testable)
# ---------------------------------------------------------------------------


async def run_chat(
    handler: Any,
    utterance: str,
    session_id: str,
    user_id: str = "User",
) -> dict[str, Any]:
    """Run one chat turn through `handler.handle_message` and return a record.

    After `handle_message` returns, drains any fire-and-forget background
    tasks (fire-and-forget extraction is spawned via `asyncio.create_task`
    inside `ConversationHandler.handle_message`). Without this drain, the
    extraction task is cancelled when the surrounding `asyncio.run` closes
    the loop, which would silently skip every graph write.

    Args:
        handler: Any object with `async handle_message(user_message, session_id,
            user_id="User", max_history=10) -> str`. Either a production
            ConversationHandler from `backend.factories.build_conversation_handler`
            or a test double.
        utterance: The user message to send.
        session_id: Conversation session identifier.
        user_id: User identifier (default "User" matches the seeded anchor).

    Returns:
        Dict with keys: utterance, session_id, user_id, response,
        duration_ms (response-only, caller-facing), extraction_duration_ms
        (background drain), total_duration_ms, ok, error. On handler
        exception: `ok=False`, `error="ExceptionType: msg"`, response=None.
    """
    start = time.time()
    try:
        response = await handler.handle_message(
            user_message=utterance,
            session_id=session_id,
            user_id=user_id,
        )
        response_duration_ms = (time.time() - start) * 1000
        ex_start = time.time()
        pending = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        extraction_duration_ms = (time.time() - ex_start) * 1000
        return {
            "utterance": utterance,
            "session_id": session_id,
            "user_id": user_id,
            "response": response,
            "duration_ms": response_duration_ms,
            "extraction_duration_ms": extraction_duration_ms,
            "total_duration_ms": response_duration_ms + extraction_duration_ms,
            "ok": True,
            "error": None,
        }
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        return {
            "utterance": utterance,
            "session_id": session_id,
            "user_id": user_id,
            "response": None,
            "duration_ms": duration_ms,
            "extraction_duration_ms": 0.0,
            "total_duration_ms": duration_ms,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }


async def run_replay(
    handler: Any,
    inputs: list[dict[str, Any]],
    default_session_id: str,
    default_user_id: str = "User",
) -> list[dict[str, Any]]:
    """Replay a list of inputs through `run_chat`, preserving per-entry metadata.

    Each input is a dict with at minimum an `utterance` key. Optional keys:
    `session_id` (override default), `user_id` (override default), `tag`
    (label propagated to result), `expected_behavior` (label propagated).

    Args:
        handler: ConversationHandler or test double.
        inputs: Ordered list of input dicts.
        default_session_id: Session used when an input lacks `session_id`.
        default_user_id: User used when an input lacks `user_id`.

    Returns:
        List of per-turn result dicts in the same order as `inputs`. Each
        carries the same keys as `run_chat`'s return plus `tag` and
        `expected_behavior` if present on input.
    """
    results: list[dict[str, Any]] = []
    for entry in inputs:
        utterance = entry.get("utterance", "")
        sid = entry.get("session_id", default_session_id)
        uid = entry.get("user_id", default_user_id)
        result = await run_chat(handler, utterance, sid, uid)
        for key in ("tag", "expected_behavior"):
            if key in entry:
                result[key] = entry[key]
        results.append(result)
    return results


async def run_extraction_only(
    handler: Any,
    utterance: str,
    session_id: str,
    user_id: str = "User",
) -> dict[str, Any]:
    """Run the production extraction pipeline for one utterance, no chat reply.

    Drives MIST's PRODUCTION extraction path (subject-scope classifier ->
    ontology extraction -> validation -> curation/graph write) WITHOUT
    generating a conversational reply and without injecting a same-turn
    assistant reply into the extraction context. This is the deterministic F2
    measurement path: the chat reply is conversational noise the gold corpus
    does not encode and is the sole source of F2 nondeterminism (flash-attn FP
    noise on long greedy generations); the extraction calls themselves are
    deterministic at temperature 0.

    Reuses the handler's extraction entry point directly rather than
    reimplementing it. Mirrors the scaffolding `handle_message` sets up before
    spawning background extraction, minus the chat generation:

    1. Step 0 -- vault session-note path pre-allocation
       (`_get_or_allocate_vault_path`), mirroring `handle_message`'s cache
       priming exactly. R1.3 retired the fact-path consumer of the returned
       path, so the value itself is not threaded any further here.
    2. Event-store turn record (`_record_turn_event`) with an EMPTY
       assistant_message, to obtain the `event_id` (fact provenance) and
       `recorded_at` (bitemporal fact-time + extraction reference_date). The
       reply slot is empty because no reply is generated.
    3. The production extraction entry point (`_extract_knowledge_async`) with
       `conversation_history=[]` -- so no prior turn enters the extraction
       "Context:" block. R1.3.1 dropped the `assistant_message` parameter
       from `_extract_knowledge_async` entirely (it only fed the retired
       per-turn vault append), so there is nothing left to pass here. This is
       the same coroutine `handle_message` spawns as a background task; running
       it inline here emits the identical `extraction.ontology` /
       `extraction.scope_classifier` `llm_call` debug records (via the
       instrumented provider) that `score_extraction_run.py` consumes.
    4. Drain step (`_drain_extraction_tasks`) for defensive symmetry with
       `run_chat`. The current production extraction pipeline does not spawn
       registry-tracked background tasks -- the inline `_extract_knowledge_async`
       await above does the work -- so on the real path this drain finds
       nothing. It is kept as parity in case the pipeline later spawns
       background tasks; it is not load-bearing here.

    The 60-probe gold corpus is single-utterance and self-contained (no prior
    turns), so an empty `conversation_history` is the faithful extraction input.
    Documented in `scripts/eval_harness/extraction_probe_set_design.md`.

    Args:
        handler: A ConversationHandler (or test double) exposing
            `_get_or_allocate_vault_path`, `_record_turn_event`,
            `_extract_knowledge_async`, and `_drain_extraction_tasks`.
        utterance: The user message to extract from.
        session_id: Conversation session identifier.
        user_id: User identifier (default "User" matches the seeded anchor).

    Returns:
        Dict with keys: utterance, session_id, user_id, ok, error,
        extraction_duration_ms (the extraction + drain wall time), and
        event_id. On handler exception: ok=False, error="ExceptionType: msg".
    """
    start = time.time()
    try:
        # Step 0: vault session-note path pre-allocation (pure path compute).
        # R1.3: the return value is no longer forwarded to extraction; the
        # call is kept for its per-session slug-cache priming side effect.
        handler._get_or_allocate_vault_path(session_id, first_utterance=utterance)

        # Event-store turn record. EMPTY assistant_message: no reply is
        # generated on the extraction-only path. Yields the event_id used for
        # fact provenance and the recorded_at that anchors both the bitemporal
        # edge fact-time and the extraction prompt's reference_date.
        event_id, recorded_at = handler._record_turn_event(
            session_id=session_id,
            user_message=utterance,
            assistant_message="",
        )

        # Production extraction entry point, run inline. conversation_history
        # is empty so no prior turn enters the extraction context. The
        # instrumented provider emits the extraction.* llm_call debug records
        # the scorer consumes.
        await handler._extract_knowledge_async(
            utterance=utterance,
            conversation_history=[],
            event_id=event_id,
            session_id=session_id,
            turn_record=None,
            recorded_at=recorded_at,
        )

        # Defensive symmetry with run_chat: drain any background extraction
        # tasks the pipeline may have registered before asyncio.run closes the
        # loop. The current production extraction pipeline does not spawn
        # registry tasks -- the inline await above does the work -- so on the
        # real path this drain finds nothing. Kept for parity in case the
        # pipeline later spawns background tasks; not load-bearing here.
        await handler._drain_extraction_tasks(session_id=session_id)

        extraction_duration_ms = (time.time() - start) * 1000
        return {
            "utterance": utterance,
            "session_id": session_id,
            "user_id": user_id,
            "ok": True,
            "error": None,
            "extraction_duration_ms": extraction_duration_ms,
            "event_id": event_id,
        }
    except Exception as e:  # noqa: BLE001
        # Intentional batch-isolation boundary: one probe's failure is
        # captured as ok=False so run_extraction_only_replay continues the
        # batch rather than aborting. Consistent with the failure-isolation
        # boundaries in run_replay / _extract_knowledge_async.
        extraction_duration_ms = (time.time() - start) * 1000
        return {
            "utterance": utterance,
            "session_id": session_id,
            "user_id": user_id,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "extraction_duration_ms": extraction_duration_ms,
            "event_id": None,
        }


async def run_extraction_only_replay(
    handler: Any,
    inputs: list[dict[str, Any]],
    default_session_id: str,
    default_user_id: str = "User",
) -> list[dict[str, Any]]:
    """Replay inputs through `run_extraction_only`, preserving per-entry metadata.

    The extraction-only sibling of `run_replay`. Each input is a dict with at
    minimum an `utterance` key; optional `session_id`, `user_id`, `tag`, and
    `expected_behavior` keys behave exactly as in `run_replay`. A single
    probe's extraction failure is captured in that probe's record and does not
    abort the batch.

    Args:
        handler: ConversationHandler or test double (see `run_extraction_only`).
        inputs: Ordered list of input dicts.
        default_session_id: Session used when an input lacks `session_id`.
        default_user_id: User used when an input lacks `user_id`.

    Returns:
        List of per-probe result dicts in input order; each carries the keys
        from `run_extraction_only` plus `tag`/`expected_behavior` if present.
    """
    results: list[dict[str, Any]] = []
    for entry in inputs:
        utterance = entry.get("utterance", "")
        sid = entry.get("session_id", default_session_id)
        uid = entry.get("user_id", default_user_id)
        result = await run_extraction_only(handler, utterance, sid, uid)
        for key in ("tag", "expected_behavior"):
            if key in entry:
                result[key] = entry[key]
        results.append(result)
    return results


def _read_replay_inputs(path: Path) -> list[dict[str, Any]]:
    """Load replay inputs from a JSONL or plain-text file.

    JSONL (`.jsonl`/`.json`): one JSON object per line, each with at least
    `utterance`. Bare strings on a line are treated as shorthand for
    `{"utterance": "<string>"}`. Blank lines and `#`-prefixed lines are
    skipped.

    Plain text (any other extension): one utterance per line. Blank lines
    and `#`-prefixed lines are skipped. No escaping; newlines within an
    utterance are not supported.

    Raises:
        ValueError: If a JSONL line is not valid JSON or yields an
            unsupported type (neither dict nor string).
    """
    text = path.read_text(encoding="utf-8")
    items: list[dict[str, Any]] = []
    is_jsonl = path.suffix.lower() in (".jsonl", ".json")
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if is_jsonl:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}") from e
            if isinstance(obj, str):
                items.append({"utterance": obj})
            elif isinstance(obj, dict):
                items.append(obj)
            else:
                raise ValueError(
                    f"Line {i} of {path}: expected object or string, " f"got {type(obj).__name__}"
                )
        else:
            items.append({"utterance": line})
    return items


def cmd_chat(args: argparse.Namespace) -> int:
    """Full end-to-end chat turn through the production ConversationHandler.

    Vault constraint (ADR-010 Phase 5 single-writer rule): cmd_chat does
    NOT construct a VaultWriter. Spinning up a second writer against a
    vault root that the running server already owns deadlocks on the git
    auto-init / consumer-loop coordination. The chat path exercises LLM
    + extraction + graph writes; vault session-note writes are produced
    only by the server's WebSocket path. To exercise vault writes from
    the CLI, stop the server first.
    """
    be = _load_backend()
    from backend.factories import build_conversation_handler

    config = be.get_config()
    session_id = args.session_id or f"admin-cli-{uuid.uuid4().hex[:8]}"
    user_id = args.user_id or "User"
    print(f"[chat] session_id={session_id} user_id={user_id}")
    print(f"[chat] utterance: {args.utterance!r}")
    print("[chat] Building conversation handler (may load embedding model)...")

    # ADR-010 Phase 9: wire the read-only vault sidecar so the CLI chat path
    # exercises the same vault auto-RAG / "Relevant Documents" retrieval the
    # server's WebSocket path does (server lifespan: build_sidecar_index ->
    # vault_sidecar= into build_conversation_handler). `_build_cli_sidecar`
    # returns None when config.sidecar_index.enabled is False, mirroring the
    # server's enablement guard, and uses a reader (no writer) per the
    # single-writer rule so it is safe to run alongside the server.
    sidecar = _build_cli_sidecar(config)
    try:
        handler = build_conversation_handler(config, vault_sidecar=sidecar)

        result = asyncio.run(run_chat(handler, args.utterance, session_id, user_id))
        _print_chat_result(result)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            print(f"\n[chat] Wrote result to {out_path}")
    finally:
        _close_cli_sidecar(sidecar)
    return 0 if result["ok"] else 1


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay utterances from a file through the chat path; aggregate results.

    Same vault constraint as `cmd_chat` -- no VaultWriter construction
    here. Replays validate LLM/extraction/retrieval quality against the
    integrated graph; conversation-side DERIVED_FROM->VaultNote emission
    is covered by the server-path unit tests
    (test_conversation_handler_vault_integration, test_factories_vault).

    With `--extraction-only`, each utterance is driven through the production
    extraction pipeline WITHOUT a chat reply (the deterministic F2 measurement
    path). The chat reply is conversational noise the F2 gold does not encode
    and is the sole source of F2 nondeterminism (flash-attn FP noise on long
    greedy generations); the extraction calls are deterministic at temperature
    0. The same `extraction.ontology` / `extraction.scope_classifier` debug
    records are emitted, so `score_extraction_run.py` works unchanged. See
    `scripts/eval_harness/extraction_probe_set_design.md`.
    """
    be = _load_backend()
    from backend.factories import build_conversation_handler

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Replay input file not found: {input_path}")
    inputs = _read_replay_inputs(input_path)
    if not inputs:
        print(f"[replay] No inputs in {input_path}")
        return 0

    config = be.get_config()
    session_id = args.session_id or f"replay-{uuid.uuid4().hex[:8]}"
    user_id = args.user_id or "User"
    extraction_only = getattr(args, "extraction_only", False)
    mode = "extraction-only" if extraction_only else "chat"
    print(
        f"[replay] {len(inputs)} inputs, session_id={session_id}, "
        f"user_id={user_id}, mode={mode}"
    )
    print("[replay] Building conversation handler (may load embedding model)...")

    # ADR-010 Phase 9: wire the read-only vault sidecar so replay validates
    # retrieval quality against the SAME vault auto-RAG path production uses
    # (server lifespan: build_sidecar_index -> vault_sidecar= into
    # build_conversation_handler). `_build_cli_sidecar` returns None when
    # config.sidecar_index.enabled is False, mirroring the server's
    # enablement guard, and is a reader (no writer) per the single-writer rule.
    #
    # The sidecar feeds the chat path's auto-RAG / "Relevant Documents"
    # injection. The extraction-only path does NOT consult the sidecar (it
    # skips chat generation entirely), but building it is harmless and keeps
    # both modes on one wiring path.
    sidecar = _build_cli_sidecar(config)
    try:
        handler = build_conversation_handler(config, vault_sidecar=sidecar)

        if extraction_only:
            results = asyncio.run(run_extraction_only_replay(handler, inputs, session_id, user_id))
            _print_extraction_only_summary(results)
        else:
            results = asyncio.run(run_replay(handler, inputs, session_id, user_id))
            _print_replay_summary(results)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            print(f"\n[replay] Wrote {len(results)} records to {out_path}")
    finally:
        _close_cli_sidecar(sidecar)

    fail_count = sum(1 for r in results if not r["ok"])
    return 0 if fail_count == 0 else 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _connect(be):
    """Build and connect a Neo4jConnection from the current config."""
    config = be.get_config()
    connection = be.Neo4jConnection(config.neo4j)
    connection.connect()
    return connection


def _build_cli_sidecar(config: Any) -> Any:
    """Build a read-only VaultSidecarIndex for CLI retrieval commands.

    ADR-010 Phase 5 single-writer rule: only ONE VaultWriter may operate
    against a vault root at a time. The server lifespan owns the
    production writer; a CLI process running concurrently with the server
    must NOT spin up a second writer. Read-side access (sidecar query)
    has no such restriction -- multiple SQLite readers are safe.

    This helper is the CLI's sole vault entry point. `cmd_chat` and
    `cmd_replay` deliberately do NOT take a writer; they exercise the
    LLM + extraction + retrieval paths but vault session-note writes
    are left to the server's WebSocket path.

    Returns None when the sidecar subsystem is disabled in config.
    """
    if not config.sidecar_index.enabled:
        return None

    from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
    from backend.vault.sidecar_index import VaultSidecarIndex

    embedder = EmbeddingGenerator(config.embedding.model_name)
    sidecar = VaultSidecarIndex(config.sidecar_index, embedder)
    sidecar.initialize()
    return sidecar


def _close_cli_sidecar(sidecar: Any) -> None:
    """Release the sidecar SQLite handle. No-op on None."""
    if sidecar is None:
        return
    try:
        sidecar.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("VaultSidecarIndex close error during CLI cleanup: %s", exc)


def _resolve_snapshot_path(user_path: str | None) -> Path:
    """Return snapshot path, defaulting to data/graph_snapshots/reset-<ts>.cypher."""
    if user_path:
        return Path(user_path)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return DEFAULT_SNAPSHOT_DIR / f"reset-{ts}.cypher"


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _print_extraction_result(result, mode: str) -> None:
    """Render a ValidationResult or CurationResult for the CLI."""
    # Prefer inner ValidationResult when present (CurationResult wraps it).
    inner = getattr(result, "validation_result", None)
    validation = inner if inner is not None else result
    entities = list(getattr(validation, "entities", []) or [])
    relationships = list(getattr(validation, "relationships", []) or [])
    warnings = list(getattr(validation, "warnings", []) or [])
    errors = list(getattr(validation, "errors", []) or [])

    print(f"\n[extract:{mode}] Entities ({len(entities)}):")
    if not entities:
        print("  (none)")
    for e in entities:
        # ValidationResult yields dicts; CurationResult may hold the same shape.
        if isinstance(e, dict):
            name = e.get("name") or e.get("id", "?")
            etype = e.get("type") or e.get("entity_type", "?")
            conf = (e.get("properties") or {}).get("confidence") or e.get("confidence")
        else:
            name = getattr(e, "name", None) or getattr(e, "id", "?")
            etype = getattr(e, "entity_type", None) or getattr(e, "type", "?")
            conf = getattr(e, "confidence", None)
        print(f"  - {name}  [{etype}]  conf={_fmt(conf)}")

    print(f"\n[extract:{mode}] Relationships ({len(relationships)}):")
    if not relationships:
        print("  (none)")
    for r in relationships:
        if isinstance(r, dict):
            subj = r.get("source") or r.get("subject", "?")
            pred = r.get("type") or r.get("predicate", "?")
            obj = r.get("target") or r.get("object", "?")
            conf = (r.get("properties") or {}).get("confidence") or r.get("confidence")
        else:
            subj = getattr(r, "source", None) or getattr(r, "subject", "?")
            pred = getattr(r, "type", None) or getattr(r, "predicate", "?")
            obj = getattr(r, "target", None) or getattr(r, "object", "?")
            conf = getattr(r, "confidence", None)
        print(f"  - {subj} -[{pred}]-> {obj}  conf={_fmt(conf)}")

    if warnings:
        print(f"\n[extract:{mode}] Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print(f"\n[extract:{mode}] Errors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")

    # CurationResult-only fields
    wr = getattr(result, "write_result", None)
    if wr is not None:
        rr = getattr(result, "reconcile_result", None)
        print(f"\n[extract:{mode}] Graph writes:")
        print(f"  entities_created:      {getattr(wr, 'entities_created', 0)}")
        print(f"  entities_updated:      {getattr(wr, 'entities_updated', 0)}")
        print(f"  relationships_appended: {getattr(rr, 'appended', 0)}")
        print(f"  relationships_closed:   {getattr(rr, 'closed', 0)}")
        print(f"  relationships_reinforced: {getattr(rr, 'reinforced', 0)}")
        print(f"  relationships_structural: {getattr(rr, 'structural', 0)}")
    curation_ms = getattr(result, "curation_time_ms", None)
    if curation_ms is not None:
        print(f"  curation_time_ms:      {curation_ms:.1f}")


def _print_retrieval_result(result, show_context: bool) -> None:
    """Render a RetrievalResult for the CLI."""
    print(f"\n[retrieve] Intent: {getattr(result, 'intent', '?')}")
    print(f"[retrieve] Entities matched (vector): {result.entities_found}")
    print(f"[retrieve] Total facts retrieved: {result.total_facts}")
    print(f"[retrieve] Document chunks used: {getattr(result, 'document_chunks_used', 0)}")
    print("\n[retrieve] Timing:")
    print(f"  retrieval_time_ms:       {result.retrieval_time_ms:.1f}")
    print(f"  vector_search_time_ms:   {result.vector_search_time_ms:.1f}")
    print(f"  graph_traversal_time_ms: {result.graph_traversal_time_ms:.1f}")

    print("\n[retrieve] Facts:")
    if not result.facts:
        print("  (none)")
    for fact in result.facts:
        sim = getattr(fact, "similarity_score", None)
        dist = getattr(fact, "graph_distance", None)
        print(
            f"  - {fact.subject} -[{fact.predicate}]-> {fact.object}  "
            f"sim={_fmt(sim)} hops={dist}"
        )

    if show_context:
        ctx = getattr(result, "formatted_context", "") or ""
        print("\n[retrieve] Formatted context:")
        if ctx:
            for line in ctx.splitlines():
                print(f"  {line}")
        else:
            print("  (empty)")


def _print_chat_result(result: dict) -> None:
    """Render a run_chat result for CLI output."""
    mark = "OK" if result["ok"] else "FAIL"
    print(f"\n[chat:{mark}] duration_ms={result['duration_ms']:.1f}")
    if result["ok"]:
        response = result["response"] or ""
        print(f"[chat:{mark}] response:")
        for line in response.splitlines() or [""]:
            print(f"  {line}")
    else:
        print(f"[chat:{mark}] error: {result['error']}")


def _print_replay_summary(results: list[dict]) -> None:
    """Render a replay batch summary for CLI output."""
    if not results:
        print("[replay] (no results)")
        return
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - ok_count
    avg_ms = sum(r["duration_ms"] for r in results) / len(results)
    p50_ms = sorted(r["duration_ms"] for r in results)[len(results) // 2]
    print(
        f"\n[replay] Results: {ok_count}/{len(results)} ok, "
        f"{fail_count} failed, avg {avg_ms:.1f}ms, p50 {p50_ms:.1f}ms"
    )
    for i, r in enumerate(results, start=1):
        mark = "OK" if r["ok"] else "FAIL"
        tag = r.get("tag", "")
        body = r["response"] if r["ok"] else r.get("error", "")
        preview = (body or "").replace("\n", " ")[:60]
        print(f"  [{i:>2}] [{mark:<4}] {tag:<22} " f"{r['duration_ms']:>8.1f}ms  {preview!r}")


def _print_extraction_only_summary(results: list[dict]) -> None:
    """Render an extraction-only replay batch summary for CLI output.

    Mirrors `_print_replay_summary` but reports `extraction_duration_ms`
    (extraction + drain wall time) rather than chat-response latency, since the
    extraction-only path never generates a reply.
    """
    if not results:
        print("[replay] (no results)")
        return
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - ok_count
    durations = [r.get("extraction_duration_ms", 0.0) for r in results]
    avg_ms = sum(durations) / len(results)
    p50_ms = sorted(durations)[len(results) // 2]
    print(
        f"\n[replay] Extraction-only results: {ok_count}/{len(results)} ok, "
        f"{fail_count} failed, avg {avg_ms:.1f}ms, p50 {p50_ms:.1f}ms"
    )
    for i, r in enumerate(results, start=1):
        mark = "OK" if r["ok"] else "FAIL"
        tag = r.get("tag", "")
        detail = "" if r["ok"] else (r.get("error", "") or "")
        preview = detail.replace("\n", " ")[:60]
        print(
            f"  [{i:>2}] [{mark:<4}] {tag:<22} "
            f"{r.get('extraction_duration_ms', 0.0):>8.1f}ms  {preview!r}"
        )


def _print_status_line(status: dict) -> None:
    service = status.get("service", "?")
    state = status.get("status", "?")
    indicator = "OK" if state == "healthy" else "FAIL"
    details: list[str] = []
    if "url" in status:
        details.append(status["url"])
    if "uri" in status:
        details.append(status["uri"])
    if "entity_count" in status:
        details.append(f"entities={status['entity_count']}")
    if "error" in status:
        details.append(f"error={status['error']}")
    detail_str = "  ".join(details)
    print(f"  [{indicator}] {service:<8} {state:<16} {detail_str}")


# ---------------------------------------------------------------------------
# Arg parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mist_admin",
        description="MIST knowledge graph admin CLI (Tier 1 + Tier 2 + Tier 3).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="SUBCOMMAND")

    p_seed = sub.add_parser("seed", help="Wipe-then-apply the versioned seed source idempotently.")
    p_seed.add_argument(
        "--seed-dir",
        default=None,
        help="Path to the seed source directory (default: <vault.root>/seed).",
    )
    p_seed.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation (faster seed; vector retrieval will miss).",
    )
    p_seed.add_argument(
        "--no-vault-bootstrap",
        action="store_true",
        help=(
            "Skip vault bootstrap (Phase 10). When omitted and "
            "config.vault.enabled is True, also writes identity/mist.md "
            "and users/<id>.md from the seed documents' bodies."
        ),
    )
    p_seed.set_defaults(func=cmd_seed)

    p_seed_verify = sub.add_parser(
        "seed-verify",
        help=(
            "Run the five gates (facts-present, node-definitions, containment, "
            "negation-proximity, embeddings) on the seed source."
        ),
    )
    p_seed_verify.add_argument(
        "--seed-dir",
        default=None,
        help="Path to the seed source directory (default: <vault.root>/seed).",
    )
    p_seed_verify.set_defaults(func=cmd_seed_verify)

    p_dump = sub.add_parser("graph-dump", help="Dump the __Entity__ subgraph.")
    p_dump.add_argument(
        "--format",
        choices=["json", "cypher"],
        default="json",
        help="Output format (default: json).",
    )
    p_dump.add_argument(
        "--output",
        default=None,
        help="Write to file instead of stdout.",
    )
    p_dump.add_argument(
        "--include-provenance",
        action="store_true",
        default=False,
        dest="include_provenance",
        help=(
            "Also emit the :__Provenance__ subgraph and cross-layer edges. "
            "Adds 'provenance' and 'cross_layer_edges' keys to JSON output; "
            "appends two additional sections to Cypher output."
        ),
    )
    p_dump.set_defaults(func=cmd_graph_dump)

    p_stats = sub.add_parser("graph-stats", help="Print node/rel counts and health.")
    p_stats.set_defaults(func=cmd_graph_stats)

    p_reset = sub.add_parser("graph-reset", help="Wipe graph with safety guards.")
    p_reset.add_argument("--confirm", action="store_true", help="Execute the wipe.")
    p_reset.add_argument("--dry-run", action="store_true", help="Preview without writing.")
    p_reset.add_argument(
        "--snapshot-to",
        default=None,
        help=f"Snapshot path (default: {DEFAULT_SNAPSHOT_DIR}/reset-<ts>.cypher).",
    )
    p_reset.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip pre-wipe snapshot (destructive without backup).",
    )
    p_reset.add_argument(
        "--include-derived",
        action="store_true",
        help="Allow wiping entities whose provenance is not 'seed'.",
    )
    p_reset.set_defaults(func=cmd_graph_reset)

    p_backfill = sub.add_parser(
        "graph-backfill-bitemporal",
        help="One-shot idempotent C1 backfill: stamp bitemporal fields on legacy edges.",
    )
    p_backfill.add_argument("--dry-run", action="store_true", help="Count candidates only.")
    p_backfill.set_defaults(func=cmd_graph_backfill_bitemporal)

    p_status = sub.add_parser("stack-status", help="Probe Neo4j + LLM + backend health.")
    p_status.add_argument(
        "--backend-url",
        default=None,
        help="Backend URL (default: http://localhost:8001).",
    )
    p_status.set_defaults(func=cmd_stack_status)

    p_extract = sub.add_parser(
        "extract",
        help="Run extraction pipeline on an utterance (dry-run by default).",
    )
    p_extract.add_argument("utterance", help="User utterance to extract from.")
    p_extract.add_argument(
        "--commit",
        action="store_true",
        help="Include curation + internal derivation; writes to graph. Default is dry-run.",
    )
    p_extract.add_argument(
        "--session-id",
        default=None,
        help="Session identifier (default: admin-cli).",
    )
    p_extract.set_defaults(func=cmd_extract)

    p_retrieve = sub.add_parser(
        "retrieve",
        help="Run hybrid (graph + vector) retrieval for a query.",
    )
    p_retrieve.add_argument("query", help="Natural-language query.")
    p_retrieve.add_argument(
        "--user-id",
        default="User",
        help="User identifier scoping the retrieval (default: User).",
    )
    p_retrieve.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum facts to retrieve (default: 10).",
    )
    p_retrieve.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold (default: 0.7).",
    )
    p_retrieve.add_argument(
        "--show-context",
        action="store_true",
        help="Also print the LLM-facing formatted_context string.",
    )
    p_retrieve.set_defaults(func=cmd_retrieve)

    p_chat = sub.add_parser(
        "chat",
        help="Full end-to-end chat turn (retrieval + LLM + extraction + graph).",
    )
    p_chat.add_argument("utterance", help="User utterance to send.")
    p_chat.add_argument(
        "--session-id",
        default=None,
        help="Session identifier (default: admin-cli-<hash>).",
    )
    p_chat.add_argument(
        "--user-id",
        default="User",
        help="User identifier (default: User).",
    )
    p_chat.add_argument(
        "--output",
        default=None,
        help="Write the JSON result to this path in addition to stdout.",
    )
    p_chat.set_defaults(func=cmd_chat)

    p_replay = sub.add_parser(
        "replay",
        help="Replay utterances from a JSONL or plain-text file through chat.",
    )
    p_replay.add_argument(
        "input",
        help="Path to .jsonl (one object per line) or .txt (one utterance per line).",
    )
    p_replay.add_argument(
        "--session-id",
        default=None,
        help="Shared session identifier (default: replay-<hash>). Can be "
        "overridden per-line via a session_id field in JSONL input.",
    )
    p_replay.add_argument(
        "--user-id",
        default="User",
        help="User identifier (default: User).",
    )
    p_replay.add_argument(
        "--output",
        default=None,
        help="Write per-turn results as JSONL to this path.",
    )
    p_replay.add_argument(
        "--extraction-only",
        action="store_true",
        dest="extraction_only",
        help=(
            "Drive each utterance through the production extraction pipeline "
            "WITHOUT generating a chat reply (deterministic F2 measurement "
            "path). Emits the same extraction.* debug records the F2 scorer "
            "consumes. The chat reply is conversational noise the F2 gold does "
            "not encode and is the sole source of F2 nondeterminism."
        ),
    )
    p_replay.set_defaults(func=cmd_replay)

    # ---- Cluster 8 Phase 11: vault subcommands -----------------------------
    p_vstatus = sub.add_parser(
        "vault-status",
        help="Report vault layer config + sidecar chunk count + on-disk note count.",
    )
    p_vstatus.set_defaults(func=cmd_vault_status)

    p_vreindex = sub.add_parser(
        "vault-reindex",
        help="Walk vault root and re-index every .md into the sidecar.",
    )
    p_vreindex.add_argument(
        "--scope",
        default=None,
        help=(
            "Optional path to a single vault note. When provided only that "
            "file is re-indexed. When omitted, all .md files under the vault "
            "root are re-indexed."
        ),
    )
    p_vreindex.set_defaults(func=cmd_vault_reindex)

    p_vrebuild = sub.add_parser(
        "vault-rebuild",
        help="Drop the sidecar (vec0 + FTS5 + main table) and re-index from disk.",
    )
    p_vrebuild.add_argument(
        "--confirm",
        action="store_true",
        help="Required to actually drop + rebuild the sidecar. Without --confirm "
        "the rebuild previews the work and exits without writing.",
    )
    p_vrebuild.set_defaults(func=_cmd_vault_rebuild_sidecar)

    p_vmigrate = sub.add_parser(
        "vault-migrate",
        help="Apply registered ontology / schema migrations to vault notes.",
    )
    p_vmigrate.add_argument(
        "--target-version",
        default=None,
        help="Target ontology version. Defaults to current config.ontology_version.",
    )
    p_vmigrate.set_defaults(func=cmd_vault_migrate)

    # ---- R1.2: graph rebuild from log (proof-first, dry-run only) ----------
    p_rebuild = sub.add_parser(
        "graph-rebuild-from-log",
        help=(
            "Proof-first: rebuild a staging graph from the event log; "
            "prove determinism (dry-run)."
        ),
    )
    p_rebuild.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Build staging + run gates; NO cutover (the only R1.2 mode).",
    )
    p_rebuild.add_argument(
        "--staging-uri",
        default="bolt://mist-neo4j-staging:7687",
        help="Staging Neo4j bolt URI (must NOT be live).",
    )
    p_rebuild.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch id (default: current).",
    )
    p_rebuild.set_defaults(func=cmd_graph_rebuild_from_log)

    return parser


# ---------------------------------------------------------------------------
# Cluster 8 Phase 11 -- vault CLI subcommands
# ---------------------------------------------------------------------------


def _walk_vault_md_files(vault_root: Path) -> list[Path]:
    """Return every `.md` file under `vault_root` in a deterministic order.

    Skips hidden files and `.git/`. Matches the file filter applied by
    VaultFilewatcher so the on-disk corpus and the live-reindex corpus
    agree.
    """
    if not vault_root.exists():
        return []
    out: list[Path] = []
    for p in vault_root.rglob("*.md"):
        if any(part.startswith(".") for part in p.relative_to(vault_root).parts):
            continue
        out.append(p)
    return sorted(out)


def _read_vault_note(path: Path) -> tuple[str, dict | None, int]:
    """Read a vault note file and return (content_after_frontmatter, frontmatter, mtime).

    Frontmatter parsing uses the same `parse_frontmatter` helper the
    VaultWriter uses, so this is symmetric with the live write path.
    Returns (raw_text, None, mtime) when frontmatter parsing fails -- the
    sidecar can still index the body in that case.
    """
    from backend.vault.models import parse_frontmatter

    raw = path.read_text(encoding="utf-8")
    mtime = int(path.stat().st_mtime)
    try:
        frontmatter, body = parse_frontmatter(raw)
    except Exception:  # noqa: BLE001 -- corrupted frontmatter shouldn't block reindex
        return raw, None, mtime
    return body, frontmatter, mtime


def cmd_vault_status(args: argparse.Namespace) -> int:
    """Report vault layer health: config, on-disk note count, sidecar chunk count.

    Read-only diagnostic. Builds a transient sidecar instance to call
    `chunk_count()` and `health_check()`; closes it before returning so
    no SQLite connections leak. Does not require Neo4j to be reachable.
    """
    be = _load_backend()
    config = be.get_config()

    print("[vault-status]")
    print(f"  config.vault.enabled:           {config.vault.enabled}")
    print(f"  config.vault.root:              {config.vault.root}")
    print(f"  config.sidecar_index.enabled:   {config.sidecar_index.enabled}")
    print(f"  config.sidecar_index.db_path:   {config.sidecar_index.db_path}")
    print(f"  config.filewatcher.enabled:     {config.filewatcher.enabled}")
    print(f"  config.filewatcher.observer:    {config.filewatcher.observer_type}")
    print(f"  config.filewatcher.debounce_ms: {config.filewatcher.debounce_ms}")

    if not config.vault.enabled:
        print("\n[vault-status] Vault layer disabled; nothing else to report.")
        return 0

    vault_root = Path(config.vault.root)
    md_files = _walk_vault_md_files(vault_root)
    print(f"\n  on-disk .md files:              {len(md_files)}")
    if md_files:
        print(f"    first: {md_files[0]}")
        print(f"    last:  {md_files[-1]}")

    if not config.sidecar_index.enabled:
        print("\n[vault-status] Sidecar disabled; skipping chunk counts.")
        return 0

    from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
    from backend.vault.sidecar_index import VaultSidecarIndex

    embedding_provider = EmbeddingGenerator(config.embedding.model_name)
    sidecar = VaultSidecarIndex(config.sidecar_index, embedding_provider)
    try:
        sidecar.initialize()
        chunk_count = sidecar.chunk_count()
        healthy = sidecar.health_check()
    finally:
        sidecar.close()

    print(f"\n  sidecar chunk_count:            {chunk_count}")
    print(f"  sidecar health_check:           {'OK' if healthy else 'DEGRADED'}")
    return 0


def cmd_vault_reindex(args: argparse.Namespace) -> int:
    """Walk the vault root and re-index every `.md` into the sidecar.

    With `--scope <path>`, only that single file is re-indexed. Without
    `--scope`, every `.md` under the vault root is re-indexed.

    Idempotent via the sidecar's MERGE semantics. Useful after editing
    notes outside MIST (e.g. via Obsidian) when filewatcher events were
    missed (Windows ReadDirectoryChangesW overflow).
    """
    be = _load_backend()
    config = be.get_config()

    if not config.vault.enabled or not config.sidecar_index.enabled:
        print("[vault-reindex] Vault or sidecar disabled; nothing to do.")
        return 0

    vault_root = Path(config.vault.root)

    if args.scope:
        scope_path = Path(args.scope).resolve()
        if not scope_path.exists():
            print(f"[vault-reindex] Scope file not found: {scope_path}")
            return 1
        targets = [scope_path]
    else:
        targets = _walk_vault_md_files(vault_root)

    if not targets:
        print("[vault-reindex] No .md files to index.")
        return 0

    from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
    from backend.vault.sidecar_index import VaultSidecarIndex

    embedding_provider = EmbeddingGenerator(config.embedding.model_name)
    sidecar = VaultSidecarIndex(config.sidecar_index, embedding_provider)
    sidecar.initialize()

    total_chunks = 0
    failures: list[tuple[Path, str]] = []
    try:
        for path in targets:
            try:
                body, frontmatter, mtime = _read_vault_note(path)
                chunks = sidecar.upsert_file(
                    path=str(path),
                    content=body,
                    mtime=mtime,
                    frontmatter=frontmatter,
                )
                total_chunks += chunks
            except Exception as exc:  # noqa: BLE001 -- per-file isolation
                failures.append((path, repr(exc)))
                print(f"[vault-reindex] FAIL {path}: {exc}")
    finally:
        sidecar.close()

    print("\n[vault-reindex] Done.")
    print(f"  files processed: {len(targets)}")
    print(f"  failures:        {len(failures)}")
    print(f"  total chunks:    {total_chunks}")
    return 0 if not failures else 1


def _cmd_vault_rebuild_sidecar(args: argparse.Namespace) -> int:
    """Drop the sidecar tables and re-index every vault note from disk.

    Heavier than `vault-reindex`: clears all chunks first so per-file
    upserts cannot accumulate stale rows from notes that have since been
    deleted. Use after schema changes to the sidecar or after suspected
    corruption.

    Requires `--confirm` because this drops the sidecar's contents
    (re-buildable from disk, but still destructive on the SQLite file).

    This is the sole `vault-rebuild` handler. The graph-aware --scope and
    --retry-orphaned modes retired with R1.3: a vault edit no longer produces
    graph facts, so there is no graph subgraph to rebuild from a vault file.
    Graph rebuilds now run from the utterance log via `graph-rebuild-from-log`.
    """
    be = _load_backend()
    config = be.get_config()

    if not config.vault.enabled or not config.sidecar_index.enabled:
        print("[vault-rebuild] Vault or sidecar disabled; nothing to do.")
        return 0

    vault_root = Path(config.vault.root)
    targets = _walk_vault_md_files(vault_root)

    if not args.confirm:
        print("[vault-rebuild] DRY RUN -- pass --confirm to execute.")
        print(f"  vault_root:       {vault_root}")
        print(f"  sidecar db:       {config.sidecar_index.db_path}")
        print(f"  files to reindex: {len(targets)}")
        return 0

    print(f"[vault-rebuild] Dropping sidecar at {config.sidecar_index.db_path}")
    sidecar_db = Path(config.sidecar_index.db_path)
    if sidecar_db.exists():
        sidecar_db.unlink()

    from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
    from backend.vault.sidecar_index import VaultSidecarIndex

    embedding_provider = EmbeddingGenerator(config.embedding.model_name)
    sidecar = VaultSidecarIndex(config.sidecar_index, embedding_provider)
    sidecar.initialize()  # Recreates schema on the freshly-deleted db

    total_chunks = 0
    failures: list[tuple[Path, str]] = []
    try:
        for path in targets:
            try:
                body, frontmatter, mtime = _read_vault_note(path)
                chunks = sidecar.upsert_file(
                    path=str(path),
                    content=body,
                    mtime=mtime,
                    frontmatter=frontmatter,
                )
                total_chunks += chunks
            except Exception as exc:  # noqa: BLE001
                failures.append((path, repr(exc)))
                print(f"[vault-rebuild] FAIL {path}: {exc}")
    finally:
        sidecar.close()

    print("\n[vault-rebuild] Done.")
    print(f"  files processed: {len(targets)}")
    print(f"  failures:        {len(failures)}")
    print(f"  total chunks:    {total_chunks}")
    return 0 if not failures else 1


def cmd_vault_migrate(args: argparse.Namespace) -> int:
    """Apply registered vault schema / ontology migrations.

    Placeholder for ADR-010's "Ontology evolution" path. The current
    ontology is v1.0.0 and there are no registered migrations. When a
    future ontology bump (e.g. 1.1.0) ships, the migration script lives
    here. Today this command is a structural no-op that reports the
    current versions and exits successfully so operators can wire it
    into runbooks now.
    """
    be = _load_backend()
    config = be.get_config()

    target = args.target_version or config.ontology_version

    print("[vault-migrate]")
    print(f"  config.ontology_version:    {config.ontology_version}")
    print(f"  config.extraction_version:  {config.extraction_version}")
    print(f"  target_version:             {target}")
    print("\n  registered migrations:      (none)")
    if target != config.ontology_version:
        print(
            f"\n[vault-migrate] No migration path registered for "
            f"{config.ontology_version} -> {target}. No-op."
        )
    else:
        print("\n[vault-migrate] Already at target version. No-op.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Lazy MistError import so `--help` works without the neo4j driver installed.
    try:
        from backend.errors import MistError
    except ModuleNotFoundError as e:
        print(
            f"[error] Missing dependency: {e}. Install with "
            "`pip install -r requirements.txt` from the MIST repo root.",
            file=sys.stderr,
        )
        return 1

    try:
        return args.func(args)
    except ModuleNotFoundError as e:
        print(
            f"[error] Missing dependency: {e}. Install with "
            "`pip install -r requirements.txt` from the MIST repo root.",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    except MistError as e:
        print(f"[error] {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
