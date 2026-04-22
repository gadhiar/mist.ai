"""MIST Admin CLI — Tier 1, Tier 2, and Tier 3 operations.

Thin wrapper around backend.knowledge.admin functions and the production DI
composition root in backend.factories so the CLI exercises the same code paths
as the running backend.

Tier 1 subcommands (graph operations):
    seed                                    Apply seed_data.yaml idempotently.
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
from typing import Any

# Make `backend` importable when running from the host (mist-ai is not pip-installed).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REPO_ROOT = _REPO_ROOT
DEFAULT_SEED_PATH = REPO_ROOT / "scripts" / "seed_data.yaml"
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
    be = _load_backend()
    seed_path = Path(args.seed_file) if args.seed_file else DEFAULT_SEED_PATH
    print(f"[seed] Loading {seed_path}")
    seed_data = be.admin.load_seed_yaml(seed_path)

    embedding_generator = None
    if not args.no_embeddings:
        from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator

        config = be.get_config()
        print(f"[seed] Loading embedding model: {config.embedding.model_name}")
        embedding_generator = EmbeddingGenerator(model_name=config.embedding.model_name)

    connection = _connect(be)
    try:
        counts = be.admin.apply_seed(connection, seed_data, embedding_generator=embedding_generator)
    finally:
        connection.disconnect()
    print("[seed] Applied (idempotent MERGE):")
    for layer, count in counts.items():
        print(f"  {layer}: {count}")
    print(f"[seed] Total writes: {sum(counts.values())}")

    # ADR-010 Cluster 8 Phase 10: vault bootstrap. Mirrors the seeded
    # identity/user data into the vault as canonical markdown notes and
    # emits DERIVED_FROM edges from each seeded entity to its source vault
    # note. Disabled with --no-vault-bootstrap. Skipped automatically when
    # the vault subsystem is disabled in config.
    if not getattr(args, "no_vault_bootstrap", False):
        config = be.get_config()
        if config.vault.enabled:
            _do_vault_bootstrap(be, config, seed_data)
        else:
            print("[seed] Vault bootstrap skipped: config.vault.enabled is False")

    return 0


def _do_vault_bootstrap(be: Any, config: Any, seed_data: dict[str, Any]) -> None:
    """Run the vault bootstrap step for `cmd_seed` (Phase 10).

    Builds and starts a VaultWriter, writes identity/mist.md +
    users/<id>.md from seed_data, then emits DERIVED_FROM edges from each
    seeded entity to its bootstrap note. All vault operations are
    idempotent so re-running `seed` is safe. Vault errors are logged but
    never propagate -- graph seed already succeeded by the time this runs.
    """
    import asyncio

    from backend.vault.writer import VaultWriter

    print("[seed] Vault bootstrap: writing identity/mist.md + users/<id>.md")

    async def _run() -> dict[str, str]:
        writer = VaultWriter(config.vault)
        await writer.start()
        try:
            return await be.admin.bootstrap_vault_from_seed(writer, seed_data)
        finally:
            await writer.stop()

    try:
        paths = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 -- ADR-010 Invariant 6
        print(f"[seed] Vault bootstrap failed (graph seed unaffected): {exc}")
        return

    print(f"[seed]   identity_path: {paths['identity_path']}")
    print(f"[seed]   user_path:     {paths['user_path']}")

    connection = _connect(be)
    try:
        edges = be.admin.emit_seed_vault_provenance(
            connection,
            seed_data,
            identity_path=paths["identity_path"],
            user_path=paths["user_path"],
        )
    finally:
        connection.disconnect()

    print(f"[seed] Vault bootstrap: wrote {edges} DERIVED_FROM edges to bootstrap notes")


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
    """Run hybrid retrieval on a query and print facts with scores + timing."""
    be = _load_backend()
    _, build_knowledge_retriever = _load_factories()
    config = be.get_config()
    print(
        f"[retrieve] query={args.query!r} user_id={args.user_id} "
        f"limit={args.limit} threshold={args.threshold}"
    )
    retriever = build_knowledge_retriever(config)
    result = asyncio.run(
        retriever.retrieve(
            query=args.query,
            user_id=args.user_id,
            limit=args.limit,
            similarity_threshold=args.threshold,
        )
    )
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
    """Full end-to-end chat turn through the production ConversationHandler."""
    be = _load_backend()
    from backend.factories import build_conversation_handler

    config = be.get_config()
    session_id = args.session_id or f"admin-cli-{uuid.uuid4().hex[:8]}"
    user_id = args.user_id or "User"
    print(f"[chat] session_id={session_id} user_id={user_id}")
    print(f"[chat] utterance: {args.utterance!r}")
    print("[chat] Building conversation handler (may load embedding model)...")
    handler = build_conversation_handler(config)

    result = asyncio.run(run_chat(handler, args.utterance, session_id, user_id))
    _print_chat_result(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"\n[chat] Wrote result to {out_path}")
    return 0 if result["ok"] else 1


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay utterances from a file through the chat path; aggregate results."""
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
    print(f"[replay] {len(inputs)} inputs, session_id={session_id}, user_id={user_id}")
    print("[replay] Building conversation handler (may load embedding model)...")
    handler = build_conversation_handler(config)

    results = asyncio.run(run_replay(handler, inputs, session_id, user_id))
    _print_replay_summary(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"\n[replay] Wrote {len(results)} records to {out_path}")

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
        print(f"\n[extract:{mode}] Graph writes:")
        print(f"  entities_created:      {getattr(wr, 'entities_created', 0)}")
        print(f"  entities_updated:      {getattr(wr, 'entities_updated', 0)}")
        print(f"  relationships_created: {getattr(wr, 'relationships_created', 0)}")
        print(f"  relationships_updated: {getattr(wr, 'relationships_updated', 0)}")
        print(f"  relationships_superseded: {getattr(wr, 'relationships_superseded', 0)}")
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

    p_seed = sub.add_parser("seed", help="Apply seed_data.yaml idempotently.")
    p_seed.add_argument(
        "--seed-file",
        default=None,
        help=f"Path to seed YAML (default: {DEFAULT_SEED_PATH}).",
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
            "and users/<id>.md and emits seed DERIVED_FROM edges."
        ),
    )
    p_seed.set_defaults(func=cmd_seed)

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
    p_replay.set_defaults(func=cmd_replay)

    return parser


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
