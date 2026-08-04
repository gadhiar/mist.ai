"""Dump the hydrated dev stack to a versioned artifact, and restore it in seconds.

R1.4.6 T4. Hydration costs 87 LLM turns. That is affordable once per ontology
bump and unaffordable once per session, so the artifact -- not the run -- is
what a developer actually consumes. A fixture nobody restores is a fixture
nobody uses.

WHAT IS CAPTURED
    graph.json          every node and relationship, plus the constraint/index
                        DDL, read back out of the dev Neo4j
    <name>.db           each SQLite store found at the dev root (event store,
                        extraction cache, vault sidecar)
    vault/              the dev vault tree
    manifest.json       the producer identity (see manifest.py)

WHY A DIRECTORY OF TEXT AND NOT A NEO4J DUMP
    `neo4j-admin database dump` is the standard tool and the wrong one here. It
    requires the database stopped, it is coupled to the server version that
    wrote it, and its output is opaque. The graph this fixture holds is small
    (hundreds of nodes), and what a reviewer needs from a regenerated artifact
    is to SEE what changed -- an ontology bump that alters two edge properties
    should show up as a two-line diff, not as a different binary blob. So the
    graph leg is canonical JSON, sorted, with content-derived node keys, and the
    whole artifact is diffable.

WHY sqlite3 .backup() AND NOT A FILE COPY
    The stores run in WAL mode. Copying `foo.db` alone silently drops whatever
    is still in `foo.db-wal`, producing an artifact that is missing the most
    recent turns -- the ones a hydration run just wrote. `Connection.backup()`
    takes a read lock and produces one consistent file.

WHY RESTORE REFUSES A NON-ISOLATED TARGET
    Restore is destructive twice over: it detach-deletes every node in the
    target graph, and it replaces the store files under the target root. Both
    guards are fail-closed and neither has an off switch:
      - the graph target must be in the dev-endpoint ALLOWLIST
        (`assert_neo4j_dev_isolated`) -- the live bolt port is host-published,
        so `bolt://localhost:7687` is a live spelling that no denylist of
        service names would catch;
      - the state root must survive `assert_isolated_root`, which refuses a root
        that is, sits under, or CONTAINS a live store directory.

Usage:
    python -m scripts.hydration.snapshot dump    [--label r1.4.6-initial]
    python -m scripts.hydration.snapshot verify  --artifact data/hydration-snapshots/<label>
    python -m scripts.hydration.snapshot restore --artifact data/hydration-snapshots/<label>
    python -m scripts.hydration.snapshot list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neo4j.time import Date, DateTime, Duration, Time

from backend.errors import Neo4jQueryError
from backend.knowledge.config import Neo4jConfig
from backend.knowledge.eval_isolation import (
    EvalIsolationError,
    IsolatedRootError,
    assert_isolated_root,
    assert_neo4j_dev_isolated,
    live_state_roots,
)
from backend.knowledge.storage.neo4j_connection import Neo4jConnection

from .manifest import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_CORPUS_PATH,
    REPO_ROOT,
    HydrationError,
    SnapshotIdentity,
    SnapshotManifest,
    assert_fresh,
)

logger = logging.getLogger(__name__)

# `REPO_ROOT / "dev-state"` is one expression that is correct in both places
# this runs: `/app/dev-state` inside mist-backend-dev, `<repo>/dev-state` on the
# host. Overridable for a second dev stack, but never to a live path -- the
# override goes through `assert_isolated_root` like everything else.
DEFAULT_DEV_ROOT = REPO_ROOT / "dev-state"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "data" / "hydration-snapshots"
DEFAULT_DEV_NEO4J_URI = "bolt://mist-neo4j-dev:7687"

GRAPH_FILENAME = "graph.json"
VAULT_DIRNAME = "vault"

# Marks a tagged temporal in graph.json. Neo4j has no map property type, so a
# dict in a property position can only ever be one of these.
NEO4J_TYPE_TAG = "__neo4j_type__"
_TEMPORAL_TYPES = (Date, Time, DateTime, Duration)
_TEMPORAL_DECODERS = {cls.__name__: cls.from_iso_format for cls in _TEMPORAL_TYPES}

# Temp label + property used to re-link relationships to their endpoints during
# a restore. Both are stripped before the restore returns; `verify` fails if any
# survive, because a leftover marker means the load aborted mid-way.
RESTORE_LABEL = "__RestoreKey__"
RESTORE_KEY_PROP = "__restore_key__"
RESTORE_INDEX_NAME = "restore_key_tmp"

# Rows per write transaction. Entity nodes carry a 384-float embedding, so the
# batch is sized for payload rather than row count.
NODE_BATCH = 200
REL_BATCH = 500

# Counted into the manifest when present. Absent tables are skipped rather than
# raising: the set of stores grows, and a snapshot tool that fails on a table it
# has not heard of would block the phase it exists to serve.
COUNTED_TABLES = (
    "conversation_sessions",
    "conversation_turn_events",
    "epoch_ledger",
    "curation_job_runs",
    "graph_health_events",
    "materialized_graph_registry",
    "extraction_cache",
    "vault_chunks",
)


@dataclass(frozen=True, slots=True)
class SnapshotReport:
    """What a dump or restore actually moved."""

    artifact_dir: Path
    nodes: int
    relationships: int
    databases: tuple[str, ...]
    vault_files: int


def _canonical(value: Any) -> str:
    """Canonical JSON for hashing and for byte-stable artifact output."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _encode_value(value: Any, key: str, where: str) -> Any:
    """Encode one property value into JSON, losslessly or not at all.

    Neo4j temporals have no JSON form, and they are NOT hypothetical: the
    self-model seeding path writes `created_at` as a zoned datetime, so a real
    hydrated graph contains them on the first boot. Stringifying one would
    restore a graph whose `created_at` is text where the original was a
    temporal -- silently changing a property's TYPE is the class of defect this
    phase exists to remove, so they are tagged instead:

        DateTime(...) -> {"__neo4j_type__": "DateTime", "iso": "2026-08-04T..."}

    `str()` / `from_iso_format` round-trips all four temporal types exactly,
    nanosecond precision and timezone included (verified against the driver).
    The tag cannot collide with a genuine property: Neo4j has no map property
    type, so a dict here can only be one of ours.

    Anything else -- spatial Points today -- still refuses, because an encoding
    whose round trip has not been verified is worse than a refusal.
    """
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, _TEMPORAL_TYPES):
        return {NEO4J_TYPE_TAG: type(value).__name__, "iso": str(value)}
    if isinstance(value, list):
        return [_encode_value(item, key, where) for item in value]
    raise HydrationError(
        f"{where}: property {key!r} has type {type(value).__name__}, which this "
        "artifact format cannot round-trip. Coercing it would restore a graph that "
        "differs from the one captured. Extend the artifact format rather than "
        "losing it."
    )


def _encode_props(props: dict[str, Any], where: str) -> dict[str, Any]:
    """Encode a whole property map. Runs before any canonicalization.

    Ordering matters: `_canonical` is `json.dumps`, so it raises a raw
    TypeError on an unencodable value. Encoding first means the operator sees
    this module's explanation rather than a traceback from the sort key.
    """
    return {key: _encode_value(value, key, where) for key, value in props.items()}


def _decode_value(value: Any) -> Any:
    """Invert `_encode_value` so restored properties keep their original types."""
    if isinstance(value, dict) and NEO4J_TYPE_TAG in value:
        type_name = value[NEO4J_TYPE_TAG]
        decoder = _TEMPORAL_DECODERS.get(type_name)
        if decoder is None:
            raise HydrationError(
                f"artifact carries an unknown tagged value type {type_name!r}; this "
                "tree cannot restore it without changing the property's type"
            )
        return decoder(value["iso"])
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _decode_props(props: dict[str, Any]) -> dict[str, Any]:
    """Decode a whole property map read back out of the artifact."""
    return {key: _decode_value(value) for key, value in props.items()}


def _node_key(labels: list[str], props: dict[str, Any]) -> str:
    """Content-derived, stable across dumps -- unlike `elementId`.

    A dump keyed on `elementId` would differ byte-for-byte after every restore
    even for an identical graph, which defeats the point of a diffable artifact.
    """
    payload = _canonical({"labels": sorted(labels), "props": props})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _quote_ident(name: str) -> str:
    """Backtick-quote a label or relationship type read back from the database.

    The values come from the server, not from user input, but a backtick inside
    one would still break out of the quoting and is refused rather than escaped.
    """
    if "`" in name or not name:
        raise HydrationError(f"refusing unquotable label/type {name!r}")
    return f"`{name}`"


def _label_clause(labels: tuple[str, ...] | list[str]) -> str:
    """Render a multi-label pattern body, e.g. ``__Entity__`:`Person``.

    Its own function because the separator is load-bearing and silent when
    wrong. Joining quoted labels with "" yields `` `A``B` ``, which Cypher reads
    as ONE label literally named ``A`B`` -- a doubled backtick is the escape for
    a literal one inside a quoted identifier. The CREATE then succeeds, the
    nodes come back with a single nonsense label, and the failure surfaces much
    later as relationships whose endpoints do not resolve.
    """
    return ":".join(_quote_ident(label) for label in labels)


# --------------------------------------------------------------------------
# graph leg
# --------------------------------------------------------------------------


def _read_schema_ddl(conn: Neo4jConnection) -> dict[str, list[str]]:
    """Capture constraint + index DDL as the server itself states it.

    Taken from `SHOW CONSTRAINTS`/`SHOW INDEXES` rather than re-derived from
    `GraphStore.initialize_schema`, for two reasons: it records the schema the
    dev stack ACTUALLY had (including the vector index at its real dimension),
    and it needs no embedding model just to construct a `GraphStore`.

    Constraint-owned indexes are excluded -- creating the constraint creates
    them -- as are LOOKUP indexes, which the server maintains itself, and this
    tool's OWN restore scaffolding index. That last exclusion is not hygiene: a
    dump taken after a restore would otherwise capture `restore_key_tmp` as if
    it were part of the fixture's schema, and the next restore would try to
    create an index it had just created, which the server rejects.
    """
    constraints = conn.execute_query(
        "SHOW CONSTRAINTS YIELD name, createStatement RETURN name, createStatement ORDER BY name"
    )
    indexes = conn.execute_query(
        "SHOW INDEXES YIELD name, type, owningConstraint, createStatement "
        "WHERE owningConstraint IS NULL AND type <> 'LOOKUP' AND name <> $scaffolding "
        "RETURN name, createStatement ORDER BY name",
        {"scaffolding": RESTORE_INDEX_NAME},
    )
    return {
        "constraints": [row["createStatement"] for row in constraints],
        "indexes": [row["createStatement"] for row in indexes],
    }


def _assert_no_restore_scaffolding(conn: Neo4jConnection) -> None:
    """Refuse to read a graph that a previous restore left half-marked.

    Leftover markers mean the last restore aborted between creating nodes and
    stripping the scaffolding, so the graph is a partial load. Snapshotting it
    would bake that partial state into an artifact and carry the marker label
    into every future restore.
    """
    rows = conn.execute_query(f"MATCH (n:{_quote_ident(RESTORE_LABEL)}) RETURN count(n) AS marked")
    marked = int(rows[0]["marked"]) if rows else 0
    if marked:
        raise HydrationError(
            f"{marked} nodes still carry the {RESTORE_LABEL} marker, so a previous "
            "restore did not finish and this graph is partially loaded. Re-run "
            "restore before snapshotting."
        )


def _read_graph(conn: Neo4jConnection) -> dict[str, Any]:
    """Read the whole graph into a canonical, byte-stable structure."""
    _assert_no_restore_scaffolding(conn)
    raw_nodes = conn.execute_query(
        "MATCH (n) RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props"
    )

    # Encode BEFORE sorting: the sort key canonicalizes, and canonicalizing an
    # unencodable value raises a bare TypeError from json instead of this
    # module's explanation of what it found and why it refused.
    encoded_nodes = [
        {
            "eid": row["eid"],
            "labels": sorted(row["labels"]),
            "props": _encode_props(dict(row["props"]), f"node {sorted(row['labels'])}"),
        }
        for row in raw_nodes
    ]

    by_element: dict[str, str] = {}
    seen: dict[str, int] = {}
    nodes: list[dict[str, Any]] = []
    for row in sorted(
        encoded_nodes, key=lambda r: (_canonical(r["labels"]), _canonical(r["props"]))
    ):
        labels = row["labels"]
        props = row["props"]
        base = _node_key(labels, props)
        ordinal = seen.get(base, 0)
        seen[base] = ordinal + 1
        # Two nodes identical in labels AND properties are interchangeable, so
        # the ordinal suffix keeps the artifact deterministic and preserves both
        # rather than collapsing them. Which duplicate a relationship re-attaches
        # to is then arbitrary; the graph restores isomorphic, and the warning
        # below makes the duplication visible, because in this ontology every
        # node carries a unique id and a duplicate is a real finding.
        if ordinal:
            logger.warning(
                "duplicate node content for labels %s (copy %d); relationship "
                "endpoints among identical nodes restore in arbitrary order",
                labels,
                ordinal + 1,
            )
        key = base if not ordinal else f"{base}#{ordinal}"
        by_element[row["eid"]] = key
        nodes.append({"key": key, "labels": labels, "props": props})

    raw_rels = conn.execute_query(
        "MATCH (a)-[r]->(b) RETURN elementId(a) AS start, elementId(b) AS end, "
        "type(r) AS type, properties(r) AS props"
    )
    relationships: list[dict[str, Any]] = []
    for row in raw_rels:
        relationships.append(
            {
                "type": row["type"],
                "start": by_element[row["start"]],
                "end": by_element[row["end"]],
                "props": _encode_props(dict(row["props"]), f"relationship {row['type']}"),
            }
        )
    relationships.sort(key=lambda r: (r["type"], r["start"], r["end"], _canonical(r["props"])))

    return {
        "schema": _read_schema_ddl(conn),
        "nodes": nodes,
        "relationships": relationships,
    }


def _clear_graph(conn: Neo4jConnection) -> int:
    """Detach-delete every node, in bounded batches.

    Batched rather than one `MATCH (n) DETACH DELETE n`, which builds the whole
    delete in one transaction and can exhaust heap on a graph large enough to be
    worth snapshotting.
    """
    deleted = 0
    while True:
        rows = conn.execute_write(
            "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS deleted"
        )
        batch = int(rows[0]["deleted"]) if rows else 0
        deleted += batch
        if batch == 0:
            return deleted


def _apply_schema_ddl(conn: Neo4jConnection, schema: dict[str, list[str]]) -> None:
    """Replay captured DDL, skipping anything already present BY NAME.

    Skipping by name rather than catching the server's "equivalent already
    exists" error: an exception-swallowing loop here would also swallow a
    genuinely malformed statement and leave the restore silently unindexed.
    """
    existing = {
        row["name"] for row in conn.execute_query("SHOW CONSTRAINTS YIELD name RETURN name")
    } | {row["name"] for row in conn.execute_query("SHOW INDEXES YIELD name RETURN name")}

    for statement in list(schema.get("constraints", [])) + list(schema.get("indexes", [])):
        name = _ddl_name(statement)
        if name and name in existing:
            continue
        conn.execute_write(statement)


def _ddl_name(statement: str) -> str | None:
    """Pull the object name out of a Neo4j `createStatement`.

    The name follows the CONSTRAINT/INDEX keyword, and its POSITION varies --
    which is the whole reason this scans instead of indexing. Neo4j 5 emits
    `CREATE CONSTRAINT `n` FOR ...` (name at token 2) but `CREATE RANGE INDEX
    `n` FOR ...` and `CREATE VECTOR INDEX `n` ...` (token 3). Reading token 2
    unconditionally returns the literal string "INDEX" for every index, so no
    index ever matches an existing name, and restore re-creates every one of
    them -- which Neo4j rejects with EquivalentSchemaRuleAlreadyExists.
    """
    parts = statement.split()
    if not parts or parts[0].upper() != "CREATE":
        return None
    for position, token in enumerate(parts):
        # Bare keyword only: an object actually NAMED "index" arrives
        # backtick-quoted and must not be mistaken for the keyword.
        if token.upper() in {"CONSTRAINT", "INDEX"} and "`" not in token:
            if position + 1 < len(parts):
                return parts[position + 1].strip("`")
            return None
    return None


def _write_graph(conn: Neo4jConnection, graph: dict[str, Any]) -> tuple[int, int]:
    """Load nodes then relationships, re-linking endpoints via a temp key.

    Labels and relationship types cannot be parameterized in Cypher, so nodes
    are grouped by label-set and relationships by type, and each group gets one
    statement with a quoted literal. This is deliberately plain Cypher: APOC
    would allow dynamic labels in a single statement, but the artifact then
    depends on a plugin, and a fixture whose restore fails on a stock Neo4j is
    a fixture with a footnote.
    """
    _apply_schema_ddl(conn, graph.get("schema", {}))
    conn.execute_write(
        f"CREATE INDEX {_quote_ident(RESTORE_INDEX_NAME)} IF NOT EXISTS "
        f"FOR (n:{_quote_ident(RESTORE_LABEL)}) ON (n.{RESTORE_KEY_PROP})"
    )

    by_labels: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for node in graph["nodes"]:
        by_labels.setdefault(tuple(node["labels"]), []).append(node)

    node_count = 0
    for labels, group in sorted(by_labels.items()):
        label_clause = _label_clause((*labels, RESTORE_LABEL))
        for start in range(0, len(group), NODE_BATCH):
            batch = group[start : start + NODE_BATCH]
            conn.execute_write(
                f"UNWIND $rows AS row CREATE (n:{label_clause}) "
                f"SET n = row.props SET n.{RESTORE_KEY_PROP} = row.key",
                {"rows": [{"key": n["key"], "props": _decode_props(n["props"])} for n in batch]},
            )
            node_count += len(batch)

    by_type: dict[str, list[dict[str, Any]]] = {}
    for rel in graph["relationships"]:
        by_type.setdefault(rel["type"], []).append(rel)

    rel_count = 0
    for rel_type, group in sorted(by_type.items()):
        for start in range(0, len(group), REL_BATCH):
            batch = group[start : start + REL_BATCH]
            rows = conn.execute_write(
                f"UNWIND $rows AS row "
                f"MATCH (a:{_quote_ident(RESTORE_LABEL)} "
                f"{{{RESTORE_KEY_PROP}: row.start}}) "
                f"MATCH (b:{_quote_ident(RESTORE_LABEL)} "
                f"{{{RESTORE_KEY_PROP}: row.end}}) "
                f"CREATE (a)-[r:{_quote_ident(rel_type)}]->(b) SET r = row.props "
                f"RETURN count(r) AS created",
                {"rows": [{**rel, "props": _decode_props(rel["props"])} for rel in batch]},
            )
            created = int(rows[0]["created"]) if rows else 0
            if created != len(batch):
                raise HydrationError(
                    f"restore created {created} of {len(batch)} {rel_type} relationships; "
                    "an endpoint key did not resolve, so the artifact is internally "
                    "inconsistent. The graph is now partially loaded -- re-run restore."
                )
            rel_count += created

    # Strip the scaffolding. If this loop does not run to completion the marker
    # label and property stay ON the restored nodes, where they are visible to
    # any query -- a restore that died half way is loud rather than plausible.
    while True:
        rows = conn.execute_write(
            f"MATCH (n:{_quote_ident(RESTORE_LABEL)}) WITH n LIMIT 10000 "
            f"REMOVE n:{_quote_ident(RESTORE_LABEL)} REMOVE n.{RESTORE_KEY_PROP} "
            f"RETURN count(n) AS cleaned"
        )
        if not rows or int(rows[0]["cleaned"]) == 0:
            break
    conn.execute_write(f"DROP INDEX {_quote_ident(RESTORE_INDEX_NAME)} IF EXISTS")
    return node_count, rel_count


# --------------------------------------------------------------------------
# sqlite + vault legs
# --------------------------------------------------------------------------


def _sqlite_files(dev_root: Path) -> list[Path]:
    """Every top-level `*.db` under the dev root, discovered rather than listed.

    Discovery over enumeration because the store set grows: the extraction cache
    is not separately configurable and only appears beside the event store, and a
    snapshot that silently omitted a store added next month would produce a
    fixture that is wrong in exactly the invisible way this phase is about.
    """
    return sorted(p for p in dev_root.glob("*.db") if p.is_file())


def _backup_sqlite(source: Path, destination: Path) -> None:
    """Consistent copy via the online backup API. See the module docstring."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(destination))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    _strip_sqlite_sidecars(destination)


def _strip_sqlite_sidecars(db_path: Path) -> None:
    """Remove the `-wal`/`-shm` files a WAL database leaves beside itself.

    The copy inherits WAL mode, so merely OPENING it recreates these -- which is
    why this is a named helper called again after the row count rather than a
    tail on the backup: doing it only there produced an artifact that still
    carried a 32KB `-shm` and an empty `-wal`, because `_count_rows` reopened
    the file afterwards. A clean close checkpoints everything into the `.db`, so
    the sidecars hold nothing; leaving them in an artifact is misleading, since
    restore globs `*.db` and ignores them while a reader would assume they
    carried data.
    """
    for suffix in ("-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)


def _count_rows(db_path: Path) -> dict[str, int]:
    """Row counts for the tables this artifact reports on, skipping absent ones."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        present = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        counts: dict[str, int] = {}
        for table in COUNTED_TABLES:
            if table in present:
                counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        return counts
    finally:
        conn.close()


def _copy_tree(source: Path, destination: Path) -> int:
    """Replace `destination` with a copy of `source`; return the file count."""
    if destination.exists():
        shutil.rmtree(destination)
    if not source.exists():
        return 0
    shutil.copytree(source, destination)
    return sum(1 for p in destination.rglob("*") if p.is_file())


# --------------------------------------------------------------------------
# artifact directory safety
# --------------------------------------------------------------------------


def _assert_artifact_dir_safe(artifact_dir: Path) -> None:
    """Refuse an output directory that is, or contains, live state.

    Narrower than `assert_isolated_root` on purpose: artifacts live UNDER
    `data/`, alongside `data/golden-log/`, following the convention this repo
    already uses for large generated fixtures. So the "sits under a live root"
    arm cannot apply here -- but "IS a live root" and "CONTAINS one" still must,
    because `dump` deletes and rewrites its output directory.
    """
    resolved = artifact_dir.resolve()
    for live in live_state_roots():
        if resolved == live or resolved in live.parents:
            raise HydrationError(
                f"refusing to write a hydration artifact to {resolved}: it is, or "
                f"contains, the live state directory {live}. dump() replaces its "
                "output directory."
            )
    if resolved.exists() and any(resolved.iterdir()) and not (resolved / "manifest.json").exists():
        raise HydrationError(
            f"refusing to write a hydration artifact to {resolved}: it is non-empty "
            "and holds no manifest.json, so it is not a hydration artifact and "
            "overwriting it would destroy whatever it is."
        )


def _connect(uri: str) -> Neo4jConnection:
    """Open a guarded connection to a dev endpoint. The guard runs FIRST."""
    assert_neo4j_dev_isolated(uri)
    config = Neo4jConfig(
        uri=uri,
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    conn = Neo4jConnection(config)
    conn.connect()
    return conn


# --------------------------------------------------------------------------
# public operations
# --------------------------------------------------------------------------


def dump(
    *,
    dev_root: Path = DEFAULT_DEV_ROOT,
    neo4j_uri: str = DEFAULT_DEV_NEO4J_URI,
    artifact_dir: Path,
    corpus_path: Path | None = None,
) -> SnapshotReport:
    """Capture the dev stack into `artifact_dir`.

    Fails closed on an empty graph, per R1.4.5's precedent: an artifact of zero
    nodes would satisfy every downstream assertion vacuously and is far more
    likely to mean "pointed at the wrong stack" than "hydration produced
    nothing".

    Raises:
        IsolatedRootError: If `dev_root` resolves onto live state.
        EvalIsolationError: If `neo4j_uri` is not a dev endpoint.
        HydrationError: On an empty graph, an unsafe artifact directory, or a
            property the artifact format cannot round-trip.
    """
    assert_isolated_root(dev_root, purpose="hydration snapshot source")
    _assert_artifact_dir_safe(artifact_dir)
    if not dev_root.exists():
        raise HydrationError(f"dev root {dev_root} does not exist; nothing to snapshot")

    # Identity FIRST, before a single byte is written. It is the step most
    # likely to refuse (a missing corpus, an unreadable config), and computing
    # it last once left a directory holding a graph, three databases and a vault
    # but no manifest -- the exact "artifact that cannot state what produced it"
    # that `SnapshotManifest.read` warns about, manufactured by the tool itself.
    identity = SnapshotIdentity.current(corpus_path)

    conn = _connect(neo4j_uri)
    try:
        graph = _read_graph(conn)
    finally:
        conn.disconnect()

    if not graph["nodes"]:
        raise HydrationError(
            f"refusing to snapshot an empty graph from {neo4j_uri}: an artifact with "
            "zero nodes restores a state indistinguishable from a fresh stack and "
            "would satisfy every downstream assertion vacuously"
        )

    # Assemble beside the destination and promote by rename, so the artifact
    # path only ever holds a COMPLETE artifact. Belt to the braces above: no
    # ordering of the writes below can leave a half-built directory behind
    # under the name a later restore would trust.
    staging = artifact_dir.with_name(artifact_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        (staging / GRAPH_FILENAME).write_text(
            json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

        contents: dict[str, int] = {
            "graph_nodes": len(graph["nodes"]),
            "graph_relationships": len(graph["relationships"]),
        }
        databases: list[str] = []
        for db_path in _sqlite_files(dev_root):
            _backup_sqlite(db_path, staging / db_path.name)
            databases.append(db_path.name)
            for table, count in _count_rows(staging / db_path.name).items():
                contents[f"{db_path.stem}.{table}"] = count
            _strip_sqlite_sidecars(staging / db_path.name)

        vault_files = _copy_tree(dev_root / VAULT_DIRNAME, staging / VAULT_DIRNAME)
        contents["vault_files"] = vault_files

        SnapshotManifest(
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            identity=identity,
            contents=contents,
            source={"neo4j_uri": neo4j_uri, "dev_root": str(dev_root)},
        ).write(staging)
    except (HydrationError, OSError, sqlite3.Error):
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    staging.rename(artifact_dir)

    return SnapshotReport(
        artifact_dir=artifact_dir,
        nodes=len(graph["nodes"]),
        relationships=len(graph["relationships"]),
        databases=tuple(databases),
        vault_files=vault_files,
    )


def restore(
    *,
    artifact_dir: Path,
    dev_root: Path = DEFAULT_DEV_ROOT,
    neo4j_uri: str = DEFAULT_DEV_NEO4J_URI,
    corpus_path: Path | None = None,
    allow_stale: bool = False,
) -> SnapshotReport:
    """Load an artifact back into the dev stack. Destructive on the target.

    Order matters: both isolation guards and the staleness check run BEFORE
    anything is deleted, so a refused restore leaves the target untouched.

    Raises:
        IsolatedRootError: If `dev_root` resolves onto live state.
        EvalIsolationError: If `neo4j_uri` is not a dev endpoint.
        HydrationError: On a stale artifact (unless `allow_stale`), a missing
            manifest, or an artifact whose relationship endpoints do not resolve.
    """
    assert_isolated_root(dev_root, purpose="hydration restore target")
    assert_neo4j_dev_isolated(neo4j_uri)

    manifest = SnapshotManifest.read(artifact_dir)
    if allow_stale:
        current = SnapshotIdentity.current(corpus_path)
        drift = manifest.identity.drift_against(current)
        if drift:
            logger.warning(
                "[WARNING] restoring a STALE artifact -- drift on %s. The restored "
                "graph does not correspond to this tree's code.",
                ", ".join(drift),
            )
    else:
        assert_fresh(manifest, corpus_path)

    graph_path = artifact_dir / GRAPH_FILENAME
    if not graph_path.exists():
        raise HydrationError(f"{graph_path} not found; the artifact is incomplete")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    conn = _connect(neo4j_uri)
    try:
        _clear_graph(conn)
        nodes, relationships = _write_graph(conn, graph)
    finally:
        conn.disconnect()

    dev_root.mkdir(parents=True, exist_ok=True)
    databases: list[str] = []
    for db_file in sorted(artifact_dir.glob("*.db")):
        target = dev_root / db_file.name
        # Copy through the backup API in this direction too, so a target left
        # with a stale -wal/-shm sidecar from a previous run cannot resurrect
        # older pages on top of the restored file.
        for sidecar in (f"{target}-wal", f"{target}-shm"):
            Path(sidecar).unlink(missing_ok=True)
        _backup_sqlite(db_file, target)
        databases.append(db_file.name)

    vault_files = _copy_tree(artifact_dir / VAULT_DIRNAME, dev_root / VAULT_DIRNAME)

    expected_nodes = manifest.contents.get("graph_nodes")
    if expected_nodes is not None and expected_nodes != nodes:
        raise HydrationError(
            f"restored {nodes} nodes but the manifest recorded {expected_nodes}; "
            "the artifact's graph.json and manifest.json disagree"
        )

    return SnapshotReport(
        artifact_dir=artifact_dir,
        nodes=nodes,
        relationships=relationships,
        databases=tuple(databases),
        vault_files=vault_files,
    )


def verify(*, artifact_dir: Path, corpus_path: Path | None = None) -> SnapshotManifest:
    """Read an artifact's manifest and check it against this tree. Reads only."""
    manifest = SnapshotManifest.read(artifact_dir)
    assert_fresh(manifest, corpus_path)
    return manifest


def list_artifacts(root: Path = DEFAULT_ARTIFACT_ROOT) -> list[tuple[Path, str]]:
    """Return `(directory, status)` for each artifact under `root`."""
    if not root.exists():
        return []
    found: list[tuple[Path, str]] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            verify(artifact_dir=child)
            found.append((child, "FRESH"))
        except HydrationError as exc:
            first = str(exc).splitlines()[0]
            found.append((child, f"STALE ({first})"))
    return found


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    # The target flags live on a PARENT parser rather than the root one, so
    # `dump --dev-root X` works. On the root parser argparse accepts them only
    # BEFORE the subcommand, and `--dev-root` is the flag that decides what gets
    # written -- an argument order that rejects the obvious spelling is a way to
    # get an operator to retry with the default.
    targets = argparse.ArgumentParser(add_help=False)
    targets.add_argument(
        "--dev-root",
        type=Path,
        default=DEFAULT_DEV_ROOT,
        help=f"dev state root (default: {DEFAULT_DEV_ROOT})",
    )
    targets.add_argument(
        "--neo4j-uri",
        default=os.getenv("MIST_DEV_NEO4J_URI", DEFAULT_DEV_NEO4J_URI),
        help=f"dev Neo4j bolt URI (default: {DEFAULT_DEV_NEO4J_URI})",
    )
    targets.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=f"corpus whose digest identifies the artifact (default: {DEFAULT_CORPUS_PATH})",
    )

    parser = argparse.ArgumentParser(
        prog="python -m scripts.hydration.snapshot",
        description="Dump and restore the R1.4.6 hydrated dev stack.",
        parents=[targets],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dump_cmd = sub.add_parser(
        "dump", parents=[targets], help="snapshot the dev stack to an artifact"
    )
    dump_cmd.add_argument("--label", default=None, help="artifact directory name")
    dump_cmd.add_argument("--artifact", type=Path, default=None, help="explicit artifact path")

    for name, help_text in (
        ("restore", "load an artifact into the dev stack (DESTRUCTIVE on the target)"),
        ("verify", "check an artifact against this tree without restoring"),
    ):
        cmd = sub.add_parser(name, parents=[targets], help=help_text)
        cmd.add_argument("--artifact", type=Path, required=True)
        if name == "restore":
            cmd.add_argument(
                "--allow-stale",
                action="store_true",
                help="restore even though producer inputs have changed",
            )

    sub.add_parser("list", parents=[targets], help="list artifacts and whether each is fresh")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    try:
        if args.command == "dump":
            label = args.label or datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
            artifact_dir = args.artifact or (DEFAULT_ARTIFACT_ROOT / label)
            report = dump(
                dev_root=args.dev_root,
                neo4j_uri=args.neo4j_uri,
                artifact_dir=artifact_dir,
                corpus_path=args.corpus,
            )
            print(f"[OK] wrote {report.artifact_dir}")
            print(f"     nodes={report.nodes} relationships={report.relationships}")
            print(f"     databases={list(report.databases)} vault_files={report.vault_files}")
        elif args.command == "restore":
            report = restore(
                artifact_dir=args.artifact,
                dev_root=args.dev_root,
                neo4j_uri=args.neo4j_uri,
                corpus_path=args.corpus,
                allow_stale=args.allow_stale,
            )
            print(f"[OK] restored {report.artifact_dir} -> {args.dev_root} / {args.neo4j_uri}")
            print(f"     nodes={report.nodes} relationships={report.relationships}")
            print(f"     databases={list(report.databases)} vault_files={report.vault_files}")
        elif args.command == "verify":
            manifest = verify(artifact_dir=args.artifact, corpus_path=args.corpus)
            print(f"[OK] {args.artifact} is FRESH against this tree")
            print(f"     created_at={manifest.created_at}")
            print(f"     {manifest.identity.to_dict()}")
        else:
            rows = list_artifacts()
            if not rows:
                print(f"[INFO] no artifacts under {DEFAULT_ARTIFACT_ROOT}")
            for path, status in rows:
                print(f"{status:<8} {path}")
    except (HydrationError, IsolatedRootError, EvalIsolationError, Neo4jQueryError) as exc:
        # A refused run is an expected outcome of a fail-closed tool, not a
        # crash. It exits 1 with the guard's own message so the operator reads
        # WHY it refused rather than a traceback.
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
