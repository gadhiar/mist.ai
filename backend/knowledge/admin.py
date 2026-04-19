"""Admin operations for MIST knowledge graph.

Functions for the `scripts/mist_admin.py` CLI: seed application, graph
introspection, full-graph dump, safety-guarded reset, and stack health probes.
All seed writes are idempotent (MERGE with ON CREATE / ON MATCH branches) and
auto-attach seed metadata (confidence=1.0, temporal_status=current,
event_id=seed, provenance=seed, first_seen_at, last_seen_at).

Spec: ~/.claude/plans/nimble-forage-cinder.md Parts 1-3.
"""

from __future__ import annotations

import contextlib
import json
import socket
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from backend.errors import Neo4jConnectionError, Neo4jQueryError
from backend.interfaces import GraphConnection

SEED_METADATA_FIELDS = (
    "confidence",
    "temporal_status",
    "event_id",
    "provenance",
    "first_seen_at",
    "last_seen_at",
)


def _seed_metadata(now_iso: str) -> dict[str, Any]:
    """Return the standard seed-metadata dict applied to every seeded node/rel.

    `first_seen_at` is create-only (see `_split_seed_metadata`); the rest apply on
    both CREATE and MATCH so that re-seeding enforces the YAML as source of truth
    and seed-metadata fields land even if the node was pre-created by a factory
    (e.g., `gs.ensure_mist_identity()` during backend startup).
    """
    return {
        "confidence": 1.0,
        "temporal_status": "current",
        "event_id": "seed",
        "provenance": "seed",
        "first_seen_at": now_iso,
        "last_seen_at": now_iso,
    }


def _split_seed_metadata(meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split seed metadata into create-only (first_seen_at) and merge (everything else)."""
    create_only = {"first_seen_at": meta["first_seen_at"]}
    merge = {k: v for k, v in meta.items() if k != "first_seen_at"}
    return create_only, merge


def load_seed_yaml(path: Path | str) -> dict[str, Any]:
    """Load and return the seed_data.yaml contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Seed file must be a YAML mapping at root: {path}")
    return data


# ---------------------------------------------------------------------------
# Seed writers
# ---------------------------------------------------------------------------


def ensure_schema(connection: GraphConnection) -> dict[str, int]:
    """Idempotently create the Neo4j constraints + vector index needed for
    extraction, curation, and hybrid retrieval.

    Mirrors `GraphStore.initialize_schema()` but runs against a raw connection
    so admin seed doesn't require building a full GraphStore (which has heavy
    sentence-transformers dependencies). Index creation uses `IF NOT EXISTS`.
    """
    counts = {"constraints": 0, "indexes": 0, "vector_indexes": 0}
    for cypher in (
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT provenance_id_unique IF NOT EXISTS FOR (p:__Provenance__) REQUIRE p.id IS UNIQUE",
    ):
        connection.execute_write(cypher)
        counts["constraints"] += 1
    for cypher in (
        "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.entity_type)",
        "CREATE INDEX provenance_type_idx IF NOT EXISTS FOR (p:__Provenance__) ON (p.entity_type)",
    ):
        connection.execute_write(cypher)
        counts["indexes"] += 1
    vector_cypher = (
        "CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS "
        "FOR (e:__Entity__) ON e.embedding "
        "OPTIONS {indexConfig: {"
        "`vector.dimensions`: 384, "
        "`vector.similarity_function`: 'cosine'}}"
    )
    try:
        connection.execute_write(vector_cypher)
        counts["vector_indexes"] += 1
    except Neo4jQueryError:
        pass  # Older Neo4j or index already exists with different config.
    return counts


def apply_seed(
    connection: GraphConnection,
    seed_data: dict[str, Any],
    embedding_generator: Any = None,
) -> dict[str, int]:
    """Apply all seed facts idempotently. Returns counts per layer.

    Layers applied in order:
        1. MistIdentity singleton
        2. Traits / Capabilities / Preferences
        3. User entity
        4. Anchor entities
        5. Identity relationships (MistIdentity -> trait/cap/pref)
        6. Anchor relationships (user -> entity)
        7. Embeddings backfill (if `embedding_generator` provided)

    Each MERGE uses ON CREATE SET for immutable bootstrap fields and ON MATCH
    SET for seed-spec fields (Task 3 fix). Immutable properties (e.g., trait
    `mutable: false`) guard against accidental overwrite by the
    InternalKnowledgeDeriver.

    Without embeddings, vector retrieval returns zero matches on seeded
    entities; pass `embedding_generator` to populate the `embedding` property
    from each node's `display_name + description` text.
    """
    now = datetime.now(UTC).isoformat()
    ontology_version = seed_data.get("ontology_version", "1.0.0")
    counts: dict[str, int] = {}

    schema_counts = ensure_schema(connection)
    counts["schema_objects"] = sum(schema_counts.values())

    counts["mist_identity"] = _seed_mist_identity(
        connection, seed_data["mist_identity"], ontology_version, now
    )
    counts["traits"] = _seed_internal_nodes(
        connection,
        seed_data.get("traits", []),
        label="MistTrait",
        ontology_version=ontology_version,
        now_iso=now,
        immutable=True,
    )
    counts["capabilities"] = _seed_internal_nodes(
        connection,
        seed_data.get("capabilities", []),
        label="MistCapability",
        ontology_version=ontology_version,
        now_iso=now,
    )
    counts["preferences"] = _seed_internal_nodes(
        connection,
        seed_data.get("preferences", []),
        label="MistPreference",
        ontology_version=ontology_version,
        now_iso=now,
    )
    counts["user"] = _seed_anchor_entity(connection, seed_data["user"], ontology_version, now)
    counts["entities"] = sum(
        _seed_anchor_entity(connection, entity, ontology_version, now)
        for entity in seed_data.get("entities", [])
    )
    counts["identity_relationships"] = _seed_identity_relationships(
        connection, seed_data.get("identity_relationships", []), ontology_version, now
    )
    counts["anchor_relationships"] = _seed_anchor_relationships(
        connection, seed_data.get("anchor_relationships", []), ontology_version, now
    )
    if embedding_generator is not None:
        counts["embeddings"] = _backfill_embeddings(connection, embedding_generator)
    return counts


def _backfill_embeddings(connection: GraphConnection, embedding_generator: Any) -> int:
    """Compute + SET embedding property on seeded nodes missing one.

    Uses `display_name + description` as the text to embed. Only touches nodes
    whose `provenance = 'seed'` so this is safe to re-run.
    """
    query = """
    MATCH (n:__Entity__)
    WHERE n.provenance = 'seed' AND n.embedding IS NULL
    RETURN n.id AS id,
           coalesce(n.display_name, n.name, n.id) AS display_name,
           n.description AS description,
           labels(n) AS labels
    """
    rows = connection.execute_query(query)
    if not rows:
        return 0
    for row in rows:
        text_parts = [row["display_name"] or row["id"]]
        if row["description"]:
            text_parts.append(row["description"])
        text = " — ".join(text_parts)
        embedding = embedding_generator.generate_embedding(text)
        connection.execute_write(
            "MATCH (n:__Entity__ {id: $id}) SET n.embedding = $embedding",
            {"id": row["id"], "embedding": list(embedding)},
        )
    return len(rows)


def _seed_mist_identity(
    connection: GraphConnection,
    identity: dict[str, Any],
    ontology_version: str,
    now_iso: str,
) -> int:
    """MERGE the MistIdentity singleton with full seed properties.

    Applies merge-params on both CREATE and MATCH so that properties land even if
    the node was pre-created by `gs.ensure_mist_identity()` during backend
    startup. `first_seen_at` is create-only.
    """
    meta = _seed_metadata(now_iso)
    create_only, merge_meta = _split_seed_metadata(meta)
    merge_params = {
        "ontology_version": ontology_version,
        **{k: v for k, v in identity.items() if k != "id"},
        **merge_meta,
    }
    query = """
    MERGE (m:__Entity__:MistIdentity {id: $id})
    ON CREATE SET m += $create_only, m += $merge_params
    ON MATCH SET m += $merge_params
    """
    connection.execute_write(
        query,
        {"id": identity["id"], "create_only": create_only, "merge_params": merge_params},
    )
    return 1


def _seed_internal_nodes(
    connection: GraphConnection,
    items: list[dict[str, Any]],
    label: str,
    ontology_version: str,
    now_iso: str,
    immutable: bool = False,
) -> int:
    """MERGE a list of MistTrait/Capability/Preference nodes.

    Applies merge-params on both branches so re-seeding enforces YAML spec.
    `immutable: false` prevents InternalKnowledgeDeriver from overwriting.
    """
    if not items:
        return 0
    meta = _seed_metadata(now_iso)
    create_only, merge_meta = _split_seed_metadata(meta)
    count = 0
    for item in items:
        merge_params = {
            "entity_type": label,
            "ontology_version": ontology_version,
            **{k: v for k, v in item.items() if k != "id"},
            **merge_meta,
        }
        if immutable:
            merge_params["mutable"] = False
        query = f"""
        MERGE (n:__Entity__:{label} {{id: $id}})
        ON CREATE SET n += $create_only, n += $merge_params
        ON MATCH SET n += $merge_params
        """
        connection.execute_write(
            query,
            {
                "id": item["id"],
                "create_only": create_only,
                "merge_params": merge_params,
            },
        )
        count += 1
    return count


def _seed_anchor_entity(
    connection: GraphConnection,
    entity: dict[str, Any],
    ontology_version: str,
    now_iso: str,
) -> int:
    """MERGE a User or anchor entity with its scalar properties.

    Applies merge-params on both branches so seed-metadata (provenance,
    confidence, temporal_status, event_id) lands even if extraction pre-created
    the node. Seed acts as source of truth for domain properties (industry,
    category, vram_gb, etc.); extraction-derived fields NOT in seed YAML are
    preserved since Neo4j `+=` is a merge not a replace.
    """
    label = entity["entity_type"]
    meta = _seed_metadata(now_iso)
    create_only, merge_meta = _split_seed_metadata(meta)
    merge_params = {
        "ontology_version": ontology_version,
        **{k: v for k, v in entity.items() if k != "id"},
        **merge_meta,
    }
    query = f"""
    MERGE (n:__Entity__:{label} {{id: $id}})
    ON CREATE SET n += $create_only, n += $merge_params
    ON MATCH SET n += $merge_params
    """
    connection.execute_write(
        query,
        {"id": entity["id"], "create_only": create_only, "merge_params": merge_params},
    )
    return 1


def _seed_identity_relationships(
    connection: GraphConnection,
    groups: list[dict[str, Any]],
    ontology_version: str,
    now_iso: str,
) -> int:
    """MERGE MistIdentity -> trait/capability/preference relationships."""
    count = 0
    for group in groups:
        source = group["source"]
        rel_type = group["type"]
        for target in group.get("targets", []):
            count += _merge_relationship(
                connection, source, rel_type, target, ontology_version, now_iso
            )
    return count


def _seed_anchor_relationships(
    connection: GraphConnection,
    rels: list[dict[str, Any]],
    ontology_version: str,
    now_iso: str,
) -> int:
    """MERGE user -> entity anchor relationships."""
    return sum(
        _merge_relationship(
            connection,
            rel["source"],
            rel["type"],
            rel["target"],
            ontology_version,
            now_iso,
        )
        for rel in rels
    )


def _merge_relationship(
    connection: GraphConnection,
    source_id: str,
    rel_type: str,
    target_id: str,
    ontology_version: str,
    now_iso: str,
) -> int:
    """MERGE a single relationship between two existing __Entity__ nodes.

    Applies merge-params on both branches so seed metadata lands even if the
    relationship was pre-created by extraction. `first_seen_at` is create-only.
    """
    meta = _seed_metadata(now_iso)
    create_only, merge_meta = _split_seed_metadata(meta)
    merge_params = {"ontology_version": ontology_version, **merge_meta}
    query = f"""
    MATCH (s:__Entity__ {{id: $source_id}}), (t:__Entity__ {{id: $target_id}})
    MERGE (s)-[r:{rel_type}]->(t)
    ON CREATE SET r += $create_only, r += $merge_params
    ON MATCH SET r += $merge_params
    """
    result = connection.execute_write(
        query,
        {
            "source_id": source_id,
            "target_id": target_id,
            "create_only": create_only,
            "merge_params": merge_params,
        },
    )
    return 1 if result is not None else 0


# ---------------------------------------------------------------------------
# Graph introspection (stats)
# ---------------------------------------------------------------------------


def count_nodes_by_type(connection: GraphConnection) -> list[dict[str, Any]]:
    """Return node counts grouped by entity_type (excluding non-__Entity__ nodes)."""
    query = """
    MATCH (n:__Entity__)
    RETURN coalesce(n.entity_type, '(unspecified)') AS entity_type, count(n) AS count
    ORDER BY count DESC, entity_type ASC
    """
    return connection.execute_query(query)


def count_relationships_by_type(connection: GraphConnection) -> list[dict[str, Any]]:
    """Return relationship counts grouped by type (only between __Entity__ nodes)."""
    query = """
    MATCH (:__Entity__)-[r]->(:__Entity__)
    RETURN type(r) AS rel_type, count(r) AS count
    ORDER BY count DESC, rel_type ASC
    """
    return connection.execute_query(query)


def get_confidence_distribution(connection: GraphConnection) -> dict[str, Any]:
    """Return avg/min/max confidence across nodes and relationships."""
    node_query = """
    MATCH (n:__Entity__)
    WHERE n.confidence IS NOT NULL
    RETURN avg(n.confidence) AS avg, min(n.confidence) AS min,
           max(n.confidence) AS max, count(n) AS n
    """
    rel_query = """
    MATCH (:__Entity__)-[r]->(:__Entity__)
    WHERE r.confidence IS NOT NULL
    RETURN avg(r.confidence) AS avg, min(r.confidence) AS min,
           max(r.confidence) AS max, count(r) AS n
    """
    nodes = connection.execute_query(node_query)
    rels = connection.execute_query(rel_query)
    return {
        "nodes": nodes[0] if nodes else {},
        "relationships": rels[0] if rels else {},
    }


def find_orphan_relationships(connection: GraphConnection) -> list[dict[str, Any]]:
    """Return relationships where either endpoint is not an __Entity__.

    Strict orphan check (missing endpoints) is enforced by Neo4j referential
    integrity, so this surfaces label/type mismatches and malformed writes.
    """
    query = """
    MATCH (s)-[r]->(t)
    WHERE NOT s:__Entity__ OR NOT t:__Entity__
    RETURN labels(s) AS source_labels, type(r) AS rel_type,
           labels(t) AS target_labels, count(r) AS count
    LIMIT 100
    """
    return connection.execute_query(query)


def count_provenance(connection: GraphConnection) -> dict[str, int]:
    """Return counts of seeded vs derived (non-seed) __Entity__ nodes."""
    query = """
    MATCH (n:__Entity__)
    RETURN coalesce(n.provenance, '(none)') AS provenance, count(n) AS count
    """
    rows = connection.execute_query(query)
    return {row["provenance"]: row["count"] for row in rows}


def count_non_seed_entities(connection: GraphConnection) -> int:
    """Return count of __Entity__ nodes whose provenance is NOT 'seed'.

    Used by graph-reset safety guard to refuse wiping derived data unless
    --include-derived is explicitly passed.
    """
    query = """
    MATCH (n:__Entity__)
    WHERE coalesce(n.provenance, '') <> 'seed'
    RETURN count(n) AS count
    """
    result = connection.execute_query(query)
    return result[0]["count"] if result else 0


def provenance_counts_by_type(connection: GraphConnection) -> list[dict[str, Any]]:
    """Return node counts grouped by entity_type for :__Provenance__ nodes only."""
    query = """
    MATCH (n:__Provenance__)
    RETURN coalesce(n.entity_type, '(unspecified)') AS entity_type, count(n) AS count
    ORDER BY count DESC, entity_type ASC
    """
    return connection.execute_query(query)


def provenance_relationship_counts_by_type(connection: GraphConnection) -> list[dict[str, Any]]:
    """Return relationship counts for edges between :__Provenance__ nodes."""
    query = """
    MATCH (:__Provenance__)-[r]->(:__Provenance__)
    RETURN type(r) AS rel_type, count(r) AS count
    ORDER BY count DESC, rel_type ASC
    """
    return connection.execute_query(query)


def cross_layer_relationship_counts(connection: GraphConnection) -> list[dict[str, Any]]:
    """Return counts of edges spanning :__Entity__ and :__Provenance__ (both directions)."""
    query = """
    MATCH (s)-[r]->(t)
    WHERE (s:__Entity__ AND t:__Provenance__) OR (s:__Provenance__ AND t:__Entity__)
    RETURN type(r) AS rel_type, count(r) AS count
    ORDER BY count DESC, rel_type ASC
    """
    return connection.execute_query(query)


# ---------------------------------------------------------------------------
# Graph dump
# ---------------------------------------------------------------------------


def _dump_subgraph(connection: GraphConnection, label: str) -> dict[str, list[dict[str, Any]]]:
    """Return nodes and internal relationships for a single label family.

    Strips the ``embedding`` field from node properties. Intended for use
    by ``dump_graph_json``; not part of the public admin API.
    """
    node_query = f"""
    MATCH (n:{label})
    RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties
    ORDER BY n.id
    """
    rel_query = f"""
    MATCH (s:{label})-[r]->(t:{label})
    RETURN s.id AS source, type(r) AS type, t.id AS target,
           properties(r) AS properties
    ORDER BY s.id, type(r), t.id
    """
    nodes = [
        {
            "id": row["id"],
            "labels": [lbl for lbl in row["labels"] if lbl != label],
            "properties": _strip_embedding(row["properties"]),
        }
        for row in connection.execute_query(node_query)
    ]
    relationships = [
        {
            "source": row["source"],
            "type": row["type"],
            "target": row["target"],
            "properties": row["properties"],
        }
        for row in connection.execute_query(rel_query)
    ]
    return {"nodes": nodes, "relationships": relationships}


def dump_graph_json(
    connection: GraphConnection,
    *,
    include_provenance: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Return the __Entity__ subgraph as a JSON-serializable dict.

    When *include_provenance* is True three additional keys are included:

    - ``provenance``: dict with ``nodes`` and ``relationships`` for the
      ``:__Provenance__`` label family.
    - ``cross_layer_edges``: list of edge dicts spanning ``:__Entity__`` and
      ``:__Provenance__`` (both directions).

    Default behaviour (``include_provenance=False``) is unchanged — only the
    entity subgraph is returned.
    """
    result = _dump_subgraph(connection, "__Entity__")

    if include_provenance:
        result["provenance"] = _dump_subgraph(connection, "__Provenance__")

        cross_query = """
        MATCH (s)-[r]->(t)
        WHERE (s:__Entity__ AND t:__Provenance__) OR (s:__Provenance__ AND t:__Entity__)
        RETURN s.id AS source, type(r) AS type, t.id AS target,
               properties(r) AS properties
        ORDER BY s.id, type(r), t.id
        """
        result["cross_layer_edges"] = [
            {
                "source": row["source"],
                "type": row["type"],
                "target": row["target"],
                "properties": row["properties"],
            }
            for row in connection.execute_query(cross_query)
        ]

    return result


def dump_graph_cypher(
    connection: GraphConnection,
    *,
    include_provenance: bool = False,
) -> str:
    """Return the __Entity__ subgraph as a Cypher script of MERGE statements.

    Intended for snapshotting and offline diffing. Embeddings are stripped.
    Re-importing produces a schema-equivalent subgraph.

    When *include_provenance* is True the ``:__Provenance__`` subgraph and
    cross-layer edges are appended as a second section in the script.
    """
    payload = dump_graph_json(connection, include_provenance=include_provenance)
    lines: list[str] = [
        "// MIST graph snapshot (Cypher)",
        f"// generated {datetime.now(UTC).isoformat()}",
        "",
        "// --- :__Entity__ subgraph ---",
    ]
    for node in payload["nodes"]:
        labels = ":".join(["__Entity__", *node["labels"]]) if node["labels"] else "__Entity__"
        props = _cypher_props(node["properties"])
        lines.append(
            f"MERGE (:{labels} {{id: {_cypher_value(node['id'])}}}) ON CREATE SET {props};"
        )
    lines.append("")
    for rel in payload["relationships"]:
        src = _cypher_value(rel["source"])
        tgt = _cypher_value(rel["target"])
        rtype = rel["type"]
        props = _cypher_props(rel["properties"], prefix="r")
        lines.append(
            f"MATCH (s:__Entity__ {{id: {src}}}), (t:__Entity__ {{id: {tgt}}}) "
            f"MERGE (s)-[r:{rtype}]->(t) ON CREATE SET {props};"
        )

    if include_provenance:
        prov = payload["provenance"]
        lines += [
            "",
            "// --- :__Provenance__ subgraph ---",
        ]
        for node in prov["nodes"]:
            labels = (
                ":".join(["__Provenance__", *node["labels"]])
                if node["labels"]
                else "__Provenance__"
            )
            props = _cypher_props(node["properties"])
            lines.append(
                f"MERGE (:{labels} {{id: {_cypher_value(node['id'])}}}) ON CREATE SET {props};"
            )
        lines.append("")
        for rel in prov["relationships"]:
            src = _cypher_value(rel["source"])
            tgt = _cypher_value(rel["target"])
            rtype = rel["type"]
            props = _cypher_props(rel["properties"], prefix="r")
            lines.append(
                f"MATCH (s:__Provenance__ {{id: {src}}}), (t:__Provenance__ {{id: {tgt}}}) "
                f"MERGE (s)-[r:{rtype}]->(t) ON CREATE SET {props};"
            )

        cross = payload["cross_layer_edges"]
        lines += [
            "",
            "// --- cross-layer edges ---",
        ]
        for rel in cross:
            src = _cypher_value(rel["source"])
            tgt = _cypher_value(rel["target"])
            rtype = rel["type"]
            props = _cypher_props(rel["properties"], prefix="r")
            lines.append(
                f"MATCH (s {{id: {src}}}), (t {{id: {tgt}}}) "
                f"MERGE (s)-[r:{rtype}]->(t) ON CREATE SET {props};"
            )

    return "\n".join(lines) + "\n"


def _strip_embedding(properties: dict[str, Any]) -> dict[str, Any]:
    """Strip the `embedding` list (large numeric vector) from property output."""
    return {k: v for k, v in properties.items() if k != "embedding"}


def _cypher_value(value: Any) -> str:
    """Render a Python value as a Cypher literal."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_cypher_value(v) for v in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _cypher_props(properties: dict[str, Any], prefix: str = "n") -> str:
    """Render properties dict as `prefix.key = value, ...`."""
    return ", ".join(
        f"{prefix}.{k} = {_cypher_value(v)}"
        for k, v in sorted(properties.items())
        if k != "embedding"
    )


# ---------------------------------------------------------------------------
# Graph reset
# ---------------------------------------------------------------------------


def reset_graph(connection: GraphConnection, include_derived: bool = False) -> dict[str, int]:
    """Wipe __Entity__ nodes and their relationships. Returns counts removed.

    When ``include_derived=True``, also wipes all ``:__Provenance__`` nodes so
    that a full reset produces a clean slate. Without the flag, provenance nodes
    survive the reset — this preserves the "keep seed, wipe conversation"
    pattern used during iterative gauntlet runs.

    Safety: caller MUST verify non-seed entity count before calling with
    include_derived=False; this function itself applies the guard and raises.
    """
    non_seed = count_non_seed_entities(connection)
    if non_seed > 0 and not include_derived:
        raise Neo4jQueryError(
            f"Refusing to reset: {non_seed} non-seed entities present. "
            "Pass include_derived=True to proceed."
        )
    before_nodes = connection.execute_query("MATCH (n:__Entity__) RETURN count(n) AS count")[0][
        "count"
    ]
    before_rels = connection.execute_query(
        "MATCH (:__Entity__)-[r]->(:__Entity__) RETURN count(r) AS count"
    )[0]["count"]
    connection.execute_write("MATCH (n:__Entity__) DETACH DELETE n")

    result: dict[str, int] = {
        "nodes_removed": before_nodes,
        "relationships_removed": before_rels,
        "provenance_nodes_removed": 0,
    }

    if include_derived:
        before_provenance = connection.execute_query(
            "MATCH (n:__Provenance__) RETURN count(n) AS count"
        )[0]["count"]
        connection.execute_write("MATCH (n:__Provenance__) DETACH DELETE n")
        result["provenance_nodes_removed"] = before_provenance

    return result


# ---------------------------------------------------------------------------
# Health probes (stack-status)
# ---------------------------------------------------------------------------


def probe_neo4j(connection: GraphConnection) -> dict[str, Any]:
    """Probe Neo4j connectivity. Returns status dict with diagnostic info."""
    try:
        connection.connect()
        result = connection.execute_query("MATCH (n:__Entity__) RETURN count(n) AS count")
        return {
            "service": "neo4j",
            "status": "healthy",
            "entity_count": result[0]["count"] if result else 0,
            "uri": connection.config.uri,
        }
    except Neo4jConnectionError as e:
        return {"service": "neo4j", "status": "unreachable", "error": str(e)}
    except Neo4jQueryError as e:
        return {"service": "neo4j", "status": "query_failed", "error": str(e)}


def probe_llm(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Probe llama-server /health endpoint."""
    url = f"{base_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310  # nosec B310
            ok = resp.status == 200
            body = resp.read().decode("utf-8", errors="replace")[:200]
        return {
            "service": "llm",
            "status": "healthy" if ok else f"http_{resp.status}",
            "url": url,
            "body": body,
        }
    except urllib.error.URLError as e:
        return {"service": "llm", "status": "unreachable", "url": url, "error": str(e)}
    except TimeoutError:
        return {"service": "llm", "status": "timeout", "url": url}


def probe_backend(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Probe MIST backend /health endpoint."""
    url = f"{base_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310  # nosec B310
            ok = resp.status == 200
            body = resp.read().decode("utf-8", errors="replace")[:200]
        payload: dict[str, Any] = {}
        with contextlib.suppress(json.JSONDecodeError):
            payload = json.loads(body)
        return {
            "service": "backend",
            "status": "healthy" if ok else f"http_{resp.status}",
            "url": url,
            **({"payload": payload} if payload else {"body": body}),
        }
    except urllib.error.URLError as e:
        return {"service": "backend", "status": "unreachable", "url": url, "error": str(e)}
    except TimeoutError:
        return {"service": "backend", "status": "timeout", "url": url}


def probe_tcp(host: str, port: int, timeout: float = 3.0) -> bool:
    """Low-level TCP reachability probe. Used when HTTP probes are inappropriate."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
