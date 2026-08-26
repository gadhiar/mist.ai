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

from backend.errors import Neo4jConnectionError, Neo4jQueryError, SeedSourceError
from backend.interfaces import GraphConnection
from backend.knowledge.embeddings.embedding_text import embedding_text_for
from backend.knowledge.ontologies import EDGE_TYPES_BY_NAME, EXTRACTABLE_RELATIONSHIP_TYPES
from backend.knowledge.seed.models import SeedDocument
from backend.knowledge.storage.partitions import (
    ENTITY_LABEL,
    PROVENANCE_LABEL,
    SELF_MODEL_LABEL,
    SELF_MODEL_TYPES,
)
from backend.knowledge.version_stamps import ONTOLOGY_VERSION

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
        "CREATE CONSTRAINT selfmodel_id_unique IF NOT EXISTS FOR (s:__SelfModel__) REQUIRE s.id IS UNIQUE",
    ):
        connection.execute_write(cypher)
        counts["constraints"] += 1
    for cypher in (
        "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.entity_type)",
        "CREATE INDEX provenance_type_idx IF NOT EXISTS FOR (p:__Provenance__) ON (p.entity_type)",
        "CREATE INDEX selfmodel_type_idx IF NOT EXISTS FOR (s:__SelfModel__) ON (s.entity_type)",
    ):
        connection.execute_write(cypher)
        counts["indexes"] += 1
    # C1: per-predicate relationship range index on the engine's probe key.
    # Neo4j 5 Community supports relationship range indexes; booleans like
    # is_latest_belief are deliberately NOT indexed (poor selectivity) --
    # reads stay entity-anchored.
    for rel_name in EXTRACTABLE_RELATIONSHIP_TYPES:
        connection.execute_write(
            f"CREATE INDEX rel_{rel_name.lower()}_src_utt_idx IF NOT EXISTS "
            f"FOR ()-[r:{rel_name}]-() ON (r.source_utterance_id)"
        )
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


def _find_document_by_partition(documents: list[SeedDocument], partition: str) -> SeedDocument:
    """Return the sole seed document carrying `partition`.

    Raises:
        SeedSourceError: Zero or more than one document carries `partition`.
            R1.4's real source is exactly one identity document
            (`seed/mist.md`, SELF_MODEL_LABEL) and one user document
            (`seed/user.md`, ENTITY_LABEL); ambiguity here is a
            seed-authoring bug, not something to silently resolve by
            picking the first match.
    """
    matches = [d for d in documents if d.partition == partition]
    if len(matches) != 1:
        raise SeedSourceError(
            f"Expected exactly one seed document with partition {partition!r}, "
            f"found {len(matches)} ({[str(d.source_path) for d in matches]})"
        )
    return matches[0]


async def bootstrap_vault_from_seed(
    vault_writer: Any,
    documents: list[SeedDocument],
    rendered_at: str | None = None,
) -> dict[str, str]:
    """Render `identity/mist.md` and `users/<id>.md` from the versioned seed source.

    R1.4 Task 10: repointed from the retired `scripts/seed_data.yaml` dict
    onto `documents: list[SeedDocument]` (`load_seed_documents` over
    `mist-memory/seed/`). Each document's body is written VERBATIM to its
    target vault note rather than assembled from structured per-field dicts
    -- the seed source is authored directly as the note content (`seed/
    mist.md`'s body IS `identity/mist.md`'s body; `seed/user.md`'s body IS
    `users/user.md`'s body; verified byte-identical against both live notes
    before this task landed). The identity document (partition
    SELF_MODEL_LABEL) goes through the new `VaultWriter.upsert_identity_body`;
    the user document (partition ENTITY_LABEL) goes through the existing
    `VaultWriter.upsert_user`, using `source_path.stem` as the user id
    (`mist-memory/seed/user.md` -> user_id `"user"`, matching the retired
    `users/user.md`'s own `user_id` frontmatter field).

    Idempotent. `upsert_identity_body` and `upsert_user` both respect
    `authored_by in {user, user-edit}` and preserve user edits (ADR-010
    Invariant 5).

    Args:
        vault_writer: Started VaultWriter instance.
        documents: Parsed seed documents (`load_seed_documents` output).
        rendered_at: Optional ISO 8601 timestamp string threaded to the
            writer so the seeded identity/user notes are byte-reproducible.
            The seeded `users/<id>.md` is read into the chat system prompt;
            pinning this value makes the F2 replay chain deterministic. None
            means wall-clock -- the unchanged production seed behavior.

    Returns:
        `{"identity_path": <abs>, "user_path": <abs>}`.

    Raises:
        SeedSourceError: `documents` does not carry exactly one
            SELF_MODEL_LABEL document and exactly one ENTITY_LABEL document.
        Whatever VaultWriter raises (typically VaultWriteError on irrecoverable
        filesystem or validation failures). The caller decides whether to
        propagate or swallow.
    """
    identity_doc = _find_document_by_partition(documents, SELF_MODEL_LABEL)
    user_doc = _find_document_by_partition(documents, ENTITY_LABEL)

    identity_path = await vault_writer.upsert_identity_body(
        body_markdown=identity_doc.body,
        source_path=str(identity_doc.source_path),
        rendered_at=rendered_at,
    )
    user_path = await vault_writer.upsert_user(
        user_id=user_doc.source_path.stem,
        body_markdown=user_doc.body,
        rendered_at=rendered_at,
    )
    return {"identity_path": identity_path, "user_path": user_path}


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
    ontology_version = seed_data.get("ontology_version", ONTOLOGY_VERSION)
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

    Uses `embedding_text_for` as the text to embed -- the same builder
    `_backfill_embeddings_for_seed` and `seed.gates.check_embeddings` use
    (I7 Task 1). Only touches nodes whose `provenance = 'seed'` so this is
    safe to re-run.
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
        text = embedding_text_for(row["display_name"], row["description"], row["id"])
        embedding = embedding_generator.generate_embedding(text)
        connection.execute_write(
            "MATCH (n:__Entity__ {id: $id}) SET n.embedding = $embedding",
            {"id": row["id"], "embedding": list(embedding)},
        )
    return len(rows)


def _backfill_embeddings_for_seed(
    connection: GraphConnection, embedding_generator: Any, seed_version: str
) -> int:
    """Compute + SET embedding property on seeded nodes (either partition) missing one.

    R1.4 Task 10: sibling to `_backfill_embeddings`, additive rather than a
    signature change to that (still separately used and tested) function.
    Matches on `seed_version` across the `:__Entity__|__SelfModel__` label
    union rather than `provenance='seed'` restricted to `:__Entity__`,
    because:

    1. `reseed()`'s wipe-then-apply cycle (Task 5) deletes every
       `seed_version`-stamped node once its edges are wiped, then recreates
       it via `MERGE ... ON CREATE SET n.created_at = $now` -- which sets
       only `created_at`/`seed_version`/`updated_at`. `provenance` (and
       `display_name`/`description`, if the node was previously touched by
       the retired `apply_seed`) do NOT survive a delete+recreate; there is
       no property memory across a Neo4j node deletion. `seed_version` DOES
       survive, because it is re-stamped by the very `reseed()` call this
       function runs after -- it is the one predicate proven to hold on
       every re-seed, first run or tenth.
    2. The self-model partition (`:__SelfModel__`) is never also
       `:__Entity__` (Task 4's partition-routing fix), so
       `_backfill_embeddings`'s `MATCH (n:__Entity__)` structurally cannot
       reach it -- whatever either partition currently holds. That
       disjointness, not any particular node count, is what makes the label
       union required rather than cosmetic: `_backfill_embeddings` alone can
       never protect the self-model's embeddings. (When this function was
       written every embedded node happened to be `:__SelfModel__`; both
       partitions carry embedded nodes today. The counts moved, the argument
       did not.)

    Uses `display_name + description` as the embedded text, same as
    `_backfill_embeddings`; falls back to bare `id` when neither is present
    (e.g. a node recreated fresh by `reseed` after losing those properties
    in the delete+recreate above -- lower-quality embedding text, but a
    real one, not a missing one).

    Args:
        connection: Sync graph connection.
        embedding_generator: Provider exposing `generate_embedding(text)`.
        seed_version: The exact stamp `reseed`/`apply_seed_documents` just
            applied -- scopes the backfill to seed-owned nodes only, the
            same discipline the wipe itself uses (Task 5).

    Returns:
        Count of nodes embedded.
    """
    query = """
    MATCH (n:__Entity__|__SelfModel__)
    WHERE n.seed_version = $seed_version AND n.embedding IS NULL
    RETURN n.id AS id,
           coalesce(n.display_name, n.name, n.id) AS display_name,
           n.description AS description
    """
    rows = connection.execute_query(query, {"seed_version": seed_version})
    if not rows:
        return 0
    for row in rows:
        text = embedding_text_for(row["display_name"], row["description"], row["id"])
        embedding = embedding_generator.generate_embedding(text)
        connection.execute_write(
            "MATCH (n:__Entity__|__SelfModel__ {id: $id}) SET n.embedding = $embedding",
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
    MERGE (m:__SelfModel__:MistIdentity {id: $id})
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
        _assert_known_entity_label(label, context=f"seed node {item.get('id', '?')!r}")
        partition = SELF_MODEL_LABEL if label in SELF_MODEL_TYPES else "__Entity__"
        query = (
            f"MERGE (n:{partition} {{id: $id}}) "
            "ON CREATE SET n += $create_only, n += $merge_params "
            "ON MATCH SET n += $merge_params "
            f"SET n:{label}"
        )
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


def _assert_known_entity_label(label: str, *, context: str) -> None:
    """Refuse Cypher label interpolation outside the ontology's node types.

    Labels cannot be parameterized; the closed ontology set is the
    allowlist that makes the f-string interpolation safe
    (deep review cypher-data-integrity-1).
    """
    from backend.knowledge.ontologies import ALL_NODE_TYPE_NAMES

    if label not in ALL_NODE_TYPE_NAMES:
        raise ValueError(
            f"Seed entity label {label!r} ({context}) is not a known ontology "
            "node type; refusing to interpolate it into Cypher."
        )


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
    _assert_known_entity_label(label, context=f"seed entity {entity.get('id', '?')!r}")
    meta = _seed_metadata(now_iso)
    create_only, merge_meta = _split_seed_metadata(meta)
    merge_params = {
        "ontology_version": ontology_version,
        **{k: v for k, v in entity.items() if k != "id"},
        **merge_meta,
    }
    # Label-safe MERGE: match on :__Entity__ {id} alone and SET the typed
    # label afterwards, so a pre-existing label-less node
    # (extraction-created) is HEALED instead of tripping the
    # entity_id_unique constraint (deep review cypher-data-integrity-2b).
    query = f"""
    MERGE (n:__Entity__ {{id: $id}})
    ON CREATE SET n += $create_only, n += $merge_params
    ON MATCH SET n += $merge_params
    SET n:{label}
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
    """MERGE a single relationship between two existing nodes (each in the
    :__Entity__ or :__SelfModel__ partition).

    Applies merge-params on both branches so seed metadata lands even if the
    relationship was pre-created by extraction. `first_seen_at` is create-only.
    """
    meta = _seed_metadata(now_iso)
    create_only, merge_meta = _split_seed_metadata(meta)
    merge_params = {"ontology_version": ontology_version, **merge_meta}
    # C1: seed edges satisfy the current-belief read filters and the engine's
    # fetch shape. 'seed' is the canonical synthetic source; NULL-valued
    # bitemporal fields are omitted (absent == null for every C1 read).
    create_only = {
        **create_only,
        "source_utterance_id": "seed",
        "recorded_at": now_iso,
        "is_latest_belief": True,
        "correction": False,
        "evidence": ["seed"],
    }
    # MERGE keys the seed VERSION so ON MATCH never clobbers engine-written
    # bitemporal versions between the same pair (post-C1 there can be many
    # edges per pair). Legacy seed edges get version_key='seed' via the
    # one-shot backfill, which the cutover runs BEFORE any re-seed.
    query = f"""
    MATCH (s:__Entity__|__SelfModel__ {{id: $source_id}}), (t:__Entity__|__SelfModel__ {{id: $target_id}})
    MERGE (s)-[r:{rel_type} {{version_key: 'seed'}}]->(t)
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


def dump_full_graph_json(connection: GraphConnection) -> dict[str, Any]:
    """Return the ENTIRE graph, every partition, embeddings retained -- a backup.

    Deliberately NOT a flag on `dump_graph_json`. That function's default output
    is what `canonical_graph_form` serialises
    (`grep -n "payload = dump_graph_json" backend/knowledge/canonical_serialize.py`),
    so widening its scope would silently change what the rebuild determinism
    gate compares. These are different jobs: `dump_graph_json` is an ANALYSIS
    view of the entity subgraph; this is a RESTORABLE copy of everything.

    Three differences from `dump_graph_json`, each deliberate:

    - LABEL-AGNOSTIC. `_dump_subgraph` matches `(n:__Entity__)` and, for edges,
      `(s:L)-[r]->(t:L)` with BOTH endpoints carrying one label. On the live
      graph that captured 11 of 32 nodes and none of the 20 intra-self-model
      relationships. A backup that silently omits two thirds of the graph is
      worse than no backup, because it is reached for in exactly the moment it
      is relied on.
    - EMBEDDINGS RETAINED. `_dump_subgraph` calls `_strip_embedding`, correct
      for an analysis view and wrong here: `canonical_serialize` also excludes
      `embedding`, so a vectorless restore is invisible to every determinism
      gate in the repo and surfaces only as degraded retrieval.
    - COUNTS EMITTED, so a truncated artifact is detectable without a graph to
      compare against.

    Ordering is by node id then by (source, type, target), so two dumps of one
    graph do not differ by row order. Nodes with no `id` property sort last
    under a stable sentinel rather than raising.
    """
    node_query = """
    MATCH (n)
    RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties
    """
    rel_query = """
    MATCH (s)-[r]->(t)
    RETURN s.id AS source, type(r) AS type, t.id AS target, properties(r) AS properties
    """
    nodes = [
        {
            "id": row["id"],
            "labels": sorted(row["labels"]),
            "properties": dict(row["properties"]),
        }
        for row in connection.execute_query(node_query)
    ]
    relationships = [
        {
            "source": row["source"],
            "type": row["type"],
            "target": row["target"],
            "properties": dict(row["properties"]),
        }
        for row in connection.execute_query(rel_query)
    ]
    nodes.sort(key=lambda n: (n["id"] is None, n["id"] or ""))
    relationships.sort(
        key=lambda r: (
            r["source"] is None,
            r["source"] or "",
            r["type"] or "",
            r["target"] or "",
        )
    )
    return {
        "nodes": nodes,
        "relationships": relationships,
        "node_count": len(nodes),
        "rel_count": len(relationships),
    }


def count_nodes_by_partition(connection: GraphConnection) -> list[dict[str, Any]]:
    """Return node counts per partition label, including unpartitioned nodes.

    `count_nodes_by_type` is `MATCH (n:__Entity__)` -- its docstring says so --
    which means `graph-stats` reported "11 total" for a 32-node graph and would
    report an unchanged 11 after the entire `:__SelfModel__` partition was
    destroyed. That is the specific blindness that made the 2026-07-31
    live-data-loss incident hard to see, and it is why this exists.

    `(unpartitioned)` is reported rather than dropped: a node carrying no
    partition label is invisible to every partition-scoped query in the
    codebase, so a census that omitted it would reproduce the same class of
    blind spot it is here to close. The counts therefore sum to `MATCH (n)`.
    """
    counts: dict[str, int] = {
        ENTITY_LABEL: 0,
        SELF_MODEL_LABEL: 0,
        PROVENANCE_LABEL: 0,
        "(unpartitioned)": 0,
    }
    rows = connection.execute_query("MATCH (n) RETURN n.id AS id, labels(n) AS labels")
    for row in rows:
        labels = set(row["labels"])
        partitions = labels & {ENTITY_LABEL, SELF_MODEL_LABEL, PROVENANCE_LABEL}
        if not partitions:
            counts["(unpartitioned)"] += 1
            continue
        # A node carrying two partition labels is a defect, not a category --
        # count it under each so the census total still reconciles loudly
        # against `MATCH (n)` rather than hiding the overlap.
        for partition in partitions:
            counts[partition] += 1
    return [{"partition": name, "count": count} for name, count in counts.items()]


def _partition_of(labels: list[str]) -> str:
    """Name the partition a node's label set belongs to, for census bucketing."""
    for label in (ENTITY_LABEL, SELF_MODEL_LABEL, PROVENANCE_LABEL):
        if label in labels:
            return label
    return "(unpartitioned)"


def count_relationships_by_partition(connection: GraphConnection) -> list[dict[str, Any]]:
    """Return relationship counts bucketed by (source partition -> target partition).

    The edge-side twin of `count_nodes_by_partition`.
    `count_relationships_by_type` is scoped to edges whose endpoints are both
    `:__Entity__`, so on the live graph it reported 10 of 30 relationships --
    the 20 intra-self-model edges were invisible, exactly like the 21 nodes
    carrying them.

    Buckets are exclusive (each edge is counted once), so the totals reconcile
    against `MATCH ()-[r]->()`. A cross-partition edge gets its own bucket
    rather than being folded into either endpoint's, because "the self-model
    points at an entity" is the interesting case, not a rounding error.
    """
    rows = connection.execute_query(
        """
        MATCH (s)-[r]->(t)
        RETURN labels(s) AS source_labels, type(r) AS type, labels(t) AS target_labels
        """
    )
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{_partition_of(row['source_labels'])} -> {_partition_of(row['target_labels'])}"
        counts[key] = counts.get(key, 0) + 1
    return [
        {"partitions": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


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


# ---------------------------------------------------------------------------
# C1 bitemporal backfill (one-shot migration)
# ---------------------------------------------------------------------------

# Seeded persona edges (HAS_TRAIT/HAS_CAPABILITY/HAS_PREFERENCE, plus the
# internal-derivation IS_UNCERTAIN_ABOUT) are not extractable types but are
# written under the same version_key='seed' MERGE contract as anchor edges.
# Backfill MUST stamp them too: an unstamped pre-C1 persona edge never
# matches the seed MERGE, so every re-seed duplicates it
# (deep review read-path-currency-4).
_BACKFILL_INTERNAL_TYPES = (
    "HAS_TRAIT",
    "HAS_CAPABILITY",
    "HAS_PREFERENCE",
    "IS_UNCERTAIN_ABOUT",
)

_BACKFILL_GUARD = "r.source_utterance_id IS NULL AND type(r) IN $backfill_types"


def backfill_bitemporal(
    connection: GraphConnection,
    ontology_version: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """One-shot, idempotent bitemporal backfill for pre-C1 fact edges.

    Stamps the C1 fields on every extractable-type edge that lacks
    `source_utterance_id` (the guard that makes re-runs no-ops). SET-to-NULL
    in Cypher removes a property -- absent equals null for every C1 read, so
    only populated fields materialize:
    - source_utterance_id + version_key: 'seed' for legacy seed edges
      (event_id='seed'), else source_event_id, else a stable
      'legacy-' + elementId(r) marker (design 4.4).
    - recorded_at: the legacy created_at (best available fact-time).
    - recorded_until / is_latest_belief: from the legacy status property
      ('superseded' edges are transaction-closed at their updated_at).
    - valid_to: legacy temporal_status='past' closes at updated_at.
      (valid_from stays absent = open lower bound; the legacy 'future'
      status is dropped -- both documented deviations from design 4.4,
      acceptable at post-reset scale.)
    - ontology_version: corrected to the current version (drift fix, 4.7).

    The graph is small (post-reset scale); a single UPDATE is sufficient --
    APOC batching is unnecessary until edge counts demand it.
    """
    # Undirected predicates the engine reads/writes ONLY in lexical
    # (min)->(max) direction. Closed ontology set -- the f-string TYPE
    # interpolation below is allowlist-bounded (Cypher cannot parameterize
    # relationship types).
    undirected_types = [
        name for name in EXTRACTABLE_RELATIONSHIP_TYPES if not EDGE_TYPES_BY_NAME[name].directional
    ]

    params: dict[str, Any] = {
        "backfill_types": list(EXTRACTABLE_RELATIONSHIP_TYPES) + list(_BACKFILL_INTERNAL_TYPES)
    }
    if dry_run:
        rows = connection.execute_query(
            f"MATCH (:__Entity__)-[r]->(:__Entity__) WHERE {_BACKFILL_GUARD} "
            "RETURN count(r) AS n",
            params,
        )
        reverse_candidates = 0
        for name in undirected_types:
            rev = connection.execute_query(
                f"MATCH (a:__Entity__)-[r:{name}]->(b:__Entity__) "
                "WHERE a.id > b.id RETURN count(r) AS n",
                None,
            )
            reverse_candidates += int(rev[0]["n"]) if rev else 0
        return {
            "candidates": int(rows[0]["n"]) if rows else 0,
            "undirected_reverse_candidates": reverse_candidates,
        }

    params["ontology_version"] = ontology_version
    rows = connection.execute_write(
        f"MATCH (:__Entity__)-[r]->(:__Entity__) WHERE {_BACKFILL_GUARD} "
        "SET r.source_utterance_id = CASE WHEN r.event_id = 'seed' THEN 'seed' "
        "ELSE coalesce(r.source_event_id, 'legacy-' + elementId(r)) END, "
        "r.version_key = CASE WHEN r.event_id = 'seed' THEN 'seed' "
        "ELSE coalesce(r.source_event_id, 'legacy-' + elementId(r)) END, "
        "r.recorded_at = coalesce(r.recorded_at, r.created_at), "
        "r.recorded_until = CASE WHEN r.status = 'superseded' "
        "THEN coalesce(r.updated_at, r.created_at) ELSE NULL END, "
        "r.is_latest_belief = (coalesce(r.status, 'active') <> 'superseded'), "
        "r.correction = false, "
        "r.valid_to = CASE WHEN r.temporal_status = 'past' "
        "THEN coalesce(r.updated_at, r.created_at) ELSE NULL END, "
        "r.ontology_version = $ontology_version "
        "RETURN count(r) AS updated",
        params,
    )

    # Canonicalize legacy undirected edges (design 4.1): a reverse-direction
    # WORKS_WITH/RELATED_TO row is invisible to the engine's same-fact
    # fetches, so re-assertions append a canonical twin instead of
    # reinforcing and the pair coexists as permanent duplicates. MERGE on
    # version_key keeps this idempotent and collision-safe: when a canonical
    # twin already exists, the twin wins and the reverse row is dropped.
    # Runs AFTER stamping so reversed rows carry version_key.
    reversed_total = 0
    for name in undirected_types:
        rev_rows = connection.execute_write(
            f"MATCH (a:__Entity__)-[r:{name}]->(b:__Entity__) WHERE a.id > b.id "
            f"MERGE (b)-[c:{name} {{version_key: r.version_key}}]->(a) "
            "ON CREATE SET c = properties(r) "
            "WITH r DELETE r "
            "RETURN count(r) AS reversed",
            None,
        )
        reversed_total += int(rev_rows[0].get("reversed", 0)) if rev_rows else 0

    return {
        "edges_backfilled": int(rows[0]["updated"]) if rows else 0,
        "undirected_canonicalized": reversed_total,
    }
