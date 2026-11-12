"""Ontology v1.4.0 migration: retype Topic -> Concept, Milestone -> Event.

Idempotent. Near-noop on the current identity-only live graph.

DEFERRED (2026-06-14, MIS-124 close-out): do NOT run this standalone against the
live graph yet. R1 (the utterance->graph regenerator -- the next sub-project A
deliverable) performs a FULL graph regeneration under ontology v1.4.0, which
subsumes this retype; running it now would be throwaway work. Retained for
completeness, eval use, and in case R1 slips. `ONTOLOGY_V1_0_0.migration_script_path`
points here.

An entity's type is the `entity_type` property on its :__Entity__ node, and
since typed labels (approved by Raj on 2026-11-06, reversing MIS-177 D2,
which kept every type but :User off :__Entity__ nodes as a label) the node
also carries the type as a label that mirrors the property
(`backend/knowledge/storage/type_labels.py`). Retired types have no label,
so no writer gives a node :Topic or :Milestone; each retype therefore sets the
new type's label and removes the old type's, in the same statement as the
property change, so the label keeps mirroring the property. The WHERE clause
guards make re-runs a no-op: after the first run, no node has
entity_type='Topic' or entity_type='Milestone'.

The migration accepts a GraphExecutor (async) and issues SET and REMOVE write
queries through `execute_write`. Compatible with the production executor
interface in backend.knowledge.storage.graph_executor.GraphExecutor.

Usage (from inside the container, after stack restart):
    python -c "
    import asyncio, sys
    sys.path.insert(0, '.')
    from backend.knowledge.config import get_config
    from backend.knowledge.storage.neo4j_connection import Neo4jConnection
    from backend.knowledge.storage.graph_executor import GraphExecutor
    from scripts.migrations.ontology_v1_4_0 import migrate

    config = get_config()
    conn = Neo4jConnection(config.neo4j)
    conn.connect()
    executor = GraphExecutor(conn)
    asyncio.run(migrate(executor))
    conn.disconnect()
    "
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.knowledge.storage.type_labels import entity_type_label

if TYPE_CHECKING:
    from backend.knowledge.storage.graph_executor import GraphExecutor


def _type_label(entity_type: str) -> str:
    """Return the label the retype sets; raise if the target type has none."""
    label = entity_type_label(entity_type)
    if label is None:
        raise ValueError(f"{entity_type!r} has no entity type label")
    return label


# ---------------------------------------------------------------------------
# Migration queries
# ---------------------------------------------------------------------------

# Each query targets only nodes that still carry the legacy entity_type value.
# After the first successful run, the WHERE clause matches zero nodes and each
# query becomes a structural no-op (no SET, no I/O beyond the MATCH scan).
# The new type's label comes from `entity_type_label`; the REMOVE names the
# retired type, which no current writer labels, so it strips only a label an
# older writer may have left.
_CYPHER_TOPIC_TO_CONCEPT = (
    "MATCH (e:__Entity__) WHERE e.entity_type = 'Topic' "
    "SET e.entity_type = 'Concept', e.ontology_version = '1.4.0', "
    f"e:{_type_label('Concept')} "
    "REMOVE e:Topic"
)

_CYPHER_MILESTONE_TO_EVENT = (
    "MATCH (e:__Entity__) WHERE e.entity_type = 'Milestone' "
    "SET e.entity_type = 'Event', "
    "e.event_type = coalesce(e.event_type, 'milestone'), "
    "e.ontology_version = '1.4.0', "
    f"e:{_type_label('Event')} "
    "REMOVE e:Milestone"
)

CYPHER: list[str] = [_CYPHER_TOPIC_TO_CONCEPT, _CYPHER_MILESTONE_TO_EVENT]


async def migrate(executor: GraphExecutor) -> None:
    """Retype Topic -> Concept and Milestone -> Event across all __Entity__ nodes.

    Idempotent: safe to run multiple times. Each query only touches nodes that
    still carry the legacy entity_type string; after the first run, zero nodes
    match and subsequent runs are structural no-ops.

    Args:
        executor: An async GraphExecutor (or compatible fake) exposing
            `execute_write(query: str, params: dict | None) -> list[dict]`.
    """
    for query in CYPHER:
        await executor.execute_write(query, {})
