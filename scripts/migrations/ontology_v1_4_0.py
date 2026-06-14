"""Ontology v1.4.0 migration: retype Topic -> Concept, Milestone -> Event.

Idempotent. Near-noop on the current identity-only live graph.

DEFERRED (2026-06-14, MIS-124 close-out): do NOT run this standalone against the
live graph yet. R1 (the utterance->graph regenerator -- the next sub-project A
deliverable) performs a FULL graph regeneration under ontology v1.4.0, which
subsumes this retype; running it now would be throwaway work. Retained for
completeness, eval use, and in case R1 slips. `ONTOLOGY_V1_0_0.migration_script_path`
points here.

Entity types in MIST.AI are stored ONLY as the `entity_type` property on
:__Entity__ nodes -- there are no :Topic or :Milestone Neo4j labels (only
:__Entity__ is the universal label, with :User and :MistIdentity as special
invariants). The WHERE clause guards make re-runs a no-op: after the first
run, no node has entity_type='Topic' or entity_type='Milestone'.

The migration accepts a GraphExecutor (async) and issues SET-only write
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

if TYPE_CHECKING:
    from backend.knowledge.storage.graph_executor import GraphExecutor

# ---------------------------------------------------------------------------
# Migration queries
# ---------------------------------------------------------------------------

# Each query targets only nodes that still carry the legacy entity_type value.
# After the first successful run, the WHERE clause matches zero nodes and each
# query becomes a structural no-op (no SET, no I/O beyond the MATCH scan).
_CYPHER_TOPIC_TO_CONCEPT = (
    "MATCH (e:__Entity__) WHERE e.entity_type = 'Topic' "
    "SET e.entity_type = 'Concept', e.ontology_version = '1.4.0'"
)

_CYPHER_MILESTONE_TO_EVENT = (
    "MATCH (e:__Entity__) WHERE e.entity_type = 'Milestone' "
    "SET e.entity_type = 'Event', "
    "e.event_type = coalesce(e.event_type, 'milestone'), "
    "e.ontology_version = '1.4.0'"
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
