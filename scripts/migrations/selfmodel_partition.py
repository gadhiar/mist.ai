"""R1.0 migration: relabel self-model nodes from :__Entity__ into :__SelfModel__.

Idempotent. Moves the five self-model entity types out of the universal
:__Entity__ partition into the dedicated :__SelfModel__ partition (mirroring
:__Provenance__) so an :__Entity__-scoped rebuild/reset can never delete the
self-model, and backfills typed labels for nodes that internal/skill derivation
created with only :__Entity__ + an entity_type property (the historical
typed-label gap).

Run inside the container after the R1.0 writers ship:
    python -c "
    import asyncio, sys
    sys.path.insert(0, '.')
    from backend.knowledge.config import get_config
    from backend.knowledge.storage.neo4j_connection import Neo4jConnection
    from backend.knowledge.storage.graph_executor import GraphExecutor
    from scripts.migrations.selfmodel_partition import migrate

    config = get_config()
    conn = Neo4jConnection(config.neo4j)
    conn.connect()
    asyncio.run(migrate(GraphExecutor(conn)))
    conn.disconnect()
    "

Required before R1.2's build-then-swap copy-forward (which copies the
:__SelfModel__ partition): the partition must exist on the live graph first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.knowledge.storage.partitions import SELF_MODEL_TYPES

if TYPE_CHECKING:
    from backend.knowledge.storage.graph_executor import GraphExecutor

_TYPES_LIST = "[" + ", ".join(f"'{t}'" for t in sorted(SELF_MODEL_TYPES)) + "]"

# 1) Move every self-model node out of :__Entity__ into :__SelfModel__.
#    After the first run, no :__Entity__ node carries a self-model entity_type,
#    so the MATCH is empty and the statement is a structural no-op.
_RELABEL = (
    f"MATCH (e:__Entity__) WHERE e.entity_type IN {_TYPES_LIST} "
    "REMOVE e:__Entity__ SET e:__SelfModel__"
)

# 2) Backfill the typed label for any self-model node that lacks it (gap-created
#    nodes carried only :__Entity__ + entity_type). One statement per type; the
#    `AND NOT e:Type` guard makes re-runs no-ops.
_BACKFILL = [
    (f"MATCH (e:__SelfModel__) WHERE e.entity_type = '{typed}' AND NOT e:{typed} " f"SET e:{typed}")
    for typed in sorted(SELF_MODEL_TYPES)
]

CYPHER: list[str] = [_RELABEL, *_BACKFILL]


async def migrate(executor: GraphExecutor) -> None:
    """Relabel self-model nodes into :__SelfModel__ and backfill typed labels.

    Idempotent: each statement only touches nodes still in the pre-migration
    state; after the first run every statement matches zero nodes.

    Args:
        executor: An async GraphExecutor (or compatible fake) exposing
            `execute_write(query: str, params: dict | None) -> list[dict]`.
    """
    for query in CYPHER:
        await executor.execute_write(query, {})
