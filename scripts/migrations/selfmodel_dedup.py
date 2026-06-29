"""R1.3 (self-model slice) migration: delete the vault->graph self-model shadows.

Idempotent. MIST's self-model exists twice in the :__SelfModel__ partition: the
canonical kebab set (`trait-warm`, `cap-*`, `pref-*`) seeded once by the ADR-008
bootstrap (`apply_seed`), carrying full content (`axis`/`description`/`embedding`)
and currency-stamped `HAS_*` edges; and a sparse mist-prefixed shadow set
(`mist-trait-Warm`, ...) written by the vault->graph regenerator on
`identity/mist.md` edits, carrying no content and no currency stamps.
The shadows double MIST's persona injection.

The :__SelfModel__ partition is canonical and preserved (R1 truth model: the
self-model is not vault-derived). This migration deletes the mist-prefixed
shadow traits/capabilities/preferences. The singleton identity root
(id 'mist-identity', :MistIdentity) is deliberately untouched: both seed paths
MERGE onto the same root node, so there is no identity shadow, and the
selfmodel_partition migration already reconciled its gap-window duplicate.

Run inside the container after the vault->graph re-derivation path is retired:
    python -c "
    import asyncio, sys
    sys.path.insert(0, '.')
    from backend.knowledge.config import get_config
    from backend.knowledge.storage.neo4j_connection import Neo4jConnection
    from backend.knowledge.storage.graph_executor import GraphExecutor
    from scripts.migrations.selfmodel_dedup import migrate

    config = get_config()
    conn = Neo4jConnection(config.neo4j)
    conn.connect()
    asyncio.run(migrate(GraphExecutor(conn)))
    conn.disconnect()
    "

Expected on live: 41 -> 21 :__SelfModel__ nodes (1 identity + 9 traits + 5
capabilities + 6 preferences); the persona reads each attribute once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.knowledge.storage.graph_executor import GraphExecutor

# Delete the sparse vault->graph shadow nodes. Double-guarded so a canonical
# kebab node (id 'trait-warm', no 'mist-' prefix) can never be caught: the node
# must BOTH carry a self-model trait/capability/preference entity_type AND have
# an id under the regenerator's 'mist-<kind>-' scheme. MistIdentity is excluded
# from the type set (the singleton root is the keeper). Idempotent: after the
# first run no 'mist-*' shadow remains, so re-runs match nothing.
_DELETE_SHADOWS = (
    "MATCH (s:__SelfModel__) "
    "WHERE s.entity_type IN ['MistTrait', 'MistCapability', 'MistPreference'] "
    "AND (s.id STARTS WITH 'mist-trait-' "
    "OR s.id STARTS WITH 'mist-cap-' "
    "OR s.id STARTS WITH 'mist-pref-') "
    "DETACH DELETE s"
)

CYPHER: list[str] = [_DELETE_SHADOWS]


async def migrate(executor: GraphExecutor) -> None:
    """Delete the mist-prefixed self-model shadow nodes (idempotent).

    Args:
        executor: An async GraphExecutor (or compatible fake) exposing
            `execute_write(query: str, params: dict | None) -> list[dict]`.
    """
    for query in CYPHER:
        await executor.execute_write(query, {})
