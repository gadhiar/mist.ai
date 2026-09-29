# backend/knowledge/storage/partitions.py
"""Canonical Neo4j partition labels and the self-model entity-type set.

Three structural partitions share the graph, each distinguished by a universal
label and kept disjoint by construction (a node carries exactly one partition
label):

- ``__Entity__``    -- user/world facts (the deterministic projection of the
                       utterance log; wiped/rebuilt by R1).
- ``__Provenance__``-- source-anchor metadata (ConversationContext, VaultNote,
                       ExternalSource, ...). Survives an ``__Entity__`` reset
                       because it never carries ``__Entity__``.
- ``__SelfModel__`` -- MIST's self-model (identity/traits/capabilities/
                       preferences/uncertainties). Preserved across rebuilds
                       for the same structural reason.

This module is the single definition of the self-model type set so writers,
schema setup, and the migration agree.
"""

from __future__ import annotations

ENTITY_LABEL = "__Entity__"
PROVENANCE_LABEL = "__Provenance__"
SELF_MODEL_LABEL = "__SelfModel__"

# The stored id of the singleton user node (:__Entity__ {id: "user"}). Neo4j
# property matching is case-sensitive, so every graph read that anchors on the
# user node by id must use exactly this value ("User" matches nothing).
# Writers of this id, established by:
#   grep -n '"id": "user"' backend/knowledge/extraction/prompts.py   (line 49)
#   grep -n "id: 'user'" backend/knowledge/curation/skill_derivation.py (line 173)
# Readers that take it by id: GraphStore.get_user_relationships_to_entities
# (MATCH (user:__Entity__ {id: $user_id})) via KnowledgeRetriever.retrieve.
USER_ENTITY_ID = "user"

# The five entity types that live in the :__SelfModel__ partition. MistIdentity
# is the singleton root; the other four hang off it via HAS_* edges.
SELF_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "MistIdentity",
        "MistTrait",
        "MistCapability",
        "MistPreference",
        "MistUncertainty",
    }
)
