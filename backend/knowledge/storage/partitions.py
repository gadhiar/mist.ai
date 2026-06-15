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
