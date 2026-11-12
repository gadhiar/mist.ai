# backend/knowledge/storage/type_labels.py
"""The ontology type labels an `:__Entity__` node carries beside its partition label.

An entity node's type is its `entity_type` property; that property is the
record every type reader filters on. Since 2026-11-06 the node also carries
the type as a Neo4j label (`:__Entity__:Person`), so the graph can be browsed
and queried by type. The label mirrors the property and never replaces it.
This reverses MIS-177 D2, which kept every type but `User` off `:__Entity__`
nodes as a label.

Neo4j cannot parameterize a label, so a writer interpolates it into the
statement text. Every type label a writer interpolates comes from
`entity_type_label`, which returns a label only for a type in the fixed set
below, each a plain identifier; a type outside the set gets no label.

The set is the ontology's external node types (`_EXTERNAL_NODE_TYPES` in
`backend/knowledge/ontologies/v1_0_0.py`) less `MistIdentity`, which belongs
to the `:__SelfModel__` partition. It is written out here, not derived, so
this module imports nothing; `tests/unit/knowledge/storage/test_type_labels.py`
pins it to the ontology. Retired types (`Topic`, `Milestone`, `RETIRED_TYPE_MAP`
in `backend/knowledge/ontologies/hierarchy.py`) and `Unknown` are not in it.
"""

from __future__ import annotations

ENTITY_TYPE_LABELS: frozenset[str] = frozenset(
    {
        "User",
        "Person",
        "Organization",
        "Technology",
        "Skill",
        "Project",
        "Concept",
        "Event",
        "Goal",
        "Preference",
        "Location",
        "Date",
        "Metric",
        "Document",
        "Abstraction",
        "Pattern",
        "Convention",
        "Mechanism",
        "Strategy",
        "DataStructure",
    }
)


def entity_type_label(entity_type: str) -> str | None:
    """Return the label an `:__Entity__` node of this type carries, or None.

    Args:
        entity_type: The node's `entity_type` value, as given.

    Returns:
        The member of `ENTITY_TYPE_LABELS` equal to `entity_type`, or None for
        any other string (no case folding, no retired-type mapping).
    """
    return entity_type if entity_type in ENTITY_TYPE_LABELS else None
