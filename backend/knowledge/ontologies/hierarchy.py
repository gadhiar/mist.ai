"""Single source of truth for the v1.4.0 entity-type hierarchy.

The reconciliation engine pattern (derive_assertion_kind) -- one derivation,
many consumers -- applied to type canonicalization. The normalizer/resolver,
the ontology edge-set build, and both F2 scorers import from here so a hierarchy
change cannot drift across consumers.
"""

from __future__ import annotations

# Types removed in v1.4.0 -> their canonical replacement. Derived consumers
# (normalizer retired-type coercion, scorer canonical_type) read this map.
RETIRED_TYPE_MAP: dict[str, str] = {
    "Topic": "Concept",
    "Milestone": "Event",
}

# Each abstract supertype maps to its anchor (merge-target) child. A type slot
# accepts the supertype iff it already accepts the anchor child. One entry today;
# future supertypes (Agent, TemporalEntity) add one line each.
ANCHOR_CHILD: dict[str, str] = {
    "Abstraction": "Concept",
}


def canonical_type(type_name: str) -> str:
    """Resolve a retired type to its canonical replacement; passthrough otherwise."""
    return RETIRED_TYPE_MAP.get(type_name, type_name)


def _node_index() -> dict[str, object]:
    # Imported lazily to avoid a circular import with v1_0_0 (which will import
    # RETIRED_TYPE_MAP / expand_allowed_with_parents from this module at build time).
    from backend.knowledge.ontologies.v1_0_0 import ALL_NODE_TYPES

    return {nt.type_name: nt for nt in ALL_NODE_TYPES}


def parent_of(type_name: str) -> str | None:
    nt = _node_index().get(type_name)
    return getattr(nt, "parent_type", None) if nt is not None else None


def children_of(parent: str) -> set[str]:
    return {
        name for name, nt in _node_index().items() if getattr(nt, "parent_type", None) == parent
    }


def anchor_child_of(parent: str) -> str | None:
    return ANCHOR_CHILD.get(parent)


def expand_allowed_with_parents(allowed: tuple[str, ...]) -> tuple[str, ...]:
    """Add a supertype P to an allowed-type tuple iff P's anchor child is present.

    Deterministic and idempotent. This is the "accepts parent iff accepts Concept"
    rule, generalized over ANCHOR_CHILD.
    """
    result = list(allowed)
    for parent, anchor in ANCHOR_CHILD.items():
        if anchor in allowed and parent not in result:
            result.append(parent)
    return tuple(result)


def dedup_type_filter(entity_type: str) -> list[str]:
    """Types that count as 'same kind' for embedding dedup.

    For abstract-cluster types, widen to the parent + all siblings so a Concept
    and a Skill of the same entity can still merge; non-cluster types match
    exactly. Result is sorted + deduped for deterministic candidate scoping.
    """
    parent = parent_of(entity_type) or (entity_type if entity_type == "Abstraction" else None)
    if parent is None:
        return [entity_type]
    return sorted({parent, *children_of(parent), entity_type})
