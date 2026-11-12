"""Single source of truth for the v1.4.0 entity-type hierarchy.

The reconciliation engine pattern (derive_assertion_kind) -- one derivation,
many consumers -- applied to type canonicalization. The normalizer/resolver,
the ontology edge-set build, and both F2 scorers import from here so a hierarchy
change cannot drift across consumers.

The data is stated once, in the v1.4.0 pack's `types.py` (`RETIRED_TYPES` and
`SUPERTYPES`), which the pack's manifest is built from too. This module reads it
from there when it is first asked, never at import: `types.py` imports this
module, and builds its edge sets with `expand_allowed_with_parents` while it is
still loading. `RETIRED_TYPE_MAP` and `ANCHOR_CHILD` are served by the module's
`__getattr__`, each as one dict built once.
"""

from __future__ import annotations

from functools import cache


def _pack_types() -> object:
    # Imported at each call: `types.py` imports this module, so the import cannot stand
    # at the top. While `types.py` is loading this returns the module as far as it has
    # run, which is past the two tuples by the time it builds an edge set.
    from backend.knowledge.ontologies.packs.v1_4_0 import types

    return types


@cache
def _retired_type_map() -> dict[str, str]:
    """Types removed in v1.4.0 -> their canonical replacement.

    Derived consumers (normalizer retired-type coercion, scorer canonical_type) read
    this map as `RETIRED_TYPE_MAP`.
    """
    return dict(_pack_types().RETIRED_TYPES)


@cache
def _anchor_child() -> dict[str, str]:
    """Each abstract supertype -> its anchor (merge-target) child, read as `ANCHOR_CHILD`.

    A type slot accepts the supertype iff it already accepts the anchor child.
    """
    return dict(_pack_types().SUPERTYPES)


_LAZY = {"RETIRED_TYPE_MAP": _retired_type_map, "ANCHOR_CHILD": _anchor_child}


def __getattr__(name: str) -> dict[str, str]:
    build = _LAZY.get(name)
    if build is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return build()


def canonical_type(type_name: str) -> str:
    """Resolve a retired type to its canonical replacement; passthrough otherwise."""
    return _retired_type_map().get(type_name, type_name)


def _node_index() -> dict[str, object]:
    # Imported lazily to avoid a circular import with v1_0_0 (which imports
    # expand_allowed_with_parents from this module at build time).
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
    return _anchor_child().get(parent)


def expand_allowed_with_parents(allowed: tuple[str, ...]) -> tuple[str, ...]:
    """Add a supertype P to an allowed-type tuple iff P's anchor child is present.

    Deterministic and idempotent. This is the "accepts parent iff accepts Concept"
    rule, generalized over ANCHOR_CHILD.
    """
    result = list(allowed)
    for parent, anchor in _anchor_child().items():
        if anchor in allowed and parent not in result:
            result.append(parent)
    return tuple(result)


def dedup_type_filter(entity_type: str) -> list[str]:
    """Types that count as 'same kind' for embedding dedup.

    For abstract-cluster types, widen to the parent + all siblings so a Concept
    and a Skill of the same entity can still merge; non-cluster types match
    exactly. Result is sorted + deduped for deterministic candidate scoping.
    """
    parent = parent_of(entity_type) or (entity_type if entity_type in _anchor_child() else None)
    if parent is None:
        return [entity_type]
    return sorted({parent, *children_of(parent), entity_type})
