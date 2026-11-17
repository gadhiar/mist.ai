"""Facets: the groups of facts a type-scoped question asks for, as one retrievable item each.

"Which organisations am I a member of?" targets every fact with the user as subject, MEMBER_OF
as relation and an Organization as object. Each member's own sentence names only its object,
so no single fact matches the question as a whole, and a fetch that finds a few of them by name
misses the rest. A facet is that group: a subject, a relation and an object type held by two or
more distinct fact keys. The retrieval view words the group, not its members
(`render_facet_view` in backend/memory/retrieval_view.py), so the facet's text stays the same as
members are added.

This module holds only the pure pieces: the key and the grouping. Facets are grouped over every
stored version, so a facet may have fewer than two members current at a given time; which
members are current is decided when the facet is fetched, not here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from .type_terms import TYPE_TERMS

# The ontology type of the user's own node (`USER_TYPE` in retrieval_view, which imports this
# module, so it is not imported from there). A facet of "user" objects would word the user as a
# type ("a user"), which no question asks for.
_USER_TYPE = "User"


def facet_key(subject_id: str, rel_type: str, object_type: str) -> str:
    """The facet's stable identity: the hex sha256 of the three parts joined by `|`.

    Args:
        subject_id: The subject node's id.
        rel_type: The relation type.
        object_type: The object node's `entity_type`.

    Returns:
        A 64-character lower-case hex digest.
    """
    joined = f"{subject_id}|{rel_type}|{object_type}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FacetSpec:
    """One facet: the group's identity, its subject as last recorded, and its members.

    Attributes:
        facet_key: `facet_key(subject_id, rel_type, object_type)`.
        subject_id: The subject node's id.
        subject_name: The subject's display name on the newest recorded member row.
        subject_type: The subject's `entity_type` on the same row; `User` makes the view second
            person.
        rel_type: The relation type.
        object_type: The objects' shared `entity_type`, a `TYPE_TERMS` key other than `User`.
        member_keys: The distinct fact keys of the group, newest recorded first (see
            `facets_of`).
    """

    facet_key: str
    subject_id: str
    subject_name: str
    subject_type: str | None
    rel_type: str
    object_type: str
    member_keys: tuple[str, ...]


class FacetMemberRow(Protocol):
    """The fields grouping reads, all of which the index's `IndexedFact` has."""

    @property
    def fact_key(self) -> str:
        """The fact's identity across versions; a facet counts distinct ones."""
        ...

    @property
    def subject_id(self) -> str:
        """The subject node's id, the first part of the facet key."""
        ...

    @property
    def subject_name(self) -> str:
        """The subject's display name, which the facet's view names."""
        ...

    @property
    def subject_type(self) -> str | None:
        """The subject node's `entity_type`."""
        ...

    @property
    def rel_type(self) -> str:
        """The relation type, the second part of the facet key."""
        ...

    @property
    def object_type(self) -> str | None:
        """The object node's `entity_type`, the third part of the facet key."""
        ...

    @property
    def recorded_at(self) -> str | None:
        """When the version was recorded, which orders the members."""
        ...


def facets_of(rows: Iterable[FacetMemberRow]) -> dict[str, FacetSpec]:
    """Group fact versions into facets.

    A group is the rows sharing a subject id, a relation type and an object type. It is a facet
    when its object type is a `TYPE_TERMS` key other than `User` and it holds two or more
    distinct fact keys: versions of one fact are one member, since a facet stands for a set of
    different objects.

    Order is independent of the order of `rows`. `recorded_at` compares as stored text, as the
    index orders versions, with None as the oldest. `member_keys` orders each fact key by its
    newest version's `recorded_at`, newest first; equal times go to the smaller fact key first.
    `subject_name` and `subject_type` come from the newest recorded row, so a renamed subject
    reads as it was last recorded: the newest version of the first member key, and among that
    key's versions recorded at the same time, the smallest (`subject_name`, `subject_type`).

    Args:
        rows: Fact versions, typically every version the index stores; read once.

    Returns:
        The facets keyed by `facet_key`, in key order.
    """
    groups: dict[tuple[str, str, str], list[FacetMemberRow]] = {}
    for row in rows:
        object_type = row.object_type
        if not object_type or object_type == _USER_TYPE or object_type not in TYPE_TERMS:
            continue
        groups.setdefault((row.subject_id, row.rel_type, object_type), []).append(row)

    facets: dict[str, FacetSpec] = {}
    for (subject_id, rel_type, object_type), members in groups.items():
        newest: dict[str, str] = {}
        for row in members:
            recorded = row.recorded_at or ""
            if recorded >= newest.get(row.fact_key, ""):
                newest[row.fact_key] = recorded
        if len(newest) < 2:
            continue
        # Two stable passes: fact key ascending, then recorded descending, so ties keep key order.
        member_keys = sorted(newest)
        member_keys.sort(key=newest.__getitem__, reverse=True)
        first_key = member_keys[0]
        subject = min(
            (
                row
                for row in members
                if row.fact_key == first_key and (row.recorded_at or "") == newest[first_key]
            ),
            key=lambda row: (row.subject_name, row.subject_type or ""),
        )
        key = facet_key(subject_id, rel_type, object_type)
        facets[key] = FacetSpec(
            facet_key=key,
            subject_id=subject_id,
            subject_name=subject.subject_name,
            subject_type=subject.subject_type,
            rel_type=rel_type,
            object_type=object_type,
            member_keys=tuple(member_keys),
        )
    return dict(sorted(facets.items()))


__all__ = ["FacetMemberRow", "FacetSpec", "facet_key", "facets_of"]
