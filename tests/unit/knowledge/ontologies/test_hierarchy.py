"""Tests for the v1.4.0 hierarchy module.

Verifies retired-type canonicalization and the parent-type expansion
rule that makes a slot accepting Concept also accept Abstraction.
"""

from __future__ import annotations


def test_retired_types_resolve_to_canonical():
    from backend.knowledge.ontologies.hierarchy import canonical_type

    assert canonical_type("Topic") == "Concept"
    assert canonical_type("Milestone") == "Event"
    assert canonical_type("Concept") == "Concept"  # passthrough


def test_expand_allowed_adds_parent_when_anchor_child_present():
    from backend.knowledge.ontologies.hierarchy import expand_allowed_with_parents

    assert "Abstraction" in expand_allowed_with_parents(("Concept", "Technology"))
    assert "Abstraction" not in expand_allowed_with_parents(("Mechanism", "Pattern"))


def test_children_of_abstraction():
    from backend.knowledge.ontologies.hierarchy import children_of

    kids = children_of("Abstraction")
    assert kids == {
        "Concept",
        "Skill",
        "Pattern",
        "Strategy",
        "Mechanism",
        "Convention",
        "DataStructure",
    }


def test_dedup_type_filter_widens_cluster_types_to_parent_and_siblings():
    from backend.knowledge.ontologies.hierarchy import dedup_type_filter

    # Concept's parent is Abstraction; widening returns Abstraction + all its children
    out = dedup_type_filter("Concept")
    assert out == sorted(set(out))  # sorted, deduped
    assert "Concept" in out
    assert "Abstraction" in out
    # All siblings must be present too
    assert "Skill" in out
    assert "Pattern" in out
    assert len(out) >= 3


def test_dedup_type_filter_returns_exact_for_non_cluster_type():
    from backend.knowledge.ontologies.hierarchy import dedup_type_filter

    assert dedup_type_filter("User") == ["User"]


def test_dedup_type_filter_abstraction_self_widens():
    # "Abstraction" itself: parent_of returns None, but the guard
    # (entity_type == "Abstraction") kicks in and widens to Abstraction + children.
    from backend.knowledge.ontologies.hierarchy import children_of, dedup_type_filter

    out = dedup_type_filter("Abstraction")
    assert out == sorted(set(out))
    assert "Abstraction" in out
    expected = sorted({"Abstraction", *children_of("Abstraction")})
    assert out == expected
