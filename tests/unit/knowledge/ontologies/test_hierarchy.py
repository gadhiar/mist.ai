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
