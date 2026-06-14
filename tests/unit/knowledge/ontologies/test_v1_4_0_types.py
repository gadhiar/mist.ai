"""Tests for v1.4.0 ontology type changes.

Verifies:
- Topic and Milestone are removed from ALL_NODE_TYPE_NAMES.
- Abstraction supertype is present with no parent.
- Seven abstract leaves declare parent_type="Abstraction".
- Event gains the significance optional property.
"""

from __future__ import annotations


def test_topic_and_milestone_removed():
    from backend.knowledge.ontologies.v1_0_0 import ALL_NODE_TYPE_NAMES

    assert "Topic" not in ALL_NODE_TYPE_NAMES
    assert "Milestone" not in ALL_NODE_TYPE_NAMES


def test_abstraction_present_and_childless():
    from backend.knowledge.ontologies.v1_0_0 import ALL_NODE_TYPE_NAMES, ALL_NODE_TYPES

    assert "Abstraction" in ALL_NODE_TYPE_NAMES
    abs_nt = next(n for n in ALL_NODE_TYPES if n.type_name == "Abstraction")
    assert abs_nt.parent_type is None


def test_seven_children_have_parent():
    from backend.knowledge.ontologies.v1_0_0 import ALL_NODE_TYPES

    idx = {n.type_name: n for n in ALL_NODE_TYPES}
    for child in (
        "Concept",
        "Skill",
        "Pattern",
        "Strategy",
        "Mechanism",
        "Convention",
        "DataStructure",
    ):
        assert idx[child].parent_type == "Abstraction", child


def test_event_gains_significance_property():
    from backend.knowledge.ontologies.v1_0_0 import ALL_NODE_TYPES

    event = next(n for n in ALL_NODE_TYPES if n.type_name == "Event")
    assert any(p.name == "significance" for p in event.optional_properties)
