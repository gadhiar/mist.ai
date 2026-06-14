"""Tests for v1.4.0 edge allowed-set transform.

Verifies:
- Topic is replaced by Concept in edge allowed-type sets.
- Milestone is dropped (mapped to Event) in temporal edges.
- Abstraction parent is added wherever Concept is present.
"""

from __future__ import annotations


def test_topic_replaced_by_concept_on_edges():
    from backend.knowledge.ontologies.v1_0_0 import EDGE_TYPES_BY_NAME

    knows = EDGE_TYPES_BY_NAME["KNOWS"]
    assert "Topic" not in knows.allowed_target_types
    assert "Concept" in knows.allowed_target_types


def test_milestone_dropped_from_temporal_edges():
    from backend.knowledge.ontologies.v1_0_0 import EDGE_TYPES_BY_NAME

    assert "Milestone" not in EDGE_TYPES_BY_NAME["OCCURRED_ON"].allowed_source_types
    assert "Event" in EDGE_TYPES_BY_NAME["OCCURRED_ON"].allowed_source_types


def test_abstraction_added_where_concept_present():
    from backend.knowledge.ontologies.v1_0_0 import EDGE_TYPES_BY_NAME

    assert "Abstraction" in EDGE_TYPES_BY_NAME["KNOWS"].allowed_target_types
    assert "Abstraction" not in EDGE_TYPES_BY_NAME["MECHANISM_OF"].allowed_source_types
