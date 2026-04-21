"""Snapshot tests for the MIST.AI v1.0.0 ontology definition.

These tests pin the ontology structure so that accidental changes to entity
types, relationship types, confidence policies, or extraction rules cause
immediate, visible test failures.
"""

from __future__ import annotations

import pytest

from backend.knowledge.ontologies.base import KnowledgeDomain
from backend.knowledge.ontologies.v1_0_0 import (
    ALL_EDGE_TYPES,
    ALL_NODE_TYPES,
    CONFIDENCE_BRIDGING,
    CONFIDENCE_EXTERNAL,
    CONFIDENCE_INTERNAL,
    EXTRACTABLE_NODE_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
    ONTOLOGY_V1_0_0,
)

# -------------------------------------------------------------------
# Entity types
# -------------------------------------------------------------------


class TestEntityTypes:
    """Verify the complete set of entity types in the ontology."""

    def test_has_21_entity_types(self):
        assert len(ALL_NODE_TYPES) == 21

    @pytest.mark.parametrize(
        "type_name",
        [
            pytest.param("MistIdentity", id="MistIdentity"),
            pytest.param("MistTrait", id="MistTrait"),
            pytest.param("MistCapability", id="MistCapability"),
            pytest.param("MistPreference", id="MistPreference"),
            pytest.param("MistUncertainty", id="MistUncertainty"),
        ],
    )
    def test_internal_domain_has_5_types(self, type_name: str):
        internal_types = [
            nt for nt in ALL_NODE_TYPES if nt.knowledge_domain == KnowledgeDomain.INTERNAL
        ]

        internal_names = [nt.type_name for nt in internal_types]

        assert len(internal_types) == 5
        assert type_name in internal_names

    @pytest.mark.parametrize(
        "type_name",
        [
            pytest.param("User", id="User"),
            pytest.param("Person", id="Person"),
            pytest.param("Organization", id="Organization"),
            pytest.param("Technology", id="Technology"),
            pytest.param("Skill", id="Skill"),
            pytest.param("Project", id="Project"),
            pytest.param("Concept", id="Concept"),
            pytest.param("Topic", id="Topic"),
            pytest.param("Event", id="Event"),
            pytest.param("Goal", id="Goal"),
            pytest.param("Preference", id="Preference"),
            pytest.param("Location", id="Location"),
        ],
    )
    def test_external_domain_has_12_types(self, type_name: str):
        external_types = [
            nt for nt in ALL_NODE_TYPES if nt.knowledge_domain == KnowledgeDomain.EXTERNAL
        ]

        external_names = [nt.type_name for nt in external_types]

        assert len(external_types) == 12
        assert type_name in external_names

    @pytest.mark.parametrize(
        "type_name",
        [
            pytest.param("LearningEvent", id="LearningEvent"),
            pytest.param("ConversationContext", id="ConversationContext"),
            pytest.param("ExternalSource", id="ExternalSource"),
            pytest.param("VectorChunk", id="VectorChunk"),
        ],
    )
    def test_bridging_domain_has_4_types(self, type_name: str):
        bridging_types = [
            nt for nt in ALL_NODE_TYPES if nt.knowledge_domain == KnowledgeDomain.BRIDGING
        ]

        bridging_names = [nt.type_name for nt in bridging_types]

        assert len(bridging_types) == 4
        assert type_name in bridging_names

    def test_extractable_types_are_external_plus_mist_identity(self):
        # Cluster 1 (Bug J): MistIdentity is promoted from INTERNAL-only to
        # also extractable so that MIST-scope facts ("MIST uses LanceDB")
        # survive extraction. The extractable set is the full 12 external
        # types plus MistIdentity.
        external_names = {
            nt.type_name for nt in ALL_NODE_TYPES if nt.knowledge_domain == KnowledgeDomain.EXTERNAL
        }
        expected = external_names | {"MistIdentity"}

        extractable_set = set(EXTRACTABLE_NODE_TYPES)

        assert len(EXTRACTABLE_NODE_TYPES) == 13
        assert extractable_set == expected


# -------------------------------------------------------------------
# Relationship types
# -------------------------------------------------------------------


class TestRelationshipTypes:
    """Verify the complete set of relationship types in the ontology."""

    def test_has_37_relationship_types(self):
        # Cluster 1 added 4 MIST-scope edges: IMPLEMENTED_WITH,
        # MIST_HAS_CAPABILITY, MIST_HAS_TRAIT, MIST_HAS_PREFERENCE.
        assert len(ALL_EDGE_TYPES) == 37

    def test_extractable_relationships_count(self):
        # 13 user-centric + 8 structural (excludes LEARNED_FROM, ABOUT, SUPERSEDES)
        # + 4 MIST-scope (IMPLEMENTED_WITH, MIST_HAS_CAPABILITY, MIST_HAS_TRAIT,
        # MIST_HAS_PREFERENCE).
        assert len(EXTRACTABLE_RELATIONSHIP_TYPES) == 25

    @pytest.mark.parametrize(
        "type_name",
        [
            pytest.param("IMPLEMENTED_WITH", id="IMPLEMENTED_WITH"),
            pytest.param("MIST_HAS_CAPABILITY", id="MIST_HAS_CAPABILITY"),
            pytest.param("MIST_HAS_TRAIT", id="MIST_HAS_TRAIT"),
            pytest.param("MIST_HAS_PREFERENCE", id="MIST_HAS_PREFERENCE"),
        ],
    )
    def test_mist_scope_edges_are_extractable(self, type_name: str):
        assert type_name in EXTRACTABLE_RELATIONSHIP_TYPES
        assert type_name in {et.type_name for et in ALL_EDGE_TYPES}


# -------------------------------------------------------------------
# Confidence policies
# -------------------------------------------------------------------


class TestConfidencePolicies:
    """Verify confidence decay configuration per knowledge domain."""

    def test_internal_policy_no_decay(self):
        assert CONFIDENCE_INTERNAL.decay_enabled is False
        assert CONFIDENCE_INTERNAL.decay_half_life_days is None

    def test_external_policy_has_180_day_half_life(self):
        assert CONFIDENCE_EXTERNAL.decay_enabled is True
        assert CONFIDENCE_EXTERNAL.decay_half_life_days == 180

    def test_bridging_policy_has_365_day_half_life(self):
        assert CONFIDENCE_BRIDGING.decay_enabled is True
        assert CONFIDENCE_BRIDGING.decay_half_life_days == 365


# -------------------------------------------------------------------
# Ontology version
# -------------------------------------------------------------------


class TestOntologyVersion:
    """Verify the top-level ontology version metadata."""

    def test_version_is_1_0_0(self):
        assert ONTOLOGY_V1_0_0.version == "1.0.0"
