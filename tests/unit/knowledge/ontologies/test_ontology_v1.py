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

    def test_has_26_entity_types(self):
        # Cluster 8 Phase 6 (ADR-010) added VaultNote as a bridging/provenance type.
        # Post-MVP additive (2026-04-22) added Date, Milestone, Metric, Document
        # as EXTERNAL extractable types.
        assert len(ALL_NODE_TYPES) == 26

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
            # Post-MVP additive (2026-04-22):
            pytest.param("Date", id="Date"),
            pytest.param("Milestone", id="Milestone"),
            pytest.param("Metric", id="Metric"),
            pytest.param("Document", id="Document"),
        ],
    )
    def test_external_domain_has_16_types(self, type_name: str):
        external_types = [
            nt for nt in ALL_NODE_TYPES if nt.knowledge_domain == KnowledgeDomain.EXTERNAL
        ]

        external_names = [nt.type_name for nt in external_types]

        assert len(external_types) == 16
        assert type_name in external_names

    @pytest.mark.parametrize(
        "type_name",
        [
            pytest.param("LearningEvent", id="LearningEvent"),
            pytest.param("ConversationContext", id="ConversationContext"),
            pytest.param("ExternalSource", id="ExternalSource"),
            pytest.param("VectorChunk", id="VectorChunk"),
            pytest.param("VaultNote", id="VaultNote"),
        ],
    )
    def test_bridging_domain_has_5_types(self, type_name: str):
        # Cluster 8 Phase 6 (ADR-010) added VaultNote.
        bridging_types = [
            nt for nt in ALL_NODE_TYPES if nt.knowledge_domain == KnowledgeDomain.BRIDGING
        ]

        bridging_names = [nt.type_name for nt in bridging_types]

        assert len(bridging_types) == 5
        assert type_name in bridging_names

    def test_vault_note_required_property_is_path(self):
        # ADR-010 Phase 6: VaultNote nodes are MERGE-idempotent keyed by `path`.
        # The schema must declare `path` as a required property.
        from backend.knowledge.ontologies.v1_0_0 import VAULT_NOTE

        required_names = [p.name for p in VAULT_NOTE.required_properties]
        assert required_names == ["path"]
        assert VAULT_NOTE.knowledge_domain == KnowledgeDomain.BRIDGING

    def test_derived_from_allows_vault_note_target(self):
        # ADR-010 Phase 6: DERIVED_FROM extends to entity -> VaultNote for the
        # vault-rebuild contract. VectorChunk and ExternalSource targets remain.
        from backend.knowledge.ontologies.v1_0_0 import DERIVED_FROM

        assert "VaultNote" in DERIVED_FROM.allowed_target_types
        assert "VectorChunk" in DERIVED_FROM.allowed_target_types
        assert "ExternalSource" in DERIVED_FROM.allowed_target_types
        # MistIdentity must be a permitted source so MIST-scope entities can
        # carry vault-note provenance.
        assert "MistIdentity" in DERIVED_FROM.allowed_source_types

    def test_extractable_types_are_external_plus_mist_identity(self):
        # Cluster 1 (Bug J): MistIdentity is promoted from INTERNAL-only to
        # also extractable so that MIST-scope facts ("MIST uses LanceDB")
        # survive extraction. The extractable set is the full external
        # types plus MistIdentity.
        # Post-MVP additive (2026-04-22): 12 -> 16 external, 13 -> 17 extractable.
        external_names = {
            nt.type_name for nt in ALL_NODE_TYPES if nt.knowledge_domain == KnowledgeDomain.EXTERNAL
        }
        expected = external_names | {"MistIdentity"}

        extractable_set = set(EXTRACTABLE_NODE_TYPES)

        assert len(EXTRACTABLE_NODE_TYPES) == 17
        assert extractable_set == expected


# -------------------------------------------------------------------
# Relationship types
# -------------------------------------------------------------------


class TestRelationshipTypes:
    """Verify the complete set of relationship types in the ontology."""

    def test_has_41_relationship_types(self):
        # Cluster 1 added 4 MIST-scope edges: IMPLEMENTED_WITH,
        # MIST_HAS_CAPABILITY, MIST_HAS_TRAIT, MIST_HAS_PREFERENCE.
        # Post-MVP additive (2026-04-22) added 4 temporal / quantified / document
        # edges: OCCURRED_ON, HAS_METRIC, REFERENCES_DOCUMENT, PRECEDED_BY.
        assert len(ALL_EDGE_TYPES) == 41

    def test_extractable_relationships_count(self):
        # 13 user-centric + 8 original structural (excludes LEARNED_FROM, ABOUT,
        # SUPERSEDES + provenance SOURCED_FROM/REFERENCES/DERIVED_FROM) + 4
        # MIST-scope (IMPLEMENTED_WITH, MIST_HAS_CAPABILITY, MIST_HAS_TRAIT,
        # MIST_HAS_PREFERENCE) + 4 post-MVP additive structural (OCCURRED_ON,
        # HAS_METRIC, REFERENCES_DOCUMENT, PRECEDED_BY).
        assert len(EXTRACTABLE_RELATIONSHIP_TYPES) == 29

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
# Post-MVP additive: temporal / quantified / document invariants (2026-04-22)
# -------------------------------------------------------------------


class TestNewNodeTypeInvariants:
    """Each post-MVP additive node type must be EXTERNAL-domain, have the
    promised required-property contract, and be present in the extractable
    set so the LLM extractor sees it as a valid target.

    `expected_allowed_values` pins the enum vocabulary for properties that
    restrict inputs (Milestone.significance, Document.doc_type). Silent
    widening of these enums would break downstream normalization and
    retrieval filters, so this test catches it.
    """

    @pytest.mark.parametrize(
        "type_const_name, expected_required, expected_optional, expected_allowed_values",
        [
            pytest.param(
                "DATE",
                ["iso_date"],
                [],
                {},
                id="Date",
            ),
            pytest.param(
                "MILESTONE",
                [],
                ["significance"],
                {"significance": ("high", "medium", "low")},
                id="Milestone",
            ),
            pytest.param(
                "METRIC",
                ["value", "unit"],
                ["as_of_date"],
                {},
                id="Metric",
            ),
            pytest.param(
                "DOCUMENT",
                ["title"],
                ["doc_type", "identifier"],
                {
                    "doc_type": (
                        "adr",
                        "spec",
                        "paper",
                        "book",
                        "article",
                        "other",
                    ),
                },
                id="Document",
            ),
        ],
    )
    def test_new_node_type_contract(
        self,
        type_const_name: str,
        expected_required: list[str],
        expected_optional: list[str],
        expected_allowed_values: dict[str, tuple[str, ...]],
    ):
        import backend.knowledge.ontologies.v1_0_0 as onto

        node = getattr(onto, type_const_name)
        assert node.knowledge_domain == KnowledgeDomain.EXTERNAL
        required_names = [p.name for p in node.required_properties]
        optional_names = [p.name for p in node.optional_properties]
        assert required_names == expected_required
        assert optional_names == expected_optional
        assert node.type_name in EXTRACTABLE_NODE_TYPES

        # Pin enum vocabularies for properties that use allowed_values.
        all_props = list(node.required_properties) + list(node.optional_properties)
        by_name = {p.name: p for p in all_props}
        for prop_name, expected_values in expected_allowed_values.items():
            actual = by_name[prop_name].allowed_values
            assert actual == expected_values, (
                f"{type_const_name}.{prop_name} allowed_values drifted: "
                f"expected {expected_values!r}, got {actual!r}"
            )


class TestNewEdgeTypeInvariants:
    """Each post-MVP additive edge must have the exact allowed source/target
    sets from the plan and be present in the extractable relationship set.
    """

    @pytest.mark.parametrize(
        "edge_const_name, expected_sources, expected_targets",
        [
            pytest.param(
                "OCCURRED_ON",
                ("Event", "Milestone"),
                ("Date",),
                id="OCCURRED_ON",
            ),
            pytest.param(
                "HAS_METRIC",
                ("User", "Project", "Technology", "Skill", "Concept", "Goal"),
                ("Metric",),
                id="HAS_METRIC",
            ),
            pytest.param(
                "REFERENCES_DOCUMENT",
                (
                    "User",
                    "MistIdentity",
                    "Project",
                    "Concept",
                    "Topic",
                    "Goal",
                    "Event",
                ),
                ("Document",),
                id="REFERENCES_DOCUMENT",
            ),
            pytest.param(
                "PRECEDED_BY",
                ("Event", "Milestone"),
                ("Event", "Milestone", "Date"),
                id="PRECEDED_BY",
            ),
        ],
    )
    def test_new_edge_type_contract(
        self,
        edge_const_name: str,
        expected_sources: tuple[str, ...],
        expected_targets: tuple[str, ...],
    ):
        import backend.knowledge.ontologies.v1_0_0 as onto

        edge = getattr(onto, edge_const_name)
        # Tuple equality (not set): catches accidental duplicates + ordering
        # drift. The constants are tuple-typed, so this is the precise check.
        assert edge.allowed_source_types == expected_sources
        assert edge.allowed_target_types == expected_targets
        assert edge.type_name in EXTRACTABLE_RELATIONSHIP_TYPES


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
