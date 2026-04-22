"""Tests for ExtractionValidator extraction pipeline stage."""

import pytest

from backend.knowledge.extraction.ontology_extractor import (
    ExtractionResult,
    OntologyConstrainedExtractor,
)
from backend.knowledge.extraction.validator import ExtractionValidator, ValidationResult


def _make_entity(*, entity_id: str = "python", entity_type: str = "Technology") -> dict:
    """Build a valid entity dict."""
    return {"id": entity_id, "type": entity_type, "name": entity_id}


def _make_relationship(
    *,
    rel_type: str = "USES",
    source: str = "user",
    target: str = "python",
    confidence: float = 0.90,
) -> dict:
    """Build a valid relationship dict."""
    return {
        "type": rel_type,
        "source": source,
        "target": target,
        "properties": {"confidence": confidence},
    }


def _make_extraction(
    *,
    entities: list[dict] | None = None,
    relationships: list[dict] | None = None,
) -> ExtractionResult:
    """Build an ExtractionResult with sensible defaults."""
    if entities is None:
        entities = [
            _make_entity(entity_id="user", entity_type="User"),
            _make_entity(entity_id="python", entity_type="Technology"),
        ]
    if relationships is None:
        relationships = [
            _make_relationship(source="user", target="python"),
        ]
    return ExtractionResult(
        entities=entities,
        relationships=relationships,
        source_utterance="test",
    )


class TestValidEntities:
    """Valid entities pass through validation."""

    def test_valid_entities_pass_through(self):
        # Arrange
        validator = ExtractionValidator()
        extraction = _make_extraction()

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.entities) == 2
        assert result.entities[0]["id"] == "user"
        assert result.entities[1]["id"] == "python"


class TestInvalidEntityType:
    """Entities with invalid types are rejected."""

    def test_invalid_entity_type_is_rejected(self):
        # Arrange
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="user", entity_type="User"),
                _make_entity(entity_id="thing", entity_type="InvalidType"),
            ],
            relationships=[],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.entities) == 1
        assert result.entities[0]["id"] == "user"
        assert any("InvalidType" in e for e in result.errors)


class TestEmptyEntityId:
    """Entities with empty IDs are rejected."""

    @pytest.mark.parametrize(
        "entity_id",
        [
            pytest.param("", id="empty-string"),
            pytest.param("   ", id="whitespace-only"),
        ],
    )
    def test_empty_entity_id_is_rejected(self, entity_id: str):
        # Arrange
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id=entity_id, entity_type="Technology"),
            ],
            relationships=[],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.entities) == 0
        assert len(result.errors) > 0


class TestInvalidRelationshipType:
    """Relationships with unknown types are rejected."""

    def test_invalid_relationship_type_is_rejected(self):
        # Arrange
        validator = ExtractionValidator()
        extraction = _make_extraction(
            relationships=[
                _make_relationship(rel_type="INVENTED_BY"),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.relationships) == 0
        assert any("INVENTED_BY" in e for e in result.errors)


class TestSelfReferencingRelationship:
    """Self-referencing relationships are rejected."""

    def test_self_referencing_relationship_is_rejected(self):
        # Arrange
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="python", entity_type="Technology"),
            ],
            relationships=[
                _make_relationship(source="python", target="python", rel_type="RELATED_TO"),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.relationships) == 0
        assert any("Self-referencing" in w for w in result.warnings)


class TestConfidenceThreshold:
    """Relationships below the confidence threshold are rejected."""

    def test_confidence_below_threshold_is_rejected(self):
        # Arrange
        validator = ExtractionValidator(min_confidence=0.5)
        extraction = _make_extraction(
            relationships=[
                _make_relationship(confidence=0.30),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.relationships) == 0
        assert any("Low confidence" in w for w in result.warnings)

    def test_confidence_at_threshold_passes(self):
        # Arrange
        validator = ExtractionValidator(min_confidence=0.5)
        extraction = _make_extraction(
            relationships=[
                _make_relationship(confidence=0.50),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.relationships) == 1


class TestSideEffectBoundary:
    """Invalid items produce warnings but valid items still pass through."""

    def test_invalid_items_do_not_block_valid_items(self):
        # Arrange -- mix of valid and invalid entities/relationships
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="user", entity_type="User"),
                _make_entity(entity_id="python", entity_type="Technology"),
                _make_entity(entity_id="bad", entity_type="FakeType"),
            ],
            relationships=[
                _make_relationship(source="user", target="python", confidence=0.90),
                _make_relationship(source="python", target="python", rel_type="RELATED_TO"),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert -- valid entity and relationship pass; invalid ones produce diagnostics
        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert len(result.errors) > 0 or len(result.warnings) > 0

    def test_validation_result_is_returned_not_raised(self):
        # Arrange -- all invalid: pipeline never halts
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="", entity_type="Technology"),
            ],
            relationships=[
                _make_relationship(rel_type="BOGUS"),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert -- returns a ValidationResult, does not raise
        assert isinstance(result, ValidationResult)
        assert len(result.entities) == 0
        assert len(result.relationships) == 0
        assert len(result.errors) > 0


class TestMistScopePredicates:
    """MIST-scope relationships let extraction attribute facts to MIST itself.

    Regression coverage for Bug J (root cause memo:
    project_mist_bug_j_root_cause.md). Before Cluster 1 the validator
    hard-required {User, Person} as the source of USES/LEARNING/etc., so
    correct emissions like "MIST USES LanceDB" were silently dropped.
    These tests pin the new contract:

      - Extended user-centric predicates accept Organization and MistIdentity
        as sources where the ontology allows (USES, DEPENDS_ON, WORKS_WITH).
      - New MIST-scope predicates (IMPLEMENTED_WITH, MIST_HAS_CAPABILITY,
        MIST_HAS_TRAIT, MIST_HAS_PREFERENCE) route MIST-owned facts to
        MistIdentity.
      - LEARNING, PREFERS, and similar user-only predicates remain
        user-scope only; MIST-scope equivalents MUST go through the
        dedicated MIST_HAS_* edges.

    The `_extend_ontology_extractor_allowlists` fixture keeps these tests
    self-contained against the validator by ensuring MistIdentity and the
    4 MIST-scope predicates are accepted at the entity-type / rel-type gate
    regardless of whether the parallel Cluster 1 allowlist update has landed.
    """

    @pytest.fixture(autouse=True)
    def _extend_ontology_extractor_allowlists(self, monkeypatch):
        # The validator's entity-type and relationship-type gates read from
        # OntologyConstrainedExtractor.ALLOWED_*. We patch those allowlists
        # for the duration of this class so the MIST-scope contract can be
        # exercised independently of parallel ontology-extractor changes.
        extended_entity_types = frozenset(
            OntologyConstrainedExtractor.ALLOWED_ENTITY_TYPES | {"MistIdentity"}
        )
        extended_relationship_types = frozenset(
            OntologyConstrainedExtractor.ALLOWED_RELATIONSHIP_TYPES
            | {
                "IMPLEMENTED_WITH",
                "MIST_HAS_CAPABILITY",
                "MIST_HAS_TRAIT",
                "MIST_HAS_PREFERENCE",
            }
        )
        monkeypatch.setattr(
            OntologyConstrainedExtractor,
            "ALLOWED_ENTITY_TYPES",
            extended_entity_types,
        )
        monkeypatch.setattr(
            OntologyConstrainedExtractor,
            "ALLOWED_RELATIONSHIP_TYPES",
            extended_relationship_types,
        )

    def test_mist_identity_uses_technology_accepted(self):
        # Arrange -- "MIST uses LanceDB" regression scenario
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="mist-identity", entity_type="MistIdentity"),
                _make_entity(entity_id="lancedb", entity_type="Technology"),
            ],
            relationships=[
                _make_relationship(
                    source="mist-identity",
                    target="lancedb",
                    rel_type="USES",
                ),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.relationships) == 1
        assert result.relationships[0]["source"] == "mist-identity"
        assert result.relationships[0]["type"] == "USES"

    def test_organization_uses_technology_accepted(self):
        # Arrange -- "Anthropic uses Python"
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="anthropic", entity_type="Organization"),
                _make_entity(entity_id="python", entity_type="Technology"),
            ],
            relationships=[
                _make_relationship(source="anthropic", target="python", rel_type="USES"),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.relationships) == 1
        assert result.relationships[0]["source"] == "anthropic"

    def test_mist_identity_implemented_with_technology_accepted(self):
        # Arrange -- "MIST is implemented with Python"
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="mist-identity", entity_type="MistIdentity"),
                _make_entity(entity_id="python", entity_type="Technology"),
            ],
            relationships=[
                _make_relationship(
                    source="mist-identity",
                    target="python",
                    rel_type="IMPLEMENTED_WITH",
                ),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.relationships) == 1
        assert result.relationships[0]["type"] == "IMPLEMENTED_WITH"

    def test_mist_identity_has_capability_skill_accepted(self):
        # Arrange -- "MIST has a reasoning capability"
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="mist-identity", entity_type="MistIdentity"),
                _make_entity(entity_id="reasoning", entity_type="Skill"),
            ],
            relationships=[
                _make_relationship(
                    source="mist-identity",
                    target="reasoning",
                    rel_type="MIST_HAS_CAPABILITY",
                ),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.relationships) == 1
        assert result.relationships[0]["type"] == "MIST_HAS_CAPABILITY"

    def test_mist_identity_has_trait_concept_accepted(self):
        # Arrange -- "MIST values transparency"
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="mist-identity", entity_type="MistIdentity"),
                _make_entity(entity_id="transparency", entity_type="Concept"),
            ],
            relationships=[
                _make_relationship(
                    source="mist-identity",
                    target="transparency",
                    rel_type="MIST_HAS_TRAIT",
                ),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.relationships) == 1
        assert result.relationships[0]["type"] == "MIST_HAS_TRAIT"

    def test_mist_identity_has_preference_accepted(self):
        # Arrange -- "MIST prefers plain-text responses"
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="mist-identity", entity_type="MistIdentity"),
                _make_entity(entity_id="plain-text", entity_type="Preference"),
            ],
            relationships=[
                _make_relationship(
                    source="mist-identity",
                    target="plain-text",
                    rel_type="MIST_HAS_PREFERENCE",
                ),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert
        assert len(result.relationships) == 1
        assert result.relationships[0]["type"] == "MIST_HAS_PREFERENCE"

    def test_user_source_on_mist_has_trait_is_rejected(self):
        # Arrange -- MIST_HAS_TRAIT is MistIdentity-only; user as source must fail
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="user", entity_type="User"),
                _make_entity(entity_id="transparency", entity_type="Concept"),
            ],
            relationships=[
                _make_relationship(
                    source="user",
                    target="transparency",
                    rel_type="MIST_HAS_TRAIT",
                ),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert -- dropped with a source type-mismatch warning
        assert len(result.relationships) == 0
        assert any("MIST_HAS_TRAIT" in w and "Source type 'User'" in w for w in result.warnings)

    def test_mist_identity_source_on_learning_is_rejected(self):
        # Arrange -- LEARNING remains user-only; MIST-scope learning must use
        # MIST_HAS_CAPABILITY instead
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="mist-identity", entity_type="MistIdentity"),
                _make_entity(entity_id="rust", entity_type="Technology"),
            ],
            relationships=[
                _make_relationship(
                    source="mist-identity",
                    target="rust",
                    rel_type="LEARNING",
                ),
            ],
        )

        # Act
        result = validator.validate(extraction)

        # Assert -- dropped because MistIdentity is not in LEARNING sources
        assert len(result.relationships) == 0
        assert any("LEARNING" in w and "MistIdentity" in w for w in result.warnings)


# -------------------------------------------------------------------
# Post-MVP additive: temporal / quantified / document constraints (2026-04-22)
# Mirrors the MIST-scope class pattern: happy-path accept + shape-mismatch
# reject for each of the 4 new edge types.
# -------------------------------------------------------------------


class TestTemporalQuantifiedDocumentConstraints:
    """RELATIONSHIP_CONSTRAINTS must mirror the ontology's EdgeTypeDefinition
    entries exactly for the 4 post-MVP additive edges. Accept valid shapes
    and reject shapes whose source or target type is outside the ontology
    allowlist.
    """

    # --- OCCURRED_ON: (Event | Milestone) -> Date ---

    def test_occurred_on_event_to_date_accepted(self):
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="v6-run", entity_type="Event"),
                _make_entity(entity_id="2026-04-22", entity_type="Date"),
            ],
            relationships=[
                _make_relationship(
                    source="v6-run",
                    target="2026-04-22",
                    rel_type="OCCURRED_ON",
                ),
            ],
        )

        result = validator.validate(extraction)

        assert len(result.relationships) == 1
        assert result.relationships[0]["type"] == "OCCURRED_ON"

    def test_occurred_on_user_source_rejected(self):
        # User is not in OCCURRED_ON.allowed_source_types
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="user", entity_type="User"),
                _make_entity(entity_id="2026-04-22", entity_type="Date"),
            ],
            relationships=[
                _make_relationship(
                    source="user",
                    target="2026-04-22",
                    rel_type="OCCURRED_ON",
                ),
            ],
        )

        result = validator.validate(extraction)

        assert len(result.relationships) == 0
        assert any("OCCURRED_ON" in w and "Source type 'User'" in w for w in result.warnings)

    # --- HAS_METRIC: (User | Project | Technology | Skill | Concept | Goal) -> Metric ---

    def test_has_metric_technology_to_metric_accepted(self):
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="gemma-4-e4b", entity_type="Technology"),
                _make_entity(entity_id="tool-selection-score", entity_type="Metric"),
            ],
            relationships=[
                _make_relationship(
                    source="gemma-4-e4b",
                    target="tool-selection-score",
                    rel_type="HAS_METRIC",
                ),
            ],
        )

        result = validator.validate(extraction)

        assert len(result.relationships) == 1
        assert result.relationships[0]["type"] == "HAS_METRIC"

    def test_has_metric_event_source_rejected(self):
        # Event is not in HAS_METRIC.allowed_source_types
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="v6-run", entity_type="Event"),
                _make_entity(entity_id="latency-ms", entity_type="Metric"),
            ],
            relationships=[
                _make_relationship(
                    source="v6-run",
                    target="latency-ms",
                    rel_type="HAS_METRIC",
                ),
            ],
        )

        result = validator.validate(extraction)

        assert len(result.relationships) == 0
        assert any("HAS_METRIC" in w and "Source type 'Event'" in w for w in result.warnings)

    # --- REFERENCES_DOCUMENT: (User | MistIdentity | Project | Concept | Topic
    #                           | Goal | Event) -> Document ---

    def test_references_document_user_to_document_accepted(self):
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="user", entity_type="User"),
                _make_entity(entity_id="adr-010", entity_type="Document"),
            ],
            relationships=[
                _make_relationship(
                    source="user",
                    target="adr-010",
                    rel_type="REFERENCES_DOCUMENT",
                ),
            ],
        )

        result = validator.validate(extraction)

        assert len(result.relationships) == 1
        assert result.relationships[0]["type"] == "REFERENCES_DOCUMENT"

    def test_references_document_person_source_rejected(self):
        # Person is not in REFERENCES_DOCUMENT.allowed_source_types
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="colleague", entity_type="Person"),
                _make_entity(entity_id="paper-x", entity_type="Document"),
            ],
            relationships=[
                _make_relationship(
                    source="colleague",
                    target="paper-x",
                    rel_type="REFERENCES_DOCUMENT",
                ),
            ],
        )

        result = validator.validate(extraction)

        assert len(result.relationships) == 0
        assert any(
            "REFERENCES_DOCUMENT" in w and "Source type 'Person'" in w for w in result.warnings
        )

    # --- PRECEDED_BY: (Event | Milestone) -> (Event | Milestone | Date) ---

    def test_preceded_by_milestone_to_date_accepted(self):
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="mvp-close", entity_type="Milestone"),
                _make_entity(entity_id="2026-04-17", entity_type="Date"),
            ],
            relationships=[
                _make_relationship(
                    source="mvp-close",
                    target="2026-04-17",
                    rel_type="PRECEDED_BY",
                ),
            ],
        )

        result = validator.validate(extraction)

        assert len(result.relationships) == 1
        assert result.relationships[0]["type"] == "PRECEDED_BY"

    def test_preceded_by_project_source_rejected(self):
        # Project is not in PRECEDED_BY.allowed_source_types
        validator = ExtractionValidator()
        extraction = _make_extraction(
            entities=[
                _make_entity(entity_id="mist-ai", entity_type="Project"),
                _make_entity(entity_id="v6-run", entity_type="Event"),
            ],
            relationships=[
                _make_relationship(
                    source="mist-ai",
                    target="v6-run",
                    rel_type="PRECEDED_BY",
                ),
            ],
        )

        result = validator.validate(extraction)

        assert len(result.relationships) == 0
        assert any("PRECEDED_BY" in w and "Source type 'Project'" in w for w in result.warnings)


# -------------------------------------------------------------------
# Standing drift guard (2026-04-22): RELATIONSHIP_CONSTRAINTS MUST
# mirror EdgeTypeDefinition in the ontology exactly. Adding a new edge
# type to the ontology without updating RELATIONSHIP_CONSTRAINTS is the
# historical #1 bug class; this test turns the manual cross-check into
# standing coverage.
# -------------------------------------------------------------------


class TestValidatorOntologyConsistency:
    """The validator's RELATIONSHIP_CONSTRAINTS dict and the ontology's
    EdgeTypeDefinition objects must agree on every extractable edge's
    allowed source and target types. Drift between them silently
    changes validation behavior and is genuinely hard to catch in
    per-edge tests.
    """

    def test_every_extractable_edge_has_a_validator_constraint(self):
        from backend.knowledge.extraction.validator import RELATIONSHIP_CONSTRAINTS
        from backend.knowledge.ontologies.v1_0_0 import (
            EXTRACTABLE_RELATIONSHIP_TYPES,
        )

        missing = [
            name for name in EXTRACTABLE_RELATIONSHIP_TYPES if name not in RELATIONSHIP_CONSTRAINTS
        ]
        assert missing == [], (
            f"Edges in EXTRACTABLE_RELATIONSHIP_TYPES but missing from "
            f"validator RELATIONSHIP_CONSTRAINTS: {missing!r}. Update "
            f"validator.py when adding a new edge to the ontology."
        )

    def test_validator_constraints_mirror_edge_type_definitions(self):
        """For each edge that has BOTH a validator entry AND an ontology
        EdgeTypeDefinition, the allowed_source_types / allowed_target_types
        must match member-for-member.

        The validator stores constraints as `tuple[set | None, set | None]`
        where None means "any type permitted". The ontology stores
        `allowed_source_types` / `allowed_target_types` as tuples. This
        test normalizes to sets for comparison; `None` validator entries
        skip the comparison (permissive by design).
        """
        from backend.knowledge.extraction.validator import RELATIONSHIP_CONSTRAINTS
        from backend.knowledge.ontologies.v1_0_0 import ALL_EDGE_TYPES

        edges_by_name = {e.type_name: e for e in ALL_EDGE_TYPES}
        drift: list[str] = []

        for edge_name, (val_src, val_tgt) in RELATIONSHIP_CONSTRAINTS.items():
            onto_edge = edges_by_name.get(edge_name)
            if onto_edge is None:
                drift.append(f"{edge_name}: in validator but not in ontology " f"ALL_EDGE_TYPES")
                continue
            onto_src = set(onto_edge.allowed_source_types)
            onto_tgt = set(onto_edge.allowed_target_types)
            if val_src is not None and val_src != onto_src:
                drift.append(
                    f"{edge_name} sources: ontology={sorted(onto_src)!r} "
                    f"validator={sorted(val_src)!r}"
                )
            if val_tgt is not None and val_tgt != onto_tgt:
                drift.append(
                    f"{edge_name} targets: ontology={sorted(onto_tgt)!r} "
                    f"validator={sorted(val_tgt)!r}"
                )

        assert drift == [], "Validator-ontology drift:\n  " + "\n  ".join(drift)
