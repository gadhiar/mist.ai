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
