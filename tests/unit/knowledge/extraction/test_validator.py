"""Tests for ExtractionValidator extraction pipeline stage."""

import pytest

from backend.knowledge.extraction.ontology_extractor import ExtractionResult
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
