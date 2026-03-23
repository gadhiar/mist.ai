"""Shared fixtures for curation pipeline tests."""

from backend.knowledge.extraction.validator import ValidationResult


def make_entity_dict(
    *,
    entity_id: str = "python",
    entity_type: str = "Technology",
    display_name: str = "Python",
    confidence: float = 0.85,
    source_type: str = "extracted",
    aliases: list[str] | None = None,
    description: str = "",
    embedding: list[float] | None = None,
) -> dict:
    """Build an entity dict matching extraction pipeline output format."""
    return {
        "id": entity_id,
        "type": entity_type,
        "name": display_name,
        "confidence": confidence,
        "source_type": source_type,
        "aliases": aliases or [],
        "description": description,
        "embedding": embedding,
    }


def make_relationship_dict(
    *,
    source: str = "user",
    target: str = "python",
    rel_type: str = "USES",
    confidence: float = 0.85,
    source_type: str = "extracted",
    temporal_status: str = "current",
    context: str = "",
    properties: dict | None = None,
) -> dict:
    """Build a relationship dict matching extraction pipeline output format."""
    result = {
        "source": source,
        "target": target,
        "type": rel_type,
        "confidence": confidence,
        "source_type": source_type,
        "temporal_status": temporal_status,
        "context": context,
    }
    if properties:
        result["properties"] = properties
    return result


def make_validation_result(
    *,
    entities: list[dict] | None = None,
    relationships: list[dict] | None = None,
    valid: bool = True,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> ValidationResult:
    """Build a ValidationResult wrapping entities and relationships."""
    return ValidationResult(
        valid=valid,
        entities=entities or [],
        relationships=relationships or [],
        warnings=warnings or [],
        errors=errors or [],
    )
