"""Extraction validation stage.

Stage 6: Validates extracted entities and relationships against the
ontology schema and constraint rules. Invalid items are dropped with
logged warnings. The pipeline never halts on validation errors.
Target <5ms.
"""

import logging
from dataclasses import dataclass, field

from backend.knowledge.extraction.ontology_extractor import (
    ExtractionResult,
    OntologyConstrainedExtractor,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of extraction validation.

    Contains the filtered valid entities/relationships and any
    warnings or errors encountered during validation.
    """

    valid: bool
    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# Maps relationship type -> (allowed_source_types, allowed_target_types).
# None means any type is allowed in that position.
# IMPORTANT: These MUST match the EdgeTypeDefinition constraints in
# backend.knowledge.ontologies.v1_0_0. If you update the ontology,
# update this table too (or generate it programmatically in the future).
RELATIONSHIP_CONSTRAINTS: dict[str, tuple[set[str] | None, set[str] | None]] = {
    "USES": ({"User", "Person"}, {"Technology"}),
    "KNOWS": ({"User", "Person"}, {"Skill", "Concept", "Technology", "Topic"}),
    "WORKS_ON": ({"User", "Person"}, {"Project"}),
    "WORKS_AT": ({"User", "Person"}, {"Organization"}),
    "INTERESTED_IN": ({"User"}, {"Technology", "Concept", "Topic", "Skill"}),
    "HAS_GOAL": ({"User"}, {"Goal"}),
    "PREFERS": ({"User"}, {"Preference", "Technology", "Concept"}),
    "DISLIKES": ({"User"}, {"Technology", "Concept", "Topic", "Preference"}),
    "EXPERT_IN": ({"User", "Person"}, {"Technology", "Skill", "Concept"}),
    "LEARNING": ({"User"}, {"Technology", "Skill", "Concept"}),
    "STRUGGLES_WITH": ({"User"}, {"Technology", "Skill", "Concept"}),
    "DECIDED": ({"User"}, {"Event"}),
    "EXPERIENCED": ({"User", "Person"}, {"Event"}),
    "IS_A": (
        {"Technology", "Skill", "Concept", "Topic", "Organization", "Location"},
        {"Technology", "Skill", "Concept", "Topic", "Organization", "Location"},
    ),
    "PART_OF": (
        {"Technology", "Concept", "Skill", "Project"},
        {"Technology", "Concept", "Project", "Organization"},
    ),
    "RELATED_TO": (None, None),  # Generic: any -> any
    "DEPENDS_ON": ({"Technology", "Project"}, {"Technology"}),
    "USED_FOR": ({"Technology", "Skill"}, {"Concept", "Topic", "Project"}),
    "WORKS_WITH": ({"Technology"}, {"Technology"}),
    "KNOWS_PERSON": ({"User"}, {"Person"}),
    "MEMBER_OF": ({"User", "Person"}, {"Organization"}),
}

VALID_TEMPORAL_STATUSES: set[str] = {"current", "past", "future", "recurring"}


class ExtractionValidator:
    """Validates extraction results against ontology constraints.

    Checks:
    - Entity type is in the allowed set.
    - Entity ID is non-empty.
    - Relationship type is in the allowed set.
    - Relationship source and target entities exist in the entity list.
    - No self-referencing relationships.
    - Source/target entity types satisfy relationship constraints.
    - Confidence >= minimum threshold (default 0.5).
    - temporal_status is a valid value if present.

    Invalid items are dropped and logged. The pipeline never halts.
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        """Initialize the validator.

        Args:
            min_confidence: Minimum confidence threshold for relationships.
                Relationships below this threshold are dropped.
        """
        self.min_confidence = min_confidence

    def validate(self, extraction: ExtractionResult) -> ValidationResult:
        """Validate an extraction result.

        Filters out invalid entities and relationships, collecting
        warnings and errors for diagnostics.

        Args:
            extraction: The ExtractionResult to validate.

        Returns:
            ValidationResult with valid items and diagnostic messages.
        """
        warnings: list[str] = []
        errors: list[str] = []

        # --- Validate entities ---
        valid_entities: list[dict] = []
        valid_entity_ids: set[str] = set()
        entity_type_map: dict[str, str] = {}

        for entity in extraction.entities:
            entity_id = entity.get("id", "")
            entity_type = entity.get("type", "")

            # Check non-empty ID
            if not entity_id or not entity_id.strip():
                errors.append(f"Entity with empty ID dropped: {entity}")
                continue

            # Check entity type is allowed
            if entity_type not in OntologyConstrainedExtractor.ALLOWED_ENTITY_TYPES:
                errors.append(f"Entity '{entity_id}' has invalid type '{entity_type}', dropped")
                continue

            valid_entities.append(entity)
            valid_entity_ids.add(entity_id)
            entity_type_map[entity_id] = entity_type

        # --- Validate relationships ---
        valid_relationships: list[dict] = []

        for rel in extraction.relationships:
            rel_type = rel.get("type", "")
            source = rel.get("source", "")
            target = rel.get("target", "")
            props = rel.get("properties", {}) or {}

            # Check relationship type is allowed
            if rel_type not in OntologyConstrainedExtractor.ALLOWED_RELATIONSHIP_TYPES:
                errors.append(
                    f"Relationship type '{rel_type}' not in ontology, dropped: "
                    f"{source} -[{rel_type}]-> {target}"
                )
                continue

            # Check source entity exists
            if source not in valid_entity_ids:
                errors.append(
                    f"Relationship source '{source}' not in entity list, dropped: "
                    f"{source} -[{rel_type}]-> {target}"
                )
                continue

            # Check target entity exists
            if target not in valid_entity_ids:
                errors.append(
                    f"Relationship target '{target}' not in entity list, dropped: "
                    f"{source} -[{rel_type}]-> {target}"
                )
                continue

            # Check no self-reference
            if source == target:
                warnings.append(
                    f"Self-referencing relationship dropped: " f"{source} -[{rel_type}]-> {target}"
                )
                continue

            # Check source/target type constraints
            constraints = RELATIONSHIP_CONSTRAINTS.get(rel_type)
            if constraints is not None:
                allowed_sources, allowed_targets = constraints

                source_type = entity_type_map.get(source, "")
                target_type = entity_type_map.get(target, "")

                if allowed_sources is not None and source_type not in allowed_sources:
                    warnings.append(
                        f"Source type '{source_type}' not valid for {rel_type} "
                        f"(expected {allowed_sources}), dropped: "
                        f"{source} -[{rel_type}]-> {target}"
                    )
                    continue

                if allowed_targets is not None and target_type not in allowed_targets:
                    warnings.append(
                        f"Target type '{target_type}' not valid for {rel_type} "
                        f"(expected {allowed_targets}), dropped: "
                        f"{source} -[{rel_type}]-> {target}"
                    )
                    continue

            # Check confidence threshold
            confidence = props.get("confidence", 0.9)
            if not isinstance(confidence, int | float):
                try:
                    confidence = float(confidence)
                except (ValueError, TypeError):
                    confidence = 0.0

            if confidence < self.min_confidence:
                warnings.append(
                    f"Low confidence ({confidence:.2f} < {self.min_confidence}), dropped: "
                    f"{source} -[{rel_type}]-> {target}"
                )
                continue

            # Check temporal_status if present
            temporal_status = props.get("temporal_status")
            if temporal_status is not None and temporal_status not in VALID_TEMPORAL_STATUSES:
                warnings.append(
                    f"Invalid temporal_status '{temporal_status}' on "
                    f"{source} -[{rel_type}]-> {target}, cleared to null"
                )
                props["temporal_status"] = None

            valid_relationships.append(rel)

        # Log summary
        dropped_entities = len(extraction.entities) - len(valid_entities)
        dropped_rels = len(extraction.relationships) - len(valid_relationships)

        if dropped_entities > 0 or dropped_rels > 0:
            logger.info(
                "Validation: dropped %d entities and %d relationships",
                dropped_entities,
                dropped_rels,
            )

        if errors:
            for error in errors:
                logger.debug("Validation error: %s", error)

        if warnings:
            for warning in warnings:
                logger.debug("Validation warning: %s", warning)

        return ValidationResult(
            valid=len(errors) == 0 and len(warnings) == 0,
            entities=valid_entities,
            relationships=valid_relationships,
            warnings=warnings,
            errors=errors,
        )
