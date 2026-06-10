"""MIST.AI Knowledge Ontology package.

Re-exports all public enums, base dataclasses, and the v1.0.0 ontology
definition so callers can import from ``backend.knowledge.ontologies`` directly.
"""

from backend.knowledge.ontologies.base import (
    Cardinality,
    ConfidencePolicy,
    EdgeTypeDefinition,
    EntityStatus,
    ExtractionRules,
    KnowledgeDomain,
    NodeTypeDefinition,
    OntologyVersion,
    PropertyDefinition,
    RelationshipStatus,
    SourceType,
    TemporalClass,
    TemporalStatus,
)
from backend.knowledge.ontologies.v1_0_0 import (
    ALL_EDGE_TYPE_NAMES,
    ALL_EDGE_TYPES,
    ALL_NODE_TYPE_NAMES,
    ALL_NODE_TYPES,
    EDGE_TYPES_BY_NAME,
    EXTRACTABLE_NODE_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
    ONTOLOGY_V1_0_0,
)

__all__ = [
    # Enums
    "KnowledgeDomain",
    "SourceType",
    "EntityStatus",
    "RelationshipStatus",
    "TemporalStatus",
    "Cardinality",
    "TemporalClass",
    # Base dataclasses
    "PropertyDefinition",
    "NodeTypeDefinition",
    "EdgeTypeDefinition",
    "ExtractionRules",
    "ConfidencePolicy",
    "OntologyVersion",
    # v1.0.0 ontology
    "ONTOLOGY_V1_0_0",
    # Helper constants
    "ALL_NODE_TYPES",
    "ALL_NODE_TYPE_NAMES",
    "ALL_EDGE_TYPES",
    "ALL_EDGE_TYPE_NAMES",
    "EDGE_TYPES_BY_NAME",
    "EXTRACTABLE_NODE_TYPES",
    "EXTRACTABLE_RELATIONSHIP_TYPES",
]
