"""MIST.AI Knowledge Ontology package.

Re-exports all public enums, base dataclasses, and the v1.0.0 ontology
definition so callers can import from ``backend.knowledge.ontologies`` directly.
"""

from backend.knowledge.ontologies.base import (
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
    TemporalStatus,
)
from backend.knowledge.ontologies.v1_0_0 import (
    ALL_EDGE_TYPE_NAMES,
    ALL_EDGE_TYPES,
    ALL_NODE_TYPE_NAMES,
    ALL_NODE_TYPES,
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
    "EXTRACTABLE_NODE_TYPES",
    "EXTRACTABLE_RELATIONSHIP_TYPES",
]
