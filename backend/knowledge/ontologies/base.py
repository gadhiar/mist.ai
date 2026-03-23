"""Base types and enums for the MIST.AI knowledge ontology system.

Defines the foundational data structures used to describe node types,
edge types, extraction rules, confidence policies, and ontology versions.
All dataclasses are frozen (immutable) to prevent accidental mutation of
ontology definitions at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class KnowledgeDomain(StrEnum):
    """Broad domain category for knowledge entities and relationships."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    BRIDGING = "bridging"


class SourceType(StrEnum):
    """How a piece of knowledge was acquired."""

    EXTRACTED = "extracted"
    STATED = "stated"
    INFERRED = "inferred"
    SELF_AUTHORED = "self_authored"
    CORRECTED = "corrected"


class EntityStatus(StrEnum):
    """Lifecycle status for an entity node."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    MERGED = "merged"
    ARCHIVED = "archived"


class RelationshipStatus(StrEnum):
    """Lifecycle status for a relationship edge."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class TemporalStatus(StrEnum):
    """Temporal qualifier for relationships."""

    CURRENT = "current"
    PAST = "past"
    FUTURE = "future"
    RECURRING = "recurring"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropertyDefinition:
    """Schema definition for a single property on a node or edge type."""

    name: str
    type: str
    required: bool = False
    description: str = ""
    allowed_values: tuple[str, ...] | None = None
    default: Any | None = None


@dataclass(frozen=True)
class NodeTypeDefinition:
    """Schema definition for a node (entity) type in the ontology."""

    type_name: str
    description: str
    knowledge_domain: KnowledgeDomain
    required_properties: tuple[PropertyDefinition, ...] = ()
    optional_properties: tuple[PropertyDefinition, ...] = ()
    singleton: bool = False


@dataclass(frozen=True)
class EdgeTypeDefinition:
    """Schema definition for an edge (relationship) type in the ontology."""

    type_name: str
    description: str
    allowed_source_types: tuple[str, ...]
    allowed_target_types: tuple[str, ...]
    directional: bool = True
    required_properties: tuple[PropertyDefinition, ...] = ()
    optional_properties: tuple[PropertyDefinition, ...] = ()


@dataclass(frozen=True)
class ExtractionRules:
    """Configuration governing how the LLM extractor operates."""

    system_prompt: str
    allowed_node_types: tuple[str, ...]
    allowed_relationship_types: tuple[str, ...]
    property_extraction_enabled: bool = True
    max_entities_per_utterance: int = 10
    max_relationships_per_utterance: int = 15
    min_confidence_threshold: float = 0.4


@dataclass(frozen=True)
class ConfidencePolicy:
    """Policy controlling confidence scoring and decay for a knowledge domain."""

    domain: KnowledgeDomain
    default_confidence: float
    decay_enabled: bool
    decay_half_life_days: int | None = None
    correction_boost: float = 0.0
    third_party_penalty: float = 0.0


@dataclass(frozen=True)
class OntologyVersion:
    """Complete, versioned snapshot of the knowledge ontology."""

    version: str
    created_at: datetime
    description: str
    node_types: tuple[NodeTypeDefinition, ...]
    edge_types: tuple[EdgeTypeDefinition, ...]
    extraction_rules: ExtractionRules
    confidence_policies: tuple[ConfidencePolicy, ...]
    universal_entity_properties: tuple[PropertyDefinition, ...]
    universal_relationship_properties: tuple[PropertyDefinition, ...]
    parent_version: str | None = None
    migration_script_path: str | None = None
    active: bool = True
    deprecated: bool = False
    deprecated_reason: str | None = None
