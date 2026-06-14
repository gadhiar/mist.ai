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


class Cardinality(StrEnum):
    """How many concurrent current targets an edge type permits per source.

    SINGLE: at most one valid-time-open edge per (source, type) across all
    targets -- a new open assertion supersedes the prior one (e.g. WORKS_AT).
    MULTI: targets accumulate independently.
    """

    SINGLE = "single"
    MULTI = "multi"


class TemporalClass(StrEnum):
    """How an edge type participates in valid-time reconciliation.

    STATIVE: a state over an interval -- reconciled with valid-time.
    EVENT: a point occurrence -- accumulates, never superseded by time.
    DURABLE: timeless structural fact -- no valid-time reconciliation.
    """

    STATIVE = "stative"
    EVENT = "event"
    DURABLE = "durable"


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
    parent_type: str | None = None


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
    # C1 reconciliation semantics (design 4.1). Declarative -- the
    # reconciliation engine reads these; it contains no predicate names
    # (Inv-A6). Defaults preserve pre-C1 behavior: accumulate-only.
    # Undirected edges use the existing `directional=False` flag; there is
    # deliberately no separate `symmetric` field. For directional=False the
    # (allowed_source_types, allowed_target_types) sets describe an UNORDERED
    # pair: validation accepts either orientation and the write path
    # canonicalizes lexically, so the stored direction carries no meaning.
    cardinality: Cardinality = Cardinality.MULTI
    temporal_class: TemporalClass = TemporalClass.STATIVE
    contradicts: tuple[str, ...] = ()
    progression_supersedes: tuple[str, ...] = ()


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
