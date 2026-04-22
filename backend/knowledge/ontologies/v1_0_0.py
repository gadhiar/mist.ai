"""MIST.AI Knowledge Ontology v1.0.0.

Defines every node type, edge type, confidence policy, extraction rule, and
universal property for the first stable release of the knowledge graph schema.

Exports
-------
ONTOLOGY_V1_0_0 : OntologyVersion
    The complete, frozen ontology definition.
ALL_NODE_TYPES / ALL_NODE_TYPE_NAMES
ALL_EDGE_TYPES / ALL_EDGE_TYPE_NAMES
EXTRACTABLE_NODE_TYPES / EXTRACTABLE_RELATIONSHIP_TYPES
"""

from __future__ import annotations

from datetime import UTC, datetime

from backend.knowledge.ontologies.base import (
    ConfidencePolicy,
    EdgeTypeDefinition,
    ExtractionRules,
    KnowledgeDomain,
    NodeTypeDefinition,
    OntologyVersion,
    PropertyDefinition,
)

# ===================================================================
# Confidence Policies
# ===================================================================

CONFIDENCE_INTERNAL = ConfidencePolicy(
    domain=KnowledgeDomain.INTERNAL,
    default_confidence=0.9,
    decay_enabled=False,
    decay_half_life_days=None,
    correction_boost=0.05,
    third_party_penalty=0.0,
)

CONFIDENCE_EXTERNAL = ConfidencePolicy(
    domain=KnowledgeDomain.EXTERNAL,
    default_confidence=0.8,
    decay_enabled=True,
    decay_half_life_days=180,
    correction_boost=0.10,
    third_party_penalty=0.15,
)

CONFIDENCE_BRIDGING = ConfidencePolicy(
    domain=KnowledgeDomain.BRIDGING,
    default_confidence=0.85,
    decay_enabled=True,
    decay_half_life_days=365,
    correction_boost=0.10,
    third_party_penalty=0.10,
)

# ===================================================================
# Universal Entity Properties (13)
# ===================================================================

UNIVERSAL_ENTITY_PROPERTIES: tuple[PropertyDefinition, ...] = (
    PropertyDefinition(
        name="id",
        type="string",
        required=True,
        description="Globally unique identifier for the entity.",
    ),
    PropertyDefinition(
        name="entity_type",
        type="string",
        required=True,
        description="Ontology node type name.",
    ),
    PropertyDefinition(
        name="display_name",
        type="string",
        required=True,
        description="Human-readable display name.",
    ),
    PropertyDefinition(
        name="knowledge_domain",
        type="string",
        required=True,
        description="Knowledge domain this entity belongs to.",
        allowed_values=("internal", "external", "bridging"),
    ),
    PropertyDefinition(
        name="confidence",
        type="float",
        required=True,
        description="Confidence score in the range [0.0, 1.0].",
    ),
    PropertyDefinition(
        name="source_type",
        type="string",
        required=True,
        description="How this entity was acquired.",
        allowed_values=("extracted", "stated", "inferred", "self_authored", "corrected"),
    ),
    PropertyDefinition(
        name="created_at",
        type="datetime",
        required=True,
        description="Timestamp when the entity was first created.",
    ),
    PropertyDefinition(
        name="updated_at",
        type="datetime",
        required=True,
        description="Timestamp of the most recent update.",
    ),
    PropertyDefinition(
        name="ontology_version",
        type="string",
        required=True,
        description="Ontology version under which this entity was created.",
    ),
    PropertyDefinition(
        name="embedding",
        type="float[]",
        required=False,
        description="Vector embedding for semantic search.",
    ),
    PropertyDefinition(
        name="description",
        type="string",
        required=False,
        description="Free-text description of the entity.",
    ),
    PropertyDefinition(
        name="aliases",
        type="string[]",
        required=False,
        description="Alternative names or spellings for the entity.",
    ),
    PropertyDefinition(
        name="status",
        type="string",
        required=False,
        description="Lifecycle status of the entity.",
        allowed_values=("active", "deprecated", "merged", "archived"),
        default="active",
    ),
)

# ===================================================================
# Universal Relationship Properties (15)
# ===================================================================

UNIVERSAL_RELATIONSHIP_PROPERTIES: tuple[PropertyDefinition, ...] = (
    PropertyDefinition(
        name="confidence",
        type="float",
        required=True,
        description="Confidence score for this relationship.",
    ),
    PropertyDefinition(
        name="source_event_id",
        type="string",
        required=True,
        description="ID of the conversation event that produced this relationship.",
    ),
    PropertyDefinition(
        name="source_type",
        type="string",
        required=True,
        description="How this relationship was acquired.",
        allowed_values=("extracted", "stated", "inferred", "self_authored", "corrected"),
    ),
    PropertyDefinition(
        name="created_at",
        type="datetime",
        required=True,
        description="Timestamp when the relationship was first created.",
    ),
    PropertyDefinition(
        name="updated_at",
        type="datetime",
        required=True,
        description="Timestamp of the most recent update.",
    ),
    PropertyDefinition(
        name="ontology_version",
        type="string",
        required=True,
        description="Ontology version under which this relationship was created.",
    ),
    PropertyDefinition(
        name="status",
        type="string",
        required=False,
        description="Lifecycle status of the relationship.",
        allowed_values=("active", "superseded", "retracted"),
    ),
    PropertyDefinition(
        name="superseded_by",
        type="string",
        required=False,
        description="ID of the relationship that superseded this one.",
    ),
    PropertyDefinition(
        name="evidence",
        type="string",
        required=False,
        description="Supporting evidence or source quote.",
    ),
    PropertyDefinition(
        name="context",
        type="string",
        required=False,
        description="Contextual note about the relationship.",
    ),
    PropertyDefinition(
        name="started",
        type="datetime",
        required=False,
        description="When the relationship began.",
    ),
    PropertyDefinition(
        name="ended",
        type="datetime",
        required=False,
        description="When the relationship ended.",
    ),
    PropertyDefinition(
        name="duration",
        type="string",
        required=False,
        description="Human-readable duration of the relationship.",
    ),
    PropertyDefinition(
        name="frequency",
        type="string",
        required=False,
        description="How often this relationship is exercised (e.g. daily, weekly).",
    ),
    PropertyDefinition(
        name="temporal_status",
        type="string",
        required=False,
        description="Temporal qualifier for the relationship.",
        allowed_values=("current", "past", "future", "recurring"),
    ),
)

# ===================================================================
# INTERNAL Entity Types (5)
# ===================================================================

MIST_IDENTITY = NodeTypeDefinition(
    type_name="MistIdentity",
    description="Singleton node representing MIST's self-model and identity.",
    knowledge_domain=KnowledgeDomain.INTERNAL,
    singleton=True,
    required_properties=(
        PropertyDefinition(
            name="personality_summary",
            type="string",
            required=True,
            description="High-level summary of MIST's personality.",
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="communication_style",
            type="string",
            required=False,
            description="Description of MIST's communication style.",
        ),
        PropertyDefinition(
            name="formality_level",
            type="string",
            required=False,
            description="Preferred level of formality.",
        ),
        PropertyDefinition(
            name="humor_style",
            type="string",
            required=False,
            description="Description of MIST's sense of humor.",
        ),
        PropertyDefinition(
            name="core_values",
            type="string[]",
            required=False,
            description="List of core values guiding MIST's behavior.",
        ),
    ),
)

MIST_TRAIT = NodeTypeDefinition(
    type_name="MistTrait",
    description="A personality, communication, behavioral, or ethical trait of MIST.",
    knowledge_domain=KnowledgeDomain.INTERNAL,
    required_properties=(
        PropertyDefinition(
            name="trait_category",
            type="string",
            required=True,
            description="Category of the trait.",
            allowed_values=("personality", "communication", "behavioral", "ethical"),
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="strength",
            type="float",
            required=False,
            description="How strongly this trait is expressed, in [0.0, 1.0].",
        ),
        PropertyDefinition(
            name="origin",
            type="string",
            required=False,
            description="How or why this trait was adopted.",
        ),
    ),
)

MIST_CAPABILITY = NodeTypeDefinition(
    type_name="MistCapability",
    description="A capability or skill that MIST has developed.",
    knowledge_domain=KnowledgeDomain.INTERNAL,
    required_properties=(
        PropertyDefinition(
            name="capability_type",
            type="string",
            required=True,
            description="Type of capability.",
            allowed_values=("technical_domain", "interaction_pattern", "knowledge_area"),
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="proficiency",
            type="float",
            required=False,
            description="Self-assessed proficiency level in [0.0, 1.0].",
        ),
        PropertyDefinition(
            name="learned_from",
            type="string",
            required=False,
            description="Source or event from which this capability was acquired.",
        ),
        PropertyDefinition(
            name="example_count",
            type="int",
            required=False,
            description="Number of examples encountered for this capability.",
        ),
    ),
)

MIST_PREFERENCE = NodeTypeDefinition(
    type_name="MistPreference",
    description="A preference MIST has developed about how to handle interactions.",
    knowledge_domain=KnowledgeDomain.INTERNAL,
    required_properties=(
        PropertyDefinition(
            name="preference_type",
            type="string",
            required=True,
            description="Type of preference.",
            allowed_values=("response_style", "topic_handling", "uncertainty_handling"),
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="strength",
            type="float",
            required=False,
            description="Strength of the preference in [0.0, 1.0].",
        ),
        PropertyDefinition(
            name="reason",
            type="string",
            required=False,
            description="Why this preference was adopted.",
        ),
    ),
)

MIST_UNCERTAINTY = NodeTypeDefinition(
    type_name="MistUncertainty",
    description="Something MIST is uncertain about and may need to verify.",
    knowledge_domain=KnowledgeDomain.INTERNAL,
    required_properties=(
        PropertyDefinition(
            name="uncertainty_type",
            type="string",
            required=True,
            description="Type of uncertainty.",
            allowed_values=("factual", "preference", "capability", "recency"),
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="related_entity_id",
            type="string",
            required=False,
            description="ID of the entity this uncertainty pertains to.",
        ),
        PropertyDefinition(
            name="last_checked",
            type="datetime",
            required=False,
            description="When this uncertainty was last checked or revisited.",
        ),
        PropertyDefinition(
            name="resolution_strategy",
            type="string",
            required=False,
            description="Strategy for resolving this uncertainty.",
        ),
    ),
)

# ===================================================================
# EXTERNAL Entity Types (16)
# ===================================================================

USER = NodeTypeDefinition(
    type_name="User",
    description="Singleton node representing the primary user.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    singleton=True,
    optional_properties=(
        PropertyDefinition(
            name="first_name",
            type="string",
            required=False,
            description="User's first name.",
        ),
        PropertyDefinition(
            name="last_name",
            type="string",
            required=False,
            description="User's last name.",
        ),
        PropertyDefinition(
            name="role",
            type="string",
            required=False,
            description="User's professional role or title.",
        ),
        PropertyDefinition(
            name="current_employer",
            type="string",
            required=False,
            description="User's current employer or organization.",
        ),
        PropertyDefinition(
            name="location",
            type="string",
            required=False,
            description="User's primary location.",
        ),
        PropertyDefinition(
            name="communication_style",
            type="string",
            required=False,
            description="User's preferred communication style.",
        ),
    ),
)

PERSON = NodeTypeDefinition(
    type_name="Person",
    description="A person other than the primary user.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    optional_properties=(
        PropertyDefinition(
            name="role",
            type="string",
            required=False,
            description="This person's professional role or title.",
        ),
        PropertyDefinition(
            name="organization",
            type="string",
            required=False,
            description="Organization this person is affiliated with.",
        ),
        PropertyDefinition(
            name="relationship_to_user",
            type="string",
            required=False,
            description="How this person relates to the user (colleague, friend, etc.).",
        ),
    ),
)

ORGANIZATION = NodeTypeDefinition(
    type_name="Organization",
    description="A company, team, institution, community, or open-source project.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    optional_properties=(
        PropertyDefinition(
            name="org_type",
            type="string",
            required=False,
            description="Type of organization.",
            allowed_values=("company", "team", "institution", "community", "open_source"),
        ),
        PropertyDefinition(
            name="url",
            type="string",
            required=False,
            description="URL for the organization's website or page.",
        ),
    ),
)

TECHNOLOGY = NodeTypeDefinition(
    type_name="Technology",
    description="A programming language, framework, library, tool, platform, service, database, or protocol.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    optional_properties=(
        PropertyDefinition(
            name="tech_category",
            type="string",
            required=False,
            description="Category of the technology.",
            allowed_values=(
                "language",
                "framework",
                "library",
                "tool",
                "platform",
                "service",
                "database",
                "protocol",
            ),
        ),
        PropertyDefinition(
            name="url",
            type="string",
            required=False,
            description="URL for official documentation or homepage.",
        ),
        PropertyDefinition(
            name="version",
            type="string",
            required=False,
            description="Relevant version number.",
        ),
    ),
)

SKILL = NodeTypeDefinition(
    type_name="Skill",
    description="A skill or competency (technical, soft, domain, or methodological).",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    optional_properties=(
        PropertyDefinition(
            name="skill_category",
            type="string",
            required=False,
            description="Category of the skill.",
            allowed_values=("technical", "soft", "domain", "methodological"),
        ),
    ),
)

PROJECT = NodeTypeDefinition(
    type_name="Project",
    description="A work, personal, open-source, or research project.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    optional_properties=(
        PropertyDefinition(
            name="project_type",
            type="string",
            required=False,
            description="Type of project.",
            allowed_values=("work", "personal", "open_source", "research"),
        ),
        PropertyDefinition(
            name="project_status",
            type="string",
            required=False,
            description="Current status of the project.",
            allowed_values=("active", "completed", "planned", "paused", "abandoned"),
        ),
        PropertyDefinition(
            name="url",
            type="string",
            required=False,
            description="URL for the project repository or page.",
        ),
        PropertyDefinition(
            name="started",
            type="datetime",
            required=False,
            description="When the project started.",
        ),
        PropertyDefinition(
            name="ended",
            type="datetime",
            required=False,
            description="When the project ended.",
        ),
    ),
)

CONCEPT = NodeTypeDefinition(
    type_name="Concept",
    description="An abstract concept such as a methodology, pattern, principle, domain, or theory.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    optional_properties=(
        PropertyDefinition(
            name="concept_category",
            type="string",
            required=False,
            description="Category of the concept.",
            allowed_values=("methodology", "pattern", "principle", "domain", "theory"),
        ),
    ),
)

TOPIC = NodeTypeDefinition(
    type_name="Topic",
    description="A general topic of discussion or interest.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
)

EVENT = NodeTypeDefinition(
    type_name="Event",
    description="A notable event such as a decision, milestone, meeting, or life event.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    optional_properties=(
        PropertyDefinition(
            name="event_type",
            type="string",
            required=False,
            description="Type of event.",
            allowed_values=(
                "decision",
                "milestone",
                "meeting",
                "life_event",
                "deadline",
                "achievement",
            ),
        ),
        PropertyDefinition(
            name="date",
            type="datetime",
            required=False,
            description="When the event occurred or is scheduled.",
        ),
        PropertyDefinition(
            name="outcome",
            type="string",
            required=False,
            description="Outcome or result of the event.",
        ),
    ),
)

GOAL = NodeTypeDefinition(
    type_name="Goal",
    description="A goal the user is pursuing.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    optional_properties=(
        PropertyDefinition(
            name="goal_type",
            type="string",
            required=False,
            description="Type of goal.",
            allowed_values=("career", "learning", "project", "personal", "financial"),
        ),
        PropertyDefinition(
            name="goal_status",
            type="string",
            required=False,
            description="Current status of the goal.",
            allowed_values=("active", "achieved", "abandoned", "deferred"),
        ),
        PropertyDefinition(
            name="target_date",
            type="datetime",
            required=False,
            description="Target completion date.",
        ),
    ),
)

PREFERENCE = NodeTypeDefinition(
    type_name="Preference",
    description="A user preference for tools, workflows, aesthetics, communication, food, or other areas.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    optional_properties=(
        PropertyDefinition(
            name="preference_category",
            type="string",
            required=False,
            description="Category of preference.",
            allowed_values=("tool", "workflow", "aesthetic", "communication", "food", "general"),
        ),
        PropertyDefinition(
            name="strength",
            type="string",
            required=False,
            description="How strongly the preference is held.",
            allowed_values=("strong", "moderate", "mild"),
        ),
    ),
)

LOCATION = NodeTypeDefinition(
    type_name="Location",
    description="A geographic or logical location.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    optional_properties=(
        PropertyDefinition(
            name="location_type",
            type="string",
            required=False,
            description="Type of location.",
            allowed_values=("city", "country", "office", "home", "region"),
        ),
    ),
)

# -------------------------------------------------------------------
# Temporal / Quantified / Document (post-MVP additive, 2026-04-22)
# Extends the ontology with entity types that appeared repeatedly in
# conversation but had no structured representation: a specific calendar
# Date, a named Milestone, a numeric Metric, and a referenced Document.
# Additive to v1.0.0 (no version bump).
# -------------------------------------------------------------------

DATE = NodeTypeDefinition(
    type_name="Date",
    description="A specific calendar date referenced in conversation.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    required_properties=(
        PropertyDefinition(
            name="iso_date",
            type="string",
            required=True,
            description="ISO 8601 date in YYYY-MM-DD format.",
        ),
    ),
)

MILESTONE = NodeTypeDefinition(
    type_name="Milestone",
    description=(
        "A discrete significant event in a user or project timeline. "
        "Distinct from Event: a Milestone carries explicit user-assigned "
        "importance and typically anchors to a Date."
    ),
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    optional_properties=(
        PropertyDefinition(
            name="significance",
            type="string",
            required=False,
            description="User-assigned importance level.",
            allowed_values=("high", "medium", "low"),
        ),
    ),
)

METRIC = NodeTypeDefinition(
    type_name="Metric",
    description=(
        "A numeric measurement or quantified property. Attached to other "
        "entities via HAS_METRIC to record benchmark scores, counts, "
        "durations, or other scalar facts."
    ),
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    required_properties=(
        PropertyDefinition(
            name="value",
            type="float",
            required=True,
            description="Numeric value of the measurement.",
        ),
        PropertyDefinition(
            name="unit",
            type="string",
            required=True,
            description=(
                "Unit of measurement (e.g. 'seconds', 'tokens/sec', "
                "'tool_selection', 'percent', 'count')."
            ),
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="as_of_date",
            type="string",
            required=False,
            description="ISO 8601 date the measurement was taken.",
        ),
    ),
)

DOCUMENT = NodeTypeDefinition(
    type_name="Document",
    description=(
        "An extractable reference to an artifact (paper, book, ADR, spec, "
        "article). Distinct from ExternalSource: Document is a content "
        "entity the user or MIST discusses; ExternalSource is a provenance "
        "record of how data entered the graph."
    ),
    knowledge_domain=KnowledgeDomain.EXTERNAL,
    required_properties=(
        PropertyDefinition(
            name="title",
            type="string",
            required=True,
            description="Human-readable title of the document.",
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="doc_type",
            type="string",
            required=False,
            description="Category of document.",
            allowed_values=("adr", "spec", "paper", "book", "article", "other"),
        ),
        PropertyDefinition(
            name="identifier",
            type="string",
            required=False,
            description=("Stable identifier: ADR-XXX, DOI, ISBN, URL, or similar."),
        ),
    ),
)

# ===================================================================
# BRIDGING Entity Types (5)
# ===================================================================

LEARNING_EVENT = NodeTypeDefinition(
    type_name="LearningEvent",
    description="A discrete learning moment where MIST acquired, corrected, or reinforced knowledge.",
    knowledge_domain=KnowledgeDomain.BRIDGING,
    required_properties=(
        PropertyDefinition(
            name="learning_event_type",
            type="string",
            required=True,
            description="Type of learning event.",
            allowed_values=(
                "new_fact",
                "correction",
                "contradiction",
                "reinforcement",
                "self_reflection",
                "capability_growth",
            ),
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="trigger",
            type="string",
            required=False,
            description="What triggered this learning event.",
        ),
        PropertyDefinition(
            name="old_value",
            type="string",
            required=False,
            description="Previous value before the learning event (for corrections).",
        ),
        PropertyDefinition(
            name="new_value",
            type="string",
            required=False,
            description="New value after the learning event.",
        ),
        PropertyDefinition(
            name="source_event_id",
            type="string",
            required=False,
            description="ID of the conversation event that caused this learning.",
        ),
    ),
)

CONVERSATION_CONTEXT = NodeTypeDefinition(
    type_name="ConversationContext",
    description="A snapshot of a conversation providing provenance context for knowledge.",
    knowledge_domain=KnowledgeDomain.BRIDGING,
    required_properties=(
        PropertyDefinition(
            name="conversation_id",
            type="string",
            required=True,
            description="Unique identifier for the conversation.",
        ),
        PropertyDefinition(
            name="date",
            type="datetime",
            required=True,
            description="Date of the conversation.",
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="topic_summary",
            type="string",
            required=False,
            description="Brief summary of the conversation topic.",
        ),
        PropertyDefinition(
            name="key_entities_discussed",
            type="string[]",
            required=False,
            description="List of entity display names discussed in the conversation.",
        ),
    ),
)

EXTERNAL_SOURCE = NodeTypeDefinition(
    type_name="ExternalSource",
    description="Provenance record for a document, MCP tool output, web resource, or upload that produced knowledge.",
    knowledge_domain=KnowledgeDomain.BRIDGING,
    required_properties=(
        PropertyDefinition(
            name="source_type",
            type="string",
            required=True,
            description="Type of external source.",
            allowed_values=("document", "mcp_tool", "web", "upload"),
        ),
        PropertyDefinition(
            name="source_uri",
            type="string",
            required=True,
            description="URI identifying the external source.",
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="title",
            type="string",
            required=False,
            description="Human-readable title of the external source.",
        ),
        PropertyDefinition(
            name="last_accessed",
            type="datetime",
            required=False,
            description="When this source was last accessed.",
        ),
        PropertyDefinition(
            name="content_hash",
            type="string",
            required=False,
            description="Hash of the source content for change detection.",
        ),
    ),
)

VECTOR_CHUNK = NodeTypeDefinition(
    type_name="VectorChunk",
    description="Lightweight reference node pointing to a chunk in the vector store.",
    knowledge_domain=KnowledgeDomain.BRIDGING,
    required_properties=(
        PropertyDefinition(
            name="vector_store_id",
            type="string",
            required=True,
            description="Identifier of this chunk in the vector store.",
        ),
        PropertyDefinition(
            name="source_id",
            type="string",
            required=True,
            description="Identifier of the parent source that produced this chunk.",
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="position",
            type="int",
            required=False,
            description="Ordinal position of this chunk within the source.",
        ),
        PropertyDefinition(
            name="section_title",
            type="string",
            required=False,
            description="Title of the section this chunk belongs to.",
        ),
    ),
)

VAULT_NOTE = NodeTypeDefinition(
    type_name="VaultNote",
    description=(
        "Provenance reference to a vault session note (ADR-010 Cluster 8). "
        "Created on conversation-turn graph writes to anchor every extracted "
        "entity to the canonical session note that produced it. Keyed by "
        "filesystem path; one node per note file regardless of turn count."
    ),
    knowledge_domain=KnowledgeDomain.BRIDGING,
    required_properties=(
        PropertyDefinition(
            name="path",
            type="string",
            required=True,
            description="Absolute filesystem path to the vault session note.",
        ),
    ),
    optional_properties=(
        PropertyDefinition(
            name="event_id",
            type="string",
            required=False,
            description=(
                "Most recent event_id that referenced this vault note. "
                "Per-turn provenance lives on the DERIVED_FROM edge, not here."
            ),
        ),
    ),
)

# ===================================================================
# Collected node type lists
# ===================================================================

_INTERNAL_NODE_TYPES: tuple[NodeTypeDefinition, ...] = (
    MIST_IDENTITY,
    MIST_TRAIT,
    MIST_CAPABILITY,
    MIST_PREFERENCE,
    MIST_UNCERTAINTY,
)

_EXTERNAL_NODE_TYPES: tuple[NodeTypeDefinition, ...] = (
    USER,
    PERSON,
    ORGANIZATION,
    TECHNOLOGY,
    SKILL,
    PROJECT,
    CONCEPT,
    TOPIC,
    EVENT,
    GOAL,
    PREFERENCE,
    LOCATION,
    # Post-MVP additive expansion 2026-04-22: temporal + quantified + document.
    DATE,
    MILESTONE,
    METRIC,
    DOCUMENT,
)

_BRIDGING_NODE_TYPES: tuple[NodeTypeDefinition, ...] = (
    LEARNING_EVENT,
    CONVERSATION_CONTEXT,
    EXTERNAL_SOURCE,
    VECTOR_CHUNK,
    VAULT_NOTE,
)

ALL_NODE_TYPES: list[NodeTypeDefinition] = list(
    _INTERNAL_NODE_TYPES + _EXTERNAL_NODE_TYPES + _BRIDGING_NODE_TYPES
)

ALL_NODE_TYPE_NAMES: list[str] = [nt.type_name for nt in ALL_NODE_TYPES]

# ===================================================================
# INTERNAL Relationships (6)
# ===================================================================

HAS_TRAIT = EdgeTypeDefinition(
    type_name="HAS_TRAIT",
    description="Links MistIdentity to one of its personality/behavioral traits.",
    allowed_source_types=("MistIdentity",),
    allowed_target_types=("MistTrait",),
)

HAS_CAPABILITY = EdgeTypeDefinition(
    type_name="HAS_CAPABILITY",
    description="Links MistIdentity to a capability it has developed.",
    allowed_source_types=("MistIdentity",),
    allowed_target_types=("MistCapability",),
)

HAS_PREFERENCE_INTERNAL = EdgeTypeDefinition(
    type_name="HAS_PREFERENCE",
    description="Links MistIdentity to a preference it has adopted.",
    allowed_source_types=("MistIdentity",),
    allowed_target_types=("MistPreference",),
)

IS_UNCERTAIN_ABOUT = EdgeTypeDefinition(
    type_name="IS_UNCERTAIN_ABOUT",
    description="Links MistIdentity to something it is uncertain about.",
    allowed_source_types=("MistIdentity",),
    allowed_target_types=("MistUncertainty",),
)

ADAPTED_FOR = EdgeTypeDefinition(
    type_name="ADAPTED_FOR",
    description="Indicates that MistIdentity has adapted its behavior for the user.",
    allowed_source_types=("MistIdentity",),
    allowed_target_types=("User",),
)

LEARNED_SELF = EdgeTypeDefinition(
    type_name="LEARNED_SELF",
    description="Links MistIdentity to a learning event about itself.",
    allowed_source_types=("MistIdentity",),
    allowed_target_types=("LearningEvent",),
)

# ===================================================================
# EXTERNAL User-Centric Relationships (13)
# ===================================================================

USES = EdgeTypeDefinition(
    type_name="USES",
    description="Indicates that a user or person actively uses a technology.",
    allowed_source_types=("User", "Person"),
    allowed_target_types=("Technology",),
)

KNOWS = EdgeTypeDefinition(
    type_name="KNOWS",
    description="Indicates familiarity with a skill, concept, technology, or topic.",
    allowed_source_types=("User", "Person"),
    allowed_target_types=("Skill", "Concept", "Technology", "Topic"),
)

WORKS_ON = EdgeTypeDefinition(
    type_name="WORKS_ON",
    description="Indicates active involvement in a project.",
    allowed_source_types=("User", "Person"),
    allowed_target_types=("Project",),
)

WORKS_AT = EdgeTypeDefinition(
    type_name="WORKS_AT",
    description="Indicates employment or membership at an organization.",
    allowed_source_types=("User", "Person"),
    allowed_target_types=("Organization",),
)

INTERESTED_IN = EdgeTypeDefinition(
    type_name="INTERESTED_IN",
    description="Indicates interest in a technology, concept, topic, or skill.",
    allowed_source_types=("User",),
    allowed_target_types=("Technology", "Concept", "Topic", "Skill"),
)

HAS_GOAL = EdgeTypeDefinition(
    type_name="HAS_GOAL",
    description="Links the user to a goal they are pursuing.",
    allowed_source_types=("User",),
    allowed_target_types=("Goal",),
)

PREFERS = EdgeTypeDefinition(
    type_name="PREFERS",
    description="Indicates a positive preference for something.",
    allowed_source_types=("User",),
    allowed_target_types=("Preference", "Technology", "Concept"),
)

DISLIKES = EdgeTypeDefinition(
    type_name="DISLIKES",
    description="Indicates a negative preference or aversion.",
    allowed_source_types=("User",),
    allowed_target_types=("Technology", "Concept", "Topic", "Preference"),
)

EXPERT_IN = EdgeTypeDefinition(
    type_name="EXPERT_IN",
    description="Indicates expert-level proficiency in a technology, skill, or concept.",
    allowed_source_types=("User", "Person"),
    allowed_target_types=("Technology", "Skill", "Concept"),
)

LEARNING = EdgeTypeDefinition(
    type_name="LEARNING",
    description="Indicates actively learning a technology, skill, or concept.",
    allowed_source_types=("User",),
    allowed_target_types=("Technology", "Skill", "Concept"),
)

STRUGGLES_WITH = EdgeTypeDefinition(
    type_name="STRUGGLES_WITH",
    description="Indicates difficulty with a technology, skill, or concept.",
    allowed_source_types=("User",),
    allowed_target_types=("Technology", "Skill", "Concept"),
)

DECIDED = EdgeTypeDefinition(
    type_name="DECIDED",
    description="Links the user to a decision event.",
    allowed_source_types=("User",),
    allowed_target_types=("Event",),
)

EXPERIENCED = EdgeTypeDefinition(
    type_name="EXPERIENCED",
    description="Links a user or person to an event they experienced.",
    allowed_source_types=("User", "Person"),
    allowed_target_types=("Event",),
)

# ===================================================================
# Structural / Social / Provenance Relationships (14)
# ===================================================================

IS_A = EdgeTypeDefinition(
    type_name="IS_A",
    description=("Hierarchical taxonomy relationship. Depth cap at 3, enforced at runtime."),
    allowed_source_types=(
        "Technology",
        "Skill",
        "Concept",
        "Topic",
        "Organization",
        "Location",
    ),
    allowed_target_types=(
        "Technology",
        "Skill",
        "Concept",
        "Topic",
        "Organization",
        "Location",
    ),
)

PART_OF = EdgeTypeDefinition(
    type_name="PART_OF",
    description="Indicates that the source entity is a component or sub-part of the target.",
    allowed_source_types=("Technology", "Concept", "Skill", "Project"),
    allowed_target_types=("Technology", "Concept", "Project", "Organization"),
)

RELATED_TO = EdgeTypeDefinition(
    type_name="RELATED_TO",
    description="General bidirectional association between any two entities.",
    allowed_source_types=tuple(ALL_NODE_TYPE_NAMES),
    allowed_target_types=tuple(ALL_NODE_TYPE_NAMES),
    directional=False,
)

DEPENDS_ON = EdgeTypeDefinition(
    type_name="DEPENDS_ON",
    description="Indicates a dependency from the source to the target.",
    allowed_source_types=("Technology", "Project"),
    allowed_target_types=("Technology",),
)

USED_FOR = EdgeTypeDefinition(
    type_name="USED_FOR",
    description="Indicates that a technology or skill is used for a concept, topic, or project.",
    allowed_source_types=("Technology", "Skill"),
    allowed_target_types=("Concept", "Topic", "Project"),
)

WORKS_WITH = EdgeTypeDefinition(
    type_name="WORKS_WITH",
    description="Bidirectional compatibility between technologies.",
    allowed_source_types=("Technology",),
    allowed_target_types=("Technology",),
    directional=False,
)

KNOWS_PERSON = EdgeTypeDefinition(
    type_name="KNOWS_PERSON",
    description="Links the user to a person they know.",
    allowed_source_types=("User",),
    allowed_target_types=("Person",),
)

MEMBER_OF = EdgeTypeDefinition(
    type_name="MEMBER_OF",
    description="Indicates membership in an organization.",
    allowed_source_types=("User", "Person"),
    allowed_target_types=("Organization",),
)

LEARNED_FROM = EdgeTypeDefinition(
    type_name="LEARNED_FROM",
    description="Links a learning event to the conversation context that produced it.",
    allowed_source_types=("LearningEvent",),
    allowed_target_types=("ConversationContext",),
)

ABOUT = EdgeTypeDefinition(
    type_name="ABOUT",
    description="Links a learning event or uncertainty to the external entity it concerns.",
    allowed_source_types=("LearningEvent", "MistUncertainty"),
    allowed_target_types=tuple(_ext.type_name for _ext in _EXTERNAL_NODE_TYPES),
)

SUPERSEDES = EdgeTypeDefinition(
    type_name="SUPERSEDES",
    description="Indicates that one learning event supersedes a prior one.",
    allowed_source_types=("LearningEvent",),
    allowed_target_types=("LearningEvent",),
)

SOURCED_FROM = EdgeTypeDefinition(
    type_name="SOURCED_FROM",
    description="Links an extractable entity to the external source it was derived from.",
    allowed_source_types=tuple(nt.type_name for nt in _EXTERNAL_NODE_TYPES),
    allowed_target_types=("ExternalSource",),
)

REFERENCES = EdgeTypeDefinition(
    type_name="REFERENCES",
    description="Links an extractable entity to a vector chunk that supports it.",
    allowed_source_types=tuple(nt.type_name for nt in _EXTERNAL_NODE_TYPES),
    allowed_target_types=("VectorChunk",),
)

DERIVED_FROM = EdgeTypeDefinition(
    type_name="DERIVED_FROM",
    description=(
        "Provenance edge linking an extractable entity (or learning event) to the "
        "source it was derived from: a vector chunk or external source for "
        "document-ingest paths, or a vault session note for conversation-turn "
        "extractions (ADR-010 Cluster 8)."
    ),
    allowed_source_types=tuple(nt.type_name for nt in _EXTERNAL_NODE_TYPES)
    + ("MistIdentity", "LearningEvent"),
    allowed_target_types=("VectorChunk", "ExternalSource", "VaultNote"),
)

# -------------------------------------------------------------------
# Temporal / Quantified / Document relationships (post-MVP additive, 2026-04-22)
# Paired with the DATE / MILESTONE / METRIC / DOCUMENT node additions.
# -------------------------------------------------------------------

OCCURRED_ON = EdgeTypeDefinition(
    type_name="OCCURRED_ON",
    description=(
        "Anchors an Event or Milestone to a specific Date. Use when an "
        "event or milestone has a known calendar date."
    ),
    allowed_source_types=("Event", "Milestone"),
    allowed_target_types=("Date",),
)

HAS_METRIC = EdgeTypeDefinition(
    type_name="HAS_METRIC",
    description=(
        "Attaches a numeric measurement (Metric) to the entity it describes. "
        "Use for benchmark scores, durations, counts, or other quantified "
        "properties of a user, project, technology, skill, concept, or goal."
    ),
    allowed_source_types=(
        "User",
        "Project",
        "Technology",
        "Skill",
        "Concept",
        "Goal",
    ),
    allowed_target_types=("Metric",),
)

REFERENCES_DOCUMENT = EdgeTypeDefinition(
    type_name="REFERENCES_DOCUMENT",
    description=(
        "Links an entity to a Document it references or discusses. Distinct "
        "from provenance edges (SOURCED_FROM, DERIVED_FROM): REFERENCES_DOCUMENT "
        "records that the subject mentioned or discussed the artifact, not "
        "that the artifact was the provenance source for a fact."
    ),
    allowed_source_types=(
        "User",
        "MistIdentity",
        "Project",
        "Concept",
        "Topic",
        "Goal",
        "Event",
    ),
    allowed_target_types=("Document",),
)

PRECEDED_BY = EdgeTypeDefinition(
    type_name="PRECEDED_BY",
    description=(
        "Temporal ordering edge: the source Event or Milestone happened "
        "after the target Event, Milestone, or Date. Use for explicit "
        "timeline relationships between events."
    ),
    allowed_source_types=("Event", "Milestone"),
    allowed_target_types=("Event", "Milestone", "Date"),
)

# ===================================================================
# MIST-Scope Relationships (4)
# ===================================================================
# These edges let extraction attribute facts to MIST.AI itself (as a
# MistIdentity node) or to third-party Organizations/Projects. Without
# them, the validator drops MIST-scope facts such as "MIST uses LanceDB".

IMPLEMENTED_WITH = EdgeTypeDefinition(
    type_name="IMPLEMENTED_WITH",
    description=(
        "Indicates the source (MIST, an organization, or a project) is "
        "implemented using the target technology."
    ),
    allowed_source_types=("MistIdentity", "Organization", "Project"),
    allowed_target_types=("Technology",),
)

MIST_HAS_CAPABILITY = EdgeTypeDefinition(
    type_name="MIST_HAS_CAPABILITY",
    description=(
        "Links MistIdentity to an external capability it exercises, such as "
        "a technology, skill, concept, or topic area."
    ),
    allowed_source_types=("MistIdentity",),
    allowed_target_types=("Technology", "Skill", "Concept", "Topic"),
)

MIST_HAS_TRAIT = EdgeTypeDefinition(
    type_name="MIST_HAS_TRAIT",
    description=(
        "Links MistIdentity to a personality-, preference-, or topic-level "
        "trait expressed as an external-domain entity."
    ),
    allowed_source_types=("MistIdentity",),
    allowed_target_types=("Concept", "Topic", "Preference", "Skill"),
)

MIST_HAS_PREFERENCE = EdgeTypeDefinition(
    type_name="MIST_HAS_PREFERENCE",
    description=(
        "Links MistIdentity to a preference it holds about tooling, concepts, "
        "technologies, or topics."
    ),
    allowed_source_types=("MistIdentity",),
    allowed_target_types=("Preference", "Concept", "Technology", "Topic"),
)

# ===================================================================
# Collected edge type lists
# ===================================================================

_INTERNAL_EDGE_TYPES: tuple[EdgeTypeDefinition, ...] = (
    HAS_TRAIT,
    HAS_CAPABILITY,
    HAS_PREFERENCE_INTERNAL,
    IS_UNCERTAIN_ABOUT,
    ADAPTED_FOR,
    LEARNED_SELF,
)

_EXTERNAL_USER_CENTRIC_EDGE_TYPES: tuple[EdgeTypeDefinition, ...] = (
    USES,
    KNOWS,
    WORKS_ON,
    WORKS_AT,
    INTERESTED_IN,
    HAS_GOAL,
    PREFERS,
    DISLIKES,
    EXPERT_IN,
    LEARNING,
    STRUGGLES_WITH,
    DECIDED,
    EXPERIENCED,
)

_STRUCTURAL_EDGE_TYPES: tuple[EdgeTypeDefinition, ...] = (
    IS_A,
    PART_OF,
    RELATED_TO,
    DEPENDS_ON,
    USED_FOR,
    WORKS_WITH,
    KNOWS_PERSON,
    MEMBER_OF,
    LEARNED_FROM,
    ABOUT,
    SUPERSEDES,
    SOURCED_FROM,
    REFERENCES,
    DERIVED_FROM,
    # Post-MVP additive (2026-04-22): temporal + quantified + document edges.
    OCCURRED_ON,
    HAS_METRIC,
    REFERENCES_DOCUMENT,
    PRECEDED_BY,
)

_MIST_SCOPE_EDGE_TYPES: tuple[EdgeTypeDefinition, ...] = (
    IMPLEMENTED_WITH,
    MIST_HAS_CAPABILITY,
    MIST_HAS_TRAIT,
    MIST_HAS_PREFERENCE,
)

ALL_EDGE_TYPES: list[EdgeTypeDefinition] = list(
    _INTERNAL_EDGE_TYPES
    + _EXTERNAL_USER_CENTRIC_EDGE_TYPES
    + _STRUCTURAL_EDGE_TYPES
    + _MIST_SCOPE_EDGE_TYPES
)

ALL_EDGE_TYPE_NAMES: list[str] = [et.type_name for et in ALL_EDGE_TYPES]

# ===================================================================
# Extractable subsets (what the LLM extractor may produce)
# ===================================================================

# All 16 external node types plus MistIdentity are extractable (17 total).
# MistIdentity is also defined as an INTERNAL singleton node, but is promoted
# to extractable so that MIST-scope facts (e.g. "MIST uses LanceDB") can be
# attributed at extraction time.
# Post-MVP additive (2026-04-22): Date, Milestone, Metric, Document joined the
# EXTERNAL set.
EXTRACTABLE_NODE_TYPES: list[str] = [nt.type_name for nt in _EXTERNAL_NODE_TYPES] + [
    MIST_IDENTITY.type_name,
]

# 13 user-centric + 8 original structural + 4 post-MVP structural
# (OCCURRED_ON, HAS_METRIC, REFERENCES_DOCUMENT, PRECEDED_BY) + 4 MIST-scope.
# Excludes provenance (LEARNED_FROM, ABOUT, SUPERSEDES, SOURCED_FROM, REFERENCES,
# DERIVED_FROM) and internal (HAS_TRAIT etc.) edges.
_EXTRACTABLE_STRUCTURAL_EDGE_TYPES: tuple[EdgeTypeDefinition, ...] = (
    IS_A,
    PART_OF,
    RELATED_TO,
    DEPENDS_ON,
    USED_FOR,
    WORKS_WITH,
    KNOWS_PERSON,
    MEMBER_OF,
    IMPLEMENTED_WITH,
    MIST_HAS_CAPABILITY,
    MIST_HAS_TRAIT,
    MIST_HAS_PREFERENCE,
    # Post-MVP additive (2026-04-22): temporal + quantified + document edges.
    OCCURRED_ON,
    HAS_METRIC,
    REFERENCES_DOCUMENT,
    PRECEDED_BY,
)

EXTRACTABLE_RELATIONSHIP_TYPES: list[str] = [
    et.type_name for et in (_EXTERNAL_USER_CENTRIC_EDGE_TYPES + _EXTRACTABLE_STRUCTURAL_EDGE_TYPES)
]

# ===================================================================
# Extraction Rules
# ===================================================================

_EXTRACTION_SYSTEM_PROMPT = """\
You are a knowledge-graph extraction engine for the MIST.AI cognitive assistant.

Your job is to read a user utterance and produce structured entities and
relationships that should be stored in MIST's persistent knowledge graph.

ENTITY TYPES YOU MAY PRODUCE (17):
  User, Person, Organization, Technology, Skill, Project, Concept, Topic,
  Event, Goal, Preference, Location, Date, Milestone, Metric, Document,
  MistIdentity

RELATIONSHIP TYPES YOU MAY PRODUCE (29):
  User-centric: USES, KNOWS, WORKS_ON, WORKS_AT, INTERESTED_IN, HAS_GOAL,
    PREFERS, DISLIKES, EXPERT_IN, LEARNING, STRUGGLES_WITH, DECIDED, EXPERIENCED
  Structural: IS_A, PART_OF, RELATED_TO, DEPENDS_ON, USED_FOR, WORKS_WITH,
    KNOWS_PERSON, MEMBER_OF
  Temporal / quantified / document: OCCURRED_ON, HAS_METRIC,
    REFERENCES_DOCUMENT, PRECEDED_BY
  MIST-scope: IMPLEMENTED_WITH, MIST_HAS_CAPABILITY, MIST_HAS_TRAIT,
    MIST_HAS_PREFERENCE

RULES:
1. Only extract entities and relationships that are clearly supported by the
   utterance. Do not infer beyond what is stated.
2. Each entity must have a display_name. Include type-specific properties only
   when explicitly mentioned.
3. Each relationship must reference source and target entities by display_name
   and include a confidence score.
4. Respect the allowed source/target type constraints for each relationship.
5. Assign confidence scores between 0.0 and 1.0 based on how explicitly the
   information was stated.
6. Do not produce internal (Mist*) entities or bridging entities -- those are
   managed by separate subsystems.
"""

EXTRACTION_RULES = ExtractionRules(
    system_prompt=_EXTRACTION_SYSTEM_PROMPT,
    allowed_node_types=tuple(EXTRACTABLE_NODE_TYPES),
    allowed_relationship_types=tuple(EXTRACTABLE_RELATIONSHIP_TYPES),
    property_extraction_enabled=True,
    max_entities_per_utterance=10,
    max_relationships_per_utterance=15,
    min_confidence_threshold=0.4,
)

# ===================================================================
# OntologyVersion v1.0.0
# ===================================================================

ONTOLOGY_V1_0_0 = OntologyVersion(
    version="1.0.0",
    created_at=datetime(2026, 3, 22, tzinfo=UTC),
    description=(
        "First stable ontology for the MIST.AI knowledge graph. "
        "Covers internal self-model, external world knowledge, and "
        "bridging provenance types."
    ),
    node_types=tuple(ALL_NODE_TYPES),
    edge_types=tuple(ALL_EDGE_TYPES),
    extraction_rules=EXTRACTION_RULES,
    confidence_policies=(
        CONFIDENCE_INTERNAL,
        CONFIDENCE_EXTERNAL,
        CONFIDENCE_BRIDGING,
    ),
    universal_entity_properties=UNIVERSAL_ENTITY_PROPERTIES,
    universal_relationship_properties=UNIVERSAL_RELATIONSHIP_PROPERTIES,
    parent_version=None,
    migration_script_path=None,
    active=True,
    deprecated=False,
    deprecated_reason=None,
)
